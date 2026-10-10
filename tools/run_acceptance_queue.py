"""Serial operator-defined acceptance runs; never edits agents, scores or source repositories.

Only advances after the preceding service exits and its native research evidence passes
the acceptance gate. A failure stops the queue for diagnosis; it never resets or retries
an experiment automatically. This is launch management, not a benchmark adapter.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path

from autosim.research.common import atomic_json, atomic_text, now, read_json

PROJECT = Path(__file__).resolve().parents[1]


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def record(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("refusing symlink record")
    try:
        value = read_json(path)
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError("record must be an object")
    return value


def acceptance(output: Path, *, require_data_campaign: bool = False) -> tuple[bool, str]:
    preparation = record(output / "preparation_derived.json")
    research = record(output / "research/derived/research_report.json")
    if preparation.get("status") != "completed" or research.get("run_status") != "completed":
        return False, "原生研究未完成；保留原 run 调查，不重置或创建替代实验"
    rounds = research.get("rounds") or []
    baseline = [r for r in rounds if r.get("round") == 0 and r.get("measured") is True
                and number(r.get("metric_value"))]
    candidate = [r for r in rounds if number(r.get("round")) and r["round"] > 0
                 and r.get("status") == "measured" and number(r.get("metric_value"))]
    from autosim.research.candidate_source_guard import historical_mismatches
    warnings = historical_mismatches(output / 'research/derived')
    if any(row.get('status') == 'unverifiable' for row in warnings):
        return False, '候选源码历史无法核验；不得自动通过验收'
    mismatched = {row['label'] for row in warnings}
    candidate = [row for row in candidate if row.get('label') not in mismatched]
    if not baseline or not candidate:
        return False, "缺少有效 baseline 或候选计分；零分是有效数值，但空指标不是实验结果"
    if (preparation.get("evaluation_verdict") or {}).get("verdict") != "real":
        return False, "独立 Supervisor 尚未确认实验有效"
    confirmation = research.get("confirmation") or {}
    if (confirmation.get("status") != "taken" or not number(confirmation.get("metric_value"))
            or not number(confirmation.get("baseline_metric_value"))):
        return False, "尚未完成独立留出确认；不能把 --confirm 参数当作确认结果"
    if require_data_campaign:
        from autosim.research.data_campaign import view
        campaigns = view(output, 'derived')
        reviews = (preparation.get('evaluation_verdict') or {}).get('data_campaign_reviews') or []
        passed = False
        for campaign in campaigns:
            if campaign.get('status') == 'evidence_unavailable':
                continue
            observations = campaign['observations']
            paired = all(any(r['arm'] == arm and r['seed'] == seed for r in observations)
                         for arm in ('targeted', 'random') for seed in campaign['comparison']['seeds'])
            reviewed = any(r.get('id') == campaign['id'] and r.get('verdict') == 'real' for r in reviews)
            passed |= paired and reviewed
        if not passed:
            return False, '失败驱动/随机数据对照或独立数据公平性审核尚未完成；普通计分闭环不能冒充数据研究验收'
    return True, "真实 baseline、候选计分、独立确认和 Supervisor 已完成；不等于 SOTA"


def service(unit: str) -> dict:
    result = subprocess.run(["systemctl", "--user", "show", unit,
        "--property=LoadState,ActiveState,SubState,MainPID,ExecMainStatus"],
        check=True, capture_output=True, text=True, timeout=15)
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def command(job: dict) -> list[str]:
    args = [str(PROJECT / ".venv/bin/autosim"), "research", job["repo"], job["output"],
            "1", json.dumps(job["settings"], ensure_ascii=False),
            "--framework", "autosota_sim_v1", "--isolated-copy", "--tracked-copy",
            "--budget-scope", "task", "--wall-seconds", "172800",
            "--agent-turn-budget-usd", "4", "--agent-total-budget-usd", "60",
            "--agent-timeout-seconds", "900", "--max-actions", "80",
            "--max-relaunch", "8", "--confirm", "--interpreter", job["interpreter"]]
    for binding in job["resources"]:
        args.extend(["--resource", binding])
    if job.get('research_brief'):
        args.extend(['--research-brief', job['research_brief']])
    return args


def validate(document: dict, manifest: Path) -> list[dict]:
    jobs = document.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("manifest requires explicit jobs")
    units, outputs = set(), set()
    for job in jobs:
        if set(job) - {"id", "repo", "output", "unit", "settings", "interpreter", "resources", "adopt", 'research_brief', 'require_data_campaign'}:
            raise ValueError("unknown job field")
        if not isinstance(job.get("settings"), dict) or not isinstance(job.get("resources"), list):
            raise ValueError("settings/resources must be explicit")
        if type(job.get('require_data_campaign', False)) is not bool:
            raise ValueError('require_data_campaign must be boolean')
        if job.get('research_brief'):
            brief = Path(job['research_brief'])
            if not brief.is_absolute() or not brief.is_file() or brief.is_symlink():
                raise ValueError('research_brief must be an explicit regular file')
        unit = str(job["unit"])
        if not unit.startswith("autosim-") or not all(c.isalnum() or c in "-_." for c in unit):
            raise ValueError("unsafe unit name")
        repo, output = Path(job["repo"]), Path(job["output"])
        if not repo.is_absolute() or not repo.is_dir() or not output.is_absolute():
            raise ValueError("require absolute repository and output")
        if output.is_symlink() or not output.resolve().is_relative_to(PROJECT / "autoresearch_runs"):
            raise ValueError("output must be an independent project run")
        if output.resolve().is_relative_to(repo.resolve()) or repo.resolve().is_relative_to(output.resolve()):
            raise ValueError("source and output must be disjoint")
        if unit in units or str(output) in outputs:
            raise ValueError("duplicate unit/output")
        units.add(unit); outputs.add(str(output))
        if not job.get("adopt") and output.exists():
            snapshot = read_json(output / "workspace_snapshot.json")
            if snapshot.get("source") != str(repo.resolve()):
                raise ValueError("existing run lacks matching source identity")
        if job.get("adopt"):
            snapshot = read_json(output / "workspace_snapshot.json")
            if snapshot.get("source") != str(repo.resolve()):
                raise ValueError("adopted run source identity differs")
    return jobs


def publish(root: Path, state: dict):
    state["updated_at"] = now()
    atomic_json(root / "queue_state.json", state)
    lines = ["# 真实实验验收队列", "", f"更新时间：{state['updated_at']}", "",
             "串行执行；前项未验收则暂停后续。不改源仓库、不自动追加预算、不清历史。", "",
             "| 仓库 / 任务 | 状态 | 说明 |", "| --- | --- | --- |"]
    if state.get('resource_wait'):
        lines[4:4] = [f"当前等待 `{state['resource_wait']['unit']}` 退出后启动，尚未消耗新实验预算。", '']
    for row in state["jobs"]:
        link = Path(row["output"]).relative_to(root.parent).as_posix() + "/RUN.md"
        label = (f"[{row['id']}](../{link})" if (Path(row["output"]) / "RUN.md").is_file()
                 else row["id"])
        note = str(row.get("note") or "等待前项验收").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {label} | {row['status']} | {note} |")
    lines += ["", "验收来自原生计分记录及独立审核；未完成不等于仓库不支持。视频和数据采集仍需逐项检查实际证据。"]
    atomic_text(root / "MATRIX.md", "\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument('--wait-unit', action='append', default=[],
                        help='resource serialization only: wait for an existing autosim service to exit; not its scientific acceptance')
    args = parser.parse_args(argv)
    if args.manifest.is_symlink():
        parser.error("refusing symlink manifest")
    manifest = args.manifest.resolve(strict=True)
    if not manifest.is_relative_to(PROJECT / "autoresearch_runs"):
        parser.error("manifest must be an operator-owned local run file")
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll interval must be between 1 and 60 seconds")
    document = read_json(manifest)
    jobs = validate(document, manifest)
    root = manifest.parent
    with (root / "queue.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = record(root / "queue_state.json") or {"schema_version": 1,
            "manifest": document, "jobs": [{"id": j["id"], "output": j["output"],
                "status": "等待"} for j in jobs]}
        if state.get("manifest") != document:
            raise ValueError("queue input is frozen; refusing to change existing runs")
        for unit in args.wait_unit:
            if not unit.startswith('autosim-') or not all(c.isalnum() or c in '-_.' for c in unit):
                raise ValueError('wait-unit must be an explicit autosim service')
            while True:
                observed = service(unit)
                if observed.get('LoadState') == 'not-found' or observed.get('ActiveState') not in {
                        'active', 'activating', 'deactivating', 'reloading'}:
                    break
                state['resource_wait'] = {'unit': unit, 'status': 'waiting',
                    'note': '只等待前一进程释放资源，不修改、终止或验收它；新实验预算尚未启动'}
                publish(root, state)
                time.sleep(args.poll_seconds)
        state.pop('resource_wait', None)
        for job, row in zip(jobs, state["jobs"]):
            if row["status"] == "已验收":
                continue
            observed = service(job["unit"])
            if observed.get("LoadState") != "not-found":
                snapshot = record(Path(job["output"]) / "workspace_snapshot.json")
                if snapshot.get("source") != str(Path(job["repo"]).resolve()):
                    raise ValueError("existing service has no matching run source identity")
            if observed.get("LoadState") == "not-found":
                if job.get("adopt") or Path(job["output"]).exists():
                    row.update(status="暂停", note="服务不存在；核对旧进程与原 run 后人工续跑")
                    publish(root, state); return 1
                subprocess.run(["systemd-run", "--user", "--unit=" + job["unit"],
                    "--working-directory=" + str(PROJECT), "--property=Restart=no",
                    "--property=KillMode=control-group",
                    "--setenv=PYTHONPATH=" + str(PROJECT / "autosim"),
                    "--setenv=PATH=" + str(PROJECT / ".venv/bin") +
                    ":" + os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                    *command(job)], check=True, timeout=30)
            while True:
                observed = service(job["unit"])
                if observed.get("ActiveState") in {"active", "activating", "deactivating"}:
                    row.update(status="运行中", note="系统自主执行；后续任务尚未启动预算")
                    publish(root, state); time.sleep(args.poll_seconds); continue
                passed, reason = acceptance(Path(job["output"]),
                    require_data_campaign=job.get('require_data_campaign', False))
                row.update(status="已验收" if passed else "暂停", note=reason,
                           service_exit=observed.get("ExecMainStatus"))
                publish(root, state)
                if not passed:
                    return 1
                break
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
