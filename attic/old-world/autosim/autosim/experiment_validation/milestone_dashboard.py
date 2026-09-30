"""Machine-checkable live dashboard for the next RoboSyn milestone."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from autosim.research.common import atomic_json, freeze_files, now, read_json
from .queue_manifest import versioned_manifest


def load(path: Path) -> dict:
    return read_json(path) if path.is_file() else {}


def click_training_progress(research_root: Path) -> dict:
    state = load(research_root / "click_bell/state.json")
    stage = state.get("stage", "missing")
    arm = None
    if stage == "long_training:auto":
        arm = research_root / "click_bell/round_2/auto"
    elif stage == "long_training:random":
        arm = research_root / "click_bell/round_1/random"
    if arm is None:
        return {"stage": stage, "approximate_logged_step": None}
    log = arm / "process_train_80000/stdout.log"
    text = log.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace") if log.is_file() else ""
    matches = re.findall(r"step:(\d+)K", text)
    process = load(arm / "process_train_80000/process.json")
    return {"stage": stage, "arm": str(arm),
            "approximate_logged_step": int(matches[-1]) * 1000 if matches else None,
            "target_step": 80000, "process_pid": process.get("pid"),
            "process_returncode": process.get("returncode"), "log": str(log)}


def decide(inputs: dict) -> tuple[list[dict], bool]:
    integration = inputs["integration"]
    baseline = inputs["baseline"]
    gap = inputs["gap"]
    comparison_state = inputs["comparison_state"]
    comparison = inputs["comparison"]
    fairness = inputs["fairness"]
    checkpoints = inputs["checkpoints"]
    gates = [
        {"gate": "native_integration", "passed": integration.get("status") == "integration_gate_reached",
         "actual": integration.get("status", "missing"), "required": "integration_gate_reached"},
        {"gate": "official_act_10x100", "passed": baseline.get("status") == "baseline_table_complete" and
         int(baseline.get("completed", 0)) == 10,
         "actual": f"{baseline.get('status', 'missing')}:{baseline.get('completed', 0)}/10",
         "required": "baseline_table_complete:10/10"},
        {"gate": "two_collection_gaps", "passed": gap.get("status") == "collection_gap_handled",
         "actual": gap.get("status", "missing"), "required": "collection_gap_handled"},
        {"gate": "three_research_loops", "passed": comparison_state.get("status") == "three_task_comparison_complete" and
         comparison.get("study_execution_complete") is True,
         "actual": f"{comparison_state.get('status', 'missing')}:{comparison.get('study_execution_complete')}",
         "required": "three_task_comparison_complete:True"},
        {"gate": "equal_total_research_budget", "passed": fairness.get("all_three_tasks_equal_budget") is True,
         "actual": fairness.get("all_three_tasks_equal_budget"), "required": True},
        {"gate": "nine_checkpoint_integrity", "passed": checkpoints.get("all_selected_checkpoints_passed") is True,
         "actual": checkpoints.get("all_selected_checkpoints_passed"), "required": True},
        {"gate": "automatic_decision_effect", "passed":
         comparison.get("automatic_decision_beats_random_control_established") is True,
         "actual": comparison.get("automatic_decision_beats_random_control_established"), "required": True},
        {"gate": "no_final_or_sota_claim", "passed": comparison.get("final_test_used") is False and
         comparison.get("sota_established") is False,
         "actual": {"final_test_used": comparison.get("final_test_used"),
                    "sota_established": comparison.get("sota_established")},
         "required": {"final_test_used": False, "sota_established": False}},
    ]
    return gates, all(row["passed"] for row in gates)


def snapshot(root: Path, paths: dict[str, Path], research_root: Path) -> dict:
    inputs = {name: load(path) for name, path in paths.items()}
    gates, complete = decide(inputs)
    value = {"updated_at": now(), "kind": "next_milestone_live_dashboard",
             "milestone_complete": complete, "passed_gates": sum(row["passed"] for row in gates),
             "total_gates": len(gates), "gates": gates,
             "click_bell_training": click_training_progress(research_root),
             "sources": {name: str(path) for name, path in paths.items()},
             "all_ten_task_sota": False,
             "scope": "first three-task development comparison, not official hidden evaluation"}
    atomic_json(root / "milestone_status.json", value)
    lines = ["# 下一里程碑实时验收表", "", f"更新时间：{value['updated_at']}。", "",
             f"当前通过：{value['passed_gates']}/{value['total_gates']}；里程碑完成：{'是' if complete else '否'}。", "",
             "| 验收门 | 当前值 | 要求 | 通过 |", "| --- | --- | --- | --- |"]
    for row in gates:
        lines.append(f"| {row['gate']} | `{row['actual']}` | `{row['required']}` | {'是' if row['passed'] else '否'} |")
    progress = value["click_bell_training"]
    lines += ["", f"ClickBell阶段：`{progress['stage']}`；日志步数："
              f"`{progress['approximate_logged_step']}` / `{progress.get('target_step')}`。",
              "", "只有8个门全部由实际产物通过时才标记完成；后台服务启动、单元测试通过或局部模型完成均不能替代这些门。"]
    (root / "milestone_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def queue(root: Path, paths: dict[str, Path], research_root: Path,
          max_hours: float, poll_seconds: float):
    if not 0 < max_hours <= 336 or not 5 <= poll_seconds <= 300:
        raise ValueError("invalid bounded dashboard settings")
    local = Path(__file__).resolve()
    versioned_manifest(root, {"inputs": {name: str(path) for name, path in paths.items()},
                              "research_root": str(research_root),
                              "sources": freeze_files([local, local.with_name("queue_manifest.py")]),
                              "sota_claim": False})
    state_path = root / "dashboard_state.json"
    prior = load(state_path)
    if prior.get("status") in {"completed", "queue_wait_budget_exhausted"}:
        return prior
    deadline = float(prior.get("deadline_epoch", time.time() + max_hours * 3600))
    state = {**prior, "status": "running", "started_at": prior.get("started_at", now()),
             "deadline_epoch": deadline}
    while time.time() < deadline:
        value = snapshot(root, paths, research_root)
        if value["milestone_complete"]:
            state.update(status="completed", passed_gates=value["passed_gates"], updated_at=now())
            atomic_json(state_path, state)
            return state
        state.update(stage="waiting_milestone_gates", passed_gates=value["passed_gates"],
                     total_gates=value["total_gates"], updated_at=now())
        atomic_json(state_path, state)
        time.sleep(poll_seconds)
    state.update(status="queue_wait_budget_exhausted", updated_at=now())
    atomic_json(state_path, state)
    return state


def parse_paths(args) -> dict[str, Path]:
    return {"integration": args.integration.absolute(), "baseline": args.baseline.absolute(),
            "gap": args.gap.absolute(), "comparison_state": args.comparison_state.absolute(),
            "comparison": args.comparison.absolute(), "fairness": args.fairness.absolute(),
            "checkpoints": args.checkpoints.absolute()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("snapshot", "queue"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--integration", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gap", type=Path, required=True)
    parser.add_argument("--comparison-state", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--fairness", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=240)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    paths = parse_paths(args)
    if args.command == "snapshot":
        snapshot(args.root.absolute(), paths, args.research_root.absolute())
    else:
        queue(args.root.absolute(), paths, args.research_root.absolute(), args.max_hours, args.poll_seconds)
