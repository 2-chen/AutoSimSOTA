#!/usr/bin/env python3
"""Collect component evidence without upgrading failures or submitting new work."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def seconds(start, end):
    return (datetime.fromisoformat(end.replace("Z", "+00:00")) -
            datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/AutoResearch/AutoSimSOTA/compute_validation"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.absolute(); output = args.output or root / "report_20260914"
    attempts, failures = [], []
    for directory in sorted(root.glob("matrix_*/*gpu")):
        submission = read(directory / "submission.json")
        if not submission:
            failure = read(directory / "submission_failure.json")
            if failure:
                failures.append({"directory":str(directory), **failure})
            continue
        summary = read(directory / "summary.json")
        description = read(directory / "job_description.json")
        inventory = read(directory / "inventory.json")
        n = int(directory.name.removesuffix("gpu"))
        row = {"directory":str(directory), "requested_gpus":n, "job":submission.get("name"),
               "platform_state":description.get("state"), "summary":summary,
               "actual_allowed_uuids":[d["uuid"] for d in inventory.get("gpus", [])
                   if d["index"] in inventory.get("allowed", [])],
               "host":inventory.get("host"), "phase_seconds":{}}
        start = description.get("start_time")
        end = description.get("complete_time") or description.get("suspend_time")
        row["platform_gpu_hours"] = n * seconds(start, end)/3600 if start and end else None
        if start:
            row["pre_start_seconds"] = seconds(description["create_time"], start)
        manifest = directory.parent / "source_manifest.json"
        row["source_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        blocks = [read(p) for p in (directory / "schedules/T2_queue").glob("block_*/fixed_cuda.json")]
        row["fixed_cuda_contract_verified"] = (len(blocks)==16 and all(b.get("steps")==1000 and b.get("status")=="passed" for b in blocks))
        row["telemetry"] = read(directory / "telemetry.json")
        row["host_capacity"] = read(directory / "schedules/T0_cuda/schedule.json").get("host_capacity",{})
        for phase in (directory / "schedules").glob("*"):
            records = [read(p) for name in ("outcome.json","failure.json") for p in phase.rglob(name)]
            records = [r for r in records if r.get("started_at") and r.get("finished_at")]
            if records:
                row["phase_seconds"][phase.name] = seconds(min(r["started_at"] for r in records),
                    max(r["finished_at"] for r in records))
        row["native_environment_setup_seconds"] = []
        for census in directory.rglob("startup_census.jsonl"):
            events = [json.loads(line) for line in census.read_text().splitlines() if line]
            times = {e["phase"]:e["time"] for e in events}
            if times.get("environment_constructing") and times.get("environment_ready"):
                row["native_environment_setup_seconds"].append({"census":str(census),
                    "seconds":seconds(times["environment_constructing"],times["environment_ready"])})
        row["fixed_bank"] = {"episodes":len(read(directory / "fixed_native_result.json").get("episodes", [])),
                             "merge_receipt":read(directory / "fixed_native/merge_receipt.json")}
        attempts.append(row)
    attempts.sort(key=lambda row: row["summary"].get("started", float("inf")))
    required = ["T0_cuda", "T2_queue", "T1_native_concurrency", "T2_native_fixed_bank",
                "T4_collection", "T3_data_audit", "T3_short_training"]
    arms = []
    for n in (1,2,4,8):
        rows = [r for r in attempts if r["requested_gpus"] == n]
        evidence = {}
        for name in required:
            evidence[name] = [r["job"] for r in rows if r["summary"].get("tests",{}).get(name,{}).get("status") == "passed"]
        if n in (2,4):
            for name in ("T5_scheduler_faults", "T5_orphan_cleanup"):
                evidence[name] = [r["job"] for r in rows if (lambda v: v.get("status") == "passed" or v.get("passed") is True)(
                    r["summary"].get("tests",{}).get(name,{}))]
        # The fixed bank also requires the integrity-checked merge artifact itself.
        evidence["fixed_bank_merged"] = [r["job"] for r in rows if r["fixed_bank"]["episodes"] == 16 and r["fixed_bank"]["merge_receipt"]]
        arms.append({"gpus":n, "jobs":[r["job"] for r in rows], "evidence":evidence,
                     "component_coverage_complete":bool(rows) and all(evidence.values()),
                     "missing":[k for k,v in evidence.items() if not v]})
    api = {"requests":0,"charged_tokens":0,"reserved_unknown_tokens":0,"ledgers":[]}
    for p in sorted(root.glob("**/api/budget.json")):
        if "/source/" in str(p) or "/implementation_baseline/" in str(p):
            continue
        d = read(p)
        if "calls" not in d:
            continue
        api["requests"] += d["calls"]
        api["charged_tokens"] += d["charged_tokens"]
        api["reserved_unknown_tokens"] += d["reserved_tokens"]
        api["ledgers"].append(str(p))
    # The isolated connectivity preflight has its own receipt outside the agent ledger.
    api["preflight"] = read(root / "api_validation_20260914/preflight.json")
    result = {"generated_at":datetime.now(timezone.utc).isoformat(),"full_autoresearch":False,
              "arms":arms,"attempts":attempts,"submission_failures":failures,"api":api,
              "completed_platform_gpu_hours":sum(r["platform_gpu_hours"] or 0 for r in attempts),
              "finished_application_ledger_gpu_hours":sum(r["summary"].get("allocated_gpu_hours",0) for r in attempts),
              "short_chain_jobs":[r["job"] for r in attempts if r["summary"].get("tests",{}).get("T4_short_chain",{}).get("status")=="passed"]}
    result["short_chain_gpus"] = [r["requested_gpus"] for r in attempts if r["job"] in result["short_chain_jobs"]]
    result["api_execution_jobs"] = [r["job"] for r in attempts if r["summary"].get("tests",{}).get("T6_api_execution",{}).get("status")=="passed"]
    result["all_required_components_passed"] = (all(a["component_coverage_complete"] for a in arms)
        and 1 in result["short_chain_gpus"] and any(n>1 for n in result["short_chain_gpus"])
        and bool(result["api_execution_jobs"]))
    real_audit = read(root / "api_validation_20260914/isolated_activation_validation.json")
    result["codegen_real_workload"] = real_audit
    result["codegen_net_gain_verified"] = bool(real_audit.get("promoted_for_production")
                                               and real_audit.get("overall_payback_verified"))
    result["plan_acceptance_complete"] = (result["all_required_components_passed"]
                                           and result["codegen_net_gain_verified"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "matrix.json").write_text(json.dumps(result, ensure_ascii=False,indent=2)+"\n")
    lines = ["# 算力调度组件验证报告", "", f"更新时间：{result['generated_at']}", "",
        "四种规格完整通过：" + ("是" if result["all_required_components_passed"] else "否，见缺失证据。"), "",
        "| 规格 | 独立 Job ID | 尚缺通过证据 |", "| --- | --- | --- |"]
    for arm in arms:
        lines.append(f"| {arm['gpus']} × 5090 | {', '.join(arm['jobs']) or '未创建'} | {', '.join(arm['missing']) or '基础组件齐全；短链路见下文'} |")
    lines += ["", "固定短链路已通过的规格：" + (", ".join(str(n) for n in sorted(set(result["short_chain_gpus"]))) or "暂无") +
              "。验收需要单卡以及至少一个多卡规格通过。", "",
              "| 固定工作量测量（仅阶段通过的尝试） | 规格 | 16 CUDA 工作块秒数 | 16 原生 episode 秒数 |", "| --- | --- | --- | --- |"]
    for r in attempts:
        tests = r["summary"].get("tests",{})
        def timing(name):
            value = r["phase_seconds"].get(name)
            if tests.get(name,{}).get("reused_without_execution"):
                return "—（复用，未重跑）"
            if value is None:
                return "—（未执行）"
            if name == "T2_queue" and not r["fixed_cuda_contract_verified"]:
                return "—（工作量不同）"
            return f"{value:.3f}" if value is not None and tests.get(name,{}).get("status")=="passed" else "—"
        lines.append(f"| {r['job']} | {r['requested_gpus']} | {timing('T2_queue')} | {timing('T2_native_fixed_bank')} |")
    lines += ["", "| 尝试 | 平台状态 / 组件状态 | 完成分配 GPU-h | 证据 |", "| --- | --- | --- | --- |"]
    for r in attempts:
        cost = f"{r['platform_gpu_hours']:.4f}" if r["platform_gpu_hours"] is not None else "尚未结算"
        lines.append(f"| {r['job']} | {r['platform_state']} / {r['summary'].get('status','未开始')} | {cost} | [目录]({r['directory']}) |")
    lines += ["", f"已结束任务的平台分配合计：{result['completed_platform_gpu_hours']:.4f} GPU-h；运行中和排队任务尚未计入终态合计。",
        f"API agent 账本：{api['requests']} 次，已结算 {api['charged_tokens']} tokens，未知响应保留 {api['reserved_unknown_tokens']} tokens；独立连通性预检另列。",
        "", "固定队列是 16 个相同 CUDA 工作块；固定原生 bank 是 16 个相同 episode。短训练/采集按每卡独立工作测试容量，不能用于固定工作量加速结论。",
        "", "各阶段仅一次测量时属于初步结果；启动开销、GPU 型号相同但节点/驱动不同等因素需与调度开销区分。",
        "", "DDP 线性模型探针与 ACT DDP 分别验收；通信失败不会启用 ACT DDP。采集失败或零产出均不等同于短链路通过。",
        "", "生成补丁的微基准收益不代表全系统净收益。当前实际审计、错误补丁回退及负收益证据见 api_validation_20260914；完整 AutoResearch 未运行。",
        "", "原始失败日志、未执行项、API 未知响应和提交配额错误均保留在 [matrix.json](matrix.json)。"]
    if real_audit:
        lines += ["", f"实际 {real_audit['episode_count']} episode 元数据审计：原函数 {real_audit['reference_seconds']:.6f} s；"
            f"隔离候选 {real_audit['isolated_candidate_seconds']:.6f} s。输出等价，原函数已恢复，未全局推广。",
            "M4 的真实业务净收益门槛尚未达成；不能以函数微基准或 API 参数成功执行代替该门槛。"]
    (output / "compute_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"report":str(output),"complete":result["all_required_components_passed"],
                      "completed_platform_gpu_hours":result["completed_platform_gpu_hours"]}))


if __name__ == "__main__":
    main()
