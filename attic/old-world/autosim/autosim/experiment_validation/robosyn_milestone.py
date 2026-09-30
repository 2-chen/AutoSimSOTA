"""Durable ten-task RoboSyn development baseline queue and milestone ledger.

This module is intentionally separate from the frozen integration runtime. It
evaluates released ACT checkpoints on a new development seed bank. These are
local reproductions, not official leaderboard scores and not final-test results.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from autosim.experiment_system.executor import Executor, Job
from autosim.research.common import atomic_json, digest, freeze_files, now, read_json
from autosim.research.ledger import SeedLedger
from autosim.research.registry import TASK_IDS, load_task
from autosim.research.runtime import Runtime
from autosim.robosyn_data import evaluation_seed_bank
from .paired_initialization_audit import evaluation_artifact_directory
from .queue_manifest import versioned_manifest


TERMINAL_INTEGRATION = {"integration_gate_reached", "requires_capability_review",
                        "queue_wait_budget_exhausted", "failed", "blocked"}


def model_paths(workspace: Path, previous_output: Path) -> dict[str, Path]:
    status = read_json(previous_output / "asset_status.json").get("tasks", {})
    result = {}
    for task in TASK_IDS:
        local = workspace / "AutoSimSOTA/RoboSynChallenge/checkpoints" / f"ACT_sim_{task}"
        downloaded = Path(status.get(task, {}).get("model", {}).get("path", ""))
        candidate = local if (local / "model.safetensors").is_file() else downloaded
        for name in ("model.safetensors", "config.json"):
            if not (candidate / name).is_file():
                raise FileNotFoundError(f"released ACT checkpoint incomplete for {task}: {candidate / name}")
        result[task] = candidate.absolute()
    return result


def integration_terminal(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    status = str(read_json(path).get("status"))
    return status in TERMINAL_INTEGRATION, status


def integration_ready(path: Path) -> tuple[bool, bool, str]:
    """Return (terminal, successful, status); terminal failures fail closed."""
    terminal, status = integration_terminal(path)
    return terminal, status == "integration_gate_reached", status


def validate_baseline_metrics(metrics: dict, *, task: str, episodes: int,
                              master_seed: int, max_episode_steps: int,
                              checkpoint: Path) -> dict:
    """Validate the actual rollout contract, not just the worker exit status."""
    config = metrics.get("config", {})
    expected_seeds = evaluation_seed_bank(master_seed, episodes)
    rows = metrics.get("episodes", [])
    actual_seeds = [int(row.get("episode_seed", -1)) for row in rows]
    expected_config = {
        "task": task,
        "setting": "random",
        "episode_count": episodes,
        "timeout_action_steps": max_episode_steps,
        "seed": master_seed,
    }
    actual_config = {key: config.get(key) for key in expected_config}
    if metrics.get("execution_mode") != "real_simulation":
        raise ValueError("baseline did not certify real simulation")
    if metrics.get("purpose") != "development":
        raise ValueError("baseline used a non-development evaluation purpose")
    if metrics.get("harness") != "official_control_loop_observation_only_rpc_v1":
        raise ValueError("baseline used an unexpected evaluation harness")
    if actual_config != expected_config:
        raise ValueError(f"baseline evaluation config mismatch: {actual_config}")
    if Path(config.get("checkpoint_path", "")).resolve() != checkpoint.resolve():
        raise ValueError("baseline evaluated a different checkpoint")
    if actual_seeds != expected_seeds:
        raise ValueError("baseline episode seeds differ from the pre-reserved ordered bank")
    if [row.get("episode_index") for row in rows] != list(range(episodes)):
        raise ValueError("baseline episode indices are incomplete or reordered")
    if any(not isinstance(row.get("success"), bool) or
           not 0 <= int(row.get("action_steps", -1)) <= max_episode_steps for row in rows):
        raise ValueError("baseline episode outcomes violate the result contract")
    success_count = sum(row["success"] for row in rows)
    summary = metrics.get("summary", {})
    if (summary.get("episode_count") != episodes or summary.get("success_count") != success_count or
            abs(float(summary.get("success_rate", -1)) - success_count / episodes) > 1e-12):
        raise ValueError("baseline summary disagrees with episode records")
    return {"ordered_seed_bank_verified": True, "episode_count": episodes,
            "first_seed": expected_seeds[0], "last_seed": expected_seeds[-1],
            "success_count": success_count, "max_episode_steps": max_episode_steps}


def validate_baseline_sidecars(artifact_directory: Path, *, task: str, episodes: int,
                               master_seed: int, checkpoint: Path) -> dict:
    """Corroborate metrics with reset records, telemetry, and frozen protocol."""
    expected_seeds = evaluation_seed_bank(master_seed, episodes)
    paths = {
        "protocol": artifact_directory / "protocol.json",
        "initializations": artifact_directory / "initializations.jsonl",
        "telemetry": artifact_directory / "telemetry.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError(f"baseline evaluation sidecars missing: {missing}")
    protocol = read_json(paths["protocol"])
    expected_protocol = {"seed": master_seed, "episodes": episodes,
                         "purpose": "development", "policy": "act"}
    if {key: protocol.get(key) for key in expected_protocol} != expected_protocol:
        raise ValueError("baseline protocol differs from the requested seed bank or purpose")
    if protocol.get("task", {}).get("name") != task:
        raise ValueError("baseline protocol task mismatch")
    if protocol.get("checkpoint_sha256") != digest(checkpoint / "model.safetensors"):
        raise ValueError("baseline protocol checkpoint hash mismatch")
    initializations = [json.loads(line) for line in paths["initializations"].read_text().splitlines()]
    initialization_seeds = [int(row.get("seed", -1)) for row in initializations]
    if initialization_seeds != expected_seeds or any(
            row.get("event") != "reset" or not row.get("allowed_observation_sha256")
            for row in initializations):
        raise ValueError("baseline reset records differ from the ordered seed bank")
    telemetry = [json.loads(line) for line in paths["telemetry"].read_text().splitlines()]
    step_zero = [row for row in telemetry if row.get("step") == 0]
    if [int(row.get("seed", -1)) for row in step_zero] != expected_seeds:
        raise ValueError("baseline step-0 telemetry differs from the ordered seed bank")
    if any(row.get("task") != task or row.get("missing") for row in step_zero):
        raise ValueError("baseline step-0 telemetry has a task mismatch or missing entities")
    return {"reset_records_verified": episodes, "step_zero_telemetry_verified": episodes,
            "protocol_verified": True}


def validate_nested_evaluation_process(process_path: Path) -> dict:
    """Require the simulator subprocess behind a successful outer worker."""
    if not process_path.is_file() or not process_path.stat().st_size:
        raise FileNotFoundError(f"nested evaluation process receipt missing: {process_path}")
    process = read_json(process_path)
    if process.get("status") != "completed" or process.get("returncode") != 0:
        raise ValueError("nested evaluation simulator process did not complete successfully")
    if not isinstance(process.get("command"), list) or "autosim.research.evaluation" not in process["command"]:
        raise ValueError("nested evaluation process command is not the frozen evaluator")
    return {"nested_evaluation_process_verified": True,
            "nested_evaluation_elapsed_seconds": process.get("elapsed_seconds")}


def reserve_development_banks(previous_output: Path, episodes: int) -> dict[str, dict]:
    """Atomically reserve every official-baseline development bank globally."""
    ledger = SeedLedger(previous_output / "seeds.sqlite")
    result = {}
    try:
        for index, task in enumerate(TASK_IDS):
            master = 95_000_000 + index * 100_000
            seeds = ledger.reserve(task, "robosyn_milestone_20260906:official_act_development",
                                   "development", master, episodes)
            result[task] = {"master_seed": master, "episode_count": episodes,
                            "first_seed": seeds[0], "last_seed": seeds[-1]}
    finally:
        ledger.close()
    return result


def task_sources(workspace: Path, task: str, checkpoint: Path) -> dict[str, str]:
    train_repo = workspace / "AutoSimSOTA/RoboSynChallenge"
    eval_repo = workspace / "AutoSimSOTA/RoboSynChallenge_eval_clean"
    local = Path(__file__).resolve()
    paths = [local, local.parents[1] / "experiment_system/executor.py",
             local.parents[1] / "experiment_system/worker.py",
             local.with_name("queue_manifest.py"), local.with_name("paired_initialization_audit.py"),
             local.parents[1] / "robosyn_data.py",
             local.parents[1] / "research/common.py", local.parents[1] / "research/runtime.py",
             local.parents[1] / "research/evaluation.py", local.parents[1] / "research/policy_rpc.py",
             local.parents[1] / "research/registry.py", checkpoint / "model.safetensors",
             checkpoint / "config.json"]
    for repo in (train_repo, eval_repo):
        paths += [repo / f"configs/{task}/random/gym_config.json",
                  repo / f"configs/{task}/action_config.json",
                  repo / "scripts/eval_policy.py", repo / "policy/act/deploy_policy.py",
                  repo / f"robosynchallenge/tasks/{task}/{task}.py"]
    return freeze_files(paths)


def baseline_job(workspace: Path, root: Path, task: str, checkpoint: Path,
                 episodes: int, master_seed: int) -> Job:
    python = workspace / "AutoSimSOTA/.venv/bin/python"
    command = [str(python), "-m", "autosim.experiment_validation.robosyn_milestone", "evaluate",
               "--workspace", str(workspace), "--attempt", "{attempt}", "--task", task,
               "--checkpoint", str(checkpoint), "--episodes", str(episodes),
               "--master-seed", str(master_seed)]
    runtime = Runtime(workspace, root)
    allowed = {"PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "EMBODICHAIN_DATA_ROOT",
               "XDG_CACHE_HOME", "MPLCONFIGDIR", "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "PYTHONFAULTHANDLER"}
    environment = {k: v for k, v in runtime.environment().items() if k in allowed}
    return Job(name=f"official_act_{task}_development_{episodes}", stage="evaluate",
               command=command, cwd=str(workspace / "AutoSimSOTA/RoboSynChallenge_eval_clean"),
               outputs={"result.json": "result"}, sources=task_sources(workspace, task, checkpoint),
               environment=environment, timeout_seconds=max(3600, episodes * 90),
               budget_seconds=max(3600, episodes * 90), max_attempts=1, gpu="0")


def evaluate(workspace: Path, attempt: Path, task: str, checkpoint: Path,
             episodes: int, master_seed: int):
    attempt.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(workspace, attempt, gpu="0")
    spec = load_task(runtime.repo, task)
    metrics = runtime.evaluate(spec, checkpoint, attempt / "evaluation", episodes=episodes,
                               master_seed=master_seed, purpose="development", policy="act")
    validated_contract = validate_baseline_metrics(
        metrics, task=task, episodes=episodes, master_seed=master_seed,
        max_episode_steps=spec.max_episode_steps, checkpoint=checkpoint)
    artifact_directory = evaluation_artifact_directory(attempt / "evaluation", metrics)
    validated_contract.update(validate_baseline_sidecars(
        artifact_directory, task=task, episodes=episodes,
        master_seed=master_seed, checkpoint=checkpoint))
    raw_metrics_path = artifact_directory / "evaluation_metrics.json"
    raw_metrics = read_json(raw_metrics_path)
    if any(raw_metrics.get(key) != metrics.get(key) for key in ("config", "episodes", "summary")):
        raise ValueError("published baseline metrics differ from the raw successful attempt")
    nested_process_path = artifact_directory / "process/process.json"
    validated_contract.update(validate_nested_evaluation_process(nested_process_path))
    verified = {}
    for path in (attempt / "evaluation/evaluation_metrics.json",
                 attempt / "evaluation/evaluation_request.json",
                 raw_metrics_path,
                 nested_process_path,
                 artifact_directory / "protocol.json",
                 artifact_directory / "initializations.jsonl",
                 artifact_directory / "telemetry.jsonl"):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"required baseline evidence missing: {path}")
        verified[str(path.absolute())] = digest(path)
    result = {"status": "completed", "benchmark": "RoboSynChallenge", "task": task,
              "policy": "official_released_ACT_checkpoint", "checkpoint": str(checkpoint),
              "episodes": episodes, "master_seed": master_seed, "purpose": "development",
              "success_count": int(metrics["summary"]["success_count"]),
              "success_rate": float(metrics["summary"]["success_rate"]),
              "validated_contract": validated_contract,
              "official_leaderboard_result": False, "sota_claim": False,
              "verified_files": verified}
    atomic_json(attempt / "result.json", result)


def snapshot(workspace: Path, root: Path, previous_output: Path, episodes: int = 100):
    executor = Executor(root / "jobs")
    admission_path = workspace / "autosim/output/experiment_system_20260906/admission_summary.json"
    admission = {r["task"]: r for r in read_json(admission_path)["tasks"]}
    assets = read_json(previous_output / "asset_status.json").get("tasks", {})
    rows = []
    for task in TASK_IDS:
        job_name = f"official_act_{task}_development_{episodes}"
        commit = root / "jobs" / job_name / "commit.json"
        baseline = None
        if commit.is_file():
            # Revalidate the immutable job signature, pinned sources, receipt,
            # result, and every result-referenced evaluation artifact.
            record = executor.committed(job_name)
            baseline = read_json(commit.parent / record["attempt"] / "result.json")
            if (baseline.get("task") != task or baseline.get("episodes") != episodes or
                    baseline.get("validated_contract", {}).get("ordered_seed_bank_verified") is not True):
                raise ValueError(f"invalid committed baseline result for {task}")
        pilot_path = previous_output / "pilot_research/train_seed_1000" / task / "state.json"
        pilot = read_json(pilot_path) if pilot_path.is_file() else {}
        model = assets.get(task, {}).get("model", {})
        data = assets.get(task, {}).get("dataset", {})
        rows.append({"task": task, "official_model_available": model.get("status") == "completed" or
                     task == "click_bell", "official_dataset_available": data.get("status") == "completed" or
                     task == "click_bell", "successful_self_collected_episodes_audited":
                     int(admission.get(task, {}).get("episodes", 0)) if admission.get(task, {}).get("passed") else 0,
                     "automatic_collection_status": "validated_small_batch" if admission.get(task, {}).get("passed")
                     else "no_successful_collection", "pilot_status": pilot.get("status", "not_started"),
                     "pilot_improvement": pilot.get("improvement", "not_established"),
                     "official_act_development_baseline": baseline,
                     "sota_established": False})
    completed = sum(r["official_act_development_baseline"] is not None for r in rows)
    value = {"updated_at": now(), "benchmark": "RoboSynChallenge", "task_count": len(rows),
             "official_assets_complete": all(r["official_model_available"] and r["official_dataset_available"] for r in rows),
             "self_collection_validated_tasks": sum(r["automatic_collection_status"] == "validated_small_batch" for r in rows),
             "official_baselines_completed": completed, "all_task_sota": False,
             "next_milestone_complete": False, "tasks": rows}
    atomic_json(root / "task_scoreboard.json", value)
    lines = ["# RoboSyn 十任务里程碑账本", "", f"更新时间：{value['updated_at']}。本表中的评测是本地开发集复测，不是官方榜单成绩。", "",
             "| 任务 | 官方模型/数据 | 合格自采 | 官方 ACT 新开发集复测 | Pilot 状态 | 已证明 SOTA |",
             "| --- | --- | ---: | --- | --- | --- |"]
    for row in rows:
        baseline = row["official_act_development_baseline"]
        score = "待评测" if not baseline else f"{baseline['success_count']}/{baseline['episodes']} = {baseline['success_rate']:.1%}"
        lines.append(f"| {row['task']} | {'完整' if row['official_model_available'] and row['official_dataset_available'] else '缺失'} | "
                     f"{row['successful_self_collected_episodes_audited']} | {score} | {row['pilot_status']} | 否 |")
    lines += ["", f"当前覆盖：官方资产 {sum(r['official_model_available'] and r['official_dataset_available'] for r in rows)}/10；"
              f"小批量自采 {value['self_collection_validated_tasks']}/10；统一官方 ACT 开发集复测 {completed}/10。",
              "", "注意：小批量自采通过结构检查不代表数据足以训练有效策略；Pilot 完成也不代表改进成立。"]
    (root / "task_scoreboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def queue(workspace: Path, root: Path, previous_output: Path, integration_state: Path,
          episodes: int, max_hours: float, poll_seconds: float):
    if not 10 <= episodes <= 500 or not 0 < max_hours <= 240 or not 1 <= poll_seconds <= 60:
        raise ValueError("invalid bounded queue parameters")
    root.mkdir(parents=True, exist_ok=True)
    models = model_paths(workspace, previous_output)
    seed_banks = reserve_development_banks(previous_output, episodes)
    local = Path(__file__).resolve()
    manifest = {"benchmark": "RoboSynChallenge", "tasks": list(TASK_IDS),
                "episodes_per_task": episodes, "purpose": "development", "policy": "official_released_ACT_checkpoint",
                "master_seed_base": 95_000_000, "integration_gate": str(integration_state),
                "max_hours": max_hours, "final_test_used": False, "sota_claim": False,
                "seed_ledger": str((previous_output / "seeds.sqlite").absolute()),
                "seed_banks": seed_banks,
                "queue_sources": freeze_files([local, local.with_name("queue_manifest.py"),
                                                 local.with_name("paired_initialization_audit.py"),
                                                 local.parents[1] / "robosyn_data.py"]),
                "model_files": {task: {name: digest(path / name) for name in ("model.safetensors", "config.json")}
                                for task, path in models.items()}}
    versioned_manifest(root, manifest)
    executor = Executor(root / "jobs")
    state_path = root / "state.json"
    prior = read_json(state_path) if state_path.is_file() else {}
    if prior.get("status") in {"baseline_table_complete", "requires_review", "queue_wait_budget_exhausted"}:
        return prior
    state = {**prior, "status": "running", "started_at": prior.get("started_at", now()),
             "completed": int(prior.get("completed", 0)), "total": len(TASK_IDS),
             "formal_milestone_complete": False,
             "deadline_epoch": float(prior.get("deadline_epoch", time.time() + max_hours * 3600))}
    if prior:
        state.update(resume_count=int(prior.get("resume_count", 0)) + 1, resumed_at=now())
    while time.time() < state["deadline_epoch"]:
        terminal, integration_succeeded, integration_status = integration_ready(integration_state)
        if not terminal:
            state.update(stage="waiting_integration_gate", integration_status=integration_status, updated_at=now())
            atomic_json(state_path, state)
            snapshot(workspace, root, previous_output, episodes)
            time.sleep(poll_seconds)
            continue
        if not integration_succeeded:
            state.update(status="requires_review", stage="integration_gate_not_reached",
                         integration_status=integration_status, updated_at=now())
            atomic_json(state_path, state)
            snapshot(workspace, root, previous_output, episodes)
            return state
        for index, task in enumerate(TASK_IDS):
            job = baseline_job(workspace, root, task, models[task], episodes, 95_000_000 + index * 100_000)
            result = executor.run(job)
            state.update(stage=f"official_act_development:{task}", current_result=result, updated_at=now(),
                         integration_status=integration_status)
            atomic_json(state_path, state)
            snapshot(workspace, root, previous_output, episodes)
            if result.get("status") == "waiting_lease":
                time.sleep(poll_seconds)
                break
            if result.get("status") != "committed":
                state.update(status="requires_review", failed_task=task, failure=result, updated_at=now())
                atomic_json(state_path, state)
                return state
        else:
            state.update(status="baseline_table_complete", completed=len(TASK_IDS),
                         remaining=["repair two collection gaps", "multi-task effective policies",
                                    "three real feedback loops", "fair system comparison"], updated_at=now())
            atomic_json(state_path, state)
            snapshot(workspace, root, previous_output, episodes)
            return state
    state.update(status="queue_wait_budget_exhausted", updated_at=now())
    atomic_json(state_path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "queue", "snapshot"):
        p = sub.add_parser(name)
        p.add_argument("--workspace", type=Path, required=True)
        p.add_argument("--root", type=Path)
        p.add_argument("--previous-output", type=Path)
        if name == "evaluate":
            p.add_argument("--attempt", type=Path, required=True)
            p.add_argument("--task", choices=TASK_IDS, required=True)
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--episodes", type=int, required=True)
            p.add_argument("--master-seed", type=int, required=True)
        elif name == "queue":
            p.add_argument("--integration-state", type=Path, required=True)
            p.add_argument("--episodes", type=int, default=100)
            p.add_argument("--max-hours", type=float, default=120)
            p.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.workspace.absolute(), args.attempt.absolute(), args.task,
                 args.checkpoint.absolute(), args.episodes, args.master_seed)
    else:
        root = (args.root or args.workspace / "autosim/output/robosyn_milestone_20260906").absolute()
        previous = (args.previous_output or args.workspace / "autosim/output/robosyn_general_20260905").absolute()
        if args.command == "snapshot":
            snapshot(args.workspace.absolute(), root, previous)
        else:
            queue(args.workspace.absolute(), root, previous, args.integration_state.absolute(),
                  args.episodes, args.max_hours, args.poll_seconds)
