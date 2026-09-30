"""Queue and summarize a fair three-task AutoResearch comparison.

The frozen research controller gives the automatic and random-control arms the
same collection, screening-training, long-training, and confirmation budgets.
This additive wrapper waits for earlier queues, resumes two representative
tasks, and reports paired confirmation results.  It never opens the final-test
seed bank and never makes an official-leaderboard or SOTA claim.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from autosim.research.common import assert_frozen, atomic_json, digest, freeze_files, now, read_json
from autosim.research.controller import ResearchConfig, ResearchController
from autosim.research.ledger import compare, holm
from autosim.research.runtime import Runtime
from autosim.robosyn_data import evaluation_seed_bank
from .paired_initialization_audit import audit_bank, evaluation_artifact_directory
from .queue_manifest import versioned_manifest


TASKS = ("click_bell", "water_pouring", "drawer_open_place")
QUEUED_TASKS = TASKS[1:]


def one_sided_exact(wins: int, losses: int) -> float:
    """Exact one-sided sign-test probability for paired discordant outcomes."""
    n = wins + losses
    if not n:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def gate_status(click_state: Path, collection_gap_state: Path) -> tuple[str, dict]:
    click = read_json(click_state) if click_state.is_file() else {}
    gap = read_json(collection_gap_state) if collection_gap_state.is_file() else {}
    detail = {"click_bell": click.get("status", "missing"),
              "collection_gap": gap.get("status", "missing")}
    if click.get("status") == "failed":
        return "requires_review", detail
    if gap.get("status") in {"blocked_by_baseline_gate", "requires_review", "queue_wait_budget_exhausted"}:
        return "requires_review", detail
    if click.get("status") != "completed_development":
        return "waiting_click_bell", detail
    if gap.get("status") != "collection_gap_handled":
        return "waiting_collection_gap", detail
    return "ready", detail


def source_manifest() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), package / "research/controller.py",
             package / "research/analysis.py", package / "research/proposals.py",
             package / "research/ledger.py", package / "research/runtime.py",
             package / "research/evaluation.py", package / "research/collection_worker.py",
             Path(__file__).with_name("paired_initialization_audit.py"),
             Path(__file__).with_name("queue_manifest.py")]
    return freeze_files(paths)


def _flag(command: list, name: str) -> str:
    matches = [index for index, value in enumerate(command) if value == name]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise ValueError(f"confirmation process has missing or repeated flag: {name}")
    return str(command[matches[0] + 1])


def _confirmation_bundle(directory: Path, checkpoint: Path, *, task: str,
                         episodes: int, master_seed: int) -> dict:
    """Bind one published confirmation result to its checkpoint and raw process."""
    directory = Path(directory).absolute().resolve()
    checkpoint = Path(checkpoint).absolute().resolve()
    published = directory / "evaluation_metrics.json"
    request_path = directory / "evaluation_request.json"
    for path in (published, request_path, checkpoint / "model.safetensors", checkpoint / "config.json"):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"confirmation identity evidence missing: {path}")
    metrics = read_json(published)
    artifact = evaluation_artifact_directory(directory, metrics)
    startup_attempt = metrics.get("startup_attempt_count")
    expected_artifact = directory if startup_attempt == 1 else directory / f"startup_attempt_{startup_attempt}"
    if startup_attempt not in {1, 2, 3} or artifact != expected_artifact:
        raise ValueError("confirmation startup-attempt provenance is inconsistent")
    protocol_path = artifact / "protocol.json"
    process_path = artifact / "process/process.json"
    raw_metrics_path = artifact / "evaluation_metrics.json"
    for path in (raw_metrics_path, protocol_path, process_path,
                 artifact / "initializations.jsonl", artifact / "telemetry.jsonl"):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"confirmation raw evidence missing: {path}")
    raw_metrics = read_json(raw_metrics_path)
    if any(raw_metrics.get(key) != metrics.get(key) for key in ("config", "episodes", "summary")):
        raise ValueError("published confirmation differs from the raw successful attempt")
    protocol = read_json(protocol_path)
    request = read_json(request_path)
    process = read_json(process_path)
    weight_hash = digest(checkpoint / "model.safetensors")
    config_hash = digest(checkpoint / "config.json")
    expected_seeds = evaluation_seed_bank(master_seed, episodes)
    rows = metrics.get("episodes", [])
    config = metrics.get("config", {})
    if metrics.get("execution_mode") != "real_simulation" or metrics.get("purpose") != "confirmation":
        raise ValueError("confirmation is not a certified real-simulation confirmation result")
    if metrics.get("harness") != "official_control_loop_observation_only_rpc_v1":
        raise ValueError("confirmation used an unexpected harness")
    expected_config = {"task": task, "setting": "random", "episode_count": episodes,
                       "timeout_action_steps": 361, "seed": master_seed}
    if {key: config.get(key) for key in expected_config} != expected_config:
        raise ValueError("confirmation metrics config differs from the frozen study")
    if Path(config.get("checkpoint_path", "")).resolve() != checkpoint:
        raise ValueError("confirmation metrics name a different checkpoint")
    if [row.get("episode_index") for row in rows] != list(range(episodes)) or [
            int(row.get("episode_seed", -1)) for row in rows] != expected_seeds:
        raise ValueError("confirmation episodes differ from the ordered reserved seed bank")
    if any(not isinstance(row.get("success"), bool) or
           not 0 <= int(row.get("action_steps", -1)) <= 361 for row in rows):
        raise ValueError("confirmation episode outcomes violate the result contract")
    success_count = sum(row["success"] for row in rows)
    summary = metrics.get("summary", {})
    if (summary.get("episode_count") != episodes or summary.get("success_count") != success_count or
            abs(float(summary.get("success_rate", -1)) - success_count / episodes) > 1e-12):
        raise ValueError("confirmation summary disagrees with episode records")
    expected_request = {"checkpoint": str(checkpoint), "weight_sha256": weight_hash,
                        "model_config_sha256": config_hash, "episodes": episodes,
                        "master_seed": master_seed, "purpose": "confirmation", "policy": "act"}
    actual_request = {**{key: request.get(key) for key in expected_request},
                      "checkpoint": str(Path(request.get("checkpoint", "")).resolve())}
    if actual_request != expected_request:
        raise ValueError("confirmation evaluation request does not bind the expected checkpoint/protocol")
    expected_protocol = {"purpose": "confirmation", "checkpoint_sha256": weight_hash,
                         "seed": master_seed, "episodes": episodes, "policy": "act"}
    if ({key: protocol.get(key) for key in expected_protocol} != expected_protocol or
            protocol.get("task", {}).get("name") != task):
        raise ValueError("confirmation protocol does not bind the expected checkpoint/seed bank")
    frozen_files = protocol.get("frozen_files")
    if not isinstance(frozen_files, dict) or not frozen_files:
        raise ValueError("confirmation protocol has no frozen evaluator evidence")
    assert_frozen(frozen_files)
    command = process.get("command", [])
    if process.get("status") != "completed" or process.get("returncode") != 0:
        raise ValueError("confirmation simulator subprocess did not complete successfully")
    expected_flags = {"--task": task, "--checkpoint": str(checkpoint), "--episodes": str(episodes),
                      "--seed": str(master_seed), "--purpose": "confirmation", "--policy": "act",
                      "--output": str(artifact)}
    actual_flags = {name: _flag(command, name) for name in expected_flags}
    actual_flags["--checkpoint"] = str(Path(actual_flags["--checkpoint"]).resolve())
    actual_flags["--output"] = str(Path(actual_flags["--output"]).resolve())
    if actual_flags != expected_flags:
        raise ValueError("confirmation simulator command differs from the published request")
    evidence_paths = [published, request_path, raw_metrics_path, protocol_path, process_path,
                      artifact / "initializations.jsonl", artifact / "telemetry.jsonl",
                      checkpoint / "model.safetensors", checkpoint / "config.json"]
    return {"metrics": metrics, "artifact_directory": str(artifact),
            "checkpoint": str(checkpoint), "checkpoint_sha256": weight_hash,
            "identity_verified": True,
            "evidence": {str(path.absolute()): digest(path) for path in evidence_paths}}


def _task_comparison(research_root: Path, task: str, report_root: Path | None = None,
                     *, confirmation_episodes: int = 100, train_seed: int = 1000) -> dict:
    state_path = research_root / task / "state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    row = {"task": task, "status": state.get("status", "missing"),
           "final_test_used": False, "official_leaderboard_result": False,
           "sota_established": False}
    if state.get("status") != "completed_development":
        return row
    selected = state["selected"]
    task_root = research_root / task
    task_index = TASKS.index(task)
    master_seed = 80_000_000 + task_index * 1_000_000 + (train_seed - 1000) * 10_000 + 2
    expected_dirs = {"controlled_baseline": task_root / "evaluations/controlled_baseline_confirmation",
                     "auto_selected": task_root / "evaluations/auto_selected_confirmation",
                     "random_selected": task_root / "evaluations/random_selected_confirmation"}
    if (Path(selected["auto"]["evaluation"]).resolve() != expected_dirs["auto_selected"].resolve() or
            Path(selected["random"]["evaluation"]).resolve() != expected_dirs["random_selected"].resolve()):
        raise ValueError(f"{task} state points outside the fixed confirmation output directories")
    bundles = {
        "controlled_baseline": _confirmation_bundle(
            expected_dirs["controlled_baseline"], Path(state["baseline_checkpoint"]), task=task,
            episodes=confirmation_episodes, master_seed=master_seed),
        "auto_selected": _confirmation_bundle(
            expected_dirs["auto_selected"], Path(selected["auto"]["checkpoint"]), task=task,
            episodes=confirmation_episodes, master_seed=master_seed),
        "random_selected": _confirmation_bundle(
            expected_dirs["random_selected"], Path(selected["random"]["checkpoint"]), task=task,
            episodes=confirmation_episodes, master_seed=master_seed),
    }
    auto_eval = bundles["auto_selected"]["metrics"]
    random_eval = bundles["random_selected"]["metrics"]
    baseline_eval = bundles["controlled_baseline"]["metrics"]
    paired = compare(auto_eval, random_eval)
    auto_vs_baseline = compare(auto_eval, baseline_eval)
    auto_summary = auto_eval["summary"]
    random_summary = random_eval["summary"]
    initialization = audit_bank({
        "controlled_baseline": expected_dirs["controlled_baseline"],
        "auto_selected": Path(selected["auto"]["evaluation"]),
        "random_selected": Path(selected["random"]["evaluation"]),
    })
    initialization_path = None
    if report_root is not None:
        initialization_path = report_root / "initialization_audits" / f"{task}_confirmation.json"
        initialization["audit_source_sha256"] = digest(Path(__file__).with_name("paired_initialization_audit.py"))
        atomic_json(initialization_path, initialization)
    row.update(
        episodes=paired["episodes"],
        auto_success_count=auto_summary["success_count"],
        auto_success_rate=auto_summary["success_rate"],
        random_success_count=random_summary["success_count"],
        random_success_rate=random_summary["success_rate"],
        auto_vs_random=paired,
        auto_vs_fixed_training=auto_vs_baseline,
        automatic_collection_validated=state.get("automatic_collection_validated", False),
        completed_rounds=state.get("completed_rounds"),
        research_mode=state.get("research_mode"),
        confirmation_identity_verified=all(bundle["identity_verified"] for bundle in bundles.values()),
        confirmation_checkpoints={name: {"path": bundle["checkpoint"],
                                         "sha256": bundle["checkpoint_sha256"]}
                                  for name, bundle in bundles.items()},
        reset_pairing_evidence={"status": initialization["status"],
                                "summary": initialization["summary"],
                                "interpretation": initialization["interpretation"],
                                "artifact": str(initialization_path) if initialization_path else None},
        evidence={str(state_path.absolute()): digest(state_path),
                  **{path: value for bundle in bundles.values() for path, value in bundle["evidence"].items()},
                  **({str(initialization_path.absolute()): digest(initialization_path)}
                     if initialization_path else {})},
    )
    return row


def write_report(root: Path, research_root: Path) -> dict:
    config = read_json(research_root / "research_config.json")
    if config.get("pilot") is not False:
        raise ValueError("three-task comparison requires the non-pilot research configuration")
    rows = [_task_comparison(research_root, task, root,
                             confirmation_episodes=int(config["confirmation_episodes"]),
                             train_seed=int(config["train_seed"])) for task in TASKS]
    completed = [row for row in rows if row["status"] == "completed_development"]
    wins = sum(row["auto_vs_random"]["candidate_only_successes"] for row in completed)
    losses = sum(row["auto_vs_random"]["baseline_only_successes"] for row in completed)
    p_values = {row["task"]: row["auto_vs_random"]["one_sided_p_value"] for row in completed}
    corrected = holm(p_values) if len(p_values) == len(TASKS) else {}
    complete = len(completed) == len(TASKS)
    aggregate = {"paired_episodes": sum(row["episodes"] for row in completed),
                 "auto_only_successes": wins, "random_only_successes": losses,
                 "success_delta": ((wins - losses) / sum(row["episodes"] for row in completed)) if completed else None,
                 "one_sided_exact_p_value": one_sided_exact(wins, losses) if completed else None}
    pairing_evidence_complete = complete and all(
        row.get("reset_pairing_evidence", {}).get("status") ==
        "measured_reset_variation_no_equivalence_claim" for row in completed)
    improvement = (complete and pairing_evidence_complete and wins > losses and
                   aggregate["one_sided_exact_p_value"] < 0.05)
    value = {"updated_at": now(), "kind": "three_task_autoresearch_development_comparison",
             "tasks": rows, "aggregate_auto_vs_random": aggregate,
             "holm_task_level_rejections_at_0_05": corrected,
             "study_execution_complete": complete,
             "reset_pairing_evidence_complete": pairing_evidence_complete,
             "automatic_decision_beats_random_control_established": improvement,
             "final_test_used": False, "official_leaderboard_result": False,
             "sota_established": False,
             "limitations": ["single train seed; replication remains required",
                             "confirmation banks are held-out development banks, not official hidden evaluation",
                             "aggregate exact test pools discordant episodes across heterogeneous tasks",
                             "same seeds match recorded scene randomization but reset-time physics may vary; numeric audits are attached"]}
    atomic_json(root / "comparison.json", value)
    lines = ["# 三任务 AutoResearch 等预算对照", "",
             "所有结果均为本地、预留确认开发种子上的配对评测，不是官方隐藏集成绩，也不构成 SOTA 声明。", "",
             "| 任务 | 自动组 | 等预算随机补采+固定训练组 | 自动-随机 | 配对单侧 p | 自动-固定官方数据训练 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["status"] != "completed_development":
            lines.append(f"| {row['task']} | 待完成 | 待完成 | — | — | — |")
            continue
        avf = row["auto_vs_fixed_training"]
        lines.append(f"| {row['task']} | {row['auto_success_count']}/{row['episodes']} ({row['auto_success_rate']:.1%}) | "
                     f"{row['random_success_count']}/{row['episodes']} ({row['random_success_rate']:.1%}) | "
                     f"{row['auto_vs_random']['success_delta']:+.1%} | {row['auto_vs_random']['one_sided_p_value']:.4f} | "
                     f"{avf['success_delta']:+.1%} |")
    lines += ["", f"执行完成：{len(completed)}/3；合并配对样本：{aggregate['paired_episodes']}；"
              f"自动独占成功/随机独占成功：{wins}/{losses}。",
              f"确认集reset-time数值审计完整：{'yes' if pairing_evidence_complete else 'no'}。"
              "该审计量化同seed重置差异，不事后设定等价容差，也不声称完整隐状态位级确定。",
              "", "判定门槛：三任务全部完成，且合并配对比较自动独占成功数更高、单侧精确检验 p<0.05。"
              "即使达到该门槛，也只表示首轮开发证据成立；独立训练种子复现和官方评测仍是后续里程碑。"]
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def queue(workspace: Path, root: Path, previous_output: Path, click_state: Path,
          collection_gap_state: Path, max_hours: float, poll_seconds: float):
    if not 0 < max_hours <= 336 or not 1 <= poll_seconds <= 60:
        raise ValueError("invalid bounded queue settings")
    root.mkdir(parents=True, exist_ok=True)
    config = ResearchConfig(rounds=2, screen_steps=20000, final_steps=80000,
                            development_episodes=40, confirmation_episodes=100,
                            collect_per_round=50, train_seed=1000, hours_per_task=40,
                            allow_llm=False, pilot=False)
    manifest = {"tasks": list(TASKS), "queued_tasks": list(QUEUED_TASKS),
                "comparison": "automatic_decision_vs_random_collection_fixed_recipe",
                "research_config": config.__dict__, "same_confirmation_seeds_within_task": True,
                "same_total_arm_budgets": True, "click_state": str(click_state),
                "collection_gap_state": str(collection_gap_state), "sources": source_manifest(),
                "final_test_used": False, "sota_claim": False}
    versioned_manifest(root, manifest)
    state_path = root / "state.json"
    prior = read_json(state_path) if state_path.is_file() else {}
    if prior.get("status") in {"three_task_comparison_complete", "requires_review", "queue_wait_budget_exhausted"}:
        return prior
    state = {**prior, "status": "running", "started_at": prior.get("started_at", now()),
             "formal_milestone_complete": False,
             "deadline_epoch": float(prior.get("deadline_epoch", time.time() + max_hours * 3600))}
    if prior:
        state.update(resume_count=int(prior.get("resume_count", 0)) + 1, resumed_at=now())
    while time.time() < state["deadline_epoch"]:
        gate, detail = gate_status(click_state, collection_gap_state)
        if gate == "requires_review":
            state.update(status=gate, gate_detail=detail, updated_at=now())
            atomic_json(state_path, state)
            return state
        if gate != "ready":
            state.update(stage=gate, gate_detail=detail, updated_at=now())
            atomic_json(state_path, state)
            write_report(root, previous_output / "research/train_seed_1000")
            time.sleep(poll_seconds)
            continue
        state.update(stage="run_research:water_pouring,drawer_open_place", gate_detail=detail, updated_at=now())
        atomic_json(state_path, state)
        runtime = Runtime(workspace, previous_output, gpu="0")
        results = ResearchController(runtime, config).run(list(QUEUED_TASKS))
        report = write_report(root, previous_output / "research/train_seed_1000")
        failed = {task: row for task, row in results.items() if row.get("status") != "completed_development"}
        if failed:
            state.update(status="requires_review", stage="research_incomplete", task_statuses={k: v.get("status") for k, v in results.items()}, updated_at=now())
        else:
            state.update(status="three_task_comparison_complete", stage="awaiting_replication_or_final_test",
                         formal_milestone_complete=report["automatic_decision_beats_random_control_established"],
                         automatic_decision_beats_random_control_established=report["automatic_decision_beats_random_control_established"],
                         updated_at=now())
        atomic_json(state_path, state)
        return state
    state.update(status="queue_wait_budget_exhausted", updated_at=now())
    atomic_json(state_path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("queue", "report"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous-output", type=Path, required=True)
    parser.add_argument("--click-state", type=Path)
    parser.add_argument("--collection-gap-state", type=Path)
    parser.add_argument("--max-hours", type=float, default=240)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.command == "report":
        write_report(args.root.absolute(), args.previous_output.absolute() / "research/train_seed_1000")
    else:
        if args.click_state is None or args.collection_gap_state is None:
            parser.error("queue requires --click-state and --collection-gap-state")
        queue(args.workspace.absolute(), args.root.absolute(), args.previous_output.absolute(),
              args.click_state.absolute(), args.collection_gap_state.absolute(),
              args.max_hours, args.poll_seconds)
