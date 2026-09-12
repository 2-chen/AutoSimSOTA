"""Bounded, training-only repair study for RoboSyn expert collection gaps.

The benchmark environment, randomization and success judge remain unchanged.
Only a copied expert action graph may be varied. Probe failures are scientific
results; production data is admitted only after 20 successful episodes and full
parquet/video validation. Official-data fallback is reported, never called self
collection.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from dataclasses import replace
from pathlib import Path

from autosim.experiment_system.executor import Executor, Job
from autosim.experiment_system.quality import audit_dataset
from autosim.research.common import atomic_json, digest, freeze_files, now, read_json, redact
from autosim.research.ledger import SeedLedger
from autosim.research.registry import load_task
from autosim.research.runtime import Runtime
from autosim.robosyn_data import evaluation_seed_bank
from .queue_manifest import versioned_manifest


TASK_VARIANTS = {
    "sample_loading": {"official_graph": None, "final_place_raise_001m": .01,
                       "final_place_raise_002m": .02},
    "item_assembly": {"official_graph": None, "left_align_offset_minus_022m": -.22,
                      "left_align_offset_minus_020m": -.20},
}


def collection_master_seed(task: str, variant: str, production: bool) -> int:
    if task not in TASK_VARIANTS or variant not in TASK_VARIANTS[task]:
        raise ValueError("undeclared task/variant")
    base = 98_000_000 if task == "sample_loading" else 98_100_000
    return base + list(TASK_VARIANTS[task]).index(variant) * 10_000 + (5_000 if production else 0)


def reserve_collection_banks(previous: Path) -> dict:
    """Reserve every possible probe/production reset stream before selection."""
    ledger = SeedLedger(previous / "seeds.sqlite")
    result = {}
    try:
        for task, variants in TASK_VARIANTS.items():
            result[task] = {}
            for variant in variants:
                result[task][variant] = {}
                for phase, production, attempts in (("probe", False, 12), ("production", True, 200)):
                    master = collection_master_seed(task, variant, production)
                    # The collector performs one final reset after the last
                    # recorded attempt, so reserve attempts+1 scene seeds.
                    count = attempts + 1
                    seeds = ledger.reserve(task,
                        f"robosyn_collection_gap_20260906:{phase}:{variant}",
                        "collection", master, count)
                    result[task][variant][phase] = {
                        "master_seed": master, "reserved_reset_count": count,
                        "first_seed": seeds[0], "last_seed": seeds[-1],
                    }
    finally:
        ledger.close()
    return result


def validate_collection_seed_trace(collection: dict, *, master_seed: int,
                                   target: int, max_attempts: int) -> dict:
    """Verify that all observed collection resets came from the reserved stream."""
    if (collection.get("master_seed") != master_seed or
            collection.get("target_successful_episodes") != target or
            collection.get("collection_mode") != "expert" or
            collection.get("profile") != "full_random"):
        raise ValueError("collection manifest contract mismatch")
    attempts = collection.get("attempts", [])
    count = int(collection.get("expert_attempt_count", -1))
    if count != len(attempts) or not 0 <= count <= max_attempts:
        raise ValueError("collection attempt count mismatch")
    expected = evaluation_seed_bank(master_seed, max_attempts + 1)
    attempt_seeds = [int(row.get("seed", -1)) for row in attempts]
    reset_seeds = [int(row.get("seed", -1)) for row in collection.get("resets", [])]
    if attempt_seeds != expected[:count]:
        raise ValueError("collection attempts differ from the reserved ordered seed stream")
    if reset_seeds != expected[:count + 1]:
        raise ValueError("collection resets differ from the reserved ordered seed stream")
    successful = [int(seed) for seed in collection.get("successful_episode_seeds", [])]
    failed = [int(seed) for seed in collection.get("failed_attempt_seeds", [])]
    saved_from_attempts = [int(row["seed"]) for row in attempts if row.get("saved") is True]
    failed_from_attempts = [int(row["seed"]) for row in attempts if row.get("saved") is False]
    if successful != saved_from_attempts or failed != failed_from_attempts:
        raise ValueError("collection success/failure seed lineage mismatch")
    if len(successful) != len(set(successful)) or len(failed) != len(set(failed)):
        raise ValueError("collection trace contains duplicate attempt seeds")
    status = collection.get("status")
    if (status == "completed") != (len(successful) == target):
        raise ValueError("collection status disagrees with successful episode target")
    return {"ordered_reserved_seed_stream_verified": True,
            "attempt_count": count, "reset_count": count + 1,
            "successful_episode_count": len(successful)}


def find_node(config: dict, name: str) -> dict:
    found = [node[name] for nodes in config["node"].values() for node in nodes if name in node]
    if len(found) != 1:
        raise ValueError(f"expected exactly one action node {name}, found {len(found)}")
    return found[0]


def derived_config(task: str, variant: str, source: Path, destination: Path) -> dict:
    if task not in TASK_VARIANTS or variant not in TASK_VARIANTS[task]:
        raise ValueError("undeclared task/variant")
    original = read_json(source)
    modified = copy.deepcopy(original)
    value = TASK_VARIANTS[task][variant]
    mutation = None
    if task == "sample_loading" and value is not None:
        node = find_node(modified, "left_arm_cube_place_qpos")
        chain = node["kwargs"]["affordance_infos"][0]["valid_funcs_name_kwargs_proc"]
        if chain[0].get("name") != "get_ik_ret" or chain[0]["kwargs"].get("control_part") != "left_arm":
            raise ValueError("sample-loading final IK contract changed")
        chain.insert(0, {"name": "no_validation", "kwargs": {}, "pass_processes": [
            {"name": "get_offset_pose", "kwargs": {"offset_value": value,
             "direction": "z", "mode": "extrinsic"}}]})
        mutation = {"node": "left_arm_cube_place_qpos", "operation": "raise_target_before_ik",
                    "axis": "world_z", "delta_m": value,
                    "reason": "existing probes repeatedly fail the final left-arm place IK"}
    elif task == "item_assembly" and value is not None:
        node = find_node(modified, "left_align_qpos")
        process = node["kwargs"]["affordance_infos"][0]["valid_funcs_name_kwargs_proc"][0]["pass_processes"][0]
        kwargs = process.get("kwargs", {})
        if process.get("name") != "get_offset_pose" or kwargs != {
                "offset_value": -.24, "direction": "x", "mode": "intrinsic"}:
            raise ValueError("item-assembly alignment contract changed")
        old = kwargs["offset_value"]
        kwargs["offset_value"] = value
        mutation = {"node": "left_align_qpos", "operation": "shorten_alignment_reach",
                    "axis": "tool_x", "old_offset_m": old, "new_offset_m": value,
                    "reason": "existing probes include late left_align_qpos IK failures"}
    if value is None and modified != original:
        raise AssertionError("official graph must be an exact copy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, modified)
    provenance = {"task": task, "variant": variant, "source": str(source.absolute()),
                  "source_sha256": digest(source), "derived_sha256": digest(destination),
                  "mutation": mutation, "environment_or_success_judge_modified": False,
                  "training_expert_only": True}
    atomic_json(destination.with_name("config_provenance.json"), provenance)
    return provenance


def collect_command(runtime: Runtime, spec, config: Path, destination: Path,
                    target: int, attempts: int, master_seed: int):
    manifest = destination / "collection.json"
    command = [str(runtime.python), "-m", "autosim.research.collection_worker",
               "--gym_config", spec.gym_config, "--action_config", str(config),
               "--num_envs", "1", "--headless", "--max_episodes", str(target),
               "--collection_seed", str(master_seed), "--collection_profile", "full_random",
               "--collection_mode", "expert", "--collection_manifest", str(manifest),
               "--dataset_save_path", str(destination / "data"),
               "--collection_max_attempts", str(attempts), "--collection_quiet"]
    return command, manifest


def run_variant(workspace: Path, attempt: Path, task: str, variant: str,
                target: int, max_attempts: int, master_seed: int, production: bool):
    runtime = Runtime(workspace, attempt, gpu="0")
    spec = load_task(runtime.repo, task)
    config = attempt / "derived_action_config.json"
    provenance = derived_config(task, variant, Path(spec.action_config), config)
    derived_spec = replace(spec, action_config=str(config), config_hashes={
                           **spec.config_hashes, str(config): digest(config)})
    command, manifest = collect_command(runtime, derived_spec, config, attempt / "collection",
                                        target, max_attempts, master_seed)
    error = None
    try:
        runtime.run(command, attempt / "process", max(900, max_attempts * 20))
    except Exception as exc:
        error = redact(f"{type(exc).__name__}: {exc}")
    collection = read_json(manifest) if manifest.is_file() else {}
    if not collection:
        atomic_json(attempt / "incomplete_result.json", {"task": task, "variant": variant,
                    "underlying_error": error, "reason": "collection_manifest_missing"})
        raise RuntimeError("collection process produced no auditable manifest")
    seed_trace = validate_collection_seed_trace(collection, master_seed=master_seed,
                                                 target=target, max_attempts=max_attempts)
    bounded_error = f"RuntimeError: Collection exceeded {max_attempts} expert attempts."
    if error is not None and collection.get("error") != bounded_error:
        atomic_json(attempt / "incomplete_result.json", {"task": task, "variant": variant,
                    "underlying_error": error, "manifest_error": collection.get("error"),
                    "reason": "unexpected_collection_process_failure"})
        raise RuntimeError("collection failed for a reason other than its predeclared attempt bound")
    successful = len(collection.get("successful_episode_seeds", []))
    result = {"status": "completed", "kind": "training_expert_probe" if not production else "training_collection",
              "task": task, "variant": variant, "target_successful_episodes": target,
              "max_attempts": max_attempts, "successful_episodes": successful,
              "attempt_count": int(collection.get("expert_attempt_count", max_attempts if error else 0)),
              "underlying_process_completed": error is None, "underlying_error": error,
              "seed_trace": seed_trace,
              "training_expert_modified": provenance["mutation"] is not None,
              "environment_or_success_judge_modified": False, "policy_score_claim": False,
              "official_data_fallback": False, "verified_files": {
                  str(config.absolute()): digest(config),
                  str(config.with_name("config_provenance.json").absolute()): digest(config.with_name("config_provenance.json"))}}
    if manifest.is_file():
        result["verified_files"][str(manifest.absolute())] = digest(manifest)
    if production:
        if successful != target or collection.get("status") != "completed" or len(collection.get("dataset_paths", [])) != 1:
            atomic_json(attempt / "incomplete_result.json", result)
            raise RuntimeError(f"bounded collection reached {successful}/{target}; official-data fallback required")
        dataset = Path(collection["dataset_paths"][0])
        audit = audit_dataset(dataset, state_dim=spec.state_dim, action_dim=spec.action_dim)
        atomic_json(attempt / "data_admission.json", audit)
        if not audit["passed"] or len(audit["episodes"]) != target:
            atomic_json(attempt / "incomplete_result.json", result)
            raise RuntimeError("full collected dataset admission failed")
        result.update(dataset=str(dataset), structural_admission_passed=True,
                      verified_files={**result["verified_files"], **audit["verified_files"],
                                      str((attempt / "data_admission.json").absolute()): digest(attempt / "data_admission.json")})
    atomic_json(attempt / "result.json", result)


def existing_evidence(previous: Path, output: Path):
    rows = {}
    for task in TASK_VARIANTS:
        directory = previous / "stepwise_history_expert_diagnostics" / task
        manifest = read_json(directory / "collection.json")
        log = (directory / "process/stdout.log").read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
        failures = re.findall(r"Node '([^']+)' generation fails", log)
        rows[task] = {"task": task, "diagnostic_attempts": manifest.get("expert_attempt_count"),
                      "successful_trajectories": len(manifest.get("successful_episode_seeds", [])),
                      "attempt_outcomes": manifest.get("attempts", []),
                      "failed_action_nodes": {name: failures.count(name) for name in sorted(set(failures))},
                      "evidence": {str((directory / "collection.json").absolute()): digest(directory / "collection.json"),
                                   str((directory / "checks.jsonl").absolute()): digest(directory / "checks.jsonl"),
                                   str((directory / "process/stdout.log").absolute()): digest(directory / "process/stdout.log")},
                      "interpretation": ("final left_arm_cube_place_qpos IK is the repeated planning failure"
                         if task == "sample_loading" else
                         "one fully generated plan failed task success; other seeds include late alignment IK failures"),
                      "causal_fix_established": False}
    report = {"created_at": now(), "kind": "existing_collection_gap_evidence", "tasks": rows,
              "judge_mutation_allowed": False, "policy_performance_claim": False}
    atomic_json(output, report)
    return report


def sources(workspace: Path, task: str) -> dict[str, str]:
    repo = workspace / "AutoSimSOTA/RoboSynChallenge"
    embodi = workspace / "AutoSimSOTA/EmbodiChain"
    local = Path(__file__).resolve()
    paths = [local, local.parents[1] / "experiment_system/executor.py",
             local.parents[1] / "experiment_system/worker.py", local.parents[1] / "experiment_system/quality.py",
             local.with_name("queue_manifest.py"), local.parents[1] / "robosyn_data.py",
             local.parents[1] / "research/common.py", local.parents[1] / "research/runtime.py",
             local.parents[1] / "research/collection_worker.py", local.parents[1] / "research/ledger.py",
             local.parents[1] / "research/policy_rpc.py",
             local.parents[1] / "research/registry.py", repo / "scripts/run_env.py",
             repo / "robosynchallenge/managers/datasets.py", repo / f"configs/{task}/action_config.json",
             repo / f"configs/{task}/random/gym_config.json",
             repo / f"robosynchallenge/tasks/{task}/{task}.py",
             repo / f"robosynchallenge/tasks/{task}/action_bank.py",
             embodi / "embodichain/lab/gym/envs/action_bank/configurable_action.py",
             embodi / "embodichain/lab/gym/utils/misc.py"]
    return freeze_files(paths)


def job(workspace: Path, root: Path, task: str, variant: str, *, production=False):
    target, attempts = (20, 200) if production else (1, 12)
    seed = collection_master_seed(task, variant, production)
    python = workspace / "AutoSimSOTA/.venv/bin/python"
    command = [str(python), "-m", "autosim.experiment_validation.collection_gap", "run-variant",
               "--workspace", str(workspace), "--attempt", "{attempt}", "--task", task,
               "--variant", variant, "--target", str(target), "--max-attempts", str(attempts),
               "--master-seed", str(seed)]
    if production:
        command.append("--production")
    runtime = Runtime(workspace, root)
    allowed = {"PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "EMBODICHAIN_DATA_ROOT",
               "XDG_CACHE_HOME", "MPLCONFIGDIR", "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "PYTHONFAULTHANDLER"}
    environment = {k: v for k, v in runtime.environment().items() if k in allowed}
    label = f"{task}_{variant}_" + ("collect20" if production else "probe")
    return Job(label, "collect" if production else "probe", command, str(runtime.repo),
               {"result.json": "result"}, sources(workspace, task), environment=environment,
               timeout_seconds=7200 if production else 1200, budget_seconds=7200 if production else 1200,
               max_attempts=1, gpu="0")


def result(executor: Executor, name: str):
    commit = executor.committed(name)
    return read_json(executor.root / name / commit["attempt"] / "result.json") if commit else None


def queue(workspace: Path, root: Path, milestone_state: Path, previous: Path,
          max_hours: float, poll_seconds: float):
    if not 0 < max_hours <= 240 or not 1 <= poll_seconds <= 60:
        raise ValueError("invalid bounded queue settings")
    root.mkdir(parents=True, exist_ok=True)
    evidence_path = root / "existing_evidence.json"
    if not evidence_path.exists():
        existing_evidence(previous, evidence_path)
    seed_banks = reserve_collection_banks(previous)
    local = Path(__file__).resolve()
    manifest = {"tasks": list(TASK_VARIANTS), "variants": TASK_VARIANTS,
                "probe_target": 1, "probe_attempts": 12, "production_target": 20,
                "production_attempts": 200, "milestone_gate": str(milestone_state),
                "seed_banks": seed_banks,
                "existing_evidence_sha256": digest(evidence_path), "environment_modified": False,
                "queue_sources": freeze_files([local, local.with_name("queue_manifest.py"),
                                                 local.parents[1] / "robosyn_data.py",
                                                 local.parents[1] / "research/ledger.py"]),
                "success_judge_modified": False, "official_data_fallback_is_self_collection": False}
    versioned_manifest(root, manifest)
    executor = Executor(root / "jobs")
    state_path = root / "state.json"
    prior = read_json(state_path) if state_path.is_file() else {}
    if prior.get("status") in {"collection_gap_handled", "blocked_by_baseline_gate", "requires_review",
                              "queue_wait_budget_exhausted"}:
        return prior
    state = {**prior, "status": "running", "started_at": prior.get("started_at", now()),
             "formal_policy_improvement_complete": False,
             "deadline_epoch": float(prior.get("deadline_epoch", time.time() + max_hours * 3600))}
    if prior:
        state.update(resume_count=int(prior.get("resume_count", 0)) + 1, resumed_at=now())
    while time.time() < state["deadline_epoch"]:
        baseline = read_json(milestone_state) if milestone_state.is_file() else {}
        if baseline.get("status") != "baseline_table_complete":
            if baseline.get("status") in {"requires_review", "queue_wait_budget_exhausted"}:
                state.update(status="blocked_by_baseline_gate", baseline_status=baseline.get("status"), updated_at=now())
                atomic_json(state_path, state)
                return state
            state.update(stage="waiting_baseline_table", baseline_status=baseline.get("status", "missing"), updated_at=now())
            atomic_json(state_path, state)
            time.sleep(poll_seconds)
            continue
        capabilities = {}
        for task, variants in TASK_VARIANTS.items():
            probe_rows = []
            for variant in variants:
                current = job(workspace, root, task, variant)
                outcome = executor.run(current)
                state.update(stage=f"probe:{task}:{variant}", current_result=outcome, updated_at=now())
                atomic_json(state_path, state)
                if outcome.get("status") == "waiting_lease":
                    time.sleep(poll_seconds)
                    break
                if outcome.get("status") != "committed":
                    state.update(status="requires_review", failed_job=current.name, failure=outcome, updated_at=now())
                    atomic_json(state_path, state)
                    return state
                probe_rows.append(result(executor, current.name))
            else:
                ranked = sorted(probe_rows, key=lambda x: (-x["successful_episodes"],
                                list(variants).index(x["variant"])))
                best = ranked[0]
                if best["successful_episodes"] < 1:
                    capabilities[task] = {"status": "official_data_fallback",
                        "automatic_collection_validated": False, "reason": "all bounded expert variants produced zero successful trajectories",
                        "probe_results": probe_rows}
                    atomic_json(root / f"{task}_capability.json", capabilities[task])
                    continue
                production_job = job(workspace, root, task, best["variant"], production=True)
                outcome = executor.run(production_job)
                state.update(stage=f"collect20:{task}:{best['variant']}", current_result=outcome, updated_at=now())
                atomic_json(state_path, state)
                if outcome.get("status") == "waiting_lease":
                    time.sleep(poll_seconds)
                    break
                if outcome.get("status") == "committed":
                    collected = result(executor, production_job.name)
                    capabilities[task] = {"status": "validated_collection_20", "automatic_collection_validated": True,
                                          "selected_variant": best["variant"], "dataset": collected["dataset"],
                                          "probe_results": probe_rows}
                else:
                    capabilities[task] = {"status": "official_data_fallback", "automatic_collection_validated": False,
                        "reason": "selected expert variant did not reach 20/200 bounded acceptance",
                        "selected_variant": best["variant"], "production_result": outcome, "probe_results": probe_rows}
                atomic_json(root / f"{task}_capability.json", capabilities[task])
                continue
            break
        else:
            state.update(status="collection_gap_handled", capabilities=capabilities,
                         remaining=["multi-task effective policies", "three real feedback loops", "fair system comparison"],
                         updated_at=now())
            atomic_json(state_path, state)
            return state
    state.update(status="queue_wait_budget_exhausted", updated_at=now())
    atomic_json(state_path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    variant = sub.add_parser("run-variant")
    variant.add_argument("--workspace", type=Path, required=True)
    variant.add_argument("--attempt", type=Path, required=True)
    variant.add_argument("--task", choices=TASK_VARIANTS, required=True)
    variant.add_argument("--variant", required=True)
    variant.add_argument("--target", type=int, required=True)
    variant.add_argument("--max-attempts", type=int, required=True)
    variant.add_argument("--master-seed", type=int, required=True)
    variant.add_argument("--production", action="store_true")
    q = sub.add_parser("queue")
    q.add_argument("--workspace", type=Path, required=True)
    q.add_argument("--root", type=Path, required=True)
    q.add_argument("--milestone-state", type=Path, required=True)
    q.add_argument("--previous-output", type=Path, required=True)
    q.add_argument("--max-hours", type=float, default=168)
    q.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.command == "run-variant":
        run_variant(args.workspace.absolute(), args.attempt.absolute(), args.task, args.variant,
                    args.target, args.max_attempts, args.master_seed, args.production)
    else:
        queue(args.workspace.absolute(), args.root.absolute(), args.milestone_state.absolute(),
              args.previous_output.absolute(), args.max_hours, args.poll_seconds)
