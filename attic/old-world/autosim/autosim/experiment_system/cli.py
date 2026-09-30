"""Independent system-version bootstrap, read-only inventory and admission audit."""
import argparse
import subprocess
from dataclasses import asdict
from pathlib import Path

from autosim.research.common import assert_frozen, atomic_json, digest, immutable_json, now, read_json
from .plugins import ROBOTWIN_TASKS, RoboSynPlugin, RoboTwinPlugin


def bootstrap(workspace, output):
    output.mkdir(parents=True, exist_ok=True)
    previous = workspace / "autosim/output/robosyn_general_20260905/production_continuation/frozen_core.json"
    old_hashes = read_json(previous)
    assert_frozen(old_hashes)
    immutable_json(output / "protected_previous_core.json", old_hashes)
    plan = {"schema_version": 1, "system_version": "0.1.0", "created_date": "2026-09-06",
            "robosyn_tasks": list(__import__("autosim.research.registry", fromlist=["TASK_IDS"]).TASK_IDS),
            "transfer_benchmark": "robotwin", "transfer_tasks": list(ROBOTWIN_TASKS),
            "transfer_setting": "demo_randomized", "policies": ["act", "dp"],
            "smoke": {"collection_episodes": 3, "train_updates": 200, "evaluation_episodes": 3,
                      "seed_base": 91006000, "policy_quality_claim": False},
            "stages": ["cpu_fault_acceptance", "native_runtime_preflight", "real_collection_admission",
                       "act_dp_train_reload_eval", "second_benchmark_full_loop", "system_ablation",
                       "independent_operator_onboarding", "budget_matched_policy_experiments"],
            "comparators": ["fixed_workflow_same_components", "full_system", "without_capability_validation",
                            "without_recovery", "without_quality_admission"],
            "constraints": {"no_mutation_of_previous_core": True, "native_judge_read_only": True,
                            "no_policy_failure_retry": True, "no_llm_api_calls_by_default": True,
                            "gpu_exclusive_lease": "/tmp/autosim-robosyn-gpu-0.lock"},
            "formal_experiments": {"status": "gated_on_real_integration_and_cost_measurement",
                                   "training_seeds": [6100, 6101, 6102], "final_test_episodes": 200,
                                   "final_seeds_created_before_formal_training": True,
                                   "operator_effort_requires_independent_human_measurement": True},
            "known_boundaries": ["RoboSyn development tasks are not held-out tasks",
                                 "smoke success is not policy quality", "local scores are not official submission"]}
    immutable_json(output / "execution_plan.json", plan)
    return plan


def inventory(workspace, output):
    from autosim.research.registry import TASK_IDS
    rows = []
    plugins = [(RoboSynPlugin(workspace), list(TASK_IDS)),
               (RoboTwinPlugin(output / "native/RoboTwin"), list(ROBOTWIN_TASKS))]
    for plugin, tasks in plugins:
        for task in tasks:
            try:
                rows.append(plugin.inventory(task))
            except Exception as exc:
                rows.append({"task": task, "configuration_loaded": False, "error": f"{type(exc).__name__}: {exc}"})
    result = {"created_at": now(), "tasks": rows, "validated_full_workflow_count": 0,
              "configuration_loaded_count": sum(r["configuration_loaded"] for r in rows)}
    atomic_json(output / "inventory.json", result)
    return result


def audit_existing(workspace, output):
    from autosim.research.registry import TASK_IDS, load_task
    from .quality import audit_dataset
    previous = workspace / "autosim/output/robosyn_general_20260905/smoke"
    rows = []
    for task in TASK_IDS:
        candidates = sorted((previous / task).glob("**/collection/collection.json"))
        good = [p for p in candidates if read_json(p).get("status") == "completed"]
        if not good:
            rows.append({"task": task, "status": "no_successful_collection", "passed": False})
            continue
        manifest = good[-1]
        dataset = Path(read_json(manifest)["dataset_paths"][0])
        target = output / "admission" / f"{task}.json"
        if target.exists():
            result = read_json(target)
        else:
            spec = load_task(workspace / "AutoSimSOTA/RoboSynChallenge", task)
            result = audit_dataset(dataset, state_dim=spec.state_dim, action_dim=spec.action_dim)
            result.update(collection_manifest=str(manifest), collection_manifest_sha256=digest(manifest))
            atomic_json(target, result)
        rows.append({"task": task, "status": result["status"], "passed": result["passed"],
                     "audit": str(target), "episodes": len(result["episodes"]), "errors": result["errors"]})
        atomic_json(output / "admission_summary.json", {"tasks": rows, "all_tasks_processed": False})
        print({"task": task, "passed": result["passed"], "errors": result["errors"]}, flush=True)
    report = {"tasks": rows, "all_tasks_processed": True, "scope": "existing real smoke collection, not new policy results"}
    atomic_json(output / "admission_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "inventory", "audit-existing", "cpu-acceptance", "status"])
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.absolute()
    output = (args.output or workspace / "autosim/output/experiment_system_20260906").absolute()
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "bootstrap":
        result = bootstrap(workspace, output)
    elif args.command == "inventory":
        result = inventory(workspace, output)
    elif args.command == "audit-existing":
        result = audit_existing(workspace, output)
    elif args.command == "cpu-acceptance":
        started = now()
        with (output / "cpu_acceptance.log").open("w") as log:
            process = subprocess.run([str(workspace / "AutoSimSOTA/.venv/bin/python"), "-m", "unittest", "discover",
                                      "-s", "tests", "-p", "test_experiment_system.py", "-v"],
                                     cwd=workspace / "autosim", stdout=log, stderr=subprocess.STDOUT, timeout=120)
        result = {"passed": process.returncode == 0, "returncode": process.returncode,
                  "started_at": started, "finished_at": now(), "log_sha256": digest(output / "cpu_acceptance.log"),
                  "test_source_sha256": digest(workspace / "autosim/tests/test_experiment_system.py"),
                  "scope": "CPU contracts, data checks and real subprocess faults; no policy scores"}
        atomic_json(output / "cpu_acceptance.json", result)
        if process.returncode:
            raise SystemExit("CPU acceptance failed; see log")
    else:
        from .executor import Executor
        assert_frozen(read_json(output / "protected_previous_core.json"))
        result = Executor(output / "jobs").summary()
        result["protected_previous_core_unchanged"] = True
    if args.command == "inventory":
        print({k: v for k, v in result.items() if k != "tasks"})
    elif args.command == "bootstrap":
        print({"plan": str(output / "execution_plan.json"), "previous_core_unchanged": True})
    else:
        print(result)


if __name__ == "__main__":
    main()
