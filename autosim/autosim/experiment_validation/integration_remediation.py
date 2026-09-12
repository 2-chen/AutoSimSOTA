"""Version-2 remediation queue for failed native integration jobs.

The original queue and attempts are immutable predecessors.  This queue reuses
only verified committed artifacts, pins a private Warp 1.12 overlay for RoboTwin,
and evaluates the already-trained RoboSyn DP checkpoint with the non-ranking
numerical-safety wrapper.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from autosim.experiment_system.executor import Executor, Job, lease
from autosim.experiment_system.plugins import ROBOTWIN_TASKS, RoboTwinPlugin
from autosim.research.common import assert_frozen, atomic_json, digest, immutable_json, now, read_json
from autosim.research.runtime import Runtime


OLD_REQUIRED = (
    "robotwin_render", "robosyn_click_collect", "robosyn_click_admit",
    "robosyn_click_act_train", "robosyn_click_act_evaluate", "robosyn_click_dp_train",
)


def file_manifest(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files[str(path.absolute())] = digest(path)
    if not files:
        raise ValueError(f"empty dependency overlay: {root}")
    return files


def prepare_manifest(workspace: Path, root: Path, old: Path) -> dict:
    overlay = root / "warp_overlay"
    metadata = next(iter(overlay.glob("warp_lang-1.12.0.dist-info/METADATA")), None)
    if metadata is None or "Version: 1.12.0" not in metadata.read_text(errors="replace"):
        raise ValueError("isolated Warp 1.12.0 overlay is missing")
    old_state = read_json(old / "queue_state.json")
    if old_state.get("status") != "requires_capability_review":
        raise ValueError("unexpected predecessor integration state")
    failed = {}
    for name in ("robotwin_pick_dual_bottles_probe", "robotwin_stack_blocks_two_probe",
                 "robotwin_open_laptop_probe", "robosyn_click_dp_evaluate"):
        attempt = old / "jobs" / name / "attempt_001"
        failed[name] = {str(p.absolute()): digest(p) for p in
                        (attempt / "request.json", attempt / "receipt.json", attempt / "stdout.log")}
    sources = [
        Path(__file__), Path(__file__).with_name("safe_evaluation.py"),
        Path(__file__).with_name("remediation_worker.py"),
        workspace / "autosim/autosim/experiment_system/backend_worker.py",
        workspace / "autosim/autosim/experiment_system/contracts.py",
        workspace / "autosim/autosim/experiment_system/executor.py",
        workspace / "autosim/autosim/experiment_system/native_policy.py",
        workspace / "autosim/autosim/experiment_system/plugins.py",
        workspace / "autosim/autosim/experiment_system/robotwin_worker.py",
        workspace / "autosim/autosim/experiment_system/trainers.py",
        workspace / "autosim/autosim/experiment_system/worker.py",
        workspace / "autosim/autosim/research/evaluation.py",
        workspace / "autosim/autosim/research/policy_rpc.py",
        workspace / "autosim/autosim/research/runtime.py",
        old / "queue_state.json", old / "protected_previous_core.json", old / "texture_state.json",
    ]
    payload = {
        "schema_version": 2, "created_for": "native_integration_remediation",
        "predecessor": {"path": str(old), "queue_state_sha256": digest(old / "queue_state.json"),
                        "status": old_state["status"], "failed_attempt_evidence": failed},
        "warp_overlay": {"path": str(overlay), "version": "1.12.0",
                         "files": file_manifest(overlay)},
        "sources": {str(p.absolute()): digest(p) for p in sources},
        "claims": {"ranking": False, "policy_quality": False, "success_threshold_changed": False},
    }
    immutable_json(root / "remediation_manifest.json", payload)
    return payload


def get_result(executor: Executor, name: str):
    record = executor.committed(name)
    return read_json(executor.root / name / record["attempt"] / "result.json") if record else None


def base_environment(workspace: Path, old: Path, overlay: Path) -> dict[str, str]:
    runtime = Runtime(workspace, old)
    keep = {"LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "EMBODICHAIN_DATA_ROOT", "MPLCONFIGDIR",
            "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "PYTHONFAULTHANDLER"}
    env = {k: v for k, v in runtime.environment().items() if k in keep}
    env.update(PYTHONPATH=os.pathsep.join((str(overlay), str(workspace / "autosim"),
               str(workspace / "AutoSimSOTA/RoboSynChallenge"), str(old / "native/RoboTwin"))),
               TORCH_EXTENSIONS_DIR=str(root_for_overlay(overlay) / "torch_extensions"),
               XDG_CACHE_HOME=str(root_for_overlay(overlay) / "runtime_cache"),
               HF_DATASETS_CACHE=str(root_for_overlay(overlay) / "dataset_cache"))
    return env


def root_for_overlay(overlay: Path) -> Path:
    return overlay.parent


def native_jobs(workspace: Path, root: Path, old: Path, executor: Executor,
                common_sources: dict[str, str], environment: dict[str, str]) -> list[Job]:
    native = old / "native/RoboTwin"
    interpreter = old / "runtime_env/bin/python"
    base = [str(interpreter), "-m", "autosim.experiment_system.robotwin_worker"]
    shared = ["--repo", str(native), "--attempt", "{attempt}"]
    jobs: list[Job] = []
    for index, task in enumerate(ROBOTWIN_TASKS):
        contract = RoboTwinPlugin(native).contract(task)
        inputs = {**common_sources, **contract.sources}
        inputs.update({str(p.absolute()): digest(p) for p in
                       (old / "runtime_env/lib/python3.10/site-packages/curobo/curobolib").glob("*.so")})

        def job(name, stage, command, dependency=(), gpu="0", timeout=1800, sources=None):
            return Job(name, stage, command, str(native), {"result.json": "result"},
                       sources or inputs, dependencies=dependency, environment=environment,
                       timeout_seconds=timeout, budget_seconds=timeout, gpu=gpu)

        prefix = f"robotwin_{task}"
        task_args = [*shared, "--task", task]
        probe = prefix + "_probe"
        jobs.append(job(probe, "probe", [*base, "probe", *task_args,
                        "--seed", str(92_006_000 + index * 10_000)], timeout=600))
        if not get_result(executor, probe):
            continue
        collect = prefix + "_collect"
        jobs.append(job(collect, "collect", [*base, "collect", *task_args, "--episodes", "3",
                        "--seed", str(92_006_100 + index * 10_000)], (probe,), timeout=2400))
        collected = get_result(executor, collect)
        if not collected:
            continue
        collection_result = executor.root / collect / executor.committed(collect)["attempt"] / "result.json"
        convert = prefix + "_convert"
        jobs.append(job(convert, "admit", [*base, "convert", *task_args,
                        "--collection-result", str(collection_result)], (collect,), gpu=None, timeout=1800))
        converted = get_result(executor, convert)
        if not converted:
            continue
        dataset = converted["dataset"]
        backend = [str(workspace / "AutoSimSOTA/.venv/bin/python"),
                   "-m", "autosim.experiment_system.backend_worker"]
        train_args = ["--workspace", str(workspace), "--benchmark", "robotwin",
                      "--native-repo", str(native), "--task", task, "--attempt", "{attempt}",
                      "--dataset", dataset]
        admit = prefix + "_admit"
        jobs.append(job(admit, "admit", [*backend, "admit", *train_args], (convert,),
                        gpu=None, timeout=600))
        admitted = get_result(executor, admit)
        if not admitted:
            continue
        train_sources = {**inputs, **admitted["verified_files"]}
        for policy in ("act", "dp"):
            train = prefix + f"_{policy}_train"
            jobs.append(job(train, "train", [*backend, "train", *train_args, "--policy", policy,
                            "--steps", "200", "--seed", "91006100"], (admit,), timeout=2400,
                            sources=train_sources))
            trained = get_result(executor, train)
            if not trained:
                continue
            evaluate = prefix + f"_{policy}_evaluate"
            eval_sources = {**train_sources, **trained["verified_files"]}
            jobs.append(job(evaluate, "evaluate", [*base, "evaluate", *task_args,
                            "--checkpoint", trained["checkpoint"], "--policy", policy,
                            "--episodes", "3", "--seed", str(94_006_000 + index * 10_000)],
                            (train,), timeout=1800, sources=eval_sources))
    return jobs


def validate_old_commits(old: Path) -> dict[str, str]:
    executor = Executor(old / "jobs")
    result = {}
    for name in OLD_REQUIRED:
        record = executor.committed(name)
        if not record:
            raise ValueError(f"required predecessor job is not committed: {name}")
        result[name] = digest(executor.root / name / "commit.json")
    return result


def build_jobs(workspace: Path, root: Path, old: Path, executor: Executor,
               manifest: dict, old_commits: dict[str, str]) -> list[Job]:
    common_sources = {**manifest["sources"], **manifest["warp_overlay"]["files"],
                      str(root / "remediation_manifest.json"): digest(root / "remediation_manifest.json")}
    environment = base_environment(workspace, old, root / "warp_overlay")
    jobs = native_jobs(workspace, root, old, executor, common_sources, environment)
    old_dp = get_result(Executor(old / "jobs"), "robosyn_click_dp_train")
    checkpoint = Path(old_dp["checkpoint"])
    safe_sources = {**common_sources, **old_dp["verified_files"]}
    safe_sources.update({str(old / "jobs" / name / "commit.json"): value for name, value in old_commits.items()})
    command = [str(workspace / "AutoSimSOTA/.venv/bin/python"),
               "-m", "autosim.experiment_validation.remediation_worker", "robosyn-safe-evaluate",
               "--workspace", str(workspace), "--attempt", "{attempt}", "--task", "click_bell",
               "--checkpoint", str(checkpoint), "--policy", "dp", "--episodes", "3",
               "--seed", "91006200"]
    jobs.append(Job("robosyn_click_dp_evaluate_safe_v2", "evaluate", command,
                    str(workspace / "AutoSimSOTA/RoboSynChallenge_eval_clean"),
                    {"result.json": "result"}, safe_sources, environment=environment,
                    timeout_seconds=1800, budget_seconds=1800, gpu="0"))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.absolute()
    root = (args.root or workspace / "autosim/output/experiment_system_remediation_20260906").absolute()
    old = (args.old_root or workspace / "autosim/output/experiment_system_20260906").absolute()
    if not 0 < args.max_hours <= 72:
        raise ValueError("remediation queue wall-clock budget must be in (0, 72] hours")
    manifest = prepare_manifest(workspace, root, old)
    old_commits = validate_old_commits(old)
    pinned = {**manifest["sources"], **manifest["warp_overlay"]["files"],
              str(root / "remediation_manifest.json"): digest(root / "remediation_manifest.json")}
    executor = Executor(root / "jobs")
    state = {"schema_version": 2, "started_at": now(), "status": "running",
             "predecessor_status": "requires_capability_review", "old_commits": old_commits,
             "ranking_eligible": False, "formal_experiments_complete": False}
    started = time.monotonic()
    with lease(root / "queue.lock"):
        while time.monotonic() - started < args.max_hours * 3600:
            assert_frozen(pinned)
            statuses = {}
            for job in build_jobs(workspace, root, old, executor, manifest, old_commits):
                statuses[job.name] = executor.run(job)
                state.update(jobs=statuses, updated_at=now())
                atomic_json(root / "queue_state.json", state)
            wanted = {"robosyn_click_dp_evaluate_safe_v2"}
            wanted.update(f"robotwin_{task}_{policy}_evaluate"
                          for task in ROBOTWIN_TASKS for policy in ("act", "dp"))
            if all(statuses.get(name, {}).get("status") == "committed" for name in wanted):
                state.update(status="integration_gate_reached", inherited_committed=list(OLD_REQUIRED),
                             remediations={"robotwin": "private_warp_1.12_overlay",
                                           "robosyn_dp": "numerical_failure_counts_as_failed_episode"},
                             remaining=["formal_policy_experiments", "independent_operator_measurement"])
                break
            terminal = {"requires_adapter_or_data_fix", "requires_artifact_audit",
                        "requires_audit_unknown_execution", "quarantine_evaluation", "budget_exhausted"}
            if statuses and all(row.get("status") == "committed" or row.get("status") in terminal
                                for row in statuses.values()):
                if {job.name for job in build_jobs(workspace, root, old, executor, manifest, old_commits)} == set(statuses):
                    state["status"] = "requires_capability_review"
                    break
            if args.once:
                state["status"] = "one_pass_finished"
                break
            time.sleep(10)
        else:
            state["status"] = "queue_wait_budget_exhausted"
        state["updated_at"] = now()
        atomic_json(root / "queue_state.json", state)


if __name__ == "__main__":
    main()
