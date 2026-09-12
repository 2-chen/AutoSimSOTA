"""Integration remediation v5, reusing verified v4 data/training commits."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from autosim.experiment_system.executor import Executor, Job, lease
from autosim.experiment_system.plugins import ROBOTWIN_TASKS, RoboTwinPlugin
from autosim.experiment_validation.integration_remediation import OLD_REQUIRED, file_manifest, get_result
from autosim.experiment_validation.integration_remediation_v4 import environment
from autosim.research.common import assert_frozen, atomic_json, digest, immutable_json, now, read_json


V4_REUSED = tuple(
    f"robotwin_{task}_{stage}_v4"
    for task in ("pick_dual_bottles", "stack_blocks_two")
    for stage in ("collect", "convert", "admit", "act_train", "dp_train")
) + ("robotwin_open_laptop_probe_v2",)


def prepare_manifest(workspace: Path, root: Path, old: Path, v3: Path,
                     v4: Path, overlay: Path) -> dict:
    for predecessor in (old, v3, v4):
        if read_json(predecessor / "queue_state.json").get("status") != "requires_capability_review":
            raise ValueError(f"predecessor is not terminal: {predecessor}")
    old_executor, v4_executor = Executor(old / "jobs"), Executor(v4 / "jobs")
    old_commits, v4_commits = {}, {}
    for name in OLD_REQUIRED:
        if not old_executor.committed(name):
            raise ValueError(f"missing original commit: {name}")
        old_commits[name] = digest(old_executor.root / name / "commit.json")
    for name in V4_REUSED:
        if not v4_executor.committed(name):
            raise ValueError(f"missing reusable v4 commit: {name}")
        v4_commits[name] = digest(v4_executor.root / name / "commit.json")
    shim = root / "bin/ffmpeg"
    ffmpeg = Path("/home/wbc/miniconda3/envs/robotwin5090/lib/python3.10/site-packages/"
                  "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
    if not shim.is_symlink() or shim.resolve() != ffmpeg:
        raise ValueError("invalid v5 ffmpeg shim")
    source_paths = [
        Path(__file__), Path(__file__).with_name("integration_remediation.py"),
        Path(__file__).with_name("integration_remediation_v4.py"),
        Path(__file__).with_name("robotwin_collect_v2.py"),
        Path(__file__).with_name("robotwin_evaluate_v2.py"),
        Path(__file__).with_name("remediation_worker_v3.py"),
        Path(__file__).with_name("remediation_worker_v2.py"),
        Path(__file__).with_name("safe_evaluation_v2.py"),
        Path(__file__).with_name("safe_evaluation.py"),
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
        old / "queue_state.json", v3 / "queue_state.json", v4 / "queue_state.json",
        old / "protected_previous_core.json", old / "texture_state.json", ffmpeg, shim,
        workspace / "AutoSimSOTA/RoboSynChallenge/policy/act/deploy_policy.py",
        workspace / "AutoSimSOTA/RoboSynChallenge/policy/dp/deploy_policy.py",
    ]
    v4_failed = {}
    for name in ("robotwin_pick_dual_bottles_act_evaluate_v4",
                 "robotwin_pick_dual_bottles_dp_evaluate_v4",
                 "robotwin_stack_blocks_two_act_evaluate_v4",
                 "robotwin_stack_blocks_two_dp_evaluate_v4",
                 "robotwin_open_laptop_collect_v4", "robosyn_click_dp_evaluate_isolated_v4"):
        attempt = v4 / "jobs" / name / "attempt_001"
        v4_failed[name] = {str(path.absolute()): digest(path) for path in
                           (attempt / "request.json", attempt / "receipt.json", attempt / "stdout.log")}
    payload = {
        "schema_version": 5, "created_for": "native_integration_remediation",
        "predecessors": {
            "original": {"path": str(old), "state_sha256": digest(old / "queue_state.json"),
                         "commits": old_commits},
            "v3": {"path": str(v3), "state_sha256": digest(v3 / "queue_state.json")},
            "v4": {"path": str(v4), "state_sha256": digest(v4 / "queue_state.json"),
                   "reused_commits": v4_commits, "failed_attempt_evidence": v4_failed},
        },
        "warp_overlay": {"path": str(overlay), "version": "1.12.0", "files": file_manifest(overlay)},
        "ffmpeg": {"shim": str(shim), "target": str(ffmpeg), "sha256": digest(ffmpeg)},
        "sources": {str(path.absolute()): digest(path) for path in source_paths},
        "remediations": {
            "robotwin_policy_namespace": "spawn_policy_child_before_native_cwd",
            "open_laptop_collection": "skip_only_official_UnStableError_before_planning",
            "robosyn_dp_startup": "fresh_process_per_episode_max_three_pre_reset_native_startups",
        },
        "claims": {"ranking": False, "policy_quality": False, "success_threshold_changed": False},
    }
    immutable_json(root / "remediation_manifest.json", payload)
    return payload


def evaluation_job(workspace: Path, root: Path, old: Path, v4: Path, task: str,
                   policy: str, sources: dict[str, str], env: dict[str, str]) -> Job:
    v4_executor = Executor(v4 / "jobs")
    train_name = f"robotwin_{task}_{policy}_train_v4"
    trained = get_result(v4_executor, train_name)
    checkpoint = Path(trained["checkpoint"])
    pinned = {**sources, **trained["verified_files"],
              str(v4_executor.root / train_name / "commit.json"):
              digest(v4_executor.root / train_name / "commit.json")}
    command = [str(old / "runtime_env/bin/python"),
               "-m", "autosim.experiment_validation.robotwin_evaluate_v2",
               "--repo", str(old / "native/RoboTwin"),
               "--policy-repo", str(workspace / "AutoSimSOTA/RoboSynChallenge"),
               "--attempt", "{attempt}", "--task", task, "--checkpoint", str(checkpoint),
               "--policy", policy, "--episodes", "3",
               "--seed", str(94_006_000 + ROBOTWIN_TASKS.index(task) * 10_000)]
    return Job(f"robotwin_{task}_{policy}_evaluate_v5", "evaluate", command, str(workspace),
               {"result.json": "result"}, pinned, environment=env,
               timeout_seconds=1800, budget_seconds=1800, gpu="0")


def open_laptop_jobs(workspace: Path, root: Path, old: Path, v4: Path, executor: Executor,
                     sources: dict[str, str], env: dict[str, str]) -> list[Job]:
    task, native = "open_laptop", old / "native/RoboTwin"
    contract = RoboTwinPlugin(native).contract(task)
    inputs = {**sources, **contract.sources}
    probe_commit = v4 / "jobs/robotwin_open_laptop_probe_v2/commit.json"
    inputs[str(probe_commit)] = digest(probe_commit)
    jobs = []

    def job(name, stage, command, dependencies=(), gpu="0", timeout=1800, pinned=None):
        return Job(name, stage, command, str(native), {"result.json": "result"}, pinned or inputs,
                   dependencies=dependencies, environment=env, timeout_seconds=timeout,
                   budget_seconds=timeout, gpu=gpu)

    collect = "robotwin_open_laptop_collect_v5"
    jobs.append(job(collect, "collect", [str(old / "runtime_env/bin/python"),
                    "-m", "autosim.experiment_validation.robotwin_collect_v2",
                    "--repo", str(native), "--attempt", "{attempt}", "--task", task,
                    "--episodes", "3", "--seed", "92026100"], timeout=2400))
    collected = get_result(executor, collect)
    if not collected:
        return jobs
    collection_result = executor.root / collect / executor.committed(collect)["attempt"] / "result.json"
    convert = "robotwin_open_laptop_convert_v5"
    base = [str(old / "runtime_env/bin/python"), "-m", "autosim.experiment_system.robotwin_worker"]
    task_args = ["--repo", str(native), "--attempt", "{attempt}", "--task", task]
    jobs.append(job(convert, "admit", [*base, "convert", *task_args,
                    "--collection-result", str(collection_result)], (collect,), gpu=None))
    converted = get_result(executor, convert)
    if not converted:
        return jobs
    dataset = converted["dataset"]
    backend = [str(workspace / "AutoSimSOTA/.venv/bin/python"),
               "-m", "autosim.experiment_system.backend_worker"]
    common = ["--workspace", str(workspace), "--benchmark", "robotwin", "--native-repo", str(native),
              "--task", task, "--attempt", "{attempt}", "--dataset", dataset]
    admit = "robotwin_open_laptop_admit_v5"
    jobs.append(job(admit, "admit", [*backend, "admit", *common], (convert,), gpu=None, timeout=600))
    admitted = get_result(executor, admit)
    if not admitted:
        return jobs
    train_sources = {**inputs, **admitted["verified_files"]}
    for policy in ("act", "dp"):
        train = f"robotwin_open_laptop_{policy}_train_v5"
        jobs.append(job(train, "train", [*backend, "train", *common, "--policy", policy,
                        "--steps", "200", "--seed", "91006100"], (admit,), timeout=2400,
                        pinned=train_sources))
        trained = get_result(executor, train)
        if not trained:
            continue
        checkpoint = Path(trained["checkpoint"])
        eval_sources = {**train_sources, **trained["verified_files"]}
        command = [str(old / "runtime_env/bin/python"),
                   "-m", "autosim.experiment_validation.robotwin_evaluate_v2",
                   "--repo", str(native), "--policy-repo", str(workspace / "AutoSimSOTA/RoboSynChallenge"),
                   "--attempt", "{attempt}", "--task", task, "--checkpoint", str(checkpoint),
                   "--policy", policy, "--episodes", "3", "--seed", "94026000"]
        jobs.append(job(f"robotwin_open_laptop_{policy}_evaluate_v5", "evaluate", command,
                        (train,), timeout=1800, pinned=eval_sources))
    return jobs


def build_jobs(workspace: Path, root: Path, old: Path, v3: Path, v4: Path,
               overlay: Path, executor: Executor, manifest: dict) -> list[Job]:
    sources = {**manifest["sources"], **manifest["warp_overlay"]["files"],
               str(root / "remediation_manifest.json"): digest(root / "remediation_manifest.json")}
    env = environment(workspace, root, old, overlay)
    jobs = [evaluation_job(workspace, root, old, v4, task, policy, sources, env)
            for task in ("pick_dual_bottles", "stack_blocks_two") for policy in ("act", "dp")]
    jobs.extend(open_laptop_jobs(workspace, root, old, v4, executor, sources, env))
    old_dp = get_result(Executor(old / "jobs"), "robosyn_click_dp_train")
    checkpoint = Path(old_dp["checkpoint"])
    command = [str(workspace / "AutoSimSOTA/.venv/bin/python"),
               "-m", "autosim.experiment_validation.remediation_worker_v3",
               "robosyn-isolated-safe-evaluate", "--workspace", str(workspace),
               "--attempt", "{attempt}", "--task", "click_bell", "--checkpoint", str(checkpoint),
               "--policy", "dp", "--episodes", "3", "--seed", "91006200"]
    jobs.append(Job("robosyn_click_dp_evaluate_isolated_v5", "evaluate", command,
                    str(workspace / "AutoSimSOTA/RoboSynChallenge_eval_clean"),
                    {"result.json": "result"}, {**sources, **old_dp["verified_files"]},
                    environment=env, timeout_seconds=5400, budget_seconds=5400, gpu="0"))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--warp-overlay", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    workspace, root, old, v3, v4, overlay = (args.workspace.absolute(), args.root.absolute(),
        args.old_root.absolute(), args.v3_root.absolute(), args.v4_root.absolute(), args.warp_overlay.absolute())
    manifest = prepare_manifest(workspace, root, old, v3, v4, overlay)
    pinned = {**manifest["sources"], **manifest["warp_overlay"]["files"],
              str(root / "remediation_manifest.json"): digest(root / "remediation_manifest.json")}
    executor = Executor(root / "jobs")
    state = {"schema_version": 5, "started_at": now(), "status": "running",
             "ranking_eligible": False, "formal_experiments_complete": False,
             "predecessors": {key: value["state_sha256"] for key, value in manifest["predecessors"].items()}}
    started = time.monotonic()
    with lease(root / "queue.lock"):
        while time.monotonic() - started < args.max_hours * 3600:
            assert_frozen(pinned)
            statuses = {}
            for job in build_jobs(workspace, root, old, v3, v4, overlay, executor, manifest):
                statuses[job.name] = executor.run(job)
                state.update(jobs=statuses, updated_at=now())
                atomic_json(root / "queue_state.json", state)
            wanted = {"robosyn_click_dp_evaluate_isolated_v5"}
            wanted.update(f"robotwin_{task}_{policy}_evaluate_v5"
                          for task in ROBOTWIN_TASKS for policy in ("act", "dp"))
            if all(statuses.get(name, {}).get("status") == "committed" for name in wanted):
                state.update(status="integration_gate_reached", inherited_original=list(OLD_REQUIRED),
                             inherited_v4=list(V4_REUSED),
                             remaining=["formal_policy_experiments", "independent_operator_measurement"])
                break
            terminal = {"requires_adapter_or_data_fix", "requires_artifact_audit",
                        "requires_audit_unknown_execution", "quarantine_evaluation", "budget_exhausted"}
            if statuses and all(row.get("status") == "committed" or row.get("status") in terminal
                                for row in statuses.values()):
                if {job.name for job in build_jobs(workspace, root, old, v3, v4, overlay, executor, manifest)} == set(statuses):
                    state["status"] = "requires_capability_review"
                    break
            time.sleep(10)
        else:
            state["status"] = "queue_wait_budget_exhausted"
        state["updated_at"] = now()
        atomic_json(root / "queue_state.json", state)


if __name__ == "__main__":
    main()
