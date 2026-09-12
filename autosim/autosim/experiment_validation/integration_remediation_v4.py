"""Native integration remediation v4: pinned media tool and task-aware probe."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from autosim.experiment_system.executor import Executor, Job, lease
from autosim.experiment_system.plugins import ROBOTWIN_TASKS, RoboTwinPlugin
from autosim.experiment_validation.integration_remediation import OLD_REQUIRED, file_manifest, get_result
from autosim.research.common import assert_frozen, atomic_json, digest, immutable_json, now, read_json
from autosim.research.runtime import Runtime


INHERITED_PROBES = ("robotwin_pick_dual_bottles_probe", "robotwin_stack_blocks_two_probe")


def prepare_manifest(workspace: Path, root: Path, old: Path, v3: Path, overlay: Path) -> dict:
    if read_json(old / "queue_state.json").get("status") != "requires_capability_review":
        raise ValueError("unexpected original predecessor state")
    if read_json(v3 / "queue_state.json").get("status") != "requires_capability_review":
        raise ValueError("v3 predecessor must be terminal before v4")
    metadata = next(iter(overlay.glob("warp_lang-1.12.0.dist-info/METADATA")), None)
    if metadata is None or "Version: 1.12.0" not in metadata.read_text(errors="replace"):
        raise ValueError("Warp 1.12 overlay is missing")
    shim = root / "bin/ffmpeg"
    expected_ffmpeg = Path("/home/wbc/miniconda3/envs/robotwin5090/lib/python3.10/site-packages/"
                           "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
    if not shim.is_symlink() or shim.resolve() != expected_ffmpeg or not expected_ffmpeg.is_file():
        raise ValueError("versioned ffmpeg shim is absent or points to an unexpected binary")
    old_executor, v3_executor = Executor(old / "jobs"), Executor(v3 / "jobs")
    old_commits, probe_commits = {}, {}
    for name in OLD_REQUIRED:
        if not old_executor.committed(name):
            raise ValueError(f"missing original committed predecessor: {name}")
        old_commits[name] = digest(old_executor.root / name / "commit.json")
    for name in INHERITED_PROBES:
        if not v3_executor.committed(name):
            raise ValueError(f"missing v3 committed probe: {name}")
        probe_commits[name] = digest(v3_executor.root / name / "commit.json")
    sources = [
        Path(__file__), Path(__file__).with_name("integration_remediation.py"),
        Path(__file__).with_name("safe_evaluation.py"),
        Path(__file__).with_name("safe_evaluation_v2.py"),
        Path(__file__).with_name("remediation_worker_v2.py"),
        Path(__file__).with_name("robotwin_probe_v2.py"),
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
        old / "queue_state.json", v3 / "queue_state.json", old / "protected_previous_core.json",
        old / "texture_state.json", expected_ffmpeg, shim,
    ]
    failed_names = ("robotwin_pick_dual_bottles_collect", "robotwin_stack_blocks_two_collect",
                    "robotwin_open_laptop_probe", "robosyn_click_dp_evaluate_safe_v2")
    failed = {}
    for name in failed_names:
        attempt = v3 / "jobs" / name / "attempt_001"
        failed[name] = {str(p.absolute()): digest(p) for p in
                        (attempt / "request.json", attempt / "receipt.json", attempt / "stdout.log")}
    payload = {
        "schema_version": 4, "created_for": "native_integration_remediation",
        "predecessors": {
            "original": {"path": str(old), "state_sha256": digest(old / "queue_state.json"),
                         "commits": old_commits},
            "v3": {"path": str(v3), "state_sha256": digest(v3 / "queue_state.json"),
                   "committed_probes": probe_commits, "failed_attempt_evidence": failed},
        },
        "warp_overlay": {"path": str(overlay), "version": "1.12.0", "files": file_manifest(overlay)},
        "ffmpeg": {"shim": str(shim), "target": str(expected_ffmpeg),
                   "sha256": digest(expected_ffmpeg)},
        "sources": {str(path.absolute()): digest(path) for path in sources},
        "remediations": {
            "media_path": "explicit_content_pinned_ffmpeg_shim",
            "open_laptop_probe": "official_play_once_arm_selection_precondition",
            "robosyn_dp": "fresh_simulator_process_per_episode_and_invalid_state_is_failure",
        },
        "claims": {"ranking": False, "policy_quality": False, "success_threshold_changed": False},
    }
    immutable_json(root / "remediation_manifest.json", payload)
    return payload


def environment(workspace: Path, root: Path, old: Path, overlay: Path) -> dict[str, str]:
    runtime = Runtime(workspace, old)
    keep = {"LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "EMBODICHAIN_DATA_ROOT", "MPLCONFIGDIR",
            "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "PYTHONFAULTHANDLER"}
    env = {key: value for key, value in runtime.environment().items() if key in keep}
    inherited_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.update(
        PATH=os.pathsep.join((str(root / "bin"), inherited_path)),
        PYTHONPATH=os.pathsep.join((str(overlay), str(workspace / "autosim"),
                   str(workspace / "AutoSimSOTA/RoboSynChallenge"), str(old / "native/RoboTwin"))),
        TORCH_EXTENSIONS_DIR=str(root / "torch_extensions"), XDG_CACHE_HOME=str(root / "runtime_cache"),
        HF_DATASETS_CACHE=str(root / "dataset_cache"),
    )
    return env


def build_native_jobs(workspace: Path, root: Path, old: Path, v3: Path, executor: Executor,
                      sources: dict[str, str], env: dict[str, str]) -> list[Job]:
    native = old / "native/RoboTwin"
    interpreter = old / "runtime_env/bin/python"
    base = [str(interpreter), "-m", "autosim.experiment_system.robotwin_worker"]
    shared = ["--repo", str(native), "--attempt", "{attempt}"]
    jobs = []
    for index, task in enumerate(ROBOTWIN_TASKS):
        contract = RoboTwinPlugin(native).contract(task)
        inputs = {**sources, **contract.sources}
        inputs.update({str(path.absolute()): digest(path) for path in
                       (old / "runtime_env/lib/python3.10/site-packages/curobo/curobolib").glob("*.so")})
        if task in {"pick_dual_bottles", "stack_blocks_two"}:
            inherited = v3 / "jobs" / f"robotwin_{task}_probe/commit.json"
            inputs[str(inherited)] = digest(inherited)

        def job(name, stage, command, dependencies=(), gpu="0", timeout=1800, pinned=None):
            return Job(name, stage, command, str(native), {"result.json": "result"}, pinned or inputs,
                       dependencies=dependencies, environment=env, timeout_seconds=timeout,
                       budget_seconds=timeout, gpu=gpu)

        prefix, task_args = f"robotwin_{task}", [*shared, "--task", task]
        dependencies = ()
        if task == "open_laptop":
            probe = prefix + "_probe_v2"
            probe_command = [str(interpreter), "-m", "autosim.experiment_validation.robotwin_probe_v2",
                             "--repo", str(native), "--attempt", "{attempt}", "--task", task,
                             "--seed", str(92_006_000 + index * 10_000)]
            jobs.append(job(probe, "probe", probe_command, timeout=600))
            if not get_result(executor, probe):
                continue
            dependencies = (probe,)
        collect = prefix + "_collect_v4"
        jobs.append(job(collect, "collect", [*base, "collect", *task_args, "--episodes", "3",
                        "--seed", str(92_006_100 + index * 10_000)], dependencies, timeout=2400))
        collected = get_result(executor, collect)
        if not collected:
            continue
        collection_result = executor.root / collect / executor.committed(collect)["attempt"] / "result.json"
        convert = prefix + "_convert_v4"
        jobs.append(job(convert, "admit", [*base, "convert", *task_args,
                        "--collection-result", str(collection_result)], (collect,), gpu=None, timeout=1800))
        converted = get_result(executor, convert)
        if not converted:
            continue
        dataset = converted["dataset"]
        backend = [str(workspace / "AutoSimSOTA/.venv/bin/python"),
                   "-m", "autosim.experiment_system.backend_worker"]
        common = ["--workspace", str(workspace), "--benchmark", "robotwin", "--native-repo", str(native),
                  "--task", task, "--attempt", "{attempt}", "--dataset", dataset]
        admit = prefix + "_admit_v4"
        jobs.append(job(admit, "admit", [*backend, "admit", *common], (convert,), gpu=None, timeout=600))
        admitted = get_result(executor, admit)
        if not admitted:
            continue
        train_sources = {**inputs, **admitted["verified_files"]}
        for policy in ("act", "dp"):
            train = prefix + f"_{policy}_train_v4"
            jobs.append(job(train, "train", [*backend, "train", *common, "--policy", policy,
                            "--steps", "200", "--seed", "91006100"], (admit,), timeout=2400,
                            pinned=train_sources))
            trained = get_result(executor, train)
            if not trained:
                continue
            evaluate = prefix + f"_{policy}_evaluate_v4"
            eval_sources = {**train_sources, **trained["verified_files"]}
            jobs.append(job(evaluate, "evaluate", [*base, "evaluate", *task_args,
                            "--checkpoint", trained["checkpoint"], "--policy", policy,
                            "--episodes", "3", "--seed", str(94_006_000 + index * 10_000)],
                            (train,), timeout=1800, pinned=eval_sources))
    return jobs


def build_jobs(workspace: Path, root: Path, old: Path, v3: Path, overlay: Path,
               executor: Executor, manifest: dict) -> list[Job]:
    sources = {**manifest["sources"], **manifest["warp_overlay"]["files"],
               str(root / "remediation_manifest.json"): digest(root / "remediation_manifest.json")}
    env = environment(workspace, root, old, overlay)
    jobs = build_native_jobs(workspace, root, old, v3, executor, sources, env)
    old_dp = get_result(Executor(old / "jobs"), "robosyn_click_dp_train")
    checkpoint = Path(old_dp["checkpoint"])
    safe_sources = {**sources, **old_dp["verified_files"]}
    command = [str(workspace / "AutoSimSOTA/.venv/bin/python"),
               "-m", "autosim.experiment_validation.remediation_worker_v2",
               "robosyn-isolated-safe-evaluate", "--workspace", str(workspace),
               "--attempt", "{attempt}", "--task", "click_bell", "--checkpoint", str(checkpoint),
               "--policy", "dp", "--episodes", "3", "--seed", "91006200"]
    jobs.append(Job("robosyn_click_dp_evaluate_isolated_v4", "evaluate", command,
                    str(workspace / "AutoSimSOTA/RoboSynChallenge_eval_clean"),
                    {"result.json": "result"}, safe_sources, environment=env,
                    timeout_seconds=5400, budget_seconds=5400, gpu="0"))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--warp-overlay", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    workspace, root, old, v3, overlay = (args.workspace.absolute(), args.root.absolute(),
        args.old_root.absolute(), args.v3_root.absolute(), args.warp_overlay.absolute())
    if not 0 < args.max_hours <= 72:
        raise ValueError("queue wall-clock budget must be in (0, 72] hours")
    manifest = prepare_manifest(workspace, root, old, v3, overlay)
    pinned = {**manifest["sources"], **manifest["warp_overlay"]["files"],
              str(root / "remediation_manifest.json"): digest(root / "remediation_manifest.json")}
    executor = Executor(root / "jobs")
    state = {"schema_version": 4, "started_at": now(), "status": "running",
             "ranking_eligible": False, "formal_experiments_complete": False,
             "predecessors": {key: value["state_sha256"] for key, value in manifest["predecessors"].items()}}
    started = time.monotonic()
    with lease(root / "queue.lock"):
        while time.monotonic() - started < args.max_hours * 3600:
            assert_frozen(pinned)
            statuses = {}
            for job in build_jobs(workspace, root, old, v3, overlay, executor, manifest):
                statuses[job.name] = executor.run(job)
                state.update(jobs=statuses, updated_at=now())
                atomic_json(root / "queue_state.json", state)
            wanted = {"robosyn_click_dp_evaluate_isolated_v4"}
            wanted.update(f"robotwin_{task}_{policy}_evaluate_v4"
                          for task in ROBOTWIN_TASKS for policy in ("act", "dp"))
            if all(statuses.get(name, {}).get("status") == "committed" for name in wanted):
                state.update(status="integration_gate_reached", inherited_original=list(OLD_REQUIRED),
                             inherited_v3=list(INHERITED_PROBES),
                             remaining=["formal_policy_experiments", "independent_operator_measurement"])
                break
            terminal = {"requires_adapter_or_data_fix", "requires_artifact_audit",
                        "requires_audit_unknown_execution", "quarantine_evaluation", "budget_exhausted"}
            if statuses and all(row.get("status") == "committed" or row.get("status") in terminal
                                for row in statuses.values()):
                if {job.name for job in build_jobs(workspace, root, old, v3, overlay, executor, manifest)} == set(statuses):
                    state["status"] = "requires_capability_review"
                    break
            time.sleep(10)
        else:
            state["status"] = "queue_wait_budget_exhausted"
        state["updated_at"] = now()
        atomic_json(root / "queue_state.json", state)


if __name__ == "__main__":
    main()
