"""Integration remediation v7: enter RoboTwin cwd before native imports."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from autosim.experiment_system.executor import Executor, Job, lease
from autosim.experiment_system.plugins import ROBOTWIN_TASKS, RoboTwinPlugin
from autosim.experiment_validation.integration_remediation import file_manifest, get_result
from autosim.experiment_validation.integration_remediation_v4 import environment
from autosim.experiment_validation.integration_remediation_v6 import V4_TRAINS, V5_REUSED
from autosim.research.common import assert_frozen, atomic_json, digest, immutable_json, now, read_json


def prepare_manifest(workspace: Path, root: Path, old: Path, v4: Path, v5: Path,
                     v6: Path, overlay: Path) -> dict:
    if read_json(v6 / "queue_state.json").get("status") != "requires_capability_review":
        raise ValueError("v6 predecessor is not terminal")
    executors = {"v4": Executor(v4 / "jobs"), "v5": Executor(v5 / "jobs")}
    reused = {"v4": {}, "v5": {}}
    for version, names in (("v4", V4_TRAINS), ("v5", V5_REUSED)):
        for name in names:
            if not executors[version].committed(name):
                raise ValueError(f"missing {version} commit: {name}")
            reused[version][name] = digest(executors[version].root / name / "commit.json")
    bridge = Path(__file__).with_name("policy_namespace")
    ffmpeg = Path("/home/wbc/miniconda3/envs/robotwin5090/lib/python3.10/site-packages/"
                  "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
    shim = root / "bin/ffmpeg"
    if not shim.is_symlink() or shim.resolve() != ffmpeg:
        raise ValueError("invalid v7 ffmpeg shim")
    sources = [
        Path(__file__), Path(__file__).with_name("integration_remediation.py"),
        Path(__file__).with_name("integration_remediation_v4.py"),
        Path(__file__).with_name("integration_remediation_v6.py"),
        Path(__file__).with_name("robotwin_evaluate_v3.py"), bridge / "policy/__init__.py",
        bridge / "README.md", workspace / "autosim/autosim/experiment_system/executor.py",
        workspace / "autosim/autosim/experiment_system/native_policy.py",
        workspace / "autosim/autosim/experiment_system/plugins.py",
        workspace / "autosim/autosim/experiment_system/robotwin_worker.py",
        workspace / "autosim/autosim/experiment_system/worker.py",
        workspace / "autosim/autosim/research/policy_rpc.py", ffmpeg, shim,
        v6 / "queue_state.json",
        workspace / "AutoSimSOTA/RoboSynChallenge/policy/act/deploy_policy.py",
        workspace / "AutoSimSOTA/RoboSynChallenge/policy/dp/deploy_policy.py",
        workspace / "AutoSimSOTA/RoboSynChallenge/policy/inference_timing.py",
    ]
    failed = {}
    for task in ROBOTWIN_TASKS:
        for policy in ("act", "dp"):
            name = f"robotwin_{task}_{policy}_evaluate_v6"
            attempt = v6 / "jobs" / name / "attempt_001"
            failed[name] = {str(path.absolute()): digest(path) for path in
                            (attempt / "request.json", attempt / "receipt.json", attempt / "stdout.log")}
    payload = {
        "schema_version": 7, "created_for": "native_integration_remediation",
        "predecessor": {"path": str(v6), "state_sha256": digest(v6 / "queue_state.json"),
                        "failed_native_import_evidence": failed},
        "reused_commits": reused,
        "warp_overlay": {"path": str(overlay), "version": "1.12.0",
                         "files": file_manifest(overlay)},
        "policy_namespace": {"bridge": str(bridge),
                             "target": str(workspace / "AutoSimSOTA/RoboSynChallenge/policy")},
        "ffmpeg": {"shim": str(shim), "target": str(ffmpeg), "sha256": digest(ffmpeg)},
        "sources": {str(path.absolute()): digest(path) for path in sources},
        "remediation": "chdir_to_native_repo_before_importing_modules_that_read_relative_assets",
        "claims": {"ranking": False, "policy_quality": False,
                   "success_threshold_changed": False, "evaluator_changed": False},
    }
    immutable_json(root / "remediation_manifest.json", payload)
    return payload


def evaluation_job(workspace: Path, old: Path, training_root: Path, task: str, policy: str,
                   sources: dict[str, str], env: dict[str, str]) -> Job:
    trainer = Executor(training_root / "jobs")
    suffix = "v5" if task == "open_laptop" else "v4"
    train_name = f"robotwin_{task}_{policy}_train_{suffix}"
    trained = get_result(trainer, train_name)
    contract = RoboTwinPlugin(old / "native/RoboTwin").contract(task)
    pinned = {**sources, **contract.sources, **trained["verified_files"],
              str(trainer.root / train_name / "commit.json"):
              digest(trainer.root / train_name / "commit.json")}
    command = [str(old / "runtime_env/bin/python"),
               "-m", "autosim.experiment_validation.robotwin_evaluate_v3",
               "--repo", str(old / "native/RoboTwin"),
               "--policy-repo", str(workspace / "AutoSimSOTA/RoboSynChallenge"),
               "--attempt", "{attempt}", "--task", task,
               "--checkpoint", trained["checkpoint"], "--policy", policy, "--episodes", "3",
               "--seed", str(96_006_000 + ROBOTWIN_TASKS.index(task) * 10_000)]
    return Job(f"robotwin_{task}_{policy}_evaluate_v7", "evaluate", command, str(workspace),
               {"result.json": "result"}, pinned, environment=env,
               timeout_seconds=1800, budget_seconds=1800, gpu="0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "root", "old-root", "v4-root", "v5-root", "v6-root", "warp-overlay"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=12)
    args = parser.parse_args()
    workspace, root, old, v4, v5, v6, overlay = (args.workspace.absolute(), args.root.absolute(),
        args.old_root.absolute(), args.v4_root.absolute(), args.v5_root.absolute(),
        args.v6_root.absolute(), args.warp_overlay.absolute())
    manifest = prepare_manifest(workspace, root, old, v4, v5, v6, overlay)
    pinned = {**manifest["sources"], **manifest["warp_overlay"]["files"],
              str(root / "remediation_manifest.json"): digest(root / "remediation_manifest.json")}
    env = environment(workspace, root, old, overlay)
    bridge = Path(__file__).with_name("policy_namespace")
    env["PYTHONPATH"] = os.pathsep.join((str(bridge), env["PYTHONPATH"]))
    env["AUTOSIM_POLICY_PACKAGE_ROOT"] = str(workspace / "AutoSimSOTA/RoboSynChallenge/policy")
    executor, started = Executor(root / "jobs"), time.monotonic()
    state = {"schema_version": 7, "started_at": now(), "status": "running",
             "ranking_eligible": False, "formal_experiments_complete": False,
             "inherited_robosyn_dp": manifest["reused_commits"]["v5"]
                                      ["robosyn_click_dp_evaluate_isolated_v5"]}
    with lease(root / "queue.lock"):
        while time.monotonic() - started < args.max_hours * 3600:
            assert_frozen(pinned)
            jobs = [evaluation_job(workspace, old, v5 if task == "open_laptop" else v4,
                                   task, policy, pinned, env)
                    for task in ROBOTWIN_TASKS for policy in ("act", "dp")]
            statuses = {}
            for job in jobs:
                statuses[job.name] = executor.run(job)
                state.update(jobs=statuses, updated_at=now())
                atomic_json(root / "queue_state.json", state)
            if all(row.get("status") == "committed" for row in statuses.values()):
                state.update(status="integration_gate_reached", inherited=[*V4_TRAINS, *V5_REUSED],
                             remaining=["formal_policy_experiments", "independent_operator_measurement"])
                break
            terminal = {"requires_adapter_or_data_fix", "requires_artifact_audit",
                        "requires_audit_unknown_execution", "quarantine_evaluation", "budget_exhausted"}
            if statuses and all(row.get("status") in terminal or row.get("status") == "committed"
                                for row in statuses.values()):
                state["status"] = "requires_capability_review"
                break
            time.sleep(10)
        else:
            state["status"] = "queue_wait_budget_exhausted"
        state["updated_at"] = now()
        atomic_json(root / "queue_state.json", state)


if __name__ == "__main__":
    main()
