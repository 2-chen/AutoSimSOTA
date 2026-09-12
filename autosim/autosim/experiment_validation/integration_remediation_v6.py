"""Integration remediation v6 with an explicit RoboSyn policy namespace."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from autosim.experiment_system.executor import Executor, Job, lease
from autosim.experiment_system.plugins import ROBOTWIN_TASKS, RoboTwinPlugin
from autosim.experiment_validation.integration_remediation import file_manifest, get_result
from autosim.experiment_validation.integration_remediation_v4 import environment
from autosim.research.common import assert_frozen, atomic_json, digest, immutable_json, now, read_json


V4_TRAINS = tuple(
    f"robotwin_{task}_{policy}_train_v4"
    for task in ("pick_dual_bottles", "stack_blocks_two")
    for policy in ("act", "dp")
)
V5_REUSED = (
    "robotwin_open_laptop_collect_v5", "robotwin_open_laptop_convert_v5",
    "robotwin_open_laptop_admit_v5", "robotwin_open_laptop_act_train_v5",
    "robotwin_open_laptop_dp_train_v5", "robosyn_click_dp_evaluate_isolated_v5",
)


def prepare_manifest(workspace: Path, root: Path, old: Path, v4: Path, v5: Path,
                     overlay: Path) -> dict:
    if read_json(v4 / "queue_state.json").get("status") != "requires_capability_review":
        raise ValueError("v4 predecessor is not terminal")
    if read_json(v5 / "queue_state.json").get("status") != "requires_capability_review":
        raise ValueError("v5 predecessor is not terminal")
    v4_executor, v5_executor = Executor(v4 / "jobs"), Executor(v5 / "jobs")
    commits = {"v4": {}, "v5": {}}
    for name in V4_TRAINS:
        if not v4_executor.committed(name):
            raise ValueError(f"missing v4 training commit: {name}")
        commits["v4"][name] = digest(v4_executor.root / name / "commit.json")
    for name in V5_REUSED:
        if not v5_executor.committed(name):
            raise ValueError(f"missing v5 commit: {name}")
        commits["v5"][name] = digest(v5_executor.root / name / "commit.json")
    bridge = Path(__file__).with_name("policy_namespace")
    ffmpeg = Path("/home/wbc/miniconda3/envs/robotwin5090/lib/python3.10/site-packages/"
                  "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
    shim = root / "bin/ffmpeg"
    if not shim.is_symlink() or shim.resolve() != ffmpeg:
        raise ValueError("invalid v6 ffmpeg shim")
    sources = [
        Path(__file__), Path(__file__).with_name("integration_remediation.py"),
        Path(__file__).with_name("integration_remediation_v4.py"),
        Path(__file__).with_name("integration_remediation_v5.py"),
        Path(__file__).with_name("robotwin_evaluate_v2.py"), bridge / "policy/__init__.py",
        bridge / "README.md", workspace / "autosim/autosim/experiment_system/executor.py",
        workspace / "autosim/autosim/experiment_system/native_policy.py",
        workspace / "autosim/autosim/experiment_system/plugins.py",
        workspace / "autosim/autosim/experiment_system/robotwin_worker.py",
        workspace / "autosim/autosim/experiment_system/worker.py",
        workspace / "autosim/autosim/research/policy_rpc.py", ffmpeg, shim,
        v4 / "queue_state.json", v5 / "queue_state.json",
        workspace / "AutoSimSOTA/RoboSynChallenge/policy/act/deploy_policy.py",
        workspace / "AutoSimSOTA/RoboSynChallenge/policy/dp/deploy_policy.py",
        workspace / "AutoSimSOTA/RoboSynChallenge/policy/inference_timing.py",
    ]
    failed = {}
    for task in ("pick_dual_bottles", "stack_blocks_two"):
        for policy in ("act", "dp"):
            name = f"robotwin_{task}_{policy}_evaluate_v5"
            attempt = v5 / "jobs" / name / "attempt_001"
            failed[name] = {str(path.absolute()): digest(path) for path in
                            (attempt / "request.json", attempt / "receipt.json", attempt / "stdout.log")}
    payload = {
        "schema_version": 6, "created_for": "native_integration_remediation",
        "predecessors": {
            "v4": {"path": str(v4), "state_sha256": digest(v4 / "queue_state.json"),
                   "reused_training_commits": commits["v4"]},
            "v5": {"path": str(v5), "state_sha256": digest(v5 / "queue_state.json"),
                   "reused_commits": commits["v5"], "failed_namespace_evidence": failed},
        },
        "warp_overlay": {"path": str(overlay), "version": "1.12.0",
                         "files": file_manifest(overlay)},
        "policy_namespace": {
            "bridge": str(bridge), "target": str(workspace / "AutoSimSOTA/RoboSynChallenge/policy"),
            "reason": "implicit_namespace_was_shadowed_by_RoboTwin_regular_policy_package",
        },
        "ffmpeg": {"shim": str(shim), "target": str(ffmpeg), "sha256": digest(ffmpeg)},
        "sources": {str(path.absolute()): digest(path) for path in sources},
        "remediation": "regular_package_bridge_with_hash_pinned_original_submodules",
        "claims": {"ranking": False, "policy_quality": False,
                   "success_threshold_changed": False, "evaluator_changed": False},
    }
    immutable_json(root / "remediation_manifest.json", payload)
    return payload


def evaluation_job(workspace: Path, root: Path, old: Path, training_root: Path,
                   task: str, policy: str, sources: dict[str, str], env: dict[str, str]) -> Job:
    trainer = Executor(training_root / "jobs")
    suffix = "v5" if task == "open_laptop" else "v4"
    train_name = f"robotwin_{task}_{policy}_train_{suffix}"
    trained = get_result(trainer, train_name)
    checkpoint = Path(trained["checkpoint"])
    contract = RoboTwinPlugin(old / "native/RoboTwin").contract(task)
    pinned = {**sources, **contract.sources, **trained["verified_files"],
              str(trainer.root / train_name / "commit.json"):
              digest(trainer.root / train_name / "commit.json")}
    command = [str(old / "runtime_env/bin/python"),
               "-m", "autosim.experiment_validation.robotwin_evaluate_v2",
               "--repo", str(old / "native/RoboTwin"),
               "--policy-repo", str(workspace / "AutoSimSOTA/RoboSynChallenge"),
               "--attempt", "{attempt}", "--task", task, "--checkpoint", str(checkpoint),
               "--policy", policy, "--episodes", "3",
               "--seed", str(95_006_000 + ROBOTWIN_TASKS.index(task) * 10_000)]
    return Job(f"robotwin_{task}_{policy}_evaluate_v6", "evaluate", command, str(workspace),
               {"result.json": "result"}, pinned, environment=env,
               timeout_seconds=1800, budget_seconds=1800, gpu="0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--warp-overlay", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=12)
    args = parser.parse_args()
    workspace, root, old, v4, v5, overlay = (args.workspace.absolute(), args.root.absolute(),
        args.old_root.absolute(), args.v4_root.absolute(), args.v5_root.absolute(),
        args.warp_overlay.absolute())
    manifest = prepare_manifest(workspace, root, old, v4, v5, overlay)
    pinned = {**manifest["sources"], **manifest["warp_overlay"]["files"],
              str(root / "remediation_manifest.json"): digest(root / "remediation_manifest.json")}
    env = environment(workspace, root, old, overlay)
    bridge = Path(__file__).with_name("policy_namespace")
    env["PYTHONPATH"] = os.pathsep.join((str(bridge), env["PYTHONPATH"]))
    env["AUTOSIM_POLICY_PACKAGE_ROOT"] = str(workspace / "AutoSimSOTA/RoboSynChallenge/policy")
    executor = Executor(root / "jobs")
    state = {"schema_version": 6, "started_at": now(), "status": "running",
             "ranking_eligible": False, "formal_experiments_complete": False,
             "inherited_robosyn_dp": manifest["predecessors"]["v5"]["reused_commits"]
                                      ["robosyn_click_dp_evaluate_isolated_v5"]}
    started = time.monotonic()
    with lease(root / "queue.lock"):
        while time.monotonic() - started < args.max_hours * 3600:
            assert_frozen(pinned)
            jobs = [evaluation_job(workspace, root, old, v5 if task == "open_laptop" else v4,
                                   task, policy, pinned, env)
                    for task in ROBOTWIN_TASKS for policy in ("act", "dp")]
            statuses = {}
            for job in jobs:
                statuses[job.name] = executor.run(job)
                state.update(jobs=statuses, updated_at=now())
                atomic_json(root / "queue_state.json", state)
            if all(row.get("status") == "committed" for row in statuses.values()):
                state.update(status="integration_gate_reached",
                             inherited=[*V4_TRAINS, *V5_REUSED],
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
