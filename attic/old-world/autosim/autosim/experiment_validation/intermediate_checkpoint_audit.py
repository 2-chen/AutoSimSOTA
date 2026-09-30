"""Audit a serialized checkpoint from a training process that is still running.

This is deliberately separate from the final-checkpoint audit: passing here only
means that a resumable intermediate snapshot was serialized consistently.  It
must never be used as evidence that the requested training budget completed or
that the policy has any behavioral performance.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from safetensors import safe_open

from autosim.research.common import atomic_json, digest, now, read_json
from .fairness_audit import command_value


def _tensor_header(path: Path) -> dict:
    with safe_open(path, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        elements = sum(math.prod(tensors.get_slice(key).get_shape()) for key in keys)
        dtypes = sorted({str(tensors.get_slice(key).get_dtype()) for key in keys})
    return {
        "tensor_count": len(keys),
        "elements": elements,
        "dtypes": dtypes,
        "bytes": path.stat().st_size,
    }


def audit_intermediate_checkpoint(
    checkpoint: Path,
    process_path: Path,
    *,
    expected_serialized_step: int,
    expected_target_step: int,
    expected_seed: int,
    require_resume: bool = False,
) -> dict:
    """Validate an intermediate ACT checkpoint without claiming final completion."""
    checkpoint = checkpoint.absolute()
    process_path = process_path.absolute()
    step_root = checkpoint.parent
    state_root = step_root / "training_state"
    paths = {
        "process": process_path,
        "training_step": state_root / "training_step.json",
        "config": checkpoint / "config.json",
        "train_config": checkpoint / "train_config.json",
        "weights": checkpoint / "model.safetensors",
        "optimizer_groups": state_root / "optimizer_param_groups.json",
        "optimizer_state": state_root / "optimizer_state.safetensors",
        "rng_state": state_root / "rng_state.safetensors",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    base = {
        "updated_at": now(),
        "kind": "in_progress_intermediate_checkpoint_integrity_audit",
        "checkpoint": str(checkpoint),
        "process_path": str(process_path),
        "expected_serialized_step": expected_serialized_step,
        "expected_target_step": expected_target_step,
        "expected_seed": expected_seed,
        "training_complete_claim": False,
        "policy_performance_claim": False,
        "official_leaderboard_result": False,
        "sota_claim": False,
    }
    if missing:
        return {**base, "status": "failed", "passed": False, "missing": missing}

    process = read_json(paths["process"])
    step = read_json(paths["training_step"])
    config = read_json(paths["config"])
    train_config = read_json(paths["train_config"])
    optimizer_groups = read_json(paths["optimizer_groups"])
    command = process.get("command", [])
    pid = process.get("pid")
    inputs = config.get("input_features", {})
    visual = {name: value for name, value in inputs.items() if value.get("type") == "VISUAL"}
    states = [value for value in inputs.values() if value.get("type") == "STATE"]
    action = config.get("output_features", {}).get("action", {})
    last = checkpoint.parents[1] / "last"
    process_alive = isinstance(pid, int) and pid > 0 and Path(f"/proc/{pid}").exists()
    checks = [
        {"name": "process_declares_running", "passed": process.get("status") == "running" and
         process.get("returncode") is None, "actual_status": process.get("status"),
         "actual_returncode": process.get("returncode")},
        {"name": "process_pid_alive_at_audit", "passed": process_alive, "pid": pid},
        {"name": "process_target_step", "passed": int(command_value(process, "--steps")) == expected_target_step,
         "actual": int(command_value(process, "--steps")), "expected": expected_target_step},
        {"name": "process_seed", "passed": int(command_value(process, "--seed")) == expected_seed,
         "actual": int(command_value(process, "--seed")), "expected": expected_seed},
        {"name": "serialized_step_precedes_target", "passed": 0 < expected_serialized_step < expected_target_step,
         "serialized": expected_serialized_step, "target": expected_target_step},
        {"name": "serialized_training_step", "passed": int(step.get("step", -1)) == expected_serialized_step,
         "actual": step.get("step"), "expected": expected_serialized_step},
        {"name": "checkpoint_directory_step", "passed": step_root.name == f"{expected_serialized_step:06d}",
         "actual": step_root.name, "expected": f"{expected_serialized_step:06d}"},
        {"name": "last_pointer_at_audit", "passed": last.is_symlink() and last.resolve() == step_root.resolve(),
         "actual": str(last.resolve()) if last.exists() else "missing", "expected": str(step_root)},
        {"name": "train_config_target_step", "passed": int(train_config.get("steps", -1)) == expected_target_step,
         "actual": train_config.get("steps"), "expected": expected_target_step},
        {"name": "train_config_seed", "passed": int(train_config.get("seed", -1)) == expected_seed,
         "actual": train_config.get("seed"), "expected": expected_seed},
        {"name": "resume_contract", "passed": (not require_resume) or command.count("--resume") == 1,
         "required": require_resume, "actual_count": command.count("--resume")},
        {"name": "state_contract", "passed": len(states) == 1 and states[0].get("shape") == [14],
         "actual": [value.get("shape") for value in states], "expected": [[14]]},
        {"name": "camera_contract", "passed": len(visual) == 3 and
         all(value.get("shape") == [3, 480, 640] for value in visual.values()),
         "actual": {name: value.get("shape") for name, value in visual.items()}, "expected_count": 3},
        {"name": "action_contract", "passed": action.get("type") == "ACTION" and
         action.get("shape") == [14], "actual": action, "expected": {"type": "ACTION", "shape": [14]}},
        {"name": "optimizer_param_groups", "passed": isinstance(optimizer_groups, list) and
         bool(optimizer_groups) and all(isinstance(group, dict) for group in optimizer_groups),
         "group_count": len(optimizer_groups) if isinstance(optimizer_groups, list) else None},
    ]
    headers = {}
    for name in ("weights", "optimizer_state", "rng_state"):
        try:
            header = _tensor_header(paths[name])
            headers[name] = header
            checks.append({"name": f"{name}_header", "passed": header["tensor_count"] > 0 and
                           header["elements"] > 0, **header})
        except Exception as exc:  # corrupted or concurrently incomplete serialization
            checks.append({"name": f"{name}_header", "passed": False,
                           "error": f"{type(exc).__name__}: {exc}"})
    failures = [check for check in checks if not check["passed"]]
    return {
        **base,
        "status": "intermediate_serialization_passed" if not failures else "failed",
        "passed": not failures,
        "checks": checks,
        "failed_checks": len(failures),
        "headers": headers,
        "evidence": {str(path): digest(path) for path in paths.values()},
        "scope_note": "serialization/resume integrity only; training is still in progress and final/behavior gates remain closed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--process", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serialized-step", type=int, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--require-resume", action="store_true")
    args = parser.parse_args()
    result = audit_intermediate_checkpoint(
        args.checkpoint,
        args.process,
        expected_serialized_step=args.serialized_step,
        expected_target_step=args.target_step,
        expected_seed=args.seed,
        require_resume=args.require_resume,
    )
    atomic_json(args.output.absolute(), result)
    raise SystemExit(0 if result["passed"] else 1)
