"""Durably audit ACT checkpoints at their valid in-progress/final moments."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from autosim.research.common import assert_frozen, atomic_json, freeze_files, now, read_json

from .checkpoint_audit import audit_checkpoint
from .intermediate_checkpoint_audit import audit_intermediate_checkpoint
from .queue_manifest import versioned_manifest


def checkpoint_path(arm: Path, step: int) -> Path:
    return arm / f"train/checkpoints/{step:06d}/pretrained_model"


def intermediate_ready(checkpoint: Path) -> bool:
    step_root = checkpoint.parent
    required = (
        checkpoint / "config.json",
        checkpoint / "train_config.json",
        checkpoint / "model.safetensors",
        step_root / "training_state/training_step.json",
        step_root / "training_state/optimizer_param_groups.json",
        step_root / "training_state/optimizer_state.safetensors",
        step_root / "training_state/rng_state.safetensors",
    )
    last = checkpoint.parents[1] / "last"
    return (all(path.is_file() for path in required) and last.is_symlink()
            and last.resolve() == step_root.resolve())


def queue(arm: Path, output: Path, *, target_step: int, seed: int,
          intermediate_steps: list[int], max_hours: float, poll_seconds: float) -> dict:
    if not 0 < max_hours <= 168 or not 1 <= poll_seconds <= 60:
        raise ValueError("invalid bounded checkpoint watcher settings")
    if any(step <= 0 or step >= target_step for step in intermediate_steps):
        raise ValueError("intermediate steps must be positive and precede target")
    local = Path(__file__).resolve()
    sources = freeze_files([local, local.with_name("checkpoint_audit.py"),
                            local.with_name("intermediate_checkpoint_audit.py"),
                            local.with_name("queue_manifest.py")])
    versioned_manifest(output, {"arm": str(arm), "target_step": target_step, "seed": seed,
                                "intermediate_steps": intermediate_steps, "sources": sources,
                                "policy_performance_claim": False, "sota_claim": False})
    state_path = output / "state.json"
    prior = read_json(state_path) if state_path.is_file() else {}
    if prior.get("status") in {"completed", "requires_review", "queue_wait_budget_exhausted"}:
        return prior
    deadline = float(prior.get("deadline_epoch", time.time() + max_hours * 3600))
    process_path = arm / f"process_train_{target_step}/process.json"
    state = {**prior, "status": "running", "started_at": prior.get("started_at", now()),
             "deadline_epoch": deadline, "audited_steps": prior.get("audited_steps", [])}
    while time.time() < deadline:
        assert_frozen(sources)
        process = read_json(process_path) if process_path.is_file() else {}
        for step in intermediate_steps:
            report_path = output / f"checkpoint_{step:06d}.json"
            checkpoint = checkpoint_path(arm, step)
            if not report_path.exists() and process.get("status") == "running" and intermediate_ready(checkpoint):
                result = audit_intermediate_checkpoint(
                    checkpoint, process_path, expected_serialized_step=step,
                    expected_target_step=target_step, expected_seed=seed,
                    require_resume="--resume" in process.get("command", []),
                )
                atomic_json(report_path, result)
                if not result["passed"]:
                    state.update(status="requires_review", stage=f"intermediate_audit_failed:{step}",
                                 failed_report=str(report_path), updated_at=now())
                    atomic_json(state_path, state)
                    return state
                state["audited_steps"] = sorted(set(state["audited_steps"] + [step]))
                atomic_json(state_path, state)
        if process.get("status") == "failed" or (process.get("returncode") not in {None, 0}):
            state.update(status="requires_review", stage="training_process_failed",
                         process_status=process.get("status"), returncode=process.get("returncode"),
                         updated_at=now())
            atomic_json(state_path, state)
            return state
        final_checkpoint = checkpoint_path(arm, target_step)
        if process.get("status") == "completed" and process.get("returncode") == 0:
            final = audit_checkpoint(final_checkpoint, target_step, seed)
            atomic_json(output / f"checkpoint_{target_step:06d}.json", final)
            if not final["passed"]:
                state.update(status="requires_review", stage="final_audit_failed",
                             failed_report=str(output / f"checkpoint_{target_step:06d}.json"),
                             updated_at=now())
            else:
                state.update(status="completed", stage="all_requested_checkpoints_audited",
                             final_step=target_step, final_checkpoint=str(final_checkpoint),
                             updated_at=now())
            atomic_json(state_path, state)
            return state
        state.update(stage="waiting_training_checkpoint", process_status=process.get("status", "missing"),
                     process_pid=process.get("pid"), audited_steps=state["audited_steps"], updated_at=now())
        atomic_json(state_path, state)
        time.sleep(poll_seconds)
    state.update(status="queue_wait_budget_exhausted", updated_at=now())
    atomic_json(state_path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--intermediate-step", type=int, action="append", default=[])
    parser.add_argument("--max-hours", type=float, default=48)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    queue(args.arm.absolute(), args.output.absolute(), target_step=args.target_step,
          seed=args.seed, intermediate_steps=args.intermediate_step,
          max_hours=args.max_hours, poll_seconds=args.poll_seconds)
