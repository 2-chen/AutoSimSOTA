"""Queue validation after production; never opens final tests or retries failures."""
import argparse
import time
from pathlib import Path

from autosim.research.common import atomic_json, exclusive, now, read_json
from autosim.research.registry import TASK_IDS
from autosim.research.runtime import Runtime


def optional(path):
    return read_json(path) if path.exists() else {}


def next_action(production, research, audit, repeats):
    if production.get("status") not in {"gated_handoff", "blocked"}:
        return "waiting_production"
    if research.get("status") != "completed_development" or research.get("pilot_only") is not False:
        return "blocked_production_incomplete"
    if research.get("completed_rounds", 0) < 2:
        return "blocked_production_incomplete"
    if not audit:
        return "determinism"
    if audit.get("status") == "failed":
        return "blocked_audit_process"
    if repeats.get("status") == "failed":
        return "blocked_repeat_training"
    if repeats.get("status") != "completed_training_repeats":
        return "train-repeats"
    return "awaiting_final_protocol_and_semantic_review"


def run(runtime, *, wait_hours=1000):
    directory = runtime.output / "validation_continuation"
    state_path = directory / "state.json"
    start = time.monotonic()
    with exclusive(directory / "controller.lock"):
        state = optional(state_path) or {"created_at": now(), "steps": []}
        while time.monotonic() - start < wait_hours * 3600:
            production = optional(runtime.output / "production_continuation/state.json")
            tasks, pending = {}, []
            for task in TASK_IDS:
                research = optional(runtime.output / "research/train_seed_1000" / task / "state.json")
                validation = runtime.output / "validation" / task
                audit = optional(validation / "determinism/audit.json")
                repeats = optional(validation / "fixed_recipe_repeats/state.json")
                action = next_action(production, research, audit, repeats)
                tasks[task] = {"next_action": action}
                if action in {"determinism", "train-repeats"}:
                    pending.append((task, action, research))
            state.update(status="running", tasks=tasks, heartbeat_at=now(), final_evaluation="not_run",
                         all_tasks_improved=False, repeat_process_hours_per_task=48,
                         remaining_acceptance=["locked multi-training-seed final evaluation", "semantic review",
                             "representative data/processing attribution", "DP training adapter"])
            atomic_json(state_path, state)
            if not pending:
                if any(r["next_action"] == "waiting_production" for r in tasks.values()):
                    time.sleep(30)
                    continue
                state.update(status="gated_handoff", finished_at=now())
                atomic_json(state_path, state)
                return state
            task, action, research = min(pending, key=lambda r: (r[1] != "determinism", list(TASK_IDS).index(r[0])))
            output = directory / f"phase_{len(state['steps']):03d}_{task}_{action}"
            command = [str(runtime.python), "-m", "autosim.research_validation.cli", action,
                       "--workspace", str(runtime.workspace), "--output", str(runtime.output), "--task", task]
            if action == "determinism":
                command += ["--checkpoint", research["baseline_checkpoint"]]
            state.update(stage=f"{task}:{action}")
            atomic_json(state_path, state)
            try:
                runtime.run(command, output, 7200 if action == "determinism" else 48 * 3600 + 600)
                result_path = runtime.output / "validation" / task / (
                    "determinism/audit.json" if action == "determinism" else "fixed_recipe_repeats/state.json")
                result = optional(result_path)
                if not result or result.get("status") in {"failed", "running"}:
                    raise RuntimeError("validation process did not commit a successful completion")
                state["steps"].append({"task": task, "action": action, "output": str(output), "finished_at": now()})
                atomic_json(state_path, state)
            except Exception as exc:
                # A partial evaluation/training is evidence, not a retry request.
                state.update(status="blocked", stage="partial_validation_requires_audit",
                             error_type=type(exc).__name__, failed_output=str(output), finished_at=now())
                atomic_json(state_path, state)
                return state
        state.update(status="blocked", stage="validation_queue_wall_clock_budget_exhausted", finished_at=now())
        atomic_json(state_path, state)
        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(Runtime(args.workspace, args.output or args.workspace / "autosim/output/robosyn_general_20260905"))
