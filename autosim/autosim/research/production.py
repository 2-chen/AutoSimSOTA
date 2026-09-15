"""Gated, restartable single-GPU continuation through development experiments.

Never runs final tests automatically: training repeats, determinism and semantic
review remain separate acceptance gates, not inferred from a successful process.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import assert_frozen, atomic_json, exclusive, freeze_files, now, read_json, redact
from .controller import source_tree_files
from .official_smoke import official_assets
from .registry import TASK_IDS
from .runtime import Runtime


def next_action(smoke, fallback, pilot, research, data_ready):
    """Pure gate logic; a failed experiment is never silently retried."""
    full = smoke.get("status") == "passed"
    policy_only = fallback.get("status") == "passed"
    if not (full or policy_only):
        if fallback.get("status") == "failed":
            return "blocked_policy_integration"
        return "official-smoke" if data_ready else "waiting_official_assets"
    if pilot.get("status") == "failed":
        return "blocked_pilot"
    if pilot.get("status") != "completed_development":
        return "pilot"
    if int(pilot.get("completed_rounds", 0)) < 2:
        return "blocked_incomplete_pilot"
    if research.get("status") == "failed":
        return "blocked_research"
    if research.get("status") == "completed_development":
        return "development_complete"
    return "research" if data_ready else "waiting_official_assets"


def optional(path):
    return read_json(path) if path.exists() else {}


def run(runtime, tasks, *, wait_hours=24, suite_hours=420):
    directory = runtime.output / "production_continuation"
    path = directory / "state.json"
    state = optional(path) or {"created_at": now(), "steps": [], "tasks": {}}
    protocol_path = directory / "frozen_core.json"
    if not protocol_path.exists():
        files = list(Path(__file__).parent.glob("*.py"))
        files += list(Path(__file__).parent.parent.glob("robosyn*.py"))
        files += source_tree_files(runtime.repo / "policy/act", ".py")
        files += [runtime.repo / "scripts/run_env.py", runtime.repo / "policy/act/scripts/train.py"]
        atomic_json(protocol_path, freeze_files(files))
    frozen = read_json(protocol_path)
    started, idle_since = time.monotonic(), time.monotonic()
    with exclusive(directory / "controller.lock"):
        while time.monotonic() - started < suite_hours * 3600:
            assert_frozen(frozen)
            state.update(status="running", heartbeat_at=now(), final_evaluation="not_run",
                         all_tasks_improved=False,
                         remaining_acceptance=["all ten tasks", "fixed-recipe training repeats",
                                               "determinism", "semantic review", "locked final tests", "DP adapter"])
            # Wait for previously launched integration/pilot controller. It owns
            # the continuation lock even in gaps between GPU subprocesses.
            try:
                with exclusive(runtime.output / "continuation/controller.lock"), exclusive(runtime.output / "suite.lock"):
                    pass
            except RuntimeError:
                state.update(stage="waiting_existing_controller")
                atomic_json(path, state)
                time.sleep(10)
                continue
            smoke = optional(runtime.output / "smoke_status.json").get("tasks", {})
            fallback = optional(runtime.output / "official_policy_smoke_status.json").get("tasks", {})
            pending = []
            for task in tasks:
                pilot = optional(runtime.output / "pilot_research/train_seed_1000" / task / "state.json")
                research = optional(runtime.output / "research/train_seed_1000" / task / "state.json")
                root, _ = official_assets(runtime, task)
                action = next_action(smoke.get(task, {}), fallback.get(task, {}), pilot, research, root is not None)
                state["tasks"][task] = {"next_action": action,
                    "automatic_collection_validated": smoke.get(task, {}).get("status") == "passed"}
                if action in {"official-smoke", "pilot", "research"}:
                    pending.append((task, action))
            atomic_json(path, state)
            if not pending:
                if any(r["next_action"] == "waiting_official_assets" for r in state["tasks"].values()):
                    if time.monotonic() - idle_since < wait_hours * 3600:
                        state.update(stage="waiting_official_assets")
                        atomic_json(path, state)
                        time.sleep(10)
                        continue
                    state.update(status="blocked", stage="official_asset_wait_budget_exhausted")
                else:
                    state.update(status="gated_handoff", stage="development_finished_or_blocked")
                break
            # Finish ready integration/pilots before starting a many-hour job.
            task, command = min(pending, key=lambda row: ({"official-smoke": 0, "pilot": 1, "research": 2}[row[1]], tasks.index(row[0])))
            state.update(stage=f"{task}:{command}")
            atomic_json(path, state)
            index = len(state["steps"])
            destination = directory / f"phase_{index:03d}_{task}_{command}"
            args = [str(runtime.python), "-m", "autosim.research.cli", command,
                    "--workspace", str(runtime.workspace), "--output", str(runtime.output),
                    "--gpu", runtime.gpu, "--tasks", task]
            try:
                runtime.run(args, destination, {"official-smoke": 7200, "pilot": 11400, "research": 145200}[command])
                result = "completed"
            except Exception as exc:
                result = redact(f"{type(exc).__name__}: {exc}")
            state["steps"].append({"task": task, "phase": command, "result": result,
                                    "directory": str(destination), "finished_at": now()})
            atomic_json(path, state)
            if result != "completed":
                # Per-task state normally carries the failure. If the process
                # died before that commit, stop rather than repeat a partial job.
                state_file = (runtime.output / "official_policy_smoke_status.json" if command == "official-smoke" else
                    runtime.output / ("pilot_research" if command == "pilot" else "research") / "train_seed_1000" / task / "state.json")
                row = optional(state_file)
                row = row.get("tasks", {}).get(task, {}) if command == "official-smoke" else row
                if row.get("status") != "failed":
                    state.update(status="blocked", stage="partial_process_requires_audit")
                    break
            idle_since = time.monotonic()
        else:
            state.update(status="blocked", stage="suite_wall_clock_budget_exhausted")
        state["finished_at"] = now()
        atomic_json(path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_IDS), default=list(TASK_IDS))
    parser.add_argument("--wait-hours", type=float, default=24)
    args = parser.parse_args()
    if not 0 < args.wait_hours <= 168:
        parser.error("wait-hours must be in (0, 168]")
    run(Runtime(args.workspace, args.output or args.workspace / "autosim/output/robosyn_general_20260905"),
        args.tasks, wait_hours=args.wait_hours)
