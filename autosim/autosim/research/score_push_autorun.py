"""Failure-isolated executor for the frozen RoboSyn score-push campaign.

Scientific decisions stay in score_push_runner and score_push_fallback.  This
module only makes their task-level execution exhaustive: one failed task is
recorded but cannot prevent later tasks from producing positive or negative
evidence.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .common import atomic_json, now, read_json
from .score_push_fallback import run as run_fallback
from .score_push_runner import DEFAULT_ORDER, run_task


def execute_task(workspace: Path, root: Path, task: str) -> dict:
    output = root / task / "task_execution_summary.json"
    if output.is_file():
        previous = read_json(output)
        if previous.get("status") in {"completed", "completed_no_improvement"}:
            return previous

    runner = run_task(workspace, root, task)
    result = {
        "task": task,
        "status": runner["status"],
        "route": "self_collection_screen",
        "runner_state": str((root / task / "runner_state.json").resolve()),
        "new_data_used": runner["status"] not in {"awaiting_expert_repair", "failed"},
    }
    if runner["status"] == "awaiting_expert_repair":
        fallback = run_fallback(workspace, root, task)
        result.update(
            status=fallback["status"],
            route="official_data_only_fallback",
            new_data_used=False,
            fallback=fallback,
        )
    result["completed_at"] = now()
    atomic_json(output, result)
    return result


def run_campaign(workspace: Path, root: Path, tasks: tuple[str, ...]) -> dict:
    workspace, root = workspace.resolve(), root.resolve()
    state_path = root / "autorun_campaign_state.json"
    state = read_json(state_path) if state_path.is_file() else {
        "schema_version": 1,
        "kind": "robosyn_score_push_failure_isolated_autorun",
        "created_at": now(),
        "tasks": {},
    }
    for task in tasks:
        state.update(status="running", current_task=task, updated_at=now())
        atomic_json(state_path, state)
        try:
            state["tasks"][task] = execute_task(workspace, root, task)
        except Exception as exc:  # preserve failure and continue the frozen task list
            failure = {
                "task": task,
                "status": "failed",
                "failed_at": now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            state["tasks"][task] = failure
            atomic_json(root / task / "task_execution_summary.json", failure)
        atomic_json(state_path, state)
    state.update(status="completed_with_failures" if any(
        item.get("status") == "failed" for item in state["tasks"].values()) else "completed",
        current_task=None, completed_at=now())
    atomic_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=DEFAULT_ORDER)
    args = parser.parse_args()
    tasks = tuple(args.task) if args.task else DEFAULT_ORDER
    result = run_campaign(args.workspace, args.root, tasks)
    print({"status": result["status"], "tasks": {
        key: value["status"] for key, value in result["tasks"].items()}})


if __name__ == "__main__":
    main()
