"""Unattended gated continuation of collection, smoke tests and pilot research.

Waits for an existing suite without interrupting it. Failed tasks remain failed;
only a declared training adapter repair gets one separately logged retry.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import atomic_json, exclusive, now, read_json, redact
from .registry import TASK_IDS, load_task
from .runtime import Runtime


def advance(runtime: Runtime, *, wait_hours=4):
    directory = runtime.output / "continuation"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state.json"
    state = {"started_at": now(), "status": "running", "steps": []}

    def phase(command: str, tasks: list[str], label: str, extra=(), timeout=36000):
        state.update(stage=label)
        atomic_json(path, state)
        args = [str(runtime.python), "-m", "autosim.research.cli", command,
                "--workspace", str(runtime.workspace), "--output", str(runtime.output),
                "--gpu", runtime.gpu, "--tasks", *tasks, *extra]
        try:
            runtime.run(args, directory / label, timeout)
            result = "completed"
        except Exception as exc:
            result = redact(f"{type(exc).__name__}: {exc}")
        state["steps"].append({"phase": label, "result": result, "finished_at": now()})
        atomic_json(path, state)

    with exclusive(directory / "controller.lock"):
        deadline = time.monotonic() + wait_hours * 3600
        while True:
            try:
                with exclusive(runtime.output / "suite.lock"):
                    break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    state.update(status="blocked", stage="existing_suite_wait_deadline")
                    atomic_json(path, state)
                    return state
                state.update(stage="waiting_for_existing_suite", heartbeat_at=now())
                atomic_json(path, state)
                time.sleep(10)
        smoke = read_json(runtime.output / "smoke_status.json")["tasks"]
        repairs = [name for name, row in smoke.items() if row["status"] == "failed"
                   and row.get("stage") == "collection" and int(row.get("attempt", 1)) == 1
                   and load_task(runtime.repo, name).expert_adapter != "official"]
        if repairs:
            phase("collect", repairs, "declared_expert_repairs", ["--retry-failed"], timeout=7200)
        phase("smoke", list(TASK_IDS), "ten_task_smoke")
        smoke = read_json(runtime.output / "smoke_status.json")["tasks"]
        representatives = [name for name in ("click_bell", "drawer_open_place", "sample_loading")
                           if smoke.get(name, {}).get("status") == "passed"]
        if representatives:
            phase("pilot", representatives, "representative_two_round_pilot", timeout=36000)
        state.update(status="gated_handoff", stage="integration_review_required",
                     blocked_tasks={name: row for name, row in smoke.items() if row["status"] != "passed"},
                     production_research_started=False,
                     reason="Review actual integration/pilot results and freeze core before multi-day production training",
                     finished_at=now())
        atomic_json(path, state)
        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.workspace / "autosim/output/robosyn_general_20260905"
    advance(Runtime(args.workspace, output))
