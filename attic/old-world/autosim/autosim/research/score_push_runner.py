"""Resume-safe task runner for the frozen RoboSyn score-push stages."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .common import atomic_json, now, read_json
from .score_push import collect_capability
from .score_push_final import confirm, evaluate_final_development, select_final, train_final
from .score_push_pipeline import analyze_screen, evaluate_screen, mixtures, production, train_screen


DEFAULT_ORDER = (
    "handle_basket", "water_pouring", "manipulate_pipette",
    "drawer_open_place", "items_handover", "mixer_operating",
    "table_rearrangement", "sample_loading", "item_assembly",
)


def _record(path: Path, state: dict, *, stage: str, status: str, detail=None) -> None:
    state.update(stage=stage, status=status, updated_at=now())
    if detail is not None:
        state.setdefault("stages", {})[stage] = detail
    atomic_json(path, state)


def run_task(workspace: Path, root: Path, task: str) -> dict:
    task_root = root / task
    state_path = task_root / "runner_state.json"
    state = read_json(state_path) if state_path.is_file() else {
        "task": task, "status": "running", "created_at": now(), "stages": {}}
    try:
        _record(state_path, state, stage="capability", status="running")
        capability = collect_capability(workspace, root, task)
        _record(state_path, state, stage="capability", status="running", detail=capability)
        if capability.get("decision") != "scale":
            reason = capability.get("decision", capability.get("status", "expert_repair_required"))
            state.update(status="awaiting_expert_repair", stage="stopped_after_capability_gate",
                         stop_reason=reason, completed_at=now())
            atomic_json(state_path, state)
            return state

        stages = (
            ("production", lambda: production(workspace, root, task)),
            ("mixtures", lambda: mixtures(workspace, root, task)),
            ("train_screen", lambda: train_screen(workspace, root, task)),
            ("evaluate_screen", lambda: evaluate_screen(workspace, root, task)),
            ("analyze_screen", lambda: analyze_screen(root, task)),
        )
        for name, function in stages:
            _record(state_path, state, stage=name, status="running")
            _record(state_path, state, stage=name, status="running", detail=function())
        screen = state["stages"]["analyze_screen"]
        if screen.get("selected_arm") is None:
            state.update(status="completed_no_improvement", stage="screen_gate_closed",
                         stop_reason="no mixed-data arm passed frozen 20K gate", completed_at=now())
            atomic_json(state_path, state)
            return state

        final_stages = (
            ("train_final", lambda: train_final(workspace, root, task)),
            ("evaluate_final_development", lambda: evaluate_final_development(workspace, root, task)),
            ("select_final", lambda: select_final(root, task)),
        )
        for name, function in final_stages:
            _record(state_path, state, stage=name, status="running")
            _record(state_path, state, stage=name, status="running", detail=function())
        if state["stages"]["select_final"].get("selected_candidate") is None:
            state.update(status="completed_no_improvement", stage="final_development_gate_closed",
                         stop_reason="no 80K arm passed frozen development gate", completed_at=now())
            atomic_json(state_path, state)
            return state
        _record(state_path, state, stage="confirmation", status="running")
        confirmation = confirm(workspace, root, task)
        _record(state_path, state, stage="confirmation", status="completed", detail=confirmation)
        state.update(status="completed", completed_at=now(), deployment=confirmation["deployment"])
        atomic_json(state_path, state)
        return state
    except Exception as exc:
        state.update(status="failed", failed_stage=state.get("stage"), finished_at=now(),
                     error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        atomic_json(state_path, state)
        raise


def run_campaign(workspace: Path, root: Path, tasks: tuple[str, ...]) -> dict:
    campaign_path = root / "runner_campaign_state.json"
    campaign = read_json(campaign_path) if campaign_path.is_file() else {
        "kind": "robosyn_score_push_runner", "created_at": now(), "tasks": {}}
    for task in tasks:
        campaign.update(current_task=task, status="running", updated_at=now())
        atomic_json(campaign_path, campaign)
        try:
            campaign["tasks"][task] = run_task(workspace, root, task)
        except Exception as exc:
            campaign["tasks"][task] = read_json(root / task / "runner_state.json")
            campaign.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_task=task)
            atomic_json(campaign_path, campaign)
            raise
        atomic_json(campaign_path, campaign)
    campaign.update(status="completed", current_task=None, completed_at=now())
    atomic_json(campaign_path, campaign)
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=DEFAULT_ORDER)
    args = parser.parse_args()
    tasks = tuple(args.task) if args.task else DEFAULT_ORDER
    result = run_campaign(args.workspace.resolve(), args.root.resolve(), tasks)
    print({"status": result["status"], "tasks": {key: value["status"]
                                                   for key, value in result["tasks"].items()}})


if __name__ == "__main__":
    main()
