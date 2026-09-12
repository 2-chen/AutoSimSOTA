"""Task-semantic observation plugins; none of these replace native judges."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze
from .common import atomic_json, read_json


def _matrix(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1, 4, 4)[0]


def _traces(evaluation: Path) -> tuple[dict, dict[int, list[dict]]]:
    metrics = read_json(evaluation / "evaluation_metrics.json")
    artifact = Path(metrics.get("artifact_directory", evaluation))
    rows: dict[int, list[dict]] = defaultdict(list)
    path = artifact / "telemetry.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[int(row["seed"])].append(row)
    return metrics, rows


def _water(metrics: dict, traces: dict[int, list[dict]]) -> dict[str, Any]:
    categories = Counter()
    episodes = []
    for episode in metrics["episodes"]:
        if episode["success"]:
            continue
        rows = [row for row in traces[episode["episode_seed"]]
                if "bottle" in row.get("entities", {}) and "cup" in row.get("entities", {})]
        if not rows:
            category = "insufficient_task_telemetry"
            detail = {}
        else:
            bottle = [_matrix(row["entities"]["bottle"]["pose"]) for row in rows]
            cup = [_matrix(row["entities"]["cup"]["pose"]) for row in rows]
            initial_z = bottle[0][2, 3]
            lift = max(pose[2, 3] - initial_z for pose in bottle)
            angles = [float(np.arccos(np.clip(pose[2, 1], -1, 1))) for pose in bottle]
            cup_angles = [float(np.arccos(np.clip(pose[2, 2], -1, 1))) for pose in cup]
            distances = []
            for bottle_pose, cup_pose in zip(bottle, cup):
                mouth = bottle_pose[:3, 3] + 0.236 * bottle_pose[:3, 1]
                distances.append(float(np.linalg.norm((cup_pose[:3, 3] - mouth)[:2])))
            if max(cup_angles) >= np.pi / 4:
                category = "cup_fall_observed"
            elif lift <= 0.03:
                category = "bottle_not_lifted"
            elif min(distances) >= 0.08:
                category = "bottle_mouth_not_aligned_over_cup"
            elif max(angles) <= np.pi / 4:
                category = "pouring_tilt_not_observed"
            elif angles[-1] >= np.pi / 4:
                category = "upright_return_not_observed"
            else:
                category = "incomplete_despite_sampled_event_sequence"
            detail = {"max_bottle_lift_m": float(lift),
                      "max_bottle_tilt_rad": max(angles),
                      "final_bottle_tilt_rad": angles[-1],
                      "min_mouth_to_cup_xy_m": min(distances),
                      "max_cup_tilt_rad": max(cup_angles)}
        categories[category] += 1
        episodes.append({"episode_seed": episode["episode_seed"], "category": category, **detail})
    return {"plugin": "water_pouring_v1", "categories": dict(categories), "episodes": episodes,
            "measurement_status": "exploratory_sampled_task_events_not_native_judge"}


def _handle(metrics: dict, traces: dict[int, list[dict]]) -> dict[str, Any]:
    categories = Counter()
    episodes = []
    for episode in metrics["episodes"]:
        if episode["success"]:
            continue
        rows = [row for row in traces[episode["episode_seed"]]
                if "basket" in row.get("entities", {}) and "milk" in row.get("entities", {})]
        if not rows:
            category = "insufficient_task_telemetry"
            detail = {}
        else:
            basket = [_matrix(row["entities"]["basket"]["pose"]) for row in rows]
            milk = [_matrix(row["entities"]["milk"]["pose"]) for row in rows]
            y_move = max(pose[1, 3] - basket[0][1, 3] for pose in basket)
            lift = max(pose[2, 3] - basket[0][2, 3] for pose in basket)
            distances = [float(np.linalg.norm(milk_pose[:2, 3] - basket_pose[:2, 3]))
                         for basket_pose, milk_pose in zip(basket, milk)]
            final_in = distances[-1] < 0.10 and milk[-1][2, 3] > basket[-1][2, 3]
            if lift <= 0.01:
                category = "basket_lift_not_observed"
            elif y_move <= 0.15:
                category = "basket_transport_not_observed"
            elif not final_in:
                category = "milk_not_observed_in_basket_at_end"
            else:
                category = "stability_or_unsampled_condition_incomplete"
            detail = {"max_basket_lift_m": float(lift), "max_basket_y_displacement_m": float(y_move),
                      "min_milk_basket_xy_m": min(distances), "final_milk_in_basket_observation": bool(final_in)}
        categories[category] += 1
        episodes.append({"episode_seed": episode["episode_seed"], "category": category, **detail})
    return {"plugin": "handle_basket_v1", "categories": dict(categories), "episodes": episodes,
            "measurement_status": "exploratory_sampled_task_events_not_native_judge"}


def task_failure_analysis(task: str, evaluation: Path, output: Path) -> dict[str, Any]:
    result = analyze(evaluation)
    if task in {"water_pouring", "handle_basket"}:
        metrics, traces = _traces(evaluation)
        result["task_diagnostics"] = (_water(metrics, traces) if task == "water_pouring"
                                      else _handle(metrics, traces))
    else:
        result["task_diagnostics"] = {
            "plugin": None, "measurement_status": "no_verified_task_semantic_plugin"}
    atomic_json(output, result)
    return result
