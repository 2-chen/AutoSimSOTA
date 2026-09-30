"""Raw task evidence for the controller to interpret.

This module deliberately names no failure modes and applies no thresholds.  It
reports *what was observed* -- the task's own success predicate, the entities the
task declared, and the trajectories those entities followed -- and leaves the
interpretation to the controller.  Hand-written per-task analyzers used to live
here; they fixed the set of recognisable failures in advance and silently
degenerated to `no_verified_task_semantic_plugin` for every task nobody had
written one for, so the system's reach was capped by how many analyzers existed.

Everything below is derived from artifacts the pipeline already records:
`telemetry.jsonl` files entities under the names the task's own config gives
them, which is why nothing here needs to know what a "bottle" or a "basket" is.
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze
from .common import atomic_json, read_json

#: Predicate functions worth quoting to the controller, in the order the
#: benchmark defines them.  Missing ones are simply absent from the evidence.
_PREDICATE_NAMES = ("is_task_success", "_evaluate_task_state", "_evaluate_success")


def _matrix(value: Any) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float64)
    if array.size < 16:
        return None
    return array.reshape(-1, 4, 4)[0]


def _traces(evaluation: Path) -> tuple[dict, dict[int, list[dict]]]:
    metrics = read_json(evaluation / "evaluation_metrics.json")
    artifact = Path(metrics.get("artifact_directory", evaluation))
    rows: dict[int, list[dict]] = defaultdict(list)
    path = artifact / "telemetry.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["seed"])].append(row)
    return metrics, rows


def success_predicate_source(repo: Path | None, task: str) -> dict[str, Any]:
    """Quote the task's own success predicate.

    The predicate is the ground truth for what "worked" means, and the pipeline
    already hashes this file for the frozen protocol.  Quoting it costs nothing
    beyond reading a file it already read, and it replaces the hand-copied
    constants that task-specific analyzers used to duplicate (and drift from).
    """
    if repo is None:
        return {"status": "repository_not_supplied"}
    path = Path(repo) / "robosynchallenge" / "tasks" / task / f"{task}.py"
    if not path.is_file():
        return {"status": "predicate_source_not_found", "expected": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {"status": "predicate_source_unparsable", "error": str(exc), "path": str(path)}
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _PREDICATE_NAMES:
            segment = ast.get_source_segment(text, node)
            if segment:
                found[node.name] = segment
    if not found:
        return {"status": "no_predicate_function_found", "path": str(path),
                "looked_for": list(_PREDICATE_NAMES)}
    return {"status": "quoted", "path": str(path), "functions": found}


def _trajectory(rows: list[dict]) -> dict[str, Any]:
    """Per-entity displacement and final state over one episode's samples."""
    if not rows:
        return {"samples": 0}
    ordered = sorted(rows, key=lambda row: row.get("step") or 0)
    names: set[str] = set()
    for row in ordered:
        names.update((row.get("entities") or {}).keys())

    entities: dict[str, Any] = {}
    for name in sorted(names):
        poses = []
        for row in ordered:
            entry = (row.get("entities") or {}).get(name) or {}
            pose = _matrix(entry.get("pose"))
            if pose is not None:
                poses.append(pose)
        if not poses:
            entities[name] = {"observed": False}
            continue
        start, end = poses[0], poses[-1]
        translations = np.asarray([pose[:3, 3] for pose in poses])
        entities[name] = {
            "observed": True,
            "sample_count": len(poses),
            "start_xyz": [round(float(v), 5) for v in start[:3, 3]],
            "final_xyz": [round(float(v), 5) for v in end[:3, 3]],
            "max_displacement_from_start_m": round(
                float(np.linalg.norm(translations - translations[0], axis=1).max()), 5),
            "net_displacement_m": round(float(np.linalg.norm(end[:3, 3] - start[:3, 3])), 5),
            "min_z_m": round(float(translations[:, 2].min()), 5),
            "max_z_m": round(float(translations[:, 2].max()), 5),
            "start_up_axis": [round(float(v), 4) for v in start[:3, 2]],
            "final_up_axis": [round(float(v), 4) for v in end[:3, 2]],
        }

    qpos = [np.asarray(row["robot_qpos"], dtype=np.float64).ravel()
            for row in ordered if row.get("robot_qpos") is not None]
    joint = {}
    if qpos:
        stacked = np.vstack(qpos)
        joint = {
            "dim": int(stacked.shape[1]),
            "max_abs_change": round(float(np.abs(stacked - stacked[0]).max()), 5),
            "final_abs_change": round(float(np.abs(stacked[-1] - stacked[0]).max()), 5),
        }

    missing: set[str] = set()
    unavailable: set[str] = set()
    for row in ordered:
        missing.update(row.get("missing") or [])
        unavailable.update(str(v) for v in (row.get("unavailable_realized_parameters") or []))
    return {
        "samples": len(ordered),
        "step_range": [ordered[0].get("step"), ordered[-1].get("step")],
        "entities": entities,
        "robot_joint": joint,
        "missing": sorted(missing),
        "unavailable_realized_parameters": sorted(unavailable),
    }


def _contrast(episodes: list[dict]) -> dict[str, Any]:
    """How measured quantities differ between episodes that succeeded and failed.

    Reported as raw distributions.  Whether any of these differences is the
    reason for failure is a judgement the controller makes, not one encoded here.
    """
    def collect(cohort: list[dict]) -> dict[str, Any]:
        totals: dict[str, list[float]] = defaultdict(list)
        for episode in cohort:
            trajectory = episode["trajectory"]
            for name, stats in (trajectory.get("entities") or {}).items():
                if not stats.get("observed"):
                    continue
                totals[f"{name}.max_displacement_from_start_m"].append(
                    stats["max_displacement_from_start_m"])
                totals[f"{name}.net_displacement_m"].append(stats["net_displacement_m"])
                totals[f"{name}.z_range_m"].append(round(stats["max_z_m"] - stats["min_z_m"], 5))
            joint = trajectory.get("robot_joint") or {}
            if "max_abs_change" in joint:
                totals["robot_joint.max_abs_change"].append(joint["max_abs_change"])
        return {
            name: {"n": len(values), "mean": round(float(np.mean(values)), 5),
                   "min": round(float(np.min(values)), 5),
                   "max": round(float(np.max(values)), 5)}
            for name, values in sorted(totals.items()) if values
        }

    succeeded = [e for e in episodes if e["success"]]
    failed = [e for e in episodes if not e["success"]]
    return {"success_cohort": collect(succeeded), "failure_cohort": collect(failed),
            "counts": {"success": len(succeeded), "failure": len(failed)}}


def task_evidence(task: str, evaluation: Path, output: Path,
                  repo: Path | None = None) -> dict[str, Any]:
    """Assemble everything the controller can legitimately reason from.

    `analyze()` is deliberately not merged in: it labels each episode with a fixed
    set of failure categories, and inheriting those would hand the controller the
    same pre-named answer this module exists to stop supplying.
    """
    legacy = analyze(evaluation)
    metrics, traces = _traces(evaluation)
    episodes = []
    for episode in metrics.get("episodes", []):
        seed = int(episode["episode_seed"])
        episodes.append({"episode_seed": seed, "success": bool(episode["success"]),
                         "action_steps": episode.get("action_steps"),
                         "trajectory": _trajectory(traces.get(seed) or [])})
    result = {
        "source": legacy.get("source"), "purpose": legacy.get("purpose"),
        "summary": legacy.get("summary"),
        "task_evidence": {
            "task": task,
            "measurement_status": "raw_observations_no_categories_assigned",
            "success_predicate_source": success_predicate_source(repo, task),
            "episodes": episodes,
            "contrast": _contrast(episodes),
            "unlike_the_old_plugin": (
                "no failure modes are named and no thresholds are applied here; derive both "
                "from success_predicate_source and the contrast distributions"),
        },
    }
    atomic_json(output, result)
    return result
