import json

import numpy as np

from autosim.research.task_diagnostics import task_failure_analysis


def _pose(x=0.0, y=0.0, z=0.0):
    value = np.eye(4)
    value[:3, 3] = [x, y, z]
    return value.tolist()


def test_water_plugin_uses_task_specific_observations(tmp_path):
    metrics = {
        "purpose": "development", "artifact_directory": str(tmp_path),
        "summary": {"episode_count": 1, "success_count": 0, "success_rate": 0.0},
        "episodes": [{"episode_seed": 7, "success": False}],
    }
    (tmp_path / "evaluation_metrics.json").write_text(json.dumps(metrics))
    row = {"seed": 7, "entities": {
        "bottle": {"pose": _pose(z=0.5)}, "cup": {"pose": _pose(z=0.5)}}}
    (tmp_path / "telemetry.jsonl").write_text(json.dumps(row) + "\n")
    result = task_failure_analysis("water_pouring", tmp_path, tmp_path / "analysis.json")
    assert result["task_diagnostics"]["plugin"] == "water_pouring_v1"
    assert result["task_diagnostics"]["categories"]["bottle_not_lifted"] == 1


def test_unknown_task_does_not_invent_semantics(tmp_path):
    metrics = {
        "purpose": "development", "artifact_directory": str(tmp_path),
        "summary": {"episode_count": 1, "success_count": 1, "success_rate": 1.0},
        "episodes": [{"episode_seed": 3, "success": True}],
    }
    (tmp_path / "evaluation_metrics.json").write_text(json.dumps(metrics))
    result = task_failure_analysis("item_assembly", tmp_path, tmp_path / "analysis.json")
    assert result["task_diagnostics"]["measurement_status"] == "no_verified_task_semantic_plugin"
