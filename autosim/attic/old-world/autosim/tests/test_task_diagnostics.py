"""The evidence layer reports raw observations for every task alike.

It used to recognise failures only for tasks somebody had written an analyzer
for, and answered `no_verified_task_semantic_plugin` otherwise. These tests pin
the replacement contract: one shape of evidence for all tasks, no invented
failure modes, and the task's own success predicate quoted rather than guessed.
"""
import json

import numpy as np

from autosim.research.task_diagnostics import task_evidence


def _pose(x=0.0, y=0.0, z=0.0):
    value = np.eye(4)
    value[:3, 3] = [x, y, z]
    return value.tolist()


def _evaluation(tmp_path, episodes, telemetry):
    metrics = {
        "purpose": "development", "artifact_directory": str(tmp_path),
        "summary": {"episode_count": len(episodes),
                    "success_count": sum(e["success"] for e in episodes),
                    "success_rate": 0.0},
        "episodes": episodes,
    }
    (tmp_path / "evaluation_metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry))
    return task_evidence


def test_evidence_names_no_failure_modes(tmp_path):
    """Raw measurements only: the controller decides what counts as a failure."""
    _evaluation(tmp_path, [{"episode_seed": 7, "success": False}],
                [{"seed": 7, "step": 0, "entities": {"bottle": {"pose": _pose(z=0.5)},
                                                     "cup": {"pose": _pose(z=0.5)}}},
                 {"seed": 7, "step": 10, "entities": {"bottle": {"pose": _pose(z=0.5)},
                                                      "cup": {"pose": _pose(z=0.5)}}}])
    result = task_evidence("water_pouring", tmp_path, tmp_path / "analysis.json")
    evidence = result["task_evidence"]
    assert evidence["measurement_status"] == "raw_observations_no_categories_assigned"
    # entity names come from the telemetry, not from a task-name branch
    assert sorted(evidence["episodes"][0]["trajectory"]["entities"]) == ["bottle", "cup"]
    assert "categories" not in evidence
    assert "plugin" not in evidence


def test_every_task_receives_the_same_shape_of_evidence(tmp_path):
    """A task nobody wrote an analyzer for is no longer a second-class citizen."""
    _evaluation(tmp_path, [{"episode_seed": 3, "success": True}], [])
    known = task_evidence("water_pouring", tmp_path, tmp_path / "a.json")["task_evidence"]
    unknown = task_evidence("item_assembly", tmp_path, tmp_path / "b.json")["task_evidence"]
    assert sorted(known) == sorted(unknown)
    assert unknown["measurement_status"] == known["measurement_status"]


def test_contrast_separates_success_from_failure_cohorts(tmp_path):
    episodes = [{"episode_seed": 1, "success": True}, {"episode_seed": 2, "success": False}]
    telemetry = [
        {"seed": 1, "step": 0, "entities": {"cube": {"pose": _pose(z=0.1)}}, "robot_qpos": [[0.0]]},
        {"seed": 1, "step": 9, "entities": {"cube": {"pose": _pose(z=0.4)}}, "robot_qpos": [[0.3]]},
        {"seed": 2, "step": 0, "entities": {"cube": {"pose": _pose(z=0.1)}}, "robot_qpos": [[0.0]]},
        {"seed": 2, "step": 9, "entities": {"cube": {"pose": _pose(z=0.11)}}, "robot_qpos": [[0.01]]},
    ]
    _evaluation(tmp_path, episodes, telemetry)
    contrast = task_evidence("item_assembly", tmp_path, tmp_path / "a.json")["task_evidence"]["contrast"]
    assert contrast["counts"] == {"success": 1, "failure": 1}
    moved = contrast["success_cohort"]["cube.max_displacement_from_start_m"]["mean"]
    still = contrast["failure_cohort"]["cube.max_displacement_from_start_m"]["mean"]
    assert moved > still


def test_success_predicate_is_quoted_not_hashed(tmp_path):
    """The predicate source is the ground truth for 'worked'; show it, don't hash it."""
    repo = tmp_path / "repo"
    task_dir = repo / "robosynchallenge" / "tasks" / "demo_task"
    task_dir.mkdir(parents=True)
    (task_dir / "demo_task.py").write_text(
        "class Demo:\n"
        "    def is_task_success(self):\n"
        "        return self._cup_filled and not self._spilled\n")
    _evaluation(tmp_path, [{"episode_seed": 1, "success": True}], [])
    evidence = task_evidence("demo_task", tmp_path, tmp_path / "a.json", repo=repo)["task_evidence"]
    source = evidence["success_predicate_source"]
    assert source["status"] == "quoted"
    assert "_cup_filled and not self._spilled" in source["functions"]["is_task_success"]


def test_missing_repository_is_reported_not_faked(tmp_path):
    _evaluation(tmp_path, [{"episode_seed": 1, "success": True}], [])
    evidence = task_evidence("demo_task", tmp_path, tmp_path / "a.json", repo=None)["task_evidence"]
    assert evidence["success_predicate_source"]["status"] == "repository_not_supplied"
