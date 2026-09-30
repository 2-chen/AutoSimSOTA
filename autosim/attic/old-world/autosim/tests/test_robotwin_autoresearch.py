from pathlib import Path
from contextlib import nullcontext

import pytest

from autosim.research.robotwin_autoresearch import RoboTwinAutoResearch


PROJECT = Path(__file__).resolve().parents[2]


def test_runner_freezes_two_native_evaluation_banks(tmp_path):
    runner = RoboTwinAutoResearch(
        project_root=PROJECT, repo=PROJECT / "RoboTwin", output_root=tmp_path,
        run_id="runner_contract", task="beat_block_hammer", controller="heuristic",
        gpu="0", hours=1, train_seed=1000, training_epochs=2000,
        evaluation_episodes=3, allow_api_egress=False)
    runner.initialize()
    import json
    protocol = json.loads((runner.run_root / "frozen_protocol.json").read_text())
    assert protocol["development_evaluation_seed"] == 0
    assert protocol["confirmation_evaluation_seed"] == 1
    assert protocol["final_seed_feedback_allowed"] is False


def test_runner_rejects_path_like_run_id(tmp_path):
    with pytest.raises(ValueError, match="path component"):
        RoboTwinAutoResearch(
            project_root=PROJECT, repo=PROJECT / "RoboTwin", output_root=tmp_path,
            run_id="../escape", task="beat_block_hammer", controller="fixed", gpu="0",
            hours=1, train_seed=1, training_epochs=2000, evaluation_episodes=1,
            allow_api_egress=False)


def test_execute_opens_confirmation_only_after_two_setting_gain(tmp_path, monkeypatch):
    runner = RoboTwinAutoResearch(
        project_root=PROJECT, repo=PROJECT / "RoboTwin", output_root=tmp_path,
        run_id="execute_contract", task="beat_block_hammer", controller="heuristic",
        gpu="0", hours=1, train_seed=1, training_epochs=2000,
        evaluation_episodes=1, allow_api_egress=False)
    monkeypatch.setattr("autosim.research.robotwin_autoresearch.gpu_lock",
                        lambda gpu: nullcontext())
    monkeypatch.setattr(runner, "initialize", lambda: {"task_contract": {}})
    monkeypatch.setattr(runner, "_baseline", lambda: Path("baseline"))
    calls = []

    def evaluate_pair(checkpoint, label, purpose, *, eval_seed=0, seed_banks=None):
        calls.append((label, purpose, eval_seed, seed_banks))
        rows = [{"episode_seed": 10, "success": label != "baseline"}]
        return {"demo_clean": {"label": label, "metrics": {"episodes": rows}},
                "demo_randomized": {"label": label, "metrics": {"episodes": rows}}}

    monkeypatch.setattr(runner, "_evaluate_pair", evaluate_pair)
    monkeypatch.setattr(runner, "_proposal", lambda evidence, discovery: {
        "decision": "experiment", "proposal_id": "p", "collection": {"setting": "demo_randomized"},
        "training": {"epochs": 2000, "lr": 1e-5, "chunk_size": 50, "kl_weight": 10}})
    monkeypatch.setattr(runner, "_acquire_and_mix", lambda proposal: "mixed")
    monkeypatch.setattr(runner, "_candidate", lambda key, proposal: Path("candidate"))
    monkeypatch.setattr("autosim.research.robotwin_autoresearch.paired",
                        lambda candidate, baseline: {"success_delta": 0.1})
    monkeypatch.setattr(runner, "_export", lambda checkpoint: (Path("export"), True))
    result = runner.execute()
    assert result["status"] == "completed"
    assert result["performance_improved"] is True
    assert result["selected_checkpoint"] == "candidate"
    assert any(row[:3] == ("baseline_confirmation", "confirmation", 1) for row in calls)
    candidate_confirmation = next(
        row for row in calls if row[:3] == ("candidate_confirmation", "confirmation", 1))
    assert candidate_confirmation[3] == {"demo_clean": [10], "demo_randomized": [10]}


def test_controller_stop_still_exports_baseline(tmp_path, monkeypatch):
    runner = RoboTwinAutoResearch(
        project_root=PROJECT, repo=PROJECT / "RoboTwin", output_root=tmp_path,
        run_id="stop_contract", task="beat_block_hammer", controller="heuristic",
        gpu="0", hours=1, train_seed=1, training_epochs=2000,
        evaluation_episodes=1, allow_api_egress=False)
    monkeypatch.setattr("autosim.research.robotwin_autoresearch.gpu_lock",
                        lambda gpu: nullcontext())
    monkeypatch.setattr(runner, "initialize", lambda: {"task_contract": {}})
    monkeypatch.setattr(runner, "_baseline", lambda: Path("baseline"))
    monkeypatch.setattr(runner, "_evaluate_pair", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_proposal", lambda *args: {"decision": "stop"})
    monkeypatch.setattr(runner, "_export", lambda checkpoint: (Path("export"), True))
    result = runner.execute()
    assert result["status"] == "completed"
    assert result["selected_checkpoint"] == "baseline"
    assert result["optimized_repo"] == "export"
    assert result["export_runtime_verified"] is True
