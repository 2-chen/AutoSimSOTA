import json

import pytest

from tools.export_external_pilot_evidence import enpire_robotwin, robosyn


def test_robosyn_audit_counts_real_episodes(monkeypatch, tmp_path):
    monkeypatch.setattr("tools.export_external_pilot_evidence.artifact", lambda p: {"path": str(p)})
    path = tmp_path / "evaluation_metrics.json"
    path.write_text(json.dumps({
        "config": {"episode_count": 2, "setting": "clear", "seed": 2026},
        "summary": {"episode_count": 2, "success_count": 1, "success_rate": .5},
        "episodes": [{"episode_seed": 1, "success": True},
                     {"episode_seed": 2, "success": False}],
    }))
    result = robosyn(path)
    assert result["successes"] == 1 and result["episodes"] == 2
    assert result["episode_seeds"] == [1, 2]
    value = json.loads(path.read_text())
    value["summary"]["success_count"] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Aggregate mismatch"):
        robosyn(path)


def test_robosyn_audit_rejects_partial_instead_of_zero(tmp_path):
    path = tmp_path / "evaluation_metrics.json"
    path.write_text(json.dumps({
        "config": {"episode_count": 20},
        "summary": {"episode_count": 20, "success_count": 0, "success_rate": 0},
        "episodes": [],
    }))
    with pytest.raises(ValueError, match="Incomplete episode count"):
        robosyn(path)


def test_enpire_audit_rejects_error_outcomes(monkeypatch, tmp_path):
    monkeypatch.setattr("tools.export_external_pilot_evidence.artifact", lambda p: {"path": str(p)})
    path = tmp_path / "summary.json"
    records = [{"episode": i, "seed": i, "success": False,
                "termination": "error", "steps": 0} for i in range(20)]
    path.write_text(json.dumps({"records": records, "episodes": 20,
                               "test_num": 20, "success": 0, "success_rate": 0}))
    episode = tmp_path / "ep000"
    episode.mkdir()
    (episode / "result.json").write_text(json.dumps({
        "termination": "error", "success": False, "steps": 0,
        "verification": {"success": False},
    }))
    with pytest.raises(ValueError, match="result mismatch/error"):
        enpire_robotwin(path)
