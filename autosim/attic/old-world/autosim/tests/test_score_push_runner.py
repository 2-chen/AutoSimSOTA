from pathlib import Path
from unittest.mock import Mock, patch

from autosim.research.score_push import collect_capability
from autosim.research.score_push_autorun import execute_task, run_campaign
from autosim.research.score_push_runner import DEFAULT_ORDER


def test_runner_covers_every_non_click_task_once():
    assert len(DEFAULT_ORDER) == 9
    assert len(set(DEFAULT_ORDER)) == len(DEFAULT_ORDER)
    assert "click_bell" not in DEFAULT_ORDER
    assert DEFAULT_ORDER[-2:] == ("sample_loading", "item_assembly")


def test_capability_audit_failure_is_persisted_and_never_admitted(tmp_path):
    workspace = tmp_path / "workspace"
    root = tmp_path / "campaign"
    dataset = tmp_path / "collected"
    workspace.mkdir()
    dataset.mkdir()
    (root / "campaign_protocol.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "campaign_protocol.json").write_text(
        '{"tasks":{"handle_basket":{"cap_attempts":10,'
        '"banks":{"capability":{"master":7}}}}}', encoding="utf-8")

    runtime = Mock()
    runtime.repo = tmp_path / "repo"
    runtime.collect_bounded.return_value = {
        "accepted_episodes": 8, "dataset_root": str(dataset)}
    runtime.prepare_data.return_value = {"passed": True}
    spec = Mock(cameras=("cam_high",))
    protocol = {"tasks": {"handle_basket": {
        "cap_attempts": 10, "banks": {"capability": {"master": 7}}}}}
    with patch("autosim.research.score_push._runtime", return_value=(runtime, protocol)), \
         patch("autosim.research.score_push.load_task", return_value=spec), \
         patch("autosim.research.score_push.audit_dataset", return_value={"passed": False}):
        result = collect_capability(workspace, root, "handle_basket")

    assert result["yield"] == .8
    assert result["decision"] == "repair_then_reprobe"
    assert result["dataset_root"] is None
    assert result["collected_dataset_root"] == str(dataset)
    assert result["training_admission"] == "rejected"
    assert (root / "handle_basket/capability_summary.json").is_file()


def test_capability_below_yield_gate_does_not_expose_dataset_to_training(tmp_path):
    workspace = tmp_path / "workspace"
    root = tmp_path / "campaign"
    dataset = tmp_path / "collected"
    workspace.mkdir()
    dataset.mkdir()
    runtime = Mock(repo=tmp_path / "repo")
    runtime.collect_bounded.return_value = {
        "accepted_episodes": 3, "dataset_root": str(dataset)}
    runtime.prepare_data.return_value = {"passed": True}
    protocol = {"tasks": {"handle_basket": {
        "cap_attempts": 10, "banks": {"capability": {"master": 7}}}}}
    spec = Mock(cameras=("cam_high",))
    with patch("autosim.research.score_push._runtime", return_value=(runtime, protocol)), \
         patch("autosim.research.score_push.load_task", return_value=spec), \
         patch("autosim.research.score_push.audit_dataset", return_value={"passed": True}):
        result = collect_capability(workspace, root, "handle_basket")

    assert result["decision"] == "repair_then_reprobe"
    assert result["dataset_root"] is None
    assert result["collected_dataset_root"] == str(dataset)
    assert result["training_admission"] == "rejected"
    assert result["admission_failure"] == "capability yield gate did not pass"


def test_autorun_routes_closed_capability_to_official_only_fallback(tmp_path):
    with patch("autosim.research.score_push_autorun.run_task", return_value={
             "status": "awaiting_expert_repair"}), \
         patch("autosim.research.score_push_autorun.run_fallback", return_value={
             "status": "completed_no_improvement"}) as fallback:
        result = execute_task(tmp_path, tmp_path / "campaign", "item_assembly")
    fallback.assert_called_once()
    assert result["route"] == "official_data_only_fallback"
    assert result["new_data_used"] is False


def test_autorun_records_failure_and_continues_next_task(tmp_path):
    outcomes = [RuntimeError("first failed"), {"status": "completed_no_improvement"}]
    with patch("autosim.research.score_push_autorun.execute_task", side_effect=outcomes):
        result = run_campaign(tmp_path, tmp_path / "campaign", (
            "manipulate_pipette", "mixer_operating"))
    assert result["status"] == "completed_with_failures"
    assert result["tasks"]["manipulate_pipette"]["status"] == "failed"
    assert result["tasks"]["mixer_operating"]["status"] == "completed_no_improvement"
