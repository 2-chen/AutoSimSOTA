"""Immutable policy copies are bounded and content-checked."""

import json
import stat
from pathlib import Path

import pytest

from autosim.research.experiment_bundle import freeze_artifact
from autosim.research.experiment_bundle import export_best


def test_freeze_a_policy_file_and_preserve_the_old_bytes(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"candidate one")
    destination = tmp_path / "experiment" / "policy.pt"
    record = freeze_artifact(source, destination, max_bytes=1024)
    source.write_bytes(b"candidate two")
    assert Path(record["path"]).read_bytes() == b"candidate one"
    assert record["files"] == 1


def test_copy_limit_is_explicit_and_never_creates_a_partial_policy(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"a" * 10)
    destination = tmp_path / "experiment" / "policy.pt"
    with pytest.raises(ValueError, match="above copy limit"):
        freeze_artifact(source, destination, max_bytes=5)
    assert not destination.exists()


def test_best_export_keeps_a_separate_policy_and_overlay(tmp_path):
    run = tmp_path / "run"
    policy = run / "experiments" / "baseline" / "attempt" / "policy.pt"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(b"best policy")
    import json
    from autosim.research.common import digest

    measurement = run / "measurements" / "baseline.json"
    measurement.parent.mkdir()
    measurement.write_text(json.dumps({"policy_artifact": {
        "path": str(policy), "content_sha256": digest(policy)}}),
                           encoding="utf-8")
    source = run / "snapshots" / "blobs"
    source.mkdir(parents=True)
    staged = tmp_path / "train.py"
    staged.write_text("print('best')\n", encoding="utf-8")
    key = digest(staged)
    (source / key).write_bytes(staged.read_bytes())
    result = export_best(run, {"name": "baseline", "scale": "success_rate", "score": 0.5,
                               "files": {"train.py": key}})
    assert result["status"] == "artifact_and_overlay_exported"
    assert (Path(result["path"]) / "policy.pt").read_bytes() == b"best policy"
    assert (Path(result["path"]) / "source_overlay" / "train.py").read_bytes() == staged.read_bytes()
    assert result["native_load_verified"] is False


def test_best_export_carries_a_source_controller_overlay_without_fake_weights(tmp_path):
    from autosim.research.snapshot import Snapshots
    from autosim.research.workspace_snapshot import capture_state, create

    output = tmp_path / "out"
    source = tmp_path / "source"
    source.mkdir()
    (source / "controller.py").write_text("score = 0.1\n", encoding="utf-8")
    checkout = output / "checkout"
    workspace = create(source, checkout)
    (checkout / "controller.py").write_text("score = 0.9\n", encoding="utf-8")
    (checkout / "controller.py").chmod(0o755)
    run = output / "research" / "derived"
    snapshots = Snapshots(run / "snapshots")
    state = capture_state(checkout, workspace, snapshots, name="source-state-baseline")
    measurement_path = run / "measurements" / "baseline.json"
    measurement_path.parent.mkdir(parents=True)
    measurement_path.write_text(json.dumps({
        "label": "baseline", "ok": True, "metric_value": 0.9,
        "scored": "a source-defined controller with no weight artifact",
        "policy_artifact": {}, "source_state": state,
        "source_policy_artifact": {"kind": "source_tree", **state}}), encoding="utf-8")

    result = export_best(run, {"name": "baseline", "scale": "success_rate",
                              "score": 0.9, "files": {}})

    assert result["status"] == "source_controller_overlay_exported"
    assert result["policy"]["kind"] == "source_tree"
    assert result["source_base"]["tree_fingerprint"] == \
        workspace["source_tree_fingerprint"]
    exported_overlay = Path(result["path"]) / "source_overlay" / "controller.py"
    assert exported_overlay.read_text() == "score = 0.9\n"
    assert stat.S_IMODE(exported_overlay.stat().st_mode) == 0o755
    assert result["portable"] is False
    assert result["native_load_verified"] is False


def test_export_rejects_policy_bytes_changed_after_measurement(tmp_path):
    run = _run_with_recipe(tmp_path)
    measured = run / "measurements" / "baseline.json"
    policy = Path(json.loads(measured.read_text())["policy_artifact"]["path"])
    policy.write_bytes(b"overwritten")
    exported = export_best(run, {"name": "baseline", "scale": "success_rate", "score": 0.5})
    assert exported["status"] == "incomplete_export"
    assert "no longer matches" in exported["policy_error"]


def _run_with_recipe(tmp_path, *, confirmed=False):
    """A run directory with an exportable policy and the records that make it re-runnable."""
    import json
    from autosim.research.common import digest

    run = tmp_path / "out" / "research" / "derived"
    policy = run / "experiments" / "baseline" / "attempt" / "policy.pt"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(b"best policy")
    (run / "measurements").mkdir()
    (run / "measurements" / "baseline.json").write_text(json.dumps({
        "label": "baseline", "ok": True, "metric_value": 0.5, "metric_utility": 0.5,
        "settings": {"seed": 0},
        "evaluate": {"device": "cpu", "compute_environment": {
            "CUDA_VISIBLE_DEVICES": ""}},
        "policy_artifact": {"path": str(policy), "content_sha256": digest(policy)},
        "metric": {"name": "success_rate", "direction": "maximize", "unit": "fraction",
                   "minimum": 0.0, "maximum": 1.0, "source": "log", "json_key": "",
                   "csv_column": "", "aggregation": "", "min_samples": 1}}), encoding="utf-8")
    (run / "comparison_protocol.json").write_text(json.dumps({
        "schema_version": 1, "target": "evaluate", "settings": {"seed": 0}}), encoding="utf-8")
    checkout = tmp_path / "out" / "checkout"
    checkout.mkdir(parents=True)
    (tmp_path / "out" / "derived_stages.json").write_text(json.dumps({
        "evaluate": {"source": "def stage_argv_evaluate(i):\n    return [i['python']]\n",
                     "parameters": {}, "row": {"available": True}}}), encoding="utf-8")
    (tmp_path / "out" / "execution.json").write_text(json.dumps({
        "repo": str(checkout), "stages": {"evaluate": {"available": True}}}), encoding="utf-8")
    import sys
    (tmp_path / "out" / "environment.json").write_text(json.dumps({
        "interpreter": sys.executable}), encoding="utf-8")
    if confirmed:
        (run / "confirmation.json").write_text(json.dumps({
            "label": "confirmation", "held_out": {"seed": 101}, "ok": True,
            "metric_value": 0.47, "at": "now"}), encoding="utf-8")
    return run


def test_an_export_carries_what_its_number_was_measured_under(tmp_path):
    """A policy file and a note saying which measurement it came from is a policy file. The
    parts that make the number readable -- the frozen settings, the metric contract, the
    search's own count -- are small and are already on disk."""
    run = _run_with_recipe(tmp_path)
    result = export_best(run, {"name": "baseline", "scale": "success_rate", "score": 0.5})
    bundle = result["bundle"]
    assert bundle["complete"] is True
    assert bundle["measured"]["metric_value"] == 0.5
    assert bundle["measured"]["settings"] == {"seed": 0}
    assert bundle["measured"]["metric"]["name"] == "success_rate"
    assert set(bundle["records"]) >= {"derived_stages.json", "execution.json",
                                      "environment.json", "comparison_protocol.json"}
    assert all(row["sha256"] for row in bundle["records"].values())


def test_an_export_says_what_it_did_not_include(tmp_path):
    """A bundle silent about the checkout, the environment and the datasets reads as
    self-contained, and it is not: nothing here can be evaluated without them."""
    run = _run_with_recipe(tmp_path)
    bundle = export_best(run, {"name": "baseline", "scale": "success_rate",
                               "score": 0.5})["bundle"]
    assert len(bundle["not_copied"]) == 3
    assert any("checkout" in one for one in bundle["not_copied"])
    assert any("environment" in one for one in bundle["not_copied"])


def test_the_export_states_the_command_that_would_check_it_again(tmp_path):
    """The bundle and the re-check are one mechanism: an export that cannot name how to
    reproduce its number has left the reader to guess."""
    run = _run_with_recipe(tmp_path)
    result = export_best(run, {"name": "baseline", "scale": "success_rate", "score": 0.5})
    assert result["bundle"]["reproduce"].startswith("autosim recheck")
    assert result["portable"] is False
    assert result["export_runtime_verified"] is False
    exported = Path(result["path"])
    for name, row in result["bundle"]["records"].items():
        assert (exported / row["copy"]).read_bytes() == Path(row["from"]).read_bytes()
    assert (exported / result["bundle"]["measurement_copy"]).is_file()


def test_an_incomplete_bundle_does_not_report_itself_as_portable(tmp_path):
    """The one property that has to hold: `portable` must be false when a record that a re-run
    needs is not there, and the manifest has to name it rather than warn in general."""
    import json
    run = _run_with_recipe(tmp_path)
    (tmp_path / "out" / "derived_stages.json").unlink()
    result = export_best(run, {"name": "baseline", "scale": "success_rate", "score": 0.5})
    assert result["portable"] is False
    assert result["bundle"]["complete"] is False
    assert any("derived_stages.json" in one for one in result["bundle"]["missing_records"])
    assert "bundle.missing_records" in result["warning"]


def test_an_export_of_a_single_arm_does_not_warn_about_a_maximum(tmp_path):
    """The warning has to be about this export. An export that always cries "best of many" is
    one a reader learns to skip, and then it is not there for the one that needed it."""
    run = _run_with_recipe(tmp_path)
    bundle = export_best(run, {"name": "baseline", "scale": "success_rate",
                               "score": 0.5})["bundle"]
    assert not any("best of several arms" in one for one in bundle["limitations"])


def test_an_export_of_many_arms_says_its_number_includes_the_highest_noise(tmp_path):
    import json
    run = _run_with_recipe(tmp_path)
    for index in range(4):
        (run / "measurements" / f"round_{index}.json").write_text(json.dumps({
            "label": f"round_{index}", "metric_value": 0.6, "metric_utility": 0.6,
            "settings": {"seed": 0}}), encoding="utf-8")
    bundle = export_best(run, {"name": "baseline", "scale": "success_rate",
                               "score": 0.5})["bundle"]
    assert bundle["search"]["best_is_the_maximum_of"] == 5
    assert any("highest noise" in one for one in bundle["limitations"])


def test_an_export_without_a_confirmation_says_nothing_was_measured_on_held_out_episodes(
        tmp_path):
    run = _run_with_recipe(tmp_path)
    bundle = export_best(run, {"name": "baseline", "scale": "success_rate",
                               "score": 0.5})["bundle"]
    assert any("held-out" in one for one in bundle["limitations"])
    confirmed = export_best(_run_with_recipe(tmp_path / "second", confirmed=True),
                            {"name": "baseline", "scale": "success_rate", "score": 0.5})
    assert not any("held-out" in one for one in confirmed["bundle"]["limitations"])
    assert "confirmation.json" in confirmed["bundle"]["records"]


def test_an_ordinary_run_with_no_confirmation_is_still_a_complete_export(tmp_path):
    """The mirror of the usual defect. `selection_record.json` and `confirmation.json` are not
    always there and are not supposed to be: a single-arm run writes no selection record, and a
    run with no declared split has no confirmation and never will. Requiring them made every
    ordinary export call itself unportable, which is a warning a reader learns to skip -- and
    then it is absent for the export that needed it."""
    run = _run_with_recipe(tmp_path)
    bundle = export_best(run, {"name": "baseline", "scale": "success_rate",
                               "score": 0.5})["bundle"]
    assert bundle["complete"] is True
    assert bundle["missing_records"] == []
    assert any("selection_record.json" in one for one in bundle["records_not_present"])
    assert any("confirmation.json" in one for one in bundle["records_not_present"])


def test_a_missing_required_record_is_never_filed_as_merely_absent(tmp_path):
    """The two lists have to stay separate, or a bundle that cannot be re-run reports itself
    as complete with a footnote."""
    run = _run_with_recipe(tmp_path)
    (tmp_path / "out" / "environment.json").unlink()
    bundle = export_best(run, {"name": "baseline", "scale": "success_rate",
                               "score": 0.5})["bundle"]
    assert bundle["complete"] is False
    assert any("environment.json" in one for one in bundle["missing_records"])
    assert not any("environment.json" in one for one in bundle["records_not_present"])
