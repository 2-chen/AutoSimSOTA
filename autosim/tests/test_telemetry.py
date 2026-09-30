from __future__ import annotations

import json
import builtins
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosim.research.telemetry import LiveTelemetryWriter, _render_svg


def _writer(tmp_path: Path, *, started_epoch: float | None = None,
            telemetry_spec: dict | None = None) -> tuple[
        LiveTelemetryWriter, Path, Path]:
    repo = tmp_path / "checkout"
    output = tmp_path / "run"
    stage = output / "research" / "derived" / "train"
    work = repo / "examples" / "train"
    stage.mkdir(parents=True)
    work.mkdir(parents=True)
    log = stage / "attempts" / ("a" * 32) / "output.log"
    log.parent.mkdir(parents=True)
    writer = LiveTelemetryWriter(
        output=output, run_root=output / "research" / "derived", repo=repo,
        working_directory=work, stage_directory=stage, log_path=log,
        stage="train", attempt_id="a" * 32,
        started_epoch=time.time() - 1 if started_epoch is None else started_epoch,
        telemetry_spec=telemetry_spec)
    return writer, log, output


def test_stdout_scalars_are_incremental_idempotent_and_not_the_score(tmp_path):
    writer, log, output = _writer(tmp_path)
    log.write_text("global_step=4 loss=0.75 return=2.5\n", encoding="utf-8")

    state = writer.poll()
    assert state["status"] == "in_progress"
    assert state["sample_count"] == 2
    by_name = {row["metric_name"]: row["samples"][0] for row in state["series"]}
    assert set(by_name) == {"loss", "return"}
    assert by_name["loss"]["native_step"] == 4
    assert by_name["loss"]["verification"] == "native_named_log"
    assert writer.poll()["sample_count"] == 2
    assert writer.finalize("completed")["status"] == "completed"
    assert (output / "report" / "telemetry" / ("a" * 32 + ".svg")).is_file()
    persisted = (output / "report" / "telemetry" / ("a" * 32 + ".json")).read_text()
    assert str(log) not in persisted
    assert "official score" in persisted


def test_partial_line_waits_for_delimiter_then_emits_without_duplicate(tmp_path):
    writer, log, _ = _writer(tmp_path)
    with log.open("wb") as stream:
        stream.write(b"global_step=1 train_loss=0.5")
    assert writer.poll()["sample_count"] == 0
    with log.open("ab") as stream:
        stream.write(b"\n")
    state = writer.poll()
    assert state["sample_count"] == 1
    assert state["series"][0]["samples"][0]["value"] == 0.5
    assert writer.poll()["sample_count"] == 1


def test_truncated_log_starts_a_new_segment(tmp_path):
    writer, log, _ = _writer(tmp_path)
    log.write_text("global_step=1 loss=1.0\n", encoding="utf-8")
    assert writer.poll()["sample_count"] == 1
    log.write_text("", encoding="utf-8")
    writer.poll()
    log.write_text("global_step=1 loss=0.25\n", encoding="utf-8")
    state = writer.poll()
    samples = state["series"][0]["samples"]
    assert state["sample_count"] == 2
    assert [sample["source_position"].split(":", 1)[0] for sample in samples] == ["0", "1"]
    assert samples[-1]["value"] == 0.25


def test_writer_restart_resumes_from_persisted_cursor_without_duplicate_samples(tmp_path):
    writer, log, output = _writer(tmp_path)
    log.write_text("global_step=1 loss=1.0\n", encoding="utf-8")
    assert writer.poll()["sample_count"] == 1

    restarted = LiveTelemetryWriter(
        output=output, run_root=writer.run_root, repo=writer.repo,
        working_directory=writer.working_directory,
        stage_directory=writer.stage_directory, log_path=log,
        stage="train", attempt_id="a" * 32,
        started_epoch=writer.started_epoch)

    assert restarted.poll()["sample_count"] == 1
    with log.open("a", encoding="utf-8") as stream:
        stream.write("global_step=2 loss=0.5\n")
    state = restarted.poll()
    assert state["sample_count"] == 2
    assert [sample["native_step"] for sample in state["series"][0]["samples"]] == [1, 2]


def test_event_discovery_is_cached_and_periodically_bounded(tmp_path, monkeypatch):
    import autosim.research.telemetry as telemetry

    writer, _, _ = _writer(tmp_path)
    calls = 0
    original = telemetry.os.walk

    def count_walk(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(telemetry.os, "walk", count_walk)
    writer._event_files()
    writer._event_files()
    writer._event_files()
    assert calls == 2  # exactly one walk for cwd and one for the stage output


def test_tensorboard_scalars_are_read_when_optional_dependency_is_available(tmp_path):
    writer, _, output = _writer(tmp_path)
    work = writer.working_directory
    try:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
    except ImportError:
        pytest.skip("TensorBoard is optional in the execution environment")

    event_writer = EventFileWriter(str(work))
    event_writer.add_event(Event(
        wall_time=time.time(), step=9,
        summary=Summary(value=[Summary.Value(tag="eval/success_once", simple_value=0.5)])))
    event_writer.flush()
    event_writer.close()

    state = writer.poll()
    assert state["sample_count"] == 1
    sample = state["series"][0]["samples"][0]
    assert sample["metric_name"] == "eval/success_once"
    assert sample["native_step"] == 9
    assert sample["verification"] == "native_tensorboard_scalar"
    serialized = json.dumps(state)
    assert str(output) not in serialized


def test_missing_tensorboard_reader_degrades_with_an_explicit_reason(tmp_path, monkeypatch):
    writer, _, _ = _writer(tmp_path)
    event_dir = writer.working_directory / "runs"
    event_dir.mkdir()
    (event_dir / "events.out.tfevents.synthetic").write_bytes(b"not parsed")
    original_import = builtins.__import__

    def without_tensorboard(name, *args, **kwargs):
        if name.startswith("tensorboard."):
            raise ImportError("synthetic optional dependency absence")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tensorboard)
    state = writer.poll()

    assert state["status"] == "in_progress_telemetry_unavailable"
    assert "tensorboard_reader_unavailable" in state["errors"]
    assert "Waiting for named trainer scalars" in writer.chart_path.read_text()


def test_fake_tensorboard_event_source_is_consumed_incrementally(tmp_path, monkeypatch):
    writer, _, output = _writer(tmp_path)
    event_dir = writer.working_directory / "runs"
    event_dir.mkdir()
    (event_dir / "events.out.tfevents.synthetic").write_bytes(b"synthetic event")

    class FakeAccumulator:
        def __init__(self, path, size_guidance):
            self.path = path
            self.size_guidance = size_guidance

        def Reload(self):
            return None

        def Tags(self):
            return {"scalars": ["eval/success_once"]}

        def Scalars(self, tag):
            assert tag == "eval/success_once"
            return [SimpleNamespace(value=0.75, wall_time=time.time(), step=8)]

    original_import = builtins.__import__

    def fake_tensorboard(name, *args, **kwargs):
        if name == "tensorboard.backend.event_processing.event_accumulator":
            return SimpleNamespace(EventAccumulator=FakeAccumulator)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_tensorboard)
    state = writer.poll()
    assert state["sample_count"] == 1
    sample = state["series"][0]["samples"][0]
    assert sample["metric_name"] == "eval/success_once"
    assert sample["native_step"] == 8
    assert sample["verification"] == "native_tensorboard_scalar"
    assert str(output) not in json.dumps(state)
    assert writer.poll()["sample_count"] == 1


def test_bad_named_scalars_and_nonfinite_values_are_not_fabricated(tmp_path):
    writer, log, _ = _writer(tmp_path)
    log.write_text("global_step=2 reward=nan loss=inf\n", encoding="utf-8")
    state = writer.poll()
    assert state["sample_count"] == 0
    assert state["status"] == "in_progress_waiting_for_native_scalars"
    assert "Waiting for named trainer scalars" in writer.chart_path.read_text()


def test_chart_prioritizes_success_loss_and_throughput_and_discloses_omissions():
    series = {
        f"misc/metric_{index:02d}": {
            "metric_name": f"misc/metric_{index:02d}", "value_unit": None,
            "samples_seen": 1,
            "samples": [{"value": float(index), "native_step": 1,
                         "source_ref": "synthetic"}],
        }
        for index in range(14)
    }
    for name in ("losses/policy_loss", "charts/SPS", "eval/success_once"):
        series[name] = {"metric_name": name, "value_unit": "unit",
                        "samples_seen": 1,
                        "samples": [{"value": 0.5, "native_step": 1,
                                     "source_ref": "synthetic"}]}

    svg = _render_svg(series, status="in_progress")

    assert "eval/success_once (unit)" in svg
    assert "losses/policy_loss (unit)" in svg
    assert "charts/SPS (unit)" in svg
    assert "Showing 12 of 17 metrics by priority" in svg
    assert "misc/metric_13" not in svg


def test_declared_metric_alias_and_unit_are_projected_without_changing_source_name(tmp_path):
    writer, log, _ = _writer(tmp_path, telemetry_spec={
        "schema_version": 1,
        "metrics": {"train_loss": {"name": "optimization_loss", "unit": "nats"}},
    })
    log.write_text("global_step=3 train_loss=0.125\n", encoding="utf-8")

    state = writer.poll()

    series = state["series"][0]
    sample = series["samples"][0]
    assert series["metric_name"] == "optimization_loss"
    assert series["value_unit"] == "nats"
    assert sample["source_metric_name"] == "train_loss"
    assert sample["value_unit"] == "nats"


def test_declared_event_roots_bound_tensorboard_discovery(tmp_path):
    writer, _, _ = _writer(tmp_path, telemetry_spec={"event_roots": ["runs"]})
    root = writer.working_directory / "runs"
    root.mkdir()
    event = root / "events.out.tfevents.synthetic"
    event.write_bytes(b"event")

    files = writer._event_files()

    assert len(files) == 1
    assert files[0][1] == event
    assert files[0][0].startswith("native-log:")


def test_invalid_optional_spec_degrades_telemetry_but_does_not_raise(tmp_path):
    writer, _, _ = _writer(tmp_path, telemetry_spec={"event_roots": ["../escape"]})

    state = writer.poll()

    assert state["status"] == "in_progress_telemetry_unavailable"
    assert "invalid_telemetry_spec" in state["errors"]
