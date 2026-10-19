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


def test_compact_native_counters_scientific_values_and_short_metric_names(tmp_path):
    writer,log,_ = _writer(tmp_path)
    log.write_text('INFO 2026-10-17 11:40:03 ts/train.py:232 '
                   'step:1K smpl:8K ep:108 epch:0.10 loss:2.4 lr:1.0e-05 updt_s:0.08\n'
                   'step:1.2K loss:2.0 lr:9.5e-06\n')
    state = writer.poll()
    series = {r['metric_name']:r for r in state['series']}
    assert [r['native_step'] for r in series['loss']['samples']] == [1000,1200]
    assert series['smpl']['samples'][0]['value'] == 8000
    assert series['lr']['samples'][-1]['value'] == 9.5e-6
    assert 'info' not in series and 'ts/train.py' not in series


def test_identity_tokens_and_configuration_are_not_numeric_training_signals(tmp_path):
    writer,log,_ = _writer(tmp_path)
    identity = '87e57cd66aa228b652f1edefcc45b59209bea0deefc33197441c709b916e2534'
    raw = ('AUTOSIM_TRAIN_DATA_CONSUMPTION '+json.dumps({
        'identity':identity,'identity_field':'sha256','samples_read':15})+'\n'
        'INFO 2026-10-17 11:39:46 ts/train.py:159 cfg.steps=80000 (80K)\n'
        'backbone=resnet18 initialization=imagenet1k checksum='+identity+'\n'
        'global_step=42 vendor_margin=3 lr=2e-5 loss=0.7\n')
    log.write_text(raw)
    state=writer.poll()
    series={r['metric_name']:r for r in state['series']}
    assert set(series)=={'vendor_margin','lr','loss'}
    assert all(s['native_step']==42 for r in series.values() for s in r['samples'])
    assert log.read_text()==raw


def test_json_only_actual_numeric_values_are_observations(tmp_path):
    writer,log,_ = _writer(tmp_path)
    log.write_text(json.dumps({'step':12,'loss':0.6,'version':'0.24',
                              'hash':'123abc','lr':2e-6,'nested':{'fake':400}})+'\n')
    state=writer.poll()
    series={r['metric_name']:r for r in state['series']}
    assert set(series)=={'loss','lr'}
    assert all(r['samples'][0]['native_step']==12 for r in series.values())


def test_parser_upgrade_archives_bad_projection_then_replays_original_evidence(tmp_path):
    writer,log,output = _writer(tmp_path)
    raw='step:1K loss:0.5 lr:1e-5\n'
    log.write_text(raw)
    old=writer.poll()
    old.pop('parser_version')
    old['series'][0]['samples'][0]['native_step']=1
    writer.state_path.write_text(json.dumps(old))
    restarted=LiveTelemetryWriter(output=output,run_root=writer.run_root,repo=writer.repo,
        working_directory=writer.working_directory,stage_directory=writer.stage_directory,
        log_path=log,stage='train',attempt_id='a'*32,started_epoch=writer.started_epoch)
    state=restarted.poll()
    assert state['parser_version']=='autosim-telemetry-v2'
    assert state['sample_count']==2
    assert all(s['native_step']==1000 for r in state['series'] for s in r['samples'])
    history=list((writer.report_dir/'parser_history').glob('*.json'))
    assert len(history)==1 and json.loads(history[0].read_text())==old
    assert log.read_text()==raw
    assert restarted.poll()['sample_count']==2


def test_nonfinite_and_oversize_integer_telemetry_does_not_break_observation(tmp_path):
    writer,log,_ = _writer(tmp_path)
    log.write_text(json.dumps({'step':2,'loss':float('nan'),'huge':10**400,
                              'flag':True,'vendor_signal':0.3})+'\n')
    state=writer.poll()
    assert [r['metric_name'] for r in state['series']]==['vendor_signal']


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


@pytest.mark.parametrize('diagnostic', [
    '/site-packages/video.py:5: UserWarning: removed in version 0.24',
    'WARNING encoder version=0.24 loss=0.9',
    '[WARN] fallback version=2',
    'ERROR retries=3',
    'RuntimeError: code=42 loss=0.9',
    'ValueError: shape=64',
    'File "/native/train.py", line 25, in train',
    'raise RuntimeError("code=42")',
])
def test_diagnostic_numbers_are_not_training_curves_and_raw_log_is_preserved(tmp_path, diagnostic):
    writer, log, _ = _writer(tmp_path)
    text = diagnostic + '\nglobal_step=5 loss=0.75 warning_count=2 vendor_margin=3\n'
    log.write_text(text, encoding='utf-8')
    state = writer.poll()
    assert {row['metric_name'] for row in state['series']} == {
        'loss', 'warning_count', 'vendor_margin'}
    assert all(sample['native_step'] == 5
               for row in state['series'] for sample in row['samples'])
    assert log.read_text(encoding='utf-8') == text


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
