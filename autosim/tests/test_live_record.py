"""The human document links only to real run artifacts and survives rebuilding."""

import json
import hashlib
import shutil
import subprocess
import threading
import time

import pytest

from autosim.research import run_record
from autosim.research.research_state import ResearchStateStore


def test_record_links_real_demo_and_embeds_real_trajectory(tmp_path):
    video = tmp_path / "eval_result" / "rollout.mp4"
    video.parent.mkdir()
    video.write_bytes(b"recorded bytes")
    trace = tmp_path / "trajectory.svg"
    trace.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")

    document = run_record.generate(tmp_path).read_text(encoding="utf-8")
    assert "[打开真实运行录像](<eval_result/rollout.mp4>)" in document
    assert "![从真实遥测生成的轨迹](<trajectory.svg>)" in document
    assert "不是场景回放" in document
    manifest = json.loads((tmp_path / "media" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["recordings"][0]["path"] == "eval_result/rollout.mp4"
    assert manifest["recordings"][0]["hash_status"] == "checked"
    assert manifest["recordings"][0]["same_scored_evaluation"] is None


def test_report_without_media_does_not_imply_a_demo_exists(tmp_path):
    document = run_record.generate(tmp_path).read_text(encoding="utf-8")
    assert "没有留下录像" in document
    assert "不能假设一个通用开关" in document
    assert json.loads((tmp_path / "media" / "manifest.json").read_text(encoding="utf-8"))[
        "status"] == "no_media_found"


def test_native_recording_automatically_gets_real_frame_in_top_run(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("real-frame preview requires ffmpeg")
    output = tmp_path / "out"
    research = output / "research" / "run-1"
    video = research / "eval_result" / "episode_000_seed_1_fail.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=160x120:d=1", "-pix_fmt", "yuv420p",
                    "-y", str(video)], check=True, capture_output=True, timeout=20)
    video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    attempt_id = "a" * 32
    receipt = research / "attempts" / attempt_id / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"attempt_id": attempt_id, "stage": "evaluate",
                                   "node_id": "evaluate", "status": "completed",
                                   "media": [{"path": "eval_result/episode_000_seed_1_fail.mp4",
                                              "sha256": video_sha}]}), encoding="utf-8")
    (output / "RUN.md").write_text(
        "# Run\n\n<!-- AUTOSIM_RESEARCH_START -->\n"
        "<!-- AUTOSIM_RESEARCH_END -->\n", encoding="utf-8")

    run_record.generate(research)
    top = (output / "RUN.md").read_text(encoding="utf-8")
    manifest = json.loads((research / "media" / "manifest.json").read_text())
    row = manifest["recordings"][0]
    preview = row["preview"]
    assert preview["status"] == "decoded_frame"
    assert preview["parent_sha256"] == video_sha
    assert hashlib.sha256((research / preview["path"]).read_bytes()).hexdigest() == preview["sha256"]
    assert f"research/run-1/{preview['path']}" in top
    assert "research/run-1/eval_result/episode_000_seed_1_fail.mp4" in top
    (research / preview["path"]).write_bytes(b"tampered")
    refreshed = run_record.refresh_research_projection(output, "run-1").read_text()
    assert "从上述真实仿真录像解码的帧" not in refreshed
    assert "episode_000_seed_1_fail.mp4" in refreshed


def test_report_explains_frozen_comparison_settings_without_claiming_confirmation(tmp_path):
    (tmp_path / "comparison_protocol.json").write_text(json.dumps({
        "schema_version": 1, "settings": {"task": "Lift", "episodes": 20}}),
        encoding="utf-8")
    document = run_record.generate(tmp_path).read_text(encoding="utf-8")
    assert "`task`=Lift" in document and "`episodes`=20" in document
    assert "相同 seed 不自动证明相同初始状态" in document


def test_external_media_symlink_is_not_presented_as_run_owned_demo(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-external.gif"
    outside.write_bytes(b"GIF89a")
    link = tmp_path / "eval_result" / "rollout.gif"
    link.parent.mkdir()
    link.symlink_to(outside)
    document = run_record.generate(tmp_path).read_text(encoding="utf-8")
    assert "rollout.gif" not in document
    manifest = json.loads((tmp_path / "media" / "manifest.json").read_text())
    assert manifest["recordings"] == []


def test_lightweight_stage_heartbeat_updates_one_live_strip(tmp_path):
    run_record.generate(tmp_path)
    (tmp_path / "running.json").write_text(json.dumps({
        "stage": "native_evaluate", "started_at": time.time() - 120}), encoding="utf-8")
    first = run_record.refresh_live_status(tmp_path).read_text(encoding="utf-8")
    second = run_record.refresh_live_status(tmp_path).read_text(encoding="utf-8")
    assert "native_evaluate" in first and "运行中" in first
    assert second.count("<!-- AUTOSIM_LIVE_START -->") == 1
    assert "约 120 秒" in second
    (tmp_path / "running.json").unlink()
    assert "AUTOSIM_LIVE_START" not in run_record.generate(tmp_path).read_text(encoding="utf-8")


def test_research_stage_heartbeat_is_visible_from_top_level_run_document(tmp_path):
    output = tmp_path / "out"
    research = output / "research" / "run-1"
    research.mkdir(parents=True)
    document = ("# Research run\n\n<!-- AUTOSIM_LIVE_START -->\nStatus: running\n"
                "<!-- AUTOSIM_LIVE_END -->\n")
    run_record.publish_document(output / "RUN.md", document)
    (research / "running.json").write_text(json.dumps({
        "stage": "native_train", "started_at": time.time() - 12}), encoding="utf-8")

    refreshed = run_record.refresh_live_status(
        research, destination=output / "RUN.md").read_text(encoding="utf-8")

    assert "Status: running" in refreshed
    assert "native_train" in refreshed
    assert (output / "report" / "health.json").is_file()


def test_top_level_research_projection_shows_measurements_and_only_verified_media(tmp_path):
    output = tmp_path / "out"
    research = output / "research" / "run-1"
    measurements = research / "measurements"
    measurements.mkdir(parents=True)
    (measurements / "baseline.json").write_text(json.dumps({
        "label": "baseline", "ok": True, "metric_value": 0.25,
        "metric": {"name": "success_rate"},
        "metric_reading": {"samples": 12},
        "evaluate": {"attempt_id": "a" * 32}}), encoding="utf-8")
    (measurements / "round_0.json").write_text(json.dumps({
        "label": "candidate-0", "ok": False, "metric_value": 0.99}), encoding="utf-8")
    (research / "media").mkdir()
    (research / "media" / "manifest.json").write_text(json.dumps({
        "recordings": [
            {"path": "experiments/baseline/demo.mp4", "trigger": "baseline",
             "hash_status": "checked", "media_identity_status": "verified_sha256_match"},
            {"path": "../../outside.mp4", "trigger": "untrusted", "hash_status": "checked",
             "media_identity_status": "verified_sha256_match"},
            {"path": "experiments/candidate/unknown.mp4", "trigger": "candidate",
             "hash_status": "deferred_size_limit", "media_identity_status": "hash_deferred_size_limit"},
        ]}), encoding="utf-8")
    (output / "RUN.md").write_text(
        "# Research run\n\n<!-- AUTOSIM_RESEARCH_START -->\n"
        "old research view\n<!-- AUTOSIM_RESEARCH_END -->\n", encoding="utf-8")

    document = run_record.refresh_research_projection(output, "run-1").read_text(encoding="utf-8")

    assert "baseline" in document and "success_rate: 0.25" in document
    assert "candidate-0" in document and "unscored" in document
    assert "success_rate: 0.99" not in document
    assert "research/run-1/experiments/baseline/demo.mp4" in document
    assert "outside.mp4" not in document
    assert "unknown.mp4" not in document
    assert "remain unverified" in document
    view = json.loads((output / "report" / "view.json").read_text(encoding="utf-8"))
    assert view["schema_version"] == 1
    assert view["source_consistency"] == "stable"
    assert view["source_revision_vector"][
        "research/run-1/measurements/baseline.json"]
    previous_revision = view["view_revision"]
    (measurements / "baseline.json").write_text(json.dumps({
        "label": "baseline", "ok": True, "metric_value": 0.3,
        "metric": {"name": "success_rate"}}), encoding="utf-8")
    refreshed = run_record.build_report_view(output, "run-1")
    assert refreshed["view_revision"] != previous_revision


def test_report_view_marks_a_measurement_arriving_during_projection_as_inconsistent(
        tmp_path, monkeypatch):
    output = tmp_path / "out"
    research = output / "research" / "run-1"
    measurements = research / "measurements"
    measurements.mkdir(parents=True)
    (measurements / "baseline.json").write_text(json.dumps({
        "label": "baseline", "ok": True, "metric_value": 0.25}), encoding="utf-8")
    original_digest = run_record.digest
    added = False

    def add_measurement_after_hash(path):
        nonlocal added
        result = original_digest(path)
        if not added and path == measurements / "baseline.json":
            added = True
            (measurements / "round_0.json").write_text(json.dumps({
                "label": "candidate", "ok": True, "metric_value": 0.5}), encoding="utf-8")
        return result

    monkeypatch.setattr(run_record, "digest", add_measurement_after_hash)

    view = run_record.build_report_view(output, "run-1")

    assert view["source_consistency"] == "changed_during_view_build"
    assert view["source_revision_vector"][
        "research/run-1/measurements/*"] == "changed_during_view_build"


def test_telemetry_projection_uses_reconciled_receipt_after_observer_was_killed(tmp_path):
    output = tmp_path / "out"
    attempt = "b" * 32
    research = output / "research" / "run-1"
    receipt_path = research / "attempts" / attempt / "receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps({
        "attempt_id": attempt, "node_id": "train", "stage": "train",
        "status": "interrupted", "termination_reason": "controller_interrupted",
    }), encoding="utf-8")
    telemetry_dir = output / "report" / "telemetry"
    telemetry_dir.mkdir(parents=True)
    (telemetry_dir / f"{attempt}.json").write_text(json.dumps({
        "schema_version": 1, "attempt_id": attempt,
        "status": "in_progress", "sample_count": 1,
        "series": [{"metric_name": "train_loss", "samples_seen": 1,
                    "value_unit": None,
                    "samples": [{"value": 0.5, "native_step": 2,
                                 "verification": "native_named_log"}]}],
        "source_aliases": ["native-log:0123456789ab"], "errors": [],
    }), encoding="utf-8")
    (telemetry_dir / f"{attempt}.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
    (output / "RUN.md").write_text(
        "# Run\n\n<!-- AUTOSIM_RESEARCH_START -->\n"
        "<!-- AUTOSIM_RESEARCH_END -->\n", encoding="utf-8")

    document = run_record.refresh_research_projection(output, "run-1").read_text()
    view = json.loads((output / "report" / "view.json").read_text())

    row = view["telemetry_attempts"][0]
    assert row["status"] == "interrupted"
    assert row["stale_observer_snapshot"] is True
    assert "stage `interrupted`" in document
    assert "observer snapshot differs from terminal stage receipt (receipt wins)" in document
    assert "running" not in document


def test_live_report_marks_unreconciled_old_running_state_as_stale(tmp_path):
    run_record.publish_document(tmp_path / "RUN.md",
        "# Research run\n\n<!-- AUTOSIM_LIVE_START -->\nStatus: running\n"
        "Current action: train\n<!-- AUTOSIM_LIVE_END -->\n")
    health = tmp_path / "report" / "health.json"
    health.parent.mkdir()
    health.write_text(json.dumps({"status": "running", "current_action": "train",
                                  "last_successful_refresh_epoch": time.time() - 180}),
                      encoding="utf-8")
    (tmp_path / "running.json").write_text(json.dumps({
        "stage": "train", "started_at": time.time() - 600}), encoding="utf-8")

    document = run_record.refresh_live_status(tmp_path).read_text(encoding="utf-8")

    assert "stale: process state requires reconciliation" in document
    assert "Current action: `train`" in document

    completed = run_record.refresh_live_status(
        tmp_path, fallback_status="completed", fallback_current="finished")
    assert "Status: completed" in completed.read_text(encoding="utf-8")


def test_markdown_render_failure_leaves_a_readable_fallback_and_error_receipt(
        tmp_path, monkeypatch):
    def fail_gather(_root):
        raise RuntimeError("synthetic renderer failure")

    monkeypatch.setattr(run_record, "gather", fail_gather)

    destination = run_record.generate(tmp_path)

    assert destination.is_file()
    assert "这份记录没能生成" in destination.read_text(encoding="utf-8")
    error = json.loads((tmp_path / "report_renderer_error.json").read_text(encoding="utf-8"))
    assert error["renderer"] == "markdown"
    assert error["error"] == "RuntimeError: synthetic renderer failure"


def test_concurrent_document_publish_and_heartbeat_keep_one_complete_live_strip(tmp_path):
    destination = tmp_path / "RUN.md"
    run_record.publish_document(destination,
        "# Research run\n\n<!-- AUTOSIM_LIVE_START -->\nStatus: running\n"
        "<!-- AUTOSIM_LIVE_END -->\nbody\n")
    errors = []
    barrier = threading.Barrier(2)

    def full_report():
        barrier.wait()
        try:
            run_record.publish_document(destination,
                "# Research run\n\n<!-- AUTOSIM_LIVE_START -->\nStatus: completed\n"
                "<!-- AUTOSIM_LIVE_END -->\nnew body\n")
        except Exception as exc:  # pragma: no cover - diagnostic on thread failure
            errors.append(exc)

    def heartbeat():
        barrier.wait()
        try:
            run_record.refresh_live_status(tmp_path, fallback_status="running",
                                           fallback_current="evaluation")
        except Exception as exc:  # pragma: no cover - diagnostic on thread failure
            errors.append(exc)

    workers = [threading.Thread(target=full_report), threading.Thread(target=heartbeat)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    document = destination.read_text(encoding="utf-8")
    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert document.count("<!-- AUTOSIM_LIVE_START -->") == 1
    assert document.count("<!-- AUTOSIM_LIVE_END -->") == 1
    assert "new body" in document


def test_the_document_says_how_many_arms_the_run_took(tmp_path):
    """The table below this sentence shows every arm as a row of equal standing, and the run's
    conclusion picks one of them. That picking is the thing a reader cannot see, and it is the
    difference between a number and the highest of several numbers."""
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    for label, value in (("baseline", 0.41), ("round_0", 0.58), ("round_1", None)):
        (measurements / f"{label}.json").write_text(json.dumps({
            "label": label, "metric_value": value, "metric_utility": value,
            "settings": {"episodes": 40, "task": "Lift"}}), encoding="utf-8")
    document = run_record.generate(tmp_path).read_text(encoding="utf-8")
    assert "这次搜索取了多少个点" in document
    assert "2 个有数字的点里的最大值" in document
    assert "共取过 3 个" in document
    assert "挑最大值这个动作本身" in document
    # The arm that produced nothing is named as an arm, because it was one.
    assert "1 条手臂没有产生数字" in document
    assert "round_1" in document


def test_nested_research_report_reads_the_parent_run_state_and_shared_timeline(tmp_path):
    output = tmp_path / "output"
    research = output / "research" / "run-1"
    research.mkdir(parents=True)
    store = ResearchStateStore(output, run_id="run-1", repository=tmp_path)
    store.record("preparation", "action_completed", status="running",
                 details={"step": "derive_stages"},
                 phase_state={"steps": [{"step": "derive_stages", "outcome": "done"}]})
    store.record("research", "stage_started", status="running",
                 details={"stage": "train", "attempt_id": "attempt-1"},
                 phase_state={"current_action": {"step": "train", "status": "running"}})
    (research / "events.json").write_text(json.dumps({"rows": [{
        "at": "2026-09-26T00:00:00Z", "event": "idea_selected", "round": 1}]}),
        encoding="utf-8")

    source = run_record.gather(research)
    section = run_record.timeline(source)

    assert source["state"]["run_id"] == "run-1"
    assert source["state"]["phases"]["preparation"]["steps"][0]["outcome"] == "done"
    assert source["run_events_path"] == "../../run_events.json"
    assert "preparation:action_completed" in section.body
    assert "research:stage_started" in section.body
    assert "idea_selected" in section.body
    assert "当前动作 `train`" in section.body


def test_a_run_with_no_arms_is_not_described_as_a_search(tmp_path):
    """An empty run directory must not get a sentence about maxima. The shape to avoid is a
    paragraph that reads as a finding when nothing was measured."""
    (tmp_path / "measurements").mkdir()
    document = run_record.generate(tmp_path).read_text(encoding="utf-8")
    assert "这次搜索取了多少个点" not in document


def test_the_document_does_not_report_a_missing_file_nothing_writes(tmp_path):
    """`selection.json` was read here and written nowhere, so every run's document carried a
    line saying it was missing. It is computed now, and a run with no arms says nothing."""
    document = run_record.generate(tmp_path).read_text(encoding="utf-8")
    assert "selection.json" not in document
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) \
        if (tmp_path / "summary.json").is_file() else {}
    assert summary.get("unreadable") is None or not any(
        "selection" in str(one) for one in summary.get("unreadable") or [])


def test_a_difference_that_was_not_established_says_so_rather_than_no_improvement(tmp_path):
    """"No improvement" and "this measurement could not tell" are different findings, and a
    run that confuses them teaches its reader to distrust a number that was never shown."""
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    for label, value in (("baseline", 0.41), ("round_0", 0.58)):
        (measurements / f"{label}.json").write_text(json.dumps({
            "label": label, "metric_value": value, "metric_utility": value,
            "settings": {"episodes": 40}}), encoding="utf-8")
    document = run_record.generate(tmp_path).read_text(encoding="utf-8")
    assert "与基线的差别没有被确立" in document
    assert "不是「没有提升」" in document
