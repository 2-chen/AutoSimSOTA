"""A native recording stage is actually invoked and its video is attributed."""

import json

from autosim.research.declarative_backend import DeclarativeBackend
from autosim.research import media_manifest
from tests.test_derived_research import research


def test_baseline_triggers_attributed_native_demo(tmp_path):
    real = research(tmp_path, stages=("train", "evaluate", "record_demo"))
    real.declaration = {"task_contract": {"demo_stage": "record_demo"}}
    real.backend.answer["stages"]["record_demo"]["artifact"] = "rollout.gif"
    real.backend.sources["record_demo"] = (
        "def stage_argv_record_demo(i):\n"
        "    return [i['python'], '-c', "
        "'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); (p/\"rollout.gif\").write_bytes(b\"GIF89a\")', i['output']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                       sources=real.backend.sources,
                                       parameters=real.backend.parameters)
    real.run(rounds=0, settings={"task": "PushCube-v1", "seed": 7})
    events = json.loads((real.run_root / "media_events.json").read_text())["rows"]
    assert len(events) == 1
    assert events[0]["status"] == "captured"
    assert events[0]["trigger"] == "first_native_baseline"
    assert events[0]["policy_sha256"]
    assert events[0]["task"] == "PushCube-v1"
    assert events[0]["seed"] == 7
    assert events[0]["episode_id"] is None
    assert events[0]["episode_identity_status"] == \
        "not_reported_by_native_recording_stage"
    assert events[0]["evaluation_attempt_id"]
    assert events[0]["evaluation_receipt_sha256"]
    assert len(events[0]["comparison_protocol_sha256"]) == 64
    assert events[0]["seconds"] >= events[0]["stage_seconds"]
    manifest = json.loads((real.run_root / "media" / "manifest.json").read_text())
    recording = next(row for row in manifest["recordings"] if row["trigger"])
    assert recording["policy_sha256"] == events[0]["policy_sha256"]
    assert recording["task"] == "PushCube-v1" and recording["seed"] == 7
    assert recording["evaluation_attempt_id"] == events[0]["evaluation_attempt_id"]
    assert recording["comparison_protocol_sha256"] == \
        events[0]["comparison_protocol_sha256"]
    assert recording["sha256"] and recording["hash_status"] == "checked"
    assert recording["episode"] is None
    assert recording["same_scored_evaluation"] is None
    document = (real.run_root / "RUN.md").read_text()
    assert "主动采集的真实 demo" in document
    assert "PushCube-v1" in document and "seed：7" in document
    assert "未报告（原生录制器未提供 episode ID）" in document
    assert "评测回执 SHA-256" in document
    assert "媒体 SHA-256" in document and "协议 SHA-256" in document
    assert f"attempts/{events[0]['evaluation_attempt_id']}/receipt.json" in document


def test_oversize_active_media_keeps_provenance_without_claiming_verified_demo(
        tmp_path, monkeypatch):
    capture = media_manifest.capture_attempt
    write = media_manifest.write

    def defer_capture(root, stage_directory, *, since, max_files=64,
                      max_hash_bytes=256 * 1024**2):
        return capture(root, stage_directory, since=since, max_files=max_files,
                       max_hash_bytes=0)

    def defer_manifest(root, survey, *, max_hash_bytes=256 * 1024**2):
        return write(root, survey, max_hash_bytes=0)

    monkeypatch.setattr(media_manifest, "capture_attempt", defer_capture)
    monkeypatch.setattr(media_manifest, "write", defer_manifest)

    real = research(tmp_path, stages=("train", "evaluate", "record_demo"))
    real.declaration = {"task_contract": {"demo_stage": "record_demo"}}
    real.backend.answer["stages"]["record_demo"]["artifact"] = "rollout.gif"
    real.backend.sources["record_demo"] = (
        "def stage_argv_record_demo(i):\n"
        "    return [i['python'], '-c', "
        "'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); (p/\"rollout.gif\").write_bytes(b\"GIF89a\")', i['output']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                       sources=real.backend.sources,
                                       parameters=real.backend.parameters)

    real.run(rounds=0, settings={"task": "PushCube-v1", "seed": 7})

    events = json.loads((real.run_root / "media_events.json").read_text())["rows"]
    assert len(events) == 1 and events[0]["status"] == "captured"
    manifest = json.loads((real.run_root / "media" / "manifest.json").read_text())
    recording = next(row for row in manifest["recordings"] if row["trigger"])
    assert recording["sha256"] is None
    assert recording["hash_status"] == "deferred_size_limit"
    assert recording["media_identity_status"] == "hash_deferred_size_limit"
    assert recording["attempt_id"] is None
    assert recording["reported_by_attempt_id"] == events[0]["attempt_id"]
    assert recording["evaluation_attempt_id"] == events[0]["evaluation_attempt_id"]
    assert recording["task"] == "PushCube-v1" and recording["seed"] == 7

    document = (real.run_root / "RUN.md").read_text()
    assert "主动媒体待验证（未纳入已验证 demo）" in document
    assert "hash_deferred_size_limit" in document
    assert "主动采集的真实 demo" not in document
