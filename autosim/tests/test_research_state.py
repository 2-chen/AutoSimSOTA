import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from autosim.research.research_state import (ResearchStateError, ResearchStateStore,
                                             verify_event_rows)


def test_one_run_state_merges_preparation_and_research_events(tmp_path):
    store = ResearchStateStore(tmp_path / "out", run_id="run-1", repository=tmp_path)
    prepared = store.record(
        "preparation", "action_completed", status="running",
        details={"step": "read_the_checkout", "outcome": "done"},
        phase_state={"steps": [{"step": "read_the_checkout", "outcome": "done"}],
                     "state": {"to_measure": {"ready": False}}},
        event_patch={"last_action": {"step": "read_the_checkout", "outcome": "done"}})
    researched = store.record(
        "research", "stage_started", status="running",
        details={"stage": "train", "attempt_id": "attempt-1",
                 "receipt": "research/run-1/attempts/attempt-1/receipt.json"},
        phase_state={"current_action": {"stage": "train", "status": "running"}},
        event_patch={"current_action": {"stage": "train", "status": "running"}})

    events = json.loads((tmp_path / "out/run_events.json").read_text())
    assert [row["phase"] for row in events["rows"]] == ["preparation", "research"]
    assert verify_event_rows(events["rows"]) == researched["event_head_hash"]
    assert researched["state_revision"] == 2
    assert researched["phase"] == "research"
    assert researched["phases"]["preparation"]["steps"][0]["outcome"] == "done"
    assert researched["phases"]["research"]["current_action"]["stage"] == "train"
    assert prepared["state_revision"] == 1


def test_decision_revision_ignores_runtime_telemetry_but_tracks_research_changes(tmp_path):
    store = ResearchStateStore(tmp_path / "out", run_id="run-1", repository=tmp_path)
    prepared = store.record("preparation", "action_completed", status="running",
                            details={"step": "read_the_checkout"})
    observed = store.record("agent_runtime", "agent_text_delta", status="running",
                            details={"role": "scheduler"}, decision_relevant=False)
    researched = store.record("research", "candidate_started", status="running",
                              details={"candidate": "candidate-1"})

    assert prepared["state_revision"] == 1
    assert observed["state_revision"] == 2
    assert observed["decision_revision"] == prepared["decision_revision"] == 1
    assert researched["state_revision"] == 3
    assert researched["decision_revision"] == 2
    assert store.load()["decision_revision"] == 2


def test_old_state_snapshot_migrates_decision_revision_from_event_history(tmp_path):
    root = tmp_path / "out"
    store = ResearchStateStore(root, run_id="run-1", repository=tmp_path)
    store.record("preparation", "action_completed", status="running")
    store.record("agent_runtime", "agent_text_delta", status="running",
                 decision_relevant=False)
    snapshot_path = root / "run_state.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot.pop("decision_revision")
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    restored = store.load()

    assert restored["state_revision"] == 2
    assert restored["decision_revision"] == 1


def test_event_log_replays_projection_after_snapshot_write_was_missed(tmp_path):
    root = tmp_path / "out"
    store = ResearchStateStore(root, run_id="run-1", repository=tmp_path)
    state = store.record("preparation", "action_started", status="running",
                         details={"step": "build_the_environment"},
                         phase_state={"state": {"environment": {"ready": False}},
                                      "steps": [{"step": "scan", "outcome": "done"}],
                                      "current_action": {"step": "build_the_environment",
                                                         "status": "running"}},
                         event_patch={"current_action": {"step": "build_the_environment",
                                                          "status": "running"}})
    state["event_sequence"] = 0
    state["state_revision"] = 0
    state["phases"] = {}
    state.pop("current_action", None)
    (root / "run_state.json").write_text(json.dumps(state), encoding="utf-8")

    restored = store.load()

    assert restored["event_sequence"] == 1
    assert restored["state_revision"] == 1
    assert restored["current_action"]["step"] == "build_the_environment"
    assert restored["phases"]["preparation"]["status"] == "running"
    assert restored["phases"]["preparation"]["steps"][0]["outcome"] == "done"
    assert restored["phases"]["preparation"]["state"]["environment"]["ready"] is False


def test_appended_event_survives_snapshot_write_failure_and_replays(monkeypatch, tmp_path):
    from autosim.research import research_state

    root = tmp_path / "out"
    store = ResearchStateStore(root, run_id="run-1", repository=tmp_path)
    original_write = research_state.atomic_json

    def fail_projection(path, value):
        if Path(path) == store.state_path:
            raise OSError("injected snapshot failure")
        return original_write(path, value)

    monkeypatch.setattr(research_state, "atomic_json", fail_projection)
    with pytest.raises(OSError, match="injected snapshot failure"):
        store.record("preparation", "action_started", status="running",
                     phase_state={"steps": [{"step": "scan", "outcome": "done"}],
                                  "current_action": {"step": "build", "status": "running"}},
                     event_patch={"current_action": {"step": "build", "status": "running"}})
    assert (root / "run_events.json").is_file()
    assert not store.state_path.exists()

    monkeypatch.undo()
    restored = ResearchStateStore(root, run_id="run-1", repository=tmp_path).load()
    assert restored["event_sequence"] == 1
    assert restored["phases"]["preparation"]["steps"][0]["outcome"] == "done"
    assert restored["current_action"]["step"] == "build"


def test_event_append_failure_before_replace_preserves_the_previous_log(monkeypatch, tmp_path):
    from autosim.research import research_state

    root = tmp_path / "out"
    store = ResearchStateStore(root, run_id="run-1", repository=tmp_path)
    first = store.record("preparation", "action_completed", status="running",
                         details={"step": "scan"})
    original_write = research_state.atomic_json

    def fail_event(path, value):
        if Path(path) == store.events_path:
            raise OSError("injected event temp-write failure")
        return original_write(path, value)

    monkeypatch.setattr(research_state, "atomic_json", fail_event)
    with pytest.raises(OSError, match="event temp-write"):
        store.record("preparation", "action_started", status="running",
                     details={"step": "build"})

    held = json.loads(store.events_path.read_text(encoding="utf-8"))
    assert [row["sequence"] for row in held["rows"]] == [1]
    assert held["rows"][0]["event_hash"] == first["event_head_hash"]
    monkeypatch.undo()
    recovered = ResearchStateStore(root, run_id="run-1", repository=tmp_path)
    assert recovered.load()["state_revision"] == 1
    second = recovered.record("preparation", "action_started", status="running",
                              details={"step": "build"})
    assert second["event_sequence"] == 2


def test_event_replace_then_error_is_reconciled_as_a_committed_event(monkeypatch, tmp_path):
    """An exception after replace is an ambiguous acknowledgement, not a safe retry signal."""
    from autosim.research import research_state

    root = tmp_path / "out"
    store = ResearchStateStore(root, run_id="run-1", repository=tmp_path)
    original_write = research_state.atomic_json

    def commit_then_raise(path, value):
        original_write(path, value)
        if Path(path) == store.events_path:
            raise OSError("injected post-replace directory sync failure")

    monkeypatch.setattr(research_state, "atomic_json", commit_then_raise)
    with pytest.raises(OSError, match="post-replace"):
        store.record("preparation", "action_started", status="running",
                     phase_state={"current_action": {"step": "build", "status": "running"}},
                     event_patch={"current_action": {"step": "build", "status": "running"}})

    monkeypatch.undo()
    recovered = ResearchStateStore(root, run_id="run-1", repository=tmp_path).load()
    assert recovered["event_sequence"] == 1
    assert recovered["state_revision"] == 1
    assert recovered["current_action"] == {"step": "build", "status": "running"}


def test_concurrent_phase_writers_keep_one_valid_shared_event_order(tmp_path):
    store = ResearchStateStore(tmp_path / "out", run_id="run-1", repository=tmp_path)

    def record(index):
        return store.record("phase" + str(index % 2), "checkpoint", status="running",
                            details={"writer": index})

    with ThreadPoolExecutor(max_workers=6) as workers:
        list(workers.map(record, range(18)))
    held = json.loads((tmp_path / "out/run_events.json").read_text(encoding="utf-8"))
    assert [row["sequence"] for row in held["rows"]] == list(range(1, 19))
    assert verify_event_rows(held["rows"]) == held["rows"][-1]["event_hash"]
    assert store.load()["state_revision"] == 18


def test_tampered_event_log_is_refused_without_rewriting_it(tmp_path):
    root = tmp_path / "out"
    store = ResearchStateStore(root, run_id="run-1", repository=tmp_path)
    store.record("preparation", "action_completed", status="running",
                 details={"outcome": "done"})
    path = root / "run_events.json"
    events = json.loads(path.read_text(encoding="utf-8"))
    events["rows"][0]["details"]["outcome"] = "changed"
    tampered = json.dumps(events)
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(ResearchStateError, match="hash is invalid"):
        store.load()
    assert path.read_text(encoding="utf-8") == tampered


def test_state_store_rejects_cross_run_reuse(tmp_path):
    store = ResearchStateStore(tmp_path / "out", run_id="run-1", repository=tmp_path)
    store.record("preparation", "action_completed", status="running")
    other = ResearchStateStore(tmp_path / "out", run_id="run-2", repository=tmp_path)

    with pytest.raises(ResearchStateError, match="another run"):
        other.record("research", "phase_started", status="running")
