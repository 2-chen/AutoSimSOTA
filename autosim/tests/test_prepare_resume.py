"""What a preparation picks up from earlier runs, and what it must not do over again.

Both defects here were found by running the system on LIBERO, which had a working interpreter,
a passing environment verdict and two verified stage commands. The preparation built the
environment again from nothing -- and LIBERO's build begins with a `conda create` into the
prefix that held the interpreter, so a run that had lost nothing destroyed a verified
environment on its way past.

The state had always been read from the records. The *steps* had not, which is why the loop's
own prompt ("a step already done does not need doing again") was not true of the code.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from autosim.research.prepare import Preparation
from autosim.research import prepare as prepare_module
from autosim.research.codepatch import Patch, apply_many
from autosim.research.ideas import CLEARED, Idea, IdeaLibrary
from autosim.research.snapshot import Snapshots
from autosim.research.process_executor import (capture_process_identity,
                                               observe_process_starts, run_process,
                                               terminate_recorded_process)
from autosim.research.research_state import ResearchStateStore


class _Client:
    """Any model call at all is a failure here: these are record reads, not decisions."""

    model = "a-test-client"

    def chat_with_metadata(self, *args, **kwargs):
        raise AssertionError("no model call was expected")


def _built(tmp_path: Path, *, interpreter_exists: bool = True) -> Path:
    out = tmp_path / "out"
    (out / "env/bin").mkdir(parents=True)
    binary = out / "env/bin/python"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    if not interpreter_exists:
        binary.unlink()
    (out / "environment.json").write_text(json.dumps({
        "verdict": {"passed": True}, "interpreter": str(binary),
        "record": [{"command": "conda create -y -p {prefix} python=3.8.13"}],
        "probes": ["import libero"]}), encoding="utf-8")
    (out / "derived_stages.json").write_text(json.dumps({
        "train": {"source": "def stage_argv_train(i):\n    return [i['python']]\n",
                  "parameters": {}},
        "evaluate": {"source": "def stage_argv_evaluate(i):\n    return [i['python']]\n",
                     "parameters": {}}}), encoding="utf-8")
    return out


def _prep(tmp_path: Path, out: Path) -> Preparation:
    return Preparation(repo=tmp_path, output=out, client=_Client(),
                       scouting=tmp_path / "scouting")


def test_isolated_copy_uses_original_repository_name_for_declaration_lookup(tmp_path):
    source = tmp_path / "robomimic"
    source.mkdir()
    output = tmp_path / "out"
    checkout = output / "checkout"
    checkout.mkdir(parents=True)
    (output / "workspace_snapshot.json").write_text(json.dumps({
        "source": str(source), "destination": str(checkout)}))
    scouting = tmp_path / "scouting" / "registered"
    scouting.mkdir(parents=True)
    (scouting / "declaration.json").write_text(json.dumps({"declaration": {
        "benchmark": "robomimic", "optimization_space": {
            "collection": [], "training": [{"name": "steps", "kind": "integer",
                "description": "train steps", "low": 1, "high": 2, "default": 1}]}}}))
    prepared = Preparation(repo=checkout, output=output, client=_Client(),
                           scouting=tmp_path / "scouting")
    assert prepared._benchmark_name() == "robomimic"
    assert prepared.declaration["benchmark"] == "robomimic"


def test_the_commands_an_earlier_derivation_verified_come_back(tmp_path):
    """They are the expensive product of the derivation and the record holds them."""
    out = _built(tmp_path)
    preparation = _prep(tmp_path, out)
    assert sorted(preparation.stages) == ["evaluate", "train"]


def test_the_interpreter_an_earlier_build_produced_comes_back(tmp_path):
    out = _built(tmp_path)
    assert _prep(tmp_path, out).interpreter == out / "env/bin/python"


def test_a_passing_build_is_not_built_over(tmp_path):
    """The defect that destroyed LIBERO's environment."""
    out = _built(tmp_path)
    result = _prep(tmp_path, out).do("build_the_environment")
    assert result["outcome"] == "already done"
    assert "nothing to rebuild" in result["because"]


def test_a_record_that_passed_but_lost_its_interpreter_is_rebuilt(tmp_path):
    """The record says the build passed; the interpreter it names is gone. Those are two
    different facts, and the second is the one that decides what can run."""
    out = _built(tmp_path, interpreter_exists=False)
    preparation = _prep(tmp_path, out)
    assert preparation.interpreter is None
    # Not "already done": there is nothing to run commands with.
    assert preparation.do("build_the_environment")["outcome"] != "already done"


def test_nothing_is_read_from_a_directory_with_no_records(tmp_path):
    """A first run has nothing to pick up, and says so by having nothing."""
    preparation = _prep(tmp_path, tmp_path / "empty")
    assert preparation.stages == {} and preparation.interpreter is None
    assert preparation.declaration == {}


def test_prior_step_history_is_restored_only_for_its_run_and_repository(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    path = out / "preparation_derived.json"
    first = {"step": "build_the_environment", "outcome": "done",
             "because": "interpreter verified"}
    path.write_text(json.dumps({"run_id": "derived", "repository": str(tmp_path),
                               "steps": [first]}), encoding="utf-8")

    preparation = _prep(tmp_path, out)

    assert preparation.steps == [first]
    assert preparation.state()["steps_taken"] == [{**first, "attempts": 1}]

    path.write_text(json.dumps({"run_id": "another-run", "repository": str(tmp_path),
                               "steps": [{"step": "declare", "outcome": "done"}]}),
                    encoding="utf-8")
    other_run = _prep(tmp_path, out)
    assert other_run.steps == []

    path.write_text(json.dumps({"run_id": "derived", "repository": str(tmp_path / "other"),
                               "steps": [{"step": "declare", "outcome": "done"}]}),
                    encoding="utf-8")
    other_repository = _prep(tmp_path, out)
    assert other_repository.steps == []


def test_interrupted_action_is_restored_as_unknown_not_as_completed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    previous = {"step": "build_the_environment", "outcome": "done",
                "because": "previous verified build"}
    active = {"step": "derive_a_command", "status": "running",
              "arguments": {"stage": "train"}, "started_at": "earlier"}
    (out / "run_state.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "derived", "repository": str(tmp_path),
        "phase": "preparation", "status": "running", "state_revision": 4,
        "steps": [previous],
        "current_action": active, "last_decision": {"step": "derive_a_command"}}),
        encoding="utf-8")

    preparation = _prep(tmp_path, out)

    assert preparation.steps == [previous]
    assert preparation.state_revision == 4
    recovery = preparation.state()["recovery_after_interruption"]
    assert recovery["action"] == active
    assert "outcome is unknown" in recovery["reason"]
    assert preparation.choose()["do"] == "reconcile_interrupted_action"
    assert preparation.do("read_the_checkout")["outcome"] == "blocked"
    assert preparation.do("reconcile_interrupted_action")["outcome"] == "blocked"
    assert preparation.recovery_after_interruption
    assert preparation.steps[-1]["outcome"] == "done"
    preparation._refresh_document(status="running", current="reconcile interrupted action")
    document = (out / "RUN.md").read_text(encoding="utf-8")
    assert "## Recovery required" in document
    assert "derive_a_command" in document


@pytest.mark.parametrize(("controller_action", "safe_to_resume"),
                         [("baseline", True), ("round_1", False)])
def test_interrupted_research_controller_action_without_child_process_is_checkpointed(
        tmp_path, controller_action, safe_to_resume):
    out = tmp_path / "out"
    research_root = out / "research" / "derived"
    research_root.mkdir(parents=True)
    session_path = research_root / "controller_session.json"
    pending_action = ({"round": 1, "phase": "preparing_idea",
                       "idea": {"label": "candidate idea"}}
                      if controller_action != "baseline" else None)
    session_path.write_text(json.dumps({
        "schema_version": 1, "run_id": "derived", "repository": str(tmp_path),
        "status": "running", "action": controller_action, "rounds": 1,
        "next_round": 1, "history": [], "pending_action": pending_action,
    }), encoding="utf-8")
    report_path = research_root / "research_report.json"
    report_path.write_text(json.dumps({"run_id": "derived", "repo": str(tmp_path),
                                       "run_status": "paused"}), encoding="utf-8")
    store = ResearchStateStore(out, run_id="derived", repository=tmp_path)
    action = {"step": controller_action, "status": "running",
              "started_at": "before-crash"}
    store.record("research", "research_action_started", status="running",
                 details={"action": action}, phase_state={"current_action": action},
                 event_patch={"current_action": action})

    restarted = _prep(tmp_path, out)
    restarted.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    restarted.interpreter = Path(sys.executable)
    restarted.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    restarted._metric_bound = lambda: True
    assert restarted.recovery_after_interruption["process_status"]["status"] == "not_recorded"
    result = restarted.do("reconcile_interrupted_action")

    assert result["outcome"] == "interrupted; outcome unknown"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["status"] == "interrupted"
    assert session["interruption"]["controller_action"] == controller_action
    assert session["interruption"]["evidence_ref"] is None
    if pending_action:
        assert session["interruption"]["pending_action"] == pending_action
        boundary = session["interruption"]["boundary"]
        assert boundary["status"] == "unresolved"
        assert boundary["reason_code"] == "candidate_interrupted_before_controller_commit"
        assert boundary["round"] == pending_action["round"]
        assert boundary["idea"]["label"] == "candidate idea"
        assert boundary["candidate_measurement"]["status"] == "missing"
        assert boundary["automatic_replay"] is False
        assert boundary["automatic_measurement_adoption"] is False
        assert boundary["automatic_source_rollback"] is False
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["run_status"] == "interrupted"
        assert report["interruption"]["boundary"] == boundary
    else:
        assert "boundary" not in session["interruption"]
    state = restarted.state()
    assert state["research_progress"]["safe_to_resume"] is safe_to_resume
    if pending_action:
        assert state["research_progress"]["interruption_boundary"] == boundary
    assert ("run_the_loop" in state["available"]) is safe_to_resume


def test_candidate_boundary_reports_unverifiable_measurement_without_adopting_it(tmp_path):
    out = tmp_path / "out"
    preparation = _prep(tmp_path, out)
    measurement_root = out / "research" / preparation.run_id / "measurements"
    measurement_root.mkdir(parents=True)
    (measurement_root / "round_3.json").write_text("{}", encoding="utf-8")

    boundary = preparation._candidate_interruption_boundary(
        controller_action="round_3",
        pending_action={"round": 3, "phase": "measurement_starting",
                        "idea": {"label": "candidate", "granularity": "code"},
                        "changed": {"files": ["policy.py"]}},
        evidence_ref="research/derived/attempts/a/receipt.json",
        evidence_status="interrupted", process_status="not_running")

    assert boundary["pending_action_identity"] == "matched"
    assert boundary["candidate_measurement"]["status"] == "unverifiable"
    assert boundary["candidate_measurement"]["ref"].endswith("round_3.json")
    assert boundary["changed_source_paths"] == ["policy.py"]
    assert boundary["automatic_measurement_adoption"] is False
    assert boundary["automatic_source_rollback"] is False


def test_candidate_interruption_boundary_exposes_unknown_finalization_side_effects(tmp_path):
    preparation = _prep(tmp_path, tmp_path / "out")
    pending = {
        "round": 1, "phase": "measurement_recorded",
        "idea": {"label": "candidate", "granularity": "param"},
        "round_finalization": {
            "schema_version": 1, "transaction_id": "f" * 32,
            "status": "unresolved", "round": 1, "idea_label": "candidate",
            "measurement_sha256": "1" * 64, "result_sha256": "2" * 64,
            "steps": {"best_state": {"status": "completed"},
                      "demo_capture": {"status": "started"},
                      "idea_outcome": {"status": "pending"}},
        },
    }

    boundary = preparation._candidate_interruption_boundary(
        controller_action="round_1", pending_action=pending,
        evidence_ref="research/derived/attempts/a/receipt.json",
        evidence_status="interrupted", process_status="terminated")

    assert boundary["status"] == "unresolved"
    assert boundary["automatic_replay"] is False
    assert boundary["automatic_measurement_adoption"] is False
    assert boundary["round_finalization"]["transaction_id"] == "f" * 32
    assert boundary["in_flight_finalization_steps"] == ["demo_capture"]
    assert boundary["round_finalization"]["steps"]["idea_outcome"]["status"] == "pending"


def _committed_candidate_recovery(preparation, output, *, attempt_id):
    """Create the outer-controller records around an already-proven round commit."""
    research_root = output / "research" / preparation.run_id
    attempts = research_root / "attempts" / attempt_id
    attempts.mkdir(parents=True)
    (attempts / "receipt.json").write_text(json.dumps({
        "run_id": preparation.run_id, "attempt_id": attempt_id,
        "status": "interrupted",
    }), encoding="utf-8")
    (research_root / "rubric.json").write_text(json.dumps({
        "checks": [{"name": "measured", "question": "Was it measured?",
                    "weight": 1.0, "passed": True, "because": "receipt verified"}],
    }), encoding="utf-8")
    history = [
        {"round": 0, "label": "baseline", "metric_value": 0.2},
        {"round": 1, "label": "round_1", "metric_value": 0.3,
         "finalization_id": "a" * 32},
    ]
    session = {
        "schema_version": 1, "run_id": preparation.run_id,
        "repository": str(preparation.repo), "status": "running",
        "action": "round_1", "rounds": 2, "next_round": 2,
        "history": history, "kinds": ["param"], "pending_action": None,
        "stopped_because": "",
    }
    session_path = research_root / "controller_session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    report_path = research_root / "research_report.json"
    report_path.write_text(json.dumps({
        "run_id": preparation.run_id, "repo": str(preparation.repo),
        "run_status": "interrupted", "rounds": history,
    }), encoding="utf-8")
    boundary = {
        "status": "resolved_committed", "round": 1,
        "round_finalization": {
            "status": "committed", "transaction_id": "a" * 32,
            "history_row_sha256": "b" * 64,
            "measurement_sha256": "c" * 64,
        },
        "round_history_sha256": "b" * 64,
        "receipt_proof": {"status": "consistent"},
        "attempt_evidence": {"attempt_id": attempt_id,
                             "receipt_status": "interrupted",
                             "outcome_known": True},
    }
    preparation._candidate_interruption_boundary = lambda **_: boundary

    class Projection:
        def build_rubric(self):
            return None

        def _controller_report(self, **kwargs):
            return {"schema_version": 1, "run_id": preparation.run_id,
                    "repo": str(preparation.repo), "rounds": kwargs["history"],
                    "run_status": kwargs["run_status"],
                    "next_round": kwargs["next_round"],
                    "planned_rounds": kwargs["rounds"],
                    "objective": {"checks": ["persisted"]}}

    preparation._research_controller = lambda: (Projection(), {})
    evidence_ref = f"research/{preparation.run_id}/attempts/{attempt_id}/receipt.json"
    preparation.recovery_after_interruption = {
        "action": {"step": "round_1", "attempt_id": attempt_id,
                   "receipt_ref": evidence_ref,
                   "parent_action": {"step": "round_1"}},
        "parent_action": {"step": "round_1"},
        "process_status": {"status": "not_running"},
    }
    return research_root, session_path, report_path, boundary


def test_committed_candidate_reconciliation_rebuilds_projection_without_replay(tmp_path):
    out = tmp_path / "out"
    preparation = _prep(tmp_path, out)
    attempt_id = "d" * 32
    research_root, session_path, report_path, boundary = _committed_candidate_recovery(
        preparation, out, attempt_id=attempt_id)

    result = preparation._step_reconcile_interrupted_action()

    assert result["outcome_known"] is True
    assert result["outcome"] == "committed candidate round verified"
    assert result["transaction_id"] == "a" * 32
    session = json.loads(session_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert session["status"] == "paused" and session["next_round"] == 2
    assert session["pending_action"] is None
    assert session["reconciliation"]["status"] == "committed_round_verified"
    assert report["run_status"] == "paused"
    assert report["rounds"] == session["history"]
    assert report["recovered_candidate_finalization"]["transaction_id"] == "a" * 32
    assert report["interruption_boundary"]["status"] == boundary["status"]
    assert len(list((research_root / "attempts").glob("*/receipt.json"))) == 1

    # A crash after committing the session but before advancing the outer run state is safe:
    # the same committed transaction may repair its projection again, never rerun its stage.
    preparation.recovery_after_interruption = {
        "action": {"step": "round_1", "attempt_id": attempt_id,
                   "receipt_ref": f"research/{preparation.run_id}/attempts/"
                                 f"{attempt_id}/receipt.json",
                   "parent_action": {"step": "round_1"}},
        "parent_action": {"step": "round_1"},
        "process_status": {"status": "not_running"},
    }
    repeated = preparation._step_reconcile_interrupted_action()
    assert repeated["outcome_known"] is True
    assert repeated["transaction_id"] == "a" * 32
    assert len(list((research_root / "attempts").glob("*/receipt.json"))) == 1


def test_committed_candidate_projection_failure_does_not_advance_session(tmp_path,
                                                                         monkeypatch):
    out = tmp_path / "out"
    preparation = _prep(tmp_path, out)
    attempt_id = "e" * 32
    _, session_path, report_path, _ = _committed_candidate_recovery(
        preparation, out, attempt_id=attempt_id)
    original_atomic_json = prepare_module.atomic_json

    def fail_report(path, value, **kwargs):
        if Path(path) == report_path:
            raise OSError("simulated projection write failure")
        return original_atomic_json(path, value, **kwargs)

    monkeypatch.setattr(prepare_module, "atomic_json", fail_report)
    with pytest.raises(OSError, match="projection write failure"):
        preparation._step_reconcile_interrupted_action()
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["status"] == "running"
    assert session["action"] == "round_1"


def test_pre_measurement_interruption_rolls_back_only_the_exact_candidate_patch(tmp_path):
    out = tmp_path / "out"
    source = tmp_path / "loader.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    patch = Patch("loader.py", "VALUE = 1", "VALUE = 2")
    preparation = _prep(tmp_path, out)
    research_root = out / "research" / preparation.run_id
    snapshots = Snapshots(research_root / "snapshots")
    snapshots.capture([patch.file], repo=tmp_path, name="before-round-1")
    apply_many([patch], repo=tmp_path)
    manifest_dir = research_root / "code_changes"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "round-1.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "round": 1, "idea_label": "candidate",
        "snapshot": "before-round-1", "patches": [patch.as_dict()],
        "status": "applying",
    }), encoding="utf-8")
    preparation.recovery_after_interruption = {"action": {"step": "round_1"}}

    boundary = preparation._candidate_interruption_boundary(
        controller_action="round_1",
        pending_action={"round": 1, "phase": "preparing_idea",
                        "idea": {"label": "candidate", "granularity": "code"}},
        evidence_ref=None, evidence_status="not_recorded", process_status="not_recorded")

    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert boundary["status"] == "unresolved"
    assert boundary["automatic_source_rollback"] is True
    assert boundary["automatic_measurement_adoption"] is False
    assert boundary["changed_source_paths"] == ["loader.py"]
    assert boundary["source_rollback"]["status"] == "rolled_back"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "rolled_back"


def test_pre_measurement_interruption_preserves_source_when_it_diverged(tmp_path):
    out = tmp_path / "out"
    source = tmp_path / "loader.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    patch = Patch("loader.py", "VALUE = 1", "VALUE = 2")
    preparation = _prep(tmp_path, out)
    research_root = out / "research" / preparation.run_id
    snapshots = Snapshots(research_root / "snapshots")
    snapshots.capture([patch.file], repo=tmp_path, name="before-round-1")
    apply_many([patch], repo=tmp_path)
    source.write_text("VALUE = 3\n", encoding="utf-8")
    manifest_dir = research_root / "code_changes"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "round-1.json").write_text(json.dumps({
        "schema_version": 1, "round": 1, "idea_label": "candidate",
        "snapshot": "before-round-1", "patches": [patch.as_dict()],
        "status": "applied",
    }), encoding="utf-8")
    preparation.recovery_after_interruption = {"action": {"step": "round_1"}}

    boundary = preparation._candidate_interruption_boundary(
        controller_action="round_1",
        pending_action={"round": 1, "phase": "preparing_idea",
                        "idea": {"label": "candidate", "granularity": "code"}},
        evidence_ref=None, evidence_status="not_recorded", process_status="not_recorded")

    assert source.read_text(encoding="utf-8") == "VALUE = 3\n"
    assert boundary["automatic_source_rollback"] is False
    assert boundary["changed_source_paths"] == ["loader.py"]
    assert boundary["source_rollback"]["status"] == "refused"
    assert "diverged" in boundary["source_rollback"]["because"]


def _discardable_candidate(tmp_path: Path, *, phase: str = "prepared"):
    out = tmp_path / "out"
    preparation = _prep(tmp_path, out)
    research_root = out / "research" / preparation.run_id
    research_root.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "loader.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    patch = Patch("loader.py", "VALUE = 1", "VALUE = 2")
    snapshots = Snapshots(research_root / "snapshots")
    snapshots.capture([patch.file], repo=tmp_path, name="before-round-1")
    apply_many([patch], repo=tmp_path)
    manifest_dir = research_root / "code_changes"
    manifest_dir.mkdir()
    (manifest_dir / "round-1.json").write_text(json.dumps({
        "schema_version": 1, "round": 1, "idea_label": "candidate",
        "snapshot": "before-round-1", "patches": [patch.as_dict()],
        "status": "applied",
    }), encoding="utf-8")
    idea = Idea(label="candidate", granularity="code", mechanism="change loader",
                change={"file": patch.file, "find": patch.find, "replace": patch.replace},
                status=CLEARED, touches=[patch.file])
    library = IdeaLibrary(research_root / "ideas.json")
    library.add(idea)
    pending = {"round": 1, "phase": phase, "idea": idea.as_dict(),
               "changed": {"file": patch.file, "files": [patch.file]}}
    boundary = {
        "schema_version": 1, "status": "unresolved",
        "reason_code": "candidate_interrupted_before_controller_commit",
        "controller_action": "round_1", "round": 1,
        "pending_action_identity": "matched", "phase": phase,
        "idea": {"label": idea.label, "granularity": idea.granularity},
        "attempt_evidence": {"attempt_id": None, "receipt_ref": None,
                             "receipt_status": "not_recorded",
                             "process_status": "not_recorded", "outcome_known": False},
        "candidate_measurement": {"status": "missing",
                                  "ref": "research/derived/measurements/round_1.json"},
        "changed_source_paths": [patch.file],
        "automatic_replay": False, "automatic_measurement_adoption": False,
    }
    session = {
        "schema_version": 1, "run_id": preparation.run_id,
        "repository": str(tmp_path), "status": "interrupted", "action": "round_1",
        "rounds": 2, "next_round": 1, "history": [{"round": 0, "label": "baseline",
                                                       "status": "measured"}],
        "kinds": [], "current_measurement_label": "baseline", "pending_action": pending,
        "interruption": {"controller_action": "round_1", "pending_action": pending,
                         "boundary": boundary},
        "reconciliation": {"status": "reconciled", "process_status": "not_recorded"},
    }
    session_path = research_root / "controller_session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    report_path = research_root / "research_report.json"
    report_path.write_text(json.dumps({
        "run_id": preparation.run_id, "repo": str(tmp_path), "run_status": "interrupted",
        "rounds": session["history"], "planned_rounds": 2,
        "best": {"name": "baseline"},
    }), encoding="utf-8")
    preparation.current_action = {"decision_id": "main-discard-decision"}
    return preparation, source, research_root, session_path, report_path


def test_main_controller_can_discard_only_a_reconciled_pre_stage_candidate(tmp_path):
    preparation, source, research_root, session_path, report_path = _discardable_candidate(
        tmp_path)

    assert "discard_interrupted_candidate" in preparation.state()["available"]
    result = preparation.do("discard_interrupted_candidate",
                            reason="No benchmark stage or measurement began; discard this idea.")

    assert result["outcome"] == "candidate discarded"
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert session["status"] == report["run_status"] == "paused"
    assert session["next_round"] == report["next_round"] == 2
    assert session["current_measurement_label"] == "baseline"
    assert session["history"][-1]["status"] == "interrupted candidate discarded"
    assert "metric_value" not in session["history"][-1]
    assert session["interruption"]["boundary"]["status"] == "resolved_discarded"
    assert report["rounds"][-1]["interruption_resolution_id"] == result["resolution_id"]
    idea = IdeaLibrary(research_root / "ideas.json").get("candidate")
    assert idea.status != CLEARED and idea.times_tried == 1
    assert idea.outcome_event_ids == [result["resolution_id"]]
    repeated = preparation._step_discard_interrupted_candidate(reason="retry")
    assert repeated["outcome"] == "already resolved"
    assert IdeaLibrary(research_root / "ideas.json").get("candidate").times_tried == 1


def test_discard_retry_after_idea_event_write_is_idempotent(tmp_path):
    preparation, _, research_root, session_path, _ = _discardable_candidate(tmp_path)
    resolution_id = "a" * 32
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["interruption_resolution"] = {
        "status": "discarding", "round": 1, "idea_label": "candidate",
        "resolution_id": resolution_id, "decision_id": "main-discard-decision",
        "reason": "the stage never started", "started_at": "test",
    }
    session_path.write_text(json.dumps(session), encoding="utf-8")
    library = IdeaLibrary(research_root / "ideas.json")
    library.note("candidate", worked=False, outcome="discard recorded before crash",
                 event_id=resolution_id)
    assert library.get("candidate").times_tried == 1

    result = preparation._step_discard_interrupted_candidate(reason="retry after restart")

    assert result["outcome"] == "candidate discarded"
    saved = IdeaLibrary(research_root / "ideas.json").get("candidate")
    assert saved.times_tried == 1
    assert saved.outcome_event_ids == [resolution_id]
    completed = json.loads(session_path.read_text(encoding="utf-8"))
    assert completed["interruption_resolution"]["resolution_id"] == resolution_id
    assert completed["status"] == "paused"


def test_discard_refuses_a_candidate_source_diverged_after_measurement_boundary(tmp_path):
    preparation, source, _, session_path, _ = _discardable_candidate(tmp_path)
    source.write_text("VALUE = 3\n", encoding="utf-8")

    result = preparation.do("discard_interrupted_candidate",
                            reason="Discard the interrupted candidate based on its boundary.")

    assert result["outcome"] == "blocked"
    assert "overwriting unknown changes" in result["because"]
    assert source.read_text(encoding="utf-8") == "VALUE = 3\n"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["status"] == "interrupted"
    assert session["interruption_resolution"]["status"] == "blocked"


def test_discard_action_is_not_offered_after_a_benchmark_attempt_started(tmp_path):
    progress = {
        "status": "interrupted",
        "interruption": {"controller_action": "round_1"},
        "reconciliation": {"status": "reconciled"},
        "interruption_boundary": {
            "status": "unresolved", "controller_action": "round_1", "round": 1,
            "phase": "measurement_starting", "pending_action_identity": "matched",
            "idea": {"label": "candidate"},
            "candidate_measurement": {"status": "missing"},
            "attempt_evidence": {"attempt_id": "attempt-1", "receipt_ref": "receipt.json",
                                 "receipt_status": "interrupted",
                                 "process_status": "terminated"},
        },
    }
    assert Preparation._interrupted_candidate_discard_available(progress) is False


def test_reconcile_stops_only_the_recorded_attempt_and_marks_its_receipt_unknown(tmp_path):
    out = tmp_path / "out"
    receipt_ref = "research/derived/attempts/attempt-1/receipt.json"
    receipt_path = out / receipt_ref
    receipt_path.parent.mkdir(parents=True)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               start_new_session=True)
    identity = capture_process_identity(
        process, run_id="derived", attempt_id="attempt-1", argv=process.args)
    action = {"step": "train", "status": "running", "attempt_id": "attempt-1",
              "receipt_ref": receipt_ref, "process_identity": identity}
    store = ResearchStateStore(out, run_id="derived", repository=tmp_path)
    store.record("research", "stage_process_started", status="running",
                 details={"stage": "train", "process_identity": identity},
                 phase_state={"current_action": action,
                              "process_identity": identity},
                 event_patch={"current_action": action,
                              "process_identity": identity})
    receipt_path.write_text(json.dumps({
        "run_id": "derived", "attempt_id": "attempt-1", "status": "running"}),
        encoding="utf-8")

    try:
        preparation = _prep(tmp_path, out)
        assert preparation.choose()["do"] == "reconcile_interrupted_action"
        result = preparation.do("reconcile_interrupted_action")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert result["outcome"] == "interrupted; outcome unknown", result
        assert not preparation.recovery_after_interruption
        assert receipt["status"] == "interrupted"
        assert receipt["process_reconciliation"]["status"] == "terminated"
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_bounded_helper_process_is_journaled_and_reconciled_by_a_new_preparer(tmp_path):
    out = tmp_path / "out"
    first = _prep(tmp_path, out)
    first.current_action = {"step": "build_the_environment", "status": "running",
                            "started_at": "2026-09-26T00:00:00Z"}
    observed = threading.Event()
    attempts = []

    def observe(process):
        first._observe_process_start(process)
        observed.set()

    def launch():
        with observe_process_starts(observe):
            attempts.append(run_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=tmp_path, env=dict(os.environ), timeout=30,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    thread = threading.Thread(target=launch, daemon=True)
    thread.start()
    assert observed.wait(timeout=5)

    restarted = _prep(tmp_path, out)
    recovery = restarted.state()["recovery_after_interruption"]
    assert recovery["process_status"]["status"] == "matching_running"
    result = restarted.do("reconcile_interrupted_action")
    thread.join(timeout=5)

    assert result["outcome"] == "interrupted; outcome unknown"
    assert attempts and attempts[0].returncode is not None
    process_record = json.loads((out / recovery["action"]["process_ref"]).read_text())
    assert process_record["status"] == "interrupted"


def test_hard_killed_controller_restart_reconciles_its_live_helper_process(tmp_path):
    out = tmp_path / "out"
    ready = tmp_path / "controller-ready"
    package_root = str(Path(__file__).resolve().parents[1])
    python_path = os.pathsep.join(
        part for part in (package_root, os.environ.get("PYTHONPATH", "")) if part)
    env = {**os.environ, "PYTHONPATH": python_path}
    code = (
        "import os,sys,subprocess; from pathlib import Path; "
        "from autosim.research.prepare import Preparation; "
        "from autosim.research.process_executor import observe_process_starts,run_process; "
        "p=Preparation(repo=Path(sys.argv[1]),output=Path(sys.argv[2]),client=object(),"
        "scouting=Path(sys.argv[1])/'scouting'); "
        "p.current_action={'step':'build_the_environment','status':'running'}\n"
        "def observe(proc):\n"
        " p._observe_process_start(proc)\n"
        " Path(sys.argv[3]).write_text('ready')\n"
        "with observe_process_starts(observe):\n"
        " run_process([sys.executable,'-c','import time; time.sleep(60)'],"
        "cwd=Path(sys.argv[1]),env=dict(os.environ),timeout=60,"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n")
    controller = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(out),
                                   str(ready)], cwd=tmp_path, env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    identity = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ready.is_file():
            if controller.poll() is not None:
                break
            time.sleep(0.02)
        assert ready.is_file(), "controller did not persist the live process identity"
        state = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
        recovery_action = state["current_action"]
        identity = recovery_action["process_identity"]

        controller.send_signal(signal.SIGKILL)
        controller.wait(timeout=5)
        restarted = _prep(tmp_path, out)
        recovery = restarted.state()["recovery_after_interruption"]
        assert recovery["process_status"]["status"] == "matching_running"
        result = restarted.do("reconcile_interrupted_action")
        process_record = json.loads((out / recovery_action["process_ref"]).read_text())
        assert result["outcome"] == "interrupted; outcome unknown"
        assert process_record["status"] == "interrupted"
        assert process_record["process_reconciliation"]["status"] == "terminated"
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()
        if identity:
            terminate_recorded_process(identity, grace_seconds=0.2)


def test_cli_hard_kill_reconciles_benchmark_stage_without_automatic_retry(tmp_path):
    """Exercise autosim research, its persisted stage identity, and restart handoff."""
    import shutil

    from autosim.research import objective
    from autosim.research.process_executor import inspect_process_identity

    if not shutil.which("bwrap"):
        pytest.skip("benchmark-stage interruption contract requires bubblewrap")

    repo = tmp_path / "toy_sim"
    repo.mkdir()
    (repo / "README.md").write_text("Synthetic simulator fixture.\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    scout = runtime / "autoresearch_runs" / "scouting" / "toy_sim"
    scout.mkdir(parents=True)
    output = tmp_path / "output"
    output.mkdir()
    (output / "environment.json").write_text(json.dumps({
        "verdict": {"passed": True}, "interpreter": sys.executable}), encoding="utf-8")
    stage_marker = "autosim-cli-stage-" + uuid.uuid4().hex
    train_program = f"import time; marker = {stage_marker!r}; time.sleep(60)"
    train_source = ('def stage_argv_train(i):\n'
                    f'    return [i["python"], "-c", {train_program!r}]\n')
    evaluate_source = ('def stage_argv_evaluate(i):\n'
                       '    return [i["python"], "-c", "print(\'success_rate: 0.5\')"]\n')
    (output / "derived_stages.json").write_text(json.dumps({
        "train": {"source": train_source, "parameters": {}},
        "evaluate": {"source": evaluate_source, "parameters": {}},
    }), encoding="utf-8")
    (output / "execution.json").write_text(json.dumps({"stages": {
        "train": {"available": True, "entrypoint": "train.py",
                  "invocation": "python train.py", "artifact": "models/*.pth"},
        "evaluate": {"available": True, "entrypoint": "evaluate.py",
                     "invocation": "python evaluate.py", "artifact": "models/*.pth"},
    }}), encoding="utf-8")
    declaration = {
        "benchmark": "toy_sim", "evidence": "synthetic CLI recovery fixture",
        "repo_markers": ["README.md"], "tasks": {"kind": "module_registry",
                                                     "pattern": "", "names": ["ToyTask"]},
        "task_contract": {"primary_metric": {"name": "success_rate",
                                               "direction": "maximize", "unit": "fraction",
                                               "source": "log", "min": 0, "max": 1}},
        "assets": {}, "capabilities": {},
        "optimization_space": {"collection": [], "training": [{
            "name": "epochs", "kind": "integer", "description": "training epochs",
            "low": 1, "high": 2, "default": 1}]},
    }
    (scout / "declaration.json").write_text(json.dumps({"declaration": declaration}),
                                            encoding="utf-8")
    research_root = output / "research" / "derived"
    research_root.mkdir(parents=True)
    (research_root / "rubric.json").write_text(
        json.dumps(objective.plain().as_dict()), encoding="utf-8")

    package_root = str(Path(__file__).resolve().parents[1])
    python_path = os.pathsep.join(
        part for part in (package_root, os.environ.get("PYTHONPATH", "")) if part)
    env = {**os.environ, "PYTHONPATH": python_path,
           "AUTOSIM_TEST_NO_BWRAP": "0"}
    wrapper = """
import sys
import os
import shutil
from pathlib import Path
import autosim.research.derive_and_run as cli
import autosim.llm_client as llm

if os.environ.get("AUTOSIM_TEST_NO_BWRAP") == "1":
    _which = shutil.which
    shutil.which = lambda name, *args, **kwargs: (
        None if name == "bwrap" else _which(name, *args, **kwargs))

class NoNetwork:
    model = "test"
    def chat(self, *args, **kwargs):
        raise AssertionError("network forbidden")
    def chat_with_metadata(self, *args, **kwargs):
        raise AssertionError("network forbidden")

llm.LLMClient = NoNetwork
llm.load_credential_file = lambda **kwargs: None
cli.ROOT = Path(sys.argv[1])
raise SystemExit(cli.main(sys.argv[2:]))
"""
    # This synthetic command tests process recovery, not GPU selection. Make its resource
    # choice explicit so it remains independent of the invoking sandbox's device nodes.
    # This fixture reconstructs an existing legacy in-place run. New runs default to
    # isolated input, so explicitly select the old mode rather than rebinding its records.
    args = [str(repo), str(output), "0", '{"device":"cpu"}', "--keep-only", "--in-place",
            "--wall-seconds", "600"]

    def launch_cli():
        return subprocess.Popen([sys.executable, "-c", wrapper, str(runtime), *args],
                                cwd=package_root, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)

    def stage_pids():
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if stage_marker.encode() in (entry / "cmdline").read_bytes():
                    found.append(int(entry.name))
            except OSError:
                continue
        return found

    controller = launch_cli()
    identity = None
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and controller.poll() is None:
            state_path = output / "run_state.json"
            if state_path.is_file():
                try:
                    action = json.loads(state_path.read_text(encoding="utf-8")).get(
                        "current_action") or {}
                except (OSError, ValueError):
                    action = {}
                candidate_identity = action.get("process_identity")
                if action.get("step") == "train" and isinstance(candidate_identity, dict):
                    identity = candidate_identity
                    break
            time.sleep(0.05)
        assert identity, "CLI never persisted the active benchmark process identity"
        assert inspect_process_identity(identity)["status"] == "matching_running"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not stage_pids():
            time.sleep(0.02)
        assert stage_pids(), "benchmark stage child did not start before the kill"
        controller.send_signal(signal.SIGKILL)
        controller.wait(timeout=5)
        # bubblewrap's PID namespace dies with its init, so the simulator subprocess is
        # already gone before a restart reconciles its still-running receipt.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and stage_pids():
            time.sleep(0.02)
        assert stage_pids() == []
        # A host PID may be reused between the kill and this observation. In that case safe
        # recovery blocks instead of signalling the new process. Namespace teardown can
        # briefly leave a host process-group member after the workload marker vanishes; poll
        # the recorded identity to distinguish teardown latency from a real orphan.
        deadline = time.monotonic() + 5
        observed = inspect_process_identity(identity)
        while (time.monotonic() < deadline and
               observed.get("status") == "unverifiable"):
            time.sleep(0.02)
            observed = inspect_process_identity(identity)
        assert observed["status"] in {"not_running", "identity_mismatch"}

        restarted = launch_cli()
        returncode = restarted.wait(timeout=20)
        assert returncode == 1  # an unknown interrupted outcome is not an L2 success
        assert inspect_process_identity(identity)["status"] in {"not_running", "identity_mismatch"}
        assert stage_pids() == []
        receipts = list((research_root / "attempts").glob("*/receipt.json"))
        assert len(receipts) == 1, "reconciliation must not launch a duplicate train attempt"
        receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
        assert receipt["containment_mode"] == "pid_namespace"
        assert receipt["status"] in {"interrupted", "running"}
        if receipt["status"] == "interrupted":
            assert receipt["process_reconciliation"]["status"] == "not_running"
        else:
            assert receipt["status"] == "running"
        controller_session = json.loads(
            (research_root / "controller_session.json").read_text(encoding="utf-8"))
        assert controller_session["status"] == "interrupted"
        assert controller_session["interruption"]["controller_action"] == "baseline"
        assert controller_session["interruption"]["attempt_id"] == receipt["attempt_id"]
        assert controller_session["reconciliation"]["status"] == "reconciled"
        report = json.loads((output / "preparation_derived.json").read_text(encoding="utf-8"))
        assert report["steps"][-1]["step"] == "reconcile_interrupted_action"
        assert report["steps"][-1]["outcome"] in {
            "interrupted; outcome unknown", "blocked"}
        assert report["state"]["research_progress"]["status"] == "interrupted"
        view = json.loads((output / "report" / "view.json").read_text(encoding="utf-8"))
        assert view["status"] != "running"
        assert view["current_action"] == "finished"
        document = (output / "RUN.md").read_text(encoding="utf-8")
        live_strip = document.split("<!-- AUTOSIM_LIVE_START -->", 1)[1].split(
            "<!-- AUTOSIM_LIVE_END -->", 1)[0]
        assert "Status: running" not in live_strip
        assert "Current action: finished" in live_strip
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()
        if identity:
            terminate_recorded_process(identity, grace_seconds=0.2)


def test_a_declaration_on_file_is_picked_up(tmp_path):
    """Otherwise the loop refuses for want of a declaration the run already has, and the
    model -- reasonably -- falls back to a step that redoes work."""
    out = _built(tmp_path)
    scouting = tmp_path / "scouting" / "somewhere"
    scouting.mkdir(parents=True)
    # Wrapped, the way the scout writes it: the file records the attempts alongside what
    # survived them, and the declaration is one key of that.
    (scouting / "declaration.json").write_text(json.dumps({
        "schema_version": 1, "attempts": [],
        "declaration": {
            "benchmark": tmp_path.name,
            "optimization_space": {
                "collection": [{"name": "episodes", "kind": "integer", "description": "n",
                                "low": 1, "high": 9, "default": 1}],
                "training": [{"name": "steps", "kind": "integer", "description": "s",
                              "low": 1, "high": 9, "default": 1}]}}}), encoding="utf-8")
    preparation = Preparation(repo=tmp_path, output=out, client=_Client(),
                              scouting=tmp_path / "scouting")
    assert preparation.declaration.get("benchmark") == tmp_path.name


def test_a_verified_stage_keeps_the_row_it_was_verified_as(tmp_path):
    """A stage is two records of one thing, and only one of them has the `PATH` that made the
    command work.

    `execution.json` says what a reading of the checkout found -- the entry point, the
    invocation. `derived_stages.json` says what the derivation settled in order to get it to
    run: the working directory, the environment, the flags. RoboTwin's evaluation ran with the
    first and died on `ModuleNotFoundError: No module named 'einops'`, because the recorded
    environment put the right interpreter first on `PATH` and the re-read row put nothing
    there. The command was verified and then run as a different command.
    """
    out = _built(tmp_path)
    kept = json.loads((out / "derived_stages.json").read_text(encoding="utf-8"))
    kept["train"]["row"] = {"entrypoint": "train.sh",
                            "environment": {"PATH": "{repo}/../.venv/bin:${PATH}"},
                            "working_directory": "{repo}/scripts"}
    (out / "derived_stages.json").write_text(json.dumps(kept), encoding="utf-8")

    preparation = _prep(tmp_path, out)
    assert preparation.verified["train"]["working_directory"] == "{repo}/scripts"
    assert "PATH" in preparation.verified["train"]["environment"]
    # And what a later reading said about the same stage is kept for the parts the verified
    # row does not override.
    assert preparation.verified["train"]["entrypoint"] == "train.sh"


def test_a_stage_with_no_verified_row_has_nothing_to_overlay(tmp_path):
    out = _built(tmp_path)
    preparation = _prep(tmp_path, out)
    assert preparation.verified == {}
