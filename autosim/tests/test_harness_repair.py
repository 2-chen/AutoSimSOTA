import json
import zipfile
from pathlib import Path

import pytest

from autosim.research import harness_repair as hr, provision as pv
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.execution_paths import ExecutionPaths
from autosim.research.installation_recovery import wheel_import_hints


def sources(root):
    for name in hr.ALLOWED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n")
    return hr.identity(root)


def test_resource_execution_handles_use_run_root_not_checkout(tmp_path):
    repo = tmp_path / "checkout"; repo.mkdir()
    paths = ExecutionPaths(repo, tmp_path)
    row = {"cache_ref": "home/cache/a.body", "suggested_destination": "work/a.whl"}
    enriched = paths.bind_inventory({"digest": "unchanged", "local_wheels": [row]})
    assert enriched["digest"] == "unchanged" and "cache_execution_ref" not in row
    wheel = enriched["local_wheels"][0]
    command = paths.decode("cp " + wheel["cache_execution_ref"] + " " + wheel["destination_execution_ref"])
    assert command == f"cp {tmp_path}/home/cache/a.body {tmp_path}/work/a.whl"
    with pytest.raises(ValueError): paths.bind_run_reference("../outside")
    (tmp_path / "escape").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError): paths.bind_run_reference("escape/file")


def test_wheel_distribution_is_not_import_name(tmp_path):
    path = tmp_path / "artifact.body"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("different_distribution-1.dist-info/top_level.txt", "actual_module\n../escape\n")
    assert wheel_import_hints(path) == ["actual_module"]
    path.write_bytes(b"not a zip")
    assert wheel_import_hints(path) == []


def test_candidate_edits_are_isolated_not_activation(tmp_path):
    base = sources(tmp_path)
    result = hr.apply_edits(tmp_path, [{"path": "research/execution_paths.py",
        "before": "VALUE = 1", "after": "VALUE = 2"}], base)
    assert result["syntax_checked"] and not result["activation_allowed"]
    assert result["validation_status"].startswith("awaiting_independent")
    assert hr.identity(tmp_path) != base


@pytest.mark.parametrize("edit", [
    {"path": "research/agent_budget.py", "before": "VALUE = 1", "after": "VALUE = 2"},
    {"path": "../outside.py", "before": "VALUE = 1", "after": "VALUE = 2"},
    {"path": "research/execution_paths.py", "before": "absent", "after": "VALUE = 2"},
    {"path": "research/execution_paths.py", "before": "VALUE = 1", "after": "if !!!"},
])
def test_candidate_rejects_protected_ambiguous_and_invalid_edits(tmp_path, edit):
    base = sources(tmp_path)
    with pytest.raises((ValueError, SyntaxError)): hr.apply_edits(tmp_path, [edit], base)
    assert hr.identity(tmp_path) == base


def test_failed_evidence_and_parent_budget_worker_required(tmp_path):
    log = tmp_path / "attempt.log"; log.write_text("wrong path\n")
    capture_attempt_evidence(tmp_path, attempt_id="a" * 32, log=log,
        receipt_ref="attempt.json", status="failed", returncode=1, termination_reason="failed")
    with pytest.raises(ValueError, match="isolated repair"):
        hr.propose(tmp_path, object(), evidence_id="a" * 32, diagnosis="path mapping fault")
    log.write_text("changed")
    with pytest.raises(ValueError, match="changed after capture"):
        hr.propose(tmp_path, object(), evidence_id="a" * 32, diagnosis="fault")


def test_readonly_fix_proposes_isolated_candidate_not_live_patch(tmp_path):
    log = tmp_path / "attempt.log"; log.write_text("resource path failure\n")
    capture_attempt_evidence(tmp_path, attempt_id="b" * 32, log=log,
        receipt_ref="attempt.json", status="failed", returncode=1, termination_reason="failed")
    live = Path(hr.__file__).resolve().parents[1]
    before = hr.identity(live)
    class Client:
        def fork_readonly(self, **kwargs):
            assert kwargs["role"] == "fix"
            assert kwargs["workspace"].is_relative_to(tmp_path / "agent_workers")
            return self
        def chat_with_metadata(self, system, user, **kwargs):
            assert kwargs["read_only"] is True
            return json.dumps({"fault_domain": "harness", "reasoning": "mapping defect",
                "expected_postcondition": "same resource reaches native executor",
                "edits": [{"path": "research/execution_paths.py", "before": "import re\n",
                    "after": "import re\n# isolated candidate\n"}]}), {}
    result = hr.propose(tmp_path, Client(), evidence_id="b" * 32, diagnosis="mapping failure")
    assert result["outcome"] == "candidate_only" and not result["activation_allowed"]
    assert hr.identity(live) == before
    assert hr.status(tmp_path)["repairs"][0]["failure_evidence_id"] == "b" * 32
    with pytest.raises(ValueError, match="already has"):
        hr.propose(tmp_path, Client(), evidence_id="b" * 32, diagnosis="same failure")


def test_rejected_probe_replacement_is_not_adopted(tmp_path, monkeypatch):
    class Client:
        output = tmp_path
        def chat_with_metadata(self, *args, **kwargs):
            return json.dumps({"probes": ["true"]}), {}
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    monkeypatch.setattr(pv, "review_probe_replacement", lambda *a, **kw: {"approved": False, "reason": "weaker"})
    transcript = []
    commands, probes = pv.resume(Client(), tmp_path, [], {"command": "false"},
        manifests={}, transcript=transcript, values={}, attempts=1)
    assert commands == [] and probes is None
    assert transcript[-1]["kind"] == "repair_proposal_rejected"
    assert "weaker" in transcript[-1]["reason"]


def test_disposition_controls_replacement_with_independent_review(tmp_path, monkeypatch):
    class Client:
        output = tmp_path
        def chat_with_metadata(self, *args, **kwargs):
            return json.dumps({"commands": ["echo correct"], "repair_mode": "prerequisites",
                "original_operation_disposition": "replace_invalid_operation",
                "install_replacement_evidence": {"failure_evidence_id": "a" * 32}}), {}
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    monkeypatch.setattr(pv, "review_install_replacement", lambda *a, **kw: {"approved": True})
    transcript = []
    commands, probes = pv.resume(Client(), tmp_path, [], {"command": "wrong", "evidence_id": "a" * 32},
        manifests={}, transcript=transcript, values={}, attempts=1)
    assert commands == ["echo correct"] and probes is None
    assert transcript[-1]["repair_mode"] == "replace_operation"
    assert transcript[-1]["replacement_review"]["approved"]


def test_recorder_failure_is_not_presented_as_main_runtime_block(tmp_path):
    from autosim.research import recorder, run_record
    from autosim.research.common import atomic_json
    atomic_json(tmp_path / "agent/runtime_failure.json", {"role": "recorder",
        "message": "writing timeout", "evidence_ref": "evidence/" + "a" * 32 + ".json"})
    view = run_record.build_report_view(tmp_path, "derived", status="running")
    snapshot = recorder.make_snapshot(tmp_path, view, {"actions": [], "plan": {}})
    document = recorder.render(tmp_path, snapshot, {})
    assert "报告写作回合异常（不代表主实验停止）" in document
    assert "### 框架 / 运行故障与恢复" not in document
    assert "writing timeout" in document


def test_scheduler_dispatch_preserves_harness_repair_arguments(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    class Client: pass
    controller = Preparation(repo=tmp_path, output=tmp_path / "out", client=Client(),
        scouting=tmp_path / "scouting")
    called = []
    def step(**kwargs):
        called.append(kwargs)
        return {"outcome": "candidate_only", "activation_allowed": False}
    monkeypatch.setattr(controller, "_step_propose_harness_repair", step)
    result = controller.do("propose_harness_repair", evidence_id="a" * 32, diagnosis="mapping")
    assert result["outcome"] == "candidate_only"
    assert called == [{"evidence_id": "a" * 32, "diagnosis": "mapping"}]


def test_live_strip_uses_preparation_action_start_not_heartbeat_time(tmp_path, monkeypatch):
    from autosim.research import run_record
    from autosim.research.common import atomic_json
    atomic_json(tmp_path / "run_state.json", {"status": "running", "current_action": {
        "step": "retry_failed_action", "at": "2026-10-01T05:00:00+00:00"}})
    monkeypatch.setattr(run_record.time, "time", lambda: 1790832600.0)  # 2026-10-01 05:30 UTC
    path = run_record.refresh_live_status(tmp_path, fallback_status="running",
        fallback_current="retry_failed_action")
    assert "stage elapsed about 1800s" in path.read_text()
