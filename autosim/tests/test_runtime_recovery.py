from __future__ import annotations

import json
from pathlib import Path
from contextlib import contextmanager

import pytest

from autosim.research import runtime_recovery as recovery
from autosim.research.agent_runtime import AgentWorkspaceBlocked, _validate_agent_workspace
from autosim.research.agent_client import AgentRuntimeClientError, RoleAwareAgentClient
from autosim.research.common import atomic_json
from autosim.research.evidence_store import read_attempt_evidence


def workspace(tmp_path):
    output = tmp_path / "run"
    source = output / "checkout"
    source.mkdir(parents=True)
    (source / "README.md").write_text("original source")
    atomic_json(output / "workspace_snapshot.json", {"copy_mode": "tracked_worktree",
        "destination": str(source), "source_entries": [["file", "README.md", 15]]})
    return output, source


def venv(source, name=".fixprobe_venv"):
    root = source / name
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (root / "bin/python3").symlink_to("/usr/bin/python3")
    (root / "bin/python").symlink_to("python3")
    return root


def fault(output, source):
    try:
        _validate_agent_workspace(source)
    except AgentWorkspaceBlocked as exc:
        return recovery.seal_failure(output, source, exc, role="scheduler",
                                     decision_attempt_id="a" * 32)
    raise AssertionError("expected blocked workspace")


class RecoveryClient:
    def __init__(self, output, source, *, reject=False):
        self.output, self.workspace = output, source
        self.reject = reject
        self.calls = []

    def fork_readonly(self, *, output, workspace, role):
        assert role == "fix"
        assert workspace != self.workspace
        assert not list(workspace.iterdir())
        assert json.loads((output / "worker_parent.json").read_text())["output"] == str(self.output)
        _validate_agent_workspace(workspace)
        return self

    def chat_with_metadata(self, system, user, **kwargs):
        packet = json.loads(user)
        self.calls.append(packet)
        assert kwargs["read_only"] and not kwargs["include_research_context"]
        assert read_attempt_evidence(self.output, packet["evidence_id"])["text"]
        ids = [row["id"] for row in packet["approved_candidates"]]
        return json.dumps({"quarantine_ids": ["invalid"] if self.reject else ids,
                           "reason": "sealed guard evidence shows a generated diagnostic environment"}), {}


def failure_row(record):
    return {"failure_category": "workspace_guard", "failure_fingerprint": record["fingerprint"],
            "evidence_id": record["evidence_id"]}


def test_guard_seals_real_prelaunch_evidence_with_specific_paths(tmp_path):
    output, source = workspace(tmp_path)
    venv(source)
    record = fault(output, source)
    assert record["category"] == "workspace_guard" and not record["launched"]
    assert record["entries"][0]["path"].startswith(".fixprobe_venv/")
    detail = read_attempt_evidence(output, record["evidence_id"])
    assert "AgentWorkspaceBlocked" in detail["text"]
    assert detail["termination_reason"] == "workspace_guard"


def test_recovery_uses_immutable_id_even_if_other_role_overwrites_latest_fault(tmp_path):
    output, source = workspace(tmp_path)
    venv(source)
    record = fault(output, source)
    recovery.seal_failure(output, source, RuntimeError("concurrent recorder failure"),
                          role="recorder", decision_attempt_id=None)
    result = recovery.recover(RecoveryClient(output, source), failure_row(record), timeout=10)
    assert result["status"] == "revalidated"


def test_public_certificate_bundle_is_not_misclassified_as_a_private_key(tmp_path):
    import ssl
    import shutil
    from autosim.research.agent_runtime import _public_certificate_bundle
    bundle = ssl.get_default_verify_paths().cafile
    if not bundle:
        pytest.skip("local public CA bundle unavailable")
    output, source = workspace(tmp_path)
    certificate = source / "public-certificates.pem"
    shutil.copyfile(bundle, certificate)
    assert _public_certificate_bundle(certificate)
    _validate_agent_workspace(source)
    certificate.write_text(certificate.read_text() + "\n-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----\n")
    assert not _public_certificate_bundle(certificate)
    with pytest.raises(AgentWorkspaceBlocked):
        _validate_agent_workspace(source)


def test_invalid_certificate_envelope_and_public_symlink_remain_blocked(tmp_path):
    from autosim.research.agent_runtime import _public_certificate_bundle
    output, source = workspace(tmp_path)
    certificate = source / "bad.pem"
    certificate.write_text("-----BEGIN CERTIFICATE-----\nc2VjcmV0\n-----END CERTIFICATE-----\n")
    assert not _public_certificate_bundle(certificate)
    with pytest.raises(AgentWorkspaceBlocked):
        _validate_agent_workspace(source)


def test_recovery_agent_selects_generated_venv_then_original_guard_passes(tmp_path):
    output, source = workspace(tmp_path)
    generated = venv(source)
    record = fault(output, source)
    result = recovery.recover(RecoveryClient(output, source), failure_row(record), timeout=10)
    assert result["status"] == "revalidated" and not generated.exists()
    destination = output / result["moved"][0]["to"]
    assert (destination / "pyvenv.cfg").is_file()
    assert (source / "README.md").read_text() == "original source"
    _validate_agent_workspace(source)


@pytest.mark.parametrize("unsafe", ["credential", "asset_link", "original", "missing_inventory"])
def test_recovery_never_moves_credentials_resources_or_original_source(tmp_path, unsafe):
    output, source = workspace(tmp_path)
    root = venv(source)
    if unsafe == "credential":
        (root / ".env").write_text("secret-content-must-not-be-read")
    elif unsafe == "asset_link":
        (root / "asset").symlink_to("/usr/bin/python3")
    elif unsafe == "original":
        document = json.loads((output / "workspace_snapshot.json").read_text())
        document["source_entries"].append(["directory", ".fixprobe_venv", 0])
        atomic_json(output / "workspace_snapshot.json", document)
    else:
        atomic_json(output / "workspace_snapshot.json", {})
    record = fault(output, source)
    client = RecoveryClient(output, source)
    result = recovery.recover(client, failure_row(record), timeout=10)
    assert result["status"] == "blocked" and not result["moved"]
    assert root.exists() and not client.calls[0]["approved_candidates"]
    assert "secret-content-must-not-be-read" not in json.dumps(record)


def test_unapproved_recovery_id_cannot_mutate_checkout(tmp_path):
    output, source = workspace(tmp_path)
    root = venv(source)
    record = fault(output, source)
    with pytest.raises(ValueError, match="contract"):
        recovery.recover(RecoveryClient(output, source, reject=True), failure_row(record), timeout=10)
    assert root.exists()


def test_recovery_refuses_active_native_job(tmp_path, monkeypatch):
    output, source = workspace(tmp_path)
    root = venv(source)
    record = fault(output, source)
    monkeypatch.setattr("autosim.research.native_jobs.active_jobs", lambda _: ["busy"])
    client = RecoveryClient(output, source)
    result = recovery.recover(client, failure_row(record), timeout=10)
    assert result["status"] == "blocked" and not client.calls and root.exists()


def test_fault_fingerprint_changes_when_relevant_link_identity_changes(tmp_path):
    output, source = workspace(tmp_path)
    root = venv(source)
    first = fault(output, source)
    (root / "bin/python3").unlink()
    (root / "bin/python3").symlink_to("/usr/bin/python3")
    second = fault(output, source)
    assert first["fingerprint"] != second["fingerprint"]


def test_agent_client_does_not_drop_startup_error_or_invent_event_reference(tmp_path, monkeypatch):
    output, source = workspace(tmp_path)
    venv(source)
    client = RoleAwareAgentClient(workspace=source, output=output, run_id="test",
                                 turn_budget_usd=1, total_budget_usd=5)
    monkeypatch.setattr("autosim.research.agent_runtime.run_coding_agent",
                        lambda **kwargs: _validate_agent_workspace(kwargs["workspace"]))
    with pytest.raises(AgentRuntimeClientError) as caught:
        client.chat_with_metadata("system", "request", decision_attempt_id="b" * 32)
    error = caught.value
    assert error.failure_category == "workspace_guard"
    assert "external-link" in str(error)
    assert error.runtime_failure["evidence_ref"].startswith("evidence/")


def test_repeated_sealed_startup_failure_stops_after_one_recovery_assessment(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    from autosim.research.common import object_digest
    output, source = workspace(tmp_path)
    class Client:
        @contextmanager
        def as_role(self, role):
            yield
    controller = Preparation(repo=source, output=output, client=Client(), scouting=tmp_path / "scout")
    calls = []
    def choose():
        calls.append("choose")
        raise AgentRuntimeClientError("permanent startup refusal", failure_category="fixture",
            decision_attempt_id="c" * 32, runtime_failure={"fingerprint": object_digest("same"),
                "message": "permanent refusal", "evidence_ref": "evidence/" + "d" * 32 + ".json"})
    monkeypatch.setattr(controller, "choose", choose)
    monkeypatch.setattr(controller, "_monitor_failed_action", lambda **_: {
        "status": "assessed", "progress": "uncertain", "summary": "try once", "guidance": "", "evidence_refs": []})
    result = controller.run(max_steps=8, max_relaunch=4)
    assert calls == ["choose", "choose"]
    assert result["status"] == "infrastructure_blocked"
    assert result["supervision"]["automatic_relaunches"] == 0


def test_recorder_accepts_cited_versions_but_rejects_invented_numbers(tmp_path):
    from autosim.research import recorder, run_record
    held = recorder.make_snapshot(tmp_path, run_record.build_report_view(tmp_path, "derived", status="paused"),
        {"actions": [{"step": "probe", "because": "torch 2.7.1 targets sm_120"}], "plan": {}})
    answer = {"event_revision": held["event_revision"], "sections": {
        key: {"text": "版本 2.7.1 支持 sm_120，仍需原生探针。", "evidence_ids": ["action-0"]}
        for key in recorder.FIELDS}, "charts": [], "demo_requests": []}
    assert recorder._validate(answer, held)
    answer["sections"]["finding"]["text"] = "成功率是99%。"
    with pytest.raises(ValueError, match="numbers absent"):
        recorder._validate(answer, held)


def test_report_displays_sealed_error_not_just_pause_limit(tmp_path):
    from autosim.research import recorder, run_record
    atomic_json(tmp_path / "agent/runtime_failure.json", {"message": "external-link guard refusal",
        "evidence_ref": "evidence/" + "e" * 32 + ".json"})
    atomic_json(tmp_path / "runtime_blocker.json", {"recovery": {"status": "blocked", "reason": "same fault"}})
    held = recorder.make_snapshot(tmp_path, run_record.build_report_view(tmp_path, "derived", status="paused"),
                                   {"actions": [], "plan": {}})
    text = recorder.render(tmp_path, held, {})
    assert "启动故障与恢复" in text and "external-link guard refusal" in text
    assert "不等同于总预算耗尽" in text and "完整错误证据" in text


def test_failed_environment_preserves_original_error_not_only_round_limit(tmp_path, monkeypatch):
    from autosim.research import provision
    output, source = workspace(tmp_path)
    monkeypatch.setattr(provision, "content_id", lambda *args: "fixture")
    failure = {"command": "native-config-probe", "ok": False, "returncode": 1,
               "excerpt": "EOFError: EOF when reading a line", "evidence_id": "f" * 32}
    result = provision._finish(output, source, "3.10", [], [failure], ["native-config-probe"],
        {"passed": False, "reason": "rounds exhausted"}, interpreter=output / "env/bin/python")
    assert result["latest_failure"] == failure
    assert json.loads((output / "environment.json").read_text())["latest_failure"] == failure
    assert "noninteractive" in provision.PLAN_SYSTEM and "replay" in provision.PLAN_SYSTEM


def test_persistent_diagnostic_scratch_never_contaminates_source(tmp_path):
    import shutil
    from autosim.research.agent_runtime import execute_agent_command
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap required")
    output, source = workspace(tmp_path)
    result = execute_agent_command(arguments={"argv": ["python3", "-c",
        "import pathlib,os;p=pathlib.Path('/tmp/diagnostics/probe');p.mkdir();"
        "os.symlink('/usr/bin/python3',p/'python')"], "timeout_seconds": 10},
        workspace=source, output=output)
    assert result["status"] == "completed", result
    next_result = execute_agent_command(arguments={"argv": ["/tmp/diagnostics/probe/python", "-c",
        "print('persistent-scratch-ok')"], "timeout_seconds": 10}, workspace=source, output=output)
    assert next_result["status"] == "completed" and "persistent-scratch-ok" in next_result["stdout"]
    _validate_agent_workspace(source)
    assert list(source.iterdir()) == [source / "README.md"]


def test_config_eof_evidence_reaches_fix_and_original_operation_is_replayed(tmp_path):
    import os
    from autosim.research import provision
    output, source = workspace(tmp_path)
    work = output / "work"
    work.mkdir()
    script = source / "native_probe.py"
    script.write_text("import os,pathlib\np=pathlib.Path(os.environ['TASK_CONFIG'])\n"
                      "if not p.exists(): input('configuration path? ')\n"
                      "assert p.read_text() == 'configured'\nprint('original-native-probe-ok')\n")
    values = {"python": "/usr/bin/python3", "repo": str(source), "workdir": str(work)}
    environment = {**os.environ, "TASK_CONFIG": str(work / "task.cfg")}
    original = "{python} {repo}/native_probe.py"
    first = provision.probe_environment(original, values=values, env=environment, cwd=source,
        output=output / "first_probe.log", timeout=10)
    assert not first["ok"] and "EOFError" in first["excerpt"]
    class Fix:
        def chat_with_metadata(self, system, user, **kwargs):
            packet = json.loads(user)
            sealed = read_attempt_evidence(output, packet["failure"]["evidence_id"])
            assert "EOFError" in sealed["text"]
            assert "run-local" in system and "failed probe" in system
            return json.dumps({"commands": [
                "{python} -c \"import pathlib;pathlib.Path('{workdir}/task.cfg').write_text('configured')\""],
                "reasoning": "prepare the native config at the exact consumer path"}), {}
    pending, replacement = provision.resume(Fix(), source, [], first, manifests={},
        transcript=[], values=values, probing=True)
    assert len(pending) == 1 and not replacement
    setup = provision.run(provision.substitute(pending[0], values), env=environment, cwd=source,
                          output=output / "build.log", timeout=10)
    assert setup["ok"]
    second = provision.probe_environment(original, values=values, env=environment, cwd=source,
        output=output / "second_probe.log", timeout=10)
    assert second["ok"] and "original-native-probe-ok" in second["excerpt"]
    assert first["evidence_id"] != second["evidence_id"]
