"""Cross-layer regressions from the stopped RoboSyn run, without API/GPU calls."""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from autosim.research.agent_roles import role_profile
from autosim.research.agent_runtime import _critical_stream_line
from autosim.research.common import atomic_json
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.process_executor import run_process_stream
from autosim.research.research_state import ResearchStateError, ResearchStateStore
from autosim.research import provision as pv


def _journal_store(tmp_path):
    store = ResearchStateStore(tmp_path / "run", run_id="run", repository=tmp_path)
    for i in range(130):
        store.record("preparation", "checkpoint", status="running", details={"index": i})
    assert json.loads(store.events_path.read_text())["schema_version"] == 2
    return store


def test_journal_append_does_not_rewrite_base_or_manifest(tmp_path):
    store = _journal_store(tmp_path)
    base = store.root / "run_events.legacy.json"
    before = (base.read_bytes(), base.stat().st_mtime_ns,
              store.events_path.read_bytes(), store.events_path.stat().st_mtime_ns)
    store.record("research", "stage_started", status="running")
    assert before == (base.read_bytes(), base.stat().st_mtime_ns,
                      store.events_path.read_bytes(), store.events_path.stat().st_mtime_ns)
    restored = ResearchStateStore(store.root, run_id="run", repository=tmp_path).load()
    assert restored["event_sequence"] == 131 and restored["phase"] == "research"


def test_journal_cache_only_reads_new_tail(tmp_path, monkeypatch):
    store = _journal_store(tmp_path)
    store.load()
    offsets = []
    original = store._journal_rows
    def observe(offset=0):
        offsets.append(offset)
        return original(offset)
    monkeypatch.setattr(store, "_journal_rows", observe)
    store.record("preparation", "checkpoint", status="running")
    store.record("preparation", "checkpoint", status="running")
    assert all(offset > 0 for offset in offsets)
    assert offsets[-1] > offsets[0]


def test_journal_replays_after_snapshot_failure(tmp_path, monkeypatch):
    store = _journal_store(tmp_path)
    from autosim.research import research_state
    original = research_state.atomic_json
    def fail_snapshot(path, value):
        if path == store.state_path:
            raise OSError("snapshot failure")
        original(path, value)
    monkeypatch.setattr(research_state, "atomic_json", fail_snapshot)
    with pytest.raises(OSError):
        store.record("research", "stage_started", status="running",
                     phase_state={"current_action": {"stage": "train"}})
    monkeypatch.undo()
    state = ResearchStateStore(store.root, run_id="run", repository=tmp_path).load()
    assert state["event_sequence"] == 131
    assert state["current_action"]["stage"] == "train"


def test_journal_tamper_and_torn_append_are_not_adopted(tmp_path):
    store = _journal_store(tmp_path)
    original = store.journal_path.read_bytes()
    store.journal_path.write_bytes(original + b'{"sequence":')
    with pytest.raises(ResearchStateError, match="incomplete append"):
        store.load()
    store.journal_path.write_bytes(original.replace(b'"index":128', b'"index":999'))
    with pytest.raises(ResearchStateError, match="hash is invalid"):
        store.load()


def test_journal_cross_process_style_writers_have_one_chain(tmp_path):
    store = _journal_store(tmp_path)
    def write(i):
        other = ResearchStateStore(store.root, run_id="run", repository=tmp_path)
        other.record("preparation", "checkpoint", status="running", details={"writer": i})
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(write, range(16)))
    assert store.load()["event_sequence"] == 146
    assert len(store._read_events()[0]) == 146


def test_immutable_legacy_base_is_checked_on_cold_load(tmp_path):
    store = _journal_store(tmp_path)
    base = store.root / "run_events.legacy.json"
    base.write_bytes(base.read_bytes().replace(b'"index": 0', b'"index": 9'))
    with pytest.raises(ResearchStateError, match="hash is invalid"):
        ResearchStateStore(store.root, run_id="run", repository=tmp_path).load()


def test_run_document_uses_bounded_verified_journal_tail(tmp_path):
    store = _journal_store(tmp_path)
    from autosim.research.run_record import gather
    gathered = gather(store.root)
    assert gathered["run_events"]["projection_scope"] == "bounded_verified_tail"
    assert gathered["run_events"]["rows"][-1]["sequence"] == 130
    assert not any("共享事件链校验失败" in x for x in gathered["unreadable"])


def test_large_patch_is_referenced_and_replayed_with_identity(tmp_path):
    store = ResearchStateStore(tmp_path / "run", run_id="run", repository=tmp_path)
    value = "x" * 100000
    state = store.record("preparation", "checkpoint", status="running",
                         phase_state={"state": {"large": value}})
    row = json.loads(store.events_path.read_text())["rows"][0]
    assert row["state_patch"] == {} and len(row["state_patch_digest"]) == 64
    assert len(store.events_path.read_bytes()) < 2000
    store.state_path.unlink()
    assert store.load()["state"]["large"] == value
    blob = store.root / "run_state_blobs" / (row["state_patch_digest"] + ".json")
    blob.write_text('{}')
    store.state_path.unlink()
    with pytest.raises(ResearchStateError, match="blob changed"):
        store.load()


def test_subtask_completion_cannot_finish_run_or_clear_owner_action(tmp_path):
    store = ResearchStateStore(tmp_path / "run", run_id="run", repository=tmp_path)
    parent = {"step": "build_the_environment", "status": "running"}
    store.record("preparation", "action_started", status="running",
                 phase_state={"current_action": parent})
    result = store.record("agent_runtime", "agent_finished", status="completed",
                          phase_state={"current_action": {}}, decision_relevant=False)
    assert result["status"] == "running" and result["phase"] == "preparation"
    assert result["current_action"] == parent
    assert result["phases"]["agent_runtime"]["status"] == "completed"


def test_old_runtime_owned_snapshot_is_normalized_on_load(tmp_path):
    store = ResearchStateStore(tmp_path / "run", run_id="run", repository=tmp_path)
    store.record("preparation", "checkpoint", status="running")
    state = store.record("agent_runtime", "agent_finished", status="completed")
    state.update(phase="agent_runtime", status="completed")
    atomic_json(store.state_path, state)
    assert store.load()["status"] == "running"


def test_readonly_fix_has_only_bounded_native_inspection(tmp_path):
    profile = role_profile("fix", read_only=True)
    assert "mcp__autosim_exec__inspect_native_environment" in profile.mcp_tools
    assert "mcp__autosim_exec__run_command" not in profile.mcp_tools
    assert "Edit" not in profile.builtin_tools and not profile.can_execute_diagnostics
    from autosim.research.agent_runtime import _claude_configuration
    config = _claude_configuration(output=tmp_path, workspace=tmp_path,
        python=sys.executable, include_executor=False, native_readonly=profile.can_inspect_native)
    args = config["mcpServers"]["autosim_exec"]["args"]
    assert "--read-only" in args and "--native-readonly" in args


def test_stream_filter_retains_unknown_errors_and_results():
    assert not _critical_stream_line('{"type":"system","subtype":"thinking_tokens"}')
    for line in ['not json', '{"type":"result"}', '{"type":"system","subtype":"error"}',
                 '{"type":"assistant"}', '{"type":"user"}', '{"type":"stream_event"}']:
        assert _critical_stream_line(line)


def test_telemetry_flood_does_not_kill_slow_consumer(tmp_path):
    received = []
    import time
    def consume(line):
        time.sleep(.01)
        received.append(json.loads(line))
    code = ('import sys,json\n'
            'for i in range(20041): print(json.dumps({"type":"system","subtype":"thinking_tokens","estimated_tokens":i}))\n'
            'print(json.dumps({"type":"assistant","message":"tool receipt"}))\n'
            'print(json.dumps({"type":"result","result":"complete"}))\n')
    result = run_process_stream([sys.executable, "-c", code], cwd=tmp_path,
        env=os.environ.copy(), timeout=5, decouple_callbacks=True,
        on_stdout_line=consume, stdout_line_filter=_critical_stream_line)
    assert result.returncode == 0 and result.error is None and not result.timed_out
    assert [x["type"] for x in received] == ["assistant", "result"]
    assert len(result.stdout.splitlines()) == 20043  # Raw evidence was not discarded.


def test_nonstream_capture_without_callback_does_not_fill_callback_queue(tmp_path):
    code = "import json; print(json.dumps({'type':'result','rows':list(range(25000))},indent=2))"
    result = run_process_stream([sys.executable, "-c", code], cwd=tmp_path,
        env=os.environ.copy(), timeout=5, decouple_callbacks=True)
    assert result.returncode == 0 and result.error is None
    assert len(json.loads(result.stdout)["rows"]) == 25000


@pytest.mark.parametrize("identity", ["4c2d4a90e97d4f049a60918f4051dab8",
    "4c549f81c2134554821d81a668dfbfc9", "2d587093cfaf4d648047532b472fd585"])
def test_real_failure_message_mix_replay(tmp_path, identity):
    root = Path(__file__).resolve().parents[2]
    log = root / "autoresearch_runs/robosyn_installrecovery_20261001_v1/agent/stream_failures" / (identity + ".log")
    if not log.exists():
        pytest.skip("optional local sealed failure output")
    sealed = json.loads(log.read_text())
    lines = sealed["stdout"].splitlines()
    # Sealed records are sanitized, not original protocol bytes. Restore only known
    # telemetry to valid synthetic records; never repair tool/control/error records.
    normalized = [json.dumps({"type": "system", "subtype": "thinking_tokens"})
                  if re.search(r'"subtype"\s*:\s*"thinking_tokens"', line) else line
                  for line in lines]
    expected = [line for line in normalized if _critical_stream_line(line)]
    received = []
    import time
    def consume(line):
        time.sleep(.003)
        received.append(line)
    result = run_process_stream([sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        cwd=tmp_path, env=os.environ.copy(), timeout=10,
        input_bytes=("\n".join(normalized) + "\n").encode(), decouple_callbacks=True,
        on_stdout_line=consume, stdout_line_filter=_critical_stream_line)
    assert result.returncode == 0 and result.error is None
    assert received == expected
    assert len(expected) < 100


def _sealed_reviewer(tmp_path, *, bad=False, legacy=False, canonical_file=False, source_ref=None):
    output = tmp_path / "run"
    repo = output / "checkout"
    repo.mkdir(parents=True)
    (repo / "pyproject.toml").write_text('name = "real-package"\n')
    log = output / "provision_attempts" / ("a" * 32 + ".log")
    log.parent.mkdir()
    log.write_text('isolated build failed for patchelf\n')
    capture_attempt_evidence(output, attempt_id="a" * 32, log=log,
        receipt_ref="provision_attempts/failure.json", status="failed", returncode=1,
        termination_reason="failed")
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            bundle = json.loads(user)["citation_bundle"]
            assert bundle["evidence:" + "a" * 32]["kind"] == "native_failure"
            assert "Installer commands are implementation choices" in system
            return json.dumps({"approved": True, "reason": "same required capability",
                "citations": [{"ref": "source:pyproject.toml", "quote": 'name = "real-package"'},
                    {("file" if legacy or canonical_file else "ref"): (str(log.relative_to(output)) if legacy
                        else "evidence:" + "a" * 32),
                     "quote": "fabricated" if bad else "isolated build failed for patchelf"}]}), {}
    client = Client()
    client.output = output
    claim = {"same_capability": "same package source", "source_refs": [source_ref or "pyproject.toml"],
             "failure_evidence_id": "a" * 32}
    return pv.review_install_replacement(client, repo, {"evidence_id": "a" * 32},
                                         ["download exact source artifact"], claim)


@pytest.mark.parametrize("legacy", [False, True])
def test_verified_failure_log_is_accepted_as_approval_evidence(tmp_path, legacy):
    result = _sealed_reviewer(tmp_path, legacy=legacy)
    assert result["approved"] and result["model_approved"]
    assert not result["validation_errors"]


def test_canonical_bundle_key_in_legacy_file_field_still_requires_exact_quote(tmp_path):
    result = _sealed_reviewer(tmp_path, canonical_file=True)
    assert result['approved'] and not result['validation_errors']
    result = _sealed_reviewer(tmp_path/'bad', canonical_file=True, bad=True)
    assert not result['approved'] and 'does not match' in result['reason']


@pytest.mark.parametrize('field', ['path','file','source','ref'])
def test_source_reference_spellings_have_same_bounded_checkout_authority(tmp_path, field):
    assert _sealed_reviewer(tmp_path, source_ref={field:'source:pyproject.toml'})['approved']


def test_conflicting_or_escaping_source_reference_never_becomes_approval(tmp_path):
    result = _sealed_reviewer(tmp_path, source_ref={'path':'pyproject.toml','source':'other.py'})
    assert not result['approved'] and 'source_refs[0]' in result['reason']
    result = _sealed_reviewer(tmp_path/'escape', source_ref={'source':'source:../outside.py'})
    assert not result['approved'] and 'unsafe' in result['reason']


def test_quote_mismatch_returns_specific_validation_error_not_model_reason(tmp_path):
    result = _sealed_reviewer(tmp_path, bad=True)
    assert not result["approved"] and result["model_approved"]
    assert "citation[1]" in result["reason"] and "does not match" in result["reason"]
    assert result["model_reason"] == "same required capability"
    from autosim.research.review_transaction import review_view
    view = review_view(tmp_path / "run")[0]
    assert view["status"] == "validation_rejected"
    assert view["model_response_status"] == "completed"
    assert view["validation_errors"] == result["validation_errors"]


def test_real_cpu_failure_reviewed_route_replacement_and_native_postcondition(tmp_path):
    """Real executor/receipt/review-validation chain; the model alone is a fixture.

    This is not a benchmark result or a claim that a real LLM completed recovery.
    """
    output = tmp_path / "run"
    repo = output / "checkout"
    repo.mkdir(parents=True)
    source = "import argparse\np=argparse.ArgumentParser()\np.add_argument('--correct', action='store_true')\na=p.parse_args()\nassert a.correct\nprint('CAPABILITY_READY')\n"
    (repo / "native_capability.py").write_text(source)
    original = "{python} native_capability.py --incorrect"
    corrected = "{python} native_capability.py --correct"
    failure = pv.run(original.format(python=sys.executable), env=os.environ.copy(),
                     cwd=repo, timeout=15, output=output / "install.log")
    assert failure["ok"] is False and failure["returncode"] == 2
    failure["template"] = original
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            payload = json.loads(user)
            if "citation_bundle" in payload:
                assert payload["required_existing_probes"] == [corrected]
                assert "evidence:" + failure["evidence_id"] in payload["citation_bundle"]
                return json.dumps({"approved": True, "reason": "same native capability, corrected CLI",
                    "citations": [{"ref": "source:native_capability.py", "quote": "p.add_argument('--correct', action='store_true')"}]}), {}
            return json.dumps({"commands": [corrected], "repair_mode": "replace_operation",
                "install_replacement_evidence": {"same_capability": "same native CLI capability",
                    "failure_evidence_id": failure["evidence_id"], "source_refs": ["native_capability.py"]}}), {}
    client = Client()
    client.output = output
    transcript = []
    commands, probes = pv.resume(client, repo, [], failure, manifests={},
        transcript=transcript, values={"python": sys.executable}, current_probes=[corrected],
        pending_commands=[original])
    assert commands == [corrected] and probes is None
    assert transcript[-1]["replacement_review"]["approved"]
    assert transcript[-1]["original_operation_disposition"] == "replace_invalid_operation"
    success = pv.run(commands[0].format(python=sys.executable), env=os.environ.copy(),
        cwd=repo, timeout=15, output=output / "install.log")
    assert success["ok"] and "CAPABILITY_READY" in success["excerpt"]
    from autosim.research.evidence_store import read_attempt_evidence
    assert read_attempt_evidence(output, success["evidence_id"])["returncode"] == 0
