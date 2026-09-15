"""Real process/kernel checks and deterministic API fixtures; no GPU or API calls."""
import json
import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

from autosim.research import probe_barrier_contract
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.incidents import IncidentStore, build_incident, evidence_view, redact_text
from autosim.research.repair_agent import RepairAgent
from autosim.research.repair_tools import RepairTools, membership_validation_spec, verify_activation_request
from autosim.research.repair_validation import (ValidationSpec, check_pure_component, isolation_capability,
                                               run_contract, validate_candidate)


OLD = "def ready_members_match(expected: list[dict], ready: list[dict], generation: str) -> bool:\n    return not (len(ready) < len(expected))\n"
NEW = Path(probe_barrier_contract.__file__).read_text()


def make_incident(**kwargs):
    return build_incident(run_id="cpu_contract_fixture", stage="native_probe", purpose="smoke", worker_roots={}, **kwargs)


def make_broker(tmp_path, **kwargs):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "component.py").write_text(OLD)
    (root / ".env").write_text("API_KEY=private-controller-value")
    spec = membership_validation_spec("component.py")
    return RepairTools(tmp_path / "tools", source_root=root, allowed_code_paths=["component.py"],
        incident=make_incident(), validators={spec.name: spec}, deadline_epoch=time.time() + 1200, **kwargs)


@pytest.fixture
def kernel_isolation():
    result = isolation_capability()
    if not result["l2_available"]:
        if os.environ.get("AUTOSIM_REQUIRE_REPAIR_SANDBOX") == "1":
            pytest.fail(f"required kernel isolation unavailable: {result}")
        pytest.skip("kernel isolation blocked by outer execution sandbox; rerun approved kernel test")
    return result


def test_first_native_failure_survives_peer_barrier_and_secrets(tmp_path):
    native, peer = tmp_path / "native", tmp_path / "peer"
    atomic_json(native / "process/process.json", {"returncode": -11, "status": "failed", "finished_at": "2026-09-15T01:00:00Z"})
    atomic_json(native / "startup.json", {"phase": "environment_constructing", "policy_has_acted": False})
    (native / "process/stdout.log").write_text('Fatal Python error: Segmentation fault\n  File "/sim.py", line 1591\n    env.load_urdf(path)\nAPI_KEY=keep-this-secret\n')
    atomic_json(peer / "process/process.json", {"returncode": -11, "status": "failed", "finished_at": "2026-09-15T01:04:00Z"})
    atomic_json(peer / "worker_failure.json", {"error": "TimeoutError: native workers did not reach the ready barrier"})
    incident = build_incident(run_id="cpu", stage="device_probe_finished", purpose="smoke",
                              worker_roots={"native_0": native, "native_1": peer})
    assert incident["failure_kind"] == "native_startup_abort"
    assert incident["primary_worker_id"] == "native_0"
    assert incident["workers"][1]["causal_role"] == "secondary"
    view = evidence_view(incident)
    serialized = json.dumps(view)
    assert "load_urdf" in serialized and "keep-this-secret" not in serialized
    assert str(tmp_path) not in serialized
    store = IncidentStore(tmp_path / "incidents")
    assert store.record(incident) == store.record(incident)


def test_incident_snapshot_keeps_api_identity_while_budget_only_tightens(tmp_path):
    first = make_incident(exception=RuntimeError("constructor failed"), budget={"deadline_epoch": 1000, "remaining_wall_seconds": 100})
    later = make_incident(exception=RuntimeError("constructor failed"), budget={"deadline_epoch": 900, "remaining_wall_seconds": 40})
    assert first["incident_id"] == later["incident_id"]
    store = IncidentStore(tmp_path)
    snapshot = store.snapshot(first)
    path = tmp_path / first["incident_id"]
    pinned = digest(path / "incident.json")
    assert store.snapshot(later) == snapshot
    looser = make_incident(exception=RuntimeError("constructor failed"), budget={"deadline_epoch": 2000})
    assert store.snapshot(looser) == snapshot
    assert digest(path / "incident.json") == pinned
    assert read_json(path / "budget_state.json")["deadline_epoch"] == 900
    assert len(list((path / "observations").glob("*.json"))) == 3
    altered = {**later, "supervisor_error": "another error under a forged same ID"}
    with pytest.raises(ValueError, match="incident changed"):
        store.snapshot(altered)


def test_distinct_unclassified_supervisor_errors_have_distinct_identities():
    first = make_incident(exception=RuntimeError("unclassified alpha condition"))
    second = make_incident(exception=RuntimeError("unclassified beta condition"))
    assert first["failure_kind"] == second["failure_kind"] == "unclassified"
    assert first["incident_id"] != second["incident_id"]


def test_supervisor_rebuild_returns_pinned_snapshot_for_api_resume(tmp_path):
    from autosim.research.infrastructure_recovery import supervisor_incident
    kwargs = dict(run_id="cpu", stage="device_probe", exception=RuntimeError("constructor failed"),
                  source_revision="source", environment_fingerprint="environment", final_opened=False)
    first = supervisor_incident(tmp_path, deadline_epoch=time.time()+600, **kwargs)
    later = supervisor_incident(tmp_path, deadline_epoch=time.time()+500, **kwargs)
    assert first == later


def test_final_opened_seals_error_even_if_caller_uses_development_purpose():
    incident = build_incident(run_id="cpu", stage="export", purpose="development", worker_roots={},
                              final_opened=True, exception=RuntimeError("sealed-score-0.875"))
    assert "sealed-score" not in json.dumps(evidence_view(incident))


@pytest.mark.parametrize("marker", ["reset_started.json", "initializations.jsonl", "episodes.jsonl"])
def test_reset_markers_prevent_pre_reset_recovery(tmp_path, marker):
    atomic_json(tmp_path / "process/process.json", {"returncode": -11, "status": "failed"})
    (tmp_path / marker).write_text("{}\n")
    incident = build_incident(run_id="cpu", stage="eval", purpose="development", worker_roots={"worker": tmp_path})
    assert incident["workers"][0]["episode_started"]
    assert incident["failure_kind"] == "native_runtime_abort"


def test_sealed_evidence_never_exports_scores_or_traceback(tmp_path):
    atomic_json(tmp_path / "process/process.json", {"returncode": -11, "status": "failed"})
    atomic_json(tmp_path / "worker_failure.json", {"error": "secret-final-score=0.875"})
    (tmp_path / "process/stdout.log").write_text("Error: secret-final-score=0.875\n")
    incident = build_incident(run_id="cpu", stage="eval", purpose="final_confirmation", worker_roots={"worker": tmp_path})
    assert "secret-final-score" not in json.dumps(evidence_view(incident))
    assert "error_excerpt" not in evidence_view(incident)["workers"][0]


def test_worker_artifact_symlink_cannot_turn_evidence_into_private_file_read(tmp_path):
    worker, private = tmp_path / "worker", tmp_path / "private"
    worker.mkdir(); private.mkdir()
    atomic_json(private / "process.json", {"returncode": -11, "status": "failed"})
    (private / "stdout.log").write_text("Error: must-not-export-private-content\n")
    (worker / "process").symlink_to(private, target_is_directory=True)
    incident = build_incident(run_id="cpu", stage="probe", purpose="smoke", worker_roots={"worker": worker})
    assert "must-not-export" not in json.dumps(evidence_view(incident))
    assert incident["artifact_references"] == {}


@pytest.mark.parametrize("text", ['API_KEY=secret-value', '"api_key": "secret-value"',
                                  "Authorization: Bearer secret-value", "https://user:secret-value@example.com"])
def test_log_redaction(text):
    assert "secret-value" not in redact_text(text)


@pytest.mark.parametrize("path", [".env", "../repo/component.py", "/etc/passwd", "selection/results.json", "other.py"])
def test_broker_file_scope_denies_credentials_and_unregistered_files(tmp_path, path):
    broker = make_broker(tmp_path)
    result = broker.execute("read_code", {"path": path})
    assert result["status"] == "failed"
    assert "private-controller-value" not in json.dumps(result)


def test_isolation_unavailable_does_not_call_api(tmp_path, monkeypatch):
    broker = make_broker(tmp_path)
    monkeypatch.setattr("autosim.research.repair_agent.isolation_capability", lambda: {"l2_available": False})
    client = FakeClient()
    result = RepairAgent(tmp_path / "agent", client, tools=broker).run(broker.incident)
    assert result["status"] == "unavailable" and client.calls == 0


def test_real_kernel_rejects_files_network_processes_and_excludes_secrets(tmp_path, kernel_isolation):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / ".env").write_text("secret-value")
    outside = tmp_path / "private.env"
    outside.write_text("secret-value")
    code = f'''import os
import socket
from pathlib import Path
def check():
    rejected = []
    for operation in [lambda: Path({str(outside)!r}).read_text(),
                      lambda: Path(__file__).with_name('.env').read_text(),
                      lambda: socket.socket(), lambda: os.fork()]:
        try:
            operation()
        except OSError:
            rejected.append(True)
        else:
            rejected.append(False)
    return rejected + ['API_KEY' not in os.environ]
'''
    (root / "component.py").write_text(code)
    result = run_contract(root, ValidationSpec("isolation", "component.py", "check",
                                               ({"args": [], "expected": [True] * 5},)))
    assert result["passed"], result


def test_trusted_differential_gate_rejects_fake_success_and_requires_reproduction(tmp_path, kernel_isolation):
    original, repaired = tmp_path / "base", tmp_path / "candidate"
    original.mkdir(); repaired.mkdir()
    (original / "component.py").write_text(OLD)
    (repaired / "component.py").write_text(NEW)
    spec = membership_validation_spec("component.py")
    result = validate_candidate(original, repaired, (spec,))
    assert result["passed"] and result["baseline_reproduced"]
    (original / "component.py").write_text(NEW)
    assert not validate_candidate(original, repaired, (spec,))["passed"]
    (repaired / "component.py").write_text("import os\nos._exit(0)\n")
    with pytest.raises(ValueError, match="registered function"):
        run_contract(repaired, spec)
    untrusted = ValidationSpec(spec.name, spec.target, spec.function, spec.cases)
    assert not run_contract(repaired, untrusted)["passed"]


@pytest.mark.parametrize("body", [
    "    import os\n    return os.system('true')\n",
    "    return expected.__class__.__mro__\n",
    "    len = ready_members_match\n    return len(expected, ready, generation)\n",
    "    while True:\n        pass\n",
])
def test_live_activation_rejects_python_escape_even_if_test_sandbox_would_contain_it(body):
    source = "def ready_members_match(expected, ready, generation):\n" + body
    with pytest.raises(ValueError):
        check_pure_component(source)


def test_action_idempotency_verified_activation_and_tamper_rejection(tmp_path, kernel_isolation):
    broker = make_broker(tmp_path)
    arguments = {"hypothesis": "Validate current-generation live membership rather than only file count.",
                 "edits": [{"path": "component.py", "old": OLD, "new": NEW}]}
    first = broker.execute("apply_patch_candidate", arguments)
    assert first["status"] == "candidate_created", first
    assert broker.execute("apply_patch_candidate", arguments) == first
    assert len(read_json(broker.root / "session.json")["candidates"]) == 1
    candidate_id = first["candidate_id"]
    validation = broker.execute("run_validation", {"candidate_id": candidate_id})
    assert validation["status"] == "validated", validation
    request = broker.execute("request_activation", {"candidate_id": candidate_id})
    assert verify_activation_request(request)["candidate_id"] == candidate_id
    candidate = Path(request["candidate_root"]) / "component.py"
    candidate.write_text(OLD)
    with pytest.raises(ValueError, match="candidate"):
        verify_activation_request(request)
    assert (broker.source_root / "component.py").read_text() == OLD


def test_prepare_only_cannot_start_or_renew_real_coding_session_budget(tmp_path, monkeypatch):
    module_path = Path(__file__).resolve().parents[2] / "tools/validate_harness_repair_api.py"
    spec = importlib.util.spec_from_file_location("harness_repair_validation_fixture", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    historical = tmp_path / "archived.py"
    historical.write_text('def historical():\n    while len(list(barrier.glob("*.json"))) < expected:\n        pass\n')
    root = tmp_path / "validation"
    argv = [str(module_path), "--root", str(root), "--historical-evaluation", str(historical), "--max-seconds", "300"]
    monkeypatch.setattr(module, "isolation_capability", lambda: {"l2_available": True})
    monkeypatch.setattr(module, "client_from_file", lambda *_: pytest.fail("preparation must not read credentials"))
    monkeypatch.setattr(sys, "argv", argv + ["--prepare-only"])
    assert module.main() == 0
    assert (root / "preparation/tools/session.json").is_file()
    assert not (root / "budget.json").exists() and not (root / "tools").exists() and not (root / "agent").exists()
    # Simulate a long pending approval. Preparation can be reviewed again, but
    # it must not consume the execution deadline or generate API calls.
    atomic_json(root / "preparation/budget.json", {"deadline_epoch": time.time()-3600})
    assert module.main() == 0
    snapshots = []
    class LocalOnlyAgent:
        def __init__(self, root, client, *, tools, **kwargs):
            snapshots.append({"tools_deadline": tools.deadline_epoch, "incident": tools.incident})
        def run(self, incident):
            return {"status": "stopped", "api_used": False, "steps": []}
    monkeypatch.setattr(module, "client_from_file", lambda *_: object())
    monkeypatch.setattr(module, "RepairAgent", LocalOnlyAgent)
    monkeypatch.setattr(sys, "argv", argv + ["--env-file", str(tmp_path / "unused_fake.env")])
    assert module.main() == 2
    assert snapshots[-1]["tools_deadline"] > time.time()+250
    original = read_json(root / "budget.json")["deadline_epoch"]
    # Tightening an execution budget preserves the original incident/API context.
    atomic_json(root / "budget.json", {"deadline_epoch": original-50})
    assert module.main() == 2
    assert snapshots[-1]["tools_deadline"] == original-50
    assert snapshots[-1]["incident"] == snapshots[-2]["incident"]
    assert read_json(root / "preparation/budget.json")["deadline_epoch"] < time.time()


class FakeClient:
    model = "deterministic-cpu-test-fixture"
    base_url = "https://fixture.invalid"
    available = True

    def __init__(self):
        self.calls = 0

    def chat_with_metadata(self, system, user, **kwargs):
        context = json.loads(user)
        history = context["history"]
        index = len(history)
        candidate = next((row["result"]["candidate_id"] for row in history if row["action"] == "apply_patch_candidate"), None)
        actions = [("read_evidence", {}), ("read_code", {"path": "component.py"}),
                   ("run_reproducer", {"validator": "native_membership"}),
                   ("apply_patch_candidate", {"hypothesis": "Reject stale, duplicate or non-live membership.",
                       "edits": [{"path": "component.py", "old": OLD, "new": NEW}]}),
                   ("run_validation", {"candidate_id": candidate}),
                   ("request_activation", {"candidate_id": candidate})]
        action, arguments = actions[index]
        self.calls += 1
        return json.dumps({"schema_version": 1, "incident_id": context["incident"]["incident_id"],
                           "action": action, "arguments": arguments,
                           "hypothesis": "File count does not establish current, live, unique membership."}), {
                               "model": self.model, "usage": {"total_tokens": 400}}


def test_tool_using_api_session_and_restart_are_idempotent(tmp_path, kernel_isolation):
    broker, client = make_broker(tmp_path), FakeClient()
    agent = RepairAgent(tmp_path / "agent", client, tools=broker)
    result = agent.run(broker.incident)
    assert result["status"] == "activation_requested", result
    assert client.calls == 6
    assert agent.run(broker.incident) == result and client.calls == 6
    assert [row["action"] for row in result["steps"]] == ["read_evidence", "read_code", "run_reproducer",
        "apply_patch_candidate", "run_validation", "request_activation"]


def test_unknown_api_outcome_not_resent_or_recharged(tmp_path, monkeypatch):
    broker, client = make_broker(tmp_path), FakeClient()
    monkeypatch.setattr("autosim.research.repair_agent.isolation_capability",
                        lambda: {"l2_available": True, "backend": "test-no-code-executed"})
    def lost(*args, **kwargs):
        client.calls += 1
        raise TimeoutError("API_KEY=private-controller-value")
    client.chat_with_metadata = lost
    agent = RepairAgent(tmp_path / "agent", client, tools=broker)
    with pytest.raises(RuntimeError, match="request failed"):
        agent.run(broker.incident)
    with pytest.raises(RuntimeError, match="unresolved API request"):
        agent.run(broker.incident)
    assert client.calls == 1
    assert read_json(tmp_path / "agent/api/budget.json")["calls"] == 1
    assert all("private-controller-value" not in path.read_text() for path in (tmp_path / "agent").rglob("*.json"))


def test_original_deadline_and_candidate_limit_are_not_reset(tmp_path):
    broker = make_broker(tmp_path, max_candidates=1)
    original = broker.deadline_epoch
    restored = RepairTools(broker.root, source_root=broker.source_root, allowed_code_paths=broker.allowed,
        incident=broker.incident, validators=broker.validators, deadline_epoch=original + 9999,
        max_candidates=4)
    assert restored.deadline_epoch == original
    assert read_json(broker.root / "session.json")["max_candidates"] == 1
