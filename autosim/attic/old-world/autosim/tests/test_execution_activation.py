"""CPU-only host promotion, actual first import, inherited workers and rollback.

The candidate here is an explicit synthetic fixture. Real API coding acceptance
is performed separately by validate_harness_repair_api.py with outbound approval.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from autosim.research import probe_barrier_contract
from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.continuation import ContinuationSupervisor
from autosim.research.execution_activation import (
    ENVIRONMENT_KEYS, HOST_FILES, TARGET, ActivationRefused, ExecutionActivation,
    active_execution_identity, install_activation_from_environment, verify_prepared_activation,
)
from autosim.research.incidents import build_incident
from autosim.research.repair_tools import RepairTools, membership_validation_spec, production_repair_tools
from autosim.research.repair_validation import isolation_capability


BASE = Path(probe_barrier_contract.__file__).read_text()
OLD = "def ready_members_match(expected: list[dict], ready: list[dict], generation: str) -> bool:\n    return not (len(ready) < len(expected))\n"
CANDIDATE = BASE + "\n# Explicit CPU fixture: independently promoted source identity, no API claim.\n"
SCIENCE = {"task": "CPU lifecycle fixture", "seeds": [1, 2], "selection": "sealed_fixture"}


@pytest.fixture(scope="module")
def kernel():
    result = isolation_capability()
    if not result["l2_available"]:
        if os.environ.get("AUTOSIM_REQUIRE_REPAIR_SANDBOX") == "1":
            pytest.fail(f"required kernel isolation unavailable: {result}")
        pytest.skip("kernel isolation blocked by outer execution sandbox")
    return result


@pytest.fixture
def manager(tmp_path):
    source = tmp_path / "frozen"
    research = source / "autosim/autosim/research"
    original = Path(probe_barrier_contract.__file__).parent
    research.mkdir(parents=True)
    # The child must actually import the frozen trusted loader, not the worktree.
    for path in original.glob("*.py"):
        shutil.copyfile(path, research / path.name)
    shutil.copyfile(original.parent / "__init__.py", research.parent / "__init__.py")
    files = {str(path.relative_to(source)): digest(path) for path in source.rglob("*.py")}
    return ExecutionActivation(tmp_path / "host", source_root=source, source_files=files,
        scientific_contract=SCIENCE, base_execution_revision=object_digest(files), deadline_epoch=time.time() + 300)


def repair_request(tmp_path):
    source = tmp_path / "legacy_control"
    target = source / TARGET
    target.parent.mkdir(parents=True)
    target.write_text(OLD)
    incident = build_incident(run_id="synthetic_activation_fixture", stage="native_probe", purpose="smoke", worker_roots={})
    spec = membership_validation_spec(TARGET)
    broker = RepairTools(tmp_path / "repair", source_root=source, allowed_code_paths=[TARGET], incident=incident,
                        validators={spec.name: spec}, deadline_epoch=time.time() + 300, max_candidates=1)
    candidate = broker.execute("apply_patch_candidate", {"hypothesis": "CPU fixture replaces the count-only predicate.",
        "edits": [{"path": TARGET, "old": OLD, "new": CANDIDATE}]})
    assert candidate["status"] == "candidate_created", candidate
    candidate_id = candidate["candidate_id"]
    validation = broker.execute("run_validation", {"candidate_id": candidate_id})
    assert validation["status"] == "validated", validation
    request = broker.execute("request_activation", {"candidate_id": candidate_id})
    assert request["status"] == "activation_requested", request
    return request


def child_environment(manager, plan):
    return {**os.environ, **plan["env"], "PYTHONPATH": str(manager.source_root / "autosim")}


CHILD = """
import json, os
from autosim.research import probe_barrier_contract as component
from autosim.research.execution_activation import active_execution_identity
expected = [dict(worker='w', uuid='GPU-u', generation='g', claim='c', attempt_id='a')]
ready = [dict(expected[0], state='ready', alive=True)]
assert component.ready_members_match(expected, ready, 'g') is True
ready[0]['generation'] = 'old'
assert component.ready_members_match(expected, ready, 'g') is False
print(json.dumps(dict(identity=active_execution_identity(), path=component.__file__, pid=os.getpid())))
"""


def run_child(manager, plan, code=CHILD, *, env=None):
    result = subprocess.run([sys.executable, "-c", code], cwd=manager.source_root,
        env=env or child_environment(manager, plan), text=True, capture_output=True, timeout=20)
    return result


def test_baseline_loads_only_on_fresh_import_and_all_child_workers_inherit(manager, kernel):
    plan = manager.register_baseline()
    assert manager.current_context(manager.base_revision) == plan
    body = verify_prepared_activation(plan)
    assert body["mode"] == "verified_baseline" and body["provenance"]["api_used"] is False
    assert plan["validation_receipt"]["component_contract"]["cases"] == 298
    assert (manager.root / ".host-signing-key").stat().st_mode & 0o777 == 0o600
    code = CHILD + "\nimport subprocess, sys\nsubprocess.run([sys.executable, '-c', " + repr(CHILD) + "], check=True)\n"
    result = run_child(manager, plan, code)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 2 and len({row["pid"] for row in rows}) == 2
    assert all(row["pid"] != os.getpid() and row["identity"]["execution_revision"] == manager.base_revision for row in rows)
    assert all(row["path"].startswith(str(manager.root / "components")) for row in rows)
    assert len(list((manager.root / "load_receipts").glob("*.json"))) == 2
    assert active_execution_identity() is None
    assert Path(probe_barrier_contract.__file__).read_text() == BASE


def test_historical_control_requires_explicit_promotion_and_keeps_both_ancestries(manager, tmp_path, kernel):
    request = repair_request(tmp_path)
    with pytest.raises(ActivationRefused, match="base differs"):
        manager.prepare_activation(request)
    plan = manager.prepare_activation(request, promote_historical_control=True)
    body = verify_prepared_activation(plan)
    proof = body["provenance"]
    assert proof["relation"] == "historical_control_to_validated_component_promotion"
    assert proof["repair_base_component_sha256"] != proof["execution_base_component_sha256"]
    assert plan["validation_receipt"]["repair_control"]["baseline_reproduced"]
    assert plan["validation_receipt"]["current_base_contract"]["passed"]
    assert plan["validation_receipt"]["candidate_contract"]["passed"]
    assert plan["execution_revision"] != manager.base_revision
    assert plan["compatibility"]["from_revision"] == manager.base_revision
    assert manager.prepare_activation(request, promote_historical_control=True) == plan
    result = run_child(manager, plan)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["identity"]["component_sha256"] == request["candidate_hashes"][TARGET]
    assert digest(manager.source_root / TARGET) == manager.files[TARGET]
    rollback = manager.prepare_rollback(from_revision=plan["execution_revision"])
    assert rollback["rollback"] and rollback["execution_revision"] == manager.base_revision
    result = run_child(manager, rollback)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["identity"]["component_sha256"] == manager.files[TARGET]


def test_continuation_executes_baseline_candidate_and_rollback_in_distinct_processes(manager, tmp_path, kernel):
    request = repair_request(tmp_path)
    base = manager.register_baseline()
    candidate = manager.prepare_activation(request, promote_historical_control=True)
    rollback = manager.prepare_rollback(from_revision=candidate["execution_revision"])
    supervisor = ContinuationSupervisor(tmp_path / "outer", deadline_epoch=time.time() + 90,
                                        max_restarts=2, max_no_progress=4)
    resume = {"safe_boundary": True, "stage": "native_probe", "scientific_contract_sha256": object_digest(SCIENCE)}
    for index, context in enumerate((base, candidate, rollback)):
        verify_prepared_activation(context)
        result = supervisor.run_once([sys.executable, "-c", CHILD + f"\nraise SystemExit({int(index < 2)})"],
            cwd=manager.source_root, execution_revision=context["execution_revision"], scientific_contract=SCIENCE,
            stage="native_probe", allocation_reconciled=True, resume_receipt=resume,
            validation_receipt=context["validation_receipt"], compatibility=context["compatibility"],
            rollback=context["rollback"], env=child_environment(manager, context), cleanup_seconds=1)
        assert result["status"] == ("completed" if index == 2 else "failed"), result
    journal = read_json(supervisor.path)
    assert [row["operation"] for row in journal["attempts"]] == ["activate", "activate", "rollback"]
    assert len({row["worker"]["pid"] for row in journal["attempts"]}) == 3
    loads = [read_json(path) for path in (manager.root / "load_receipts").glob("*.json")]
    assert len(loads) == 3
    assert {row["execution_revision"] for row in loads} == {manager.base_revision, candidate["execution_revision"]}


def test_environment_cannot_hot_reload_an_existing_research_process(manager, kernel, monkeypatch):
    plan = manager.register_baseline()
    for name, value in plan["env"].items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ActivationRefused, match="after research package import"):
        install_activation_from_environment()
    with pytest.raises(ActivationRefused, match="after import"):
        active_execution_identity()


@pytest.mark.parametrize("mutation", ["component", "validation", "signature", "frozen_source", "registry_index"])
def test_host_evidence_and_frozen_source_tampering_prevent_child_activation(manager, kernel, mutation):
    plan = manager.register_baseline()
    sidecar = Path(plan["activation_file"])
    envelope = read_json(sidecar)
    body = envelope["body"]
    if mutation == "component":
        (manager.root / body["component_file"]).write_text(OLD)
    elif mutation == "validation":
        receipt = manager.root / body["validation_file"]
        atomic_json(receipt, {**read_json(receipt), "passed": False})
    elif mutation == "signature":
        envelope["body"]["deadline_epoch"] += 900
        atomic_json(sidecar, envelope)
        plan["env"][ENVIRONMENT_KEYS[1]] = digest(sidecar)
    elif mutation == "frozen_source":
        (manager.source_root / TARGET).write_text(OLD)
    else:
        state = read_json(manager.root / "registry.json")
        state["revisions"] = {}
        atomic_json(manager.root / "registry.json", state)
    with pytest.raises(ActivationRefused):
        verify_prepared_activation(plan)
    result = run_child(manager, plan)
    assert result.returncode != 0 and "ActivationRefused" in result.stderr
    assert not list((manager.root / "load_receipts").glob("*.json"))


def test_trusted_loader_must_be_loaded_from_the_frozen_execution_root(manager, kernel):
    plan = manager.register_baseline()
    env = child_environment(manager, plan)
    env["PYTHONPATH"] = str(Path(probe_barrier_contract.__file__).parents[2])
    result = run_child(manager, plan, env=env)
    assert result.returncode != 0 and "not the pinned frozen execution source" in result.stderr


def test_host_restart_recovers_signed_publication_without_revalidating_or_renewing(manager, kernel, monkeypatch):
    baseline = manager.register_baseline()
    state = read_json(manager.root / "registry.json")
    original_deadline = state["deadline_epoch"]
    state["revisions"] = {}  # Crash after sidecar rename, before registry-index rename.
    atomic_json(manager.root / "registry.json", state)
    recovered = ExecutionActivation(manager.root, source_root=manager.source_root, source_files=manager.files,
        scientific_contract=SCIENCE, base_execution_revision=manager.base_revision, deadline_epoch=original_deadline + 3600)
    monkeypatch.setattr("autosim.research.execution_activation.run_contract", lambda *_: pytest.fail("must reuse signed evidence"))
    assert recovered.register_baseline() == baseline
    assert recovered.deadline_epoch == original_deadline


def test_tightened_global_deadline_invalidates_previously_prepared_environment(manager, kernel):
    baseline = manager.register_baseline()
    state = read_json(manager.root / "registry.json")
    state["deadline_epoch"] = time.time() - 1
    atomic_json(manager.root / "registry.json", state)
    with pytest.raises(ActivationRefused, match="budget exhausted"):
        verify_prepared_activation(baseline)
    result = run_child(manager, baseline)
    assert result.returncode != 0 and "budget exhausted" in result.stderr


def test_missing_loader_manifest_and_unregistered_rollback_are_rejected(manager):
    files = {name: value for name, value in manager.files.items() if name not in HOST_FILES}
    with pytest.raises(ActivationRefused, match="trusted import hook"):
        ExecutionActivation(manager.root / "other", source_root=manager.source_root, source_files=files,
            scientific_contract=SCIENCE, base_execution_revision=manager.base_revision, deadline_epoch=time.time() + 60)
    with pytest.raises(ActivationRefused, match="already validated"):
        manager.prepare_rollback(from_revision="0" * 64)


def test_production_tool_factory_preserves_policy_candidate_limit(manager, tmp_path):
    incident = build_incident(run_id="factory", stage="native_probe", purpose="smoke", worker_roots={})
    broker = production_repair_tools(incident, manager.source_root, tmp_path / "limited", time.time() + 300,
                                    max_candidates=1)
    assert read_json(broker.root / "session.json")["max_candidates"] == 1
