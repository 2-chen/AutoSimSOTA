import json

import pytest

from autosim.research.action_receipt import (
    ActionReceipt, ActionReceiptError, read_action_receipt, verify_action_receipt,
    write_action_receipt,
)
from autosim.research.common import object_digest


def receipt(**changes):
    fields = {
        "action_id": "0123456789abcdef", "run_id": "run-a",
        "repository": "/tmp/repository-a", "action": "derive_a_command",
        "decision_id": "0123456789abcdef", "state_revision": 7,
        "started_at": "2026-09-27T12:00:00Z", "finished_at": "2026-09-27T12:00:02Z",
        "status": "done", "reason": "the evaluate stage passed its command check",
        "arguments": {"stage": "evaluate"}, "evidence_refs": ["derived_stages.json"],
        "child_attempt_ids": ["attempt-1"], "reported_postconditions": ["command exits zero"],
        "verification_level": "L1", "resource_limits": {"wall_seconds": 30},
        "costs": {"wall_seconds": 2.0, "gpu_seconds": None,
                  "gpu_seconds_status": "not_separately_metered"},
        "role_handoff": {},
    }
    fields.update(changes)
    return ActionReceipt(**fields)


def test_action_receipt_is_immutable_verifiable_and_idempotently_writable(tmp_path):
    repository = tmp_path / "repository-a"
    sample = receipt(repository=str(repository))
    reference, record = write_action_receipt(tmp_path, sample)
    assert reference == "action_receipts/0123456789abcdef.json"
    verified = read_action_receipt(tmp_path, reference, run_id="run-a",
                                   repository=repository)
    assert verified == record
    assert verified["costs"]["gpu_seconds"] is None

    same_reference, same_record = write_action_receipt(tmp_path, sample)
    assert (same_reference, same_record) == (reference, record)
    with pytest.raises(ActionReceiptError, match="different content"):
        write_action_receipt(tmp_path, receipt(repository=str(repository), status="failed"))


def test_action_receipt_rejects_tampering_and_identity_mismatch(tmp_path):
    repository = tmp_path / "repository-a"
    reference, record = write_action_receipt(tmp_path, receipt(repository=str(repository)))
    tampered = {**record, "status": "failed"}
    with pytest.raises(ActionReceiptError, match="hash is invalid"):
        verify_action_receipt(tampered)
    with pytest.raises(ActionReceiptError, match="another run"):
        read_action_receipt(tmp_path, reference, run_id="run-b",
                            repository=repository)
    with pytest.raises(ActionReceiptError, match="another repository"):
        read_action_receipt(tmp_path, reference, run_id="run-a",
                            repository=tmp_path / "repository-b")


def test_action_receipt_path_and_symlink_escape_are_refused(tmp_path):
    with pytest.raises(ActionReceiptError, match="unsafe"):
        read_action_receipt(tmp_path, "../outside.json", run_id="run-a",
                            repository=tmp_path / "repository-a")

    outside = tmp_path / "outside"
    outside.mkdir()
    directory = tmp_path / "run" / "action_receipts"
    directory.parent.mkdir()
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ActionReceiptError, match="symlink"):
        write_action_receipt(directory.parent, receipt())


def test_action_receipt_store_is_readable_json(tmp_path):
    reference, _ = write_action_receipt(tmp_path, receipt())
    path = tmp_path / reference
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == 1
    assert parsed["action"] == "derive_a_command"


def test_action_receipt_persists_and_validates_role_handoff(tmp_path):
    repository = tmp_path / "repository-a"
    handoff = {"to": "fix", "trigger": "native stage contradicted environment",
               "failure": {"stage": "train", "evidence_ref": "train/output.log"}}
    reference, record = write_action_receipt(
        tmp_path, receipt(repository=str(repository), role_handoff=handoff))

    assert read_action_receipt(tmp_path, reference, run_id="run-a",
                               repository=repository)["role_handoff"] == handoff
    invalid = {**record, "role_handoff": "fix"}
    invalid.pop("receipt_sha256")
    invalid["receipt_sha256"] = object_digest(invalid)
    with pytest.raises(ActionReceiptError, match="role_handoff must be an object"):
        verify_action_receipt(invalid)
