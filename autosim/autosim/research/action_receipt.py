"""Immutable receipts for actions owned by the outer research controller."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common import atomic_json, object_digest, read_json


class ActionReceiptError(ValueError):
    """A controller action receipt is malformed, unsafe, or conflicts with an existing one."""


@dataclass(frozen=True)
class ActionReceipt:
    """What one bounded controller action returned, bound to the decision that requested it.

    Child process receipts remain authoritative for argv, environment, and artifact details.
    This parent receipt binds those child identities to the outer action and records only
    measured wall time here; GPU seconds remain unknown unless a separate meter supplies them.
    """

    action_id: str
    run_id: str
    repository: str
    action: str
    decision_id: str
    state_revision: int
    started_at: str
    finished_at: str
    status: str
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    child_attempt_ids: list[str] = field(default_factory=list)
    reported_postconditions: list[str] = field(default_factory=list)
    postcondition_verification: str = "not_independently_verified"
    verification_level: str | None = None
    resource_limits: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)
    role_handoff: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}


def _sealed(receipt: ActionReceipt) -> dict[str, Any]:
    row = receipt.as_dict()
    row["receipt_sha256"] = object_digest(row)
    return row


def verify_action_receipt(record: Any, *, run_id: str | None = None,
                          repository: Path | None = None) -> dict[str, Any]:
    """Verify immutable content hash, schema, and optional run/repository identity."""
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ActionReceiptError("unsupported action receipt schema")
    material = {key: value for key, value in record.items() if key != "receipt_sha256"}
    if record.get("receipt_sha256") != object_digest(material):
        raise ActionReceiptError("action receipt hash is invalid")
    action_id = str(record.get("action_id") or "")
    if not re.fullmatch(r"[0-9a-f]{16}", action_id):
        raise ActionReceiptError("action receipt id is unsafe")
    if run_id is not None and record.get("run_id") != str(run_id):
        raise ActionReceiptError("action receipt belongs to another run")
    if repository is not None:
        recorded = Path(str(record.get("repository") or "")).expanduser().resolve()
        if recorded != Path(repository).expanduser().resolve():
            raise ActionReceiptError("action receipt belongs to another repository")
    required = ("action", "decision_id", "started_at", "finished_at", "status", "reason")
    if any(not isinstance(record.get(key), str) or not record[key] for key in required):
        raise ActionReceiptError("action receipt has missing identity or outcome fields")
    if not isinstance(record.get("state_revision"), int) or isinstance(
            record.get("state_revision"), bool):
        raise ActionReceiptError("action receipt state revision is invalid")
    for name in ("arguments", "resource_limits", "costs"):
        if not isinstance(record.get(name), dict):
            raise ActionReceiptError(f"action receipt {name} must be an object")
    if "role_handoff" in record and not isinstance(record.get("role_handoff"), dict):
        raise ActionReceiptError("action receipt role_handoff must be an object")
    for name in ("evidence_refs", "child_attempt_ids", "reported_postconditions"):
        if (not isinstance(record.get(name), list) or
                any(not isinstance(item, str) for item in record[name])):
            raise ActionReceiptError(f"action receipt {name} must be a string list")
    return dict(record)


def write_action_receipt(root: Path, receipt: ActionReceipt) -> tuple[str, dict[str, Any]]:
    """Persist once; an identical retry is idempotent, a conflicting rewrite is rejected."""
    if not re.fullmatch(r"[0-9a-f]{16}", receipt.action_id):
        raise ActionReceiptError("action receipt id is unsafe")
    root = Path(root).resolve()
    directory = root / "action_receipts"
    if directory.is_symlink():
        raise ActionReceiptError("action receipt directory is a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.resolve().parent != root:
        raise ActionReceiptError("action receipt directory escapes the run root")
    path = directory / f"{receipt.action_id}.json"
    if path.is_symlink():
        raise ActionReceiptError("action receipt path is a symlink")
    sealed = _sealed(receipt)
    if path.exists():
        existing = verify_action_receipt(
            read_json(path), run_id=receipt.run_id, repository=Path(receipt.repository))
        if existing.get("receipt_sha256") != sealed["receipt_sha256"]:
            raise ActionReceiptError("action receipt already exists with different content")
    else:
        atomic_json(path, sealed)
    verify_action_receipt(sealed, run_id=receipt.run_id,
                          repository=Path(receipt.repository))
    return path.relative_to(root).as_posix(), sealed


def read_action_receipt(root: Path, reference: str, *, run_id: str,
                        repository: Path) -> dict[str, Any]:
    """Read one safe relative receipt reference and verify its identity and hash."""
    relative = Path(reference)
    if (relative.is_absolute() or ".." in relative.parts or
            relative.parts[:1] != ("action_receipts",) or relative.suffix != ".json"):
        raise ActionReceiptError("action receipt reference is unsafe")
    root = Path(root).resolve()
    path = root / relative
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ActionReceiptError("action receipt escapes the run root") from exc
    if (path.is_symlink() or path.parent.is_symlink() or not path.is_file() or
            resolved.parent != (root / "action_receipts")):
        raise ActionReceiptError("action receipt is missing or unsafe")
    try:
        record = read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        raise ActionReceiptError(f"action receipt cannot be read: {type(exc).__name__}") from exc
    return verify_action_receipt(record, run_id=run_id, repository=repository)
