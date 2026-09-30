"""Run-owned, immutable failure evidence with stable, bounded read references.

The executor captures facts. Agents may inspect them but cannot choose their own host path.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .common import atomic_json, read_json, sanitize_model_text


_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_LOG = 64 * 1024 * 1024


def capture_attempt_evidence(output: Path, *, attempt_id: str, log: Path,
                             receipt_ref: str, status: str, returncode: int | None,
                             termination_reason: str, artifact: dict[str, Any] | None = None,
                             metric_artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Seal an already run-owned log; never accept an external /tmp file as evidence."""
    if not _ID.fullmatch(attempt_id):
        raise ValueError("unsafe evidence id")
    output = Path(output).resolve(strict=True)
    resolved = Path(log).resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(output):
        raise ValueError("stage log is not stored inside this run")
    size = resolved.stat().st_size
    if size > _MAX_LOG:
        raise ValueError("stage log exceeds evidence store limit")
    data = resolved.read_bytes()
    relative = resolved.relative_to(output).as_posix()
    record = {
        "schema_version": 1, "evidence_id": attempt_id,
        "log_ref": relative, "log_sha256": hashlib.sha256(data).hexdigest(),
        "log_bytes": size, "receipt_ref": receipt_ref,
        "status": status, "returncode": returncode,
        "termination_reason": termination_reason,
        "artifact": artifact or {}, "metric_artifact": metric_artifact or {},
        "excerpt": sanitize_model_text(data[-3000:].decode("utf-8", "replace"))[-3000:],
    }
    destination = output / "evidence" / f"{attempt_id}.json"
    if destination.exists() or destination.is_symlink():
        if read_json(destination) != record:
            raise ValueError("attempt evidence already exists with different content")
    else:
        atomic_json(destination, record)
    return {"evidence_id": attempt_id,
            "evidence_ref": destination.relative_to(output).as_posix(),
            "log_ref": relative, "log_sha256": record["log_sha256"],
            "excerpt": record["excerpt"]}


def read_attempt_evidence(output: Path, evidence_id: str, *, offset: int = 0,
                          limit: int = 8000) -> dict[str, Any]:
    """Read a verified log slice by ID, never by a model-supplied filesystem path."""
    if not isinstance(evidence_id, str) or not _ID.fullmatch(evidence_id):
        raise ValueError("unsafe evidence id")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a nonnegative byte position")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 12000:
        raise ValueError("limit must be 1..12000 bytes")
    output = Path(output).resolve(strict=True)
    document = output / "evidence" / f"{evidence_id}.json"
    if document.is_symlink() or not document.is_file():
        raise ValueError("evidence record is missing or unsafe")
    record = read_json(document)
    if (not isinstance(record, dict) or record.get("evidence_id") != evidence_id or
            record.get("schema_version") != 1):
        raise ValueError("evidence identity is invalid")
    relative = Path(str(record.get("log_ref") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence log reference is unsafe")
    log = (output / relative).resolve(strict=True)
    if not log.is_relative_to(output) or not log.is_file() or log.is_symlink():
        raise ValueError("evidence log escaped or disappeared")
    data = log.read_bytes()
    if (len(data) != record.get("log_bytes") or
            hashlib.sha256(data).hexdigest() != record.get("log_sha256")):
        raise ValueError("evidence log changed after capture")
    return {"evidence_id": evidence_id, "status": record.get("status"),
            "returncode": record.get("returncode"),
            "termination_reason": record.get("termination_reason"),
            "receipt_ref": record.get("receipt_ref"),
            "artifact": record.get("artifact"),
            "metric_artifact": record.get("metric_artifact"),
            "log_bytes": len(data), "offset": offset,
            "next_offset": min(len(data), offset + limit),
            "text": sanitize_model_text(data[offset:offset + limit].decode(
                "utf-8", "replace"))}
