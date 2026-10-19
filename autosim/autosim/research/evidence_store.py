"""Run-owned, immutable failure evidence with stable, bounded read references.

The executor captures facts. Agents may inspect them but cannot choose their own host path.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Any

from .common import atomic_json, read_json, sanitize_model_text


_ID = re.compile(r"[0-9a-f]{32}\Z")
# Disk evidence and model delivery are different bounds. Native training may emit
# hundreds of MiB; hash it incrementally, never load it into memory or the prompt.
_MAX_LOG = 1024 * 1024 * 1024
_HASH_CHUNK = 1024 * 1024


def _log_slice(log: Path, *, offset: int | None, limit: int,
               expected_bytes: int | None = None, expected_sha256: str | None = None
               ) -> tuple[int, str, bytes]:
    """Hash the entire bounded regular file; retain only the requested byte slice."""
    descriptor = os.open(log, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        size = before.st_size
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('stage log must be a regular file')
        if size > _MAX_LOG:
            raise ValueError('stage log exceeds evidence store limit')
        if expected_bytes is not None and size != expected_bytes:
            raise ValueError('evidence log changed after capture')
        start = max(0, size-limit) if offset is None else min(offset, size)
        end = min(size, start+limit)
        hasher = hashlib.sha256()
        excerpt = bytearray()
        position = 0
        while position < size:
            chunk = stream.read(min(_HASH_CHUNK, size-position))
            if not chunk:
                raise ValueError('evidence log changed while reading')
            hasher.update(chunk)
            left, right = max(position, start), min(position+len(chunk), end)
            if left < right:
                excerpt.extend(chunk[left-position:right-position])
            position += len(chunk)
        fingerprint = hasher.hexdigest()
        if expected_sha256 is not None and fingerprint != expected_sha256:
            raise ValueError('evidence log changed after capture')
        # Fast same-length writes can share a filesystem timestamp tick. A second
        # bounded pass catches those changes without materializing the whole log.
        stream.seek(0)
        confirmation = hashlib.sha256()
        position = 0
        while position < size:
            chunk = stream.read(min(_HASH_CHUNK, size-position))
            if not chunk:
                raise ValueError('evidence log changed while reading')
            confirmation.update(chunk)
            position += len(chunk)
        if confirmation.hexdigest() != fingerprint:
            raise ValueError('evidence log changed while reading')
        after = os.fstat(stream.fileno())
        current = log.stat()
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                                row.st_mtime_ns, row.st_ctime_ns)
        if (stream.read(1) or identity(before) != identity(after)
                or identity(after) != identity(current)):
            raise ValueError('evidence log changed while reading')
        return size, fingerprint, bytes(excerpt)


def capture_attempt_evidence(output: Path, *, attempt_id: str, log: Path,
                             receipt_ref: str, status: str, returncode: int | None,
                             termination_reason: str, artifact: dict[str, Any] | None = None,
                             metric_artifact: dict[str, Any] | None = None,
                             native_context: dict[str, str] | None = None) -> dict[str, Any]:
    """Seal an already run-owned log; never accept an external /tmp file as evidence."""
    if not _ID.fullmatch(attempt_id):
        raise ValueError("unsafe evidence id")
    output = Path(output).resolve(strict=True)
    resolved = Path(log).resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(output):
        raise ValueError("stage log is not stored inside this run")
    size, fingerprint, excerpt = _log_slice(resolved, offset=None, limit=3000)
    relative = resolved.relative_to(output).as_posix()
    if native_context is not None and (not isinstance(native_context,dict)
            or set(native_context) != {'stage','output_directory','working_directory'}
            or any(not isinstance(v,str) or not v or len(v)>4000 for v in native_context.values())):
        raise ValueError('native evidence context requires bounded stage/output/working directory')
    record = {
        "schema_version": 1, "evidence_id": attempt_id,
        "log_ref": relative, "log_sha256": fingerprint,
        "log_bytes": size, "receipt_ref": receipt_ref,
        "status": status, "returncode": returncode,
        "termination_reason": termination_reason,
        "artifact": artifact or {}, "metric_artifact": metric_artifact or {},
        **({'native_context':native_context} if native_context is not None else {}),
        "excerpt": sanitize_model_text(excerpt.decode("utf-8", "replace"))[-3000:],
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
                          limit: int = 8000, path_encoder=None) -> dict[str, Any]:
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
    expected = record.get('log_bytes')
    fingerprint = record.get('log_sha256')
    if (isinstance(expected, bool) or not isinstance(expected, int)
            or not 0 <= expected <= _MAX_LOG or not isinstance(fingerprint, str)
            or not re.fullmatch(r'[0-9a-f]{64}', fingerprint)):
        raise ValueError('invalid bounded evidence log identity')
    size, _, data = _log_slice(log, offset=offset, limit=limit,
        expected_bytes=expected, expected_sha256=fingerprint)
    if size != expected:
        raise ValueError("evidence log changed after capture")
    excerpt = data.decode('utf-8', 'replace')
    if path_encoder is not None:
        excerpt = path_encoder(excerpt)
    return {"evidence_id": evidence_id, "status": record.get("status"),
            "returncode": record.get("returncode"),
            "termination_reason": record.get("termination_reason"),
            "receipt_ref": record.get("receipt_ref"),
            "artifact": record.get("artifact"),
            "metric_artifact": record.get("metric_artifact"),
            "native_context": record.get("native_context") or {},
            "log_bytes": size, "offset": offset,
            "next_offset": min(size, offset + limit),
            "text": sanitize_model_text(excerpt)}
