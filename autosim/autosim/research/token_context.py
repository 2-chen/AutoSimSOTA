"""Lossless on-demand context paging; never summarize scientific constraints away."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import atomic_json
from .evidence_store import capture_attempt_evidence


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ": "), default=str)


def project_context(context: dict, user: str, output: Path, *, pageable: bool = True
                    ) -> tuple[dict, dict]:
    """Deduplicate exact supplied fields and page bulky leaves with verified read IDs.

    Dictionaries retain all keys; short scalar facts, IDs, protocol and error fields
    remain visible. Long strings/lists have previews plus an immutable full receipt.
    Tool-less roles must receive the full context instead of inaccessible references.
    """
    try:
        request = json.loads(user)
    except (ValueError, TypeError):
        request = None
    supplied = []

    def collect(value):
        if isinstance(value, dict):
            supplied.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(request)
    counts = {"deduplicated_fields": 0, "paged_leaves": 0,
              "original_context_bytes": len(compact_json(context).encode())}

    def page(value):
        encoded = compact_json(value)
        identity = hashlib.sha256(encoded.encode()).hexdigest()[:32]
        path = output / "agent" / "context_blobs" / (identity + ".json")
        if any(parent.is_symlink() for parent in (output / "agent", path.parent)):
            raise ValueError("unsafe context blob directory")
        # Existing immutable blobs must not be overwritten (including symlinks).
        if not path.exists() and not path.is_symlink():
            atomic_json(path, value)
        elif path.is_symlink() or json.loads(path.read_text()) != value:
            raise ValueError("context blob identity collision or unsafe path")
        receipt = capture_attempt_evidence(output, attempt_id=identity, log=path,
            receipt_ref=path.relative_to(output).as_posix(), status="context_snapshot",
            returncode=None, termination_reason="paged_model_context")
        counts["paged_leaves"] += 1
        preview = value[-6:] if isinstance(value, list) else value[:1500] + " … " + value[-1000:]
        return {"context_paged": True, "items_or_characters": len(value),
                "preview": visit(preview), "evidence_id": receipt["evidence_id"],
                "read_instruction": "Use read_evidence with this ID and byte offsets for the full JSON; preview is incomplete, not authoritative."}

    def visit(value):
        if pageable and ((isinstance(value, str) and len(value) > 6000) or
                         (isinstance(value, list) and len(value) > 16 and
                          len(compact_json(value).encode()) > 4000)):
            return page(value)
        if isinstance(value, dict):
            return {key: (item if any(word in str(key).lower() for word in
                ("protocol", "constraint", "rubric", "objective", "red_line", "redline",
                 "error", "failure", "rule", "permission", "contract", "schema"))
                else visit(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    projected = {}
    for key, value in context.items():
        matches = [obj[key] for obj in supplied if key in obj]
        if matches and all(item == value for item in matches):
            counts["deduplicated_fields"] += 1
            continue
        projected[key] = visit({key: value})[key]
    counts["projected_context_bytes"] = len(compact_json(projected).encode())
    return projected, counts
