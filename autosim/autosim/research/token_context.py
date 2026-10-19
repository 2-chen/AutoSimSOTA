"""Lossless on-demand context paging; never summarize scientific constraints away."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import atomic_json
from .evidence_store import capture_attempt_evidence


_INLINE_KEYS = ("protocol", "constraint", "rubric", "objective", "red_line", "redline",
                "error", "failure", "stderr", "rule", "permission", "contract", "schema",
                "budget", "instruction", "capability", "active_job", "current_action")


def _inline_key(key: Any) -> bool:
    return any(word in str(key).lower() for word in _INLINE_KEYS)


def _contains_inline(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_inline_key(key) or _contains_inline(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_inline(item) for item in value)
    return False


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ": "), default=str)


def request_json(value: Any) -> str:
    """Stable methods/instructions before changing event evidence, without duplicates.

    Only JSON member ordering changes. Every member/value remains present, and
    nested objects are canonical. This is not a promise of provider prefix reuse.
    """
    if not isinstance(value, dict):
        return compact_json(value)
    leading = [key for key in ('methods', 'selected_methods', 'skill_catalog', 'instruction')
               if key in value]
    keys = [*leading, *sorted(key for key in value if key not in leading)]
    return '{' + ','.join(compact_json(key) + ': ' + compact_json(value[key]) for key in keys) + '}'


def project_context(context: dict, user: str, output: Path, *, pageable: bool = True
                    ) -> tuple[dict, dict]:
    """Deduplicate exact supplied fields and page bulky leaves with verified read IDs.

    Short facts and protocol/error subtrees remain visible. Bulky unguarded maps,
    strings and lists have previews plus an immutable full receipt.
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
        if isinstance(value, dict):
            keys = sorted(value)
            preview = {key: value[key] for key in [*keys[:4], *keys[-4:]]}
        elif isinstance(value, list):
            preview = value[-6:]
            if len(compact_json(preview).encode()) > 4000:
                # Few huge records are not a short catalog. Avoid recursively
                # paging the same list as its own preview, and do not copy giant
                # patches/observations into a supposedly small head document.
                def bounded(item, depth=0):
                    if isinstance(item, str):
                        return item if len(item) <= 240 else item[:240] + ' [excerpt; read full evidence]'
                    if isinstance(item, dict):
                        if depth >= 2:
                            return {'preview_omitted':True,'kind':'object','items':len(item)}
                        keys = sorted(item)
                        return {k:bounded(item[k],depth+1) for k in dict.fromkeys([*keys[:4],*keys[-4:]])}
                    if isinstance(item, list):
                        if depth >= 2:
                            return {'preview_omitted':True,'kind':'array','items':len(item)}
                        return [bounded(v,depth+1) for v in item[:2]]
                    return item
                selected = value if len(value) <= 3 else [value[0], *value[-2:]]
                preview = [bounded(item) for item in selected]
        else:
            preview = value[:1500] + " … " + value[-1000:]
        return {"context_paged": True, "items_or_characters": len(value),
                "preview": preview if isinstance(value,list) else visit(preview),
                "evidence_id": receipt["evidence_id"],
                "read_instruction": "Use read_evidence with this ID and byte offsets for the full JSON; preview is incomplete, not authoritative."}

    def visit(value):
        if pageable and ((isinstance(value, str) and len(value) > 6000) or
                         (isinstance(value, list) and (
                             (len(value) > 16 and len(compact_json(value).encode()) > 4000) or
                             (len(compact_json(value).encode()) > 6000 and not _contains_inline(value))))):
            return page(value)
        if isinstance(value, dict):
            if (pageable and len(value) > 32 and not _contains_inline(value) and
                    len(compact_json(value).encode()) > 6000):
                return page(value)
            return {key: (item if _inline_key(key)
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
        projected[key] = value if _inline_key(key) else visit(value)
    counts["projected_context_bytes"] = len(compact_json(projected).encode())
    return projected, counts


def project_request(user: str, output: Path, *, pageable: bool) -> tuple[str, dict]:
    """Page structured request evidence as well as shared context, without an LLM call.

    Free-form instructions and tool-less requests are never replaced by references.
    The complete original JSON remains in the evidence store, not a model summary.
    """
    try:
        request = json.loads(user)
    except (ValueError, TypeError):
        return user, {"request_paged_leaves": 0, "original_request_bytes": len(user.encode()),
                      "projected_request_bytes": len(user.encode())}
    if not isinstance(request, dict):
        return request_json(request), {"request_paged_leaves": 0,
            "original_request_bytes": len(compact_json(request).encode()),
            "projected_request_bytes": len(compact_json(request).encode())}
    projected, metrics = project_context(request, "null", output, pageable=pageable)
    return request_json(projected), {"request_paged_leaves": metrics["paged_leaves"],
        "original_request_bytes": metrics["original_context_bytes"],
        "projected_request_bytes": metrics["projected_context_bytes"]}


def project_tool_result(value: dict, output: Path) -> dict:
    """Offload bulky successful tool evidence BEFORE it enters CLI history.

    Failed operations keep their immediate error surface intact. The dispatcher must
    not call this for read_evidence itself (otherwise reading could page forever).
    """
    if value.get('status') in {'failed', 'timed_out', 'rejected', 'unknown'}:
        return value
    projected, _ = project_context(value, 'null', output, pageable=True)
    return projected
