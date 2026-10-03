"""Recover only a read-only review; never replay installation or infer approval."""
import json
import time
from pathlib import Path

from .common import atomic_json, object_digest, read_json, now, digest


def review_call(client, *, instructions, payload, output, timeout=180):
    identity = object_digest({"version": 1, "instructions": instructions,
        "payload": payload, "model": getattr(client, "model", "unknown"),
        "transport_implementation": {name: digest(Path(__file__).with_name(name)) for name in
            ("agent_runtime.py", "process_executor.py", "deepseek_gateway.py")}})
    root = Path(output) / "review_transactions"
    if root.is_symlink():
        raise ValueError("review transaction directory is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{identity}.json"
    if path.is_symlink():
        raise ValueError("review transaction record is a symlink")
    previous = read_json(path) if path.exists() else {}
    if not isinstance(previous, dict) or (previous and previous.get("identity") != identity):
        raise ValueError("review transaction identity invalid; refusing cached result")
    if previous.get("identity") == identity and previous.get("status") == "completed":
        return previous["content"], {"review_identity": identity, "cache_hit": True}
    from .agent_client import AgentRuntimeClientError
    if previous.get("identity") == identity and previous.get("status") == "transport_unavailable":
        raise AgentRuntimeClientError("read-only review transport remains unavailable for this "
            "unchanged evidence bundle; no native operation launched", failure_category="review_transport",
            runtime_failure=previous.get("runtime_failure") or {})
    deadline = time.monotonic() + max(1, timeout)
    attempts = list(previous.get("attempts") or [])
    # Legacy/fake clients preserve their ordinary contract. The real adapter keeps all
    # calls on the existing gateway, cost ledger and hard wall budget, with zero tools.
    compact = getattr(client, "supports_main_agent", False)
    for index in range(len(attempts), 2):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            break
        record = {"identity": identity, "status": "running", "attempts": attempts,
                  "updated_at": now(), "authority": "read_only_review_not_execution"}
        attempts.append({"status": "started", "at": now()})
        atomic_json(path, record)
        try:
            options = {"max_tokens": 1500, "timeout": remaining, "read_only": True}
            if compact:
                options.update(decision_only=True, output_format="json" if index == 0 else "stream-json",
                    auto_skills=False, include_research_context=False)
            content, metadata = client.chat_with_metadata(instructions,
                json.dumps(payload, ensure_ascii=False), **options)
            # Final approval and exact citations are validated by the caller. Cache
            # only complete parseable objects; partial output never becomes approval.
            from .execution_derive import _object
            parsed = _object(content)
            if not isinstance(parsed.get("approved"), bool) or not isinstance(parsed.get("reason"), str):
                raise ValueError("review response requires approved:boolean and reason:string")
            attempts[-1] = {"status": "completed", "turn_id": metadata.get("turn_id"), "at": now()}
            atomic_json(path, {**record, "status": "completed", "attempts": attempts,
                "content": content, "updated_at": now()})
            return content, {**metadata, "review_identity": identity, "cache_hit": False}
        except AgentRuntimeClientError as exc:
            if exc.failure_category not in {"stream_protocol", "missing_final_result",
                    "missing_result_or_nonzero_exit", "provider_transport", "wall_timeout"}:
                raise
            attempts[-1] = {"status": "transport_failed", "category": exc.failure_category,
                "evidence_ref": exc.runtime_failure.get("evidence_ref"), "at": now()}
            atomic_json(path, {**record, "status": "transport_interrupted", "attempts": attempts,
                "runtime_failure": exc.runtime_failure, "updated_at": now()})
            if index == 0 and compact and deadline-time.monotonic() > 1:
                continue
            atomic_json(path, {**record, "status": "transport_unavailable", "attempts": attempts,
                "runtime_failure": exc.runtime_failure, "updated_at": now()})
            raise
    atomic_json(path, {"identity": identity, "status": "transport_unavailable",
        "attempts": attempts, "updated_at": now()})
    raise AgentRuntimeClientError("read-only review local window exhausted; no native operation launched",
                                  failure_category="review_transport")


def record_review_validation(output, identity, result):
    """Keep model response completion separate from trusted citation validation."""
    import re
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("invalid review validation identity")
    root = Path(output) / "review_validations"
    path = root / f"{identity}.json"
    if root.is_symlink() or path.is_symlink():
        raise ValueError("unsafe review validation path")
    atomic_json(path, {"identity": identity, "updated_at": now(),
        "status": "validated" if result.get("approved") is True else "validation_rejected",
        "model_approved": result.get("model_approved"), "reason": result.get("reason"),
        "validation_errors": result.get("validation_errors", []),
        "authority": "citation_validation_only_not_native_recovery"})
    return result


def review_view(output):
    root = Path(output)/"review_transactions"
    if not root.is_dir() or root.is_symlink():
        return []
    rows = []
    for path in sorted(root.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:4]:
        if path.is_symlink() or path.stat().st_size > 65536:
            continue
        try:
            record = read_json(path)
            if not isinstance(record, dict):
                raise ValueError("non-object review record")
            validation_path = Path(output) / "review_validations" / path.name
            validation = (read_json(validation_path) if validation_path.is_file()
                          and not validation_path.is_symlink() else {})
            if validation and validation.get("identity") != record.get("identity"):
                raise ValueError("review validation identity mismatch")
            rows.append({"identity": record.get("identity"),
                "status": validation.get("status", record.get("status")),
                "model_response_status": record.get("status"),
                "validation_errors": validation.get("validation_errors", []),
                "validation_reason": validation.get("reason"),
                "attempts": record.get("attempts", []),
                "record_ref": f"review_transactions/{path.name}",
                "authority": "review_only_not_native_install_or_score"})
        except (ValueError, OSError):
            rows.append({"status": "unreadable", "record_ref": f"review_transactions/{path.name}"})
    return rows
