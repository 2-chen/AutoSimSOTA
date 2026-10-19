"""Content-free cache diagnostics. Never rewrite provider history or save raw prompts."""
from __future__ import annotations

import hashlib
import json


def request_fingerprint(request: dict, *, body_bytes: int) -> dict:
    """Digest the system/tools and initial message; byte counts are NOT token counts.

    No key, path, instruction, tool argument or response content is returned. These
    fingerprints diagnose envelope churn; provider usage alone measures cache hits.
    """
    def encoded(value):
        # Preserve ordering inside arrays and objects: changing tool/schema order can
        # affect the provider prefix even when semantically equivalent.
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()

    def identity(value):
        return hashlib.sha256(encoded(value)).hexdigest()

    system = request.get("system")
    tools = request.get("tools")
    messages = request.get("messages") or []
    envelope = {key: request.get(key) for key in
                ("model", "system", "tools", "tool_choice", "thinking", "output_config")}
    first = messages[0] if isinstance(messages, list) and messages else None
    return {"schema_version": 1, "not_provider_cache_measurement": True,
            "envelope_sha256": identity(envelope), "system_sha256": identity(system),
            "tools_sha256": identity(tools), "first_message_sha256": identity(first),
            "body_bytes": body_bytes, "system_bytes": len(encoded(system)),
            "tools_bytes": len(encoded(tools)), "messages_bytes": len(encoded(messages)),
            "message_count": len(messages) if isinstance(messages, list) else 0,
            "max_output_tokens": request.get("max_tokens")}
