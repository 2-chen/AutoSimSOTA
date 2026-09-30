"""Persistent, bounded API requests for compute planning and code proposals."""
from __future__ import annotations

import json
from pathlib import Path

from autosim.llm_client import LLMClient
from .common import atomic_json, exclusive, now, object_digest, read_json
from .compute_proposals import LIMITS, validate_proposal


def client_from_file(path: Path) -> LLMClient:
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise PermissionError("API credential file must be private")
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    def get(suffix):
        return next((values[prefix + suffix] for prefix in
                     ("DEEPSEEK_", "AUTOSIM_LLM_", "OPENAI_") if values.get(prefix + suffix)), None)
    if not get("API_KEY"):
        raise ValueError("configured credential file has no API key")
    return LLMClient(api_key=get("API_KEY"), base_url=get("BASE_URL"), model=get("MODEL"))


class ComputeAgent:
    def __init__(self, root: Path, client: LLMClient, *, max_calls=40, max_tokens=200000):
        if type(max_calls) is not int or type(max_tokens) is not int or min(max_calls, max_tokens) < 0:
            raise ValueError("compute API limits must be nonnegative integers")
        self.root, self.client = Path(root), client
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.max_tokens = max_calls, max_tokens

    def request(self, role: str, context: dict, *, system: str, output_tokens=3000,
                thinking="disabled") -> dict:
        identity = {"role": role, "context": context, "system": system,
                    "output_tokens": output_tokens, "thinking": thinking,
                    "model": self.client.model, "endpoint": self.client.base_url}
        key = object_digest(identity)
        destination = self.root / key
        with exclusive(self.root / "requests.lock"):
            ledger_path = self.root / "budget.json"
            ledger = read_json(ledger_path) if ledger_path.exists() else {
                "calls": 0, "charged_tokens": 0, "reserved_tokens": 0, "requests": []}
            original = ledger.get("limits", {})
            ledger["limits"] = {"max_calls": min(self.max_calls, original.get("max_calls", self.max_calls)),
                                "max_tokens": min(self.max_tokens, original.get("max_tokens", self.max_tokens))}
            atomic_json(ledger_path, ledger)
            if (destination / "response.json").exists():
                response = read_json(destination / "response.json")
                ledger_path = self.root / "budget.json"
                ledger = read_json(ledger_path)
                row = next(r for r in ledger["requests"] if r["request_id"] == key)
                if row["status"] != "completed":
                    used = (response.get("metadata", {}).get("usage") or {}).get("total_tokens")
                    ledger["charged_tokens"] += int(used or row["reserved_tokens"])
                    ledger["reserved_tokens"] -= row["reserved_tokens"]
                    row["status"] = "completed"
                    atomic_json(ledger_path, ledger)
                return response
            if (destination / "request.json").exists():
                raise RuntimeError("unresolved API request; reconcile before making another charged attempt")
            ledger_path = self.root / "budget.json"
            ledger = read_json(ledger_path) if ledger_path.exists() else {
                "calls": 0, "charged_tokens": 0, "reserved_tokens": 0, "requests": []}
            # UTF-8 bytes are a conservative local token admission bound; provider usage settles it.
            user = json.dumps(context, ensure_ascii=False)
            estimate = len((system + user).encode()) + output_tokens + 1024
            if ledger["calls"] >= ledger["limits"]["max_calls"] or (
                    ledger["charged_tokens"] + ledger["reserved_tokens"] + estimate > ledger["limits"]["max_tokens"]):
                raise RuntimeError("compute API budget exhausted")
            ledger["calls"] += 1
            ledger["reserved_tokens"] += estimate
            ledger["requests"].append({"request_id": key, "reserved_tokens": estimate, "status": "pending"})
            atomic_json(ledger_path, ledger)
            atomic_json(destination / "request.json", {"request_id": key, **identity, "started": now()})
            try:
                content, metadata = self.client.chat_with_metadata(
                    system, user, max_tokens=output_tokens, timeout=180, retries=0, thinking=thinking)
                response = {"request_id": key, "content": content, "metadata": metadata, "finished": now()}
                atomic_json(destination / "response.json", response)
            except Exception as exc:
                # Preserve conservative charge when the remote outcome is unknown; never save echoed bodies.
                safe_reason = str(exc) if str(exc).startswith("LLM API ") else type(exc).__name__
                atomic_json(destination / "failure.json", {"exception_type": type(exc).__name__,
                                                            "safe_reason": safe_reason, "time": now()})
                ledger["requests"][-1]["status"] = "unknown_remote_outcome"
                atomic_json(ledger_path, ledger)
                raise RuntimeError(f"compute API request failed: {type(exc).__name__}") from None
            usage = metadata.get("usage") or {}
            ledger["charged_tokens"] += int(usage.get("total_tokens") or estimate)
            ledger["reserved_tokens"] -= estimate
            ledger["requests"][-1]["status"] = "completed"
            atomic_json(ledger_path, ledger)
            return response

    def plan(self, snapshot: dict, limits: dict) -> dict:
        def paths(value, prefix=""):
            if isinstance(value, dict):
                return [p for k, v in value.items() for p in paths(v, f"{prefix}.{k}".strip("."))]
            if isinstance(value, list):
                return [p for i, v in enumerate(value) for p in paths(v, f"{prefix}.{i}")]
            return [prefix]
        context = {"snapshot": snapshot, "snapshot_digest": object_digest(snapshot),
                   "allowed_evidence_refs": paths(snapshot),
                   "parameter_bounds": {k: [a, min(b, limits.get(k, b))] for k, (a, b) in LIMITS.items()}}
        system = (
            "Analyze only the measured compute evidence. Return JSON with schema_version=1, "
            "based_on_snapshot copied from snapshot_digest, bottleneck, evidence_refs naming actual "
            "snapshot fields copied EXACTLY from allowed_evidence_refs (no equals sign or values), "
            "and actions (1..8). Each action has exactly type="
            "set_execution_parameter, parameter, integer value within the supplied bounds. "
            "Never change data, model, precision, evaluation protocol or budget. No shell commands.")
        response = self.request("compute_planner", context, system=system)
        for attempt in range(2):
            try:
                return self._validate_plan(response["content"], context, limits)
            except (ValueError, TypeError, KeyError) as exc:
                rejection = {"request_id":response["request_id"],"attempt":attempt,
                             "error_type":type(exc).__name__,"error":str(exc)[:300]}
                atomic_json(self.root / "rejections" / (object_digest(rejection)+".json"), rejection)
                if attempt:
                    raise
                response = self.request("compute_planner_repair", {**context,
                    "rejected_response":response["content"][:8000],"validation_error":rejection["error"]},
                    system=system + " Correct the rejected response once. schema_version must be the JSON integer 1, not a string or boolean.")

    @staticmethod
    def _validate_plan(content: str, context: dict, limits: dict) -> dict:
        snapshot = context["snapshot"]
        proposal = validate_proposal(json.loads(content), snapshot_digest=context["snapshot_digest"], limits=limits)
        for ref in proposal["evidence_refs"]:
            if not isinstance(ref, str):
                raise ValueError("evidence reference must be a snapshot field path")
            path = ref.removeprefix("snapshot.").replace("[", ".").replace("]", "").split(".")
            value = snapshot
            try:
                for part in path:
                    value = value[int(part)] if isinstance(value, list) else value[part]
            except (KeyError, IndexError, ValueError, TypeError):
                raise ValueError(f"unmeasured evidence reference: {ref}") from None
        return proposal
