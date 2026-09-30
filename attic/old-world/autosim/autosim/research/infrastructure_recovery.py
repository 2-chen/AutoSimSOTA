"""API-selected operational recovery, constrained by the existing probe attempt budget."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .common import atomic_json, object_digest, read_json
from .compute_agent import ComputeAgent, client_from_file
from .incidents import IncidentStore, build_incident, evidence_view, redact_text


def choose_probe_recovery(*, incident: dict, output: Path,
                          current_construction_concurrency: int, max_workers: int,
                          deadline_epoch: float | None = None, remaining_seconds: float | None = None,
                          env_file: Path | str | None = None,
                          max_calls: int = 8, max_tokens: int = 80000) -> dict:
    """Choose only the construction order/cache of an already-admitted next attempt.

    Caller must have reaped the failed generation and must still enforce its own
    total attempt/deadline limits. This function cannot extend them or change GPUs.
    """
    remaining = deadline_epoch - time.time() if deadline_epoch is not None else float(remaining_seconds or 0)
    fallback = {"action": "restart_generation", "next_construction_concurrency": current_construction_concurrency,
                "fresh_cache": True, "api_used": False, "status": "deterministic_fallback"}
    root = Path(output) / "infrastructure_recovery"
    root.mkdir(parents=True, exist_ok=True)
    keys = {"returncode", "signal", "phase", "failure_kind", "worker", "worker_id", "error_type",
            "generation", "attempt_id", "episode_started", "retryable", "primary_failure",
            "startup_receipt", "termination", "startup", "last_phase", "error",
            "traceback_excerpt", "cleanup_confirmed", "remaining_attempts", "reset_started"}
    def safe(value, depth=0):
        if depth > 4:
            return None
        if isinstance(value, dict):
            return {k: safe(v, depth+1) for k, v in value.items() if k in keys}
        if isinstance(value, str):
            return redact_text(value, limit=1000)
        return value if value is None or type(value) in (int, float, bool) else None
    snapshot = {"failure": safe(incident), "current_construction_concurrency": current_construction_concurrency,
                "allocated_workers": max_workers,
                "constraints": "A new attempt is already budget-admitted. Keep every worker, seed, model and physics unchanged."}
    incident_id = object_digest(snapshot)
    receipt = root / f"{incident_id}.json"
    if receipt.exists():
        return read_json(receipt)
    if remaining < 185:
        atomic_json(receipt, {**fallback, "reason": "insufficient remaining budget for API"})
        return read_json(receipt)
    source = Path(env_file or os.environ.get("AUTOSIM_ENV_FILE", ""))
    if not source.is_file():
        atomic_json(receipt, {**fallback, "reason": "no configured API credential source"})
        return read_json(receipt)
    api_root = Path(os.environ.get("AUTOSIM_RUN_ROOT", str(output))) / "recovery/api"
    try:
        agent = ComputeAgent(api_root, client_from_file(source), max_calls=max_calls, max_tokens=max_tokens)
        response = agent.request("native_startup_recovery", {"incident_id": incident_id, **snapshot},
            output_tokens=1500, system=(
                "You diagnose a native simulation startup failure after all failed peers were cleaned up. "
                "Return JSON with exactly incident_id, action, construction_concurrency, hypothesis. "
                "Copy incident_id. action is restart_generation or stop. construction_concurrency is an integer "
                "between 1 and current_construction_concurrency; one is serial construction followed by fully "
                "concurrent rollout. Choose a bounded useful next startup strategy; never change GPU count, "
                "seed, model, physics, scientific results, deadline or number of attempts. "
                "Native C++ cause is unproven; hypothesis must not claim it is established."))
        value = json.loads(response["content"])
        if (not isinstance(value, dict) or set(value) != {"incident_id", "action", "construction_concurrency", "hypothesis"}
                or value["incident_id"] != incident_id or value["action"] not in {"restart_generation", "stop"}
                or type(value["construction_concurrency"]) is not int
                or not 1 <= value["construction_concurrency"] <= min(current_construction_concurrency, max_workers)
                or not isinstance(value["hypothesis"], str) or len(value["hypothesis"]) > 2000):
            raise ValueError("invalid infrastructure recovery proposal")
        result = {"action": value["action"], "next_construction_concurrency": value["construction_concurrency"],
                  "fresh_cache": True, "api_used": True, "status": "validated",
                  "hypothesis": redact_text(value["hypothesis"]), "request_id": response["request_id"],
                  "incident_id": incident_id}
    except Exception as exc:
        result = {**fallback, "reason": type(exc).__name__, "incident_id": incident_id}
    atomic_json(receipt, result)
    return result


def supervisor_incident(run_root: Path, *, run_id: str, stage: str, exception: Exception,
                        source_revision: str, environment_fingerprint: str,
                        deadline_epoch: float, final_opened: bool) -> dict:
    """Discover only process failure roots and let the incident builder seal purposes."""
    purpose = ("final_confirmation" if final_opened or "final" in stage else
               "selection_validation" if "selection" in stage else
               "smoke" if "probe" in stage or "initialize" in stage else "development")
    roots = {}
    search = Path(run_root) / "device_probe" if "probe" in stage or "initializ" in stage else Path(run_root) / "evaluations"
    if search.exists():
        for index, path in enumerate(sorted(search.rglob("process.json"))[-128:]):
            if path.parent.name == "process":
                roots[f"worker_{index}"] = path.parent.parent
    result = build_incident(run_id=run_id, stage=stage, purpose=purpose, worker_roots=roots,
        source_revision=source_revision, environment_fingerprint=environment_fingerprint,
        budget={"remaining_wall_seconds": max(0, deadline_epoch-time.time()), "deadline_epoch": deadline_epoch},
        exception=exception, final_opened=final_opened)
    store = IncidentStore(Path(run_root) / "infrastructure/incidents")
    result = store.snapshot(result)
    path = store.root / result["incident_id"]
    return {"incident": result, "evidence": evidence_view(result), "path": str(path)}
