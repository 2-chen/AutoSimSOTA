"""Causal infrastructure incidents and purpose-limited views for repair tools.

Raw artifacts stay local.  Only explicitly supplied worker directories are read;
this module never scans a research run or its sealed evaluation banks.
"""
from __future__ import annotations

import json
import re
import signal
from pathlib import Path
from typing import Mapping

from .common import atomic_json, digest, exclusive, object_digest, read_json

SEALED_PURPOSES = {"selection", "selection_validation", "final", "final_confirmation"}
_SECRET = re.compile(
    r"(?i)(?:[A-Z_]*(?:API[_-]?KEY|AUTH[_-]?TOKEN|ACCESS[_-]?TOKEN|PASSWORD|SECRET)[A-Z_]*"
    r"[\"']?\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)|"
    r"(?:bearer\s+)[^\s,;]+|(?:sk-|hf_)[A-Za-z0-9_-]{8,}")
_SENSITIVE_PATH = re.compile(r"(?i)(?:^|[/\\_.-])(?:sealed|selection|final(?:_confirmation)?|credentials?)(?:$|[/\\_.-])")


def redact_text(value: str, *, limit: int = 12000) -> str:
    """Remove credential assignments/tokens, URL credentials and control chars."""
    value = _SECRET.sub("[REDACTED]", str(value))
    value = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    return value[:limit]


def permitted_relative_path(value: str) -> str:
    path = Path(value)
    if (not isinstance(value, str) or not value or path.is_absolute() or
            any(part in {"", ".", ".."} for part in path.parts) or
            any(part.startswith(".") for part in path.parts) or
            _SENSITIVE_PATH.search(value) or path.suffix.lower() in {".env", ".pem", ".key"}):
        raise PermissionError("path is outside the repair evidence scope")
    return path.as_posix()


def _read(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 262144:
        return {}
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _artifact(root: Path, relative: str) -> Path | None:
    path = root / relative
    if (not path.is_file() or path.is_symlink() or
            not path.resolve().is_relative_to(root.resolve())):
        return None
    return path


def _log_excerpt(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        lines = stream.read(32768).decode("utf-8", errors="replace").splitlines()
    # Tracebacks include enough neighboring source context without exporting the
    # unfiltered environment dump, observations, scores or entire worker stdout.
    selected = set()
    for i, line in enumerate(lines):
        if any(token in line for token in ("Traceback", "Fatal Python", 'File "', "SIGSEGV",
                                           "Segmentation fault", "Error:", "Exception:",
                                           "load_urdf", "native workers did not", "peer_failed")):
            selected.update(range(max(0, i - 1), min(len(lines), i + 3)))
    return redact_text("\n".join(lines[i] for i in sorted(selected)))


def classify_failure(*, returncode: int | None, phase: str, message: str,
                     episode_started: bool) -> str:
    lower = message.lower()
    if "peer_failed" in lower or "peer failed" in lower:
        return "peer_failed"
    if "ready barrier" in lower or "barrier" in lower and "timeout" in lower:
        return "barrier_timeout"
    if "out of memory" in lower:
        return "out_of_memory"
    if "deadline" in lower or "budget" in lower:
        return "budget_or_deadline"
    if "non-finite" in lower or "nonfinite" in lower:
        return "nonfinite_state"
    if returncode in (-signal.SIGSEGV, -signal.SIGABRT):
        return "native_runtime_abort" if episode_started else "native_startup_abort"
    if "no such file" in lower or "modulenotfounderror" in lower:
        return "missing_dependency"
    if returncode not in (0, None):
        return "worker_failure"
    return "unclassified"


def build_incident(*, run_id: str, stage: str, purpose: str,
                   worker_roots: Mapping[str, Path], source_revision: str = "unknown",
                   environment_fingerprint: str = "unknown", budget: dict | None = None,
                   exception: BaseException | str | None = None, generation: str | int | None = None,
                   final_opened: bool = False, prior_actions: list[dict] | None = None) -> dict:
    """Preserve the first native cause and label peer/cleanup failures secondary."""
    if len(worker_roots) > 128:
        raise ValueError("too many incident workers")
    workers, references = [], {}
    sealed = bool(final_opened) or purpose in SEALED_PURPOSES or any(_SENSITIVE_PATH.search(str(root)) for root in worker_roots.values())
    for name, raw_root in sorted(worker_roots.items()):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name):
            raise ValueError("invalid worker evidence identifier")
        root = Path(raw_root).absolute()
        process_path = _artifact(root, "process/process.json")
        startup_path = _artifact(root, "startup.json")
        failure_path = _artifact(root, "worker_failure.json")
        process = _read(process_path) if process_path else {}
        startup = _read(startup_path) if startup_path else {}
        failure = _read(failure_path) if failure_path else {}
        episode_started = bool(startup.get("policy_has_acted") or startup.get("episode_started") or
                               startup.get("phase") in {"episode_running", "rollout_running", "episode_finished"} or
                               (root / "reset_started.json").is_file() or
                               any((root / name).is_file() and (root / name).stat().st_size > 0
                                   for name in ("initializations.jsonl", "episodes.jsonl")))
        phase = str(startup.get("phase", "unknown"))[:100]
        # Even for sealed purposes, classification is local and safe error codes
        # survive.  No message/traceback crosses the purpose boundary after reset.
        message = str(failure.get("error", ""))
        log_path = _artifact(root, "process/stdout.log")
        excerpt = _log_excerpt(log_path) if log_path and not sealed else ""
        kind = classify_failure(returncode=process.get("returncode"), phase=phase,
                                message=message + "\n" + excerpt, episode_started=episode_started)
        failed = bool(failure or process.get("returncode") not in (0, None) or
                      process.get("status") in {"failed", "interrupted", "failed_to_start"})
        row = {"worker_id": name, "failed": failed, "failure_kind": kind if failed else None,
               "returncode": process.get("returncode"), "phase": phase,
               "episode_started": episode_started, "gpu_uuid": process.get("device_uuid"),
               "pid": process.get("worker_pid", process.get("pid")),
               "process_start_identity": process.get("process_start_identity"),
               "started_at": process.get("started_at"), "finished_at": process.get("finished_at"),
               "evidence_refs": []}
        for relative in ("process/process.json", "startup.json", "worker_failure.json", "process/stdout.log"):
            path = _artifact(root, relative)
            if path:
                key = f"{name}:{relative}"
                references[key] = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
                row["evidence_refs"].append(key)
        if not sealed:
            row["error_excerpt"] = redact_text(message, limit=2000)
            row["traceback_excerpt"] = excerpt
        workers.append(row)
    failed = [row for row in workers if row["failed"]]
    # A peer timeout and a cleanup crash must not replace an earlier construction
    # abort.  Timestamps establish ordering among independent primary candidates.
    causes = [row for row in failed if row["failure_kind"] not in {"peer_failed", "barrier_timeout"}]
    primary = min(causes or failed, key=lambda row: (row.get("finished_at") or "9999", row["worker_id"])) if failed else None
    failure_kind = primary["failure_kind"] if primary else classify_failure(
        returncode=None, phase=stage, message=str(exception or ""), episode_started=False)
    exception_type = type(exception).__name__ if isinstance(exception, BaseException) else None
    supervisor_error = redact_text(str(exception or ""), limit=2000) if not sealed else None
    identity = {"run_id": run_id, "stage": stage, "purpose": purpose, "generation": generation,
                "source_revision": source_revision, "environment_fingerprint": environment_fingerprint,
                "workers": workers, "failure_kind": failure_kind,
                "exception_type": exception_type, "supervisor_error": supervisor_error,
                "final_opened": bool(final_opened)}
    incident_id = object_digest(identity)
    for row in workers:
        row["causal_role"] = "primary" if row is primary else "secondary" if row["failed"] else "peer"
        row["primary_incident_id"] = incident_id if row["failed"] and row is not primary else None
    budget_keys = {"remaining_wall_seconds", "remaining_gpu_hours", "remaining_api_calls", "remaining_api_tokens",
                   "deadline_epoch", "charged_gpu_hours"}
    safe_budget = {k: v for k, v in (budget or {}).items()
                   if k in budget_keys and (v is None or type(v) in (int, float))}
    return {"schema_version": 1, "incident_id": incident_id, **identity,
            "primary_worker_id": primary["worker_id"] if primary else None,
            "artifact_references": references, "budget": safe_budget, "final_opened": bool(final_opened),
            "prior_actions": list(prior_actions or [])[-8:],
            "causal_limit": "Observed process failure and location; underlying native root cause is not proven."}


def evidence_view(incident: dict) -> dict:
    """The API gets field allowlists, never arbitrary nested run metadata."""
    allowed = {"schema_version", "incident_id", "run_id", "stage", "purpose", "generation",
               "source_revision", "environment_fingerprint", "failure_kind", "primary_worker_id",
               "exception_type", "budget", "final_opened", "causal_limit"}
    view = {key: incident[key] for key in allowed if key in incident}
    sealed = bool(incident.get("final_opened")) or incident.get("purpose") in SEALED_PURPOSES
    worker_keys = {"worker_id", "failed", "failure_kind", "returncode", "phase", "episode_started",
                   "gpu_uuid", "pid", "started_at", "finished_at", "causal_role", "primary_incident_id",
                   "evidence_refs"}
    if not sealed:
        worker_keys |= {"error_excerpt", "traceback_excerpt"}
        view["supervisor_error"] = redact_text(incident.get("supervisor_error") or "", limit=2000)
    view["workers"] = [{key: redact_text(value) if isinstance(value, str) else value
                        for key, value in row.items() if key in worker_keys}
                       for row in incident.get("workers", [])[:128]]
    view["artifact_references"] = {key: {"sha256": row["sha256"], "bytes": row["bytes"]}
                                   for key, row in incident.get("artifact_references", {}).items()}
    # Prior actions can contain Agent-supplied strings; export only authoritative
    # identifiers and terminal statuses, not arbitrary rationales or tool payloads.
    view["prior_actions"] = [{k: row[k] for k in ("action_id", "action", "status", "candidate_id") if k in row}
                             for row in incident.get("prior_actions", [])[-8:]]
    return view


class IncidentStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def record(self, incident: dict) -> Path:
        """Keep first evidence immutable; subsequent budget observations cannot reset it."""
        identity = incident["incident_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", identity):
            raise ValueError("invalid incident identity")
        destination = self.root / identity
        with exclusive(self.root / "incidents.lock"):
            path = destination / "incident.json"
            if path.exists():
                existing = read_json(path)
                stable = lambda row: {key: value for key, value in row.items() if key not in {"budget", "prior_actions"}}
                if object_digest(stable(existing)) != object_digest(stable(incident)):
                    raise ValueError("incident changed under the same identity")
            else:
                atomic_json(path, incident)
                atomic_json(destination / "evidence_view.json", evidence_view(incident))
            observation = {key: incident.get(key) for key in ("budget", "prior_actions")}
            atomic_json(destination / "observations" / (object_digest(observation) + ".json"), observation)
            budget_path = destination / "budget_state.json"
            state = read_json(budget_path) if budget_path.exists() else {}
            deadline = incident.get("budget", {}).get("deadline_epoch")
            if type(deadline) in (int, float):
                state["deadline_epoch"] = min(state.get("deadline_epoch", deadline), deadline)
            atomic_json(budget_path, state)
        return destination

    def snapshot(self, incident: dict) -> dict:
        """Return the durable first API view, with budget changes journaled separately.

        Live admission uses the broker/global ledger, not the snapshot's historical
        budget. Rebuilding an incident must not alter an in-flight API request key.
        """
        return read_json(self.record(incident) / "incident.json")
