"""Pre-launch evidence and a narrow, reversible Agent-directed workspace recovery.

The recovery Agent sees an empty read-only workspace and sealed evidence, never the
blocked checkout. It may select generated venv IDs; the executor rechecks provenance
and moves them to quarantine. It cannot delete files or loosen the credential guard.
"""
from __future__ import annotations

import json
import os
import traceback
import uuid
from pathlib import Path

from .common import atomic_json, atomic_text, now, object_digest, read_json, sanitize_model_text, digest
from .evidence_store import capture_attempt_evidence


def seal_failure(output: Path, workspace: Path, exc: Exception, *, role: str,
                 decision_attempt_id: str | None) -> dict:
    output = output.resolve(strict=True)
    identity = uuid.uuid4().hex
    directory = output / "agent" / "startup_failures"
    if (directory.is_symlink() or (output / "agent").is_symlink() or
            (output / "evidence").is_symlink()):
        raise ValueError("startup evidence directory is unsafe")
    directory.mkdir(parents=True, exist_ok=True)
    entries = getattr(exc, "entries", [])
    category = "workspace_guard" if entries else type(exc).__name__
    message = sanitize_model_text(str(exc), local_roots=(workspace, output))
    detail = sanitize_model_text("".join(traceback.format_exception(exc)),
                                 local_roots=(workspace, output))
    record = {"schema_version": 1, "id": identity, "role": role,
              "category": category, "message": message, "entries": entries,
              "decision_attempt_id": decision_attempt_id, "at": now(),
              "launched": False if category == "workspace_guard" else None}
    identities = []
    for entry in entries:
        path = workspace / entry["path"]
        stat = path.lstat()
        identities.append([entry["path"], stat.st_dev, stat.st_ino, stat.st_mtime_ns,
                           stat.st_size])
    context = {}
    if category != "workspace_guard":
        for path in (output / "native_context.json", output / "settings.json",
                     Path(__file__).with_name("agent_runtime.py")):
            if path.is_file() and not path.is_symlink():
                context[path.name] = (read_json(path).get("identity") if path.name == "native_context.json"
                                      else digest(path))
    record["fingerprint"] = object_digest({"category": category, "message": message,
                                          "entries": entries, "identities": identities,
                                          **({"execution_context": context} if context else {})})
    log = directory / (identity + ".log")
    atomic_text(log, detail)
    relative = str((directory / (identity + ".json")).relative_to(output))
    record.update(capture_attempt_evidence(output, attempt_id=identity, log=log,
        receipt_ref=relative, status="startup_failed", returncode=None,
        termination_reason=category))
    atomic_json(output / relative, record)
    atomic_json(output / "agent/runtime_failure.json", record)
    return record


def _generated_venv(output: Path, workspace: Path, relative: str) -> dict:
    """Fail closed: complete initial inventory is required to prove generated provenance."""
    manifest = read_json(output / "workspace_snapshot.json")
    if (manifest.get("copy_mode") != "tracked_worktree" or
            Path(manifest.get("destination", "")).resolve() != workspace or
            not manifest.get("source_entries")):
        raise ValueError("no complete original source inventory for quarantine")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 1:
        raise ValueError("quarantine is restricted to top-level generated diagnostic envs")
    path = workspace / rel
    if path.is_symlink() or not path.is_dir():
        raise ValueError("generated environment is missing or linked")
    if any(Path(e[1]) == rel or rel in Path(e[1]).parents
           for e in manifest["source_entries"] if len(e) > 1):
        raise ValueError("original source cannot be quarantined")
    cfg = path / "pyvenv.cfg"
    if cfg.is_symlink() or not cfg.is_file() or cfg.stat().st_size > 16384:
        raise ValueError("not a generated Python virtual environment")
    # Only interpreter links into standard system prefixes are repairable here. Other
    # links/credentials are not touched and never exposed to the recovery Agent.
    from .agent_runtime import AgentWorkspaceBlocked, _validate_agent_workspace
    try:
        _validate_agent_workspace(path)
    except AgentWorkspaceBlocked as exc:
        for entry in exc.entries:
            link = path / entry["path"]
            if (entry["reason"] != "external_link" or
                    link.parent != path / "bin" or not link.name.startswith("python") or
                    not link.resolve(strict=True).is_relative_to(Path("/usr/bin"))):
                raise ValueError("environment contains non-interpreter unsafe entries") from exc
    stat = path.stat()
    return {"path": relative, "device": stat.st_dev, "inode": stat.st_ino}


def recover(client, failure: dict, *, timeout: float) -> dict:
    output, workspace = client.output, client.workspace
    if failure.get("failure_category") != "workspace_guard":
        return {"status": "unsupported"}
    # The latest-fault pointer is a presentation projection, not recovery identity.
    # Concurrent Recorder/worker failures must not replace the Scheduler's evidence.
    identity = str(failure.get("runtime_failure", {}).get("id") or
                   failure.get("failure_evidence_id") or failure.get("evidence_id") or "")
    if not identity:
        identity = str((failure.get("runtime_failure") or {}).get("evidence_id") or "")
    if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("recovery requires an immutable startup failure id")
    path = output / "agent/startup_failures" / (identity + ".json")
    if path.is_symlink():
        raise ValueError("startup failure record is a symlink")
    fault = read_json(path)
    if fault.get("fingerprint") != failure.get("failure_fingerprint"):
        raise ValueError("runtime fault changed before recovery")
    from .native_jobs import active_jobs
    from .agent_tasks import records, snapshot
    if active_jobs(output) or any(r.get("status") in {"submitted", "running", "outcome_unknown"}
                                 for r in records(output)):
        return {"status": "blocked", "reason": "active or unresolved jobs hold the checkout"}
    candidates = []
    for name in sorted({Path(e["path"]).parts[0] for e in fault.get("entries", [])}):
        try:
            row = _generated_venv(output, workspace, name)
        except (OSError, ValueError, RuntimeError):
            continue
        candidates.append({**row, "id": object_digest(row)[:16]})
    identity = uuid.uuid4().hex
    root, clean = snapshot(output, workspace, identity, empty=True)
    child = client.fork_readonly(output=root, workspace=clean, role="fix")
    content, _metadata = child.chat_with_metadata(
        "You are a read-only runtime recovery Agent. The normal checkout is blocked. "
        "Read the sealed evidence by evidence_id. You cannot inspect that checkout or "
        "execute commands. Choose only executor-approved generated diagnostic environment "
        "IDs to move out of source into reversible quarantine, or choose none. Never "
        "request disabling guards or moving original source, data, credentials or weights. "
        'Return JSON {"quarantine_ids":[],"reason":"简体中文、基于证据的说明"}.',
        json.dumps({"fault": fault, "approved_candidates": candidates,
                    "evidence_id": fault["evidence_id"]}, ensure_ascii=False),
        timeout=timeout, read_only=True, include_research_context=False)
    from .execution_derive import _object
    answer = _object(content)
    requested = answer.get("quarantine_ids")
    approved = {row["id"]: row for row in candidates}
    if (not isinstance(requested, list) or len(requested) > len(candidates) or
            any(not isinstance(k, str) or k not in approved for k in requested) or
            len(set(requested)) != len(requested) or
            not isinstance(answer.get("reason"), str) or not answer["reason"].strip()):
        raise ValueError("recovery proposal failed its quarantine contract")
    receipt = {"id": identity, "status": "proposed", "at": now(),
               "fault_fingerprint": fault["fingerprint"], "answer": answer,
               "worker_ref": str(root.relative_to(output)), "moved": []}
    if active_jobs(output) or any(r.get("status") in {"submitted", "running", "outcome_unknown"}
                                 for r in records(output)):
        return {"status": "blocked", "reason": "job ownership changed during recovery",
                "worker_ref": str(root.relative_to(output))}
    if (output / "workspace_snapshot.json").is_symlink():
        raise ValueError("source inventory changed to a symlink")
    target = output / "diagnostics/quarantine" / identity
    if (output / "diagnostics").is_symlink() or target.parent.is_symlink():
        raise ValueError("quarantine directory is unsafe")
    target.mkdir(parents=True, exist_ok=False)
    receipt_path = target / "recovery.json"
    atomic_json(receipt_path, receipt)
    for key in requested:
        row = approved[key]
        if _generated_venv(output, workspace, row["path"]) != {
                k: row[k] for k in ("path", "device", "inode")}:
            raise ValueError("quarantine source identity changed")
        source = workspace / row["path"]
        destination = target / row["path"]
        # Persist intention before the reversible filesystem side effect. An interrupted
        # move must remain inspectable rather than being mistaken for no operation.
        receipt["pending_move"] = {"from": row["path"], "to": str(destination.relative_to(output)),
                                   "device": row["device"], "inode": row["inode"]}
        atomic_json(receipt_path, receipt)
        os.rename(source, destination)
        receipt["moved"].append({"from": row["path"],
            "to": str(destination.relative_to(output)), "device": row["device"], "inode": row["inode"]})
        receipt.pop("pending_move", None)
        atomic_json(receipt_path, receipt)
    from .agent_runtime import _validate_agent_workspace
    try:
        _validate_agent_workspace(workspace)
        receipt["status"] = "revalidated"
    except RuntimeError as exc:
        receipt.update(status="blocked", reason=sanitize_model_text(str(exc)))
    receipt["receipt_ref"] = str(receipt_path.relative_to(output))
    atomic_json(receipt_path, receipt)
    return receipt
