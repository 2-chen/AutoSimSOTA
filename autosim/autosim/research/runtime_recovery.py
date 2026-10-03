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


def seal_protocol_stream(output: Path, workspace: Path, *, turn_id: str, attempt,
                         secrets=()) -> dict:
    """Keep bounded raw protocol evidence before lossy event projection."""
    directory = output / "agent" / "stream_failures"
    from .agent_runtime import _worker_evidence_root
    evidence_root = _worker_evidence_root(output)
    identity = uuid.uuid4().hex
    stdout, stderr = str(attempt.stdout or ""), str(attempt.stderr or "")
    for secret in secrets:
        if secret:
            stdout, stderr = stdout.replace(secret, "[REDACTED]"), stderr.replace(secret, "[REDACTED]")
    record = {"turn_id": turn_id, "returncode": attempt.returncode,
        "timed_out": attempt.timed_out,
        "stdout_bytes": len(str(attempt.stdout or "").encode("utf-8")),
        "stderr_bytes": len(str(attempt.stderr or "").encode("utf-8")),
        "stdout_newline_terminated": not stdout or stdout.endswith("\n"),
        "stdout": stdout, "stderr": stderr, "authority": "protocol_diagnostic_not_action"}
    log = directory / (identity + ".log")
    atomic_text(log, sanitize_model_text(json.dumps(record, ensure_ascii=False),
                                        local_roots=(workspace, output)))
    ref = (directory / f"{identity}.json").relative_to(evidence_root).as_posix()
    sealed = capture_attempt_evidence(evidence_root, attempt_id=identity, log=log,
        receipt_ref=ref, status="protocol_diagnostic", returncode=attempt.returncode,
        termination_reason="failed_model_turn")
    atomic_json(evidence_root / ref, {**sealed, **{k:v for k,v in record.items() if k not in {"stdout", "stderr"}}})
    return sealed


def seal_turn_failure(output: Path, workspace: Path, result: dict, *, role: str,
                      timeout: float) -> dict:
    """Keep interrupted-model evidence distinct from a failed native environment probe.

    Partial/final provider text is diagnostic only and is never adopted as an action.
    The stable log lets Fix see tool progress, the local timeout and terminal process
    facts instead of guessing that a simulator/configuration operation timed out.
    """
    output = output.resolve(strict=True)
    identity = uuid.uuid4().hex
    directory = output / "agent" / "turn_failures"
    if directory.is_symlink() or directory.parent.is_symlink():
        raise ValueError("turn failure evidence directory is unsafe")
    turn_id = str(result.get("turn_id") or "")
    process_ref = str(result.get("process_ref") or "")
    trace = []
    events = output / "agent/events.jsonl"
    if events.is_file() and not events.is_symlink() and events.stat().st_size <= 32 * 1024**2:
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                # A killed writer may leave an incomplete final JSONL record.
                continue
            if not isinstance(event, dict):
                continue
            if event.get("turn_id") == turn_id and turn_id:
                trace.append({k: event.get(k) for k in ("at", "type", "tool", "status", "summary")})
    category = str(result.get("failure_category") or "provider_or_tool_error")
    message = f"{role} provider turn {result.get('status')}: {category}; native readiness not evaluated"
    record = {"schema_version": 1, "id": identity, "role": role, "category": category,
              "message": message, "at": now(), "provider_launched": True,
              "native_operation_status": "not_inferred", "turn_id": turn_id,
              "process_ref": process_ref, "timeout_seconds": timeout,
              "provider_duration_ms": result.get("provider_duration_ms"),
              "provider_final_response_present": bool(result.get("final_text")),
              "stream_evidence": result.get("stream_evidence"),
              "decision_attempt_id": result.get("decision_attempt_id")}
    record["fingerprint"] = object_digest({"role": role, "category": category,
        "timeout_seconds": timeout, "runtime": digest(Path(__file__).with_name("agent_runtime.py"))})
    detail = {**record, "returncode": result.get("returncode"),
              "stream_evidence": result.get("stream_evidence"),
              "timed_out": result.get("timed_out"), "recent_turn_events": trace[-20:],
              "unadopted_provider_response": str(result.get("final_text") or "")[:24000],
              "guidance": "Inspect provider progress and timeout first. This is not a failed "
                  "native build/reset/rollout. Do not patch benchmark configuration merely "
                  "because a planning turn timed out. Reuse source findings; request a "
                  "compact executable plan under the configured window."}
    log = directory / f"{identity}.log"
    atomic_text(log, sanitize_model_text(json.dumps(detail, ensure_ascii=False),
                                        local_roots=(workspace, output)))
    ref = f"agent/turn_failures/{identity}.json"
    record.update(capture_attempt_evidence(output, attempt_id=identity, log=log,
        receipt_ref=ref, status="provider_turn_failed", returncode=result.get("returncode"),
        termination_reason=category))
    atomic_json(output / ref, record)
    atomic_json(output / "agent/runtime_failure.json", record)
    return record


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


def _generated_interpreter_link(output: Path, workspace: Path, relative: str) -> dict:
    """Quarantine only a new link to the currently bound native interpreter.

    This never allows the link through the workspace guard. The target remains
    untouched and the original source inventory must prove the link is generated.
    """
    import re
    manifest_path = output / 'workspace_snapshot.json'
    context_path = output / 'native_context.json'
    if manifest_path.is_symlink() or context_path.is_symlink():
        raise ValueError('linked recovery provenance')
    manifest = read_json(manifest_path)
    rel = Path(relative)
    if (manifest.get('copy_mode') != 'tracked_worktree' or
            Path(manifest.get('destination', '')).resolve() != workspace or
            not manifest.get('source_entries') or rel.is_absolute() or '..' in rel.parts or
            not re.fullmatch(r'python(?:\d+(?:\.\d+)?)?', rel.name)):
        raise ValueError('unproven generated interpreter link')
    if any(Path(e[1]) == rel or Path(e[1]) in rel.parents
           for e in manifest['source_entries'] if len(e) > 1):
        raise ValueError('original source cannot be quarantined')
    path = workspace / rel
    if not path.is_symlink() or any(p.is_symlink() for p in path.parents if p != workspace and workspace in p.parents):
        raise ValueError('not a direct generated link')
    interpreter = Path(read_json(context_path).get('interpreter') or '')
    if (not interpreter.is_absolute() or not interpreter.is_file() or
            path.resolve(strict=True) != interpreter.resolve(strict=True)):
        raise ValueError('link does not name the bound native interpreter')
    stat = path.lstat()
    return {'path': relative, 'device': stat.st_dev, 'inode': stat.st_ino,
            'kind': 'generated_interpreter_link', 'target_sha256': digest(interpreter)}


def _recheck_candidate(output: Path, workspace: Path, row: dict) -> dict:
    if row.get('kind') == 'generated_interpreter_link':
        return _generated_interpreter_link(output, workspace, row['path'])
    return _generated_venv(output, workspace, row['path'])


def view(output: Path) -> dict:
    """Expose verified recovery facts without replaying recovery or claiming readiness."""
    output = Path(output).resolve()
    pointer = output/'runtime_blocker.json'
    if not pointer.is_file() or pointer.is_symlink():
        return {'status': 'not_recorded'}
    try:
        failure = read_json(pointer)
        recovery = failure.get('recovery') or {}
        relative = Path(recovery.get('receipt_ref') or '')
        receipt = output/relative
        if (relative.is_absolute() or '..' in relative.parts or
                relative.parts[:2] != ('diagnostics', 'quarantine') or
                receipt.is_symlink() or not receipt.is_file() or
                not receipt.resolve().is_relative_to(output/'diagnostics/quarantine') or
                receipt.stat().st_size > 1024*1024):
            return {'status': 'unverified', 'guidance': 'inspect sealed runtime failure before inferring recovery'}
        row = read_json(receipt)
        if (row.get('fault_fingerprint') != failure.get('failure_fingerprint') or
                row.get('status') != recovery.get('status') or
                row.get('receipt_ref') != str(relative)):
            return {'status': 'unverified'}
        return {'status': row['status'], 'receipt_ref': str(relative),
                'receipt_sha256': digest(receipt), 'at': row.get('at'),
                'recovered_fault_fingerprint': row['fault_fingerprint'],
                'guidance': 'This is a sealed runtime-guard revalidation, not native readiness. '
                    'Earlier workspace_guard errors are historical, not evidence that every '
                    'Agent role is currently blocked. Guards still run before each Agent turn. '
                    'Research-round and wall-budget limits are separate facts.'}
    except (OSError, ValueError, TypeError, AttributeError):
        return {'status': 'unverified'}


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
    for entry in fault.get('entries', []):
        if entry.get('reason') != 'external_link':
            continue
        relative = entry['path']
        if any(Path(row['path']) in Path(relative).parents for row in candidates):
            continue
        try:
            row = _generated_interpreter_link(output, workspace, relative)
        except (OSError, ValueError, RuntimeError):
            continue
        candidates.append({**row, 'id': object_digest(row)[:16]})
    identity = uuid.uuid4().hex
    root, clean = snapshot(output, workspace, identity, empty=True)
    child = client.fork_readonly(output=root, workspace=clean, role="fix")
    content, _metadata = child.chat_with_metadata(
        "You are a read-only runtime recovery Agent. The normal checkout is blocked. "
        "Read the sealed evidence by evidence_id. You cannot inspect that checkout or "
        "execute commands. Choose only executor-approved generated diagnostic environment "
        "or generated interpreter-link IDs to move into reversible quarantine, or choose none. "
        "Link targets remain untouched. Never "
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
        if _recheck_candidate(output, workspace, row) != {k: v for k, v in row.items() if k != 'id'}:
            raise ValueError("quarantine source identity changed")
        source = workspace / row["path"]
        destination = target / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
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
