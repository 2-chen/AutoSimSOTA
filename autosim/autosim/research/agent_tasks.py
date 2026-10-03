"""Asynchronous read-only investigations; only Scheduler merges their handoffs."""
from __future__ import annotations

import json
import os
import math
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import atomic_json, digest, now, read_json, object_digest
from .main_agent import READ_ONLY_ROLES, TASK_SYSTEM, validate_report, validate_task
from .native_jobs import source_identity
from .scheduling import locked, note, policy

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="autosim-reader")
_futures = {}
_guard = threading.Lock()
MAX_READER_BYTES = 64 * 1024**2
PUBLIC_SOURCE_SUFFIXES = {".py", ".toml", ".md", ".rst", ".sh", ".yaml", ".yml", ".ini", ".cfg", ".lock", ".txt"}


def _materialized_source_files(repo: Path, scope: Path) -> list[Path]:
    from .source_tracking import materialized_files
    return materialized_files(repo, scope)



def _scope_identity(repo: Path, names: list[str]) -> str:
    rows = []
    for name in names:
        relative = Path(name); path = repo / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(repo.resolve()) or not path.is_file()):
            raise ValueError("reader source changed or escaped")
        rows.append((name, digest(path)))
    return object_digest(rows)


def snapshot(output: Path, repo: Path, task_id: str, *, empty: bool = False,
             source_paths: list[str] | None = None) -> tuple[Path, Path]:
    root = Path(output) / "agent_workers" / task_id
    checkout = root / "checkout"
    checkout.mkdir(parents=True)
    if not empty:
        before = source_identity(output, repo)
        manifest = read_json(Path(output) / "workspace_snapshot.json")
        scopes = [Path(item) for item in source_paths] if source_paths is not None else None
        if scopes is not None and (not scopes or len(scopes) > 64 or any(
                path.is_absolute() or ".." in path.parts for path in scopes)):
            raise ValueError("unsafe reader source scope")
        selected = [entry for entry in manifest.get("source_entries") or []
                    if entry[0] == "file" and (scopes is None or any(
                        Path(entry[1]).is_relative_to(scope) for scope in scopes))]
        if scopes is not None:
            for scope in scopes:
                additional = _materialized_source_files(repo, scope)
                if not additional and not any(Path(row[1]).is_relative_to(scope) for row in selected):
                    raise ValueError("source scope contains no public files; use actual checkout-relative source paths")
                selected.extend(("file", str(name)) for name in additional)
        else:
            from .source_tracking import materialized_files
            selected.extend(("file", str(name)) for name in materialized_files(repo))
        selected = list({row[1]: row for row in selected}.values())
        total = 0
        for entry in selected:
            relative = Path(entry[1])
            source = repo / relative
            if (relative.is_absolute() or ".." in relative.parts or source.is_symlink()
                    or not source.resolve().is_relative_to(repo.resolve())):
                raise ValueError("unsafe reader source inventory")
            total += source.stat().st_size
        if total > MAX_READER_BYTES:
            raise ValueError("reader source snapshot exceeds 64 MiB; supply narrower source_paths (files/directories), not just narrower prose")
        source_names = sorted(row[1] for row in selected)
        scope_identity = _scope_identity(repo, source_names)
        for entry in selected:
            if entry[0] != "file":
                continue
            relative = Path(entry[1])
            source = repo / relative
            if (relative.is_absolute() or ".." in relative.parts or source.is_symlink()
                    or not source.resolve().is_relative_to(repo.resolve())):
                raise ValueError("unsafe reader source inventory")
            if not source.is_file():
                continue
            target = checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if (before != source_identity(output, repo) or scope_identity != _scope_identity(repo, source_names)
                or scope_identity != _scope_identity(checkout, source_names)):
            raise ValueError("source changed while making reader snapshot; retry when stable")
    atomic_json(root / "workspace_snapshot.json", {"schema_version": 2,
        "source": str(repo), "destination": str(checkout),
        "reader_source_identity": before if not empty else None,
        "reader_source_files": source_names if not empty else [],
        "reader_scope_identity": scope_identity if not empty else None,
        "source_paths": source_paths, "source_bytes": total if not empty else 0,
        "snapshot_scope": "selected inventoried files" if source_paths else "all inventoried files",
        "note": "read-only inventoried-source snapshot, not a runnable benchmark copy"})
    atomic_json(root / "worker_parent.json", {"output": str(Path(output).resolve()),
        "read_only": True})
    return root, checkout


def records(output: Path) -> list[dict]:
    rows = []
    for path in sorted((Path(output) / "agent_tasks").glob("*/task.json")):
        if not path.is_symlink() and path.resolve().is_relative_to(Path(output).resolve()):
            rows.append(read_json(path))
    return sorted(rows, key=lambda row: (row.get("submitted_at") or "", row["id"]))


def submit(output: Path, repo: Path, client, assignment: dict, facts: dict, timeout: float) -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("read-only task needs positive remaining wall time")
    task = validate_task(assignment, allow_source_paths=True)
    if task["role"] not in READ_ONLY_ROLES:
        raise ValueError("parallel tasks are read-only; edits require exclusive Scheduler operations")
    if not hasattr(client, "fork_readonly") or not policy(output):
        raise ValueError("isolated parallel worker runtime is unavailable")
    with locked(Path(output) / "agent_tasks/submission.lock"):
        active = [r for r in records(output) if r["status"] in {"submitted", "running", "outcome_unknown"}]
        if len(active) >= int(policy(output)["readonly_workers"]):
            raise ValueError("read-only worker slots are occupied")
        task_id = uuid.uuid4().hex
        root, checkout = snapshot(output, repo, task_id, source_paths=task.get("source_paths"))
        identity = source_identity(output, repo)
        if not identity or read_json(root / "workspace_snapshot.json")["reader_source_identity"] != identity:
            raise ValueError("read-only worker needs a stable inventoried source snapshot")
        record = {"id": task_id, **task, "status": "submitted", "submitted_at": now(),
                  "source_identity": identity, "input_state_revision": facts.get("state_revision"),
                  "worker_ref": str(root.relative_to(output)), "cancel_requested": False}
        reader = read_json(root / "workspace_snapshot.json")
        record["reader_source_files"] = reader.get("reader_source_files") or []
        record["reader_scope_identity"] = reader.get("reader_scope_identity")
        path = Path(output) / "agent_tasks" / task_id / "task.json"
        atomic_json(path, record)
        child = client.fork_readonly(output=root, workspace=checkout, role=task["role"])

        def work():
            start = time.monotonic()
            with locked(path.with_suffix(".lock")):
                current = read_json(path)
                if current.get("cancel_requested"):
                    current.update(status="cancelled", finished_at=now())
                    atomic_json(path, current)
                    return
                current.update(status="running", started_at=now())
                atomic_json(path, current)
            try:
                content, metadata = child.chat_with_metadata(TASK_SYSTEM,
                    json.dumps({"assignment": task, "state": facts,
                        "snapshot_scope": "inventoried source only; read native errors by evidence ID"},
                        ensure_ascii=False, default=str), timeout=timeout, read_only=True,
                    include_research_context=False)
                from .main_agent import receive_report
                report = receive_report(content, client=child, output=root, workspace=checkout,
                    timeout=max(1, timeout - (time.monotonic()-start)), metadata=metadata)
                changes = {"status": "completed", "report": report, "runtime": metadata}
            except Exception as exc:
                changes = {"status": "failed", "error": type(exc).__name__ + ": " + str(exc)[:300]}
            with locked(path.with_suffix(".lock")):
                current = read_json(path)
                current.update(changes, finished_at=now())
                if current.get("cancel_requested"):
                    current["status"] = "cancelled"
                atomic_json(path, current)
            note(output, kind="readonly_task", identity=task_id,
                 seconds=time.monotonic()-start, status=current["status"])
        with _guard:
            _futures[(str(output), task_id)] = _pool.submit(work)
    return record


def cancel(output: Path, task_id: str, reason: str) -> dict:
    if len(task_id) != 32 or any(c not in "0123456789abcdef" for c in task_id):
        raise ValueError("unsafe task id")
    path = Path(output) / "agent_tasks" / task_id / "task.json"
    with locked(path.with_suffix(".lock")):
        record = read_json(path)
        if record["status"] not in {"submitted", "running", "outcome_unknown"}:
            raise ValueError("task is not active")
        # Provider calls cannot be safely undone. Discard the result and retain the charge;
        # the bounded turn will finish rather than signalling the controller process group.
        record.update(cancel_requested=True, cancel_reason=reason[:400])
        if record["status"] == "outcome_unknown":
            record.update(status="cancelled", finished_at=now(),
                          warning="Lost worker result discarded; any recorded usage remains charged")
        atomic_json(path, record)
    return record


def collect(output: Path, repo: Path) -> list[dict]:
    ready = []
    identity = None
    for row in records(output):
        path = Path(output) / "agent_tasks" / row["id"] / "task.json"
        key = (str(output), row["id"])
        if row["status"] not in {"submitted", "running"}:
            with _guard:
                future = _futures.get(key)
                if future is not None and future.done():
                    _futures.pop(key, None)
        if row["status"] in {"submitted", "running"}:
            with _guard:
                future = _futures.get(key)
            if future is None:
                with locked(path.with_suffix(".lock")):
                    row = read_json(path)
                    if row["status"] in {"submitted", "running"}:
                        row.update(status="outcome_unknown", error="worker owner restarted; inspect worker receipts")
                        atomic_json(path, row)
        if row["status"] == "completed" and not row.get("collected_at"):
            identity = identity or source_identity(output, repo)
            row.update(stale=row["source_identity"] != identity)
            if row.get("reader_scope_identity"):
                try:
                    row["stale"] |= row["reader_scope_identity"] != _scope_identity(repo, row["reader_source_files"])
                except (OSError, ValueError):
                    row["stale"] = True
            ready.append(row)
    return ready


def acknowledge(output: Path, task_id: str):
    """Ack only after Scheduler persisted the handoff; replay is deduplicated by task ID."""
    path = Path(output) / "agent_tasks" / task_id / "task.json"
    with locked(path.with_suffix(".lock")):
        row = read_json(path)
        if row["status"] == "completed":
            row.update(collected_at=now())
            atomic_json(path, row)
