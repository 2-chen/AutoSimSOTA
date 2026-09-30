"""Asynchronous read-only investigations; only Scheduler merges their handoffs."""
from __future__ import annotations

import json
import math
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import atomic_json, digest, now, read_json
from .main_agent import READ_ONLY_ROLES, TASK_SYSTEM, validate_report, validate_task
from .native_jobs import source_identity
from .scheduling import locked, note, policy

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="autosim-reader")
_futures = {}
_guard = threading.Lock()


def snapshot(output: Path, repo: Path, task_id: str, *, empty: bool = False) -> tuple[Path, Path]:
    root = Path(output) / "agent_workers" / task_id
    checkout = root / "checkout"
    checkout.mkdir(parents=True)
    if not empty:
        before = source_identity(output, repo)
        manifest = read_json(Path(output) / "workspace_snapshot.json")
        total = 0
        for entry in manifest.get("source_entries") or []:
            if entry[0] != "file":
                continue
            relative = Path(entry[1])
            source = repo / relative
            if (relative.is_absolute() or ".." in relative.parts or source.is_symlink()
                    or not source.resolve().is_relative_to(repo.resolve())):
                raise ValueError("unsafe reader source inventory")
            if not source.is_file():
                continue
            total += source.stat().st_size
            if total > 64 * 1024**2:
                raise ValueError("reader source snapshot exceeds 64 MiB; narrow the assignment")
            target = checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if before != source_identity(output, repo):
            raise ValueError("source changed while making reader snapshot; retry when stable")
    atomic_json(root / "workspace_snapshot.json", {"schema_version": 2,
        "source": str(repo), "destination": str(checkout),
        "reader_source_identity": before if not empty else None,
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
    task = validate_task(assignment)
    if task["role"] not in READ_ONLY_ROLES:
        raise ValueError("parallel tasks are read-only; edits require exclusive Scheduler operations")
    if not hasattr(client, "fork_readonly") or not policy(output):
        raise ValueError("isolated parallel worker runtime is unavailable")
    with locked(Path(output) / "agent_tasks/submission.lock"):
        active = [r for r in records(output) if r["status"] in {"submitted", "running", "outcome_unknown"}]
        if len(active) >= int(policy(output)["readonly_workers"]):
            raise ValueError("read-only worker slots are occupied")
        task_id = uuid.uuid4().hex
        root, checkout = snapshot(output, repo, task_id)
        identity = source_identity(output, repo)
        if not identity or read_json(root / "workspace_snapshot.json")["reader_source_identity"] != identity:
            raise ValueError("read-only worker needs a stable inventoried source snapshot")
        record = {"id": task_id, **task, "status": "submitted", "submitted_at": now(),
                  "source_identity": identity, "input_state_revision": facts.get("state_revision"),
                  "worker_ref": str(root.relative_to(output)), "cancel_requested": False}
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
                report = validate_report(json.loads(content))
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
