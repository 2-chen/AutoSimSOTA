"""Durable records, immutable specifications, and bounded subprocess execution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def redact(text: str) -> str:
    return re.sub(r"(?:hf_|sk-)[A-Za-z0-9_-]{16,}", "[REDACTED]", text)


def immutable_json(path: Path, value: Any) -> None:
    """Treat tuples/lists identically after JSON serialization, never redefine inputs."""
    if path.exists():
        if object_digest(read_json(path)) != object_digest(value):
            raise ValueError(f"immutable artifact changed: {path}")
        return
    atomic_json(path, value)


def event(path: Path, kind: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(redact(json.dumps({"time": now(), "event": kind, **payload}, ensure_ascii=False)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def freeze_files(paths: list[Path]) -> dict[str, str]:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen files missing: {missing}")
    return {str(p.absolute()): digest(p) for p in sorted(set(paths))}


def assert_frozen(hashes: dict[str, str]) -> None:
    bad = [p for p, h in hashes.items() if not Path(p).is_file() or digest(Path(p)) != h]
    if bad:
        raise RuntimeError(f"frozen protocol changed: {bad[:10]}")


def run_command(command: list[str], *, cwd: Path, env: dict[str, str],
                output: Path, timeout: int, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Never retry a partial run or infer success merely from exit code zero.

    ``metadata`` (the device binding, the job name) is stamped onto the record and onto
    every partial write of it, but never enters ``request``/``request_hash``: the timeout
    may shrink when a run is resumed, so the request hash already excludes it, and a job
    re-bound to a *different* device is a different fact that must be refused rather than
    silently overwritten.
    """
    metadata = dict(metadata or {})
    collisions = set(metadata) & {"command", "cwd", "timeout", "request_hash", "status",
                                  "started_at", "finished_at", "elapsed_seconds", "pid",
                                  "returncode", "error"}
    if collisions:
        raise ValueError(f"metadata may not shadow process record fields: {sorted(collisions)}")
    output.mkdir(parents=True, exist_ok=True)
    request = {"command": command, "cwd": str(cwd), "timeout": timeout}
    request_hash = object_digest(request)
    result_path = output / "process.json"
    if result_path.exists():
        previous = read_json(result_path)
        if previous["command"] != command or previous["cwd"] != str(cwd):
            raise RuntimeError(f"refusing to reuse output for a different command: {output}")
        conflicts = {key: value for key, value in metadata.items() if previous.get(key, value) != value}
        if conflicts:
            raise RuntimeError(f"refusing to rebind an existing process record: {output} {conflicts}")
        if metadata:
            atomic_json(result_path, {**previous, **metadata})
        if previous["status"] == "completed":
            return {**previous, **metadata}
        raise RuntimeError(f"prior process incomplete/failed; audit and use a new attempt directory: {output}")
    record = {**request, **metadata, "request_hash": request_hash, "started_at": now(), "status": "running"}
    atomic_json(result_path, record)
    started = time.monotonic()
    with (output / "stdout.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            record.update(status="failed_to_start", returncode=None,
                          error=f"{type(exc).__name__}: {exc}", finished_at=now(),
                          elapsed_seconds=time.monotonic() - started)
            atomic_json(result_path, record)
            raise RuntimeError(f"process failed to start: {output / 'stdout.log'}") from exc
        record["pid"] = process.pid
        atomic_json(result_path, record)
        try:
            code = process.wait(timeout=timeout)
            record.update(returncode=code, status="completed" if code == 0 else "failed")
        except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit) as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record.update(returncode=process.returncode, status="interrupted",
                          error=type(exc).__name__)
        finally:
            record.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
            atomic_json(result_path, record)
    if record["status"] != "completed":
        raise RuntimeError(f"process {record['status']} ({record.get('returncode')}): {output / 'stdout.log'}")
    return record
