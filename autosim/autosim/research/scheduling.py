"""Opt-in scheduling policy and durable admission, independent of research decisions.

The LLM decides tasks and priorities. This layer enforces ownership, declared capacity,
and finite waits. Reservations are cooperative, not a replacement for OS isolation.
"""
from __future__ import annotations

import fcntl
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from .common import atomic_json, now, read_json
from .host_resources import capacity
from .process_executor import capture_process_identity, inspect_process_identity

DEFAULTS = {"schema_version": 1, "readonly_workers": 2, "model_slots": 3,
            "async_recorder": True, "native_slots": 4, "cpu_fraction": .75,
            "memory_fraction": .8, "wait_seconds": 30, "resource_poll_seconds": 2}


def configure(output: Path) -> dict:
    path = Path(output) / "scheduler_policy.json"
    if path.is_file():
        return read_json(path)
    atomic_json(path, {**DEFAULTS, "created_at": now()})
    return read_json(path)


def policy(output: Path) -> dict:
    path = Path(output) / "scheduler_policy.json"
    return read_json(path) if path.is_file() else {}


@contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def model_slot(output: Path, *, timeout: float, foreground: bool = True):
    settings = policy(output)
    if not settings:
        yield
        return
    root = Path(output) / "agent" / "admission"
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    admitted = None
    try:
        while admitted is None:
            # Keep one slot for the main controller; background narration/investigations
            # cannot occupy every provider slot and delay the next research decision.
            slots = int(settings["model_slots"])
            indexes = range(slots) if foreground else range(1, slots)
            for index in indexes:
                fd = os.open(root / f"{index}.lock", os.O_CREAT | os.O_RDWR |
                             getattr(os, "O_NOFOLLOW", 0), 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    admitted = fd
                    break
                except BlockingIOError:
                    os.close(fd)
            if admitted is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("model admission deadline reached; no provider call started")
                time.sleep(.1)
        yield
    finally:
        if admitted is not None:
            os.close(admitted)


def resource_profile(value: dict | None, *, gpu: bool = False) -> dict:
    raw = {} if value is None else value
    if not isinstance(raw, dict) or set(raw) - {"cpu", "memory_mib", "gpu", "priority"}:
        raise ValueError("resources admits cpu, memory_mib, gpu, priority only")
    answer = {"cpu": raw.get("cpu", 2), "memory_mib": raw.get("memory_mib", 4096),
              "gpu": raw.get("gpu", gpu), "priority": raw.get("priority", 0)}
    for key in ("cpu", "memory_mib", "priority"):
        if not isinstance(answer[key], int) or isinstance(answer[key], bool):
            raise ValueError("resource counts must be integers")
    if not 1 <= answer["cpu"] <= 256 or not 1 <= answer["memory_mib"] <= 2**24:
        raise ValueError("resource request exceeds supported bounds")
    if not -10 <= answer["priority"] <= 10 or not isinstance(answer["gpu"], bool):
        raise ValueError("invalid resource priority or GPU request")
    return answer


class HostLease:
    """Cross-run CPU/memory admission with strong PID reconciliation."""
    def __init__(self, identity: str, resources: dict, *, directory: Path | None = None,
                 limits: dict | None = None):
        self.resources = resource_profile(resources)
        self.identity = identity
        self.limits = {key: (limits or DEFAULTS).get(key, DEFAULTS[key])
                       for key in ("cpu_fraction", "memory_fraction")}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1
               for v in self.limits.values()):
            raise ValueError("invalid host admission fractions")
        self.root = directory or Path(f"/tmp/autosimsota-host-{os.getuid()}")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (self.root.is_symlink() or self.root.stat().st_uid != os.getuid()
                or self.root.stat().st_mode & 0o077):
            raise ValueError("unsafe host admission directory")
        self.path = self.root / "leases.json"
        self.held = False
        self.cores = []

    def acquire(self) -> bool:
        facts = capacity()
        try:
            namespace = os.readlink("/proc/self/ns/pid")
        except OSError:
            namespace = ""
        if facts["memory"]["available_mib"] is None or facts["memory"]["total_mib"] is None:
            raise ValueError("host memory capacity is unknown; cannot admit declared resources")
        cpu = max(1, int(facts["cpu"]["effective_cpus"] * self.limits["cpu_fraction"]))
        memory = int(facts["memory"]["available_mib"] * self.limits["memory_fraction"])
        # Live available memory excludes resident jobs already. Also use a stable total
        # ceiling for declared reservations, so these are not subtracted twice.
        memory_cap = int(facts["memory"]["total_mib"] * self.limits["memory_fraction"])
        if self.resources["cpu"] > cpu or self.resources["memory_mib"] > memory_cap:
            raise ValueError("resource request cannot fit host capacity")
        with locked(self.root / "leases.lock"):
            rows = read_json(self.path) if self.path.is_file() else {}
            rows = {key: row for key, row in rows.items()
                    if not namespace or row.get("pid_namespace") != namespace or
                       inspect_process_identity(row["process"]).get("status") != "not_running"}
            # Different runs may request stricter ceilings. Apply the strictest live
            # reservation; never let a permissive newcomer invalidate a neighbor's cap.
            cpu = min([cpu, *(max(1, int(facts["cpu"]["effective_cpus"] *
                row.get("limits", DEFAULTS)["cpu_fraction"])) for row in rows.values())])
            memory_cap = min([memory_cap, *(int(facts["memory"]["total_mib"] *
                row.get("limits", DEFAULTS)["memory_fraction"]) for row in rows.values())])
            if self.identity in rows:
                self.held = True
                self.cores = rows[self.identity].get("cores") or []
                return True
            used_cpu = sum(row["resources"]["cpu"] for row in rows.values())
            used_memory = sum(row["resources"]["memory_mib"] for row in rows.values())
            if (used_cpu + self.resources["cpu"] > cpu or
                    used_memory + self.resources["memory_mib"] > memory_cap or
                    self.resources["memory_mib"] > memory):
                atomic_json(self.path, rows)
                return False
            process = capture_process_identity(SimpleNamespace(pid=os.getpid(), poll=lambda: None),
                run_id="host-admission", attempt_id=self.identity, argv=[])
            assigned = {core for row in rows.values() for core in row.get("cores") or []}
            allowed = facts["cpu"].get("affinity") or list(range(int(facts["cpu"]["effective_cpus"])))
            self.cores = [core for core in allowed if core not in assigned][:self.resources["cpu"]]
            if len(self.cores) < self.resources["cpu"]:
                atomic_json(self.path, rows)
                return False
            rows[self.identity] = {"resources": self.resources, "process": process,
                                   "cores": self.cores, "limits": self.limits,
                                   "pid_namespace": namespace}
            atomic_json(self.path, rows)
            self.held = True
            return True

    def release(self):
        if self.held:
            with locked(self.root / "leases.lock"):
                rows = read_json(self.path)
                rows.pop(self.identity, None)
                atomic_json(self.path, rows)
            self.held = False


def note(output: Path, *, kind: str, identity: str, seconds: float, **facts):
    path = Path(output) / "scheduling_metrics.json"
    with locked(path.with_suffix(".lock")):
        doc = read_json(path) if path.is_file() else {"schema_version": 1, "events": []}
        aggregates = doc.setdefault("aggregates", {})
        aggregate = aggregates.setdefault(kind, {"count": 0, "seconds": 0.0})
        aggregate["count"] += 1
        aggregate["seconds"] += max(0.0, float(seconds))
        doc["events"] = [*doc["events"], {"kind": kind, "id": identity,
            "seconds": seconds, "at": now(), **facts}][-2000:]
        atomic_json(path, doc)


def view(output: Path) -> dict:
    settings = policy(output)
    if not settings:
        return {}
    facts = capacity()
    return {"policy": settings, "host_capacity": facts,
            "metrics_ref": "scheduling_metrics.json", "failure_ref": "scheduling_failures.json",
            "resource_contract": "CPU affinity/native thread caps; cooperative memory reservation, not an RSS hard limit"}
