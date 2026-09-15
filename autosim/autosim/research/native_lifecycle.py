"""Native worker identity, progress and bounded startup recovery contract.

The lifecycle observes execution; it never changes an episode, action or seed.
Synchronized probes have one recovery owner (their coordinator), while isolated
Runtime workers use the same pre-reset retry predicate locally.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

from .common import atomic_json, event, read_json


class PeerCancelled(RuntimeError):
    """The coordinator revoked this generation before it could commit."""


def process_identity(pid: int) -> str | None:
    """Linux process start ticks distinguish a reused PID from its old owner."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except (OSError, ValueError, IndexError):
        return None


def retryable_startup(directory: Path) -> bool:
    """Only positively identified native failures before the first reset."""
    directory = Path(directory)
    process, startup = directory / "process/process.json", directory / "startup.json"
    if not process.is_file() or not startup.is_file():
        return False
    record = read_json(process)
    native = record.get("status") == "failed" and record.get("returncode") in {-11, -6}
    hang = (record.get("status") == "interrupted" and record.get("error") == "TimeoutExpired"
            and record.get("returncode") in {-15, -9})
    reset = directory / "reset_started.json"
    return (bool(native or hang)
            and read_json(startup).get("phase") in {"environment_constructing", "environment_ready",
                                                   "native_barrier_released"}
            and not reset.exists()
            and not (directory / "initializations.jsonl").exists()
            and not (directory / "evaluation_metrics.json").exists())


def startup_attempt_limit(requested: int, config: dict | None = None, *, coordinated=False) -> int:
    """Apply one attempt budget; group recovery never nests worker retries."""
    if type(requested) is not int or not 1 <= requested <= 3:
        raise ValueError("startup_attempts must be in [1, 3]")
    settings = (config or {}).get("lifecycle", config or {})
    configured = settings.get("startup_attempts", requested)
    if type(configured) is not int or not 1 <= configured <= 3:
        raise ValueError("harness startup_attempts must be in [1, 3]")
    return 1 if coordinated else min(requested, configured)


class NativeLifecycle:
    """One real process, optionally participating in an authenticated local wave."""

    def __init__(self, output: Path, *, config: dict | None = None):
        self.output = Path(output)
        self.config = config if config is not None else json.loads(
            os.environ.get("AUTOSIM_NATIVE_COORDINATOR", "{}"))
        self.identity = {"worker": self.config.get("worker", os.environ.get("AUTOSIM_JOB", "worker")),
                         "uuid": self.config.get("uuid"), "generation": self.config.get("generation"),
                         "claim": self.config.get("claim"), "attempt_id": self.config.get("attempt_id"),
                         "pid": os.getpid(), "process_start_identity": process_identity(os.getpid()),
                         "pgid": os.getpgrp(), "host": os.uname().nodename}
        self.phase = "created"
        self.reset_started = False
        self.steps = 0
        self.work_started = None
        self.work_ended = None
        self.cancelled: str | None = None
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.heartbeat: threading.Thread | None = None

    def _request(self, action: str, **values) -> dict:
        if not self.config:
            return {"state": "released", "construction_allowed": True}
        payload = {**self.identity, "action": action, "token": self.config["token"], **values}
        host, port = self.config["address"]
        try:
            with socket.create_connection((host, int(port)), timeout=3) as connection:
                connection.settimeout(3)
                connection.sendall(json.dumps(payload).encode() + b"\n")
                answer = connection.makefile("rb").readline(65537)
            result = json.loads(answer)
        except (OSError, ValueError) as exc:
            self.cancelled = "coordinator_unavailable"
            raise PeerCancelled(self.cancelled) from exc
        if result.get("state") in {"cancelled", "rejected"}:
            self.cancelled = str(result.get("reason") or "peer_failed")
            raise PeerCancelled(self.cancelled)
        return result

    def record(self, phase: str, **values) -> None:
        with self.lock:
            self.phase = phase
            row = {**self.identity, "phase": phase, "monotonic": time.monotonic(),
                   "reset_started": self.reset_started, "steps": self.steps,
                   "work_started": self.work_started, "work_ended": self.work_ended, **values}
            atomic_json(self.output / "lifecycle.json", row)
            event(self.output / "lifecycle_events.jsonl", "native_lifecycle", **row)
        self._request("phase", **{k: v for k, v in row.items() if k not in self.identity})

    def start(self) -> None:
        self.record("starting")
        if self.config:
            def beat():
                while not self.stop.wait(.5):
                    try:
                        self._request("heartbeat")
                    except PeerCancelled:
                        return
            self.heartbeat = threading.Thread(target=beat, daemon=True)
            self.heartbeat.start()

    def _wait(self, action: str) -> None:
        if not self.config:
            return
        deadline = time.monotonic() + float(self.config.get("timeout_seconds", 900))
        while True:
            if self.cancelled:
                raise PeerCancelled(self.cancelled)
            response = self._request(action)
            if response.get("state") == "released":
                return
            if time.monotonic() >= deadline:
                self._request("failure", reason="native_startup_timeout")
                raise TimeoutError("native coordinator startup deadline exhausted")
            self.stop.wait(.1)

    def constructing(self) -> None:
        self._wait("construction")
        self.record("constructing")

    def ready(self) -> None:
        self.record("ready")
        self._wait("environment_ready")

    def before_reset(self) -> None:
        if not self.reset_started:
            self._wait("policy_ready")
            # This is durable BEFORE calling native reset, even if reset itself crashes.
            self.reset_started = True
            atomic_json(self.output / "reset_started.json", {**self.identity, "monotonic": time.monotonic()})
            self.record("running")

    def step_started(self) -> None:
        if self.cancelled:
            raise PeerCancelled(self.cancelled)
        if self.work_started is None:
            self.work_started = time.monotonic()
            self.record("running")

    def step_completed(self) -> None:
        self.steps += 1
        self.work_ended = time.monotonic()

    def work_finished(self) -> None:
        # Window ends at the last actual step, excluding close/cleanup/barrier waiting.
        self.record("work_finished")

    def completed(self) -> None:
        self.record("completed")
        self.close()

    def failed(self, error: BaseException) -> None:
        try:
            self.record("cancelled" if isinstance(error, PeerCancelled) else "failed",
                        error_type=type(error).__name__)
            self._request("failure", reason=type(error).__name__)
        except PeerCancelled:
            pass
        self.close()

    def close(self) -> None:
        self.stop.set()
        if self.heartbeat and self.heartbeat is not threading.current_thread():
            self.heartbeat.join(timeout=3.5)
