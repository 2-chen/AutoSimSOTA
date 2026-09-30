"""Local authenticated native-probe generations; artifacts are evidence, not IPC."""
from __future__ import annotations

import json
import os
import secrets
import signal
import socketserver
import threading
import time
import uuid
from pathlib import Path

from .common import atomic_json, event
from .native_lifecycle import process_identity
from .probe_barrier_contract import ready_members_match


def _descends_from(pid: int, parent: int) -> bool:
    for _ in range(32):
        if pid == parent:
            return True
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            return False
        if pid <= 1:
            return False
    return False


def group_alive(pgid: int) -> bool:
    """Ignore zombies, but include descendants whose group leader has exited."""
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            fields = (item / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


class ProbeCoordinator:
    """One generation owns two barriers, cancellation, and construction admission."""

    def __init__(self, output: Path, workers: list[str], *, timeout_seconds=900,
                 construction_concurrency: int | None = None, minimum_overlap_seconds=5,
                 termination_grace_seconds=10, attempt_id=1):
        if not workers or len(set(workers)) != len(workers):
            raise ValueError("probe needs unique worker names")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.workers = list(workers)
        self.generation = uuid.uuid4().hex
        self.token = secrets.token_hex(32)
        self.owner = os.getpid()
        self.attempt_id = str(attempt_id)
        self.timeout = float(timeout_seconds)
        self.deadline = time.monotonic() + self.timeout
        self.limit = len(workers) if construction_concurrency is None else int(construction_concurrency)
        if not 1 <= self.limit <= len(workers):
            raise ValueError("invalid construction concurrency")
        self.minimum_overlap = float(minimum_overlap_seconds)
        if self.minimum_overlap < 0 or self.timeout <= 0:
            raise ValueError("invalid probe timing contract")
        self.grace = float(termination_grace_seconds)
        self.lock = threading.RLock()
        self.bindings: dict[str, str] = {}
        self.claims: dict[str, str] = {}
        self.rows: dict[str, dict] = {}
        self.constructing: set[str] = set()
        self.environment_ready: set[str] = set()
        self.policy_ready: set[str] = set()
        self.cancelled: dict | None = None
        self.terminated: set[int] = set()
        self.stop = threading.Event()
        coordinator = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(3)
                try:
                    raw = self.rfile.readline(65537)
                    if len(raw) > 65536:
                        raise ValueError("oversized request")
                    response = coordinator.receive(json.loads(raw))
                except (ValueError, TypeError, KeyError, OSError):
                    response = {"state": "rejected", "reason": "invalid_lifecycle_message"}
                try:
                    self.wfile.write(json.dumps(response).encode() + b"\n")
                except OSError:
                    pass

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = False
            def handle_error(self, request, client_address):
                pass

        self.server = Server(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .1}, daemon=True)
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self._write()

    @property
    def configuration(self) -> dict:
        return {"address": list(self.server.server_address), "token": self.token,
                "generation": self.generation, "timeout_seconds": self.timeout,
                "attempt_id": self.attempt_id,
                "recovery_owner": "coordinator"}

    def worker_configuration(self, worker: str) -> dict:
        return {**self.configuration, "claim": self.claims[worker],
                "worker": worker, "uuid": self.bindings[worker]}

    def bind(self, worker: str, gpu_uuid: str) -> None:
        with self.lock:
            if worker not in self.workers or worker in self.bindings:
                raise ValueError("worker already bound or not expected")
            if gpu_uuid in self.bindings.values():
                raise ValueError("duplicate probe GPU UUID")
            self.bindings[worker] = gpu_uuid
            self.claims[worker] = secrets.token_hex(16)
            self._write()

    def _write(self):
        atomic_json(self.output / "generation.json", {"schema_version": 1, "generation": self.generation,
                    "owner_pid": self.owner, "workers": self.workers, "bindings": self.bindings,
                    "construction_concurrency": self.limit, "rows": self.rows,
                    "environment_ready": sorted(self.environment_ready), "policy_ready": sorted(self.policy_ready),
                    "cancelled": self.cancelled, "minimum_overlap_seconds": self.minimum_overlap})

    def cancel(self, worker: str | None, reason: str) -> None:
        with self.lock:
            if self.cancelled is None:
                self.cancelled = {"worker": worker, "reason": reason, "monotonic": time.monotonic()}
                event(self.output / "events.jsonl", "generation_cancelled", generation=self.generation,
                      primary=self.cancelled)
                self._write()

    def receive(self, message: dict) -> dict:
        with self.lock:
            worker = message.get("worker")
            if (message.get("generation") != self.generation
                    or not secrets.compare_digest(str(message.get("token", "")), self.token)
                    or worker not in self.bindings or message.get("uuid") != self.bindings[worker]
                    or message.get("claim") != self.claims.get(worker)
                    or message.get("attempt_id") != self.attempt_id
                    or message.get("host") != os.uname().nodename):
                return {"state": "rejected", "reason": "generation_or_membership_mismatch"}
            pid = int(message["pid"])
            identity = process_identity(pid)
            if (identity is None or identity != message.get("process_start_identity")
                    or int(message.get("pgid", 0)) != pid or os.getpgid(pid) != pid
                    or not _descends_from(pid, self.owner)):
                return {"state": "rejected", "reason": "process_identity_mismatch"}
            row = self.rows.get(worker)
            if row and (row["pid"] != pid or row["process_start_identity"] != identity):
                return {"state": "rejected", "reason": "worker_claim_already_owned"}
            if row is None:
                row = {"worker": worker, "uuid": message["uuid"], "pid": pid, "pgid": pid,
                       "host": os.uname().nodename,
                       "claim": self.claims[worker], "attempt_id": self.attempt_id,
                       "process_start_identity": identity, "phase": "created", "reset_started": False,
                       "generation": self.generation, "work_started": None, "work_ended": None, "steps": 0}
                self.rows[worker] = row
            row["heartbeat"] = time.monotonic()
            if self.cancelled:
                return {"state": "cancelled", "reason": "peer_failed", "primary": self.cancelled}
            action = message["action"]
            if action == "phase":
                transitions = {"created": {"starting"}, "starting": {"constructing"},
                    "constructing": {"ready"}, "ready": {"running", "work_finished"},
                    "running": {"work_finished"}, "work_finished": {"completed"},
                    "completed": set(), "failed": set(), "cancelled": set()}
                phase = message.get("phase")
                if (phase not in transitions or (phase != row["phase"] and
                        phase not in transitions[row["phase"]] and phase not in {"failed", "cancelled"})):
                    self.cancel(worker, "invalid_lifecycle_transition")
                    return {"state": "rejected", "reason": "invalid_lifecycle_transition"}
                if row.get("reset_started") and message.get("reset_started") is not True:
                    self.cancel(worker, "reset_state_regressed")
                    return {"state": "rejected", "reason": "reset_state_regressed"}
                for key in ("phase", "reset_started", "work_started", "work_ended", "steps"):
                    if key in message:
                        row[key] = message[key]
                event(self.output / "events.jsonl", "worker_phase", **row)
                self._write()
            elif action == "failure":
                self.cancel(worker, str(message.get("reason") or "worker_failed"))
                return {"state": "cancelled", "reason": "peer_failed", "primary": self.cancelled}
            elif action == "construction":
                if worker in self.constructing or worker in self.environment_ready:
                    return {"state": "released"}
                if len(self.constructing) < self.limit:
                    self.constructing.add(worker)
                    return {"state": "released"}
                return {"state": "waiting"}
            elif action in {"environment_ready", "policy_ready"}:
                if row["phase"] not in {"ready", "running", "work_finished", "completed"}:
                    return {"state": "rejected", "reason": "worker_not_constructed"}
                ready = self.environment_ready if action == "environment_ready" else self.policy_ready
                if action == "policy_ready" and len(self.environment_ready) != len(self.workers):
                    return {"state": "rejected", "reason": "environment_barrier_not_released"}
                if worker not in ready:
                    ready.add(worker)
                    if action == "environment_ready":
                        self.constructing.discard(worker)
                    self._write()
                # Revalidate all process identities at release, not just the latest sender.
                if len(ready) == len(self.workers):
                    dead = next((name for name in ready if process_identity(self.rows[name]["pid"])
                                 != self.rows[name]["process_start_identity"]), None)
                    if dead:
                        self.cancel(dead, "worker_exited")
                        return {"state": "cancelled", "reason": "peer_failed", "primary": self.cancelled}
                    fields = ("worker", "uuid", "generation", "claim", "attempt_id")
                    expected = [{"worker": name, "uuid": self.bindings[name], "generation": self.generation,
                                 "claim": self.claims[name], "attempt_id": self.attempt_id} for name in self.workers]
                    members = [{**{key: self.rows[name][key] for key in fields}, "state": "ready",
                                "alive": process_identity(self.rows[name]["pid"]) ==
                                         self.rows[name]["process_start_identity"]} for name in ready]
                    if not ready_members_match(expected, members, self.generation):
                        self.cancel(None, "barrier_membership_rejected")
                        return {"state": "cancelled", "reason": "barrier_membership_rejected"}
                    return {"state": "released"}
                return {"state": "waiting"}
            elif action != "heartbeat":
                return {"state": "rejected", "reason": "unknown_lifecycle_action"}
            return {"state": "active"}

    def _monitor(self):
        while not self.stop.wait(.1):
            with self.lock:
                if self.cancelled is None:
                    for worker, row in self.rows.items():
                        if row["phase"] == "completed":
                            continue
                        if process_identity(row["pid"]) != row["process_start_identity"]:
                            self.cancel(worker, "worker_exited")
                            break
                    if time.monotonic() >= self.deadline:
                        waiting = [name for name in self.workers if name not in self.environment_ready]
                        if not waiting:
                            waiting = [name for name in self.workers if name not in self.policy_ready]
                        self.cancel(waiting[0] if waiting else None,
                                    "native_startup_timeout" if waiting else "native_execution_timeout")
                cancelled = self.cancelled
                rows = list(self.rows.values())
            if cancelled:
                elapsed = time.monotonic() - cancelled["monotonic"]
                for row in rows:
                    # Each registered worker owns a new session; group identity is frozen.
                    # A disappeared leader may still have descendants that require cleanup.
                    pgid = row["pgid"]
                    current_identity = process_identity(row["pid"])
                    if current_identity is not None and current_identity != row["process_start_identity"]:
                        # Never signal a recycled PID/group; cleanup remains unconfirmed.
                        continue
                    if elapsed >= .5 and pgid not in self.terminated:
                        try:
                            os.killpg(pgid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        self.terminated.add(pgid)
                    if elapsed >= self.grace + .5:
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def overlap_receipt(self) -> dict:
        with self.lock:
            rows = [dict(self.rows[name]) for name in self.workers if name in self.rows]
            valid = (self.cancelled is None and len(rows) == len(self.workers)
                     and len({row["uuid"] for row in rows}) == len(self.workers)
                     and all(row["phase"] == "completed" and row.get("steps", 0) > 0
                             and row.get("work_started") is not None and row.get("work_ended") is not None
                             for row in rows))
            seconds = (min(row["work_ended"] for row in rows) - max(row["work_started"] for row in rows)
                       if valid else 0.0)
            return {"generation": self.generation, "passed": valid and seconds >= self.minimum_overlap,
                    "overlap_seconds": max(0.0, seconds), "minimum_overlap_seconds": self.minimum_overlap,
                    "verified_rollout_concurrency": len(rows) if valid and seconds >= self.minimum_overlap else 0,
                    "construction_concurrency": self.limit, "workers": rows, "primary_failure": self.cancelled}

    def __enter__(self):
        self.thread.start()
        self.monitor.start()
        return self

    def __exit__(self, kind, error, traceback):
        if error:
            self.cancel(None, type(error).__name__)
        if self.cancelled:
            end = time.monotonic() + self.grace + 5
            while time.monotonic() < end and any(group_alive(row["pgid"]) for row in self.rows.values()):
                time.sleep(.1)
        alive = [row["worker"] for row in self.rows.values() if group_alive(row["pgid"])]
        atomic_json(self.output / "cleanup.json", {"generation": self.generation,
                    "confirmed_reaped": not alive, "unreaped_workers": alive})
        self.stop.set()
        self.monitor.join(timeout=2)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self._write()
        if alive:
            raise RuntimeError(f"native generation cleanup unconfirmed: {alive}")
