"""Job-level scheduling over a device set: dependency order, one heavy job per device.

The pipeline's rounds are a dependency chain and stay a dependency chain -- multi-device
work never reorders scientific dependencies, it only lets *independent* jobs inside one
round run at the same time (``collect_targeted_r`` beside ``collect_original_r``, the
official evaluation beside the candidate evaluations).  Three rules keep that safe:

* one heavy job per device, and a trainer never shares a device with a simulator -- two
  environments on one card is how a "parallel" run turns into a corrupted measurement;
* a job starts only if both budgets afford its estimate *plus* the reserved verification
  cost, so a parallel job can never eat the wall clock needed to check what it produced;
* every decision is written down: ``outcome.json`` / ``failure.json`` per job and a
  ``schedule.json`` snapshot that names, for each job not started, the reason it waits.

Resume is idempotent because it is derived from those same files: a job with an outcome is
done and is never re-run, a job with a failure stays failed (nothing score-bearing is
retried), and a job with a partial directory is simply pending again.
"""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .accounting import CATEGORIES, Charge
from .common import atomic_json, now, read_json
from .leases import device_leases, occupancy

# A device runs at most one of these at a time; "api" jobs are light and may share.
HEAVY_CATEGORIES = ("training", "collection", "evaluation", "audit")
SIMULATION_CATEGORIES = ("collection", "evaluation", "audit")


class ScheduleRefused(RuntimeError):
    """A job cannot be placed, and the message says which rule stopped it."""


@dataclass(frozen=True)
class Job:
    """One unit the pipeline may run; ``name`` is its identity in every record."""

    name: str
    category: str
    kind: str = ""
    depends_on: tuple[str, ...] = ()
    device_count: int = 1
    estimate_seconds: float = 0.0
    reserve_seconds: float = 0.0
    priority: int = 0

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown accounting category: {self.category}")
        if self.device_count < 1:
            raise ValueError("a job needs at least one device")

    @property
    def heavy(self) -> bool:
        return self.category in HEAVY_CATEGORIES

    def as_dict(self) -> dict:
        record = asdict(self)
        record["depends_on"] = list(self.depends_on)
        return record


@dataclass(frozen=True)
class Assignment:
    job: Job
    devices: tuple[dict, ...]


@dataclass(frozen=True)
class Waiting:
    job: Job
    reason: str
    detail: str


@dataclass
class _Running:
    job: Job
    devices: tuple[dict, ...]
    started: float
    started_at: str
    stack: ExitStack
    leases: list = field(default_factory=list)
    waiting_seconds: float = 0.0


class Scheduler:
    """The device set, the job graph and the two budgets, in one decision point."""

    def __init__(self, *, jobs: Sequence[Job], devices: Sequence[Mapping[str, Any]],
                 output: Path, ledger=None, max_parallel_jobs: int | None = None,
                 run_id: str = "run", lease_root: Path | None = None,
                 clock=time.monotonic, use_leases: bool = True):
        self.jobs = {job.name: job for job in jobs}
        if len(self.jobs) != len(list(jobs)):
            raise ValueError("duplicate job name in the schedule")
        unknown = {dep for job in jobs for dep in job.depends_on if dep not in self.jobs}
        if unknown:
            raise ValueError(f"dependencies name unknown jobs: {sorted(unknown)}")
        self.devices = [dict(device) for device in devices]
        if not self.devices:
            raise ScheduleRefused("no usable device: refusing to schedule anything")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger
        self.run_id = run_id
        self.lease_root = lease_root
        self.clock = clock
        self.use_leases = use_leases
        self.limit = len(self.devices) if max_parallel_jobs is None else int(max_parallel_jobs)
        if self.limit < 1:
            raise ValueError("max_parallel_jobs must be at least 1")
        self.running: dict[str, _Running] = {}
        self.done: dict[str, dict] = {}
        self.failed: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.decisions: list[dict] = []

    # ---- job bookkeeping -------------------------------------------------
    def job_output(self, name: str) -> Path:
        return self.output / name

    def outcome_path(self, name: str) -> Path:
        return self.job_output(name) / "outcome.json"

    def failure_path(self, name: str) -> Path:
        return self.job_output(name) / "failure.json"

    def state(self, name: str) -> str:
        with self.lock:
            if name in self.done:
                return "completed"
            if name in self.failed:
                return "failed"
            if name in self.running:
                return "running"
        job = self.jobs[name]
        if any(self.state(dep) in ("failed",) for dep in job.depends_on):
            return "blocked"
        if all(self.state(dep) == "completed" for dep in job.depends_on):
            return "ready"
        return "waiting"

    def restore(self) -> dict:
        """Re-derive the state from the recorded files; calling it twice changes nothing."""
        found = {"completed": 0, "failed": 0}
        for name in self.jobs:
            if name in self.done or name in self.failed or name in self.running:
                continue
            outcome, failure = self.outcome_path(name), self.failure_path(name)
            if outcome.is_file():
                record = read_json(outcome)
                if record.get("status") == "completed":
                    self.done[name] = record
                    found["completed"] += 1
            elif failure.is_file():
                record = read_json(failure)
                if record.get("status") == "failed":
                    self.failed[name] = record
                    found["failed"] += 1
        return found

    # ---- decisions -------------------------------------------------------
    def free_devices(self) -> list[dict]:
        with self.lock:
            busy = {uuid for run in self.running.values()
                    for uuid in (str(device["uuid"]) for device in run.devices)}
        return [device for device in self.devices if str(device["uuid"]) not in busy]

    def _budget(self, job: Job, devices: Sequence[dict]) -> dict:
        if self.ledger is None:
            return {"allowed": True, "reason": "unbudgeted"}
        return self.ledger.may_start(devices=len(devices), estimate_seconds=job.estimate_seconds,
                                     reserve_seconds=job.reserve_seconds, category=job.category)

    def plan_step(self) -> dict:
        """What may start right now, and why everything else does not.

        Deterministic: ready jobs are considered by (priority, name), devices by index.
        """
        assignments: list[Assignment] = []
        waiting: list[Waiting] = []
        with self.lock:
            free = self.free_devices()
            heavy_running = sum(1 for run in self.running.values() if run.job.heavy)
            for name in sorted(self.jobs, key=lambda n: (self.jobs[n].priority, n)):
                job = self.jobs[name]
                state = self.state(name)
                if state in ("completed", "failed", "running"):
                    continue
                if state == "blocked":
                    failed = [dep for dep in job.depends_on if self.state(dep) == "failed"]
                    waiting.append(Waiting(job, "failed_dependency", f"depends on failed {failed}"))
                    continue
                if state == "waiting":
                    pending = [dep for dep in job.depends_on if self.state(dep) != "completed"]
                    waiting.append(Waiting(job, "dependency", f"waiting for {pending}"))
                    continue
                # Devices first: "no free card" and "the parallelism cap is reached" are
                # different findings and the report has to name the binding one.
                if len(free) < job.device_count:
                    waiting.append(Waiting(job, "devices_busy",
                                           f"needs {job.device_count} of {len(free)} free device(s)"))
                    continue
                if job.heavy and heavy_running >= self.limit:
                    waiting.append(Waiting(job, "parallel_limit",
                                           f"{heavy_running} heavy job(s) already placed or running, "
                                           f"limit {self.limit}"))
                    continue
                chosen = tuple(free[:job.device_count])
                verdict = self._budget(job, chosen)
                if not verdict.get("allowed"):
                    waiting.append(Waiting(job, "budget", f"{verdict.get('reason')}: {verdict}"))
                    continue
                assignments.append(Assignment(job, chosen))
                free = free[job.device_count:]
                if job.heavy:
                    heavy_running += 1
        return {"assignments": assignments, "waiting": waiting}

    # ---- lifecycle -------------------------------------------------------
    def start(self, assignment: Assignment, *, wait_seconds: float = 0.0) -> dict:
        """Take the devices (leases included) and mark the job running."""
        job, devices = assignment.job, assignment.devices
        with self.lock:
            if self.state(job.name) not in ("ready",):
                raise ScheduleRefused(f"job {job.name} is {self.state(job.name)}, not ready")
            stack = ExitStack()
            leases = []
            if self.use_leases:
                leases = stack.enter_context(device_leases(
                    [dict(device) for device in devices], job=job.name, run_id=self.run_id,
                    root=self.lease_root, wait_seconds=wait_seconds))
            started = self.clock()
            record = _Running(job=job, devices=tuple(devices), started=started,
                              started_at=now(), stack=stack, leases=leases)
            self.running[job.name] = record
        return {"job": job.name, "devices": [str(device["uuid"]) for device in devices],
                "leases": [lease.as_dict() for lease in leases], "waiting_seconds": wait_seconds}

    def complete(self, name: str, *, status: str = "completed", detail: Mapping[str, Any] | None = None,
                 episodes: int | None = None) -> dict:
        """Publish the job's own record, charge its devices, release them."""
        with self.lock:
            if name not in self.running:
                raise ScheduleRefused(f"job {name} is not running")
            run = self.running.pop(name)
            wall = max(0.0, self.clock() - run.started)
            record = {"job": name, "kind": run.job.kind or run.job.category,
                      "category": run.job.category, "status": status, "finished_at": now(),
                      "started_at": run.started_at, "wall_seconds": wall,
                      "devices": [str(device["uuid"]) for device in run.devices],
                      "device_indices": [int(device["index"]) for device in run.devices],
                      "waiting_seconds": run.waiting_seconds, "detail": dict(detail or {})}
            target = self.outcome_path(name) if status == "completed" else self.failure_path(name)
            atomic_json(target, record)
            (self.done if status == "completed" else self.failed)[name] = record
        try:
            if self.ledger is not None:
                self.ledger.charge(Charge(job=name, category=run.job.category, status=status,
                                          devices=tuple(str(d["uuid"]) for d in run.devices),
                                          classes={str(d["uuid"]): d.get("class_name")
                                                   for d in run.devices},
                                          wall_seconds=wall, waiting_seconds=run.waiting_seconds,
                                          episodes=episodes))
                self.ledger.write()
        finally:
            run.stack.close()      # leases are flock-based: closing frees exactly these
        return record

    def abandon(self, name: str, *, reason: str) -> dict:
        """A crash path: the job's devices go back, the job itself stays unresolved."""
        with self.lock:
            run = self.running.pop(name, None)
        if run is None:
            raise ScheduleRefused(f"job {name} is not running")
        run.stack.close()
        record = {"job": name, "abandoned_at": now(), "reason": reason,
                  "devices": [str(device["uuid"]) for device in run.devices]}
        atomic_json(self.job_output(name) / "abandoned.json", record)
        return record

    # ---- reporting -------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            step = self.plan_step()
            return {
                "schema_version": 1, "created_at": now(), "run_id": self.run_id,
                "devices": [{"uuid": str(device["uuid"]), "index": int(device["index"]),
                             "class_name": device.get("class_name")} for device in self.devices],
                "max_parallel_jobs": self.limit,
                "running": {name: {"devices": [str(d["uuid"]) for d in run.devices],
                                   "started_at": run.started_at}
                            for name, run in sorted(self.running.items())},
                "completed": sorted(self.done), "failed": sorted(self.failed),
                "decisions": list(self.decisions[-50:]),
                "next": {
                    "assignments": [{"job": item.job.name,
                                     "devices": [str(device["uuid"]) for device in item.devices]}
                                    for item in step["assignments"]],
                    "waiting": [{"job": item.job.name, "reason": item.reason, "detail": item.detail}
                                for item in step["waiting"]],
                },
            }

    def write_snapshot(self) -> Path:
        path = self.output / "schedule.json"
        atomic_json(path, self.snapshot())
        return path

    def occupancy(self) -> list[dict]:
        """Who holds which device, from the lease sidecars rather than from our own state."""
        return occupancy(self.lease_root)

    def runnable(self) -> bool:
        return any(self.state(name) == "ready" for name in self.jobs)

    def unresolved(self) -> dict:
        return {name: self.state(name) for name in self.jobs
                if self.state(name) not in ("completed", "failed")}
