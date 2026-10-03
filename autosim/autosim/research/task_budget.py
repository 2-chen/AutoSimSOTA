"""Per-study GPU lease occupancy, independent of model and controller wall time.

Unfinished reservations remain held after a crash. They are never silently refunded;
the operator must reconcile the corresponding process before releasing one.
"""
from __future__ import annotations

import fcntl
import math
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .common import atomic_json, read_json


class TaskGPUBudget:
    def __init__(self, output: Path):
        self.output = Path(output).resolve()
        self.path = self.output / "task_gpu_budget.json"

    @contextmanager
    def locked(self):
        if self.path.is_symlink() or self.path.with_suffix(".lock").is_symlink():
            raise ValueError("GPU budget path is a symlink")
        with self.path.with_suffix(".lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            record = read_json(self.path) if self.path.exists() else None
            if record is not None and (record.get("output") != str(self.output) or
                                       record.get("schema_version") != 1):
                raise ValueError("GPU task budget identity changed")
            yield record

    def initialize(self, repo: Path, cap_seconds: float | None = None):
        if cap_seconds is not None and (not math.isfinite(cap_seconds) or not 0 < cap_seconds <= 86400):
            raise ValueError("task GPU cap must be within 24 hours")
        with self.locked() as record:
            if record is not None:
                saved_cap = record.get('cap_seconds')
                if (not isinstance(saved_cap, (int, float)) or isinstance(saved_cap, bool)
                        or not math.isfinite(saved_cap) or not 0 < saved_cap <= 86400):
                    raise ValueError('saved task GPU cap must remain within 24 hours')
                if record["repository"] != str(Path(repo).resolve()) or (cap_seconds is not None and record["cap_seconds"] != cap_seconds):
                    raise ValueError("cannot change a task's repository or GPU cap")
                return
            atomic_json(self.path, {
                "schema_version": 1, "task_id": uuid.uuid4().hex,
                "output": str(self.output), "repository": str(Path(repo).resolve()),
                "cap_seconds": 86400 if cap_seconds is None else cap_seconds, "charged_seconds": 0.0, "leases": {},
                "accounting": "physical-device exclusive lease occupancy; not kernel utilization"})

    @staticmethod
    def available(record):
        held = sum(row["reserved_seconds"] for row in record["leases"].values()
                   if row["status"] == "active")
        return max(0.0, record["cap_seconds"] - record["charged_seconds"] - held)

    def snapshot(self):
        with self.locked() as record:
            return {"scope": "task", "task_id": record["task_id"],
                    "cap_gpu_seconds": record["cap_seconds"],
                    "settled_gpu_seconds": record["charged_seconds"],
                    "available_gpu_seconds": self.available(record),
                    "active_leases": sum(row["status"] == "active"
                                         for row in record["leases"].values()),
                    "accounting": record["accounting"]}

    def reserve(self, identity: str, device_uuid: str, requested_seconds: float | None = None):
        with self.locked() as record:
            if record is None:
                raise ValueError("task GPU budget missing")
            if identity in record["leases"]:
                raise ValueError("duplicate GPU lease reservation")
            left = self.available(record)
            if left <= 0:
                raise ValueError("task GPU budget exhausted or reserved by another job")
            if requested_seconds is not None:
                if (not math.isfinite(requested_seconds) or requested_seconds <= 0 or
                        requested_seconds > left):
                    raise ValueError("requested GPU window exceeds available task budget")
                left = requested_seconds
            row = {"status": "active", "reserved_seconds": left,
                   "device_uuid": device_uuid, "started_epoch": time.time(),
                   "deadline_epoch": time.time() + left}
            record["leases"][identity] = row
            atomic_json(self.path, record)
            return row

    def extend(self, identity: str, seconds_from_now: float):
        if not math.isfinite(seconds_from_now) or seconds_from_now <= 0:
            raise ValueError("invalid GPU extension")
        with self.locked() as record:
            row = record["leases"][identity]
            if row["status"] != "active":
                raise ValueError("GPU lease is not active")
            elapsed = max(0, time.time() - row["started_epoch"])
            requested = elapsed + seconds_from_now
            additional = max(0, requested - row["reserved_seconds"])
            if additional > self.available(record):
                raise ValueError("GPU extension exceeds task budget")
            row.update(reserved_seconds=max(row["reserved_seconds"], requested),
                       deadline_epoch=time.time() + seconds_from_now)
            atomic_json(self.path, record)
            return row

    def finish(self, identity: str, elapsed_seconds: float):
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("invalid GPU lease duration")
        with self.locked() as record:
            row = record["leases"][identity]
            if row["status"] != "active":
                return
            row.update(status="finished", charged_seconds=elapsed_seconds,
                       reserved_seconds=0.0, finished_epoch=time.time())
            record["charged_seconds"] += elapsed_seconds
            atomic_json(self.path, record)
