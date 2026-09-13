"""Two budgets, four categories, and never a discount for idle hardware.

The wall clock already has an authority in this pipeline (``start_or_resume_budget`` sets
``Runtime.deadline`` and is never reset on resume); this module is the *second* budget that
multi-device work makes necessary -- accumulated GPU hours -- plus the occupancy and
utilization record that makes a resource claim auditable.

Charging is deliberately crude in one direction: a device is charged for the whole wall
time it is leased, whatever its utilization.  Low utilization is reported, never deducted,
because a held device is unavailable to every other job whether or not it is busy.
Heterogeneous devices are counted per class and never summed as if they were equivalent.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .common import atomic_json, now, read_json

CATEGORIES = ("collection", "evaluation", "training", "audit", "retry", "api", "probe",
              "waiting")


@dataclass
class Charge:
    """One finished job: what it held, for how long, and what it produced."""

    job: str
    category: str
    status: str = "completed"
    attempt: int = 1
    devices: tuple[str, ...] = ()
    classes: Mapping[str, str] = field(default_factory=dict)
    wall_seconds: float = 0.0
    waiting_seconds: float = 0.0
    peak_memory_mib: Mapping[str, int] = field(default_factory=dict)
    mean_utilization_pct: Mapping[str, float] = field(default_factory=dict)
    episodes: int | None = None
    detail: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        record = asdict(self)
        record["devices"] = list(self.devices)
        return record


class BudgetLedger:
    """GPU-hours and wall clock enforced together; the snapshot is the receipt."""

    def __init__(self, path: Path, *, wall_limit_seconds: float,
                 gpu_hours_limit: float | None, devices: Sequence[dict],
                 clock: Callable[[], float] = time.time):
        self.path = path
        self.wall_limit = float(wall_limit_seconds)
        self.gpu_hours_limit = None if gpu_hours_limit is None else float(gpu_hours_limit)
        self.clock = clock
        self.devices = [{"uuid": d.get("uuid"), "index": d.get("index"),
                         "model": d.get("model"), "class_name": d.get("class_name")}
                        for d in devices]
        self.charges: list[dict] = []
        self.started_at = self.clock()
        self.wall_charged = 0.0
        self.gpu_seconds = 0.0
        self.waiting_gpu_seconds = 0.0
        self.device_seconds: dict[str, float] = {}
        self.by_category: dict[str, float] = {name: 0.0 for name in CATEGORIES}
        self.by_class: dict[str, float] = {}
        if path.is_file():
            self._restore(read_json(path))

    def _restore(self, record: dict) -> None:
        """Resume continues the same budgets; nothing is reset by restarting."""
        self.started_at = record.get("wall_clock", {}).get("budget_started_at_epoch", self.started_at)
        self.wall_charged = float(record.get("wall_clock", {}).get("charged_seconds", 0.0))
        hours = record.get("gpu_hours", {})
        self.gpu_seconds = float(hours.get("charged", 0.0)) * 3600.0
        self.waiting_gpu_seconds = float(hours.get("waiting_gpu_hours", 0.0)) * 3600.0
        self.device_seconds = {k: float(v) for k, v in record.get("occupancy",
                                                                  {}).get("device_seconds", {}).items()}
        self.by_category = {name: float(hours.get("by_category", {}).get(name, 0.0))
                            for name in CATEGORIES}
        self.by_class = {k: float(v) for k, v in hours.get("by_class", {}).items()}
        self.charges = list(record.get("jobs", []))

    # ---- budgets ---------------------------------------------------------
    @property
    def elapsed_wall(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    @property
    def remaining_wall(self) -> float:
        return self.wall_limit - self.elapsed_wall

    @property
    def charged_gpu_hours(self) -> float:
        return self.gpu_seconds / 3600.0

    @property
    def remaining_gpu_hours(self) -> float | None:
        if self.gpu_hours_limit is None:
            return None
        return self.gpu_hours_limit - self.charged_gpu_hours

    def may_start(self, *, devices: int = 1, estimate_seconds: float,
                  reserve_seconds: float = 0.0, category: str = "evaluation") -> dict:
        """Both budgets must afford the estimate plus the reserved verification cost."""
        need = float(estimate_seconds) + float(reserve_seconds)
        if category not in CATEGORIES:
            raise ValueError(f"unknown accounting category: {category}")
        if self.remaining_wall < need:
            return {"allowed": False, "reason": "wall_clock",
                    "remaining_wall_seconds": self.remaining_wall, "needed_seconds": need}
        remaining_hours = self.remaining_gpu_hours
        if remaining_hours is not None and remaining_hours * 3600.0 < devices * need:
            return {"allowed": False, "reason": "gpu_hours",
                    "remaining_gpu_hours": remaining_hours,
                    "needed_gpu_hours": devices * need / 3600.0}
        return {"allowed": True, "reason": "affordable",
                "remaining_wall_seconds": self.remaining_wall,
                "remaining_gpu_hours": remaining_hours}

    # ---- charging --------------------------------------------------------
    def charge(self, charge: Charge) -> dict:
        if charge.category not in CATEGORIES:
            raise ValueError(f"unknown accounting category: {charge.category}")
        class_of = {**{d["uuid"]: d.get("class_name") for d in self.devices if d.get("uuid")},
                    **{str(k): v for k, v in (charge.classes or {}).items()}}
        hours = len(charge.devices) * float(charge.wall_seconds) / 3600.0
        self.gpu_seconds += hours * 3600.0
        self.wall_charged += float(charge.wall_seconds)
        self.by_category[charge.category] += hours
        for uuid in charge.devices:
            self.device_seconds[uuid] = self.device_seconds.get(uuid, 0.0) + float(charge.wall_seconds)
            name = class_of.get(uuid) or "unknown"
            self.by_class[name] = self.by_class.get(name, 0.0) + float(charge.wall_seconds) / 3600.0
        self.waiting_gpu_seconds += len(charge.devices) * float(charge.waiting_seconds)
        self.by_category["waiting"] += len(charge.devices) * float(charge.waiting_seconds) / 3600.0
        record = charge.as_dict()
        record["charged_gpu_hours"] = hours
        self.charges.append(record)
        return record

    def record_waiting(self, *, job: str, devices: Sequence[str], seconds: float) -> None:
        """Waiting is charged to the wall clock only -- it holds no device."""
        self.waiting_gpu_seconds += len(devices) * float(seconds)
        self.by_category["waiting"] += len(devices) * float(seconds) / 3600.0
        self.charges.append({"job": job, "category": "waiting", "status": "waiting",
                             "devices": list(devices), "waiting_seconds": float(seconds),
                             "charged_gpu_hours": 0.0})

    # ---- receipts --------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "autosim_run_accounting",
            "updated_at": now(),
            "wall_clock": {
                "limit_seconds": self.wall_limit,
                "budget_started_at_epoch": self.started_at,
                "elapsed_seconds": self.elapsed_wall,
                "charged_seconds": self.wall_charged,
                "remaining_seconds": self.remaining_wall,
                "authority": "Runtime.deadline (start_or_resume_budget); never reset on resume",
            },
            "gpu_hours": {
                "limit": self.gpu_hours_limit,
                "charged": self.charged_gpu_hours,
                "remaining": self.remaining_gpu_hours,
                "by_class": dict(sorted(self.by_class.items())),
                "by_category": {k: v for k, v in sorted(self.by_category.items()) if v},
                "waiting_gpu_hours": self.waiting_gpu_seconds / 3600.0,
                "note": "2 devices for 1 h is 2 GPU-hours; heterogeneous classes are reported "
                        "separately and never summed as equivalent",
            },
            "occupancy": {"device_seconds": dict(sorted(self.device_seconds.items()))},
            "utilization": {
                "sampled": False,
                "note": "occupancy is charged regardless of utilization; low utilization never "
                        "offsets budget",
            },
            "devices": self.devices,
            "jobs": self.charges,
        }

    def write(self) -> Path:
        atomic_json(self.path, self.snapshot())
        return self.path


def describe_process(pid: int, *, proc: Path = Path("/proc")) -> dict:
    """Name a holder by its command line and its parent, or say why it cannot be named.

    A pid is not an answer: by the time anyone reads the receipt the process is usually gone,
    and an unattributable 1 GiB is what left the last probe with a finding nobody could act
    on.  The command line is read *while* the process is alive, so the receipt carries the
    name even after the process exits.  Never raises: a diagnostic that can fail is a
    diagnostic that turns a measurement into a crash.
    """
    try:
        raw = (proc / str(pid) / "cmdline").read_bytes()
    except OSError as exc:
        return {"cmd": None, "note": f"unreadable: {exc.strerror or exc}"}
    argv = [part for part in raw.decode("utf-8", "replace").split("\0") if part]
    return {"cmd": " ".join(argv)[:300] if argv else None,
            "parent": _parent_of(pid, proc=proc)}


def _parent_of(pid: int, *, proc: Path) -> int | None:
    try:
        fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    try:
        return int(fields[1])          # ppid, the first field after the comm field
    except (IndexError, ValueError):
        return None


class UtilizationSampler:
    """Sample per-device utilization while jobs run; never affects the charge.

    Every sample also names the *processes* holding memory on each card.  A card that was
    supposed to stay idle while another one ran an episode is the one failure mode this
    ladder cannot see from utilization alone, and "1030 MiB appeared on card 0, owner
    unknown" is not a finding anyone can act on -- the pid is.
    """

    def __init__(self, *, interval_seconds: float = 15.0, runner=None,
                 clock: Callable[[], float] = time.monotonic):
        from .devices import run_text

        self.interval = interval_seconds
        self.runner = runner or run_text
        self.clock = clock
        self.samples: dict[str, dict] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        text = self.runner(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"])
        if text.strip().startswith("__error__"):
            return
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                utilization, memory = float(parts[1]), int(parts[2])
            except ValueError:
                continue
            key = parts[0]      # the UUID as reported; joins with leases, plans and charges
            row = self.samples.setdefault(key, {"count": 0, "utilization_sum": 0.0,
                                                "peak_memory_mib": 0, "processes": {}})
            row["count"] += 1
            row["utilization_sum"] += utilization
            row["peak_memory_mib"] = max(row["peak_memory_mib"], memory)
        apps = self.runner(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                            "--format=csv,noheader,nounits"])
        if apps.strip().startswith("__error__"):
            return
        from .devices import parse_compute_apps

        for app in parse_compute_apps(apps):
            uuid, pid = app.get("gpu_uuid"), app.get("pid")
            if uuid is None or pid is None:
                continue
            row = self.samples.setdefault(uuid, {"count": 0, "utilization_sum": 0.0,
                                                 "peak_memory_mib": 0, "processes": {}})
            held = row["processes"].setdefault(str(pid), {"mib": 0})
            held["mib"] = max(held.get("mib", 0), app.get("used_memory_mib") or 0)
            if "cmd" not in held:
                held.update(describe_process(pid))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample_once()
            except Exception:            # sampling must never break a run
                continue

    def __enter__(self) -> "UtilizationSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval))

    def summary(self) -> dict:
        return {uuid: {"mean_utilization_pct": row["utilization_sum"] / row["count"],
                       "peak_memory_mib": row["peak_memory_mib"], "samples": row["count"],
                       # Biggest holder first: the pid that made a "should be idle" card move
                       # is the first thing a reader looks for, and it is rarely the biggest.
                       "processes": dict(sorted((row.get("processes") or {}).items(),
                                                key=lambda item: -item[1].get("mib", 0)))}
                for uuid, row in self.samples.items() if row["count"]}
