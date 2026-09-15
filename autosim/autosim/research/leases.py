"""Atomic per-device leases, keyed by GPU UUID.

Two properties matter and both come from ``flock`` rather than from bookkeeping:

* a lease dies with its holder, so a crashed or OOM-killed job frees exactly its own
  devices and nothing else -- there is no reaper to get wrong;
* the lock is taken on ``gpu-<uuid>.lock``, never on an index, so a device keeps its
  identity across container renumbering and across a resume.

The sidecar ``.owner.json`` and the append-only ``lease_receipts.jsonl`` exist only to
make the lock *legible* (who held what, when, why a job waited); they are never the
authority.  A live holder is never displaced.
"""

from __future__ import annotations

import fcntl
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Iterator, Sequence

from .common import atomic_json, event, now, read_json
from .devices import normalize_uuid


class LeaseUnavailable(RuntimeError):
    """Another process holds a device; the message names it."""


@dataclass(frozen=True)
class Lease:
    uuid: str
    index: int
    run_id: str
    job: str
    pid: int
    host: str
    started_at: str
    lock_path: Path
    owner_path: Path

    def as_dict(self) -> dict:
        record = asdict(self)
        record["lock_path"], record["owner_path"] = str(self.lock_path), str(self.owner_path)
        return record


def lease_root(host: str | None = None) -> Path:
    return Path("/tmp/autosim-robosyn-gpu-lease") / (host or os.uname().nodename)


def receipts_path(lease: Lease) -> Path:
    return lease.lock_path.parent / "lease_receipts.jsonl"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def legacy_lock_path(index: int) -> Path:
    """``robosyn_mvp.gpu_lock``'s path: the index-keyed lock this module replaces."""
    return Path(f"/tmp/autosim-robosyn-gpu-{int(index)}.lock")


def legacy_index_holder(index: int) -> dict | None:
    """Evidence that a *legacy* single-device run holds this physical index, or ``None``.

    Leases are keyed by UUID and the legacy lock by index, so ignoring the legacy lock
    would let a ``--gpu 0`` run and a leased multi-device run sit on one card.  The flock
    stays the authority -- a dead holder's lock is already released by the kernel, so this
    probes rather than trusting the pid the holder wrote -- and the file is opened ``a+``
    because opening it ``w`` (as the legacy writer does) would truncate a live holder's
    own evidence line.
    """
    path = legacy_lock_path(index)
    try:
        stream = path.open("a+")
    except OSError:
        return None
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            holder = stream.read().strip()[:200]
            # ``flock`` denies a process its own lock taken through another descriptor, so
            # a probe from the holder itself looks blocked.  It cannot be a conflict: the
            # lock is exclusive, so whoever holds it holds it alone.
            match = re.search(r"pid=(\d+)", holder)
            if match and int(match.group(1)) == os.getpid():
                return None
            return {"path": str(path), "index": int(index), "holder": holder,
                    "probed_by_pid": os.getpid()}
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return None
    finally:
        stream.close()


def _acquire(lease: Lease) -> tuple[bool, IO[str] | None]:
    """Take the lock once.  The returned fd must live as long as the lease."""
    stream = lease.lock_path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return False, None
    previous = None
    if lease.owner_path.is_file():
        try:
            previous = read_json(lease.owner_path)
        except (ValueError, OSError):
            previous = None
    if previous and previous.get("pid") not in (None, lease.pid):
        if not pid_alive(int(previous.get("pid") or 0)):
            event(receipts_path(lease), "stale_lease_reclaimed", uuid=lease.uuid,
                  previous=previous, by_pid=lease.pid, job=lease.job)
    atomic_json(lease.owner_path, lease.as_dict())
    event(receipts_path(lease), "device_leased", uuid=lease.uuid, index=lease.index,
          run_id=lease.run_id, job=lease.job, pid=lease.pid, host=lease.host)
    return True, stream


def _release(lease: Lease, stream: IO[str]) -> None:
    try:
        if lease.owner_path.is_file():
            try:
                holder = read_json(lease.owner_path)
            except (ValueError, OSError):
                holder = {}
            if holder.get("pid") == lease.pid and holder.get("job") == lease.job:
                lease.owner_path.unlink()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()
    event(receipts_path(lease), "device_released", uuid=lease.uuid, job=lease.job,
          pid=lease.pid, released_at=now())


@contextmanager
def device_lease(device: dict, *, job: str, run_id: str, root: Path | None = None,
                 wait_seconds: float = 0.0, poll_seconds: float = 2.0) -> Iterator[Lease]:
    """Hold one device for the duration of the block, or raise ``LeaseUnavailable``."""
    with device_leases([device], job=job, run_id=run_id, root=root,
                       wait_seconds=wait_seconds, poll_seconds=poll_seconds) as leases:
        yield leases[0]


@contextmanager
def device_leases(devices: Sequence[dict], *, job: str, run_id: str, root: Path | None = None,
                  wait_seconds: float = 0.0, poll_seconds: float = 2.0,
                  consult_legacy: bool = True) -> Iterator[list[Lease]]:
    """All-or-nothing acquisition over a UUID-sorted device set.

    Sorting by UUID removes the ordering that makes multi-device acquisition deadlock, and
    taking the whole set at once (releasing everything on a partial failure) removes the
    hold-and-wait that makes it deadlock the other way.

    ``consult_legacy`` also treats a held ``robosyn_mvp.gpu_lock`` on any requested index as
    occupancy: that lock is not ours to take, but a device it holds is not free for us
    either, and saying so here is cheaper than discovering it as a crashed episode.
    """
    if not devices:
        raise ValueError("no devices requested")
    base = root or lease_root()
    base.mkdir(parents=True, exist_ok=True)
    # Sort on the folded form so the order is stable, but key on the UUID as reported:
    # leases, plans and accounting all join on that exact string.
    ordered = sorted(devices, key=lambda device: normalize_uuid(device["uuid"]))
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while True:
        held: list[tuple[Lease, IO[str]]] = []
        blocked: list[str] = []
        legacy: list[dict] = []
        for device in ordered:
            uuid = str(device["uuid"])
            found = legacy_index_holder(device["index"]) if consult_legacy else None
            if found is not None:
                legacy.append(found)
                blocked.append(uuid)
                break
            lease = Lease(uuid=uuid, index=int(device["index"]), run_id=run_id, job=job,
                          pid=os.getpid(), host=os.uname().nodename, started_at=now(),
                          lock_path=base / f"gpu-{uuid}.lock",
                          owner_path=base / f"gpu-{uuid}.owner.json")
            taken, stream = _acquire(lease)
            if not taken:
                blocked.append(uuid)
                break
            held.append((lease, stream))
        if not blocked:
            try:
                yield [lease for lease, _ in held]
            finally:
                for lease, stream in held:
                    _release(lease, stream)
            return
        for lease, stream in held:
            _release(lease, stream)
        if legacy:
            event(base / "lease_receipts.jsonl", "legacy_index_lock_blocks_lease",
                  job=job, run_id=run_id, holders=legacy)
        if time.monotonic() >= deadline:
            reason = (f"legacy index lock(s) {legacy} " if legacy else "")
            raise LeaseUnavailable(
                f"device(s) {blocked} held by another process {reason}while requesting "
                f"{[str(d['uuid']) for d in ordered]} for job {job}")
        time.sleep(min(poll_seconds, max(0.05, deadline - time.monotonic())))


def sweep_stale(root: Path | None = None, *, alive=pid_alive) -> list[dict]:
    """Report sidecars whose holder is gone.  The lock itself is the truth, never this."""
    base = root or lease_root()
    reclaimed: list[dict] = []
    if not base.is_dir():
        return reclaimed
    for path in sorted(base.glob("gpu-*.owner.json")):
        try:
            holder = read_json(path)
        except (ValueError, OSError):
            continue
        pid = int(holder.get("pid") or 0)
        if not alive(pid):
            record = {"uuid": holder.get("uuid"), "job": holder.get("job"), "pid": pid,
                      "reclaimed_at": now()}
            reclaimed.append(record)
            event(base / "lease_receipts.jsonl", "stale_sidecar", **record)
    return reclaimed


def occupancy(root: Path | None = None) -> list[dict]:
    """Who holds what right now -- evidence for the resource report, not a gate."""
    base = root or lease_root()
    held: list[dict] = []
    if not base.is_dir():
        return held
    for path in sorted(base.glob("gpu-*.owner.json")):
        try:
            held.append(read_json(path))
        except (ValueError, OSError):
            continue
    return held
