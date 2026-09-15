"""CPU and memory capacity visible to this process, including cgroup limits."""

from __future__ import annotations

import os
from pathlib import Path


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _number(text: str) -> int | None:
    try:
        value = int(text)
        return value if 0 <= value < 2**60 else None
    except ValueError:
        return None


def _hierarchy(root: Path, relative: str):
    # A namespace may report /../../host/path. Never traverse outside its mount.
    path = root.joinpath(*[p for p in relative.split("/") if p not in ("", ".", "..")])
    while path != root:
        yield path
        path = path.parent
    yield root


def capacity(*, cgroup: Path = Path("/sys/fs/cgroup"), proc: Path = Path("/proc"),
             affinity: set[int] | None = None) -> dict:
    if affinity is None:
        try:
            affinity = set(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = set(range(os.cpu_count() or 1))
    quotas, memory_limits, memory_available = [], [], []
    memberships = _read(proc / "self/cgroup").splitlines()
    paths: dict[str, str] = {}
    for row in memberships:
        parts = row.split(":", 2)
        if len(parts) == 3:
            for controller in parts[1].split(","):
                paths[controller] = parts[2]
    v2 = (cgroup / "cgroup.controllers").exists() or "" in paths
    cpu_root = cgroup if v2 else cgroup / "cpu"
    if not v2 and not cpu_root.exists():
        cpu_root = cgroup / "cpu,cpuacct"
    for directory in _hierarchy(cpu_root, paths.get("" if v2 else "cpu", "/")):
        values = (_read(directory / "cpu.max").split() if v2 else
                  [_read(directory / "cpu.cfs_quota_us"), _read(directory / "cpu.cfs_period_us")])
        if len(values) == 2:
            quota, period = _number(values[0]), _number(values[1])
            if quota is not None and period:
                quotas.append(quota / period)
    mem_root = cgroup if v2 else cgroup / "memory"
    for directory in _hierarchy(mem_root, paths.get("" if v2 else "memory", "/")):
        limit = _number(_read(directory / ("memory.max" if v2 else "memory.limit_in_bytes")))
        used = _number(_read(directory / ("memory.current" if v2 else "memory.usage_in_bytes")))
        if limit is not None:
            memory_limits.append(limit)
            # Unknown usage is not free memory. A capacity limit alone cannot prove availability.
            if used is not None:
                memory_available.append(max(0, limit - used))
    meminfo = {}
    for row in _read(proc / "meminfo").splitlines():
        name, _, value = row.partition(":")
        if name in {"MemTotal", "MemAvailable"} and value.split():
            amount = _number(value.split()[0])
            if amount is not None:
                meminfo[name] = amount * 1024
    if "MemTotal" in meminfo:
        memory_limits.append(meminfo["MemTotal"])
    if "MemAvailable" in meminfo:
        memory_available.append(meminfo["MemAvailable"])
    effective_cpus = min([float(len(affinity)), *quotas])
    return {
        "cpu": {"logical_count": os.cpu_count(), "affinity": sorted(affinity),
                "quota_cpus": min(quotas) if quotas else None, "effective_cpus": effective_cpus},
        "memory": {"total_mib": min(memory_limits) // 2**20 if memory_limits else None,
                   "available_mib": min(memory_available + memory_limits) // 2**20
                   if memory_available else None},
        "cgroup_version": 2 if v2 else 1,
    }
