"""Validated resource requests shared by the planner, scheduler and workers."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class ResourceRequest:
    cpu_cores: float = 0
    ram_mib: int = 0
    shm_mib: int = 0
    scratch_mib: int = 0
    io_slots: int = 0
    per_gpu_memory_mib: int = 0
    capabilities: tuple[str, ...] = ()
    homogeneous: bool = False

    def __post_init__(self):
        for name in ("cpu_cores", "ram_mib", "shm_mib", "scratch_mib", "io_slots",
                     "per_gpu_memory_mib"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid resource {name}: {value}")
        if set(self.capabilities) - {"train", "physics", "render", "policy_inference",
                                     "distributed_train", "cuda"}:
            raise ValueError("unknown resource capability")

    def as_dict(self) -> dict:
        return asdict(self)

    def host(self) -> dict:
        return {k: getattr(self, k) for k in
                ("cpu_cores", "ram_mib", "shm_mib", "scratch_mib", "io_slots")}


def device_compatible(device: Mapping, request: ResourceRequest) -> bool:
    if request.per_gpu_memory_mib:
        total, used = device.get("memory_total_mib"), device.get("memory_used_mib")
        if total is None or used is None or total - used < request.per_gpu_memory_mib:
            return False
    capabilities = device.get("capabilities") or {}
    for capability in request.capabilities:
        receipt = capabilities.get(capability)
        if isinstance(receipt, dict):
            receipt = receipt.get("status")
        if receipt != "verified":
            return False
    return True


def host_inventory(path=None) -> dict:
    import os
    import shutil
    from pathlib import Path
    from .host_resources import capacity
    raw = capacity()
    shm = shutil.disk_usage("/dev/shm").free // 2**20 if Path("/dev/shm").is_dir() else 0
    return {"cpu_cores": raw["cpu"]["effective_cpus"],
            "ram_mib": raw["memory"]["available_mib"], "shm_mib": shm,
            "scratch_mib": shutil.disk_usage("/tmp").free // 2**20,
            "output_free_mib": shutil.disk_usage(path or "/tmp").free // 2**20,
            "io_slots": max(1, min(4, int(raw["cpu"]["effective_cpus"]))),
            "host": os.uname().nodename, "raw": raw}


def host_fits(request: ResourceRequest, capacity: Mapping, used: Mapping) -> bool:
    return all(not amount or (capacity.get(key) is not None and
                              amount + used.get(key, 0) <= capacity[key])
               for key, amount in request.host().items())
