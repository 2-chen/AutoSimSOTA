"""Measured workload costs; environment-scoped samples, never model-name guesses."""
from __future__ import annotations

import math
import statistics
import fcntl
from pathlib import Path

from .common import atomic_json, now, object_digest, read_json, exclusive


class ProfileStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, context: dict) -> str:
        required = {"workload", "code_digest", "environment_id", "device_class"}
        if not required <= context.keys():
            raise ValueError(f"profile context requires {sorted(required)}")
        return object_digest(context)

    def record(self, context: dict, *, seconds: float, units: int = 1,
               cold_start_seconds: float = 0, metrics: dict | None = None) -> dict:
        if not math.isfinite(seconds) or seconds < 0 or units < 1:
            raise ValueError("invalid measured workload")
        path = self.root / (self.key(context) + ".json")
        with path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            value = read_json(path) if path.exists() else {"context": context, "samples": []}
            value["samples"].append({"seconds": seconds, "units": units,
                                  "cold_start_seconds": cold_start_seconds,
                                  "metrics": metrics or {}, "time": now()})
            value["samples"] = value["samples"][-100:]
            atomic_json(path, value)
        return self.estimate(context)

    def estimate(self, context: dict) -> dict:
        path = self.root / (self.key(context) + ".json")
        rows = read_json(path)["samples"] if path.exists() else []
        samples = sorted(row["seconds"] / row["units"] for row in rows)
        if not samples:
            return {"p50": None, "p90": None, "confidence": "unknown", "count": 0}
        return {"p50": statistics.median(samples),
                "p90": samples[max(0, math.ceil(len(samples) * .9) - 1)],
                "confidence": "measured" if len(samples) >= 3 else "preliminary",
                "count": len(samples)}

    def summary(self) -> list[dict]:
        result = []
        for path in sorted(self.root.glob("*.json")):
            row = read_json(path)
            context = row["context"]
            result.append({"profile_id":path.stem, "workload":context["workload"],
                           "device_class":context["device_class"], **self.estimate(context),
                           "latest_runtime_timings":row["samples"][-1].get("metrics", {}).get("runtime_timings", {})})
        return result[-100:]
