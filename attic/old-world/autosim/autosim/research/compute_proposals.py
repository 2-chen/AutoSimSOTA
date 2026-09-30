"""Small, explicit action space for execution tuning; no scientific parameters."""
from __future__ import annotations
import math

LIMITS = {"loader_workers": (0, 32), "io_slots": (1, 16),
          "max_parallel_jobs": (1, 64), "shard_min_episodes": (1, 128),
          "omp_threads": (1, 32)}


def validate_proposal(raw: dict, *, snapshot_digest: str, limits: dict) -> dict:
    if (not isinstance(raw, dict) or type(raw.get("schema_version")) is not int
            or raw["schema_version"] != 1):
        raise ValueError("invalid compute proposal schema")
    if raw.get("based_on_snapshot") != snapshot_digest:
        raise ValueError("stale compute proposal")
    actions = raw.get("actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
        raise ValueError("proposal requires 1..8 actions")
    seen = set()
    for action in actions:
        if set(action) != {"type", "parameter", "value"} or action["type"] != "set_execution_parameter":
            raise ValueError("unknown compute action")
        name, value = action["parameter"], action["value"]
        if name not in LIMITS or name in seen or isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("invalid or duplicate execution parameter")
        lower, upper = LIMITS[name]
        if not lower <= value <= min(upper, limits.get(name, upper)):
            raise ValueError(f"execution parameter exceeds resource bounds: {name}")
        seen.add(name)
    if not isinstance(raw.get("evidence_refs"), list) or not raw["evidence_refs"]:
        raise ValueError("proposal needs measurable evidence references")
    return {"schema_version": 1, "based_on_snapshot": snapshot_digest,
            "actions": actions, "evidence_refs": raw["evidence_refs"],
            "bottleneck": str(raw.get("bottleneck", ""))[:1000]}
