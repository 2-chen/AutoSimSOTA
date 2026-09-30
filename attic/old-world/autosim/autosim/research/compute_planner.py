"""Execution planning is versioned separately from the scientific protocol."""
from __future__ import annotations
import os
from pathlib import Path

from .common import atomic_json, exclusive, now, object_digest, read_json
from .compute_agent import ComputeAgent, client_from_file
from .resource_contracts import host_inventory


def upgrade_plan(plan: dict, *, output: Path) -> dict:
    host = host_inventory(output)
    devices = [dict(d) for d in plan["usable"]]
    for d in devices:
        caps = dict(d.get("capabilities") or {})
        if d.get("capability") == "verified_sim":
            caps.update(physics="verified", render="verified", policy_inference="verified")
        elif d.get("capability") == "verified_train":
            caps["train"] = "verified"
        d["capabilities"] = caps
    result = {**plan, "schema_version": 2, "unified_scheduler": True,
              "host_capacity": host, "usable": devices, "legacy_equivalence": False,
              "parallel_cap": plan.get("max_parallel_jobs") or len(devices)}
    result["plan_digest"] = object_digest({k: v for k, v in result.items() if k != "plan_digest"})
    return result


def publish_execution_plan(output: Path, plan: dict, *, protocol_digest: str = "") -> Path:
    root = Path(output) / "compute/execution_plans"
    root.mkdir(parents=True, exist_ok=True)
    payload = {"protocol_digest": protocol_digest, "plan": plan}
    identity = object_digest(payload)
    with exclusive(root / "publish.lock"):
        target = root / f"{identity}.json"
        if not target.exists():
            atomic_json(target, {**payload, "execution_digest": identity, "created_at": now()})
        atomic_json(root / "current.json", {"path": str(target), "execution_digest": identity})
    return target


class ComputePlanner:
    def __init__(self, root: Path, *, controller="heuristic", env_file: Path | None = None,
                 min_interval_seconds=300):
        self.root = Path(root)
        self.controller = controller
        self.min_interval = min_interval_seconds
        self.agent = (ComputeAgent(self.root / "api", client_from_file(env_file))
                      if controller == "api" and env_file is not None else None)

    def propose(self, plan: dict, telemetry: dict | None = None, *, force=False) -> dict:
        import time
        path = self.root / "planner_state.json"
        previous = read_json(path) if path.exists() else None
        snapshot = {"devices": [{k: d.get(k) for k in ("uuid", "class_name", "memory_total_mib",
                      "memory_used_mib", "capabilities")} for d in plan["usable"]],
                    "host": plan["host_capacity"], "measurements": telemetry or {}}
        resource_digest = object_digest({"devices": [{k: d.get(k) for k in
            ("uuid", "class_name", "memory_total_mib", "capabilities")} for d in snapshot["devices"]],
            "cpu": snapshot["host"]["cpu_cores"], "host": snapshot["host"].get("host")})
        if previous and not force and previous["resource_digest"] == resource_digest and (
                time.time() - previous["epoch"] < self.min_interval):
            return previous
        count = max(1, min(len(plan["usable"]), plan.get("max_parallel_jobs") or len(plan["usable"])))
        cpus = max(1, int(plan["host_capacity"]["cpu_cores"]))
        limits = {"max_parallel_jobs": max(1, min(len(plan["usable"]), plan.get("parallel_cap", len(plan["usable"])))),
                  "loader_workers": min(32, max(0, cpus // count - 1)),
                  "omp_threads": min(32, max(1, cpus // count)),
                  "io_slots": plan["host_capacity"]["io_slots"], "shard_min_episodes": 128}
        parameters = {"max_parallel_jobs": count, "loader_workers": min(4, limits["loader_workers"]),
                      "omp_threads": min(4, limits["omp_threads"]),
                      "io_slots": min(count, limits["io_slots"]), "shard_min_episodes": 8}
        proposal, mode, error = None, "heuristic", None
        if self.agent:
            try:
                proposal = self.agent.plan(snapshot, limits)
                parameters.update({a["parameter"]: a["value"] for a in proposal["actions"]})
                per_job_cpu = max(1, cpus // parameters["max_parallel_jobs"])
                parameters["omp_threads"] = min(parameters["omp_threads"], per_job_cpu)
                parameters["loader_workers"] = min(parameters["loader_workers"], max(0, per_job_cpu-1))
                mode = "api"
            except Exception as exc:
                mode, error = "heuristic_fallback", type(exc).__name__
        result = {"parameters": parameters, "mode": mode, "error_type": error, "proposal": proposal,
                  "snapshot_digest": object_digest(snapshot), "resource_digest": resource_digest,
                  "epoch": time.time(), "time": now()}
        atomic_json(self.root / "decisions" / (object_digest(result) + ".json"), result)
        atomic_json(path, result)
        return result
