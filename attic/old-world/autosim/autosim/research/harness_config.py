"""Versioned execution policy, separate from the scientific experiment contract."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .common import object_digest, read_json


DEFAULT_POLICY = {
    "schema_version": 1,
    "lifecycle": {"startup_attempts": 3, "incident_timeout": 1800,
                  "construction_concurrency": None, "minimum_overlap_seconds": 5.0},
    "repair": {"enabled": True, "max_candidates": 2, "max_seconds": 1200},
    "continuation": {"enabled": True, "max_relaunches": 4, "max_no_progress": 2},
    "validation": {"stage": "research", "force_probe": False, "probe_repetitions": 1,
                   "inject_failure_worker": None, "inject_failure_generation": 1},
}


def load_harness_policy(path: Path | str | None = None) -> dict:
    """Reject unrecognized knobs instead of silently weakening a validation gate."""
    policy = deepcopy(DEFAULT_POLICY)
    value = read_json(Path(path)) if path is not None else {}
    if not isinstance(value, dict) or set(value) - set(policy):
        raise ValueError("unknown harness policy fields")
    if value.get("schema_version", 1) != 1 or isinstance(value.get("schema_version"), bool):
        raise ValueError("unsupported harness policy schema")
    for section in ("lifecycle", "repair", "continuation", "validation"):
        supplied = value.get(section, {})
        if not isinstance(supplied, dict) or set(supplied) - set(policy[section]):
            raise ValueError(f"unknown harness {section} fields")
        policy[section].update(supplied)
    def integer(section, name, lower, upper, optional=False):
        number = policy[section][name]
        if optional and number is None:
            return
        if type(number) is not int or not lower <= number <= upper:
            raise ValueError(f"harness {section}.{name} must be in [{lower},{upper}]")
    integer("lifecycle", "startup_attempts", 1, 3)
    integer("lifecycle", "incident_timeout", 30, 3600)
    integer("lifecycle", "construction_concurrency", 1, 8, optional=True)
    overlap = policy["lifecycle"]["minimum_overlap_seconds"]
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or not 0 < overlap <= 60:
        raise ValueError("invalid minimum overlap")
    integer("repair", "max_candidates", 1, 2)
    integer("repair", "max_seconds", 30, 1200)
    integer("continuation", "max_relaunches", 0, 8)
    integer("continuation", "max_no_progress", 1, 2)
    integer("validation", "probe_repetitions", 1, 3)
    integer("validation", "inject_failure_worker", 0, 7, optional=True)
    integer("validation", "inject_failure_generation", 1, 3)
    for section, name in (("repair", "enabled"), ("continuation", "enabled"),
                          ("validation", "force_probe")):
        if type(policy[section][name]) is not bool:
            raise ValueError(f"harness {section}.{name} must be a boolean")
    validation = policy["validation"]
    if validation["stage"] not in {"research", "native_admission"}:
        raise ValueError("unknown harness validation stage")
    if validation["inject_failure_worker"] is not None and validation["stage"] != "native_admission":
        raise ValueError("fault injection is confined to native admission validation")
    if validation["probe_repetitions"] > 1 and validation["stage"] != "native_admission":
        raise ValueError("repeated probing requires an explicit validation run")
    return policy


def policy_identity(policy: dict) -> str:
    return object_digest(policy)
