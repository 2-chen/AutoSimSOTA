"""Derive a conservative continuation boundary from trusted local receipts.

This module does not rerun a phase, rewrite a probe generation, or invent a
training checkpoint contract. It returns a host-only authorization receipt;
uncertainty about processes, episode progress or durable commits means stop.
"""
from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Callable

from .artifact_commits import lifecycle_compatible
from .common import atomic_json, digest, object_digest, read_json
from .continuation import process_gone
from .native_lifecycle import retryable_startup
from .repository_harness import scientific_identity


class BoundaryRefused(RuntimeError):
    pass


_PRUNE = {".git", ".venv", "__pycache__", "node_modules", "source", "optimized_repo",
          "data", "videos", "images", "assets", "embodichain_data", "lerobot_dataset", "checkpoints"}
_NAMES = {"process.json", "worker_identity.json", "lifecycle.json", "generation.json", "claim.json",
          "startup.json", "reset_started.json", "initializations.jsonl", "episodes.jsonl", "scene_resets.jsonl",
          "collection.json", "bounded_collection_result.json", "proposal.json", "training_exposure_audit.json",
          "training_data_audit.json", "training_result.json", "training_request.json"}
_SCIENCE_MARKERS = {"proposal.json", "collection.json", "bounded_collection_result.json",
                    "training_exposure_audit.json", "training_data_audit.json", "training_result.json",
                    "training_request.json", "scene_resets.jsonl"}
_TERMINAL = {"completed", "completed_without_target_improvement", "interrupted"}
_PRE_RESET = {"environment_constructing", "environment_ready", "native_barrier_released"}


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise BoundaryRefused("missing, linked or oversized control receipt")
    value = read_json(path)
    if not isinstance(value, dict):
        raise BoundaryRefused("invalid control receipt shape")
    return value


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise BoundaryRefused("artifact reference escaped the research run")
    return resolved


def _controls(root: Path, *, limit: int = 20000) -> list[Path]:
    found, seen = [], 0
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        names[:] = [name for name in names if name not in _PRUNE and not name.startswith(".")
                    and not (parent / name).is_symlink()]
        seen += len(files) + len(names)
        if seen > limit:
            raise BoundaryRefused("control inventory limit exceeded; explicit reconciliation required")
        for name in files:
            relative = (parent / name).relative_to(root)
            if (name in _NAMES or re.fullmatch(r"recipe_\d+\.json", name)
                    or parent.name == "workers" and name.endswith(".json")
                    or "api" in relative.parts and name.startswith("request") and name.endswith(".json")):
                path = parent / name
                if path.is_symlink():
                    raise BoundaryRefused("linked control receipt is not trusted")
                found.append(path)
    return found


def _covered(path: Path, roots: list[Path]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _budget(path: Path, clock: Callable[[], float]) -> dict:
    row = _json(path)
    wall, gpu = row.get("wall_clock", {}), row.get("gpu_hours", {})
    started, duration = wall.get("budget_started_at_epoch"), wall.get("limit_seconds")
    if not all(type(n) in (int, float) and math.isfinite(n) for n in (started, duration)):
        raise BoundaryRefused("budget lacks a stable original wall-clock origin")
    remaining = float(started) + float(duration) - clock()
    intervals = row.get("allocation_intervals", [])
    spent = (sum(item["devices"] * max(0, (item["ended"] if item.get("ended") is not None else clock())
                                      - item["started"]) / 3600 for item in intervals)
             if intervals else float(gpu.get("charged", 0)))
    limit = gpu.get("limit")
    remaining_gpu = float(limit) - spent if limit is not None else None
    if remaining <= 0 or remaining_gpu is not None and remaining_gpu <= 0:
        raise BoundaryRefused("original persisted budget exhausted")
    return {"artifact": str(path), "sha256": digest(path), "started_epoch": started,
            "deadline_epoch": started + duration, "remaining_seconds": remaining,
            "remaining_gpu_hours": remaining_gpu}


def _commits(run_root: Path, science_hash: str, revision: str, compatibility: dict | None) -> tuple[list[Path], list[Path]]:
    roots, receipts = [], []
    for path in sorted((run_root / "harness/artifact_commits").glob("*.json")):
        row = _json(path)
        if row.get("scientific_contract_sha256") != science_hash:
            raise BoundaryRefused("artifact science differs from the frozen continuation contract")
        if row.get("state") != "committed":
            continue
        if row.get("commit_sha256") != object_digest({k: v for k, v in row.items() if k != "commit_sha256"}):
            raise BoundaryRefused("committed artifact receipt changed")
        if row.get("execution_revision") != revision and not lifecycle_compatible(
                compatibility, row.get("execution_revision"), revision, science_hash):
            raise BoundaryRefused("committed producer revision lacks lifecycle compatibility")
        if not row.get("outputs"):
            raise BoundaryRefused("committed artifact lists no outputs")
        for output in row["outputs"].values():
            target = _inside(run_root / output["path"], run_root)
            if not target.is_file() or target.stat().st_size != output["bytes"] or digest(target) != output["sha256"]:
                raise BoundaryRefused("committed output is missing or changed")
        roots.append(_inside(run_root / row["key"], run_root))
        receipts.append(path)
    return roots, receipts


def _identities(controls: list[Path], previous: dict, host: str) -> list[dict]:
    identities = [previous[name] for name in ("process", "worker") if previous.get(name)]
    for path in controls:
        if path.name not in {"process.json", "worker_identity.json", "lifecycle.json", "claim.json", "generation.json"} and path.parent.name != "workers":
            continue
        row = _json(path)
        rows = list(row.get("rows", {}).values()) if path.name == "generation.json" else [row]
        for item in rows:
            for key in ("worker_pid", "guard_pid", "owner_pid", "pid"):
                pid = item.get(key)
                if pid is None:
                    continue
                if type(pid) is not int or pid <= 0:
                    raise BoundaryRefused("invalid old worker identity")
                pgid = (item.get("worker_pgid") if key == "worker_pid" else item.get("pgid") if key == "pid" else None)
                identity = {"pid": pid, "host": item.get("host", host), "pgid": pgid,
                            "start_ticks": item.get("start_ticks", item.get("process_start_identity")),
                            "source": str(path)}
                identities.append(identity)
    unique = {object_digest({k: row.get(k) for k in ("host", "pid", "pgid", "start_ticks")}): row for row in identities}
    return list(unique.values())


def _probe_limits(run_root: Path, limit: int, clock: Callable[[], float]) -> tuple[list[dict], list[Path]]:
    views, completed = [], []
    base = run_root / "device_probe"
    directories = ([base] if base.is_dir() else []) + sorted((base / "repetitions").glob("probe_*"))
    for root in directories:
        contract_path, result_path = root / "probe_contract.json", root / "device_probe.json"
        if not contract_path.is_file():
            if (root / "attempts").is_dir():
                raise BoundaryRefused("probe generations exist without their frozen contract")
            continue
        contract = _json(contract_path)
        actual_limit = min(limit, int(contract["startup_attempts"]))
        attempts = sorted((root / "attempts").glob("attempt_*"), key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        if [p.name for p in attempts] != [f"attempt_{i}" for i in range(1, len(attempts) + 1)]:
            raise BoundaryRefused("probe generation history has a gap")
        records = []
        for index, directory in enumerate(attempts, 1):
            record, cleanup, generation = (_json(directory / name) for name in ("attempt.json", "cleanup.json", "generation.json"))
            if (int(record.get("attempt_id", -1)) != index or not record.get("generation")
                    or cleanup.get("generation") != record["generation"]
                    or generation.get("generation") != record["generation"]
                    or cleanup.get("confirmed_reaped") is not True or cleanup.get("unreaped_workers")):
                raise BoundaryRefused("probe generation is not durably reconciled and reaped")
            records.append(record)
        result = _json(result_path) if result_path.is_file() else {}
        passed = (result.get("passed") is True and bool(result.get("requested_devices"))
                  and set(result.get("verified_devices", [])) == set(result["requested_devices"]))
        if passed:
            if not records or records[-1].get("passed") is not True:
                raise BoundaryRefused("probe pass lacks its completed generation")
            completed.append(root)
            budget = None
        else:
            if len(records) >= actual_limit:
                raise BoundaryRefused("probe startup attempt budget exhausted; cannot reset it")
            if records and (records[-1].get("passed") or records[-1].get("retryable") is not True):
                raise BoundaryRefused("last probe generation is terminal")
            if any(row.get("reset_started") or row.get("steps", 0) for row in
                   (_json(attempts[-1] / "generation.json").get("rows", {}).values() if attempts else [])):
                raise BoundaryRefused("failed probe generation already began reset or rollout")
            budget = _budget(root / "budget.json", clock)
            if budget["remaining_seconds"] < 50:
                raise BoundaryRefused("original probe budget cannot afford construction and cleanup")
        views.append({"root": str(root), "passed": passed, "used_attempts": len(records),
                      "attempt_limit": actual_limit, "next_attempt": len(records) + 1 if not passed else None,
                      "probe_contract_sha256": digest(contract_path), "budget": budget,
                      "preserve_output_directory": True, "reset_budget_permitted": False})
    return views, completed


def inspect_resume_boundary(allocation_root: Path, run_root: Path, *, scientific_contract: dict,
                            execution_revision: str, previous_attempt: dict, deadline_epoch: float,
                            probe_attempt_limit: int = 3, compatibility: dict | None = None,
                            clock: Callable[[], float] = time.time) -> dict:
    """Return a persistent host receipt. No model receives the inspected raw files.

    The caller must retain the same allocation/run/probe directories and pass
    workers/resume_receipt into ContinuationSupervisor.run_once. A new probe
    launcher process directory is allowed; its logical attempts and budget are not.
    """
    allocation_root, run_root = Path(allocation_root).resolve(), Path(run_root).resolve()
    if (not run_root.is_relative_to(allocation_root) or not 1 <= probe_attempt_limit <= 3
            or not math.isfinite(deadline_epoch)):
        raise ValueError("invalid trusted allocation boundary or probe limit")
    science_hash = object_digest(scientific_contract)
    result = {"schema_version": 1, "allowed": False, "reason": "not_examined", "workers": [],
              "progress_receipts": [], "probe_limits": [], "resume_receipt": None,
              "allocation_root": str(allocation_root), "run_root": str(run_root),
              "deadline_epoch": deadline_epoch,
              "scientific_contract_sha256": science_hash, "execution_revision": execution_revision,
              "previous_attempt": previous_attempt.get("attempt"), "final_opened": False,
              "compatibility": None}
    try:
        if not math.isfinite(deadline_epoch) or deadline_epoch - clock() <= 30:
            raise BoundaryRefused("absolute allocation budget cannot afford continuation")
        state = _json(run_root / "run_state.json")
        stage = state.get("stage")
        if not isinstance(stage, str) or not stage:
            raise BoundaryRefused("research stage is unknown")
        result["stage"] = stage
        result["final_opened"] = bool(state.get("final_confirmation_opened"))
        if state.get("status") in _TERMINAL or stage in {"complete", "native_admission_complete", "dry_run_complete"}:
            raise BoundaryRefused("research is already terminal")
        if previous_attempt.get("status") not in {"failed", "failed_to_start", "reconciled_terminated"}:
            raise BoundaryRefused("previous supervisor is not a recoverable failed attempt")
        prior_revision = previous_attempt.get("execution_revision")
        if prior_revision != execution_revision and not lifecycle_compatible(
                compatibility, prior_revision, execution_revision, science_hash):
            raise BoundaryRefused("supervisor revision changed without trusted compatibility")
        protocol_path = run_root / "protocol.json"
        if protocol_path.is_file() and object_digest(scientific_identity(_json(protocol_path))) != science_hash:
            raise BoundaryRefused("run protocol differs from frozen continuation science")
        accounting = run_root / "accounting.json"
        if accounting.is_file():
            result["run_budget"] = _budget(accounting, clock)
        controls = _controls(run_root)
        host = previous_attempt.get("process", {}).get("host") or previous_attempt.get("worker", {}).get("host")
        if not host:
            raise BoundaryRefused("prior allocation host identity is missing")
        workers = _identities(controls, previous_attempt, host)
        result["workers"] = workers
        if any(not process_gone(worker) for worker in workers):
            raise BoundaryRefused("old worker, guardian, dispatcher or process group is alive or unknown")
        committed, receipts = _commits(run_root, science_hash, execution_revision, compatibility)
        result["progress_receipts"] = [str(path) for path in receipts]
        probe_views, passed_probes = _probe_limits(run_root, probe_attempt_limit, clock)
        result["probe_limits"] = probe_views
        if prior_revision != execution_revision and any(not view["passed"] for view in probe_views):
            raise BoundaryRefused("probe execution contract migration is not implemented; preserve attempts and stop")
        probe_root = run_root / "device_probe"
        native_roots = []
        scientific_incomplete = False
        for path in controls:
            if _covered(path, committed):
                continue
            is_probe = path.is_relative_to(probe_root)
            if is_probe:
                parts = path.relative_to(probe_root).parts
                owning_probe = (probe_root / "repetitions" / parts[1]
                                if len(parts) > 1 and parts[0] == "repetitions" else probe_root)
                if owning_probe in passed_probes:
                    continue
            if path.name in {"reset_started.json", "initializations.jsonl", "episodes.jsonl"}:
                if path.name == "reset_started.json" or path.stat().st_size:
                    raise BoundaryRefused("uncommitted reset or episode progress forbids automatic replay")
            if path.name == "lifecycle.json":
                lifecycle = _json(path)
                if lifecycle.get("reset_started") or lifecycle.get("steps", 0):
                    raise BoundaryRefused("uncommitted native lifecycle already entered scientific execution")
            if path.name == "startup.json":
                startup = _json(path)
                if startup.get("phase") == "evaluation_reset_started" or startup.get("policy_has_acted"):
                    raise BoundaryRefused("uncommitted startup receipt shows reset or policy action")
                if (path.parent / "process/process.json").is_file():
                    native_roots.append(path.parent)
            if not is_probe and (path.name in _SCIENCE_MARKERS or re.fullmatch(r"recipe_\d+\.json", path.name)
                                 or "api" in path.relative_to(run_root).parts and path.name.startswith("request")):
                scientific_incomplete = True
            if path.name == "process.json" and not is_probe:
                relative = path.relative_to(run_root).parts
                if "collection" in relative or any(part.startswith("process_train_") for part in relative):
                    scientific_incomplete = True
        if scientific_incomplete:
            raise BoundaryRefused("uncommitted collection, training or proposal work requires explicit phase reconciliation")
        startup_failures = [root for root in native_roots if retryable_startup(root)]
        policy_path = run_root / "harness_policy.json"
        native_limit = (_json(policy_path).get("lifecycle", {}).get("startup_attempts", 3)
                        if policy_path.is_file() else 3)
        if type(native_limit) is not int or not 1 <= native_limit <= 3:
            raise BoundaryRefused("invalid frozen native startup attempt limit")
        for root in startup_failures:
            if root.is_relative_to(probe_root):
                continue
            match = re.fullmatch(r"startup_attempt_(\d+)", root.name)
            used = int(match[1]) if match else 1
            if used >= native_limit:
                raise BoundaryRefused("native startup attempt budget exhausted; cannot renew it outside Runtime")
        final = result["final_opened"]
        if final and stage not in {"final_confirmation", "export_validation"}:
            raise BoundaryRefused("final opened: cannot return to a research stage")
        if final:
            expected_fragment = "final_confirmation" if stage == "final_confirmation" else "export"
            relevant = [root for root in startup_failures if expected_fragment in str(root.relative_to(run_root))]
            if not relevant:
                raise BoundaryRefused("sealed-stage continuation lacks a certified pre-reset failure")
        elif not startup_failures:
            match = re.fullmatch(r"round_(\d+)_(?:completed|quarantined)", stage)
            boundary_stage = bool(match and run_root / "rounds" / f"round_{match[1]}" in committed)
            if not boundary_stage:
                raise BoundaryRefused("no certified startup failure or fully committed phase boundary")
        if stage.startswith("device_probe") and not any(not view["passed"] for view in probe_views):
            raise BoundaryRefused("probe startup recovery lacks preserved attempt and budget records")
        receipt = {"schema_version": 1, "stage": stage, "safe_boundary": True, "episode_started": False,
                   "scientific_contract_sha256": science_hash, "execution_revision": execution_revision,
                   "final_opened": final, "previous_workers_reaped": True,
                   "recovery_scope": "before_first_reset" if startup_failures else "committed_phase_boundary",
                   "startup_evidence": [{"startup_sha256": digest(root / "startup.json"),
                                         "process_sha256": digest(root / "process/process.json")}
                                        for root in startup_failures]}
        result.update(allowed=True, reason="verified_safe_boundary", resume_receipt=receipt)
        if compatibility is not None and prior_revision != execution_revision:
            result["compatibility"] = {key: compatibility[key] for key in (
                "passed", "scope", "from_revision", "to_revision", "scientific_contract_sha256",
                "validation_receipt_sha256")}
            result["compatibility"]["previous_workers_reaped"] = True
    except (BoundaryRefused, OSError, ValueError, KeyError, TypeError) as exc:
        result["reason"] = str(exc) if isinstance(exc, BoundaryRefused) else f"invalid_control_artifact:{type(exc).__name__}"
    result["evidence_sha256"] = object_digest(result)
    if result["resume_receipt"] is not None:
        result["resume_receipt"]["boundary_evidence_sha256"] = result["evidence_sha256"]
    destination = allocation_root / "continuation/boundaries" / (result["evidence_sha256"] + ".json")
    result["receipt_path"] = str(destination)
    atomic_json(destination, result)
    return {**result, "receipt_sha256": digest(destination)}


def load_resume_receipt(path: Path, *, expected_sha256: str, run_root: Path,
                        scientific_contract: dict, execution_revision: str,
                        clock: Callable[[], float] = time.time) -> dict:
    """Validate the host-selected resume envelope before admitting a new process.

    AUTOSIM_RESUME_RECEIPT + AUTOSIM_RESUME_RECEIPT_SHA256 can carry these
    arguments, but the caller supplies run/science/revision from its frozen
    manifest. Environment values alone are not a scientific authority.
    """
    path = Path(path)
    row = _json(path)
    if digest(path) != expected_sha256 or row.get("allowed") is not True:
        raise BoundaryRefused("resume envelope is unapproved or changed")
    if (Path(row.get("run_root", "")).resolve() != Path(run_root).resolve()
            or row.get("scientific_contract_sha256") != object_digest(scientific_contract)
            or row.get("execution_revision") != execution_revision):
        raise BoundaryRefused("resume envelope run, science or revision does not match")
    allocation = Path(row["allocation_root"]).resolve()
    if not path.resolve().is_relative_to(allocation / "continuation/boundaries"):
        raise BoundaryRefused("resume envelope is outside the trusted host journal")
    receipt = row.get("resume_receipt") or {}
    if (receipt.get("safe_boundary") is not True or receipt.get("episode_started") is not False
            or receipt.get("scientific_contract_sha256") != object_digest(scientific_contract)
            or receipt.get("execution_revision") != execution_revision):
        raise BoundaryRefused("resume envelope contains no compatible safe boundary")
    if row.get("deadline_epoch", 0) <= clock():
        raise BoundaryRefused("resume envelope's original deadline expired")
    if any(not process_gone(worker) for worker in row.get("workers", [])):
        raise BoundaryRefused("old worker became live or termination is unknown")
    return row
