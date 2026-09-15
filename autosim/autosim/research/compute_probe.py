"""Bounded allocation-wide native admission using the shared worker lifecycle."""
from __future__ import annotations

import argparse
import os
import threading
import time
import uuid
from pathlib import Path

from .accounting import BudgetLedger, UtilizationSampler
from .common import atomic_json, digest, immutable_json, now, read_json
from .device_cli import receipt_lookup_key
from .device_probe import judge, run_arm
from .devices import discover, identity_is_safe, select, write_probe_receipt
from .job_runner import PhaseJob, run_phase
from .native_lifecycle import retryable_startup
from .probe_coordinator import ProbeCoordinator
from .registry import load_task
from .resource_contracts import ResourceRequest, host_inventory
from .runtime import Runtime, startup_receipt


def retryable_generation(results: dict, primary: dict | None, root: Path) -> bool:
    """Known pre-reset primary failures may recover; peers are consequences."""
    if not primary or not primary.get("worker"):
        return False
    for row in results.values():
        directory = Path(row.get("artifact_directory", root / "missing"))
        if (directory / "reset_started.json").exists() or (directory / "initializations.jsonl").exists():
            return False
    row = results.get(primary["worker"], {})
    if not row:
        return False
    directory = Path(row["artifact_directory"])
    if retryable_startup(directory):
        return True
    process, startup = directory / "process/process.json", directory / "startup.json"
    # A coordinator's deadline can terminate a still-live native constructor before
    # run_command's slightly later process timeout; retain this measured distinction.
    return (primary.get("reason") == "native_startup_timeout" and process.is_file() and startup.is_file()
            and read_json(process).get("returncode") in {-15, -9}
            and read_json(startup).get("phase") in {"environment_constructing", "environment_ready",
                                                   "native_barrier_released"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="click_bell")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--mode", choices=("identity", "pinned_index"), default="identity")
    parser.add_argument("--smoke-timeout", type=int, default=900)
    parser.add_argument("--gpu-hours", type=float)
    parser.add_argument("--startup-attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--incident-timeout", type=float, default=1800)
    parser.add_argument("--construction-concurrency", type=int)
    parser.add_argument("--minimum-overlap-seconds", type=float, default=5)
    parser.add_argument("--harness-config", type=Path)
    parser.add_argument("--cache-mode", choices=("cold", "warm"), default="cold")
    parser.add_argument("--cache-namespace")
    parser.add_argument("--probe-repetition", type=int, default=0)
    parser.add_argument("--recovery-max-calls", type=int, default=8)
    parser.add_argument("--recovery-max-tokens", type=int, default=80000)
    args = parser.parse_args()
    if args.cache_mode == "warm" and not args.cache_namespace:
        parser.error("warm probes require the previous successful receipt's --cache-namespace")
    if args.smoke_timeout < 30 or args.incident_timeout < 30 or args.minimum_overlap_seconds < 0:
        parser.error("invalid native timing budget")
    if args.recovery_max_calls < 0 or args.recovery_max_tokens < 0:
        parser.error("invalid recovery API budget")
    config = read_json(args.harness_config) if args.harness_config else {}
    validation = config.get("validation", {})
    injection = validation.get("inject_failure_worker")
    if injection is not None and validation.get("stage") != "native_admission":
        parser.error("fault injection is restricted to native_admission validation")
    if args.probe_repetition not in range(3):
        parser.error("probe repetition must be in [0, 2]")
    requested_injection = injection
    # One explicitly requested fault per validation run. A warm repetition must
    # not be silently converted to another recovered cold-cache attempt.
    if args.probe_repetition:
        injection = None
    args.output.mkdir(parents=True, exist_ok=True)
    report = discover(disk_path=args.output)
    atomic_json(args.output / "inventory.json", report)
    if args.mode == "identity":
        safe, reason = identity_is_safe(report)
        if not safe:
            raise RuntimeError(f"identity addressing unavailable: {reason}")
    devices = [d for d in report["gpus"] if d["index"] in report["allowed"]]
    if args.gpus != "auto":
        requested = {int(part) for part in args.gpus.split(",")}
        if not requested <= {d["index"] for d in devices}:
            raise ValueError("requested probe GPU lies outside the container allocation")
        devices = [d for d in devices if d["index"] in requested]
    if not devices:
        raise RuntimeError("no visible GPU in the container allocation")
    if injection is not None and (type(injection) is not int or not 0 <= injection < len(devices)):
        raise ValueError("injected worker lies outside the probe allocation")
    host = host_inventory(args.output)
    cores = max(1, min(4, int(host["cpu_cores"] // len(devices))))
    if host["ram_mib"] < len(devices) * 4096 or host["cpu_cores"] < len(devices):
        raise RuntimeError("host allocation cannot support one native worker per GPU")
    for device in devices:
        device["selection"] = select(device, args.mode).as_dict()
    plan = {"unified_scheduler": True, "usable": devices, "mode": args.mode,
            "max_parallel_jobs": len(devices), "host_capacity": host}
    ledger = BudgetLedger(args.output / "budget.json", wall_limit_seconds=args.incident_timeout,
                          gpu_hours_limit=min(args.gpu_hours, len(devices)*args.incident_timeout/3600)
                          if args.gpu_hours is not None else len(devices)*args.incident_timeout/3600, devices=devices)
    deadline = time.monotonic() + ledger.remaining_wall
    runtime = Runtime(args.workspace, args.output, plan=plan, deadline=deadline, harness_config=config)
    spec = load_task(runtime.repo, args.task)
    checkpoint = runtime.repo / f"checkpoints/ACT_sim_{args.task}"
    immutable_json(args.output / "probe_contract.json", {"task": args.task, "workspace": str(args.workspace),
        "gpus": args.gpus, "mode": args.mode, "startup_attempts": args.startup_attempts,
        "construction_concurrency": args.construction_concurrency, "minimum_overlap_seconds": args.minimum_overlap_seconds,
        "cache_mode": args.cache_mode, "cache_namespace": args.cache_namespace,
        "probe_repetition": args.probe_repetition,
        "recovery_max_calls": args.recovery_max_calls, "recovery_max_tokens": args.recovery_max_tokens,
        "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
        "execution_digest": runtime._execution_code_digest(), "harness_config": config})
    prior_receipt = args.output / "device_probe.json"
    if prior_receipt.is_file():
        prior = read_json(prior_receipt)
        current_key = receipt_lookup_key(report, image=os.environ.get("AUTOSIM_IMAGE"),
                                        worker_spec=os.environ.get("SCO_WORKER_SPEC"))
        if prior.get("probe_receipt_key") != current_key:
            raise RuntimeError("prior native admission belongs to another allocation; use a new output")
        if prior.get("passed"):
            return 0
    sampler = UtilizationSampler(interval_seconds=2)
    sampler._sample_once()
    baseline = sampler.summary()
    names = [f"native_{i}" for i in range(len(devices))]
    attempts, results, overlap = [], {}, {}
    for directory in sorted((args.output / "attempts").glob("attempt_*"),
                            key=lambda p: int(p.name.rsplit("_", 1)[-1])):
        record = directory / "attempt.json"
        if not record.is_file():
            raise RuntimeError(f"interrupted probe generation requires cleanup reconciliation: {directory}")
        attempts.append(read_json(record))
    if attempts and (attempts[-1].get("passed") or not attempts[-1].get("retryable")):
        raise RuntimeError("committed native admission is terminal; use a new explicitly budgeted validation")
    if attempts:
        results = {row["name"]: row for row in attempts[-1]["arms"]}
        overlap = attempts[-1]["concurrency_evidence"]
    error = None
    cache_namespace = None
    effective_cache_mode = args.cache_mode
    construction_concurrency = args.construction_concurrency or len(devices)
    for prior in attempts:
        action_path = Path(prior["artifact_directory"]) / "recovery_action.json"
        if action_path.is_file():
            action = read_json(action_path)
            if action.get("status") == "stopped":
                raise RuntimeError("recorded infrastructure Agent stop is terminal")
            selected = action.get("next_construction_concurrency")
            if type(selected) is not int or not 1 <= selected <= construction_concurrency:
                raise RuntimeError("recorded recovery construction limit is invalid")
            construction_concurrency = selected
    with sampler:
        for attempt in range(len(attempts) + 1, args.startup_attempts + 1):
            remaining = deadline - time.monotonic() - 20
            if remaining < 30:
                error = "incident_deadline_exhausted"
                break
            timeout = min(args.smoke_timeout, int(remaining))
            wave_root = args.output / "attempts" / f"attempt_{attempt}"
            effective_cache_mode = args.cache_mode if attempt == 1 else "cold"
            cache_namespace = args.cache_namespace if effective_cache_mode == "warm" else uuid.uuid4().hex
            bound_config = {**config, "_native_cache_namespace": cache_namespace}
            atomic_json(wave_root / "cache_identity.json", {"mode": effective_cache_mode,
                        "native_cache_namespace": cache_namespace})
            runtime.harness_config = bound_config
            cuda_barrier = threading.Barrier(len(devices), timeout=min(120, timeout))
            results = {}
            with ProbeCoordinator(wave_root, names, timeout_seconds=timeout,
                    construction_concurrency=construction_concurrency,
                    minimum_overlap_seconds=args.minimum_overlap_seconds, attempt_id=attempt) as coordinator:
                def probe(bound):
                    coordinator.bind(bound.job, bound.selection.uuid)
                    bound.native_barrier = coordinator.worker_configuration(bound.job)
                    if (injection is not None and bound.job == f"native_{injection}"
                            and attempt == validation.get("inject_failure_generation", 1)):
                        bound.native_barrier.update(validation_injection=True, inject_before_ready=True)
                    try:
                        bound.run([str(bound.python), "-m", "autosim.research.validation_worker", "--uuid",
                                   bound.selection.uuid, "--output", str(bound.output / "cuda_train.json"),
                                   "--steps", "50", "--training"], bound.output / "cuda_train_process",
                                  min(120, timeout), selection=bound.training_selection())
                        cuda_barrier.wait()
                        result = run_arm(name=bound.job, runtime=bound, spec=spec, checkpoint=checkpoint,
                            output=bound.output / "native", master_seed=2000000001,
                            smoke_timeout=timeout, selection=bound.selection, sampler=sampler)
                        result.pop("_baseline", None)
                        result.pop("_peak", None)
                        result["cuda_verified"] = True
                        if result.get("status") != "completed":
                            coordinator.cancel(bound.job, "native_worker_failed")
                        return result
                    except Exception as exc:
                        cuda_barrier.abort()
                        coordinator.cancel(bound.job, "cuda_or_dispatch_failure")
                        return {"name": bound.job, "status": "failed", "error_type": type(exc).__name__,
                                "error": str(exc)[:400], "selection": bound.selection.as_dict(),
                                "artifact_directory": str(bound.output / "native"), "startup_census": [],
                                "cuda_verified": False}
                results = run_phase(phase="native_probe", runtime=runtime, plan=plan,
                    output=wave_root / "schedule", ledger=ledger,
                    run_id=f"probe_{coordinator.generation}",
                    jobs=[PhaseJob(name, "probe", probe, reserve_seconds=0, estimate_seconds=timeout,
                          resources=ResourceRequest(cpu_cores=cores, ram_mib=4096, per_gpu_memory_mib=2048))
                          for name in names])
                overlap = coordinator.overlap_receipt()
            retryable = retryable_generation(results, overlap.get("primary_failure"), wave_root)
            attempt_record = {"attempt_id": attempt, "generation": coordinator.generation,
                "passed": overlap["passed"] and all(row.get("status") == "completed" for row in results.values()),
                "primary_failure": overlap.get("primary_failure"), "retryable": retryable,
                "cache_mode": effective_cache_mode, "native_cache_namespace": cache_namespace,
                "artifact_directory": str(wave_root), "concurrency_evidence": overlap,
                "arms": list(results.values())}
            attempts.append(attempt_record)
            atomic_json(wave_root / "attempt.json", attempt_record)
            if attempt_record["passed"] or not retryable:
                break
            # Cleanup has already completed. The API can select a more conservative
            # construction strategy; it cannot extend this incident or re-run a score.
            if attempt < args.startup_attempts and deadline - time.monotonic() >= 185:
                try:
                    from .infrastructure_recovery import choose_probe_recovery
                    primary = overlap.get("primary_failure") or {}
                    failed_row = results.get(primary.get("worker"), {})
                    incident = {"scope": "native_startup", "generation": coordinator.generation,
                        "attempt_id": attempt, "remaining_attempts": args.startup_attempts - attempt,
                        "primary_failure": primary, "reset_started": False,
                        "startup_receipt": startup_receipt(Path(failed_row["artifact_directory"]))}
                    decision = choose_probe_recovery(incident=incident,
                        output=wave_root / "infrastructure_recovery",
                        current_construction_concurrency=construction_concurrency, max_workers=len(devices),
                        remaining_seconds=deadline - time.monotonic(),
                        max_calls=args.recovery_max_calls, max_tokens=args.recovery_max_tokens,
                        env_file=Path(os.environ["AUTOSIM_ENV_FILE"]) if os.environ.get("AUTOSIM_ENV_FILE") else None)
                    if decision.get("action") == "stop":
                        error = "infrastructure_recovery_stopped"
                        atomic_json(wave_root / "recovery_action.json", {"status": "stopped",
                                    "primary_failure": primary, "budget_extended": False})
                        break
                    selected = decision.get("next_construction_concurrency", construction_concurrency)
                    if type(selected) is not int or not 1 <= selected <= construction_concurrency:
                        raise ValueError("infrastructure recovery exceeded construction limits")
                    construction_concurrency = selected
                    recovery = {"status": "applied", "next_construction_concurrency": selected,
                                "fresh_cache": True}
                except Exception as exc:
                    recovery = {"status": "deterministic_fallback", "error_type": type(exc).__name__,
                                "next_construction_concurrency": construction_concurrency, "fresh_cache": True}
                atomic_json(wave_root / "recovery_action.json", recovery)
    uuids = [device["uuid"] for device in devices]
    verified = []
    for result in results.values():
        own = result["selection"]["uuid"]
        result["judged"] = judge(result, own_uuid=own, allocation=uuids, baseline=baseline,
                                 peak=sampler.summary(), idle_uuids=[], census=result.get("startup_census", []))
        if result["judged"]["passed"]:
            verified.append(own)
    passed = set(verified) == set(uuids) and bool(overlap.get("passed"))
    primary = overlap.get("primary_failure")
    if primary and primary.get("worker") in results:
        primary = {**primary, "startup_receipt": startup_receipt(Path(results[primary["worker"]]["artifact_directory"]))}
    cuda_verified = {r["selection"]["uuid"] for r in results.values() if r.get("cuda_verified")}
    injection_evidence = []
    for attempt_record in attempts:
        for arm in attempt_record.get("arms", []):
            if injection is None or arm.get("name") != f"native_{injection}":
                continue
            directory = Path(arm["artifact_directory"])
            marker, process = directory / "injected_failure.json", directory / "process/process.json"
            if marker.is_file() and process.is_file():
                observed = read_json(marker)
                if (observed.get("generation") == attempt_record["generation"]
                        and observed.get("worker") == arm["name"]
                        and observed.get("uuid") == arm["selection"]["uuid"]
                        and observed.get("reset_started") is False
                        and read_json(process).get("returncode") == -11):
                    injection_evidence.append({"generation": attempt_record["generation"],
                        "worker": arm["name"], "artifact": str(marker), "sha256": digest(marker),
                        "process_artifact": str(process), "process_sha256": digest(process), "returncode": -11})
    receipt = {"schema_version": 3, "mode": args.mode, "passed": passed,
        "allocation": devices, "requested_devices": uuids, "verified_devices": verified,
        "inventory_evidence": {"artifact": str(args.output / "inventory.json"),
            "sha256": digest(args.output / "inventory.json"), "host": report.get("host"),
            "requested_indices": [device["index"] for device in devices]},
        "arms": list(results.values()), "startup_attempts": attempts,
        "devices": {u: "verified_sim" if u in verified else "unknown" for u in uuids},
        "device_capabilities": {u: {"train": "verified" if u in cuda_verified else "unknown",
            "cuda": "verified" if u in cuda_verified else "unknown",
            "physics": "verified" if u in verified else "unknown",
            "render": "verified" if u in verified else "unknown",
            "policy_inference": "verified" if u in verified else "unknown"} for u in uuids},
        "verified_concurrent_workers": len(verified) if passed else 0,
        "verified_rollout_concurrency": len(verified) if passed else 0,
        "construction_concurrency": construction_concurrency,
        "concurrency_evidence": overlap, "cache_mode": effective_cache_mode,
        "primary_failure": primary,
        "validation_injection": {"requested_worker": requested_injection,
            "repetition": args.probe_repetition, "applied": bool(injection_evidence),
            "evidence": injection_evidence},
        "native_cache_namespace": cache_namespace, "finished_at": now(), "error": error,
        "scope": "native one-episode workers with measured rollout overlap; ACT DDP separately gated"}
    key = receipt_lookup_key(report, image=os.environ.get("AUTOSIM_IMAGE"), worker_spec=os.environ.get("SCO_WORKER_SPEC"))
    receipt["probe_receipt_key"] = key
    atomic_json(args.output / "device_probe.json", receipt)
    if passed:
        write_probe_receipt(args.workspace, key, receipt)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
