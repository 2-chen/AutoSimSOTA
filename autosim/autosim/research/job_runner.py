"""Run one phase's independent jobs over the run's device set.

A *phase* is a set of jobs nothing inside it depends on: a round's targeted collection
beside its original collection, the official baseline beside the candidate evaluations,
the three final-confirmation evaluations.  Everything *between* phases stays a dependency
chain -- multi-device work never reorders scientific dependencies, it only stops
independent work from queueing behind work it does not need.

Adaptive execution routes one, many, and zero-GPU jobs through ``Scheduler``. Legacy
single-device calls retain their sequential path. The scheduler records the devices
held, reasons work could not start, and measured cost.

Two deliberate properties:

* **A completed job is not re-run.** ``Scheduler.restore`` reads ``outcome.json`` /
    ``failure.json``; adaptive results are recovered from the committed typed cache
  without calling the worker, and a failed job stays failed -- a
  score-bearing evaluation is never silently retried by a resume.
* **A phase fails as a whole.** The first failure is recorded against its job and raised
  once the other jobs have finished, so the run cannot continue on a partial phase while
  other jobs are still writing into it.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from .common import now, atomic_json, read_json, exclusive, object_digest
from .devices import SimDeviceSelection, select
from .scheduler import Job, Scheduler, ScheduleRefused
from .resource_contracts import ResourceRequest

# A job's own wall-clock bound, used for the budget gate only; a real episode evaluation is
# minutes of environment construction plus per-episode marginal cost (see devices.shard_count).
DEFAULT_RESERVE_SECONDS = 900.0


@dataclass(frozen=True)
class PhaseJob:
    """One job in a phase: what to run, and what it is expected to cost."""

    name: str
    category: str
    run: Callable[[Any], Any]
    estimate_seconds: float = 0.0
    reserve_seconds: float = DEFAULT_RESERVE_SECONDS
    kind: str = ""
    device_count: int = 1
    depends_on: tuple[str, ...] = ()
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    input_digest: str = ""
    code_digest: str = ""
    profile_digest: str = ""
    estimate_by_class: dict[str, float] = field(default_factory=dict)

    def as_job(self) -> Job:
        return Job(self.name, category=self.category, kind=self.kind or self.category,
                   estimate_seconds=self.estimate_seconds, reserve_seconds=self.reserve_seconds,
                   device_count=self.device_count, depends_on=self.depends_on,
                   resources=self.resources, input_digest=self.input_digest,
                   code_digest=self.code_digest, estimate_by_class=self.estimate_by_class)


def plan_devices(plan: dict) -> list[dict]:
    """The plan's usable rows, each carrying the addressing it froze."""
    return [dict(device) for device in (plan.get("usable") or [])]


def schedule_limit(plan: dict, max_parallel_jobs: int | None) -> int:
    """How many heavy jobs may run at once: the plan's cap, never wider than the plan."""
    requested = plan.get("max_parallel_jobs") if max_parallel_jobs is None else max_parallel_jobs
    return max(1, min(int(requested or 1), len(plan_devices(plan)) or 1))


def is_scheduled(plan: dict | None, max_parallel_jobs: int | None) -> bool:
    """True only for a multi-device plan that permits more than one heavy job.

    Anything else -- no plan (every legacy invocation), a plan equivalent to the legacy
    single-device run, or ``--max-parallel-jobs 1`` -- takes the sequential path and
    produces exactly the artifacts it produced before this module existed.
    """
    if plan is not None and plan.get("unified_scheduler"):
        return True
    if plan is None or plan.get("legacy_equivalence"):
        return False
    if len(plan_devices(plan)) < 2:
        return False
    return schedule_limit(plan, max_parallel_jobs) > 1


def episodes_of(result: Any) -> int | None:
    """Charge the job with the episodes it actually produced, when it reports them."""
    if not isinstance(result, dict):
        return None
    rows = result.get("episodes")
    if isinstance(rows, list):
        return len(rows)
    for key in ("accepted_episodes", "accepted", "episodes"):
        value = result.get(key)
        if isinstance(value, int):
            return value
    return None


def _selection_of(device: dict, mode: str) -> SimDeviceSelection:
    frozen = device.get("selection")
    return SimDeviceSelection(**frozen) if frozen else select(device, mode)


def _encode(value):
    if isinstance(value, Path):
        return {"__autosim_path__": str(value)}
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__autosim_path__"}:
            return Path(value["__autosim_path__"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def run_phase(*, phase: str, jobs: Sequence[PhaseJob], runtime, plan: dict | None,
              output: Path, max_parallel_jobs: int | None = None, ledger=None,
              run_id: str = "run", lease_root: Path | None = None) -> dict[str, Any]:
    """Run every job, on the plan's devices when there is more than one, else in order.

    Returns ``{job name: result}``.  Raises ``RuntimeError`` naming each failure once the
    phase has stopped -- a phase is not partially usable, because the pipeline's next step
    reads all of its outputs.
    """
    if is_scheduled(plan, max_parallel_jobs):
        with exclusive(Path(output) / phase / "dispatcher.lock"):
            if plan.get("unified_scheduler"):
                from .accounting import UtilizationSampler
                sampler = UtilizationSampler(interval_seconds=2)
                try:
                    with sampler:
                        return _run_phase(phase=phase, jobs=jobs, runtime=runtime, plan=plan,
                            output=output,max_parallel_jobs=max_parallel_jobs,ledger=ledger,
                            run_id=run_id,lease_root=lease_root,telemetry_sampler=sampler)
                finally:
                    atomic_json(Path(output) / phase / "telemetry.json", sampler.summary())
            return _run_phase(phase=phase, jobs=jobs, runtime=runtime, plan=plan,
                              output=output, max_parallel_jobs=max_parallel_jobs,
                              ledger=ledger, run_id=run_id, lease_root=lease_root)
    return _run_phase(phase=phase, jobs=jobs, runtime=runtime, plan=plan, output=output,
                      max_parallel_jobs=max_parallel_jobs, ledger=ledger, run_id=run_id,
                      lease_root=lease_root)


def _run_phase(*, phase, jobs, runtime, plan, output, max_parallel_jobs, ledger, run_id, lease_root,
               telemetry_sampler=None):
    jobs = list(jobs)
    names = [job.name for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate job name in phase {phase}: {names}")
    if not is_scheduled(plan, max_parallel_jobs):
        return {job.name: job.run(runtime) for job in jobs}

    devices = plan_devices(plan)
    mode = plan.get("mode") or "pinned_index"
    limit = schedule_limit(plan, max_parallel_jobs)
    root = Path(output) / phase
    from .profiling import ProfileStore
    profiles = ProfileStore(Path(output).parent / "profiles")
    environment_id = object_digest({"host": (plan.get("host_capacity") or {}).get("host"),
                                   "driver": [d.get("driver_version") for d in devices],
                                   "probe": plan.get("probe_receipt_key")})
    def profile_context(job, device_class):
        return {"workload": job.kind or job.category,
                "workload_shape_digest": job.profile_digest or job.input_digest,
                "cpu_cores":job.resources.cpu_cores,"parallel_limit":limit,
                "code_digest": job.code_digest, "environment_id": environment_id,
                "device_class": device_class}
    if plan.get("unified_scheduler"):
        enriched = []
        for job in jobs:
            estimates = dict(job.estimate_by_class)
            for device_class in {d.get("class_name") or "unknown" for d in devices}:
                measured = profiles.estimate(profile_context(job, device_class))
                if measured["confidence"] == "measured":
                    estimates[device_class] = measured["p90"]
            enriched.append(replace(job, estimate_by_class=estimates))
        jobs = enriched
    effective_host = dict(plan.get("host_capacity") or {})
    io_limit = plan.get("execution_parameters", {}).get("io_slots")
    if plan.get("unified_scheduler") and io_limit is not None:
        effective_host["io_slots"] = min(effective_host.get("io_slots", 0), int(io_limit))
    scheduler = Scheduler(jobs=[job.as_job() for job in jobs], devices=devices, output=root,
                          ledger=ledger, max_parallel_jobs=limit, run_id=run_id,
                          lease_root=lease_root, host_capacity=effective_host or None)
    by_name = {job.name: job for job in jobs}
    restored = scheduler.restore()
    results: dict[str, Any] = {}
    pending: list[PhaseJob] = []
    for job in jobs:
        state = scheduler.state(job.name)
        if state == "completed":
            # Its own cache answers this without touching a device; the scheduler's record
            # is the reason we know it is safe to ask rather than to re-run.
            cache = scheduler.job_output(job.name) / "result.json"
            if cache.exists() and plan.get("unified_scheduler"):
                results[job.name] = _decode(read_json(cache))
            elif plan.get("unified_scheduler"):
                raise RuntimeError(f"completed job is missing its result: {job.name}")
            else:
                results[job.name] = job.run(runtime)
        elif state == "failed":
            raise RuntimeError(
                f"phase {phase} job {job.name} is recorded failed; refusing to retry "
                f"score-bearing work ({scheduler.failure_path(job.name)})")
        else:
            pending.append(job)
    if not pending:
        scheduler.write_snapshot()
        return results

    failures: dict[str, str] = {}
    try:
        cpu_slots = max(1, int(scheduler.host_capacity.get("cpu_cores") or 1))
        with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), limit + cpu_slots))) as pool:
            futures: dict[Any, Any] = {}
            while pending or futures:
                for assignment in scheduler.plan_step()["assignments"]:
                    job = by_name[assignment.job.name]
                    device = assignment.devices[0] if assignment.devices else None
                    bound = runtime.for_job(_selection_of(device, mode) if device else None, job=job.name,
                                            output=scheduler.job_output(job.name))
                    if hasattr(bound, "execution_resources"):
                        bound.execution_resources = {**job.resources.as_dict(), "gpu_count": job.device_count}
                        bound.assigned_devices = tuple(assignment.devices)
                    # The lease is taken *before* the work is submitted: a job that starts
                    # running first and leases second would be, for that window, work on a
                    # device the run does not hold.
                    scheduler.start(assignment)
                    if hasattr(bound, "lease_journal"):
                        bound.lease_journal = str(scheduler.job_output(job.name) / "workers")
                    futures[pool.submit(job.run, bound)] = assignment
                    pending.remove(job)
                if not futures:
                    blocked = scheduler.plan_step()["waiting"]
                    raise ScheduleRefused(
                        f"phase {phase} cannot place any job: "
                        + "; ".join(f"{item.job.name} waits for {item.reason} ({item.detail})"
                                    for item in blocked))
                done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    assignment = futures.pop(future)
                    name = assignment.job.name
                    try:
                        result = future.result()
                        atomic_json(scheduler.job_output(name) / "result.json", _encode(result))
                    except Exception as exc:          # recorded, then raised as a group
                        failures[name] = f"{type(exc).__name__}: {exc}"
                        scheduler.complete(name, status="failed",
                                           detail={"phase": phase, "error": failures[name][:400]})
                    except BaseException:
                        raise                          # Ctrl-C: the abandon path below
                    else:
                        results[name] = result
                        outcome = scheduler.complete(name, episodes=episodes_of(result),
                                           detail={"phase": phase,
                                                   "devices": [str(device["uuid"])
                                                               for device in assignment.devices]})
                        if plan.get("unified_scheduler"):
                            classes = {d.get("class_name") or "unknown" for d in assignment.devices} or {"cpu"}
                            for device_class in classes:
                                profiles.record(profile_context(by_name[assignment.job.name], device_class),
                                                seconds=outcome["wall_seconds"],
                                                metrics={"gpu_count": len(assignment.devices),
                                                         "episodes": episodes_of(result),
                                                         "phase_gpu_peak_upper_bound": telemetry_sampler.summary()
                                                         if telemetry_sampler else {},
                                                         "runtime_timings": result.get("timings", {})
                                                         if isinstance(result,dict) else {}})
                scheduler.write_snapshot()
    except BaseException:
        # Nothing is left holding a device or looking runnable once the phase is over.
        for name in list(scheduler.running):
            with suppress(Exception):
                scheduler.abandon(name, reason=f"phase {phase} aborted")
        raise
    scheduler.decisions.append({"phase": phase, "finished_at": now(), "restored": restored,
                                "completed": sorted(results), "failed": sorted(failures)})
    scheduler.write_snapshot()
    if failures:
        raise RuntimeError(f"phase {phase} jobs failed: {failures}")
    return results
