"""Which device a stage runs on, decided from the machine rather than assumed.

The derived-execution path had no compute decision at all: `device` came from the run's
settings and defaulted to `"cpu"`. That default is not neutral. It is a claim about a machine
the code has never seen -- and on the machine this system actually runs on it was wrong by
the whole wall clock. LIBERO's own configuration says `cuda`; a baseline that takes about
ninety seconds on the card ran for over eleven hours on the CPU nobody asked for, and had not
finished. A default that overrides the benchmark's is not conservative, it is a different
experiment.

The resource table this needs already existed. `devices.discover()` reports every GPU with
its UUID, model, memory, utilization and the PIDs currently on it; which cards the container
actually exposes to CUDA; an outer `CUDA_VISIBLE_DEVICES` filter; whether nvidia-smi's
ordering and torch's disagree; cgroup-aware CPU quota and memory; and NVLink topology. None
of it was being consulted on this path. So this is not a new scheduler -- it is the decision
that was missing, made from evidence that was already being gathered.

Three things it deliberately does not do.

It does not probe capability. `device_probe` answers a harder question -- does a *smoke test
of this engine* run on this card -- and pays for it with a real render. A stage is a different
kind of work, and the cheap table is enough to choose; when a stage fails on the device it was
given, `reconsider` is where that is handled, from the failure rather than from a guess.

It does not assume a GPU exists. A machine with none is a machine this runs on, so the CPU is
a real answer with a real reason attached, and the reason is recorded -- the difference
between choosing the CPU and defaulting to it is the whole of this module.

It does not silently narrow what a process can see. `CUDA_VISIBLE_DEVICES` is set to the one
leased card, and the ordinal a bare `"cuda"` resolves to is pointed at that same card, because
those are two independent levers and setting only the first leaves a library resolving to
ordinal 0 -- a neighbour's card under identity addressing. That pairing is `devices`'s
existing machinery, not a second invention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .devices import (BUSY_MEMORY_MIB, DEFAULT_DEVICE_ENV, NoCompatibleDevice,
                      discover, normalize_uuid, select)


#: Thread-count variables worth setting from the cgroup quota. A stage that reads the host's
#: 32 cores while the cgroup allows 4 oversubscribes itself into being slower, which then
#: looks like the benchmark being slow.
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")

#: Failures that are the device's fault, and the answer to each. Read from the program's own
#: output rather than anticipated: a stage that ran out of memory on one card is a stage that
#: should be tried on another, and one that says there is no CUDA device is a stage that
#: should be told to use the CPU.
_DEVICE_FAILURES = (
    ("out of memory", "another card"),
    ("cuda error: out of memory", "another card"),
    ("no cuda-capable device", "cpu"),
    ("no cuda gpus are available", "cpu"),
    ("cuda driver version is insufficient", "cpu"),
    ("cuda unknown error", "another card"),
)


@dataclass
class ComputeDecision:
    """One stage's device, the environment that pins it, and the evidence for it."""

    device: str                       # "cuda" / "cpu" / "unavailable" / a CUDA ordinal
    device_index: int
    environment: dict[str, str] = field(default_factory=dict)
    why: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def on_gpu(self) -> bool:
        return self.device.startswith("cuda")

    def as_dict(self) -> dict[str, Any]:
        return {"device": self.device, "device_index": self.device_index,
                "environment": dict(self.environment), "why": self.why,
                "evidence": self.evidence}


def _rank(device: Mapping[str, Any]) -> tuple:
    """The order cards are tried in: nothing running on it, then room, then idle.

    Ordered rather than scored, so the reason for a choice is a comparison a person can
    follow: a card with no compute processes beats one with them, whatever their sizes.
    """
    free = not device.get("compute_pids")
    total = int(device.get("memory_total_mib") or 0)
    used = int(device.get("memory_used_mib") or 0)
    utilisation = float(device.get("utilization_gpu_pct") or 0.0)
    # What is *free* is total minus used, and a card whose usage cannot be read is not a card
    # with room -- `parse_gpu_csv` leaves what it cannot read as None rather than guessing,
    # and the ranking has to respect that rather than read None as zero used.
    room = max(0, total - used) if (total and device.get("memory_used_mib") is not None) else 0
    return (0 if free else 1, -room, utilisation)


def _busy_reason(device: Mapping[str, Any]) -> str:
    """Whether a visible card already has enough external work to exclude this run.

    Small desktop/rendering allocations are common and do not by themselves make a card
    unavailable. Larger allocations or active utilization do: a single-card research run
    must not silently share the only GPU with another experiment.
    """
    used = device.get("memory_used_mib")
    utilization = device.get("utilization_gpu_pct")
    if used is None:
        return "GPU memory occupancy is unreadable"
    if int(used) > BUSY_MEMORY_MIB:
        return f"{int(used)} MiB already allocated (busy threshold {BUSY_MEMORY_MIB} MiB)"
    if utilization is None:
        return "GPU utilization is unreadable"
    if float(utilization) > 0.0:
        return f"GPU utilization is {float(utilization):g}%"
    return ""


def _threads(report: Mapping[str, Any]) -> dict[str, str]:
    effective = (report.get("cpu") or {}).get("effective_cpus")
    if not effective or float(effective) < 1:
        return {}
    count = str(max(1, int(float(effective))))
    return {name: count for name in _THREAD_VARS}


def decide(*, runner: Callable[..., str] | None = None,
           environ: Mapping[str, str] | None = None,
           python: Path | str | None = None,
           prefer: str | None = None,
           avoid: tuple[int, ...] = (),
           require_gpu: bool = False) -> ComputeDecision:
    """Choose a device for one stage, from what is on this machine right now.

    `avoid` names cards this stage has already failed on -- a card that ran out of memory
    is not a card to choose again, and the only thing that knows so is the failure.

    `prefer` is the caller overriding the decision -- `"cpu"`, `"cuda"`, or an index. It is
    honoured rather than second-guessed, because a caller that has a reason the table cannot
    show (a benchmark known to be slower on the card, a card reserved for something else) is
    the one that can see it. What it does not do is skip the table: the environment still has
    to pin whatever was chosen, and a caller asking for a card that is not there is told so.
    """
    report: dict[str, Any] = {}
    try:
        kwargs: dict[str, Any] = {}
        if runner is not None:
            kwargs["runner"] = runner
        if environ is not None:
            kwargs["environ"] = environ
        if python is not None:
            kwargs["python"] = python
        report = discover(**kwargs)
    except Exception as exc:                                     # noqa: BLE001
        # An unreadable device table is not evidence that the host has no GPU. Keep the
        # controller usable for source-only work, but make execution opt into CPU rather than
        # silently turning a potentially GPU-bound experiment into a very slow CPU run.
        explicitly_cpu = (prefer is not None and
                          str(prefer).strip().lower() in ("cpu", ""))
        explicitly_gpu = (prefer is not None and
                          str(prefer).strip().lower().startswith("cuda"))
        error = f"{type(exc).__name__}: {exc}"
        if require_gpu or explicitly_gpu:
            raise NoCompatibleDevice(
                f"the device table could not be read and this stage requires a GPU: "
                f"{error}", evidence={"error": error}) from None
        if explicitly_cpu:
            return ComputeDecision(
                device="cpu", device_index=0,
                environment={"CUDA_VISIBLE_DEVICES": ""},
                why=f"the caller asked for CPU; device discovery also failed ({error})",
                evidence={"error": error, "caller_requested_cpu": True})
        return ComputeDecision(
            device="unavailable", device_index=0,
            environment={"CUDA_VISIBLE_DEVICES": ""},
            why=(f"the device table could not be read ({error}); GPU availability is "
                 "unknown, so automatic execution is blocked rather than silently falling "
                 "back to CPU"),
            evidence={"error": error, "resource_unavailable": True})

    allowed = [gpu for gpu in (report.get("gpus") or [])
               if gpu.get("index") in set(report.get("allowed") or [])]
    visible = [gpu for gpu in allowed if gpu.get("torch_index") is not None]
    cuda_had_visible_gpu = bool(visible)
    evidence = {
        "host": report.get("host"),
        "gpus_seen": len(report.get("gpus") or []),
        "allowed": report.get("allowed"),
        "excluded": report.get("excluded"),
        "outer_cuda_visible_devices": (report.get("outer") or {}).get("CUDA_VISIBLE_DEVICES"),
        "cuda_order_mismatch": report.get("cuda_order_mismatch"),
        "cpu": report.get("cpu"),
        "memory_mib": (report.get("memory") or {}).get("available_mib"),
        "choices": [{"index": g.get("index"), "model": g.get("model"),
                     "memory_total_mib": g.get("memory_total_mib"),
                     "memory_used_mib": g.get("memory_used_mib"),
                     "utilization_gpu_pct": g.get("utilization_gpu_pct"),
                     "compute_pids": g.get("compute_pids"),
                     "uuid": g.get("uuid"),
                     "torch_index": g.get("torch_index")} for g in visible],
    }

    if prefer is not None and str(prefer).strip().lower() in ("cpu", ""):
        return ComputeDecision(
            device="cpu", device_index=0,
            environment={"CUDA_VISIBLE_DEVICES": "", **_threads(report)},
            why="the caller asked for the CPU", evidence=evidence)

    if avoid:
        remaining = [g for g in visible if int(g.get("index") or -1) not in avoid]
        if remaining != visible and not remaining:
            visible = []
        elif remaining:
            visible = remaining

    if not visible:
        why = ("no card is usable: " + (report.get("outer") or {}).get(
            "reason", "") if report.get("outer") else "") or \
            str(report.get("excluded") or "the machine reports no GPU")
        explicitly_gpu = (prefer is not None and
                          str(prefer).strip().lower().startswith("cuda"))
        if require_gpu or explicitly_gpu:
            raise NoCompatibleDevice(why)
        query_error = ((report.get("errors") or {}).get("gpu_query")
                       if isinstance(report.get("errors"), dict) else None)
        if query_error and not report.get("gpus"):
            evidence["resource_unavailable"] = True
            evidence["gpu_query_error"] = str(query_error)
            return ComputeDecision(
                device="unavailable", device_index=0,
                environment={"CUDA_VISIBLE_DEVICES": "", **_threads(report)},
                why=("nvidia-smi could not report whether GPU hardware exists; automatic "
                     "execution is blocked rather than silently falling back to CPU"),
                evidence=evidence)
        if not cuda_had_visible_gpu and report.get("gpus"):
            # nvidia-smi can see a physical card while this execution context has no CUDA
            # device nodes or has filtered every GPU out. Silently calling that "CPU" made
            # a GPU experiment appear runnable and could turn minutes into many hours.
            evidence["resource_unavailable"] = True
            return ComputeDecision(
                device="unavailable", device_index=0,
                environment={"CUDA_VISIBLE_DEVICES": "", **_threads(report)},
                why=("physical GPU hardware is reported but no card is visible to CUDA in "
                     "this execution context; refusing an implicit CPU fallback"),
                evidence=evidence)
        return ComputeDecision(device="cpu", device_index=0,
                               environment={"CUDA_VISIBLE_DEVICES": "", **_threads(report)},
                               why=f"{why}; the CPU is the answer that does not require a card",
                               evidence=evidence)

    busy = [(gpu, _busy_reason(gpu)) for gpu in visible]
    busy = [(gpu, reason) for gpu, reason in busy if reason]
    if busy:
        busy_indexes = {int(gpu.get("index", -1)) for gpu, _ in busy}
        eligible = [gpu for gpu in visible if int(gpu.get("index", -1)) not in busy_indexes]
        evidence["busy_devices"] = [
            {"index": gpu.get("index"), "uuid": gpu.get("uuid"), "reason": reason,
             "compute_pids": gpu.get("compute_pids")}
            for gpu, reason in busy]
        if not eligible:
            summary = "; ".join(f"card {gpu.get('index')}: {reason}"
                                 for gpu, reason in busy)
            raise NoCompatibleDevice(
                "all visible GPUs are busy; refusing to run a second workload: " + summary,
                evidence=evidence)
        visible = eligible

    if prefer is not None and str(prefer).strip().lower().startswith("cuda"):
        wanted = str(prefer).split(":", 1)[1] if ":" in str(prefer) else None
        if wanted is not None:
            matching = [g for g in visible if str(g.get("torch_index")) == wanted
                        or str(g.get("index")) == wanted]
            if not matching:
                raise NoCompatibleDevice(
                    f"the caller asked for device {prefer!r} and this container exposes "
                    f"{[g.get('index') for g in visible]}", evidence=evidence)
            order = matching + [g for g in visible if g not in matching]
        else:
            order = sorted(visible, key=_rank)
    else:
        order = sorted(visible, key=_rank)

    chosen = order[0]
    selection = select(chosen, "pinned_index")
    evidence["selected_device"] = {
        "index": chosen.get("index"), "uuid": chosen.get("uuid"),
        "torch_index": chosen.get("torch_index"),
        "process_torch_index": selection.torch_index,
    }
    environment = {
        "CUDA_VISIBLE_DEVICES": (selection.cuda_visible
                                 if selection.cuda_visible is not None else ""),
        # Two levers, not one: narrowing what is visible does not stop a library asking for a
        # bare "cuda" and getting ordinal 0, so the default ordinal is pointed at the same
        # card. `devices.align_process_defaults` is what consumes this in a CUDA child.
        DEFAULT_DEVICE_ENV: str(selection.torch_index),
        **_threads(report),
    }
    busy = chosen.get("compute_pids") or []
    total = chosen.get("memory_total_mib")
    used = chosen.get("memory_used_mib")
    room = (f"{max(0, total - used)} of {total} MiB free"
            if total and used is not None else "memory usage unreadable")
    return ComputeDecision(
        device=f"cuda:{selection.torch_index}", device_index=selection.torch_index,
        environment=environment,
        why=(f"card {chosen.get('index')} ({chosen.get('model')}), {room}, "
             f"{chosen.get('utilization_gpu_pct')}% utilisation, chosen from "
             f"{len(visible)} visible to CUDA"
             + (f"; {len(busy)} other process(es) are on it" if busy else
                "; nothing else is running on it")),
        evidence=evidence)


def recheck_gpu(decision: ComputeDecision, *,
                discoverer: Callable[..., dict[str, Any]] = discover,
                settle_seconds: float = 0.0) -> dict[str, Any]:
    """Re-read occupancy for the exact physical card immediately before a stage.

    A research run keeps one device decision so a train/evaluate pair remains comparable.
    That cached decision must not make a later stage blind to a service or another user's
    workload that appeared after the first stage. This check neither changes the card nor
    assumes that a lease file describes external processes.
    """
    if not decision.on_gpu:
        return {}
    selected = decision.evidence.get("selected_device") or {}
    target_uuid = str(selected.get("uuid") or "")
    if not target_uuid:
        raise NoCompatibleDevice(
            "the chosen GPU has no recorded physical UUID; refusing to run without a "
            "device identity", evidence=decision.evidence)
    wait_limit = max(0.0, float(settle_seconds))
    deadline = time.monotonic() + wait_limit
    while True:
        report = discoverer()
        allowed = set(report.get("allowed") or [])
        visible = [row for row in report.get("gpus") or []
                   if row.get("index") in allowed and row.get("torch_index") is not None]
        matches = [row for row in visible
                   if normalize_uuid(str(row.get("uuid") or "")) ==
                   normalize_uuid(target_uuid)]
        if len(matches) != 1:
            raise NoCompatibleDevice(
                f"the selected GPU UUID {target_uuid} is no longer uniquely visible to CUDA",
                evidence={"selected_device": selected,
                          "visible_uuids": [row.get("uuid") for row in visible],
                          "allowed": report.get("allowed"),
                          "outer": report.get("outer")})
        gpu = matches[0]
        reason = _busy_reason(gpu)
        waited = max(0.0, wait_limit - max(0.0, deadline - time.monotonic()))
        fresh = {"selected_device": selected, "observed_device": {
            "index": gpu.get("index"), "uuid": gpu.get("uuid"),
            "memory_used_mib": gpu.get("memory_used_mib"),
            "utilization_gpu_pct": gpu.get("utilization_gpu_pct"),
            "compute_pids": gpu.get("compute_pids")},
            "settle_wait_seconds": round(waited, 3)}
        if not reason:
            return fresh
        remaining = deadline - time.monotonic()
        # A just-finished local GPU stage can leave a brief utilization sample behind. Give
        # only that telemetry a short bounded quiet-window; memory pressure and persistent
        # utilization still fail closed before launch.
        if not reason.startswith("GPU utilization is ") or remaining <= 0:
            raise NoCompatibleDevice(
                f"selected GPU {target_uuid} became busy before stage start: {reason}",
                evidence=fresh)
        time.sleep(min(0.25, remaining))


def reconsider(decision: ComputeDecision, failure: str) -> ComputeDecision | None:
    """Another device, when the failure is the device's fault and there is another.

    Returns `None` when the failure says nothing about the device, which is most failures.
    The stage's own output decides -- the same discipline as reading any other failure --
    and a decision that has no alternative says so by returning `None` rather than repeating
    itself, so a caller cannot loop on it.
    """
    lowered = str(failure or "").lower()
    answer = next((what for signal, what in _DEVICE_FAILURES if signal in lowered), None)
    if answer is None:
        return None
    if answer == "cpu":
        if not decision.on_gpu:
            return None
        return decide(prefer="cpu")
    if not decision.on_gpu:
        return None
    # The card it failed on is not a card to try again, and re-deciding without that fact is
    # how a loop spends its whole budget on the one card that cannot hold the run.
    try:
        return decide(avoid=(decision.device_index,), prefer="cuda")
    except NoCompatibleDevice:
        return decide(prefer="cpu")
