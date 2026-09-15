"""Resolve the multi-device CLI against real hardware, or refuse with a reason.

``devices.py`` answers "what does this node expose and how would each card be addressed";
this module answers the invocation-level questions: which addressing mode is *verified* on
this node, which subset of cards this run may take, whether the result is still the legacy
single-device form, and whether a partial plan needs the operator's acknowledgement.

The ``--gpus`` family changes the run's frozen identity, so a plan that is not legacy
equivalent adds a ``devices`` block to ``protocol.json``.  A legacy invocation must
therefore never build a plan at all -- that is what keeps ``--gpu 0`` byte-identical, and
why :func:`resolve_plan` is only ever called when a multi-device knob was passed.

Unknown capability is never assumed available: without a receipt that was measured under
the addressing mode in use, the run refuses to start rather than guessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import atomic_json, object_digest, now
from .devices import (NoCompatibleDevice, apply_capability, build_plan, describe, discover,
                      load_probe_receipt, normalize_uuid, normalize_uuid_mapping,
                      probe_cache_dir, probe_receipt_key, run_text)

# The fields that make two plans the *same experiment*; occupancy is runtime state, and a
# card becoming free or busy must not by itself invalidate a resume.
IDENTITY_FIELDS = ("mode", "requested", "allowed_uuids", "max_parallel_jobs", "gpu_hours_limit")

PROBE_MODULE = "autosim.research.device_probe"


class DevicePlanRefused(RuntimeError):
    """The run refuses to start: the plan is unverified, partial, or unavailable."""


class DeviceReceiptMissing(DevicePlanRefused):
    """No receipt was measured for *this* container's device set.

    Its own class because it is the one refusal a run can *fix by measuring*: a receipt is
    keyed by the node it was taken on, and in this cluster the container's hostname is
    job-scoped (``pt-<job uid>-worker-0``), so a receipt published by another job can never
    authorize this one.  ``--probe-if-needed`` therefore runs the ladder in this container
    rather than treating a missing receipt as an unavailability.
    """


def plan_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: plan.get(key) for key in IDENTITY_FIELDS}


def plan_digest(plan: Mapping[str, Any]) -> str:
    return object_digest(plan_identity(plan))


def driver_of(report: Mapping[str, Any]) -> str | None:
    for gpu in report.get("gpus", []):
        if gpu.get("driver_version"):
            return str(gpu["driver_version"])
    return None


def uuids_of(report: Mapping[str, Any]) -> list[str]:
    return [str(gpu["uuid"]) for gpu in report.get("gpus", []) if gpu.get("uuid")]


def probe_hint(*, platform_root: Path, output: Path | None = None) -> str:
    return (f"run the device probe on this node first: python -m {PROBE_MODULE} "
            f"--workspace {platform_root} --output {output or '<dir>'} --gpus auto "
            f"--mode pinned_index  (see ROBOSYN_5090_RUNBOOK.md)")


def receipt_lookup_key(report: Mapping[str, Any], *, image: str | None,
                       worker_spec: str | None) -> str:
    return probe_receipt_key(host=str(report.get("host") or ""), driver=driver_of(report),
                             image=image, worker_spec=worker_spec, uuids=uuids_of(report))


def resolve_plan(*, platform_root: Path, requested: str | None, mode: str,
                 max_parallel_jobs: int | None = None, gpu_hours: float | None = None,
                 hours: float = 24.0, accept_partial: bool = False,
                 image: str | None = None, worker_spec: str | None = None,
                 report: Mapping[str, Any] | None = None,
                 receipt: Mapping[str, Any] | None = None,
                 environ: Mapping[str, str] | None = None,
                 runner: Callable[..., str] = run_text,
                 python: Path | str | None = None,
                 torch_rows: Sequence[dict] | None = None,
                 disk_path: Path | None = None) -> dict:
    """Build the plan this run executes under, or explain why it cannot start.

    ``report``/``receipt`` exist so the decision logic is testable without hardware; when
    they are omitted they are measured (``nvidia-smi`` + a torch child) and loaded (the
    content-addressed receipt under ``autoresearch_cache/device_probes``).
    """
    if report is None:
        report = discover(runner=runner, environ=environ, python=python,
                          torch_rows=torch_rows, disk_path=disk_path)
    key = receipt_lookup_key(report, image=image, worker_spec=worker_spec)
    if receipt is None:
        receipt = load_probe_receipt(platform_root, key)
    if receipt is None:
        raise DeviceReceiptMissing(
            f"no passing device receipt for this node/driver/device set "
            f"(key {key[:12]}, cache {probe_cache_dir(platform_root)}); a multi-device "
            f"plan is only built from a measured receipt, and unknown capability is never "
            f"assumed available. " + probe_hint(platform_root=platform_root))
    # "auto" means "the mode this node was actually verified under", not "pick one".
    resolved = str(receipt.get("mode") or "")
    if not resolved:
        raise DevicePlanRefused(
            f"the receipt for key {key[:12]} names no addressing mode; re-run the probe")
    if mode != "auto" and resolved != mode:
        raise DevicePlanRefused(
            f"the receipt for key {key[:12]} was verified under mode {resolved!r}, "
            f"not the requested {mode!r}; re-run the probe in the requested mode")
    report = apply_capability(dict(report), receipt)
    capabilities = normalize_uuid_mapping(receipt.get("device_capabilities", {}))
    for d in report["gpus"]:
        d["capabilities"] = dict(capabilities.get(normalize_uuid(d.get("uuid") or ""), {}))
    if not report.get("usable"):
        raise NoCompatibleDevice(describe(report))
    plan = build_plan(report=report, requested=requested, mode=resolved,
                      max_parallel_jobs=max_parallel_jobs, gpu_hours=gpu_hours, hours=hours,
                      probe_receipt_sha256=object_digest(receipt))
    waiting = list(plan["waiting"])
    if waiting and not accept_partial:
        reasons = "; ".join(f"index {row['index']} {row['capability']} ({row['reason']})"
                            for row in waiting)
        raise DevicePlanRefused(
            f"{len(waiting)} requested device(s) are not allocatable: {reasons}. "
            f"Re-run with --accept-device-plan to proceed on the "
            f"{len(plan['usable'])} usable device(s), or wait and retry.")
    plan.update(schema_version=1, probe_receipt_key=key, probe_receipt_mode=resolved,
                requested_mode=mode, accepted_partial=bool(waiting),
                legacy_equivalence=legacy_equivalence(plan),
                resource_summary=describe(report), created_at=now())
    plan["plan_digest"] = plan_digest(plan)
    return plan


def legacy_equivalence(plan: Mapping[str, Any]) -> bool:
    """A plan is legacy equivalent only if it addresses the one card legacy would.

    ``--gpus 0`` on a four-card node resolves to a single device at index 0, which is
    exactly the card a plain ``--gpu 0`` run uses and exactly how it addresses it
    (``CUDA_VISIBLE_DEVICES=0``, ``gpu_id=0``, torch index 0).  Such a run keeps the
    legacy command shape and the legacy protocol: it must not pay for an explicit renderer
    or a device block it does not use.
    """
    usable = plan.get("usable") or []
    return (len(usable) == 1 and plan.get("mode") == "pinned_index"
            and int(usable[0].get("index", -1)) == 0)


def plan_summary(plan: Mapping[str, Any]) -> str:
    # Defensive on purpose: this runs while an exception is being *raised*, and a formatter
    # that can itself fail turns a clear refusal into a confusing traceback.
    usable = ", ".join(f"{gpu.get('index')}:{gpu.get('uuid')}"
                       for gpu in plan.get("usable") or [])
    return (f"mode={plan.get('mode')} usable={len(plan.get('usable') or [])} [{usable}] "
            f"waiting={len(plan.get('waiting') or [])} "
            f"max_parallel_jobs={plan.get('max_parallel_jobs')} "
            f"gpu_hours_limit={plan.get('gpu_hours_limit'):g} "
            f"legacy_equivalence={plan.get('legacy_equivalence')}")


def write_plan(run_root: Path, plan: Mapping[str, Any]) -> Path:
    path = Path(run_root) / "device_plan.json"
    atomic_json(path, dict(plan))
    return path


def devices_protocol_block(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The identity-bearing part of the plan, as it enters the frozen protocol."""
    return {
        "plan_digest": plan_digest(plan),
        "mode": plan.get("mode"),
        "requested": plan.get("requested"),
        "max_parallel_jobs": plan.get("max_parallel_jobs"),
        "gpu_hours_limit": plan.get("gpu_hours_limit"),
        "usable": [{"index": g.get("index"), "uuid": g.get("uuid"),
                    "class_name": g.get("class_name"), "renderer": g.get("renderer"),
                    "selection": g.get("selection")} for g in plan.get("usable") or []],
        "waiting": [{"index": g.get("index"), "uuid": g.get("uuid"),
                     "capability": g.get("capability"), "reason": g.get("reason")}
                    for g in plan.get("waiting") or []],
        "accepted_partial": bool(plan.get("accepted_partial")),
        "probe_receipt_key": plan.get("probe_receipt_key"),
        "probe_receipt_mode": plan.get("probe_receipt_mode"),
        "resource_summary": plan.get("resource_summary"),
    }


def protocol_mismatch(recorded: Mapping[str, Any], resolved: Mapping[str, Any]) -> dict:
    """Compare the recorded plan identity with this invocation's, field by field."""
    return {key: {"recorded": recorded.get(key), "requested": resolved.get(key)}
            for key in IDENTITY_FIELDS if recorded.get(key) != resolved.get(key)}
