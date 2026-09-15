"""Device discovery, capability receipts, and per-job simulator device selection.

DexSim renders through Vulkan but runs physics/CUDA interop on the GPU named by
``gpu_id``.  The engine resolves ``gpu_id`` by matching that index's **NVML UUID**
against each Vulkan candidate's ``deviceUUID`` (``VulkanCore::DX_PickPhysicalDevice``,
``librender-hybrid.so``), so ``gpu_id`` lives in the physical/NVML index space while
``select_default_renderer`` reads the same integer in the CUDA space.  ``CUDA_VISIBLE_DEVICES``
remaps only the CUDA space: setting it to ``i`` while leaving ``gpu_id`` at 0 makes the two
name different cards and the process aborts importing a semaphore (``DFGpuSemaphore.cpp:215``).

Everything here is therefore expressed in two explicitly separated numbers:

* ``vulkan_gpu_id`` -- the physical/NVML index handed to the engine;
* ``torch_index``    -- the index that same physical card has *inside* the subprocess
  (its ``CUDA_VISIBLE_DEVICES`` remapping applied).

A capability is never inferred from a model name: a device is usable only when a probe
receipt recorded for this node's driver and UUID set says a real episode ran on it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import atomic_json, object_digest, read_json

NVIDIA_SMI = "nvidia-smi"
GPU_QUERY = "index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version,compute_cap"
COMPUTE_APPS_QUERY = "gpu_uuid,pid,used_memory"

#: A device is treated as occupied by someone else above this idle footprint.
BUSY_MEMORY_MIB = 2048
#: Capability states.  ``unknown`` is deliberately *not* usable.
CAPABILITY_STATES = ("unknown", "verified_sim", "verified_train", "incompatible", "busy")
#: Renderer fragments that prefer full ray tracing, mirroring EmbodiChain's own rule
#: (``render_utils._FAST_RT_GPU_KEYWORDS``) minus the silent ``auto`` fallback.
FAST_RT_KEYWORDS = ("A100", "A800", "H100", "H800", "H200", "H20")

_ABSENT = {"", "n/a", "[n/a]", "unknown", "[unknown]", "not supported", "none"}


class NoCompatibleDevice(RuntimeError):
    """No device may be used for GPU work; the reason is in ``device_plan.json``."""


def _int_or_none(value: str) -> int | None:
    text = value.strip()
    return None if text.lower() in _ABSENT else int(text)


def _float_or_none(value: str) -> float | None:
    text = value.strip()
    return None if text.lower() in _ABSENT else float(text)


def normalize_uuid(value: str) -> str:
    """Fold a UUID for *comparison* only -- torch omits the ``GPU-`` prefix nvidia-smi prints.

    Never use this to name or key anything: the raw UUID is the identity shared by leases,
    plans, receipts and accounting records.
    """
    text = value.strip()
    return text[4:] if text.upper().startswith("GPU-") else text


def normalize_uuid_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Build a comparison index without changing persisted GPU identities."""
    result: dict[str, Any] = {}
    for uuid, value in values.items():
        key = normalize_uuid(uuid)
        if key in result and result[key] != value:
            raise ValueError(f"conflicting receipt values for GPU UUID {key}")
        result[key] = value
    return result


def gpu_class(model: str) -> str:
    return "".join(ch for ch in model.lower() if ch.isalnum())[-12:] or "unknown"


def renderer_for(model: str) -> str:
    """An explicit renderer, never ``auto``.

    ``renderer == "auto"`` routes through ``select_default_renderer(gpu_id)``, which calls
    ``torch.cuda.get_device_name(gpu_id)`` -- a physical index read in CUDA space.  Under a
    remapped ``CUDA_VISIBLE_DEVICES`` that raises, and the exception is swallowed into a
    silent ``hybrid`` fallback (``render_utils.py:62-69``): wrong renderer, no error.
    """
    upper = model.upper()
    return "fast-rt" if any(key in upper for key in FAST_RT_KEYWORDS) else "hybrid"


@dataclass(frozen=True)
class SimDeviceSelection:
    """How one subprocess is pointed at exactly one physical device."""

    mode: str                     # "legacy" | "pinned_index" | "identity" | "node_isolated"
    index: int                    # position inside the outer visible range
    uuid: str
    vulkan_gpu_id: int            # config["gpu_id"] / run_env.py --gpu_id
    torch_index: int              # index inside the subprocess, after CUDA_VISIBLE_DEVICES
    cuda_visible: str | None      # value to export; None unsets it (full device set)
    renderer: str
    extra_env: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


#: Set by ``Runtime.environment`` to the *torch-space* index a GPU child should treat as
#: its default device.  Unset on the legacy single-GPU path, where ordinal 0 already is the
#: leased card and there is nothing to align.
DEFAULT_DEVICE_ENV = "AUTOSIM_DEFAULT_CUDA_ORDINAL"


def ordinal_of(device: object) -> int | None:
    """The ordinal in a torch device spelling: ``"cuda"`` -> 0, ``"cuda:3"`` -> 3.

    Anything else -- a ``torch.device``, ``"cpu"``, an empty value -- has no ordinal here and
    returns ``None``; callers treat that as "this process has no CUDA default to align".
    """
    text = str(device or "").strip().lower()
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return None
    return None


def default_device_index(environ: Mapping[str, str] | None = None) -> int | None:
    """The ordinal ``Runtime.environment`` asked this process to default to, if any."""
    value = ((os.environ if environ is None else environ).get(DEFAULT_DEVICE_ENV) or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def align_process_defaults(index: int, *, warp: bool = True,
                           importer: Callable[[str], Any] | None = None) -> dict:
    """Make "the default device" mean the leased card in this process, not card zero.

    Two libraries pick a card when a call does not name one, and both pick ordinal 0: warp
    sets ``cuda:0`` as its default at init (``Runtime.__init__`` calls
    ``set_default_device("cuda:0")`` when no context is current) and torch resolves a bare
    ``"cuda"`` through ``torch.cuda.current_device()``.  Under *identity* addressing, ordinal
    0 is physical card 0 -- not the card this job leased -- so one unqualified call lands on
    a neighbour's card.  That is the residual the second multi-GPU probe measured: 1030 MiB
    appearing on card 0 at 0.0% utilization while an identity arm ran on card 3.  Under the
    pinned modes the same call is invisible, because there ordinal 0 *is* the leased card,
    which is why the leak only ever showed up under identity.

    This is a second, independent lever from ``CUDA_VISIBLE_DEVICES``: narrowing the visible
    set cannot help an identity process (the engine needs every card visible to resolve its
    physical index), so the *default* has to be pointed at the leased card explicitly.

    Nothing here may be fatal.  A process without CUDA, without warp, or with an ordinal its
    (possibly narrowed) view does not contain must still start, so every step is guarded and
    the outcome is returned as a record -- which is also what the tests assert on.

    ``warp=False`` skips a library the caller's process never uses (the policy child is torch
    only); on the legacy path no caller aligns at all, because ordinal 0 is already the leased
    card there and initializing a library nobody asked for would edit the golden path for
    nothing.
    """
    import importlib

    load = importer or importlib.import_module
    record: dict[str, Any] = {"torch_index": int(index), "torch_default": None,
                              "warp_default": None}
    try:
        torch = load("torch")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(index))
            record["torch_default"] = int(index)
        else:
            record["torch_default"] = "no_cuda"
    except Exception as exc:                      # missing torch, driver, or a bad ordinal
        record["torch_default"] = f"unavailable: {type(exc).__name__}"
    if not warp:
        return record
    try:
        warp_module = load("warp")
        warp_module.set_device(f"cuda:{int(index)}")
        record["warp_default"] = int(index)
    except Exception as exc:
        record["warp_default"] = f"unavailable: {type(exc).__name__}"
    return record


def parse_gpu_csv(text: str) -> list[dict]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``.

    Anything unreadable stays ``None`` rather than being guessed -- a device whose UUID or
    memory cannot be read is reported as ``unknown`` and is not usable.
    """
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        index, uuid = _int_or_none(parts[0]), parts[1]
        rows.append({
            "index": index,
            "uuid": None if uuid.lower() in _ABSENT else uuid,
            "model": None if parts[2].lower() in _ABSENT else parts[2],
            "memory_total_mib": _int_or_none(parts[3]),
            "memory_used_mib": _int_or_none(parts[4]),
            "utilization_gpu_pct": _float_or_none(parts[5]),
            "driver_version": None if parts[6].lower() in _ABSENT else parts[6],
            "compute_capability": None if parts[7].lower() in _ABSENT else parts[7],
        })
    return rows


def parse_compute_apps(text: str) -> list[dict]:
    """Parse ``nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory``."""
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        memory = _int_or_none(parts[2])
        rows.append({"gpu_uuid": parts[0], "pid": _int_or_none(parts[1]),
                     "used_memory_mib": memory})
    return rows


def run_text(command: Sequence[str], *, env: Mapping[str, str] | None = None,
             timeout: int = 30) -> str:
    """Run a read-only query; a missing/failing tool is reported, never raised."""
    try:
        completed = subprocess.run(list(command), capture_output=True, text=True,
                                   env=dict(env) if env is not None else None, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"__error__: {type(exc).__name__}: {exc}"
    return completed.stdout if completed.returncode == 0 else f"__error__: rc={completed.returncode}"


def _error(text: str) -> str | None:
    return text.strip() if text.strip().startswith("__error__") else None


def torch_devices(python: Path | str = sys.executable,
                  env: Mapping[str, str] | None = None) -> list[dict]:
    """The CUDA-space truth: index -> UUID, straight from torch."""
    script = (
        "import json, torch\n"
        "rows = []\n"
        "for i in range(torch.cuda.device_count()):\n"
        "    p = torch.cuda.get_device_properties(i)\n"
        "    rows.append({'index': i, 'uuid': str(p.uuid), 'name': p.name,\n"
        "                 'memory_total_mib': int(p.total_memory // (1024 * 1024))})\n"
        "print(json.dumps(rows))\n")
    text = run_text([str(python), "-c", script], env=env, timeout=180)
    if _error(text):
        return []
    try:
        import json
        return json.loads(text.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return []


def outer_allowed(environ: Mapping[str, str], gpus: Sequence[dict]) -> tuple[list[dict], dict]:
    """Respect the container's own device filtering before choosing anything.

    ``NVIDIA_VISIBLE_DEVICES`` (the container runtime's filter) is authoritative when it
    lists UUIDs; ``CUDA_VISIBLE_DEVICES`` then restricts further, expressed in the indices
    the container itself reports.
    """
    record = {"CUDA_VISIBLE_DEVICES": environ.get("CUDA_VISIBLE_DEVICES"),
              "NVIDIA_VISIBLE_DEVICES": environ.get("NVIDIA_VISIBLE_DEVICES")}
    allowed, reason = list(gpus), "no outer filter"
    for variable in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if variable not in environ:
            continue
        visible = environ[variable].strip()
        if variable == "NVIDIA_VISIBLE_DEVICES" and visible.lower() == "all":
            continue
        if not visible or visible.lower() in {"none", "void", "-1"}:
            allowed, reason = [], f"{variable}={visible!r} exposes no device"
            continue
        selected = []
        for token in visible.split(","):
            token = token.strip()
            if token.isdecimal():
                matches = [g for g in allowed if g.get("index") == int(token)]
            elif token.startswith(("GPU-", "MIG-")):
                matches = [g for g in allowed if g.get("uuid") and
                           normalize_uuid(g["uuid"]).startswith(normalize_uuid(token))]
            else:
                matches = []
            if len(matches) != 1 or matches[0] in selected:
                # CUDA stops enumeration at the first invalid identifier. Never widen it.
                break
            selected.append(matches[0])
        allowed = selected
        reason = f"restricted by {variable}"
    record["allowed_indices"] = [g.get("index") for g in allowed]
    record["reason"] = reason
    return allowed, record


def cuda_child_env(environ: Mapping[str, str], outer: Mapping[str, Any]) -> dict[str, str]:
    """The environment a torch query must run in to report the truth about this process.

    Empty is a deliberate mask; only an absent variable may be removed.
    """
    child = dict(environ)
    visible = outer.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        child["CUDA_VISIBLE_DEVICES"] = str(visible)
    else:
        child.pop("CUDA_VISIBLE_DEVICES", None)
    return child


def legacy_index_lock_held(index: int) -> bool:
    """A legacy single-GPU run holds ``/tmp/autosim-robosyn-gpu-<i>.lock`` while working.

    The old ``robosyn_mvp.gpu_lock`` is keyed by index, so it cannot be consulted through
    the UUID lease layer; it is read here as *evidence of occupancy* only.
    """
    import fcntl
    path = Path(f"/tmp/autosim-robosyn-gpu-{index}.lock")
    if not path.is_file():
        return False
    try:
        with path.open("a+") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError:
        return True
    return False


def discover(*, runner: Callable[..., str] = run_text,
             environ: Mapping[str, str] | None = None,
             python: Path | str | None = None,
             torch_rows: Sequence[dict] | None = None,
             disk_path: Path | None = None,
             host: str | None = None) -> dict:
    """Collect the resource table before any GPU-bound work starts."""
    environ = dict(os.environ if environ is None else environ)
    table = runner([NVIDIA_SMI, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"])
    apps = runner([NVIDIA_SMI, f"--query-compute-apps={COMPUTE_APPS_QUERY}",
                   "--format=csv,noheader,nounits"])
    gpus = parse_gpu_csv(table)
    compute = [] if _error(apps) else parse_compute_apps(apps)
    by_uuid: dict[str, list[dict]] = {}
    for row in compute:
        by_uuid.setdefault(normalize_uuid(row["gpu_uuid"]), []).append(row)
    allowed, outer = outer_allowed(environ, gpus)
    cuda_rows = list(torch_rows) if torch_rows is not None else torch_devices(
        python or sys.executable, env=cuda_child_env(environ, outer))
    torch_by_uuid = {normalize_uuid(str(r["uuid"])): r for r in cuda_rows if r.get("uuid")}
    excluded = [{"index": g["index"], "reason": "no UUID reported by nvidia-smi" if not g.get("uuid")
                 else "not visible to CUDA"} for g in allowed
                if normalize_uuid(g.get("uuid") or "") not in torch_by_uuid]
    allowed = [g for g in allowed if normalize_uuid(g.get("uuid") or "") in torch_by_uuid]
    order_mismatch = None
    for position, row in enumerate(cuda_rows):
        if row.get("index") == position and position < len(gpus):
            if gpus[position].get("uuid") and normalize_uuid(gpus[position]["uuid"]) != normalize_uuid(
                    str(row["uuid"])):
                order_mismatch = (f"position {position}: nvidia-smi {gpus[position]['uuid']} != "
                                  f"torch {row['uuid']}")
    for gpu in gpus:
        key = normalize_uuid(gpu["uuid"]) if gpu.get("uuid") else None
        gpu["class_name"] = gpu_class(gpu.get("model") or "")
        gpu["compute_pids"] = [r["pid"] for r in by_uuid.get(key or "", [])]
        gpu["capability"] = "unknown"
        if key and key in torch_by_uuid:
            gpu["torch_index"] = torch_by_uuid[key]["index"]
        gpu["busy_evidence"] = []
    disk = disk_path or Path.cwd()
    usage = shutil.disk_usage(disk)
    from .host_resources import capacity
    limits = capacity()
    topology = runner([NVIDIA_SMI, "topo", "-m"])
    return {
        "schema_version": 1,
        "host": host or os.uname().nodename,
        "gpus": gpus,
        "compute_apps": compute,
        "outer": outer,
        "allowed": [g["index"] for g in allowed],
        "excluded": excluded,
        "visible_index_to_uuid": {str(g["index"]): g["uuid"] for g in allowed},
        "cuda_index_to_uuid": {str(r["index"]): str(r["uuid"]) for r in cuda_rows},
        "cuda_order_mismatch": order_mismatch,
        "errors": {"gpu_query": _error(table), "compute_apps": _error(apps)},
        "cpu": {"model": _cpu_model(), **limits["cpu"]},
        "memory": limits["memory"],
        "cgroup_version": limits["cgroup_version"],
        "disk": {"path": str(disk), "free_bytes": usage.free},
        "renderers": {str(g["index"]): renderer_for(g.get("model") or "") for g in gpus},
        "topology": {"source":"nvidia-smi topo -m", "raw":topology[:24000],
                     "error":_error(topology), "nvlink_assumed":False},
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _memory_info() -> dict:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.split(":")[0] in {"MemTotal", "MemAvailable"}:
                values[line.split(":")[0]] = int(line.split()[1]) // 1024
    except OSError:
        pass
    return {"total_mib": values.get("MemTotal"), "available_mib": values.get("MemAvailable")}


def apply_capability(report: dict, receipt: dict | None, *,
                     index_lock_held: Callable[[int], bool] | None = None) -> dict:
    """Fold a probe receipt into the resource table; unprobed devices stay ``unknown``.

    Receipt keys are compared UUID-normalized (torch omits the ``GPU-`` prefix); the
    device's own ``uuid`` field is untouched and stays the identity every other record
    (lease, plan, accounting) is keyed by.
    """
    per_uuid = normalize_uuid_mapping((receipt or {}).get("devices", {}))
    lock_held = index_lock_held or legacy_index_lock_held
    for gpu in report["gpus"]:
        # Without a UUID the device cannot be leased (the lock is keyed by it), cannot be
        # matched to a receipt, and cannot be named in a receipt of our own: unusable.
        if not gpu.get("uuid"):
            gpu["capability"] = "unknown"
            gpu["busy_evidence"] = ["no UUID reported by nvidia-smi"]
            continue
        key = normalize_uuid(gpu["uuid"])
        evidence = []
        gpu["occupancy_unverified"] = gpu.get("memory_used_mib") is None
        if gpu.get("memory_used_mib") is not None and gpu["memory_used_mib"] > BUSY_MEMORY_MIB:
            evidence.append(f"{gpu['memory_used_mib']} MiB in use")
        if gpu.get("compute_pids"):
            evidence.append(f"compute pids {gpu['compute_pids']}")
        if gpu.get("index") is not None and lock_held(int(gpu["index"])):
            evidence.append("legacy index lock held")
        if evidence:
            gpu["capability"], gpu["busy_evidence"] = "busy", evidence
            continue
        verdict = per_uuid.get(key, "")
        if verdict in ("verified_sim", "verified_train", "incompatible"):
            gpu["capability"] = verdict
    usable = [g for g in report["gpus"]
              if g.get("index") in report["allowed"] and g["capability"] == "verified_sim"]
    waiting = [g for g in report["gpus"] if g.get("index") in report["allowed"] and g not in usable]
    report["usable"], report["waiting"] = usable, waiting
    report["reason"] = describe(report)
    return report


def describe(report: dict) -> str:
    """Every scenario names a reason: a device is either allocated or waiting with why."""
    if not report["allowed"]:
        return f"no device is allowed: {report['outer'].get('reason')}"
    waiting = "; ".join(
        f"index {g['index']} {g['capability']}"
        + (f" ({', '.join(g.get('busy_evidence') or [])})" if g.get("busy_evidence") else "")
        for g in report.get("waiting", []))
    if not report.get("usable"):
        return f"0 of {len(report['allowed'])} allowed devices usable: {waiting}"
    base = f"{len(report['usable'])} of {len(report['allowed'])} allowed devices usable"
    return f"{base}; waiting: {waiting}" if waiting else base


def select(device: dict, mode: str = "pinned_index") -> SimDeviceSelection:
    """Build the two-number device selection for one physical device."""
    index, uuid = int(device["index"]), device["uuid"]
    renderer = renderer_for(device.get("model") or "")
    if mode == "identity":
        # The process keeps the whole visible set; CUDA index == physical index.
        return SimDeviceSelection(mode, index, uuid, index, index, None, renderer, {})
    if mode == "node_isolated":
        # One visible device *and* physical index 0 (device nodes for others hidden).
        return SimDeviceSelection(mode, index, uuid, 0, 0, "0", renderer, {})
    # pinned_index: the process sees one card, the engine is addressed by physical index.
    return SimDeviceSelection("pinned_index", index, uuid, index, 0, str(index), renderer, {})


def identity_is_safe(report: Mapping[str, Any]) -> tuple[bool, str]:
    """Identity addressing assumes the process's CUDA ordinals *are* the physical indices.

    That holds while the container exposes every card, in order.  An outer
    ``CUDA_VISIBLE_DEVICES`` that lists a subset -- or the same cards in a different order --
    renumbers what the process sees, so the engine's physical index stops being a valid
    ordinal and the run dies in ``OptixDevice.cpp``.  Refusing here names the filter instead
    of letting the plan die in its first episode.
    """
    outer = report.get("outer") or {}
    visible = str(outer.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if outer.get("CUDA_VISIBLE_DEVICES") is None:
        mismatch = report.get("cuda_order_mismatch")
        return (False, str(mismatch)) if mismatch else (True, "")
    if not visible:
        return False, "CUDA_VISIBLE_DEVICES is empty (no CUDA device is visible)"
    try:
        listed = [int(part) for part in visible.split(",")]
    except ValueError:
        return False, f"CUDA_VISIBLE_DEVICES={visible!r} is not a list of device indices"
    order = [int(gpu["index"]) for gpu in (report.get("gpus") or [])
             if gpu.get("index") is not None]
    if listed != order:
        return False, (f"outer CUDA_VISIBLE_DEVICES={visible!r} does not expose this node's "
                       f"cards in their own order ({order}); identity addressing means the "
                       f"engine's index *is* the process's CUDA ordinal, so a filter that "
                       f"hides or reorders cards would point the engine at another device")
    return True, ""


def build_plan(*, report: dict, requested: str, mode: str, max_parallel_jobs: int | None,
               gpu_hours: float | None, hours: float, probe_receipt_sha256: str | None) -> dict:
    """Turn the resource table into the plan the run is executed under."""
    allowed = set(report["allowed"])
    if requested not in (None, "auto"):
        try:
            positions = [int(part) for part in str(requested).split(",")]
        except ValueError as exc:
            raise ValueError(f"unparseable --gpus {requested!r}") from exc
        outside = [p for p in positions if p not in allowed]
        if outside:
            raise ValueError(f"--gpus {requested} outside the allowed range {sorted(allowed)}")
        wanted = set(positions)
    else:
        wanted = allowed
    usable = [g for g in report.get("usable", []) if g["index"] in wanted]
    waiting = [g for g in report.get("waiting", []) if g["index"] in wanted]
    if not usable:
        raise NoCompatibleDevice(report.get("reason", "no usable device"))
    if mode == "identity":
        safe, why = identity_is_safe(report)
        if not safe:
            raise NoCompatibleDevice(why)
    for gpu in usable:
        gpu.setdefault("class_name", gpu_class(gpu.get("model") or ""))
        gpu["renderer"] = renderer_for(gpu.get("model") or "")
        gpu["selection"] = select(gpu, mode).as_dict()
    limit = max_parallel_jobs if max_parallel_jobs is not None else len(usable)
    if limit < 1:
        raise ValueError("--max-parallel-jobs must be at least 1")
    return {
        "schema_version": 1,
        "kind": "autosim_device_plan",
        "mode": mode,
        "requested": requested or "auto",
        "usable": usable,
        "waiting": [{"index": g["index"], "uuid": g["uuid"], "capability": g["capability"],
                     "reason": ", ".join(g.get("busy_evidence") or []) or g["capability"]}
                    for g in waiting] + ([{"index": g["index"], "uuid": None, "capability": "unknown",
                        "reason": g["reason"]} for g in report.get("excluded", [])]
                        if requested in (None, "auto") else []),
        "allowed_uuids": [g["uuid"] for g in report["gpus"] if g["index"] in allowed],
        "visible_index_to_uuid": report["visible_index_to_uuid"],
        "cuda_order_mismatch": report.get("cuda_order_mismatch"),
        "probe_receipt_sha256": probe_receipt_sha256,
        "max_parallel_jobs": min(limit, len(usable)),
        "gpu_hours_limit": (gpu_hours if gpu_hours is not None else hours * len(usable)),
        "gpu_hours_requested": gpu_hours,
        "hours": hours,
        "policy": {"max_heavy_jobs_per_device": 1,
                   "co_schedule_training_and_simulation": False,
                   "shared_writable_cache": "~/.cache/embodichain_cache (material cache, "
                                            "deliberately shared; per-job caches cover /tmp)"},
    }


def plan_digest(plan: dict) -> str:
    return object_digest(plan)


def probe_receipt_key(*, host: str, driver: str | None, image: str | None,
                      worker_spec: str | None, uuids: Sequence[str]) -> str:
    """A receipt is only valid for the node + driver + device set it was measured on."""
    return object_digest({"host": host, "driver": driver, "image": image,
                          "worker_spec": worker_spec,
                          "uuids": sorted(normalize_uuid(u) for u in uuids)})


def probe_cache_dir(platform_root: Path) -> Path:
    return platform_root / "autoresearch_cache/device_probes"


def load_probe_receipt(platform_root: Path, key: str, mode: str | None = None) -> dict | None:
    """A receipt is only usable for the addressing scheme it was measured under.

    A device verified while pinned by physical index says nothing about the identity mode
    (which leaves every card visible to torch), so the mode is part of the query, not a
    detail of the caller.
    """
    path = probe_cache_dir(platform_root) / f"{key}.json"
    if not path.is_file():
        return None
    receipt = read_json(path)
    if not receipt.get("passed"):
        return None
    if mode is not None and receipt.get("mode") != mode:
        return None
    return receipt


def write_probe_receipt(platform_root: Path, key: str, receipt: dict) -> Path:
    path = probe_cache_dir(platform_root) / f"{key}.json"
    atomic_json(path, receipt)
    return path


def shard_plan(episodes: int, count: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal, ordered blocks covering ``[0, episodes)`` exactly."""
    if count < 1:
        raise ValueError("shard count must be positive")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if count > episodes:
        raise ValueError("cannot have more shards than episodes")
    base, extra = divmod(episodes, count)
    blocks, offset = [], 0
    for k in range(count):
        size = base + (1 if k < extra else 0)
        blocks.append((offset, size))
        offset += size
    return blocks


def shard_count(episodes: int, ready_devices: int, *, max_parallel_jobs: int,
                cost_model: Mapping[str, float], min_gain: float = 0.20,
                min_episodes_per_shard: int = 8) -> int:
    """How many shards pay off, given that every shard re-pays environment construction.

    ``T(n) = construction + ceil(episodes/n) * marginal``.  Each further shard must cut
    the estimate by at least ``min_gain`` to be worth the extra construction.
    """
    import math

    construction = float(cost_model.get("construction_seconds", 360.0))
    marginal = float(cost_model.get("marginal_seconds", 40.0))
    ceiling = min(ready_devices, max_parallel_jobs, episodes // max(1, min_episodes_per_shard))
    previous = construction + episodes * marginal
    best = 1
    for n in range(2, ceiling + 1):
        estimate = construction + math.ceil(episodes / n) * marginal
        if estimate > (1.0 - min_gain) * previous:
            break
        best, previous = n, estimate
    return best
