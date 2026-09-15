"""Bounded CPU environment inventory and embedded ACT normalization checks.

This does not certify rendering, GPU execution, or prediction parity. Native and
reference-inference receipts remain separate gates. No environment tree is hashed.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .common import atomic_json, digest, now, object_digest, read_json
from .host_resources import capacity


PACKAGES = ("torch", "numpy", "lerobot", "safetensors", "embodichain", "dexsim")
_VERSION_CODE = """
import importlib.metadata as m, importlib.util, json, platform, sys
names = json.loads(sys.argv[1])
versions = {}
for name in names:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python':platform.python_version(),'executable':sys.executable,
                 'prefix':sys.prefix,'versions':versions,
                 'module_available':{n:importlib.util.find_spec(n) is not None for n in names}}))
"""


def interpreter_contract(python: Path, *, timeout: float = 20) -> dict:
    """Ask the selected interpreter for metadata without importing GPU libraries."""
    if timeout <= 0:
        raise ValueError("metadata timeout must be positive")
    result = {"requested_executable": str(Path(python).absolute())}
    try:
        completed = subprocess.run([str(python), "-I", "-c", _VERSION_CODE, json.dumps(PACKAGES)],
                                   capture_output=True, text=True, timeout=timeout,
                                   env={k: os.environ[k] for k in ("PATH", "LANG", "LD_LIBRARY_PATH")
                                        if k in os.environ})
        if completed.returncode != 0:
            return {**result, "status": "failed", "returncode": completed.returncode}
        return {**result, "status": "passed", **json.loads(completed.stdout)}
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {**result, "status": "failed", "error_type": type(exc).__name__}


def _normalization_values(stream, data_start: int, header: dict, name: str, expected_shape: list) -> list:
    item = header.get(name)
    if not isinstance(item, dict) or item.get("shape") != expected_shape:
        raise ValueError(f"missing or incompatible normalization tensor: {name}")
    offsets = item.get("data_offsets", [])
    width, fmt = {"F64": (8, "d"), "F32": (4, "f"), "F16": (2, "e"),
                  "BF16": (2, None)}.get(item.get("dtype"), (0, None))
    count = math.prod(expected_shape)
    if not width or len(offsets) != 2 or any(type(v) is not int for v in offsets):
        raise ValueError(f"invalid normalization layout: {name}")
    start, end = offsets
    if start < 0 or end - start != count * width or not 0 < end - start <= 1024 * 1024:
        raise ValueError(f"invalid normalization extent: {name}")
    stream.seek(data_start + start)
    raw = stream.read(end - start)
    if len(raw) != end - start:
        raise ValueError(f"truncated normalization data: {name}")
    if fmt is None:
        values = [struct.unpack("<f", b"\x00\x00" + raw[i:i + 2])[0]
                  for i in range(0, len(raw), 2)]
    else:
        values = list(struct.unpack("<" + str(count) + fmt, raw))
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"nonfinite normalization: {name}")
    if name.endswith(".std") and any(v < 0 for v in values):
        raise ValueError(f"negative normalization standard deviation: {name}")
    return values


def inspect_act_checkpoint(checkpoint: Path, *, compatibility_file: Path | None = None,
                           verified_file_hashes: Mapping[str, str] | None = None) -> dict:
    """Inspect JSON/header/stats; full inference remains a separate check.

    A trusted caller that just hashed these files may supply those identities to
    avoid another full weight-file read. This never skips structural validation;
    externally supplied/model-authored hashes must not be passed here.
    """
    checkpoint = Path(checkpoint)
    config_path, weights = checkpoint / "config.json", checkpoint / "model.safetensors"
    result = {"scope": "cpu_checkpoint_normalization_structure", "checkpoint": str(checkpoint),
              "prediction_parity_verified": False, "native_runtime_verified": False}
    try:
        known = dict(verified_file_hashes or {})
        if (set(known) - {"config_sha256", "checkpoint_sha256", "compatibility_sha256"}
                or any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v)
                       for v in known.values())):
            raise ValueError("invalid host-verified checkpoint identity")
        config = read_json(config_path)
        if config.get("type") not in (None, "act"):
            raise ValueError("checkpoint is not ACT")
        mapping = config.get("normalization_mapping")
        if not isinstance(mapping, dict):
            raise ValueError("checkpoint normalization mapping is missing")
        input_features, output_features = config.get("input_features", {}), config.get("output_features", {})
        if "observation.state" not in input_features or "action" not in output_features:
            raise ValueError("ACT state/action feature contract is missing")
        checked = []
        with weights.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError("truncated safetensors header length")
            header_size = struct.unpack("<Q", size_bytes)[0]
            if not 0 < header_size <= 16 * 1024 * 1024:
                raise ValueError("unbounded safetensors header")
            raw = stream.read(header_size)
            if len(raw) != header_size:
                raise ValueError("truncated safetensors header")
            header = json.loads(raw)
            for prefix, features in (("normalize_inputs", input_features),
                                     ("normalize_targets", output_features),
                                     ("unnormalize_outputs", output_features)):
                for key, feature in features.items():
                    mode = mapping.get(feature["type"], "IDENTITY")
                    if mode == "IDENTITY":
                        continue
                    names = {"MEAN_STD": ("mean", "std"), "MIN_MAX": ("min", "max")}.get(mode)
                    if names is None:
                        raise ValueError("unsupported normalization mapping")
                    shape = feature["shape"]
                    if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
                        raise ValueError("invalid feature shape")
                    if feature["type"] == "VISUAL":
                        shape = [shape[0], 1, 1]
                    fields = {}
                    for stat in names:
                        name = f"{prefix}.buffer_{key.replace('.', '_')}.{stat}"
                        fields[stat] = _normalization_values(stream, 8 + header_size, header, name, shape)
                        checked.append(name)
                    if mode == "MIN_MAX" and any(a > b for a, b in zip(fields["min"], fields["max"])):
                        raise ValueError("normalization minimum exceeds maximum")
        helper = Path(compatibility_file) if compatibility_file is not None else None
        if helper is not None and not helper.is_file():
            raise ValueError("ACT compatibility implementation is missing")
        result.update(status="passed", normalization="checkpoint_inline_v1",
                      config_sha256=known.get("config_sha256") or digest(config_path),
                      checkpoint_sha256=known.get("checkpoint_sha256") or digest(weights),
                      compatibility_sha256=(known.get("compatibility_sha256") or digest(helper)) if helper else None,
                      input_features=input_features, output_features=output_features,
                      normalization_mapping=mapping, checked_statistics=checked,
                      extra_unused_fields="skip when absent, matching legacy Normalize")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:300])
    return result


def collect_environment_contract(repo: Path, *, interpreters: Mapping[str, Path],
                                 assets: Mapping[str, dict] | None = None,
                                 source_files: Sequence[Path] = (), inventory: dict | None = None,
                                 checkpoint: Path | None = None, output: Path | None = None,
                                 interpreter_timeout: float = 20) -> dict:
    """Inventory explicit inputs only; missing assets fail the lightweight gate."""
    repo = Path(repo).absolute()
    sources = {}
    for path in source_files:
        path = Path(path)
        if path.name.startswith(".env") or ".venv" in path.parts or path.is_dir():
            raise ValueError("source inventory accepts explicit code/lock files, never credentials or environments")
        sources[str(path.absolute())] = digest(path)
    runtimes = {role: interpreter_contract(Path(python), timeout=interpreter_timeout)
                for role, python in interpreters.items()}
    asset_records = {}
    for name, spec in (assets or {}).items():
        path = Path(spec["path"])
        row = {"path": str(path.absolute()), "exists": path.exists(),
               "kind": "directory" if path.is_dir() else "file", "status": "passed" if path.exists() else "failed"}
        if spec.get("sha256"):
            row["expected_sha256"] = spec["sha256"]
            row["sha256"] = digest(path) if path.is_file() else None
            if row["sha256"] != spec["sha256"]:
                row["status"] = "failed"
        elif path.is_file():
            row["bytes"] = path.stat().st_size
        asset_records[name] = row
    disks = {}
    for label, path in (("repository", repo), ("temporary", Path("/tmp")), ("shared_memory", Path("/dev/shm"))):
        if path.exists():
            usage = shutil.disk_usage(path)
            disks[label] = {"path": str(path), "total_bytes": usage.total, "free_bytes": usage.free,
                            "readable": os.access(path, os.R_OK), "writable": os.access(path, os.W_OK)}
    policy = (inspect_act_checkpoint(checkpoint, compatibility_file=repo / "policy/act/checkpoint_compat.py")
              if checkpoint is not None else {"status": "not_requested"})
    stable = {"source_files": sources, "interpreters": runtimes, "assets": asset_records,
              "checkpoint_contract": policy,
              "container_image": os.environ.get("AUTOSIM_CONTAINER_IMAGE")}
    passed = bool(runtimes) and all(row["status"] == "passed" for row in runtimes.values())
    passed = passed and all(row["status"] == "passed" for row in asset_records.values())
    passed = passed and policy["status"] != "failed"
    result = {"schema_version": 1, "kind": "harness_environment_contract", "created_at": now(),
              **stable, "fingerprint": object_digest(stable), "host_resources": capacity(),
              "disk": disks, "allocation_inventory": inventory,
              "status": "passed" if passed else "failed", "inventory_only": True,
              "native_runtime_verified": False, "prediction_parity_verified": False,
              "scope": "metadata, named offline assets and optional ACT statistics; separate native/parity gates required"}
    if output is not None:
        atomic_json(Path(output), result)
    return result
