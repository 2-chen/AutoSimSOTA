"""Explicit read-only resource attachments, independent of the editable source copy.

Mounts exist only inside controlled processes. No symlinks/hardlinks to the source are
created. Metadata identities detect replacement, not in-place dataset content changes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .common import read_json


def parse_binding(value: str) -> dict[str, str]:
    target, separator, source = value.partition("=")
    if not separator or not source:
        raise ValueError("resource must be CHECKOUT_RELATIVE_PATH=/absolute/source")
    return {"target": target, "source": source}


def validate_bindings(source: Path, checkout: Path,
                      bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(bindings) > 64:
        raise ValueError("at most 64 explicit resource bindings are supported")
    result: list[dict[str, Any]] = []
    for item in bindings:
        relative = Path(str(item.get("target", "")))
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts or
                any(part in {".git", ".autosim"} for part in relative.parts)):
            raise ValueError("resource target must be a safe checkout-relative path")
        raw = Path(str(item.get("source", ""))).expanduser()
        if not raw.is_absolute():
            raise ValueError("resource source must be absolute")
        resource = raw.resolve(strict=True)
        if (len(resource.parts) < 3 or source.is_relative_to(resource) or
                resource.is_relative_to(checkout.parent) or
                checkout.parent.is_relative_to(resource)):
            raise ValueError("resource must not expose a repository ancestor or run directory")
        if not resource.is_dir() and not resource.is_file():
            raise ValueError("resource must be a regular file or directory")
        for previous in result:
            other = Path(previous["target"])
            if relative.is_relative_to(other) or other.is_relative_to(relative):
                raise ValueError("overlapping resource targets are not allowed")
        stat = resource.stat()
        if (("device" in item and item["device"] != stat.st_dev) or
                ("inode" in item and item["inode"] != stat.st_ino)):
            raise ValueError("resource identity changed; create a new run to rebind")
        result.append({"target": relative.as_posix(), "source": str(resource),
                       "kind": "directory" if resource.is_dir() else "file",
                       "device": stat.st_dev, "inode": stat.st_ino,
                       "access": "read_only"})
    return result


def bindings_for(checkout: Path) -> list[dict[str, Any]]:
    checkout = Path(checkout).resolve()
    marker = checkout.parent / "workspace_snapshot.json"
    if not marker.is_file():
        return []
    manifest = read_json(marker)
    bindings = manifest.get("resource_bindings", [])
    if not bindings:
        return []
    if manifest.get("destination") != str(checkout):
        raise ValueError("resource checkout identity mismatch")
    current = validate_bindings(Path(manifest["source"]), checkout, bindings)
    if current != bindings:
        raise ValueError("resource identity changed; create a new run to rebind")
    for item in bindings:
        target = checkout / item["target"]
        if target.resolve() != target or not target.exists():
            raise ValueError("resource mount target is missing or traverses a symlink")
        if (item["kind"] == "directory" and (not target.is_dir() or any(target.iterdir())) or
                item["kind"] == "file" and (not target.is_file() or target.stat().st_size)):
            raise ValueError("resource placeholder was modified; refusing to hide local outputs")
    return bindings


def mount_args(checkout: Path, *, mounted_at: Path | None = None) -> list[str]:
    args: list[str] = []
    for item in bindings_for(checkout):
        args.extend(("--ro-bind", item["source"],
                     str((mounted_at or checkout) / item["target"])))
    return args


def resource_view(checkout: Path) -> dict[str, Any]:
    marker = Path(checkout).parent / "workspace_snapshot.json"
    if not marker.is_file():
        return {}
    manifest = read_json(marker)
    try:
        bindings = bindings_for(checkout)
        error = ""
    except (OSError, ValueError, TypeError, KeyError) as exc:
        bindings, error = [], str(exc)
    return {"bindings": [{"target": row["target"], "kind": row["kind"],
                           "access": "read_only"} for row in bindings],
            "validation_error": error,
            "omitted_candidates": [str(name)[:256] for name in
                                   manifest.get("omitted_candidates", [])[:16]],
            "omitted_inventory_ref": "workspace_snapshot.json#omitted_candidates",
            "readiness": "requires_native_probe",
            "note": "Code isolation is not environment readiness. Resources are visible at "
                    "these relative paths only inside controlled shell/inspection/stage "
                    "processes, not host file readers. Verify native consumers; write derived "
                    "data, caches and new checkpoints to unmounted run-local paths. Resource "
                    "contents are live, not immutable snapshots; verify experiment inputs."}


def omitted_candidates(source: Path, copied: set[Path], limit: int = 128) -> list[str]:
    """Bounded names-only inventory, pruning omitted trees without reading their data."""
    from .workspace_snapshot import SKIP_DIRECTORIES
    rows: list[str] = []
    for parent, directories, files in os.walk(source, followlinks=False):
        relative = Path(parent).relative_to(source)
        kept = []
        for name in sorted(directories):
            path = relative / name
            if name in SKIP_DIRECTORIES:
                continue
            if path not in copied:
                rows.append(path.as_posix() + "/")
            elif not (source / path).is_symlink():
                kept.append(name)
            if len(rows) >= limit:
                return rows[:limit]
        directories[:] = kept
        for name in sorted(files):
            path = relative / name
            if path not in copied:
                rows.append(path.as_posix())
            if len(rows) >= limit:
                return rows[:limit]
    return rows
