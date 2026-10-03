"""Bounded metadata inspection of explicitly authorized resource bindings.

File readers see empty mount placeholders; this tool reads actual bound metadata without
granting execution, environment access, arbitrary host paths or resource file contents.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import uuid

from .common import atomic_json, atomic_text, now, read_json
from .evidence_store import capture_attempt_evidence
from .workspace_resources import bindings_for


def inspect(output: Path, *, target: str, directory: str = ".", limit: int = 100) -> dict:
    if not isinstance(target, str) or not target or len(target) > 512:
        raise ValueError("choose an exact target from the explicit resource catalog")
    if not isinstance(directory, str) or len(directory) > 512:
        raise ValueError("directory must be a bounded resource-relative path")
    relative = Path(directory)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("resource directory must not escape its binding")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("resource inventory limit must be 1..200")
    manifest = read_json(output / "workspace_snapshot.json")
    checkout = Path(manifest["destination"]).resolve(strict=True)
    if checkout != output.resolve() / "checkout":
        raise ValueError("resource inventory requires the run's isolated checkout")
    bindings = bindings_for(checkout)
    binding = next((row for row in bindings if row["target"] == target), None)
    if binding is None:
        raise ValueError("target is not an explicitly authorized resource binding")
    root = Path(binding["source"])
    path = root / relative
    # No symlink traversal, including links that currently point back into the binding.
    cursor = root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError("resource inventory does not traverse symlinks")
    if not path.resolve().is_relative_to(root):
        raise ValueError("resource directory escaped binding")
    entries, truncated = [], False
    from .agent_runtime import _PRIVATE_BASENAMES, _PRIVATE_SUFFIXES
    if path.is_dir():
        with os.scandir(path) as iterator:
            batch = list(itertools.islice(iterator, limit + 1))
        truncated = len(batch) > limit
        for entry in sorted(batch[:limit], key=lambda item: item.name):
            if entry.name in _PRIVATE_BASENAMES or Path(entry.name).suffix.lower() in _PRIVATE_SUFFIXES:
                continue
            stat = entry.stat(follow_symlinks=False)
            kind = "link_not_followed" if entry.is_symlink() else "directory" if entry.is_dir(follow_symlinks=False) else "file"
            entries.append({"name": entry.name, "kind": kind, "bytes": stat.st_size})
        status = "enumerated"
    elif path.is_file():
        entries = [{"name": path.name, "kind": "file", "bytes": path.stat().st_size}]
        status = "file_metadata"
    elif not path.exists():
        status = "missing_relative_directory"
    else:
        raise ValueError("resource is not a regular file or directory")
    identity = uuid.uuid4().hex
    result = {"id": identity, "target": target, "directory": relative.as_posix(),
              "status": status, "entries": entries, "truncated": truncated,
              "scope": "metadata only; not dataset/loader/strategy readiness", "at": now()}
    log = output / "resource_inspections" / f"{identity}.log"
    if log.parent.is_symlink():
        raise ValueError("resource inventory evidence path is unsafe")
    atomic_text(log, json.dumps(result, ensure_ascii=False))
    ref = f"resource_inspections/{identity}.json"
    result.update(capture_attempt_evidence(output, attempt_id=identity, log=log,
        receipt_ref=ref, status=status, returncode=None, termination_reason="metadata_only"))
    atomic_json(output / ref, result)
    return result
