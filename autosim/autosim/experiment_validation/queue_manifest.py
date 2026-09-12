"""Append-only, restart-safe manifests for long-running validation queues."""
from __future__ import annotations

from pathlib import Path

from autosim.research.common import digest, immutable_json, now, read_json


_LINEAGE_KEYS = {"created_at", "supersedes", "supersedes_sha256"}


def _protocol(document: dict) -> dict:
    return {key: value for key, value in document.items() if key not in _LINEAGE_KEYS}


def versioned_manifest(root: Path, payload: dict) -> Path:
    """Reuse an exact protocol manifest or append a new immutable version."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    base = root / "manifest.json"
    if base.is_file():
        paths.append(base)
    index = 2
    while (root / f"manifest.v{index}.json").is_file():
        paths.append(root / f"manifest.v{index}.json")
        index += 1
    for path in reversed(paths):
        if _protocol(read_json(path)) == payload:
            return path
    target = base if not paths else root / f"manifest.v{index}.json"
    document = {"created_at": now(), **payload}
    if paths:
        predecessor = paths[-1]
        document.update(supersedes=str(predecessor.absolute()),
                        supersedes_sha256=digest(predecessor))
    immutable_json(target, document)
    return target
