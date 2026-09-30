"""Classify pre-existing policy assets against an immutable source inventory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import digest, object_digest, read_json


def source_snapshot_policy(path: Path, *, repo: Path, output: Path) -> dict[str, Any]:
    """A policy generated after workspace creation must never become 'shipped'."""
    path = Path(path)
    repo = Path(repo).resolve(strict=True)
    output = Path(output).resolve(strict=True)
    if not path.is_absolute():
        path = repo / path
    manifest_path = output / "workspace_snapshot.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return {"status": "unknown", "why": "no immutable source inventory"}
    manifest = read_json(manifest_path)
    entries = manifest.get("source_entries") if isinstance(manifest, dict) else None
    if (not isinstance(entries, list) or
            object_digest(entries) != manifest.get("source_tree_fingerprint") or
            manifest.get("destination") != str(repo)):
        return {"status": "invalid_inventory", "why": "source inventory is untrusted"}
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(repo).as_posix()
    except (OSError, ValueError, RuntimeError):
        return {"status": "not_source", "why": "artifact is outside the source snapshot"}
    if not resolved.is_file() or path.is_symlink():
        return {"status": "not_source", "why": "only inventoried regular files qualify"}
    row = next((item for item in entries if isinstance(item, (list, tuple)) and
                len(item) == 5 and item[0] == "file" and item[1] == relative), None)
    if row is None:
        return {"status": "run_generated", "why": "artifact was absent from the source snapshot"}
    if row[2] != resolved.stat().st_size or row[3] != digest(resolved):
        return {"status": "changed", "why": "artifact bytes differ from the source snapshot"}
    return {"status": "source_snapshot", "path": str(resolved),
            "sha256": row[3], "source_tree_fingerprint": manifest["source_tree_fingerprint"]}
