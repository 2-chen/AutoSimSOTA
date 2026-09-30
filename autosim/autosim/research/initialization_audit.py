"""Preserve bounded source provenance before any research Agent can change code."""
from pathlib import Path
import os
from .common import atomic_json, digest, read_json
from .snapshot import Snapshots


def preserve(output: Path, repo: Path):
    root = output / "initialization_audit"
    if root.is_symlink():
        raise ValueError("initialization audit directory is unsafe")
    if (root / "provenance.json").exists():
        return
    inventory_path = output / "workspace_snapshot.json"
    inventory = read_json(inventory_path) if inventory_path.is_file() else {}
    original = {r[1]: r[3] for r in inventory.get("source_entries", [])
                if isinstance(r, (list, tuple)) and len(r) >= 5 and r[0] == "file"
                and isinstance(r[1], str) and isinstance(r[3], str)}
    paths, total, mismatched, omitted = [], 0, [], 0
    suffixes = {".py", ".sh", ".toml", ".yaml", ".yml", ".cfg"}
    def candidates():
        if original:
            # Never traverse generated environments or linked resource trees when the
            # tracked source inventory already specifies the audit's scope.
            for name in sorted(original):
                relative = Path(name)
                if not relative.is_absolute() and ".." not in relative.parts:
                    yield repo / relative
            return
        for parent, dirs, names in os.walk(repo, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {".git", ".venv", "venv", "node_modules", "__pycache__"}
                       and not (Path(parent) / d).is_symlink() and
                       not (Path(parent) / d).resolve().is_relative_to(output.resolve())]
            for name in names:
                yield Path(parent) / name
    for path in candidates():
        if (path.suffix not in suffixes or path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(repo.resolve())):
            continue
        relative = path.relative_to(repo).as_posix()
        if any(word in path.name.lower() for word in ("credential", "secret", "private_key")):
            omitted += 1; continue
        size = path.stat().st_size
        if size > 65536 or total + size > 8 * 1024**2 or len(paths) >= 10000:
            omitted += 1; continue
        if original and digest(path) != original[relative]:
            mismatched.append(relative)
        total += size; paths.append(relative)
    Snapshots(root).capture(paths, repo=repo, name="controller_start",
                            why="before first Agent call; bounded source-only provenance")
    atomic_json(root / "provenance.json", {"schema_version": 1,
        "status": "source_copy_verified" if original and not mismatched else "controller_start_only",
        "preexisting_changes": mismatched[:100], "omitted_files": omitted,
        "scope": "bounded source text, not resources/checkpoints/demonstrations"})
