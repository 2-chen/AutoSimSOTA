"""A bounded, independent code checkout for experiments that may edit source files.

This copies bytes, not hard links. An external symlink is refused rather than becoming a
write-through escape from the working copy. The result is a source snapshot, not a complete
container sandbox: subprocesses may still access absolute paths and external services.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .common import atomic_json, bounded_run, digest, now, object_digest, read_json


SKIP_DIRECTORIES = frozenset({".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                              ".mypy_cache", ".ruff_cache"})
DEFAULT_COPY_LIMIT_BYTES = 4 * 1024**3


def _copied_link_target(root: Path, link: Path, *, kind: str = "source") -> str:
    """Rebase absolute in-tree symlinks so the isolated copy does not read the source tree."""
    root, link = Path(root).resolve(), Path(link)
    raw = os.readlink(link)
    target = link.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"external {kind} symlink in source: {link}")
    if Path(raw).is_absolute():
        return os.path.relpath(target, start=link.parent)
    return raw


def _git_revision(source: Path) -> str:
    try:
        done = bounded_run(["git", "-C", str(source), "rev-parse", "HEAD"],
                           cwd=source, timeout=10,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _inventory(root: Path, *, max_bytes: int, max_files: int) -> list[tuple[Any, ...]]:
    """Hash the materialized tree with the same exclusions and link rules as create()."""
    root = Path(root).resolve()
    rows: list[tuple[Any, ...]] = []
    total = 0
    for parent, dirnames, filenames in os.walk(root, followlinks=False):
        parent_path = Path(parent)
        relative_parent = parent_path.relative_to(root)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            path = parent_path / name
            if name in SKIP_DIRECTORIES:
                continue
            if path.is_symlink():
                rows.append(("symlink", str(path.relative_to(root)),
                             _copied_link_target(root, path, kind="directory")))
            else:
                rows.append(("directory", str(path.relative_to(root)),
                             stat.S_IMODE(path.stat().st_mode)))
                kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            path = parent_path / name
            relative = path.relative_to(root)
            if path.is_symlink():
                rows.append(("symlink", str(relative),
                             _copied_link_target(root, path, kind="file")))
                continue
            if not path.is_file():
                raise ValueError(f"non-regular source entry cannot be inventoried: {path}")
            size = path.stat().st_size
            total += size
            if total > max_bytes:
                raise ValueError(f"source tree exceeds inventory byte limit {max_bytes}")
            rows.append(("file", str(relative), size, digest(path),
                         stat.S_IMODE(path.stat().st_mode)))
            if len(rows) > max_files:
                raise ValueError(f"source tree exceeds inventory file limit {max_files}")
    rows.sort(key=lambda row: row[1])
    if len(rows) > max_files:
        raise ValueError(f"source tree exceeds inventory file limit {max_files}")
    return rows


def _validated_base(manifest: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    entries = manifest.get("source_entries")
    fingerprint = str(manifest.get("source_tree_fingerprint") or "")
    if not isinstance(entries, list) or not fingerprint or object_digest(entries) != fingerprint:
        raise ValueError("workspace source inventory is missing or has an invalid fingerprint")
    base: dict[str, tuple[Any, ...]] = {}
    for row in entries:
        if (not isinstance(row, (list, tuple)) or len(row) not in {3, 4, 5} or
                not isinstance(row[1], str)):
            raise ValueError("workspace source inventory contains a malformed entry")
        relative = Path(row[1])
        if relative.is_absolute() or ".." in relative.parts or not row[1]:
            raise ValueError("workspace source inventory contains an unsafe path")
        if row[1] in base:
            raise ValueError("workspace source inventory contains a duplicate path")
        if row[0] == "file" and len(row) in {4, 5}:
            if (not isinstance(row[2], int) or row[2] < 0 or
                    not isinstance(row[3], str) or len(row[3]) != 64):
                raise ValueError("workspace source inventory contains a malformed file")
            if len(row) == 5 and (not isinstance(row[4], int) or row[4] < 0):
                raise ValueError("workspace source inventory contains a malformed file mode")
        elif row[0] == "symlink" and len(row) == 3 and isinstance(row[2], str):
            pass
        elif row[0] == "directory" and len(row) == 3 and isinstance(row[2], int) and row[2] >= 0:
            pass
        else:
            raise ValueError("workspace source inventory contains an unknown entry type")
        base[row[1]] = tuple(row)
    return base


def _source_overlay(destination: Path, manifest: dict[str, Any], *,
                    max_bytes: int, max_files: int
                    ) -> tuple[dict[str, str], dict[str, int], dict[str, int | None]]:
    destination = Path(destination).resolve()
    base = _validated_base(manifest)
    current_rows = _inventory(destination, max_bytes=max_bytes, max_files=max_files)
    current = {str(row[1]): tuple(row) for row in current_rows}
    changed_files: dict[str, str] = {}
    changed_modes: dict[str, int] = {}
    changed_directories: dict[str, int | None] = {}
    for relative in sorted(set(base) | set(current)):
        before, after = base.get(relative), current.get(relative)
        if before == after:
            continue
        if ((before and before[0] == "symlink") or (after and after[0] == "symlink")):
            raise ValueError(f"source-policy snapshot cannot preserve symlink change: {relative}")
        if (before and after and before[0] != after[0]):
            raise ValueError(f"source-policy snapshot cannot preserve path type change: {relative}")
        if (before and before[0] == "file") or (after and after[0] == "file"):
            changed_files[relative] = str(after[3]) if after and after[0] == "file" else ""
            if after and after[0] == "file" and len(after) >= 5:
                changed_modes[relative] = int(after[4])
        if (before and before[0] == "directory") or (after and after[0] == "directory"):
            changed_directories[relative] = (int(after[2]) if after and
                                             after[0] == "directory" else None)
    overlay_bytes = sum(int(current[path][2]) for path, key in changed_files.items()
                        if key and path in current and current[path][0] == "file")
    overlay_limit = int(os.environ.get("AUTOSIM_SOURCE_OVERLAY_LIMIT_BYTES",
                                      str(2 * 1024**3)))
    if overlay_limit <= 0 or overlay_bytes > overlay_limit:
        raise ValueError(f"source overlay is {overlay_bytes} bytes, above limit {overlay_limit}")
    return changed_files, changed_modes, changed_directories


def _state_identity(manifest: dict[str, Any], name: str,
                    changed: dict[str, str], changed_modes: dict[str, int],
                    changed_directories: dict[str, int | None]) -> str:
    return object_digest({
        "schema_version": 1,
        "source_tree_fingerprint": manifest["source_tree_fingerprint"],
        "source_snapshot": name,
        "overlay_files": changed,
        "overlay_modes": changed_modes,
        "overlay_directories": changed_directories,
        "copy_mode": manifest.get("copy_mode", "full_worktree"),
        "untracked_assets_omitted": bool(manifest.get("untracked_assets_omitted")),
        **({"resource_bindings": manifest["resource_bindings"]}
           if manifest.get("resource_bindings") else {}),
    })


def capture_state(destination: Path, manifest: dict[str, Any], snapshots: Any, *, name: str,
                  max_bytes: int = DEFAULT_COPY_LIMIT_BYTES,
                  max_files: int = 100000) -> dict[str, Any]:
    """Freeze the exact source-tree delta from an isolated run's verified starting tree.

    Snapshot blobs preserve only changed files; the full base remains identified by the
    workspace manifest. Unexpected symlink changes are rejected because the file snapshot
    format cannot safely restore them.
    """
    destination = Path(destination).resolve()
    if manifest.get("schema_version") != 2:
        raise ValueError("source-state freezing requires workspace snapshot schema version 2")
    if manifest.get("destination") != str(destination):
        raise ValueError("workspace manifest does not identify this isolated checkout")
    changed, modes, directories = _source_overlay(
        destination, manifest, max_bytes=max_bytes, max_files=max_files)
    snapshot = snapshots.get(name)
    if snapshot is None:
        snapshot = snapshots.capture(changed, repo=destination, name=name,
                                     why="source tree used by a measured policy")
    elif dict(snapshot.files) != changed:
        raise ValueError("the persisted source snapshot differs from the measured checkout")
    elif any(key and (not (snapshots.blobs / key).is_file() or
                      digest(snapshots.blobs / key) != key)
             for key in snapshot.files.values()):
        snapshot = snapshots.capture(changed, repo=destination, name=name,
                                     why="source tree used by a measured policy")
    if dict(snapshot.files) != changed:
        raise ValueError("the persisted source snapshot differs from the measured checkout")
    identity = _state_identity(manifest, name, changed, modes, directories)
    return {"schema_version": 1, "source_tree_fingerprint":
            manifest["source_tree_fingerprint"], "source_snapshot": name,
            "overlay_files": changed, "overlay_modes": modes,
            "overlay_directories": directories, "identity_sha256": identity,
            "copy_mode": manifest.get("copy_mode", "full_worktree"),
            "untracked_assets_omitted": bool(manifest.get("untracked_assets_omitted"))}


def verify_state(destination: Path, manifest: dict[str, Any], state: dict[str, Any],
                 snapshots: Any, *, max_bytes: int = DEFAULT_COPY_LIMIT_BYTES,
                 max_files: int = 100000) -> bool:
    """Check that a checkout is exactly the recorded base plus its content-addressed delta."""
    name = str(state.get("source_snapshot") or "")
    snapshot = snapshots.get(name)
    if (manifest.get("schema_version") != 2 or state.get("schema_version") != 1 or
            not name or snapshot is None or
            state.get("source_tree_fingerprint") != manifest.get("source_tree_fingerprint")):
        return False
    try:
        changed, modes, directories = _source_overlay(
            destination, manifest, max_bytes=max_bytes, max_files=max_files)
    except (OSError, TypeError, ValueError, RuntimeError):
        return False
    return (dict(snapshot.files) == changed == dict(state.get("overlay_files") or {}) and
            modes == dict(state.get("overlay_modes") or {}) and
            directories == dict(state.get("overlay_directories") or {}) and
            _state_identity(manifest, name, changed, modes, directories) ==
            state.get("identity_sha256") and
            bool(state.get("identity_sha256")))


def apply_state(destination: Path, state: dict[str, Any], snapshots: Any) -> None:
    """Reconstruct the recorded file, directory and permission overlay on a verified base."""
    root = Path(destination).resolve()
    files = dict(state.get("overlay_files") or {})
    modes = dict(state.get("overlay_modes") or {})
    directories = dict(state.get("overlay_directories") or {})
    snapshot = snapshots.get(str(state.get("source_snapshot") or ""))
    if snapshot is None or dict(snapshot.files) != files:
        raise ValueError("source state and its content-addressed file snapshot do not match")

    def safe(relative: str) -> Path:
        path = Path(relative)
        target = (root / path).resolve()
        if path.is_absolute() or ".." in path.parts or not target.is_relative_to(root):
            raise ValueError(f"source overlay path escapes checkout: {relative}")
        return target

    for relative, mode in sorted(directories.items(), key=lambda row: len(Path(row[0]).parts)):
        if mode is not None:
            target = safe(relative)
            target.mkdir(parents=True, exist_ok=True)
            # Keep changed directories writable until the file blobs have been restored.
            os.chmod(target, int(mode) | 0o700)
    snapshots.restore(str(state.get("source_snapshot") or ""), repo=root)
    for relative, mode in modes.items():
        target = safe(relative)
        if mode is not None:
            if not target.is_file() or target.is_symlink():
                raise ValueError(f"source overlay mode has no regular file: {relative}")
            os.chmod(target, int(mode))
    for relative, mode in sorted(directories.items(),
                                 key=lambda row: len(Path(row[0]).parts), reverse=True):
        target = safe(relative)
        if mode is None:
            try:
                target.rmdir()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ValueError(f"source overlay directory cannot be removed: {relative}") from exc
    for relative, mode in sorted(directories.items(),
                                 key=lambda row: len(Path(row[0]).parts), reverse=True):
        if mode is not None:
            os.chmod(safe(relative), int(mode))


def create(source: Path, destination: Path, *, max_bytes: int = DEFAULT_COPY_LIMIT_BYTES,
           max_files: int = 100000, tracked_only: bool = False,
           resources: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Materialize a source snapshot, refusing ambiguous or unbounded source trees."""
    source = Path(source).resolve()
    destination = Path(destination).parent.resolve() / Path(destination).name
    if not source.is_dir():
        raise ValueError("benchmark source is not a directory")
    if destination.exists():
        raise FileExistsError(f"working copy already exists: {destination}")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("working copy must be disjoint from the benchmark source")
    if max_bytes <= 0 or max_files <= 0:
        raise ValueError("working-copy byte and file limits must be positive")
    from .workspace_resources import validate_bindings, omitted_candidates
    bindings = validate_bindings(source, destination, resources or [])
    # User-declared resources are not code, even when Git tracks them. Do not use
    # benchmark names or file extensions to guess this boundary.
    resource_prefixes = [Path(row["source"]).relative_to(source) for row in bindings
                         if Path(row["source"]).is_relative_to(source)]
    tracked: set[Path] | None = None
    tracked_directories: set[Path] = set()
    if tracked_only:
        listed = subprocess.run(["git", "-C", str(source), "ls-files", "--cached", "-z"],
                                capture_output=True, check=False, timeout=30)
        if listed.returncode != 0:
            raise ValueError("tracked-only copy requires a readable Git index; "
                             "use --full-copy explicitly for a bounded non-Git source copy")
        tracked = {Path(os.fsdecode(item)) for item in listed.stdout.split(b"\0") if item}
        if not tracked:
            raise ValueError("tracked-only copy found no files in the Git index")
        tracked_directories = {parent for item in tracked for parent in item.parents}
    files: list[tuple[Path, Path, int, str, int]] = []
    links: list[tuple[Path, str]] = []
    directories: list[tuple[Path, int]] = []
    total = 0
    for parent, dirnames, filenames in os.walk(source, followlinks=False):
        parent_path = Path(parent)
        relative_parent = parent_path.relative_to(source)
        kept_dirs = []
        for name in sorted(dirnames):
            path = parent_path / name
            if name in SKIP_DIRECTORIES:
                continue
            if any((relative_parent / name).is_relative_to(prefix)
                   for prefix in resource_prefixes):
                continue
            if tracked is not None and relative_parent / name not in tracked_directories:
                continue
            if path.is_symlink():
                links.append((path.relative_to(source),
                              _copied_link_target(source, path, kind="directory")))
            else:
                directories.append((relative_parent / name,
                                    stat.S_IMODE(path.stat().st_mode)))
                kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            path = parent_path / name
            relative = path.relative_to(source)
            if any(relative.is_relative_to(prefix) for prefix in resource_prefixes):
                continue
            if tracked is not None and relative not in tracked:
                continue
            if path.is_symlink():
                links.append((relative, _copied_link_target(source, path, kind="file")))
                continue
            if not path.is_file():
                raise ValueError(f"non-regular source entry cannot be copied: {path}")
            size = path.stat().st_size
            total += size
            if total > max_bytes:
                raise ValueError(f"working copy exceeds byte limit {max_bytes}")
            if len(files) + len(links) >= max_files:
                raise ValueError(f"working copy exceeds file limit {max_files}")
            files.append((relative, path, size, digest(path),
                          stat.S_IMODE(path.stat().st_mode)))
    included = {Path("."), *(relative for relative, _, _, _, _ in files)} | \
        {relative for relative, _ in directories}
    copied_paths = included | {relative for relative, _ in links}
    for binding in bindings:
        target = Path(binding["target"])
        if (target in copied_paths or any(
                path in target.parents for path, *_ in files) or any(
                path in target.parents for path, _ in links)):
            raise ValueError(f"resource target collides with copied source: {target}")
    for relative, target in links:
        resolved = (source / relative.parent / target).resolve().relative_to(source)
        if resolved not in included:
            raise ValueError(f"source symlink targets an excluded path: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.mkdir()
        for relative, _ in directories:
            (temporary / relative).mkdir(parents=True, exist_ok=True)
        for relative, path, _, expected, _ in files:
            target = temporary / relative
            shutil.copy2(path, target)
            if digest(target) != expected:
                raise RuntimeError(f"source changed while making working copy: {relative}")
        for relative, target in links:
            (temporary / relative).symlink_to(target)
        for binding in bindings:
            target = temporary / binding["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if binding["kind"] == "directory":
                target.mkdir()
            else:
                target.touch()
        for relative, mode in sorted(directories,
                                     key=lambda row: len(row[0].parts), reverse=True):
            os.chmod(temporary / relative, mode)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            for relative, _ in directories:
                candidate = temporary / relative
                if candidate.is_dir() and not candidate.is_symlink():
                    try:
                        os.chmod(candidate, 0o700)
                    except OSError:
                        pass
            shutil.rmtree(temporary)
    inventory = ([('file', str(relative), size, checksum, mode)
                  for relative, _, size, checksum, mode in files] +
                 [('directory', str(relative), mode) for relative, mode in directories] +
                 [('symlink', str(relative), target) for relative, target in links])
    inventory.sort(key=lambda one: one[1])
    if bindings:
        inventory = _inventory(destination, max_bytes=max_bytes, max_files=max_files)
    manifest = {"schema_version": 2, "created_at": now(), "source": str(source),
                "destination": str(destination), "source_git_revision": _git_revision(source),
                "source_tree_fingerprint": object_digest(inventory),
                "source_entries": inventory,
                "files": len(files), "symlinks": len(links),
                "directories": len(directories), "bytes": total,
                "copy_mode": "tracked_worktree" if tracked_only else "full_worktree",
                "untracked_assets_omitted": bool(tracked_only),
                "resource_bindings": bindings,
                "resource_source_exclusions": [path.as_posix() for path in resource_prefixes],
                "omitted_candidates": (omitted_candidates(source, copied_paths)
                                       if tracked_only else []),
                "resource_readiness": "unverified",
                "excluded_directory_names": sorted(SKIP_DIRECTORIES),
                "warning": "A byte copy isolates relative source edits, not arbitrary absolute "
                           "paths, subprocess mounts or external services."}
    atomic_json(destination.parent / "workspace_snapshot.json", manifest)
    return manifest


def existing(destination: Path, *, source: Path) -> dict[str, Any]:
    """Resume the same copy; never silently replace a possibly modified experiment."""
    destination = Path(destination).parent.resolve() / Path(destination).name
    manifest = read_json(destination.parent / "workspace_snapshot.json")
    if not destination.is_dir() or manifest.get("destination") != str(destination) or (
            manifest.get("source") != str(Path(source).resolve())):
        raise ValueError("working-copy manifest does not match this source and destination")
    from .workspace_resources import bindings_for
    bindings_for(destination)
    return manifest
