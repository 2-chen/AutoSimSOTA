"""Bound CPU observations to repository code/config, without reading resource links.

This is not policy, dataset or installed-package identity. Unknown/oversized trees
remain diagnostic-only: inability to fingerprint never prevents running a probe.
"""
import os
from pathlib import Path

from .common import digest, object_digest

SKIP = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
TEXT_SUFFIXES = {'.py', '.pyi', '.sh', '.bash', '.yaml', '.yml', '.json', '.toml',
                '.ini', '.cfg', '.xml', '.urdf', '.mjcf', '.c', '.cpp', '.h', '.hpp'}


def source_identity(repo: Path, *, max_entries=20000, max_bytes=64*1024**2):
    """Hash bounded source/config contents and link targets; never follow symlinks."""
    try:
        repo = Path(repo)
        if repo.is_symlink() or not repo.is_dir():
            return None
        rows, total, count = [], 0, 0
        def inaccessible(error):
            raise error
        for current, directories, files in os.walk(repo, followlinks=False, onerror=inaccessible):
            directories[:] = sorted(name for name in directories if name not in SKIP)
            for name in sorted([*directories, *files]):
                count += 1
                if count > max_entries:
                    return None
                path = Path(current)/name
                relative = path.relative_to(repo).as_posix()
                if path.is_symlink():
                    rows.append([relative, 'link', os.readlink(path)])
                elif path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                    before = path.stat()
                    total += before.st_size
                    if total > max_bytes or before.st_size > 4*1024**2:
                        return None
                    sha = digest(path)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                            after.st_size, after.st_mtime_ns, after.st_ino):
                        return None
                    rows.append([relative, 'source', sha])
        return object_digest({'version': 1, 'sources_and_links': rows})
    except (OSError, ValueError, RuntimeError):
        return None
