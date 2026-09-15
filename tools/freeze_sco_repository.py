"""Freeze benchmark code beside AutoSim code; share only cached data/runtimes."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from sco_compute_validation import BASE, prepare, save


def freeze_repository(root: Path) -> Path:
    source = prepare(root)
    repository = source / "RoboSynChallenge"
    if not repository.is_symlink():
        return source
    original = repository.resolve()
    repository.unlink()
    repository.mkdir()
    for name in ("configs", "scripts", "robosynchallenge", "policy", "docs", "launch"):
        shutil.copytree(original / name, repository / name,
            ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info", ".env"))
    for name in ("README.md", "LICENSE", "VERSION", "pyproject.toml", ".gitignore"):
        shutil.copy2(original / name, repository / name)
    for name in ("assets", "checkpoints", "lerobot_dataset", ".venv", "policy/act/.venv"):
        (repository / name).symlink_to(original / name, target_is_directory=True)
    manifest = json.loads((root / "source_manifest.json").read_text())
    for path in repository.rglob("*"):
        if path.is_file() and not path.is_symlink():
            manifest[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(root / "source_manifest.json", manifest)
    save(root / "benchmark_source.json", {"original_repository": str(original),
        "frozen_code": str(repository), "shared_assets_and_runtimes": [
            "assets", "checkpoints", "lerobot_dataset", ".venv", "policy/act/.venv"],
        "git_metadata": "not copied; source_manifest.json identifies exact source bytes"})
    return source
