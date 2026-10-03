"""Bounded public source discovery, not a snapshot of data or model outputs."""
import os
import time
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".toml", ".md", ".rst", ".sh", ".yaml", ".yml", ".ini", ".cfg", ".lock", ".txt"}
SKIP = {".git", "__pycache__", ".venv", "venv", "assets", "datasets", "dataset",
        "checkpoints", "outputs", "output", "logs", "log", "videos", "trajectories",
        "node_modules", "build", "dist", "wandb", "runs", "lerobot_dataset"}


def materialized_files(repo: Path, scope: Path = Path(".")) -> list[Path]:
    from .agent_runtime import _PRIVATE_BASENAMES, _PRIVATE_SUFFIXES
    repo = repo.resolve()
    target = repo / scope
    if (scope.is_absolute() or ".." in scope.parts or target.is_symlink()
            or not target.resolve().is_relative_to(repo)
            or any(part.lower() in _PRIVATE_BASENAMES or part.lower().startswith(".env") for part in scope.parts)):
        raise ValueError("unsafe materialized source scope")
    if not target.exists():
        return []
    def accept(path):
        return (not path.is_symlink() and path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
            and path.suffix.lower() not in _PRIVATE_SUFFIXES
            and path.name.lower() not in _PRIVATE_BASENAMES
            and not path.name.lower().startswith(".env"))
    if target.is_file():
        return [target.relative_to(repo)] if accept(target) else []
    found, count, total = [], 0, 0
    deadline = time.monotonic() + 10
    for parent, directories, files in os.walk(target, followlinks=False):
        # Generated diagnostic Python envs are not benchmark source dependencies.
        if (Path(parent) / "pyvenv.cfg").exists() or (Path(parent) / "conda-meta").is_dir():
            directories[:] = []
            continue
        count += len(directories) + len(files)
        if count > 100000 or time.monotonic() >= deadline:
            raise ValueError("source scan incomplete; narrow/register actual source dependencies")
        directories[:] = sorted(name for name in directories if name.lower() not in SKIP
            and name.lower() not in _PRIVATE_BASENAMES and not name.startswith(".env")
            and not (Path(parent) / name).is_symlink())
        for name in sorted(files):
            path = Path(parent) / name
            if accept(path):
                total += path.stat().st_size
                if total > 64 * 1024**2:
                    raise ValueError("materialized source exceeds 64 MiB; register narrower source dependencies")
                found.append(path.relative_to(repo))
    return sorted(found)
