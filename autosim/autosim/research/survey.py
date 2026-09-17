"""What an unknown repository says about itself, before anyone interprets it.

The system used to learn a benchmark's shape from a hand-written adapter: a person read
the repository and typed its facts into a file. That works, but it means the system can
only ever optimize benchmarks somebody has already onboarded, and the cost of the next
one falls on whoever wants it.

This module does the reading. It is deliberately not an interpreter. It reports what
files exist, what the manifests declare, and which signals appear in which sources -- and
it names no benchmark. Deciding that a particular file is "the evaluator" is the scout's
job, where a model can be argued with and wrong in a way the report makes visible.

Two rules:

* **Facts, not verdicts.** A file is reported with the signals it contains and where they
  were found, never with a label like "this is the expert". A wrong label in here would be
  invisible; a wrong label in the scout's proposal is checked against these facts.
* **Bounded, always.** A benchmark checkout can be hundreds of gigabytes of assets. The
  survey reads a fixed budget of directories, entries and bytes, and says what it skipped
  rather than silently sampling.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import read_json


#: Directories that are never worth walking: version control, caches, virtualenvs, and
#: the build products of packaging. Listed rather than discovered so the cost is fixed.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", "env", ".tox", "build", "dist", ".eggs",
})

#: Extensions whose contents are read as text when they are small enough.
TEXT_SUFFIXES = frozenset({
    ".py", ".toml", ".cfg", ".ini", ".txt", ".md", ".rst", ".yaml", ".yml", ".json",
    ".sh", ".bddl", ".xml", ".cfg",
})

#: Extensions that are plausibly a dataset or a learned model. Reported by size, not read.
DATA_SUFFIXES = frozenset({".hdf5", ".h5", ".pkl", ".pickle", ".npz", ".npy", ".parquet",
                           ".zarr", ".lmdb", ".tfrecord", ".bag", ".mcap", ".csv", ".jsonl"})
MODEL_SUFFIXES = frozenset({".ckpt", ".pt", ".pth", ".safetensors", ".bin", ".onnx",
                            ".pkl", ".joblib", ".msgpack"})

#: Signal -> phrases worth reporting. These are *evidence*, not rules: the scout is told
#: which phrases were found where, and decides what they mean. Adding a phrase here can
#: only make the report more informative; nothing branches on any of them.
SIGNALS: dict[str, tuple[str, ...]] = {
    "evaluation": ("def evaluate", "def eval_", "success_rate", "is_success", "check_success",
                   "rollout", "num_episodes", "eval_episodes", "initial_state", "init_state"),
    "training": ("optimizer", "loss", "backward()", "checkpoint", "save_checkpoint",
                 "lr_scheduler", "dataloader", "DataLoader"),
    "collection": ("collect", "dataset_save", "save_episode", "record_episode", "to_hdf5",
                   "add_episode"),
    "expert": ("expert", "scripted", "oracle", "motion_plan", "MotionPlanning", "ik_",
               "inverse_kinematics", "controller_config", "OSC", "operational_space"),
    # Teleoperation is the signal that matters most and is the easiest to mistake for an
    # expert: a collector that waits for a human produces no trajectories on its own.
    "teleoperation": ("input2action", "input_utils", "SpaceNav", "keyboard", "start_control",
                      "pynput", "getch", "device", "joystick"),
    "simulation": ("robosuite", "mujoco", "sapien", "isaacgym", "isaacsim", "pybullet",
                   "gymnasium", "gym.make", "habitat", "genesis", "embodichain", "warp"),
    "language": ("language_instruction", "instruction", "task_description", "prompt",
                 "bddl", "pddl", "goal_predicate"),
}


def _is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def _read_text(path: Path, limit: int) -> str | None:
    try:
        if path.stat().st_size > limit:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _tree(repo: Path, *, max_depth: int, max_entries: int) -> dict[str, Any]:
    """A bounded listing: depth, extension counts, and the largest files.

    Sizes are reported because they distinguish a task definition from a dataset, and
    because a directory that is 99% of the checkout by weight is a fact about the
    benchmark worth having before asking a model anything.
    """
    extensions: Counter[str] = Counter()
    largest: list[dict[str, Any]] = []
    directories: set[str] = set()
    counted = 0
    truncated = False
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        depth = len(Path(root).relative_to(repo).parts)
        if depth >= max_depth:
            dirs[:] = []
        directories.add(str(Path(root).relative_to(repo)) or ".")
        for name in sorted(files):
            counted += 1
            if counted > max_entries:
                truncated = True
                break
            path = Path(root) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            extensions[path.suffix.lower() or "(none)"] += 1
            largest.append({"path": str(path.relative_to(repo)), "bytes": size})
        if truncated:
            break
    largest.sort(key=lambda row: -row["bytes"])
    return {"entries": counted, "truncated": truncated,
            "directories": sorted(directories)[:200],
            "extensions": dict(extensions.most_common(40)),
            "largest_files": largest[:40],
            "total_bytes": sum(row["bytes"] for row in largest)}


def _manifests(repo: Path, *, limit: int) -> dict[str, str]:
    """Files whose contents are a declaration of what this project is and needs."""
    names = ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
             "environment.yml", "environment.yaml", "conda.yml", "install.sh",
             "Makefile", "Dockerfile", "README.md", "README.rst", "LICENSE")
    found: dict[str, str] = {}
    for name in names:
        path = repo / name
        if not path.is_file():
            continue
        text = _read_text(path, limit)
        if text is not None:
            found[name] = text
    return found


def _signal_scan(repo: Path, *, max_depth: int, max_files: int,
                 max_bytes: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Which signals appear in which sources, with a line of context for each.

    Every file that matches at least one signal is reported with every phrase it matched,
    so the scout can tell a file that *collects demonstrations* from one that *waits for a
    human to collect demonstrations* -- a distinction that decides whether this benchmark
    can supply new data at all, and one a filename cannot express.
    """
    hits: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    scanned = 0
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        if len(Path(root).relative_to(repo).parts) >= max_depth:
            dirs[:] = []
        for name in sorted(files):
            path = Path(root) / name
            if not _is_text(path):
                continue
            scanned += 1
            if scanned > max_files:
                break
            text = _read_text(path, max_bytes)
            if text is None:
                continue
            lowered = text.lower()
            matched = sorted(signal for signal, phrases in SIGNALS.items()
                             if any(phrase.lower() in lowered for phrase in phrases))
            if not matched:
                continue
            totals.update(matched)
            hits.append({"path": str(path.relative_to(repo)),
                         "lines": text.count("\n") + 1,
                         "signals": matched,
                         "phrases": {signal: sorted(
                             phrase for phrase in SIGNALS[signal]
                             if phrase.lower() in lowered)
                             for signal in matched}})
    hits.sort(key=lambda row: (-len(row["signals"]), row["path"]))
    return hits[:120], dict(totals)


def _fragment(repo: Path, path: str) -> str | None:
    """The shortest leading path fragment that could name where an outside asset lives.

    `datasets/libero/libero_10/x_demo.hdf5` yields `datasets/libero`: long enough to be
    distinctive, short enough that a script naming the download location will contain it.
    """
    target = Path(path)
    try:
        relative = target.relative_to(repo.parent)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 2 or parts[0] == repo.name:
        return None
    return "/".join(parts[:2])


def _asset_roots(repo: Path, extra: Iterable[Path]) -> list[Path]:
    """Where to look for datasets and checkpoints: the checkout, then its neighbours.

    A benchmark's code and its data are routinely kept apart -- downloaded assets land in
    a sibling directory the code does not mention -- so searching only inside the checkout
    would report "no dataset" for a benchmark whose dataset is right there.
    """
    roots = [repo, *extra, repo.parent]
    seen: list[Path] = []
    for root in roots:
        resolved = Path(root).expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in seen:
            seen.append(resolved)
    return seen


def _datasets(roots: list[Path], *, max_depth: int, max_candidates: int) -> list[dict[str, Any]]:
    """Large structured files, plus directories that look like a dataset root."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            if len(Path(directory).relative_to(root).parts) >= max_depth:
                dirs[:] = []
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() not in DATA_SUFFIXES:
                    continue
                if str(path) in seen:
                    continue
                seen.add(str(path))
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size < 1_000_000:
                    continue
                found.append({"path": str(path), "bytes": size,
                              "suffix": path.suffix.lower(),
                              "found_under": str(root)})
        if len(found) >= max_candidates:
            break
    found.sort(key=lambda row: -row["bytes"])
    return found[:max_candidates]


def _models(roots: list[Path], *, max_depth: int, max_candidates: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            if len(Path(directory).relative_to(root).parts) >= max_depth:
                dirs[:] = []
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() not in MODEL_SUFFIXES:
                    continue
                if str(path) in seen:
                    continue
                seen.add(str(path))
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                # A learned policy is never a few kilobytes; small pickles are caches.
                if size < 100_000:
                    continue
                found.append({"path": str(path), "bytes": size,
                              "suffix": path.suffix.lower(),
                              "found_under": str(root)})
        if len(found) >= max_candidates:
            break
    found.sort(key=lambda row: -row["bytes"])
    return found[:max_candidates]


def _reference_search(repo: Path, fragments: list[str], *, max_depth: int,
                      max_bytes: int, limit: int = 6) -> dict[str, Any]:
    """Which files in the repository mention where an outside asset actually lives.

    An asset found outside the checkout is not evidence about this benchmark -- a sibling
    project's checkpoints sit on the same disk and look the same. What distinguishes them is
    whether the repository *itself* names that location: a download script, a config, a
    default path. Without that, the asset is somebody else's and the honest answer is no;
    with it, the asset is this benchmark's and merely stored elsewhere.
    """
    found: dict[str, list[str]] = {}
    for fragment in fragments:
        hits: list[str] = []
        for root, dirs, files in os.walk(repo):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            if len(Path(root).relative_to(repo).parts) >= max_depth:
                dirs[:] = []
            for name in files:
                path = Path(root) / name
                if not _is_text(path):
                    continue
                text = _read_text(path, max_bytes)
                if text is None or fragment not in text:
                    continue
                hits.append(str(path.relative_to(repo)))
                if len(hits) >= limit:
                    break
            if len(hits) >= limit:
                break
        found[fragment] = hits
    return found


def _structured_roots(roots: list[Path], *, max_depth: int) -> list[dict[str, Any]]:
    """Directories that carry a dataset manifest, whatever the manifest is called.

    `meta/info.json` is LeRobot, `dataset_info.json` is another convention, and neither is
    universal; this reports the file, not the format, and lets the scout read it.
    """
    names = ("info.json", "dataset_info.json", "episodes.jsonl", "manifest.json",
             "modality.json", "stats.json")
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            if len(Path(directory).relative_to(root).parts) >= max_depth:
                dirs[:] = []
            present = [name for name in names if name in files]
            if not present:
                continue
            key = str(directory)
            if key in seen:
                continue
            seen.add(key)
            found.append({"directory": key, "manifests": present})
    return found[:60]


def _hdf5_structure(path: Path, *, max_groups: int, max_datasets: int) -> dict[str, Any]:
    """The shape of an HDF5 file: group names, dataset names, dtypes and dimensions.

    Observation and action dimensionality live in the recorded data, not in the source
    that wrote it. A reader who only sees the code ends up guessing, which is how a
    contract acquires a state dimension of one.
    """
    try:
        import h5py
    except ImportError:
        return {"error": "h5py is not installed, so the structure could not be read"}
    out: dict[str, Any] = {"groups": {}}
    try:
        with h5py.File(path, "r") as handle:
            out["root_attrs"] = {str(k): str(v)[:200] for k, v in handle.attrs.items()}
            out["root_keys"] = list(handle.keys())[:max_groups]

            def describe(group: Any, prefix: str, budget: list[int]) -> dict[str, Any]:
                rows: dict[str, Any] = {}
                for name in sorted(group.keys()):
                    if budget[0] <= 0:
                        break
                    item = group[name]
                    budget[0] -= 1
                    if isinstance(item, h5py.Group):
                        rows[name] = {"group": describe(item, f"{prefix}/{name}", budget)}
                    else:
                        rows[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}
                return rows

            budget = [max_datasets]
            for key in out["root_keys"]:
                item = handle[key]
                if isinstance(item, h5py.Group):
                    members = sorted(item.keys())
                    out["groups"][key] = {
                        "count": len(members),
                        "examples": members[:3],
                        # A recorded file usually carries its provenance in its own
                        # attributes -- the task definition it was generated from, the tag of
                        # the release it belongs to. That outranks any inference from where
                        # the file happens to sit on disk, and it is the only thing that
                        # settles ownership for data stored outside the checkout.
                        "attrs": {str(k): str(v)[:300] for k, v in item.attrs.items()},
                        "first": describe(item[members[0]], key, budget) if members else {},
                    }
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _table_structure(path: Path, *, limit: int) -> dict[str, Any]:
    """Parquet/JSONL columns and types -- the tabular analogue of the HDF5 report."""
    if path.suffix.lower() == ".jsonl":
        head = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for _ in range(limit):
                    line = stream.readline()
                    if not line:
                        break
                    head.append(json.loads(line))
        except (OSError, ValueError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"records": head}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"error": "pyarrow is not installed"}
    try:
        schema = pq.read_schema(path)
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"columns": {name: str(schema.field(name).type) for name in schema.names}}


def artifact_structure(path: Path, *, max_groups: int = 6, max_datasets: int = 30,
                       limit: int = 2) -> dict[str, Any]:
    """What is inside a data artifact, bounded, without loading the data itself."""
    suffix = path.suffix.lower()
    if suffix in {".hdf5", ".h5"}:
        body = _hdf5_structure(path, max_groups=max_groups, max_datasets=max_datasets)
    elif suffix in {".parquet", ".jsonl"}:
        body = _table_structure(path, limit=limit)
    else:
        body = {"note": "no reader is registered for this format; only its size is known"}
    return {"path": str(path), "bytes": path.stat().st_size, "suffix": suffix, **body}


def recorded_provenance(structures: list[dict[str, Any]], repo: Path) -> dict[str, Any]:
    """Paths an asset records about itself, and whether each one resolves in the checkout.

    A demonstration file written by a benchmark names the task definition it was generated
    from. If that name resolves inside the repository, the file belongs to the repository
    wherever it was stored. This is the only ownership evidence that survives relocation,
    which is why it decides the question rather than the directory the file sits in.
    """
    recorded: list[str] = []
    resolved: list[str] = []
    for row in structures:
        for group in (row.get("groups") or {}).values():
            for value in (group.get("attrs") or {}).values():
                for candidate in _embedded_paths(value):
                    recorded.append(candidate)
                    if (repo / candidate).exists():
                        resolved.append(candidate)
    return {"recorded_paths": sorted(set(recorded))[:20],
            "resolve_inside_repo": sorted(set(resolved))[:20]}


def _embedded_paths(value: str) -> list[str]:
    """Paths named in an attribute, including inside a nested JSON-looking string."""
    found: list[str] = []
    text = value.strip()
    if text.startswith("{"):
        try:
            found.extend(_walk_paths(json.loads(text)))
        except ValueError:
            pass
    if "/" in text and " " not in text and len(text) < 300:
        found.append(text)
    return found


def _walk_paths(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node] if "/" in node and " " not in node and len(node) < 300 else []
    if isinstance(node, dict):
        return [path for value in node.values() for path in _walk_paths(value)]
    if isinstance(node, list):
        return [path for value in node for path in _walk_paths(value)]
    return []


def _peek(path: str, *, lines: int = 60) -> dict[str, Any]:
    """The head of a file the scout asked to see, so it never needs filesystem access."""
    target = Path(path)
    if not target.is_file():
        return {"path": path, "exists": False}
    text = _read_text(target, 2_000_000)
    if text is None:
        return {"path": path, "exists": True, "readable": False,
                "bytes": target.stat().st_size}
    body = text.splitlines()
    return {"path": path, "exists": True, "readable": True,
            "bytes": target.stat().st_size, "total_lines": len(body),
            "head": "\n".join(body[:lines])}


def peek_many(paths: Iterable[str], *, lines: int = 60, limit: int = 12) -> list[dict[str, Any]]:
    """Read the heads of at most `limit` files. Bounded because a model asked for them."""
    return [_peek(path, lines=lines) for path in list(paths)[:limit]]


def survey(repo: Path, *, extra_roots: Iterable[Path] = (), max_depth: int = 4,
           max_entries: int = 6000, max_scan_files: int = 900,
           max_bytes: int = 400_000, asset_depth: int = 3,
           max_assets: int = 40, manifest_bytes: int = 60_000,
           structure_samples: int = 3) -> dict[str, Any]:
    """Everything a reader would want before forming an opinion about this repository.

    The report is a fact sheet. It names no benchmark and reaches no conclusion, which is
    what makes it reusable for the next repository as well as this one.
    """
    repo = Path(repo).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"not a directory: {repo}")
    roots = _asset_roots(repo, extra_roots)
    signals, totals = _signal_scan(repo, max_depth=max_depth + 2,
                                   max_files=max_scan_files, max_bytes=max_bytes)
    datasets = _datasets(roots, max_depth=asset_depth, max_candidates=max_assets)
    models = _models(roots, max_depth=asset_depth, max_candidates=max_assets)
    fragments = sorted({_fragment(repo, row["path"]) for row in datasets + models} - {None})
    return {
        "repo": str(repo),
        "project_manifests": _manifests(repo, limit=manifest_bytes),
        "tree": _tree(repo, max_depth=max_depth, max_entries=max_entries),
        "signal_sources": signals,
        "signal_totals": totals,
        "datasets": datasets,
        "model_artifacts": models,
        # Which of these are actually *this* repository's is not something a filename or an
        # existence check can settle: a sibling benchmark's checkpoint sits on the same disk
        # and looks identical. So the provenance is reported rather than judged, and the
        # scout is told that anything outside the checkout needs a reason to be believed.
        "asset_provenance": {
            "repository_mentions": _reference_search(repo, [f for f in fragments if f],
                                                     max_depth=max_depth + 2,
                                                     max_bytes=max_bytes),
            "inside_repo": sum(1 for row in datasets + models
                               if str(row["path"]).startswith(str(repo) + os.sep)),
            "outside_repo": sum(1 for row in datasets + models
                                if not str(row["path"]).startswith(str(repo) + os.sep)),
            "note": "a file inside the checkout is this benchmark's. One outside it is "
                    "evidence only if something else ties it here: a script in the "
                    "repository naming that location (see repository_mentions), or "
                    "provenance recorded in the file's own attributes (see "
                    "dataset_structure.*.groups.*.attrs). A neighbour's files look "
                    "identical otherwise.",
        },
        # The dimensions, camera names and horizon a contract needs are recorded in the
        # data, so one representative artifact is opened and described. Structure only:
        # the contents of a 2 GB demonstration file are never read.
        "dataset_structure": [artifact_structure(Path(row["path"]))
                              for row in datasets[:structure_samples]],
        # Resolved here rather than left to the reader: a recorded path either resolves
        # inside the checkout or it does not, and that verdict is what settles who owns a
        # file stored elsewhere. Reporting the raw attribute and expecting the reader to
        # test it puts a filesystem question in front of a model that has no filesystem.
        "recorded_provenance": recorded_provenance(
            [artifact_structure(Path(row["path"])) for row in datasets[:structure_samples]],
            repo),
        "structured_roots": _structured_roots(roots, max_depth=asset_depth),
        "searched_roots": [str(root) for root in roots],
        "budget": {"max_depth": max_depth, "max_entries": max_entries,
                   "max_scan_files": max_scan_files, "max_bytes_per_file": max_bytes},
    }


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    """The survey at a size that fits in one message, for the scout's first pass.

    The full report stays on disk next to it: a model that wants a file's contents says so,
    and the caller reads it. Compressing rather than truncating keeps the shape visible.
    """
    return {
        "repo": report["repo"],
        "project_manifests": sorted(report["project_manifests"]),
        "manifest_excerpts": {name: text[:1200] for name, text in
                              report["project_manifests"].items()
                              if name in ("setup.py", "pyproject.toml", "requirements.txt",
                                          "environment.yml", "README.md")},
        "tree_extensions": report["tree"]["extensions"],
        "tree_truncated": report["tree"]["truncated"],
        "largest_files": report["tree"]["largest_files"][:15],
        "signal_totals": report["signal_totals"],
        "signal_sources": [{k: row[k] for k in ("path", "signals")}
                           for row in report["signal_sources"][:40]],
        "dataset_count": len(report["datasets"]),
        "datasets": report["datasets"][:10],
        "model_count": len(report["model_artifacts"]),
        "model_artifacts": report["model_artifacts"][:10],
        "asset_provenance": report["asset_provenance"],
        "structured_roots": report["structured_roots"][:10],
        "dataset_structure": report["dataset_structure"],
        "recorded_provenance": report["recorded_provenance"],
        "asset_provenance": report["asset_provenance"],
        "searched_roots": report["searched_roots"],
    }


def load_declaration(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None
