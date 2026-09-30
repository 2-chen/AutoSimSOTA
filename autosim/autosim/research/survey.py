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
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

from .common import read_json, sanitize_model_text


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


def _walk_by_depth(root: Path, *, max_depth: int | None = None):
    """Every directory under `root`, shallowest first, alphabetically within a depth.

    Yields `(directory, subdirectory names, file names)`, like `os.walk`, but breadth-first.

    **Why this exists rather than `os.walk`.** `os.walk` descends depth-first in sorted order,
    so a walk with a finite budget spends the whole of it inside whichever large directory
    sorts first -- and a capital letter sorts before every lower-case one, so that is usually
    a vendored library. On RoboTwin it was `XPolicyLab/`: 119 of the 120 signal sources the
    scan reached were inside it, and `envs/`, `script/` and `task_config/` were never opened
    at all. The declaration that came out of that survey had an **empty optimisation space**,
    because the survey had read a vendored policy library instead of the benchmark.

    That fault was found and corrected three separate times, in three separate walks: the
    file listing was reordered by depth, the dataset ranking grew a per-directory cap, and
    each time the next walk turned out to have the same problem. Three fixes to one bug is
    the argument for one traversal: a walk with a budget has to spend it on the shallow
    entries first, and that is a property of *walking*, not of what the walk is looking for.
    """
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    while queue:
        directory, depth = queue.popleft()
        try:
            entries = sorted(os.scandir(directory), key=lambda one: one.name)
        except OSError:
            # An unreadable directory yields nothing rather than ending the walk. A survey
            # that dies on one bad permission reports nothing about the rest of the checkout.
            continue
        dirs: list[str] = []
        files: list[str] = []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in SKIP_DIRS:
                        continue
                    if max_depth is not None and depth >= max_depth:
                        continue
                    dirs.append(entry.name)
                elif entry.is_file(follow_symlinks=False):
                    files.append(entry.name)
            except OSError:
                continue
        yield directory, dirs, files
        for name in dirs:
            queue.append((directory / name, depth + 1))


def _tree(repo: Path, *, max_depth: int, max_entries: int) -> dict[str, Any]:
    """A bounded listing: depth, extension counts, and the largest files.

    Sizes are reported because they distinguish a task definition from a dataset, and because
    a directory that is 99% of the checkout by weight is a fact about the benchmark worth
    having before asking a model anything.

    Walks with `_walk_by_depth`, which is where the argument for doing so is written down.
    """
    by_depth: dict[int, list[dict[str, Any]]] = {}
    directories: set[str] = set()
    extensions: Counter[str] = Counter()
    walked = 0
    truncated = False
    for root, dirs, files in _walk_by_depth(repo, max_depth=max_depth):
        depth = len(Path(root).relative_to(repo).parts)
        directories.add(str(Path(root).relative_to(repo)) or ".")
        for name in files:
            walked += 1
            if walked > max_entries * 4:
                # A hard stop so a huge checkout cannot be walked forever. Four times the
                # reporting budget, because the budget now bounds what is *reported* and this
                # bounds what is *looked at*.
                truncated = True
                break
            path = Path(root) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            row = {"path": str(path.relative_to(repo)), "bytes": size}
            by_depth.setdefault(depth, []).append(row)
        if truncated:
            break

    seen = 0
    largest: list[dict[str, Any]] = []
    for depth in sorted(by_depth):
        for row in by_depth[depth]:
            if seen >= max_entries:
                truncated = True
                break
            seen += 1
            extensions[Path(row["path"]).suffix.lower() or "(none)"] += 1
            largest.append(row)
        if seen >= max_entries:
            break
    shallow = sorted(largest, key=lambda row: -row["bytes"])
    return {"entries": seen, "truncated": truncated, "walked": walked,
            # Shallowest first, and the count of what was left out. Alphabetical order put
            # `XPolicyLab/...` at the front here too -- the same capital-letter accident that
            # ordered the walk -- so a reader looking for the checkout's own directories found
            # the vendored library's instead, and had no way to tell that `envs/` and
            # `task_config/` were on the list at all.
            "directories": sorted(directories, key=lambda one: (one.count("/"), one))[:400],
            "directory_count": len(directories),
            "extensions": dict(extensions.most_common(40)),
            # The shallow files, by size: what a reader needs first is the top of the
            # checkout, and the largest things in it, both of which are now guaranteed to
            # have been seen.
            "largest_files": shallow[:40],
            "seen_files": [row["path"] for row in largest][:400],
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


def _candidate_sources(repo: Path, *, max_depth: int) -> dict[str, list[Path]]:
    """Text files worth reading, grouped by the top-level directory they sit under.

    Grouped rather than listed because the caller has a budget and needs to spread it. The
    key is the top-level directory, so `envs/beat_block_hammer.py` and `envs/tools/x.py`
    share one group while `XPolicyLab/` has its own -- which is the unit a budget has to be
    spent across.
    """
    groups: dict[str, list[Path]] = {}
    for directory, dirs, files in _walk_by_depth(repo, max_depth=max_depth):
        for name in files:
            path = directory / name
            if not _is_text(path):
                continue
            relative = path.relative_to(repo)
            key = relative.parts[0] if len(relative.parts) > 1 else "(the checkout itself)"
            groups.setdefault(key, []).append(path)
    return groups


def _spread_over_subtrees(groups: dict[str, list[Path]], *, limit: int):
    """Take from every group in turn, rather than exhausting one and moving on.

    Depth order alone was not enough, and the reason is worth stating: within one depth the
    directories are still visited alphabetically, so a vendored `XPolicyLab/` with four
    hundred files at depth 1 spent the whole budget before `envs/` -- which has one file at
    the same depth -- was reached. Ordering fixed *which* directory is visited first; only
    interleaving fixes *how much* of the budget any one of them can take.

    The groups are visited in round-robin, so a checkout gets read in proportion to how many
    top-level directories it has rather than in proportion to how large one of them is.
    """
    cursors = {key: 0 for key in groups}
    order = sorted(groups, key=lambda key: (":(the checkout itself)" != ":" + key, key))
    yielded = 0
    while yielded < limit:
        progressed = False
        for key in order:
            if yielded >= limit:
                break
            index = cursors[key]
            if index >= len(groups[key]):
                continue
            cursors[key] = index + 1
            progressed = True
            yield groups[key][index]
            yielded += 1
        if not progressed:
            return


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
    for path in _spread_over_subtrees(_candidate_sources(repo, max_depth=max_depth),
                                      limit=max_files):
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


def _datasets(roots: list[Path], *, max_depth: int, max_candidates: int,
              max_directories: int = 60000) -> list[dict[str, Any]]:
    """Large structured files, plus directories that look like a dataset root.

    The depth is bounded by the *cost* rather than by a number chosen for one layout. A
    dataset is recognised by its suffix and its size, not by how deep it sits, and how deep
    it sits is not a constant: LIBERO ships its demonstrations three levels down and RoboTwin
    six, so a limit of three found the first and walked past the second entirely. What that
    cost: with no demonstration found, nothing could tell the scout what `state_dim` is, the
    model wrote zeros, the contract validator rejected them as a value, and the run raised
    after three attempts -- a benchmark the scout could otherwise describe produced nothing.

    Bounded by directories visited instead, which is what the depth was standing in for.
    """
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    visited = 0
    for root in roots:
        for directory, dirs, files in _walk_by_depth(root, max_depth=max_depth):
            visited += 1
            if visited > max_directories:
                break
            for name in files:
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
                relative = path.relative_to(root if path.is_relative_to(root) else path.parent)
                found.append({"path": str(path), "bytes": size,
                              "suffix": path.suffix.lower(),
                              "found_under": str(root),
                              "depth": len(relative.parts) - 1,
                              "top": relative.parts[0] if len(relative.parts) > 1 else ""})
        if len(found) >= max_candidates * 4:
            break

    # Shallow first, then big: what a benchmark ships sits near its own root, and what a
    # vendored library *produced* sits deep inside it. Ranking by size alone put forty files
    # under `XPolicyLab/policy/ACT/processed_data/` at the top -- 348 MB each against the
    # benchmark's own 6.6 MB -- and not one of RoboTwin's own demonstrations made the list,
    # so the contract reader had nothing of this benchmark's to read.
    #
    # And a cap per top-level directory, because "shallow first" still lets one subtree take
    # everything if it is flat. This is the same correction as the file listing above: a
    # budget that one directory can spend entirely is a budget that will be.
    # This checkout's own assets first, and only then depth and size.
    #
    # The parent directory is searched because a benchmark's data is routinely kept beside
    # its code rather than inside it -- and on any machine with more than one benchmark
    # installed, "beside it" is where the *others* are. Measured on four checkouts sitting
    # side by side: `robomimic`, which ships 29 of its own demonstrations, was shown
    # twenty-five files belonging to LIBERO, every one of them found under the shared parent.
    # Its own six gigabytes lost to LIBERO's thirty-two because depth is identical between
    # siblings and the tiebreak is size.
    #
    # The contract reader is then handed another benchmark's HDF5 and reports *its* state and
    # action dimensions as this one's -- which is a wrong declaration produced by a correct
    # reading of the wrong file. `found_under` already records which root a file came from;
    # this is that fact being used instead of being carried and ignored.
    own = str(roots[0]) if roots else ""
    found.sort(key=lambda row: (row["found_under"] != own, row["depth"], -row["bytes"]))
    per_top: Counter[str] = Counter()
    room = max(4, max_candidates // 4)
    kept: list[dict[str, Any]] = []
    for row in found:
        # The allowance is per subtree, and "subtree" means a directory under the root this
        # file was found in. Under the checkout that is its top-level directory, which is
        # what it has always been.
        #
        # Under the *parent* -- searched because a benchmark's assets are routinely kept
        # beside its code rather than inside it -- the top-level directory is shared by every
        # benchmark on the machine, so one benchmark's data can spend the whole allowance and
        # another's never appears. Measured on four checkouts side by side: `datasets/libero`
        # filled all ten slots of the `datasets` group, and `robomimic` -- which ships 29 of
        # its own demonstrations, six gigabytes of them -- was handed none of them, then had
        # LIBERO's HDF5 read as its own contract. One level deeper separates them, which is
        # the level the two benchmarks actually differ at.
        key = row["top"]
        if row["found_under"] != own:
            parts = Path(row["path"]).relative_to(row["found_under"]).parts
            key = "/".join(parts[:2]) if len(parts) > 2 else row["top"]
        if per_top[key] >= room:
            continue
        per_top[key] += 1
        kept.append((key, row))
        if len(kept) >= max_candidates:
            break

    # Interleaved across groups, because the caller shows the first ten.
    #
    # A cap decides *how much* of the budget one subtree can take; it does not decide what a
    # reader sees first, and `summarise` shows `datasets[:10]`. So even with the cap working,
    # the ten were whatever sorted biggest -- and on this machine that was ten of LIBERO's
    # for every benchmark surveyed. Interleaving is the same correction the file listing
    # needed for the same reason: ordering fixes which directory is visited first, only
    # interleaving fixes how much of the reader's attention any one of them can take.
    order: list[str] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for key, row in kept:
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)
    out: list[dict[str, Any]] = []
    for index in range(max((len(rows) for rows in buckets.values()), default=0)):
        for key in order:
            if index < len(buckets[key]):
                out.append(buckets[key][index])
    return out


def _models(roots: list[Path], *, max_depth: int, max_candidates: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for directory, dirs, files in _walk_by_depth(root, max_depth=max_depth):
            for name in files:
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
        for root, dirs, files in _walk_by_depth(repo, max_depth=max_depth):
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
        for directory, dirs, files in _walk_by_depth(root, max_depth=max_depth):
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

            def describe(item: Any, prefix: str, budget: list[int]) -> dict[str, Any]:
                """Whatever this node is, described as what it is.

                A group is recursed into; anything else is reported by shape and dtype. The
                previous version called `.keys()` on the node it was handed, which is right
                for a group and raises `AttributeError` for a dataset -- and a dataset is
                what a group's first member usually is. RoboTwin's file has
                `observations/cam_head`, the reader was handed the dataset, and the whole
                survey died with it: one unrecognised shape in one art, object, and nothing
                else was reported at all.
                """
                if not isinstance(item, h5py.Group):
                    shape = getattr(item, "shape", None)
                    return {"shape": list(shape) if shape is not None else None,
                            "dtype": str(getattr(item, "dtype", ""))}
                rows: dict[str, Any] = {}
                for name in sorted(item.keys()):
                    if budget[0] <= 0:
                        break
                    budget[0] -= 1
                    rows[name] = describe(item[name], f"{prefix}/{name}", budget)
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
                else:
                    # A top-level dataset rather than a group. Reported like any other node
                    # instead of being skipped: a file whose root is flat is a shape this
                    # reader has to survive, whether or not it is one it expected.
                    shape = getattr(item, "shape", None)
                    out["groups"][key] = {
                        "count": 1, "examples": [key],
                        "shape": list(shape) if shape is not None else None,
                        "dtype": str(getattr(item, "dtype", ""))}
    except Exception as exc:                                     # noqa: BLE001
        # Anything at all: a file this reader does not understand is a file it has nothing
        # to say about, and it is not a reason to lose the rest of the survey. The previous
        # `except OSError` let an `AttributeError` from one unrecognised dataset take down
        # the whole run.
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


def _json_schema(value: Any, *, depth: int = 0) -> dict[str, Any]:
    """Return JSON field/type structure without values or record counts."""
    if depth >= 6:
        if isinstance(value, dict):
            return {"type": "object", "keys": sorted(map(str, value))[:80]}
        if isinstance(value, list):
            return {"type": "array"}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, (int, float)):
            return {"type": "number"}
        if value is None:
            return {"type": "null"}
        return {"type": "string" if isinstance(value, str) else type(value).__name__}
    if isinstance(value, dict):
        return {"type": "object", "fields": {
            str(key): _json_schema(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))[:80]}}
    if isinstance(value, list):
        shapes: list[dict[str, Any]] = []
        for item in value[:2]:
            shape = _json_schema(item, depth=depth + 1)
            if shape not in shapes:
                shapes.append(shape)
        return {"type": "array", "item_shapes": shapes}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, (int, float)):
        return {"type": "number"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, str):
        return {"type": "string"}
    return {"type": type(value).__name__}


_PRIVATE_PATH_PARTS = frozenset({
    ".ssh", "secrets", "secret", "credentials", "credential", "private_keys",
    "data", "dataset", "datasets", "demonstrations", "demonstration", "demos",
    "demo", "episodes", "episode", "trajectories", "trajectory", "samples", "sample",
    "checkpoints", "checkpoint", "weights", "weight",
    "videos", "video", "recordings", "recording",
})
_PRIVATE_SUFFIXES = DATA_SUFFIXES | MODEL_SUFFIXES | frozenset({
    ".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".gif", ".png", ".jpg",
    ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".wav", ".mp3", ".flac", ".pem",
    ".p12", ".pfx",
})


def _peek(path: str, *, repo: Path, lines: int = 60) -> dict[str, Any]:
    """Read only bounded, non-sensitive text beneath the repository root.

    Triage output is model-controlled. Resolving it against a checkout and checking the
    final target prevents ``../`` and symlink escapes; suffix/path checks keep checkpoints,
    demonstrations, trajectories, media, and credentials out of the prompt entirely.
    """
    root = Path(repo).expanduser().resolve(strict=True)
    requested = Path(path).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    try:
        target = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return {"path": "[invalid-path]", "exists": False, "readable": False,
                "why": "requested path could not be resolved"}
    if not target.is_relative_to(root):
        # Do not even report whether an arbitrary host target exists. The model only needs
        # to know that its request was outside the authorized checkout.
        return {"path": "[outside-checkout-path]", "readable": False,
                "why": "requested file is outside the repository checkout"}
    relative = target.relative_to(root).as_posix()
    if not target.is_file():
        return {"path": relative, "exists": False}
    parts = [part.casefold() for part in Path(relative).parts]
    if (target.suffix.casefold() in _PRIVATE_SUFFIXES or
            any(part in _PRIVATE_PATH_PARTS or part == ".env" or part.startswith(".env.")
                for part in parts) or
            target.name.casefold() in {"id_rsa", "id_ed25519", "authorized_keys"} or
            target.suffix.casefold() in {".key", ".pem", ".p12", ".pfx"}):
        return {"path": "[private-artifact]", "exists": True, "readable": False,
                "bytes": target.stat().st_size,
                "why": "private data, model, media, or credential content is withheld"}
    if target.suffix.casefold() not in TEXT_SUFFIXES:
        return {"path": relative, "exists": True, "readable": False,
                "bytes": target.stat().st_size,
                "why": "only bounded source/config text and schema-only JSON are readable"}
    try:
        if target.stat().st_size > 2_000_000:
            text = None
        else:
            raw = target.read_bytes()
            # Extension alone is not proof of text: a binary artifact renamed to `.py`
            # must not be decoded with replacement characters and sent to an API.
            text = raw.decode("utf-8") if b"\x00" not in raw else None
    except (OSError, UnicodeError):
        text = None
    if text is None:
        return {"path": relative, "exists": True, "readable": False,
                "why": "file is oversized, binary, or not valid UTF-8 text"}
    if target.suffix.lower() == ".json":
        # A small JSON file may be a task config, a trajectory, a credential-bearing
        # manifest, or a demonstration. The scout only needs its shape to decide what to
        # inspect next; values (and array lengths, which can reveal sample counts) are not
        # sent to the model. Malformed JSON fails closed instead of falling back to raw text.
        try:
            structure = _json_schema(json.loads(text))
        except (ValueError, TypeError):
            return {"path": relative, "exists": True, "readable": False,
                    "bytes": target.stat().st_size,
                    "why": "JSON schema-only preview failed; raw contents withheld"}
        return {"path": relative, "exists": True, "readable": True,
                "bytes": target.stat().st_size, "values_redacted": True,
            "structure": _safe_model_value(structure, root)}
    body = text.splitlines()
    return {"path": relative, "exists": True, "readable": True,
            "bytes": target.stat().st_size, "total_lines": len(body),
            "head": sanitize_model_text("\n".join(body[:lines]), local_roots=(root,))}


def peek_many(paths: Iterable[str], *, repo: Path, lines: int = 60,
              limit: int = 12) -> list[dict[str, Any]]:
    """Read at most `limit` safe files, rooted to the checkout that supplied the survey."""
    return [_peek(path, repo=repo, lines=lines) for path in list(paths)[:limit]]


def _safe_model_value(value: Any, repo: Path) -> Any:
    """Recursively scrub path/credential strings, including strings used as JSON keys."""
    if isinstance(value, dict):
        return {sanitize_model_text(key, local_roots=(repo,)): _safe_model_value(item, repo)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_model_value(item, repo) for item in value]
    if isinstance(value, tuple):
        return [_safe_model_value(item, repo) for item in value]
    if isinstance(value, str):
        return sanitize_model_text(value, local_roots=(repo,))
    return value


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
    # Deeper for data than for the rest: a demonstration set is buried under however many
    # levels its own layout happens to use, and the walk is bounded below by directories
    # rather than by depth.
    datasets = _datasets(roots, max_depth=max(asset_depth, 8), max_candidates=max_assets)
    models = _models(roots, max_depth=max(asset_depth, 6), max_candidates=max_assets)
    fragments = sorted({_fragment(repo, row["path"]) for row in datasets + models} - {None})
    from .workspace_resources import resource_view
    return {
        "repo": str(repo),
        "workspace_resources": resource_view(repo),
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
        "workspace_resources": report.get("workspace_resources", {}),
        "project_manifests": sorted(report["project_manifests"]),
        "manifest_excerpts": {name: text[:1200] for name, text in
                              report["project_manifests"].items()
                              if name in ("setup.py", "pyproject.toml", "requirements.txt",
                                          "environment.yml", "README.md")},
        "tree_extensions": report["tree"]["extensions"],
        "tree_truncated": report["tree"]["truncated"],
        # The file names, which this never passed. The triage step is asked to *name up to
        # twelve repository-relative files* and it was being shown extension counts, the
        # largest files and the directory names -- so it was guessing, and on RoboTwin it
        # guessed directories, a PNG, and four files inside a vendored library. Shallow
        # first, because that is where entry points are.
        "files": (report["tree"].get("seen_files") or [])[:300],
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


def summarise_for_model(report: dict[str, Any]) -> dict[str, Any]:
    """A model-facing survey projection with paths and artifact values withheld.

    The full survey is retained locally for verification. The LLM sees repository-relative
    source names, bounded source excerpts, and structure-only data descriptions; host paths,
    checkpoint filenames, dataset/media payloads, and arbitrary HDF5/JSONL attributes are
    never included.
    """
    repo = Path(report["repo"]).expanduser().resolve()
    local = summarise(report)

    def source_path(value: Any) -> str | None:
        text = str(value or "")
        candidate = Path(text)
        if candidate.is_absolute():
            try:
                relative = candidate.resolve(strict=False).relative_to(repo)
            except (OSError, RuntimeError, ValueError):
                return None
        else:
            relative = candidate
        if (relative.is_absolute() or ".." in relative.parts or
                any(part.casefold() in _PRIVATE_PATH_PARTS for part in relative.parts) or
                relative.suffix.casefold() not in TEXT_SUFFIXES):
            return None
        return relative.as_posix()

    def shape_only(value: Any) -> Any:
        if isinstance(value, dict):
            output: dict[str, Any] = {}
            for key, item in value.items():
                name = str(key)
                lowered = name.casefold()
                if lowered in {"path", "root_attrs", "attrs", "count", "examples",
                               "length", "records"}:
                    if lowered in {"root_attrs", "attrs"} and isinstance(item, dict):
                        output["attribute_names"] = sorted(map(str, item.keys()))
                    elif lowered == "records" and isinstance(item, list):
                        shapes = [_json_schema(row) for row in item[:2]]
                        output["record_shapes"] = _safe_model_value(shapes, repo)
                    continue
                if re.fullmatch(r"(?:demo|episode|trajectory)[_-]?\d+", name, re.I):
                    name = "example_group"
                if name in output:
                    name = f"{name}_group"
                output[name] = shape_only(item)
            return output
        if isinstance(value, list):
            return [shape_only(item) for item in value[:30]]
        if isinstance(value, str):
            return sanitize_model_text(value, local_roots=(repo,))
        return value

    assets = []
    for index, row in enumerate(local["datasets"]):
        raw_path = str(row.get("path") or "")
        candidate = Path(raw_path)
        inside = False
        try:
            candidate.resolve(strict=False).relative_to(repo)
            inside = True
        except (OSError, RuntimeError, ValueError):
            pass
        assets.append({"asset": f"dataset_{index + 1}", "suffix": row.get("suffix"),
                       "bytes": row.get("bytes"), "inside_checkout": inside})

    structures = []
    for index, row in enumerate(local["dataset_structure"]):
        structures.append({"asset": f"dataset_{index + 1}",
                           "suffix": row.get("suffix"), "bytes": row.get("bytes"),
                           "structure": shape_only({key: value for key, value in row.items()
                                                     if key not in {"path", "bytes", "suffix"}})})

    structured_roots = []
    for row in local["structured_roots"]:
        path = source_path(row.get("directory"))
        structured_roots.append({"location": "inside_checkout" if path else "outside_or_private",
                                 **({"path": path} if path else {}),
                                 "manifest_types": row.get("manifests", [])})

    safe = {
        "repo": "{repository}",
        "project_manifests": local["project_manifests"],
        "manifest_excerpts": local["manifest_excerpts"],
        "tree_extensions": local["tree_extensions"],
        "tree_truncated": local["tree_truncated"],
        "files": [path for path in (source_path(item) for item in local["files"]) if path],
        "largest_files": [{"path": path, "bytes": row.get("bytes")}
                          for row in local["largest_files"]
                          if (path := source_path(row.get("path")))],
        "signal_totals": local["signal_totals"],
        "signal_sources": [{"path": path, "signals": row.get("signals", [])}
                           for row in local["signal_sources"]
                           if (path := source_path(row.get("path")))],
        "dataset_count": len(report.get("datasets") or []),
        "workspace_resources": local.get("workspace_resources", {}),
        "datasets": assets,
        "model_artifacts": {"inside_checkout": sum(
            1 for row in report.get("model_artifacts") or []
            if str(row.get("path", "")).startswith(str(repo) + os.sep)),
            "outside_checkout": sum(
                1 for row in report.get("model_artifacts") or []
                if not str(row.get("path", "")).startswith(str(repo) + os.sep))},
        "asset_provenance": {
            "inside_repo": (report.get("asset_provenance") or {}).get("inside_repo", 0),
            "outside_repo": (report.get("asset_provenance") or {}).get("outside_repo", 0),
            "repository_mentions_present": bool(
                (report.get("asset_provenance") or {}).get("repository_mentions")),
        },
        "structured_roots": structured_roots,
        "dataset_structure": structures,
        "recorded_provenance": {
            "recorded_path_count": len((report.get("recorded_provenance") or {}).get(
                "recorded_paths") or []),
            "resolved_inside_checkout_count": len((report.get("recorded_provenance") or {}).get(
                "resolve_inside_repo") or []),
        },
        "searched_roots": ["{repository}", "{external_asset_roots_withheld}"],
    }
    return _safe_model_value(safe, repo)


def load_declaration(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None
