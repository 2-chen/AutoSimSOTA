"""Discover recorded runs without treating datasets and model weights as runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .run_record import gather

_MARKERS = {"events.json", "research_report.json", "preparation_derived.json",
            "declaration.json", "verification.json", "process.json", "run_state.json",
            "final_confirmation_report.json", "RUN.md"}
_BULK = {"env", "datasets", "data", "checkpoints", "blobs", "videos", "media",
         "site-packages", "__pycache__", ".git"}


def _walk(root: Path):
    root = Path(root)
    if not root.is_dir():
        return
    for parent, dirs, files in os.walk(root, followlinks=False):
        here = Path(parent)
        depth = len(here.relative_to(root).parts)
        dirs[:] = [name for name in dirs if name not in _BULK and depth < 6]
        if "RUN.md" in files:
            # A prepared run owns its nested research document. Index the human entry
            # point once, rather than reporting one run as two independent experiments.
            dirs[:] = []
        yield here, files


def find_runs(root: Path) -> list[Path]:
    """Find directories with primary records, newest first."""
    found = [here for here, files in _walk(root) if _MARKERS.intersection(files)]
    return sorted(found, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)


def runs_index(root: Path) -> dict[str, Any]:
    """A compact index; unreadable records and empty directories remain visible."""
    root = Path(root)
    runs = []
    unreadable = []
    with_no_records = []
    for here, files in _walk(root):
        if here == root:
            continue
        relative = str(here.relative_to(root))
        if _MARKERS.intersection(files):
            source = gather(here)
            broken = [entry for entry in source["unreadable"] if "文件在，但读不出来" in entry]
            row = {"directory": relative, "record": str(here / "RUN.md"),
                   "has_report": bool(source.get("report")), "unreadable": broken}
            runs.append(row)
            if broken:
                unreadable.append(relative)
        elif not files and not any(here.iterdir()):
            with_no_records.append(relative)
    return {"schema_version": 1, "runs": runs,
            "with_unreadable_records": unreadable,
            "directories_with_no_records": with_no_records}
