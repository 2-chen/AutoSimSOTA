"""Every run under one root, one line each.

Finding a run meant listing a directory and opening what looked promising, and comparing two
runs meant doing that twice. This is the enumeration that did not exist, written beside the
runs rather than into a database, for the same reason nothing else here is a service: the runs
are the record and this is an index of them.

It covers three shapes, and it found the third by accident -- thirty-eight onboarding runs sat
under `scouting/` with a completely different set of records and were in no index at all, so
most of this machine's work was invisible to anything that reads runs. That is the argument for
an index built from what a run *left* rather than from a list of the shapes someone remembered.

**The row must not invent a number.** A run that measured nothing has no score, and printing
`0.0` for it states that the policy failed every episode -- a different and false claim. The
same rule applies to the reads that failed: a file that could not be parsed is named, and a
file this shape never had is not reported as a fault.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import atomic_json, now
from .run_record import _IS_THE_DOCUMENT, _load, gather

#: What makes a directory under the runs root a run. A run wrote at least one of these; a
#: checkout, a cache or a dataset did not. Listing every directory and guessing would put a
#: repository clone in an index of experiments.
RUN_MARKERS = ("run_state.json", "events.json", "research_report.json", "rounds", "measurements",
               "onboarding.json", "declaration.json")


def find_runs(runs_root: Path) -> list[Path]:
    """Every run under a root, newest first, by what it left rather than by its name."""
    root = Path(runs_root)
    found = [path for path in root.glob("*/*")
             if path.is_dir() and any((path / marker).exists() for marker in RUN_MARKERS)]
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def one_line_about(root: Path) -> dict[str, Any]:
    """The row a cross-run index keeps for one run: enough to decide whether to open it.

    Deliberately not a summary of the run -- a summary is the document's job. This is the
    line you scan to find the run you want, and the two things it must not do are invent a
    number for a run that produced none and silently omit a run whose records are unreadable.
    """
    root = Path(root)
    source = gather(root)
    state = source.get("state") if isinstance(source.get("state"), dict) else {}
    report = source.get("report") if isinstance(source.get("report"), dict) else {}
    rates: list[float] = [row["record"]["success_rate"]
                          for row in (source.get("measurements") or [])
                          if row["record"].get("success_rate") is not None]
    rates += [row["success_rate"] for row in (source.get("rounds") or [])
              if row.get("success_rate") is not None]
    summary = _load(root / "summary.json") or {}
    check = summary.get("checked") or {}
    # Only the faults. `gather` also reports every file this run's shape does not have, which
    # for a scout run is all the event and measurement records -- true, and useless in an index
    # whose job is to point at what is worth opening.
    faults = [one for one in (source.get("unreadable") or []) if "读不出来" in one]
    scout = source.get("scout") or {}
    return {
        "run": root.name,
        "path": str(root),
        "shape": ("scouting" if scout else "derived" if source.get("events") else
                  "research" if state or report else "unknown"),
        "record": str(root / "RUN.md") if (root / "RUN.md").is_file() else "",
        "page": str(root / "RUN.html") if (root / "RUN.html").is_file() else "",
        "status": (state.get("status") or ("scouted" if scout else None)
                   or ("stopped" if report.get("stopped_because") else "?")),
        "stage": state.get("stage") or "",
        "created_at": (state.get("created_at") or report.get("created_at")
                       or (scout.get("onboarding") or {}).get("created_at") or ""),
        "rounds": len(source.get("rounds") or []),
        "measurements": len(source.get("measurements") or []),
        "decisions": (source.get("decisions") or {}).get("rows") and
                     len([row for row in source["decisions"]["rows"]
                          if row.get("kind") == "decision"]) or 0,
        "drafts": len(scout.get("drafts") or []),
        "recordings": sum(bucket.count for bucket in (source.get("survey") or {}).values()
                          if bucket.kind == "recording"),
        # The best number this run produced, and None where it produced none. A run that
        # measured nothing has no score, and a zero would be a different claim.
        "best_success_rate": max(rates) if rates else None,
        "summary_written": bool(summary.get("text")),
        "unsourced_numbers": check.get("untraced") or [],
        "unreadable": faults,
        "error": str(state.get("error") or "")[:400] or None,
    }


def runs_index(runs_root: Path) -> dict[str, Any]:
    """Every run under a root, one line each, newest first.

    The enumeration that did not exist: finding a run meant listing a directory and opening
    what looked promising, and comparing two runs meant doing that twice. It is written
    beside the runs rather than in a database, for the same reason nothing else here is a
    service -- the runs are the record and this is an index of them.
    """
    root = Path(runs_root)
    listed = find_runs(root)
    rows = [one_line_about(path) for path in listed]
    unreadable = [row["run"] for row in rows if row["unreadable"]]
    # Directories that look like a run's position but hold none of a run's records. Named
    # rather than skipped: an attempt that produced nothing is not the same as no attempt, and
    # an index that silently omits it makes the work look tidier than it was.
    empty = sorted(str(path.relative_to(root)) for path in root.glob("*/*")
                   if path.is_dir() and path not in set(listed))
    return {"schema_version": 1, "at": now(), "runs_root": str(root),
            "runs": rows, "with_unreadable_records": unreadable,
            "directories_with_no_records": empty}
