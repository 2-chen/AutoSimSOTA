"""What is the system doing, and does its own record agree with itself?

Usage: `.venv/bin/python tools/observe.py <run-root> [<run-root> ...]`

Reads only what the system has already written: the liveness marker of a stage that is
running, the measurements it has taken, and the events it has recorded. Reports anything that
does not fit -- a stage running long past what the same command has cost before, a record that
says it ran for hours and named no numbers, a failure that nothing has read.

This exists because every fault that cost real time here was found by a person reading output,
and the material was always in the record already. It is the reading, not the material, that
was missing.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autosim"))

from autosim.research import awareness                                  # noqa: E402


def main() -> int:
    roots = [Path(one).expanduser() for one in sys.argv[1:]]
    if not roots:
        print(__doc__.strip())
        return 2
    total = 0
    for root in roots:
        print(f"=== {root}")
        events = []
        events_path = root / "events.json"
        if events_path.is_file():
            try:
                events = json.loads(events_path.read_text(encoding="utf-8")).get("rows") or []
            except (ValueError, OSError):
                pass
        measurements = sorted((root / "measurements").glob("*.json")) \
            if (root / "measurements").is_dir() else []
        # What is running right now, before what it will say about it.
        live = awareness.liveness_path(root)
        if live.is_file():
            try:
                row = json.loads(live.read_text(encoding="utf-8"))
                print(f"  running: {row.get('stage')} since {row.get('started')} "
                      f"on {row.get('device') or '?'} "
                      f"(expected {row.get('expected_seconds') or 'unknown'})")
            except (ValueError, OSError):
                pass
        found = awareness.observe(root, measurements=measurements, events=events)
        print(awareness.describe(found))
        for row in found:
            print(f"    evidence: {json.dumps(row.evidence, ensure_ascii=False)[:300]}")
        total += len(found)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
