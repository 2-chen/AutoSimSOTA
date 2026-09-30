"""What the system can tell about its own state, from what it has already written down.

Every fault that cost real time here was found the same way, and it was found by a person
reading the output: look at what happened, compare it against what should have happened, and
follow the difference to its source. The system had all the material and none of the reading.
Three specific things it could not see.

**It had no notion of "too long".** A run that finished task 0 in eleven minutes sat for four
hours with nothing more written, and before that one sat for ten. Both were indistinguishable
from slow work, because nothing in the system knew how long that command takes. A timeout is
not a detector: set at seventy-two hours it will eventually fire, having already cost the day.

**It did not read its own records for contradictions.** Four runs printed losses and success
rates for hours and were written down with `readings: {}`. A stage that printed `train loss:
5.36` cannot have produced no named numbers, and that contradiction was in the record at the
moment it was written, looked at by nobody.

**A failure was diagnosed or not depending on which loop it landed in.** The derivation loop
reads failures, reads the source, and revises. The research loop records failures and moves
on. So an error that would have been chased down while generating a command was filed and
forgotten when it arrived two minutes later from a running measurement -- and one of them, a
declared axis whose values the program contradicts, stayed wrong across two full attempts.

None of this needed a better model. It needed the same reading applied to a wider set of
things, and that is what this module is: an observer over the system's own artifacts.

Two rules it holds itself to. **It never raises.** A diagnostic that can fail is worse than no
diagnostic, because it fails exactly when something has already gone wrong -- this has already
happened once in this repository, where a check added to catch a class of error crashed on a
`None` and took down a measurement that had screened two candidates successfully. **It never
acts.** It reports an observation with its evidence and leaves the response to a caller: the
system's habit is to state what it found and let the layer that owns the decision decide.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_json, read_json


#: How far past the expected duration a run may go before it is called overdue. Generous,
#: because the cost model is built from a handful of runs and contention moves it: the point
#: is to catch ten hours where two were expected, not to police twenty percent.
OVERDUE_FACTOR = 3.0

#: Below this, an expectation is too thin to judge by. A stage that has never run has no
#: expected duration, and inventing one would make the first run of anything an anomaly.
MIN_SAMPLES = 2


@dataclass
class Observation:
    """Something the record says that does not fit the rest of the record."""

    kind: str                     # "overdue" | "contradiction" | "undiagnosed" | "stalled"
    subject: str                  # the stage or run it is about
    detail: str                   # one sentence a person can act on
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "subject": self.subject, "detail": self.detail,
                "evidence": self.evidence}


def liveness_path(root: Path) -> Path:
    return Path(root) / "running.json"


def began(root: Path, *, stage: str, argv: list[str], expected_seconds: float | None,
          device: str = "") -> None:
    """Write down that a stage is running, so a reader can tell running from stuck.

    Overwritten per stage and removed when the stage ends. What it is for is the case where
    the stage does *not* end: without this, a hung run and a working run look the same from
    outside, which is how ten hours passed unnoticed.
    """
    try:
        atomic_json(liveness_path(root), {
            "stage": stage, "argv": [str(a) for a in argv], "device": device,
            "started_at": time.time(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "expected_seconds": expected_seconds})
    except Exception:                                            # noqa: BLE001
        pass


def ended(root: Path) -> None:
    try:
        liveness_path(root).unlink(missing_ok=True)
    except Exception:                                            # noqa: BLE001
        pass


def cost_of(history: Iterable[dict[str, Any]], stage: str) -> float | None:
    """How long this stage has taken before, from the runs the system already recorded.

    The median rather than the mean, because one run that hung for ten hours would drag a
    mean far enough to make the next hang look normal -- and the hang is the thing being
    watched for. `None` when there is not enough history to say, which is the honest answer
    for a stage that has run once and is a different answer from "it takes no time".
    """
    durations = sorted(float(row["seconds"]) for row in history
                       if row.get("stage") == stage and row.get("seconds")
                       and row.get("returncode") in (0, None))
    if len(durations) < MIN_SAMPLES:
        return None
    middle = len(durations) // 2
    return (durations[middle] if len(durations) % 2
            else (durations[middle - 1] + durations[middle]) / 2)


def readings_contradiction(record: dict[str, Any]) -> Observation | None:
    """A stage that named no numbers while saying it ran for hours.

    `readings: {}` after a stage that printed `[info] Epoch: 0 | train loss: 5.36` is a
    contradiction between two fields of the same record. It is also the signature of a
    specific fault -- the output was discarded on the way to being written down -- and it was
    sitting in four records, looked at by nobody.
    """
    said = str(record.get("said") or "")
    readings = record.get("readings")
    seconds = float(record.get("seconds") or 0)
    if not isinstance(readings, dict) or readings:
        return None
    if seconds < 300 or not said:
        return None
    return Observation(
        kind="contradiction", subject=str(record.get("stage") or "?"),
        detail=(f"ran {seconds / 60:.0f} minutes and reported for {len(said)} characters, "
                f"and the record holds no number from any of it"),
        evidence={"stage": record.get("stage"), "seconds": seconds,
                  "said_excerpt": said[-300:]})


def diagnosed(record: dict[str, Any], events: Iterable[dict[str, Any]]) -> bool:
    """Did anything look at this failure, or was it only filed?

    The derivation loop reads a failure and revises; the research loop records one and moves
    on. The same error gets two different responses depending on which loop it lands in, and
    a declaration error that survived two whole attempts landed in the one that only files.
    """
    stage = str(record.get("stage") or "")
    if record.get("returncode") in (0, None):
        return True
    return any(str(row.get("stage") or "") == stage
               and str(row.get("event") or "") in ("device_reconsidered", "declaration_"
                                                   "contradicted", "stage_diagnosed")
               for row in events)


def stalled(root: Path, *, now: float | None = None) -> Observation | None:
    """A stage that is running and has run past what the same command has cost before."""
    try:
        live = read_json(liveness_path(root))
    except Exception:                                            # noqa: BLE001
        return None
    if not isinstance(live, dict) or not live.get("started_at"):
        return None
    expected = live.get("expected_seconds")
    if not expected:
        return None
    elapsed = (now or time.time()) - float(live["started_at"])
    if elapsed < float(expected) * OVERDUE_FACTOR:
        return None
    return Observation(
        kind="overdue", subject=str(live.get("stage") or "?"),
        detail=(f"running {elapsed / 3600:.1f}h against an expected "
                f"{float(expected) / 3600:.1f}h from {MIN_SAMPLES}+ earlier runs of this "
                f"command; either it is stuck or the machine is contended"),
        evidence={"argv": live.get("argv"), "device": live.get("device"),
                  "elapsed_seconds": round(elapsed), "expected_seconds": expected})


def observe(root: Path, *, measurements: Iterable[Path] | None = None,
            events: Iterable[dict[str, Any]] = (), history: Iterable[dict[str, Any]] = (),
            now: float | None = None) -> list[Observation]:
    """Everything the record says about itself that does not fit.

    Never raises. Each check runs in isolation and one that fails is reported as a blind spot
    rather than allowed to take the others down -- a diagnostic that dies on bad input dies at
    the worst moment, which is a failure this repository has already had once.
    """
    found: list[Observation] = []
    events = list(events)
    paths = [Path(one) for one in (measurements or [])]
    checks = (
        ("the liveness record", lambda: stalled(root, now=now)),
        ("named numbers", lambda: _first_contradiction(paths)),
        ("failed runs that nothing read", lambda: _undiagnosed(paths, events)),
    )
    for label, check in checks:
        try:
            found.extend(_as_list(check()))
        except Exception as exc:                                 # noqa: BLE001
            found.append(Observation(
                kind="blind", subject=label,
                detail=f"this check could not be run: {type(exc).__name__}: {exc}"))
    return found


def _as_list(value: Any) -> list[Observation]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _first_contradiction(paths: Iterable[Path]) -> Observation | None:
    for path in paths:
        record = read_json(path)
        if isinstance(record, dict):
            found = readings_contradiction(record)
            if found is not None:
                return found
    return None


def _undiagnosed(paths: Iterable[Path],
                 events: Iterable[dict[str, Any]]) -> list[Observation]:
    out: list[Observation] = []
    for path in paths:
        record = read_json(path)
        if not isinstance(record, dict) or diagnosed(record, events):
            continue
        out.append(Observation(
            kind="undiagnosed", subject=str(record.get("stage") or "?"),
            detail=(f"failed with return code {record.get('returncode')} and nothing has "
                    f"read the failure"),
            evidence={"measurement": str(path),
                      "said_excerpt": str(record.get("said") or "")[-300:]}))
    return out


def describe(observations: Iterable[Observation]) -> str:
    """One line, or lines, a person can read without opening the JSON."""
    rows = list(observations)
    if not rows:
        return "nothing in the record contradicts itself"
    return "\n".join(f"[{row.kind}] {row.subject}: {row.detail}" for row in rows)


def record(root: Path, observations: Iterable[Observation]) -> None:
    """Keep the observations, so a reader of the run's tree sees what the run saw."""
    try:
        rows = [row.as_dict() for row in observations]
        atomic_json(Path(root) / "observations.json", {"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                       "rows": rows})
    except Exception:                                            # noqa: BLE001
        pass
