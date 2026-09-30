"""What the loop is doing, whether it is getting anywhere, and what to say when it is not.

A derivation loop's own record is a list of attempts, and each attempt knows only why *it*
failed. Nothing in it knows that the last eleven attempts were all the same kind of change,
or that the stage's state has not moved since round four. So the loop proposes the same
plausible fix in slightly different words until the budget ends -- which is what happened
here: nineteen rounds spent reinstalling a package that was never the problem, and sixty-five
on a checkpoint argument the caller was supplying correctly all along.

AutoSOTA names this role and puts it deliberately outside the loop. Its AgentMonitor infers
the phase from the execution trace, watches for dead ends -- "repeated environment edits with
no successful state transition", repeated installs of the same failing dependency, long
stretches of output without progress -- and emits **"high-level corrective guidance rather
than low-level patch instructions"**, explicitly to preserve the agent's autonomy rather than
direct it. It is also the budget holder: wall clock, round caps, rollback.

That last distinction is the one worth copying exactly, because it is the opposite of what a
rule-based system does. A monitor that says "set PYTHONPATH to X" has taken the decision away
from the thing that can read the repository; one that says "the last six attempts have all
been environment changes and none of them changed the environment" has told the agent
something it could not see about itself, and left the decision where it belongs.

So this module is an *observer*. It reads the trace and returns a picture -- which phase the
search is in, whether the state has moved, what has been tried too often -- and a sentence at
that level. It does not decide, does not act, and does not name a fix.

Two signals, both from AutoSOTA's list and both answerable from what this loop already
records:

* **No state transition.** The stage's own configuration -- where it runs, in what
  environment, how it is called, what values it is given -- is a value, and a round that
  leaves it unchanged has moved nothing however much reasoning it contains.
* **The same kind of change, repeatedly.** Which *field* each attempt touched is in the
  record. Six environment edits in a row that did not fix it is a finding about the approach,
  and it is a different finding from six argument edits that did not fix it.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

#: The fields of a stage's configuration, and what changing each one means. The labels are
#: what a reader would call the attempt, and they are what the phase is named from.
FIELDS = {
    "environment": "environment",
    "working_directory": "where it runs",
    "invocation": "how it is called",
    "parameters": "the values it is given",
    "stdin": "what is fed to it",
    "probes": "what counts as working",
}

#: How many consecutive attempts of one kind, with no state transition, before it is worth
#: saying so. Three is where a pattern is visible and where a fourth attempt of the same kind
#: is unlikely to be the one that works; it is a reading of the trace, not a limit on trying.
REPEATS_BEFORE_SAYING_SO = 3


def _touched(change: dict[str, Any]) -> str:
    """Which field a revision changed, named for a reader."""
    for field in FIELDS:
        if change.get(field) not in (None, "", {}, []):
            return field
    return ""


def observe(rounds: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """What the search has been doing, from the rounds it has recorded.

    Each row is one attempt: `change` is the revision it tried, `state` is the stage's
    configuration afterwards, `failure` is what came back. Nothing here reads the program
    output -- this is about the shape of the search, and the shape is in the loop's own
    record.
    """
    rows = list(rounds)
    if not rows:
        return {"rounds": 0, "phase": "nothing tried yet", "moved": False, "signals": []}

    # `moved` is about the *state*, not about the round: a round that rewrote the invocation
    # into a different spelling of the same command has not moved the state, and counting it
    # as progress is how a loop reports effort where there was none.
    states = [json.dumps(row.get("state") or {}, sort_keys=True, default=str) for row in rows]
    moved = len(set(states)) > 1

    kinds = [_touched(row.get("change") or {}) for row in rows]
    trailing = 1
    for kind in reversed(kinds[:-1]):
        if kind and kind == kinds[-1]:
            trailing += 1
        else:
            break

    counts: dict[str, int] = {}
    for kind in kinds:
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    looked = sum(1 for row in rows if row.get("looked"))

    signals: list[str] = []
    if not moved and len(rows) >= 2:
        signals.append(
            f"{len(rows)} attempts and the stage's own configuration has not changed. Every "
            f"one has been a different way of saying the same command.")
    if trailing >= REPEATS_BEFORE_SAYING_SO and kinds[-1]:
        signals.append(
            f"the last {trailing} attempts changed {FIELDS.get(kinds[-1], kinds[-1])} and "
            f"nothing else, and the stage still does not run. The fault may not be in "
            f"{FIELDS.get(kinds[-1], kinds[-1])}.")
    if len(rows) >= 4 and not looked:
        signals.append(
            f"{len(rows)} attempts and nothing has been looked at. The machine and the "
            f"checkout hold the answer to almost every one of these; ask to see something "
            f"before proposing another change.")

    frequent = max(counts.values()) if counts else 0
    return {
        "rounds": len(rows),
        "phase": (f"{frequent} of {len(rows)} attempts changed "
                  f"{FIELDS.get(max(counts, key=lambda k: counts[k]), 'nothing')}"
                  if counts else "no attempt changed anything"),
        "attempts_that_looked": looked,
        "moved": moved,
        "changed_most": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "signals": signals,
    }


def guidance(picture: dict[str, Any]) -> str:
    """The high-level sentence, for the reviser's payload.

    Deliberately not a fix. What a monitor can see that the agent cannot is the shape of its
    own search -- that it has been circling one field, that nothing has moved, that it has not
    looked at anything -- and naming that is the whole contribution. A monitor that named the
    fix would be the rule-based system this is not.
    """
    signals = picture.get("signals") or []
    if not signals:
        return ""
    return ("How the search is going, from the loop's own record -- not a diagnosis of the "
            "program:\n" + "\n".join(f"* {one}" for one in signals))
