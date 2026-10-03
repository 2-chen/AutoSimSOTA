"""What could be changed, thought about once, audited, and then chosen from.

The shape of a proposal decides the shape of the search. This loop's proposal shape was "name
a value from the declared optimisation space", so its search could only ever be a walk through
that space -- and the space is a list of settings read out of a repository. AutoSOTA's report
puts numbers on what that costs: of the improvements its runs found, **33% were hyperparameter
tuning and 51% were changes to the algorithm's code**. A loop that can only do the first is
missing the larger half, and no amount of better reasoning about which value to try next
reaches it.

Three things follow from that, and AutoSOTA has all three. They are one mechanism, so they
live in one module.

**An idea has a granularity.** A value from the declared space is `param`; a change to the
benchmark's own code is `code`; a change to *what the run does* -- which stages, which data,
which order -- is `algo`. The typing is not bookkeeping: it is what makes "this class of move
has been tried enough" a thing the loop can notice, and what makes a leap to a different class
a thing it can decide to take.

**The ideas are produced once and chosen from, not reinvented every round.** AutoSOTA builds a
library of at least ten in its second phase, each with a type, a risk and the reasoning, and
the main loop then *selects*. This loop generated a fresh proposal per round from the evidence:
a plausible idea that lost the round was gone, nothing compared two candidates with each other,
and every round re-derived what the previous round had already worked out.

**When one class of move is all that has been tried, the loop leaves the library.** AutoSOTA's
Leap Path: three consecutive `PARAM` iterations force a structurally different idea rather than
another selection, and the leap gets a honeymoon -- a few iterations to be explored and
debugged before anything is rolled back. What this loop did instead was *stop*: today every one
of RoboTwin's three stages ended on "three obstacles in a row", each with a correct diagnosis
and no move of a different kind available to make.

That is the difference between a brake and a gear change, and it is the reason this module
exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_json, now, redact

#: The three kinds of move, coarse enough that "we have only been doing one of them" is a
#: thing worth saying. AutoSOTA's `τ(h)`, under this system's names.
GRANULARITIES = ("param", "code", "algo")

#: How much a failed idea costs. A low-risk idea is one whose failure is a lost round; a
#: high-risk one can leave the checkout in a state another round has to recover from, which is
#: why risk orders the queue rather than being a label.
RISKS = ("low", "medium", "high")

#: What an idea's status can be. `rejected` is set by the audit and never by a run: an idea
#: that met a red line did not run and did not fail.
NEW, CLEARED, REJECTED, TRIED, WORKED = "new", "cleared", "rejected", "tried", "worked"


@dataclass
class Idea:
    """One thing the run could do, with what it costs and what it rests on."""

    label: str
    granularity: str = "param"
    mechanism: str = ""
    #: What to do, in the shape its granularity takes. `param` carries exact declared axis names
    #: as settings, e.g. `{"n_epochs": 50}` (preserve a prefix only when it is part of the
    #: declared axis name); `code` carries `{"file": ..., "find": ..., "replace": ...}`;
    #: `algo` carries the stages or axes the run should use.
    change: dict[str, Any] = field(default_factory=dict)
    risk: str = "medium"
    #: Which red line this change would cross, or `"none"`. Answered by the idea's author
    #: when it is written, alongside the evidence and the risk, and read by the audit.
    #:
    #: It used to be inferred from the words of the idea's description against a table of
    #: thirty phrases -- which is a model of how a model phrases things, and it was wrong in
    #: both directions. This is the same fact, asked for instead of guessed.
    crosses: str = "none"
    why: str = ""
    evidence: list[str] = field(default_factory=list)
    #: Files this would change. Read by the audit, and by nothing else.
    touches: list[str] = field(default_factory=list)
    status: str = NEW
    outcome: str = ""
    times_tried: int = 0
    parent: str = ""            # the leap that produced it, when one did
    outcome_event_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "granularity": self.granularity,
                "mechanism": self.mechanism, "change": self.change, "risk": self.risk,
                "crosses": self.crosses,
                "why": self.why, "evidence": list(self.evidence),
                "touches": list(self.touches), "status": self.status,
                "outcome": self.outcome, "times_tried": self.times_tried,
                "parent": self.parent,
                "outcome_event_ids": list(self.outcome_event_ids)}


def execution_compatibility(idea: Any, *, stage_parameters: Any, repo: Any = None
                            ) -> tuple[bool, str]:
    """Check a candidate against the selected run's verified stage interface."""
    granularity = str(getattr(idea, "granularity", "") or "")
    change = getattr(idea, "change", {})
    if not isinstance(change, dict):
        return False, "candidate change is not a settings object"
    if granularity == "code":
        patches = change.get("patches") if isinstance(change.get("patches"), list) else [change]
        if not any(isinstance(row, dict) and row.get("find") != row.get("replace")
                   for row in patches):
            return False, "code candidate has no effective source change"
        if repo is not None:
            from pathlib import Path
            root = Path(repo).resolve()
            for patch in patches:
                if not isinstance(patch, dict) or not isinstance(patch.get('find'), str):
                    return False, 'code candidate requires an exact source find string'
                target = root/str(patch.get('file') or '')
                try:
                    if (not target.resolve().is_relative_to(root) or target.is_symlink()
                            or not target.is_file() or target.stat().st_size>4*1024*1024):
                        return False, 'code candidate source file is missing or unsafe'
                    find = patch['find']
                    if not find or target.read_text().count(find)!=1:
                        return False, ('code candidate must cite one exact unique source block; '
                                       'read the file and propose a literal patch before consuming a round')
                except (OSError, UnicodeError):
                    return False, 'code candidate source is unreadable'
        return True, ""
    if granularity not in {"param", "algo"}:
        return False, "candidate has an unsupported granularity"

    if isinstance(stage_parameters, dict):
        rows = stage_parameters.keys()
    elif isinstance(stage_parameters, list):
        rows = [row.get("name") for row in stage_parameters if isinstance(row, dict)]
    else:
        rows = []
    allowed = set()
    for raw in rows:
        name = str(raw or "").strip().lstrip("-").split(".")[-1]
        if name:
            allowed.add(re.sub(r"[-\s]+", "_", name).lower())
    if not allowed:
        return True, "selected stage has no parameter contract; compatibility is unknown"

    names: set[str] = set()

    def collect(value: dict[str, Any]) -> None:
        for raw, item in value.items():
            if isinstance(item, dict):
                collect(item)
                continue
            name = str(raw or "").strip().lstrip("-").split(".")[-1]
            if name:
                names.add(re.sub(r"[-\s]+", "_", name).lower())

    collect(change)
    if not names:
        return False, "settings candidate does not name a parameter"
    unsupported = sorted(names - allowed)
    if unsupported:
        return False, ("settings are not exposed by the selected verified train command: "
                       + ", ".join(unsupported[:8]))
    return True, ""


def declared_space_compatibility(idea: Any, *, space: Any
                                 ) -> tuple[bool, str]:
    """Check a settings idea against the declared axes and their accepted values.

    A trainer flag is not automatically a research axis: the repository declaration is the
    contract for what may vary. Conversely, an axis that the selected trainer does not expose
    cannot be executed. The controller surfaces a settings idea only when both contracts
    accept it; the research kernel repeats this check before consuming a round.
    """
    granularity = str(getattr(idea, "granularity", "") or "")
    if granularity == "code":
        return True, ""
    change = getattr(idea, "change", {})
    if not isinstance(change, dict):
        return False, "candidate change is not a settings object"
    try:
        from .decision import EVIDENCE_FIELD, validate_proposal
        sections = set(space.sections())
        if any(key in sections for key in change):
            named = {name: dict(change.get(name) or {}) for name in sections}
            remainder = {key: value for key, value in change.items()
                         if key not in sections}
        else:
            named = {name: {} for name in sections}
            named["training"] = dict(change)
            remainder = {}
        envelope: dict[str, Any] = {
            "decision": "experiment", "proposal_id": str(getattr(idea, "label", "idea"))[:80],
            "hypothesis": str(getattr(idea, "mechanism", "") or "idea"),
            "expected_validation": str(getattr(idea, "why", "") or "validate the change"),
            EVIDENCE_FIELD: "declared-space-check",
        }
        for section, axes in space.sections().items():
            supplied = dict(named.get(section) or {})
            supplied.update({key: value for key, value in remainder.items()
                             if any(axis.name == key for axis in axes)})
            for axis in axes:
                if axis.group or axis.name in supplied or axis.optional:
                    continue
                if axis.default is not None:
                    supplied[axis.name] = axis.default
                elif axis.accepts(None):
                    supplied[axis.name] = None
            envelope[section] = supplied
        validate_proposal(envelope, space=space, evidence_id="declared-space-check")
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"change does not fit the declared optimization space: {exc}"
    return True, ""


class IdeaLibrary:
    """The ideas for one run, written down where a reader and the next round both find them.

    Kept as a document rather than in memory because the run is resumed, and because the
    library is what a person reads to see what the run considered -- AutoSOTA keeps it as
    `idea_library.md` beside the scores for the same reason. An idea that was never selected is
    as informative as one that worked, and it exists nowhere else.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.ideas: list[Idea] = []
        self.leaps: list[dict[str, Any]] = []
        if self.path.is_file():
            try:
                document = json.loads(self.path.read_text(encoding="utf-8"))
                self.ideas = [Idea(**row) for row in document.get("ideas") or []]
                self.leaps = list(document.get("leaps") or [])
            except (OSError, ValueError, TypeError):
                self.ideas, self.leaps = [], []

    # -- reading ------------------------------------------------------------------------

    def get(self, label: str) -> Idea | None:
        return next((one for one in self.ideas if one.label == label), None)

    def usable(self) -> list[Idea]:
        """Ideas that may be selected: audited clear, and not already known to have failed."""
        return [one for one in self.ideas if one.status == CLEARED]

    def tried(self) -> list[Idea]:
        return [one for one in self.ideas if one.times_tried]

    def summary(self) -> dict[str, Any]:
        by: dict[str, int] = {}
        for one in self.ideas:
            by[one.status] = by.get(one.status, 0) + 1
        kinds: dict[str, int] = {}
        for one in self.ideas:
            kinds[one.granularity] = kinds.get(one.granularity, 0) + 1
        return {"ideas": len(self.ideas), "by_status": by, "by_granularity": kinds,
                "leaps": len(self.leaps)}

    # -- writing ------------------------------------------------------------------------

    def add(self, idea: Idea, *, parent: str = "") -> Idea:
        """Add one, or return the existing one with the same label.

        A label is an identity, not a name: two ideas with one label are one idea, and letting
        the second replace the first would lose what the first was for.
        """
        existing = self.get(idea.label)
        if existing is not None:
            return existing
        idea.parent = parent or idea.parent
        self.ideas.append(idea)
        self._write()
        return idea

    def audit(self, lines: Any) -> list[Any]:
        """Mark every unaudited idea against the red lines, and return the verdicts.

        Before anything is selected, which is the ordering AutoSOTA enforces: an idea that
        meets a red line is `REJECTED` in the library and is not a candidate. Refusing it at
        the point of use instead -- which is what `assert_frozen` did -- spends the round and
        reports a failed run rather than a refused idea.
        """
        from . import redlines
        verdicts = []
        for idea in self.ideas:
            if idea.status != NEW:
                continue
            found = redlines.audit(idea, lines)
            idea.status = CLEARED if found.cleared else REJECTED
            idea.outcome = found.because if not found.cleared else ""
            verdicts.append(found)
        if verdicts:
            self._write()
        return verdicts

    def note(self, label: str, *, worked: bool, outcome: str = "",
             event_id: str = "") -> None:
        """What came of choosing this idea. A library without outcomes is a wish list."""
        idea = self.get(label)
        if idea is None:
            return
        if event_id and event_id in idea.outcome_event_ids:
            return
        idea.times_tried += 1
        idea.status = WORKED if worked else TRIED
        idea.outcome = redact(str(outcome))[:600]
        if event_id:
            idea.outcome_event_ids.append(event_id)
        self._write()

    def record_leap(self, *, because: str, tried: list[str], produced: str) -> None:
        self.leaps.append({"at": now(), "because": because, "tried": tried,
                           "produced": produced})
        self._write()

    def _write(self) -> None:
        atomic_json(self.path, {"schema_version": 1, "at": now(),
                                "ideas": [one.as_dict() for one in self.ideas],
                                "leaps": self.leaps})


# -- choosing ------------------------------------------------------------------------------

#: How many consecutive selections of one granularity before the library stops being the
#: place to look. AutoSOTA's number, and its reason: three of a kind is where a pattern is
#: visible, and a fourth of the same kind is the round that was already tried.
SAME_KIND_BEFORE_LEAP = 3

_RISK_ORDER = {one: index for index, one in enumerate(RISKS)}


def select(library: IdeaLibrary, *, history: Iterable[str] = (),
           of_kind: Iterable[str] | None = None) -> Idea | None:
    """The next idea: the least risky that has not been tried, of a kind not over-used.

    Deterministic, and deliberately unexciting. A selection that reasoning could improve is a
    selection that would need reasoning about the library rather than about the benchmark, and
    the reasoning is better spent on the ideas themselves -- which is why they are produced in
    a batch and chosen from by a rule.

    `of_kind` narrows it to kinds that could move the thing the run is stuck on. A run whose
    evaluation produces no number is stuck on something a *value* cannot fix: `train.n_epochs`
    is a number the trainer reads once it starts, and a trainer that never starts reads
    nothing. What can fix it is a change to a file or to what the run does.
    """
    usable = library.usable()
    if of_kind is not None:
        allowed = {one for one in of_kind}
        usable = [one for one in usable if one.granularity in allowed]
    if not usable:
        return None
    recent = list(history)[-SAME_KIND_BEFORE_LEAP:]
    overused = (len(recent) == SAME_KIND_BEFORE_LEAP and len(set(recent)) == 1)
    if overused:
        # Prefer a kind that has not been tried. If there is none the leap is what produces
        # one; selecting another of the same kind here is the round AutoSOTA's Leap Path
        # exists to prevent.
        other = [one for one in usable if one.granularity != recent[0]]
        if other:
            usable = other
    untried = [one for one in usable if not one.times_tried]
    pool = untried or usable
    return sorted(pool, key=lambda one: (_RISK_ORDER.get(one.risk, 9),
                                         one.granularity != "param", one.label))[0]


def needs_a_leap(history: Iterable[str]) -> bool:
    """Whether the recent action types are repetitive (not sufficient by itself to leap)."""
    recent = list(history)[-SAME_KIND_BEFORE_LEAP:]
    return len(recent) == SAME_KIND_BEFORE_LEAP and len(set(recent)) == 1


def evidence_stalled(evidence: dict[str, Any], *, attempts: int = 3) -> bool:
    """Did repeated rounds leave both the measurement and failure diagnosis unchanged?"""
    rows = [row for row in evidence.get("round_history") or []
            if isinstance(row, dict) and int(row.get("round") or 0) > 0]
    if len(rows) < attempts:
        return False
    recent = rows[-attempts:]

    def signature(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("status"), row.get("measured"),
                row.get("metric_utility", row.get("metric_value", row.get("success_rate"))),
                row.get("why_not"))

    return len({signature(row) for row in recent}) == 1


# -- the batch that fills the library ------------------------------------------------------

LIBRARY_SYSTEM = """You are deciding what a research run could try, before it tries any of it.

You are given a benchmark's declared optimisation space, what its stages are, what is known \
about how it runs, and a library of methods. Produce a set of candidate changes, not one \
change -- the run will choose from them and come back to them, and an idea that loses one \
round is worth having on the list for the next.

Every candidate carries two things that matter more than its description:

`granularity` -- the KIND of move it is:
  * `param` — a value from the declared space. An epoch count, a sampling mass, a policy family.
  * `code`  — a change to the benchmark's own source. A path computed one way and read \
another, a loader that opens a name the data does not use, a guard whose condition is subtly \
wrong. This is where most real improvements are, and it is not reachable from the declared \
space.
  * `algo`  — a change to what the run does rather than to a value or a file: which stages it \
runs, what data it trains on, in what order.

`selected_stage_parameters` lists settings exposed by the verified execution path. A parameter \
candidate must change one or more of those names. Do not borrow axes from an unselected trainer, \
dataset path, or stage merely to fill the list. Suggest only interventions supported by the \
selected workflow and its evidence; copy axis names verbatim from `declared_space` and never \
synthesize a `train.` or `training.` prefix from the section name; fewer sound ideas are better \
than a quota of unusable ones. \
Code candidates must be non-no-op and causally relevant to an active stage.

The user prioritizes data acquisition when a native autonomous producer is feasible, allowed \
by the frozen data protocol and connected to the actual training loader. Include an early \
bounded data candidate when the declared collection axes and verified path support it; cite \
the producer/consumer evidence and explain the expected coverage or quality improvement. \
Keep the original-data baseline and account for collection plus retraining cost. If no such \
candidate is supported, state the missing prerequisite instead of inventing an axis or \
equating collector success with a policy-score gain.

`stages` says which stages this benchmark has, and `declared_space` says what its settings are \
called. Both are facts about the thing you are looking at, not conventions: a stage marked \
`"available": false` has no command to run, and an axis belongs to one section and no other.

`risk` -- `low`, `medium` or `high`. Low is a lost round if it fails. High can leave the \
checkout in a state later rounds have to recover from. Risk orders the queue; it is not a \
reason to leave something out.

For a `code` candidate, `change` is {"file": "<repository-relative path>", "find": "<the \
exact text to replace>", "replace": "<what to put there>"}; a multi-file change uses \
{"patches": [<one such object per file>]}. At most eight files are allowed, and all changes \
must pass together or none are applied. Each `file` must be a path that is in \
`files_in_the_checkout` -- the checkout's own files are listed there. `find` must occur \
exactly once in that file, and the file must still parse afterwards. You have not been shown \
the file's contents, so a `find` you cannot quote exactly will be refused -- it is then \
corrected against the file itself, so name the file and say what the change is for. For a \
`param` candidate, `change` is the settings as a map. For an `algo` candidate, `change` is a \
map from section name to the axes to run under, e.g. {"<section>": {"<axis>": <value>}} -- \
where the sections and the axes are the ones `declared_space` has, which are not always the \
same two a benchmark like this usually has.

**Every candidate says which red line it would cross.** The lines are listed in \
`what_may_not_change`, and they are what must not change whatever else does. `crosses` is the \
id of the one this change would cross, or `"none"` for a change that leaves all of them alone.

Answer it against **what the change actually does**, not against what the idea is for. A \
change that leaves the metric alone says `"none"` even if it is about measurement; a change \
that makes an evaluation see ten episodes instead of a hundred crosses R1 however reasonable \
its purpose. The audit reads this answer and refuses the idea on it, so a wrong `"none"` is \
not a shortcut -- it is a false statement about the run, in a record a reader has to be able \
to trust.

`touches` means files the action will actually write, not source files cited as evidence. \
For a `param` or `algo` value change that only passes settings, use an empty list. A `code` \
change lists every file it will patch.

Return one JSON object: {"ideas": [{"label": "<short and unique>", \
"granularity": "<param|code|algo>", "mechanism": "<what it changes, one line>", \
"change": {...}, "risk": "<low|medium|high>", "crosses": "<none|R1|R2|R3|R4|R5|R6>", \
"why": "<the reasoning>", "evidence": ["<what in the material supports it>"], \
"touches": ["<files it would change>"]}], \
"reasoning": "<how you chose these>"}"""


def build(client: Any, material: dict[str, Any], *, attempts: int = 2,
          on_event: Any = None) -> list[Idea]:
    """The batch, once, from everything known before any round has run.

    Never raises and returns [] when it gets nothing: a run with no ideas is a run that reports
    why, and the alternative -- losing the run to a parse failure in the step that was supposed
    to fill the library -- is the failure mode this whole module's neighbours keep finding.
    """
    payload = json.dumps(material, ensure_ascii=False, default=str)[:60000]
    for attempt in range(attempts):
        try:
            content, _ = client.chat_with_metadata(
                LIBRARY_SYSTEM, payload, max_tokens=6000, timeout=300, thinking="disabled")
        except Exception:                                            # noqa: BLE001
            return []
        try:
            from .execution_derive import _object
            value = _object(content)
            rows = value.get("ideas")
            if not isinstance(rows, list) or not rows:
                raise ValueError("ideas must be a non-empty list")
            out: list[Idea] = []
            for row in rows:
                if not isinstance(row, dict) or not str(row.get("label") or "").strip():
                    continue
                granularity = str(row.get("granularity") or "param").lower()
                out.append(Idea(
                    label=str(row["label"]).strip()[:80],
                    granularity=granularity if granularity in GRANULARITIES else "param",
                    mechanism=str(row.get("mechanism") or "")[:300],
                    change=row.get("change") if isinstance(row.get("change"), dict) else {},
                    risk=str(row.get("risk") or "medium").lower()
                    if str(row.get("risk") or "").lower() in RISKS else "medium",
                    crosses=str(row.get("crosses") or "").strip(),
                    why=str(row.get("why") or "")[:2000],
                    evidence=[str(one)[:300] for one in (row.get("evidence") or [])][:6],
                    # Param/algo actions pass validated settings and never patch a file.
                    # The model may cite the file defining an axis as `touches`; that is
                    # evidence, not a write target. Explicit `change.file`/patches still
                    # go through the red-line audit.
                    touches=([str(one) for one in (row.get("touches") or [])][:12]
                             if granularity == "code" else [])))
            if out:
                return out
            raise ValueError("no idea had a label")
        except Exception as exc:                                     # noqa: BLE001
            if on_event:
                on_event("library", [{"status": "the library draft was refused",
                                      "error": redact(f"{type(exc).__name__}: {exc}")[:400]}])
            payload = payload + "\n\n### YOUR PREVIOUS ANSWER WAS REFUSED\n" + json.dumps(
                {"error": redact(str(exc))[:300],
                 "instruction": "Return one JSON object with a non-empty `ideas` list."},
                ensure_ascii=False)
    return []


LEAP_SYSTEM = """A research run has made the same KIND of change several times and the stage \
still does not run. Another change of that kind is the round that was already tried.

You are given the run's situation, the ideas it has already tried and what happened, and its \
library. **Produce one idea of a kind it has not been making** -- synthetically, from what the \
material shows, and not by selecting from the library.

Say why this kind of move is the one the evidence asks for, and what it would change.

`crosses` is the id of the red line this change would cross, or `"none"`. The lines are in \
`what_may_not_change`, and the audit refuses an idea on this answer -- so answer it against \
what the change does, not against what it is for.

`touches` lists files this action will write. A parameter or algorithm settings change does \
not write the source file that defines the setting; use an empty list unless a file is patched.

Return one JSON object: {"label": "<short and unique>", "granularity": "<param|code|algo>", \
"mechanism": "<what it changes>", "change": {...}, "risk": "<low|medium|high>", \
"crosses": "<none|R1|R2|R3|R4|R5|R6>", \
"why": "<why this kind, from the evidence>", "evidence": ["<what supports it>"], \
"touches": ["<files it would change>"]}"""


def leap(client: Any, material: dict[str, Any], *, avoid: str = "",
         on_event: Any = None) -> Idea | None:
    """One idea of a different kind, made rather than chosen.

    AutoSOTA's Leap Path, and the half of it that matters is the honeymoon: a leap is a
    departure from everything the run has learned about this benchmark, so it gets a few
    iterations to be explored and debugged before anything rolls it back. A run that judged a
    leap by its first measurement would judge it the way it judged the moves that failed.
    """
    payload = json.dumps({**material, "the_kind_that_has_not_worked": avoid},
                         ensure_ascii=False, default=str)[:60000]
    for _ in range(2):
        try:
            content, _ = client.chat_with_metadata(
                LEAP_SYSTEM, payload, max_tokens=3000, timeout=300, thinking="disabled")
            from .execution_derive import _object
            row = _object(content)
            label = str(row.get("label") or "").strip()
            if not label:
                raise ValueError("a leap needs a label")
            granularity = str(row.get("granularity") or "").lower()
            if granularity not in GRANULARITIES or granularity == avoid:
                raise ValueError(f"a leap has to be a different kind of move from {avoid!r}")
            return Idea(label=label[:80], granularity=granularity,
                        mechanism=str(row.get("mechanism") or "")[:300],
                        change=row.get("change") if isinstance(row.get("change"), dict) else {},
                        risk=str(row.get("risk") or "medium").lower()
                        if str(row.get("risk") or "").lower() in RISKS else "medium",
                        crosses=str(row.get("crosses") or "").strip(),
                        why=str(row.get("why") or "")[:2000],
                        evidence=[str(one)[:300] for one in (row.get("evidence") or [])][:6],
                        touches=([str(one) for one in (row.get("touches") or [])][:12]
                                 if granularity == "code" else []))
        except Exception as exc:                                     # noqa: BLE001
            if on_event:
                on_event("leap", [{"status": "the leap was refused",
                                   "error": redact(f"{type(exc).__name__}: {exc}")[:400]}])
            payload = payload + "\n\n### YOUR PREVIOUS ANSWER WAS REFUSED\n" + json.dumps(
                {"error": redact(str(exc))[:300]}, ensure_ascii=False)
    return None


# -- repairing a change that does not fit the file it names ---------------------------------

#: How much of a file a repair is shown. A patch is a correction to a few lines, and the model
#: has to be able to find those lines exactly -- but showing a whole checkout's worth of source
#: to fix one `find` is a prompt that costs more than the round it is repairing, and the
#: middle of the file is what gets dropped first.
FILE_SHOWN = 24000

REPAIR_SYSTEM = """A change was refused before anything ran. Nothing was applied and nothing \
was written.

You are given the change, why it was refused, and whatever you need to correct it: the file it \
names, when it names one, and the sections and axes this benchmark's settings live in, always.

There are two ways a change is refused, and the message says which.

**It does not fit this benchmark's settings.** A `param` or `algo` change names axes, and the \
axes belong to sections. The refusal names the section and lists the axes it does have. Read \
the list you were given rather than the names such a benchmark usually uses: an axis belongs \
to exactly one section, and the sections are not always the same two.

**It does not fit the file it names.** `find` must be text that is in the file **exactly as \
the file spells it** -- copy it from what you were shown, including indentation -- and it must \
occur exactly once.

Correct it. Do not change what the idea is for: if the file does not contain what the idea \
assumed, or the benchmark has no axis for it, that is a fact about the idea and `give_up` is \
the honest answer, not a reason to return something you have not checked.

Return one JSON object: {"change": <the corrected change, in the same shape>, \
"why": "<what you corrected>", "give_up": "<when it cannot be done>"}"""


def _excerpt(path: Path, *, limit: int = FILE_SHOWN) -> str:
    """The file, or the parts of it a `find` could be made from.

    Whole when it fits. When it does not, the head and the tail rather than the head alone:
    a loader's bug is as often in what it does at the end as in what it says at the top, and a
    model shown only the first N characters answers confidently about a region it has not
    seen and cannot tell that it has not.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n\n... [{len(text) - limit} characters not shown] ...\n\n" \
                         + text[-half:]


def repair(client: Any, idea: "Idea", problems: list[str], *, repo: Path,
           space: Any = None, attempts: int = 2, on_event: Any = None) -> "Idea | None":
    """The same idea with a change that fits, or nothing.

    One channel for both ways a change is refused, because the model needs the same two
    things either way: what was wrong with it, and the facts it lacked when it wrote it. For
    a `code` idea those facts are the file it named -- it wrote an exact `find` without ever
    having seen the file. For a `param` or `algo` idea they are the sections and axes the
    benchmark actually declares.

    This is the framework's answer to a model that spells something the system cannot read,
    and it replaces the alternative, which is a growing set of rules that reshape what the
    model wrote. Rules of that kind cover the spellings their author saw; this covers the ones
    nobody has seen yet, because the model corrects itself against the facts rather than
    against a list of accepted forms.

    Returns `None` when the repair was refused or the model gave up. Never raises: a change
    that cannot be repaired is a round that does not happen, not the end of the run.
    """
    change_for_files = idea.change or {}
    files = [str(change_for_files.get("file") or "")]
    files += [str(row.get("file") or "") for row in
              (change_for_files.get("patches") or []) if isinstance(row, dict)]
    root = Path(repo).resolve()
    payload_for: dict[str, Any] = {
        "idea": {"label": idea.label, "granularity": idea.granularity,
                 "mechanism": idea.mechanism, "why": idea.why, "change": idea.change},
        "refused_because": problems,
    }
    shown_files = []
    for file in dict.fromkeys(files):
        if not file:
            continue
        target = (root / file).resolve()
        if Path(file).is_absolute() or not target.is_relative_to(root) or not target.is_file():
            continue
        shown = _excerpt(target)
        shown_files.append({"path": file, "content": shown,
                            "shown_in_full": len(shown) < FILE_SHOWN})
        if len(shown_files) >= 4:
            break
    if shown_files:
        payload_for["the_files"] = shown_files
        payload_for["the_file"] = shown_files[0]
    if space is not None and hasattr(space, "sections"):
        payload_for["the_sections_this_benchmark_declares"] = {
            name: [{"axis": axis.name, "kind": axis.kind, "default": axis.default}
                   for axis in axes]
            for name, axes in space.sections().items()}
    payload = json.dumps(payload_for, ensure_ascii=False, default=str)[:60000]
    for _ in range(attempts):
        try:
            content, _ = client.chat_with_metadata(REPAIR_SYSTEM, payload, max_tokens=4000,
                                                   timeout=300, thinking="disabled")
            from .execution_derive import _object
            row = _object(content)
            if str(row.get("give_up") or "").strip():
                # Kept on the idea as well as logged, because it is the finding. "This cannot
                # be made to fit, and here is why" is what a reader wants; a caller that
                # recorded only `None` would write down that the change failed and nothing
                # about what the model learned trying.
                idea.outcome = redact(f"gave up: {str(row['give_up']).strip()}")[:600]
                if on_event:
                    on_event("idea_repair", [{"idea": idea.label, "status": "abandoned",
                                              "why": redact(str(row["give_up"]))[:400]}])
                return None
            change = row.get("change")
            if not isinstance(change, dict) or not change:
                raise ValueError("a repair has to return a change")
            idea.change = change
            if str(row.get("why") or "").strip():
                idea.outcome = redact(str(row["why"]))[:600]
            if on_event:
                on_event("idea_repair", [{"idea": idea.label, "status": "repaired",
                                          "why": redact(str(row.get("why") or ""))[:300]}])
            return idea
        except Exception as exc:                                     # noqa: BLE001
            if on_event:
                on_event("idea_repair", [{"idea": idea.label, "status": "refused",
                                          "error": redact(f"{type(exc).__name__}: "
                                                          f"{exc}")[:300]}])
            payload = payload + "\n\n### YOUR PREVIOUS ANSWER WAS REFUSED\n" + json.dumps(
                {"error": redact(str(exc))[:300],
                 "instruction": "Return one JSON object with a `change`, or a `give_up`."},
                ensure_ascii=False)
    return None


def render(library: IdeaLibrary) -> str:
    """The library as a reader sees it, for the run's document.

    Every idea and its outcome, including the ones never chosen: AutoSOTA keeps this as
    `idea_library.md` beside the scores, and the reason is that an idea that was considered
    and not run is a finding about the search -- it is the difference between a run that tried
    everything and a run that ran out.
    """
    rows = library.ideas
    if not rows:
        return "_这次运行没有建想法库。_"
    out = ["| 想法 | 类型 | 风险 | 机制 | 结果 |", "| --- | --- | --- | --- | --- |"]
    for one in rows:
        outcome = one.outcome if one.times_tried else f"（{one.status}，未采用）"
        out.append(f"| {one.label} | {one.granularity} | {one.risk} | "
                   f"{one.mechanism[:110]} | {outcome[:110]} |")
    summary = library.summary()
    kinds = "、".join(f"{k} {v}" for k, v in sorted(summary["by_granularity"].items()))
    out += ["", f"共 {summary['ideas']} 条（{kinds}）；"
                f"其中 {summary['by_status'].get(WORKED, 0)} 条奏效。"]
    if library.leaps:
        out.append(f"跳变 {len(library.leaps)} 次 —— 同一类动作连续 {SAME_KIND_BEFORE_LEAP} "
                   f"轮没有效果时，换一类而不是停车。")
    return "\n".join(out)
