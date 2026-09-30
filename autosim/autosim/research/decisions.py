"""What the run decided, why, and what came of it.

A run leaves a great deal behind: stages that ran, numbers that came out, artifacts that
appeared, failures with their tracebacks. What it did not leave was the *reasoning*, and the
reasoning is the thing that exists nowhere else. It is written once, into a model's response,
and if it is not kept at that moment it is gone -- while every number it produced stays.

There is a precedent worth naming, because it is the same failure in a different system. An
agent framework wrote down a `DecisionLog` schema, connected it to nothing, and left its tracer
a no-op; the result was that "why did the system do X" could not be answered from any record
it kept, however complete the rest of them were. This module exists so that question has an
answer here.

**A decision without an outcome is a diary.** The reason a decision record is worth more than a
transcript is that it can be read back against what followed: this was chosen, and this is what
happened. So every decision is opened when it is made and closed when its consequence is
known, and the closing is a separate act -- a decision whose outcome is never filled in is
reported as such rather than read as one that had no consequences.

Two kinds of decisions are recorded, and they are not the same kind of claim. `by="model"` is a
judgement, with the reasoning quoted in the model's own words. `by="tool"` is a rule the system
applied -- a device chosen from a resource table, a failure classified. `by="human"` is
anything a person supplied. Keeping them apart is what lets a reader ask "which of these were
reasoned about" and get an answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_json, now, object_digest


#: Who made the decision. Not decoration: a model's judgement and a rule's application are
#: different kinds of claim and a reader weighing them needs to know which they have.
MADE_BY = ("model", "tool", "human")

#: Nothing is recorded for a decision taken by a caller that never closes it -- the record says
#: it is open, which is a different statement from "it had no consequences".
OPEN = "open"


@dataclass
class Decision:
    """One thing the run decided, and what came of it."""

    activity: str                    # what was decided, in the imperative
    why: str                         # the reasoning, in whoever's words
    by: str = "tool"                 # "model" | "tool" | "human"
    agent: str = ""                  # the model, tool or person
    used: list[str] = field(default_factory=list)      # the evidence it was decided on
    produced: list[str] = field(default_factory=list)  # what it made
    outcome: Any = None              # filled in when the consequence is known
    round: int | None = None
    id: str = ""
    state_revision: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "at": now(), "kind": "decision", "round": self.round,
                "activity": self.activity, "by": self.by, "agent": self.agent,
                "why": self.why, "used": list(self.used), "produced": list(self.produced),
                "outcome": self.outcome, "state_revision": self.state_revision}


@dataclass
class ResearchDecision(Decision):
    """The outer controller's version-bound action proposal and its evidence contract.

    Empty or absent model fields remain visible in `contract_missing`; callers must not
    silently turn an incomplete proposal into verified evidence. `resource_limits` records
    the model's proposal separately from limits enforced by the run kernel.
    """

    action: str = ""
    question: str = ""
    hypothesis: str = ""
    input_evidence: list[str] = field(default_factory=list)
    resource_limits: dict[str, Any] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    stop_condition: str = ""
    contract_missing: list[str] = field(default_factory=list)
    method_selection: list[dict[str, Any]] = field(default_factory=list)
    method_review_missing: list[str] = field(default_factory=list)
    skill_selection_issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        row = super().as_dict()
        row["research_decision"] = {
            "action": self.action,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "input_evidence": list(self.input_evidence),
            "resource_limits": dict(self.resource_limits),
            "expected_outputs": list(self.expected_outputs),
            "postconditions": list(self.postconditions),
            "stop_condition": self.stop_condition,
            "contract_missing": list(self.contract_missing),
            # Skill selection is a model judgment with a source-linked candidate body; it is
            # not an execution fact. Keep the verdict, evidence refs, version and body hash
            # together so a reviewer can distinguish "considered" from "verified".
            "method_selection": [dict(item) for item in self.method_selection
                                 if isinstance(item, dict)],
            "method_review_missing": list(self.method_review_missing),
            "skill_selection_issues": list(self.skill_selection_issues),
        }
        return row


class Decisions:
    """The run's decision record, written as it happens.

    Kept as a list of rows beside `events.json` rather than in a log file of its own: the
    system already rewrites a small JSON document on every append elsewhere, and a second
    convention for the same job is one more thing to know.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "decisions.json"
        self.rows: list[dict[str, Any]] = []
        if self.path.is_file():
            try:
                document = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError(f"decision log cannot be read: {type(exc).__name__}") from exc
            if (not isinstance(document, dict) or document.get("schema_version") != 1 or
                    not isinstance(document.get("rows"), list) or
                    not all(isinstance(row, dict) for row in document["rows"])):
                raise ValueError("decision log has an unsupported or invalid schema")
            self.rows = list(document["rows"])

    # -- writing ------------------------------------------------------------------------

    def record(self, decision: Decision) -> str:
        """Write a decision down and return its id, which is how its outcome finds it."""
        if decision.by not in MADE_BY:
            raise ValueError(f"a decision is made by one of {MADE_BY}, not {decision.by!r}")
        if not decision.id:
            decision.id = object_digest({"activity": decision.activity, "why": decision.why,
                                         "round": decision.round,
                                         "n": len(self.rows)})[:16]
        row = decision.as_dict()
        if row["outcome"] is None:
            row["outcome"] = {"state": OPEN}
        self.rows.append(row)
        self._write()
        return decision.id

    def resolve(self, decision_id: str, outcome: Any) -> None:
        """Say what came of a decision. The half that makes the record worth keeping.

        A decision whose outcome is never filled in stays open and is reported as open. That
        is a finding about the run -- something was decided and nobody looked at what
        happened -- rather than a decision that turned out to have no consequences.
        """
        for row in self.rows:
            if row.get("id") == decision_id:
                row["outcome"] = {"state": "known", "at": now(), "what": outcome}
                self._write()
                return
        # A decision nobody recorded is worth saying out loud rather than dropping: it means
        # a caller resolved something it never opened.
        self.rows.append({"id": decision_id, "at": now(), "kind": "outcome_only",
                          "activity": "", "why": "", "outcome":
                          {"state": "known", "at": now(), "what": outcome}})
        self._write()

    def _write(self) -> None:
        atomic_json(self.path, {"schema_version": 1, "rows": self.rows})

    # -- reading ------------------------------------------------------------------------

    def open_ids(self) -> list[str]:
        return [row["id"] for row in self.rows
                if (row.get("outcome") or {}).get("state") == "open"]

    def summary(self) -> dict[str, Any]:
        """What the record says about itself, for a document that has to be honest about it."""
        by: dict[str, int] = {}
        for row in self.rows:
            if row.get("kind") == "decision":
                by[row["by"]] = by.get(row["by"], 0) + 1
        opened = sum(1 for row in self.rows if row.get("kind") == "decision")
        return {"decisions": opened, "by": by, "unresolved": len(self.open_ids()),
                "outcome_only": sum(1 for row in self.rows if row.get("kind") == "outcome_only")}


def outcome_of(result: dict[str, Any]) -> dict[str, Any]:
    """What a stage run decided, reduced to what a reader of the record needs.

    Not the whole record: the argv and the output are already in `measurements/`, and a
    duplicate of them here would be the second copy that drifts. This is the verdict --
    whether it ran, what it returned, what number came out.
    """
    if not result.get("ran"):
        return {"ran": False, "why": result.get("why") or "the stage did not run"}
    return {"ran": True, "returncode": result.get("returncode"),
            "seconds": result.get("seconds"), "success_rate": result.get("success_rate"),
            "artifact": (result.get("artifact") or {}).get("matched"),
            "readings": result.get("readings") or {}}
