"""How far the run got, when the number at the end is not the only thing known.

A research loop on a hard benchmark spends most of its time in states where the final metric
does not exist yet. The environment is not built, or a stage does not run, or the evaluation
produces nothing -- and a loop whose only objective is the success rate has nothing to say
about any of those. It cannot tell "twelve rounds of progress toward a working evaluation"
from "twelve rounds of nothing", so it cannot prefer one, and a reader cannot either: the
record says `success_rate: null` for a run that got a long way and for one that got nowhere.

AutoSOTA's answer is an objective that is not a single number. It decomposes the macro goal
into a tree of binary sub-tasks by breadth-first expansion, each node asking a Pass/Fail
question, with the root's weight **conserved and distributed downward** -- so partial progress
is a real quantity and reaching "the environment is configured" is worth something. Context is
injected by depth: the shallow nodes are answered from the summary, the deep ones from the
repository.

**This system already had the ladder and only ever used its top rung.** `declared → verified`
is exactly this shape: is the entry point there, does the command run, does the artifact
appear, does the number come out. Every one of those was checked -- and then thrown away, and
the round reported only whether a success rate existed. RoboTwin's runs spent hundreds of
rounds in the lower rungs and the record says nothing about them beyond "no runnable command".

Two things this does that the ladder did not.

**It is scored, not just checked.** Each node carries a weight and the score is what passed,
so two runs that both failed to measure can be compared by how far they got -- and `_best` can
advance on that, which is what makes a run that cannot measure yet still able to improve.

**It is decomposed for the benchmark, not only for the system.** The spine is this system's --
environment, stages, measurement, comparison -- because those are questions every benchmark
asks. The leaves under them are this benchmark's, produced by reading it: a benchmark whose
evaluator needs its own seeds, or whose trainer needs a table a previous step writes, says so
in its own terms and the rubric carries it.

What it is not: a replacement for the success rate. The metric is the answer, and a rubric is
how a run knows it is getting closer to being able to give one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .common import now, redact

#: The spine, in the order a run climbs it. Each is a question with a yes or no, and the
#: weights are the whole hundred: reaching the end of the spine is what "this run measured
#: something" means.
#:
#: The order matters more than the numbers. A run cannot have measured a policy before it has
#: an evaluation that returns a number, cannot have that before each stage runs, and cannot
#: have that before the environment holds what the stages need. A checklist where the later
#: items can pass while the earlier ones fail is a checklist that reports nonsense.
SPINE: tuple[tuple[str, str, float], ...] = (
    ("environment", "the environment holds what the benchmark's stages need, and something "
                    "has run in it", 15.0),
    ("stages", "every stage the loop needs has a command that was run and finished",
     25.0),
    ("measurement", "the evaluation produced a number, on the benchmark's own initial states",
     25.0),
    ("comparison", "a candidate was measured against a baseline on the same conditions",
     20.0),
    ("improvement", "the candidate's number is better than the baseline's, by more than the "
                    "spread between repeats", 15.0),
)


@dataclass
class Check:
    """One question, with what it is worth and what it rests on."""

    name: str
    question: str
    weight: float
    passed: bool | None = None          # None is "not reached", which is not the same as False
    because: str = ""
    children: list["Check"] = field(default_factory=list)
    #: Who answered it. Empty when the run's own code read the answer off a record, `"model"`
    #: when the run was asked. The document has to be able to say which, because the two are
    #: different kinds of evidence and a reader cannot tell them apart from the value.
    by: str = ""

    def state(self) -> str:
        if self.passed is None:
            return "not reached"
        return "yes" if self.passed else "no"

    def reached(self) -> bool:
        """Whether anything at or under this question was observed at all.

        The distinction between this and `passed` is what makes a partly-explored rung
        scoreable. A rung whose own question nobody answered, but under which a bench-
        mark-specific question *was* answered, is not "not reached" -- something happened
        there, and the run that did it is further along than one that did nothing.
        """
        return self.passed is not None or any(one.reached() for one in self.children)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "question": self.question, "weight": self.weight,
                "passed": self.passed, "state": self.state(), "because": self.because,
                "by": self.by,
                "children": [one.as_dict() for one in self.children]}


@dataclass
class Rubric:
    """The tree, and the run's position in it."""

    checks: list[Check]

    def score(self) -> float:
        """How much of the objective is met, in [0, 1].

        A parent is met only when it is met, and its children's weight is *inside* it rather
        than added: a benchmark-specific leaf under `stages` is a way for `stages` to be
        partly satisfied, not a separate thing to be scored. That is the conservation
        AutoSOTA's decomposition holds to -- the root's weight is distributed down and no node
        creates weight of its own.
        """
        total = sum(one.weight for one in self.checks)
        if not total:
            return 0.0
        return sum(self._earned(one) for one in self.checks) / total

    def _earned(self, check: Check) -> float:
        if check.passed is True:
            return check.weight
        # Not met -- or not answered -- and its children say how far in it got,
        # proportionally, so a leaf cannot be worth more than the thing it sits under. A rung
        # nobody answered still earns what was observed under it, which is the case this whole
        # module is for: an evaluation that never produced a number, whose invocation was
        # nevertheless got right, is progress and has to be visible as progress.
        if not check.children or not any(one.reached() for one in check.children):
            return 0.0
        inner = sum(one.weight for one in check.children)
        if not inner:
            return 0.0
        return check.weight * sum(self._earned(one) for one in check.children) / inner

    def frontier(self) -> Check | None:
        """The first thing not yet met, which is what the run should be working on.

        The spine is ordered, so the frontier is a position and not a set: a run whose
        environment is not ready has no business being told about comparing candidates. It
        descends into a rung only when something under it has been observed -- so a rung that
        is simply unanswered is the frontier, while a rung whose own question failed and whose
        children say which part failed points at that part.
        """
        for check in self.checks:
            if check.passed is True:
                continue
            if any(one.reached() for one in check.children):
                for child in check.children:
                    if child.passed is not True:
                        return child
            return check
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"score": round(self.score(), 4), "at": now(),
                "checks": [one.as_dict() for one in self.checks]}


def evaluate(rubric: Rubric, facts: dict[str, Any]) -> Rubric:
    """Answer as much of the tree as the facts settle, and leave the rest unreached.

    `facts` is what the run knows about itself, under the spine's names. A question with no
    fact behind it is *not reached* rather than failed -- the distinction the whole module
    turns on, because a run that has not got as far as measuring has not failed to measure.
    """
    for check in rubric.checks:
        answer = facts.get(check.name)
        if isinstance(answer, dict):
            check.passed = bool(answer.get("passed"))
            check.because = str(answer.get("because") or "")[:400]
            for child in check.children:
                given = (answer.get("children") or {}).get(child.name)
                if isinstance(given, dict):
                    child.passed = bool(given.get("passed"))
                    child.because = str(given.get("because") or "")[:400]
        elif answer is not None:
            check.passed = bool(answer)
            check.because = ""
    return rubric


def plain() -> Rubric:
    """The spine with nothing under it, for a run whose benchmark has not been read yet."""
    return Rubric([Check(name=name, question=question, weight=weight)
                   for name, question, weight in SPINE])


# -- the benchmark's own leaves ------------------------------------------------------------

DECOMPOSE_SYSTEM = """You are breaking a research goal into sub-questions a run can answer \
about itself, before it can answer the final one.

You are given the benchmark, the stages it has, what it declares about its evaluation, and \
what is known about how it runs. The top-level questions are fixed and given to you:

{tops}

For each of them, propose the benchmark-specific questions that sit underneath -- what has to \
be true *for this benchmark* before that top-level question can be answered yes. Be concrete \
and make each one answerable from what a run records: "the evaluator used the benchmark's own \
initial states" is answerable; "the environment is good" is not.

Do not propose anything about how a metric is computed, the evaluation entry point, the \
splits, or the data. Those are not sub-goals; they are what must not change.

Return one JSON object: {{"<top-level name>": [{{"name": "<short>", "question": "<yes/no, \
answerable from the run's records>", "weight": <a number>}}]}}. Weight the children of one \
top-level question against each other; their total is that question's weight, not more."""


def decompose(client: Any, material: dict[str, Any], *, attempts: int = 2,
              on_event: Any = None) -> Rubric:
    """The spine with this benchmark's own questions under it.

    Never raises and falls back to the bare spine: a run whose rubric could not be decomposed
    is a run that scores less finely, and the alternative -- losing the run to a parse failure
    in the step that was supposed to enrich a scoreboard -- is the failure this module's
    neighbours keep producing.
    """
    tops = "\n".join(f"* `{name}` — {question}" for name, question, _ in SPINE)
    payload = json.dumps(material, ensure_ascii=False, default=str)[:40000]
    for _ in range(attempts):
        try:
            content, _ = client.chat_with_metadata(
                DECOMPOSE_SYSTEM.format(tops=tops), payload, max_tokens=3000, timeout=300,
                thinking="disabled")
        except Exception:                                            # noqa: BLE001
            return plain()
        try:
            from .execution_derive import _object
            value = _object(content)
            rubric = plain()
            for check in rubric.checks:
                rows = value.get(check.name)
                if not isinstance(rows, list):
                    continue
                for row in rows[:8]:
                    if not isinstance(row, dict) or not str(row.get("question") or "").strip():
                        continue
                    try:
                        weight = float(row.get("weight") or 1.0)
                    except (TypeError, ValueError):
                        weight = 1.0
                    check.children.append(Check(
                        name=str(row.get("name") or f"child-{len(check.children)}")[:60],
                        question=str(row["question"])[:300], weight=max(0.0, weight)))
            return rubric
        except Exception as exc:                                     # noqa: BLE001
            if on_event:
                on_event("rubric", [{"status": "the decomposition was refused",
                                     "error": redact(f"{type(exc).__name__}: {exc}")[:400]}])
    return plain()


ANSWER_SYSTEM = """A research run wrote itself a list of yes/no questions about what has to be \
true for its work to count, before it started. Answer them from what the run has recorded.

You are given the questions and the run's records: what each round did, what it was refused \
for, what came out of the measurements, and which stages exist.

Every answer is one of three things, and they are not the same:

* `"yes"` -- the records show it. Give `because` as the fact that shows it, and name where in \
the records it is.
* `"no"` -- the records show it is false, or show that a run happened without it.
* `null` -- the run has not got far enough for this question to have an answer. **This is the \
answer for anything the run has not reached**, and it is not a failure. A run that has not \
yet evaluated anything has not failed to use the benchmark's own initial states.

Do not answer `"yes"` from what the run *intends* to do, or from what it says it did. Answer \
from what the records contain. If you cannot point at the record that shows it, the answer is \
`null`.

Return one JSON object: {"answers": [{"name": "<the question's name>", "passed": true | \
false | null, "because": "<the record that shows it, or why there is none>"}]}. Answer every \
question you were given."""


def answer_children(client: Any, rubric: Rubric, *, material: dict[str, Any],
                    attempts: int = 2, on_event: Any = None) -> int:
    """Have the run answer the benchmark-specific questions it wrote for itself.

    They are questions about the run's own records -- "does the run log which config file it
    loaded", "did both evaluations finish with exit status 0" -- and no code can answer them,
    because the code does not know which facts about this benchmark decide its own success.
    The run decided that when it wrote them. This is where it says what it found.

    `None` is a first-class answer and is the most common one. A run that has not reached a
    question has not failed it, and a scorer that made those two the same would report a run
    that got two rounds in as having failed twenty things.

    Never raises, and returns how many questions it answered: a run whose scoreboard could not
    be filled in is a run with a coarser scoreboard, not a lost run.
    """
    pending = [child for check in rubric.checks for child in check.children]
    if not pending:
        return 0
    payload = json.dumps({
        "questions": [{"name": one.name, "question": one.question,
                       "under": check.name, "weight": one.weight}
                      for check in rubric.checks for one in check.children],
        "what_the_run_has_recorded": material,
    }, ensure_ascii=False, default=str)[:60000]
    for attempt in range(attempts):
        try:
            content, _ = client.chat_with_metadata(ANSWER_SYSTEM, payload, max_tokens=4000,
                                                   timeout=300, thinking="disabled")
            from .execution_derive import _object
            rows = _object(content).get("answers")
            if not isinstance(rows, list):
                raise ValueError("`answers` has to be a list")
            by_name = {str(one.name): one for check in rubric.checks for one in check.children}
            answered = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                one = by_name.get(str(row.get("name") or ""))
                if one is None:
                    continue
                passed = row.get("passed")
                one.passed = None if passed is None else bool(passed)
                one.because = str(row.get("because") or "")[:400]
                one.by = "model"
                answered += 1
            if answered:
                return answered
            raise ValueError("no answer named a question that was asked")
        except Exception as exc:                                     # noqa: BLE001
            if on_event:
                on_event("objective", [{"status": "the questions could not be answered",
                                        "error": redact(f"{type(exc).__name__}: {exc}")[:300]}])
            payload = payload + "\n\n### YOUR PREVIOUS ANSWER WAS REFUSED\n" + json.dumps(
                {"error": redact(str(exc))[:300],
                 "instruction": "Return one JSON object with an `answers` list."},
                ensure_ascii=False)
    return 0


def render(rubric: Rubric) -> str:
    """The rubric as a reader sees it: what is met, what is not, and where the run is.

    Every node, including the unmeasured ones. A scoreboard that shows only what passed
    cannot be told from one whose other rows were never looked at, and the question a reader
    arrives with is "how far did this get", which is answered by the holes.
    """
    if not rubric.checks:
        return "_没有评分表。_"
    out = [f"**完成度 {rubric.score():.0%}**", "",
           "| 问题 | 权重 | 状态 | 依据 |", "| --- | ---: | --- | --- |"]
    for check in rubric.checks:
        out.append(f"| **{check.question}** | {check.weight:.0f} | {check.state()} | "
                   f"{check.because[:120]} |")
        for child in check.children:
            out.append(f"| └ {child.question} | {child.weight:.0f} | {child.state()} | "
                       f"{child.because[:120]} |")
    frontier = rubric.frontier()
    if frontier is not None:
        out += ["", f"下一步在：**{frontier.question}**"]
    return "\n".join(out)
