"""What a run may not do, stated before anything is proposed and checked against every idea.

A research loop that can change both the thing being measured and the way it is measured will
find improvements, and they will not be improvements. The failure is not hypothetical and it
is not rare: the whole difficulty of automating it is that a run which edits its evaluator
reports a better number *and* a clean record.

AutoSOTA draws this line explicitly, calls it the Red Line System, and enforces it in three
places rather than one: an enumerated list in the prompt, an audit table that marks every
candidate idea `REJECTED` or `CLEARED` **before it is run**, and a re-check when a candidate
is synthesised off the usual path. Their own formulation is worth keeping: the constraints are
"absolute prohibitions, with no exceptions under any circumstances".

**This system checked at the point of use and not before.** `assert_frozen` compares the
files an evaluation depends on at the moment it runs, and `values_the_program_contradicts`
compares a declaration against the program's own output. Both are good checks and both arrive
late: an idea that will touch the evaluator is proposed, selected, applied and only then
refused -- and the round is spent, and what a reader sees is a failed run rather than a
refused idea.

So this module is the missing half. It states the constraints, and it audits an idea against
them *before* anything runs, with a reason that names the line and the file.

**R1 to R6 are the system's, and R7 is the benchmark's.** That split is AutoSOTA's and it is
the right one: the first six are properties of any honest measurement, and the seventh is
whatever the particular benchmark makes non-negotiable -- which is found by reading it, not by
inheriting it. A benchmark whose evaluator must be run from its own directory, or whose seeds
must come from its own bank, says so in its own terms.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .common import digest

#: What an audited idea is. Two values and no middle: AutoSOTA's table marks each candidate
#: one or the other, and a third value meaning "probably fine" would be read as clearance.
#:
#: Spelled the same as `ideas.CLEARED` and `ideas.REJECTED`, and deliberately: a verdict is
#: what the audit found and a status is what the library holds, and they are the same word.
#: Written in two cases they were two different strings, so `idea.status != redlines.CLEARED`
#: was true for a cleared idea -- a comparison that is never true and never raises, which is
#: how a leap gets treated as refused while everything looks like it agreed.
CLEARED = "cleared"
REJECTED = "rejected"

#: The constraints every benchmark's run holds to, in this system's terms.
#:
#: Written as (id, what it forbids, why it is not negotiable) rather than as a list of rules
#: to test against files: the *why* is what lets a reader tell whether a new constraint belongs
#: here or whether an existing one is being stretched to cover something it does not.
STANDING: tuple[tuple[str, str, str], ...] = (
    ("R1", "changing how a metric is computed -- its parameters, its window, its aggregation, "
           "or reporting a best-of-N in place of the whole",
     "the number stops being the benchmark's number, and every later comparison is against a "
     "different ruler"),
    ("R2", "editing the evaluation entry point, the score aggregation, or the success "
           "predicate; in general, changing anything downstream of the point where a policy "
           "is run",
     "optimisation has to stay upstream of the evaluation boundary or it is not optimisation"),
    ("R3", "supplying predictions the policy did not produce -- caching results, reading "
           "recorded outputs, hard-coding a success",
     "a result that did not come from running the policy is not a result"),
    ("R4", "dropping a metric because it moved the wrong way, or trading one against another "
           "silently",
     "every metric is reported every round; a run that reports only its wins is not reporting"),
    ("R5", "moving data between the benchmark's splits -- training on evaluation seeds, "
           "calibrating on them, augmenting with them",
     "the split is what makes the evaluation an evaluation"),
    ("R6", "altering the contents of the benchmark's data or of an evaluation's initial "
           "states -- filtering, re-labelling, re-sampling, or substituting one episode for "
           "another",
     "the distribution being measured is the benchmark's, not the run's"),
)

#: Paths whose modification is a red line by default, matched against a repository-relative
#: path. A benchmark declares more of these -- R7 -- and this is the floor that holds when it
#: declares none. Deliberately broad: an idea that would edit a file matching one of these is
#: refused with the line named, and the way past is to say why the file is not an evaluator
#: rather than to widen the pattern.
_NEVER_EDIT = (
    "*eval*.py", "*eval*.sh", "*evaluate*", "*evaluation*", "*success*", "*metric*",
    "*score*.py", "*score*.sh", "*benchmark*.py", "*test*.py", "*verify*",
)


def _matches_protected(relative: str, pattern: str) -> bool:
    """A filename guard must not turn a matching output directory into a protocol file.

    Declaration-supplied paths may include directories and match the whole path. The
    standing filename patterns only match the basename: a trainer writing under a
    directory called ``verify`` cannot thereby mutate the evaluator or its protocol.
    """
    target = relative if "/" in pattern else Path(relative).name
    return fnmatch.fnmatch(target, pattern)

@dataclass
class Audit:
    """One idea against one line, with the reason."""

    idea: str
    verdict: str
    line: str = ""            # the red line it met, when it met one
    because: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"idea": self.idea, "verdict": self.verdict, "line": self.line,
                "because": self.because}

    @property
    def cleared(self) -> bool:
        return self.verdict == CLEARED


@dataclass
class RedLines:
    """The constraints for one benchmark's run: the standing six, plus what it declares.

    R7 is not written here because it cannot be. It is whatever this benchmark makes
    non-negotiable, and it is found by reading the repository and running it -- AutoSOTA
    generates it in its exploration phase and carries it into every later prompt. What this
    holds is the place to put it and the machinery to enforce it.
    """

    benchmark: str = ""
    #: (id, what it forbids, why) for this benchmark alone.
    particular: tuple[tuple[str, str, str], ...] = ()
    #: Repository-relative globs whose modification is refused. The evaluator's own files.
    protected: tuple[str, ...] = ()
    #: Free text: what the benchmark's own documents say must not be changed.
    notes: str = ""

    def all_of_them(self) -> list[tuple[str, str, str]]:
        return list(STANDING) + list(self.particular)

    def protected_paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*_NEVER_EDIT, *self.protected)))

    def as_dict(self) -> dict[str, Any]:
        return {"benchmark": self.benchmark,
                "standing": [{"id": one[0], "forbids": one[1], "why": one[2]}
                             for one in STANDING],
                "particular": [{"id": one[0], "forbids": one[1], "why": one[2]}
                               for one in self.particular],
                "protected": list(self.protected_paths()), "notes": self.notes}

    def states_itself(self) -> str:
        """The lines as text, for a prompt. The *why* travels with each one."""
        out = []
        for line, forbids, why in self.all_of_them():
            out.append(f"*{line}* — {forbids}\n  Because {why}.")
        if self.notes:
            out.append(self.notes)
        return "\n".join(out)


def audit(idea: Any, lines: RedLines) -> Audit:
    """Whether this idea may be run, and if not, which line it met.

    Audited **before** anything runs, so a change that reads as a change to measurement is
    refused in the library and never becomes a candidate.

    Two kinds of check, and the difference between them is the point.

    **What the system can see for itself is checked in code.** A change to a file matching a
    protected pattern is refused whether or not the change is to the part that measures,
    because deciding which part of `evaluate.py` measures is exactly the judgement a run must
    not be allowed to make about itself. That is enforcement and it stays.

    **What the idea *means* is the idea's own answer.** `crosses` is a field the idea was
    asked for when it was written, alongside its evidence and its risk, and it names the line
    the change would cross or nothing. It used to be inferred here from the words of the
    idea's description, against a table of thirty phrases -- and a phrase table is a model of
    how a model phrases things. It was wrong in both directions: "this does not touch the
    metric" was refused as changing the metric because the sentence contains the word, and
    "evaluate on 10 episodes rather than 100" was cleared because it contains none of the
    phrases. The first kills a good idea permanently, since the verdict is written into the
    library; the second runs a measurement change and reports it as optimisation.

    An idea that answers nothing is not thereby cleared. It is refused for not saying, which
    is what the record should say about it.
    """
    name = str(getattr(idea, "label", "") or getattr(idea, "mechanism", "") or "an idea")
    touches = [str(one) for one in (getattr(idea, "touches", None) or [])]
    change = getattr(idea, "change", None) or {}
    if isinstance(change, dict):
        if change.get("file"):
            touches.append(str(change["file"]))
        for patch in change.get("patches") or []:
            if isinstance(patch, dict) and patch.get("file"):
                touches.append(str(patch["file"]))

    for path in touches:
        for pattern in lines.protected_paths():
            if _matches_protected(path, pattern):
                return Audit(name, REJECTED, "R2",
                             f"it would change `{path}`, which matches `{pattern}`. "
                             f"Optimisation has to stay upstream of the evaluation boundary, "
                             f"and which part of that file measures is not a judgement this "
                             f"run may make about itself.")

    said = str(getattr(idea, "crosses", "") or "").strip().upper()
    by_line = {line: (forbids, why) for line, forbids, why in lines.all_of_them()}
    if said in by_line:
        forbids, why = by_line[said]
        return Audit(name, REJECTED, said,
                     f"it says of itself that it would be {forbids.split(';')[0].strip()}. "
                     f"{why.capitalize()}.")
    if said and said not in ("NONE", "NO", ""):
        # A line it named that this benchmark does not have. Refused rather than ignored: an
        # answer nobody understands is not an answer.
        return Audit(name, REJECTED, said,
                     f"it names the line {said!r}, which is not one of this run's lines "
                     f"({', '.join(sorted(by_line))}). An answer nobody can read is not one.")
    if not said:
        return Audit(name, REJECTED, "",
                     "it does not say whether it changes anything that measures, so it cannot "
                     "be cleared. Every idea carries that answer, the way it carries its "
                     "evidence and its risk.")
    return Audit(name, CLEARED, "",
                 "by its own account it changes nothing that measures: not a metric, not an "
                 "evaluator, not a split, not the data.")


def from_declaration(declaration: dict[str, Any], *, repo: Path) -> RedLines:
    """The lines for a benchmark, from what it says about itself and what it holds.

    Two sources and neither is a person. What the declaration records about the evaluation
    entry point becomes a protected path -- the file the run must not edit is named by the
    same reading that found it, so a benchmark that moves its evaluator moves its protection
    with it. And `notes` carries whatever the declaration's evidence said must not change.
    """
    particular: list[tuple[str, str, str]] = []
    protected: list[str] = []
    for capability in ("native_evaluation", "training", "new_trajectory_generation"):
        row = (declaration.get("capabilities") or {}).get(capability) or {}
        entry = str(row.get("entrypoint") or "").strip()
        if entry and row.get("status") == "declared":
            if capability == "native_evaluation":
                protected.append(entry)
                particular.append((
                    "R7", f"changing `{entry}`, which this benchmark declares as its own "
                          f"evaluator",
                    "the benchmark's evaluator is what makes its number the benchmark's "
                    "number; a run that fixes it to pass is measuring itself"))
    contract = declaration.get("task_contract") or {}
    if contract.get("instruction") or contract.get("success") or contract.get("metrics"):
        particular.append((
            "R7", "changing the task contract the declaration records -- the instruction "
                  "source, the success definition, the metric set",
            "those are what the benchmark is; an experiment that changes them is a "
            "different experiment"))
    return RedLines(benchmark=str(declaration.get("benchmark") or ""),
                    particular=tuple(particular), protected=tuple(protected),
                    notes=str(declaration.get("evidence") or "")[:2000])


def protected_hashes(repo: Path, lines: RedLines, *, max_files: int = 20000,
                     exclude_roots: tuple[Path, ...] = ()) -> dict[str, str]:
    """Freeze existing evaluator/protocol files without traversing bulk assets."""
    repo = Path(repo).resolve()
    exclusions = tuple(Path(one).resolve() for one in exclude_roots)
    skipped = {".git", ".venv", "venv", "env", "datasets", "data", "assets",
               "checkpoints", "outputs", "experiments", "site-packages", "__pycache__"}
    hashes: dict[str, str] = {}
    seen = 0
    for parent, dirs, files in os.walk(repo, followlinks=False):
        if any(Path(parent).resolve().is_relative_to(one) for one in exclusions):
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if name not in skipped and not any(
            (Path(parent) / name).resolve().is_relative_to(one) for one in exclusions)]
        for name in files:
            seen += 1
            if seen > max_files:
                raise RuntimeError("protocol freeze exceeded file inspection limit")
            target = Path(parent) / name
            if target.is_symlink() or not target.is_file():
                continue
            relative = str(target.relative_to(repo))
            if any(_matches_protected(relative, pattern)
                   for pattern in lines.protected_paths()):
                hashes[str(target)] = digest(target)
    return hashes


def table(audits: Iterable[Audit]) -> str:
    """The audit as a reader sees it, for the run's document.

    Every idea, its verdict, and the line it met -- including the cleared ones, because a
    table that lists only refusals cannot be told from a table that only some ideas reached.
    """
    rows = list(audits)
    if not rows:
        return "_没有想法经过审计。_"
    out = ["| 想法 | 判定 | 红线 | 理由 |", "| --- | --- | --- | --- |"]
    for one in rows:
        out.append(f"| {one.idea} | **{one.verdict}** | {one.line or '—'} | {one.because} |")
    cleared = sum(1 for one in rows if one.cleared)
    out.append("")
    out.append(f"{cleared} 条通过审计，{len(rows) - cleared} 条被拒绝。"
               f"被拒绝的想法**没有跑过** —— 它们在这一步就被挡住了。")
    return "\n".join(out)
