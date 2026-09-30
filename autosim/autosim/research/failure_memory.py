"""What went wrong before, and what fixed it -- kept across runs.

Every run of this system starts as if nothing had ever been attempted. The same shape of
failure arrives at a different repository, or at the same one a week later, and the loop
reasons its way from nothing to the same fix it found last time. Measured here: within one
session the same class of fault recurred four times -- a `PATH` overwritten by a value that
did not carry the old one, a name guessed forsomething the machine already had, a checkout
placed one directory up from where it is -- and each occurrence cost a full budget of model
calls and process launches to rediscover.

AutoSOTA (arXiv:2604.05550) names the mechanism and it is worth copying exactly. Its AgentFix
maps a failure signal to a repair through `M_err`, "a reusable memory of historical failures
and remedies", under the rule of **retrieval before repair**: an error that matches a known
category is answered from memory first, and only then reasoned about. Their memory has three
levels -- per-task, **cross-task for recurring operational failures**, and notes that distil
repeated successes into explicit skills -- and it is made to generalise by the step this
module is built around: **raw tracebacks are normalised into reusable failure signatures**,
so that one repository's `ModuleNotFoundError` retrieves what fixed another repository's.

Two things it does beyond remembering.

**It remembers what did not work.** A signature carries the remedies that were tried and the
signature that came back afterwards, so a reviser can be shown the branches already exhausted
and told not to spend the round on them. AutoSOTA's phrase for this is ruling out
"already-exhausted branches", and the failure it prevents is the cyclic one: the same
plausible fix proposed, tried and refuted once per round until the budget ends.

**It stores the whole change, not a sentence about it.** A remedy that says "fix the PATH"
helps a reader and does not help a loop. What is stored is the invocation delta that
accepted -- the same fields the reviser returns -- so the next run can apply it, not read
about it.

What is *not* stored is anything that would change what is being measured. AutoSOTA draws a
Red Line around the evaluation protocol, the dataset split and the methodological
constraints, and nothing here records a remedy that touches them: a memory that remembers
editing an evaluator would hand that on as a fix.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_json, now, read_json


#: How many remedies one signature keeps. A signature with twenty remedies is a signature
#: whose text is too coarse to be one -- and the oldest are the least likely to be the answer.
PER_SIGNATURE = 8

#: Where the memory lives when the caller does not say. Beside the runs, not inside one: the
#: whole point is that it outlives the run that learned it.
def default_path(project_root: Path | None = None) -> Path:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    return root / "autoresearch_runs" / "failures.json"


#: Terminal colour. Stripped first, and this is not cosmetic: a program that colours its
#: errors emits `\x1b[1;35mModuleNotFoundError\x1b[0m`, and without this the escape becomes
#: part of the signature -- so the same failure with and without colour are two entries, and
#: the memory learns each of them separately.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

#: Text that varies between two instances of the same failure and must not be part of its
#: key, or nothing would ever match anything. Order matters: a path is replaced before the
#: numbers inside it are, or `x/1/y` becomes `x/<n>/y` and then a second pattern has to know
#: about it.
_VARYING: tuple[tuple[str, str], ...] = (
    # A quoted name: the module, the key, the path a program could not open. The opening
    # quote must not follow a letter, or the apostrophe in an English word is read as one:
    # `python: can't open file 'train.py'` became `can'…'train.py'`, the rule having eaten
    # from the contraction to the next quote it found.
    (r"(?<![\w])'(?:[^']*)'|(?<![\w])\"(?:[^\"]*)\"", "'…'"),
    (r"(?:/[\w.@+-]+){2,}", "<path>"),               # an absolute path
    (r"\b[\w.@+-]+/[\w.@+-/]+\b", "<path>"),         # a relative path with two segments
    (r"\b0x[0-9a-fA-F]+\b", "<addr>"),
    (r"\b\d+(?:\.\d+)?\b", "<n>"),
    (r"\s+", " "),
)

#: A line that names an exception, which is what a failure is when it is a traceback. Not
#: anchored to the start: a program may prefix it (`python: ...`), and a colour code may have
#: preceded it before `_ANSI` ran.
_EXCEPTION = re.compile(r"(?:^|[\s:])(\w+(?:\.\w+)*Error|Exception|KeyboardInterrupt|"
                        r"SystemExit)\b[:\s]*(.*)$")

#: The line a traceback opens with. It contains the word "exception" nowhere, but a program
#: that prints its own traceback as a message -- which Python does whenever it renders one
#: into a string, and which this system's own error wrapping does -- produces a line that
#: reads `Exception: Traceback (most recent call last)`. That is a heading, not a fault, and
#: keying on it put eight unrelated failures under one signature called
#: `exception: traceback (most recent call last)`.
_TRACEBACK_HEADER = re.compile(r"^Traceback \(most recent call last\):?$", re.I)


def signature(text: str, *, limit: int = 160) -> str:
    """The name of a failure, stable across two repositories that hit the same one.

    The exception type and the shape of its message, with everything that varies replaced:
    quoted names, paths, addresses, numbers. `ModuleNotFoundError: No module named 'einops'`
    and `ModuleNotFoundError: No module named 'XPolicyLab'` are one signature, because the
    fix for the first -- put the package's root on the search path -- is what the second one
    needed, and a memory that kept them apart would have to learn it twice.

    The last line, not the first: a traceback's opening frames are library internals and its
    final line is the thing that went wrong. Where there is no traceback at all, the text is
    normalised whole, which is the honest reading of a failure that is one line of prose.
    """
    said = _ANSI.sub("", str(text or "")).strip()
    if not said:
        return ""
    lines = [line.strip() for line in said.splitlines() if line.strip()]
    if not lines:
        return ""
    # The exception, when there is one, and the *last* one: a chained traceback ends with the
    # failure that actually stopped the program, and the lines it raised through are above it.
    #
    # Not simply the last line, which is what this was. A program that prints `[cleanup]
    # killing server pid=123` as it exits puts that after the traceback, and the signature
    # then describes a log line: the memory filled up with entries like
    # `[cleanup] killing server pid=<n>` and `[info] action dim: <n>`, none of which is a
    # failure and none of which can ever retrieve a remedy.
    found = [(_EXCEPTION.search(line), line) for line in lines]

    def _is_heading(match: Any, line: str) -> bool:
        """A traceback's opening line, as its own line or as an exception's message.

        The second form is what a program produces when it renders a traceback into a string,
        which Python does whenever one is put into an `f"{exc}"` -- and which this system's
        own error wrapping does. Matching only at the start of the line left that one through.
        """
        if match is None:
            return False
        return bool(_TRACEBACK_HEADER.match(line.strip())
                    or _TRACEBACK_HEADER.match((match.group(2) or "").strip()))

    named = [(m, line) for m, line in found if m and not _is_heading(m, line)]
    if named:
        match, line = named[-1]
        kind, message = match.group(1), match.group(2)
        if not message.strip():
            # A bare type name and nothing else -- `Exception` raised with no argument. It is
            # a real exception line and it says nothing: every argless raise in every program
            # shares it, and a signature that coarse retrieves the wrong remedy for all of
            # them. A failure the record cannot name is one it should not pretend to remember.
            return ""
    elif any(_is_heading(m, line) for m, line in found):
        # Everything that looked like an exception was a heading. There is no fault in this
        # text to name -- falling back to the last line would name the heading again, which
        # is what put eight unrelated failures under one signature called
        # `exception: traceback (most recent call last)`.
        return ""
    else:
        kind, message = "", lines[-1]
    out = f"{kind}: {message}" if kind else message
    for pattern, replacement in _VARYING:
        out = re.sub(pattern, replacement, out)
    out = out.strip().strip(": ").lower()
    return out[:limit]


def _remedy_of(change: dict[str, Any]) -> dict[str, Any]:
    """The part of a revision that is a change, with nothing about why it was made.

    `why` is prose and belongs with the example, not with the remedy: a stored sentence about
    one repository's problem is what makes a memory look specific when it is general.
    """
    return {k: v for k, v in (change or {}).items()
            if k not in ("why", "obstacle", "reasoning") and v not in (None, "", {}, [])}


class FailureMemory:
    """Remedies by signature, and the signatures they were tried against.

    One JSON document rather than a log, written whole on every change. It is small -- a
    signature, a handful of remedies each -- and the alternative is a reader that has to
    replay an append-only file to answer "what fixes this".
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_path()
        self.rows: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            try:
                document = read_json(self.path)
                self.rows = {str(k): v for k, v in (document.get("signatures") or {}).items()}
            except (OSError, ValueError, AttributeError):
                self.rows = {}

    # -- reading ------------------------------------------------------------------------

    def recall(self, text: str, *, limit: int = 3) -> dict[str, Any]:
        """What this failure was, and what has been done about it before.

        Retrieved *before* the reviser is asked to reason, which is the whole ordering: a
        failure that matches something already solved is answered from memory, and the model
        spends its turn on what memory does not cover.
        """
        key = signature(text)
        row = self.rows.get(key) or {}
        accepted = [r for r in (row.get("remedies") or []) if r.get("outcome") == "accepted"]
        refuted = [r for r in (row.get("remedies") or []) if r.get("outcome") == "refuted"]
        out: dict[str, Any] = {"signature": key}
        if accepted:
            out["remedies_that_worked"] = [
                {"change": r.get("change"), "times": r.get("times"),
                 "where": (r.get("repos") or [])[:3]} for r in accepted[:limit]]
        if refuted:
            # Named so that a reviser does not spend the round on them. AutoSOTA's term is
            # ruling out already-exhausted branches; the failure it prevents is a plausible
            # fix proposed, tried and refuted once per round until the budget ends.
            out["already_tried_and_did_not_work"] = [
                {"change": r.get("change"), "times": r.get("times")} for r in refuted[:limit]]
        if row.get("examples"):
            out["where_it_has_been_seen"] = (row.get("examples") or [])[:2]
        return out

    def known(self, text: str) -> bool:
        return signature(text) in self.rows

    # -- writing ------------------------------------------------------------------------

    def record(self, text: str, change: dict[str, Any], *, outcome: str, repo: str = "",
               detail: str = "") -> None:
        """Store a remedy against a signature, or mark one already stored.

        `outcome` is `accepted` when the signature stopped recurring after the change, and
        `refuted` when the same signature came back -- the second is not a failure of the
        record, it is the half of it that stops the loop trying the same thing again.
        """
        key = signature(text)
        if not key:
            return
        remedy = _remedy_of(change)
        if not remedy:
            return
        row = self.rows.setdefault(key, {"signature": key, "remedies": [], "examples": []})
        for existing in row["remedies"]:
            if existing.get("change") == remedy:
                existing["times"] = int(existing.get("times") or 1) + 1
                existing["outcome"] = outcome
                existing["at"] = now()
                if repo and repo not in (existing.get("repos") or []):
                    existing.setdefault("repos", []).append(repo)
                break
        else:
            row["remedies"].insert(0, {"change": remedy, "outcome": outcome, "times": 1,
                                       "at": now(), "repos": [repo] if repo else []})
            del row["remedies"][PER_SIGNATURE:]
        example = _example(text, repo=repo, detail=detail)
        if example and example not in row["examples"]:
            row["examples"].insert(0, example)
            del row["examples"][3:]
        self._write()

    def _write(self) -> None:
        try:
            atomic_json(self.path, {"schema_version": 1, "at": now(),
                                    "signatures": self.rows})
        except OSError:
            # A memory that cannot be written is a memory that will be relearned. It is not a
            # reason to end the run that was trying to write it.
            pass

    def summary(self) -> dict[str, Any]:
        remedies = sum(len(r.get("remedies") or []) for r in self.rows.values())
        worked = sum(1 for r in self.rows.values() for one in (r.get("remedies") or [])
                     if one.get("outcome") == "accepted")
        return {"signatures": len(self.rows), "remedies": remedies, "that_worked": worked}


#: How many separate repositories a remedy has to have worked in before it is written down as
#: a method. Two, because one is an anecdote about one repository and the claim being made is
#: that it transfers -- which is the whole reason the memory is keyed on signatures rather
#: than on repositories.
DISTIL_AFTER_REPOS = 2

#: Where distilled entries are written. Their own directory, not the curated library:
#: `autosim/skills/` is a person's writing with a person's evidence and the two should be
#: distinguishable at a glance. A distilled entry that turns out to be worth keeping can be
#: promoted by moving it, which is a decision a reader makes and not one the system makes for
#: them.
DISTILLED_DIR = "distilled_skills"


def distilled_dir(path: Path | None = None) -> Path:
    """Where this memory's distilled entries live -- beside the memory itself."""
    base = Path(path) if path else default_path()
    return base.parent / DISTILLED_DIR


def distil(path: Path | None = None, *, into: Path | None = None) -> list[Path]:
    """Write down every remedy that has worked in more than one place, as a skill entry.

    AutoSOTA's third level of memory, and the only one that changes what the system can do at
    a repository it has never seen: a failure signature and the change that fixed it are
    stored once, and when the same pairing holds in a second repository the pairing has
    stopped being an anecdote. What is written is exactly that -- the signature, the change,
    and where it worked -- offered to the controller like any other method and marked for what
    it is.

    **Marked, not disguised.** The entry says in its own confidence field that no person wrote
    it and how many repositories it rests on. A distilled entry that looked like a curated one
    would be believed at the weight of the curated library, which is the failure this whole
    system keeps finding in other forms: something that reads as more than it is.
    """
    memory = FailureMemory(path)
    directory = Path(into) if into else distilled_dir(path)
    written: list[Path] = []
    for key, row in sorted(memory.rows.items()):
        for remedy in row.get("remedies") or []:
            repos = [r for r in (remedy.get("repos") or []) if r]
            if remedy.get("outcome") != "accepted" or len(set(repos)) < DISTIL_AFTER_REPOS:
                continue
            written.append(_write_skill(directory, key, remedy, row, sorted(set(repos))))
    return written


def _write_skill(directory: Path, key: str, remedy: dict[str, Any], row: dict[str, Any],
                 repos: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    slug = _slug(key)
    identity = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    path = directory / f"{slug}-{identity}.md"
    change = json.dumps(remedy.get("change") or {}, ensure_ascii=False, sort_keys=True)
    examples = "\n".join(f"  - {one}" for one in (row.get("examples") or [])) or "  - (none kept)"
    # Quoted with `json.dumps`, which is valid YAML and which a signature cannot escape from:
    # the description carries the failure's own text, and that text contains colons. Written
    # raw, `description: When a command fails with `keyerror: '…'`` is read as a mapping and
    # the entry does not parse -- a skill file the library reports as unreadable rather than
    # one it offers.
    description = f"When a command fails with `{key}`, this is the invocation change that made it run."
    keywords = sorted(set(re.findall(r"[a-z0-9]+", key.casefold())))
    confidence = (
        f"distilled -- no person wrote this entry. It rests on {len(repos)} repositories and "
        f"{remedy.get('times', 1)} occurrences, and it says what the change was and not why it "
        f"works. Treat it as a starting point that has been right twice, not as a rule.")
    path.write_text(
        "---\n"
        f"name: {json.dumps(_slug(key))}\n"
        f"description: {json.dumps(description)}\n"
        "scope: general\n"
        f"confidence: {json.dumps(confidence)}\n"
        "evidence: |\n"
        f"  Signature: `{key}`\n"
        f"  Worked at: {', '.join(repos)}\n"
        f"  Seen here:\n{examples}\n"
        f"id: autosimsota.distilled.{identity}\n"
        "version: \"0.1.0\"\n"
        "capabilities: [failure.diagnosis, execution.command_repair]\n"
        f"keywords: {json.dumps(keywords, ensure_ascii=False)}\n"
        "inputs: [Current failure text, invocation, and repository evidence.]\n"
        "outputs: [A candidate invocation change with an applicability rationale.]\n"
        "preconditions: [The current failure signature matches and the recorded change is legal.]\n"
        "verification: [Apply only through the execution kernel, then rerun the same bounded stage.]\n"
        "permissions: [Advisory only; this entry grants no tool, filesystem, network, or process permission.]\n"
        "resources: [No direct compute; any rerun must reserve resources through the execution kernel.]\n"
        "invalid_when: [The failure cause differs, the change crosses a red line, or source evidence contradicts it.]\n"
        "---\n\n"
        f"# {key}\n\n"
        "A command failed this way at more than one repository, and the same change to its "
        "invocation was what made it run. The change, as the invocation carries it:\n\n"
        "```json\n"
        f"{change}\n"
        "```\n\n"
        "**What this does not say.** It does not say the change caused the fix -- the loop "
        "recorded it as the last revision before the command ran, and another difference "
        "would have been recorded the same way. It does not say the same change is right "
        "here: the signature is deliberately blind to the names in a failure, so two "
        "repositories that failed for different reasons under the same sentence are one "
        "entry. Read the failure before applying this, and say so in `why` if you do "
        "something else.\n", encoding="utf-8")
    return path


def _slug(key: str) -> str:
    """A name a file can carry, from a sentence about a failure."""
    words = re.findall(r"[a-z0-9]+", key.lower())[:8]
    return "-".join(words) or "unnamed-failure"


def _example(text: str, *, repo: str, detail: str) -> str:
    """One concrete instance, short enough to read and specific enough to recognise."""
    first = next((line.strip() for line in str(text or "").splitlines() if line.strip()), "")
    where = Path(repo).name if repo else ""
    said = (first or detail)[:160]
    return f"{where}: {said}" if where else said
