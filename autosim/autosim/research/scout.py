"""Read an unfamiliar benchmark and write down what it can do, then check the claims.

Onboarding a benchmark used to be a person reading a repository and typing its facts into
an adapter. Everything downstream was general -- the decision layer never learned a
benchmark's name -- but the doorway was narrow, and its width was "does a human have time
to onboard you".

This closes the doorway. The flow is the same one a careful person follows:

1. **Survey** the checkout. Deterministic, no model, no conclusions (`survey.py`).
2. **Triage**: the model reads the survey and asks for the specific files it needs.
3. **Propose**: the model reads those files and returns a declaration.
4. **Check**: every claim is tested against the filesystem, and the report separates what
   the model asserted from what the system confirmed.

The order matters. Asking for a declaration before the model has seen anything produces a
plausible fiction; asking for the files first makes the proposal answerable to evidence.
And checking afterwards is what makes the whole thing safe: a scout that hallucinates an
evaluator gets a `failed` check, not a broken run.

What this deliberately does not do is invent capability. A benchmark with no way to
produce a successful trajectory without a person cannot supply training data, and the
honest output is a declaration that says so -- which is a *result*, and a more useful one
than a run that fails eight hours later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import (atomic_json, event, now, object_digest, redact,
                     sanitize_model_text)
from .declaration import CAPABILITIES, STATUSES, DeclarativeAdapter, problems_in, space_from, verify
from .skills import skills_reference
from .strategy import executable_first_step, plan
from .survey import peek_many, summarise, summarise_for_model, survey


#: The task and the shape of the answer. How to *read* a repository -- where a task's
#: numbers live, what a path field is for, how ownership of an outside file is decided, when
#: to answer `unknown` -- is method knowledge and lives in the skill library, offered
#: alongside this. Keeping it there means it can be extended without touching this string,
#: and that the same reading applies whether a scout, a planner or a person is doing it.
SYSTEM = (chr(10) + chr(10)).join([
    "You are onboarding an unfamiliar robotics simulation benchmark into a research "
    "system, by reading its repository. You have no prior knowledge of this project. "
    "Everything you assert may be checked against the filesystem, and what you write down "
    "becomes the shape of every experiment that follows -- so assert only what the "
    "supplied facts support, and read the method library you were given before answering.",
    "The system needs seven things from a benchmark, described in `capabilities_needed`. "
    "Answer every one of them, including by saying `unsupported` or `unknown`.",
    "A field you cannot determine from the material is left as null, and a null is an "
    "answer: it says the material did not state it. Do not fill one in with a plausible "
    "number, and do not copy a placeholder. In `task_contract` every dimension is stated by "
    "`dataset_structure` in the survey, which is a recording of the actual file -- read the "
    "shapes there rather than inferring one.\n\n"
    "`benchmark`, `evidence`, `repo_markers`, `tasks` and `task_contract` are answered in "
    "one pass; `capabilities` and `optimization_space` in a second.",
    "For `optimization_space` you declare what the system may vary within this benchmark's "
    "own implementation -- training parameters its trainer accepts, and any collection "
    "setting its collector accepts. Declare only settings the source shows are implemented; "
    "the ranges are what the system will later be allowed to explore, so a range wider than "
    "the code supports becomes a run that dies at the trainer. Keep this executable core "
    "small: at most six training axes and four collection axes. Every numeric axis needs "
    "finite low and high bounds with low < high and a default inside them; omit a knob if "
    "you cannot justify safe bounds. Use an empty list, not null, for a section with no "
    "supported axes; omit `extra` unless a real third section exists. Every capability "
    "status must be `declared`, `unsupported` or `unknown`, never null. A declared "
    "capability needs a source-backed entrypoint and evidence; when unsure, use unknown "
    "with a reason rather than a guess.",
    "Investigate data acquisition before the optimization space is frozen. Trace the native "
    "collector's action provider (expert/planner, policy, or human), asset requirements, "
    "output/converter and actual trainer loader. Missing resources mean a conditional or "
    "unverified path, not proof that no collector exists. Include source-supported collection "
    "axes when the protocol permits data changes; do not silently omit them because shipped "
    "data already suffices for a baseline. Source discovery is not a successful runtime probe.",
    "In `task_contract.policy_representation`, answer `source` only when the native "
    "evaluator runs a source-defined controller or planner without loading trained weights; "
    "answer `artifact` when it loads a policy artifact; otherwise use null. This changes "
    "whether an evaluation with no training stage can be a measurement.",
    "If the repository establishes an official primary score, describe it in "
    "`task_contract.primary_metric` with name, maximize/minimize direction, unit and "
    "source (`log`, `json` or `csv`); otherwise leave it null. Do not assume every "
    "benchmark uses success rate. If a native CSV contains one row per completed episode, "
    "state its metric column, aggregation, task/episode identity columns, and optionally "
    "an initial-state SHA-256 column only when repository code shows where those bytes "
    "come from. A shared seed or invented hash is not proof of paired initial states.",
    "If repository source and experiment constraints establish important secondary "
    "objectives, optionally supply research_goal.guardrail_metrics with explicit name, "
    "direction, unit, source, source-backed evidence and max_regression in native units. "
    "Use the primary result artifact or an exact native log label. Do not guess tolerances; "
    "secondary mappings are rechecked before binding and frozen before optimization.",
    "Return exactly one JSON object and nothing else.",
]) + chr(10)


def _capability_lines() -> dict[str, str]:
    return dict(CAPABILITIES)


TRIAGE_SYSTEM = (
    "You are deciding what to read. A repository survey has been taken and is given to "
    "you; it contains file names, sizes, the structure of one representative recorded "
    "trajectory, and the signals found in each source -- but not the contents of files. "
    "Return exactly one JSON object with the key `files_to_read`: up to twelve "
    "repository-relative paths whose contents would let you answer what this benchmark can "
    "and cannot do. Prefer the files that decide capability -- anything that generates or "
    "records trajectories, evaluates a policy, or trains one -- over generic utilities and "
    "data files. Include at most one representative task definition. Do not ask for a file "
    "only to confirm its name."
)


def _section(name: str, payload: Any) -> str:
    return f"### {name}\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"


def _files_section(files: list[dict[str, Any]]) -> str:
    """The files the model asked for, as text rather than as JSON values.

    Kept out of the JSON envelope deliberately. When the whole request is one JSON
    document, a model that wants to be helpful mirrors it, and a reply that re-emits the
    survey and the file contents both wastes its output budget on the answer and -- because
    the mirrored text is where quoting and brace counting go wrong -- is the most common way
    a draft fails to parse.
    """
    parts = []
    for row in files:
        path = row.get("requested_as") or row["path"]
        if not row.get("readable"):
            parts.append(f"--- {path} (not readable)")
            continue
        body = "\n".join(row.get("head", "").splitlines()[:120])
        parts.append(f"--- {path} ({row.get('total_lines')} lines total)\n{body}")
    return "\n\n".join(parts) if parts else "(the model asked for no files)"


def _envelope(instruction: str, keys: tuple[str, ...], **sections: Any) -> str:
    head = (f"### INSTRUCTION\n{instruction}\n\n"
            f"### RETURN EXACTLY THESE TOP-LEVEL KEYS\n"
            f"{json.dumps(list(keys))}\n")
    body = "\n\n".join(_section(name, payload) for name, payload in sections.items()
                        if payload is not None)
    return head + "\n" + body


def triage(report: dict[str, Any], client: Any, *, limit: int = 12) -> dict[str, Any]:
    """Ask which files to read. A model that has not read anything yet is guessing."""
    methods = skills_reference(
        query="repository discovery benchmark workflow train evaluate collect entrypoint "
              "task assets source successful trajectories expert policy teleoperation")
    user = _envelope("List the files worth reading.", ("files_to_read",),
                     METHODS=methods,
                     SURVEY=summarise_for_model(report))
    content, metadata = client.chat_with_metadata(
        sanitize_model_text(TRIAGE_SYSTEM),
        sanitize_model_text(user, local_roots=(Path(report["repo"]),)),
        max_tokens=1200, timeout=120, thinking="disabled")
    try:
        chosen = _json_object(content).get("files_to_read") or []
    except ValueError:
        chosen = []
    return {"files_to_read": [str(p) for p in chosen if isinstance(p, str)][:limit],
            "method_selection": methods["selection"],
            "provider": metadata, "response_sha256": object_digest(content)}


IDENTITY_KEYS = ("benchmark", "evidence", "repo_markers", "tasks", "task_contract", "assets")
DECISION_KEYS = ("capabilities", "optimization_space")


def skeleton(capabilities: bool = True) -> dict[str, Any]:
    """The shape of a declaration, split so a draft is two answerable questions.

    One object covering identity, capability and the optimization space runs to several
    thousand tokens of output, and a model asked for that much in one reply drops the tail
    or malforms the middle -- which reads as a model that does not understand the task when
    it is really a model that ran out of room. Identity is a question about the repository;
    capability is a question about what it can do. They are asked separately and merged.
    """
    identity = {
        "benchmark": "<the project's own name for itself, from its manifests>",
        "evidence": "<which facts in the survey support this reading, in one paragraph>",
        "repo_markers": ["<files that only exist in this project>"],
        "tasks": {
            "kind": "one of: glob | directory | module_registry | explicit",
            "pattern": "<a real glob relative to the repository root, or the task directory>",
            "task_id_from": "<how a task's identifier is derived, e.g. file stem>",
            "names": ["<only when kind is explicit or module_registry>"],
            "instruction_source": "<where a task's natural-language instruction lives>",
        },
        # Nulls, not zeros. This block used to be filled with zeros as placeholders, which
        # taught the model to answer zero: a number it did not know came back as the number
        # it was shown, and the validator -- which requires a positive integer, because a
        # dimension of zero is not a dimension -- rejected it three times and ended the run.
        # The placeholders were the fault, not the model.
        "task_contract": {
            "state_dim": None,
            "action_dim": None,
            "cameras": {"<camera_name>": "<the shape recorded in the trajectory>"},
            "max_episode_steps": None,
            "control_parts": ["<arm/gripper groups the action vector drives>"],
            "recorded_fps": None,
            "instruction": "<a sentence, or {\"from\": \"<path to a catalogue>\"}>",
            "policy_representation": None,
            "primary_metric": None,
        },
        "assets": {
            "dataset": {"path": "<path within the checkout>", "format": "<what it is>"},
            "checkpoint": {"path": "<path, or omit and give why>"},
        },
    }
    if not capabilities:
        return identity
    return {
        "capabilities": {
            name: {"status": "one of: " + " | ".join(STATUSES),
                   "entrypoint": "<path, when status is declared>",
                   "evidence": "<what in the source shows this, when status is declared>",
                   "why": "<when status is unsupported or unknown>"}
            for name in CAPABILITIES
        },
        "optimization_space": {
            "collection": [{"name": "<setting>", "kind": "choice | integer | number | structure",
                            "description": "<what it does>", "values": ["<choice options>"],
                            "low": 0, "high": 1, "default": None}],
            "training": [{"name": "<parameter>", "kind": "choice | integer | number",
                          "description": "<what it does>", "values": ["<choice options>"],
                          "low": 0, "high": 1, "default": None, "group": "<optional nesting>"}],
            # Written out because `{}` said nothing, and a model looking for somewhere to put
            # a fact that was not a collection knob and not a training knob put a *capability*
            # name here. The shape is the same as the other two: a section the controller can
            # see, holding axes. It is for a knob that belongs to neither half -- how many
            # episodes this round is judged on, say -- and it stays `{}` when there is none.
            "extra": {
                "<a third section's name, if this benchmark needs one>": [
                    {"name": "<setting>", "kind": "choice | integer | number | structure",
                     "description": "<what it does>", "values": ["<choice options>"],
                     "low": 0, "high": 1, "default": None}],
            },
        },
    }


def _balanced_object(text: str) -> dict[str, Any] | None:
    """The first complete JSON object in a reply, ignoring anything around it.

    Models wrap the object in fences, append an explanation after it, or -- most often of
    all -- emit the whole object correctly and leave off the final closing brace. The first
    two are handled by locating the object with a bracket stack; the third by closing the
    stack and trying again. That last repair is safe because nothing downstream trusts the
    parse: the declaration it yields still has to satisfy `problems_in`, so a reply that was
    genuinely cut off mid-value fails there instead of being accepted here.

    **Raises when it found an object and the object is malformed.** Returning `None` for that
    case -- which this did -- reports "nothing was found" when the truth is "an object was
    found and it has a `;` where a `,` belongs". The distinction is not cosmetic: the message
    is the whole of what the model is told when its draft is rejected, and the repair loop
    runs on that message. A model told "no complete JSON object found" re-emits the same
    object; a model told "Expecting ',' delimiter at column 2569" fixes it.
    """
    body = text.strip()
    for opener in ("```json", "```"):
        if body.startswith(opener):
            body = body[len(opener):]
    end = body.rfind("```")
    if end != -1:
        body = body[:end]
    start = body.find("{")
    if start == -1:
        return None
    stack: list[str] = []
    in_string, escaped = False, False
    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack:
                continue
            stack.pop()
            if not stack:
                candidate = body[start:index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise _Malformed(candidate, exc) from None
    if in_string or not stack:
        return None
    try:
        return json.loads(body[start:] + "".join(reversed(stack)))
    except json.JSONDecodeError as exc:
        raise _Malformed(body[start:], exc) from None


class _Malformed(ValueError):
    """A JSON object was found and does not parse, with enough of it to act on.

    Carries the fault, its position, and the text either side of it. The surrounding text is
    the part that makes the message usable: a column number alone does not say what to change,
    and the characters around the fault usually do.
    """

    def __init__(self, candidate: str, error: json.JSONDecodeError):
        position = error.pos
        window = candidate[max(0, position - 120):position + 120]
        super().__init__(
            f"the JSON object is malformed: {error.msg} at line {error.lineno} "
            f"column {error.colno}. Around the fault: ...{window}...")
        self.candidate = candidate
        self.error = error


def _json_object(content: str) -> dict[str, Any]:
    """The object in a reply, or a message saying what is wrong with it.

    The message is the model's only feedback, so the two failures are kept apart: nothing
    that looks like an object was found at all, versus an object was found and it is
    malformed at a named place. The second is the common one and it is fixable in one turn.
    """
    value = _balanced_object(content)
    if value is None:
        raise ValueError("no complete JSON object found in the response")
    return value


def _faults(problems: list[str]) -> None:
    if problems:
        raise ValueError("; ".join(problems[:6]))


def _ask(system: str, payload: dict[str, Any], client: Any, *, keys: tuple[str, ...],
         stage: str, output: Path | None, attempts: list[dict[str, Any]],
         repair_attempts: int, validate: Any = None) -> dict[str, Any]:
    """Ask for one part of a declaration, repairing against the reported fault.

    ``validate`` runs on the parsed draft and may raise. That is what lets a fault the
    system only notices after assembling the whole document -- an axis whose default its own
    range rejects, say -- come back as a correction instead of ending the run. The model
    gets the same sentence a person would.
    """
    user = payload["__prompt__"]
    for repair in range(repair_attempts):
        current = user if repair == 0 else user + (
            "\n\n### YOUR PREVIOUS REPLY WAS REJECTED\n"
            + json.dumps(attempts[-1]["error"]) + "\n"
            + "Return one JSON object with exactly " + json.dumps(list(keys)) + " as its "
            "top-level keys and nothing else. Do not repeat the survey, the file contents or "
            "any other input back to me. Do not wrap the object in prose.")
        content, metadata = client.chat_with_metadata(
            sanitize_model_text(system),
            sanitize_model_text(current), max_tokens=8192,
                                                      timeout=300, thinking="disabled")
        if output is not None:
            atomic_json(output / f"draft_{stage}_{repair + 1}.json",
                        {"stage": stage, "attempt": repair + 1, "content": content,
                         "provider": metadata, "prompt_chars": len(current),
                         "response_chars": len(content),
                         "prompt": sanitize_model_text(current) if repair == 0
                         or attempts[-1]["status"] == "rejected" else None})
        try:
            draft = _json_object(content)
            missing = sorted(set(keys) - set(draft))
            if missing:
                raise ValueError(f"missing required key: {missing}")
            if validate is not None:
                validate(draft)
            attempts.append({"stage": stage, "attempt": repair + 1, "status": "accepted",
                             "response_sha256": object_digest(content)})
            return draft
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            attempts.append({"stage": stage, "attempt": repair + 1, "status": "rejected",
                             "error": redact(f"{type(exc).__name__}: {exc}"),
                             "finish_reason": (metadata or {}).get("finish_reason"),
                             "response_chars": len(content),
                             "response_sha256": object_digest(content)})
    raise ValueError(f"no usable {stage} draft: {attempts[-1].get('error')}")


def declared_content(declaration: dict[str, Any]) -> tuple[int, int]:
    """How much a declaration claims: axes it offers, capabilities it settles.

    A repair is meant to change a claim, not to remove it, and the two look the same to a
    check that only counts what is wrong: deleting the thing that failed also reduces the
    faults. RoboTwin is exactly that case. A repair replaced thirty-seven declared axes --
    twenty-five collection knobs and twelve training ones, each citing the file it was read
    from -- with none at all, and it was accepted, because the only thing compared was the
    number of path faults and a shorter document has fewer of those.

    Nothing here is specific to a benchmark: a declaration claims axes and capability
    statuses, and this counts both.
    """
    space = declaration.get("optimization_space") or {}
    axes = sum(len(rows) for rows in space.values() if isinstance(rows, list))
    capabilities = declaration.get("capabilities") or {}
    settled = sum(1 for row in capabilities.values()
                  if isinstance(row, dict) and str(row.get("status")) != "unknown")
    return axes, settled


def propose(report: dict[str, Any], files: list[dict[str, Any]], client: Any, *,
            output: Path | None = None, feedback: dict[str, Any] | None = None,
            repair_attempts: int = 5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Draft a declaration in two passes and check it as one document.

    Every attempt is written to disk as it happens, including the raw response. A rejected
    draft is the most informative thing a scout produces -- it is the difference between a
    model that misunderstood the task and one that ran out of room -- and keeping only the
    final error throws that away.
    """
    attempts: list[dict[str, Any]] = []

    coding_runtime = bool(getattr(client, "supports_main_agent", False))
    identity_methods = skills_reference(
        query="benchmark contract task repository assets dataset checkpoint metric success "
              "trajectories expert policy source evidence",
        include_index=not coding_runtime,
        max_selected=2 if coding_runtime else 3)
    identity = _ask(SYSTEM, {"__prompt__": _envelope(
        "Identify this project and describe one of its tasks. Answer from the survey, the "
        "file contents and the recorded trajectory structure. For each assets.dataset and "
        "assets.checkpoint, choose exactly one: provide a real repository-relative path and "
        "omit why, or provide why and omit path. Never copy an angle-bracket placeholder as "
        "a path. If the workflow needs no such asset, explain that under why instead of "
        "inventing a location.",
        IDENTITY_KEYS,
        WHAT_THE_SYSTEM_NEEDS=_capability_lines(),
        METHODS=identity_methods,
        SURVEY=summarise_for_model(report),
        FILES_YOU_ASKED_FOR=_files_section(files),
        FAILED_CHECKS_ON_YOUR_PREVIOUS_ANSWER=feedback,
        REQUIRED_SHAPE=skeleton(capabilities=False))},
        client, keys=IDENTITY_KEYS, stage="identity", output=output,
        attempts=attempts, repair_attempts=repair_attempts,
        validate=lambda draft: _faults(problems_in(draft, required=IDENTITY_KEYS)))
    for attempt in attempts:
        if attempt.get("stage") == "identity":
            attempt["method_selection"] = identity_methods["selection"]
    def assembled(draft: dict[str, Any]) -> None:
        """Validate as the whole document, because that is how it will be consumed."""
        whole = {**identity, **draft}
        _faults(problems_in(whole))
        space_from(whole)

    capability_methods = skills_reference(
        benchmark=identity.get("benchmark"),
        query="capability mapping optimization space available interventions data collection "
              "training evaluation parameter source evidence",
        include_index=not coding_runtime,
        max_selected=2 if coding_runtime else 3)
    decision = _ask(SYSTEM, {"__prompt__": _envelope(
        "Answer what this benchmark can and cannot do, and declare what the system may "
        "vary when it works with it.",
        DECISION_KEYS,
        WHAT_THE_SYSTEM_NEEDS=_capability_lines(),
        METHODS=capability_methods,
        ALREADY_ESTABLISHED=identity,
        SURVEY=summarise_for_model(report),
        FILES_YOU_ASKED_FOR=_files_section(files),
        REQUIRED_SHAPE=skeleton(capabilities=True))},
        client, keys=DECISION_KEYS, stage="capabilities", output=output,
        attempts=attempts, repair_attempts=repair_attempts,
        validate=assembled)
    for attempt in attempts:
        if attempt.get("stage") == "capabilities":
            attempt["method_selection"] = capability_methods["selection"]

    declaration = {**identity, **decision}
    assembled(decision)
    return declaration, attempts


def _neighbours(repo: Path, pattern: str, limit: int = 4) -> list[str]:
    """Real paths in the repository whose name resembles a pattern that matched nothing."""
    name = Path(str(pattern)).name
    if not name or name == "*":
        return []
    found: list[str] = []
    for path in sorted(Path(repo).rglob(name))[:limit]:
        if path.is_file():
            found.append(str(path.relative_to(repo)))
    if found:
        return found
    # The basename may itself be too specific; fall back to the suffix, which is what the
    # reader was reaching for when it wrote the pattern.
    suffix = Path(name).suffix
    if suffix:
        for path in sorted(Path(repo).rglob(f"*{suffix}"))[:limit]:
            if path.is_file():
                found.append(str(path.relative_to(repo)))
    return found


def path_faults(result: dict[str, Any]) -> dict[str, Any]:
    """Checks that failed for a reason the draft could fix, with what was looked for.

    A missing file the model was right about is not a fault to repair -- it is a finding,
    and re-asking would only produce a different way of saying the same thing. What is
    worth repairing is a check that failed because the answer was never a path in the first
    place, or because a relative path pointed at the wrong directory.
    """
    faults: dict[str, Any] = {}
    for row in result.get("checks", []):
        if row.get("check") == "path_exists" and not row.get("passed"):
            faults.setdefault("repo_markers", []).append(row["subject"])
    for name, row in result.get("assets", {}).items():
        if row.get("status") == "failed":
            faults.setdefault("assets", {})[name] = {
                "you_wrote": row.get("path"),
                "files_the_survey_found": row.get("nearest_surveyed", [])}
    if not result.get("tasks"):
        # Same treatment a failed asset path gets: the point is not that it failed but where
        # the files actually are. A pattern that misses by one directory is a correction, and
        # a model told only "no match" rewrites the same wrong guess -- which is what a
        # repair round did here before this was added.
        faults["tasks"] = {"reason": "your pattern matched no files",
                           "you_wrote": result.get("task_pattern"),
                           "note": "a pattern containing < or > matches nothing",
                           "files_like_it": result.get("task_pattern_neighbours", [])}
    for name, row in result.get("capabilities", {}).items():
        limitation = row.get("limitation") or ""
        if row.get("status") == "failed" and "does not exist" in limitation:
            faults.setdefault("capabilities", {})[name] = limitation
    return faults


def run(repo: Path, *, client: Any, output: Path, extra_roots: tuple[Path, ...] = (),
        read_limit: int = 12, peek_lines: int = 80,
        verification_repairs: int = 1) -> dict[str, Any]:
    """Survey, ask, propose, check -- and write every step down.

    The artifacts are written as they are produced rather than at the end, because the
    interesting failure is a scout that ran out of budget halfway and left nothing behind
    to show which step it reached.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = survey(repo, extra_roots=extra_roots)
    atomic_json(output / "survey.json", report)

    triage_result = triage(report, client, limit=read_limit)
    atomic_json(output / "triage.json", triage_result)
    requested = [path for path in triage_result["files_to_read"]]
    files = peek_many(requested, repo=Path(report["repo"]),
                      lines=peek_lines, limit=read_limit)
    for row in files:
        row["requested_as"] = row.get("path", "[withheld-path]")
    atomic_json(output / "file_contents.json",
                {"requested": requested,
                 "read": [row for row in files if row.get("readable")]})

    declaration, attempts = propose(report, files, client, output=output)
    result = verify(declaration, repo, survey_report=report)
    result["surveyed_assets"] = [row["path"] for row in
                                 report["datasets"] + report["model_artifacts"]]
    result["task_pattern_neighbours"] = _neighbours(repo, declaration["tasks"].get("pattern", ""))

    # A check that failed because a path was mistyped is worth one retry: the model can see
    # what was looked for and fix it, which is the difference between a declaration that
    # says "this dataset does not exist" and one that found it. Bounded, because a second
    # attempt that fails the same way is a finding, not a mistake.
    for round_index in range(verification_repairs):
        faults = path_faults(result)
        if not faults:
            break
        try:
            repair, more = propose(
                report, files, client, output=output,
                feedback={"round": round_index + 1, "what_was_looked_for": faults})
        except ValueError as exc:
            # A repair is an attempt to improve an answer that already exists. Treating its
            # failure as fatal would throw away the whole declaration to punish a retry, so
            # the first answer stands and the attempt is recorded as what it was.
            event(output / "scout_events.jsonl", "verification_repair_failed",
                  round=round_index + 1, error=redact(str(exc)))
            break
        attempts += more
        repaired = verify(repair, repo, survey_report=report)
        repaired["surveyed_assets"] = result["surveyed_assets"]
        repaired["task_pattern_neighbours"] = _neighbours(
            repo, repair["tasks"].get("pattern", ""))
        atomic_json(output / f"verification_repair_{round_index + 1}.json", repaired)
        # Fewer faults is necessary and not sufficient. A repair that reduces the faults by
        # claiming less has not repaired anything -- it has deleted the part that was
        # checked -- so a repair is taken only when it does not shrink what the declaration
        # declares. Either way the attempt is recorded, because a repair that was refused is
        # the most informative thing about why the round happened twice.
        before, after = declared_content(declaration), declared_content(repair)
        improved = len(path_faults(repaired)) < len(faults)
        kept_its_claims = all(new >= old for new, old in zip(after, before))
        if improved and kept_its_claims:
            declaration, result = repair, repaired
        else:
            event(output / "scout_events.jsonl", "verification_repair_refused",
                  round=round_index + 1, fewer_faults=improved, kept_its_claims=kept_its_claims,
                  claims_before=list(before), claims_after=list(after))
            break

    atomic_json(output / "declaration.json",
                {"schema_version": 1, "created_at": now(), "declaration": declaration,
                 "attempts": attempts,
                 "declaration_sha256": object_digest(declaration)})
    atomic_json(output / "verification.json", result)

    adapter = DeclarativeAdapter(declaration=declaration, repo=repo, verification=result)
    from .adapter_protocol import check_adapter
    structural = check_adapter(adapter)
    summary = _summary(declaration, result, structural)

    # What the loop should be is a decision about this benchmark, so it is asked for after
    # the facts are established and checked against them. Without a plan the system has an
    # adapter and nothing to do with it; with one it has a research programme, and a
    # benchmark that cannot collect data gets one built from the families it does have.
    # Gated on the adapter protocol alone, not on the blocking list. A plan is reasoning
    # about what to do with what exists, and it is most useful precisely when something is
    # missing -- a plan that reads "the dataset path is wrong, fix it, then do X" is
    # better than no plan and a complaint. What the plan must satisfy is checked against
    # the declared space and the established capabilities, which is a real check.
    research_plan, plan_log = None, []
    if not structural:
        try:
            research_plan, plan_log = plan(
                adapter.optimization_space(adapter.select_task("auto")),
                result["capabilities"], client, benchmark=declaration["benchmark"],
                methods=skills_reference(
                    benchmark=declaration["benchmark"],
                    query="research strategy experiment design intervention family comparison "
                          "metric resource budget control causal attribution"),
                evidence={"declaration_reasoning": declaration["evidence"],
                          "capability_notes": {name: row.get("limitation") for name, row
                                               in result["capabilities"].items()}},
                output=output)
        except ValueError as exc:
            summary["plan_fault"] = redact(str(exc))
    if research_plan is not None:
        summary["plan"] = {
            "strategy_id": research_plan["strategy_id"],
            "families": {name: bool(row.get("available"))
                         for name, row in research_plan["intervention_families"].items()},
            "first_step": executable_first_step(research_plan),
            "measurement": research_plan["measurement"],
            "stop_condition": research_plan["stop_condition"],
        }
    atomic_json(output / "strategy.json",
                {"schema_version": 1, "created_at": now(), "strategy": research_plan,
                 "attempts": plan_log})
    atomic_json(output / "onboarding.json",
                {"schema_version": 1, "created_at": now(), "repo": str(repo),
                 "benchmark": declaration["benchmark"], "summary": summary,
                 "readiness": "ready" if not summary["blocking"] else "blocked",
                 "checks_passed": result["state"] == "verified",
                 "structural_faults": structural})
    return {"declaration": declaration, "verification": result, "strategy": research_plan,
            "summary": summary, "structural_faults": structural}


def _summary(declaration: dict[str, Any], result: dict[str, Any],
             structural: list[str]) -> dict[str, Any]:
    """What the system can and cannot count on, in the system's own vocabulary.

    `blocking` holds only what stops *any* research on this benchmark: an unconfirmed
    identity, no tasks, no way to measure, no way to train. Everything else here is a
    fact about what is available, and which facts matter depends on the research plan --
    which is a question for the planner, not for this function.

    An earlier version of this function also refused a benchmark with no automated
    trajectory producer, on the grounds that a round begins by collecting data. That was
    true of the one benchmark the loop was built on and false in general: a benchmark that
    ships demonstrations and an evaluator supports research into which demonstrations are
    used and how they are weighted, which needs no new trajectories at all. Encoding one
    benchmark's shape as a precondition for every benchmark is the mistake this whole
    layer exists to stop making, so the availability of each intervention is reported as a
    fact and the choice between them belongs to the plan.
    """
    capabilities = result["capabilities"]
    blocking = []
    if result["identity"] != "confirmed":
        blocking.append({"reason": "the repository does not match the declared identity",
                         "detail": result.get("identity_failures", [])})
    if not result.get("tasks"):
        blocking.append({"reason": "no tasks were enumerated",
                         "detail": declaration["tasks"].get("pattern")})
    if result["state"] != "verified":
        blocking.append({"reason": "at least one declared capability failed its checks",
                         "detail": sorted(name for name, row in capabilities.items()
                                          if row["status"] == "failed")})
    for name in ("native_evaluation", "training"):
        if capabilities.get(name, {}).get("status") not in {"declared", "verified"}:
            blocking.append({"reason": f"{name} is not confirmed",
                             "detail": capabilities.get(name, {}).get("limitation")})
    for name, row in result["assets"].items():
        if row.get("status") == "failed":
            blocking.append({"reason": f"the declared {name} does not resolve",
                             "detail": row.get("path"),
                             "nearest_surveyed": row.get("nearest_surveyed", [])})
    if capabilities.get("native_evaluation", {}).get("status") == "failed":
        blocking.append({"reason": "the declared evaluator could not be located",
                         "detail": capabilities["native_evaluation"].get("limitation")})
    if structural:
        blocking.append({"reason": "the declaration does not satisfy the adapter protocol",
                         "detail": structural})
    return {"capabilities": {name: {"status": row["status"],
                                    "limitation": row.get("limitation")}
                             for name, row in capabilities.items()},
            "tasks": result.get("task_count", 0),
            "assets": {name: row.get("status") for name, row in result["assets"].items()},
            "blocking": blocking}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autosim scout",
        description="Read an unfamiliar benchmark repository and report what it can do.")
    parser.add_argument("repo", help="Path to the benchmark checkout")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--extra-root", type=Path, action="append", default=[],
                        help="Further directories to search for this benchmark's data and "
                             "checkpoints; repeatable")
    parser.add_argument("--read-limit", type=int, default=12,
                        help="How many files the scout may ask to read")
    parser.add_argument("--peek-lines", type=int, default=80)
    parser.add_argument("--report-only", action="store_true",
                        help="Survey without calling a model; writes survey.json only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    # One line, and it used to reach a whole parallel world: this borrowed two general
    # helpers from a module that carried one benchmark's task table, its epoch counts and a
    # finished research loop, so importing the scout imported all of it. The helpers live
    # beside the client that needs them now, and the scout's closure is its own business.
    from autosim.llm_client import LLMClient, PROJECT_ROOT, load_credential_file

    if args.report_only:
        from .survey import survey as take_survey
        report = take_survey(args.repo, extra_roots=tuple(args.extra_root))
        print(json.dumps(summarise(report), ensure_ascii=False, indent=2))
        return 0

    environment = load_credential_file(project_root=PROJECT_ROOT)
    client = LLMClient()
    if not client.available:
        raise SystemExit("DEEPSEEK_API_KEY is not set; the scout has no model to ask")
    output_root = args.output_root or PROJECT_ROOT / "autoresearch_runs" / "scouting"
    result = run(Path(args.repo), client=client, output=output_root / args.run_id,
                 extra_roots=tuple(args.extra_root), read_limit=args.read_limit,
                 peek_lines=args.peek_lines)
    print(json.dumps({"repository": str(Path(args.repo).expanduser().resolve()),
                      "model": {"requested": client.model, "base_url": client.base_url,
                                "key_available": True},
                      "summary": result["summary"],
                      "artifacts": str(output_root / args.run_id)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
