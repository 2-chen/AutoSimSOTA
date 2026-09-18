"""Find how a benchmark runs, so the system does not have to be told.

The execution layer was one hand-written backend per benchmark: a file naming RoboSyn's
trainer, its evaluator, its collector and its converter, and a second file naming another
benchmark's. That is the same shape the adapters had before they were derived, and it has
the same cost -- a benchmark nobody has onboarded cannot run at all.

This asks the same two questions the rest of the system asks, in the same order. A survey
reports what is there; the model chooses what to read; the model answers with entry points
and invocations; and every path it names is checked against the filesystem. What it cannot
do is check that an entry point is the *right* one -- a repository with six policy families
has six trainers and all six exist. Settling that needs the stage to actually run, which is
the declared-to-verified ladder's job and not a static check's.

The separation matters for reading the output: `entrypoint` is a claim about this
repository, `available` is a claim about the benchmark's capability, and neither is
evidence until something runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import atomic_json, now, object_digest, read_json, redact
from .skills import skills_reference
from .survey import peek_many, summarise, survey


#: The stages a research loop needs, in the system's own words. A benchmark that cannot do
#: one of these says so; it is not a gap for the system to paper over. Which stages exist is
#: the same question `declaration.CAPABILITIES` asks, asked again here at the level of *how*
#: rather than *whether* -- a benchmark can declare it trains and still be unclear where.
STAGES: dict[str, str] = {
    "prepare_data": "turn the benchmark's shipped demonstrations into the form its trainer "
                    "reads. Not every benchmark has this step; one whose data is already in "
                    "that form does not.",
    "train": "fit a policy on demonstrations and write a checkpoint",
    "evaluate": "score a policy on the benchmark's own initial states and its own success "
                "predicate",
    "collect": "produce new trajectories. If the benchmark cannot produce any without a "
               "person, report that here rather than naming a script that needs one.",
}

SYSTEM = (
    "You are reading a benchmark repository to find how each stage of a research loop is "
    "invoked. For every stage listed, name the entry point and how it is called: the file to "
    "run, the arguments that matter, and what file appears under the output directory when "
    "it succeeds -- a path or glob, not a description of one, because that is what will be "
    "looked for. If a stage "
    "has no entry point in this repository, say so and say why -- a missing stage is a "
    "finding, not a failure, and a stage you could not find is a different finding from a "
    "stage the benchmark does not have.\n\n"
    "Name only files you have seen in the survey, the excerpts or the file contents, and "
    "give paths relative to the repository root. A file belonging to a different project "
    "that happens to be vendored inside this one is not this repository's entry point.\n\n"
    "Every invocation needs values the caller will supply, and some need values only this "
    "benchmark knows -- which policy implementation, which configuration, which dataset "
    "identifier. Declare those in `parameters`: one entry per value the invocation needs that "
    "is not a path the caller passes in, with the value you found and the evidence for it. A "
    "parameter whose value you inferred rather than read must say what you inferred it from.\n\n"
    "Return one JSON object of the form {\"stages\": {\"<stage>\": {\"available\": true or "
    "false, \"entrypoint\": \"<path>\", \"invocation\": \"<how it is called>\", \"artifact\": "
    "\"<a path or glob, relative to the output directory, that exists only on success>\", "
    "\"level\": \"entry point, or the path this wraps\", "
    "\"parameters\": [{\"name\": \"<as the invocation spells it>\", \"value\": \"<the value "
    "for this repository>\", \"evidence\": \"<where that value comes from>\"}], "
    "\"why\": \"<when unavailable>\"}}, \"reasoning\": \"<how you found them>\"}."
)

TRIAGE_SYSTEM = (
    "You are deciding what to read. A benchmark repository has been surveyed; the survey "
    "reports file names, sizes, and the signals found in each source, but not their "
    "contents. Return one JSON object {\"files_to_read\": [\"<path>\", ...]} naming up to "
    "twelve repository-relative files that would show how each of these stages is invoked: "
    "prepare_data, train, evaluate, collect. Prefer a file that runs a stage over one that "
    "merely mentions it, prefer this repository's own code over third-party code vendored "
    "inside it, and prefer documentation that shows an invocation. Do not ask for a file "
    "only to confirm its name."
)


#: What a stage function may read. The system's vocabulary, not the benchmark's: a stage is
#: told what it is being asked to do, and it says how this particular benchmark expresses
#: that. A benchmark needing something outside this list cannot be driven, which is a real
#: limit and is reported as one rather than papered over with a free-form command string.
INPUT_VOCABULARY: dict[str, str] = {
    "python": "the interpreter to run with",
    "repo": "absolute path to the benchmark checkout",
    "task": "the task name this stage is for",
    "dataset": "absolute path to training data, when the stage consumes it",
    "checkpoint": "absolute path to a policy checkpoint, when the stage consumes one",
    "output": "absolute directory the stage should write into",
    "steps": "training steps, as an int",
    "episodes": "evaluation episodes, as an int",
    "seed": "the seed, as an int",
    "device": "the device string this stage should use",
    "extra": "a dict of anything the caller was told to pass through; may be empty",
}

ARGV_SYSTEM = (
    "You write one pure Python function that returns the command line for one stage of a "
    "research loop on a benchmark you have been shown.\n\n"
    "The function takes one argument `i`, a dict with the keys listed in "
    "`inputs_available`, and returns a list of strings: the argv, starting with the program "
    "to run. Build it from the entry point and the invocation you were given, using the "
    "arguments that invocation shows.\n\n"
    "Constraints, enforced by a validator that rejects the draft outright:\n"
    "* exactly one function, named for the stage, taking one positional argument\n"
    "* no imports, no decorators, no nested functions, no recursion, no exception handling\n"
    "* no attribute access or method calls of any kind; subscripts and comprehensions only\n"
    "* to collect values use `xs = xs + [v]`; there is no append\n"
    "* the only callable names are int, float, str, len, sum, min, max, range, enumerate, "
    "zip, abs, list, dict, tuple, set, sorted, bool, all, any\n"
    "* no f-strings; build strings with + and str()\n\n"
    "Every element of the returned list must be a string. A value the caller did not supply "
    "must not be invented: if the entry point needs something outside the vocabulary, say so "
    "in the reasoning rather than guessing it.\n\n"
    "Return only {\"source\": \"<the function>\", \"reasoning\": \"<one or two "
    "sentences, including anything the vocabulary could not express>\"}."
)


def error_excerpt(output: str, *, limit: int = 1200) -> str:
    """The part of a program's output that says what went wrong.

    Not the tail. A program that prints a version banner on import and then refuses its
    arguments has its cause *above* output that looks more recent, and a caller that reports
    the last few lines sends the model a header instead of an error -- which is what
    happened, four attempts in a row, each regenerating the same wrong command from the same
    uninformative feedback.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return "(no output)"
    markers = ("error:", "usage:", "Traceback (most recent call last)", "Error:", "Exception:",
               "cannot", "No such file", "not found", "unrecognized", "expected one argument")
    hits = [index for index, line in enumerate(lines)
            if any(marker in line for marker in markers)
            or line.lstrip().startswith(("File \"", "raise "))]
    if hits:
        start = max(0, hits[0] - 1)
        return "\n".join(lines[start:start + 14])[:limit]
    return "\n".join(lines[-8:])[:limit]


def argv_function_name(stage: str) -> str:
    return f"stage_argv_{stage}"


def generate_argv(client: Any, stage: str, *, entrypoint: str, invocation: str,
                  repository_files: list[str],
                  declared_parameters: dict[str, str] | None = None,
                  verify: Any = None, attempts: int = 3
                  ) -> tuple[str | None, list[dict[str, Any]]]:
    """Write the function that turns the system's inputs into this benchmark's command.

    Generated rather than templated because invocations differ in kind rather than in
    spelling: one benchmark takes flags, another a config file and a name, a third a shell
    wrapper with positional arguments. A template language general enough to cover them
    would be a programming language, and generating the function is the smaller thing.
    """
    from .patch_validation import checked_function
    name = argv_function_name(stage)
    user = json.dumps({
        "stage": stage,
        "function_name": name,
        "entrypoint": entrypoint,
        "invocation_as_written_in_the_repository": invocation,
        "inputs_available": INPUT_VOCABULARY,
        "declared_parameters": declared_parameters or {},
        "files_in_the_repository": repository_files[:200],
        "note": "Values in `declared_parameters` are already settled for this repository and "
                "must be used as given; do not ask the caller for them. Return the argv only.",
    }, ensure_ascii=False, sort_keys=True)
    log: list[dict[str, Any]] = []
    for repair in range(attempts):
        current = user if repair == 0 else user + json.dumps({
            "rejected": log[-1]["error"],
            "instruction": f"Return only the corrected {name}, satisfying every constraint. "
                           "If the error is from running the command, the program's own "
                           "usage line is the authority on what it accepts -- build the "
                           "argv to match it exactly."},
            ensure_ascii=False)
        content, metadata = client.chat_with_metadata(ARGV_SYSTEM, current, max_tokens=3000,
                                                      timeout=180, thinking="disabled")
        try:
            payload = _object(content)
            source = str(payload["source"])
            try:
                checked_function(source, name)
            except (ValueError, SyntaxError) as exc:
                from .contract_codegen import _offending_call
                raise ValueError(f"{exc}{_offending_call(source)}") from None
            if verify is not None:
                # The command is run before it is accepted. Nothing static decides whether
                # a flag exists on a program: the generated argv passed every check and was
                # refused by the program itself, printing the usage line that says so.
                outcome = verify(source)
                if not outcome.get("ok"):
                    raise ValueError("the command did not run. The program said:\n"
                                     + error_excerpt(str(outcome.get("error") or "")))
            log.append({"stage": stage, "attempt": repair + 1, "status": "accepted",
                        "reasoning": payload.get("reasoning"),
                        "response_sha256": object_digest(content)})
            return source, log
        except (ValueError, KeyError, SyntaxError, json.JSONDecodeError) as exc:
            log.append({"stage": stage, "attempt": repair + 1, "status": "rejected",
                        "error": redact(f"{type(exc).__name__}: {exc}"),
                        "response_sha256": object_digest(content)})
    return None, log


def problems_in(value: dict[str, Any], repo: Path) -> list[str]:
    """Faults in a derived answer, as messages. Empty means well-formed, not correct.

    Two questions, both answerable without running anything: is the answer complete enough
    to act on, and do the paths it names exist? Neither settles whether the entry point is
    the right one, and the report does not pretend otherwise.
    """
    if not isinstance(value, dict):
        return ["answer must be one JSON object"]
    stages = value.get("stages")
    if not isinstance(stages, dict):
        return ["stages must be an object"]
    problems: list[str] = []
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        problems.append(f"stages has no such entry: {unknown}; answer exactly {sorted(STAGES)}")
    for name in STAGES:
        row = stages.get(name)
        if not isinstance(row, dict):
            problems.append(f"stages.{name} must be answered, including as unavailable")
            continue
        if not isinstance(row.get("available"), bool):
            problems.append(f"stages.{name}.available must be true or false")
            continue
        if not row["available"]:
            if not str(row.get("why", "")).strip():
                problems.append(f"stages.{name} is reported unavailable and must say why")
            continue
        entrypoint = str(row.get("entrypoint", "")).strip()
        if not entrypoint:
            problems.append(f"stages.{name} claims to be available and names no entry point")
            continue
        target = repo / entrypoint
        if not target.exists():
            problems.append(f"stages.{name}.entrypoint does not exist: {entrypoint}")
        elif target.is_dir():
            # A directory is where to look, not what to run. Reported rather than rejected:
            # it is a weaker answer, and the difference is worth keeping visible.
            problems.append(f"stages.{name}.entrypoint is a directory, not a file to run: "
                            f"{entrypoint}")
        for key in ("invocation",):
            if not str(row.get(key, "")).strip():
                problems.append(f"stages.{name}.{key} is required when available")
        artifact = str(row.get("artifact", "")).strip()
        if not artifact:
            problems.append(f"stages.{name}.artifact is required when available")
        elif " " in artifact:
            # A sentence cannot be looked for. The artifact is what the execution check
            # waits for, so it has to be a path or a glob relative to the stage's output,
            # not a description of one.
            problems.append(f"stages.{name}.artifact must be a path or glob relative to the "
                            f"output directory, not a sentence: {artifact[:60]!r}")
        parameters = row.get("parameters")
        if parameters is not None and not isinstance(parameters, list):
            problems.append(f"stages.{name}.parameters must be a list when present")
            continue
        for index, parameter in enumerate(parameters or []):
            if not isinstance(parameter, dict):
                problems.append(f"stages.{name}.parameters[{index}] must be an object")
                continue
            for key in ("name", "value", "evidence"):
                if not str(parameter.get(key, "")).strip():
                    # A value without evidence is the guess this field exists to expose: the
                    # disambiguation between several implementations is exactly a claim a
                    # repository can be asked about, and the answer shown to be checked.
                    problems.append(f"stages.{name}.parameters[{index}].{key} is required")
    return problems


def _object(content: str) -> dict[str, Any]:
    text = content.strip()
    for opener in ("```json", "```"):
        if text.startswith(opener):
            text = text[len(opener):]
    text = text.removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the response")
    return json.loads(text[start:end + 1])


def triage(report: dict[str, Any], client: Any, *, limit: int = 12) -> dict[str, Any]:
    """Which files to read. A reader that has seen only a file listing is guessing."""
    content, metadata = client.chat_with_metadata(
        TRIAGE_SYSTEM,
        json.dumps({"stages_to_find": STAGES, "survey": summarise(report)},
                   ensure_ascii=False),
        max_tokens=1200, timeout=120, thinking="disabled")
    try:
        chosen = [str(p) for p in (_object(content).get("files_to_read") or [])][:limit]
    except (ValueError, AttributeError):
        chosen = []
    return {"files_to_read": chosen, "provider": metadata}


def derive(repo: Path, *, client: Any, report: dict[str, Any] | None = None,
           read_limit: int = 12, peek_lines: int = 70,
           attempts: int = 3) -> dict[str, Any]:
    """Survey, choose what to read, answer with entry points, and check what can be checked.

    The loop is the one the rest of the system uses, and for the same reason: a fault the
    filesystem can see is worth a retry, and the retry is worth a specific message. It
    cannot fix a wrong-but-real entry point, and does not claim to.
    """
    repo = Path(repo).expanduser().resolve()
    report = report or survey(repo)
    triage_result = triage(report, client, limit=read_limit)
    files = peek_many([str(repo / p) for p in triage_result["files_to_read"]],
                      lines=peek_lines, limit=read_limit)
    for row, path in zip(files, triage_result["files_to_read"]):
        row["requested_as"] = path
    excerpts = [{k: row[k] for k in ("requested_as", "readable", "head") if k in row}
                for row in files]

    payload = json.dumps({"stages_to_find": STAGES, "survey": summarise(report),
                          "file_contents": excerpts,
                          "method_library": skills_reference()}, ensure_ascii=False)
    log: list[dict[str, Any]] = []
    for repair in range(attempts):
        current = payload if repair == 0 else payload + json.dumps({
            "rejected": log[-1]["error"],
            "instruction": "Return the complete corrected object. Name only files that "
                           "exist in this repository."}, ensure_ascii=False)
        content, metadata = client.chat_with_metadata(SYSTEM, current, max_tokens=4000,
                                                      timeout=240, thinking="disabled")
        try:
            value = _object(content)
            faults = problems_in(value, repo)
            if faults:
                raise ValueError("; ".join(faults[:6]))
            log.append({"attempt": repair + 1, "status": "accepted",
                        "reasoning": value.get("reasoning"),
                        "response_sha256": object_digest(content)})
            return {"stages": value["stages"], "reasoning": value.get("reasoning"),
                    "read": triage_result["files_to_read"], "attempts": log,
                    # Carried on the answer, not only in the prose around it. Every path is
                    # checked and no entry point is: a reader who takes this for settled has
                    # been told otherwise here, in the object they are reading.
                    "evidence": "static_only",
                    "unverified": "no stage has been executed; an entry point that exists "
                                  "and is not the right one passes every check here",
                    "provider": metadata}
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            log.append({"attempt": repair + 1, "status": "rejected",
                        "error": redact(f"{type(exc).__name__}: {exc}"),
                        "response_sha256": object_digest(content)})
    return {"stages": None, "reasoning": None, "read": triage_result["files_to_read"],
            "attempts": log, "evidence": "static_only", "provider": None}


def run(repo: Path, *, client: Any, output: Path) -> dict[str, Any]:
    """Derive and record, so a failure is legible rather than a stack trace."""
    result = derive(repo, client=client)
    output = Path(output)
    atomic_json(output / "execution.json",
                {"schema_version": 1, "created_at": now(), "repo": str(repo), **result})
    return result
