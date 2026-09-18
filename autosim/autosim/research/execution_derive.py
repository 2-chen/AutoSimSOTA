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
    "run, the arguments that matter, and what appears on disk when it succeeds. If a stage "
    "has no entry point in this repository, say so and say why -- a missing stage is a "
    "finding, not a failure, and a stage you could not find is a different finding from a "
    "stage the benchmark does not have.\n\n"
    "Name only files you have seen in the survey, the excerpts or the file contents, and "
    "give paths relative to the repository root. A file belonging to a different project "
    "that happens to be vendored inside this one is not this repository's entry point.\n\n"
    "Return one JSON object of the form {\"stages\": {\"<stage>\": {\"available\": true or "
    "false, \"entrypoint\": \"<path>\", \"invocation\": \"<how it is called>\", \"artifact\": "
    "\"<what appears on success>\", \"level\": \"entry point, or the path this wraps\", "
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
        for key in ("invocation", "artifact"):
            if not str(row.get(key, "")).strip():
                problems.append(f"stages.{name}.{key} is required when available")
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
