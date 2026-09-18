"""Let the system write the reader, instead of shipping one per benchmark.

A hand-written adapter contains two kinds of thing. Some of it is *facts* -- which profiles a
task offers, whether an expert exists, where the checkpoint lives -- and the scout already
derives those by reading. The rest is a *program*: walk a config document, sum a dimension
across control parts, collect camera resolutions. That is what keeps an adapter hand-written,
and it is what this module replaces.

The split it is built on:

* The **declaration** says which documents belong to a task and supplies what cannot be
  derived. Some things genuinely have no config equivalent -- a task's semantic family is a
  label, not a property -- and those are declared rather than guessed.
* The **generated function** turns parsed documents into the task contract. It receives
  parsed data and returns a dict; it reads no files and imports nothing.

That second restriction is not a style preference. `patch_validation.checked_function`
admits only a fixed node set, only eighteen arithmetic builtins, no imports, no recursion,
no nested functions. A function that passes it cannot open a file, import a module or reach
the network, which is why the kernel sandbox the repair path uses would add nothing here:
the allow-list is already the boundary. What it does not bound is time and memory, so the
differential run happens in a subprocess with a timeout.

What makes the replacement safe is that it is not trusted. The function is checked against
the frozen contracts in `tests/fixtures/robosyn_task_contracts.json` for every task, and
only an exact match on every task and every field counts. It is given one worked example and
must generalize to the rest, so a function that reproduces the example by special-casing it
fails on the others rather than passing on one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import object_digest, read_json, redact
from .patch_validation import checked_function

#: The name the generated function must have. `checked_function` requires exactly one
#: top-level function and refuses recursion, so the name is part of the contract.
FUNCTION = "task_contract"


SYSTEM = (
    "You write one pure Python function that reads a benchmark's own configuration "
    "documents and returns the contract of a task.\n\n"
    "You will be given the documents for several tasks: for one of them the expected output "
    "as well, and for the others only the input. Your function is run against all of them "
    "and must reproduce the expected output exactly, including the tasks whose answers you "
    "were not shown. A function that special-cases the example fails on the rest.\n\n"
    "Constraints on the code, enforced by a validator that will reject it outright:\n"
    "* exactly one function, named `task_contract`, taking one positional argument\n"
    "* no imports, no decorators, no nested functions, no recursion, no exception handling\n"
    "* only these builtins may be called: int, float, len, sum, min, max, range, enumerate, "
    "zip, abs, list, dict, tuple, set, sorted, bool, all, any\n"
    "* no attribute access of any kind: write d[\"key\"], never d.get(\"key\"), and never "
    "call a method on a value -- no .get(), .values(), .items(), .startswith(), .append(). "
    "This is the constraint most often violated; everything must be done with subscripts, "
    "comprehensions and the builtins listed above\n"
    "* no f-strings, no lambda, no while, no augmented assignment to an attribute\n"
    "* to collect values, use `xs = xs + [v]` and `d[k] = d[k] + [v]`. There is no append; "
    "this is the idiom most often missed, and three of eight drafts were refused for it\n"
    "* dictionaries and lists keep their document order -- do not sort unless the expected "
    "output is sorted\n\n"
    "Return the JSON object only: {\"source\": \"<the function>\", \"reasoning\": \"<what "
    "varies between tasks and how you handle it>\"}."
)


def document_roles(setting: str = "random") -> dict[str, str]:
    """Which documents a task's contract is read from, as role -> repository-relative path.

    A role name rather than a path, so the declaration can say where each document lives
    without the generated function needing to know the layout.
    """
    return {"gym": f"configs/{{task}}/{setting}/gym_config.json",
            "action": "configs/{task}/action_config.json"}


def documents_for(repo: Path, task: str, *, declared: dict[str, Any] | None = None,
                  setting: str = "random") -> dict[str, Any]:
    """The parsed documents for one task, keyed by role, plus its name and declarations."""
    documents: dict[str, Any] = {"task": task}
    for role, template in document_roles(setting).items():
        documents[role] = read_json(Path(repo) / template.format(task=task))
    documents["declared"] = dict(declared or {})
    return documents


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value))


def build_request(worked: list[tuple[dict[str, Any], dict[str, Any]]],
                  blind: list[dict[str, Any]], fields: list[str]) -> tuple[str, str]:
    """Some worked examples, several inputs without answers, and the output shape.

    More than one worked example because the shape of a document is not always visible from
    a single instance of it: a mapping keyed by one field while the document is keyed by
    another looks identical whether you index it by the right field or the wrong one, and
    only a second example shows the difference. One example states an answer; two state a
    rule.
    """
    user = json.dumps({
        "required_output_fields": fields,
        "worked_examples": [{"input": given, "expected_output": expected}
                            for given, expected in worked],
        "inputs_without_answers": blind,
        "note": "Your function must return every required field for every input, including "
                "the ones whose answers are not shown.",
    }, ensure_ascii=False, sort_keys=True)
    return SYSTEM, user


def _offending_call(source: str) -> str:
    """Name the construct the validator refused, so a retry can fix it.

    The validator's message is accurate and says nothing about *where*. A model told only
    "only registered numeric builtins may be called" rewrites the same call a different way;
    told it wrote `d.get("key")` it stops writing method calls at all.
    """
    import ast as _ast
    from .patch_validation import BUILTINS
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return ""
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        # Two ways to be refused and both need naming: a method call, which the node set
        # excludes outright, and a call to a name that is not one of the registered
        # builtins. Reporting only the first left the model rewriting `str(x)` unchanged.
        if isinstance(node.func, _ast.Name) and node.func.id in BUILTINS:
            continue
        try:
            rendered = _ast.unparse(node)[:120]
        except Exception:
            rendered = type(node.func).__name__
        return f" (the call it refused was `{rendered}`; the only callable names are "                f"{sorted(BUILTINS)})"
    return ""


def _extract_object(content: str) -> dict[str, Any]:
    text = content.strip()
    for opener in ("```json", "```"):
        if text.startswith(opener):
            text = text[len(opener):]
    text = text.removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the response")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not one JSON object")
    return value


def generate(client: Any, *, worked: list[tuple[dict[str, Any], dict[str, Any]]],
             blind: list[dict[str, Any]], fields: list[str],
             verify: Any = None, attempts: int = 4) -> tuple[str | None, list[dict[str, Any]]]:
    """Ask for the function until it both runs and passes the differential.

    Two gates, and they reject different things. The validator rejects code that is not
    allowed to run at all. The differential rejects code that runs and is wrong -- which is
    the more interesting failure and the more common one: a function that fits the worked
    example and is wrong about the other nine is syntactically perfect.

    ``verify`` is what makes that second gate teach. Its disagreements are the error the
    model is given, naming the task and the field, so a retry corrects a specific mistake
    rather than rewriting everything and hoping.
    """
    _, user = build_request(worked, blind, fields)
    log: list[dict[str, Any]] = []
    for repair in range(attempts):
        current = user if repair == 0 else user + json.dumps({
            "rejected": log[-1]["error"],
            "instruction": "Return the corrected object. The code must satisfy every "
                           "constraint in the system message and reproduce every field. "
                           "Do not sort unless the expected output is sorted, and do not "
                           "re-derive anything the declared block already answers."},
            ensure_ascii=False)
        content, metadata = client.chat_with_metadata(SYSTEM, current, max_tokens=6000,
                                                      timeout=240, thinking="disabled")
        try:
            payload = _extract_object(content)
            source = str(payload["source"])
            try:
                checked_function(source, FUNCTION)
            except ValueError as exc:
                raise ValueError(f"{exc}{_offending_call(source)}") from None
            if verify is not None:
                report = verify(source)
                if not report["passed"]:
                    raise ValueError("the function runs but disagrees with the expected "
                                     "output: " + _disagreement_summary(report))
            log.append({"attempt": repair + 1, "status": "accepted",
                        "reasoning": payload.get("reasoning"),
                        "response_sha256": object_digest(content)})
            return source, log
        except (ValueError, KeyError, SyntaxError, json.JSONDecodeError) as exc:
            # SyntaxError is a candidate failure like any other: a function that does not
            # parse is a rejected draft, not a crash. Leaving it out made the one failure
            # the loop exists to repair the only one it could not survive.
            log.append({"attempt": repair + 1, "status": "rejected",
                        "error": redact(f"{type(exc).__name__}: {exc}"),
                        "response_sha256": object_digest(content)})
    # Returning rather than raising: the attempts are the record of what the model tried
    # and why each was refused, and a caller that only receives an exception loses the one
    # thing that says whether the next attempt should change the prompt or the model.
    return None, log


def _disagreement_summary(report: dict[str, Any], limit: int = 6) -> str:
    """The first few concrete disagreements, so a retry knows what to change."""
    lines = []
    for row in report.get("rows", []):
        if row["passed"]:
            continue
        lines.append(f"{row['task']}: " + "; ".join(row["disagreements"][:2]))
        if len(lines) >= limit:
            break
    return " | ".join(lines) if lines else (report.get("error") or "no rows")


def differential(source: str, cases: list[dict[str, Any]], *,
                 timeout: float = 60.0) -> dict[str, Any]:
    """Run the function against the frozen contracts and report every disagreement."""
    checked_function(source, FUNCTION)
    payload = json.dumps({"source": source, "cases": [case["input"] for case in cases]})
    runner = Path(__file__).with_name("contract_runner.py")
    result = subprocess.run([sys.executable, "-I", str(runner)],
                            input=payload, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        return {"passed": False, "error": (result.stderr or "").strip()[-2000:], "rows": []}
    produced = json.loads(result.stdout)
    rows = []
    for case, got in zip(cases, produced):
        expected = case["expected"]
        returned = got.get("returned")
        disagreements = ([f"raised {got['raised']}"] if "raised" in got else
                         [f"{field}: expected {expected.get(field)!r}, got "
                          f"{(returned or {}).get(field)!r}"
                          for field in sorted(set(expected) | set(returned or {}))
                          if (returned or {}).get(field) != expected.get(field)])
        rows.append({"task": case["task"], "passed": not disagreements,
                     "disagreements": disagreements})
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "rows": rows}


def gold_cases(repo: Path, oracle: dict[str, Any], *,
               declared: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Every task as (input documents, expected contract) -- the acceptance criterion."""
    return [{"task": task,
             "input": documents_for(repo, task, declared=(declared or {}).get(task)),
             "expected": expected}
            for task, expected in sorted(oracle["tasks"].items())]


def oracle(path: Path) -> dict[str, Any]:
    return read_json(path)


