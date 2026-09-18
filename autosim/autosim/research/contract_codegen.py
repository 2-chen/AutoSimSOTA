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


#: The fields a task contract is made of. One function each, generated and checked
#: separately: a fifteen-field function is one thing to get right and fourteen ways to get it
#: wrong, and a failure in any of them rejects all of it. Split, a wrong field costs one
#: regeneration and the other fourteen are already frozen.
FIELD_NAMES = ("name", "env_id", "setting", "max_episode_steps", "state_dim", "action_dim",
               "cameras", "camera_shapes", "control_parts", "recorded_fps", "instruction",
               "event_families", "roles", "correction_supported", "expert_adapter")

#: Easy to hard, because a field that passes becomes an anchor for the ones after it. The
#: ordering is a curriculum, not a difficulty model: a field read straight off the document
#: is a place to start, and a field assembled from several places is not.
FIELD_ORDER = ("name", "setting", "env_id", "max_episode_steps", "correction_supported",
               "expert_adapter", "recorded_fps", "instruction", "control_parts", "cameras",
               "camera_shapes", "action_dim", "state_dim", "roles", "event_families")


def function_name(field: str) -> str:
    return f"field_{field}"


SYSTEM = (
    "You write one small pure Python function that reads a benchmark's configuration "
    "documents and returns a single value from them -- one field of a task's contract.\n\n"
    "You will be shown the shape of the documents (asked for, in pieces), the field you are "
    "responsible for, worked examples of the answer, and the tasks whose answers you are not "
    "shown. Your function is run against all of them and must reproduce the value exactly "
    "everywhere, including where you were not shown the answer.\n\n"
    "Constraints, enforced by a validator that rejects the draft outright:\n"
    "* exactly one function, named for the field, taking one positional argument `d`\n"
    "* no imports, no decorators, no nested functions, no recursion, no exception handling\n"
    "* no attribute access of any kind: `d[\"key\"]`, never `d.get(\"key\")`, and never a "
    "method on a value -- no .get(), .values(), .keys(), .startswith(), .append()\n"
    "* to collect values use `xs = xs + [v]` and `d[k] = d[k] + [v]`; there is no append, "
    "and this is the idiom most often missed\n"
    "* the only callable names are: int, float, len, sum, min, max, range, enumerate, zip, "
    "abs, list, dict, tuple, set, sorted, bool, all, any\n"
    "* no f-strings, no lambda, no while\n"
    "* lists and dicts keep their document order -- do not sort unless the expected value is "
    "sorted\n\n"
    "Return only {\"source\": \"<the function>\", \"reasoning\": \"<one sentence>\"}."
)


PATHS_SYSTEM = (
    "You are deciding what to read. You are given the top of a benchmark's configuration "
    "documents and must name the subtrees that would let you write a reader for a task's "
    "contract. Return only one JSON object of the form {\"paths\": [\"<dotted path>\", ...]}, "
    "with between four and twelve entries, each naming a subtree by its dotted path as "
    "listed. Ask for what you need to find the values, not for the whole document."
)


def request_paths(client: Any, index: list[tuple[str, str]], *,
                  limit: int = 12) -> tuple[list[str], dict[str, Any]]:
    """One triage call: which subtrees the reader will need."""
    content, metadata = client.chat_with_metadata(
        PATHS_SYSTEM,
        json.dumps({"document_shape": [{"path": p, "type": k} for p, k in index],
                    "note": "name subtrees by their dotted path; deeper paths exist under "
                            "each of these"}, ensure_ascii=False),
        max_tokens=800, timeout=120, thinking="disabled")
    try:
        value = _extract_object(content)
        paths = [str(p) for p in (value.get("paths") or [])][:limit]
    except (ValueError, AttributeError):
        paths = []
    return paths, {"provider": metadata, "requested": paths}


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
        return f" (the call it refused was `{rendered}`; the only callable names are " \
               f"{sorted(BUILTINS)})"
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


def _merge(node: dict, value: Any, depth: int, max_depth: int) -> None:
    """Fold one value into a structural node, recording types and optional keys."""
    kind = ("null" if value is None else "bool" if isinstance(value, bool) else
            "int" if isinstance(value, int) else "float" if isinstance(value, float) else
            "str" if isinstance(value, str) else "list" if isinstance(value, list) else
            "dict" if isinstance(value, dict) else type(value).__name__)
    node.setdefault("types", set()).add(kind)
    if depth >= max_depth:
        return
    if isinstance(value, dict):
        node.setdefault("keys", {})
        for key in value:
            _merge(node["keys"].setdefault(key, {}), value[key], depth + 1, max_depth)
    elif isinstance(value, list) and value:
        node.setdefault("items", {})
        for item in value:
            _merge(node["items"], item, depth + 1, max_depth)


def _walk(node: dict, prefix: str, out: list[tuple[str, str]], depth: int,
          max_depth: int) -> None:
    kinds = sorted(node.get("types", ()))
    if prefix:
        out.append((prefix, kinds[0] if len(kinds) == 1 else "|".join(kinds)))
    if depth >= max_depth:
        return
    for key, child in sorted(node.get("keys", {}).items()):
        _walk(child, f"{prefix}.{key}" if prefix else key, out, depth + 1, max_depth)
    if "items" in node:
        _walk(node["items"], f"{prefix}[]", out, depth + 1, max_depth)


def structure_schema(documents: list[dict[str, Any]], *, max_depth: int = 12) -> dict[str, Any]:
    """The shape of the input documents, merged across examples into one navigable tree.

    Inferring structure from whole documents is the part of writing a reader that has
    nothing to do with the benchmark -- which keys exist, at what depth, of what type -- and
    a program does it perfectly where a model does it approximately.

    It is returned as a tree rather than a flattened listing because flattening does not
    compress: a benchmark's configuration is mostly machinery the contract never looks at
    (one action graph here is 1,874 of 3,123 paths), so a complete listing is as large as
    the documents it summarises. What compresses is *asking* -- the model names the subtrees
    it wants and gets those, which is how a person reads an unfamiliar config too.
    """
    root: dict[str, Any] = {}
    key_counts: dict[str, int] = {}
    for document in documents:
        _merge(root, document, 0, max_depth)
        for key in document:
            key_counts[key] = key_counts.get(key, 0) + 1
    return {"tree": root, "documents_examined": len(documents),
            "top_level_optional": sorted(k for k, n in key_counts.items()
                                         if n < len(documents))}


def index_of(schema: dict[str, Any], *, depth: int = 1) -> list[tuple[str, str]]:
    """The top of the tree: enough to choose what to ask for, not enough to drown in."""
    out: list[tuple[str, str]] = []
    _walk(schema["tree"], "", out, 0, depth)
    return out


def paths_under(schema: dict[str, Any], prefixes: list[str], *,
                depth: int = 3, limit: int = 400) -> dict[str, Any]:
    """The schema restricted to the requested subtrees, plus which requests found nothing.

    Naming a subtree that does not exist is reported rather than ignored: it is the
    difference between a model that has understood the shape and one that is guessing at
    key names, and it costs one line to say so.
    """
    found: list[tuple[str, str]] = []
    _walk(schema["tree"], "", found, 0, 64)
    by_path = dict(found)
    kept: dict[str, str] = {}
    missing: list[str] = []
    for prefix in prefixes:
        selected = {path: kind for path, kind in by_path.items()
                    if path == prefix or path.startswith(prefix + ".") or
                    path.startswith(prefix + "[]")}
        if not selected:
            missing.append(prefix)
            continue
        for path in sorted(selected)[:limit]:
            kept[path] = selected[path]
    return {"paths": kept, "not_found": missing}


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


def differential_reader(source: str, functions: list[str], cases: list[dict[str, Any]], *,
                        timeout: float = 60.0) -> dict[str, Any]:
    """Run a reader made of per-field functions and report every field that disagrees.

    Field granularity in the report is what makes a repair local: "cameras is wrong on three
    tasks" names one function to regenerate, where "the contract is wrong" names fifteen.
    """
    payload = json.dumps({"source": source, "functions": functions,
                          "cases": [case["input"] for case in cases]})
    runner = Path(__file__).with_name("contract_runner.py")
    try:
        result = subprocess.run([sys.executable, "-I", str(runner)], input=payload, text=True,
                                capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"the reader did not finish within {timeout}s",
                "rows": [], "field_failures": {name: ["did not finish"] for name in functions}}
    if result.returncode:
        return {"passed": False, "error": (result.stderr or "").strip()[-2000:],
                "rows": [], "field_failures": {}}
    produced = json.loads(result.stdout)
    rows, failures = [], {}
    # The expected contract is keyed by field; the runner returns function names. The
    # mapping is fixed by `function_name`, so it is derived rather than carried.
    pairs = [(name, name[len("field_"):]) for name in functions]
    for case, got in zip(cases, produced):
        expected, returned = case["expected"], got.get("returned") or {}
        errors = got.get("errors") or {}
        disagreements = []
        for name, field in pairs:
            if name in errors:
                # Attributed by the runner, so the blame lands on the function that raised.
                # Guessing leaves a broken field looking correct, which is how one got
                # accepted here.
                disagreements.append(f"{field}: raised {errors[name]}")
            elif returned.get(name) != expected.get(field):
                disagreements.append(f"{field}: expected {expected.get(field)!r}, "
                                     f"got {returned.get(name)!r}")
        for disagreement in disagreements:
            failures.setdefault(disagreement.split(":")[0], []).append(
                f"{case['task']}: {disagreement}")
        rows.append({"task": case["task"], "passed": not disagreements,
                     "disagreements": disagreements})
    field_of = {field: name for name, field in pairs}
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "rows": rows,
            "field_failures": {field_of[k]: v[:3] for k, v in failures.items()
                               if k in field_of}}


def generate_field(client: Any, field: str, *, shape: dict[str, Any],
                   worked: list[dict[str, Any]], blind: list[dict[str, Any]],
                   settled: list[str], attempts: int = 4) -> tuple[str | None, list[dict[str, Any]]]:
    """Write one field's function, with what is already correct held out as settled.

    ``settled`` is the anchor. Naming the fields that already pass does two things: it stops
    the model spending attention on them, and it makes the remaining failure the only thing
    in the request that is wrong -- which is what turns a rewrite into a repair.
    """
    user = json.dumps({
        "field": field,
        "function_name": function_name(field),
        "document_shape": shape.get("paths"),
        "document_shape_not_found": shape.get("not_found"),
        "declared_block_contains": ["setting", "family", "correction_supported",
                                    "expert_adapter", "event_families"],
        "worked_examples": worked,
        "inputs_without_answers": blind,
        "already_settled_do_not_change": settled,
    }, ensure_ascii=False, sort_keys=True)
    log: list[dict[str, Any]] = []
    for repair in range(attempts):
        current = user if repair == 0 else user + json.dumps({
            "rejected": log[-1]["error"],
            "instruction": f"Return only the corrected {function_name(field)}. Every other "
                           "field is already correct; change only this one."},
            ensure_ascii=False)
        content, metadata = client.chat_with_metadata(SYSTEM, current, max_tokens=3000,
                                                      timeout=180, thinking="disabled")
        try:
            payload = _extract_object(content)
            source = str(payload["source"])
            try:
                checked_function(source, function_name(field))
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"{exc}{_offending_call(source)}") from None
            log.append({"field": field, "attempt": repair + 1, "status": "accepted",
                        "reasoning": payload.get("reasoning"),
                        "response_sha256": object_digest(content)})
            return source, log
        except (ValueError, KeyError, SyntaxError, json.JSONDecodeError) as exc:
            log.append({"field": field, "attempt": repair + 1, "status": "rejected",
                        "error": redact(f"{type(exc).__name__}: {exc}"),
                        "response_sha256": object_digest(content)})
    return None, log


def generate_reader(client: Any, *, cases: list[dict[str, Any]], worked_tasks: list[str],
                    shape: dict[str, Any], fields: list[str] | None = None,
                    attempts: int = 4) -> dict[str, Any]:
    """Generate one function per field, in easy-to-hard order, holding what passes fixed."""
    order = [f for f in (fields or FIELD_ORDER) if f in FIELD_NAMES]
    by_task = {case["task"]: case for case in cases}
    worked = [by_task[t] for t in worked_tasks]
    blind = [case["input"] for case in cases if case["task"] not in worked_tasks]

    sources: dict[str, str] = {}
    log: list[dict[str, Any]] = []
    settled: list[str] = []
    for field in order:
        source, field_log = generate_field(
            client, field, shape=shape,
            worked=[{"input": c["input"], "expected": c["expected"][field]} for c in worked],
            blind=blind, settled=list(settled), attempts=attempts)
        log += field_log
        if source is None:
            continue
        trial = dict(sources)
        trial[field] = source
        report = differential_reader(_assemble(trial), [function_name(f) for f in trial],
                                     cases)
        if report["passed"] or field not in report["field_failures"]:
            sources[field] = source
            settled.append(field)
        else:
            log.append({"field": field, "attempt": 0, "status": "rejected",
                        "error": "generated but disagrees: " +
                                 " | ".join(report["field_failures"][field][:3])})
    assembled = _assemble(sources)
    names = [function_name(f) for f in sources]
    report = differential_reader(assembled, names, cases) if names else {"passed": False,
                                                                        "rows": []}
    return {"source": assembled, "functions": names, "fields": sorted(sources),
            "settled": settled, "report": report, "log": log}


def _assemble(sources: dict[str, str]) -> str:
    """The field functions as one module. Order is fixed so the source is reproducible."""
    return "\n\n".join(sources[field].rstrip() + "\n" for field in sorted(sources))


def _merge(node: dict, value: Any, depth: int, max_depth: int) -> None:
    """Fold one value into a structural node, recording types and optional keys."""
    kind = ("null" if value is None else "bool" if isinstance(value, bool) else
            "int" if isinstance(value, int) else "float" if isinstance(value, float) else
            "str" if isinstance(value, str) else "list" if isinstance(value, list) else
            "dict" if isinstance(value, dict) else type(value).__name__)
    node.setdefault("types", set()).add(kind)
    if depth >= max_depth:
        return
    if isinstance(value, dict):
        node.setdefault("keys", {})
        for key in value:
            _merge(node["keys"].setdefault(key, {}), value[key], depth + 1, max_depth)
    elif isinstance(value, list) and value:
        node.setdefault("items", {})
        for item in value:
            _merge(node["items"], item, depth + 1, max_depth)


def _walk(node: dict, prefix: str, out: list[tuple[str, str]], depth: int,
          max_depth: int) -> None:
    kinds = sorted(node.get("types", ()))
    if prefix:
        out.append((prefix, kinds[0] if len(kinds) == 1 else "|".join(kinds)))
    if depth >= max_depth:
        return
    for key, child in sorted(node.get("keys", {}).items()):
        _walk(child, f"{prefix}.{key}" if prefix else key, out, depth + 1, max_depth)
    if "items" in node:
        _walk(node["items"], f"{prefix}[]", out, depth + 1, max_depth)


def structure_schema(documents: list[dict[str, Any]], *, max_depth: int = 12) -> dict[str, Any]:
    """The shape of the input documents, merged across examples into one navigable tree.

    Inferring structure from whole documents is the part of writing a reader that has
    nothing to do with the benchmark -- which keys exist, at what depth, of what type -- and
    a program does it perfectly where a model does it approximately.

    It is returned as a tree rather than a flattened listing because flattening does not
    compress: a benchmark's configuration is mostly machinery the contract never looks at
    (one action graph here is 1,874 of 3,123 paths), so a complete listing is as large as
    the documents it summarises. What compresses is *asking* -- the model names the subtrees
    it wants and gets those, which is how a person reads an unfamiliar config too.
    """
    root: dict[str, Any] = {}
    key_counts: dict[str, int] = {}
    for document in documents:
        _merge(root, document, 0, max_depth)
        for key in document:
            key_counts[key] = key_counts.get(key, 0) + 1
    return {"tree": root, "documents_examined": len(documents),
            "top_level_optional": sorted(k for k, n in key_counts.items()
                                         if n < len(documents))}


def index_of(schema: dict[str, Any], *, depth: int = 1) -> list[tuple[str, str]]:
    """The top of the tree: enough to choose what to ask for, not enough to drown in."""
    out: list[tuple[str, str]] = []
    _walk(schema["tree"], "", out, 0, depth)
    return out


def paths_under(schema: dict[str, Any], prefixes: list[str], *,
                depth: int = 3, limit: int = 400) -> dict[str, Any]:
    """The schema restricted to the requested subtrees, plus which requests found nothing.

    Naming a subtree that does not exist is reported rather than ignored: it is the
    difference between a model that has understood the shape and one that is guessing at
    key names, and it costs one line to say so.
    """
    found: list[tuple[str, str]] = []
    _walk(schema["tree"], "", found, 0, 64)
    by_path = dict(found)
    kept: dict[str, str] = {}
    missing: list[str] = []
    for prefix in prefixes:
        selected = {path: kind for path, kind in by_path.items()
                    if path == prefix or path.startswith(prefix + ".") or
                    path.startswith(prefix + "[]")}
        if not selected:
            missing.append(prefix)
            continue
        for path in sorted(selected)[:limit]:
            kept[path] = selected[path]
    return {"paths": kept, "not_found": missing}


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


