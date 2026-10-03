"""Bounded working memory and open-ended task contracts for the main controller.

These are model-authored hypotheses and handoffs, never execution receipts or scores.
The controller retains the plan; specialists cannot silently replace it.
"""

from __future__ import annotations

import json
from typing import Any

READ_ONLY_ROLES = frozenset({"resource", "objective", "monitor", "ideator", "supervisor"})
TASK_ROLES = READ_ONLY_ROLES | {"scheduler", "init", "fix"}

TASK_SYSTEM = """Work on the main Scheduler's explicit research assignment.
Use your actual tools to inspect source, configuration and available evidence; do not guess
an interface from filenames. Follow producer-to-consumer connections when relevant (data
root -> loader, checkpoint -> policy loader, server -> client). The shared research context
is a snapshot, not permission and not guaranteed current after a tool action. Consult the
referenced evidence and distinguish observed facts, hypotheses, and missing prerequisites.

You may take multiple tool steps within this turn's budget. Editable roles may repair only
the isolated checkout before research begins and run bounded CPU diagnostics. Do not launch
training, collection, simulation evaluation, services or GPU work here: request a trusted
executor operation from the Scheduler. Never edit evaluation rules or authoritative records.
The live run environment/prefix and package cache are NOT writable through your diagnostic
tools. For environment installs/configuration, return an executable repair proposal to
Scheduler's build_the_environment(repair_proposal=...), linked to the sealed failure ID.
Do not repeatedly attempt prefix writes here. A diagnostic report is not an installation.
Read-only roles must not edit or execute. Never claim that a repair is verified merely because
you edited a file. Do not change the shared plan or select the next experiment yourself.
Native stage failures carry a stable evidence_id. Use read_evidence with that ID to inspect
the original log; an inaccessible relative path is not proof that the log is absent.
For policy/rollout/metric witnesses, state.native_identity_contract gives the executor's
exact field names and file/directory digest recipe. Follow that contract, not an invented
serialization; syntax or a CPU witness fixture does not prove native consumption.

For a data-supply assignment, Resource owns discovery and Init/Fix owns connection repair.
Use inspect_workspace_resources to enumerate actual explicit resource bindings; empty
checkout mount placeholders and gitignored Glob results are NOT missing-asset evidence.
Trace the collector's action provider and prerequisites, the native output/conversion, and
the trainer's actual resolved data root and sample reader. Distinguish unknown capability,
missing resources, a successful collection probe and verified training consumption. Propose
a bounded trusted-executor probe with source citations; a read-only report cannot certify
that collection or a loader probe ran. State whether new training data is protocol-allowed
and why it deserves an early experiment or should be deferred.
Distinguish native expert/planner, trajectory adaptation, success-filtered training policy
rollout and teleoperation. If only the recorder is missing, report the minimal source-backed
training-side bridge Init could implement, not "autonomous collection impossible". Label
possible versus implemented routes explicitly. Do not research optional families once a
safe next action is identified; return a compact handoff immediately.

Return one JSON object with summary (string), findings (list of strings),
uncertainties (list of strings), evidence_refs (list of relative references), and
recommended_next_actions (list of strings). This is a model report, not an accepted score.
Each list must contain at most 20 entries; the entire JSON object must be at most 16000
UTF-8 bytes. Group related citations by source instead of listing every line separately.
"""


def _text(value: Any, name: str, limit: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _items(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > 20:
        raise ValueError(f"{name} must be a list with at most 20 entries")
    return [_text(item, name, 1500) for item in value]


def validate_plan(value: Any) -> dict[str, Any]:
    required = {"objective", "hypotheses", "open_questions", "next_actions", "evidence_refs"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"data_strategy"}:
        raise ValueError("plan requires objective, hypotheses, open_questions, next_actions, "
                         "and evidence_refs; optional data_strategy")
    if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 16000:
        raise ValueError("working plan exceeds 16000 bytes; keep source evidence in files")
    result = {"objective": _text(value["objective"], "objective"),
            **{key: _items(value[key], key) for key in (
                "hypotheses", "open_questions", "next_actions", "evidence_refs")},
            "authority": "model_working_plan_not_verified_facts"}
    if "data_strategy" in value:
        result["data_strategy"] = validate_data_strategy(value["data_strategy"])
    return result


def validate_data_strategy(value: Any) -> dict[str, Any]:
    """A persistent Agent-authored acquisition plan, never collection authorization.

    Route/control choices remain model reasoning; native stage/axis/protocol validators
    decide what may actually execute. No benchmark names or collector recipes live here.
    """
    keys = {"route", "why", "targets", "producer_to_loader", "evidence_refs", "next_probe"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("data_strategy requires route, why, targets, producer_to_loader, evidence_refs and next_probe")
    if len(json.dumps(value, ensure_ascii=False).encode()) > 8000:
        raise ValueError("data_strategy exceeds 8000 bytes; cite source/evidence instead")
    return {"route": _text(value["route"], "route", 400),
            "why": _text(value["why"], "why", 1500),
            "next_probe": _text(value["next_probe"], "next_probe", 1500),
            **{key: _items(value[key], key) for key in (
                "targets", "producer_to_loader", "evidence_refs")},
            "authority": "model_data_strategy_not_collection_verification"}


def validate_task(arguments: dict[str, Any], *, allow_source_paths: bool = False) -> dict[str, Any]:
    required = {"role", "task", "expected_result"}
    optional = {"source_paths"} if allow_source_paths else {"mode", "timeout_seconds"}
    if not required <= set(arguments) or set(arguments) - required - optional:
        missing = sorted(required - set(arguments))
        unexpected = sorted(set(arguments) - required - optional)
        raise ValueError(
            "research_task requires role, task, expected_result; optional fields: "
            + ", ".join(sorted(optional)) + f"; missing={missing}; unexpected={unexpected}")
    role = arguments["role"]
    if not isinstance(role, str) or role not in TASK_ROLES:
        raise ValueError("research_task role is not supported")
    result = {"role": role, "task": _text(arguments["task"], "task"),
              "expected_result": _text(arguments["expected_result"], "expected_result")}
    if "mode" in arguments:
        if arguments["mode"] not in {"inspect", "edit"}:
            raise ValueError("research_task mode must be inspect or edit")
        result["mode"] = arguments["mode"]
    if "timeout_seconds" in arguments:
        import math
        value = arguments["timeout_seconds"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("research task window must be finite and positive")
        result["timeout_seconds"] = value
    if "source_paths" in arguments:
        from pathlib import Path
        paths = arguments["source_paths"]
        if not isinstance(paths, list) or not 1 <= len(paths) <= 64:
            raise ValueError("source_paths must contain 1..64 checkout-relative files/directories")
        for item in paths:
            if (not isinstance(item, str) or not item or len(item) > 512 or
                    Path(item).is_absolute() or ".." in Path(item).parts or "\\" in item):
                raise ValueError("unsafe reader source path")
        result["source_paths"] = list(dict.fromkeys(paths))
    return result


def validate_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("research task must return an object")
    if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 16000:
        raise ValueError("research report exceeds 16000 bytes; cite evidence instead")
    return {"summary": _text(value.get("summary"), "summary"),
            **{key: _items(value.get(key), key) for key in (
                "findings", "uncertainties", "evidence_refs", "recommended_next_actions")},
            "authority": "model_report_not_execution_verification"}


def receive_report(content: str, *, client, output, workspace, timeout: float,
                   metadata: dict | None = None) -> dict:
    """Recover only malformed delivery, not repeat the completed investigation."""
    from pathlib import Path
    import uuid
    from .common import atomic_text, atomic_json, sanitize_model_text
    from .evidence_store import capture_attempt_evidence
    from .execution_derive import _object
    try:
        return validate_report(_object(content))
    except (ValueError, TypeError, KeyError) as exc:
        identity = uuid.uuid4().hex
        output = Path(output)
        from .agent_runtime import _worker_evidence_root
        evidence_root = _worker_evidence_root(output)
        log = output / "report_handoffs" / (identity + ".log")
        atomic_text(log, sanitize_model_text(content, local_roots=(Path(workspace), output)))
        receipt_ref = (output / f"report_handoffs/{identity}.json").relative_to(evidence_root).as_posix()
        evidence = capture_attempt_evidence(evidence_root, attempt_id=identity, log=log,
            receipt_ref=receipt_ref, status="report_delivery_rejected", returncode=None,
            termination_reason="invalid_report_contract")
        atomic_json(evidence_root / receipt_ref, {**evidence, "reason": str(exc),
            "original_turn_id": (metadata or {}).get("turn_id"),
            "authority": "unaccepted_model_report"})
        corrected, _ = client.chat_with_metadata(
            "Repair ONLY the JSON delivery of an already completed investigation. "
            "Do not inspect, execute, edit, research or claim new evidence. Preserve uncertainty "
            "and evidence IDs. Return summary plus findings, uncertainties, evidence_refs, "
            "recommended_next_actions. Use short strings and <=8 entries per list; "
            "the ENTIRE JSON must be <=8000 UTF-8 bytes. Never infer recovery or a score.",
            json.dumps({"validation_error": str(exc), "original_report_evidence_id": identity,
                "original_report": sanitize_model_text(content[:128000],
                    local_roots=(Path(workspace), output)),
                "original_report_complete": len(content) <= 128000}, ensure_ascii=False),
            timeout=timeout, max_tokens=2000, read_only=True, auto_skills=False,
            include_research_context=False)
        report = validate_report(_object(corrected))
        report["delivery_recovery_evidence_ref"] = evidence["evidence_ref"]
        return report


def memory_view(memory: dict[str, Any]) -> dict[str, Any]:
    """A bounded prompt projection; the event journal retains the complete history."""
    handoffs = memory.get("handoffs") or []
    recent = []
    for row in handoffs[-3:]:
        report = row.get("report") or {}
        recent.append({
            "task_id": row.get("task_id"), "role": row.get("role"),
            "stale": bool(row.get("stale")),
            "input_state_revision": row.get("input_state_revision"),
            "task": str(row.get("task") or "")[:600],
            "expected_result": str(row.get("expected_result") or "")[:600],
            "runtime_evidence_refs": row.get("runtime_evidence_refs") or [],
            "report": {"summary": str(report.get("summary") or "")[:800],
                       "authority": "model_report_not_execution_verification",
                       **{key: [str(item)[:400] for item in (report.get(key) or [])[:5]]
                          for key in ("findings", "uncertainties", "evidence_refs",
                                      "recommended_next_actions")}},
        })
    return {**{key: value for key, value in memory.items() if key != "handoffs"},
            "handoffs": recent, "retained_handoff_count": len(handoffs),
            "full_memory_ref": "run_state.json#phases.main_agent.memory",
            "history_ref": "run_events.json#main_agent",
            "projection": "bounded_excerpt_not_complete_history"}
