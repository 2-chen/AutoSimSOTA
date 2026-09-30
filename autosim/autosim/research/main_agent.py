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
Read-only roles must not edit or execute. Never claim that a repair is verified merely because
you edited a file. Do not change the shared plan or select the next experiment yourself.
Native stage failures carry a stable evidence_id. Use read_evidence with that ID to inspect
the original log; an inaccessible relative path is not proof that the log is absent.

For a data-supply assignment, Resource owns discovery and Init/Fix owns connection repair.
Trace the collector's action provider and prerequisites, the native output/conversion, and
the trainer's actual resolved data root and sample reader. Distinguish unknown capability,
missing resources, a successful collection probe and verified training consumption. Propose
a bounded trusted-executor probe with source citations; a read-only report cannot certify
that collection or a loader probe ran. State whether new training data is protocol-allowed
and why it deserves an early experiment or should be deferred.

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
    if not isinstance(value, dict) or set(value) != {
            "objective", "hypotheses", "open_questions", "next_actions", "evidence_refs"}:
        raise ValueError("plan requires objective, hypotheses, open_questions, next_actions, "
                         "and evidence_refs, with no extra keys")
    if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 16000:
        raise ValueError("working plan exceeds 16000 bytes; keep source evidence in files")
    return {"objective": _text(value["objective"], "objective"),
            **{key: _items(value[key], key) for key in (
                "hypotheses", "open_questions", "next_actions", "evidence_refs")},
            "authority": "model_working_plan_not_verified_facts"}


def validate_task(arguments: dict[str, Any]) -> dict[str, str]:
    if set(arguments) != {"role", "task", "expected_result"}:
        raise ValueError("research_task requires exactly role, task, expected_result")
    role = arguments["role"]
    if not isinstance(role, str) or role not in TASK_ROLES:
        raise ValueError("research_task role is not supported")
    return {"role": role, "task": _text(arguments["task"], "task"),
            "expected_result": _text(arguments["expected_result"], "expected_result")}


def validate_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("research task must return an object")
    if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 16000:
        raise ValueError("research report exceeds 16000 bytes; cite evidence instead")
    return {"summary": _text(value.get("summary"), "summary"),
            **{key: _items(value.get(key), key) for key in (
                "findings", "uncertainties", "evidence_refs", "recommended_next_actions")},
            "authority": "model_report_not_execution_verification"}


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
