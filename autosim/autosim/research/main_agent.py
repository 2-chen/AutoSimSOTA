"""Bounded working memory and open-ended task contracts for the main controller.

These are model-authored hypotheses and handoffs, never execution receipts or scores.
The controller retains the plan; specialists cannot silently replace it.
"""

from __future__ import annotations

import json
from typing import Any

from .prompt_library import load_prompt

READ_ONLY_ROLES = frozenset({"resource", "data", "objective", "monitor", "ideator", "supervisor"})
TASK_ROLES = READ_ONLY_ROLES | {"scheduler", "init", "fix"}

TASK_SYSTEM = load_prompt("specialist")


def task_system_for_role(role: str) -> str:
    """Share the specialist contract, adding the dedicated data responsibility."""
    if role not in TASK_ROLES:
        raise ValueError("research_task role is not supported")
    return TASK_SYSTEM + "\n\n" + load_prompt(role) if role in {"data", "resource"} else TASK_SYSTEM


def retain_handoff(memory: dict[str, Any], handoff: dict[str, Any]) -> dict[str, Any]:
    """Keep data ownership visible even after unrelated reports roll out of the window.

    The retained report remains advisory and may be stale; it grants no job, data deletion
    or scoring authority. Native receipts must be inspected again before execution.
    """
    result = {**memory, "handoffs": [*(memory.get("handoffs") or []), handoff][-12:]}
    if handoff.get("role") == "data":
        result["data_agent_handoff"] = handoff
    if handoff.get("role") == "resource":
        result["resource_agent_handoff"] = handoff
    return result


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
    if not isinstance(value, dict):
        raise ValueError('plan must be an object')
    missing = required - set(value)
    extra = set(value) - required - {"data_strategy", "budget_strategy"}
    if missing or extra:
        raise ValueError(f'plan: missing fields={sorted(missing)}; unexpected fields={sorted(extra)}; '
                         'authority is server-owned metadata; submit main_agent.writable_plan instead')
    size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    if size > 16000:
        raise ValueError(f"working plan exceeds 16000 bytes: actual={size}; keep source evidence in files")
    result = {"objective": _text(value["objective"], "objective"),
            **{key: _items(value[key], key) for key in (
                "hypotheses", "open_questions", "next_actions", "evidence_refs")},
            "authority": "model_working_plan_not_verified_facts"}
    if "data_strategy" in value:
        result["data_strategy"] = validate_data_strategy(value["data_strategy"])
    if "budget_strategy" in value:
        result["budget_strategy"] = validate_budget_strategy(value["budget_strategy"])
    return result


def validate_budget_strategy(value: Any) -> dict[str, Any]:
    """Resource-authored guidance, never permission to change hard ceilings."""
    keys = {"why", "allocations", "evidence_refs", "revisit_condition", "next_action"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("budget_strategy requires why, allocations, evidence_refs, revisit_condition and next_action")
    if len(json.dumps(value, ensure_ascii=False).encode()) > 6000:
        raise ValueError("budget_strategy exceeds 6000 bytes")
    return {"why": _text(value["why"], "why", 1500),
            "revisit_condition": _text(value["revisit_condition"], "revisit_condition", 1000),
            "next_action": _text(value["next_action"], "next_action", 1000),
            "allocations": _items(value["allocations"], "allocations"),
            "evidence_refs": _items(value["evidence_refs"], "evidence_refs"),
            "authority": "resource_allocation_advice_not_hard_budget_authorization"}


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
        if role in {"data", "resource"} and arguments["mode"] != "inspect":
            raise ValueError(f"{'Data' if role == 'data' else 'Resource'} Agent is read-only; request Init/Fix repairs through Scheduler")
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
    result = {"summary": _text(value.get("summary"), "summary"),
            **{key: _items(value.get(key), key) for key in (
                "findings", "uncertainties", "evidence_refs", "recommended_next_actions")},
            "authority": "model_report_not_execution_verification"}
    if 'operation_proposals' in value:
        import re
        proposals = value['operation_proposals']
        if not isinstance(proposals, list) or len(proposals)>3:
            raise ValueError('operation_proposals must contain at most three proposals')
        checked = []
        for proposal in proposals:
            if not isinstance(proposal, dict) or set(proposal) != {'do','arguments','why','evidence_refs','expected_postcondition'}:
                raise ValueError('operation proposal requires do, arguments, why, evidence_refs, expected_postcondition')
            if not isinstance(proposal['do'], str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,79}', proposal['do']):
                raise ValueError('operation proposal has an invalid operation name')
            if not isinstance(proposal['arguments'], dict) or len(json.dumps(proposal['arguments'], ensure_ascii=False).encode())>6000:
                raise ValueError('operation proposal arguments must be a bounded object')
            checked.append({'do':proposal['do'], 'arguments':proposal['arguments'],
                'why':_text(proposal['why'],'why',1000),
                'expected_postcondition':_text(proposal['expected_postcondition'],'expected_postcondition',1000),
                'evidence_refs':_items(proposal['evidence_refs'],'evidence_refs')})
        result['operation_proposals'] = checked
    return result


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


def _proposal_view(proposals: list, *, compact: bool) -> list:
    from .common import object_digest
    rows = []
    for proposal in proposals[:1 if compact else 3]:
        if len(json.dumps(proposal,ensure_ascii=False).encode())<=1800:
            rows.append(proposal)
        else:
            rows.append({k:proposal.get(k) for k in ('do','why','expected_postcondition')})
            rows[-1].update(arguments_omitted=True,
                arguments_sha256=object_digest(proposal.get('arguments')),
                full_handoff_ref='run_state.json#phases.main_agent.memory',
                guidance='read exact retained handoff before deciding; do not reconstruct omitted arguments')
    return rows


def _handoff_view(row: dict[str, Any], *, compact: bool = False) -> dict[str, Any]:
    report = row.get("report") or {}
    return {
        "task_id": row.get("task_id"), "role": row.get("role"),
        "at": row.get("at"), "stale": bool(row.get("stale")),
        "input_state_revision": row.get("input_state_revision"),
        "task": str(row.get("task") or "")[:400 if compact else 600],
        "expected_result": str(row.get("expected_result") or "")[:400 if compact else 600],
        "runtime_evidence_refs": row.get("runtime_evidence_refs") or [],
        "report": {"summary": str(report.get("summary") or "")[:600 if compact else 800],
                   "authority": "model_report_not_execution_verification",
                   **{key: [str(item)[:240 if compact else 400]
                            for item in (report.get(key) or [])[:3 if compact else 5]]
                      for key in ("findings", "uncertainties", "evidence_refs",
                                  "recommended_next_actions")},
                   **({'operation_proposals': _proposal_view(report['operation_proposals'],compact=compact)}
                      if report.get('operation_proposals') else {})},
    }


def memory_view(memory: dict[str, Any]) -> dict[str, Any]:
    """A bounded prompt projection; the event journal retains the complete history."""
    handoffs = memory.get("handoffs") or []
    recent = [_handoff_view(row) for row in handoffs[-3:]]
    data_handoff = memory.get("data_agent_handoff") or next(
        (row for row in reversed(handoffs) if row.get("role") == "data"), None)
    data_view = _handoff_view(data_handoff, compact=True) if data_handoff else None
    resource_handoff = memory.get("resource_agent_handoff") or next(
        (row for row in reversed(handoffs) if row.get("role") == "resource"), None)
    resource_view = _handoff_view(resource_handoff, compact=True) if resource_handoff else None
    writable = {k:v for k,v in (memory.get('plan') or {}).items() if k != 'authority'}
    for key in ('data_strategy', 'budget_strategy'):
        if isinstance(writable.get(key), dict):
            writable[key] = {k:v for k,v in writable[key].items() if k != 'authority'}
    return {**{key: value for key, value in memory.items()
               if key not in {"handoffs", "data_agent_handoff", "resource_agent_handoff"}},
            "handoffs": recent, "retained_handoff_count": len(handoffs),
            "writable_plan": writable,
            **({"data_agent_handoff": data_view} if data_view else {}),
            **({"resource_agent_handoff": resource_view} if resource_view else {}),
            "full_memory_ref": "run_state.json#phases.main_agent.memory",
            "history_ref": "run_events.json#main_agent",
            "projection": "bounded_excerpt_not_complete_history"}
