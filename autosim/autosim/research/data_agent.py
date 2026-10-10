"""Explicit, bounded skill selection for delegated Data Agent turns.

No execution, data mutation or automatic lexical method selection happens here.
"""
from __future__ import annotations

import json
import time

from .execution_derive import _object
from .prompt_library import load_prompt
from .skills import skills_catalog, read_selected_skills


def select_methods(client, assignment: dict, facts: dict, *, timeout: float) -> dict:
    benchmark = (facts.get("benchmark") or {}).get("name") if isinstance(facts.get("benchmark"), dict) else None
    catalog = skills_catalog(query=assignment["task"], benchmark=benchmark)
    role = assignment.get("role", "data")
    if role not in {"data", "resource"}:
        raise ValueError("explicit selection supports data/resource specialists only")
    system = load_prompt("skill-selection").replace(
        "You are the main research Agent", f"You are the delegated {role.title()} Agent", 1)
    content, _ = client.chat_with_metadata(system,
        json.dumps({"delegated_task": assignment, "skill_catalog": catalog}, ensure_ascii=False),
        timeout=timeout, max_tokens=1200, read_only=True, auto_skills=False,
        include_research_context=False)
    issues = []
    try:
        answer = _object(content)
        chosen = answer.get("skill_reads")
        if not isinstance(chosen, list) or len(chosen) > 3:
            raise ValueError("skill_reads must be a list of at most three IDs with reasons")
    except (ValueError, TypeError, AttributeError):
        chosen = []
        issues.append("Invalid skill selection; no bodies were loaded and no retry was made")
    available = {row["id"] for row in catalog["entries"]}
    reasons = {}
    for row in chosen:
        identity = row.get("id") if isinstance(row, dict) else None
        why = row.get("why") if isinstance(row, dict) else None
        if (not isinstance(identity, str) or identity not in available or identity in reasons
                or not isinstance(why, str) or not why.strip() or len(why) > 400):
            issues.append("Unknown, duplicate or unexplained skill choice was withheld")
            continue
        reasons[identity] = why.strip()
    methods = read_selected_skills(list(reasons), benchmark=benchmark)
    for row in methods["selection"]:
        row.update(selected_by=f"{role}_agent", selection_reason=reasons[row["id"]])
    if len(methods["selection"]) != len(reasons):
        issues.append("A chosen skill was unavailable; inspect remaining evidence without it")
    return {"method_library": methods, "skill_selection_issues": issues}


def remaining_window(started: float, timeout: float) -> float:
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Data Agent / Resource Agent local window exhausted during skill selection")
    return remaining
