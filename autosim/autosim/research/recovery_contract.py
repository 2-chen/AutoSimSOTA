"""Small, receipt-linked recovery view; no benchmark names or inferred success."""
from typing import Any


def pending(steps: list[dict[str, Any]], fix: dict[str, Any]) -> dict[str, Any]:
    """Offer one revalidation, not permission to replay a scored experiment."""
    failed_ref = str(fix.get("failure_receipt_ref") or "")
    fix_id = str(fix.get("fix_attempt_id") or "")
    if fix.get("status") == "assessed" and fix.get("assessment") == "repair_attempted":
        for index in range(len(steps) - 1, -1, -1):
            original = steps[index]
            if not failed_ref or not fix_id or original.get("receipt_ref") != failed_ref:
                continue
            later = steps[index + 1:]
            if (any(row.get("fix_attempt_id") == fix_id and
                    row.get("step") == "retry_failed_action" for row in later) or
                    any(row.get("step") == original.get("step") and
                        row.get("arguments", {}) == original.get("arguments", {}) and
                        row.get("outcome") in {"done", "already done"} for row in later)):
                break
            return {"owner": "source_fix", "original": original,
                    "fix_attempt_id": fix_id, "failure_receipt_ref": failed_ref}

    # Planning never executed a native operation. Retrying means generating a corrected
    # proposal with sealed feedback, not repeating an unknown external side effect.
    builds = [(index, row) for index, row in enumerate(steps)
              if row.get("step") == "build_the_environment" or
              (row.get("step") == "retry_failed_action" and
               row.get("replayed_step") == "build_the_environment")]
    if not builds:
        return {}
    index, original = builds[-1]
    domain = original.get("failure_domain")
    if domain == "repair_contract":
        # Proposal rejection launched nothing. It remains Scheduler-owned, not another
        # replay charge against the old native failure or a vanished recovery obligation.
        return {}
    if (domain not in {"framework_plan", "native_install", "native_probe"} or
            not original.get("receipt_ref") or not original.get("evidence_id")):
        return {}
    consecutive = 0
    for _, row in reversed(builds):
        if row.get("failure_domain") != domain:
            break
        if domain in {"native_install", "native_probe"} and (row.get("native_operation") or {}).get("template") != (original.get("native_operation") or {}).get("template"):
            break
        consecutive += 1
    if consecutive >= 2 or any(
            row.get("recovery_parent_ref") == original["receipt_ref"]
            for row in steps[index + 1:]):
        return {}
    return {"owner": "environment_executor" if domain in {"native_install", "native_probe"} else "environment_planner", "original": original,
            "failure_receipt_ref": original["receipt_ref"],
            "evidence_id": original["evidence_id"], "retry_limit": 1}


def view(steps: list[dict[str, Any]], fix: dict[str, Any]) -> dict[str, Any]:
    target = pending(steps, fix)
    trials = [row for row in steps if row.get("step") == "retry_failed_action"]
    latest = trials[-1] if trials else {}
    unresolved = {}
    for row in reversed(steps):
        if row.get("step") != "build_the_environment" and row.get("replayed_step") != "build_the_environment":
            continue
        if row.get("outcome") in {"done", "already done", "reverified"}:
            break
        if row.get("failure_domain") == "repair_contract":
            unresolved = {key: row.get(key) for key in (
                "failure_domain", "repair_owner", "parent_evidence_id", "receipt_ref", "recovery_revision")}
            unresolved["next_action"] = "build_the_environment"
            unresolved["required_input"] = "revised evidence-backed repair_proposal"
            break
        if row.get("failure_domain") in {"native_install", "native_probe", "framework_plan"}:
            unresolved = {key: row.get(key) for key in ("failure_domain", "repair_owner", "evidence_id", "receipt_ref", "native_operation")}
            break
    return {"status": "revalidation_required" if target else
            "unresolved_requires_new_evidence" if unresolved else "no_pending_revalidation",
            "unresolved_failure": unresolved or None,
            "pending": ({key: value for key, value in target.items() if key != "original"}
                        | {"original_step": target["original"].get("step"),
                           "original_arguments": target["original"].get("arguments", {}),
                           "next_action": "retry_failed_action"}) if target else None,
            "latest_revalidation": ({key: latest.get(key) for key in (
                "outcome", "replayed_step", "original_outcome", "recovery_parent_ref",
                "receipt_ref", "evidence_refs")} if latest else None),
            "boundary": "A Fix report is not recovery. Revalidate the failed capability with "
                        "native receipts; an independently approved installation route may "
                        "replace its obsolete command, but never weaken consumer probes or "
                        "scientific protocol. Scored candidates require a new audited round."}
