"""Receipt-derived warning signals; advisory, never a benchmark-specific policy."""
from collections import Counter


def execution_progress(steps, *, window=8):
    recent = list(steps[-window:])
    failures = {"repair proposal rejected", "checkpoint failure", "raised", "failed",
                "reverification_failed", "rejected", "could not be asked", "decision_rejected",
                "blocked"}
    # Inspection, plans, new proposal text and failure IDs do not certify capabilities.
    trailing = []
    for row in reversed(recent):
        if (row.get("outcome") not in failures and (
                row.get("verification_level") in {"native", "L2", "L3"} or
                (row.get("outcome") == "checkpoint" and row.get("evidence_id")))):
            break
        if row.get("outcome") in failures:
            trailing.append(row)
    counts = Counter((row.get("step"), row.get("outcome")) for row in trailing)
    repeated = [{"step": step, "outcome": outcome, "count": count}
                for (step, outcome), count in counts.items() if count >= 2]
    return {"window_actions": len(recent), "failure_count": len(trailing),
            "needs_strategy_change": len(trailing) >= 3,
            "repeated_failures": repeated,
            "guidance": ("Use existing evidence. Name the changed executable remedy and its "
                "next native receipt/postcondition. For a contract rejection correct only the "
                "identified field; for missing resources verify an acquisition/reuse route. "
                "Do not repeat unchanged installs or delegate another scan of settled facts. "
                "If no route is executable, report the specific boundary, not progress.")
                if len(trailing) >= 3 else "",
            "authority": "advisory_only_not_a_score_or_automatic_stop"}
