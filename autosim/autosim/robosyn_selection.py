"""Deterministic development-only selection for RoboSyn data ablations.

This module deliberately does not evaluate policies.  It consumes completed
development summaries, selects on each run's final confirmation rung, and
writes an auditable promotion manifest for the one candidate allowed to reach
the frozen final bank.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from autosim.robosyn_mvp import metric_order


def _confirmation_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    if summary.get("status") != "completed":
        raise ValueError("candidate run is not completed")
    if summary.get("evaluation_protocol") != "development":
        raise ValueError("selection accepts development summaries only")
    records = [
        record
        for record in summary.get("records", [])
        if record.get("status") == "completed" and record.get("metrics")
    ]
    if not records:
        raise ValueError("candidate run has no completed evaluation records")
    max_rung = max(int(record.get("rung", -1)) for record in records)
    matches = [record for record in records if int(record.get("rung", -1)) == max_rung]
    if len(matches) != 1:
        raise ValueError(f"expected one final confirmation record, found {len(matches)}")
    return matches[0]


def select_confirmed_candidate(summary_paths: Iterable[str | Path]) -> Dict[str, Any]:
    candidates = []
    for raw_path in summary_paths:
        path = Path(raw_path).expanduser().resolve()
        summary = json.loads(path.read_text())
        record = _confirmation_record(summary)
        metrics = record["metrics"]
        candidates.append(
            {
                "experiment": summary.get("name"),
                "summary_path": str(path),
                "run_dir": str(path.parent),
                "checkpoint": record.get("checkpoint"),
                "rung": int(record.get("rung", -1)),
                "training_steps": int(record.get("training_steps", 0)),
                "episode_count": int(metrics.get("episode_count", 0)),
                "success_count": int(metrics.get("success_count", 0)),
                "success_rate": float(metrics.get("success_rate", 0.0)),
                "average_action_steps": float(
                    metrics.get("average_action_steps", float("inf"))
                ),
                "average_inference_time_per_episode_seconds": metrics.get(
                    "average_inference_time_per_episode_seconds"
                ),
                "evaluation_seed": (metrics.get("evaluation_config") or {}).get("seed"),
                "metrics_path": metrics.get("metrics_path"),
            }
        )
    if not candidates:
        raise ValueError("at least one summary is required")
    checkpoints = [candidate["checkpoint"] for candidate in candidates]
    if any(not checkpoint for checkpoint in checkpoints):
        raise ValueError("every candidate must have a checkpoint")
    seeds = {candidate["evaluation_seed"] for candidate in candidates}
    episodes = {candidate["episode_count"] for candidate in candidates}
    steps = {candidate["training_steps"] for candidate in candidates}
    if len(seeds) != 1 or None in seeds:
        raise ValueError("candidates do not share one explicit confirmation seed")
    if len(episodes) != 1:
        raise ValueError("candidates do not share the same confirmation episode count")
    if len(steps) != 1:
        raise ValueError("candidates do not share the same training budget")

    def order(candidate: Dict[str, Any]) -> tuple[Any, ...]:
        return metric_order(candidate) + (-candidate["training_steps"], candidate["experiment"])

    ranked = sorted(candidates, key=order, reverse=True)
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "development_selection_completed",
        "ranking_eligible": False,
        "selection_rule": [
            "higher confirmation success rate",
            "lower average action steps",
            "lower average inference time per episode",
            "lexical experiment name tie-break",
        ],
        "confirmation_seed": next(iter(seeds)),
        "confirmation_episodes": next(iter(episodes)),
        "training_steps": next(iter(steps)),
        "selected": ranked[0],
        "ranking": ranked,
        "frozen_evaluation_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = select_confirmed_candidate(args.summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
