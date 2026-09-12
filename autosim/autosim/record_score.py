#!/usr/bin/env python3
"""
AutoSim score recorder — mirrors AutoSOTA's record_score.sh.

Atomic score recording: appends JSON line to scores.jsonl,
tracks best score, supports idempotency.
"""

import os, sys, json, argparse
from datetime import datetime
from pathlib import Path


def record_score(output_path: str, iteration: int, idea_id: str,
                 title: str, status: str, primary_score: float,
                 params: dict = None, metrics: dict = None,
                 notes: str = "", is_best: bool = False):
    """Record a score entry to scores.jsonl."""

    entry = {
        "iter": iteration,
        "idea_id": idea_id,
        "idea_title": title,
        "status": status,
        "primary_metric": primary_score,
        "params": params or {},
        "metrics": metrics or {},
        "notes": notes,
        "is_best": is_best,
        "timestamp": datetime.now().isoformat(),
    }

    scores_path = Path(output_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    # Idempotency check: skip if this iter+idea already recorded
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                    if (existing.get("iter") == iteration and
                        existing.get("idea_id") == idea_id):
                        print(f"[Record] Iter {iteration} ({idea_id}) already recorded — skip")
                        return
                except json.JSONDecodeError:
                    pass

    with open(scores_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    marker = "★ BEST" if is_best else ""
    print(f"[Record] Iter {iteration}: {primary_score:.4f} ({status}) {marker}")


def get_best_score(scores_path: str, direction: str = "higher") -> float:
    """Get the best score from scores.jsonl."""
    if not os.path.exists(scores_path):
        return None

    best = None
    with open(scores_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                score = entry.get("primary_metric")
                if score is None:
                    continue
                if best is None:
                    best = score
                elif direction == "higher" and score > best:
                    best = score
                elif direction == "lower" and score < best:
                    best = score
            except json.JSONDecodeError:
                pass
    return best


def main():
    parser = argparse.ArgumentParser(description="AutoSim Score Recorder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--idea-id", required=True)
    parser.add_argument("--title", default="Untitled")
    parser.add_argument("--status", default="success")
    parser.add_argument("--primary", type=float, required=True)
    parser.add_argument("--params", default="{}")
    parser.add_argument("--metrics", default="{}")
    parser.add_argument("--notes", default="")
    parser.add_argument("--is-best", action="store_true", default=False)
    args = parser.parse_args()

    params = json.loads(args.params) if isinstance(args.params, str) else args.params
    metrics = json.loads(args.metrics) if isinstance(args.metrics, str) else args.metrics

    record_score(
        output_path=args.output,
        iteration=args.iter,
        idea_id=args.idea_id,
        title=args.title,
        status=args.status,
        primary_score=args.primary,
        params=params,
        metrics=metrics,
        notes=args.notes,
        is_best=args.is_best,
    )


if __name__ == "__main__":
    main()
