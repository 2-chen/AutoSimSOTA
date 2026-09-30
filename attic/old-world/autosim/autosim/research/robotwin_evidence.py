"""Turn RoboTwin native evaluation traces into policy-specific evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import atomic_json, digest, now, object_digest
from .finalize import wilson
from .ledger import compare


def load_trace(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty evaluation trace")
    return rows


def analyze(trace: Path, checkpoint: Path, *, task: str, setting: str,
            max_actions: int, purpose: str, evaluation_seed: int) -> dict:
    rows = load_trace(trace)
    episodes = [row for row in rows if row.get("status") == "episode_counted"]
    if not episodes:
        raise ValueError("trace contains no counted episodes")
    seeds = [int(row["seed"]) for row in episodes]
    if len(seeds) != len(set(seeds)):
        raise ValueError("counted evaluation seeds are not unique")
    successes = sum(bool(row["success"]) for row in episodes)
    skipped = Counter(row.get("reason", "unknown") for row in rows
                      if row.get("status") == "seed_skipped")
    failures = [row for row in episodes if not row["success"]]
    categories = {
        "timeout_or_step_limit": sum(int(row.get("action_steps", max_actions)) >= max_actions
                                     for row in failures),
        "failed_before_step_limit": sum(int(row.get("action_steps", max_actions)) < max_actions
                                        for row in failures),
    }
    metrics = {
        "execution_mode": "real_simulation", "harness": "RoboTwin_XPolicyLab_native_v2",
        "purpose": purpose,
        "config": {"task": task, "setting": setting, "timeout_action_steps": max_actions,
                   "seed": evaluation_seed},
        "summary": {"episode_count": len(episodes), "success_count": successes,
                    "success_rate": successes / len(episodes),
                    "wilson_95_interval": wilson(successes, len(episodes))},
        "episodes": [{"episode_index": index, "episode_seed": int(row["seed"]),
                      "success": bool(row["success"]),
                      "action_steps": row.get("action_steps")}
                     for index, row in enumerate(episodes)],
    }
    identity = {"task": task, "setting": setting, "checkpoint_sha256": digest(checkpoint),
                "trace_sha256": digest(trace), "purpose": purpose, "evaluation_seed": evaluation_seed}
    return {"schema_version": 1, "kind": "robotwin_policy_evidence", "created_at": now(),
            "evidence_id": object_digest(identity), "identity": identity,
            "metrics": metrics, "failure_categories": categories,
            "skipped_seed_reasons": dict(sorted(skipped.items())),
            "observation_limit": "aggregate action-step outcomes; causal failure stage remains unknown"}


def paired(candidate: dict, baseline: dict) -> dict:
    return compare(candidate["metrics"], baseline["metrics"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--max-actions", type=int, required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.checkpoint, task=args.task, setting=args.setting,
                     max_actions=args.max_actions, purpose=args.purpose,
                     evaluation_seed=args.evaluation_seed)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
