"""Summarize diagnostic action/state traces without generating benchmark scores."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def analyze(directory: Path) -> dict:
    path = directory / "dynamics_trace.jsonl"
    if not path.exists():
        return {"status": "trace_missing"}
    contract = json.loads((directory / "dynamics_contract.json").read_text())
    ids = [int(i) for i in contract["active_joint_ids"]]
    limits = contract["qpos_limits"][0]
    names = [contract["joint_names"][i] for i in ids]
    counts = [0] * len(ids)
    max_target = [0.] * len(ids)
    max_jump = [0.] * len(ids)
    previous, first_invalid = None, None
    last = []
    episode_first_actions = []
    lines = 0
    for line in path.open():
        row = json.loads(line)
        lines += 1
        if previous is None or previous["seed"] != row["seed"]:
            previous = None
            episode_first_actions.append({"seed": row["seed"], "step": row["step"], "action": row["action"]})
            last = []
        last.append(row)
        last = last[-6:]
        for i, raw_id in enumerate(ids):
            action = row["action"][i]
            if action is not None:
                max_target[i] = max(max_target[i], abs(action))
                if action < limits[raw_id][0] or action > limits[raw_id][1]:
                    counts[i] += 1
                if previous is not None and previous["action"][i] is not None:
                    max_jump[i] = max(max_jump[i], abs(action - previous["action"][i]))
        if first_invalid is None and any(x is None or not math.isfinite(x) for x in row["qpos"]):
            first_invalid = {"seed": row["seed"], "step": row["step"], "last_six_steps": list(last)}
        previous = row
    return {"status": "analyzed", "rows": lines, "first_invalid": first_invalid,
            "first_actions": episode_first_actions, "joint_names": names,
            "targets_outside_joint_limits_count": counts, "max_abs_action": max_target,
            "max_consecutive_action_change": max_jump,
            "note": "diagnostic only; normalized gripper actions require separate unit interpretation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    results = {}
    for path in args.root.glob("*/replay_result.json"):
        result = json.loads(path.read_text())
        results[path.parent.name] = analyze(Path(result["artifact"]))
    output = args.root / "dynamics_analysis.json"
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {"status": v["status"], "rows": v.get("rows"),
        "first_invalid": {a: b for a, b in (v.get("first_invalid") or {}).items() if a != "last_six_steps"},
        "max_abs_action": v.get("max_abs_action"), "max_action_change": v.get("max_consecutive_action_change")}
        for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
