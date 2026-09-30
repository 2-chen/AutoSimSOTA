"""Backend for versioned integration-remediation jobs."""

import argparse
from pathlib import Path

from autosim.research.common import atomic_json, digest, read_json
from autosim.research.runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["robosyn-safe-evaluate"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--task", default="click_bell")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", choices=["act", "dp"], required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    output = args.attempt / "outputs"
    runtime = Runtime(args.workspace, args.attempt)
    command = [str(runtime.python), "-m", "autosim.experiment_validation.safe_evaluation",
               "--task", args.task, "--checkpoint", str(args.checkpoint),
               "--output", str(output), "--episodes", str(args.episodes),
               "--seed", str(args.seed), "--policy", args.policy]
    runtime.run(command, output / "process", max(1800, args.episodes * 600), evaluation=True)
    metrics = read_json(output / "evaluation_metrics.json")
    safety = read_json(output / "safety_summary.json")
    result = {
        "status": "completed", "stage": "evaluate", "task": args.task,
        "policy": args.policy, "metrics": metrics, "safety": safety,
        "execution_mode": "real_simulation", "policy_quality_claim": False,
        "ranking_eligible": False,
        "verified_files": {str(output / name): digest(output / name) for name in
                           ("evaluation_metrics.json", "safety_summary.json", "protocol.json")},
    }
    atomic_json(args.attempt / "result.json", result)


if __name__ == "__main__":
    main()
