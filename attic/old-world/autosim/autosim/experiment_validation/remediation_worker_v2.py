"""Fresh-process episode aggregation for remediation v4."""

import argparse
from pathlib import Path

import numpy as np

from autosim.research.common import atomic_json, digest, freeze_files, read_json
from autosim.research.runtime import Runtime


def aggregate_metrics(metrics: list[dict], master_seed: int) -> dict:
    episodes, timings, episode_totals = [], [], []
    action_steps = []
    success_count = 0
    for index, item in enumerate(metrics):
        episode = dict(item["episodes"][0], episode_index=index)
        episodes.append(episode)
        summary = item["summary"]
        success_count += int(summary["success_count"])
        action_steps.append(float(summary["average_action_steps"]))
        count = int(summary["inference_call_count"])
        if count:
            timings.extend([float(summary["average_inference_time_seconds"])] * count)
        episode_totals.append(float(summary["average_inference_time_per_episode_seconds"]))
    count = len(metrics)
    max_steps = int(metrics[0]["config"]["timeout_action_steps"])
    total_calls = len(timings)
    summary = {
        "episode_count": count, "success_count": success_count,
        "success_rate": success_count / count,
        "average_action_steps": float(np.mean(action_steps)),
        "average_action_steps_ratio": float(np.mean(action_steps)) / max_steps,
        "inference_call_count": total_calls,
        "average_inference_calls_per_episode": total_calls / count,
        "average_inference_time_seconds": float(np.mean(timings)) if timings else None,
        "average_inference_time_per_episode_seconds": float(np.mean(episode_totals)),
    }
    config = dict(metrics[0]["config"], episode_count=count, seed=master_seed,
                  ranking_eligible=False)
    return {
        "schema_version": 3, "created_at": metrics[-1]["created_at"], "config": config,
        "inference_timing_scope": metrics[0]["inference_timing_scope"],
        "platform": metrics[0]["platform"], "diagnostic": None, "summary": summary,
        "episodes": episodes, "process_isolation": "one_fresh_simulator_process_per_episode",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["robosyn-isolated-safe-evaluate"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--task", default="click_bell")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", choices=["act", "dp"], required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        raise ValueError("bounded smoke episode count required")
    output = args.attempt / "outputs"
    runtime = Runtime(args.workspace, args.attempt)
    rng = np.random.RandomState(args.seed)
    episode_seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(args.episodes)]
    metrics, safety, verified = [], [], {}
    for index, episode_seed in enumerate(episode_seeds):
        child = output / f"episode_{index:03d}"
        command = [str(runtime.python), "-m", "autosim.experiment_validation.safe_evaluation_v2",
                   "--task", args.task, "--checkpoint", str(args.checkpoint),
                   "--output", str(child), "--master-seed", str(args.seed),
                   "--episode-seed", str(episode_seed), "--policy", args.policy]
        runtime.run(command, child / "process", 1800, evaluation=True)
        metric_path, safety_path, protocol_path = (child / "evaluation_metrics.json",
                                                    child / "safety_summary.json", child / "protocol.json")
        metric, safe, protocol = read_json(metric_path), read_json(safety_path), read_json(protocol_path)
        if metric["episodes"][0]["episode_seed"] != episode_seed or protocol["episode_seed"] != episode_seed:
            raise ValueError("fixed episode seed lineage mismatch")
        metrics.append(metric)
        safety.append(safe)
        for path in (metric_path, safety_path, protocol_path):
            verified[str(path)] = digest(path)
    aggregate = aggregate_metrics(metrics, args.seed)
    aggregate_safety = {
        "status": "completed", "ranking_eligible": False, "policy_score_claim": False,
        "success_criterion_modified": False, "horizon_modified": False,
        "seed_protocol_modified": False, "episode_process_isolation": True,
        "episode_seeds": episode_seeds,
        "action_count": sum(row["action_count"] for row in safety),
        "clipped_action_count": sum(row["clipped_action_count"] for row in safety),
        "clipped_element_count": sum(row["clipped_element_count"] for row in safety),
        "maximum_abs_raw_action": max(row["maximum_abs_raw_action"] for row in safety),
        "invalid_state_episodes": [dict(event, episode_index=index)
                                   for index, row in enumerate(safety)
                                   for event in row["invalid_state_episodes"]],
    }
    atomic_json(output / "evaluation_metrics.json", aggregate)
    atomic_json(output / "safety_summary.json", aggregate_safety)
    atomic_json(output / "protocol.json", {
        "schema_version": 3, "purpose": "integration_smoke", "ranking_eligible": False,
        "task": args.task, "policy": args.policy, "episodes": args.episodes,
        "master_seed": args.seed, "ordered_episode_seeds": episode_seeds,
        "checkpoint_sha256": digest(args.checkpoint / "model.safetensors"),
        "children": verified, "frozen_files": freeze_files([Path(__file__),
            Path(__file__).with_name("safe_evaluation_v2.py"),
            Path(__file__).with_name("safe_evaluation.py")]),
    })
    for name in ("evaluation_metrics.json", "safety_summary.json", "protocol.json"):
        verified[str(output / name)] = digest(output / name)
    atomic_json(args.attempt / "result.json", {
        "status": "completed", "stage": "evaluate", "task": args.task,
        "policy": args.policy, "metrics": aggregate, "safety": aggregate_safety,
        "execution_mode": "real_simulation", "policy_quality_claim": False,
        "ranking_eligible": False, "verified_files": verified,
    })


if __name__ == "__main__":
    main()
