"""Isolated smoke episodes with bounded pre-reset native startup recovery."""

import argparse
from pathlib import Path

import numpy as np

from autosim.experiment_validation.remediation_worker_v2 import aggregate_metrics
from autosim.research.common import atomic_json, digest, freeze_files, read_json
from autosim.research.runtime import Runtime, retryable_evaluation_startup


def run_episode(runtime: Runtime, command_prefix: list[str], parent: Path,
                episode_seed: int, max_startups: int = 3) -> tuple[Path, int]:
    if not 1 <= max_startups <= 3:
        raise ValueError("startup attempt limit must be in [1, 3]")
    for number in range(1, max_startups + 1):
        attempt = parent / f"startup_attempt_{number}"
        command = [*command_prefix, "--output", str(attempt),
                   "--episode-seed", str(episode_seed)]
        try:
            runtime.run(command, attempt / "process", 1800, evaluation=True)
            return attempt, number
        except RuntimeError:
            if number == max_startups or not retryable_evaluation_startup(attempt):
                raise
    raise AssertionError("unreachable")


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
    metrics, safety, verified, startup_attempts = [], [], {}, []
    prefix = [str(runtime.python), "-m", "autosim.experiment_validation.safe_evaluation_v2",
              "--task", args.task, "--checkpoint", str(args.checkpoint),
              "--master-seed", str(args.seed), "--policy", args.policy]
    for index, episode_seed in enumerate(episode_seeds):
        child, count = run_episode(runtime, prefix, output / f"episode_{index:03d}", episode_seed)
        startup_attempts.append(count)
        metric_path, safety_path, protocol_path = (child / "evaluation_metrics.json",
                                                    child / "safety_summary.json", child / "protocol.json")
        metric, safe, protocol = read_json(metric_path), read_json(safety_path), read_json(protocol_path)
        if metric["episodes"][0]["episode_seed"] != episode_seed or protocol["episode_seed"] != episode_seed:
            raise ValueError("fixed episode seed lineage mismatch")
        metrics.append(metric)
        safety.append(safe)
        for path in (metric_path, safety_path, protocol_path, child / "process/process.json"):
            verified[str(path)] = digest(path)
        for failed in range(1, count):
            failed_root = output / f"episode_{index:03d}" / f"startup_attempt_{failed}"
            for path in (failed_root / "process/process.json", failed_root / "startup.json"):
                verified[str(path)] = digest(path)
    aggregate = aggregate_metrics(metrics, args.seed)
    aggregate["startup_attempts_per_episode"] = startup_attempts
    aggregate_safety = {
        "status": "completed", "ranking_eligible": False, "policy_score_claim": False,
        "success_criterion_modified": False, "horizon_modified": False,
        "seed_protocol_modified": False, "episode_process_isolation": True,
        "startup_retry_rule": "only_sigabrt_or_sigsegv_before_first_reset_max_three",
        "episode_seeds": episode_seeds, "startup_attempts_per_episode": startup_attempts,
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
        "schema_version": 4, "purpose": "integration_smoke", "ranking_eligible": False,
        "task": args.task, "policy": args.policy, "episodes": args.episodes,
        "master_seed": args.seed, "ordered_episode_seeds": episode_seeds,
        "startup_attempts_per_episode": startup_attempts,
        "checkpoint_sha256": digest(args.checkpoint / "model.safetensors"),
        "children": verified, "frozen_files": freeze_files([Path(__file__),
            Path(__file__).with_name("remediation_worker_v2.py"),
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
