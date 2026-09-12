"""Audited same-condition comparison of the released and 2,500-episode ACT models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from autosim.robosyn_data import evaluation_seed_bank
from autosim.robosyn_mvp import (
    MVPConfig,
    RoboSynMVPRunner,
    gpu_lock,
    paired_success_comparison,
)


FROZEN_FILES = [
    "scripts/eval_policy.py",
    "policy/act/deploy_policy.yml",
    "policy/act/deploy_policy.py",
    "policy/inference_timing.py",
    "robosynchallenge/tasks/click_bell/click_bell.py",
    "configs/click_bell/random/gym_config.json",
    "configs/click_bell/action_config.json",
]

# All evaluation master seeds used in the preceding local AutoResearch rounds.
PREVIOUS_EVAL_BANKS = {
    "seed_zero": (0, 100),
    "v2_dev_a": (910_001, 40),
    "v2_dev_b": (920_001, 40),
    "v2_confirm_a": (931_001, 100),
    "v2_confirm_b": (932_001, 100),
    "v2_confirm_c": (933_001, 100),
    "v2_confirm_d": (934_001, 100),
    "v3_dev_a": (1_031_001, 40),
    "v3_dev_b": (1_032_001, 40),
    "v3_confirmation": (1_033_001, 100),
    "v3_appearance": (1_034_101, 30),
    "v3_camera": (1_034_201, 30),
    "v3_robot_pose": (1_034_301, 30),
    "v3_clutter": (1_034_401, 30),
    "v3_contact": (1_034_501, 30),
    "v3_internal_frozen": (1_039_001, 200),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def summarize_prefix(metrics: dict[str, Any], episodes: int) -> dict[str, Any]:
    rows = list(metrics["episodes"][:episodes])
    if len(rows) != episodes:
        raise RuntimeError(f"expected {episodes} episode rows, got {len(rows)}")
    successes = sum(bool(row["success"]) for row in rows)
    action_steps = [int(row["action_steps"]) for row in rows]
    inference_totals = [float(row["total_inference_time_seconds"]) for row in rows]
    inference_calls = [int(row["inference_call_count"]) for row in rows]
    return {
        "episode_count": episodes,
        "success_count": successes,
        "success_rate": successes / episodes,
        "average_action_steps": sum(action_steps) / episodes,
        "average_inference_time_per_episode_seconds": sum(inference_totals) / episodes,
        "inference_call_count": sum(inference_calls),
        "episodes": rows,
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [centre - radius, centre + radius]


def public_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_count": int(metrics["episode_count"]),
        "success_count": int(metrics["success_count"]),
        "success_rate": float(metrics["success_rate"]),
        "success_rate_wilson_95ci": wilson(
            int(metrics["success_count"]), int(metrics["episode_count"])
        ),
        "average_action_steps": float(metrics["average_action_steps"]),
        "average_inference_time_per_episode_seconds": float(
            metrics["average_inference_time_per_episode_seconds"]
        ),
    }


def parse_args() -> argparse.Namespace:
    workspace = Path("/home/wbc/下载/autoresearch")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=workspace / "AutoSimSOTA/RoboSynChallenge_eval_clean",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=workspace / "AutoSimSOTA/.venv/bin/python",
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=workspace / "AutoSimSOTA/RoboSynChallenge/checkpoints/ACT_sim_click_bell",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=workspace
        / "autosim/output/robosyn_click_bell_v3/mixture_ablation/M1_append/candidates/"
        "policy/train/checkpoints/080000/pretrained_model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace / "autosim/output/official_vs_2500_clean_20260902",
    )
    parser.add_argument("--fresh-master-seed", type=int, default=1_099_001)
    parser.add_argument("--fresh-episodes", type=int, default=500)
    parser.add_argument("--seed-zero-episodes", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.python = Path(os.path.abspath(args.python))
    args.official = args.official.resolve()
    args.candidate = args.candidate.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    fresh_seeds = evaluation_seed_bank(args.fresh_master_seed, args.fresh_episodes)
    previous_seeds = {
        seed
        for master, count in PREVIOUS_EVAL_BANKS.values()
        for seed in evaluation_seed_bank(master, count)
    }
    overlap = sorted(set(fresh_seeds) & previous_seeds)
    if overlap:
        raise RuntimeError(f"fresh bank overlaps previous eval banks: {overlap[:5]}")
    if len(set(fresh_seeds)) != len(fresh_seeds):
        raise RuntimeError("fresh bank contains duplicate episode seeds")

    frozen_hashes = {name: sha256_file(args.repo / name) for name in FROZEN_FILES}
    protocol = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "official released ACT vs 2,500-episode ACT, same-condition rerun",
        "repo": str(args.repo),
        "repo_commit": git_output(args.repo, "rev-parse", "HEAD"),
        "tracked_diff": git_output(args.repo, "diff", "--", *FROZEN_FILES),
        "frozen_hashes": frozen_hashes,
        "checkpoints": {
            "official_released_act": {
                "path": str(args.official),
                "model_sha256": sha256_file(args.official / "model.safetensors"),
                "config_sha256": sha256_file(args.official / "config.json"),
            },
            "model_2500_episodes": {
                "path": str(args.candidate),
                "model_sha256": sha256_file(args.candidate / "model.safetensors"),
                "config_sha256": sha256_file(args.candidate / "config.json"),
            },
        },
        "evaluation": {
            "task": "click_bell",
            "setting": "random",
            "max_episode_steps": 361,
            "renderer": "hybrid",
            "headless": True,
            "video": False,
            "eval_reset_sync_steps": 0,
            "policy_device": "cuda",
            "gpu_id": "0",
            "fresh_bank": {
                "master_seed": args.fresh_master_seed,
                "episodes": args.fresh_episodes,
                "episode_seed_sha256": hashlib.sha256(
                    json.dumps(fresh_seeds, separators=(",", ":")).encode()
                ).hexdigest(),
                "report_prefixes": [200, 500],
                "overlap_with_previous_eval_banks": len(overlap),
            },
            "seed_zero_bank": {
                "master_seed": 0,
                "episodes": args.seed_zero_episodes,
                "role": "compatibility result; not fresh held-out evidence",
            },
        },
        "instrumentation_note": (
            "Only per-episode result serialization and write-before-env.close were added; "
            "environment dynamics, policy observations/actions, horizon, and success test are unchanged."
        ),
    }
    write_json(args.output / "protocol.json", protocol)

    config = MVPConfig(
        name="official-vs-2500-clean",
        repo=args.repo,
        python=args.python,
        gpu_id="0",
        task="click_bell",
        setting="random",
        baseline_checkpoint=args.official,
        dataset_root=None,
        dataset_mixture_manifest=None,
        output_root=args.output,
        embodichain_data_root=None,
        extra_library_paths=[Path("/home/wbc/miniconda3/envs/robotwin5090/lib")],
        evaluation={
            "rungs": [100],
            "training_steps": [80_000],
            "seed": args.fresh_master_seed,
            "timeout_seconds": 7_200,
            "renderer": "hybrid",
            "video": False,
            "startup_retries": 2,
            "startup_retry_exit_codes": [-11, -6],
            "startup_retry_delay_seconds": 5.0,
        },
        frozen_files=FROZEN_FILES,
    )
    run_dir = args.output / "runner"
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = RoboSynMVPRunner(config, run_dir=run_dir)

    jobs = [
        ("official_fresh500", args.official, args.fresh_episodes, args.fresh_master_seed),
        ("model2500_fresh500", args.candidate, args.fresh_episodes, args.fresh_master_seed),
        ("official_seed0_100", args.official, args.seed_zero_episodes, 0),
        ("model2500_seed0_100", args.candidate, args.seed_zero_episodes, 0),
    ]
    results: dict[str, dict[str, Any]] = {}
    with gpu_lock("0"):
        for name, checkpoint, episodes, seed in jobs:
            result_file = args.output / f"{name}.json"
            if result_file.exists():
                results[name] = json.loads(result_file.read_text(encoding="utf-8"))
                print(f"RESUME {name}: {results[name]['success_count']}/{episodes}", flush=True)
                continue
            print(f"START {name}: episodes={episodes} master_seed={seed}", flush=True)
            results[name] = runner.evaluate(
                checkpoint, episodes, name=name, seed=seed
            )
            write_json(result_file, results[name])
            print(
                f"DONE {name}: {results[name]['success_count']}/{episodes} "
                f"({100 * results[name]['success_rate']:.2f}%)",
                flush=True,
            )
            runner.guard.assert_unchanged()

    official_200 = summarize_prefix(results["official_fresh500"], 200)
    model_200 = summarize_prefix(results["model2500_fresh500"], 200)
    comparisons = {
        "fresh_prefix_200": {
            "official": public_summary(official_200),
            "model_2500": public_summary(model_200),
            "paired": paired_success_comparison(model_200, official_200),
        },
        "fresh_full_500": {
            "official": public_summary(results["official_fresh500"]),
            "model_2500": public_summary(results["model2500_fresh500"]),
            "paired": paired_success_comparison(
                results["model2500_fresh500"], results["official_fresh500"]
            ),
        },
        "seed_zero_100": {
            "official": public_summary(results["official_seed0_100"]),
            "model_2500": public_summary(results["model2500_seed0_100"]),
            "paired": paired_success_comparison(
                results["model2500_seed0_100"], results["official_seed0_100"]
            ),
        },
    }
    fresh_official_seeds = [
        int(row["episode_seed"]) for row in results["official_fresh500"]["episodes"]
    ]
    fresh_model_seeds = [
        int(row["episode_seed"]) for row in results["model2500_fresh500"]["episodes"]
    ]
    zero_official_seeds = [
        int(row["episode_seed"]) for row in results["official_seed0_100"]["episodes"]
    ]
    zero_model_seeds = [
        int(row["episode_seed"]) for row in results["model2500_seed0_100"]["episodes"]
    ]
    if fresh_official_seeds != fresh_seeds or fresh_model_seeds != fresh_seeds:
        raise RuntimeError("fresh-bank episode seeds differ from the locked protocol")
    if zero_official_seeds != zero_model_seeds:
        raise RuntimeError("seed-zero episode seeds differ between checkpoints")
    runner.guard.assert_unchanged()
    final_hashes = {name: sha256_file(args.repo / name) for name in FROZEN_FILES}
    if final_hashes != frozen_hashes:
        raise RuntimeError("frozen evaluator/config hash changed during evaluation")
    summary = {
        "schema_version": 1,
        "completed_at": datetime.now().astimezone().isoformat(),
        "protocol_path": str(args.output / "protocol.json"),
        "same_seed_assertions": {"fresh500": True, "seed0_100": True},
        "frozen_hashes_unchanged": True,
        "comparisons": comparisons,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
