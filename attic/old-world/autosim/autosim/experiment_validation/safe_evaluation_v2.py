"""Single-episode, fixed-seed form of the non-ranking safe evaluator."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from autosim.experiment_validation.safe_evaluation import NumericalSafetyAdapter, NumericalSafetyEnv
from autosim.research.common import atomic_json, digest, freeze_files
from autosim.research.registry import load_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--episode-seed", type=int, required=True)
    parser.add_argument("--policy", choices=["act", "dp"], required=True)
    args = parser.parse_args()
    repo, output = Path.cwd(), args.output.absolute()
    if (output / "evaluation_metrics.json").exists():
        raise FileExistsError("refusing to overwrite an existing evaluation")
    output.mkdir(parents=True, exist_ok=True)
    spec = load_task(repo, args.task)
    source = repo / "scripts/eval_policy.py"
    module_spec = importlib.util.spec_from_file_location("official_robosyn_evaluator", source)
    official = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(official)
    import yaml

    config = yaml.safe_load((repo / f"policy/{args.policy}/deploy_policy.yml").read_text())
    config.update(task_name=args.task, setting="random", policy_name=args.policy,
                  checkpoint_path=str(args.checkpoint.absolute()), max_episodes=1,
                  seed=args.master_seed, eval_fixed_episode_seed=args.episode_seed,
                  eval_video_log=False, eval_reset_sync_steps=0, headless=True,
                  renderer="hybrid", pytorch_device="cuda", gpu_id=0)
    adapter = NumericalSafetyAdapter(spec.as_dict())
    official.parse_args_and_config = lambda: config
    official.load_policy_adapter = lambda unused: adapter
    official.create_eval_run_dir = lambda unused: output
    make_env = official.make_env_from_configs

    def wrapped_factory(*factory_args, **factory_kwargs):
        atomic_json(output / "startup.json", {"phase": "environment_constructing", "policy_has_acted": False})
        env, gym_config = make_env(*factory_args, **factory_kwargs)
        wrapped = NumericalSafetyEnv(env, spec, output, adapter, "smoke")
        atomic_json(output / "startup.json", {"phase": "environment_ready", "policy_has_acted": False})
        return wrapped, gym_config

    official.make_env_from_configs = wrapped_factory
    v1 = Path(__file__).with_name("safe_evaluation.py")
    files = [source, repo / f"policy/{args.policy}/deploy_policy.py", Path(__file__), v1,
             Path(__file__).parents[1] / "research/evaluation.py",
             Path(__file__).parents[1] / "research/policy_rpc.py", Path(spec.gym_config),
             Path(spec.action_config), repo / f"robosynchallenge/tasks/{args.task}/{args.task}.py"]
    atomic_json(output / "protocol.json", {
        "schema_version": 3, "task": spec.as_dict(), "purpose": "integration_smoke",
        "ranking_eligible": False, "frozen_files": freeze_files(files),
        "checkpoint_sha256": digest(args.checkpoint / "model.safetensors"),
        "master_seed": args.master_seed, "episode_seed": args.episode_seed,
        "episodes": 1, "policy": args.policy,
        "process_isolation": "one_fresh_simulator_process_per_episode",
        "compatibility_remediation": "numerical_failure_is_failed_episode_v3",
    })
    try:
        official.main()
    except BaseException as exc:
        atomic_json(output / "worker_failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
