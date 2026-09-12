"""RoboTwin evaluator with isolated policy import and native import cwd."""

import argparse
import importlib
import os
import sys
from pathlib import Path

from autosim.experiment_system.plugins import ROBOTWIN_TASKS, RoboTwinPlugin
from autosim.experiment_system.robotwin_worker import native_args
from autosim.research.common import atomic_json, digest
from autosim.research.policy_rpc import RemotePolicy


def evaluate(repo: Path, policy_repo: Path, task_name: str, destination: Path,
             checkpoint: Path, policy: str, episodes: int, seed: int):
    repo, policy_repo = repo.absolute(), policy_repo.absolute()
    sys.path.insert(0, str(policy_repo))
    cameras = ("head_camera", "left_camera", "right_camera")
    contract = {"state_dim": 14, "action_dim": 14, "cameras": cameras,
                "camera_shapes": {camera: [240, 320, 3] for camera in cameras}}
    config = {"checkpoint_path": str(checkpoint), "policy_name": policy,
              "pytorch_device": "cuda", "act_step": 1, "dp_step": 1}
    adapter_path = policy_repo / f"policy/{policy}/deploy_policy.py"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"policy adapter missing: {adapter_path}")
    # The model child resolves the regular bridge before the native repository
    # is added or selected as cwd.
    model = RemotePolicy(config, contract)
    sys.path.insert(0, str(repo))
    original_cwd = Path.cwd()
    try:
        # RoboTwin reads ./assets and ./task_config during module import.
        os.chdir(repo)
        from script.eval_policy import eval_policy

        native = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
        args = native_args(repo, task_name, destination)
        args.update(policy_name="autosim.experiment_system.native_policy",
                    ckpt_setting="system_smoke")
        final_seed, successes = eval_policy(task_name, native, args, model, seed,
                                           test_num=episodes, video_size=None,
                                           instruction_type="unseen")
    finally:
        os.chdir(original_cwd)
        model.close()
    return {
        "status": "completed", "execution_mode": "real_simulation", "purpose": "smoke",
        "episodes": episodes, "successes": int(successes), "success_rate": successes / episodes,
        "start_seed": seed, "next_seed": final_seed,
        "seed_filtering": "native_expert_check_preserved",
        "native_main_loop": "script.eval_policy.eval_policy", "policy_quality_claim": False,
        "ranking_eligible": False,
        "namespace_remediation": "regular_bridge_policy_child_before_native_import",
        "native_cwd_remediation": "repo_selected_before_native_module_import",
        "policy_adapter": str(adapter_path), "policy_adapter_sha256": digest(adapter_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--policy-repo", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--task", choices=ROBOTWIN_TASKS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", choices=["act", "dp"], required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    contract = RoboTwinPlugin(args.repo).contract(args.task)
    contract.assert_sources()
    result = evaluate(args.repo, args.policy_repo, args.task, args.attempt, args.checkpoint,
                      args.policy, args.episodes, args.seed)
    result["contract_signature"] = contract.signature
    atomic_json(args.attempt / "result.json", result)
    contract.assert_sources()


if __name__ == "__main__":
    main()
