"""Task-aware native semantic probe without modifying benchmark sources."""

import argparse
import importlib
from pathlib import Path

import numpy as np

from autosim.experiment_system.plugins import RoboTwinPlugin
from autosim.experiment_system.robotwin_worker import native_args
from autosim.research.common import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--task", choices=["open_laptop"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    contract = RoboTwinPlugin(args.repo).contract(args.task)
    contract.assert_sources()
    config = native_args(args.repo, args.task, args.attempt / "probe")
    module = importlib.import_module(f"envs.{args.task}")
    native = getattr(module, args.task)()
    try:
        native.setup_demo(now_ep_num=0, seed=args.seed, eval_mode=True, **config)
        observation = native.get_obs()
        state = np.asarray(observation["joint_action"]["vector"])
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("native state mismatch")
        for camera, shape in contract.observation["cameras"].items():
            rgb = observation["observation"][camera]["rgb"]
            if list(rgb.shape) != shape or not np.isfinite(rgb).all():
                raise ValueError("native camera mismatch")
        # open_laptop.check_success requires the same arm choice established at
        # the start of its official play_once().  Reproduce that precondition;
        # do not replace or bypass the task's success predicate.
        face_prod = module.get_face_prod(native.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        native.arm_tag = module.ArmTag("left" if face_prod > 0 else "right")
        before = native.take_action_cnt
        native.take_action(state)
        if native.take_action_cnt != before + 1:
            raise ValueError("native action step counter mismatch")
        result = {
            "status": "completed", "execution_mode": "real_simulation",
            "native_observation_action_probe": True, "complete_semantic_parity_verified": False,
            "success_predicate_modified": False,
            "probe_precondition": "official_open_laptop_play_once_arm_selection",
            "task_source_sha256": digest(args.repo / "envs/open_laptop.py"),
            "contract_signature": contract.signature,
        }
        atomic_json(args.attempt / "result.json", result)
    finally:
        native.close_env()
    contract.assert_sources()


if __name__ == "__main__":
    main()
