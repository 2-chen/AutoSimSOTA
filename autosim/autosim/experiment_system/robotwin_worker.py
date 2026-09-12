"""Native RoboTwin integration: bounded expert collection and original evaluator.

No mock planner, task patch, success-threshold change, or expert-success override.
"""
import argparse
import importlib
import sys
from pathlib import Path

from autosim.research.common import atomic_json, digest, read_json
from .plugins import ROBOTWIN_TASKS, RoboTwinPlugin


def native_args(repo, task, destination):
    import yaml
    config = yaml.safe_load((repo / "task_config/demo_randomized.yml").read_text())
    embodiments = yaml.safe_load((repo / "task_config/_embodiment_config.yml").read_text())
    if config["embodiment"] != ["aloha-agilex"]:
        raise ValueError("unvalidated embodiment; new contract required")
    robot = repo / embodiments["aloha-agilex"]["file_path"]
    embodiment = yaml.safe_load((robot / "config.yml").read_text())
    config.update(task_name=task, task_config="demo_randomized", save_path=str(destination),
                  left_robot_file=str(robot), right_robot_file=str(robot),
                  left_embodiment_config=embodiment, right_embodiment_config=embodiment,
                  dual_arm_embodied=True, embodiment_name="aloha-agilex")
    return config


def native_collect(repo, task_name, destination, episodes, seed):
    config = native_args(repo, task_name, destination / "raw")
    records, accepted = [], []
    for offset in range(episodes * 12):
        if len(accepted) >= episodes:
            break
        seed_now = seed + offset
        idx = len(accepted)
        task = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
        row = {"seed": seed_now, "saved": False}
        try:
            args = dict(config, need_plan=True, save_data=False)
            task.setup_demo(now_ep_num=idx, seed=seed_now, **args)
            task.play_once()
            if not task.plan_success or not task.check_success():
                row["reason"] = "native_expert_failed"
                continue
            task.save_traj_data(idx)
            task.close_env()
            args.update(need_plan=False, save_data=True)
            task.setup_demo(now_ep_num=idx, seed=seed_now, **args)
            trajectory = task.load_tran_data(idx)
            args.update(left_joint_path=trajectory["left_joint_path"], right_joint_path=trajectory["right_joint_path"])
            task.set_path_lst(args)
            task.play_once()
            replay_success = bool(task.check_success())
            task.close_env()
            if not replay_success:
                row["reason"] = "native_replay_failed"
                # Do not reuse an index over partial output from a failed replay.
                raise RuntimeError("replay left partial data; independent attempt and audit required")
            task.merge_pkl_to_hdf5_video()
            path = Path(config["save_path"]) / "data" / f"episode{idx}.hdf5"
            if not path.is_file() or not path.stat().st_size:
                raise ValueError("native collector produced no HDF5")
            accepted.append(str(path))
            row.update(saved=True, path=str(path), sha256=digest(path))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            records.append(row)
            atomic_json(destination / "collection.json", {"status": "completed" if len(accepted) == episodes else "incomplete",
                        "task": task_name, "target_episodes": episodes, "accepted_hdf5": accepted, "attempts": records,
                        "native_judge_modified": False, "mock_planner_used": False})
            try:
                task.close_env()
            except Exception:
                pass
    if len(accepted) != episodes:
        raise RuntimeError("bounded native expert budget exhausted")
    return {"status": "completed", "hdf5": accepted,
            "verified_files": {p: digest(Path(p)) for p in accepted}, "automatic_collection": True}


def convert(hdf5, destination, task):
    import cv2
    import h5py
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    cameras = ["head_camera", "left_camera", "right_camera"]
    features = {"observation.state": {"dtype": "float32", "shape": (14,), "names": None},
                "action": {"dtype": "float32", "shape": (14,), "names": None}}
    features.update({f"observation.images.{c}": {"dtype": "video", "shape": (240, 320, 3),
                    "names": ["height", "width", "channels"]} for c in cameras})
    dataset = LeRobotDataset.create(f"autosim/{task}", fps=25, features=features, root=destination,
                                   robot_type="aloha-agilex", use_videos=True, image_writer_threads=2)
    for path in hdf5:
        with h5py.File(path, "r") as raw:
            states = np.concatenate([raw["joint_action/left_arm"][:], raw["joint_action/left_gripper"][:].reshape(-1, 1),
                                     raw["joint_action/right_arm"][:], raw["joint_action/right_gripper"][:].reshape(-1, 1)], axis=1).astype("float32")
            if len(states) < 2 or not np.isfinite(states).all():
                raise ValueError("invalid native states")
            for index in range(len(states) - 1):
                frame = {"observation.state": states[index], "action": states[index + 1]}
                for camera in cameras:
                    bits = raw[f"observation/{camera}/rgb"][index]
                    decoded = cv2.imdecode(np.frombuffer(bits, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is None or decoded.shape != (240, 320, 3):
                        raise ValueError("native camera decode/shape mismatch")
                    # Native recorder uses cv2.imencode on its RGB array; preserve
                    # channel order on inverse decode (do not add a BGR swap).
                    frame[f"observation.images.{camera}"] = decoded
                dataset.add_frame(frame, task=task)
            dataset.save_episode()
    dataset.stop_image_writer()
    atomic_json(destination / "conversion_provenance.json", {
        "sources": {p: digest(Path(p)) for p in hdf5}, "action_label": "next_recorded_joint_state",
        "last_frame_dropped": True, "nominal_encoding_fps": 25,
        "physical_control_hz_claim": False, "policy_observation": "joint_state_and_three_rgb_only"})
    return destination


def native_evaluate(repo, task_name, destination, checkpoint, policy, episodes, seed):
    import numpy as np
    from autosim.research.policy_rpc import RemotePolicy
    sys.path.append(str(repo / "description/utils"))
    from script.eval_policy import eval_policy
    native = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
    args = native_args(repo, task_name, destination)
    args.update(policy_name="autosim.experiment_system.native_policy", ckpt_setting="system_smoke")
    cameras = ("head_camera", "left_camera", "right_camera")
    contract = {"state_dim": 14, "action_dim": 14, "cameras": cameras,
                "camera_shapes": {c: [240, 320, 3] for c in cameras}}
    config = {"checkpoint_path": str(checkpoint), "policy_name": policy, "pytorch_device": "cuda",
              "act_step": 1, "dp_step": 1}
    model = RemotePolicy(config, contract)
    try:
        final_seed, successes = eval_policy(task_name, native, args, model, seed,
                                           test_num=episodes, video_size=None, instruction_type="unseen")
    finally:
        model.close()
    return {"status": "completed", "execution_mode": "real_simulation", "purpose": "smoke",
            "episodes": episodes, "successes": int(successes), "success_rate": successes / episodes,
            "start_seed": seed, "next_seed": final_seed, "seed_filtering": "native_expert_check_preserved",
            "native_main_loop": "script.eval_policy.eval_policy", "policy_quality_claim": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["imports", "probe", "collect", "convert", "evaluate"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--task", choices=ROBOTWIN_TASKS, default=ROBOTWIN_TASKS[0])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=92006000)
    parser.add_argument("--collection-result", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy", choices=["act", "dp"], default="act")
    args = parser.parse_args()
    args.attempt.mkdir(parents=True, exist_ok=True)
    contract = RoboTwinPlugin(args.repo).contract(args.task)
    contract.assert_sources()
    if args.stage == "imports":
        import envs.robot.planner as planner
        import envs.pick_dual_bottles
        import envs.stack_blocks_two
        import envs.open_laptop
        if not hasattr(planner, "CuroboPlanner") or "_Mock" in planner.CuroboPlanner.__name__:
            raise ValueError("real native cuRobo required")
        result = {"status": "completed", "native_imports": True, "native_runtime_validated": False}
    elif args.stage == "probe":
        import numpy as np
        config = native_args(args.repo, args.task, args.attempt / "probe")
        native = getattr(importlib.import_module(f"envs.{args.task}"), args.task)()
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
            before = native.take_action_cnt
            native.take_action(state)
            if native.take_action_cnt != before + 1:
                raise ValueError("native action step counter mismatch")
            result = {"status": "completed", "execution_mode": "real_simulation", "native_observation_action_probe": True,
                      "complete_semantic_parity_verified": False, "contract_signature": contract.signature}
        finally:
            native.close_env()
    elif args.stage == "collect":
        result = native_collect(args.repo, args.task, args.attempt, args.episodes, args.seed)
    elif args.stage == "convert":
        collected = read_json(args.collection_result)
        result = {"status": "completed", "dataset": str(convert(collected["hdf5"], args.attempt / "dataset", args.task))}
    else:
        result = native_evaluate(args.repo, args.task, args.attempt, args.checkpoint, args.policy, args.episodes, args.seed)
    contract.assert_sources()
    result["contract_signature"] = contract.signature
    atomic_json(args.attempt / "result.json", result)


if __name__ == "__main__":
    main()
