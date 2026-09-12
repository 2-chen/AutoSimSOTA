"""Run RoboTwin ACT deployment evaluation with a configurable test count."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _install_lightweight_import_stubs() -> None:
    """Stub optional RoboTwin visualization imports that are unused in eval."""
    import types

    if "open3d" not in sys.modules:
        open3d = types.ModuleType("open3d")
        open3d.geometry = type("geometry", (), {"PointCloud": type("PointCloud", (), {})})()
        open3d.utility = type("utility", (), {"Vector3dVector": lambda x: x})()
        open3d.io = type("io", (), {"write_point_cloud": lambda *a, **k: None})()
        sys.modules["open3d"] = open3d

    if "test_render" not in sys.modules:
        test_render = types.ModuleType("test_render")
        test_render.Sapien_TEST = lambda: None
        sys.modules["test_render"] = test_render


def _apply_checkpoint_overrides(usr_args: dict, ckpt_dir: str) -> None:
    """Infer deploy-time architecture knobs from the checkpoint when possible."""
    ckpt_path = Path(ckpt_dir) / "policy_last.ckpt"
    if not ckpt_path.exists():
        return
    try:
        import torch

        state_dict = torch.load(str(ckpt_path), map_location="cpu")
        query = state_dict.get("model.query_embed.weight")
        if query is not None and len(query.shape) == 2:
            usr_args["chunk_size"] = int(query.shape[0])
            print(f"AUTOSIM inferred chunk_size={usr_args['chunk_size']} from {ckpt_path}")
    except Exception as exc:
        print(f"AUTOSIM checkpoint override skipped: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--ckpt-setting", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--temporal-agg", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args_ns = parser.parse_args()

    repo = Path(args_ns.repo).resolve()
    os.chdir(repo)
    for rel in (".", "policy", "description/utils"):
        p = str(repo / rel)
        if p not in sys.path:
            sys.path.insert(0, p)

    _install_lightweight_import_stubs()

    from envs import CONFIGS_PATH
    import script.eval_policy as robotwin_eval

    usr_args = _load_yaml(repo / "policy" / "ACT" / "deploy_policy.yml")
    usr_args.update({
        "task_name": args_ns.task_name,
        "task_config": args_ns.task_config,
        "ckpt_setting": args_ns.ckpt_setting,
        "ckpt_dir": args_ns.ckpt_dir,
        "seed": args_ns.seed,
        "instruction_type": args_ns.instruction_type,
        "temporal_agg": bool(args_ns.temporal_agg),
        "device": args_ns.device,
    })
    _apply_checkpoint_overrides(usr_args, args_ns.ckpt_dir)

    task_args = _load_yaml(repo / "task_config" / f"{args_ns.task_config}.yml")
    task_args["task_name"] = args_ns.task_name
    task_args["task_config"] = args_ns.task_config
    task_args["ckpt_setting"] = args_ns.ckpt_setting

    embodiment_type = task_args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    embodiment_types = _load_yaml(Path(embodiment_config_path))

    def embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {name}")
        return robot_file

    camera_config = _load_yaml(Path(CONFIGS_PATH) / "_camera_config.yml")
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    task_args["left_embodiment_config"] = robotwin_eval.get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = robotwin_eval.get_embodiment_config(task_args["right_robot_file"])
    task_args["policy_name"] = usr_args["policy_name"]

    task_env = robotwin_eval.class_decorator(task_args["task_name"])
    usr_args["left_arm_dim"] = len(task_args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(task_args["right_embodiment_config"]["arm_joints_name"][1])

    get_model = robotwin_eval.eval_function_decorator(usr_args["policy_name"], "get_model")
    model = get_model(usr_args)
    st_seed = 100000 * (1 + int(args_ns.seed))

    next_seed, success_count = robotwin_eval.eval_policy(
        args_ns.task_name,
        task_env,
        task_args,
        model,
        st_seed,
        test_num=int(args_ns.test_num),
        video_size=None,
        instruction_type=args_ns.instruction_type,
    )
    success_rate = float(success_count) / float(args_ns.test_num) if args_ns.test_num else 0.0
    print("AUTOSIM_EVAL_RESULT=" + json.dumps({
        "success_rate": success_rate,
        "success_count": int(success_count),
        "test_num": int(args_ns.test_num),
        "next_seed": int(next_seed),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
