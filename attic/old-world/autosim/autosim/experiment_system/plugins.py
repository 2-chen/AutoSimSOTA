"""Read-only native benchmark discovery. No simulator imported during inventory."""
from dataclasses import asdict
from pathlib import Path

from autosim.research.common import digest, read_json
from .contracts import TaskContract


ROBOTWIN_TASKS = ("pick_dual_bottles", "stack_blocks_two", "open_laptop")


class RoboSynPlugin:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.repo = self.workspace / "AutoSimSOTA/RoboSynChallenge"

    def contract(self, task):
        from autosim.research.registry import load_task

        spec = load_task(self.repo, task)
        action = read_json(Path(spec.action_config))
        action = action.get("action_config", action)
        units = []
        for part in spec.control_parts:
            units.extend(["rad" if part.endswith("arm") else "unverified_native_joint_unit"] * int(action["scope"][part]["dim"][0]))
        sources = dict(spec.config_hashes)
        for path in (self.repo / "policy/act/deploy_policy.py", self.repo / "scripts/eval_policy.py"):
            sources[str(path.absolute())] = digest(path)
        return TaskContract("robosyn", task,
            {"state_dim": spec.state_dim, "cameras": spec.camera_shapes, "image_format": "HWC_RGB"},
            {"dimension": spec.action_dim, "mode": "absolute_joint_target", "units": units,
             "order": list(spec.control_parts), "frame": "robot_joint_coordinates"},
            {"max_actions": spec.max_episode_steps, "recorded_hz": spec.recorded_fps,
             "control_hz": "requires_runtime_probe"},
            {"native_entry": "scripts/eval_policy.py", "success_authority": "official_task",
             "setting": "random", "mutation_allowed": False}, sources).validate()

    def inventory(self, task):
        contract = self.contract(task)
        from autosim.research.registry import load_task
        spec = load_task(self.repo, task)
        return {"contract": asdict(contract), "signature": contract.signature,
                "configuration_loaded": True, "semantic_probe": "pending",
                "expert_collection": "requires_current_version_probe",
                "correction_collection": "unsupported", "policy_training": ["act", "dp"],
                "official_data_metadata_present": (self.repo / "lerobot_dataset/RoboSynChallenge" / f"cobotmagic_Sim_{task}/meta/info.json").is_file(),
                "native_policy_contract": spec.as_dict(), "full_workflow_validated": False}


class RoboTwinPlugin:
    def __init__(self, repo):
        self.repo = Path(repo)

    def contract(self, task):
        import yaml
        if task not in ROBOTWIN_TASKS:
            raise ValueError("task outside predeclared transfer suite")
        config_paths = [self.repo / "task_config" / name for name in
                        ("demo_randomized.yml", "_camera_config.yml", "_eval_step_limit.yml", "_embodiment_config.yml")]
        config, cameras, limits, _ = [yaml.safe_load(p.read_text()) for p in config_paths]
        shapes = {name: [cameras[config["camera"][kind]]["h"], cameras[config["camera"][kind]]["w"], 3]
                  for name, kind in (("head_camera", "head_camera_type"), ("left_camera", "wrist_camera_type"), ("right_camera", "wrist_camera_type"))}
        sources = config_paths + [self.repo / "envs" / f"{task}.py", self.repo / "envs/_base_task.py",
                                  self.repo / "envs/robot/robot.py", self.repo / "script/eval_policy.py"]
        return TaskContract("robotwin", task,
            {"state_dim": 14, "cameras": shapes, "image_format": "HWC_RGB"},
            {"dimension": 14, "mode": "absolute_joint_target_with_TOPP", "units": (["rad"] * 6 + ["normalized_gripper"]) * 2,
             "order": ["left_arm", "left_gripper", "right_arm", "right_gripper"], "frame": "robot_joint_coordinates"},
            {"max_actions": int(limits[task]), "recorded_hz": "variable_native_save_frequency",
             "control_hz": "variable_TOPP_duration"},
            {"native_entry": "script/eval_policy.py", "success_authority": "official_task",
             "setting": "demo_randomized", "mutation_allowed": False},
            {str(p.absolute()): digest(p) for p in sources}).validate()

    def inventory(self, task):
        contract = self.contract(task)
        return {"contract": asdict(contract), "signature": contract.signature,
                "configuration_loaded": True, "semantic_probe": "pending", "native_evaluation_parity": "pending",
                "expert_collection": "requires_native_probe", "full_workflow_validated": False}
