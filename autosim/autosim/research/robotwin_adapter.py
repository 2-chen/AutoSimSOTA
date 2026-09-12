"""Evidence-backed RoboTwin 2.0 adapter and capability probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .common import digest, object_digest
from .contracts import CapabilityRecord


class RoboTwinAdapter:
    benchmark = "RoboTwin"

    def __init__(self, repo: Path):
        self.repo = repo.absolute()
        required = [self.repo / "README.md", self.repo / "collect_data.sh",
                    self.repo / "scripts/collect_data.py",
                    self.repo / "env_cfg/task_config/demo_randomized.yml"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"repository is not a RoboTwin 2.0 checkout; missing {missing}")

    def tasks(self) -> list[str]:
        return sorted(path.stem for path in (self.repo / "envs").glob("*.py")
                      if not path.name.startswith("_"))

    def select_task(self, requested: str) -> str:
        task = "beat_block_hammer" if requested == "auto" else requested
        if task not in self.tasks():
            raise ValueError(f"unknown RoboTwin task: {task}")
        return task

    @staticmethod
    def _tree_inventory(root: Path) -> dict[str, Any]:
        files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
        rows = [(str(path.relative_to(root)), path.stat().st_size) for path in files]
        return {"path": str(root), "file_count": len(rows),
                "total_bytes": sum(size for _, size in rows),
                "manifest_sha256": object_digest(rows)}

    def _task_data_inventory(self, task: str, embodiment: str) -> dict[str, Any]:
        root = self.repo / "data/demo_clean" / task / embodiment.replace("-", "_").lower()
        groups = {name: sorted((root / name).glob(pattern)) for name, pattern in
                  (("data", "episode_*.hdf5"), ("video", "episode_*.mp4"),
                   ("instruction", "episode_*.json"))}
        rows = [(str(path.relative_to(root)), path.stat().st_size)
                for paths in groups.values() for path in paths]
        return {"path": str(root), "episode_count": len(groups["data"]),
                "video_count": len(groups["video"]),
                "instruction_count": len(groups["instruction"]),
                "total_bytes": sum(size for _, size in rows),
                "manifest_sha256": object_digest(sorted(rows)),
                "contiguous_episode_names": [path.name for path in groups["data"]] ==
                    [f"episode_{index:07d}.hdf5" for index in range(len(groups["data"]))]}

    def discover(self, task: str) -> dict[str, Any]:
        config_path = self.repo / "env_cfg/task_config/demo_randomized.yml"
        step_path = self.repo / "env_cfg/task_config/_eval_step_limit.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        limits = yaml.safe_load(step_path.read_text(encoding="utf-8"))
        task_source = self.repo / "envs" / f"{task}.py"
        embodiment = config["embodiment"]
        embodiment_name = embodiment[0] if len(embodiment) == 1 else f"{embodiment[0]}_{embodiment[1]}"
        task_data = self._task_data_inventory(task, embodiment_name)
        cameras = ["head_camera"] if config["camera"]["collect_head_camera"] else []
        if config["camera"]["collect_wrist_camera"]:
            cameras += ["left_wrist_camera", "right_wrist_camera"]
        contract = {
            "name": task, "setting": "demo_randomized",
            "embodiment": embodiment, "cameras": cameras,
            "observation_modalities": [key for key, enabled in config["data_type"].items() if enabled],
            "max_episode_steps": int(limits[task]),
            "domain_randomization": config["domain_randomization"],
            "action_semantics": "14D dual-arm joint-position targets: 6 arm joints plus 1 gripper joint per arm",
            "control_frequency_hz": 15 if task_data["episode_count"] else "unknown_until_data_or_native_probe",
        }
        return {
            "schema_version": 1, "benchmark": self.benchmark,
            "recognition": "verified_known_adapter_static_probe",
            "selected_task": task, "task_contract": contract,
            "task_signature": object_digest(contract), "task_count": len(self.tasks()),
            "evidence": {
                "task_source": {"path": str(task_source), "sha256": digest(task_source)},
                "task_config": {"path": str(config_path), "sha256": digest(config_path)},
                "step_limits": {"path": str(step_path), "sha256": digest(step_path)},
                "assets": {name: self._tree_inventory(self.repo / "assets" / name)
                           for name in ("background_texture", "embodiments", "objects")},
                "task_data": task_data,
            },
        }

    def capabilities(self, task: str) -> list[CapabilityRecord]:
        xpolicy_ready = (self.repo / "XPolicyLab/pyproject.toml").is_file()
        discovery = self.discover(task)
        asset_inventory = discovery["evidence"]["assets"]
        asset_ready = all(row["file_count"] > 0 for row in asset_inventory.values())
        data_inventory = discovery["evidence"]["task_data"]
        data_ready = (data_inventory["episode_count"] > 0
                      and data_inventory["episode_count"] == data_inventory["video_count"]
                      == data_inventory["instruction_count"]
                      and data_inventory["contiguous_episode_names"])
        native_probe_data = sorted((self.repo / "autoresearch_runs/native_probe_data").glob(
            f"*/{task}/*/data/episode_*.hdf5"))
        native_probe_videos = sorted((self.repo / "autoresearch_runs/native_probe_data").glob(
            f"*/{task}/*/video/episode_*.mp4"))
        expert_ready = bool(native_probe_data and native_probe_videos)
        native_eval_results = sorted((self.repo / "eval_result" / task).glob(
            "*/**/_result.txt"))
        native_eval_ready = bool(native_eval_results)
        project_runs = self.repo.parent / "autoresearch_runs" / "RoboTwin"
        training_receipts = sorted(project_runs.glob(
            f"{task}_*/act_train_smoke_receipt.json"))
        trained_checkpoints = sorted(project_runs.glob(
            f"{task}_*/checkpoint/policy_*.ckpt"))
        act_training_ready = bool(training_receipts or trained_checkpoints)
        runtime_receipts = sorted(project_runs.glob("*/runtime_isolation_probe.json"))
        runtime_receipt = runtime_receipts[-1] if runtime_receipts else project_runs / "runtime_isolation_probe.json"
        runtime_ready = False
        if runtime_receipt.is_file():
            import json
            runtime_ready = json.loads(runtime_receipt.read_text(encoding="utf-8")).get("status") == "passed"
        return [
            CapabilityRecord("repository_recognition", task, "verified",
                             {"task_count": len(self.tasks())}),
            CapabilityRecord("official_expert", task, "verified" if expert_ready else "declared",
                             {"entrypoint": str(self.repo / "collect_data.sh"),
                              "native_probe_hdf5": [str(path) for path in native_probe_data],
                              "native_probe_video": [str(path) for path in native_probe_videos]},
                             None if expert_ready else "native bounded collection has not completed"),
            CapabilityRecord("native_evaluation", task,
                             "verified" if xpolicy_ready and native_eval_ready else
                             "declared" if xpolicy_ready else "unsupported",
                             {"entrypoint": str(self.repo / "scripts/eval_policy.sh"),
                              "native_result_files": [str(path) for path in native_eval_results]},
                             None if xpolicy_ready else "pinned XPolicyLab submodule is not initialized"),
            CapabilityRecord("action_unit_and_timing", task,
                             "verified" if expert_ready and native_eval_ready else "unknown",
                             {"dataset_action_semantics": discovery["task_contract"]["action_semantics"],
                              "demonstration_frequency_hz": discovery["task_contract"]["control_frequency_hz"],
                              "native_policy_round_trip": native_eval_ready},
                             limitation=None if expert_ready and native_eval_ready else
                             "dataset contract is known; native server/simulator round-trip remains unverified"),
            CapabilityRecord("embodiment_assets", task,
                             "verified" if asset_ready else "unsupported", asset_inventory,
                             limitation=None if asset_ready else "embodiment assets are not downloaded"),
            CapabilityRecord("training_data", task,
                             "verified" if data_ready else "unsupported", data_inventory,
                             limitation=None if data_ready else "task data are not imported"),
            CapabilityRecord("runtime_environment", task,
                             "verified" if runtime_ready else "unknown",
                             {"receipt": str(runtime_receipt)},
                             limitation=None if runtime_ready else
                             "project-local runtime isolation probe has not passed"),
            CapabilityRecord("act_training", task,
                             "verified" if xpolicy_ready and act_training_ready else
                             "declared" if xpolicy_ready else "unsupported",
                             {"entrypoint": str(self.repo / "XPolicyLab/policy/ACT/train.sh"),
                              "training_receipts": [str(path) for path in training_receipts],
                              "trained_checkpoints": [str(path) for path in trained_checkpoints]},
                             limitation=None if xpolicy_ready and act_training_ready else
                             "no completed ACT training receipt or checkpoint was found"),
        ]
