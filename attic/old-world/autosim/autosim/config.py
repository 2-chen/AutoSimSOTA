"""
Configuration management for AutoSim.

Mirrors AutoSOTA's config.yaml pattern but adapted for embodied AI tasks.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class TaskConfig:
    """Configuration for a specific embodied AI task."""
    task_name: str
    task_type: str  # e.g., "franka_reach", "franka_grasp"
    robot_type: str = "franka"

    # Optimization target
    primary_metric: str = "distance_to_target"
    metric_direction: str = "lower"  # "lower" or "higher"
    baseline_value: Optional[float] = None
    target_improvement_pct: float = 10.0

    # Task parameters
    params: Dict[str, Any] = field(default_factory=dict)

    # Parameter ranges for optimization
    param_ranges: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Configuration for the optimization loop."""
    max_iterations: int = 10
    max_debug_attempts: int = 3
    seed: int = 42

    # Isaac Sim connection
    isaac_host: str = "localhost"
    isaac_port: int = 8766

    # Mode
    mock_mode: bool = False  # Use mock Isaac Sim instead of real


@dataclass
class AutoSimConfig:
    """Top-level configuration."""
    task: TaskConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    output_dir: str = "./output"

    @classmethod
    def from_yaml(cls, path: str) -> "AutoSimConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        task_data = data.get("task", {})
        task = TaskConfig(
            task_name=task_data.get("task_name", "default"),
            task_type=task_data.get("task_type", "franka_reach"),
            robot_type=task_data.get("robot_type", "franka"),
            primary_metric=task_data.get("primary_metric", "distance_to_target"),
            metric_direction=task_data.get("metric_direction", "lower"),
            baseline_value=task_data.get("baseline_value"),
            target_improvement_pct=task_data.get("target_improvement_pct", 10.0),
            params=task_data.get("params", {}),
            param_ranges=task_data.get("param_ranges", {}),
        )

        opt_data = data.get("optimizer", {})
        optimizer = OptimizerConfig(
            max_iterations=opt_data.get("max_iterations", 10),
            max_debug_attempts=opt_data.get("max_debug_attempts", 3),
            seed=opt_data.get("seed", 42),
            isaac_host=opt_data.get("isaac_host", "localhost"),
            isaac_port=opt_data.get("isaac_port", 8766),
            mock_mode=opt_data.get("mock_mode", False),
        )

        return cls(
            task=task,
            optimizer=optimizer,
            output_dir=data.get("output_dir", "./output"),
        )


# Default parameter ranges for common robot types
DEFAULT_PARAM_RANGES = {
    "franka": {
        "joint_1": [-2.8973, 2.8973],
        "joint_2": [-1.7628, 1.7628],
        "joint_3": [-2.8973, 2.8973],
        "joint_4": [-3.0718, -0.0698],
        "joint_5": [-2.8973, 2.8973],
        "joint_6": [-0.0175, 3.7525],
        "joint_7": [-2.8973, 2.8973],
        "gripper_open": [0.0, 0.04],  # 0 = closed, 0.04 = open
    },
}
