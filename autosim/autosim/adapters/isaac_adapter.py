"""
Enhanced Isaac Sim Adapter — full AutoSOTA-style optimization interface.

Supports complex robotics tasks in Isaac Sim:
  - reach:        Move end-effector to target position
  - grasp:        Reach + close gripper around an object
  - pick_place:   Full pick-and-place pipeline
  - insertion:    Precision peg-in-hole insertion

Each task generates rich diagnostics for LLM-driven optimization:
  - Per-seed success/failure tracking
  - Error breakdown by failure mode
  - Trajectory statistics
  - Object poses at key moments
  - IK convergence quality

Works in mock mode (analytical FK/IK) and real mode (MCP → Isaac Sim).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from autosim.adapters.base import TaskAdapter, ParamDef, EvalResult
from autosim.mcp_client import MCPClient, ObjectSpec, RobotConfig, CameraConfig


class TaskType(Enum):
    REACH = "reach"
    GRASP = "grasp"
    PICK_PLACE = "pick_place"
    INSERTION = "insertion"


TASK_TYPE_DESCRIPTIONS = {
    TaskType.REACH: "Move robot end-effector to a target 3D position",
    TaskType.GRASP: "Reach, align, and close gripper on a target object",
    TaskType.PICK_PLACE: "Pick object and place at a target location",
    TaskType.INSERTION: "Precision peg-in-hole insertion",
}


@dataclass
class SceneConfig:
    """Configuration for the simulation scene."""
    floor: bool = True
    table: bool = True
    table_height: float = 0.75
    table_size: Tuple[float, float] = (0.8, 1.2)
    lighting: str = "default"      # default, bright, dim, random
    background: str = "default"    # default, lab, warehouse, empty
    camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        position=[1.2, 0.5, 1.0],
        target=[0.3, 0.0, 0.3],
    ))


@dataclass
class TaskDiagnostics:
    """Rich diagnostics from a task evaluation."""
    success_rate: float = 0.0
    num_seeds: int = 0
    num_success: int = 0
    error_breakdown: Dict[str, int] = field(default_factory=dict)
    trajectory_stats: Dict[str, float] = field(default_factory=dict)
    seed_details: List[Dict] = field(default_factory=list)
    key_observations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "success_rate": self.success_rate,
            "num_seeds": self.num_seeds,
            "num_success": self.num_success,
            "error_breakdown": self.error_breakdown,
            "trajectory_stats": self.trajectory_stats,
            "failed_seeds": [s for s in self.seed_details if not s["success"]][:5],
            "key_observations": self.key_observations,
        }


# ═══════════════════════════════════════════════════════════════════
# Parameter space definitions per task type
# ═══════════════════════════════════════════════════════════════════

def get_param_space_for_task(task_type: TaskType,
                             robot_type: str = "franka") -> Dict[str, ParamDef]:
    """Get parameter space for a given task and robot type."""
    common = {
        "seed": ParamDef("seed", 0, (0, 100), "Random seed", "int"),
    }

    if task_type == TaskType.REACH:
        return {
            **common,
            "target_x": ParamDef("target_x", 0.5, (0.2, 0.7), "Target X", "float"),
            "target_y": ParamDef("target_y", 0.0, (-0.3, 0.3), "Target Y", "float"),
            "target_z": ParamDef("target_z", 0.5, (0.3, 0.7), "Target Z", "float"),
            "approach_speed": ParamDef("approach_speed", 0.5, (0.1, 1.0),
                                        "Movement speed factor", "float"),
        }

    elif task_type == TaskType.GRASP:
        return {
            **common,
            "target_x": ParamDef("target_x", 0.5, (0.2, 0.7), "Object X", "float"),
            "target_y": ParamDef("target_y", 0.0, (-0.25, 0.25), "Object Y", "float"),
            "target_z": ParamDef("target_z", 0.35, (0.25, 0.45), "Object Z", "float"),
            "gripper_open_width": ParamDef("gripper_open_width", 0.04,
                                            (0.02, 0.08), "Pre-grasp opening", "float"),
            "grasp_force": ParamDef("grasp_force", 0.5, (0.1, 1.0),
                                     "Grasp force", "float"),
            "approach_distance": ParamDef("approach_distance", 0.08,
                                           (0.03, 0.15), "Approach distance", "float"),
            "lift_height": ParamDef("lift_height", 0.12, (0.05, 0.25),
                                     "Post-grasp lift", "float"),
        }

    elif task_type == TaskType.PICK_PLACE:
        return {
            **common,
            "pick_x": ParamDef("pick_x", 0.5, (0.3, 0.6), "Pick X", "float"),
            "pick_y": ParamDef("pick_y", 0.0, (-0.2, 0.2), "Pick Y", "float"),
            "pick_z": ParamDef("pick_z", 0.35, (0.25, 0.45), "Pick Z", "float"),
            "place_x": ParamDef("place_x", 0.3, (0.1, 0.5), "Place X", "float"),
            "place_y": ParamDef("place_y", -0.3, (-0.4, 0.0), "Place Y", "float"),
            "place_z": ParamDef("place_z", 0.35, (0.25, 0.45), "Place Z", "float"),
            "lift_height": ParamDef("lift_height", 0.15, (0.08, 0.30),
                                     "Lift height", "float"),
            "approach_distance": ParamDef("approach_distance", 0.08,
                                           (0.03, 0.15), "Approach distance", "float"),
        }

    elif task_type == TaskType.INSERTION:
        return {
            **common,
            "hole_x": ParamDef("hole_x", 0.5, (0.35, 0.65), "Hole X", "float"),
            "hole_y": ParamDef("hole_y", 0.0, (-0.15, 0.15), "Hole Y", "float"),
            "hole_z": ParamDef("hole_z", 0.35, (0.30, 0.45), "Hole Z", "float"),
            "insert_depth": ParamDef("insert_depth", 0.03, (0.01, 0.08),
                                      "Insertion depth", "float"),
            "approach_angle_deg": ParamDef("approach_angle_deg", 0.0,
                                            (-15.0, 15.0), "Approach angle off-nominal", "float"),
            "search_radius": ParamDef("search_radius", 0.005, (0.0, 0.02),
                                       "Spiral search radius", "float"),
        }

    return common


# ═══════════════════════════════════════════════════════════════════
# Enhanced Isaac Sim Adapter
# ═══════════════════════════════════════════════════════════════════

class IsaacSimAdapter(TaskAdapter):
    """
    Universal Isaac Sim adapter for robot learning tasks.

    Supports reach, grasp, pick-and-place, and insertion tasks.
    Generates rich diagnostics for LLM-driven closed-loop optimization.

    Usage:
        adapter = IsaacSimAdapter(task_type="grasp", robot_type="franka")
        result = adapter.evaluate({"target_x": 0.5, "target_y": 0.0, ...})
        print(result.score, result.metrics)
    """

    def __init__(
        self,
        repo_path: str = "",
        task_type: str = "reach",
        robot_type: str = "franka",
        host: str = "localhost",
        port: int = 8766,
        mock: bool = True,
        scene_config: Optional[SceneConfig] = None,
        num_seeds: int = 5,
        seed_offset: int = 0,
    ):
        super().__init__(repo_path or str(Path(__file__).parent.parent.parent))
        self.task_type = TaskType(task_type)
        self.robot_type = robot_type
        self.host = host
        self.port = port
        self.mock = mock
        self.scene_config = scene_config or SceneConfig()
        self.num_seeds = num_seeds
        self.seed_offset = seed_offset

        self._client: Optional[MCPClient] = None
        self._scene_ready = False
        self._eval_log: List[Dict] = []

    # ── Client ───────────────────────────────────────────────────

    def _get_client(self) -> MCPClient:
        if self._client is None:
            self._client = MCPClient(
                host=self.host,
                port=self.port,
                mock=self.mock,
                auto_reconnect=True,
            )
            if not self.mock:
                self._client.connect()
        return self._client

    # ── Param Space ──────────────────────────────────────────────

    def get_param_space(self) -> Dict[str, ParamDef]:
        return get_param_space_for_task(self.task_type, self.robot_type)

    # ── Source Files (for LLM analysis) ──────────────────────────

    def get_source_files(self) -> Dict[str, str]:
        return {
            "adapters/isaac_adapter.py": Path(__file__).read_text(),
            "mcp_client.py": (Path(__file__).parent.parent / "mcp_client.py").read_text(),
            "tasks/task_base.py": (Path(__file__).parent.parent / "tasks" / "task_base.py").read_text(),
        }

    def get_primary_metric_name(self) -> str:
        return "success_rate"

    def get_metric_direction(self) -> str:
        return "higher"

    # ══════════════════════════════════════════════════════════════
    # Scene Setup
    # ══════════════════════════════════════════════════════════════

    def setup_scene(self, params: Optional[Dict] = None) -> bool:
        """Set up the simulation scene for the task."""
        if self._scene_ready:
            return True

        client = self._get_client()
        cfg = self.scene_config

        # Create scene
        client.create_scene(
            floor=cfg.floor,
            scene_name=f"autosim_{self.task_type.value}",
        )

        # Add table if needed
        if cfg.table:
            table_x = 0.5
            table_z = cfg.table_height / 2.0
            client.add_object(ObjectSpec(
                prim_type="Cube",
                name="table",
                position=[table_x, 0.0, table_z],
                scale=[cfg.table_size[0], cfg.table_size[1], cfg.table_height],
                color=[0.5, 0.5, 0.5],
                mass=0.0,  # static
            ))

        # Create robot
        robot_pos = [0.0, 0.0, cfg.table_height]
        if "ur" in self.robot_type:
            # UR robots are often ceiling or wall mounted
            robot_pos = [0.5, 0.0, cfg.table_height + 0.5]

        client.create_robot(
            robot_type=self.robot_type,
            position=robot_pos,
        )

        # Set camera
        client.set_camera(
            position=cfg.camera.position,
            target=cfg.camera.target,
        )

        self._scene_ready = True
        return True

    def reset_scene(self):
        """Reset scene for a new evaluation seed."""
        client = self._get_client()
        if self.mock:
            self._scene_ready = False
            self._client = None
        else:
            # In real mode, move robot to home and reset object poses
            client.move_to_joints(self.robot_type,
                                  [0.0, -0.3, 0.0, -2.6, 0.0, 2.9, 0.7],
                                  plan=False)
            client.control_gripper(self.robot_type, open=True)

    def cleanup(self):
        """Clean up the scene."""
        self._scene_ready = False
        if self._client and not self.mock:
            self._client.disconnect()
        self._client = None

    # ══════════════════════════════════════════════════════════════
    # Task Execution
    # ══════════════════════════════════════════════════════════════

    def evaluate(self, params: Dict = None) -> EvalResult:
        """
        Execute the task with given parameters across N seeds.

        For mock mode: uses analytical FK/IK with noise.
        For real mode: uses MCP to control Isaac Sim.

        Returns EvalResult with score (success_rate) and detailed metrics.
        """
        params = params or {}
        seed = int(params.get("seed", self.seed_offset))

        # Set up scene once
        self.setup_scene(params)

        seed_results = []
        error_counts: Dict[str, int] = {}

        for trial in range(self.num_seeds):
            actual_seed = seed + trial

            # Add noise for robustness evaluation
            noisy_params = self._apply_noise(params, actual_seed)

            # Execute a single trial
            result = self._execute_single_trial(noisy_params, actual_seed)

            # Track errors
            if not result["success"]:
                error_type = result.get("error_type", "unknown")
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            seed_results.append(result)

            # Reset between seeds
            if trial < self.num_seeds - 1:
                self.reset_scene()

        # Compute metrics
        success_count = sum(1 for r in seed_results if r["success"])
        success_rate = success_count / self.num_seeds if self.num_seeds > 0 else 0.0

        # Trajectory stats
        avg_distance = float(np.mean([
            r.get("final_distance", 0.0) for r in seed_results
        ]))
        avg_path_length = float(np.mean([
            r.get("path_length", 0.0) for r in seed_results if r.get("path_length")
        ]))

        # Build diagnostics
        diagnostics = TaskDiagnostics(
            success_rate=success_rate,
            num_seeds=self.num_seeds,
            num_success=success_count,
            error_breakdown=error_counts,
            trajectory_stats={
                "avg_final_distance": round(avg_distance, 4),
                "avg_path_length": round(avg_path_length, 4),
            },
            seed_details=seed_results,
            key_observations=self._generate_observations(
                success_rate, error_counts, seed_results
            ),
        )

        # Store for feedback
        self._eval_log.append({
            "params": params,
            "diagnostics": diagnostics.to_dict(),
        })

        return EvalResult(
            score=success_rate,
            success=True,
            metrics={
                "success_rate": success_rate,
                "success_count": success_count,
                "total_seeds": self.num_seeds,
                "avg_distance": round(avg_distance, 4),
                **diagnostics.to_dict(),
            },
            info=json.dumps(diagnostics.to_dict()),
        )

    def _execute_single_trial(self, params: Dict, seed: int) -> Dict:
        """Execute a single trial of the task.

        Returns:
            Dict with success, error_type, final_distance, path_length, etc.
        """
        try:
            if self.task_type == TaskType.REACH:
                return self._execute_reach(params, seed)
            elif self.task_type == TaskType.GRASP:
                return self._execute_grasp(params, seed)
            elif self.task_type == TaskType.PICK_PLACE:
                return self._execute_pick_place(params, seed)
            elif self.task_type == TaskType.INSERTION:
                return self._execute_insertion(params, seed)
            else:
                return {"success": False, "error_type": "unknown_task",
                        "final_distance": 999.0}
        except Exception as e:
            return {"success": False, "error_type": f"exception_{type(e).__name__}",
                    "final_distance": 999.0, "error": str(e)[:200]}

    # ── Task: Reach ──────────────────────────────────────────────

    def _compute_reachability(self, target_pos: np.ndarray) -> float:
        """Estimate reachability difficulty (0=easy, 1=impossible)."""
        # Center of robot workspace
        center = np.array([0.4, 0.0, 0.4])
        max_radius = 0.45

        dist_from_center = float(np.linalg.norm(target_pos - center))

        # Workspace boundary penalty
        if dist_from_center > max_radius:
            return 0.0  # unreachable

        # Low height penalty
        height_penalty = max(0, 0.25 - target_pos[2]) * 3.0

        # Distance penalty
        dist_penalty = dist_from_center / max_radius

        return float(np.clip(1.0 - height_penalty - dist_penalty * 0.3, 0.0, 1.0))

    def _execute_reach(self, params: Dict, seed: int) -> Dict:
        """Move end-effector to target position."""
        client = self._get_client()
        target = np.array([
            float(params.get("target_x", 0.5)),
            float(params.get("target_y", 0.0)),
            float(params.get("target_z", 0.5)),
        ])
        speed = float(params.get("approach_speed", 0.5))
        rng = np.random.RandomState(seed)

        if self.mock:
            # Reachability model
            reachability = self._compute_reachability(target)
            if reachability < 0.1:
                return {"success": False, "error_type": "unreachable",
                        "final_distance": 0.5, "reachability": reachability}

            # Analytical IK
            q_start = self._get_home_with_noise(seed)
            q_sol, ee_pos, ik_success = self._mock_ik(
                target, self.robot_type, q_start
            )
            final_dist = float(np.linalg.norm(ee_pos - target))

            # Speed affects accuracy (faster = less precise)
            speed_noise = (1.0 - speed) * 0.01 + rng.uniform(0, 0.02 * (1.0 - speed))
            final_dist += speed_noise

            # Reachability affects success probability
            success_prob = reachability * (1.0 - min(final_dist / 0.08, 1.0))
            success = rng.uniform() < success_prob

            return {
                "success": success,
                "error_type": "" if success else ("ik_failure" if not ik_success else "position_error"),
                "final_distance": final_dist,
                "ee_position": ee_pos.tolist(),
                "target": target.tolist(),
                "path_length": float(np.linalg.norm(ee_pos - np.array([0.5, 0.0, 0.5]))),
                "joint_positions": q_sol.tolist(),
                "reachability": round(reachability, 4),
            }
        else:
            # Real Isaac Sim
            client.move_to_joints(self.robot_type,
                                   [0, -0.3, 0, -2.6, 0, 2.9, 0.7], plan=False)
            client.control_gripper(self.robot_type, open=True)

            target_pose = target.tolist() + [0, 1, 0, 0]  # x,y,z,qx,qy,qz,qw
            result = client.move_to_pose(self.robot_type, target_pose, plan=True)

            ee_pose = client.get_end_effector_pose(self.robot_type)
            ee_pos = ee_pose.get("position", [0, 0, 0])
            final_dist = float(np.linalg.norm(np.array(ee_pos) - target))
            success = result.get("success", False) and final_dist < 0.05

            return {
                "success": success,
                "error_type": "" if success else "move_failed",
                "final_distance": final_dist,
                "ee_position": ee_pos,
                "target": target.tolist(),
            }

    # ── Task: Grasp ──────────────────────────────────────────────

    def _execute_grasp(self, params: Dict, seed: int) -> Dict:
        """Reach, align, and close gripper on target object."""
        client = self._get_client()
        target = np.array([
            float(params.get("target_x", 0.5)),
            float(params.get("target_y", 0.0)),
            float(params.get("target_z", 0.35)),
        ])
        gripper_width = float(params.get("gripper_open_width", 0.04))
        approach_dist = float(params.get("approach_distance", 0.08))
        lift_height = float(params.get("lift_height", 0.12))
        grasp_force = float(params.get("grasp_force", 0.5))

        grasp_prob = 0.0
        reachability = 0.0
        error_type = ""
        phases = {}
        final_dist = 0.0

        # Phase 1: Pre-grasp (above target)
        pre_pos = [target[0], target[1], target[2] + approach_dist + lift_height]

        if self.mock:
            rng = np.random.RandomState(seed)

            # Reachability check
            reachability = self._compute_reachability(target)
            if reachability < 0.15:
                return {"success": False, "error_type": "unreachable",
                        "final_distance": 0.5}

            # IK check
            q_start = self._get_home_with_noise(seed)
            _, _, ik_ok = self._mock_ik(target + np.array([0, 0, approach_dist + lift_height]),
                                         self.robot_type, q_start)
            if not ik_ok:
                return {"success": False, "error_type": "ik_failure_pre_grasp",
                        "final_distance": 0.5}

            # Grasp success model:
            # - gripper_width too large → can't close around object
            # - approach_dist too small → bump object
            # - grasp_force too low → drop
            # - target position near boundary → less stable
            width_penalty = abs(gripper_width - 0.04) * 5.0  # 0.04 is optimal
            approach_penalty = max(0, 0.02 - approach_dist) * 3.0
            force_benefit = min(grasp_force / 0.5, 1.0) * 0.3
            boundary_penalty = (1.0 - reachability) * 0.4

            grasp_prob = 0.6 + force_benefit - width_penalty - approach_penalty - boundary_penalty
            grasp_prob = float(np.clip(grasp_prob, 0.0, 0.95))
            error_type = ""
            grasp_success = rng.uniform() < grasp_prob

            # Lift after grasp
            lift_target = [target[0], target[1], target[2] + lift_height + 0.1]
            _, ee_pos_lift, ik_ok3 = self._mock_ik(
                np.array(lift_target), self.robot_type, q_start
            )
            lift_ok = ik_ok3

            # Dropping during lift (grasp_force too low)
            drop_during_lift = (not grasp_success) or (grasp_force < 0.2 and rng.uniform() < 0.4)

            success = grasp_success and lift_ok and not drop_during_lift
            final_dist = float(np.linalg.norm(ee_pos_lift - np.array(lift_target))) if lift_ok else 0.3

            error_type = ""
            if not success:
                if not lift_ok:
                    error_type = "ik_failure_lift"
                elif not grasp_success:
                    error_type = "grasp_failed"
                else:
                    error_type = "dropped_during_lift"

        else:
            # Real Isaac Sim
            try:
                client.move_to_joints(self.robot_type,
                                       [0, -0.3, 0, -2.6, 0, 2.9, 0.7], plan=False)
                client.control_gripper(self.robot_type, open=True, width=gripper_width)

                pre_pose = pre_pos + [0, 1, 0, 0]
                phases["pre_grasp"] = client.move_to_pose(self.robot_type, pre_pose, plan=True)

                grasp_target = [target[0], target[1], target[2] + 0.02]
                phases["approach"] = client.move_to_pose(
                    self.robot_type, grasp_target + [0, 1, 0, 0], plan=True
                )

                phases["grasp"] = client.control_gripper(self.robot_type, open=False)
                client.step_simulation(30)

                lift_target = [target[0], target[1], target[2] + lift_height + 0.1]
                phases["lift"] = client.move_to_pose(
                    self.robot_type, lift_target + [0, 1, 0, 0], plan=True
                )

                ee_pose = client.get_end_effector_pose(self.robot_type)
                ee_pos = ee_pose.get("position", [0, 0, 0])
                final_dist = float(np.linalg.norm(np.array(ee_pos) - np.array(lift_target)))
                success = True

            except (ConnectionError, TimeoutError, RuntimeError) as e:
                return {"success": False, "error_type": f"execution_error",
                        "final_distance": 999.0, "error": str(e)[:200]}

        return {
            "success": success,
            "error_type": error_type,
            "final_distance": final_dist,
            "phases": phases if not self.mock else {},
            "reachability": round(reachability, 4),
            "grasp_probability": round(grasp_prob, 4),
        }

    def _execute_pick_place(self, params: Dict, seed: int) -> Dict:
        """Full pick-and-place task."""
        client = self._get_client()
        pick = np.array([
            float(params.get("pick_x", 0.5)),
            float(params.get("pick_y", 0.0)),
            float(params.get("pick_z", 0.35)),
        ])
        place = np.array([
            float(params.get("place_x", 0.3)),
            float(params.get("place_y", -0.3)),
            float(params.get("place_z", 0.35)),
        ])
        lift_height = float(params.get("lift_height", 0.15))
        approach_dist = float(params.get("approach_distance", 0.08))

        if self.mock:
            rng = np.random.RandomState(seed)

            # Reachability for both pick and place
            pick_reach = self._compute_reachability(pick)
            place_reach = self._compute_reachability(place)

            if pick_reach < 0.1:
                return {"success": False, "error_type": "pick_unreachable",
                        "final_distance": 0.5}
            if place_reach < 0.1:
                return {"success": False, "error_type": "place_unreachable",
                        "final_distance": 0.3}

            # Pick success (similar to grasp model)
            approach_penalty = max(0, 0.04 - approach_dist) * 3.0
            boundary_penalty = (1.0 - pick_reach) * 0.4
            pick_prob = float(np.clip(0.85 - approach_penalty - boundary_penalty, 0.0, 0.95))
            pick_ok = rng.uniform() < pick_prob

            # Lift stability
            lift_penalty = max(0, lift_height - 0.20) * 3.0 + max(0, 0.05 - lift_height) * 2.0
            lift_ok = rng.uniform() < 0.9 or (lift_height > 0.08 and lift_height < 0.22)

            # Place success
            place_dist = float(np.linalg.norm(pick - place))
            place_difficulty = place_dist / 0.8  # normalized by max travel
            place_boundary = (1.0 - place_reach) * 0.5
            place_prob = float(np.clip(0.85 - place_difficulty * 0.3 - place_boundary, 0.0, 0.95))
            place_ok = rng.uniform() < place_prob

            success = pick_ok and lift_ok and place_ok
            error_type = ""
            if not success:
                if not pick_ok:
                    error_type = "pick_failed"
                elif not lift_ok:
                    error_type = "drop_during_lift"
                else:
                    error_type = "place_failed"

            return {
                "success": success,
                "error_type": error_type,
                "final_distance": float(np.linalg.norm(pick - place)),
                "pick_probability": round(pick_prob, 4),
                "place_probability": round(place_prob, 4),
            }
        else:
            try:
                result = client.pick_and_place(
                    robot_name=self.robot_type,
                    object_name="grasp_object",
                    target_position=place.tolist(),
                    lift_height=lift_height,
                    approach_distance=approach_dist,
                )
                return {
                    "success": result.get("success", False),
                    "error_type": "" if result.get("success") else "pick_place_failed",
                    "final_distance": 0.0,
                    "phases": result.get("phases", {}),
                }
            except Exception as e:
                return {"success": False, "error_type": "execution_error",
                        "final_distance": 999.0, "error": str(e)[:200]}

    # ── Task: Insertion ──────────────────────────────────────────

    def _execute_insertion(self, params: Dict, seed: int) -> Dict:
        """Precision peg-in-hole insertion."""
        client = self._get_client()
        hole = np.array([
            float(params.get("hole_x", 0.5)),
            float(params.get("hole_y", 0.0)),
            float(params.get("hole_z", 0.35)),
        ])
        insert_depth = float(params.get("insert_depth", 0.03))
        approach_angle = float(params.get("approach_angle_deg", 0.0))
        search_radius = float(params.get("search_radius", 0.005))

        if self.mock:
            # Mock: check IK reachability with alignment
            q_start = self._get_home_with_noise(seed)
            _, ee_pos, ik_ok = self._mock_ik(hole, self.robot_type, q_start)
            if not ik_ok:
                return {"success": False, "error_type": "ik_failure",
                        "final_distance": 0.5}

            # Success depends on search_radius (larger = easier) and angle (smaller = easier)
            angle_penalty = abs(approach_angle) / 15.0  # 0-1
            search_help = min(search_radius / 0.01, 1.0)  # 0-1
            success_prob = 0.3 + 0.5 * search_help - 0.3 * angle_penalty
            success = np.random.RandomState(seed).uniform() < success_prob

            return {
                "success": success,
                "error_type": "" if success else "failed_alignment",
                "final_distance": float(np.linalg.norm(ee_pos - hole)),
            }
        else:
            # Real mode: complex insertion trajectory
            above_hole = [hole[0], hole[1], hole[2] + 0.1]

            try:
                client.move_to_joints(self.robot_type,
                                       [0, -0.3, 0, -2.6, 0, 2.9, 0.7], plan=False)

                # Move above hole
                client.move_to_pose(self.robot_type,
                                    above_hole + [0, 1, 0, 0], plan=True)

                # Move to insertion start
                insert_start = [hole[0], hole[1], hole[2] + 0.02]
                client.move_to_pose(self.robot_type,
                                    insert_start + [0, 1, 0, 0], plan=False)

                # Insert
                insert_target = [hole[0], hole[1], hole[2] - insert_depth]
                client.move_to_pose(self.robot_type,
                                    insert_target + [0, 1, 0, 0], plan=False)

                ee_pose = client.get_end_effector_pose(self.robot_type)
                ee_pos = ee_pose.get("position", [0, 0, 0])
                final_dist = float(np.linalg.norm(np.array(ee_pos) - np.array(insert_target)))
                success = final_dist < 0.02

                return {
                    "success": success,
                    "error_type": "" if success else "insertion_failed",
                    "final_distance": final_dist,
                }
            except Exception as e:
                return {"success": False, "error_type": "execution_error",
                        "final_distance": 999.0, "error": str(e)[:200]}

    # ══════════════════════════════════════════════════════════════
    # Mock Helpers
    # ══════════════════════════════════════════════════════════════

    def _get_home_with_noise(self, seed: int) -> np.ndarray:
        """Get home joint positions with slight noise for seed variation."""
        rng = np.random.RandomState(seed)
        home = {
            "franka": [0.0, -0.3, 0.0, -2.6, 0.0, 2.9, 0.7],
            "ur5": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
        }.get(self.robot_type, [0.0, -0.3, 0.0, -2.6, 0.0, 2.9, 0.7])
        noise = rng.normal(0, 0.05, len(home))
        return np.array(home) + noise

    def _mock_ik(self, target_pos: np.ndarray, robot_type: str,
                 q_start: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Compute mock IK."""
        from autosim.mock.mock_isaac import compute_ik_numerical
        return compute_ik_numerical(target_pos, robot_type, q_start)

    def _apply_noise(self, params: Dict, seed: int) -> Dict:
        """Add small noise to params for robustness evaluation."""
        rng = np.random.RandomState(seed)
        noisy = dict(params)
        for key in params:
            if key in ("target_x", "pick_x", "hole_x"):
                noisy[key] = params[key] + rng.uniform(-0.02, 0.02)
            elif key in ("target_y", "pick_y", "hole_y"):
                noisy[key] = params[key] + rng.uniform(-0.02, 0.02)
            elif key in ("target_z", "pick_z", "hole_z"):
                noisy[key] = params[key] + rng.uniform(-0.01, 0.01)
        return noisy

    # ══════════════════════════════════════════════════════════════
    # Diagnostics & Observations
    # ══════════════════════════════════════════════════════════════

    def _generate_observations(
        self,
        success_rate: float,
        error_breakdown: Dict[str, int],
        seed_results: List[Dict],
    ) -> List[str]:
        """Generate human-readable observations from evaluation results."""
        obs = []

        # Overall performance
        obs.append(f"Success rate: {success_rate * 100:.0f}% "
                   f"({sum(1 for r in seed_results if r['success'])}/{len(seed_results)})")

        # Error analysis
        if error_breakdown:
            main_error = max(error_breakdown, key=error_breakdown.get)
            obs.append(f"Main failure mode: {main_error} "
                       f"({error_breakdown[main_error]}/{len(seed_results)} seeds)")

        # Per-type observations
        if success_rate == 0:
            obs.append("CRITICAL: All seeds failed — task may be impossible in current config")
        elif success_rate > 0.8:
            obs.append("Good performance — parameters are well-tuned")
        elif success_rate > 0.5:
            obs.append("Moderate performance — room for improvement")

        # Distance stats
        distances = [r.get("final_distance", 0) for r in seed_results
                     if "final_distance" in r]
        if distances:
            avg_dist = float(np.mean(distances))
            obs.append(f"Average final distance: {avg_dist:.4f}")

        # IK failures
        ik_fails = sum(1 for r in seed_results if "ik" in r.get("error_type", ""))
        if ik_fails > 0:
            obs.append(f"IK failures: {ik_fails}/{len(seed_results)} — "
                       "target may be unreachable")

        return obs

    def get_last_diagnostics(self) -> Optional[Dict]:
        """Get the diagnostics from the most recent evaluation."""
        if self._eval_log:
            return self._eval_log[-1].get("diagnostics")
        return None

    def get_eval_history(self) -> List[Dict]:
        """Get full evaluation history."""
        return list(self._eval_log)

    def summarize_experience(self) -> str:
        """Summarize all evaluations for LLM feedback context."""
        if not self._eval_log:
            return "No evaluations yet."

        lines = ["## Isaac Sim Evaluation History", ""]
        for i, entry in enumerate(self._eval_log):
            diag = entry.get("diagnostics", {})
            lines.append(f"### Trial {i + 1}")
            lines.append(f"- Params: {json.dumps(entry.get('params', {}))}")
            lines.append(f"- Success rate: {diag.get('success_rate', 0) * 100:.0f}%")
            lines.append(f"- Error breakdown: {diag.get('error_breakdown', {})}")
            for obs in diag.get("key_observations", []):
                lines.append(f"  - {obs}")
            lines.append("")

        return "\n".join(lines)
