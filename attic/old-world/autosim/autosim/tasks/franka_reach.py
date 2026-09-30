"""
Franka robot reaching task.

The Franka robot must move its end-effector as close as possible to a
target position. The optimization parameters are the joint angles.
Success is measured by the Euclidean distance from end-effector to target.

This mirrors RoboTwin's task pattern (load_actors → play_once → check_success)
but executes within Isaac Sim via the MCP interface.
"""

import json
import logging
import re
import numpy as np
from typing import Dict, Any, List, Optional

from autosim.tasks.task_base import BaseTask, TaskResult
from autosim.config import DEFAULT_PARAM_RANGES

logger = logging.getLogger(__name__)

# Default target position for the reaching task
DEFAULT_TARGET = [0.4, 0.2, 0.3]

# Default joint configuration (home position)
DEFAULT_JOINTS = {
    "joint_1": 0.0,
    "joint_2": -0.785,
    "joint_3": 0.0,
    "joint_4": -2.356,
    "joint_5": 0.0,
    "joint_6": 1.571,
    "joint_7": 0.785,
}


class FrankaReachTask(BaseTask):
    """
    Franka robot reaches for a target position.

    Optimization parameters: 7 joint angles + gripper state
    Primary metric: Euclidean distance from end-effector to target (lower is better)
    """

    TASK_TYPE = "franka_reach"

    def __init__(self, client, task_config=None):
        super().__init__(client, task_config)
        self._target_position = DEFAULT_TARGET

    @property
    def param_space(self) -> Dict[str, List[float]]:
        if self.config and self.config.param_ranges:
            return self.config.param_ranges
        return DEFAULT_PARAM_RANGES["franka"]

    def set_target(self, position: List[float]):
        """Set the target position for reaching."""
        self._target_position = position

    # ── Script builders ────────────────────────────────────────────

    def _build_setup_script(self) -> str:
        """Create target marker in the scene."""
        tx, ty, tz = self._target_position
        return f'''
# === AutoSim: FrankaReach Setup ===
import omni.usd
from pxr import UsdGeom, Gf, Sdf
from omni.isaac.core.prims import XFormPrim
import numpy as np

stage = omni.usd.get_context().get_stage()

# Create target marker (small red sphere)
target_path = "/World/Target"
if not stage.GetPrimAtPath(target_path):
    sphere = UsdGeom.Sphere.Define(stage, target_path)
    sphere.GetRadiusAttr().Set(0.03)
    sphere.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])

# Position the target
target_prim = XFormPrim(prim_path=target_path)
target_prim.set_world_pose(position=np.array([{tx}, {ty}, {tz}]))

print("AUTOSIM_SETUP_COMPLETE: target at [{tx}, {ty}, {tz}]")
'''

    def _build_execute_script(self, params: Dict[str, Any]) -> str:
        """Execute the reach task with given joint angles."""
        # Extract joint values from params, falling back to defaults
        joints = {}
        for j in [f"joint_{i}" for i in range(1, 8)]:
            joints[j] = params.get(j, DEFAULT_JOINTS.get(j, 0.0))

        tx, ty, tz = self._target_position

        # Build the script
        return f'''
# === AutoSim: FrankaReach Execute ===
from omni.isaac.core import SimulationContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.nucleus import get_assets_root_path
from pxr import UsdGeom, Gf
import numpy as np
import json

assets_root_path = get_assets_root_path()
stage = omni.usd.get_context().get_stage()

# Ensure Franka exists
if not stage.GetPrimAtPath("/Franka"):
    asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
    add_reference_to_stage(asset_path, "/Franka")

# Initialize physics
simulation_context = SimulationContext()
simulation_context.initialize_physics()

art = Articulation("/Franka")
art.initialize()

# Set joint positions
joint_positions = np.array([
    {joints['joint_1']},   # panda_joint1
    {joints['joint_2']},   # panda_joint2
    {joints['joint_3']},   # panda_joint3
    {joints['joint_4']},   # panda_joint4
    {joints['joint_5']},   # panda_joint5
    {joints['joint_6']},   # panda_joint6
    {joints['joint_7']},   # panda_joint7
    0.0,                   # panda_finger_joint1
    0.0,                   # panda_finger_joint2
])

art.set_joint_positions(joint_positions)
simulation_context.play()

# Step simulation to settle
for _ in range(60):
    simulation_context.step(render=True)

# Get end-effector position (panda_hand link)
ee_prim = stage.GetPrimAtPath("/Franka/panda_link0/panda_link1/panda_link2/panda_link3/panda_link4/panda_link5/panda_link6/panda_link7/panda_hand")
if ee_prim:
    xform = UsdGeom.Xformable(ee_prim)
    world_transform = xform.ComputeLocalToWorldTransform(0)
    ee_pos = world_transform.ExtractTranslation()
    ee_pos = np.array([ee_pos[0], ee_pos[1], ee_pos[2]])
else:
    # Fallback: use link0 as reference
    ee_pos = np.array([0.0, 0.0, 0.5])

# Target position
target_pos = np.array([{tx}, {ty}, {tz}])

# Compute distance
distance = float(np.linalg.norm(ee_pos - target_pos))

# Output metrics as JSON for parsing
result = {{
    "ee_position": ee_pos.tolist(),
    "target_position": target_pos.tolist(),
    "distance_to_target": distance,
    "joint_positions": joint_positions.tolist(),
}}
print("AUTOSIM_RESULT: " + json.dumps(result))

simulation_context.stop()
'''

    def _extract_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Parse AUTOSIM_RESULT from the execution output."""
        # Strategy 1: If 'result' field is a JSON string, parse it directly
        if isinstance(result, dict):
            raw = result.get("result", "")
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict) and "distance_to_target" in data:
                        return {"distance_to_target": data["distance_to_target"]}
                except (json.JSONDecodeError, TypeError):
                    pass

            # Strategy 2: Look for AUTOSIM_RESULT in message field
            message = result.get("message", "")
            if isinstance(message, str):
                match = re.search(r'AUTOSIM_RESULT:\s*(.+)', message)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        if isinstance(data, dict) and "distance_to_target" in data:
                            return {"distance_to_target": data["distance_to_target"]}
                    except (json.JSONDecodeError, TypeError):
                        pass

        # Strategy 3: Search the whole string representation
        output_str = json.dumps(result) if isinstance(result, dict) else str(result)
        match = re.search(r'AUTOSIM_RESULT:\s*(\{.*?\})\s*$', output_str)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "distance_to_target" in data:
                    return {"distance_to_target": data["distance_to_target"]}
            except json.JSONDecodeError:
                pass

        # Strategy 4: Try nested parsing
        if isinstance(result, dict):
            if "distance_to_target" in result:
                return {"distance_to_target": result["distance_to_target"]}
            inner = result.get("result", {})
            if isinstance(inner, dict) and "distance_to_target" in inner:
                return {"distance_to_target": inner["distance_to_target"]}

        logger.warning(f"Could not extract metrics from result: {str(result)[:200]}")
        return {}

    def _check_success(self, metrics: Dict[str, float]) -> bool:
        distance = metrics.get("distance_to_target", float("inf"))
        # Success if within 5cm
        return distance < 0.05
