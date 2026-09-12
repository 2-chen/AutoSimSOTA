"""
Enhanced Mock Isaac Sim — supports all MCPClient commands for offline testing.

Simulates:
  - Scene management (objects, physics)
  - Robot kinematics (Franka FK, UR5)
  - Camera, diagnostics
  - Motion planning (IK, path planning)
  - Domain randomization
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Forward Kinematics
# ═══════════════════════════════════════════════════════════════════

# Franka DH parameters (modified DH)
FRANKA_DH = [
    (0.0, 0.0, 0.333, 0.0),
    (0.0, -np.pi / 2, 0.0, 0.0),
    (0.0, np.pi / 2, 0.316, 0.0),
    (0.0825, np.pi / 2, 0.0, 0.0),
    (-0.0825, -np.pi / 2, 0.384, 0.0),
    (0.0, np.pi / 2, 0.0, 0.0),
    (0.088, np.pi / 2, 0.0, 0.0),
]
EE_OFFSET = np.array([0.0, 0.0, 0.107])

# UR5 DH parameters
UR5_DH = [
    (0.0, 0.0, 0.089159, 0.0),
    (0.0, -np.pi / 2, 0.0, 0.0),
    (0.425, 0.0, 0.0, 0.0),
    (0.39225, 0.0, 0.0, 0.0),
    (0.0, -np.pi / 2, 0.09465, 0.0),
    (0.0, np.pi / 2, 0.0823, 0.0),
]

JOINT_LIMITS = {
    "franka": [
        (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
        (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    ],
    "ur5": [
        (-6.283, 6.283), (-6.283, 6.283), (-6.283, 6.283),
        (-6.283, 6.283), (-6.283, 6.283), (-6.283, 6.283),
    ],
}

DEFAULT_HOME = {
    "franka": [0.0, -0.3, 0.0, -2.6, 0.0, 2.9, 0.7],
    "ur5": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
}


def forward_kinematics(dh_params: List[Tuple[float, float, float, float]],
                       joint_angles: List[float],
                       ee_offset: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute end-effector position via DH forward kinematics."""
    T = np.eye(4)
    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        if i >= len(joint_angles):
            break
        theta_total = joint_angles[i] + theta_offset
        ct, st = np.cos(theta_total), np.sin(theta_total)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T_i = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d],
            [0,   0,        0,       1],
        ])
        T = T @ T_i
    if ee_offset is not None:
        return T[:3, 3] + T[:3, :3] @ ee_offset
    return T[:3, 3]


def franka_fk(joint_angles: List[float]) -> np.ndarray:
    """Franka FK: 7 joint angles → [x, y, z] end-effector position."""
    return forward_kinematics(FRANKA_DH, joint_angles[:7], EE_OFFSET)


def ur5_fk(joint_angles: List[float]) -> np.ndarray:
    """UR5 FK: 6 joint angles → [x, y, z] end-effector position."""
    return forward_kinematics(UR5_DH, joint_angles[:6])


def compute_ik_numerical(target_pos: np.ndarray, robot_type: str,
                         q_start: Optional[np.ndarray] = None,
                         steps: int = 300, lr: float = 0.3,
                         tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Numerical IK via damped least squares."""
    if robot_type == "franka":
        dh = FRANKA_DH
        limits = JOINT_LIMITS["franka"]
        n_joints = 7
    elif robot_type == "ur5":
        dh = UR5_DH
        limits = JOINT_LIMITS["ur5"]
        n_joints = 6
    else:
        dh = FRANKA_DH
        limits = JOINT_LIMITS["franka"]
        n_joints = 7

    fk_func = lambda q: forward_kinematics(dh, q)

    if q_start is None:
        q_start = np.array(DEFAULT_HOME.get(robot_type, DEFAULT_HOME["franka"])[:n_joints])
    q = q_start.astype(np.float64).copy()

    for _ in range(steps):
        ee = fk_func(q)
        err = target_pos - ee
        if np.linalg.norm(err) < tol:
            break

        # Numerical Jacobian
        J = np.zeros((3, n_joints))
        ee0 = fk_func(q)
        for i in range(n_joints):
            qp = q.copy()
            qp[i] += 1e-4
            J[:, i] = (fk_func(qp) - ee0) / 1e-4

        # Damped least squares
        M = J @ J.T + 0.01 * np.eye(3)
        try:
            dq = lr * J.T @ np.linalg.solve(M, err)
        except np.linalg.LinAlgError:
            break
        dq = np.clip(dq, -0.3, 0.3)

        for i in range(n_joints):
            lo, hi = limits[i]
            q[i] = np.clip(q[i] + dq[i], lo, hi)

    ee_final = fk_func(q)
    success = bool(np.linalg.norm(ee_final - target_pos) < tol)
    return q, ee_final, success


# ═══════════════════════════════════════════════════════════════════
# Mock Simulator
# ═══════════════════════════════════════════════════════════════════

class MockIsaacSim:
    """Mock Isaac Sim that simulates the full MCP API."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._scene_ready = False
        self._scene_name = "autosim_scene"
        self._objects: Dict[str, Dict] = {}
        self._robots: Dict[str, Dict] = {}
        self._robot_joints: Dict[str, List[float]] = {}
        self._physics = {
            "gravity": [0.0, 0.0, -9.81],
            "dt": 1.0 / 60.0,
            "substeps": 2,
            "simulating": False,
        }
        self._cameras: Dict[str, Dict] = {
            "/Camera": {
                "position": [1.5, 0.5, 1.0],
                "target": [0.0, 0.0, 0.0],
            }
        }
        self._joint_noise = 0.001
        self._next_id = 0

    def handle_command(self, cmd_type: str,
                       params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch command to handler."""
        handler_map = {
            # Scene
            "get_scene_info": self._get_scene_info,
            "create_physics_scene": self._create_physics_scene,
            "reset_scene": self._reset_scene,
            "add_object": self._add_object,
            "remove_object": self._remove_object,
            "set_object_pose": self._set_object_pose,
            "get_object_pose": self._get_object_pose,
            # Robot
            "create_robot": self._create_robot,
            "set_joint_positions": self._set_joint_positions,
            "get_joint_positions": self._get_joint_positions,
            "control_gripper": self._control_gripper,
            "set_robot_pose": self._set_robot_pose,
            "get_robot_pose": self._get_robot_pose,
            "get_end_effector_pose": self._get_end_effector_pose,
            # Camera
            "set_camera": self._set_camera,
            "capture_image": self._capture_image,
            "capture_depth": self._capture_depth,
            "get_camera_info": self._get_camera_info,
            # Physics
            "apply_force": self._apply_force,
            "set_gravity": self._set_gravity,
            "step_simulation": self._step_simulation,
            "pause_simulation": self._pause_simulation,
            "resume_simulation": self._resume_simulation,
            "get_physics_context": self._get_physics_context,
            # Motion planning
            "compute_ik": self._compute_ik,
            "plan_path": self._plan_path,
            "execute_trajectory": self._execute_trajectory,
            # Diagnostics
            "get_transform": self._get_transform,
            "get_bounding_box": self._get_bounding_box,
            "check_collision": self._check_collision,
            "get_distance": self._get_distance,
            "ray_cast": self._ray_cast,
            "get_prim_paths": self._get_prim_paths,
            # Domain randomization
            "randomize_lighting": self._randomize_lighting,
            "randomize_textures": self._randomize_textures,
            "randomize_pose": self._randomize_pose,
            # Script
            "execute_script": self._execute_script,
            # Utility
            "get_sim_performance": self._get_sim_performance,
            "get_asset_root": self._get_asset_root,
            "omni_kit_command": self._omni_kit_command,
        }
        handler = handler_map.get(cmd_type)
        if handler:
            try:
                return handler(params)
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "error", "message": f"Unknown command: {cmd_type}"}

    # ── Helpers ──────────────────────────────────────────────────

    def _next_name(self, prefix: str = "obj") -> str:
        self._next_id += 1
        return f"{prefix}_{self._next_id}"

    def _robot_info(self, robot_type: str) -> Dict:
        return {
            "joints": 7 if "franka" in robot_type else 6,
            "home": DEFAULT_HOME.get(robot_type, DEFAULT_HOME["franka"]),
            "limits": JOINT_LIMITS.get(robot_type, JOINT_LIMITS["franka"]),
        }

    def _get_fk_func(self, robot_type: str):
        if "franka" in robot_type:
            return franka_fk
        elif "ur5" in robot_type or "ur10" in robot_type:
            return ur5_fk
        return franka_fk

    # ── Scene ────────────────────────────────────────────────────

    def _get_scene_info(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "message": "pong" if self._scene_ready else "scene not ready",
            "scene_name": self._scene_name,
            "objects_count": len(self._objects),
            "robots_count": len(self._robots),
            "simulating": self._physics["simulating"],
        }

    def _create_physics_scene(self, params: Dict) -> Dict:
        self._scene_ready = True
        self._scene_name = params.get("scene_name", "autosim_scene")
        if "gravity" in params:
            self._physics["gravity"] = params["gravity"]
        return {
            "status": "success",
            "message": f"Created physics scene: {self._scene_name}",
            "scene_name": self._scene_name,
        }

    def _reset_scene(self, params: Dict) -> Dict:
        self._objects.clear()
        self._robots.clear()
        self._robot_joints.clear()
        self._scene_ready = True
        return {"status": "success", "message": "Scene reset"}

    def _add_object(self, params: Dict) -> Dict:
        name = params.get("name", self._next_name())
        obj = {
            "prim_type": params.get("prim_type", "Cube"),
            "name": name,
            "position": params.get("position", [0.0, 0.0, 0.0]),
            "scale": params.get("scale", [1.0, 1.0, 1.0]),
            "color": params.get("color", [0.5, 0.5, 0.5]),
            "mass": params.get("mass", 0.0),
            "collision": params.get("collision", True),
            "usd_path": params.get("usd_path", ""),
        }
        self._objects[name] = obj
        return {
            "status": "success",
            "message": f"Created {obj['prim_type']}: {name}",
            "name": name,
            "prim_path": f"/World/{name}",
        }

    def _remove_object(self, params: Dict) -> Dict:
        name = params.get("name", "")
        self._objects.pop(name, None)
        return {"status": "success", "message": f"Removed: {name}"}

    def _set_object_pose(self, params: Dict) -> Dict:
        name = params.get("name", "")
        if name in self._objects:
            self._objects[name]["position"] = params.get("position",
                                                         self._objects[name]["position"])
        return {"status": "success"}

    def _get_object_pose(self, params: Dict) -> Dict:
        name = params.get("name", "")
        obj = self._objects.get(name)
        if obj:
            return {
                "status": "success",
                "name": name,
                "position": obj["position"],
                "orientation": [0, 0, 0, 1],
            }
        return {"status": "error", "message": f"Object not found: {name}"}

    # ── Robot ────────────────────────────────────────────────────

    def _create_robot(self, params: Dict) -> Dict:
        robot_type = params.get("robot_type", "franka")
        name = params.get("name", robot_type)
        info = self._robot_info(robot_type)

        self._robots[name] = {
            "robot_type": robot_type,
            "position": params.get("position", [0, 0, 0]),
            "orientation": params.get("orientation", [0, 0, 0, 1]),
        }
        self._robot_joints[name] = params.get(
            "joint_positions", info["home"][:info["joints"]]
        )
        return {
            "status": "success",
            "message": f"Created {robot_type}: {name}",
            "name": name,
            "prim_path": f"/World/{name}",
        }

    def _set_joint_positions(self, params: Dict) -> Dict:
        name = params.get("robot_name", "")
        joints = params.get("joint_positions", [])
        if name in self._robot_joints:
            self._robot_joints[name] = list(joints)
        return {"status": "success", "joint_positions": list(joints)}

    def _get_joint_positions(self, params: Dict) -> Dict:
        name = params.get("robot_name", "")
        joints = self._robot_joints.get(name, [])
        return {
            "status": "success",
            "joint_positions": list(joints),
            "joint_names": [f"joint_{i}" for i in range(len(joints))],
        }

    def _control_gripper(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "open": params.get("open", True),
            "width": params.get("width", 0.04),
        }

    def _set_robot_pose(self, params: Dict) -> Dict:
        name = params.get("robot_name", "")
        if name in self._robots:
            self._robots[name]["position"] = params.get("position",
                                                         self._robots[name]["position"])
        return {"status": "success"}

    def _get_robot_pose(self, params: Dict) -> Dict:
        name = params.get("robot_name", "")
        robot = self._robots.get(name)
        if robot:
            return {"status": "success", **robot}
        return {"status": "error", "message": f"Robot not found: {name}"}

    def _get_end_effector_pose(self, params: Dict) -> Dict:
        name = params.get("robot_name", "")
        robot = self._robots.get(name)
        if not robot:
            return {"status": "error", "message": f"Robot not found: {name}"}
        joints = self._robot_joints.get(name, [])
        fk_func = self._get_fk_func(robot["robot_type"])
        ee_pos = fk_func(joints)
        return {
            "status": "success",
            "position": ee_pos.tolist(),
            "orientation": [0, 0, 0, 1],
        }

    # ── Camera ───────────────────────────────────────────────────

    def _set_camera(self, params: Dict) -> Dict:
        cam_path = params.get("camera_path", "/Camera")
        self._cameras[cam_path] = {
            "position": params.get("position", [1.5, 0.5, 1.0]),
            "target": params.get("target", [0, 0, 0]),
        }
        return {"status": "success"}

    def _capture_image(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "data": "base64_encoded_png_placeholder",
            "width": params.get("width", 1280),
            "height": params.get("height", 720),
            "format": "png",
        }

    def _capture_depth(self, params: Dict) -> Dict:
        w, h = params.get("width", 1280), params.get("height", 720)
        depth = self.rng.uniform(0.5, 2.0, (h, w)).tolist()
        return {
            "status": "success",
            "data": depth,
            "width": w,
            "height": h,
        }

    def _get_camera_info(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "cameras": list(self._cameras.keys()),
        }

    # ── Physics ──────────────────────────────────────────────────

    def _apply_force(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "prim_path": params.get("prim_path", ""),
            "force": params.get("force", [0, 0, 0]),
        }

    def _set_gravity(self, params: Dict) -> Dict:
        self._physics["gravity"] = params.get("gravity", [0, 0, -9.81])
        return {"status": "success", "gravity": self._physics["gravity"]}

    def _step_simulation(self, params: Dict) -> Dict:
        num_steps = params.get("num_steps", 1)
        self._physics["simulating"] = True
        # Simulate physics: update object positions slightly
        for obj_name, obj in self._objects.items():
            if obj["mass"] > 0:
                obj["position"][2] -= 0.001 * num_steps  # gravity pull
        return {"status": "success", "steps_taken": num_steps}

    def _pause_simulation(self, params: Dict) -> Dict:
        self._physics["simulating"] = False
        return {"status": "success"}

    def _resume_simulation(self, params: Dict) -> Dict:
        self._physics["simulating"] = True
        return {"status": "success"}

    def _get_physics_context(self, params: Dict) -> Dict:
        return {"status": "success", **self._physics}

    # ── Motion Planning ──────────────────────────────────────────

    def _compute_ik(self, params: Dict) -> Dict:
        robot_name = params.get("robot_name", "franka")
        target_pose = params.get("target_pose", [0.5, 0.0, 0.5, 0, 1, 0, 0])
        robot = self._robots.get(robot_name, {"robot_type": "franka"})

        target_pos = np.array(target_pose[:3])
        start_joints = params.get("start_qpos")
        q_start = np.array(start_joints) if start_joints else None
        max_iter = params.get("max_iterations", 200)

        q, ee_pos, success = compute_ik_numerical(
            target_pos, robot["robot_type"],
            q_start=q_start, steps=max_iter,
        )
        return {
            "status": "success",
            "joint_positions": q.tolist(),
            "end_effector_position": ee_pos.tolist(),
            "success": success,
            "error": "" if success else "IK did not converge",
        }

    def _plan_path(self, params: Dict) -> Dict:
        target = params.get("target_joints", [])
        start = params.get("start_joints", None)
        robot_name = params.get("robot_name", "franka")
        current = self._robot_joints.get(robot_name,
                                          DEFAULT_HOME.get(robot_name, [0]*7))
        start_j = start or list(current)

        if not target:
            return {"status": "error", "message": "No target joints specified"}
        n_waypoints = 10
        path = [
            [start_j[i] + (target[i] - start_j[i]) * t / n_waypoints
             for i in range(min(len(start_j), len(target)))]
            for t in range(1, n_waypoints + 1)
        ]
        return {
            "status": "success",
            "path": path,
            "success": True,
        }

    def _execute_trajectory(self, params: Dict) -> Dict:
        robot_name = params.get("robot_name", "")
        trajectory = params.get("trajectory", [])
        if trajectory and robot_name in self._robot_joints:
            self._robot_joints[robot_name] = list(trajectory[-1])
        return {
            "status": "success",
            "waypoints_executed": len(trajectory),
            "success": True,
        }

    # ── Diagnostics ──────────────────────────────────────────────

    def _get_transform(self, params: Dict) -> Dict:
        prim_path = params.get("prim_path", "")
        return {
            "status": "success",
            "prim_path": prim_path,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }

    def _get_bounding_box(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "min": [-0.5, -0.5, -0.5],
            "max": [0.5, 0.5, 0.5],
            "extent": [1.0, 1.0, 1.0],
        }

    def _check_collision(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "colliding": False,
            "distance": 0.5,
        }

    def _get_distance(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "distance": 0.3,
            "closest_points": [[0, 0, 0], [0.3, 0, 0]],
        }

    def _ray_cast(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "hit": True,
            "hit_position": params.get("origin", [0, 0, 0]),
            "hit_normal": [0, 0, 1],
            "hit_distance": 1.0,
        }

    def _get_prim_paths(self, params: Dict) -> Dict:
        pattern = params.get("pattern", "/World/**")
        paths = [f"/World/{name}" for name in self._objects]
        paths += [f"/World/{name}" for name in self._robots]
        return {
            "status": "success",
            "prim_paths": paths,
            "count": len(paths),
        }

    # ── Domain Randomization ─────────────────────────────────────

    def _randomize_lighting(self, params: Dict) -> Dict:
        return {"status": "success", "lighting_randomized": True}

    def _randomize_textures(self, params: Dict) -> Dict:
        return {"status": "success", "textures_randomized": True}

    def _randomize_pose(self, params: Dict) -> Dict:
        prim_paths = params.get("prim_paths", [])
        trans_range = params.get("translation_range", 0.1)
        for path in prim_paths:
            name = path.split("/")[-1]
            if name in self._objects:
                obj = self._objects[name]
                obj["position"] = [
                    obj["position"][i] + self.rng.uniform(-trans_range, trans_range)
                    for i in range(3)
                ]
        return {"status": "success", "poses_randomized": len(prim_paths)}

    # ── Script Execution ─────────────────────────────────────────

    def _execute_script(self, params: Dict) -> Dict:
        """Execute script - parse commands and simulate results."""
        code = params.get("code", "")
        result_data = {}

        # Extract target position
        target = self._extract_array(code, "target", [0.5, 0.0, 0.5])

        # Extract joint positions
        joints = self._extract_array(code, "joint_positions", None)

        # Extract robot info
        robot_name = "franka"
        m = re.search(r'robot_name\s*=\s*["\'](\w+)["\']', code)
        if m:
            robot_name = m.group(1)

        robot_info = self._robots.get(robot_name, {"robot_type": "franka"})
        fk_func = self._get_fk_func(robot_info["robot_type"])

        if joints and len(joints) >= 6:
            ee_pos = fk_func(list(joints))
            # Add noise
            ee_pos += self.rng.normal(0, self._joint_noise, 3)
            dist = float(np.linalg.norm(ee_pos - np.array(target)))
            result_data = {
                "ee_position": ee_pos.tolist(),
                "target_position": target,
                "distance_to_target": dist,
                "joint_positions": list(joints),
                "success": dist < 0.05,
            }
        else:
            # No joints found, try to simulate result by extracting from code
            result_data = self._extract_result_from_code(code)
            if not result_data:
                result_data = {"note": "script executed (mock)", "success": True}

        message = json.dumps(result_data)
        return {
            "status": "success",
            "message": f"AUTOSIM_RESULT={message}",
            "result": message,
        }

    def _extract_array(self, code: str, name: str,
                       default: Any = None) -> Optional[List[float]]:
        """Extract a numpy array or list from code by variable name."""
        patterns = [
            rf'{name}\s*=\s*np\.array\(\[(.*?)\]\)',
            rf'{name}\s*=\s*\[(.*?)\]',
            rf'["\']{name}["\']\s*:\s*\[(.*?)\]',
        ]
        for pattern in patterns:
            m = re.search(pattern, code, re.DOTALL)
            if m:
                vals = re.sub(r'#.*', '', m.group(1))
                vals = vals.replace('\n', ' ').replace('\r', ' ')
                parts = [v.strip() for v in vals.split(',') if v.strip()]
                try:
                    return [float(p) for p in parts]
                except (ValueError, IndexError):
                    pass
        return default

    def _extract_result_from_code(self, code: str) -> Dict:
        """Extract structured result from code that prints AUTOSIM_RESULT."""
        m = re.search(r'AUTOSIM_RESULT[=:]\s*(\{.*\})', code, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return {}

    # ── Utility ──────────────────────────────────────────────────

    def _get_sim_performance(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "fps": 60.0,
            "sim_time": 0.0,
            "num_prims": len(self._objects) + len(self._robots),
        }

    def _get_asset_root(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "asset_root": "/Isaac/Assets",
        }

    def _omni_kit_command(self, params: Dict) -> Dict:
        return {"status": "success", "message": "command executed"}
