"""
Mock motion planner for RoboTwin tasks.

Provides MplibPlanner-compatible interface that generates
simple linear interpolation paths instead of full RRT planning.
This allows RoboTwin tasks to run without the heavy mplib dependency.

The mock generates straight-line paths in joint space, which is
sufficient for testing strategy parameter optimization.
"""

import numpy as np
from typing import Any, Dict, List


class MockMplibPlanner:
    """Mock MplibPlanner that generates linear interpolation paths."""

    def __init__(self, urdf_path: str = "", srdf_path: str = "",
                 move_group: str = "", **kwargs):
        self.urdf_path = urdf_path
        self.move_group = move_group
        self._qpos = np.zeros(7)  # current joint positions

    def plan_qpos(self, target_qpos: np.ndarray,
                  current_qpos: np.ndarray = None,
                  time_step: float = 1.0 / 250.0) -> Dict[str, Any]:
        """
        Plan a linear interpolation to target_qpos.
        Returns format matching MplibPlanner.
        """
        if current_qpos is None:
            current_qpos = self._qpos.copy()

        target = np.asarray(target_qpos, dtype=np.float64).flatten()
        current = np.asarray(current_qpos, dtype=np.float64).flatten()

        # Generate intermediate waypoints
        n_steps = max(20, int(np.max(np.abs(target - current)) / 0.02))
        path = np.linspace(current, target, n_steps)

        # Track this as current position
        self._qpos = target.copy()

        return {
            "status": "Success",
            "position": path,
            "velocity": np.zeros_like(path),
            "time": np.arange(n_steps) * time_step,
        }

    def plan_screw(self, target_pose: np.ndarray,
                   qpos: np.ndarray = None,
                   time_step: float = 1.0 / 250.0) -> Dict[str, Any]:
        """Mock screw motion planner — falls back to joint interpolation."""
        return self.plan_qpos(target_pose, qpos, time_step)

    def plan_qpos_screw(self, target_qpos: np.ndarray,
                        current_qpos: np.ndarray = None) -> Dict[str, Any]:
        """Alias for plan_qpos."""
        return self.plan_qpos(target_qpos, current_qpos)

    def TOPP(self, path: np.ndarray, time_step: float = 1.0 / 250.0,
             verbose: bool = False):
        """
        Time-Optimal Path Parameterization mock.
        Returns times, positions, velocities, accelerations, duration.
        """
        path = np.atleast_2d(path)
        n = path.shape[0]
        times = np.linspace(0, n * time_step, n)
        vels = np.zeros_like(path)
        # Simple finite difference velocities
        if n > 1:
            vels[:-1] = (path[1:] - path[:-1]) / time_step
        acc = np.zeros_like(path)
        if n > 2:
            acc[:-2] = (vels[1:] - vels[:-1]) / time_step
        duration = n * time_step
        return times, path, vels, acc, duration


class MockCuroboPlanner(MockMplibPlanner):
    """Mock CuroboPlanner — same as MockMplibPlanner."""
    pass
