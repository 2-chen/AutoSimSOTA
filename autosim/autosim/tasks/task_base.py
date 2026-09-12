"""
Base task class for embodied AI optimization.

Inspired by RoboTwin's Base_Task pattern: each task defines its
scene setup, execution logic, and success criteria. Adapted to work
with Isaac Sim through the MCP interface.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from autosim.mcp_client import MCPClient

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a single task execution."""
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def primary_score(self) -> float:
        """Get the primary metric value (first metric)."""
        if not self.metrics:
            return float("inf")
        return list(self.metrics.values())[0]


class BaseTask(ABC):
    """
    Abstract base class for embodied AI tasks.

    Each task encapsulates:
    - Scene setup (robots, objects, physics)
    - Parameter space definition
    - Execution logic (via Isaac Sim scripts)
    - Success/quality measurement

    Subclasses should implement:
    - _build_setup_script(): Generate Isaac Sim Python code for scene setup
    - _build_execute_script(params): Generate code to run the task with given params
    - _extract_metrics(result): Parse execution results into metric values
    """

    def __init__(self, client: MCPClient, task_config: "TaskConfig" = None):
        self.client = client
        self.config = task_config
        self._scene_ready = False

    # ── Public API ─────────────────────────────────────────────────

    def setup_scene(self) -> bool:
        """Set up the scene in Isaac Sim. Call once before execute()."""
        if self._scene_ready:
            logger.info("Scene already set up")
            return True

        logger.info(f"Setting up scene for task: {self.name}")

        # Verify connection
        scene_info = self.client.get_scene_info()
        logger.info(f"Scene info: {scene_info}")

        # Create physics scene
        self.client.create_physics_scene(
            objects=[],
            floor=True,
            scene_name=f"{self.name}_scene",
        )

        # Create the robot
        self.client.create_robot(
            robot_type=self.config.robot_type if self.config else "franka",
            position=[0, 0, 0],
        )

        # Run task-specific setup
        setup_code = self._build_setup_script()
        if setup_code:
            result = self.client.execute_script(setup_code)
            if isinstance(result, dict) and result.get("status") == "error":
                logger.error(f"Scene setup failed: {result}")
                return False

        self._scene_ready = True
        logger.info(f"Scene setup complete for task: {self.name}")
        return True

    def execute(self, params: Dict[str, Any]) -> TaskResult:
        """Execute the task with given parameters and return results."""
        if not self._scene_ready:
            if not self.setup_scene():
                return TaskResult(success=False, error="Scene setup failed")

        logger.info(f"Executing {self.name} with params: {params}")

        try:
            # Build and execute the task script
            code = self._build_execute_script(params)
            logger.debug(f"Execute script:\n{code[:500]}...")

            result = self.client.execute_script(code)

            # Extract metrics from the result
            metrics = self._extract_metrics(result)
            success = self._check_success(metrics)

            return TaskResult(
                success=success,
                metrics=metrics,
                info={"params": params, "raw_result": result},
            )
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResult(success=False, error=str(e))

    def teardown(self):
        """Clean up the scene."""
        self._scene_ready = False

    # ── Properties ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def param_space(self) -> Dict[str, List[float]]:
        """Return the parameter space for optimization. [min, max] per param."""
        return self.config.param_ranges if self.config else {}

    @property
    def param_names(self) -> List[str]:
        return list(self.param_space.keys())

    # ── Abstract methods ───────────────────────────────────────────

    @abstractmethod
    def _build_setup_script(self) -> str:
        """Build Python code string for scene setup (beyond robot creation)."""
        ...

    @abstractmethod
    def _build_execute_script(self, params: Dict[str, Any]) -> str:
        """Build Python code string that executes the task and prints metrics."""
        ...

    @abstractmethod
    def _extract_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Parse execution result into metric name → value mapping."""
        ...

    def _check_success(self, metrics: Dict[str, float]) -> bool:
        """Check if the task succeeded based on metrics. Override per task."""
        return len(metrics) > 0
