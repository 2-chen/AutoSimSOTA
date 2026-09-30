"""
Scene Builder — converts StructuredSceneConfig into MCP commands.

Takes a scene configuration (from template or LLM generation) and
builds it in Isaac Sim via the MCP client. Supports both mock and real modes.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from autosim.mcp_client import MCPClient, ObjectSpec, RobotConfig, CameraConfig, PhysicsConfig
from autosim.scene.scene_schema import (
    StructuredSceneConfig, ObjectConfig, RobotPlacement,
    apply_variation,
)


class SceneBuilder:
    """Builds Isaac Sim scenes from StructuredSceneConfig objects.

    Usage:
        builder = SceneBuilder(client)
        result = builder.build(scene_config)
        builder.teardown()
    """

    def __init__(self, client: MCPClient, verbose: bool = False):
        self.client = client
        self.verbose = verbose
        self.built_objects: List[str] = []
        self.built_robots: List[str] = []

    def build(self, config: StructuredSceneConfig) -> Dict[str, Any]:
        """Build a complete scene from a StructuredSceneConfig.

        Args:
            config: The scene configuration to build

        Returns:
            Dict with scene, robot, objects, camera results
        """
        self.built_objects.clear()
        self.built_robots.clear()
        results = {}

        if self.verbose:
            print(f"\n  Building scene: {config.name}")

        # 1. Create physics scene
        scene_result = self.client.create_scene(
            floor=config.floor,
            gravity=config.gravity,
            scene_name=config.name,
        )
        results["scene"] = scene_result
        if self.verbose:
            print(f"  ✓ Scene created: {config.name}")

        # 2. Add table
        if config.table.present:
            table_spec = ObjectSpec(
                prim_type="Cube",
                name="table",
                position=config.table.position,
                scale=[config.table.size[0], config.table.size[1], config.table.height],
                color=config.table.color,
                mass=0.0,
            )
            table_result = self.client.add_object(table_spec)
            self.built_objects.append("table")
            results["table"] = table_result
            if self.verbose:
                print(f"  ✓ Table added")

        # 3. Add objects
        for obj_config in config.objects:
            spec = ObjectSpec(
                prim_type=obj_config.shape,
                name=obj_config.name,
                position=obj_config.position,
                scale=obj_config.scale,
                color=obj_config.color,
                mass=obj_config.mass,
                collision=obj_config.collision and not obj_config.visual_only,
            )
            obj_result = self.client.add_object(spec)
            self.built_objects.append(obj_config.name)
            results.setdefault("objects", [])
            results["objects"].append(obj_result)
            if self.verbose:
                print(f"  ✓ Object: {obj_config.name} ({obj_config.shape})")

        # 4. Add robots
        for robot_cfg in config.robots:
            robot_name = robot_cfg.name or robot_cfg.model
            robot_result = self.client.create_robot(
                robot_type=robot_cfg.model,
                position=robot_cfg.position,
                orientation=robot_cfg.orientation,
                name=robot_name,
                joint_positions=robot_cfg.home_joints,
            )
            self.built_robots.append(robot_name)
            results.setdefault("robots", [])
            results["robots"].append(robot_result)
            if self.verbose:
                print(f"  ✓ Robot: {robot_name} ({robot_cfg.model})")

            # Gripper
            if robot_cfg.gripper_open_on_start:
                self.client.control_gripper(robot_name, open=True, width=0.04)

        # 5. Set camera
        cam_result = self.client.set_camera(
            position=config.camera.position,
            target=config.camera.target,
        )
        results["camera"] = cam_result
        if self.verbose:
            print(f"  ✓ Camera set")

        # 6. Configure domain randomization if enabled
        if config.domain_rand.enabled:
            if config.domain_rand.lighting_variation > 0:
                self.client.randomize_lighting(
                    intensity_range=(
                        1.0 - config.domain_rand.lighting_variation,
                        1.0 + config.domain_rand.lighting_variation,
                    )
                )
            if self.verbose:
                print(f"  ✓ Domain randomization configured")

        results["name"] = config.name
        results["task"] = {
            "category": config.task.category.value if hasattr(config.task.category, 'value') else config.task.category,
            "description": config.task.description,
            "target_object": config.task.target_object,
        }

        if self.verbose:
            print(f"  ✓ Scene build complete: {len(self.built_objects)} objects, {len(self.built_robots)} robots")

        return results

    def teardown(self):
        """Teardown the built scene."""
        if self.verbose:
            print(f"  Tearing down scene")
        self.client.reset_scene()
        self.built_objects.clear()
        self.built_robots.clear()

    def randomize(self, config: StructuredSceneConfig, seed: int = 0,
                  variation_scale: float = 0.1) -> Dict[str, Any]:
        """Build a randomized variation of a scene."""
        varied = apply_variation(config, seed=seed, variation_scale=variation_scale)
        return self.build(varied)
