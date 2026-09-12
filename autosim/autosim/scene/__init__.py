"""
AutoSim Scene System — LLM-driven scene generation, registry, building, and continuous learning.

Flow:
  Natural Language Description
    → SceneGenerator.from_description()  (LLM or keyword fallback)
      → StructuredSceneConfig            (data model)
        → SceneRegistry.save()           (persistent storage)
          → SceneBuilder.build()         (MCP commands → Isaac Sim)
            → TaskAdapter.evaluate()     (run optimization)
              → ContinuousLearner.ingest_run()  (learn from results)
                → SceneGenerator.refine_from_feedback()  (improve next scene)
"""

from autosim.scene.scene_schema import (
    StructuredSceneConfig,
    ObjectConfig,
    RobotPlacement,
    TaskDefinition,
    DomainRandomizationConfig,
    ObjectShape,
    RobotModel,
    TaskCategory,
    get_template,
    list_templates,
    apply_variation,
    TEMPLATES,
)
from autosim.scene.scene_registry import SceneRegistry
from autosim.scene.scene_generator import SceneGenerator, generate_scene, GenerationResult
from autosim.scene.scene_builder import SceneBuilder
from autosim.scene.continuous_learning import ContinuousLearner, AutoCurriculum, SkillMemory, ParameterInsight

__all__ = [
    "StructuredSceneConfig",
    "ObjectConfig",
    "RobotPlacement",
    "TaskDefinition",
    "DomainRandomizationConfig",
    "SceneRegistry",
    "SceneGenerator",
    "generate_scene",
    "GenerationResult",
    "SceneBuilder",
    "ContinuousLearner",
    "AutoCurriculum",
    "get_template",
    "list_templates",
    "apply_variation",
    "TEMPLATES",
]
