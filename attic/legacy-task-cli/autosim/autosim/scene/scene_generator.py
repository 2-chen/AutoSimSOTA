"""
Scene Generator — LLM-driven scene configuration generation.

Transforms natural language task descriptions into structured Isaac Sim
scene configurations that can be built and evaluated in the MCP client.

Architecture:
  1. User provides a free-text task description
  2. LLM analyzes the description and produces a structured scene config
  3. Scene config is validated and saved to the registry
  4. The scene can then be instantiated in Isaac Sim and optimized

Supports:
  - Template-based: modify an existing template to match the description
  - Zero-shot: generate a completely new scene from scratch
  - Iterative refinement: take feedback from simulation results
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autosim.scene.scene_schema import (
    StructuredSceneConfig,
    get_template,
    list_templates,
)
from autosim.llm_client import LLMClient


def load_env():
    """Load environment variables from .env."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


@dataclass
class GenerationResult:
    """Result of a scene generation call."""
    config: StructuredSceneConfig
    raw_response: str = ""
    template_used: str = ""
    analysis: str = ""
    success: bool = True
    error: str = ""


class SceneGenerator:
    """
    LLM-driven scene configuration generator.

    Uses a language model to transform natural language task descriptions
    into structured Isaac Sim scene configurations.

    Usage:
        gen = SceneGenerator()
        result = gen.from_description(
            "A Franka robot grasping a blue cylinder on a table"
        )
        scene = result.config
        scene_registry.save(scene)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        load_env()

        self.llm = LLMClient(api_key=api_key, base_url=api_base_url, model=model)
        self._unavailable = False

    # ── LLM API ─────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str,
                  max_tokens: int = 8192) -> str:
        if self._unavailable or not self.llm.available:
            return ""

        try:
            return self.llm.chat(system, user, max_tokens=max_tokens, timeout=120)
        except Exception:
            self._unavailable = True
            return ""

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{[\s\S]*"objects"[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None

    # ── Prompt Building ─────────────────────────────────────────

    def _build_zero_shot_prompt(self, description: str) -> Tuple[str, str]:
        """Build prompt for zero-shot scene generation."""
        system = """You are a robotics scene designer. You create structured Isaac Sim scene configurations from natural language descriptions.

Output ONLY valid JSON matching this exact schema, no markdown or explanation:
```json
{
  "name": "scene_name",
  "description": "human-readable description",
  "objects": [
    {
      "name": "unique_name",
      "shape": "Cube|Sphere|Cylinder|Capsule|Cone",
      "position": [x, y, z],
      "scale": [sx, sy, sz],
      "color": [r, g, b],
      "mass": 0.05,
      "semantic_tag": "grasp_target|obstacle|fixture|container"
    }
  ],
  "robots": [
    {
      "model": "franka|ur5",
      "position": [x, y, z]
    }
  ],
  "task": {
    "category": "reach|grasp|pick_place|insertion",
    "description": "task description",
    "target_object": "object_name"
  },
  "camera": {
    "position": [x, y, z],
    "target": [tx, ty, tz]
  }
}
```

Rules:
- Objects on a table should be at z ≈ 0.4 (table surface)
- The robot base is at z = 0.75 (table height)
- grasp_target objects are for picking; obstacles block the path
- Use descriptive object names
- Place objects within reachable range (x: 0.3-0.7, y: -0.25 to 0.25)"""

        user = f"Create an Isaac Sim scene for this task:\n\n{description}"
        return system, user

    def _build_template_prompt(self, description: str,
                               template: StructuredSceneConfig) -> Tuple[str, str]:
        """Build prompt for template-based scene generation."""
        template_json = json.dumps(template.to_dict(), indent=2)

        system = """You are a robotics scene designer. Given a template scene and a task description, modify the template to create an appropriate scene configuration.

Output ONLY valid JSON. Follow the exact same schema as the template but with values modified to match the new task.

Rules:
- Adjust object positions, shapes, colors, sizes to match the description
- Keep the table and robot unless the description says otherwise
- Match the task category to the type of manipulation described
- Objects must be within reachable range"""

        user = f"""Task: {description}

Template scene:
```json
{template_json}
```

Modify the template to match the task. Output the complete modified JSON."""
        return system, user

    # ── Generation Methods ───────────────────────────────────────

    def from_description(
        self,
        description: str,
        template_name: Optional[str] = None,
    ) -> GenerationResult:
        """Generate a scene configuration from a natural language description.

        Args:
            description: Free-text task description
            template_name: Optional template to base the generation on

        Returns:
            GenerationResult with the generated config
        """
        try:
            if template_name and template_name in list_templates():
                # Template-based
                template = get_template(template_name)
                system, prompt = self._build_template_prompt(description, template)
                analysis = f"Based on template: {template_name}"
            else:
                # Zero-shot
                template = None
                system, prompt = self._build_zero_shot_prompt(description)
                analysis = "Zero-shot generation"

            if self.llm.available and not self._unavailable:
                response = self._call_llm(system, prompt)
                if response:
                    parsed = self._parse_json(response)
                    if parsed:
                        config = StructuredSceneConfig.from_dict(parsed)
                        return GenerationResult(
                            config=config,
                            raw_response=response,
                            template_used=template_name or "zero_shot",
                            analysis=analysis,
                            success=True,
                        )

            # Fallback: template matching by keyword
            return self._keyword_fallback(description)

        except Exception as e:
            return GenerationResult(
                config=StructuredSceneConfig(
                    name="error",
                    description=f"Generation failed: {e}",
                ),
                success=False,
                error=str(e),
            )

    def _keyword_fallback(self, description: str) -> GenerationResult:
        """Fallback: match keywords in description to best template."""
        desc_lower = description.lower()

        # Keyword matching
        if "stack" in desc_lower or "pile" in desc_lower:
            template_name = "stack_blocks"
        elif "insert" in desc_lower or "peg" in desc_lower or "hole" in desc_lower:
            template_name = "peg_insertion"
        elif "bin" in desc_lower or "multiple" in desc_lower or "scatter" in desc_lower:
            template_name = "bin_picking"
        elif "place" in desc_lower or "pick and place" in desc_lower or "move" in desc_lower:
            template_name = "pick_and_place"
        elif "grasp" in desc_lower or "grip" in desc_lower or "hold" in desc_lower:
            template_name = "single_object_grasp"
        elif "reach" in desc_lower or "touch" in desc_lower:
            template_name = "empty_workspace"
        else:
            template_name = "single_object_grasp"

        config = get_template(template_name)
        config.description = f"Scene for: {description}"
        config.name = f"auto_{template_name}"

        return GenerationResult(
            config=config,
            analysis=f"Keyword-matched to template: {template_name}",
            template_used=template_name,
            success=True,
        )

    def refine_from_feedback(
        self,
        previous_config: StructuredSceneConfig,
        feedback: str,
        eval_results: Optional[Dict] = None,
    ) -> GenerationResult:
        """Refine a scene configuration based on simulation feedback.

        Args:
            previous_config: The scene config that was evaluated
            feedback: Natural language feedback on what to change
            eval_results: Optional evaluation metrics from the last run

        Returns:
            Updated scene configuration
        """
        # For now, use keyword fallback
        # In full implementation, this would call LLM with previous results
        if eval_results:
            # Check if the scene was too easy/hard and adjust
            success_rate = eval_results.get("success_rate", None)
            if success_rate is not None:
                if success_rate > 0.9:
                    # Too easy — make it harder
                    varied = apply_variation(previous_config, seed=42, variation_scale=0.15)
                    varied.description = f"Harder variant: {feedback}"
                    return GenerationResult(config=varied, success=True)
                elif success_rate < 0.2:
                    # Too hard — make it easier
                    varied = apply_variation(previous_config, seed=0, variation_scale=0.02)
                    varied.description = f"Easier variant: {feedback}"
                    return GenerationResult(config=varied, success=True)

        # Default: just apply keyword matching
        return self.from_description(feedback)

    def batch_generate(
        self,
        descriptions: List[str],
        template_name: Optional[str] = None,
    ) -> List[GenerationResult]:
        """Generate multiple scene configurations from a list of descriptions."""
        results = []
        for desc in descriptions:
            result = self.from_description(desc, template_name)
            results.append(result)
        return results


# Convenience alias for import
def generate_scene(description: str, **kwargs) -> StructuredSceneConfig:
    """One-liner: generate a scene from description."""
    gen = SceneGenerator()
    result = gen.from_description(description, **kwargs)
    return result.config
