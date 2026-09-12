"""
Continuous Learning — self-improving optimizer that learns from every run.

Core capabilities:
  1. Mines historical optimization runs for actionable lessons
  2. Maintains a "skill memory" of what parameter ranges work for which tasks
  3. Recommends promising starting points for new tasks
  4. Automatically prunes parameter ranges that consistently fail
  5. Generates smarter initial candidates based on accumulated experience

This is the "learning" layer on top of the experience graph.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ParameterInsight:
    """Learned knowledge about a single parameter."""
    name: str
    task_type: str
    robot_type: str
    best_value: Optional[float] = None
    best_score: Optional[float] = None
    suggested_range: Tuple[float, float] = (0.0, 1.0)
    n_trials: int = 0
    successful_values: List[float] = field(default_factory=list)
    failed_values: List[float] = field(default_factory=list)
    correlation: Optional[float] = None  # correlation with success

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "robot_type": self.robot_type,
            "best_value": self.best_value,
            "best_score": self.best_score,
            "suggested_range": self.suggested_range,
            "n_trials": self.n_trials,
            "successful_values": self.successful_values[-20:],
            "failed_values": self.failed_values[-20:],
            "correlation": self.correlation,
        }


@dataclass
class SkillMemory:
    """Learned skill for a specific (task_type, robot) combination."""
    task_type: str
    robot_type: str
    n_runs: int = 0
    best_score_ever: Optional[float] = None
    params: Dict[str, ParameterInsight] = field(default_factory=dict)
    successful_patterns: List[str] = field(default_factory=list)
    failed_patterns: List[str] = field(default_factory=list)
    last_updated: str = ""

    def suggest_params(self) -> Dict[str, float]:
        """Suggest parameter values based on accumulated knowledge."""
        suggestion = {}
        for name, insight in self.params.items():
            if insight.n_trials > 0:
                # Use best value if available and reliable
                if insight.best_value is not None and insight.n_trials >= 3:
                    suggestion[name] = insight.best_value
                else:
                    # Use center of suggested range
                    lo, hi = insight.suggested_range
                    suggestion[name] = (lo + hi) / 2
            else:
                # Use center of range
                lo, hi = insight.suggested_range
                suggestion[name] = (lo + hi) / 2
        return suggestion

    def summary(self) -> str:
        """Produce a text summary for LLM context."""
        lines = [f"## Skill: {self.task_type}/{self.robot_type}",
                 f"Runs: {self.n_runs}, Best: {self.best_score_ever:.4f}"]
        if self.params:
            lines.append("\n### Parameter Insights")
            for name, p in sorted(self.params.items()):
                success_rate = len(p.successful_values) / max(p.n_trials, 1) * 100
                lines.append(
                    f"- {name}: {p.n_trials} trials, {success_rate:.0f}% success, "
                    f"best={p.best_value} (range {p.suggested_range[0]:.3f}-{p.suggested_range[1]:.3f})"
                )
        if self.successful_patterns:
            lines.append(f"\n### What Works")
            for pat in self.successful_patterns[-3:]:
                lines.append(f"- {pat}")
        if self.failed_patterns:
            lines.append(f"\n### What Doesn't")
            for pat in self.failed_patterns[-3:]:
                lines.append(f"- {pat}")
        return "\n".join(lines)


class ContinuousLearner:
    """Self-improving optimizer that learns from every run.

    Usage:
        learner = ContinuousLearner()
        learner.ingest_run(result_dict, task_type="grasp", robot="franka")
        suggestions = learner.suggest_for(task_type="grasp", robot="franka")
        summary = learner.get_context_for_llm(task_type="grasp")
    """

    def __init__(self, memory_dir: str = ".autosim/continuous_learning"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Skills: {(task_type, robot): SkillMemory}
        self.skills: Dict[Tuple[str, str], SkillMemory] = {}
        self._load()

    def _path_for(self) -> Path:
        return self.memory_dir / "skill_memory.json"

    def _save(self):
        """Serialize all skills to disk."""
        data = {}
        for (task_type, robot), skill in self.skills.items():
            key = f"{task_type}|{robot}"
            data[key] = {
                "task_type": skill.task_type,
                "robot_type": skill.robot_type,
                "n_runs": skill.n_runs,
                "best_score_ever": skill.best_score_ever,
                "params": {n: p.to_dict() for n, p in skill.params.items()},
                "successful_patterns": skill.successful_patterns,
                "failed_patterns": skill.failed_patterns,
                "last_updated": skill.last_updated,
            }
        self._path_for().write_text(json.dumps(data, indent=2, default=str))

    def _load(self):
        """Load all skills from disk."""
        path = self._path_for()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for key, info in data.items():
                task_type = info["task_type"]
                robot = info["robot_type"]
                params = {}
                for name, p_info in info.get("params", {}).items():
                    params[name] = ParameterInsight(
                        name=p_info["name"],
                        task_type=p_info["task_type"],
                        robot_type=p_info["robot_type"],
                        best_value=p_info.get("best_value"),
                        best_score=p_info.get("best_score"),
                        suggested_range=tuple(p_info.get("suggested_range", [0, 1])),
                        n_trials=p_info.get("n_trials", 0),
                        successful_values=p_info.get("successful_values", []),
                        failed_values=p_info.get("failed_values", []),
                        correlation=p_info.get("correlation"),
                    )
                skill = SkillMemory(
                    task_type=task_type,
                    robot_type=robot,
                    n_runs=info.get("n_runs", 0),
                    best_score_ever=info.get("best_score_ever"),
                    params=params,
                    successful_patterns=info.get("successful_patterns", []),
                    failed_patterns=info.get("failed_patterns", []),
                    last_updated=info.get("last_updated", ""),
                )
                self.skills[(task_type, robot)] = skill
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [ContinuousLearner] Error loading memory: {e}")

    def _get_skill(self, task_type: str, robot: str) -> SkillMemory:
        """Get or create a SkillMemory for a (task_type, robot) pair."""
        key = (task_type, robot)
        if key not in self.skills:
            self.skills[key] = SkillMemory(
                task_type=task_type,
                robot_type=robot,
            )
        return self.skills[key]

    def ingest_run(self, result: Dict, task_type: str,
                   robot: str = "franka") -> None:
        """Ingest an optimization run result and extract lessons.

        Args:
            result: Dict from LLMOptimizer.run() or ClosedLoopOptimizer.run()
            task_type: Task type (reach, grasp, pick_place, etc.)
            robot: Robot model
        """
        skill = self._get_skill(task_type, robot)
        skill.n_runs += 1
        skill.last_updated = datetime.now().isoformat()

        history = result.get("history", [])
        best_score = result.get("best_score")
        baseline_score = result.get("baseline_score")

        # Update best score ever
        if best_score is not None:
            if skill.best_score_ever is None or best_score > skill.best_score_ever:
                skill.best_score_ever = best_score

        # Analyze each trial
        for entry in history:
            params = entry.get("params", {})
            score = entry.get("score", 0)
            entry_type = entry.get("type", "PARAM")
            success = entry.get("success", False)
            is_best = entry.get("is_best", False)

            for pname, pvalue in params.items():
                if not isinstance(pvalue, (int, float)):
                    continue

                insight = skill.params.setdefault(pname, ParameterInsight(
                    name=pname,
                    task_type=task_type,
                    robot_type=robot,
                ))

                insight.n_trials += 1

                # Track successful vs failed values
                if success:
                    insight.successful_values.append(pvalue)
                else:
                    insight.failed_values.append(pvalue)

                # Track best
                if is_best and score is not None:
                    if insight.best_value is None or score > (insight.best_score or 0):
                        insight.best_value = pvalue
                        insight.best_score = score

            # Extract patterns from CODE/ALGO entries
            if entry_type in ("CODE", "ALGO"):
                title = entry.get("title", "")
                improved = entry.get("improved", entry.get("is_best", False))
                pattern_text = f"[{entry_type}] {title} ({score:.4f})"

                if improved:
                    if pattern_text not in skill.successful_patterns:
                        skill.successful_patterns.append(pattern_text)
                else:
                    if pattern_text not in skill.failed_patterns:
                        skill.failed_patterns.append(pattern_text)

        # Prune to keep lists manageable
        skill.successful_patterns = skill.successful_patterns[-20:]
        skill.failed_patterns = skill.failed_patterns[-20:]

        # Update suggested ranges based on successful values
        for insight in skill.params.values():
            if len(insight.successful_values) >= 2:
                vals = insight.successful_values
                lo = min(vals)
                hi = max(vals)
                margin = (hi - lo) * 0.2  # 20% margin
                insight.suggested_range = (lo - margin, hi + margin)

        self._save()

    def suggest_for(self, task_type: str, robot: str = "franka") -> Dict[str, float]:
        """Get suggested parameter values for a task type.

        Returns dict of {param_name: value} or empty dict if no experience.
        """
        skill = self._get_skill(task_type, robot)
        return skill.suggest_params()

    def get_context_for_llm(self, task_type: str,
                            robot: str = "franka") -> str:
        """Generate context string for LLM prompts with learned experience.

        Intended to be injected into LLM optimization prompts so the model
        benefits from past runs.
        """
        skill = self._get_skill(task_type, robot)
        if skill.n_runs == 0:
            return "No prior experience with this task type."

        return skill.summary()

    def get_all_summaries(self) -> str:
        """Get summaries of all learned skills."""
        if not self.skills:
            return "No skills learned yet."
        parts = []
        for (task_type, robot), skill in self.skills.items():
            parts.append(skill.summary())
        return "\n\n".join(parts)

    def suggest_param_ranges(self, task_type: str, robot: str = "franka",
                             original_ranges: Optional[Dict[str, Tuple[float, float]]] = None
                             ) -> Dict[str, Tuple[float, float]]:
        """Suggest narrowed parameter ranges based on experience.

        Args:
            task_type: Task type
            robot: Robot model
            original_ranges: Original parameter ranges (fallback if no experience)

        Returns:
            Dict of {param_name: (lo, hi)} with ranges potentially narrowed
        """
        skill = self._get_skill(task_type, robot)
        suggested = {}

        for pname, insight in skill.params.items():
            if insight.n_trials >= 3:
                # Narrow range to successful region
                lo, hi = insight.suggested_range
                suggested[pname] = (lo, hi)
            elif original_ranges and pname in original_ranges:
                suggested[pname] = original_ranges[pname]

        # Fill in missing params from original
        if original_ranges:
            for pname, range_ in original_ranges.items():
                if pname not in suggested:
                    suggested[pname] = range_

        return suggested

    def get_best_params(self, task_type: str,
                        robot: str = "franka") -> Dict[str, float]:
        """Get best-known parameter combination for a task."""
        skill = self._get_skill(task_type, robot)
        best = {}
        for name, insight in skill.params.items():
            if insight.best_value is not None:
                best[name] = insight.best_value
        return best


class AutoCurriculum:
    """Dynamic difficulty adjustment based on learner progress.

    Automatically adjusts scene difficulty to maintain an optimal
    learning challenge (~50-70% success rate).
    """

    def __init__(self, learner: ContinuousLearner):
        self.learner = learner
        self.difficulty_history: Dict[str, List[float]] = defaultdict(list)

    def adjust_difficulty(self, task_type: str, robot: str,
                          current_success_rate: float,
                          params: Dict[str, float]) -> Dict[str, float]:
        """Adjust parameters to target ideal difficulty.

        If success rate is too high, make task harder.
        If too low, make it easier.

        Args:
            task_type: Task type
            robot: Robot model
            current_success_rate: Current evaluation success rate
            params: Current parameter values

        Returns:
            Adjusted parameters
        """
        key = f"{task_type}|{robot}"
        self.difficulty_history[key].append(current_success_rate)

        adjusted = dict(params)
        rate = current_success_rate

        # Target zone: 50-70% success
        if rate > 0.8:
            # Too easy — make harder
            for pname in params:
                if "distance" in pname or "height" in pname:
                    adjusted[pname] = params[pname] * 1.15
                elif "force" in pname:
                    adjusted[pname] = params[pname] * 0.85
        elif rate < 0.3:
            # Too hard — make easier
            for pname in params:
                if "distance" in pname or "height" in pname:
                    adjusted[pname] = params[pname] * 0.85
                elif "force" in pname:
                    adjusted[pname] = params[pname] * 1.15

        return adjusted

    def recent_progress(self, task_type: str, robot: str) -> Optional[float]:
        """Compute recent progress trend (-1 to 1, positive = improving)."""
        key = f"{task_type}|{robot}"
        history = self.difficulty_history[key]
        if len(history) < 5:
            return None
        recent = np.mean(history[-3:])
        older = np.mean(history[-6:-3])
        return recent - older
