"""Persistent optimization experience graph.

The graph records every attempted optimization as a node, links related
attempts, and maintains family-level lessons. It is intentionally lightweight
JSON so runs can resume and future candidate generators can use the history.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from autosim.adapters.base import Candidate, EvalResult


@dataclass
class ExperienceNode:
    id: str
    candidate: str
    family: str
    tags: List[str]
    hypothesis: str
    score: float
    baseline_score: float
    delta_abs: float
    delta_pct: Optional[float]
    effect: str
    lesson: str
    metrics: Dict[str, Any]
    timestamp: str


@dataclass
class ExperienceEdge:
    source: str
    target: str
    relation: str
    reason: str


@dataclass
class FamilyStats:
    family: str
    attempts: int = 0
    best_score: Optional[float] = None
    best_candidate: Optional[str] = None
    positive: int = 0
    neutral: int = 0
    negative: int = 0
    lessons: List[str] = field(default_factory=list)


def classify_candidate(candidate: Candidate) -> Dict[str, Any]:
    """Infer an optimization method family from candidate metadata."""
    name = candidate.name.lower()
    text = " ".join(
        str(v).lower() for v in [candidate.name, candidate.kind, candidate.description]
    )
    text += " " + " ".join(f"{k}={v}".lower() for k, v in candidate.payload.items())

    if candidate.kind in {"patch_train_recipe", "algo_recipe", "code_recipe"}:
        tier = str(candidate.payload.get("tier", "")).upper()
        recipe_type = str(candidate.payload.get("recipe_type", "")).lower()
        if tier == "ALGO" or "algo" in recipe_type:
            return {
                "family": "llm_algo_patch",
                "tags": sorted({candidate.kind, "ALGO", recipe_type}),
                "hypothesis": candidate.description or "LLM-generated algorithm patch.",
                "params": {},
            }
        if tier == "CODE" or "code" in recipe_type:
            return {
                "family": "llm_code_patch",
                "tags": sorted({candidate.kind, "CODE", recipe_type}),
                "hypothesis": candidate.description or "LLM-generated implementation patch.",
                "params": {},
            }

    rules = [
        ("learning_rate_tuning", ["lr", "learning rate"], "Tune optimizer learning rate."),
        ("kl_weight_tuning", ["kl", "kld"], "Tune ACT CVAE KL regularization."),
        ("scheduler", ["scheduler", "cosine", "anneal_lr"], "Use an LR schedule for training."),
        ("gradient_clipping", ["gradclip", "grad_clip", "clip"], "Clip gradients for transformer stability."),
        ("kl_annealing", ["klanneal", "kl_anneal"], "Warm up KL regularization during training."),
        ("gradient_accumulation", ["gradaccum", "grad_accum"], "Accumulate gradients to alter effective batch size."),
        ("combined_training_recipe", ["combo", "combined"], "Combine multiple training changes."),
        ("baseline", ["baseline"], "Reference candidate, not an optimization."),
    ]
    family = "unknown"
    hypothesis = candidate.description or "Unspecified optimization attempt."
    tags = []
    for fam, keys, default_hypothesis in rules:
        if any(k in text for k in keys):
            family = fam
            hypothesis = candidate.description or default_hypothesis
            tags.append(fam)
            break

    params = {}
    for key in ("lr", "kl", "chunk"):
        match = re.search(rf"{key}([0-9.eE+-]+)", name)
        if match:
            params[key] = match.group(1)
    return {
        "family": family,
        "tags": sorted(set(tags + [candidate.kind])),
        "hypothesis": hypothesis,
        "params": params,
    }


class OptimizationExperienceGraph:
    """Read/write graph of optimization attempts and lessons."""

    def __init__(self, output_dir: str | Path, direction: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.direction = direction
        self.path = self.output_dir / "optimization_experience_graph.json"
        self.summary_path = self.output_dir / "optimization_experience_summary.md"
        self.nodes: Dict[str, ExperienceNode] = {}
        self.edges: List[ExperienceEdge] = []
        self.families: Dict[str, FamilyStats] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.nodes = {
            k: ExperienceNode(**v) for k, v in data.get("nodes", {}).items()
        }
        self.edges = [ExperienceEdge(**e) for e in data.get("edges", [])]
        self.families = {
            k: FamilyStats(**v) for k, v in data.get("families", {}).items()
        }

    def _is_better(self, score: float, baseline: float) -> bool:
        return score > baseline if self.direction == "higher" else score < baseline

    def _effect(self, score: float, baseline: float, delta_abs: float) -> str:
        if self._is_better(score, baseline):
            return "positive"
        if abs(delta_abs) < 1e-12:
            return "neutral"
        return "negative"

    def _lesson(self, candidate: Candidate, family: str, effect: str, score: float, baseline: float) -> str:
        if effect == "positive":
            return (
                f"Keep and expand {family}: {candidate.name} beat baseline "
                f"({score:.4f} vs {baseline:.4f})."
            )
        if effect == "neutral":
            return (
                f"{family} matched baseline for {candidate.name}; keep as reference, "
                "but require a stronger variant before spending more budget."
            )
        return (
            f"Do not repeat {family} blindly: {candidate.name} underperformed "
            f"baseline ({score:.4f} vs {baseline:.4f}); change the hypothesis or combine with new evidence."
        )

    def prior_for(self, candidate: Candidate) -> Dict[str, Any]:
        meta = classify_candidate(candidate)
        family = meta["family"]
        stats = self.families.get(family)
        if not stats:
            return {"family": family, "seen": False, "advice": "No prior attempts in this family."}
        advice = "Promising family; prioritize variants." if stats.positive else (
            "Repeatedly weak family; only continue with a materially different hypothesis."
            if stats.negative >= 2 and stats.positive == 0 else
            "Mixed/limited evidence; evaluate cautiously."
        )
        return {
            "family": family,
            "seen": True,
            "attempts": stats.attempts,
            "best_candidate": stats.best_candidate,
            "best_score": stats.best_score,
            "positive": stats.positive,
            "neutral": stats.neutral,
            "negative": stats.negative,
            "advice": advice,
            "lessons": stats.lessons[-3:],
        }

    def should_skip_known_bad(self, candidate: Candidate, min_attempts: int = 2) -> bool:
        prior = self.prior_for(candidate)
        if not prior.get("seen"):
            return False
        return (
            prior.get("attempts", 0) >= min_attempts
            and prior.get("positive", 0) == 0
            and prior.get("negative", 0) >= min_attempts
        )

    def record(
        self,
        candidate: Candidate,
        result: EvalResult,
        baseline_score: float,
        delta_abs: float,
        delta_pct: Optional[float],
    ) -> Dict[str, Any]:
        meta = classify_candidate(candidate)
        family = meta["family"]
        effect = self._effect(result.score, baseline_score, delta_abs)
        lesson = self._lesson(candidate, family, effect, result.score, baseline_score)
        node_id = f"{candidate.name}:{len(self.nodes) + 1}"
        node = ExperienceNode(
            id=node_id,
            candidate=candidate.name,
            family=family,
            tags=meta["tags"],
            hypothesis=meta["hypothesis"],
            score=result.score,
            baseline_score=baseline_score,
            delta_abs=delta_abs,
            delta_pct=delta_pct,
            effect=effect,
            lesson=lesson,
            metrics=result.metrics,
            timestamp=datetime.now().isoformat(),
        )
        for other in self.nodes.values():
            if other.family == family:
                self.edges.append(
                    ExperienceEdge(
                        source=other.id,
                        target=node_id,
                        relation="same_method_family",
                        reason=f"Both attempts belong to {family}.",
                    )
                )
        self.nodes[node_id] = node
        stats = self.families.setdefault(family, FamilyStats(family=family))
        stats.attempts += 1
        if effect == "positive":
            stats.positive += 1
        elif effect == "neutral":
            stats.neutral += 1
        else:
            stats.negative += 1
        if stats.best_score is None or self._is_better(result.score, stats.best_score):
            stats.best_score = result.score
            stats.best_candidate = candidate.name
        if lesson not in stats.lessons:
            stats.lessons.append(lesson)
        self.write()
        return {
            "family": family,
            "tags": meta["tags"],
            "hypothesis": meta["hypothesis"],
            "effect": effect,
            "lesson": lesson,
            "prior": self.prior_for(candidate),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "direction": self.direction,
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges],
            "families": {k: asdict(v) for k, v in self.families.items()},
            "recommendations": self.recommendations(),
        }

    def recommendations(self) -> Dict[str, List[str]]:
        promising = []
        discouraged = []
        uncertain = []
        for family, stats in sorted(self.families.items()):
            if stats.positive:
                promising.append(f"{family}: best={stats.best_candidate} ({stats.best_score})")
            elif stats.attempts >= 2 and stats.negative >= stats.attempts:
                discouraged.append(f"{family}: {stats.attempts} negative attempts")
            else:
                uncertain.append(f"{family}: {stats.attempts} attempts")
        return {
            "promising": promising,
            "discouraged": discouraged,
            "uncertain": uncertain,
        }

    def write(self) -> None:
        self.path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False))
        lines = ["# Optimization Experience Summary", ""]
        recs = self.recommendations()
        for title, items in [
            ("Promising", recs["promising"]),
            ("Discouraged", recs["discouraged"]),
            ("Uncertain", recs["uncertain"]),
        ]:
            lines.append(f"## {title}")
            if items:
                lines.extend(f"- {item}" for item in items)
            else:
                lines.append("- None")
            lines.append("")
        lines.append("## Family Lessons")
        for family, stats in sorted(self.families.items()):
            lines.append(f"### {family}")
            lines.append(
                f"- attempts={stats.attempts}, positive={stats.positive}, "
                f"neutral={stats.neutral}, negative={stats.negative}, "
                f"best={stats.best_candidate}:{stats.best_score}"
            )
            for lesson in stats.lessons[-5:]:
                lines.append(f"- {lesson}")
            lines.append("")
        self.summary_path.write_text("\n".join(lines))
