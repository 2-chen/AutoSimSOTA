"""
Metric tracking and visualization utilities.

Provides:
- Score recording in JSONL format (compatible with AutoSOTA)
- Optimization curve plotting
- Simple statistical analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and persist optimization metrics."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._scores: List[Dict[str, Any]] = []

    def record(self, iteration: int, params: Dict[str, Any],
               metrics: Dict[str, float], primary_score: float,
               success: bool, note: str = ""):
        """Record a single evaluation result."""
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "primary_score": primary_score,
            "metrics": metrics,
            "params": params,
            "success": success,
            "note": note,
        }
        self._scores.append(entry)

    def save(self, filename: str = "scores.jsonl"):
        """Save all scores to JSONL file."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            for entry in self._scores:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Scores saved to {filepath}")

    def get_best(self, direction: str = "lower") -> Optional[Dict[str, Any]]:
        """Get the best score entry."""
        if not self._scores:
            return None
        if direction == "lower":
            return min(self._scores, key=lambda x: x["primary_score"])
        else:
            return max(self._scores, key=lambda x: x["primary_score"])

    def get_score_series(self) -> List[float]:
        """Get list of primary scores in iteration order."""
        return [s["primary_score"] for s in self._scores]

    def plot_curve(self, output_path: Optional[str] = None):
        """Plot optimization curve using matplotlib (if available)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return

        scores = self.get_score_series()
        if not scores:
            return

        iterations = list(range(len(scores)))
        best_so_far = []
        best = float("inf")
        for s in scores:
            best = min(best, s)
            best_so_far.append(best)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(iterations, scores, "o-", alpha=0.5, label="Per-iteration", markersize=6)
        ax.plot(iterations, best_so_far, "r-", linewidth=2, label="Best so far")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3, label="Baseline")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title("Optimization Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = output_path or str(self.output_dir / "optimization_curve.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Optimization curve saved to {path}")
