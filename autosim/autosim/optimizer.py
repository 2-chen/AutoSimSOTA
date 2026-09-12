"""
Closed-loop optimizer for embodied AI tasks.

Inspired by AutoSOTA's optimization loop:
  1. Evaluate baseline configuration
  2. Generate candidate parameter sets (ideas)
  3. Execute each candidate in simulation
  4. Record scores and track best
  5. Select next candidates based on history
  6. Iterate until convergence or max iterations

Supports multiple optimization strategies:
  - random: Random uniform sampling within parameter ranges
  - grid: Exhaustive grid search over discretized parameter space
  - hill_climb: Simple local search with perturbation
  - cma_es: Covariance Matrix Adaptation Evolution Strategy (lightweight)
"""

import json
import logging
import time
import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

import numpy as np

from autosim.tasks.task_base import BaseTask, TaskResult
from autosim.config import OptimizerConfig

logger = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    """Record of a single optimization iteration."""
    iteration: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    primary_score: float
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    note: str = ""


@dataclass
class OptimizationRun:
    """Complete record of an optimization run."""
    task_name: str
    primary_metric: str
    metric_direction: str  # "lower" or "higher"
    baseline_score: Optional[float] = None
    best_score: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None
    best_iteration: int = -1
    iterations: List[IterationRecord] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    def is_better(self, new_score: float) -> bool:
        """Check if new_score is better than current best."""
        if self.best_score is None:
            return True
        if self.metric_direction == "lower":
            return new_score < self.best_score
        else:
            return new_score > self.best_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "best_iteration": self.best_iteration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_iterations": len(self.iterations),
            "improvement_pct": self._improvement_pct(),
            "iterations": [
                {
                    "iteration": r.iteration,
                    "params": r.params,
                    "metrics": r.metrics,
                    "primary_score": r.primary_score,
                    "success": r.success,
                    "timestamp": r.timestamp,
                    "note": r.note,
                }
                for r in self.iterations
            ],
        }

    def _improvement_pct(self) -> Optional[float]:
        if self.baseline_score is None or self.best_score is None or self.baseline_score == 0:
            return None
        change = self.best_score - self.baseline_score
        return (change / abs(self.baseline_score)) * 100


# ═══════════════════════════════════════════════════════════════════
# Optimization Strategies
# ═══════════════════════════════════════════════════════════════════

class OptimizationStrategy(ABC):
    """Abstract base for optimization strategies."""

    def __init__(self, param_space: Dict[str, List[float]], seed: int = 42):
        self.param_space = param_space
        self.param_names = list(param_space.keys())
        self.rng = np.random.RandomState(seed)
        self.history: List[Tuple[Dict[str, Any], float]] = []

    @abstractmethod
    def suggest(self) -> Dict[str, Any]:
        """Suggest the next parameter set to try."""
        ...

    def observe(self, params: Dict[str, Any], score: float):
        """Record the result of a trial."""
        self.history.append((params.copy(), score))

    def _sample_uniform(self) -> Dict[str, Any]:
        """Sample a random point from the parameter space."""
        params = {}
        for name, (lo, hi) in self.param_space.items():
            params[name] = float(self.rng.uniform(lo, hi))
        return params

    def _clip(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clip parameters to their allowed ranges."""
        clipped = {}
        for name, val in params.items():
            if name in self.param_space:
                lo, hi = self.param_space[name]
                clipped[name] = max(lo, min(hi, val))
            else:
                clipped[name] = val
        return clipped


class RandomSearch(OptimizationStrategy):
    """Simple random uniform sampling."""

    def suggest(self) -> Dict[str, Any]:
        return self._sample_uniform()


class GridSearch(OptimizationStrategy):
    """Exhaustive grid search over discretized parameter space."""

    def __init__(self, param_space: Dict[str, List[float]], grid_points: int = 5, seed: int = 42):
        super().__init__(param_space, seed)
        self.grid_points = grid_points
        self._grid: List[Dict[str, Any]] = []
        self._index = 0
        self._build_grid()

    def _build_grid(self):
        """Build the Cartesian product grid."""
        axes = []
        for name, (lo, hi) in self.param_space.items():
            axes.append(list(np.linspace(lo, hi, self.grid_points)))

        # Use itertools.product equivalent
        self._grid = []
        self._build_grid_recursive(axes, 0, {})

    def _build_grid_recursive(self, axes: List[List[float]], depth: int, current: Dict[str, Any]):
        if depth == len(axes):
            self._grid.append(current.copy())
            return
        param_name = self.param_names[depth]
        for val in axes[depth]:
            current[param_name] = float(val)
            self._build_grid_recursive(axes, depth + 1, current)

    def suggest(self) -> Dict[str, Any]:
        if self._index >= len(self._grid):
            # Wrap around or return random
            return self._sample_uniform()
        params = self._grid[self._index]
        self._index += 1
        return params

    @property
    def total_points(self) -> int:
        return len(self._grid)


class HillClimbing(OptimizationStrategy):
    """
    Simple hill climbing with random restarts.

    Starts from best-known point and perturbs one parameter at a time.
    If no improvement after `patience` steps, restarts from random position.
    """

    def __init__(self, param_space: Dict[str, List[float]], seed: int = 42,
                 step_size: float = 0.1, patience: int = 5):
        super().__init__(param_space, seed)
        self.step_size = step_size
        self.patience = patience
        self._no_improvement = 0
        self._current: Optional[Dict[str, Any]] = None
        self._best_score: Optional[float] = None

    def suggest(self) -> Dict[str, Any]:
        if self._current is None or self._no_improvement >= self.patience:
            # Random restart
            self._current = self._sample_uniform()
            self._no_improvement = 0
            logger.info("Hill climbing: random restart")
            return self._current

        # Perturb one random parameter from current best
        param_to_perturb = self.rng.choice(self.param_names)
        lo, hi = self.param_space[param_to_perturb]
        range_width = hi - lo
        delta = self.rng.normal(0, self.step_size * range_width)

        new_params = self._current.copy()
        new_params[param_to_perturb] += float(delta)

        return self._clip(new_params)

    def observe(self, params: Dict[str, Any], score: float):
        super().observe(params, score)
        if self._best_score is None or score < self._best_score:
            self._best_score = score
            self._current = params.copy()
            self._no_improvement = 0
        else:
            self._no_improvement += 1


class SimpleCMAES(OptimizationStrategy):
    """
    Lightweight CMA-ES-like strategy.

    Maintains a Gaussian distribution over the parameter space and
    updates the mean/covariance based on the best-performing samples.
    """

    def __init__(self, param_space: Dict[str, List[float]], seed: int = 42,
                 population_size: int = 5, elite_frac: float = 0.4):
        super().__init__(param_space, seed)
        self.population_size = population_size
        self.elite_count = max(1, int(population_size * elite_frac))

        # Initialize mean at center of each range
        self.mean = np.array([
            (lo + hi) / 2.0 for lo, hi in param_space.values()
        ])
        # Initial std: 1/6 of range (covers ~99.7% within bounds with 3-sigma)
        self.std = np.array([
            (hi - lo) / 6.0 for lo, hi in param_space.values()
        ])

        self._pending_batch: List[Dict[str, Any]] = []
        self._batch_scores: List[float] = []
        self._generation = 0

    def suggest(self) -> Dict[str, Any]:
        if not self._pending_batch:
            # Generate new batch
            self._generation += 1
            self._pending_batch = []
            self._batch_scores = []

            samples = self.rng.normal(
                self.mean, self.std,
                size=(self.population_size, len(self.mean))
            )

            for i in range(self.population_size):
                params = {}
                for j, name in enumerate(self.param_names):
                    params[name] = float(samples[i, j])
                self._pending_batch.append(self._clip(params))

        return self._pending_batch.pop(0)

    def observe(self, params: Dict[str, Any], score: float):
        super().observe(params, score)
        self._batch_scores.append(score)

        # Once we have a full batch, update distribution
        if len(self._batch_scores) >= self.population_size and not self._pending_batch:
            self._update_distribution()

    def _update_distribution(self):
        """Update mean and std based on elite samples."""
        # Sort history by score (lower is better, take last pop_size entries)
        batch = self.history[-self.population_size:]
        batch_sorted = sorted(batch, key=lambda x: x[1])

        # Elite: best `elite_count` samples
        elite = batch_sorted[:self.elite_count]
        elite_params = np.array([
            [p[name] for name in self.param_names] for p, _ in elite
        ])

        # Update mean toward elite center
        new_mean = elite_params.mean(axis=0)
        lr_mean = 0.3
        self.mean = (1 - lr_mean) * self.mean + lr_mean * new_mean

        # Update std based on elite variance
        if len(elite) > 1:
            new_std = elite_params.std(axis=0)
            lr_std = 0.2
            self.std = (1 - lr_std) * self.std + lr_std * new_std
            # Don't let std collapse
            min_std = np.array([
                (hi - lo) * 0.01 for lo, hi in self.param_space.values()
            ])
            self.std = np.maximum(self.std, min_std)

        logger.info(f"CMA-ES gen {self._generation}: mean={self.mean.round(3)}, "
                     f"best_score={batch_sorted[0][1]:.4f}")


# ═══════════════════════════════════════════════════════════════════
# Strategy Factory
# ═══════════════════════════════════════════════════════════════════

STRATEGIES = {
    "random": RandomSearch,
    "grid": GridSearch,
    "hill_climb": HillClimbing,
    "cma_es": SimpleCMAES,
}


def create_strategy(name: str, param_space: Dict[str, List[float]], **kwargs) -> OptimizationStrategy:
    """Create an optimization strategy by name."""
    strategy_cls = STRATEGIES.get(name)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return strategy_cls(param_space, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# Main Optimizer
# ═══════════════════════════════════════════════════════════════════

class EmbodiedOptimizer:
    """
    Main optimization loop for embodied AI tasks.

    Usage:
        task = FrankaReachTask(client, config.task)
        optimizer = EmbodiedOptimizer(task, config.optimizer)
        run = optimizer.optimize(strategy="cma_es")
    """

    def __init__(self, task: BaseTask, opt_config: OptimizerConfig):
        self.task = task
        self.config = opt_config
        self.run: Optional[OptimizationRun] = None

    def optimize(
        self,
        strategy: str = "random",
        output_dir: Optional[str] = None,
        callback: Optional[Callable[[int, Dict[str, Any], float], None]] = None,
    ) -> OptimizationRun:
        """
        Run the closed-loop optimization.

        Args:
            strategy: Optimization strategy name
            output_dir: Directory for saving results
            callback: Optional callback(iteration, params, score) after each eval

        Returns:
            OptimizationRun with full history and best result
        """
        # Initialize run tracking
        task_config = self.task.config
        self.run = OptimizationRun(
            task_name=self.task.name,
            primary_metric=task_config.primary_metric if task_config else "score",
            metric_direction=task_config.metric_direction if task_config else "lower",
        )

        # Set up the scene
        logger.info("Setting up simulation scene...")
        if not self.task.setup_scene():
            logger.error("Scene setup failed")
            return self.run

        # Create optimization strategy
        param_space = self.task.param_space
        opt = create_strategy(
            strategy,
            param_space,
            seed=self.config.seed,
        )
        logger.info(f"Optimization strategy: {strategy}")
        logger.info(f"Parameter space: {list(param_space.keys())}")
        logger.info(f"Max iterations: {self.config.max_iterations}")

        # ── Iteration 0: Baseline ──
        logger.info("\n" + "=" * 60)
        logger.info("ITERATION 0: BASELINE")
        logger.info("=" * 60)

        baseline_params = self._get_default_params()
        baseline_result = self._evaluate_with_retry(baseline_params, iteration=0)
        baseline_score = baseline_result.primary_score

        self.run.baseline_score = baseline_score
        self.run.best_score = baseline_score
        self.run.best_params = baseline_params
        self.run.best_iteration = 0

        self.run.iterations.append(IterationRecord(
            iteration=0,
            params=baseline_params,
            metrics=baseline_result.metrics,
            primary_score=baseline_score,
            success=baseline_result.success,
            note="baseline",
        ))

        opt.observe(baseline_params, baseline_score)

        logger.info(f"Baseline score ({self.run.primary_metric}): {baseline_score:.6f}")
        logger.info(f"Baseline params: {self._format_params(baseline_params)}")

        # ── Optimization loop ──
        target_delta = task_config.target_improvement_pct / 100.0 if task_config else 0.05

        for iteration in range(1, self.config.max_iterations + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"ITERATION {iteration}/{self.config.max_iterations}")
            logger.info(f"{'=' * 60}")

            # Get next candidate
            params = opt.suggest()
            logger.info(f"Trying params: {self._format_params(params)}")

            # Evaluate
            result = self._evaluate_with_retry(params, iteration=iteration)
            score = result.primary_score

            # Record
            note = ""
            if self.run.is_better(score):
                note = "★ NEW BEST"
                self.run.best_score = score
                self.run.best_params = params.copy()
                self.run.best_iteration = iteration
                logger.info(f"  {note}: {score:.6f} (was {self.run.best_score})")

            improvement = ""
            if self.run.baseline_score and self.run.baseline_score != 0:
                pct = ((score - self.run.baseline_score) / abs(self.run.baseline_score)) * 100
                improvement = f"  (Δ={pct:+.2f}% vs baseline)"

            logger.info(f"Score: {score:.6f}{improvement}")

            self.run.iterations.append(IterationRecord(
                iteration=iteration,
                params=params,
                metrics=result.metrics,
                primary_score=score,
                success=result.success,
                note=note,
            ))

            # Feed back to strategy
            opt.observe(params, score)

            # Callback
            if callback:
                callback(iteration, params, score)

            # Check convergence
            if self._check_converged(target_delta):
                logger.info(f"\n✓ Optimization converged! Target improvement reached.")
                break

        self.run.end_time = datetime.now().isoformat()

        # ── Summary ──
        self._print_summary()

        # Save results
        if output_dir:
            self._save_results(output_dir)

        # Cleanup
        self.task.teardown()

        return self.run

    def _evaluate_with_retry(self, params: Dict[str, Any], iteration: int) -> TaskResult:
        """Evaluate with retry on failure."""
        last_error = None
        for attempt in range(self.config.max_debug_attempts):
            try:
                result = self.task.execute(params)
                if result.success or result.metrics:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)

            if attempt < self.config.max_debug_attempts - 1:
                logger.warning(f"  Attempt {attempt + 1} failed: {last_error}. Retrying...")
                time.sleep(1)

        logger.error(f"  All {self.config.max_debug_attempts} attempts failed")
        return TaskResult(
            success=False,
            metrics={"distance_to_target": float("inf")},
            error=last_error,
        )

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default (center-of-range) parameters."""
        params = {}
        for name, (lo, hi) in self.task.param_space.items():
            params[name] = (lo + hi) / 2.0
        return params

    def _check_converged(self, target_delta: float) -> bool:
        """Check if optimization has converged."""
        if self.run is None or self.run.baseline_score is None or self.run.best_score is None:
            return False
        if self.run.baseline_score == 0:
            return False

        actual_delta = abs((self.run.best_score - self.run.baseline_score) / self.run.baseline_score)
        direction = self.run.metric_direction
        improved = (direction == "lower" and self.run.best_score < self.run.baseline_score) or \
                   (direction == "higher" and self.run.best_score > self.run.baseline_score)

        return improved and actual_delta >= target_delta

    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters for display."""
        items = [f"{k}={v:.4f}" for k, v in params.items()]
        return "{" + ", ".join(items[:4]) + ("..." if len(items) > 4 else "") + "}"

    def _print_summary(self):
        """Print optimization summary."""
        run = self.run
        if run is None:
            return

        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"  Task:              {run.task_name}")
        print(f"  Metric:            {run.primary_metric} ({run.metric_direction} is better)")
        print(f"  Iterations:        {len(run.iterations)}")
        print(f"  Baseline score:    {run.baseline_score:.6f}")
        print(f"  Best score:        {run.best_score:.6f} (iter {run.best_iteration})")

        if run._improvement_pct() is not None:
            pct = run._improvement_pct()
            direction = "↓" if run.metric_direction == "lower" else "↑"
            print(f"  Improvement:       {pct:+.2f}% {direction}")

        print(f"  Best params:       {self._format_params(run.best_params or {})}")
        print("=" * 60)

    def _save_results(self, output_dir: str):
        """Save optimization results to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save full run
        run_file = output_path / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(run_file, "w") as f:
            json.dump(self.run.to_dict(), f, indent=2)
        logger.info(f"Results saved to {run_file}")

        # Save scores as JSONL (compatible with AutoSOTA format)
        scores_file = output_path / "scores.jsonl"
        with open(scores_file, "w") as f:
            for rec in self.run.iterations:
                f.write(json.dumps({
                    "iteration": rec.iteration,
                    "primary_score": rec.primary_score,
                    "metrics": rec.metrics,
                    "success": rec.success,
                    "params": rec.params,
                }) + "\n")
        logger.info(f"Scores saved to {scores_file}")
