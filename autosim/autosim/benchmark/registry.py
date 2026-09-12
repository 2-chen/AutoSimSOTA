"""Benchmark registry helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from autosim.benchmark.spec import BenchmarkSpec


def default_robotwin_spec(
    task: str = "beat_block_hammer",
    task_config: str = "demo_clean",
    baseline_ckpt: str = "demo_clean-50",
    eval_episodes: int = 10,
    max_rounds: int = 1,
    candidate_limit: int = 4,
    sota_score: Optional[float] = None,
) -> BenchmarkSpec:
    """Return a conservative default spec for RoboTwin ACT benchmarks."""
    return BenchmarkSpec(
        suite="robotwin",
        task=task,
        task_config=task_config,
        baseline_ckpt=baseline_ckpt,
        expert_data_num=50 if task_config == "demo_clean" else 15,
        sota_score=sota_score,
        eval_episodes=eval_episodes,
        max_rounds=max_rounds,
        candidate_limit=candidate_limit,
        search_space={
            "kl_weight": [6, 8, 10, 14],
            "lr": [8e-6, 1e-5, 2.5e-5],
            "chunk_size": [40, 50, 60],
        },
        checkpoint_candidates=[
            "best_200ep",
            "optimized_clean50",
            "finetuned_clean50",
            "autosim_algo1_scheduler",
            "autosim_algo2_gradclip",
            "autosim_algo3_combined",
            "autosim_algo4_klanneal",
            "autosim_code1_gradaccum",
            "autosim_combo_best",
        ],
        notes="Default RoboTwin ACT search grid. Use a YAML spec for benchmark-specific budgets.",
    )


def load_benchmark_spec(
    path: Optional[str],
    task: str,
    task_config: str,
    baseline_ckpt: str,
    eval_episodes: Optional[int],
    max_rounds: Optional[int],
    candidate_limit: Optional[int],
    target_score: Optional[float],
) -> BenchmarkSpec:
    """Load a spec file or construct the default RoboTwin spec from CLI args."""
    if path:
        spec = BenchmarkSpec.from_file(path)
        if eval_episodes is not None:
            spec.eval_episodes = eval_episodes
        if max_rounds is not None:
            spec.max_rounds = max_rounds
        if candidate_limit is not None:
            spec.candidate_limit = candidate_limit
        if target_score is not None:
            spec.sota_score = target_score
        return spec
    return default_robotwin_spec(
        task=task,
        task_config=task_config,
        baseline_ckpt=baseline_ckpt,
        eval_episodes=eval_episodes if eval_episodes is not None else 10,
        max_rounds=max_rounds if max_rounds is not None else 1,
        candidate_limit=candidate_limit if candidate_limit is not None else 4,
        sota_score=target_score,
    )
