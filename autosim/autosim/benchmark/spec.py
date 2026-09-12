"""Benchmark specification models.

BenchmarkSpec is intentionally small and file-friendly. It captures the
contract needed for a benchmark run: what task to evaluate, how to evaluate it,
what baseline to compare against, and the budget/target used by the optimizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BenchmarkSpec:
    """A reproducible benchmark target for system-level optimization."""

    suite: str
    task: str
    task_config: str
    metric: str = "success_rate"
    direction: str = "higher"
    baseline_ckpt: str = "demo_clean-50"
    data_name: Optional[str] = None
    expert_data_num: int = 50
    sota_score: Optional[float] = None
    eval_episodes: int = 10
    eval_seed: int = 0
    eval_seeds: List[int] = field(default_factory=list)
    instruction_type: str = "unseen"
    temporal_agg: bool = True
    max_rounds: int = 1
    candidate_limit: int = 4
    search_space: Dict[str, List[Any]] = field(default_factory=dict)
    checkpoint_candidates: List[str] = field(default_factory=list)
    allowed_candidate_kinds: List[str] = field(
        default_factory=lambda: ["patch_train_recipe", "train_recipe", "checkpoint", "params"]
    )
    notes: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkSpec":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        values = {k: v for k, v in data.items() if k in known}
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))
        return out

    @property
    def target_score(self) -> Optional[float]:
        return self.sota_score
