"""Deterministic benchmark candidate generation."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Dict, Iterable, List, Set

from autosim.adapters.base import Candidate, ParamDef
from autosim.benchmark.spec import BenchmarkSpec


class BenchmarkCandidateGenerator:
    """Generate benchmark candidates from a spec and adapter parameter space.

    This is intentionally deterministic so benchmark runs are reproducible.
    Later reflectors can reorder or append candidates, but the base generator
    provides a stable starting point.
    """

    def __init__(
        self,
        spec: BenchmarkSpec,
        param_space: Dict[str, ParamDef],
        llm_candidates: List[Candidate] | None = None,
    ):
        self.spec = spec
        self.param_space = param_space
        self._llm_candidates = llm_candidates or []
        self._train_candidates = self._build_train_candidates()
        self._checkpoint_candidates = self._build_checkpoint_candidates()
        self._llm_cursor = 0
        self._train_cursor = 0
        self._checkpoint_cursor = 0
        self._calls = 0

    def _default_params(self) -> Dict[str, float]:
        return {name: pdef.default for name, pdef in self.param_space.items()}

    def _space_values(self) -> Dict[str, List[float]]:
        defaults = self._default_params()
        values = {}
        for name, default in defaults.items():
            configured = self.spec.search_space.get(name)
            if configured:
                values[name] = list(configured)
            else:
                values[name] = [default]
        return values

    def _candidate_name(self, params: Dict[str, float]) -> str:
        key = json.dumps(params, sort_keys=True)
        digest = hashlib.sha1(key.encode()).hexdigest()[:8]
        return f"bench_{self.spec.task}_{digest}"

    def _build_train_candidates(self) -> List[Candidate]:
        values = self._space_values()
        names = list(values)
        candidates = []
        for combo in itertools.product(*(values[name] for name in names)):
            params = dict(zip(names, combo))
            if params == self._default_params():
                continue
            candidates.append(
                Candidate(
                    name=self._candidate_name(params),
                    kind="train_recipe",
                    payload={
                        "params": params,
                        "benchmark": self.spec.to_dict(),
                        "recipe_type": "act_hparam_train",
                    },
                    description="benchmark-generated ACT training recipe",
                )
            )
        return candidates

    def _build_checkpoint_candidates(self) -> List[Candidate]:
        candidates = []
        for name in self.spec.checkpoint_candidates:
            candidates.append(
                Candidate(
                    name=name,
                    kind="checkpoint",
                    payload={"ckpt_setting": name},
                    description="benchmark checkpoint candidate from prior ALGO/CODE/ARCH runs",
                )
            )
        return candidates

    def _take_from(self, candidates: List[Candidate], cursor_attr: str, seen: Set[str], limit: int) -> List[Candidate]:
        batch = []
        cursor = getattr(self, cursor_attr)
        while cursor < len(candidates) and len(batch) < limit:
            candidate = candidates[cursor]
            cursor += 1
            if candidate.name in seen:
                continue
            batch.append(candidate)
        setattr(self, cursor_attr, cursor)
        return batch

    def next_batch(self, seen: Set[str], limit: int) -> List[Candidate]:
        self._calls += 1
        batch = self._take_from(self._llm_candidates, "_llm_cursor", seen, limit)
        if batch:
            return batch
        if self._calls == 2 and self._checkpoint_candidates:
            batch = self._take_from(self._checkpoint_candidates, "_checkpoint_cursor", seen, limit)
            if len(batch) < limit:
                batch.extend(
                    self._take_from(self._train_candidates, "_train_cursor", seen, limit - len(batch))
                )
            return batch
        batch = self._take_from(self._train_candidates, "_train_cursor", seen, limit)
        if not batch and self._checkpoint_candidates:
            batch = self._take_from(self._checkpoint_candidates, "_checkpoint_cursor", seen, limit)
        return batch

    def all(self) -> Iterable[Candidate]:
        return list(self._llm_candidates) + list(self._checkpoint_candidates) + list(self._train_candidates)
