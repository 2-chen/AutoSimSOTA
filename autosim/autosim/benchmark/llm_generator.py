"""LLM-generated ALGO/CODE benchmark candidates."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from autosim.adapters.base import Candidate, ParamDef, TaskAdapter
from autosim.benchmark.spec import BenchmarkSpec
from autosim.llm_client import LLMClient


class LLMBenchmarkCandidateGenerator:
    """Generate concrete ALGO/CODE patch candidates from adapter source files."""

    def __init__(
        self,
        spec: BenchmarkSpec,
        adapter: TaskAdapter,
        param_space: Dict[str, ParamDef],
        limit: int = 4,
        model: Optional[str] = None,
    ):
        self.spec = spec
        self.adapter = adapter
        self.param_space = param_space
        self.limit = int(limit)
        self.llm = LLMClient(model=model)

    def _default_params(self) -> Dict[str, Any]:
        return {name: pdef.default for name, pdef in self.param_space.items()}

    def _source_context(self, max_chars: int = 24000) -> str:
        chunks = []
        used = 0
        for fname, content in self.adapter.get_source_files().items():
            remaining = max_chars - used
            if remaining <= 0:
                break
            header = f"\n### FILE: {fname}\n```python\n"
            footer = "\n```\n"
            budget = max(0, remaining - len(header) - len(footer))
            chunk = header + content[:budget] + footer
            chunks.append(chunk)
            used += len(chunk)
        return "\n".join(chunks)

    def _param_desc(self) -> str:
        if not self.param_space:
            return "(none declared)"
        return "\n".join(
            f"- {p.name}: default={p.default}, range={p.range}, type={p.dtype}"
            for p in self.param_space.values()
        )

    def _prompt(self) -> tuple[str, str]:
        system = """You generate benchmark candidates for an embodied AI AutoSOTA loop.
Return ONLY valid JSON.

Rules:
- Generate ALGO/CODE changes only; do not produce PARAM-only ideas.
- old_code must be copied verbatim from SOURCE CONTEXT.
- Do not use ellipses, placeholders, prose, or pseudocode in old_code.
- new_code must be a complete replacement for old_code.
- Each idea must be safe to train and evaluate in simulation.
- Prefer changes that can plausibly improve RoboTwin ACT success_rate."""

        user = f"""Benchmark:
- suite: {self.spec.suite}
- task: {self.spec.task}
- task_config: {self.spec.task_config}
- metric: {self.spec.metric} ({self.spec.direction} is better)
- baseline_ckpt: {self.spec.baseline_ckpt}
- target_sota_score: {self.spec.sota_score}

ACT training hyperparameters available for each candidate:
{self._param_desc()}

SOURCE CONTEXT:
{self._source_context()}

Generate up to {self.limit} candidates.

Return JSON:
{{
  "ideas": [
    {{
      "tier": "ALGO",
      "title": "short title",
      "file": "exact file path from SOURCE CONTEXT",
      "old_code": "verbatim source block to replace",
      "new_code": "complete replacement block",
      "params": {{"kl_weight": 10, "lr": 1e-5, "chunk_size": 50}},
      "rationale": "why this should improve simulation success"
    }}
  ]
}}"""
        return system, user

    def _parse(self, text: str) -> List[Dict[str, Any]]:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        payload = match.group(1) if match else text
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\"ideas\"[\s\S]*\}", text)
            if not match:
                return []
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        ideas = data.get("ideas", [])
        return ideas if isinstance(ideas, list) else []

    def _candidate_name(self, idx: int, tier: str, title: str) -> str:
        safe_title = re.sub(r"[^A-Za-z0-9_.-]+", "_", title).strip("._").lower()
        safe_title = safe_title[:48] or "idea"
        return f"llm_{tier.lower()}_{idx:02d}_{safe_title}"

    def _coerce_params(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        params = self._default_params()
        for name, value in (raw or {}).items():
            if name not in self.param_space:
                continue
            pdef = self.param_space[name]
            lo, hi = pdef.range
            try:
                if pdef.dtype == "int":
                    coerced = int(round(float(value)))
                    params[name] = max(int(lo), min(int(hi), coerced))
                else:
                    coerced = float(value)
                    params[name] = max(float(lo), min(float(hi), coerced))
            except (TypeError, ValueError):
                continue
        return params

    def generate(self) -> List[Candidate]:
        if not self.llm.available or self.limit <= 0:
            return []
        system, user = self._prompt()
        response = self.llm.chat(system, user, max_tokens=12000, timeout=240)
        ideas = self._parse(response)
        candidates = []
        for idea in ideas:
            tier = str(idea.get("tier", "CODE")).upper()
            if tier not in {"ALGO", "CODE"}:
                continue
            old_code = idea.get("old_code")
            new_code = idea.get("new_code")
            file_path = idea.get("file")
            if not old_code or not new_code or not file_path:
                continue
            title = str(idea.get("title") or f"{tier} idea")
            idx = len(candidates) + 1
            candidates.append(
                Candidate(
                    name=self._candidate_name(idx, tier, title),
                    kind="patch_train_recipe",
                    payload={
                        "tier": tier,
                        "params": self._coerce_params(idea.get("params", {})),
                        "patches": [{
                            "file": file_path,
                            "old_code": old_code,
                            "new_code": new_code,
                        }],
                        "recipe_type": f"llm_{tier.lower()}_patch_train",
                        "idea": {
                            "title": title,
                            "rationale": idea.get("rationale", ""),
                        },
                        "benchmark": self.spec.to_dict(),
                    },
                    description=f"LLM-generated {tier} patch candidate: {title}",
                )
            )
            if len(candidates) >= self.limit:
                break
        return candidates
