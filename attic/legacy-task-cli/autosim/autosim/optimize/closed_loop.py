"""
通用闭环优化器 — 对标 AutoSOTA Phase 3

任何 TaskAdapter 都可以用这个优化器，LLM 每轮参与决策。
"""

import os, sys, json, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from autosim.adapters.base import TaskAdapter, ParamDef, EvalResult
from autosim.llm_client import LLMClient

AUTOSIM_HOME = Path(__file__).parent.parent.parent  # autosim/ root


def _load_env():
    env_file = AUTOSIM_HOME / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


class ClosedLoopOptimizer:
    """
    通用闭环优化器。

    Usage:
        adapter = ACTAdapter()
        opt = ClosedLoopOptimizer(adapter)
        result = opt.run(max_iterations=8)
    """

    def __init__(
        self,
        adapter: TaskAdapter,
        output_dir: str = "output",
        allow_llm: bool = True,
        idea_file: Optional[str] = None,
    ):
        _load_env()
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.allow_llm = allow_llm
        self.idea_file = Path(idea_file) if idea_file else None
        self.llm = LLMClient() if allow_llm else None
        self._llm_failure_reported = False

    def _call_llm(self, system: str, user: str) -> str:
        if not self.llm or not self.llm.available:
            return ""
        try:
            return self.llm.chat(system, user, max_tokens=4096, timeout=120)
        except Exception as exc:
            if not self._llm_failure_reported:
                print(f"  LLM unavailable, falling back to local search: {exc}")
                self._llm_failure_reported = True
            self.llm = None
            return ""

    def _analyze_code(self, sources: Dict[str, str]) -> str:
        """Phase 1: LLM analyzes source code."""
        source_text = "\n\n".join(
            f"### {fn}\n```python\n{fc[:4000]}\n```" for fn, fc in sources.items()
        )
        system = "Expert ML/robotics engineer. Analyze this codebase for optimization."
        user = f"""Analyze this code:
{source_text}

Identify:
1. All tunable parameters with defaults
2. Architecture bottlenecks
3. Most promising optimization directions
4. Red lines (must not change)"""
        return self._call_llm(system, user)

    def _suggest_next(self, history: List[Dict], param_space: Dict,
                      best_score: float, direction: str) -> Dict:
        """LLM suggests next hyperparams based on history."""
        param_history = [h for h in history if isinstance(h.get("params"), dict)]
        hist_text = "\n".join(
            f"  [{h['iter']}] params={h['params']} -> score={h['score']:.4f}"
            + (" ★" if h['score'] == best_score else "")
            for h in param_history
        )

        param_desc = "\n".join(
            f"- {p.name}: default={p.default}, range={p.range}, type={p.dtype}"
            for p in param_space.values()
        )

        system = "Expert optimization engineer. Suggest the NEXT hyperparameters to try."
        user = f"""History ({direction} is better):
{hist_text}

Best: score={best_score:.4f}

Parameter space:
{param_desc}

Analyze trends and suggest next params. If improving → continue direction.
If plateauing → try structural change (Leap). If failing → revert direction.

Return JSON: {{"params": {{"name": value, ...}}, "analysis": "why this choice"}}"""

        response = self._call_llm(system, user)
        m = re.search(r'\{[\s\S]*"params"[\s\S]*\}', response)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        # Fallback: deterministic local search around the best known parameter set.
        import random
        if not param_history:
            best_params = {name: pdef.default for name, pdef in param_space.items()}
        else:
            best_entry = max(param_history, key=lambda h: h["score"]) if direction == "higher" else \
                min(param_history, key=lambda h: h["score"])
            best_params = best_entry["params"]
        seed_material = json.dumps(param_history, sort_keys=True, default=str)
        rng = random.Random(seed_material)
        fallback = {}
        for name, pdef in param_space.items():
            lo, hi = pdef.range
            val = best_params.get(name, pdef.default)
            scale = 0.15 if len(param_history) < 4 else 0.08
            perturb = rng.uniform(-scale, scale) * (hi - lo)
            new_val = val + perturb
            if pdef.dtype == "int":
                fallback[name] = max(int(lo), min(int(hi), int(round(new_val))))
            else:
                fallback[name] = round(max(lo, min(hi, new_val)), 6)
        return {"params": fallback, "analysis": f"fallback: perturb from best ({best_params})"}

    def _generate_code_ideas(self, sources: Dict[str, str], analysis: str,
                            baseline: float) -> list:
        """Phase 2: LLM generates CODE/ALGO ideas with exact code diffs."""
        source_text = "\n\n".join(
            f"### {fn}\n```python\n{fc}\n```" for fn, fc in sources.items()
        )
        system = "Expert engineer. Propose code changes to improve this model. Output ONLY JSON."
        direction = self.adapter.get_metric_direction()
        user = f"""Code analysis:
{analysis[:4000]}

Source files:
{source_text[:6000]}

Baseline score: {baseline:.4f} ({direction} is better)

Propose 4-6 CODE/ALGO improvements. For each, provide EXACT old_code → new_code diff.
Return JSON:
{{"ideas": [
  {{"tier": "CODE", "title": "...", "file": "policy/ACT/xxx.py",
    "old_code": "exact original lines from source", "new_code": "exact replacement",
    "rationale": "why this helps"}}
]}}
IMPORTANT: old_code MUST be copied verbatim from the source files above."""

        response = self._call_llm(system, user)
        # Parse JSON from response
        m = re.search(r'\{[\s\S]*"ideas"[\s\S]*\}', response)
        ideas = []
        if m:
            try:
                ideas = json.loads(m.group()).get("ideas", [])
            except json.JSONDecodeError:
                pass
        return ideas

    def _load_offline_ideas(self) -> list:
        """Load ALGO/CODE/PARAM ideas from a local JSON file for offline tests."""
        if not self.idea_file:
            return []
        payload = json.loads(self.idea_file.read_text())
        if isinstance(payload, list):
            return payload
        return payload.get("ideas", [])

    def run(self, max_iterations: int = 8) -> Dict:
        """Run closed-loop optimization with CODE/ALGO + PARAM layers."""
        param_space = self.adapter.get_param_space()
        direction = self.adapter.get_metric_direction()
        metric_name = self.adapter.get_primary_metric_name()

        print("=" * 60)
        print(f"  AutoSim Closed-Loop Optimizer (ALGO→CODE→PARAM)")
        print(f"  Metric: {metric_name} ({direction} is better)")
        print("=" * 60)

        # ── Phase 1: Code Analysis ──
        print("\n── Phase 1: Code Analysis ──")
        sources = self.adapter.get_source_files()
        analysis = self._analyze_code(sources) if sources and self.allow_llm else ""
        mem = self.output_dir / "memory"; mem.mkdir(exist_ok=True)
        if analysis: (mem / "code_analysis.md").write_text(analysis)
        if analysis:
            print(f"  Analysis: {len(analysis)} chars")
        elif not self.allow_llm:
            print("  Analysis skipped (--no-llm)")
        else:
            print("  Analysis skipped (no API key or no source files)")

        # ── Phase 2: Baseline ──
        print("\n── Phase 2: Baseline ──")
        defaults = {p.name: p.default for p in param_space.values()}
        result = self.adapter.evaluate(defaults)
        best_score = result.score
        history = [{"iter": 0, "params": dict(defaults), "score": result.score,
                     "success": result.success, "type": "baseline",
                     "info": result.info, "metrics": result.metrics}]
        print(f"  {metric_name}={best_score:.4f} success={result.success}")
        if not result.success and result.info:
            print(f"  info: {result.info}")

        # ── Phase 3: CODE/ALGO ideas via LLM ──
        print("\n── Phase 3: CODE/ALGO Optimization ──")
        offline_ideas = self._load_offline_ideas()
        if offline_ideas:
            code_ideas = offline_ideas
            print(f"  Loaded {len(code_ideas)} offline CODE/ALGO ideas")
        else:
            code_ideas = self._generate_code_ideas(sources, analysis, best_score) if analysis else []
            print(f"  LLM generated {len(code_ideas)} CODE/ALGO ideas")

        algo_code_count = 0
        if code_ideas:
            skipped_code_algo = 0
            for i, idea in enumerate(code_ideas):
                tier = idea.get("tier", "CODE")
                title = idea.get("title", f"idea-{i}")
                old = idea.get("old_code", "")
                new = idea.get("new_code", "")
                fname = idea.get("file", "")

                if not old or not new:
                    continue

                attempt_no = algo_code_count + skipped_code_algo + 1
                print(f"\n  [{attempt_no}] [{tier}] {title[:80]}")
                print(f"      file: {fname}")

                # 应用代码变更
                applied = self.adapter.apply_change(fname, old, new)
                if not applied:
                    skipped_code_algo += 1
                    print(f"      ✗ old_code not found in source — skipping")
                    continue

                print(f"      ✓ code change applied")
                algo_code_count += 1

                # 评估；代码变更失败时必须回滚并继续下一条 idea。
                try:
                    result = self.adapter.evaluate(defaults)
                    score = result.score
                except Exception as exc:
                    self.adapter.revert_all()
                    print(f"      ✗ evaluation failed, reverted: {exc}")
                    continue

                is_better = (direction == "lower" and score < best_score) or \
                           (direction == "higher" and score > best_score)

                if is_better:
                    best_score = score
                    self.adapter.accept_all_changes()
                    print(f"      ★ BEST: {metric_name}={score:.4f} (kept)")
                else:
                    self.adapter.revert_all()
                    print(f"      {metric_name}={score:.4f} (reverted)")

                history.append({
                    "iter": algo_code_count, "type": tier, "title": title,
                    "file": fname, "score": score, "success": result.success,
                    "improved": is_better,
                    "old_code": old[:80], "new_code": new[:80],
                })

        # ── Phase 4: PARAM fine-tuning ──
        print(f"\n── Phase 4: PARAM Fine-tuning ({max_iterations} iters) ──")
        no_improve = 0
        param_iter = algo_code_count

        for it in range(1, max_iterations + 1):
            suggestion = self._suggest_next(history, param_space, best_score, direction)
            idea_params = suggestion.get("params", {})
            analysis_text = suggestion.get("analysis", "")[:120]

            trial_params = dict(defaults)
            for k, v in idea_params.items():
                if k in trial_params:
                    pdef = param_space[k]
                    lo, hi = pdef.range
                    trial_params[k] = max(lo, min(hi,
                        int(float(v)) if pdef.dtype == "int" else float(v)))

            param_iter += 1
            print(f"\n  [{param_iter}] PARAM {analysis_text}")
            print(f"      params={trial_params}")

            result = self.adapter.evaluate(trial_params)
            score = result.score
            is_better = (direction == "lower" and score < best_score) or \
                       (direction == "higher" and score > best_score)

            if is_better:
                best_score = score; no_improve = 0
                print(f"      ★ BEST: {metric_name}={score:.4f}")
            else:
                no_improve += 1
                print(f"      {metric_name}={score:.4f} (best={best_score:.4f})")

            history.append({"iter": param_iter, "type": "PARAM", "params": dict(trial_params),
                           "score": score, "success": result.success,
                           "metrics": result.metrics, "info": result.info,
                           "improved": is_better, "note": analysis_text[:100]})

            if no_improve >= 3:
                print(f"      ⚠ {no_improve} rounds no improvement")

            if direction == "lower" and history[0]['score'] > 0 and best_score < 0.3 * history[0]['score']:
                print(f"\n  🎉 Converged!")
                break

        # ── Report ──
        baseline_score = history[0]['score']
        if baseline_score in (0, 999.0):
            improv = 0.0
        elif direction == "higher":
            improv = (best_score - history[0]['score']) / history[0]['score'] * 100
        else:
            improv = (history[0]['score'] - best_score) / history[0]['score'] * 100

        print(f"\n{'=' * 60}")
        print(f"  Optimization Complete")
        print(f"{'=' * 60}")
        print(f"  Baseline:        {history[0]['score']:.4f}")
        print(f"  Best:            {best_score:.4f} ({improv:+.1f}%)")
        print(f"  CODE/ALGO tried: {algo_code_count}")
        param_count = sum(1 for h in history if h.get("type") == "PARAM")
        print(f"  PARAM iterations: {param_count}")
        print(f"  History:")
        for h in history:
            m = "★" if h.get("improved") or h.get("type") == "baseline" else " "
            t = h.get("type", "?")
            if t in ("CODE", "ALGO"):
                print(f"    [{h['iter']}] {m} [{t}] {h.get('title','?')[:60]} → {h['score']:.4f}")
            else:
                print(f"    [{h['iter']}] {m} [{t}] {h.get('params','?')} → {h['score']:.4f}")

        meta = {"baseline_score": history[0]['score'], "best_score": best_score,
                "improvement_pct": improv, "metric": metric_name,
                "code_ideas_tried": algo_code_count,
                "algo_code_count": algo_code_count,
                "param_iterations": param_count,
                "history": history}
        with open(self.output_dir / "optimization_result.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return meta


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    from autosim.adapters.act_adapter import ACTAdapter
    adapter = ACTAdapter(epochs=80)
    opt = ClosedLoopOptimizer(adapter, output_dir="output/act_generic_test")
    opt.run(max_iterations=5)
