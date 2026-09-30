"""
Generic LLM-driven optimizer — AutoSOTA-style closed-loop optimization.

Completely decoupled from RoboTwin/SAPIEN. Works with ANY TaskAdapter.

Architecture:
  Phase 1 — Code Analysis:     LLM reads adapter source files, produces deep analysis
  Phase 2 — Idea Generation:   LLM generates ALGO/CODE/PARAM ideas with exact diffs
  Phase 3 — Closed-Loop:       Apply → Evaluate → Reflect → Refine
  Phase 4 — PARAM Fine-tune:   Local search around best config after CODE/ALGO

AutoSOTA priority: ALGO > CODE > PARAM
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autosim.adapters.base import TaskAdapter, ParamDef, EvalResult
from autosim.llm_client import LLMClient


def load_env(env_path: Optional[str] = None):
    """Load .env file for API keys."""
    env_file = Path(env_path or (Path(__file__).parent.parent / ".env"))
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


class LLMOptimizer:
    """
    AutoSOTA-style LLM-driven optimizer for embodied AI tasks.

    Uses a TaskAdapter to interact with any simulation/training environment.
    The LLM (DeepSeek/Claude API) drives the optimization loop.

    Usage:
        adapter = IsaacSimAdapter(task_type="grasp", robot_type="franka")
        opt = LLMOptimizer(adapter, output_dir="output/isaac_opt")
        result = opt.run(max_iterations=10)
    """

    def __init__(
        self,
        adapter: TaskAdapter,
        output_dir: str = "output/llm_opt",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model: Optional[str] = None,
        eval_seeds: int = 5,
    ):
        load_env()
        self.adapter = adapter
        self.eval_seeds = eval_seeds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "memory").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "results").mkdir(parents=True, exist_ok=True)

        # LLM config (OpenAI-compatible API, with env fallback)
        self.llm = LLMClient(api_key=api_key, base_url=api_base_url, model=model)
        self._llm_unavailable = False

        # Evaluation state
        self.history: List[Dict] = []
        self.best_score: Optional[float] = None
        self.best_params: Optional[Dict] = None
        self.baseline_score: Optional[float] = None

    # ══════════════════════════════════════════════════════════════
    # LLM API
    # ══════════════════════════════════════════════════════════════

    def _call_llm(self, system: str, user: str,
                  max_tokens: int = 8192, timeout: int = 180) -> str:
        """Call the LLM API. Returns empty string on failure."""
        if self._llm_unavailable or not self.llm.available:
            return ""

        try:
            return self.llm.chat(system, user, max_tokens=max_tokens, timeout=timeout)
        except Exception as e:
            print(f"  LLM API exception: {e}")
            self._llm_unavailable = True
            return ""

    def _parse_json_from_text(self, text: str) -> Dict:
        """Extract JSON object from LLM response text."""
        # Try code block first
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Try bare JSON
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {}

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Deep Code Analysis
    # ══════════════════════════════════════════════════════════════

    def _build_analysis_prompt(self, sources: Dict[str, str]) -> Tuple[str, str]:
        """Build prompts for deep code analysis."""
        # Format source files for LLM consumption
        ctx_parts = []
        total_chars = 0
        for fname, content in sources.items():
            ctx_parts.append(f"### FILE: {fname} ({len(content)} chars)")
            ctx_parts.append(f"```python\n{content}\n```")
            total_chars += len(content)

        ctx = "\n\n".join(ctx_parts)

        metric_name = self.adapter.get_primary_metric_name()
        metric_dir = self.adapter.get_metric_direction()

        system = """You are an expert robotics/AI engineer analyzing a complete codebase.
Be extremely precise: reference exact function names, line numbers, and variable names.
Use ONLY information from the provided source code."""

        user = f"""Analyze this COMPLETE codebase for a robot learning / embodied AI task.

Primary metric: {metric_name} ({metric_dir} is better)
Number of source files: {len(sources)}
Total code: {total_chars} chars

{ctx}

Write a thorough code analysis (markdown format) covering:

1. **Task/System Architecture**: Break down the overall pipeline step by step.
2. **Core Logic**: How does the main execution loop work? Quote exact code lines.
3. **ALL Parameters**: Table of EVERY tunable numeric value:
   | Variable | Default | File:Line | Description | Suggested Range |
4. **Decision Points**: Every if/else branch, every choice the code makes.
5. **Optimization Levers**: What can be changed to improve performance?
6. **Red Lines**: Code that must NOT be modified (evaluation protocol, metrics).
7. **Failure Modes**: Based on code logic, what can go wrong?
8. **Initial Hypotheses**: 3-5 specific, testable ideas for improvement."""

        return system, user

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Idea Generation
    # ══════════════════════════════════════════════════════════════

    def _build_ideas_prompt(self, analysis: str, baseline: float) -> Tuple[str, str]:
        """Build prompts for generating optimization ideas."""
        sources = self.adapter.get_source_files()
        source_context = self._build_source_context(sources, max_chars=18000)
        task_type = getattr(self.adapter, "task_type", "")
        task_focus = getattr(task_type, "value", task_type) or "unknown"

        system = """You are an expert robotics engineer generating optimization ideas.
CRITICAL rules:
- Every old_code must be copied from SOURCE CONTEXT exactly.
- Do not invent APIs, functions, or code that is not present in SOURCE CONTEXT.
- Do not use ellipses, placeholders, or pseudocode in old_code.
- Focus on the active task type only.
- ALGO ideas change strategy/architecture. CODE ideas change implementation details.
- Do not generate PARAM-only ideas in this phase.
- Priority: ALGO > CODE.
- Return ONLY valid JSON."""

        user = f"""Based on this code analysis:

{analysis[:6000]}

SOURCE CONTEXT (copy old_code verbatim from here only):
{source_context}

Primary metric: {self.adapter.get_primary_metric_name()}
Direction: {self.adapter.get_metric_direction()} is better
Baseline: {baseline:.4f}
Active task type: {task_focus}

Generate 6-10 optimization ideas. At least 4 ALGO and 2 CODE.

For each idea provide:
- "tier": "ALGO" or "CODE"
- "title": Short title
- "file": Target file path from SOURCE CONTEXT
- "old_code": EXACT code to replace (verbatim from source)
- "new_code": EXACT replacement code
- "rationale": Why this should improve the metric

Return valid JSON:
{{"ideas": [
  {{"tier": "ALGO", "title": "...", "file": "...", "old_code": "...", "new_code": "...", "rationale": "..."}}
]}}

IMPORTANT: old_code MUST be copied VERBATIM from the source files shown above."""

        return system, user

    def _build_source_context(self, sources: Dict[str, str], max_chars: int) -> str:
        """Build a bounded source bundle for exact patch generation."""
        chunks = []
        used = 0
        for fname, content in sources.items():
            remaining = max_chars - used
            if remaining <= 0:
                break
            header = f"\n### FILE: {fname}\n```python\n"
            footer = "\n```\n"
            budget = max(0, remaining - len(header) - len(footer))
            body = content[:budget]
            chunks.append(header + body + footer)
            used += len(chunks[-1])
        return "\n".join(chunks)

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Feedback
    # ══════════════════════════════════════════════════════════════

    def _build_feedback_prompt(
        self,
        idea: Dict,
        eval_result: EvalResult,
        iteration: int,
    ) -> str:
        """Build prompt for LLM to reflect on evaluation results."""
        metrics_str = json.dumps(eval_result.metrics, indent=2)
        return f"""## Simulation Feedback for Idea #{iteration}

Idea: {idea.get('title', '?')} [{idea.get('tier', '?')}]
Code change: {idea.get('old_code', '')[:120]}... → {idea.get('new_code', '')[:120]}...

Results:
- Primary score: {eval_result.score:.4f} (target: {'higher' if self.adapter.get_metric_direction() == 'higher' else 'lower'} is better)
- Success: {eval_result.success}
- Metrics: {metrics_str}
- Info: {eval_result.info[:300]}

Based on these results:
1. Was this idea effective? (score improved/declined vs baseline)
2. What does the result tell you about the failure cause?
3. What should we try next? (specific suggestion with exact code change)

Respond with JSON:
{{"effective": true/false, "analysis": "...", "next_suggestion": "..."}}"""

    # ══════════════════════════════════════════════════════════════
    # Main Optimization Loop
    # ══════════════════════════════════════════════════════════════

    def run(self, max_iterations: int = 10) -> Dict:
        """
        Run the full AutoSOTA-style optimization pipeline.

        Returns:
            Dict with baseline_score, best_score, improvement_pct, history, etc.
        """
        param_space = self.adapter.get_param_space()
        metric_name = self.adapter.get_primary_metric_name()
        direction = self.adapter.get_metric_direction()

        print("=" * 60)
        print(f"  AutoSim LLM Optimizer (ALGO → CODE → PARAM)")
        print(f"  Adapter: {self.adapter.__class__.__name__}")
        print(f"  Metric:  {metric_name} ({direction} is better)")
        print(f"  Model:   {self.llm.model}")
        print("=" * 60)

        # ═══ Phase 1: Deep Code Analysis ═══
        print(f"\n{'=' * 60}")
        print("  PHASE 1: Deep Code Analysis")
        print(f"{'=' * 60}")

        sources = self.adapter.get_source_files()
        analysis = ""

        if sources and self.llm.available and not self._llm_unavailable:
            s1, u1 = self._build_analysis_prompt(sources)
            analysis = self._call_llm(s1, u1, max_tokens=16384)
            if analysis:
                (self.output_dir / "memory" / "code_analysis.md").write_text(analysis)
                print(f"  Analysis: {len(analysis)} chars, {len(sources)} files analyzed")
            else:
                print("  Code analysis failed — skipping LLM phases")
        else:
            print(f"  {'No source files available' if not sources else 'No LLM API configured'}")
            if not sources:
                print("  (Adapter.get_source_files() returned empty)")

        # ═══ Phase 2: Baseline ═══
        print(f"\n{'=' * 60}")
        print("  PHASE 2: Baseline")
        print(f"{'=' * 60}")

        defaults = {p.name: p.default for p in param_space.values()}
        baseline_result = self.adapter.evaluate(defaults)
        self.baseline_score = baseline_result.score
        self.best_score = baseline_result.score
        self.best_params = dict(defaults)

        self.history.append({
            "iteration": 0,
            "type": "baseline",
            "title": "Baseline evaluation",
            "params": dict(defaults),
            "score": baseline_result.score,
            "success": baseline_result.success,
            "metrics": baseline_result.metrics,
            "is_best": True,
        })
        print(f"  {metric_name}={baseline_result.score:.4f} success={baseline_result.success}")
        if baseline_result.info:
            print(f"  Info: {baseline_result.info[:200]}")

        # ═══ Phase 3: CODE/ALGO Ideas (via LLM) ═══
        code_algo_count = 0
        if analysis and self.llm.available and not self._llm_unavailable:
            print(f"\n{'=' * 60}")
            print("  PHASE 3: ALGO/CODE Optimization (LLM-driven)")
            print(f"{'=' * 60}")

            s2, u2 = self._build_ideas_prompt(analysis, self.baseline_score)
            raw_ideas = self._call_llm(s2, u2, max_tokens=16384)
            ideas = []

            if raw_ideas:
                (self.output_dir / "memory" / "idea_library_raw.txt").write_text(raw_ideas)
                parsed = self._parse_json_from_text(raw_ideas)
                ideas = parsed.get("ideas", [])
                print(f"  LLM generated {len(ideas)} ideas")

            # Sort by AutoSOTA priority: ALGO > CODE > PARAM
            tier_order = {"ALGO": 0, "CODE": 1, "PARAM": 2}
            ideas.sort(key=lambda i: tier_order.get(i.get("tier", "PARAM"), 3))

            skipped_code_algo = 0
            for i, idea in enumerate(ideas):
                tier = idea.get("tier", "CODE")
                title = idea.get("title", f"idea_{i}")
                fname = idea.get("file", "")
                old = idea.get("old_code", "")
                new = idea.get("new_code", "")

                if not old or not new:
                    continue
                if code_algo_count >= max_iterations:
                    break

                attempt_no = code_algo_count + skipped_code_algo + 1
                print(f"\n  [{attempt_no}] [{tier}] {title[:80]}")
                print(f"      file: {fname}")

                # Apply change
                applied = self.adapter.apply_change(fname, old, new)
                if not applied:
                    skipped_code_algo += 1
                    print(f"      ✗ old_code not found — skipping")
                    continue
                print(f"      ✓ code change applied")
                code_algo_count += 1

                # Evaluate; broken code ideas must not poison later trials.
                try:
                    result = self.adapter.evaluate(defaults)
                    score = result.score
                except Exception as exc:
                    self.adapter.revert_all()
                    print(f"      ✗ evaluation failed, reverted: {exc}")
                    self.history.append({
                        "iteration": code_algo_count,
                        "type": tier,
                        "title": title,
                        "file": fname,
                        "params": dict(defaults),
                        "score": self.best_score,
                        "success": False,
                        "metrics": {"error": str(exc)},
                        "is_best": False,
                        "old_code_preview": old[:80],
                        "new_code_preview": new[:80],
                    })
                    continue

                # Check if improvement
                is_better = (
                    (direction == "higher" and score > self.best_score) or
                    (direction == "lower" and score < self.best_score)
                )

                if is_better:
                    self.best_score = score
                    self.best_params = dict(defaults)
                    self.adapter.accept_all_changes()
                    print(f"      ★ BEST: {metric_name}={score:.4f} (kept)")
                else:
                    self.adapter.revert_all()
                    print(f"      {metric_name}={score:.4f} (reverted, best={self.best_score:.4f})")

                self.history.append({
                    "iteration": code_algo_count,
                    "type": tier,
                    "title": title,
                    "file": fname,
                    "params": dict(defaults),
                    "score": score,
                    "success": result.success,
                    "metrics": result.metrics,
                    "is_best": is_better,
                    "old_code_preview": old[:80],
                    "new_code_preview": new[:80],
                })

        # ═══ Phase 4: PARAM Fine-tuning ═══
        if param_space:
            print(f"\n{'=' * 60}")
            print(f"  PHASE 4: PARAM Fine-tuning (up to {max_iterations} iters)")
            print(f"{'=' * 60}")

            if analysis and code_algo_count == 0:
                print("  PARAM skipped because no ALGO/CODE change was evaluated")
                return self._finalize()

            # Use best state from previous phases
            current_params = dict(self.best_params)
            no_improve = 0
            param_count = code_algo_count

            for it in range(1, max_iterations + 1):
                if no_improve >= 4:
                    print(f"  ⚠ No improvement for 4 rounds — stopping PARAM search")
                    break

                # Suggest next params (LLM-guided or fallback)
                suggestion = self._suggest_params(history=self.history)
                if suggestion:
                    # Apply suggestion within param bounds
                    trial_params = dict(current_params)
                    for k, v in suggestion.items():
                        if k in trial_params and k in {p.name for p in param_space.values()}:
                            pdef = param_space[k]
                            lo, hi = pdef.range
                            trial_params[k] = max(lo, min(hi, float(v)))

                    param_count += 1
                    print(f"\n  [{param_count}] PARAM trial: {trial_params}")

                    result = self.adapter.evaluate(trial_params)
                    score = result.score

                    is_better = (
                        (direction == "higher" and score > self.best_score) or
                        (direction == "lower" and score < self.best_score)
                    )

                    if is_better:
                        self.best_score = score
                        self.best_params = dict(trial_params)
                        current_params = dict(trial_params)
                        no_improve = 0
                        print(f"      ★ BEST: {metric_name}={score:.4f}")
                    else:
                        no_improve += 1
                        print(f"      {metric_name}={score:.4f} (best={self.best_score:.4f})")

                    self.history.append({
                        "iteration": param_count,
                        "type": "PARAM",
                        "title": f"PARAM iteration {it}",
                        "params": dict(trial_params),
                        "score": score,
                        "success": result.success,
                        "metrics": result.metrics,
                        "is_best": is_better,
                    })

        # ═══ Phase 5: Report ═══
        result = self._finalize()
        return result

    def _suggest_params(self, history: List[Dict]) -> Optional[Dict]:
        """
        Suggest next parameter set using LLM or fallback local search.

        Uses LLM if available, otherwise perturbs best params with random noise.
        """
        param_space = self.adapter.get_param_space()
        if not param_space:
            return None

        direction = self.adapter.get_metric_direction()

        # Try LLM-guided suggestion
        if self.llm.available and not self._llm_unavailable and len(history) > 1:
            hist_text = "\n".join(
                f"  [{h['iteration']}] {h.get('type','?')} params={h.get('params',{})} "
                f"→ score={h['score']:.4f}{' ★' if h.get('is_best') else ''}"
                for h in history[-6:]
            )
            param_desc = "\n".join(
                f"- {p.name}: default={p.default}, range={p.range}, type={p.dtype}"
                for p in param_space.values()
            )

            system = "Expert optimization engineer. Suggest next hyperparameters."
            user = f"""History ({direction} is better):
{hist_text}

Best score: {self.best_score:.4f}

Parameter space:
{param_desc}

Analyze trends and suggest next params.
If improving → continue direction.
If plateauing → try a structural change.
If failing → revert direction.

Return JSON: {{"params": {{"name": value, ...}}, "analysis": "why"}}"""

            response = self._call_llm(system, user, max_tokens=2048)
            if response:
                parsed = self._parse_json_from_text(response)
                params = parsed.get("params")
                if params:
                    return params

        # Fallback: perturb best params
        import random as _random
        seed_str = json.dumps([h.get("score", 0) for h in history[-3:]], sort_keys=True)
        rng = _random.Random(seed_str)

        params = {}
        for pdef in param_space.values():
            lo, hi = pdef.range
            base = self.best_params.get(pdef.name, pdef.default)
            scale = 0.12 if len(history) < 6 else 0.06
            perturb = rng.uniform(-scale, scale) * (hi - lo)
            val = base + perturb
            if pdef.dtype == "int":
                params[pdef.name] = max(int(lo), min(int(hi), int(round(val))))
            else:
                params[pdef.name] = round(max(lo, min(hi, val)), 6)

        return params

    # ══════════════════════════════════════════════════════════════
    # Finalization
    # ══════════════════════════════════════════════════════════════

    def _finalize(self) -> Dict:
        """Compute final results and write reports."""
        if self.baseline_score is None or self.baseline_score == 0:
            improv_pct = 0.0
        else:
            direction = self.adapter.get_metric_direction()
            if direction == "higher":
                improv_pct = ((self.best_score - self.baseline_score)
                              / abs(self.baseline_score)) * 100
            else:
                improv_pct = ((self.baseline_score - self.best_score)
                              / abs(self.baseline_score)) * 100

        metric_name = self.adapter.get_primary_metric_name()

        # Print summary
        print(f"\n{'=' * 60}")
        print("  OPTIMIZATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Baseline:  {self.baseline_score:.4f}")
        print(f"  Best:      {self.best_score:.4f} ({improv_pct:+.1f}%)")
        print(f"  Iterations: {len(self.history)}")

        # ALGO/CODE stats
        algo_count = sum(1 for h in self.history if h.get("type") in ("ALGO", "CODE"))
        param_count = sum(1 for h in self.history if h.get("type") == "PARAM")
        print(f"  ALGO/CODE:  {algo_count}")
        print(f"  PARAM:      {param_count}")

        # Write report
        report = f"""# AutoSim LLM Optimization Report

**Adapter**: {self.adapter.__class__.__name__}
**Metric**:  {metric_name}
**Direction**: {self.adapter.get_metric_direction()} is better
**Date**:     {datetime.now().isoformat()}

## Results

| | Score |
|---|---|
| Baseline | {self.baseline_score:.4f} |
| Best | {self.best_score:.4f} |
| Improvement | {improv_pct:+.1f}% |

Best params: {json.dumps(self.best_params, indent=2)}

## History

| Iter | Type | Title | Score | Best |
|------|------|-------|-------|------|
"""
        for h in self.history:
            report += (
                f"| {h['iteration']} | {h.get('type', '?')} | "
                f"{h.get('title', '')[:60]} | {h['score']:.4f} | "
                f"{'★' if h.get('is_best') else ''} |\n"
            )

        report_path = self.output_dir / "results" / "final_report.md"
        report_path.write_text(report)
        print(f"  Report: {report_path}")

        # Write scores.jsonl
        scores_path = self.output_dir / "results" / "scores.jsonl"
        with open(scores_path, "w") as f:
            for h in self.history:
                f.write(json.dumps({
                    "iteration": h["iteration"],
                    "type": h.get("type"),
                    "title": h.get("title"),
                    "score": h["score"],
                    "success": h.get("success", True),
                    "params": h.get("params", {}),
                    "is_best": h.get("is_best", False),
                }) + "\n")

        meta = {
            "adapter": self.adapter.__class__.__name__,
            "metric": metric_name,
            "direction": self.adapter.get_metric_direction(),
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "improvement_pct": improv_pct,
            "num_iterations": len(self.history),
            "algo_code_count": sum(1 for h in self.history if h.get("type") in ("ALGO", "CODE")),
            "history": self.history,
        }

        meta_path = self.output_dir / "optimization_result.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Result: {meta_path}")

        return meta


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoSim LLM Optimizer")
    parser.add_argument("--task-type", default="reach",
                        choices=["reach", "grasp", "pick_place", "insertion"],
                        help="Isaac Sim task type")
    parser.add_argument("--robot", default="franka",
                        help="Robot type (franka, ur5)")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use mock mode (no Isaac Sim)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-mock", action="store_false", dest="mock")
    parser.add_argument("--output", default="output/llm_isaac_opt")
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    from autosim.adapters.isaac_adapter import IsaacSimAdapter

    adapter = IsaacSimAdapter(
        task_type=args.task_type,
        robot_type=args.robot,
        host=args.host,
        port=args.port,
        mock=args.mock,
        num_seeds=args.seeds,
    )

    opt = LLMOptimizer(
        adapter=adapter,
        output_dir=args.output,
        eval_seeds=args.seeds,
    )
    result = opt.run(max_iterations=args.max_iter)
    print(f"\nDone. Best score: {result['best_score']:.4f} "
          f"({result['improvement_pct']:+.1f}%)")
