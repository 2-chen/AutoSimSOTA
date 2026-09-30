"""System-level optimizer for embodied simulation candidates.

This module keeps AutoSim focused on orchestration: define candidates, run the
same simulation evaluator for each one, rank by a task metric, and persist a
reproducible manifest. The adapter owns environment-specific details.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from autosim.adapters.base import Candidate, EvalResult, TaskAdapter
from autosim.optimize.experience_graph import OptimizationExperienceGraph


@dataclass
class CandidateRecord:
    """A persisted evaluation record for one candidate."""

    rank: Optional[int]
    round_index: int
    candidate: Candidate
    score: float
    success: bool
    improved_vs_baseline: bool
    delta_abs: float
    delta_pct: Optional[float]
    metrics: Dict[str, Any]
    info: str
    experience: Dict[str, Any]


class SimulationOptimizer:
    """Evaluate and rank simulator candidates through a TaskAdapter.

    The optimizer does not know how ACT, RoboTwin, Isaac, or a model checkpoint
    works. It only enforces the system loop:

    baseline -> candidate evaluations -> ranking -> manifest.
    """

    def __init__(
        self,
        adapter: TaskAdapter,
        output_dir: str | Path = "output/simulation_optimizer",
        skip_known_bad: bool = False,
    ):
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = adapter.get_primary_metric_name()
        self.direction = adapter.get_metric_direction()
        self.experience = OptimizationExperienceGraph(self.output_dir, self.direction)
        self.skip_known_bad = skip_known_bad
        self._bootstrap_experience_from_manifest()

    def _bootstrap_experience_from_manifest(self) -> None:
        """Import older flat manifests into the experience graph once."""
        if self.experience.nodes:
            return
        manifest_path = self.output_dir / "simulation_optimization_result.json"
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text())
            baseline_score = float(manifest["baseline"]["score"])
        except Exception:
            return
        imported = 0
        for record in manifest.get("records", []):
            try:
                candidate = Candidate(**record["candidate"])
                result = EvalResult(
                    score=float(record["score"]),
                    success=bool(record.get("success", True)),
                    metrics=record.get("metrics", {}),
                    info=record.get("info", ""),
                )
                self.experience.record(
                    candidate=candidate,
                    result=result,
                    baseline_score=baseline_score,
                    delta_abs=float(record.get("delta_abs", 0.0)),
                    delta_pct=record.get("delta_pct"),
                )
                imported += 1
            except Exception:
                continue
        if imported:
            print(f"  Imported {imported} prior attempts into experience graph")

    def _is_better(self, score: float, baseline: float) -> bool:
        if self.direction == "higher":
            return score > baseline
        return score < baseline

    def _delta(self, score: float, baseline: float) -> tuple[float, Optional[float]]:
        delta_abs = score - baseline if self.direction == "higher" else baseline - score
        if baseline == 0:
            return delta_abs, None
        return delta_abs, delta_abs / abs(baseline) * 100.0

    def _evaluate_candidate(self, candidate: Candidate) -> EvalResult:
        if hasattr(self.adapter, "evaluate_candidate"):
            return self.adapter.evaluate_candidate(candidate)
        if candidate.kind == "params":
            return self.adapter.evaluate(candidate.payload)
        raise TypeError(
            f"{self.adapter.__class__.__name__} does not support candidate kind "
            f"{candidate.kind!r}. Implement evaluate_candidate(candidate)."
        )

    def _rank(self, records: List[CandidateRecord]) -> None:
        reverse = self.direction == "higher"
        records.sort(key=lambda r: r.score, reverse=reverse)
        for rank, record in enumerate(records, start=1):
            record.rank = rank

    def _make_manifest(
        self,
        started_at: str,
        baseline: Candidate,
        baseline_result: EvalResult,
        records: List[CandidateRecord],
        status: str,
        target_score: Optional[float] = None,
        rounds_completed: int = 1,
        stop_reason: str = "",
    ) -> Dict[str, Any]:
        baseline_score = baseline_result.score
        best = records[0] if records else None
        if best and self._is_better(best.score, baseline_score):
            best_overall = {
                "source": "candidate",
                "candidate": asdict(best.candidate),
                "score": best.score,
                "success": best.success,
                "metrics": best.metrics,
                "info": best.info,
            }
        else:
            best_overall = {
                "source": "baseline",
                "candidate": asdict(baseline),
                "score": baseline_score,
                "success": baseline_result.success,
                "metrics": baseline_result.metrics,
                "info": baseline_result.info,
            }
        return {
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "adapter": self.adapter.__class__.__name__,
            "metric": self.metric_name,
            "direction": self.direction,
            "status": status,
            "target_score": target_score,
            "rounds_completed": rounds_completed,
            "stop_reason": stop_reason,
            "baseline": {
                "candidate": asdict(baseline),
                "score": baseline_score,
                "success": baseline_result.success,
                "metrics": baseline_result.metrics,
                "info": baseline_result.info,
            },
            "experience_graph": {
                "path": str(self.experience.path),
                "summary_path": str(self.experience.summary_path),
                "recommendations": self.experience.recommendations(),
            },
            "best": asdict(best) if best else None,
            "best_overall": best_overall,
            "records": [asdict(r) for r in records],
        }

    def _write_manifest(self, manifest: Dict[str, Any]) -> Path:
        result_path = self.output_dir / "simulation_optimization_result.json"
        result_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        return result_path

    def _target_reached(self, score: float, target_score: Optional[float]) -> bool:
        if target_score is None:
            return False
        if self.direction == "higher":
            return score >= target_score
        return score <= target_score

    def _eval_batch(
        self,
        candidates: List[Candidate],
        baseline_score: float,
        round_index: int,
        records: List[CandidateRecord],
    ) -> bool:
        improved_this_round = False
        previous_best = records[0].score if records else baseline_score
        for idx, candidate in enumerate(candidates, start=1):
            print(f"\n── Round {round_index} Candidate {idx}/{len(candidates)}: {candidate.name} ──")
            prior = self.experience.prior_for(candidate)
            print(f"  method_family={prior['family']} prior: {prior['advice']}")
            if self.skip_known_bad and self.experience.should_skip_known_bad(candidate):
                print("  skipped: prior attempts in this family are consistently negative")
                continue
            result = self._evaluate_candidate(candidate)
            delta_abs, delta_pct = self._delta(result.score, baseline_score)
            improved_vs_baseline = self._is_better(result.score, baseline_score)
            experience = self.experience.record(
                candidate=candidate,
                result=result,
                baseline_score=baseline_score,
                delta_abs=delta_abs,
                delta_pct=delta_pct,
            )
            print(
                f"  {self.metric_name}={result.score:.4f} "
                f"delta={delta_abs:+.4f} success={result.success}"
            )
            print(f"  lesson: {experience['lesson']}")
            if result.info:
                print(f"  info: {result.info}")
            records.append(
                CandidateRecord(
                    rank=None,
                    round_index=round_index,
                    candidate=candidate,
                    score=result.score,
                    success=result.success,
                    improved_vs_baseline=improved_vs_baseline,
                    delta_abs=delta_abs,
                    delta_pct=delta_pct,
                    metrics=result.metrics,
                    info=result.info,
                    experience=experience,
                )
            )
            self._rank(records)
            if self._is_better(records[0].score, previous_best):
                previous_best = records[0].score
                improved_this_round = True
        return improved_this_round

    def run(self, baseline: Candidate, candidates: Iterable[Candidate]) -> Dict[str, Any]:
        return self.run_until(
            baseline=baseline,
            candidate_provider=lambda _round, _seen: list(candidates) if _round == 1 else [],
            max_rounds=1,
        )

    def run_until(
        self,
        baseline: Candidate,
        candidate_provider: Callable[[int, Set[str]], Iterable[Candidate]],
        target_score: Optional[float] = None,
        max_rounds: int = 1,
        patience: int = 3,
    ) -> Dict[str, Any]:
        """Run a continuing simulation optimization loop.

        candidate_provider is called each round with (round_index, seen_names).
        This lets the system rescan a directory, poll a generator, or read a
        queue of newly produced candidates without changing the optimizer.
        """
        started_at = datetime.now().isoformat()

        print("=" * 60)
        print("  AutoSim Simulation Optimizer")
        print(f"  Metric: {self.metric_name} ({self.direction} is better)")
        if target_score is not None:
            print(f"  Target: {target_score:.4f}")
        print("=" * 60)

        print(f"\n── Baseline: {baseline.name} ──")
        baseline_result = self._evaluate_candidate(baseline)
        baseline_score = baseline_result.score
        print(f"  {self.metric_name}={baseline_score:.4f} success={baseline_result.success}")
        if baseline_result.info:
            print(f"  info: {baseline_result.info}")

        records: List[CandidateRecord] = []
        seen: Set[str] = {baseline.name}
        no_improve_rounds = 0
        status = "running"
        stop_reason = ""

        if self._target_reached(baseline_score, target_score):
            status = "target_reached"
            stop_reason = "baseline already satisfies target"

        rounds_completed = 0
        for round_index in range(1, max_rounds + 1):
            if status != "running":
                break
            raw_batch = list(candidate_provider(round_index, seen))
            batch = [c for c in raw_batch if c.name not in seen]
            for candidate in batch:
                seen.add(candidate.name)

            print(f"\n── Round {round_index}/{max_rounds}: {len(batch)} new candidates ──")
            rounds_completed = round_index
            if not batch:
                no_improve_rounds += 1
                stop_reason = "no new candidates"
            else:
                improved = self._eval_batch(batch, baseline_score, round_index, records)
                no_improve_rounds = 0 if improved else no_improve_rounds + 1
                best_score = records[0].score if records else baseline_score
                if self._target_reached(best_score, target_score):
                    status = "target_reached"
                    stop_reason = f"best score reached target {target_score}"

            if status == "running" and no_improve_rounds >= patience:
                status = "stalled"
                stop_reason = f"no improvement for {patience} rounds"

            self._rank(records)
            manifest = self._make_manifest(
                started_at=started_at,
                baseline=baseline,
                baseline_result=baseline_result,
                records=records,
                status=status,
                target_score=target_score,
                rounds_completed=rounds_completed,
                stop_reason=stop_reason,
            )
            result_path = self._write_manifest(manifest)

        if status == "running":
            status = "max_rounds_reached"
            stop_reason = f"reached max_rounds={max_rounds}"

        self._rank(records)
        manifest = self._make_manifest(
            started_at=started_at,
            baseline=baseline,
            baseline_result=baseline_result,
            records=records,
            status=status,
            target_score=target_score,
            rounds_completed=rounds_completed,
            stop_reason=stop_reason,
        )
        result_path = self._write_manifest(manifest)

        print(f"\n{'=' * 60}")
        print("  Simulation Optimization Complete")
        print(f"{'=' * 60}")
        print(f"  Baseline: {baseline_score:.4f}")
        best_overall = manifest.get("best_overall", {})
        best_candidate = best_overall.get("candidate", {})
        if best_candidate:
            print(
                f"  Best:     {best_candidate.get('name')} -> "
                f"{best_overall.get('score', 0):.4f} ({best_overall.get('source')})"
            )
        print(f"  Status:   {status} ({stop_reason})")
        print(f"  Result:   {result_path}")
        return manifest
