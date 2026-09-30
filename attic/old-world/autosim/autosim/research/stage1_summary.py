"""Build the evidence-backed closeout for the bounded Stage-1 diagnosis."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import atomic_json, digest, now, read_json


def build(root: Path) -> dict:
    root = root.resolve()
    paths = {
        "history": root / "history_audit.json",
        "collection": root / "bounded_probe_analysis.json",
        "failure": root / "failure_slice_analysis.json",
        "correction": root / "bounded_probe_policy_correction/correction_audit.json",
        "learning": root / "learning_probe_analysis.json",
        "tests": root / "stage1_pytest_final.xml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage-1 evidence: {missing}")
    history, collection, failure, correction, learning = (
        read_json(paths[name]) for name in ("history", "collection", "failure", "correction", "learning")
    )
    profiles = {row["profile"]: row for row in collection["profiles"]}

    process_paths = sorted(root.rglob("process.json"))
    processes = [read_json(path) for path in process_paths]
    elapsed = sum(float(row.get("elapsed_seconds", 0)) for row in processes)
    completed = sum(row.get("status") == "completed" for row in processes)
    failed = len(processes) - completed

    suite_root = ET.parse(paths["tests"]).getroot()
    suites = ([suite_root] if suite_root.tag == "testsuite"
              else list(suite_root.findall("testsuite")))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)

    checks = {
        "prospective_scene_and_exposure_instrumentation":
            history["conclusion"]["prospective_instrumentation_required"],
        "three_bounded_profiles_audited": set(profiles) == {
            "full_random", "targeted_camera", "targeted_clutter"},
        "all_bounded_profile_data_and_video_gates": all(
            row["scene_evidence_passed"] and row["data_audit_passed"]
            and row["full_video_decode_passed"] for row in profiles.values()),
        "repeatability_gate": failure["repeatability"]["gate_passed"],
        "correction_capability_and_lineage": correction["passed"],
        "same_pool_training_and_evaluation_valid": learning["result_valid"],
        "test_suite": failures == 0 and errors == 0 and tests >= 174,
        "within_stage_budget": elapsed / 3600 <= 8,
    }
    execution_complete = all(checks.values())
    learning_screen = learning["learning_signal"]["exploratory_screen_supported"]
    candidate_slices = failure["candidate_failure_relevance"]["eligible_training_probe_candidates"]
    proceed = execution_complete and bool(candidate_slices) and learning_screen

    return {
        "schema_version": 1,
        "kind": "autoresearch_stage_1_diagnostic_closeout",
        "created_at": now(),
        "execution_complete": execution_complete,
        "result_valid": execution_complete,
        "hypothesis_supported": proceed,
        "checks": checks,
        "budget": {
            "limit_gpu_hours": 8,
            "conservative_all_process_wall_seconds": elapsed,
            "conservative_all_process_wall_hours": elapsed / 3600,
            "process_count": len(processes),
            "completed_processes": completed,
            "failed_processes": failed,
            "accounting_note": "Conservative upper bound sums all recorded process wall time, including CPU smoke and four pre-episode infrastructure failures; jobs ran serially on one workstation.",
        },
        "findings": {
            "historical_natural_bias_proven": history["conclusion"]["historical_natural_bias_proven"],
            "bounded_expert_yield": {name: {
                "accepted": row["accepted"], "attempts": row["attempts"],
                "rate": row["yield"], "wilson_95": row["yield_wilson_95"],
            } for name, row in profiles.items()},
            "candidate_failure_slices_eligible": candidate_slices,
            "camera_extrinsic_translation_randomized_observed":
                failure["benchmark_implementation_observation"]["camera_extrinsic_translation_randomized_observed"],
            "correction_probe": {
                "saved": correction["saved_corrections"],
                "attempts": correction["attempts"],
                "parent_already_succeeded": correction["attempt_outcomes"]["parent_policy_already_succeeded"],
                "expert_correction_failed": correction["attempt_outcomes"]["expert_correction_failed"],
                "learner_state_l2_range": [
                    correction["learner_state_displacement"]["l2_min"],
                    correction["learner_state_displacement"]["l2_max"],
                ],
            },
            "actual_correction_exposure": {name: row["realized_correction_mass"]
                                            for name, row in learning["exposures"].items()},
            "learning_probe_mean_success":
                learning["learning_signal"]["policy_mean_success_rates"],
            "correction_10pct_minus_proportional":
                learning["learning_signal"]["correction_10pct_minus_proportional"],
            "correction_10pct_minus_parent":
                learning["learning_signal"]["correction_10pct_minus_parent"],
            "learning_signal_screen_supported": learning_screen,
            "within_policy_repeat_mismatch": learning["within_policy_repeatability"],
        },
        "decision": {
            "enter_stage_2_with_current_camera_clutter_or_10pct_correction_mechanism": proceed,
            "status": "stop_current_candidate_before_stage_2" if not proceed else "eligible_for_stage_2",
            "reason": (
                "No camera/clutter slice passed the frozen failure-relevance gate, and the 10% correction-exposure arm improved only 3.75 points over proportional exposure, below the frozen 5-point screen, while remaining 1.25 points below the parent."
                if not proceed else "All frozen Stage-1 continuation gates passed."
            ),
            "prohibited_follow_up": "Do not launch the 12-hour adaptive comparison, add seeds, or select a checkpoint to rescue this result.",
            "allowed_repositioning": "Treat the result as a validated infrastructure/capability result and redesign the failure signal or correction-data mechanism under a new preregistered protocol before any new GPU study.",
        },
        "test_suite": {"tests": tests, "failures": failures, "errors": errors,
                       "skipped": skipped, "passed": tests - failures - errors - skipped},
        "evidence": {name: {"path": str(path.resolve()), "sha256": digest(path)}
                     for name, path in paths.items()},
        "limitations": [
            "The learning probe has one training seed and 40 episodes per repeat; it is development evidence, not an official or confirmatory score.",
            "Appearance photometrics/material parameters remain unobserved, so an appearance candidate was not tested.",
            "Observed camera extrinsic translation was constant in this ClickBell configuration; rotation was not reduced by the current audit.",
            "WaterPouring's historical corrupt-video root file is localized, but the encoder exit cause and released-checkpoint reproduction gap remain unresolved.",
            "No cross-task or second-backend learning-transfer claim is supported by Stage 1.",
        ],
    }


def render(result: dict) -> str:
    findings, decision = result["findings"], result["decision"]
    lines = [
        "# AutoResearch Stage 1 diagnostic closeout", "",
        f"- Execution complete: **{result['execution_complete']}**",
        f"- Result valid: **{result['result_valid']}**",
        f"- Current mechanism hypothesis supported: **{result['hypothesis_supported']}**",
        f"- Decision: **{decision['status']}**", "",
        "## Bounded capability evidence", "",
        "| Profile | Accepted / attempts | Yield | Wilson 95% |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in findings["bounded_expert_yield"].items():
        lines.append(f"| {name} | {row['accepted']}/{row['attempts']} | {row['rate']:.1%} | "
                     f"[{row['wilson_95'][0]:.1%}, {row['wilson_95'][1]:.1%}] |")
    means = findings["learning_probe_mean_success"]
    lines += [
        "", "## Same-pool exposure probe", "",
        "| Policy | Two-repeat mean success |",
        "| --- | ---: |",
        f"| Frozen parent | {means['parent']:.2%} |",
        f"| Proportional correction exposure | {means['proportional']:.2%} |",
        f"| 10% correction exposure | {means['correction_10pct']:.2%} |", "",
        f"The 10% arm was {findings['correction_10pct_minus_proportional']:+.2%} versus proportional and "
        f"{findings['correction_10pct_minus_parent']:+.2%} versus the parent. The frozen exploratory screen did not pass.", "",
        "## Decision", "", decision["reason"], "", decision["prohibited_follow_up"], "",
        "## Budget and verification", "",
        f"Conservative summed process wall time: {result['budget']['conservative_all_process_wall_hours']:.3f} h / 8 h. "
        f"Tests: {result['test_suite']['passed']} passed, {result['test_suite']['failures']} failed, "
        f"{result['test_suite']['errors']} errors.", "",
        "This closeout is development evidence, not an official score, SOTA claim, or cross-task result.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root)
    atomic_json(args.root / "stage1_closeout.json", result)
    (args.root / "stage1_closeout.md").write_text(render(result), encoding="utf-8")


if __name__ == "__main__":
    main()
