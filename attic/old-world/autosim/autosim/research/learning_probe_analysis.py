"""Audit and summarize the frozen Stage-1 same-pool learning probe.

This module deliberately separates execution validity from the exploratory
learning-signal screen.  Missing evaluations yield an incomplete report rather
than silently changing the preregistered run set.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from autosim.experiment_validation.checkpoint_audit import audit_checkpoint

from .common import atomic_json, digest, now, read_json
from .ledger import compare


ARMS = ("proportional", "correction_10pct")
RUN_ORDER = (
    "parent_repeat_1",
    "correction_10pct_repeat_1",
    "proportional_repeat_1",
    "proportional_repeat_2",
    "parent_repeat_2",
    "correction_10pct_repeat_2",
)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _canonical_train_config(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("output_dir", None)
    return value


def _exposure(path: Path) -> dict:
    data = read_json(path)
    parts = {row["source_kind"]: row for row in data["parts"]}
    correction = parts["targeted_training_only"]
    return {
        "status": data.get("status"),
        "batches_yielded": data.get("batches_yielded"),
        "samples_yielded": data.get("samples_yielded"),
        "correction_samples": correction["yielded_samples"],
        "expected_correction_mass": correction["expected_sampling_mass"],
        "realized_correction_mass": correction["realized_sampling_mass"],
        "absolute_sampling_error": abs(
            correction["realized_sampling_mass"] - correction["expected_sampling_mass"]),
        "path": str(path.resolve()),
        "sha256": digest(path),
    }


def analyze(root: Path, contract_path: Path) -> dict:
    root, contract_path = root.resolve(), contract_path.resolve()
    protocol_path = root / "learning_probe_protocol.json"
    protocol, contract = read_json(protocol_path), read_json(contract_path)
    parent = Path(protocol["parent_checkpoint"])
    checkpoints = {
        arm: root / f"learning_probe_{arm}/train/checkpoints/005000/pretrained_model"
        for arm in ARMS
    }
    checkpoint_audits = {
        arm: audit_checkpoint(path, protocol["training_contract"]["updates_each"],
                              protocol["training_contract"]["training_seed"])
        for arm, path in checkpoints.items()
    }
    recipes = {arm: read_json(root / f"learning_probe_{arm}/recipe_5000.json") for arm in ARMS}
    exposures = {
        arm: _exposure(root / f"learning_probe_{arm}/training_exposure_audit.json")
        for arm in ARMS
    }
    configs = {arm: read_json(path / "train_config.json") for arm, path in checkpoints.items()}
    model_hashes = {"parent": digest(parent / "model.safetensors"), **{
        arm: digest(path / "model.safetensors") for arm, path in checkpoints.items()}}

    expected_runs = tuple(protocol["evaluation"]["frozen_run_order"])
    execution_manifest_path = root / "learning_probe_evaluation_execution.json"
    execution_manifest = read_json(execution_manifest_path)
    run_directories = execution_manifest["run_directories"]
    if set(run_directories) != set(expected_runs):
        raise ValueError("evaluation execution manifest does not exactly cover frozen runs")
    completed, missing = {}, []
    for name in expected_runs:
        path = root / run_directories[name] / "evaluation_metrics.json"
        if path.is_file():
            completed[name] = read_json(path)
        else:
            missing.append(name)

    checks = [
        {"name": "frozen_run_order", "passed": expected_runs == RUN_ORDER,
         "actual": list(expected_runs), "expected": list(RUN_ORDER)},
        {"name": "all_checkpoint_audits", "passed": all(
            row["passed"] for row in checkpoint_audits.values())},
        {"name": "same_parent", "passed": all(
            row["pretrained_model_sha256"] == protocol["frozen_inputs_sha256"]["parent_model"]
            for row in recipes.values())},
        {"name": "parent_hash_current", "passed": model_hashes["parent"] ==
         protocol["frozen_inputs_sha256"]["parent_model"]},
        {"name": "same_model_config", "passed": digest(checkpoints[ARMS[0]] / "config.json") ==
         digest(checkpoints[ARMS[1]] / "config.json")},
        {"name": "same_train_config_except_output", "passed":
         _canonical_train_config(configs[ARMS[0]]) == _canonical_train_config(configs[ARMS[1]])},
        {"name": "child_weights_are_distinct", "passed": len(set(model_hashes.values())) == 3,
         "hashes": model_hashes},
        {"name": "exact_updates_and_batches", "passed": all(
            row["batches_yielded"] == protocol["training_contract"]["updates_each"]
            for row in exposures.values())},
        {"name": "exposure_matches_frozen_mass", "passed": all(
            row["absolute_sampling_error"] <= contract["max_absolute_exposure_error"]
            for row in exposures.values()),
         "max_absolute_error": contract["max_absolute_exposure_error"]},
    ]
    training_valid = all(row["passed"] for row in checks)

    evaluation = {name: {
        "success_count": data["summary"]["success_count"],
        "episodes": data["summary"]["episode_count"],
        "success_rate": data["summary"]["success_rate"],
        "wilson_95": _wilson(data["summary"]["success_count"], data["summary"]["episode_count"]),
    } for name, data in completed.items()}
    pairwise = {}
    within_repeat_mismatch = {}
    signal = None
    if not missing:
        for policy in ("parent", *ARMS):
            a, b = completed[f"{policy}_repeat_1"], completed[f"{policy}_repeat_2"]
            rows_a, rows_b = a["episodes"], b["episodes"]
            if [r["episode_seed"] for r in rows_a] != [r["episode_seed"] for r in rows_b]:
                raise ValueError(f"seed order changed between {policy} repeats")
            mismatches = sum(bool(x["success"]) != bool(y["success"])
                             for x, y in zip(rows_a, rows_b))
            within_repeat_mismatch[policy] = {
                "mismatches": mismatches, "episodes": len(rows_a),
                "rate": mismatches / len(rows_a),
            }
        for candidate in ARMS:
            for cr in (1, 2):
                for pr in (1, 2):
                    key = f"{candidate}_r{cr}_vs_parent_r{pr}"
                    pairwise[key] = compare(completed[f"{candidate}_repeat_{cr}"],
                                            completed[f"parent_repeat_{pr}"])
        for cr in (1, 2):
            for br in (1, 2):
                key = f"correction_10pct_r{cr}_vs_proportional_r{br}"
                pairwise[key] = compare(completed[f"correction_10pct_repeat_{cr}"],
                                        completed[f"proportional_repeat_{br}"])

        means = {policy: sum(evaluation[f"{policy}_repeat_{r}"]["success_rate"]
                             for r in (1, 2)) / 2
                 for policy in ("parent", *ARMS)}
        indexed_deltas = [
            evaluation[f"correction_10pct_repeat_{r}"]["success_rate"] -
            evaluation[f"proportional_repeat_{r}"]["success_rate"] for r in (1, 2)]
        weighted_vs_proportional = means["correction_10pct"] - means["proportional"]
        weighted_vs_parent = means["correction_10pct"] - means["parent"]
        signal_supported = (
            weighted_vs_proportional >= contract["practical_delta"]
            and all(delta >= 0 for delta in indexed_deltas)
            and weighted_vs_parent >= -contract["noncatastrophic_margin"]
        )
        signal = {
            "policy_mean_success_rates": means,
            "correction_10pct_minus_proportional": weighted_vs_proportional,
            "correction_10pct_minus_parent": weighted_vs_parent,
            "same_index_correction_minus_proportional": indexed_deltas,
            "exploratory_screen_supported": signal_supported,
            "screen_rule": contract["learning_signal_rule"],
            "not_a_significance_or_final_benchmark_claim": True,
        }

    result_valid = training_valid and not missing
    result = {
        "schema_version": 1,
        "kind": "stage_1_same_pool_exposure_learning_probe_analysis",
        "updated_at": now(),
        "status": "complete" if result_valid else "incomplete",
        "execution_complete": not missing,
        "result_valid": result_valid,
        "hypothesis_supported": signal["exploratory_screen_supported"] if signal else None,
        "training_valid": training_valid,
        "checks": checks,
        "checkpoint_audits": checkpoint_audits,
        "model_hashes": model_hashes,
        "exposures": exposures,
        "evaluation": evaluation,
        "within_policy_repeatability": within_repeat_mismatch,
        "pairwise": pairwise,
        "learning_signal": signal,
        "missing_evaluations": missing,
        "frozen_protocol": str(protocol_path),
        "frozen_protocol_sha256": digest(protocol_path),
        "analysis_contract": str(contract_path),
        "analysis_contract_sha256": digest(contract_path),
        "evaluation_execution_manifest": str(execution_manifest_path),
        "evaluation_execution_manifest_sha256": digest(execution_manifest_path),
        "limitations": [
            "Forty episodes per repeat are an exploratory development probe, not confirmatory evidence.",
            "A single training seed cannot establish stability across independent research runs.",
            "The screen decides whether a larger controlled mechanism study is warranted; it does not select a checkpoint.",
        ],
    }
    return result


def render(result: dict) -> str:
    lines = ["# Stage-1 same-pool exposure learning probe", "",
             f"- Status: **{result['status']}**",
             f"- Training audit: **{'passed' if result['training_valid'] else 'failed'}**",
             f"- Missing frozen evaluations: {', '.join(result['missing_evaluations']) or 'none'}", "",
             "## Actual training exposure", "",
             "| Arm | Samples | Correction samples | Expected | Realized |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        row = result["exposures"][arm]
        lines.append(f"| {arm} | {row['samples_yielded']} | {row['correction_samples']} | "
                     f"{row['expected_correction_mass']:.3%} | {row['realized_correction_mass']:.3%} |")
    if result["evaluation"]:
        lines += ["", "## Development evaluations", "",
                  "| Frozen run | Success | Rate | Wilson 95% |",
                  "| --- | ---: | ---: | ---: |"]
        for name in RUN_ORDER:
            if name not in result["evaluation"]:
                continue
            row = result["evaluation"][name]
            lines.append(f"| {name} | {row['success_count']}/{row['episodes']} | "
                         f"{row['success_rate']:.1%} | "
                         f"[{row['wilson_95'][0]:.1%}, {row['wilson_95'][1]:.1%}] |")
    if result["learning_signal"]:
        signal = result["learning_signal"]
        lines += ["", "## Frozen exploratory screen", "",
                  f"- 10% exposure minus proportional: {signal['correction_10pct_minus_proportional']:+.1%}",
                  f"- 10% exposure minus parent: {signal['correction_10pct_minus_parent']:+.1%}",
                  f"- Screen supported: **{signal['exploratory_screen_supported']}**", ""]
    lines += ["", "This is development evidence only. It is neither an official score nor a significance claim.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.contract)
    atomic_json(args.root / "learning_probe_analysis.json", result)
    (args.root / "learning_probe_analysis.md").write_text(render(result), encoding="utf-8")


if __name__ == "__main__":
    main()
