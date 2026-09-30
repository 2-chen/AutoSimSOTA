"""Quantify reset-time variation across policies evaluated on one seed bank.

This audit deliberately reports measurements instead of inventing a post-hoc
equivalence tolerance.  It is additive validation code and does not change the
frozen policy, simulator, collection, training, or evaluation path.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from autosim.research.common import digest, immutable_json, read_json


def evaluation_artifact_directory(directory: Path, metrics: dict) -> Path:
    """Resolve sidecars for a first attempt or an allowed startup-only retry.

    ``Runtime.evaluate`` publishes the successful metrics at the stable outer
    directory, while a permitted second/third startup attempt keeps its raw
    protocol and telemetry in a child directory.  Restricting the child name
    prevents a metrics file from redirecting this audit to arbitrary evidence.
    """
    root = Path(directory).absolute().resolve()
    declared = metrics.get("artifact_directory")
    artifact = Path(declared).absolute().resolve() if declared else root
    if artifact == root:
        return artifact
    if artifact.parent == root and artifact.name in {"startup_attempt_2", "startup_attempt_3"}:
        return artifact
    raise ValueError("evaluation artifact_directory is outside the allowed startup-attempt lineage")


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None,
                "exact_zero_count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
        "exact_zero_count": int(np.count_nonzero(array == 0.0)),
    }


def _max_pairwise_translation(poses: list[np.ndarray]) -> float:
    maximum = 0.0
    for first, second in combinations(poses, 2):
        first = first.reshape(-1, 4, 4)
        second = second.reshape(-1, 4, 4)
        if first.shape != second.shape:
            raise ValueError("different reset-time entity pose shapes")
        difference = np.linalg.norm(first[:, :3, 3] - second[:, :3, 3], axis=-1)
        maximum = max(maximum, float(difference.max(initial=0.0)))
    return maximum


def _maximum_component_range(values: list[np.ndarray], message: str) -> float:
    shapes = {value.shape for value in values}
    if len(shapes) != 1:
        raise ValueError(message)
    stacked = np.stack(values, axis=0)
    return float((stacked.max(axis=0) - stacked.min(axis=0)).max(initial=0.0))


def _load(label: str, directory: Path) -> dict:
    directory = Path(directory).absolute().resolve()
    published_metrics = directory / "evaluation_metrics.json"
    if not published_metrics.is_file():
        raise FileNotFoundError(f"{label} is missing required evidence: {[str(published_metrics)]}")
    metrics = read_json(published_metrics)
    artifact = evaluation_artifact_directory(directory, metrics)
    required = {
        "published_metrics": published_metrics,
        "protocol": artifact / "protocol.json",
        "initializations": artifact / "initializations.jsonl",
        "telemetry": artifact / "telemetry.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is missing required evidence: {missing}")
    initialization_rows = [json.loads(line) for line in required["initializations"].read_text().splitlines()]
    telemetry_rows = [json.loads(line) for line in required["telemetry"].read_text().splitlines()]
    seeds = [int(row["episode_seed"]) for row in metrics["episodes"]]
    initialization_seeds = [int(row["seed"]) for row in initialization_rows]
    if len(seeds) != len(set(seeds)) or initialization_seeds != seeds:
        raise ValueError(f"{label} has duplicate, missing, or reordered initialization seeds")
    initial = [row for row in telemetry_rows if int(row["step"]) == 0]
    if [int(row["seed"]) for row in initial] != seeds:
        raise ValueError(f"{label} has duplicate, missing, or reordered step-0 telemetry")
    return {
        "label": label,
        "directory": str(directory.absolute()),
        "artifact_directory": str(artifact),
        "metrics": metrics,
        "initialization_rows": initialization_rows,
        "initial": {int(row["seed"]): row for row in initial},
        "evidence_sha256": {name: digest(path) for name, path in required.items()},
    }


def audit_bank(evaluations: dict[str, Path]) -> dict:
    """Audit two or more evaluation directories sharing an ordered seed bank."""
    if len(evaluations) < 2:
        raise ValueError("at least two evaluations are required")
    loaded = [_load(label, Path(directory)) for label, directory in evaluations.items()]
    reference = loaded[0]["metrics"]
    seeds = [int(row["episode_seed"]) for row in reference["episodes"]]
    contract = {
        "task": reference["config"]["task"],
        "setting": reference["config"]["setting"],
        "purpose": reference.get("purpose"),
        "timeout_action_steps": int(reference["config"]["timeout_action_steps"]),
        "ordered_episode_seeds": seeds,
    }
    for item in loaded[1:]:
        metrics = item["metrics"]
        candidate = {
            "task": metrics["config"]["task"],
            "setting": metrics["config"]["setting"],
            "purpose": metrics.get("purpose"),
            "timeout_action_steps": int(metrics["config"]["timeout_action_steps"]),
            "ordered_episode_seeds": [int(row["episode_seed"]) for row in metrics["episodes"]],
        }
        if candidate != contract:
            raise ValueError(f"{item['label']} does not share the evaluation protocol and ordered seed bank")

    per_seed, robot_differences, observation_matches = [], [], []
    entity_translation: dict[str, list[float]] = {}
    entity_rotation: dict[str, list[float]] = {}
    entity_qpos: dict[str, list[float]] = {}
    discordant_robot, concordant_robot = [], []
    for index, seed in enumerate(seeds):
        rows = [item["initial"][seed] for item in loaded]
        robot = [np.asarray(row["robot_qpos"], dtype=np.float64) for row in rows]
        robot_difference = _maximum_component_range(robot, "different reset-time robot qpos shapes")
        hashes = [item["initialization_rows"][index]["allowed_observation_sha256"] for item in loaded]
        observation_match = len(set(hashes)) == 1
        outcomes = [bool(item["metrics"]["episodes"][index]["success"]) for item in loaded]
        discordant = len(set(outcomes)) > 1
        (discordant_robot if discordant else concordant_robot).append(robot_difference)
        entity_names = [set(row["entities"]) for row in rows]
        if any(names != entity_names[0] for names in entity_names[1:]):
            raise ValueError(f"different reset-time entity sets for seed {seed}")
        entity_details = {}
        for name in sorted(entity_names[0]):
            entries = [row["entities"][name] for row in rows]
            poses = [np.asarray(entry["pose"], dtype=np.float64) for entry in entries]
            translation = _max_pairwise_translation(poses)
            rotation = _maximum_component_range(
                [pose.reshape(-1, 4, 4)[:, :3, :3] for pose in poses],
                "different reset-time entity rotation shapes",
            )
            detail = {
                "translation_max_pairwise_difference_m": translation,
                "rotation_matrix_max_component_range": rotation,
            }
            entity_translation.setdefault(name, []).append(translation)
            entity_rotation.setdefault(name, []).append(rotation)
            if all("qpos" in entry for entry in entries):
                qpos = _maximum_component_range(
                    [np.asarray(entry["qpos"], dtype=np.float64) for entry in entries],
                    "different reset-time articulation qpos shapes",
                )
                detail["qpos_max_component_range"] = qpos
                entity_qpos.setdefault(name, []).append(qpos)
            entity_details[name] = detail
        per_seed.append({
            "seed": seed,
            "robot_qpos_max_component_range_rad": robot_difference,
            "allowed_policy_observation_hashes_all_equal": observation_match,
            "outcomes": dict(zip((item["label"] for item in loaded), outcomes)),
            "outcomes_discordant": discordant,
            "entities": entity_details,
        })
        robot_differences.append(robot_difference)
        observation_matches.append(observation_match)

    return {
        "schema_version": 1,
        "status": "measured_reset_variation_no_equivalence_claim",
        "evaluation_contract": contract,
        "evaluations": [{
            "label": item["label"],
            "directory": item["directory"],
            "artifact_directory": item["artifact_directory"],
            "evidence_sha256": item["evidence_sha256"],
        } for item in loaded],
        "summary": {
            "evaluation_count": len(loaded),
            "episode_count": len(seeds),
            "ordered_seed_bank_identical": True,
            "robot_qpos_max_component_range_rad": _quantiles(robot_differences),
            "allowed_policy_observation_exact_match_count": int(sum(observation_matches)),
            "outcome_discordant_episode_count": int(sum(row["outcomes_discordant"] for row in per_seed)),
            "robot_qpos_range_by_outcome_concordance": {
                "discordant": _quantiles(discordant_robot),
                "concordant": _quantiles(concordant_robot),
            },
            "entities": {name: {
                "translation_max_pairwise_difference_m": _quantiles(values),
                "rotation_matrix_max_component_range": _quantiles(entity_rotation[name]),
                **({"qpos_max_component_range": _quantiles(entity_qpos[name])}
                   if name in entity_qpos else {}),
            } for name, values in sorted(entity_translation.items())},
        },
        "episodes": per_seed,
        "interpretation": {
            "what_is_established": "same ordered seeds and evaluation protocol; measured reset-time variation",
            "what_is_not_established": "bitwise-identical observations or complete hidden physics/RNG state",
            "pass_tolerance": "not_predeclared_no_pass_claim",
            "statistical_boundary": "same-seed pairing blocks recorded scene randomization but does not remove reset-time simulator variation",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", nargs=2, metavar=("LABEL", "DIRECTORY"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = [label for label, _ in args.evaluation]
    if len(labels) != len(set(labels)):
        raise ValueError("evaluation labels must be unique")
    result = audit_bank({label: Path(directory) for label, directory in args.evaluation})
    result["audit_source_sha256"] = digest(Path(__file__))
    immutable_json(args.output, result)


if __name__ == "__main__":
    main()
