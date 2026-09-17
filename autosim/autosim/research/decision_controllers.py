"""Non-API research controllers used for matched-budget causal comparisons."""

from __future__ import annotations

import random
from typing import Any


def control_proposal(strategy: str, round_index: int, context: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Return a complete proposal from exactly the same registered action space.

    These controls never inspect final-confirmation data. Random choices are
    reproducible from the frozen run seed and round index.
    """
    if strategy not in {"fixed", "random", "heuristic"}:
        raise ValueError(f"not a non-API controller: {strategy}")
    profiles = list(context["allowed_collection_profiles"])
    modes = list(context["allowed_collection_modes"])
    attempts = int(context["attempts_per_round"])
    minimum_original = max(1, int(
        attempts * float(context["min_original_fraction"]) + 0.999999))
    targeted = attempts - minimum_original
    rng = random.Random(seed + round_index * 104_729)

    preferred_profiles = (
        "composite_hard", "targeted_recovery", "targeted_clutter",
        "targeted_camera", "targeted_appearance",
    )
    profile = next((candidate for candidate in preferred_profiles if candidate in profiles), profiles[0])
    mode = "expert"
    params: dict[str, Any] = {"action_loss_profile": "valid_mean"}
    targeted_mass = 0.35
    phase_weights = {"early": 1.0, "middle": 1.0, "late": 2.0}
    hypothesis = "a fixed full-randomization data and valid-label recipe is a strong execution baseline"

    if strategy == "random":
        profile = rng.choice(profiles)
        mode = rng.choice(modes)
        params = {
            "optimizer_lr": rng.choice([5e-6, 1e-5, 2e-5]),
            "n_action_steps": rng.choice([10, 25, 50]),
            "action_loss_profile": rng.choice(["legacy_mask_mean", "valid_mean"]),
            "image_augmentation_profile": rng.choice(
                ["none", "photometric_mild", "camera_geometry_mild"]),
        }
        targeted_mass = rng.choice([0.1, 0.2, 0.35, 0.5])
        phase_weights = {name: rng.choice([0.75, 1.0, 2.0, 3.0])
                         for name in ("early", "middle", "late")}
        hypothesis = "a uniformly sampled legal intervention provides the matched random-search control"
    elif strategy == "heuristic":
        evidence_text = str(context.get("development_evidence", {})).lower()
        for keyword, candidate in (
            ("little_object_motion", "targeted_recovery"),
            ("camera", "targeted_camera"),
            ("clutter", "targeted_clutter"),
            ("appearance", "targeted_appearance"),
        ):
            if keyword in evidence_text and candidate in profiles:
                profile = candidate
                break
        hypothesis = "a low-cost deterministic mapping from observed failure slices should improve allocation"

    return {
        "decision": "experiment",
        "proposal_id": f"round_{round_index}_{strategy}_{profile}",
        "parent_checkpoint_sha256": context["parent_checkpoint_sha256"],
        "parent_data_version": context["parent_data_version"],
        "development_evidence_id": context["development_evidence_id"],
        "hypothesis": hypothesis,
        "primary_intervention": "policy_correction" if mode == "policy_correction" else "targeted_data",
        "collection": {
            "enabled": True, "mode": mode, "profile": profile,
            "targeted_attempts": targeted,
            "original_attempts": minimum_original,
            "target_episodes": targeted,
        },
        "training": {
            "steps": int(context["training_steps"]), "params": params,
            "targeted_sampling_mass": targeted_mass,
            "phase_weights": phase_weights, "horizon_floor": 0.25,
        },
        # Resolution is a knob every arm chooses, so a control that left it out would be
        # comparing against a treatment held to a different evidence budget. The controls
        # take the run's own development resolution: they are matched on what they measure.
        "resolution": {
            "development_episodes": int(context["development_episodes"]),
            "note": "held at the run's development resolution so the arm is budget-matched",
        },
        "expected_validation": "higher paired development success under the frozen evaluator",
    }
