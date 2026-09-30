"""Build M0-M3 manifests after v3 corrections and processing selection."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _dataset_entry(role, profile, root):
    root = Path(root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    return {
        "role": role,
        "profile": profile,
        "root": str(root),
        "episode_count": int(info["total_episodes"]),
        "frame_count": int(info["total_frames"]),
    }


def _manifest(entries, sampling):
    total_episodes = sum(item["episode_count"] for item in entries)
    total_frames = sum(item["frame_count"] for item in entries)
    targeted = sum(
        item["episode_count"]
        for item in entries
        if item["role"] == "targeted_training_only"
    )
    fraction = targeted / total_episodes
    if fraction > 0.6000001:
        raise ValueError(f"targeted fraction exceeds cap: {fraction}")
    return {
        "schema_version": 3,
        "kind": "robosyn_training_mixture",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling": sampling,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "targeted_episode_fraction": fraction,
        "datasets": entries,
    }


def build_mixture_experiments(
    old_mixture_path: str | Path,
    corrections_path: str | Path,
    processing_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    old = json.loads(Path(old_mixture_path).expanduser().resolve().read_text())
    corrections = json.loads(Path(corrections_path).expanduser().resolve().read_text())
    processing = json.loads(Path(processing_path).expanduser().resolve().read_text())
    if corrections.get("status") != "completed" or processing.get("status") != "completed":
        raise ValueError("correction collection and processing selection must be complete")
    selected_name = processing["selected"]["name"]
    selected = processing["candidates"][selected_name]
    base_sampling = json.loads(
        Path(selected["dataset_mixture_manifest"]).read_text()
    )["sampling"]
    if isinstance(base_sampling, str):
        base_sampling = {"strategy": base_sampling}
    old_profile_names = {
        "full_random": "full_random",
        "targeted_clutter": "targeted_clutter_old",
        "targeted_appearance": "targeted_appearance_old",
        "targeted_recovery": "targeted_recovery_old",
        "policy_correction": "policy_correction_old",
        "composite_hard": "composite_hard_old",
    }
    old_entries = [
        _dataset_entry(
            item["role"], old_profile_names[item["profile"]], item["root"]
        )
        for item in old["datasets"]
    ]
    new_profile_names = {
        "targeted_camera": "correction_v3_camera",
        "targeted_recovery": "correction_v3_recovery",
        "targeted_clutter": "correction_v3_clutter",
        "composite_contact": "correction_v3_contact",
    }
    new_entries = [
        _dataset_entry(
            "targeted_training_only",
            new_profile_names[label],
            corrections["shards"][label]["dataset_root"],
        )
        for label in new_profile_names
    ]
    output_dir = Path(output_dir).expanduser().resolve()

    def sampling(masses):
        value = copy.deepcopy(base_sampling)
        value["strategy"] = "stratified_phase"
        value["profile_masses"] = masses
        return value

    all_entries = old_entries + new_entries
    allocations = corrections["allocation"]
    new_fraction = {
        new_profile_names[label]: allocations[label] / 400 for label in new_profile_names
    }
    m1_masses = {
        "full_random": 0.52,
        "targeted_clutter_old": 0.04,
        "targeted_appearance_old": 0.04,
        "targeted_recovery_old": 0.04,
        "policy_correction_old": 0.08,
        "composite_hard_old": 0.08,
        **{key: 0.20 * fraction for key, fraction in new_fraction.items()},
    }
    reduced_entries = [
        item for item in all_entries if not item["profile"].startswith("targeted_")
    ]
    m2_masses = {
        "full_random": 0.55,
        "policy_correction_old": 0.10,
        "composite_hard_old": 0.12,
        **{key: 0.23 * fraction for key, fraction in new_fraction.items()},
    }
    m3_masses = {
        "full_random": 0.50,
        "policy_correction_old": 0.10,
        "composite_hard_old": 0.12,
        **{key: 0.07 for key in new_fraction},
    }
    manifests = {}
    for name, entries, masses in (
        ("M1_append", all_entries, m1_masses),
        ("M2_replace_old_targeted", reduced_entries, m2_masses),
        ("M3_failure_balanced", reduced_entries, m3_masses),
    ):
        path = output_dir / f"{name}.json"
        payload = _manifest(entries, sampling(masses))
        if abs(sum(masses.values()) - 1.0) > 1e-9:
            raise ValueError(f"{name} sampling masses do not sum to one")
        _write_json_atomic(path, payload)
        manifests[name] = str(path)

    spec = {
        "schema_version": 3,
        "stage": "mixture_ablation",
        "processing_winner": selected_name,
        "candidates": [],
    }
    for name, manifest in [
        ("M0_current", selected["dataset_mixture_manifest"]),
        *manifests.items(),
    ]:
        spec["candidates"].append(
            {
                "name": name,
                "hypothesis": name,
                "dataset_mixture_manifest": manifest,
                "train_overrides": {
                    "action_loss_profile": selected["train_params"]["action_loss_profile"],
                    "image_augmentation_profile": selected["train_params"][
                        "image_augmentation_profile"
                    ],
                },
            }
        )
    spec_path = output_dir / "mixture_experiments.json"
    _write_json_atomic(spec_path, spec)
    return {"manifests": manifests, "specification": str(spec_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-mixture", required=True, type=Path)
    parser.add_argument("--corrections", required=True, type=Path)
    parser.add_argument("--processing", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_mixture_experiments(
        args.old_mixture, args.corrections, args.processing, args.output_dir
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
