"""Quality gates and manifests for RoboSyn expert demonstration datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_ALIASES = {
    "cam_high": ("observation.images.cam_high", "cam_high.color"),
    "cam_left_wrist": (
        "observation.images.cam_left_wrist",
        "cam_left_wrist.color",
    ),
    "cam_right_wrist": (
        "observation.images.cam_right_wrist",
        "cam_right_wrist.color",
    ),
}
STATE_ALIASES = ("observation.state", "observation.qpos")


def evaluation_seed_bank(master_seed: int = 0, episodes: int = 100) -> list[int]:
    rng = np.random.RandomState(int(master_seed))
    return [int(rng.randint(0, 2**31 - 1)) for _ in range(int(episodes))]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_lengths(dataset_root: Path, version: str) -> list[int]:
    if version == "v2.1":
        rows = []
        episodes_path = dataset_root / "meta" / "episodes.jsonl"
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return [int(row["length"]) for row in rows]

    import pyarrow.parquet as pq

    lengths = []
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["length"])
        lengths.extend(int(value) for value in table["length"].to_pylist())
    return lengths


def _numeric_data_audit(
    dataset_root: Path, info: dict[str, Any], lengths: list[int], *,
    state_dim: int, action_dim: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    data_paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError("no data parquet files")
    schema_names = set(pq.read_schema(data_paths[0]).names)
    state_key = next((key for key in STATE_ALIASES if key in schema_names), None)
    if state_key is None or "action" not in schema_names:
        raise ValueError(
            f"state/action columns missing from parquet: state={state_key}, action={'action' in schema_names}"
        )
    columns = [state_key, "action"]
    table = pq.read_table(data_paths, columns=columns)
    report = {}
    arrays = {}
    expected_dims = {state_key: state_dim, "action": action_dim}
    for key in columns:
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        expected_dim = expected_dims[key]
        if values.ndim != 2 or values.shape[1] != expected_dim:
            raise ValueError(f"{key} sample shape is {values.shape}, expected (*, {expected_dim})")
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains NaN or Inf")
        arrays[key] = values
        report[key] = {
            "audited_frames": int(values.shape[0]),
            "shape": list(values.shape),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }
    expected_frames = int(sum(lengths))
    if len(arrays["action"]) != expected_frames:
        raise ValueError(
            f"parquet frames={len(arrays['action'])}, episode lengths sum={expected_frames}"
        )

    action = arrays["action"]
    state = arrays[state_key]
    episode_hashes: dict[str, list[int]] = {}
    initial_states = []
    action_deltas = []
    offset = 0
    for episode_index, length in enumerate(lengths):
        episode_action = np.ascontiguousarray(action[offset : offset + length])
        digest = hashlib.sha256(episode_action.tobytes()).hexdigest()
        episode_hashes.setdefault(digest, []).append(episode_index)
        initial_states.append(state[offset])
        if length > 1:
            action_deltas.append(np.abs(np.diff(episode_action, axis=0)).mean(axis=0))
        offset += length
    duplicate_groups = [indices for indices in episode_hashes.values() if len(indices) > 1]
    initial_states = np.asarray(initial_states, dtype=np.float32)
    delta_mean = (
        np.asarray(action_deltas, dtype=np.float32).mean(axis=0)
        if action_deltas
        else np.zeros(action_dim, dtype=np.float32)
    )
    passive_dims = np.flatnonzero(delta_mean < 1e-5).astype(int).tolist()
    report["trajectory_diversity"] = {
        "unique_exact_action_trajectories": len(episode_hashes),
        "duplicate_exact_action_groups": duplicate_groups,
        "initial_state_std": initial_states.std(axis=0).tolist(),
        "mean_absolute_action_delta": delta_mean.tolist(),
        "effectively_static_action_dims": passive_dims,
    }
    return report


def _padding_audit(lengths: list[int], chunk_size: int = 50) -> dict[str, Any]:
    total_anchors = int(sum(lengths))
    padded_anchors = 0
    valid_labels = 0
    for length in lengths:
        for anchor in range(length):
            valid = min(chunk_size, length - anchor)
            valid_labels += valid
            padded_anchors += int(valid < chunk_size)
    return {
        "chunk_size": int(chunk_size),
        "total_anchors": total_anchors,
        "padded_anchor_fraction": (
            padded_anchors / total_anchors if total_anchors else None
        ),
        "mean_valid_actions_per_anchor": (
            valid_labels / total_anchors if total_anchors else None
        ),
        "valid_label_fraction": (
            valid_labels / (total_anchors * chunk_size) if total_anchors else None
        ),
    }


def _sample_videos(
    dataset_root: Path, lengths: list[int], *, max_episodes: int | None = 3,
    expected_shapes: dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    import av

    report = []
    for camera_dir in sorted((dataset_root / "videos").glob("chunk-*/*")):
        if not camera_dir.is_dir():
            continue
        paths = sorted(camera_dir.glob("episode_*.mp4"))
        if max_episodes is not None:
            paths = paths[:max_episodes]
        for path in paths:
            episode_index = int(path.stem.rsplit("_", 1)[-1])
            first = None
            last = None
            frame_count = 0
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    array = frame.to_ndarray(format="rgb24")
                    if first is None:
                        first = array
                    last = array
                    frame_count += 1
            expected = lengths[episode_index]
            if frame_count != expected:
                raise ValueError(
                    f"{path} decoded {frame_count} frames, expected {expected}"
                )
            expected_shape = (expected_shapes or {}).get(camera_dir.name, [480, 640, 3])
            if first is None or first.shape != tuple(expected_shape):
                raise ValueError(f"{path} has invalid RGB shape")
            pixel_std = float(first.std())
            if pixel_std < 1.0:
                raise ValueError(f"{path} appears blank/constant (std={pixel_std:.3f})")
            report.append(
                {
                    "path": str(path.relative_to(dataset_root)),
                    "episode_index": episode_index,
                    "decoded_frames": frame_count,
                    "first_frame_mean": float(first.mean()),
                    "first_frame_std": pixel_std,
                    "first_last_mean_absolute_delta": float(
                        np.abs(first.astype(np.float32) - last.astype(np.float32)).mean()
                    ),
                }
            )
    return report


def validate_dataset(
    dataset_root: str | Path,
    *,
    collection_manifest: str | Path | None = None,
    eval_master_seed: int = 0,
    eval_episodes: int = 100,
    state_dim: int = 14,
    action_dim: int = 14,
    max_episode_frames: int | None = 361,
    camera_aliases: dict[str, tuple[str, ...]] | None = None,
    chunk_size: int = 50,
    full_video_decode: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}")
    info = _read_json(info_path)
    version = str(info.get("codebase_version", "unknown"))
    if version not in {"v2.1", "v3.0"}:
        errors.append(f"unsupported codebase_version={version}")

    features = info.get("features", {})
    state_key = next((key for key in STATE_ALIASES if key in features), None)
    aliases = CAMERA_ALIASES if camera_aliases is None else camera_aliases
    for key, expected_dim in ((state_key, state_dim), ("action", action_dim)):
        shape = features.get(key, {}).get("shape") if key is not None else None
        if shape != [expected_dim]:
            errors.append(f"{key} shape={shape}, expected [{expected_dim}]")
    resolved_cameras = {
        semantic: next((key for key in aliases if key in features), None)
        for semantic, aliases in aliases.items()
    }
    if any(value is None for value in resolved_cameras.values()):
        errors.append(f"RGB camera aliases unresolved: {resolved_cameras}")

    try:
        lengths = _episode_lengths(root, version)
    except Exception as exc:
        errors.append(f"episode metadata unreadable: {type(exc).__name__}: {exc}")
        lengths = []
    total_episodes = int(info.get("total_episodes", len(lengths)) or len(lengths))
    if total_episodes != len(lengths):
        errors.append(f"info total_episodes={total_episodes}, metadata rows={len(lengths)}")
    if any(length <= 0 or (max_episode_frames is not None and length > max_episode_frames)
           for length in lengths):
        errors.append(f"episode length is outside [1, {max_episode_frames}]")

    video_count = len(list((root / "videos").glob("**/*.mp4")))
    minimum_video_files = len(aliases) if total_episodes else 0
    if video_count < minimum_video_files:
        errors.append(f"video files={video_count}, expected at least {minimum_video_files}")
    elif version == "v2.1" and video_count != total_episodes * len(aliases):
        errors.append(
            f"v2.1 video files={video_count}, expected {total_episodes * len(aliases)}"
        )

    try:
        numeric_sample = _numeric_data_audit(
            root, info, lengths, state_dim=state_dim, action_dim=action_dim)
    except Exception as exc:
        errors.append(f"numeric sample invalid: {type(exc).__name__}: {exc}")
        numeric_sample = None

    video_sample = None
    if version == "v2.1" and lengths:
        try:
            video_sample = _sample_videos(root, lengths,
                max_episodes=None if full_video_decode else 3, expected_shapes={
                key: value["shape"] for key, value in features.items()
                if value.get("dtype") == "video"
            })
        except Exception as exc:
            errors.append(f"video sample invalid: {type(exc).__name__}: {exc}")

    seed_audit = None
    if collection_manifest is not None:
        manifest_path = Path(collection_manifest).expanduser().resolve()
        manifest = _read_json(manifest_path)
        success_seeds = [int(value) for value in manifest.get("successful_episode_seeds", [])]
        failed_seeds = [int(value) for value in manifest.get("failed_attempt_seeds", [])]
        eval_seeds = set(evaluation_seed_bank(eval_master_seed, eval_episodes))
        overlaps = sorted(set(success_seeds + failed_seeds) & eval_seeds)
        if manifest.get("status") != "completed":
            errors.append(f"collection manifest status={manifest.get('status')}")
        if len(success_seeds) != total_episodes:
            errors.append(
                f"successful seeds={len(success_seeds)}, dataset episodes={total_episodes}"
            )
        if len(success_seeds) != len(set(success_seeds)):
            errors.append("successful episode seeds contain duplicates")
        if overlaps:
            errors.append(f"collection/evaluation seed overlap: {overlaps}")
        declared_paths = {str(Path(value).resolve()) for value in manifest.get("dataset_paths", [])}
        if str(root) not in declared_paths:
            errors.append("dataset path is not declared by collection manifest")
        seed_audit = {
            "successful_seed_count": len(success_seeds),
            "failed_seed_count": len(failed_seeds),
            "evaluation_master_seed": int(eval_master_seed),
            "evaluation_episode_count": int(eval_episodes),
            "overlap": overlaps,
            "lineage": manifest.get("lineage"),
        }
        if manifest.get("schema_version", 1) < 2:
            warnings.append("collection manifest has no v2 correction/composite lineage")
        if manifest.get("collection_mode") == "policy_correction":
            saved_attempts = [
                row for row in manifest.get("attempts", []) if row.get("saved")
            ]
            missing_prefix = [
                row.get("seed")
                for row in saved_attempts
                if int(row.get("policy_prefix_steps", 0)) <= 0
            ]
            lineage = manifest.get("lineage") or {}
            if missing_prefix:
                errors.append(
                    f"saved corrections missing policy-induced prefix: {missing_prefix}"
                )
            if not lineage.get("parent_checkpoint") or not lineage.get(
                "parent_model_sha256"
            ):
                errors.append("policy corrections lack parent checkpoint lineage")
            seed_audit["policy_correction"] = {
                "saved_prefix_step_min": (
                    min(int(row["policy_prefix_steps"]) for row in saved_attempts)
                    if saved_attempts
                    else None
                ),
                "saved_prefix_step_max": (
                    max(int(row["policy_prefix_steps"]) for row in saved_attempts)
                    if saved_attempts
                    else None
                ),
                "safe_return_episode_count": sum(
                    bool(row.get("safe_return_used")) for row in saved_attempts
                ),
            }

    if lengths and len(set(lengths)) == 1:
        warnings.append(
            "all trajectories have identical length; verify this is task structure, not recorder truncation"
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset": {
            "codebase_version": version,
            "total_episodes": total_episodes,
            "total_frames": int(info.get("total_frames", sum(lengths)) or sum(lengths)),
            "episode_length_min": min(lengths) if lengths else None,
            "episode_length_max": max(lengths) if lengths else None,
            "episode_length_mean": float(np.mean(lengths)) if lengths else None,
            "video_file_count": video_count,
            "semantic_schema": {
                "state": state_key,
                "cameras": resolved_cameras,
                "action": "action",
            },
            "numeric_sample": numeric_sample,
            "video_sample": video_sample,
            "video_decode_scope": "all" if full_video_decode else "first_three_per_camera",
            "action_chunk_padding": _padding_audit(lengths, chunk_size=chunk_size),
        },
        "seed_audit": seed_audit,
    }


def write_mixture_manifest(
    output: str | Path,
    official_dataset: str | Path,
    targeted: list[str],
    *,
    sampling: dict[str, Any] | str = "proportional_to_frames",
    max_targeted_fraction: float = 0.60,
) -> dict[str, Any]:
    datasets = [
        {
            "role": "official_full_random",
            "profile": "full_random",
            "root": str(Path(official_dataset).expanduser().resolve()),
        }
    ]
    for value in targeted:
        if "=" not in value:
            raise ValueError("--targeted entries must be PROFILE=DATASET_ROOT")
        profile, path = value.split("=", 1)
        datasets.append(
            {
                "role": "targeted_training_only",
                "profile": profile,
                "root": str(Path(path).expanduser().resolve()),
            }
        )
    for item in datasets:
        info = _read_json(Path(item["root"]) / "meta" / "info.json")
        item["episode_count"] = int(info["total_episodes"])
        item["frame_count"] = int(info["total_frames"])
    total_episodes = sum(item["episode_count"] for item in datasets)
    total_frames = sum(item["frame_count"] for item in datasets)
    targeted_episodes = sum(
        item["episode_count"]
        for item in datasets
        if item["role"] == "targeted_training_only"
    )
    targeted_fraction = targeted_episodes / total_episodes
    if targeted_fraction > float(max_targeted_fraction):
        raise ValueError(
            f"targeted episode fraction {targeted_fraction:.3f} exceeds "
            f"{max_targeted_fraction:.2f} guardrail"
        )
    payload = {
        "schema_version": 2,
        "kind": "robosyn_training_mixture",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling": sampling,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "targeted_episode_fraction": targeted_fraction,
        "datasets": datasets,
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def audit_collection_manifests(
    manifests: list[str | Path],
    *,
    output: str | Path | None = None,
    eval_master_seed: int = 0,
    eval_episodes: int = 100,
    excluded_seed_banks: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    bank_specs = dict(excluded_seed_banks or {})
    frozen_spec = (int(eval_master_seed), int(eval_episodes))
    # Callers may already provide the frozen bank under a protocol-specific
    # name (for example ``internal_frozen_v3``).  Do not introduce a second
    # alias for the exact same bank: aliases are not independent banks and
    # must not be reported as a seed-bank collision.
    if frozen_spec not in bank_specs.values():
        bank_specs.setdefault("frozen_final", frozen_spec)
    banks = {
        name: set(evaluation_seed_bank(master_seed, episodes))
        for name, (master_seed, episodes) in bank_specs.items()
    }
    bank_overlaps = []
    bank_names = sorted(banks)
    for index, left in enumerate(bank_names):
        for right in bank_names[index + 1 :]:
            overlap = sorted(banks[left] & banks[right])
            if overlap:
                bank_overlaps.append(
                    {"banks": [left, right], "seeds": overlap}
                )
    owner_by_seed: dict[int, str] = {}
    cross_profile_overlaps = []
    evaluation_overlaps = []
    profiles = []
    for value in manifests:
        path = Path(value).expanduser().resolve()
        manifest = _read_json(path)
        profile = str(manifest["profile"])
        success = [int(seed) for seed in manifest.get("successful_episode_seeds", [])]
        failed = [int(seed) for seed in manifest.get("failed_attempt_seeds", [])]
        attempted = success + failed
        for seed in attempted:
            previous = owner_by_seed.get(seed)
            if previous is not None and previous != profile:
                cross_profile_overlaps.append(
                    {"seed": seed, "profiles": sorted({previous, profile})}
                )
            owner_by_seed[seed] = profile
            for bank_name, bank in banks.items():
                if seed in bank:
                    evaluation_overlaps.append(
                        {"seed": seed, "profile": profile, "bank": bank_name}
                    )
        profiles.append(
            {
                "profile": profile,
                "manifest": str(path),
                "successful_episodes": len(success),
                "failed_attempts": len(failed),
                "expert_attempt_success_rate": (
                    len(success) / len(attempted) if attempted else None
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "robosyn_collection_seed_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": (
            not cross_profile_overlaps
            and not evaluation_overlaps
            and not bank_overlaps
        ),
        "evaluation_master_seed": int(eval_master_seed),
        "evaluation_episode_count": int(eval_episodes),
        "excluded_seed_banks": {
            name: {"master_seed": spec[0], "episodes": spec[1]}
            for name, spec in bank_specs.items()
        },
        "seed_bank_overlaps": bank_overlaps,
        "unique_collection_attempt_seeds": len(owner_by_seed),
        "cross_profile_overlaps": cross_profile_overlaps,
        "evaluation_overlaps": evaluation_overlaps,
        "profiles": profiles,
    }
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    qc = subparsers.add_parser("qc")
    qc.add_argument("--dataset-root", required=True)
    qc.add_argument("--collection-manifest")
    qc.add_argument("--output")
    qc.add_argument("--eval-master-seed", type=int, default=0)
    qc.add_argument("--eval-episodes", type=int, default=100)
    mixture = subparsers.add_parser("mixture")
    mixture.add_argument("--official-dataset", required=True)
    mixture.add_argument("--targeted", action="append", default=[])
    mixture.add_argument("--output", required=True)
    mixture.add_argument(
        "--sampling-config",
        help="JSON file containing a sampling object (e.g. stratified_phase).",
    )
    mixture.add_argument("--max-targeted-fraction", type=float, default=0.60)
    seed_audit = subparsers.add_parser("seed-audit")
    seed_audit.add_argument("--manifest", action="append", required=True)
    seed_audit.add_argument("--output", required=True)
    seed_audit.add_argument("--eval-master-seed", type=int, default=0)
    seed_audit.add_argument("--eval-episodes", type=int, default=100)
    seed_audit.add_argument(
        "--exclude-bank",
        action="append",
        default=[],
        metavar="NAME=MASTER_SEED:EPISODES",
        help="Also prohibit collection overlap with a development seed bank.",
    )
    args = parser.parse_args()

    if args.command == "qc":
        report = validate_dataset(
            args.dataset_root,
            collection_manifest=args.collection_manifest,
            eval_master_seed=args.eval_master_seed,
            eval_episodes=args.eval_episodes,
        )
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0 if report["passed"] else 1

    if args.command == "seed-audit":
        banks = {}
        for value in args.exclude_bank:
            name, specification = value.split("=", 1)
            master_seed, episodes = specification.split(":", 1)
            banks[name] = (int(master_seed), int(episodes))
        payload = audit_collection_manifests(
            args.manifest,
            output=args.output,
            eval_master_seed=args.eval_master_seed,
            eval_episodes=args.eval_episodes,
            excluded_seed_banks=banks,
        )
        print(json.dumps(payload, indent=2))
        return 0 if payload["passed"] else 1

    sampling = (
        _read_json(Path(args.sampling_config).expanduser().resolve())
        if args.sampling_config
        else "proportional_to_frames"
    )
    payload = write_mixture_manifest(
        args.output,
        args.official_dataset,
        args.targeted,
        sampling=sampling,
        max_targeted_fraction=args.max_targeted_fraction,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
