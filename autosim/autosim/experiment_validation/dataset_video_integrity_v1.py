"""Strict, full-decode admission gate for local LeRobot video datasets.

File counts and container headers are insufficient: a damaged packet can remain
latent until a shuffled DataLoader reaches it.  This gate decodes every expected
frame before a collected dataset is admitted to policy training.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

from autosim.research.common import atomic_json, digest, now


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _decode_video(dataset: Path, path: Path, expected_frames: int,
                  expected_shape: tuple[int, int]) -> dict:
    import av

    decoded = 0
    observed_shapes: set[tuple[int, int]] = set()
    error = None
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                decoded += 1
                observed_shapes.add((int(frame.height), int(frame.width)))
    except Exception as exc:  # PyAV exposes codec-specific exception classes.
        error = f"{type(exc).__name__}: {exc}"
    passed = error is None and decoded == expected_frames and observed_shapes == {expected_shape}
    return {
        "path": str(path.relative_to(dataset)),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "expected_frames": expected_frames,
        "decoded_frames": decoded,
        "expected_shape": list(expected_shape),
        "observed_shapes": [list(shape) for shape in sorted(observed_shapes)],
        "error": error,
        "passed": passed,
    }


def audit_dataset(dataset: Path, *, expected_episodes: int | None = None,
                  expected_video_keys: list[str] | None = None,
                  workers: int = 4) -> dict:
    dataset = dataset.absolute()
    info_path = dataset / "meta/info.json"
    episodes_path = dataset / "meta/episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        missing = [str(path) for path in (info_path, episodes_path) if not path.is_file()]
        return {"kind": "lerobot_full_video_decode_audit_v1", "created_at": now(),
                "dataset": str(dataset), "passed": False, "missing_metadata": missing,
                "training_admission": "rejected"}

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = _read_jsonl(episodes_path)
    features = info.get("features", {})
    video_features = {key: value for key, value in features.items()
                      if value.get("dtype") == "video"}
    video_keys = sorted(video_features)
    if expected_video_keys is not None:
        expected_keys = sorted(expected_video_keys)
    else:
        expected_keys = video_keys

    declared_episodes = int(info.get("total_episodes", -1))
    requested_episodes = declared_episodes if expected_episodes is None else int(expected_episodes)
    indices = [int(row.get("episode_index", -1)) for row in episodes]
    lengths = {int(row.get("episode_index", -1)): int(row.get("length", -1)) for row in episodes}
    checks = [
        {"name": "episode_count", "passed": declared_episodes == requested_episodes == len(episodes),
         "declared": declared_episodes, "metadata_rows": len(episodes), "expected": requested_episodes},
        {"name": "episode_indices_contiguous", "passed": indices == list(range(requested_episodes)),
         "actual": indices, "expected_start": 0, "expected_end": requested_episodes - 1},
        {"name": "video_keys", "passed": video_keys == expected_keys,
         "actual": video_keys, "expected": expected_keys},
    ]

    expected_paths: list[tuple[Path, int, tuple[int, int]]] = []
    for episode_index in range(max(requested_episodes, 0)):
        chunk = episode_index // int(info.get("chunks_size", 1000))
        for key in expected_keys:
            feature = video_features.get(key, {})
            shape = feature.get("shape", [-1, -1])
            path = dataset / str(info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )).format(episode_chunk=chunk, video_key=key, episode_index=episode_index)
            expected_paths.append((path, lengths.get(episode_index, -1),
                                   (int(shape[0]), int(shape[1]))))

    actual_paths = set(dataset.glob("videos/chunk-*/*/episode_*.mp4"))
    expected_path_set = {item[0] for item in expected_paths}
    missing_videos = sorted(str(path.relative_to(dataset)) for path in expected_path_set - actual_paths)
    unexpected_videos = sorted(str(path.relative_to(dataset)) for path in actual_paths - expected_path_set)
    checks.append({"name": "video_file_set", "passed": not missing_videos and not unexpected_videos,
                   "expected_count": len(expected_path_set), "actual_count": len(actual_paths),
                   "missing": missing_videos, "unexpected": unexpected_videos})

    decodable = [item for item in expected_paths if item[0].is_file()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        decoded = list(executor.map(lambda item: _decode_video(dataset, *item), decodable))
    failures = [row for row in decoded if not row["passed"]]
    checks.append({"name": "full_video_decode", "passed": not failures and len(decoded) == len(expected_paths),
                   "decoded_files": len(decoded), "expected_files": len(expected_paths),
                   "failed_files": len(failures)})

    aggregate = hashlib.sha256()
    for row in sorted(decoded, key=lambda item: item["path"]):
        aggregate.update(row["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(row["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    passed = all(check["passed"] for check in checks)
    return {
        "kind": "lerobot_full_video_decode_audit_v1",
        "created_at": now(),
        "dataset": str(dataset),
        "passed": passed,
        "training_admission": "admitted" if passed else "rejected",
        "checks": checks,
        "episode_count": len(episodes),
        "video_key_count": len(expected_keys),
        "expected_video_count": len(expected_paths),
        "decoded_video_count": len(decoded),
        "decoded_frame_count": sum(row["decoded_frames"] for row in decoded),
        "failed_videos": failures,
        "video_set_sha256": aggregate.hexdigest(),
        "metadata_sha256": {str(info_path): digest(info_path), str(episodes_path): digest(episodes_path)},
        "scope_note": "Every expected video was fully decoded and checked against per-episode frame counts; this validates serialization, not task success or policy performance.",
        "policy_performance_claim": False,
        "official_leaderboard_result": False,
        "sota_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-video-key", action="append", dest="expected_video_keys")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = audit_dataset(args.dataset, expected_episodes=args.expected_episodes,
                           expected_video_keys=args.expected_video_keys, workers=args.workers)
    atomic_json(args.output.absolute(), result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
