"""Read-only trajectory admission. Structural validity is not learning utility."""
import hashlib
import math
from pathlib import Path

from autosim.research.common import digest, read_json


def audit_episode(rows: list[dict], *, state_key: str, state_dim: int, action_dim: int, fps: float):
    import numpy as np

    errors, warnings = [], []
    if not rows or not math.isfinite(fps) or fps <= 0:
        return {"passed": False, "errors": ["empty episode or invalid fps"]}
    try:
        states = np.asarray([r[state_key] for r in rows], dtype=np.float64)
        actions = np.asarray([r["action"] for r in rows], dtype=np.float64)
        if states.shape != (len(rows), state_dim) or actions.shape != (len(rows), action_dim):
            errors.append("state_action_shape")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            errors.append("nonfinite_values")
        frames = np.asarray([r["frame_index"] for r in rows])
        times = np.asarray([r["timestamp"] for r in rows], dtype=float)
        if not np.array_equal(frames, np.arange(len(rows))):
            errors.append("frame_index_gap_or_duplicate")
        if not np.isfinite(times).all() or not np.allclose(times, frames / fps, atol=max(1e-4, .02 / fps), rtol=0):
            errors.append("timestamp_frequency_mismatch")
        if len({r["episode_index"] for r in rows}) != 1:
            errors.append("mixed_episode_indices")
        if len(rows) > 1 and actions.ndim == 2:
            stationary = float(np.mean(np.max(np.abs(np.diff(actions, axis=0)), axis=1) < 1e-6))
        else:
            stationary = 1.0
        if stationary > .8:
            warnings.append("mostly_stationary_actions_may_be_legitimate_hold")
        fingerprint = hashlib.sha256(states.tobytes() + actions.tobytes()).hexdigest()
    except (KeyError, TypeError, ValueError) as exc:
        return {"passed": False, "errors": [f"malformed_episode:{exc}"]}
    return {"passed": not errors, "errors": errors, "warnings": warnings, "frames": len(rows),
            "numeric_trajectory_sha256": fingerprint, "stationary_action_fraction": stationary}


def audit_dataset(root: Path, *, state_dim: int, action_dim: int, decode_videos=True):
    import pyarrow.parquet as pq

    info = read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("admission currently requires canonical v2.1; convert into a separate version first")
    features = info["features"]
    state_key = "observation.state" if "observation.state" in features else "observation.qpos"
    cameras = [k for k, v in features.items() if v.get("dtype") == "video"]
    fps = float(info["fps"])
    rows, errors, seen, episode_ids, total_frames = [], [], {}, set(), 0
    files = sorted((root / "data").rglob("*.parquet"))
    for path in files:
        try:
            records = pq.read_table(path).to_pylist()
            result = audit_episode(records, state_key=state_key, state_dim=state_dim, action_dim=action_dim, fps=fps)
            result["path"] = str(path)
            result["parquet_sha256"] = digest(path)
            if result["passed"]:
                idx = int(records[0]["episode_index"])
                if idx in episode_ids:
                    result["errors"].append("duplicate_episode_index")
                episode_ids.add(idx)
                total_frames += len(records)
                duplicate = seen.get(result["numeric_trajectory_sha256"])
                if duplicate:
                    result["warnings"].append(f"identical_numeric_trajectory:{duplicate}")
                seen[result["numeric_trajectory_sha256"]] = str(path)
                video_records = []
                if decode_videos:
                    import av
                    chunk = idx // int(info.get("chunks_size", 1000))
                    for camera in cameras:
                        template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
                        video = root / template.format(episode_chunk=chunk, video_key=camera, episode_index=idx)
                        if not video.resolve().is_relative_to(root.resolve()):
                            raise ValueError("video path escapes dataset")
                        count, first_pts, previous_pts = 0, None, None
                        with av.open(str(video)) as container:
                            for frame in container.decode(video=0):
                                timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else None
                                if timestamp is None or (previous_pts is not None and timestamp <= previous_pts):
                                    raise ValueError(f"nonmonotonic video timestamps:{camera}")
                                if first_pts is None:
                                    first_pts = timestamp
                                if abs((timestamp - first_pts) - count / fps) > .51 / fps:
                                    raise ValueError(f"video/parquet timebase mismatch:{camera}")
                                shape = features[camera]["shape"]
                                if [frame.height, frame.width, 3] != list(shape):
                                    raise ValueError(f"video shape mismatch:{camera}")
                                previous_pts, count = timestamp, count + 1
                        if count != len(records):
                            raise ValueError(f"video/parquet length mismatch:{camera}:{count}/{len(records)}")
                        video_records.append({"camera": camera, "path": str(video), "frames": count, "sha256": digest(video)})
                result["videos"] = video_records
                result["passed"] = not result["errors"]
        except Exception as exc:
            result = {"path": str(path), "passed": False, "errors": [f"{type(exc).__name__}:{exc}"]}
        rows.append(result)
        errors.extend(result.get("errors", []))
    if len(files) != info.get("total_episodes") or total_frames != info.get("total_frames"):
        errors.append("metadata_count_mismatch")
    if not cameras or not files:
        errors.append("empty_data_or_missing_cameras")
    verified_files = {str(root / "meta/info.json"): digest(root / "meta/info.json")}
    for row in rows:
        if row.get("passed"):
            verified_files[row["path"]] = row["parquet_sha256"]
            verified_files.update({v["path"]: v["sha256"] for v in row["videos"]})
    return {"status": "completed", "passed": not errors, "root": str(root), "errors": errors,
            "verified_files": verified_files,
            "episodes": rows, "full_video_decode": decode_videos,
            "semantic_action_units_verified": False, "physical_camera_action_synchrony_verified": False,
            "limitations": ["timebase agreement cannot prove sensor/action physical synchrony",
                            "numeric duplicates are warnings, not automatic data deletion",
                            "structural admission does not prove downstream learning benefit"]}
