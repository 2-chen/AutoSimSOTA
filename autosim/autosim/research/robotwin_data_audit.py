"""Strict, machine-readable audit for imported RoboTwin demonstration data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist", "cam_third_view")


def _joint_vector(root: h5py.File, group: str) -> np.ndarray:
    packed = f"{group}/joint_states"
    if packed in root:
        return root[packed][()]
    keys = ("left_arm_joint_states", "left_ee_joint_states",
            "right_arm_joint_states", "right_ee_joint_states")
    return np.concatenate([root[f"{group}/{key}"][()] for key in keys], axis=-1)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_dataset(dataset: Path, *, decode_all_images: bool = True) -> dict:
    files = sorted((dataset / "data").glob("episode_*.hdf5"))
    instructions = sorted((dataset / "instruction").glob("episode_*.json"))
    videos = sorted((dataset / "video").glob("episode_*.mp4"))
    if not files:
        raise ValueError(f"no HDF5 episodes under {dataset / 'data'}")
    expected = [f"episode_{index:07d}.hdf5" for index in range(len(files))]
    if [path.name for path in files] != expected:
        raise ValueError("episode files are not contiguous from zero")

    episodes = []
    total_frames = 0
    decoded_frames = 0
    for path in files:
        with h5py.File(path, "r") as root:
            state = _joint_vector(root, "state")
            action = _joint_vector(root, "action")
            length = int(state.shape[0])
            if state.shape != action.shape or state.shape[1:] != (14,):
                raise ValueError(f"state/action contract mismatch in {path}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"non-finite state/action in {path}")
            frequency = int(root["additional_info/frequency"][()])
            camera_shapes = {}
            for camera in CAMERAS:
                colors = root[f"vision/{camera}/colors"]
                if colors.shape != (length,):
                    raise ValueError(f"camera length mismatch for {camera} in {path}")
                shape = tuple(int(x) for x in root[f"vision/{camera}/shape"][()])
                camera_shapes[camera] = list(shape)
                indexes = range(length) if decode_all_images else {0, length - 1}
                for index in indexes:
                    image = cv2.imdecode(np.frombuffer(colors[index], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None or tuple(image.shape) != shape:
                        raise ValueError(f"invalid encoded image {camera}[{index}] in {path}")
                    decoded_frames += 1
            total_frames += length
            episodes.append({"file": path.name, "bytes": path.stat().st_size,
                             "sha256": _sha256(path), "steps": length,
                             "frequency_hz": frequency, "camera_shapes": camera_shapes})

    result = {
        "schema_version": 1,
        "kind": "robotwin_official_dataset_audit",
        "status": "passed",
        "dataset": str(dataset.absolute()),
        "episode_count": len(files),
        "instruction_file_count": len(instructions),
        "video_file_count": len(videos),
        "total_steps": total_frames,
        "decoded_image_count": decoded_frames,
        "image_decode_scope": "all_frames" if decode_all_images else "first_and_last_per_camera",
        "state_dim": 14,
        "action_dim": 14,
        "frequencies_hz": sorted({row["frequency_hz"] for row in episodes}),
        "episodes": episodes,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    result["audit_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-images", action="store_true")
    args = parser.parse_args()
    result = audit_dataset(args.dataset.absolute(), decode_all_images=not args.sample_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "episode_count", "total_steps", "decoded_image_count", "audit_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
