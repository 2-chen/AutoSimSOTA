"""Build an auditable, hard-linked ACT dataset from official and new episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .common import atomic_json, digest, now, object_digest, read_json


def episode_files(root: Path) -> list[Path]:
    files = sorted(root.glob("episode_*.hdf5"), key=lambda path: int(path.stem.split("_")[-1]))
    if not files:
        raise ValueError(f"no processed ACT episodes in {root}")
    return files


def build_mix(official: Path, extra: Path, output: Path, registry: Path,
              task_config_key: str) -> dict:
    official, extra, output, registry = map(Path, (official, extra, output, registry))
    sources = [("official", path) for path in episode_files(official)]
    sources += [("new_collection", path) for path in episode_files(extra)]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (lineage, source) in enumerate(sources):
        destination = output / f"episode_{index}.hdf5"
        if destination.exists():
            if not os.path.samefile(source, destination):
                raise FileExistsError(f"refusing to replace non-matching mixed episode: {destination}")
        else:
            os.link(source, destination)
        rows.append({"episode_index": index, "lineage": lineage, "source": str(source.absolute()),
                     "destination": str(destination.absolute()), "bytes": source.stat().st_size,
                     "sha256": digest(source), "hard_linked": os.path.samefile(source, destination)})

    act_root = registry.parent
    try:
        dataset_dir = str(output.absolute().relative_to(act_root.absolute()))
    except ValueError as exc:
        raise ValueError("mixed dataset must be below the ACT directory") from exc
    config = read_json(registry) if registry.is_file() else {}
    expected = {"dataset_dir": dataset_dir, "num_episodes": len(rows), "episode_len": 5000,
                "camera_names": ["cam_head", "cam_right_wrist", "cam_left_wrist"]}
    if task_config_key in config and config[task_config_key] != expected:
        raise ValueError(f"existing TASK_CONFIGS entry differs: {task_config_key}")
    config[task_config_key] = expected
    atomic_json(registry, config)
    result = {"schema_version": 1, "kind": "robotwin_act_mixed_dataset", "created_at": now(),
              "status": "passed", "task_config_key": task_config_key,
              "official_episode_count": sum(row["lineage"] == "official" for row in rows),
              "new_episode_count": sum(row["lineage"] == "new_collection" for row in rows),
              "episode_count": len(rows), "all_hard_linked": all(row["hard_linked"] for row in rows),
              "episodes": rows, "manifest_sha256": object_digest(rows)}
    atomic_json(output / "AUTOSIM_DATA_MANIFEST.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--task-config-key", required=True)
    args = parser.parse_args()
    result = build_mix(args.official, args.extra, args.output, args.registry, args.task_config_key)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
