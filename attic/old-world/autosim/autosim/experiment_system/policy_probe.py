"""Reload a checkpoint and infer on real recorded RGB/state, without simulation."""
import argparse
import time
from pathlib import Path

from autosim.research.common import atomic_json, digest, read_json


def main():
    import av
    import numpy as np
    import pyarrow.parquet as pq
    from autosim.research.policy_rpc import RemotePolicy
    from autosim.research.registry import load_task
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="click_bell")
    parser.add_argument("--policy", choices=["act", "dp"], required=True)
    args = parser.parse_args()
    spec = load_task(args.workspace / "AutoSimSOTA/RoboSynChallenge", args.task)
    parquet = sorted((args.dataset / "data").rglob("*.parquet"))[0]
    row = pq.read_table(parquet).to_pylist()[0]
    info = read_json(args.dataset / "meta/info.json")
    state = row.get("observation.state", row.get("observation.qpos"))
    observation = {"robot": {"qpos": np.asarray(state, dtype=np.float32)}, "sensor": {}}
    sources = {str(parquet): digest(parquet)}
    for camera in spec.cameras:
        key = f"observation.images.{camera}" if f"observation.images.{camera}" in info["features"] else f"{camera}.color"
        path = args.dataset / info["video_path"].format(video_key=key, episode_chunk=0, episode_index=int(row["episode_index"]))
        with av.open(str(path)) as container:
            rgb = next(container.decode(video=0)).to_ndarray(format="rgb24")
        observation["sensor"][camera] = {"color": rgb}
        sources[str(path)] = digest(path)
    config = {"checkpoint_path": str(args.checkpoint), "policy_name": args.policy, "pytorch_device": "cpu",
              "act_step": 1, "dp_step": 1, "dp_num_inference_steps": 10}
    start = time.monotonic()
    model = RemotePolicy(config, spec.as_dict())
    try:
        model.reset()
        result = model.predict(observation)
        if result["action"].shape != (1, spec.action_dim) or not np.isfinite(result["action"]).all():
            raise ValueError("invalid real-data policy output")
    finally:
        model.close()
    atomic_json(args.output, {"status": "completed", "device": "cpu", "policy": args.policy,
                "action_shape": list(result["action"].shape), "action_finite": True,
                "elapsed_seconds": time.monotonic() - start, "input_sources": sources,
                "checkpoint_sha256": digest(args.checkpoint / "model.safetensors"),
                "execution_mode": "recorded_observation_inference", "simulation_success_rate": None})
    print({"reloaded": True, "real_observation_inference": True, "simulation_evaluated": False})


if __name__ == "__main__":
    main()
