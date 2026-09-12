"""Explicit real backend stages. All writes are in the supplied attempt directory."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from autosim.research.common import atomic_json, digest, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["collect", "admit", "train", "evaluate", "robotwin-render"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--task", default="click_bell")
    parser.add_argument("--benchmark", choices=["robosyn", "robotwin"], default="robosyn")
    parser.add_argument("--native-repo", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy", choices=["act", "dp"], default="act")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=91006000)
    args = parser.parse_args()
    args.attempt.mkdir(parents=True, exist_ok=True)
    if args.stage == "robotwin-render":
        import numpy as np
        import sapien.core as sapien
        engine = sapien.Engine()
        engine.set_renderer(sapien.SapienRenderer())
        scene = engine.create_scene()
        scene.set_ambient_light([.5, .5, .5])
        scene.add_ground(0)
        camera = scene.add_camera("preflight", 64, 64, 1.0, .01, 10)
        camera.set_pose(sapien.Pose([0, 0, 1]))
        scene.step()
        scene.update_render()
        camera.take_picture()
        pixels = camera.get_picture("Color")
        if pixels.shape != (64, 64, 4) or not np.isfinite(pixels).all():
            raise ValueError("invalid native renderer output")
        result = {"status": "completed", "kind": "native_renderer_probe", "shape": list(pixels.shape),
                  "task_workflow_validated": False, "execution_mode": "real_simulation"}
    else:
        from autosim.research.registry import load_task
        from autosim.research.runtime import Runtime
        runtime = Runtime(args.workspace, args.attempt)
        if args.benchmark == "robotwin":
            from .plugins import RoboTwinPlugin
            contract = RoboTwinPlugin(args.native_repo).contract(args.task)
            if args.stage == "train":
                from .trainers import train_canonical
                result = train_canonical(runtime, contract, args.dataset, args.attempt / "outputs", args.policy, args.steps, args.seed)
            elif args.stage == "admit":
                from .quality import audit_dataset
                result = audit_dataset(args.dataset, state_dim=contract.observation["state_dim"], action_dim=contract.action["dimension"])
                if not result["passed"]:
                    atomic_json(args.attempt / "rejected_data.json", result)
                    raise ValueError("dataset admission failed")
            else:
                raise ValueError("native RoboTwin collection/evaluation must use native worker")
            atomic_json(args.attempt / "result.json", result)
            return
        spec = load_task(runtime.repo, args.task)
        output = args.attempt / "outputs"
        result = {"status": "completed", "stage": args.stage, "task": args.task, "policy": args.policy}
        if args.stage == "collect":
            dataset = runtime.collect(spec, output, episodes=args.episodes, master_seed=args.seed)
            runtime.prepare_data(spec, dataset, output / "prepare")
            result.update(dataset=str(dataset), verified_files={str(output / "collection.json"): digest(output / "collection.json")})
        elif args.stage == "admit":
            from .quality import audit_dataset
            result = audit_dataset(args.dataset, state_dim=spec.state_dim, action_dim=spec.action_dim)
            if not result["passed"]:
                atomic_json(args.attempt / "rejected_data.json", result)
                raise ValueError("dataset admission failed")
        elif args.stage == "train":
            if args.policy == "act":
                checkpoint = runtime.train(spec, args.dataset, output, steps=args.steps, seed=args.seed,
                                           params={"batch_size": 8, "num_workers": 2})
            else:
                command = [str(runtime.python), "policy/dp/scripts/train.py", "--dataset-root", str(args.dataset),
                           "--output-dir", str(output / "train"), "--steps", str(args.steps),
                           "--save-freq", str(args.steps), "--batch-size", "2", "--num-workers", "2",
                           "--seed", str(args.seed), "--img-micro-bs", "2", "--log-freq", "10"]
                runtime.run(command, output / "process", max(1800, args.steps * 5))
                checkpoint = output / "train/checkpoints" / f"{args.steps:06d}" / "pretrained_model"
            result.update(checkpoint=str(checkpoint), verified_files={str(checkpoint / f): digest(checkpoint / f)
                          for f in ("model.safetensors", "config.json")},
                          reload_validated=False, native_evaluation_validated=False)
        else:
            metrics = runtime.evaluate(spec, args.checkpoint, output, episodes=args.episodes,
                                       master_seed=args.seed, purpose="smoke", policy=args.policy)
            result.update(metrics=metrics, verified_files={str(output / "evaluation_metrics.json"): digest(output / "evaluation_metrics.json")},
                          execution_mode="real_simulation", policy_quality_claim=False)
    atomic_json(args.attempt / "result.json", result)


if __name__ == "__main__":
    main()
