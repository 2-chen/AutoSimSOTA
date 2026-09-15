"""Reload a real short-trained ACT checkpoint and execute one bounded inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    args = parser.parse_args()
    import torch
    from policy.act.deploy_policy import get_model
    from .common import atomic_json, digest
    step = json.loads((args.checkpoint.parent / "training_state/training_step.json").read_text())["step"]
    if int(step) != args.expected_updates:
        raise ValueError("saved optimizer update count differs from the validation contract")
    policy = get_model({"checkpoint_path":str(args.checkpoint),"device":"cuda"})
    config = json.loads((args.checkpoint / "config.json").read_text())
    batch = {key:torch.zeros((1,*value["shape"]),device="cuda",dtype=torch.float32)
             for key,value in config["input_features"].items()}
    policy.reset()
    with torch.inference_mode():
        action = policy.select_action(batch)
    torch.cuda.synchronize()
    expected = (1,*config["output_features"]["action"]["shape"])
    if tuple(action.shape) != expected or not torch.isfinite(action).all().item():
        raise ValueError("reloaded ACT checkpoint produced invalid actions")
    atomic_json(args.output, {"passed":True,"optimizer_updates":int(step),
        "checkpoint_sha256":digest(args.checkpoint / "model.safetensors"),
        "action_shape":list(action.shape),"all_finite":True,
        "scope":"saved ACT model reload and inference; not exact resumed-training equivalence"})


if __name__ == "__main__":
    main()
