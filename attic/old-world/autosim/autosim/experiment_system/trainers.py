"""ACT/DP training on a canonical dataset, without benchmark task assumptions."""
from pathlib import Path

from autosim.research.common import atomic_json, digest


def train_canonical(runtime, contract, dataset, destination, policy, steps, seed):
    if policy not in {"act", "dp"} or steps < 1:
        raise ValueError("invalid trainer request")
    contract.assert_sources()
    command = [str(runtime.python), f"policy/{policy}/scripts/train.py", "--dataset-root", str(dataset),
               "--output-dir", str(destination / "train"), "--steps", str(steps),
               "--save-freq", str(steps), "--batch-size", "8" if policy == "act" else "2",
               "--num-workers", "2", "--seed", str(seed), "--log-freq", "10"]
    if policy == "act":
        path = destination / "observation_contract.json"
        atomic_json(path, {"cameras": list(contract.observation["cameras"]),
                          "state_dim": contract.observation["state_dim"], "action_dim": contract.action["dimension"]})
        command.extend(["--observation-contract", str(path)])
    else:
        command.extend(["--img-micro-bs", "2"])
    atomic_json(destination / "recipe.json", {"command": command, "task_signature": contract.signature,
                "dataset": str(dataset), "policy": policy, "steps": steps, "seed": seed})
    runtime.run(command, destination / "process", max(1800, steps * 5))
    checkpoint = destination / "train/checkpoints" / f"{steps:06d}" / "pretrained_model"
    return {"status": "completed", "checkpoint": str(checkpoint), "policy": policy,
            "verified_files": {str(checkpoint / name): digest(checkpoint / name) for name in ("config.json", "model.safetensors")},
            "reload_validated": False, "native_evaluation_validated": False}
