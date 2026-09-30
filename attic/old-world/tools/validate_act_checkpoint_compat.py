"""CPU inference evidence comparing embedded ACT normalization across runtimes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", choices=("reference", "fixed"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(713)
    sys.path.insert(0, str(args.repo))
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from policy.act.checkpoint_compat import CheckpointACTPolicy

    config = PreTrainedConfig.from_pretrained(args.checkpoint, cli_overrides=["--device=cpu"])
    cls = CheckpointACTPolicy if args.implementation == "fixed" else ACTPolicy
    policy = cls.from_pretrained(args.checkpoint, config=config)
    batch = {}
    with safe_open(args.checkpoint / "model.safetensors", framework="pt") as tensors:
        for name, feature in policy.config.input_features.items():
            if name != "observation.state" and not name.startswith("observation.images."):
                # Match the real RPC/native adapter's observation-only input.
                # Official metadata also lists qvel/qf, which ACT never consumes.
                continue
            prefix = "normalize_inputs.buffer_" + name.replace(".", "_")
            if name.startswith("observation.images."):
                # Nonconstant RGB; dimensions may vary without changing statistics.
                batch[name] = torch.rand((1, feature.shape[0], 64, 80))
            else:
                mean, std = tensors.get_tensor(prefix + ".mean"), tensors.get_tensor(prefix + ".std")
                batch[name] = (mean + std * .15).unsqueeze(0)
    before = {k: v.clone() for k, v in batch.items()}
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
        policy.reset()
        selected = torch.stack([policy.select_action(batch) for _ in range(3)], dim=1)
        policy.reset()
        first_after_reset = policy.select_action(batch)
    assert torch.isfinite(chunk).all()
    assert all(torch.equal(v, before[k]) for k, v in batch.items())
    torch.testing.assert_close(selected, chunk[:, :3], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(first_after_reset, chunk[:, 0], rtol=1e-6, atol=1e-6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "implementation": args.implementation, "interpreter": sys.executable,
        "runtime_has_inline_normalization": hasattr(policy, "normalize_inputs"),
        "compatibility_active": getattr(policy, "_restore_inline_normalization", False),
        "checkpoint_sha256": hashlib.sha256((args.checkpoint / "model.safetensors").read_bytes()).hexdigest(),
        "input_sha256": {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in batch.items()},
        "chunk": chunk.tolist(), "first_action": chunk[0, 0].tolist(),
        "queue_and_reset_passed": True, "inputs_unchanged": True,
    }, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "compatibility_active": getattr(policy, "_restore_inline_normalization", False),
                      "first_action": chunk[0, 0].tolist()}))


if __name__ == "__main__":
    main()
