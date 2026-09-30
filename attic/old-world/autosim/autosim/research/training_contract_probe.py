"""Measured NCCL/global-batch and checkpoint equivalence; not ACT certification."""
from __future__ import annotations
import argparse
import copy
import os
from datetime import timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .common import atomic_json
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    args.output.mkdir(parents=True, exist_ok=True)
    def progress(stage):
        atomic_json(args.output / f"rank_{rank}_progress.json", {"stage":stage,"rank":rank,"world_size":world})
        print(f"rank={rank} stage={stage}",flush=True)
    progress("importing_torch")
    import torch
    import torch.distributed as dist
    from .common import atomic_json
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    progress("cuda_ready")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("nccl",timeout=timedelta(seconds=60))
    progress("process_group_ready")
    torch.manual_seed(781)
    base = torch.nn.Linear(8, 4).cuda()
    model = copy.deepcopy(base)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    progress("model_ready")
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    reference = copy.deepcopy(base)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01, momentum=.9)
    generator = torch.Generator().manual_seed(823)
    batches = [(torch.randn(32, 8, generator=generator).cuda(), torch.randn(32, 4, generator=generator).cuda())
               for _ in range(10)]
    checkpoint = args.output / f"rank_{rank}.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    for step, (inputs, targets) in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs.chunk(world)[rank]) - targets.chunk(world)[rank]).square().mean()
        loss.backward(); optimizer.step()
        progress(f"update_{step+1}")
        ref_optimizer.zero_grad(set_to_none=True)
        ref_loss = (reference(inputs)-targets).square().mean()
        ref_loss.backward(); ref_optimizer.step()
        if step == 4:
            raw = model.module if world > 1 else model
            torch.save({"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                        "next_batch": step+1, "updates": step+1}, checkpoint)
            # Reload into a new model/optimizer, then continue the same frozen batch stream.
            restored = torch.load(checkpoint, weights_only=True)
            fresh = copy.deepcopy(base); fresh.load_state_dict(restored["model"])
            model = (torch.nn.parallel.DistributedDataParallel(fresh, device_ids=[local]) if world > 1 else fresh)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            optimizer.load_state_dict(restored["optimizer"])
            torch.set_rng_state(restored["rng_cpu"]); torch.cuda.set_rng_state(restored["rng_cuda"])
            assert restored["next_batch"] == restored["updates"] == 5
    actual = model.module if world > 1 else model
    delta = max((a-b).abs().max().item() for a,b in zip(actual.parameters(), reference.parameters()))
    assert delta < 1e-6, delta
    atomic_json(args.output / f"rank_{rank}.json", {"passed": True, "rank": rank, "world_size": world,
        "global_batch": 32, "micro_batch": 32//world, "updates": 10, "resume_update": 5,
        "parameter_max_abs_error": delta, "precision": "fp32", "transport": "nccl" if world>1 else "local",
        "scope": "linear model transport/global-batch/checkpoint probe; ACT remains separately gated"})
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
