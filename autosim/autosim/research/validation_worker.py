"""Bounded GPU workload used by the SCO compute matrix, never a research loop."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    import torch
    from .common import atomic_json
    from .devices import normalize_uuid
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    actual = str(props.uuid)
    assert normalize_uuid(actual) == normalize_uuid(args.uuid), (actual, args.uuid)
    torch.manual_seed(args.seed)
    a = torch.randn(1024, 1024, device="cuda")
    for _ in range(5):
        result = a @ a
    torch.cuda.synchronize()
    start = time.time()
    steps = 0
    while (steps < args.steps) if args.steps else (time.time() - start < args.seconds):
        result = a @ a
        torch.cuda.synchronize()
        steps += 1
    assert bool(torch.isfinite(result).all())
    if args.training:
        parameter = torch.randn(128,128,device="cuda",requires_grad=True)
        optimizer = torch.optim.SGD([parameter],lr=.01)
        for _ in range(3):
            optimizer.zero_grad(); loss=parameter.square().mean(); loss.backward(); optimizer.step()
        assert bool(torch.isfinite(parameter.grad).all())
    atomic_json(args.output, {"status": "passed", "uuid": args.uuid, "actual_uuid": actual,
                              "started": start, "ended": time.time(), "steps": steps,
                              "pid": os.getpid(), "device": props.name,
                              "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                              "torch_version": torch.__version__, "omp_threads":torch.get_num_threads()})


if __name__ == "__main__":
    main()
