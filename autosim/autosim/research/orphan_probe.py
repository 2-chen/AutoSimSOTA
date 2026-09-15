"""Kill an owned dispatcher after CUDA allocation and verify its worker is reaped."""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .common import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dispatcher", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        import torch
        tensor = torch.ones(1024*1024, device="cuda")
        torch.cuda.synchronize()
        atomic_json(args.output / "cuda_ready.json", {"pid":os.getpid(),"memory":torch.cuda.memory_allocated()})
        while True:
            tensor.add_(1); torch.cuda.synchronize(); time.sleep(.1)
    if args.dispatcher:
        from .common import run_command
        run_command([sys.executable,"-m","autosim.research.orphan_probe","--worker","--output",str(args.output)],
            cwd=args.output,env=os.environ.copy(),output=args.output / "child",timeout=120,
            metadata={"job":"owned_orphan_probe"})
        return 0
    dispatcher = subprocess.Popen([sys.executable,"-m","autosim.research.orphan_probe","--dispatcher",
                                   "--output",str(args.output)])
    try:
        deadline = time.monotonic()+75
        while not (args.output / "cuda_ready.json").exists():
            if dispatcher.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError("owned worker did not reach CUDA readiness")
            time.sleep(.1)
        worker = read_json(args.output / "cuda_ready.json")
        assert worker["memory"] > 0
        os.kill(dispatcher.pid, signal.SIGKILL); dispatcher.wait(timeout=10)
        deadline = time.monotonic()+20
        def running(pid):
            try:
                status = Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[0]
                return status != "Z"
            except FileNotFoundError:
                return False
        while running(worker["pid"]):
            if time.monotonic()>deadline:
                raise RuntimeError("orphan GPU worker survived dispatcher death")
            time.sleep(.1)
        atomic_json(args.output / "result.json", {"passed":True,"killed_owned_dispatcher":dispatcher.pid,
            "gpu_worker":worker["pid"],"allocated_bytes_before_kill":worker["memory"],
            "worker_dead_or_zombie_after_kill":True,"scope":"real CUDA process-group parent-death cleanup"})
        return 0
    finally:
        if dispatcher.poll() is None:
            dispatcher.terminate()
            try: dispatcher.wait(timeout=15)
            except subprocess.TimeoutExpired: dispatcher.kill(); dispatcher.wait()


if __name__ == "__main__":
    raise SystemExit(main())
