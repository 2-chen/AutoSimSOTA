"""Bounded, observation-only replay of original eight-episode failure blocks."""
from __future__ import annotations

import argparse
import os
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from autosim.research.common import atomic_json, digest, now, read_json
from autosim.research.devices import discover, identity_is_safe, select
from autosim.research.runtime import Runtime, retryable_evaluation_startup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=2100)
    parser.add_argument("--all-blocks", action="store_true", help="replay all five blocks for both checkpoints")
    parser.add_argument("--two-blocks", action="store_true", help="diagnostic only: split each 40-seed bank into two 20-episode workers")
    parser.add_argument("--require-clean", action="store_true", help="fail unless every planned episode completes")
    args = parser.parse_args()
    root = args.output.absolute()
    root.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).absolute().parents[1]
    inventory = discover(disk_path=root)
    atomic_json(root / "inventory.json", inventory)
    devices = [d for d in inventory["gpus"] if d["index"] in inventory["allowed"]]
    if len(devices) != 4 or not all("5090" in d["model"] for d in devices):
        raise RuntimeError("four RTX 5090 devices required for the four independent replay blocks")
    safe, reason = identity_is_safe(inventory)
    if not safe:
        raise RuntimeError(f"identity addressing refused: {reason}")
    started = time.monotonic()
    runtime = Runtime(project, root, plan={"unified_scheduler": True},
                      deadline=started + args.seconds)
    candidate = args.source_run / "rounds/round_1/candidate/train/checkpoints/020000/pretrained_model"
    official = project / "RoboSynChallenge/checkpoints/ACT_sim_click_bell"
    request = read_json(args.source_run / "evaluations/round_1_candidate_development/evaluation_request.json")
    if digest(candidate / "model.safetensors") != request["weight_sha256"]:
        raise RuntimeError("historical candidate checkpoint changed")
    block_plan = read_json(args.source_run / "evaluations/round_1_candidate_development/shard_plan.json")
    original_block_plan = block_plan
    if args.two_blocks:
        if not args.all_blocks or sum(block["size"] for block in block_plan["blocks"]) != 40:
            raise ValueError("two-block diagnostic requires both complete 40-episode banks")
        block_plan = {"count": 2, "blocks": [{"offset": 0, "size": 20}, {"offset": 20, "size": 20}]}
    jobs = [("official_block_02", 2, official, devices[0]),
            ("candidate_block_01", 1, candidate, devices[1]),
            ("candidate_block_02", 2, candidate, devices[2]),
            ("candidate_block_03", 3, candidate, devices[3])]
    if args.all_blocks:
        jobs = [(f"{label}_block_{block:02d}", block, cp, None)
                for label, cp in (("candidate", candidate), ("official", official))
                for block in range(block_plan["count"])]
    leases = Queue()
    for device in devices:
        leases.put(device)
    atomic_json(root / "replay_manifest.json", {
        "started": now(), "full_autoresearch": False, "diagnostic_only": True,
        "checkpoint_sha256": request["weight_sha256"], "master_seed": request["master_seed"],
        "original_block_plan": original_block_plan, "replay_block_plan": block_plan,
        "max_seconds": args.seconds,
        "require_clean": args.require_clean, "all_blocks": args.all_blocks,
        "jobs": [{"label": label, "block": block, "checkpoint": str(cp), "gpu": gpu["index"] if gpu else "exclusive_lease"}
                 for label, block, cp, gpu in jobs]})

    def replay(label, block_index, checkpoint, device):
        output = root / label
        output.mkdir(parents=True, exist_ok=True)
        bound = runtime.for_job(select(device, "identity"), job=label, output=output)
        size = block_plan["blocks"][block_index]["size"]
        for attempt in range(1, 3):
            active = output if attempt == 1 else output / "startup_attempt_2"
            command = [str(runtime.python), "-m", "autosim.research.evaluation", "--task", "click_bell",
                "--checkpoint", str(checkpoint), "--output", str(active), "--episodes", str(size),
                "--seed", str(request["master_seed"]), "--purpose", "development", "--policy", "act",
                "--seed-offset", str(block_plan["blocks"][block_index]["offset"]),
                "--shard-index", str(block_index), "--shard-count", str(block_plan["count"]),
                *bound.evaluation_device_flags()]
            try:
                bound.run(command, active / "process", args.seconds, evaluation=True)
                metrics = read_json(active / "evaluation_metrics.json")
                if args.require_clean and (active / "nonfinite_dynamics.json").exists():
                    raise RuntimeError("episode ended with non-finite dynamics")
                result = {"status": "completed", "episodes": len(metrics["episodes"]),
                          "artifact": str(active), "diagnostic_only": True}
                break
            except Exception as exc:
                if attempt == 1 and retryable_evaluation_startup(active):
                    atomic_json(output / "startup_retry.json", {"reason": "native pre-reset crash", "next": 2})
                    continue
                failure = active / "worker_failure.json"
                result = {"status": "failed", "artifact": str(active), "error_type": type(exc).__name__,
                          "worker_failure": read_json(failure) if failure.exists() else None,
                          "nonfinite_dynamics_recorded": (active / "nonfinite_dynamics.json").is_file()}
                break
        atomic_json(output / "replay_result.json", result)
        return result

    def leased_replay(label, block, checkpoint, device):
        if device is not None:
            return replay(label, block, checkpoint, device)
        device = leases.get()
        try:
            return replay(label, block, checkpoint, device)
        finally:
            leases.put(device)

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(leased_replay, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            atomic_json(root / "progress.json", results)
    clean = all(result["status"] == "completed" and result["episodes"] == block_plan["blocks"][block]["size"]
                for label, block, _, _ in jobs for result in [results[label]])
    atomic_json(root / "summary.json", {"finished": now(), "jobs": results,
        "container_seconds": time.monotonic() - started, "score_claim": False,
        "all_planned_episodes_completed": clean, "require_clean": args.require_clean})
    # Reproducing the fault is an expected diagnostic outcome, not a benchmark pass.
    return 0 if len(results) == len(jobs) and (clean or not args.require_clean) else 1


if __name__ == "__main__":
    raise SystemExit(main())
