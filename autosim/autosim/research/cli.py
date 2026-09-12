"""Unified, resumable single-GPU task suite entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from autosim.robosyn_mvp import gpu_lock
from .common import atomic_json, event, exclusive, now, read_json, redact
from .registry import TASK_IDS, inventory, load_task
from .runtime import Runtime


def smoke(runtime: Runtime, tasks: list[str], *, episodes: int, steps: int,
          collect_only: bool = False, retry_failed: bool = False) -> dict:
    status_path = runtime.output / "smoke_status.json"
    with exclusive(runtime.output / "suite.lock"), gpu_lock(runtime.gpu):
        state = read_json(status_path) if status_path.exists() else {"created_at": now(), "tasks": {}}
        for name in tasks:
            row = state["tasks"].setdefault(name, {"status": "pending", "improvement": "not_evaluated"})
            native_collection_retry = False
            if row["status"] == "failed" and row.get("stage") == "collection" and int(row.get("attempt", 1)) == 1:
                process_path = runtime.output / "smoke" / name / "collection/process/process.json"
                if process_path.exists():
                    native_collection_retry = read_json(process_path).get("returncode") in {-11, -6}
            # This is training data generation, never a policy score retry.
            # Count/preserve the crashed attempt and allow only one native-crash
            # retry; no assumption that the crash necessarily preceded reset.
            if row["status"] == "failed" and (retry_failed or native_collection_retry):
                retry_checkpoint = row.get("checkpoint") if row.get("stage") == "evaluation" else None
                prior_dataset, prior_frames = row.get("dataset"), row.get("frames")
                history = list(row.get("history", [])) + [{k: v for k, v in row.items() if k != "history"}]
                row = {"status": "pending", "improvement": "not_evaluated", "history": history,
                       "attempt": int(row.get("attempt", 1)) + 1,
                       "retry_reason": "explicit_retry" if retry_failed else "one_training_collection_native_crash_retry"}
                if retry_checkpoint and Path(retry_checkpoint, "model.safetensors").is_file():
                    row.update(retry_checkpoint=retry_checkpoint, dataset=prior_dataset, frames=prior_frames)
                state["tasks"][name] = row
            if row["status"] in {"passed", "failed"}:
                continue
            try:
                spec = load_task(runtime.repo, name)
                destination = runtime.output / "smoke" / name
                if int(row.get("attempt", 1)) > 1:
                    destination /= f"attempt_{row['attempt']}"
                destination.mkdir(parents=True, exist_ok=True)
                row.update(status="running", stage="collection", started_at=now())
                atomic_json(status_path, state)
                print(f"[{name}] collecting {episodes} expert episodes", flush=True)
                root = (Path(row["dataset"]) if row.get("retry_checkpoint") else
                        runtime.collect(spec, destination / "collection", episodes=episodes,
                                        master_seed=50_000_000 + list(TASK_IDS).index(name) * 100_000))
                row.update(stage="data_audit", dataset=str(root))
                atomic_json(status_path, state)
                audit = runtime.prepare_data(spec, root, destination)
                row.update(stage="training", frames=audit["dataset"]["total_frames"])
                atomic_json(status_path, state)
                if collect_only:
                    row.update(status="collected", stage="data_audit_complete", finished_at=now())
                    atomic_json(status_path, state)
                    print(f"[{name}] collection and data audit passed", flush=True)
                    continue
                print(f"[{name}] training {steps} updates", flush=True)
                checkpoint = (Path(row["retry_checkpoint"]) if row.get("retry_checkpoint") else
                              runtime.train(spec, root, destination / "candidate", steps=steps))
                row.update(stage="evaluation", checkpoint=str(checkpoint))
                atomic_json(status_path, state)
                result = runtime.evaluate(spec, checkpoint, destination / "evaluation", episodes=3,
                                          master_seed=60_000_000 + list(TASK_IDS).index(name) * 100_000,
                                          purpose="smoke")
                row.update(status="passed", stage="complete", summary=result["summary"], finished_at=now())
                print(f"[{name}] pipeline passed (not a performance claim)", flush=True)
            except Exception as exc:
                row.update(status="failed", error=redact(f"{type(exc).__name__}: {exc}"), finished_at=now())
                print(f"[{name}] {row['stage']} failed: {row['error']}", flush=True)
            atomic_json(status_path, state)
            event(runtime.output / "events.jsonl", "smoke_task_finished", task=name, status=row["status"])
    state["all_tasks_passed"] = len(state["tasks"]) == len(TASK_IDS) and all(
        r["status"] == "passed" for r in state["tasks"].values())
    atomic_json(status_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "collect", "smoke", "official-smoke", "research", "pilot", "final", "status"])
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_IDS), default=list(TASK_IDS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Preserve failed attempts and start new, separately logged attempts")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--allow-llm", action="store_true")
    parser.add_argument("--train-seed", type=int, default=1000)
    args = parser.parse_args()
    # research/cli.py -> autosim package -> autosim repository -> workspace
    workspace = args.workspace.absolute()
    output = (args.output or workspace / "autosim/output/robosyn_general_20260905").absolute()
    output.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(workspace, output, args.gpu)
    if args.command == "official-smoke":
        from .official_smoke import official_smoke
        result = official_smoke(runtime, args.tasks, steps=args.steps)
        return int(any(result["tasks"][task]["status"] != "passed" for task in args.tasks))
    if args.command == "final":
        from .finalize import finalize
        result = finalize(runtime, args.tasks, train_seed=args.train_seed)
        return int(any(row["status"] != "completed" for row in result["tasks"].values()))
    if args.command == "inventory":
        rows = inventory(runtime.repo)
        atomic_json(output / "task_inventory.json", rows)
        for row in rows:
            print(row["task"], row["contract_status"], "data=", row.get("dataset_available"),
                  "checkpoint=", row.get("checkpoint_available"))
        return int(any(r["contract_status"] != "passed" for r in rows))
    if args.command in {"collect", "smoke"}:
        if args.episodes < 1 or args.steps < 1:
            parser.error("episodes and steps must be positive")
        result = smoke(runtime, args.tasks, episodes=args.episodes, steps=args.steps,
                       collect_only=args.command == "collect", retry_failed=args.retry_failed)
        return int(any(result["tasks"][t]["status"] not in {"passed", "collected"} for t in args.tasks))
    if args.command in {"research", "pilot"}:
        from .controller import ResearchConfig, ResearchController

        kwargs = dict(rounds=args.rounds, allow_llm=args.allow_llm, train_seed=args.train_seed)
        if args.command == "pilot":
            kwargs.update(pilot=True, screen_steps=200, final_steps=400,
                          development_episodes=3, confirmation_episodes=3,
                          collect_per_round=5, hours_per_task=3)
        result = ResearchController(runtime, ResearchConfig(**kwargs)).run(args.tasks)
        return int(any(row["status"] != "completed_development" for row in result.values()))
    path = output / "smoke_status.json"
    print(path.read_text() if path.exists() else "no smoke run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
