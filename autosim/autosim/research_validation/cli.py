"""Validation entry points; they do not mutate the running search core."""
import argparse
from pathlib import Path

from autosim.research.runtime import Runtime
from autosim.research.registry import TASK_IDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["lock-repeats", "train-repeats", "determinism", "compare-initializations"])
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", choices=list(TASK_IDS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    args = parser.parse_args()
    runtime = Runtime(args.workspace, args.output or args.workspace / "autosim/output/robosyn_general_20260905")
    if args.command == "compare-initializations":
        from .determinism import audit_pair
        if not args.first or not args.second:
            parser.error("--first and --second are required")
        print(audit_pair(args.first, args.second))
        return
    if not args.task:
        parser.error("--task is required")
    if args.command == "determinism":
        from .determinism import run_audit
        if not args.checkpoint:
            parser.error("--checkpoint is required")
        print(run_audit(runtime, args.task, args.checkpoint))
    else:
        from .repeats import lock_recipes, run_repeats
        print(lock_recipes(runtime, args.task)[0] if args.command == "lock-repeats" else run_repeats(runtime, args.task))


if __name__ == "__main__":
    main()
