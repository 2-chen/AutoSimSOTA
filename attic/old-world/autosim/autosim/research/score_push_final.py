"""Frozen final-stage executor for the RoboSyn score-push campaign.

Only a mixed-data arm that passed the two-bank 20K screen may enter this
stage.  It compares an 80K continuation with an 80K from-scratch control,
selects on the same development banks, and uses the untouched confirmation
bank exactly once for the selected candidate and official checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import atomic_json, digest, now, read_json
from .ledger import compare
from .registry import load_task
from .runtime import Runtime


FINAL_UPDATES = 80_000
TRAINING_SEED = 1000
MIN_DEVELOPMENT_GAIN = .03
MIN_CONFIRMATION_GAIN = .03


def _context(workspace: Path, root: Path, task: str):
    workspace, root = workspace.resolve(), root.resolve()
    protocol = read_json(root / "campaign_protocol.json")
    if task not in protocol["tasks"]:
        raise ValueError(f"task outside frozen campaign: {task}")
    runtime = Runtime(workspace, root, gpu="0")
    return root, protocol["tasks"][task], runtime, load_task(runtime.repo, task)


def train_final(workspace: Path, root: Path, task: str) -> dict:
    root, row, runtime, spec = _context(workspace, root, task)
    screen = read_json(root / task / "screen_analysis.json")
    selected_arm = screen.get("selected_arm")
    if selected_arm not in {"mixed_proportional", "mixed_20pct"}:
        result = {"task": task, "status": "not_run", "reason": "no_mixed_arm_passed_20k_gate"}
        atomic_json(root / task / "final_training_summary.json", result)
        return result
    mixture = root / task / f"mixture_{selected_arm}.json"
    official_data = Path(row["official_dataset"])
    outputs = {
        "finetune_80k": root / task / f"screen_{selected_arm}",
        "scratch_80k": root / task / f"final_{selected_arm}_scratch_80k",
    }
    checkpoints = {
        "finetune_80k": runtime.train(
            spec, official_data, outputs["finetune_80k"], steps=FINAL_UPDATES,
            mixture=mixture, seed=TRAINING_SEED, resume=True),
        "scratch_80k": runtime.train(
            spec, official_data, outputs["scratch_80k"], steps=FINAL_UPDATES,
            mixture=mixture, seed=TRAINING_SEED),
    }
    result = {
        "task": task, "status": "completed", "selected_screen_arm": selected_arm,
        "mixture": str(mixture), "updates": FINAL_UPDATES, "seed": TRAINING_SEED,
        "checkpoints": {label: {"path": str(path),
                                "model_sha256": digest(path / "model.safetensors")}
                        for label, path in checkpoints.items()},
    }
    atomic_json(root / task / "final_training_summary.json", result)
    return result


def evaluate_final_development(workspace: Path, root: Path, task: str) -> dict:
    root, row, runtime, spec = _context(workspace, root, task)
    trained = read_json(root / task / "final_training_summary.json")
    if trained.get("status") != "completed":
        result = {"task": task, "status": "not_run", "reason": trained.get("reason")}
        atomic_json(root / task / "final_development_evaluation_summary.json", result)
        return result
    checkpoints = {label: Path(value["path"])
                   for label, value in trained["checkpoints"].items()}
    results = {}
    for repeat in (1, 2):
        # Reuse the already-completed official evaluation on the exact same
        # frozen bank; never rerun it merely to obtain a different draw.
        official_metrics = root / task / f"screen_eval_official_development_{repeat}/evaluation_metrics.json"
        if not official_metrics.is_file():
            runtime.evaluate(
                spec, Path(row["official_checkpoint"]), official_metrics.parent,
                episodes=100, master_seed=row["banks"][f"development_{repeat}"]["master"],
                purpose="development")
        results[f"official_development_{repeat}"] = read_json(official_metrics)["summary"]
        for label, checkpoint in checkpoints.items():
            destination = root / task / f"final_eval_{label}_development_{repeat}"
            metrics = destination / "evaluation_metrics.json"
            data = read_json(metrics) if metrics.is_file() else runtime.evaluate(
                spec, checkpoint, destination, episodes=100,
                master_seed=row["banks"][f"development_{repeat}"]["master"],
                purpose="development")
            results[f"{label}_development_{repeat}"] = data["summary"]
    result = {"task": task, "status": "completed", "results": results}
    atomic_json(root / task / "final_development_evaluation_summary.json", result)
    return result


def select_final(root: Path, task: str) -> dict:
    root = root.resolve()
    labels = ("official", "finetune_80k", "scratch_80k")

    def path(label: str, repeat: int) -> Path:
        prefix = "screen_eval_official" if label == "official" else f"final_eval_{label}"
        return root / task / f"{prefix}_development_{repeat}/evaluation_metrics.json"

    metrics = {(label, repeat): read_json(path(label, repeat))
               for label in labels for repeat in (1, 2)}
    means = {label: sum(metrics[label, repeat]["summary"]["success_rate"]
                        for repeat in (1, 2)) / 2 for label in labels}
    eligible = []
    comparisons = {}
    for label in labels[1:]:
        deltas = []
        for repeat in (1, 2):
            key = f"{label}_vs_official_development_{repeat}"
            comparisons[key] = compare(metrics[label, repeat], metrics["official", repeat])
            deltas.append(comparisons[key]["success_delta"])
        if means[label] - means["official"] >= MIN_DEVELOPMENT_GAIN and min(deltas) >= 0:
            eligible.append(label)
    # The tie break is fixed: continuation is cheaper than discarding the
    # shared 20K screen, so it wins an exact score tie.
    selected = max(eligible, key=lambda label: (means[label], label == "finetune_80k")) if eligible else None
    result = {
        "task": task, "status": "completed", "means": means,
        "comparisons": comparisons, "eligible_for_confirmation": eligible,
        "selected_candidate": selected,
        "selection_rule": (
            "80K mean >= official mean +3pp and nonnegative paired success delta on both "
            "frozen development banks; maximize mean; exact tie favors 20K-to-80K continuation"),
    }
    atomic_json(root / task / "final_selection.json", result)
    return result


def confirm(workspace: Path, root: Path, task: str) -> dict:
    root, row, runtime, spec = _context(workspace, root, task)
    selection = read_json(root / task / "final_selection.json")
    selected = selection.get("selected_candidate")
    if selected is None:
        result = {"task": task, "status": "not_run", "reason": "no_80k_candidate_passed_development_gate",
                  "deployment": "official"}
        atomic_json(root / task / "confirmation_summary.json", result)
        return result
    trained = read_json(root / task / "final_training_summary.json")
    checkpoints = {"official": Path(row["official_checkpoint"]),
                   selected: Path(trained["checkpoints"][selected]["path"])}
    metrics = {}
    for label, checkpoint in checkpoints.items():
        destination = root / task / f"confirmation_{label}"
        metrics[label] = runtime.evaluate(
            spec, checkpoint, destination, episodes=200,
            master_seed=row["banks"]["confirmation"]["master"], purpose="confirmation")
    paired = compare(metrics[selected], metrics["official"])
    candidate_rate = metrics[selected]["summary"]["success_rate"]
    official_rate = metrics["official"]["summary"]["success_rate"]
    accepted = candidate_rate - official_rate >= MIN_CONFIRMATION_GAIN and paired["success_delta"] >= MIN_CONFIRMATION_GAIN
    result = {
        "task": task, "status": "completed", "completed_at": now(),
        "selected_candidate": selected,
        "official_summary": metrics["official"]["summary"],
        "candidate_summary": metrics[selected]["summary"],
        "paired_comparison": paired,
        "accepted": accepted, "deployment": selected if accepted else "official",
        "acceptance_rule": "candidate exceeds official by at least 3pp on the untouched paired 200-episode confirmation bank",
        "claim_scope": "local RoboSyn simulation under the frozen evaluator; not an official leaderboard score",
    }
    atomic_json(root / task / "confirmation_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate-development", "select", "confirm"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    commands = {"train": train_final, "evaluate-development": evaluate_final_development,
                "confirm": confirm}
    result = select_final(args.root, args.task) if args.command == "select" else commands[args.command](
        args.workspace, args.root, args.task)
    print(result)


if __name__ == "__main__":
    main()
