"""Official-data-only fallback for tasks whose expert collection gate closes."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import atomic_json, digest, now, read_json
from .ledger import compare
from .registry import load_task
from .runtime import Runtime


SCREEN_UPDATES = 20_000
FINAL_UPDATES = 80_000
TRAINING_SEED = 1000
MIN_GAIN = .03


def _context(workspace: Path, root: Path, task: str):
    workspace, root = workspace.resolve(), root.resolve()
    protocol = read_json(root / "campaign_protocol.json")
    if task not in protocol["tasks"]:
        raise ValueError(f"task outside frozen campaign: {task}")
    capability = read_json(root / task / "capability_summary.json")
    if capability.get("decision") == "scale":
        raise ValueError("official-data fallback is forbidden after collection passed the scale gate")
    runtime = Runtime(workspace, root, gpu="0")
    return root, protocol["tasks"][task], runtime, load_task(runtime.repo, task)


def train(workspace: Path, root: Path, task: str, *, final: bool = False) -> dict:
    root, row, runtime, spec = _context(workspace, root, task)
    output = root / task / "fallback_official_continue"
    updates = FINAL_UPDATES if final else SCREEN_UPDATES
    checkpoint = runtime.train(
        spec, Path(row["official_dataset"]), output, steps=updates,
        seed=TRAINING_SEED, resume=final,
        pretrained=None if final else Path(row["official_checkpoint"]))
    result = {"task": task, "status": "completed", "updates": updates,
              "initialization": "resume_20k" if final else "official_published_checkpoint",
              "new_data_used": False, "checkpoint": str(checkpoint),
              "model_sha256": digest(checkpoint / "model.safetensors")}
    atomic_json(root / task / f"fallback_training_{updates}.json", result)
    return result


def evaluate(workspace: Path, root: Path, task: str, *, final: bool = False) -> dict:
    root, row, runtime, spec = _context(workspace, root, task)
    updates = FINAL_UPDATES if final else SCREEN_UPDATES
    checkpoint = Path(read_json(root / task / f"fallback_training_{updates}.json")["checkpoint"])
    label = f"fallback_official_continue_{updates}"
    results = {}
    for repeat in (1, 2):
        master = row["banks"][f"development_{repeat}"]["master"]
        checkpoints = {"official": Path(row["official_checkpoint"]), label: checkpoint}
        for name, model in checkpoints.items():
            destination = root / task / f"fallback_eval_{name}_development_{repeat}"
            metrics = destination / "evaluation_metrics.json"
            data = read_json(metrics) if metrics.is_file() else runtime.evaluate(
                spec, model, destination, episodes=100, master_seed=master,
                purpose="development")
            results[f"{name}_development_{repeat}"] = data["summary"]
    result = {"task": task, "status": "completed", "updates": updates,
              "new_data_used": False, "results": results}
    atomic_json(root / task / f"fallback_evaluation_{updates}.json", result)
    return result


def select(root: Path, task: str, *, final: bool = False) -> dict:
    root = root.resolve()
    updates = FINAL_UPDATES if final else SCREEN_UPDATES
    label = f"fallback_official_continue_{updates}"
    metrics = {}
    for repeat in (1, 2):
        for name in ("official", label):
            metrics[name, repeat] = read_json(
                root / task / f"fallback_eval_{name}_development_{repeat}/evaluation_metrics.json")
    means = {name: sum(metrics[name, repeat]["summary"]["success_rate"]
                       for repeat in (1, 2)) / 2 for name in ("official", label)}
    comparisons = {f"development_{repeat}": compare(metrics[label, repeat], metrics["official", repeat])
                   for repeat in (1, 2)}
    deltas = [comparisons[f"development_{repeat}"]["success_delta"] for repeat in (1, 2)]
    selected = means[label] - means["official"] >= MIN_GAIN and min(deltas) >= 0
    result = {"task": task, "status": "completed", "updates": updates,
              "means": means, "comparisons": comparisons, "selected": selected,
              "new_data_used": False,
              "selection_rule": "mean >= official +3pp and nonnegative paired delta on both frozen development banks"}
    atomic_json(root / task / f"fallback_selection_{updates}.json", result)
    return result


def confirm(workspace: Path, root: Path, task: str) -> dict:
    root, row, runtime, spec = _context(workspace, root, task)
    selected = read_json(root / task / f"fallback_selection_{FINAL_UPDATES}.json")
    if not selected["selected"]:
        result = {"task": task, "status": "not_run", "deployment": "official",
                  "reason": "80K official-data continuation failed development gate",
                  "new_data_used": False}
        atomic_json(root / task / "fallback_confirmation.json", result)
        return result
    candidate = Path(read_json(root / task / f"fallback_training_{FINAL_UPDATES}.json")["checkpoint"])
    metrics = {}
    for name, checkpoint in {"official": Path(row["official_checkpoint"]),
                             "official_continue_80k": candidate}.items():
        destination = root / task / f"fallback_confirmation_{name}"
        metrics[name] = runtime.evaluate(
            spec, checkpoint, destination, episodes=200,
            master_seed=row["banks"]["confirmation"]["master"], purpose="confirmation")
    paired = compare(metrics["official_continue_80k"], metrics["official"])
    accepted = paired["success_delta"] >= MIN_GAIN
    result = {"task": task, "status": "completed", "completed_at": now(),
              "official_summary": metrics["official"]["summary"],
              "candidate_summary": metrics["official_continue_80k"]["summary"],
              "paired_comparison": paired, "accepted": accepted,
              "deployment": "official_continue_80k" if accepted else "official",
              "new_data_used": False,
              "acceptance_rule": "at least +3pp paired improvement on untouched 200-episode confirmation bank",
              "claim_scope": "local RoboSyn simulation, not an official leaderboard score"}
    atomic_json(root / task / "fallback_confirmation.json", result)
    return result


def run(workspace: Path, root: Path, task: str) -> dict:
    screen_train = train(workspace, root, task)
    evaluate(workspace, root, task)
    screen_select = select(root, task)
    if not screen_select["selected"]:
        return {"task": task, "status": "completed_no_improvement", "screen": screen_select}
    final_train = train(workspace, root, task, final=True)
    evaluate(workspace, root, task, final=True)
    final_select = select(root, task, final=True)
    confirmation = confirm(workspace, root, task) if final_select["selected"] else {
        "task": task, "status": "not_run", "deployment": "official",
        "reason": "80K official-data continuation failed development gate", "new_data_used": False}
    if confirmation["status"] == "not_run":
        atomic_json(root / task / "fallback_confirmation.json", confirmation)
    return {"task": task, "status": "completed", "screen_training": screen_train,
            "final_training": final_train, "screen": screen_select,
            "final": final_select, "confirmation": confirmation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    print(run(args.workspace, args.root, args.task))


if __name__ == "__main__":
    main()
