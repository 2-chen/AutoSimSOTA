"""Later stages for the frozen RoboSyn score-push campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from autosim.experiment_validation.dataset_video_integrity_v1 import audit_dataset

from .common import atomic_json, digest, now, read_json
from .ledger import compare
from .registry import load_task
from .runtime import Runtime


SCREEN_ARMS = ("official_continue", "mixed_proportional", "mixed_20pct")


def _context(workspace: Path, root: Path, task: str):
    workspace, root = workspace.resolve(), root.resolve()
    protocol = read_json(root / "campaign_protocol.json")
    if task not in protocol["tasks"]:
        raise ValueError(f"task outside frozen campaign: {task}")
    runtime = Runtime(workspace, root, gpu="0")
    return workspace, root, protocol, protocol["tasks"][task], runtime, load_task(runtime.repo, task)


def _audit_collected(runtime: Runtime, spec, destination: Path, bounded: dict) -> Path | None:
    accepted = int(bounded["accepted_episodes"])
    if not bounded["dataset_root"]:
        return None
    data_root = Path(bounded["dataset_root"])
    data = runtime.prepare_data(spec, data_root, destination)
    videos = audit_dataset(
        data_root, expected_episodes=accepted,
        expected_video_keys=[f"observation.images.{camera}" for camera in spec.cameras])
    atomic_json(destination / "video_integrity.json", videos)
    if not data["passed"] or not videos["passed"]:
        raise RuntimeError(f"collected data admission failed: {destination}")
    return data_root


def production(workspace: Path, root: Path, task: str) -> dict:
    _, root, _, row, runtime, spec = _context(workspace, root, task)
    capability = read_json(root / task / "capability_summary.json")
    if capability["decision"] != "scale":
        result = {"task": task, "status": "not_run", "reason": capability["decision"]}
        atomic_json(root / task / "production_summary.json", result)
        return result
    destination = root / task / "production_collection"
    bounded = runtime.collect_bounded(
        spec, destination, attempt_budget=int(row["production_attempts"]),
        target_episodes=int(row["production_target"]),
        master_seed=row["banks"]["production"]["master"], profile="full_random",
        timeout=max(14400, int(row["production_attempts"]) * 180))
    data_root = _audit_collected(runtime, spec, destination, bounded)
    result = {"task": task, "status": "completed", "attempts": bounded["attempts_consumed"],
              "accepted": bounded["accepted_episodes"],
              "yield": bounded["accepted_episodes"] / bounded["attempts_consumed"],
              "capability_state": bounded["capability_state"],
              "dataset_root": str(data_root) if data_root else None}
    atomic_json(root / task / "production_summary.json", result)
    return result


def _entry(path: Path, profile: str, source_kind: str) -> dict:
    info = read_json(path / "meta/info.json")
    return {"root": str(path.resolve()), "profile": profile, "source_kind": source_kind,
            "episode_count": int(info["total_episodes"]), "frame_count": int(info["total_frames"]),
            "info_sha256": digest(path / "meta/info.json")}


def mixtures(workspace: Path, root: Path, task: str) -> dict:
    _, root, _, row, _, _ = _context(workspace, root, task)
    official = Path(row["official_dataset"])
    capability = read_json(root / task / "capability_summary.json")
    production_row = read_json(root / task / "production_summary.json")
    new_roots = [Path(value) for value in
                 (capability.get("dataset_root"), production_row.get("dataset_root")) if value]
    if not new_roots:
        raise RuntimeError(f"no admitted new dataset for {task}")
    entries = [_entry(official, "official", "official_full_random")]
    entries.extend(_entry(path, "new", "self_collected_full_random") for path in new_roots)
    total_episodes = sum(item["episode_count"] for item in entries)
    total_frames = sum(item["frame_count"] for item in entries)
    common = {"schema_version": 1, "kind": "robosyn_score_push_training_mixture",
              "task": task, "total_episodes": total_episodes, "total_frames": total_frames}
    manifests = {
        "official_continue": {**common, "datasets": entries[:1],
                              "total_episodes": entries[0]["episode_count"],
                              "total_frames": entries[0]["frame_count"],
                              "sampling": "proportional_to_frames"},
        "mixed_proportional": {**common, "datasets": entries,
                               "sampling": "proportional_to_frames"},
        "mixed_20pct": {**common, "datasets": entries, "sampling": {
            "strategy": "stratified_phase", "profile_masses": {"official": .8, "new": .2},
            "phase_bins": [{"name": "all", "start": 0, "end": 1.01, "weight": 1}],
            "horizon_weighting": {"mode": "none"}}},
    }
    paths = {}
    for arm, value in manifests.items():
        path = root / task / f"mixture_{arm}.json"
        if path.is_file() and read_json(path) != value:
            raise RuntimeError(f"refusing to redefine mixture: {path}")
        atomic_json(path, value)
        paths[arm] = str(path)
    result = {"task": task, "created_at": now(), "manifests": paths,
              "new_episodes": sum(item["episode_count"] for item in entries[1:]),
              "new_frames": sum(item["frame_count"] for item in entries[1:])}
    atomic_json(root / task / "mixture_summary.json", result)
    return result


def train_screen(workspace: Path, root: Path, task: str) -> dict:
    _, root, protocol, row, runtime, spec = _context(workspace, root, task)
    official_data, parent = Path(row["official_dataset"]), Path(row["official_checkpoint"])
    summary = read_json(root / task / "mixture_summary.json")
    checkpoints = {}
    for arm in SCREEN_ARMS:
        output = root / task / f"screen_{arm}"
        checkpoint = runtime.train(
            spec, official_data, output, steps=int(protocol["screen_training"]["updates"]),
            mixture=Path(summary["manifests"][arm]), seed=int(protocol["screen_training"]["seed"]),
            pretrained=parent)
        checkpoints[arm] = {"path": str(checkpoint),
                            "model_sha256": digest(checkpoint / "model.safetensors")}
    result = {"task": task, "status": "completed", "checkpoints": checkpoints}
    atomic_json(root / task / "screen_training_summary.json", result)
    return result


def evaluate_screen(workspace: Path, root: Path, task: str) -> dict:
    _, root, _, row, runtime, spec = _context(workspace, root, task)
    trained = read_json(root / task / "screen_training_summary.json")["checkpoints"]
    checkpoints = {"official": Path(row["official_checkpoint"]), **{
        arm: Path(value["path"]) for arm, value in trained.items()}}
    results = {}
    for repeat in (1, 2):
        master = row["banks"][f"development_{repeat}"]["master"]
        for label, checkpoint in checkpoints.items():
            destination = root / task / f"screen_eval_{label}_development_{repeat}"
            metrics = destination / "evaluation_metrics.json"
            data = read_json(metrics) if metrics.is_file() else runtime.evaluate(
                spec, checkpoint, destination, episodes=100, master_seed=master,
                purpose="development")
            results[f"{label}_development_{repeat}"] = data["summary"]
    atomic_json(root / task / "screen_evaluation_summary.json", results)
    return results


def analyze_screen(root: Path, task: str) -> dict:
    root = root.resolve()
    labels = ("official", *SCREEN_ARMS)
    metrics = {(label, repeat): read_json(
        root / task / f"screen_eval_{label}_development_{repeat}/evaluation_metrics.json")
        for label in labels for repeat in (1, 2)}
    means = {label: sum(metrics[label, r]["summary"]["success_rate"] for r in (1, 2)) / 2
             for label in labels}
    comparisons = {}
    for arm in SCREEN_ARMS:
        for baseline in ("official", "official_continue"):
            if arm == baseline:
                continue
            for repeat in (1, 2):
                comparisons[f"{arm}_vs_{baseline}_development_{repeat}"] = compare(
                    metrics[arm, repeat], metrics[baseline, repeat])
    reference = max(means["official"], means["official_continue"])
    eligible = []
    for arm in ("mixed_proportional", "mixed_20pct"):
        deltas = [metrics[arm, r]["summary"]["success_rate"] -
                  max(metrics["official", r]["summary"]["success_rate"],
                      metrics["official_continue", r]["summary"]["success_rate"])
                  for r in (1, 2)]
        if means[arm] - reference >= .03 and min(deltas) >= 0:
            eligible.append(arm)
    selected = max(eligible, key=means.get) if eligible else None
    result = {"task": task, "status": "completed", "means": means,
              "comparisons": comparisons, "eligible_for_80k": eligible,
              "selected_arm": selected,
              "selection_rule": "mean >= best official/official-continue mean +3pp and nonnegative on both frozen development banks"}
    atomic_json(root / task / "screen_analysis.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("production", "mixtures", "train-screen",
                                            "evaluate-screen", "analyze-screen"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    functions = {"production": production, "mixtures": mixtures,
                 "train-screen": train_screen, "evaluate-screen": evaluate_screen}
    if args.command == "analyze-screen":
        value = analyze_screen(args.root, args.task)
    else:
        value = functions[args.command](args.workspace, args.root, args.task)
    print(value)


if __name__ == "__main__":
    main()
