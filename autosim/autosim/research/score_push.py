"""Resume-safe score-first RoboSyn data and policy improvement campaign.

The campaign keeps evaluation and policy contracts fixed while admitting only
fully decoded self-collected demonstrations.  Commands are deliberately small
so a long single-GPU campaign can resume without overwriting completed work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autosim.experiment_validation.dataset_video_integrity_v1 import audit_dataset

from .common import atomic_json, digest, freeze_files, immutable_json, now, read_json
from .ledger import SeedLedger
from .registry import load_task
from .runtime import Runtime


CAMPAIGN = {
    "handle_basket": {"priority": 0, "baseline": .33, "cap_attempts": 100,
                      "production_target": 500, "production_attempts": 800},
    "water_pouring": {"priority": 0, "baseline": .40, "cap_attempts": 100,
                      "production_target": 500, "production_attempts": 1100},
    "manipulate_pipette": {"priority": 0, "baseline": .49, "cap_attempts": 100,
                           "production_target": 500, "production_attempts": 700},
    "items_handover": {"priority": 1, "baseline": .08, "cap_attempts": 100,
                       "production_target": 1000, "production_attempts": 2500},
    "drawer_open_place": {"priority": 1, "baseline": .04, "cap_attempts": 100,
                          "production_target": 500, "production_attempts": 800},
    "mixer_operating": {"priority": 2, "baseline": .71, "cap_attempts": 100,
                        "production_target": 300, "production_attempts": 400},
    "table_rearrangement": {"priority": 3, "baseline": .95, "cap_attempts": 100,
                            "production_target": 200, "production_attempts": 250},
    "sample_loading": {"priority": 4, "baseline": .01, "cap_attempts": 200,
                       "production_target": 0, "production_attempts": 0,
                       "requires_expert_repair": True},
    "item_assembly": {"priority": 4, "baseline": .18, "cap_attempts": 0,
                      "production_target": 0, "production_attempts": 0,
                      "requires_expert_repair": True},
}

OFFSETS = {"development_1": 1, "development_2": 2, "confirmation": 3,
           "capability": 4, "production": 5}


def _asset_root(workspace: Path, task: str, kind: str) -> Path:
    return workspace / "autosim/output/robosyn_general_20260905/assets" / task / kind


def initialize(workspace: Path, root: Path) -> dict:
    workspace, root = workspace.resolve(), root.resolve()
    runtime = Runtime(workspace, root)
    rows = {}
    ledger = SeedLedger(workspace / "autosim/output/robosyn_general_20260905/seeds.sqlite")
    try:
        for index, (task, settings) in enumerate(CAMPAIGN.items(), start=1):
            spec = load_task(runtime.repo, task)
            base = 98_000_000 + index * 100
            banks = {}
            for name, offset in OFFSETS.items():
                count = ({"development_1": 100, "development_2": 100,
                          "confirmation": 200,
                          "capability": settings["cap_attempts"],
                          "production": settings["production_attempts"]}[name])
                if count == 0:
                    continue
                master = base + offset
                seeds = ledger.reserve(task, f"score_push_20260908:{task}:{name}",
                                       "development" if name.startswith("development") else
                                       "confirmation" if name == "confirmation" else "collection",
                                       master, count)
                from .common import object_digest
                banks[name] = {"master": master, "count": count,
                               "ordered_seed_sha256": object_digest(seeds)}
            dataset, model = _asset_root(workspace, task, "dataset"), _asset_root(workspace, task, "model")
            required = [dataset / "meta/info.json", model / "model.safetensors", model / "config.json"]
            if any(not path.is_file() for path in required):
                raise FileNotFoundError(f"missing official asset for {task}: {required}")
            rows[task] = {**settings, "task_signature": spec.signature,
                          "official_dataset": str(dataset), "official_checkpoint": str(model),
                          "official_dataset_info_sha256": digest(dataset / "meta/info.json"),
                          "official_model_sha256": digest(model / "model.safetensors"),
                          "banks": banks}
    finally:
        ledger.close()
    protocol = {
        "schema_version": 1, "kind": "robosyn_score_push_campaign",
        "created_at": now(), "workspace": str(workspace), "tasks": rows,
        "policy_contract": {"architecture": "ACT", "state_dim": 14,
                            "action_dim": 14, "rgb_cameras": 3,
                            "evaluator_success_modified": False},
        "capability_gate": {"attempts": "per task table", "scale_if_yield_at_least": .5,
                            "repair_then_reprobe_if_yield_between": [.2, .5],
                            "do_not_scale_if_below": .2},
        "screen_training": {"updates": 20000, "seed": 1000,
                            "arms": ["official_continue", "mixed_proportional", "mixed_20pct"],
                            "same_parent": True},
        "evaluation": {"development_repeats": 2, "episodes_each": 100,
                       "confirmation_episodes": 200},
        "sources": freeze_files([Path(__file__), Path(__file__).with_name("runtime.py"),
                                  runtime.repo / "policy/act/scripts/train.py"]),
    }
    path = root / "campaign_protocol.json"
    if path.is_file():
        previous = read_json(path)
        # Timestamp is provenance, not a redefinition opportunity.
        candidate, old = dict(protocol), dict(previous)
        candidate["created_at"] = old.get("created_at")
        if candidate != old:
            raise RuntimeError("campaign protocol already exists with different inputs")
        return previous
    atomic_json(path, protocol)
    return protocol


def _runtime(workspace: Path, root: Path) -> tuple[Runtime, dict]:
    protocol = read_json(root / "campaign_protocol.json")
    return Runtime(workspace.resolve(), root.resolve(), gpu="0"), protocol


def evaluate_drawer_existing(workspace: Path, root: Path) -> dict:
    runtime, protocol = _runtime(workspace, root)
    task = "drawer_open_place"
    spec, row = load_task(runtime.repo, task), protocol["tasks"][task]
    controlled = workspace / ("autosim/output/robosyn_general_20260905/research/train_seed_1000/"
                              "drawer_open_place/controlled_baseline/train/checkpoints/080000/pretrained_model")
    checkpoints = {"official": Path(row["official_checkpoint"]), "official_retrain_80k": controlled}
    result = {}
    for repeat in (1, 2):
        master = row["banks"][f"development_{repeat}"]["master"]
        for label, checkpoint in checkpoints.items():
            destination = root / task / f"existing_{label}_development_{repeat}"
            result[f"{label}_development_{repeat}"] = runtime.evaluate(
                spec, checkpoint, destination, episodes=100, master_seed=master,
                purpose="development")
    summary = {name: value["summary"] for name, value in result.items()}
    atomic_json(root / task / "existing_drawer_summary.json", summary)
    return summary


def collect_capability(workspace: Path, root: Path, task: str) -> dict:
    runtime, protocol = _runtime(workspace, root)
    if task not in CAMPAIGN:
        raise ValueError(f"unknown campaign task: {task}")
    row = protocol["tasks"][task]
    attempts = int(row["cap_attempts"])
    if attempts <= 0:
        result = {"task": task, "status": "requires_expert_repair", "attempts": 0,
                  "accepted": 0, "yield": None}
        atomic_json(root / task / "capability_summary.json", result)
        return result
    spec = load_task(runtime.repo, task)
    destination = root / task / "capability_collection"
    bounded = runtime.collect_bounded(
        spec, destination, attempt_budget=attempts, target_episodes=attempts,
        master_seed=row["banks"]["capability"]["master"], profile="full_random",
        timeout=max(7200, attempts * 180))
    accepted = int(bounded["accepted_episodes"])
    data_root = Path(bounded["dataset_root"]) if bounded["dataset_root"] else None
    data_audit = video_audit = None
    if data_root is not None:
        data_audit = runtime.prepare_data(spec, data_root, destination)
        video_audit = audit_dataset(
            data_root, expected_episodes=accepted,
            expected_video_keys=[f"observation.images.{camera}" for camera in spec.cameras])
        atomic_json(destination / "video_integrity.json", video_audit)
    rate = accepted / attempts
    threshold_state = (
        "scale" if rate >= .5 else "repair_then_reprobe" if rate >= .2 else "do_not_scale")
    data_passed = bool(data_audit and data_audit["passed"])
    video_passed = bool(video_audit and video_audit["passed"])
    integrity_passed = data_root is not None and data_passed and video_passed
    # A yield that clears the scale threshold is still unusable when the saved
    # dataset fails either mandatory audit.  Persist the negative result instead
    # of raising before capability_summary.json exists; downstream stages see no
    # admitted dataset path and can take the preregistered official-only fallback.
    state = threshold_state if integrity_passed or threshold_state != "scale" else "repair_then_reprobe"
    admitted = integrity_passed and state == "scale"
    admission_failure = None
    if not integrity_passed:
        admission_failure = "mandatory data/video integrity audit failed"
    elif state != "scale":
        admission_failure = "capability yield gate did not pass"
    result = {"task": task, "status": "completed", "attempts": attempts,
              "accepted": accepted, "yield": rate, "decision": state,
              "dataset_root": str(data_root) if admitted else None,
              "collected_dataset_root": str(data_root) if data_root else None,
              "data_audit_passed": data_passed,
              "video_audit_passed": video_passed,
              "training_admission": "admitted" if admitted else "rejected",
              "admission_failure": admission_failure}
    atomic_json(root / task / "capability_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "drawer-existing", "capability"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(CAMPAIGN))
    args = parser.parse_args()
    if args.command == "init":
        value = initialize(args.workspace, args.root)
    elif args.command == "drawer-existing":
        value = evaluate_drawer_existing(args.workspace, args.root)
    else:
        if args.task is None:
            parser.error("capability requires --task")
        value = collect_capability(args.workspace, args.root, args.task)
    print(value)


if __name__ == "__main__":
    main()
