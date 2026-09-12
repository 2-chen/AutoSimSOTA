"""Independent integrity audit for selected ACT research checkpoints."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from safetensors import safe_open

from autosim.research.common import atomic_json, digest, freeze_files, now, read_json
from .fairness_audit import TASKS, command_value
from .queue_manifest import versioned_manifest


def audit_checkpoint(checkpoint: Path, expected_step: int, expected_seed: int) -> dict:
    checkpoint = checkpoint.absolute()
    step_root = checkpoint.parent
    arm_root = checkpoint.parents[3]
    process_path = arm_root / f"process_train_{expected_step}/process.json"
    step_path = step_root / "training_state/training_step.json"
    config_path = checkpoint / "config.json"
    train_config_path = checkpoint / "train_config.json"
    weights_path = checkpoint / "model.safetensors"
    optimizer_groups_path = step_root / "training_state/optimizer_param_groups.json"
    optimizer_state_path = step_root / "training_state/optimizer_state.safetensors"
    rng_state_path = step_root / "training_state/rng_state.safetensors"
    required = (process_path, step_path, config_path, train_config_path, weights_path,
                optimizer_groups_path, optimizer_state_path, rng_state_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"checkpoint": str(checkpoint), "status": "failed", "passed": False,
                "missing": missing, "expected_step": expected_step, "expected_seed": expected_seed}

    process = read_json(process_path)
    step = read_json(step_path)
    config = read_json(config_path)
    train_config = read_json(train_config_path)
    optimizer_groups = read_json(optimizer_groups_path)
    inputs = config.get("input_features", {})
    visual = {name: value for name, value in inputs.items() if value.get("type") == "VISUAL"}
    state_features = [value for value in inputs.values() if value.get("type") == "STATE"]
    action = config.get("output_features", {}).get("action", {})
    last = checkpoint.parents[1] / "last"
    checks = [
        {"name": "process_exit_zero", "passed": process.get("returncode") == 0,
         "actual": process.get("returncode")},
        {"name": "process_step_budget", "passed": int(command_value(process, "--steps")) == expected_step,
         "actual": int(command_value(process, "--steps")), "expected": expected_step},
        {"name": "process_train_seed", "passed": int(command_value(process, "--seed")) == expected_seed,
         "actual": int(command_value(process, "--seed")), "expected": expected_seed},
        {"name": "serialized_training_step", "passed": int(step.get("step", -1)) == expected_step,
         "actual": step.get("step"), "expected": expected_step},
        {"name": "checkpoint_directory_step", "passed": step_root.name == f"{expected_step:06d}",
         "actual": step_root.name, "expected": f"{expected_step:06d}"},
        {"name": "last_pointer", "passed": last.is_symlink() and last.resolve() == step_root.resolve(),
         "actual": str(last.resolve()) if last.exists() else "missing", "expected": str(step_root)},
        {"name": "train_config_step", "passed": int(train_config.get("steps", -1)) == expected_step,
         "actual": train_config.get("steps"), "expected": expected_step},
        {"name": "train_config_seed", "passed": int(train_config.get("seed", -1)) == expected_seed,
         "actual": train_config.get("seed"), "expected": expected_seed},
        {"name": "state_contract", "passed": len(state_features) == 1 and state_features[0].get("shape") == [14],
         "actual": [value.get("shape") for value in state_features], "expected": [[14]]},
        {"name": "action_contract", "passed": action.get("type") == "ACTION" and action.get("shape") == [14],
         "actual": action, "expected": {"type": "ACTION", "shape": [14]}},
        {"name": "camera_contract", "passed": len(visual) == 3 and
         all(value.get("shape") == [3, 480, 640] for value in visual.values()),
         "actual": {name: value.get("shape") for name, value in visual.items()}, "expected_count": 3},
        {"name": "optimizer_param_groups", "passed": isinstance(optimizer_groups, list) and
         bool(optimizer_groups) and all(isinstance(group, dict) for group in optimizer_groups),
         "actual_type": type(optimizer_groups).__name__,
         "group_count": len(optimizer_groups) if isinstance(optimizer_groups, list) else None},
    ]
    tensor_count = total_elements = 0
    dtypes = set()
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                view = tensors.get_slice(key)
                tensor_count += 1
                total_elements += math.prod(view.get_shape())
                dtypes.add(str(view.get_dtype()))
        checks.append({"name": "safetensors_header", "passed": tensor_count > 0 and total_elements > 0,
                       "tensor_count": tensor_count, "parameter_elements": total_elements,
                       "dtypes": sorted(dtypes)})
    except Exception as exc:
        checks.append({"name": "safetensors_header", "passed": False,
                       "error": f"{type(exc).__name__}: {exc}"})
    training_state_summaries = {}
    for name, path in (("optimizer_state_header", optimizer_state_path),
                       ("rng_state_header", rng_state_path)):
        try:
            with safe_open(path, framework="pt", device="cpu") as tensors:
                keys = list(tensors.keys())
                elements = sum(math.prod(tensors.get_slice(key).get_shape()) for key in keys)
                state_dtypes = sorted({str(tensors.get_slice(key).get_dtype()) for key in keys})
            summary = {"tensor_count": len(keys), "elements": elements,
                       "dtypes": state_dtypes, "bytes": path.stat().st_size}
            training_state_summaries[name] = summary
            checks.append({"name": name, "passed": bool(keys) and elements > 0, **summary})
        except Exception as exc:
            checks.append({"name": name, "passed": False,
                           "error": f"{type(exc).__name__}: {exc}"})
    evidence = {str(path.absolute()): digest(path) for path in required}
    failures = [check for check in checks if not check["passed"]]
    return {"checkpoint": str(checkpoint), "status": "passed" if not failures else "failed",
            "passed": not failures, "expected_step": expected_step, "expected_seed": expected_seed,
            "checks": checks, "failed_checks": len(failures), "tensor_count": tensor_count,
            "parameter_elements": total_elements, "weights_bytes": weights_path.stat().st_size,
            "training_state": training_state_summaries,
            "evidence": evidence,
            "scope_note": "model/optimizer/RNG header, hash, and contract audit; finite-value and behavior checks are supplied by native evaluation"}


def write_audit(root: Path, research_root: Path) -> dict:
    rows = []
    for task in TASKS:
        state_path = research_root / task / "state.json"
        state = read_json(state_path) if state_path.is_file() else {}
        task_row = {"task": task, "research_status": state.get("status", "missing"), "checkpoints": {}}
        if state.get("status") == "completed_development":
            paths = {"controlled_baseline": Path(state["baseline_checkpoint"]),
                     "auto_selected": Path(state["selected"]["auto"]["checkpoint"]),
                     "random_selected": Path(state["selected"]["random"]["checkpoint"])}
            task_row["checkpoints"] = {name: audit_checkpoint(path, 80000, 1000)
                                        for name, path in paths.items()}
            task_row["passed"] = all(row["passed"] for row in task_row["checkpoints"].values())
        else:
            task_row.update(passed=False, status="pending")
        rows.append(task_row)
    complete = all(row["passed"] for row in rows)
    value = {"updated_at": now(), "kind": "selected_checkpoint_integrity_audit",
             "tasks": rows, "all_selected_checkpoints_passed": complete,
             "policy_performance_claim": False, "official_leaderboard_result": False, "sota_claim": False}
    atomic_json(root / "checkpoint_audit.json", value)
    lines = ["# 三任务选中 checkpoint 完整性审计", "",
             "检查真实训练回执、步数、seed、模型结构、last指针及权重文件，不用目录存在代替训练完成。", "",
             "| 任务 | 研究状态 | 固定基线 | 自动选中 | 随机选中 |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        def label(name):
            item = row["checkpoints"].get(name)
            return "待完成" if item is None else ("通过" if item["passed"] else f"失败({item['failed_checks']})")
        lines.append(f"| {row['task']} | {row['research_status']} | {label('controlled_baseline')} | "
                     f"{label('auto_selected')} | {label('random_selected')} |")
    lines += ["", f"九个选中 checkpoint 全部通过：{'是' if complete else '否/待完成'}。",
              "", "注意：本审计证明文件和训练契约完整，不证明policy行为有效；行为由原生仿真确认评测验证。"]
    (root / "checkpoint_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def queue(root: Path, research_root: Path, comparison_state: Path,
          max_hours: float, poll_seconds: float):
    if not 0 < max_hours <= 336 or not 1 <= poll_seconds <= 60:
        raise ValueError("invalid bounded queue settings")
    local = Path(__file__).resolve()
    versioned_manifest(root, {"tasks": list(TASKS), "expected_step": 80000, "expected_seed": 1000,
                              "comparison_state": str(comparison_state),
                              "sources": freeze_files([local, local.with_name("queue_manifest.py")]),
                              "policy_performance_claim": False, "sota_claim": False})
    state_path = root / "checkpoint_state.json"
    prior = read_json(state_path) if state_path.is_file() else {}
    if prior.get("status") in {"completed", "requires_review", "queue_wait_budget_exhausted"}:
        return prior
    deadline = float(prior.get("deadline_epoch", time.time() + max_hours * 3600))
    state = {**prior, "status": "running", "started_at": prior.get("started_at", now()),
             "deadline_epoch": deadline}
    while time.time() < deadline:
        comparison = read_json(comparison_state) if comparison_state.is_file() else {}
        if comparison.get("status") == "three_task_comparison_complete":
            result = write_audit(root, research_root)
            state.update(status="completed" if result["all_selected_checkpoints_passed"] else "requires_review",
                         all_selected_checkpoints_passed=result["all_selected_checkpoints_passed"], updated_at=now())
            atomic_json(state_path, state)
            return state
        if comparison.get("status") in {"requires_review", "queue_wait_budget_exhausted"}:
            state.update(status="requires_review", upstream_status=comparison.get("status"), updated_at=now())
            atomic_json(state_path, state)
            return state
        state.update(stage="waiting_three_task_comparison", upstream_status=comparison.get("status", "missing"),
                     updated_at=now())
        atomic_json(state_path, state)
        time.sleep(poll_seconds)
    state.update(status="queue_wait_budget_exhausted", updated_at=now())
    atomic_json(state_path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("checkpoint", "audit", "queue"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--comparison-state", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int, default=80000)
    parser.add_argument("--expected-seed", type=int, default=1000)
    parser.add_argument("--max-hours", type=float, default=240)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.command == "checkpoint":
        if args.checkpoint is None or args.root is None:
            parser.error("checkpoint requires --checkpoint and --root")
        atomic_json(args.root.absolute(), audit_checkpoint(args.checkpoint, args.expected_step, args.expected_seed))
    elif args.command == "audit":
        if args.root is None or args.research_root is None:
            parser.error("audit requires --root and --research-root")
        write_audit(args.root.absolute(), args.research_root.absolute())
    else:
        if args.root is None or args.research_root is None or args.comparison_state is None:
            parser.error("queue requires --root, --research-root and --comparison-state")
        queue(args.root.absolute(), args.research_root.absolute(), args.comparison_state.absolute(),
              args.max_hours, args.poll_seconds)
