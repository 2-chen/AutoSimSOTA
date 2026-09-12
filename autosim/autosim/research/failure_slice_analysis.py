"""Audit repeated development rollouts against pre-frozen scene slices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .bounded_probe_analysis import _wilson
from .common import atomic_json, digest, now, read_json


NOMINAL_CAMERA = {"fx": 606.315186, "fy": 606.100952,
                  "x": 0.257046, "y": 0.049382, "z": 1.459689}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def scene_slice(row: dict[str, Any]) -> dict[str, Any]:
    camera = row["cameras"]["cam_high"]
    intrinsics, pose = _first(camera["intrinsics"]), _first(camera["local_pose"])
    fx, fy = float(intrinsics[0][0]), float(intrinsics[1][1])
    x, y, z = float(pose[0][3]), float(pose[1][3]), float(pose[2][3])
    intrinsic = max(abs(fx - NOMINAL_CAMERA["fx"]) / 50.0,
                    abs(fy - NOMINAL_CAMERA["fy"]) / 50.0)
    position = max(abs(x - NOMINAL_CAMERA["x"]) / .02,
                   abs(y - NOMINAL_CAMERA["y"]) / .02,
                   abs(z - NOMINAL_CAMERA["z"]) / .02)
    button = _first(row["entities"]["button"]["pose"])
    bx, by = float(button[0][3]), float(button[1][3])
    distances = []
    for item in row.get("active_distractors", []):
        distractor = _first(item["pose"])
        dx, dy = float(distractor[0][3]) - bx, float(distractor[1][3]) - by
        distances.append(math.sqrt(dx * dx + dy * dy))
    distance = min(distances) if distances else None
    return {
        "camera_fx": fx, "camera_fy": fy,
        "camera_x": x, "camera_y": y, "camera_z": z,
        "camera_intrinsic_severity": intrinsic,
        "camera_position_severity": position,
        "camera_high": intrinsic >= .5 or position >= .5,
        "active_distractor_count": len(distances),
        "clutter_distance": distance,
        "clutter_near": distance is not None and distance <= .18,
        "button_x": bx, "button_y": by,
    }


def _slice_result(episodes: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups = {}
    for value in (False, True):
        rows = [row for row in episodes if row[field] is value]
        successes = sum(bool(row["success"]) for row in rows)
        groups["high_or_near" if value else "low_or_far"] = {
            "episodes": len(rows), "successes": successes,
            "success_rate": successes / len(rows) if rows else None,
            "wilson_95": _wilson(successes, len(rows)),
        }
    low, high = groups["low_or_far"], groups["high_or_near"]
    gap = (low["success_rate"] - high["success_rate"]
           if low["success_rate"] is not None and high["success_rate"] is not None else None)
    gate = (min(low["episodes"], high["episodes"]) >= 8
            and gap is not None and gap >= .15)
    return {"field": field, "groups": groups,
            "low_or_far_minus_high_or_near_success_rate": gap,
            "exploratory_gate_passed": gate}


def _load_repeat(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    metrics_path, telemetry_path = directory / "evaluation_metrics.json", directory / "telemetry.jsonl"
    metrics, telemetry = read_json(metrics_path), _jsonl(telemetry_path)
    step_zero = [row for row in telemetry if int(row.get("step", -1)) == 0]
    by_seed = {int(row["seed"]): row for row in step_zero}
    sampled_depth_by_seed: dict[int, float] = {}
    for row in telemetry:
        seed = int(row["seed"])
        qpos = row.get("entities", {}).get("button", {}).get("qpos")
        values, stack = [], [qpos]
        while stack:
            value = stack.pop()
            if isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            sampled_depth_by_seed[seed] = max(
                sampled_depth_by_seed.get(seed, 0.0), max(0.0, -min(values)))
    seeds = [int(row["episode_seed"]) for row in metrics["episodes"]]
    if len(step_zero) != len(seeds) or set(by_seed) != set(seeds):
        raise ValueError(f"step-zero telemetry does not cover episodes in {directory}")
    episodes = []
    for metric in metrics["episodes"]:
        seed = int(metric["episode_seed"])
        success = bool(metric["success"])
        depth = sampled_depth_by_seed.get(seed)
        if success:
            contact_class = "success"
        elif depth is None:
            contact_class = "unclassified"
        elif depth >= .0040:
            contact_class = "near_threshold_press"
        elif depth >= .0001:
            contact_class = "contact_insufficient_press"
        else:
            contact_class = "no_button_contact"
        episodes.append({"seed": seed, "success": success,
                         "max_sampled_press_depth": depth,
                         "contact_class": contact_class,
                         **scene_slice(by_seed[seed])})
    protocol_path = directory / "protocol.json"
    return metrics, episodes, [metrics_path, telemetry_path, protocol_path]


def analyze_repeats(root: Path, repeat_names: list[str]) -> dict[str, Any]:
    loaded, inputs = [], []
    for name in repeat_names:
        metrics, episodes, paths = _load_repeat(root / name)
        loaded.append((name, metrics, episodes))
        inputs.extend(paths)
    if len(loaded) != 2:
        raise ValueError("the frozen diagnostic requires exactly two repeats")
    seed_orders = [[row["seed"] for row in item[2]] for item in loaded]
    same_order = seed_orders[0] == seed_orders[1]
    outcomes = [[row["success"] for row in item[2]] for item in loaded]
    mismatches = [seed for seed, left, right in zip(seed_orders[0], outcomes[0], outcomes[1])
                  if left != right]

    repeat_rows = []
    for name, metrics, episodes in loaded:
        contact_counts = {label: sum(row["contact_class"] == label for row in episodes)
                          for label in ("success", "near_threshold_press",
                                        "contact_insufficient_press", "no_button_contact",
                                        "unclassified")}
        repeat_rows.append({
            "name": name, "summary": metrics["summary"],
            "camera": _slice_result(episodes, "camera_high"),
            "clutter": _slice_result(episodes, "clutter_near"),
            "camera_position_max_severity": max(row["camera_position_severity"] for row in episodes),
            "camera_intrinsic_severity_range": [
                min(row["camera_intrinsic_severity"] for row in episodes),
                max(row["camera_intrinsic_severity"] for row in episodes)],
            "active_distractor_count_range": [
                min(row["active_distractor_count"] for row in episodes),
                max(row["active_distractor_count"] for row in episodes)],
            "contact_outcomes_from_ten_step_sampled_qpos": contact_counts,
        })

    numeric = ("camera_fx", "camera_fy", "camera_x", "camera_y", "camera_z",
               "clutter_distance", "button_x", "button_y")
    scene_max_abs_difference = {}
    for key in numeric:
        differences = []
        for left, right in zip(loaded[0][2], loaded[1][2]):
            if left[key] is not None and right[key] is not None:
                differences.append(abs(float(left[key]) - float(right[key])))
        scene_max_abs_difference[key] = max(differences) if differences else None
    camera_gate = all(row["camera"]["exploratory_gate_passed"] for row in repeat_rows)
    clutter_gate = all(row["clutter"]["exploratory_gate_passed"] for row in repeat_rows)
    repeatability = same_order and len(mismatches) / len(seed_orders[0]) <= .10
    return {
        "schema_version": 1, "kind": "stage_1_failure_slice_repeatability_analysis",
        "created_at": now(), "root": str(root.resolve()), "repeats": repeat_rows,
        "repeatability": {
            "same_ordered_seed_bank": same_order,
            "paired_outcome_mismatch_count": len(mismatches),
            "paired_outcome_mismatch_rate": len(mismatches) / len(seed_orders[0]),
            "mismatch_seeds": mismatches,
            "scene_feature_max_absolute_difference": scene_max_abs_difference,
            "gate_passed": repeatability,
        },
        "candidate_failure_relevance": {
            "camera_gate_passed_in_both_repeats": camera_gate,
            "clutter_gate_passed_in_both_repeats": clutter_gate,
            "eligible_training_probe_candidates": [name for name, passed in
                (("targeted_camera", camera_gate), ("targeted_clutter", clutter_gate)) if passed],
            "exploratory_not_confirmatory": True,
        },
        "benchmark_implementation_observation": {
            "camera_intrinsics_randomized": any(
                row["camera_intrinsic_severity_range"][1] > 1e-6 for row in repeat_rows),
            "camera_extrinsic_translation_randomized_observed": any(
                row["camera_position_max_severity"] > .01 for row in repeat_rows),
            "scope": "Only this frozen ClickBell config/code and these 40 seeds; rotation is not reduced to the translation statistic.",
        },
        "decision": {
            "learning_probe_authorized": repeatability and (camera_gate or clutter_gate),
            "reason": ("At least one pre-frozen failure slice passed the exploratory gate in both repeats."
                       if repeatability and (camera_gate or clutter_gate)
                       else "No candidate met both the repeatability and pre-frozen failure-relevance gates; do not train on candidate labels alone."),
        },
        "frozen_input_manifest": {str(path): {"bytes": path.stat().st_size, "sha256": digest(path)}
                                  for path in sorted(inputs)},
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# 第一阶段失败切片与重复性结果", "",
             "同一个官方 ACT checkpoint 在同一组 40 个 development seed 上完整运行两次。", "",
             "| 重复 | 总成功率 | camera 低/高成功率（n） | clutter 远/近成功率（n） |",
             "| --- | ---: | --- | --- |"]
    for row in result["repeats"]:
        camera, clutter = row["camera"]["groups"], row["clutter"]["groups"]
        c0, c1 = camera["low_or_far"], camera["high_or_near"]
        d0, d1 = clutter["low_or_far"], clutter["high_or_near"]
        lines.append(f"| {row['name']} | {row['summary']['success_rate']:.1%} | "
                     f"{c0['success_rate']:.1%} ({c0['episodes']}) / {c1['success_rate']:.1%} ({c1['episodes']}) | "
                     f"{d0['success_rate']:.1%} ({d0['episodes']}) / {d1['success_rate']:.1%} ({d1['episodes']}) |")
    repeat = result["repeatability"]
    relevance = result["candidate_failure_relevance"]
    observation = result["benchmark_implementation_observation"]
    lines += ["", "## 门结论", "",
              f"- 同 seed 结果不一致：{repeat['paired_outcome_mismatch_count']}/40（{repeat['paired_outcome_mismatch_rate']:.1%}）；重复性门：{repeat['gate_passed']}。",
              f"- camera 候选两遍均过探索门：{relevance['camera_gate_passed_in_both_repeats']}。",
              f"- clutter 候选两遍均过探索门：{relevance['clutter_gate_passed_in_both_repeats']}。",
              f"- 本次观察到内参随机化：{observation['camera_intrinsics_randomized']}；观察到相机平移外参随机化：{observation['camera_extrinsic_translation_randomized_observed']}。",
              f"- 10 步采样按钮 qpos 的失败类别（第一遍）：{result['repeats'][0]['contact_outcomes_from_ten_step_sampled_qpos']}。",
              f"- 是否允许进入 5K 学习探针：{result['decision']['learning_probe_authorized']}。",
              "- 这些是 development 探索门，不是因果证明、官方成绩或最终统计结论。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeat", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = analyze_repeats(args.root, args.repeat)
    atomic_json(args.output, result)
    if args.report:
        args.report.write_text(render_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
