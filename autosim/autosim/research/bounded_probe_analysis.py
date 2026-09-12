"""Analyze bounded collection probes without turning them into policy results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from .common import atomic_json, digest, now, read_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _first_env(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _features(row: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    camera = row.get("cameras", {}).get("cam_high", {})
    intrinsics = _first_env(camera.get("intrinsics"))
    pose = _first_env(camera.get("local_pose"))
    if isinstance(intrinsics, list) and len(intrinsics) >= 2:
        result.update(camera_fx=float(intrinsics[0][0]), camera_fy=float(intrinsics[1][1]))
    if isinstance(pose, list) and len(pose) >= 3:
        result.update(camera_x=float(pose[0][3]), camera_y=float(pose[1][3]),
                      camera_z=float(pose[2][3]))
    button = _first_env(row.get("entities", {}).get("button", {}).get("pose"))
    if isinstance(button, list) and len(button) >= 3:
        result.update(button_x=float(button[0][3]), button_y=float(button[1][3]))
    robot = _first_env(row.get("robot_qpos"))
    if isinstance(robot, list) and robot:
        result["robot_qpos_l2"] = math.sqrt(sum(float(value) ** 2 for value in robot))
    result["active_distractor_count"] = float(len(row.get("active_distractors", [])))
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    keys = sorted({key for row in rows for key in row["features"]})
    result = {}
    for key in keys:
        values = [float(row["features"][key]) for row in rows if key in row["features"]]
        mean = fmean(values)
        result[key] = {
            "count": len(values), "mean": mean,
            "std": math.sqrt(fmean([(value - mean) ** 2 for value in values])),
            "min": min(values), "max": max(values),
        }
    return result


def analyze_probe_root(root: Path, profiles: list[str]) -> dict[str, Any]:
    root = root.resolve()
    profile_rows, inputs = [], []
    for profile in profiles:
        directory = root / f"bounded_probe_{profile}"
        paths = {
            name: directory / relative for name, relative in {
                "manifest": "collection.json", "scene": "scene_resets.jsonl",
                "scene_audit": "scene_evidence_audit.json", "bounded": "bounded_collection_result.json",
                "data_audit": "data_audit.json", "video_audit": "video_integrity_v1.json",
                "process": "process/process.json",
            }.items()
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{profile} missing required artifacts: {missing}")
        inputs.extend(paths.values())
        manifest, scene_rows = read_json(paths["manifest"]), _jsonl(paths["scene"])
        by_seed = {int(row["seed"]): row for row in scene_rows if row.get("seed") is not None}
        joined = [{"seed": int(attempt["seed"]), "saved": bool(attempt["saved"]),
                   "reason": attempt.get("reason"), "features": _features(by_seed[int(attempt["seed"])])}
                  for attempt in manifest["attempts"]]
        accepted = [row for row in joined if row["saved"]]
        rejected = [row for row in joined if not row["saved"]]
        overall, accepted_summary, rejected_summary = (
            _summary(joined), _summary(accepted), _summary(rejected))
        mean_differences = {
            key: accepted_summary[key]["mean"] - rejected_summary[key]["mean"]
            for key in sorted(set(accepted_summary) & set(rejected_summary))
        }
        count = len(joined)
        profile_rows.append({
            "profile": profile, "attempts": count, "accepted": len(accepted),
            "yield": len(accepted) / count if count else None,
            "yield_wilson_95": _wilson(len(accepted), count),
            "elapsed_seconds": float(read_json(paths["process"])["elapsed_seconds"]),
            "scene_evidence_passed": bool(read_json(paths["scene_audit"])["passed"]),
            "data_audit_passed": bool(read_json(paths["data_audit"])["passed"]),
            "full_video_decode_passed": bool(read_json(paths["video_audit"])["passed"]),
            "all_attempts": overall, "accepted_attempts": accepted_summary,
            "rejected_attempts": rejected_summary,
            "accepted_minus_rejected_means_descriptive_only": mean_differences,
            "inference_warning": "n=20 was frozen as a capability probe; post-hoc feature differences are descriptive, not confirmatory evidence.",
        })

    by_profile = {row["profile"]: row for row in profile_rows}
    camera = by_profile.get("targeted_camera", {}).get("all_attempts", {})
    clutter = by_profile.get("targeted_clutter", {}).get("all_attempts", {})
    camera_varies = any(float(camera.get(key, {}).get("std", 0.0)) > 1e-6
                        for key in ("camera_fx", "camera_fy", "camera_x", "camera_y", "camera_z"))
    clutter_present = float(clutter.get("active_distractor_count", {}).get("min", 0.0)) >= 2.0
    all_admitted = all(row["scene_evidence_passed"] and row["data_audit_passed"]
                       and row["full_video_decode_passed"] for row in profile_rows)
    return {
        "schema_version": 1, "kind": "stage_1_bounded_collection_probe_analysis",
        "created_at": now(), "root": str(root), "profiles": profile_rows,
        "candidate_checks": {
            "targeted_camera_realized_geometry_varies": camera_varies,
            "targeted_clutter_realized_two_distractors": clutter_present,
            "two_non_synonymous_measurable_options": camera_varies and clutter_present,
            "all_collection_and_data_gates_passed": all_admitted,
        },
        "conclusion": {
            "observed_expert_yield_range": [
                min(row["yield"] for row in profile_rows),
                max(row["yield"] for row in profile_rows),
            ],
            "confirmatory_profile_yield_heterogeneity_proven": False,
            "natural_slice_selection_bias_proven": False,
            "failure_relevance_proven": False,
            "reason": "The bounded probe makes attempt-conditioned realized state observable and validates two distinct interventions, but n=20 post-hoc differences and no matched policy-failure slice cannot establish a learning bottleneck.",
            "next_gate": "Freeze development slice definitions, collect matching telemetry from repeated rollouts, then test whether camera/clutter slices are associated with failures before training probes."
        },
        "frozen_input_manifest": {str(path): {"bytes": path.stat().st_size, "sha256": digest(path)}
                                  for path in sorted(inputs)},
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# 第一阶段有界采集能力结果", "",
             "本报告是训练前诊断，不是 policy 得分或候选胜出结论。", "",
             "| profile | 尝试 | 合格 | 产出率（Wilson 95% CI） | 场景/数据/全视频门 | 耗时 |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for row in result["profiles"]:
        interval = row["yield_wilson_95"]
        checks = "/".join("通过" if row[key] else "失败" for key in
                          ("scene_evidence_passed", "data_audit_passed", "full_video_decode_passed"))
        lines.append(f"| `{row['profile']}` | {row['attempts']} | {row['accepted']} | "
                     f"{row['yield']:.1%} [{interval[0]:.1%}, {interval[1]:.1%}] | "
                     f"{checks} | {row['elapsed_seconds']:.1f}s |")
    checks = result["candidate_checks"]
    lines += ["", "## 可解释结论", "",
              f"- 相机切片真实内外参发生变化：{checks['targeted_camera_realized_geometry_varies']}。",
              f"- 干扰物切片每次实际存在两个干扰物：{checks['targeted_clutter_realized_two_distractors']}。",
              f"- 两个非同义候选及全部准入门可用：{checks['two_non_synonymous_measurable_options'] and checks['all_collection_and_data_gates_passed']}。",
              "- 产出率差异只说明专家在不同生成配置下能力不同；尚未证明这些切片对应 policy 关键失败，也未证明定向数据能提高成功率。",
              "- 下一门是先冻结开发评测切片，再采集带相机/干扰物遥测的重复 rollout；未通过失败相关性门时不启动 5K 训练。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = analyze_probe_root(args.root, args.profile)
    atomic_json(args.output, result)
    if args.report:
        args.report.write_text(render_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
