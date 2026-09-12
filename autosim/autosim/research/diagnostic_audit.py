"""Freeze historical evidence and audit collection-to-training observability.

This module never upgrades heuristic evidence to validated evidence.  It reports
what can and cannot be identified from existing artifacts so a new experiment
cannot silently inherit unsupported claims.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import atomic_json, digest


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{number}")
        rows.append(value)
    return rows


def audit_collection_scene_evidence(manifest_path: Path, trace_path: Path) -> dict[str, Any]:
    """Verify that every recorded attempt has realized state joined by seed."""
    manifest = _read(manifest_path)
    attempts = list(manifest.get("attempts", []))
    rows = _read_jsonl(trace_path) if trace_path.is_file() else []
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("seed") is not None:
            by_seed[int(row["seed"])].append(row)

    attempt_seeds = [int(row["seed"]) for row in attempts]
    missing_seeds = [seed for seed in attempt_seeds if seed not in by_seed]
    joined = [by_seed[seed][-1] for seed in attempt_seeds if seed in by_seed]
    non_measurable_seeds = [
        seed for seed in attempt_seeds if seed in by_seed
        and not (by_seed[seed][-1].get("entities")
                 or by_seed[seed][-1].get("robot_qpos") is not None)
    ]
    metadata_mismatch_seeds = [
        seed for seed in attempt_seeds if seed in by_seed and (
            by_seed[seed][-1].get("task") != manifest.get("task")
            or by_seed[seed][-1].get("requested_profile") != manifest.get("profile")
            or by_seed[seed][-1].get("privileged_training_diagnostics_only") is not True
        )
    ]
    duplicate_attempt_seed_rows = {
        str(seed): len(by_seed[seed]) for seed in attempt_seeds
        if len(by_seed.get(seed, [])) > 1
    }
    extra_reset_seeds = sorted(set(by_seed) - set(attempt_seeds))
    passed = bool(attempts) and not (
        missing_seeds or non_measurable_seeds or metadata_mismatch_seeds
        or duplicate_attempt_seed_rows
    )
    profile = manifest.get("profile")
    readback_checks = []
    if profile == "targeted_camera":
        readback_checks = [bool(row.get("cameras")) for row in joined]
    elif profile == "targeted_clutter":
        readback_checks = [bool(row.get("active_distractors")) for row in joined]
    elif profile == "targeted_recovery":
        readback_checks = [row.get("robot_qpos") is not None and
                           bool(row.get("active_distractors")) for row in joined]
    elif profile == "targeted_appearance":
        # The simulator wrapper currently exposes light pose but not realized
        # color/intensity or material parameters.
        readback_checks = []
    elif profile in {"full_random", "composite_hard"}:
        readback_checks = [bool(row.get("entities")) for row in joined]
    factor_status = ("verified" if readback_checks and all(readback_checks)
                     else "failed" if readback_checks
                     else "unknown")
    return {
        "schema_version": 1,
        "kind": "collection_scene_evidence_audit",
        "passed": passed,
        "task": manifest.get("task"),
        "requested_profile": manifest.get("profile"),
        "attempt_count": len(attempts),
        "trace_reset_count": len(rows),
        "joined_attempt_count": len(joined),
        "missing_attempt_seeds": missing_seeds,
        "non_measurable_attempt_seeds": non_measurable_seeds,
        "metadata_mismatch_seeds": metadata_mismatch_seeds,
        "duplicate_attempt_seed_rows": duplicate_attempt_seed_rows,
        "extra_reset_seeds_not_attempts": extra_reset_seeds,
        "extra_resets_are_excluded_from_attempt_statistics": True,
        "unavailable_realized_parameter_fields": sorted({
            str(field) for row in joined
            for field in row.get("unavailable_realized_parameters", [])
        }),
        "manifest_sha256": digest(manifest_path),
        "trace_sha256": digest(trace_path) if trace_path.is_file() else None,
        "policy_observation_unchanged": True,
        "requested_factor_readback": {
            "status": factor_status,
            "checked_attempts": len(readback_checks),
            "reason": ("realized factor fields present for every joined attempt"
                       if factor_status == "verified" else
                       "realized appearance/material values are unavailable"
                       if profile == "targeted_appearance" else
                       "one or more realized factor fields are missing"),
        },
    }


def _expected_masses(mixture: dict[str, Any]) -> list[float | None]:
    entries = list(mixture.get("datasets", []))
    if not entries:
        return []
    sampling = mixture.get("sampling", "proportional_to_frames")
    if isinstance(sampling, str):
        if sampling != "proportional_to_frames":
            return [None for _ in entries]
        frames = [int(row.get("frame_count", 0)) for row in entries]
        total = sum(frames)
        return [value / total if total else None for value in frames]
    if sampling.get("strategy") != "stratified_phase":
        return [None for _ in entries]
    profile_masses = dict(sampling.get("profile_masses", {}))
    if not profile_masses:
        frames = [int(row.get("frame_count", 0)) for row in entries]
        total = sum(frames)
        return [value / total if total else None for value in frames]
    frames_by_profile: dict[str, int] = defaultdict(int)
    for row in entries:
        frames_by_profile[str(row.get("profile", "unknown"))] += int(row.get("frame_count", 0))
    present = {str(row.get("profile", "unknown")) for row in entries}
    denominator = sum(float(profile_masses.get(profile, 0.0)) for profile in present)
    if denominator <= 0 or any(profile not in profile_masses for profile in present):
        return [None for _ in entries]
    return [
        float(profile_masses[str(row.get("profile", "unknown"))])
        / denominator
        * int(row.get("frame_count", 0))
        / frames_by_profile[str(row.get("profile", "unknown"))]
        for row in entries
    ]


def _source_rows(mixture: dict[str, Any]) -> list[dict[str, Any]]:
    masses = _expected_masses(mixture)
    return [
        {
            "root": row.get("root"),
            "profile": row.get("profile", "unknown"),
            "source_kind": row.get("source_kind", row.get("role", "unknown")),
            "episodes": row.get("episode_count"),
            "frames": row.get("frame_count"),
            "expected_sampling_mass": masses[index] if index < len(masses) else None,
        }
        for index, row in enumerate(mixture.get("datasets", []))
    ]


def _collection_row(path: Path) -> tuple[dict[str, Any], list[Path]]:
    data = _read(path)
    attempts = list(data.get("attempts", []))
    saved = sum(bool(row.get("saved")) for row in attempts)
    reasons = Counter(str(row.get("reason", "unknown")) for row in attempts)
    extra_attempt_fields = sorted(
        set().union(*(set(row) for row in attempts)) - {"seed", "saved", "reason"}
    ) if attempts else []
    scene_fields = sorted(
        key for key in extra_attempt_fields
        if key in {"scene", "scene_parameters", "scene_slice", "randomization", "initialization"}
    )
    scene_path = path.parent / "scene_resets.jsonl"
    scene_rows = _read_jsonl(scene_path) if scene_path.is_file() else []
    scene_by_seed = {
        int(row["seed"]): row for row in scene_rows if row.get("seed") is not None
    }
    joined_scene_rows = [scene_by_seed.get(int(row["seed"])) for row in attempts]
    joined_scene_count = sum(row is not None for row in joined_scene_rows)
    measurable_scene_count = sum(
        bool(row and (row.get("entities") or row.get("robot_qpos") is not None))
        for row in joined_scene_rows
    )
    inline_scene_complete = bool(attempts) and all(
        any(key in row for key in scene_fields) for row in attempts
    )
    sidecar_scene_complete = bool(attempts) and joined_scene_count == len(attempts)
    scene_outcomes_identifiable = (
        inline_scene_complete
        or (sidecar_scene_complete and measurable_scene_count == len(attempts))
    )
    audit_path = path.parent / "data_audit.json"
    audit = _read(audit_path) if audit_path.is_file() else None
    round_dir = path.parents[2]
    arm = path.parents[1].name
    mixture_path = round_dir / f"{arm}_mixture.json"
    mixture = _read(mixture_path) if mixture_path.is_file() else None
    pipeline_path = path.parents[1] / "training_data_audit.json"
    exposure_path = path.parents[1] / "training_exposure_audit.json"
    recipe_paths = sorted(path.parents[1].glob("recipe_*.json"))
    recipe = _read(recipe_paths[-1]) if recipe_paths else None
    inputs = [path]
    if scene_path.is_file():
        inputs.append(scene_path)
    inputs += [candidate for candidate in (audit_path, mixture_path, pipeline_path, exposure_path) if candidate.is_file()]
    inputs += recipe_paths
    row = {
        "task": data.get("task"),
        "round": round_dir.name,
        "arm": arm,
        "requested_profile": data.get("profile"),
        "status": data.get("status"),
        "attempts": len(attempts),
        "accepted_episodes": saved,
        "attempt_yield": saved / len(attempts) if attempts else None,
        "attempt_outcomes": dict(sorted(reasons.items())),
        "attempt_scene_parameter_fields": scene_fields,
        "scene_evidence": {
            "artifact_present": scene_path.is_file(),
            "artifact": str(scene_path.resolve()) if scene_path.is_file() else None,
            "reset_rows": len(scene_rows),
            "attempts_joined_by_seed": joined_scene_count,
            "attempts_with_measurable_realized_state": measurable_scene_count,
            "complete_for_attempt_outcomes": scene_outcomes_identifiable,
            "unsupported_realized_parameter_fields": sorted({
                str(field)
                for row in joined_scene_rows if row
                for field in row.get("unavailable_realized_parameters", [])
            }),
        },
        "admission": {
            "artifact_present": audit is not None,
            "passed": audit.get("passed") if audit else None,
            "episodes": (audit.get("dataset") or {}).get("total_episodes") if audit else None,
            "errors": audit.get("errors") if audit else None,
        },
        "training": {
            "mixture_present": mixture is not None,
            "sampling_strategy": mixture.get("sampling") if mixture else None,
            "sources": _source_rows(mixture) if mixture else [],
            "configured_pipeline_audit_present": pipeline_path.is_file(),
            "actual_exposure_audit_present": exposure_path.is_file(),
            "actual_exposure": _read(exposure_path) if exposure_path.is_file() else None,
            "recipe_content_ids_present": bool(recipe and recipe.get("training_data_content_ids")),
        },
        "identifiability": {
            "requested_to_accepted_overall_yield": bool(attempts),
            "requested_slice_to_accepted_slice": scene_outcomes_identifiable,
            "accepted_source_to_training_exposure": exposure_path.is_file(),
            "failure_slice_to_learning_effect": False,
        },
        "source_manifest": str(path.resolve()),
    }
    return row, inputs


def audit_history(research_root: Path, tasks: list[str]) -> dict[str, Any]:
    research_root = research_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    input_paths: set[Path] = set()
    for task in tasks:
        task_root = research_root / task
        for path in sorted(task_root.glob("round_*/auto/collection/collection.json")) + sorted(
            task_root.glob("round_*/random/collection/collection.json")
        ):
            row, inputs = _collection_row(path)
            rows.append(row)
            input_paths.update(inputs)
    rows.sort(key=lambda row: (str(row["task"]), str(row["round"]), str(row["arm"])))

    by_task = []
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        profiles = sorted({str(row["requested_profile"]) for row in task_rows})
        scene_identifiable = any(
            row["identifiability"]["requested_slice_to_accepted_slice"] for row in task_rows
        )
        exposure_identifiable = any(
            row["identifiability"]["accepted_source_to_training_exposure"] for row in task_rows
        )
        by_task.append(
            {
                "task": task,
                "collection_runs": len(task_rows),
                "requested_profiles": profiles,
                "attempts": sum(int(row["attempts"]) for row in task_rows),
                "accepted_episodes": sum(int(row["accepted_episodes"]) for row in task_rows),
                "scene_slice_outcomes_identifiable": scene_identifiable,
                "actual_training_exposure_identifiable": exposure_identifiable,
                "natural_selection_bias_status": (
                    "observable" if scene_identifiable and exposure_identifiable
                    else "not_identifiable_from_historical_artifacts"
                ),
            }
        )

    manifest = {
        str(path): {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(input_paths)
    }
    return {
        "schema_version": 1,
        "kind": "collection_to_training_diagnostic_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_root": str(research_root),
        "tasks": by_task,
        "runs": rows,
        "frozen_input_manifest": manifest,
        "conclusion": {
            "historical_natural_bias_proven": all(
                row["natural_selection_bias_status"] == "observable" for row in by_task
            ) if by_task else False,
            "prospective_instrumentation_required": any(
                row["natural_selection_bias_status"] != "observable" for row in by_task
            ),
            "reason": (
                "Historical artifacts expose aggregate expert yield and configured sampling, but "
                "scene-slice attempt outcomes and actual DataLoader exposure are incomplete."
            ),
        },
        "limitations": [
            "requested profile labels are not measurements of realized scene parameters",
            "configured sampling mass is not actual batch exposure",
            "aggregate expert yield cannot establish slice-conditioned selection bias",
            "this audit does not estimate causal policy benefit",
        ],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# 第一阶段：历史采集到训练证据审计",
        "",
        f"生成时间：{audit['created_at']}。该报告冻结并汇总现有产物，不代表新训练或评测。",
        "",
        "## 结论",
        "",
        "历史产物可以核算总体专家产出率和配置期望采样占比，但缺少逐尝试场景参数与实际 DataLoader 曝光，因而不能据此证明关键切片发生了自然选择偏移，也不能证明它导致 policy 失败。新训练已增加前瞻性来源曝光审计；场景切片仍需在有界探针中记录。",
        "",
        "## 任务汇总",
        "",
        "| 任务 | 采集运行 | 尝试 | 准入轨迹 | 请求 profile | 自然偏移可识别 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in audit["tasks"]:
        lines.append(
            f"| {row['task']} | {row['collection_runs']} | {row['attempts']} | "
            f"{row['accepted_episodes']} | {', '.join(row['requested_profiles']) or '无'} | "
            f"{row['natural_selection_bias_status']} |"
        )
    lines += ["", "## 逐运行证据", "", "| 任务/轮次/臂 | 请求 | 尝试→准入 | 数据审计 | 期望新增曝光 | 实际曝光 |",
              "| --- | --- | ---: | --- | ---: | --- |"]
    for row in audit["runs"]:
        new_mass = sum(
            float(source["expected_sampling_mass"] or 0.0)
            for source in row["training"]["sources"]
            if source["source_kind"] != "official_full_random"
        )
        lines.append(
            f"| {row['task']}/{row['round']}/{row['arm']} | {row['requested_profile']} | "
            f"{row['attempts']}→{row['accepted_episodes']} | "
            f"{row['admission']['passed']} | {new_mass:.4f} | "
            f"{'有' if row['training']['actual_exposure_audit_present'] else '缺失'} |"
        )
    lines += [
        "",
        "## 边界与下一动作",
        "",
        "- 旧实验所有采集请求均为 `full_random`，不能作为定向采集效果证据。",
        "- 期望采样占比来自 mixture；它不等于实际进入训练循环的比例。",
        "- 在新有界采集前先定义少量可记录切片，并将参数与每次尝试的 seed/结局绑定。",
        "- 新训练使用 `training_exposure_audit.json` 记录 DataLoader 实际产出；该记录仍不能证明一次已产出的 batch 必然完成参数更新。",
        "- 未取得上述前瞻性证据前，核心假设状态保持“不可识别”，不能升级为已支持。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", required=True, type=Path)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    audit = audit_history(args.research_root, args.task)
    atomic_json(args.output, audit)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps(audit["conclusion"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
