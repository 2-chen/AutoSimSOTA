"""Validate policy-induced ClickBell correction trajectories and lineage."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .bounded_probe_analysis import _wilson
from .common import atomic_json, digest, now, read_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _flat(value: Any) -> list[float]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [float(item) for item in value]


def audit_correction_probe(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    paths = {name: directory / relative for name, relative in {
        "manifest": "collection.json", "bounded": "bounded_collection_result.json",
        "scene": "scene_resets.jsonl", "scene_audit": "scene_evidence_audit.json",
        "data_audit": "data_audit.json", "video_audit": "video_integrity_v1.json",
        "process": "process/process.json",
    }.items()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing correction artifacts: {missing}")
    manifest, bounded = read_json(paths["manifest"]), read_json(paths["bounded"])
    attempts = list(manifest.get("attempts", []))
    saved = [row for row in attempts if row.get("saved")]
    reasons = Counter(str(row.get("reason")) for row in attempts)
    lineage = manifest.get("lineage") or {}
    parent = Path(str(lineage.get("parent_checkpoint", "")))
    lineage_valid = (parent.is_dir() and (parent / "model.safetensors").is_file()
                     and lineage.get("parent_model_sha256") == digest(parent / "model.safetensors"))

    scene_by_seed = {int(row["seed"]): row for row in _jsonl(paths["scene"])
                     if row.get("seed") is not None}
    dataset = Path(bounded["dataset_root"])
    import pyarrow.parquet as pq
    displacement_rows = []
    success_seeds = [int(seed) for seed in manifest.get("successful_episode_seeds", [])]
    parquet_paths = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    for episode_index, seed in enumerate(success_seeds):
        matches = []
        for path in parquet_paths:
            table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state"])
            episode_ids = table["episode_index"].to_pylist()
            frame_ids = table["frame_index"].to_pylist()
            states = table["observation.state"].to_pylist()
            matches.extend(states[i] for i, (episode, frame) in enumerate(zip(episode_ids, frame_ids))
                           if int(episode) == episode_index and int(frame) == 0)
        if len(matches) != 1:
            raise ValueError(f"episode {episode_index} has {len(matches)} initial states")
        reset = _flat(scene_by_seed[seed]["robot_qpos"])
        initial = _flat(matches[0])
        if len(reset) != 14 or len(initial) != 14:
            raise ValueError("correction/reset state is not 14D")
        delta = [left - right for left, right in zip(initial, reset)]
        displacement_rows.append({
            "episode_index": episode_index, "seed": seed,
            "l2": math.sqrt(sum(value * value for value in delta)),
            "max_abs": max(abs(value) for value in delta),
        })

    suffix_attempts = reasons["saved_policy_induced_expert_correction"] + reasons["expert_correction_failed"]
    checks = {
        "mode_is_policy_correction": manifest.get("collection_mode") == "policy_correction",
        "full_random_profile": manifest.get("profile") == "full_random",
        "bounded_attempts_exact": len(attempts) == int(bounded["attempt_budget"]) == 20,
        "at_least_one_saved": len(saved) >= 1,
        "saved_count_matches_dataset": len(saved) == len(success_seeds) == len(displacement_rows),
        "saved_prefix_positive_and_expert_valid": all(
            int(row.get("policy_prefix_steps", 0)) > 0 and row.get("expert_plan_valid") is True
            for row in saved),
        "parent_lineage_valid": lineage_valid,
        "seed_joined_scene_evidence": bool(read_json(paths["scene_audit"])["passed"]),
        "numeric_and_schema_audit": bool(read_json(paths["data_audit"])["passed"]),
        "full_video_decode": bool(read_json(paths["video_audit"])["passed"]),
        "learner_state_displacement_observed": bool(displacement_rows) and all(
            row["max_abs"] > 1e-3 for row in displacement_rows),
    }
    return {
        "schema_version": 1, "kind": "click_bell_policy_correction_probe_audit",
        "created_at": now(), "directory": str(directory), "passed": all(checks.values()),
        "checks": checks, "attempts": len(attempts), "saved_corrections": len(saved),
        "attempt_outcomes": dict(sorted(reasons.items())),
        "overall_saved_yield": len(saved) / len(attempts),
        "overall_saved_yield_wilson_95": _wilson(len(saved), len(attempts)),
        "expert_suffix_attempts_after_unsuccessful_parent_prefix": suffix_attempts,
        "expert_suffix_saved_yield": len(saved) / suffix_attempts if suffix_attempts else None,
        "learner_state_displacement": {
            "rows": displacement_rows,
            "l2_min": min(row["l2"] for row in displacement_rows),
            "l2_max": max(row["l2"] for row in displacement_rows),
            "max_abs_min": min(row["max_abs"] for row in displacement_rows),
            "comparison": "first recorded expert-suffix state minus same-seed post-reset state",
        },
        "elapsed_seconds": float(read_json(paths["process"])["elapsed_seconds"]),
        "scope": "Capability and data-integrity evidence only; no policy improvement, cross-task, official-score, or SOTA claim.",
        "frozen_input_manifest": {str(path): {"bytes": path.stat().st_size, "sha256": digest(path)}
                                  for path in sorted(paths.values())},
    }


def render_markdown(result: dict[str, Any]) -> str:
    low, high = result["overall_saved_yield_wilson_95"]
    displacement = result["learner_state_displacement"]
    return "\n".join([
        "# ClickBell 纠正轨迹能力审计", "",
        f"- 20 次固定尝试中保存 {result['saved_corrections']} 条纠正轨迹："
        f"{result['overall_saved_yield']:.1%}（Wilson 95% CI {low:.1%}–{high:.1%}）。",
        f"- 排除 parent 已自行成功的尝试后，专家后缀保存率：{result['expert_suffix_saved_yield']:.1%}。",
        f"- 初始纠正状态相对同 seed reset 状态的 L2 范围："
        f"{displacement['l2_min']:.4f}–{displacement['l2_max']:.4f} rad（关节表示）。",
        f"- 场景 seed、父 checkpoint、14 维标签、三路 RGB 和全视频解码总门：{result['passed']}。",
        "- 该结果只证明 ClickBell 可生成真实 learner-state 专家后缀；尚未证明加入训练会提升成功率。", ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit_correction_probe(args.directory)
    atomic_json(args.output, result)
    if args.report:
        args.report.write_text(render_markdown(result), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
