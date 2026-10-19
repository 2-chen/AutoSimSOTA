#!/usr/bin/env python3
"""Read-only audit of the October 16 external native-port pilot; no new trials.

This is an evidence exporter for one development study, not benchmark routing
logic. Agent-authored summaries are discovery aids, never the numeric authority.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "autoresearch_runs/external_systems_20261016_v1"
DEST = ROOT / "docs/paper/evidence/external_native_port_20261016.json"


def load(path):
    return json.loads(path.read_text())


def artifact(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    name = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {"path": str(name), "sha256": digest.hexdigest()}


def robosyn(path):
    value = load(path)
    records, summary, config = value["episodes"], value["summary"], value["config"]
    n, k = len(records), sum(bool(r["success"]) for r in records)
    if not n or n != summary["episode_count"] or n != config["episode_count"]:
        raise ValueError(f"Incomplete episode count: {path}")
    if k != summary["success_count"] or abs(k / n - summary["success_rate"]) > 1e-9:
        raise ValueError(f"Aggregate mismatch: {path}")
    return {"successes": k, "episodes": n, "success_rate": k / n,
            "setting": config["setting"], "seed": config["seed"],
            "episode_seeds": [r["episode_seed"] for r in records],
            "native_config": config, "evidence": [artifact(path)]}


def enpire_robotwin(path):
    value = load(path)
    records = value["records"]
    n, k = len(records), sum(bool(r["success"]) for r in records)
    if n != 20 or value["episodes"] != n or value["test_num"] != n:
        raise ValueError(f"Incomplete RoboTwin evaluation: {path}")
    if k != value["success"] or abs(k / n - value["success_rate"]) > 1e-9:
        raise ValueError(f"Aggregate mismatch: {path}")
    evidence = [artifact(path)]
    for record in records:
        episode = path.parent / f"ep{record['episode']:03d}" / "result.json"
        result = load(episode)
        if (record["termination"] not in {"truncated", "success", "terminated"}
                or result["termination"] != record["termination"]
                or result["success"] != record["success"]
                or result["verification"]["success"] != record["success"]
                or result["steps"] != record["steps"]):
            raise ValueError(f"Episode result mismatch/error: {episode}")
        evidence.append(artifact(episode))
    return {"successes": k, "episodes": n, "success_rate": k / n,
            "seed": value["seed"], "setting": value["task_config"],
            "episode_seeds": [r["seed"] for r in records],
            "terminations": dict(Counter(r["termination"] for r in records)),
            "evidence": evidence}


def native_robotwin(trial, variant, timestamp):
    log = trial / f"trials/{variant}/eval_{variant}_seed2026.log"
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    ratios = [(int(a), int(b)) for a, b in re.findall(r"Success rate:.*?(\d+)\s*/\s*(\d+)", text)]
    if [b for _, b in ratios] != list(range(1, 21)):
        raise ValueError(f"Missing native episode completions: {log}")
    loaded = re.findall(r"Loaded policy weights from ([^\r\n]+)", text)
    if len(loaded) != 1 or "Loading status: <All keys matched successfully>" not in text:
        raise ValueError(f"Unverified checkpoint loading: {log}")
    checkpoint = Path(loaded[0]).resolve()
    if not (checkpoint.is_relative_to(trial) or checkpoint.is_relative_to(ROOT.parent / "test/RoboTwin")):
        raise ValueError(f"Unexpected checkpoint location: {checkpoint}")
    result = trial / "benchmark/eval_result/beat_block_hammer/ACT/demo_clean/act-beat_block_hammer" / timestamp / "_result.txt"
    k, n = ratios[-1]
    if abs(float(result.read_text().strip().splitlines()[-1]) - k / n) > 1e-9:
        raise ValueError(f"Native text result mismatch: {result}")
    return {"successes": k, "episodes": n, "success_rate": k / n,
            "seed": 2026, "setting": "demo_clean",
            "episode_seeds": [int(v) for v in re.findall(r"Success rate:.*?current seed: (\d+)", text)],
            "loaded_checkpoint": artifact(checkpoint),
            "evidence": [artifact(log), artifact(result)]}


def main():
    report = {"schema_version": 1, "exported_at": datetime.now(timezone.utc).isoformat(),
              "study": str(STUDY.relative_to(ROOT)), "formal_comparison_eligible": False,
              "scope": "Operator-assisted native-port development pilots; no system ranking or SOTA claim",
              "measurements": [], "trials": []}

    def add(arm, name, scope, measurement):
        report["measurements"].append(dict(arm=arm, name=name, scope=scope, **measurement))

    trial = STUDY / "enpire_robosyn"
    submitted = load(trial / "pilot_results.json")
    paths = submitted["native_artifacts"]["native_metrics"]
    add(trial.name, "baseline", "text_diagnostic", robosyn(trial / paths["baseline_dev"]))
    for candidate in submitted["development"]["candidates"]:
        add(trial.name, candidate["name"], "text_diagnostic", robosyn(trial / candidate["native_metrics"]))
    add(trial.name, "baseline_confirmation", "text_diagnostic", robosyn(trial / paths["baseline_confirm"]))
    for probe in submitted["development"].get("exploratory_probes", []):
        add(trial.name, probe["name"], "post_confirmation_exploration_excluded", robosyn(trial / probe["native_metrics"]))

    trial = STUDY / "enpire_robotwin"
    for name in ["baseline_dev20", "cand_bestckpt_dev20", "cand_notemporal_dev20"]:
        add(trial.name, name, "native_port_development", enpire_robotwin(trial / "benchmark/outputs" / name / "summary.json"))

    trial = STUDY / "autoresearch_robotics_robotwin"
    for variant, timestamp in [("baseline", "2026-10-16 19:27:08"), ("cand1_ta", "2026-10-16 19:46:00"),
                               ("cand2_ft", "2026-10-16 20:09:55"), ("cand3_ft_ta", "2026-10-16 20:24:23")]:
        add(trial.name, variant, "native_port_development", native_robotwin(trial, variant, timestamp))
    report["training"] = load(trial / "trials/cand2_ft/train_meta.json")
    report["training"]["evidence"] = artifact(trial / "trials/cand2_ft/train_meta.json")
    report["checkpoint_report_discrepancy"] = (
        "autoresearch_robotics_robotwin pilot_results.json names policy_best.ckpt; "
        "native load logs show policy_last.ckpt for both retrained-policy evaluations. "
        "The actual loaded file is authoritative; best-validation loss is not assigned to it.")

    trial = STUDY / "autoresearch_robotics_robosyn"
    submitted = load(trial / "benchmark/pilot_results.json")
    for name, result in submitted["measured"].items():
        if result.get("metrics_json"):
            add(trial.name, name, "off_protocol_clear_diagnostic", robosyn(trial / "benchmark" / result["metrics_json"]))
    trial = STUDY / "enpire_robosyn_vision"
    submitted = load(trial / "pilot_results.json")
    add(trial.name, "clear_smoke", "off_protocol_clear_smoke", robosyn(trial / submitted["diagnostic_control"]["metrics_path"]))

    for spec in load(STUDY / "manifest.json")["trials"]:
        trial = STUDY / spec["name"]
        state, usage = load(trial / "state.json"), load(trial / "model_usage.json")
        report["trials"].append({"name": trial.name, "upstream_commit": spec["upstream_commit"],
            "launcher_status": state["status"], "model": usage["model"],
            "provider_priced_usd": usage["cost_usd"], "unsettled_held_usd": usage["held_usd"],
            "unknown_reserved_usd": usage["unknown_reserved_usd"], "cost_complete": not usage["unknown"],
            "requests": usage["requests"],
            "image_request_receipts": sum(r.get("request_profile", {}).get("input_modality") == "image_and_text" for r in usage["receipts"]),
            "conservative_device_occupancy_seconds": state["conservative_device_occupancy_seconds"],
            "evidence": [artifact(trial / "state.json"), artifact(trial / "model_usage.json")]})
    report["sources_unchanged"] = {name: load(STUDY / f"source_{name}_after.json")["unchanged"] for name in ["robosyn", "robotwin"]}
    report["unresolved"] = [
        "Both image-enabled RoboSyn random arms lack a measured baseline; native material initialization failed/hung. Earlier text arm succeeded: environment equivalence/root cause unresolved.",
        "RoboTwin accepted seed vectors differ across some runs; a common seed label does not establish paired initial states.",
        "RoboTwin arms start with different temporal_agg defaults; not a frozen matched-head-to-head comparison.",
        "Launcher missed benchmark/pilot_results.json; needs_followup is not evidence that the agent produced no report.",
        "RUN.md native metric index omits ENPIRE summary.json and RoboSyn research_out; index counts are not scientific results.",
        "These trials do not verify targeted new-data acquisition, batch consumption, or held-out candidate improvement.",
    ]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(DEST.relative_to(ROOT)), "audited_measurements": len(report["measurements"]),
                      "priced_usd": sum(r["provider_priced_usd"] for r in report["trials"]),
                      "sources_unchanged": report["sources_unchanged"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
