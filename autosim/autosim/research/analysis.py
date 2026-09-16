"""Evidence-first failure summaries; progress heuristics are not success judges."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import atomic_json, read_json


def analyze(evaluation: Path, output: Path | None = None) -> dict:
    import json

    metrics = read_json(evaluation / "evaluation_metrics.json")
    if metrics.get("purpose") not in {"smoke", "development", "selection_validation"}:
        raise ValueError("final/unknown evaluation data cannot enter failure-driven research")
    traces = defaultdict(list)
    trace_path = Path(metrics.get("artifact_directory", evaluation)) / "telemetry.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            row = json.loads(line)
            traces[int(row["seed"])].append(row)
    # Measure, do not label.  This used to bucket every failure into one of three
    # names chosen here (large_object_height_drop / little_object_motion /
    # incomplete_after_motion), which meant the set of recognisable failures was
    # fixed in advance for every task at once.  The numbers below are what the
    # controller reasons from; naming and thresholding them is its job.
    measurements = []
    for episode in metrics["episodes"]:
        rows = traces[episode["episode_seed"]]
        max_move, max_drop, max_joint = 0.0, 0.0, 0.0
        missing = sum(len(r.get("missing", [])) for r in rows)
        observed = 0
        if rows:
            initial = rows[0]["entities"]
            for row in rows[1:]:
                for name, data in row["entities"].items():
                    if name not in initial or "pose" not in data:
                        continue
                    start = np.asarray(initial[name]["pose"]).reshape(-1, 4, 4)[0, :3, 3]
                    end = np.asarray(data["pose"]).reshape(-1, 4, 4)[0, :3, 3]
                    observed += 1
                    max_move = max(max_move, float(np.linalg.norm(end - start)))
                    max_drop = max(max_drop, float(start[2] - end[2]))
                    if "qpos" in data and "qpos" in initial[name]:
                        max_joint = max(max_joint, float(np.abs(np.asarray(data["qpos"]) -
                                                               np.asarray(initial[name]["qpos"])).max()))
        measurements.append({"episode_seed": episode["episode_seed"],
                             "success": bool(episode["success"]),
                             "max_object_displacement_m": max_move,
                             "max_height_drop_m": max_drop,
                             "max_articulation_change": max_joint,
                             "missing_measurements": missing,
                             "measured": bool(observed),
                             "confidence": "low" if missing or not observed else "heuristic"})
    result = {"source": str(evaluation), "purpose": metrics["purpose"], "summary": metrics["summary"],
              "measurements": measurements,
              "limitations": ["progress observations are not official stage labels",
                              "10-step sampling may miss transient contacts",
                              "visual causes cannot be inferred from motion alone"]}
    if output:
        atomic_json(output, result)
    return result
