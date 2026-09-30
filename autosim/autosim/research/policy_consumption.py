"""Verify a native evaluator's own policy-load event against frozen policy bytes.

The research agent chooses how to expose the event at the repository's actual loader.
The executor only verifies the event's relationship to the scored artifact; it never
guesses a benchmark-specific checkpoint directory or policy class name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .experiment_bundle import artifact_identity


MARKER = "AUTOSIM_POLICY_LOADED "
ROLLOUT_MARKER = "AUTOSIM_ROLLOUT_COMPLETED "
METRIC_MARKER = "AUTOSIM_METRIC_REPORTED "


def verify_policy_consumption(log: Path, policy: Path) -> dict[str, Any]:
    """Accept one exact native loader event, or return an explicit unverified reason."""
    policy = Path(policy).resolve(strict=True)
    identity_field, expected = artifact_identity(policy)
    log = Path(log).resolve(strict=True)
    if not log.is_file() or log.stat().st_size > 64 * 1024 * 1024:
        return {"status": "unverified", "reason": "evaluation log is missing or too large"}
    matches: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(MARKER):
            continue
        try:
            row = json.loads(line[len(MARKER):])
            if isinstance(row, dict):
                matches.append(row)
        except ValueError:
            continue
    if not matches:
        return {"status": "unverified", "reason":
                "native evaluator emitted no policy-load event"}
    for observed in matches:
        try:
            actual_path = Path(str(observed.get("path") or "")).resolve(strict=True)
        except (OSError, ValueError, RuntimeError):
            actual_path = Path("/__autosim_missing_policy__")
        if actual_path != policy or observed.get(identity_field) != expected:
            return {"status": "mismatch", "reason":
                    "native loader event does not match the frozen candidate identity",
                    "identity_field": identity_field, "expected": expected,
                    "observed": str(observed.get(identity_field) or "")[:128]}
    return {"status": "verified", "reason": "native loader event matched frozen bytes",
            "identity_field": identity_field, "sha256": expected,
            "events": len(matches), "log_ref": str(log)}


def verify_rollout_evidence(log: Path, *, policy_sha256: str,
                            min_episodes: int = 1) -> dict[str, Any]:
    """Require completed native episodes tied to the same loaded policy identity."""
    if not isinstance(min_episodes, int) or min_episodes < 1:
        raise ValueError("minimum completed episodes must be positive")
    log = Path(log).resolve(strict=True)
    if not log.is_file() or log.stat().st_size > 64 * 1024 * 1024:
        return {"status": "unverified", "reason": "rollout log is missing or too large"}
    identities: set[str] = set()
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(ROLLOUT_MARKER):
            continue
        try:
            row = json.loads(line[len(ROLLOUT_MARKER):])
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        episode = str(row.get("episode_id") or "")
        if row.get("policy_sha256") != policy_sha256:
            return {"status": "mismatch", "reason":
                    "completed rollout names a different loaded policy"}
        if not episode or len(episode) > 160:
            return {"status": "unverified", "reason":
                    "completed rollout lacks a bounded episode identity"}
        identities.add(episode)
    if len(identities) < min_episodes:
        return {"status": "unverified", "reason":
                f"only {len(identities)} distinct completed episodes observed",
                "episodes_completed": len(identities)}
    return {"status": "verified", "episodes_completed": len(identities),
            "episode_ids": sorted(identities), "policy_sha256": policy_sha256,
            "log_ref": str(log)}


def verify_metric_lineage(log: Path, *, policy_sha256: str,
                          episode_ids: list[str], value: float) -> dict[str, Any]:
    """Bind the parsed native metric to exactly the witnessed completed episodes."""
    log = Path(log).resolve(strict=True)
    if not log.is_file() or log.stat().st_size > 64 * 1024 * 1024:
        return {"status": "unverified", "reason": "metric log is missing or too large"}
    matches = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(METRIC_MARKER):
            try:
                row = json.loads(line[len(METRIC_MARKER):])
                if isinstance(row, dict):
                    matches.append(row)
            except ValueError:
                continue
    if len(matches) != 1:
        return {"status": "unverified", "reason":
                "expected exactly one native metric-lineage event"}
    row = matches[0]
    observed = row.get("value")
    try:
        observed = float(observed)
        target = float(value)
    except (TypeError, ValueError):
        return {"status": "unverified", "reason": "metric lineage has no numeric value"}
    if (not math.isfinite(observed) or not math.isfinite(target) or
            row.get("policy_sha256") != policy_sha256 or
            not isinstance(row.get("episode_ids"), list) or
            not all(isinstance(item, str) for item in row["episode_ids"]) or
            sorted(row["episode_ids"]) != sorted(episode_ids) or
            len(row["episode_ids"]) != len(set(row["episode_ids"])) or
            not math.isclose(observed, target, rel_tol=1e-6, abs_tol=1e-8)):
        return {"status": "mismatch", "reason":
                "native metric value, episode set, or policy identity differs"}
    return {"status": "verified", "value": target,
            "episodes_completed": len(episode_ids), "policy_sha256": policy_sha256}
