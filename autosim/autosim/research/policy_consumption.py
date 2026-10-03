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


def identity_contract() -> dict[str, Any]:
    """Publish exact verifier syntax, not expected hashes or fabricated evidence."""
    return {
        "authority": "executor interface only; not native loading or score evidence",
        "loaded_marker": MARKER,
        "loaded_fields": {"path": "absolute path actually consumed by native loader",
                          "file": "content_sha256", "directory": "sha256"},
        "rollout_marker": ROLLOUT_MARKER,
        "rollout_fields": ["episode_id", "policy_sha256"],
        "metric_marker": METRIC_MARKER,
        "metric_fields": ["policy_sha256", "episode_ids", "value"],
        "instruction": (
            "Emit at the real successful loader, completed episode and metric aggregation sites. "
            "Use the exact field names: episode_index is NOT episode_id or episode_ids. "
            "Completed IDs must be nonempty strings, unique in this evaluation. "
            "Directory hash is the canonical JSON manifest below, NOT a custom concatenation "
            "of paths and hashes. All three witnesses use the same digest. Recompute from "
            "actual loaded bytes; never echo a supplied expected digest. CPU-test syntax and "
            "digest recipe before a minimal native smoke; preserve actions and scoring."),
        "identity_recipe_python": '''def loaded_artifact_identity(path):
    import hashlib, json
    from pathlib import Path
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symbolic policy artifact")
    def file_sha256(file):
        hasher = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()
    if path.is_file():
        return "content_sha256", file_sha256(path)
    if not path.is_dir() or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("missing directory or symbolic entry")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("empty artifact directory")
    manifest = [(str(p.relative_to(path)), file_sha256(p)) for p in files]
    return "sha256", hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
''',
    }


def audit_identity_schema(text: str, *, policy_kind: str | None = None) -> dict[str, Any]:
    """Pre-freeze interface feedback only; never certify loading or performance."""
    records = {MARKER: [], ROLLOUT_MARKER: [], METRIC_MARKER: []}
    issues = []
    for line in text.splitlines():
        for marker in records:
            if not line.startswith(marker):
                continue
            try:
                row = json.loads(line[len(marker):])
                if not isinstance(row, dict):
                    raise ValueError('expected JSON object')
                records[marker].append(row)
            except (ValueError, TypeError):
                issues.append(marker.strip() + ' must contain a JSON object')
    if not any(records.values()) and not issues:
        return {'status':'unavailable', 'issues':[],
                'authority':'schema inspection only; not consumption or scoring evidence'}
    loaded, rollouts, metrics = records[MARKER], records[ROLLOUT_MARKER], records[METRIC_MARKER]
    for row in loaded:
        fields = ['sha256'] if policy_kind == 'directory' else ['content_sha256'] if policy_kind == 'file' else ['sha256', 'content_sha256']
        if not isinstance(row.get('path'), str) or not row['path']:
            issues.append('AUTOSIM_POLICY_LOADED requires the actual loaded path')
        if not any(isinstance(row.get(field), str) and len(row[field]) == 64
                   and all(c in '0123456789abcdef' for c in row[field]) for field in fields):
            issues.append('AUTOSIM_POLICY_LOADED requires ' + ' or '.join(fields)
                          + '; use native_identity_contract directory recipe, not a custom serialization')
    for row in rollouts:
        episode = row.get('episode_id')
        if not isinstance(episode, str) or not episode or len(episode) > 160:
            issues.append('AUTOSIM_ROLLOUT_COMPLETED requires a nonempty episode_id string; episode_index is not that field')
        if not isinstance(row.get('policy_sha256'), str):
            issues.append('AUTOSIM_ROLLOUT_COMPLETED requires policy_sha256')
    for row in metrics:
        ids = row.get('episode_ids')
        if (not isinstance(ids, list) or not ids or
                not all(isinstance(item, str) and item for item in ids)):
            issues.append('AUTOSIM_METRIC_REPORTED requires episode_ids as a nonempty list of strings; episode_index is not that field')
        value = row.get('value')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            issues.append('AUTOSIM_METRIC_REPORTED requires a finite numeric value')
    return {'status':'incompatible' if issues else 'schema_observed',
            'issues':list(dict.fromkeys(issues))[:8],
            'authority':'schema inspection only; not consumption or scoring evidence',
            'instruction':'Repair logging at real native sites using native_identity_contract before freezing a new research session; do not alter actions, episodes, success or scoring. A compatible schema still requires independent byte/rollout/metric verification.'}


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
