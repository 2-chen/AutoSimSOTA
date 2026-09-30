"""Index actual run media without asserting an unproven policy/episode identity."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import shutil
import subprocess
import re

from .common import atomic_json, digest, now, read_json

MEDIA_SUFFIXES = frozenset({".mp4", ".webm", ".mov", ".avi", ".gif"})
_MAX_PREVIEWS = 3


def _preview(root: Path, relative: str, parent_sha256: str) -> dict[str, Any]:
    """Extract a bounded real video frame; never synthesize a simulation image."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"status": "unavailable", "reason": "ffmpeg_unavailable"}
    if not re.fullmatch(r"[a-f0-9]{64}", parent_sha256):
        return {"status": "unavailable", "reason": "invalid_media_digest"}
    video = (root / relative).resolve()
    if (not video.is_relative_to(root) or not video.is_file() or video.is_symlink()):
        return {"status": "unavailable", "reason": "media_path_unavailable"}
    target = root / "media" / "previews" / f"{parent_sha256[:24]}.jpg"
    if ((root / "media").is_symlink() or target.parent.is_symlink()
            or not target.parent.resolve().is_relative_to(root.resolve()) or target.is_symlink()):
        return {"status": "unavailable", "reason": "unsafe_preview_destination"}
    if target.is_file() and not target.is_symlink() and 100 <= target.stat().st_size <= 2 * 1024**2:
        return {"status": "decoded_frame", "path": target.relative_to(root).as_posix(),
                "parent_sha256": parent_sha256, "sha256": digest(target)}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.jpg")
    if temporary.is_symlink():
        return {"status": "unavailable", "reason": "unsafe_preview_destination"}
    try:
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(video),
                        "-frames:v", "1", "-vf", "scale=480:-2", "-y", str(temporary)],
                       check=True, capture_output=True, timeout=20)
        if not 100 <= temporary.stat().st_size <= 2 * 1024**2:
            raise ValueError("preview_size_out_of_bounds")
        os.replace(temporary, target)
        return {"status": "decoded_frame", "path": target.relative_to(root).as_posix(),
                "parent_sha256": parent_sha256, "sha256": digest(target)}
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "reason": "video_decode_failed"}
    finally:
        temporary.unlink(missing_ok=True)


def capture_attempt(root: Path, stage_directory: Path, *, since: float,
                    max_files: int = 64, max_hash_bytes: int = 256 * 1024**2
                    ) -> list[dict[str, Any]]:
    """Attribute media created in this stage's own output to its attempt, not an episode."""
    root, stage_directory = Path(root).resolve(), Path(stage_directory).resolve()
    if not stage_directory.is_relative_to(root) or not stage_directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for inspected, path in enumerate(stage_directory.rglob("*")):
        if len(rows) >= max_files or inspected >= 20000:
            break
        if path.suffix.lower() not in MEDIA_SUFFIXES or path.is_symlink() or not path.is_file():
            continue
        try:
            stat = path.stat()
            if stat.st_mtime < since:
                continue
            rows.append({"path": str(path.relative_to(root)), "bytes": stat.st_size,
                         "sha256": digest(path) if stat.st_size <= max_hash_bytes else None,
                         "hash_status": "checked" if stat.st_size <= max_hash_bytes else
                                        "deferred_size_limit"})
        except OSError:
            continue
    return rows


def write(root: Path, survey: dict[Any, Any], *, max_hash_bytes: int = 256 * 1024**2) -> Path:
    root = Path(root).resolve()
    media = root / "media" / "manifest.json"
    scored_attempts: dict[str, dict[str, Any]] = {}
    for path in (root / "measurements").glob("*.json"):
        try:
            measurement = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(measurement, dict):
            continue
        evaluation = measurement.get("evaluate") or {}
        if not isinstance(evaluation, dict):
            continue
        attempt = evaluation.get("attempt_id")
        if measurement.get("ok") and attempt:
            scored_attempts[str(attempt)] = measurement
    produced: dict[str, dict[str, Any]] = {}
    for path in (root / "attempts").glob("*/receipt.json"):
        if path.is_symlink():
            continue
        try:
            receipt = read_json(path)
            receipt_sha256 = digest(path)
        except (OSError, ValueError):
            continue
        if not isinstance(receipt, dict):
            continue
        captured_media = receipt.get("media") or []
        if not isinstance(captured_media, list):
            continue
        for item in captured_media:
            if isinstance(item, dict) and item.get("path"):
                produced[str(item["path"])] = {"receipt": receipt, "media": item,
                                                "receipt_sha256": receipt_sha256}
    demo_events: dict[str, dict[str, Any]] = {}
    try:
        event_doc = read_json(root / "media_events.json")
    except (OSError, ValueError):
        event_doc = {}
    for event in (event_doc.get("rows") or []) if isinstance(event_doc, dict) else []:
        if not isinstance(event, dict) or event.get("status") != "captured":
            continue
        for item in event.get("media") or []:
            if isinstance(item, dict) and item.get("path"):
                demo_events[str(item["path"])] = event
    rows = []
    for bucket in survey.values():
        if getattr(bucket, "kind", "") != "recording":
            continue
        for relative, size in sorted(bucket.examples):
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                continue
            current_hash = digest(path) if size <= max_hash_bytes else None
            lineage = produced.get(relative) or {}
            receipt, captured = lineage.get("receipt") or {}, lineage.get("media") or {}
            same_bytes = (bool(current_hash) and captured.get("sha256") == current_hash)
            reported_attempt = str(receipt.get("attempt_id") or "") if lineage else ""
            attempt = reported_attempt if same_bytes else ""
            measurement = scored_attempts.get(attempt) if attempt else None
            demo_candidate = demo_events.get(relative)
            demo = (demo_candidate if demo_candidate and reported_attempt and
                    str(demo_candidate.get("attempt_id") or "") == reported_attempt else None)
            media_identity_status = (
                "verified_sha256_match" if same_bytes else
                "hash_deferred_size_limit" if size > max_hash_bytes else
                "media_hash_mismatch" if captured.get("sha256") else
                "media_hash_unavailable" if lineage else "no_producer_receipt")
            demo_settings = (demo.get("settings") or {}) if demo else {}
            measurement_settings = ((measurement.get("settings") or {})
                                    if measurement else {})
            if not isinstance(demo_settings, dict):
                demo_settings = {}
            if not isinstance(measurement_settings, dict):
                measurement_settings = {}
            identity_settings = demo_settings if demo else measurement_settings
            task = ((demo.get("task") if demo else None) or
                    identity_settings.get("task") or identity_settings.get("task_name"))
            seed = demo.get("seed") if demo else measurement_settings.get("seed")
            policy = (measurement.get("policy_artifact") or {}) if measurement else {}
            if not isinstance(policy, dict):
                policy = {}
            stage_seconds = (demo.get("stage_seconds") if demo else
                             receipt.get("seconds") if measurement else None)
            media_capture_seconds = (demo.get("media_capture_seconds") if demo else
                                     receipt.get("media_capture_seconds") if measurement else None)
            capture_seconds = demo.get("seconds") if demo else None
            if (capture_seconds is None and isinstance(stage_seconds, (int, float)) and
                    not isinstance(stage_seconds, bool) and
                    isinstance(media_capture_seconds, (int, float)) and
                    not isinstance(media_capture_seconds, bool)):
                capture_seconds = round(stage_seconds + media_capture_seconds, 3)
            episode_status = (
                demo.get("episode_identity_status") if demo else
                "aggregate_evaluation_media_not_bound_to_one_episode" if measurement else
                "not_attributed_to_a_native_recording_event")
            rows.append({"path": relative, "bytes": size,
                         "sha256": current_hash,
                         "hash_status": "checked" if size <= max_hash_bytes else
                                        "deferred_size_limit",
                         "media_identity_status": media_identity_status,
                         "classification": bucket.what,
                         "limitation": bucket.caveat,
                         "task": task,
                         "seed": seed,
                         "episode": (demo.get("episode_id") if demo else None),
                         "episode_identity_status": episode_status,
                         "attempt_id": attempt or None,
                         "reported_by_attempt_id": reported_attempt or None,
                         "capture_event_status": demo.get("status") if demo else None,
                         "evaluation_attempt_id": (demo.get("evaluation_attempt_id") if demo
                                                   else attempt if measurement else None),
                         "evaluation_receipt_sha256": (
                             demo.get("evaluation_receipt_sha256") if demo else
                             lineage.get("receipt_sha256") if measurement else None),
                         "comparison_protocol_sha256": (
                             demo.get("comparison_protocol_sha256") if demo else
                             receipt.get("comparison_protocol_sha256") if measurement else None),
                         "policy_sha256": (demo.get("policy_sha256") if demo else
                                           policy.get("sha256") if policy else None),
                         "trigger": demo.get("trigger") if demo else None,
                         "measurement_label": demo.get("measurement_label") if demo else None,
                         "capture_seconds": capture_seconds,
                         "stage_seconds": stage_seconds,
                         "media_capture_seconds": media_capture_seconds,
                         "same_scored_evaluation": True if measurement else None,
                         "selection_rule": (str(demo.get("trigger")) +
                                            "; not a representative episode sample") if demo else
                                           "bounded discovery sample; not representative"})
    previewed = 0
    for row in sorted(rows, key=lambda item: (
            not bool(item.get("same_scored_evaluation")),
            not bool(item.get("trigger")), str(item.get("path")))):
        if (previewed >= _MAX_PREVIEWS or
                row.get("media_identity_status") != "verified_sha256_match" or
                row.get("hash_status") != "checked" or
                not row.get("sha256") or
                Path(str(row.get("path"))).suffix.lower() == ".gif"):
            continue
        row["preview"] = _preview(root, str(row["path"]), str(row["sha256"]))
        previewed += 1
    atomic_json(media, {"schema_version": 1, "updated_at": now(),
                        "status": "sampled_real_files" if rows else "no_media_found",
                        "recordings": rows,
                        "warning": "A file's existence is not proof of candidate identity, "
                                   "episode outcome or representativeness."})
    return media
