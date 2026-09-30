"""Freeze a policy artifact so later rounds cannot overwrite the candidate being compared."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .common import atomic_json, digest, now, object_digest, read_json


def artifact_identity(path: Path) -> tuple[str, str]:
    """Return the identity kind and digest of exactly the archived policy bytes."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("policy artifact is a symlink")
    if path.is_file():
        return "content_sha256", digest(path)
    if not path.is_dir() or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("policy directory is missing or contains symlinks")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("policy directory has no files")
    return "sha256", object_digest([(str(p.relative_to(path)), digest(p)) for p in files])


def freeze_artifact(source: Path, destination: Path, *, max_bytes: int) -> dict[str, Any]:
    """Copy a file or directory with a bounded size and verify the copied bytes."""
    source, destination = Path(source), Path(destination)
    if not source.exists() or source.is_symlink():
        raise ValueError("policy artifact is absent or is a symlink")
    files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
    if source.is_dir() and any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("policy artifact contains symlinks and is not self-contained")
    if not files:
        raise ValueError("policy artifact directory is empty")
    total = sum(p.stat().st_size for p in files)
    if total > max_bytes:
        raise ValueError(f"policy artifact is {total} bytes, above copy limit {max_bytes}")
    fingerprint = object_digest([(str(p.relative_to(source)) if source.is_dir() else source.name,
                                  digest(p)) for p in sorted(files)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"experiment artifact destination already exists: {destination}")
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        if source.is_file():
            shutil.copy2(source, temporary)
        else:
            shutil.copytree(source, temporary, symlinks=False)
        copied = [temporary] if temporary.is_file() else [p for p in temporary.rglob("*")
                                                        if p.is_file()]
        after = object_digest([(str(p.relative_to(temporary)) if temporary.is_dir() else
                                source.name, digest(p)) for p in sorted(copied)])
        if after != fingerprint:
            raise RuntimeError("policy artifact changed while it was copied")
        os.replace(temporary, destination)
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        else:
            temporary.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": fingerprint,
            # A single file has two useful identities: the archive manifest identity above
            # and the raw bytes a native evaluator opens. Never call the first one a byte hash.
            **({"content_sha256": digest(destination)} if destination.is_file() else {}),
            "bytes": total, "files": len(files), "source": str(source)}


#: The records without which an export cannot be re-run at all: the recipe that builds the
#: commands, the interpreter, and the protocol the evaluation was frozen at. Named here rather
#: than read one at a time because a reader of the manifest needs the same list -- knowing
#: which is missing is the difference between "this cannot be re-run" and a guess about why.
REQUIRED_RECORDS = {
    "derived_stages.json": "how each stage's command is built, and the values the derivation "
                           "settled for this repository",
    "execution.json": "which stages exist and what each one does",
    "environment.json": "the interpreter and the probes that established it works",
    "comparison_protocol.json": "the evaluation settings, metric and evaluator, frozen before "
                                "the search began",
}

#: Records that are worth carrying and are not always there. Absence of one of these is not an
#: incomplete export and must not be reported as one -- a single-arm run writes no selection
#: record, and a run with no declared split has no confirmation and never will. Treating them
#: as required made every ordinary export report itself unportable, which is a check failing on
#: a healthy state, and it would have been learned and ignored within a week.
OPTIONAL_RECORDS = {
    "selection_record.json": "how many arms the run measured, and which of them produced a "
                             "number; the measurement files give this too, less exactly",
    "confirmation.json": "the one measurement taken on the settings the search was not allowed "
                         "to look at, when the run took one",
}
RECIPE_ROOT_FILES = ("derived_stages.json", "execution.json", "environment.json")


def _bundle(run_root: Path, label: str, export_root: Path | None = None) -> dict[str, Any]:
    """Everything needed to say what this export is, and what it still needs from elsewhere.

    **The number travels with its provenance or it is not worth exporting.** An export holding
    a policy file and a note that says which measurement it came from is a policy file; a
    reader cannot tell whether it is the best of one arm or the best of thirty, whether its
    settings were frozen before the search or chosen during it, whether anything was confirmed
    on episodes the search never saw, or how to run it again. All of that is already on disk
    in the run directory. This copies the parts that are small and hashes them, and lists by
    name the parts that are not copied -- because the base checkout, the conda environment, the
    datasets and the GPU are real dependencies and a bundle that is silent about them reads as
    self-contained.
    """
    recipe_root = run_root
    for candidate in (run_root, *run_root.parents):
        if (candidate / "derived_stages.json").is_file():
            recipe_root = candidate
            break
        if candidate == candidate.parent:
            break

    records: dict[str, Any] = {}
    missing: list[str] = []
    absent_optional: list[str] = []
    for name, purpose in {**REQUIRED_RECORDS, **OPTIONAL_RECORDS}.items():
        path = (recipe_root if name in RECIPE_ROOT_FILES else run_root) / name
        if not path.is_file():
            (missing if name in REQUIRED_RECORDS else absent_optional).append(
                f"{name} -- {purpose}")
            continue
        try:
            key = digest(path)
            row = {"purpose": purpose, "sha256": key, "bytes": path.stat().st_size,
                   "from": str(path), "required": name in REQUIRED_RECORDS}
            if export_root is not None:
                target = export_root / "records" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                if digest(target) != key:
                    raise RuntimeError(f"record changed during export: {name}")
                row["copy"] = str(target.relative_to(export_root))
            records[name] = row
        except OSError as exc:
            missing.append(f"{name} -- unreadable ({type(exc).__name__})")

    measurement = run_root / "measurements" / f"{label}.json"
    try:
        measured = read_json(measurement) if measurement.is_file() else {}
    except (OSError, ValueError):
        measured = {}
    if not isinstance(measured, dict):
        measured = {}

    bundle: dict[str, Any] = {
        "records": records,
        "missing_records": missing,
        "measured": {
            "label": label,
            "metric_value": measured.get("metric_value"),
            "metric": measured.get("metric") or {},
            "settings": measured.get("settings") or {},
            "scored": measured.get("scored") or "",
        },
        "not_copied": [
            "the benchmark checkout itself, which the evaluator reads and which this export "
            "does not contain",
            "the conda environment or interpreter the stages ran under",
            "the datasets and any checkpoint the benchmark ships",
        ],
        "reproduce": "",
        "limitations": [],
    }

    # The arms and the confirmation, as the run's own records state them -- read through the
    # same functions the run uses, so the export cannot describe a search differently from the
    # way the run described it.
    try:
        from . import selection as selection_module

        summary = selection_module.summary(run_root)
        bundle["search"] = {
            "attempted": summary.get("attempted"), "scored": summary.get("scored"),
            "best_is_the_maximum_of": summary.get("best_is_the_maximum_of"),
            "one_protocol": summary.get("one_protocol"),
            "comparison": summary.get("comparison"),
            "reading": summary.get("reading"),
        }
        if (summary.get("best_is_the_maximum_of") or 0) > 1:
            bundle["limitations"].append(
                "the exported policy is the best of several arms measured at one protocol, so "
                "the number beside it includes the highest noise among them")
        if not summary.get("one_protocol") and (summary.get("scored") or 0) > 1:
            bundle["limitations"].append(
                "the arms were not all measured under one frozen protocol, so they are not "
                "comparable with one another")
    except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
        bundle["search"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        from .reevaluate import plan as recheck_plan

        decision = recheck_plan(run_root, label)
        if decision.get("status") == "ready":
            bundle["reproduce"] = (f"autosim recheck {run_root} {label} --output <fresh-dir>")
        else:
            bundle["limitations"].append(
                f"the evaluator cannot be run again from these records: {decision.get('why')}")
    except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
        bundle["limitations"].append(f"re-runnability could not be judged: {exc}")

    if bundle["measured"]["metric_value"] is None:
        bundle["limitations"].append(
            "this measurement produced no number, so there is no result for an export of it "
            "to carry")
    if "confirmation.json" not in records:
        bundle["limitations"].append(
            "no confirmation was taken on held-out settings, so nothing here has been "
            "measured on episodes the search did not see")
    bundle["missing_records"] = missing
    bundle["records_not_present"] = absent_optional
    bundle["complete"] = not missing
    return bundle


def export_best(run_root: Path, best: dict[str, Any] | None, *,
                max_bytes: int = 2 * 1024**3) -> dict[str, Any]:
    """Export the selected artifact and source overlay, preserving prior exports.

    This is not a native-load confirmation: the base checkout and environment are still
    external dependencies, and the manifest says so explicitly.
    """
    run_root = Path(run_root).resolve()
    export_root = run_root / "export" / uuid.uuid4().hex
    export_root.mkdir(parents=True)
    manifest: dict[str, Any] = {"schema_version": 1, "created_at": now(),
                                "selected": best or {}, "status": "no_valid_candidate",
                                "source_overlay": {}, "policy": {},
                                "native_load_verified": False,
                                "warning": "Base checkout, environment and independent evaluation "
                                           "are not packaged or verified."}
    if best is None or best.get("scale") == "progress":
        atomic_json(export_root / "manifest.json", manifest)
        return manifest
    name = str(best.get("name") or "")
    label = "baseline" if name == "baseline" else name.replace("-", "_", 1)
    measurement_path = run_root / "measurements" / f"{label}.json"
    try:
        measurement = read_json(measurement_path)
    except (OSError, ValueError):
        measurement = {}
    policy_row = measurement.get("policy_artifact") or {}
    source_state = (measurement.get("source_state")
                    if isinstance(measurement.get("source_state"), dict) else {})
    source_policy = (measurement.get("source_policy_artifact")
                     if isinstance(measurement.get("source_policy_artifact"), dict) else {})
    source_files: dict[str, str] | None = None
    source_state_valid = False
    if source_state:
        try:
            workspace_path = run_root.parent.parent / "workspace_snapshot.json"
            workspace = read_json(workspace_path)
            snapshot_name = str(source_state.get("source_snapshot") or "")
            overlay_files = source_state.get("overlay_files")
            expected = object_digest({
                "schema_version": 1,
                "source_tree_fingerprint": workspace.get("source_tree_fingerprint"),
                "source_snapshot": snapshot_name,
                "overlay_files": overlay_files,
                "overlay_modes": source_state.get("overlay_modes") or {},
                "overlay_directories": source_state.get("overlay_directories") or {},
                "copy_mode": source_state.get("copy_mode", "full_worktree"),
                "untracked_assets_omitted": bool(
                    source_state.get("untracked_assets_omitted")),
            }) if snapshot_name and isinstance(overlay_files, dict) else ""
            from .snapshot import Snapshots
            snapshots = Snapshots(run_root / "snapshots")
            snapshot = snapshots.get(snapshot_name) if snapshot_name else None
            source_state_valid = bool(
                expected and source_state.get("identity_sha256") == expected and
                source_state.get("source_tree_fingerprint") ==
                workspace.get("source_tree_fingerprint") and snapshot and
                dict(snapshot.files) == overlay_files and
                all(not key or ((snapshots.blobs / key).is_file() and
                                digest(snapshots.blobs / key) == key)
                    for key in snapshot.files.values()))
            if source_state_valid:
                source_files = dict(overlay_files)
                manifest["source_overlay_modes"] = dict(
                    source_state.get("overlay_modes") or {})
                manifest["source_overlay_directories"] = dict(
                    source_state.get("overlay_directories") or {})
                manifest["source_base"] = {
                    "tree_fingerprint": workspace.get("source_tree_fingerprint"),
                    "git_revision": workspace.get("source_git_revision") or "",
                    "copy_mode": source_state.get("copy_mode"),
                    "untracked_assets_omitted": bool(
                        source_state.get("untracked_assets_omitted")),
                    "source_state_sha256": expected,
                }
            else:
                manifest.setdefault("overlay_errors", []).append(
                    "the measured source-tree identity or one of its content blobs is invalid")
        except (OSError, KeyError, TypeError, ValueError):
            manifest.setdefault("overlay_errors", []).append(
                "the measured source-tree identity cannot be verified")
    if source_policy:
        if (source_policy.get("kind") == "source_tree" and source_state_valid and
                source_policy.get("identity_sha256") == source_state.get("identity_sha256") and
                not policy_row):
            manifest["policy"] = {"kind": "source_tree",
                                  "identity_sha256": source_state["identity_sha256"],
                                  "source_snapshot": source_state["source_snapshot"]}
        else:
            manifest["policy_error"] = (
                "the source-defined policy does not match its frozen source-tree identity")
    path = Path(str(policy_row.get("path") or ""))
    if policy_row and path.exists() and path.resolve().is_relative_to(run_root):
        suffix = path.suffix if path.is_file() else ""
        try:
            identity_field, identity = artifact_identity(path)
            if policy_row.get(identity_field) != identity:
                raise ValueError("the selected policy no longer matches its measurement")
            manifest["policy"] = freeze_artifact(
                path, export_root / f"policy{suffix}", max_bytes=max_bytes)
        except (OSError, ValueError, RuntimeError) as exc:
            manifest["policy_error"] = f"{type(exc).__name__}: {exc}"
    files = (source_files if source_state_valid else
             {} if source_policy else best.get("files") or {})
    if isinstance(files, dict):
        overlay_bytes = 0
        validated: list[tuple[str, Path, str]] = []
        for relative, key in files.items():
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                manifest.setdefault("overlay_errors", []).append(str(relative))
                continue
            if not key:
                manifest["source_overlay"][str(relative)] = {"absent": True}
                continue
            blob = run_root / "snapshots" / "blobs" / str(key)
            if not blob.is_file() or digest(blob) != key:
                manifest.setdefault("overlay_errors", []).append(str(relative))
                continue
            overlay_bytes += blob.stat().st_size
            validated.append((str(relative), blob, key))
        if overlay_bytes > max_bytes:
            manifest.setdefault("overlay_errors", []).append(
                f"source overlay is {overlay_bytes} bytes, above export limit {max_bytes}")
            validated = []
        for relative, blob, key in validated:
            relative_path = Path(relative)
            target = export_root / "source_overlay" / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(blob, target)
            if digest(target) != key:
                manifest.setdefault("overlay_errors", []).append(relative)
            else:
                mode = (manifest.get("source_overlay_modes") or {}).get(relative)
                if mode is not None:
                    os.chmod(target, int(mode))
                manifest["source_overlay"][relative] = {"sha256": key}
        if source_state_valid:
            for relative, mode in (manifest.get("source_overlay_directories") or {}).items():
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    manifest.setdefault("overlay_errors", []).append(str(relative))
                    continue
                if mode is None:
                    manifest["source_overlay"][str(relative)] = {"absent": True,
                                                                    "kind": "directory"}
                    continue
                directory = export_root / "source_overlay" / relative_path
                directory.mkdir(parents=True, exist_ok=True)
                os.chmod(directory, int(mode))
    is_source_policy = manifest["policy"].get("kind") == "source_tree"
    manifest["status"] = (
        "source_controller_overlay_exported" if is_source_policy and
        not manifest.get("overlay_errors") and source_state_valid else
        "artifact_and_overlay_exported" if manifest["policy"] and
        not manifest.get("overlay_errors") else "incomplete_export")
    manifest["bundle"] = _bundle(run_root, label, export_root)
    # The selected measurement is required even though it is one of many arms. Keeping its
    # bytes in the package makes the policy's claimed score auditable after the run moves.
    if measurement_path.is_file():
        target = export_root / "records" / "measurements" / f"{label}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(measurement_path, target)
        manifest["bundle"]["measurement_copy"] = str(target.relative_to(export_root))
    # Two different questions, and the export answers both rather than one. `status` is "is
    # this a usable policy with its sources"; `portable` is "could somebody else take this
    # directory and know what it is, what it measured, and how to run it again". An export can
    # be the first without the second, and the manifest has to be able to say which.
    # Completeness and a command template are not relocation verification. The checkout,
    # interpreter and datasets still live outside this export and no moved copy was run.
    manifest["portable"] = False
    manifest["export_runtime_verified"] = False
    manifest["warning"] = (
        "Base checkout, environment and independent evaluation are not packaged or verified."
        + ("" if manifest["portable"] else " This export is also missing records it would need "
                                           "to be re-run from: see bundle.missing_records."))
    atomic_json(export_root / "manifest.json", manifest)
    atomic_json(run_root / "export" / "latest.json", {
        "path": str(export_root.relative_to(run_root)), "status": manifest["status"],
        "created_at": manifest["created_at"]})
    return {**manifest, "path": str(export_root)}
