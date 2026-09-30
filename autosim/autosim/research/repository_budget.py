"""Conservative, persisted per-repository GPU occupancy upper bound.

Every reserved run is charged its whole wall time until it finishes, even when the GPU was
idle. This deliberately overcounts usage when a device ledger is unavailable. A crashed run
keeps its reservation until it resumes, so restarting cannot silently reset the cap.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .common import atomic_json, object_digest, read_json


def source_repository(repo: Path) -> Path:
    """Follow bounded isolated-copy manifests to the repository that owns the budget.

    A budget keyed by an isolated path can be reset by making another copy of the same
    checkout. The manifest is the copy's identity boundary: follow it only when its recorded
    destination is exactly the path being charged, and stop on missing, malformed, cyclic or
    over-deep provenance.
    """
    current = Path(repo).expanduser().resolve()
    visited = {current}
    for _ in range(8):
        manifest = current.parent / "workspace_snapshot.json"
        if manifest.is_symlink() or not manifest.is_file():
            break
        try:
            row = read_json(manifest)
            source_value = str(row.get("source") or "").strip()
            destination_value = str(row.get("destination") or "").strip()
            if not source_value or not destination_value:
                break
            destination = Path(destination_value).expanduser().resolve()
            source = Path(source_value).expanduser().resolve()
            if destination != current or source == current or source in visited or not source.is_dir():
                break
        except (OSError, TypeError, ValueError, AttributeError):
            break
        current = source
        visited.add(current)
    return current


def repository_key(repo: Path) -> str:
    repo = source_repository(repo)
    root_probe = subprocess.run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                                capture_output=True, text=True, timeout=5, check=False)
    is_git_root = (root_probe.returncode == 0 and
                   Path(root_probe.stdout.strip()).resolve() == repo)
    remote = ""
    if is_git_root:
        probe = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"],
                               capture_output=True, text=True, timeout=5, check=False)
        remote = probe.stdout.strip() if probe.returncode == 0 else ""
    identity = remote.removesuffix(".git").rstrip("/") if remote else str(repo)
    return object_digest({"repository": identity})


class RepositoryBudget:
    def __init__(self, ledger_root: Path, *, repo: Path, cap_seconds: float = 86400):
        if cap_seconds <= 0 or cap_seconds > 86400:
            raise ValueError("repository GPU budget must be within the 24-hour user cap")
        self.key = repository_key(repo)
        self.path = Path(ledger_root) / f"{self.key}.json"
        self.lock_path = self.path.with_suffix(".lock")
        self.cap_seconds = float(cap_seconds)

    def _matching_legacy_ledgers(self) -> list[tuple[Path, bytes]]:
        """Find old path-keyed ledgers whose run manifests prove this repository identity.

        An alias ledger is not imported merely because its filename, run name, or a
        caller-provided repository string looks familiar. Every run must point to a real
        isolated checkout with an exact workspace_snapshot destination, and following that
        bounded provenance chain must produce this budget's canonical repository key.
        """
        matches: list[tuple[Path, bytes]] = []
        for candidate in sorted(self.path.parent.glob("*.json")):
            if candidate == self.path or candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                raw = candidate.read_bytes()
                record = json.loads(raw)
                if (not isinstance(record, dict) or record.get("schema_version") != 1 or
                        record.get("repository_key") != candidate.stem or
                        not isinstance(record.get("runs"), dict)):
                    continue
                runs = record["runs"]
                if not runs:
                    continue
                verified = 0
                for output_value in runs:
                    output = Path(str(output_value)).expanduser()
                    if not output.is_absolute():
                        continue
                    try:
                        output = output.resolve(strict=True)
                        manifest_path = output / "workspace_snapshot.json"
                        if (not output.is_dir() or manifest_path.is_symlink() or
                                not manifest_path.is_file()):
                            continue
                        manifest = read_json(manifest_path)
                        destination_value = str(manifest.get("destination") or "").strip()
                        source_value = str(manifest.get("source") or "").strip()
                        if not destination_value or not source_value:
                            continue
                        destination = Path(destination_value).expanduser().resolve(strict=True)
                        source = Path(source_value).expanduser().resolve(strict=True)
                        if (destination.parent != output or not destination.is_dir() or
                                not source.is_dir() or destination_value != str(destination)):
                            continue
                        if repository_key(destination) == self.key:
                            verified += 1
                    except (OSError, TypeError, ValueError, AttributeError,
                            subprocess.SubprocessError):
                        continue
                if verified:
                    matches.append((candidate, raw))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
        return matches

    def _merge_legacy_ledgers(self, record: dict[str, Any]) -> None:
        """Conservatively merge provenance-proven aliases without mutating their files."""
        runs = record.get("runs")
        sources = record.setdefault("legacy_sources", {})
        if not isinstance(runs, dict) or not isinstance(sources, dict):
            raise ValueError("repository budget ledger has invalid runs or legacy_sources")
        for candidate, preliminary_bytes in self._matching_legacy_ledgers():
            lock_path = candidate.with_suffix(".lock")
            with lock_path.open("a+b") as alias_lock:
                fcntl.flock(alias_lock, fcntl.LOCK_EX)
                try:
                    # Re-read after taking the old writer's lock; do not import a moving file.
                    raw = candidate.read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    alias = json.loads(raw)
                    alias_key = candidate.stem
                    prior = sources.get(alias_key)
                    if prior is not None:
                        if not isinstance(prior, dict) or prior.get("sha256") != digest:
                            raise ValueError(
                                f"legacy repository budget ledger changed after import: {alias_key}; "
                                "reconcile its new runs before reserving more time")
                        continue
                    if raw != preliminary_bytes:
                        # The candidate changed between provenance scan and lock acquisition.
                        # Re-scan on a later explicit attempt rather than guessing.
                        raise ValueError(
                            f"legacy repository budget ledger changed during import: {alias_key}")
                    if (not isinstance(alias, dict) or alias.get("schema_version") != 1 or
                            alias.get("repository_key") != alias_key or
                            float(alias.get("cap_seconds", 0)) != self.cap_seconds or
                            not isinstance(alias.get("runs"), dict)):
                        raise ValueError(f"legacy repository budget ledger is invalid: {alias_key}")
                    alias_runs = alias["runs"]
                    if not alias_runs:
                        continue
                    # A mixed/partially unverifiable alias cannot safely be charged in part.
                    verified_outputs = set()
                    for output_value in alias_runs:
                        output = Path(str(output_value)).expanduser()
                        if not output.is_absolute():
                            continue
                        try:
                            output = output.resolve(strict=True)
                            manifest_path = output / "workspace_snapshot.json"
                            if (not output.is_dir() or manifest_path.is_symlink() or
                                    not manifest_path.is_file()):
                                continue
                            manifest = read_json(manifest_path)
                            destination_value = str(manifest.get("destination") or "").strip()
                            source_value = str(manifest.get("source") or "").strip()
                            if not destination_value or not source_value:
                                continue
                            destination = Path(destination_value).expanduser().resolve(strict=True)
                            source = Path(source_value).expanduser().resolve(strict=True)
                            if (destination.parent != output or not destination.is_dir() or
                                    not source.is_dir() or destination_value != str(destination)):
                                continue
                            if repository_key(destination) == self.key:
                                verified_outputs.add(str(output))
                        except (OSError, TypeError, ValueError, AttributeError,
                                subprocess.SubprocessError):
                            continue
                    if len(verified_outputs) != len(alias_runs):
                        raise ValueError(
                            f"legacy repository budget ledger mixes verified and unknown runs: "
                            f"{alias_key}; reconcile before reserving more time")
                    for output_value, row in alias_runs.items():
                        output = Path(str(output_value)).expanduser().resolve(strict=True)
                        if not isinstance(row, dict):
                            raise ValueError(f"legacy repository budget run is invalid: {output}")
                        charged = float(row.get("charged_seconds", 0))
                        reserved = float(row.get("reserved_seconds", 0))
                        if (not math.isfinite(charged) or not math.isfinite(reserved) or
                                charged < 0 or reserved < 0 or
                                row.get("status") not in {"active", "finished"}):
                            raise ValueError(f"legacy repository budget run is invalid: {output}")
                        key = str(output)
                        existing = runs.get(key)
                        if existing is not None and existing != row:
                            raise ValueError(
                                f"legacy repository budget run conflicts with canonical ledger: {key}")
                        runs[key] = dict(row)
                    sources[alias_key] = {
                        "filename": candidate.name,
                        "sha256": digest,
                        "imported_runs": len(alias_runs),
                    }
                finally:
                    fcntl.flock(alias_lock, fcntl.LOCK_UN)

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                record = read_json(self.path) if self.path.is_file() else {
                    "schema_version": 1, "repository_key": self.key,
                    "cap_seconds": self.cap_seconds, "runs": {}}
                if not isinstance(record, dict) or record.get("repository_key") != self.key:
                    raise ValueError("repository budget ledger has the wrong identity")
                if float(record.get("cap_seconds", 0)) != self.cap_seconds:
                    raise ValueError("cannot silently change a repository GPU budget")
                self._merge_legacy_ledgers(record)
                yield record
                atomic_json(self.path, record)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, output: Path, *, wall_seconds: float) -> dict[str, Any]:
        if wall_seconds <= 0:
            raise ValueError("run reservation needs a positive wall-clock limit")
        key = str(Path(output).resolve())
        with self._locked() as record:
            runs = record["runs"]
            existing = runs.get(key)
            if existing:
                if float(existing["wall_seconds"]) != wall_seconds:
                    raise ValueError("resumed run has a different GPU reservation limit")
                if existing["status"] == "active":
                    return dict(existing)
                left = max(0.0, wall_seconds -
                           (time.time() - float(existing["budget_started_epoch"])))
                if left <= 0:
                    raise ValueError("resumed run's wall-clock deadline has expired")
                committed = sum(float(row.get("charged_seconds", 0)) for row in runs.values())
                reserved = sum(float(row.get("reserved_seconds", 0)) for row in runs.values()
                               if row.get("status") == "active")
                if committed + reserved + left > self.cap_seconds:
                    raise ValueError("repository 24 GPU-hour cap has insufficient unreserved time")
                existing.update(status="active", started_epoch=time.time(),
                                reserved_seconds=left)
                return dict(existing)
            committed = sum(float(row.get("charged_seconds", 0)) for row in runs.values())
            reserved = sum(float(row.get("reserved_seconds", 0)) for row in runs.values()
                           if row.get("status") == "active")
            if committed + reserved + wall_seconds > self.cap_seconds:
                raise ValueError("repository 24 GPU-hour cap has insufficient unreserved time")
            row = {"status": "active", "started_epoch": time.time(),
                   "budget_started_epoch": time.time(), "wall_seconds": float(wall_seconds),
                   "reserved_seconds": float(wall_seconds), "charged_seconds": 0.0,
                   "accounting": "one-GPU full wall-time upper bound"}
            runs[key] = row
            return dict(row)

    def finish(self, output: Path) -> dict[str, Any]:
        key = str(Path(output).resolve())
        with self._locked() as record:
            row = record["runs"].get(key)
            if not row:
                raise ValueError("run has no repository GPU reservation")
            if row["status"] == "active":
                elapsed = max(0.0, time.time() - float(row["started_epoch"]))
                # If a stage outlived the reservation, record the overrun rather than hide it.
                row["charged_seconds"] = float(row.get("charged_seconds", 0)) + elapsed
                row["reserved_seconds"] = 0.0
                row["status"] = "finished"
                row["finished_epoch"] = time.time()
            return dict(row)

    def state(self) -> dict[str, Any]:
        with self._locked() as record:
            runs = record["runs"]
            return {"cap_seconds": self.cap_seconds,
                    "charged_seconds": sum(float(row.get("charged_seconds", 0))
                                           for row in runs.values()),
                    "reserved_seconds": sum(float(row.get("reserved_seconds", 0))
                                            for row in runs.values()
                                            if row.get("status") == "active"),
                    "runs": len(runs), "accounting": "one-GPU full wall-time upper bound"}
