"""Shared, replayable state for one preparation-to-research run.

Component-local records remain the detailed evidence. This store supplies the common
timeline and a compact projection of which phase owns control, what action is active, and
which evidence can reconstruct that action after a restart.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .common import atomic_json, digest, now, read_json

# Cache only verified identities, never the potentially enormous parsed legacy history.
_VERIFIED_BASES: dict[tuple, tuple[int, str]] = {}


class ResearchStateError(RuntimeError):
    """The shared event stream or run identity cannot be trusted."""


class ResearchStatePersistenceError(ResearchStateError):
    """An action could not durably record its state transition.

    A different ``ResearchStateError`` can describe a rejected research contract, such as
    changed frozen inputs. Callers must not treat every such rejection as a broken event
    store: doing so prevents the outer controller from persisting the rejection receipt and
    clearing its active-action marker.
    """


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def verify_event_rows(rows: list[dict[str, Any]], *, start_sequence: int = 1,
                      previous_hash: str = "") -> str:
    """Verify ordered sequence numbers and the hash chain; return its head hash."""
    previous = previous_hash
    for expected, row in enumerate(rows, start=start_sequence):
        if row.get("sequence") != expected or row.get("previous_hash") != previous:
            raise ResearchStateError(f"run event chain breaks at sequence {expected}")
        material = {key: value for key, value in row.items() if key != "event_hash"}
        actual = hashlib.sha256(_canonical(material)).hexdigest()
        if row.get("event_hash") != actual:
            raise ResearchStateError(f"run event hash is invalid at sequence {expected}")
        previous = actual
    return previous


class ResearchStateStore:
    """Atomic event log and recoverable snapshot shared by preparation and research."""

    def __init__(self, root: Path, *, run_id: str, repository: Path):
        self.root = Path(root).expanduser().resolve()
        self.run_id = str(run_id)
        self.repository = Path(repository).expanduser().resolve()
        self.state_path = self.root / "run_state.json"
        self.events_path = self.root / "run_events.json"
        self.lock_path = self.root / "run_state.lock"
        self.journal_path = self.root / "run_events.jsonl"
        self._journal_checkpoint = None

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _identity(self, record: dict[str, Any], *, path: Path) -> None:
        if record.get("run_id") and record["run_id"] != self.run_id:
            raise ResearchStateError(f"{path.name} belongs to another run")
        recorded_repo = record.get("repository")
        if recorded_repo and Path(str(recorded_repo)).expanduser().resolve() != self.repository:
            raise ResearchStateError(f"{path.name} belongs to another repository")

    def _read_events(self) -> tuple[list[dict[str, Any]], str]:
        if not self.events_path.is_file():
            return [], ""
        try:
            record = read_json(self.events_path)
        except (OSError, ValueError, TypeError) as exc:
            raise ResearchStateError(f"run event log cannot be read: {exc}") from exc
        if isinstance(record, dict) and record.get("schema_version") == 2:
            self._identity(record, path=self.events_path)
            base = read_json(self.root / "run_events.legacy.json")
            self._identity(base, path=self.root / "run_events.legacy.json")
            rows = base["rows"] + self._journal_rows()
            return rows, verify_event_rows(rows)
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ResearchStateError("run event log has an unsupported schema")
        self._identity(record, path=self.events_path)
        rows = record.get("rows")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ResearchStateError("run event log rows are invalid")
        head = verify_event_rows(rows)
        return rows, head

    def _journal_rows(self, offset: int = 0) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        rows = []
        with self.journal_path.open("rb") as stream:
            stream.seek(offset)
            for line in stream:
                # A torn append is not a committed event; never silently adopt it.
                if not line.endswith(b"\n"):
                    raise ResearchStateError("run journal has an incomplete append")
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ResearchStateError("run journal has an invalid record") from exc
                if not isinstance(row, dict):
                    raise ResearchStateError("run journal record is not an object")
                self._identity(row, path=self.journal_path)
                rows.append(row)
        return rows

    def _current(self) -> tuple[dict[str, Any], int, str]:
        """Reconcile the append journal without repeatedly parsing the legacy history."""
        manifest = read_json(self.events_path) if self.events_path.is_file() else {}
        if manifest.get("schema_version") != 2:
            rows, head = self._read_events()
            return self._reconcile(self._read_snapshot(), rows), len(rows), head
        self._identity(manifest, path=self.events_path)
        base_path = self.root / "run_events.legacy.json"
        stat = base_path.stat()
        key = (str(base_path), stat.st_ino, stat.st_size, stat.st_mtime_ns,
               stat.st_ctime_ns, self.run_id, str(self.repository))
        base_rows = None
        base_identity = _VERIFIED_BASES.get(key)
        if base_identity is None:
            if manifest.get("base_sha256"):
                # Migration validated the chain before committing this manifest. Verify
                # its immutable bytes in bounded memory on a cold process, not by loading
                # the entire historic object graph again.
                if digest(base_path) != manifest["base_sha256"]:
                    raise ResearchStateError("legacy event hash is invalid")
                base_identity = (manifest["base_sequence"], manifest["base_head"])
            else:
                base = read_json(base_path)
                self._identity(base, path=base_path)
                base_rows = base.get("rows")
                if not isinstance(base_rows, list):
                    raise ResearchStateError("legacy event rows are invalid")
                base_identity = (len(base_rows), verify_event_rows(base_rows))
            if len(_VERIFIED_BASES) >= 32:
                _VERIFIED_BASES.clear()
            _VERIFIED_BASES[key] = base_identity
        count, head = base_identity
        if (manifest.get("base_sequence"), manifest.get("base_head")) != (count, head):
            raise ResearchStateError("journal base does not match its manifest")
        state = self._read_snapshot()
        if int(state.get("event_sequence") or 0) < count:
            if base_rows is None:
                base_rows = read_json(base_path)["rows"]
            state = self._reconcile(state, base_rows)
        offset = 0
        checkpoint = self._journal_checkpoint
        stat = self.journal_path.stat() if self.journal_path.exists() else None
        if checkpoint and stat:
            inode, size, modified, changed, sequence, previous = checkpoint
            if (stat.st_ino == inode and stat.st_size == size
                    and int(state.get("event_sequence") or 0) >= sequence
                    and (stat.st_mtime_ns, stat.st_ctime_ns) == (modified, changed)):
                offset, count, head = size, sequence, previous
        rows = self._journal_rows(offset)
        heads = {count: head}
        for row in rows:
            count += 1
            if row.get("sequence") != count or row.get("previous_hash") != head:
                raise ResearchStateError(f"run event chain breaks at sequence {count}")
            actual = hashlib.sha256(_canonical({k: v for k, v in row.items()
                                               if k != "event_hash"})).hexdigest()
            if row.get("event_hash") != actual:
                raise ResearchStateError(f"run event hash is invalid at sequence {count}")
            head = actual
            heads[count] = head
        applied = int(state.get("event_sequence") or 0)
        if applied > count or state.get("event_head_hash", "") != heads.get(applied):
            raise ResearchStateError("run snapshot does not match its event chain")
        for row in rows:
            if row["sequence"] > applied:
                self._apply_event(state, row)
        if stat:
            self._journal_checkpoint = (stat.st_ino, stat.st_size, stat.st_mtime_ns,
                                        stat.st_ctime_ns, count, head)
        state.update(schema_version=1, run_id=self.run_id, repository=str(self.repository))
        return self._refresh_compat_fields(state), count, head

    def _append(self, row: dict[str, Any]) -> None:
        # Keep the old format for small histories and legacy tooling; switch once, before
        # it can enter quadratic rewrite growth. Preserve old evidence byte-for-byte.
        manifest = read_json(self.events_path) if self.events_path.is_file() else {}
        if manifest.get("schema_version") != 2:
            rows = manifest.get("rows", [])
            if len(rows) < 128 and (not self.events_path.exists() or
                                  self.events_path.stat().st_size < 1024 * 1024):
                atomic_json(self.events_path, {"schema_version": 1, "run_id": self.run_id,
                    "repository": str(self.repository), "rows": [*rows, row]})
                return
            base_path = self.root / "run_events.legacy.json"
            if base_path.exists() and not os.path.samefile(base_path, self.events_path):
                raise ResearchStateError("uncommitted legacy journal migration exists")
            # Hard-link first: a crash before manifest replacement leaves the original
            # readable. A retry may safely finish this migration only for identical files.
            if not base_path.exists():
                os.link(self.events_path, base_path)
            atomic_json(self.events_path, {"schema_version": 2, "run_id": self.run_id,
                "repository": str(self.repository), "base_sequence": len(rows),
                "base_head": rows[-1]["event_hash"] if rows else "",
                "base_sha256": digest(base_path),
                "base_ref": base_path.name, "journal_ref": self.journal_path.name})
        descriptor = os.open(self.journal_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            material = _canonical(row) + b"\n"
            view = memoryview(material)
            while view:
                written = os.write(descriptor, view)
                if not written:
                    raise OSError("journal append made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        stat = self.journal_path.stat()
        self._journal_checkpoint = (stat.st_ino, stat.st_size, stat.st_mtime_ns,
                                    stat.st_ctime_ns, row["sequence"], row["event_hash"])

    def _read_snapshot(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {}
        try:
            state = read_json(self.state_path)
        except (OSError, ValueError, TypeError) as exc:
            raise ResearchStateError(f"run state snapshot cannot be read: {exc}") from exc
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ResearchStateError("run state snapshot has an unsupported schema")
        self._identity(state, path=self.state_path)
        # Migrate the preparation-only schema written before the shared store existed.
        if not isinstance(state.get("phases"), dict):
            old_phase = str(state.get("phase") or "preparation")
            old_phase_state = {key: state[key] for key in (
                "status", "state", "steps", "last_decision", "current_action", "last_action",
                "process_identity", "recovery_after_interruption",
                "state_persistence_error") if key in state}
            state["phases"] = {old_phase: old_phase_state}
            state["event_sequence"] = 0
        return state

    @staticmethod
    def _refresh_compat_fields(state: dict[str, Any]) -> dict[str, Any]:
        """Keep older readers working while preserving per-phase snapshots."""
        phases = state.get("phases") or {}
        if state.get("phase") not in {"preparation", "research"}:
            owners = [(name, value) for name, value in phases.items()
                      if name in {"preparation", "research"} and isinstance(value, dict)]
            if owners:
                name, value = max(owners, key=lambda item: int(item[1].get("event_sequence") or 0))
                state.update(phase=name, status=value.get("status"))
        active = phases.get(str(state.get("phase") or ""), {})
        preparation = phases.get("preparation", {})
        if not isinstance(active, dict):
            active = {}
        if not isinstance(preparation, dict):
            preparation = {}
        for key in ("state", "steps", "last_decision", "current_action", "last_action",
                    "process_identity", "recovery_after_interruption",
                    "state_persistence_error"):
            source = active if key in active else preparation
            if key in source:
                state[key] = source[key]
            else:
                state.pop(key, None)
        return state

    def _apply_event(self, state: dict[str, Any], row: dict[str, Any]) -> None:
        phase = str(row.get("phase") or "unknown")
        phases = state.setdefault("phases", {})
        phase_state = phases.setdefault(phase, {})
        patch = row.get("state_patch") or {}
        if row.get("state_patch_digest"):
            identity = row["state_patch_digest"]
            if not isinstance(identity, str) or len(identity) != 64 or any(
                    c not in "0123456789abcdef" for c in identity):
                raise ResearchStateError("invalid state patch identity")
            patch = read_json(self.root / "run_state_blobs" / f"{identity}.json")
            if hashlib.sha256(_canonical(patch)).hexdigest() != identity:
                raise ResearchStateError("state patch blob changed")
        if isinstance(patch, dict):
            phase_state.update(patch)
        phase_state.update(status=row.get("status"), updated_at=row.get("at"),
                           last_event_id=row.get("event_id"),
                           event_sequence=row.get("sequence"))
        # Subtasks cannot declare the research completed or replace its active action.
        if phase in {"preparation", "research"} or not state.get("phase"):
            state.update(phase=phase, status=row.get("status"))
        state.update(updated_at=row.get("at"),
                     last_event_id=row.get("event_id"),
                     event_head_hash=row.get("event_hash"),
                     event_sequence=row.get("sequence"))
        revision = row.get("state_revision")
        if isinstance(revision, int) and not isinstance(revision, bool):
            state["state_revision"] = max(int(state.get("state_revision") or 0), revision)
        decision_revision = row.get("decision_revision")
        if isinstance(decision_revision, int) and not isinstance(decision_revision, bool):
            state["decision_revision"] = max(
                int(state.get("decision_revision") or 0), decision_revision)
        elif row.get("decision_relevant") is not False:
            # Older event rows predate the explicit relevance marker. Treat them as
            # decision-changing so an upgrade never weakens stale-decision protection.
            state["decision_revision"] = int(state.get("decision_revision") or 0) + 1

    def _reconcile(self, state: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        applied = int(state.get("event_sequence") or 0)
        if applied > len(rows):
            raise ResearchStateError("run state is ahead of its event log")
        if "decision_revision" not in state:
            # Migrate snapshots produced before decision and telemetry revisions were
            # separated. Legacy events all count as decision-relevant unless explicitly
            # marked otherwise by a newer writer.
            state["decision_revision"] = sum(
                1 for row in rows[:applied] if row.get("decision_relevant") is not False)
        for row in rows[applied:]:
            self._apply_event(state, row)
        state["state_revision"] = int(state.get("state_revision") or 0)
        state["decision_revision"] = int(state.get("decision_revision") or 0)
        state["run_id"] = self.run_id
        state["repository"] = str(self.repository)
        state["schema_version"] = 1
        return self._refresh_compat_fields(state)

    def load(self) -> dict[str, Any] | None:
        """Read and replay events newer than the snapshot, repairing only the projection."""
        # A fresh preparation constructor is a read, not a write: do not create its output
        # directory (or lock file) until there is state to reconcile or the first event is
        # recorded. This also keeps probing a missing run side-effect free.
        if not self.state_path.is_file() and not self.events_path.is_file():
            return None
        with self._locked():
            before = int(self._read_snapshot().get("event_sequence") or 0)
            state, count, head = self._current()
            if not count and not state:
                return None
            state["event_head_hash"] = head
            if int(state.get("event_sequence") or 0) != before:
                atomic_json(self.state_path, state)
            return state

    def record(self, phase: str, event: str, *, status: str,
               details: dict[str, Any] | None = None,
               phase_state: dict[str, Any] | None = None,
               event_patch: dict[str, Any] | None = None,
               decision_relevant: bool = True) -> dict[str, Any]:
        """Append one integrity-checked event and atomically refresh the state projection."""
        if not phase or not event or not status:
            raise ValueError("phase, event, and status are required")
        if not isinstance(decision_relevant, bool):
            raise ValueError("decision_relevant must be a bool")
        with self._locked():
            state, count, previous = self._current()
            at = now()
            sequence = count + 1
            decision_revision = int(state.get("decision_revision") or 0) + int(
                decision_relevant)
            row = {"sequence": sequence, "event_id": uuid.uuid4().hex,
                   "state_revision": int(state.get("state_revision") or 0) + 1,
                   "decision_revision": decision_revision,
                   "decision_relevant": decision_relevant,
                   "at": at, "run_id": self.run_id,
                   "repository": str(self.repository), "phase": phase,
                   "event": event, "status": status,
                   "details": dict(details or {}),
                   # The event log is the recovery source if the process dies after the
                   # append but before the snapshot write. Keep the whole phase projection
                   # patch here, not just the small compatibility patch for older readers.
                   "state_patch": {"status": status, **dict(phase_state or {}),
                                   **dict(event_patch or {})},
                   "previous_hash": previous}
            # Repeated phase projections are evidence references, not copies in the
            # journal. The snapshot still exposes the full current phase to old readers.
            for field, limit in (("state_patch", 65536), ("details", 16384)):
                material = _canonical(row[field])
                if len(material) > limit:
                    identity = hashlib.sha256(material).hexdigest()
                    destination = self.root / "run_state_blobs" / f"{identity}.json"
                    if not destination.exists():
                        atomic_json(destination, row[field])
                    row[field + "_digest"] = identity
                    row[field] = {}
            row["event_hash"] = hashlib.sha256(_canonical(row)).hexdigest()
            self._append(row)
            self._apply_event(state, row)
            self._refresh_compat_fields(state)
            atomic_json(self.state_path, state)
            return state
