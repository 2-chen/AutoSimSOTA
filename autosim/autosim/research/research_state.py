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

from .common import atomic_json, now, read_json


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


def verify_event_rows(rows: list[dict[str, Any]]) -> str:
    """Verify ordered sequence numbers and the hash chain; return its head hash."""
    previous = ""
    for expected, row in enumerate(rows, start=1):
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
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ResearchStateError("run event log has an unsupported schema")
        self._identity(record, path=self.events_path)
        rows = record.get("rows")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ResearchStateError("run event log rows are invalid")
        head = verify_event_rows(rows)
        return rows, head

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

    @staticmethod
    def _apply_event(state: dict[str, Any], row: dict[str, Any]) -> None:
        phase = str(row.get("phase") or "unknown")
        phases = state.setdefault("phases", {})
        phase_state = phases.setdefault(phase, {})
        patch = row.get("state_patch") or {}
        if isinstance(patch, dict):
            phase_state.update(patch)
        phase_state.update(status=row.get("status"), updated_at=row.get("at"),
                           last_event_id=row.get("event_id"),
                           event_sequence=row.get("sequence"))
        state.update(phase=phase, status=row.get("status"), updated_at=row.get("at"),
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
            rows, head = self._read_events()
            state = self._read_snapshot()
            if not rows and not state:
                return None
            before = int(state.get("event_sequence") or 0)
            state = self._reconcile(state, rows)
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
            rows, previous = self._read_events()
            state = self._reconcile(self._read_snapshot(), rows)
            at = now()
            sequence = len(rows) + 1
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
            row["event_hash"] = hashlib.sha256(_canonical(row)).hexdigest()
            rows.append(row)
            atomic_json(self.events_path, {"schema_version": 1, "run_id": self.run_id,
                                          "repository": str(self.repository),
                                          "rows": rows})

            phases = state.setdefault("phases", {})
            existing = phases.get(phase) or {}
            updated = {**existing, **dict(phase_state or {}),
                       "status": status, "updated_at": at,
                       "last_event_id": row["event_id"], "event_sequence": sequence}
            phases[phase] = updated
            state.update(schema_version=1, run_id=self.run_id,
                         repository=str(self.repository), phase=phase, status=status,
                         state_revision=int(state.get("state_revision") or 0) + 1,
                         decision_revision=decision_revision,
                         event_sequence=sequence, event_head_hash=row["event_hash"],
                         last_event_id=row["event_id"], updated_at=at, phases=phases)
            self._refresh_compat_fields(state)
            atomic_json(self.state_path, state)
            return state
