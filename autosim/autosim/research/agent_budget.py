"""Persistent, conservative USD accounting for coding-agent turns in one run.

    Provider per-turn limits can be exceeded slightly and interrupted turns may not return a
    usage receipt. Reservations therefore count against the run ceiling before a request starts;
    only a usage-backed cost for the frozen accounting basis releases the unused portion. A final partial reservation is
    refused by default: giving a provider a tiny remaining cap is not a safe substitute for reserving the
    configured full turn, because its final response may exceed that cap. Unknown usage remains
    reserved until explicitly reconciled from provider usage or billing evidence. DeepSeek's
    request-level gate may opt into partial allowances and atomically extend live reservations,
    because it reserves each request ceiling before forwarding; other providers stay strict.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import uuid
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .common import atomic_json, now, read_json


class AgentCostBudgetError(RuntimeError):
    """The persistent model-cost budget cannot safely admit another provider turn."""


def recovery_floor(output: Path, limit_usd: float) -> float:
    """Keep a configurable portion of the existing ceiling for execution/recovery."""
    path = Path(output)/'scheduler_policy.json'
    raw = read_json(path) if path.is_file() and not path.is_symlink() else {}
    value = raw.get('recovery_model_reserve_usd', min(2.0, float(limit_usd)*.1))
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
        raise AgentCostBudgetError('invalid recovery_model_reserve_usd')
    return min(float(value), float(limit_usd))


class AgentCostLedger:
    SCHEMA_VERSION = 1
    MAX_ENTRIES = 10_000

    def wait_for_settlement(self, *, timeout: float = 30, minimum_usd: float = .25) -> dict:
        """Read-only bounded admission wait; unknown costs are NEVER released here."""
        if not math.isfinite(timeout) or not 0 <= timeout <= 60 or not math.isfinite(minimum_usd) or minimum_usd <= 0:
            raise ValueError('invalid bounded settlement wait')
        deadline = time.monotonic() + timeout
        while True:
            with self._locked():
                raw = self._read()
                spent, held = self._totals(raw)
                remaining = max(0.0, self.limit_usd-spent-held)
                pending = any(row['status'] == 'reserved' for row in raw['entries'])
            if remaining >= minimum_usd:
                return {'status':'available', 'remaining_usd':remaining}
            if not pending or time.monotonic() >= deadline:
                return {'status':'waiting' if pending else 'blocked', 'remaining_usd':remaining,
                        'reason':'insufficient safe allowance; usage holds preserved'}
            time.sleep(min(.25, max(0, deadline-time.monotonic())))

    def __init__(self, output: Path, *, run_id: str, limit_usd: float,
                 cost_basis: str = "cli_reported"):
        if (not isinstance(limit_usd, (int, float)) or isinstance(limit_usd, bool) or
                not math.isfinite(float(limit_usd)) or float(limit_usd) <= 0):
            raise ValueError("run-level model budget must be a finite positive USD amount")
        self.output = Path(output).expanduser().resolve()
        self.run_id = str(run_id)
        if not self.run_id or len(self.run_id) > 160:
            raise ValueError("run id must contain 1–160 characters")
        self.limit_usd = float(limit_usd)
        if cost_basis not in {"cli_reported", "deepseek_official_estimate_v1"}:
            raise ValueError("unknown model-cost accounting basis")
        self.cost_basis = cost_basis
        self.directory = self.output / "agent"
        self.path = self.directory / "cost_ledger.json"
        self.lock_path = self.directory / "cost_ledger.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if self.output.is_symlink():
            raise AgentCostBudgetError("run output may not be a symlink")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise AgentCostBudgetError("agent ledger directory may not be a symlink")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
        except OSError as exc:
            raise AgentCostBudgetError("cannot safely open model-cost ledger lock") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read(self) -> dict[str, Any]:
        if self.path.is_symlink():
            raise AgentCostBudgetError("model-cost ledger may not be a symlink")
        if not self.path.is_file():
            return {"schema_version": self.SCHEMA_VERSION, "run_id": self.run_id,
                    "limit_usd": self.limit_usd, "cost_basis": self.cost_basis,
                    "entries": []}
        if self.path.stat().st_size > 8 * 1024 * 1024:
            raise AgentCostBudgetError("model-cost ledger exceeds its size limit")
        try:
            ledger = read_json(self.path)
        except (OSError, ValueError, TypeError) as exc:
            raise AgentCostBudgetError("model-cost ledger is unreadable") from exc
        if (not isinstance(ledger, dict) or
                ledger.get("schema_version") != self.SCHEMA_VERSION or
                ledger.get("run_id") != self.run_id or
                not isinstance(ledger.get("entries"), list)):
            raise AgentCostBudgetError("model-cost ledger identity or schema is invalid")
        old_limit = ledger.get("limit_usd")
        if (not isinstance(old_limit, (int, float)) or isinstance(old_limit, bool) or
                not math.isclose(float(old_limit), self.limit_usd, rel_tol=0, abs_tol=1e-9)):
            raise AgentCostBudgetError("cannot silently change a run's total model-cost budget")
        if len(ledger["entries"]) > self.MAX_ENTRIES:
            raise AgentCostBudgetError("model-cost ledger has too many turn entries")
        if ledger.get("cost_basis", "cli_reported") != self.cost_basis:
            raise AgentCostBudgetError(
                "model-cost accounting basis changed; start a fresh run instead of "
                "reinterpreting the historical ledger")
        return ledger

    @staticmethod
    def _totals(ledger: dict[str, Any]) -> tuple[float, float]:
        settled = 0.0
        reserved = 0.0
        for row in ledger["entries"]:
            if not isinstance(row, dict):
                raise AgentCostBudgetError("model-cost ledger contains a malformed entry")
            status = row.get("status")
            value = row.get("actual_usd")
            hold = row.get("reserved_usd")
            if (not isinstance(hold, (int, float)) or isinstance(hold, bool) or
                    not math.isfinite(float(hold)) or float(hold) < 0):
                raise AgentCostBudgetError("model-cost ledger contains an invalid reservation")
            if status in {"reserved", "unknown"}:
                reserved += float(hold)
            elif status == "settled":
                if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                        not math.isfinite(float(value)) or float(value) < 0):
                    raise AgentCostBudgetError("settled model-cost entry has invalid usage")
                settled += float(value)
            elif status != "cancelled":
                raise AgentCostBudgetError("model-cost ledger contains an unknown status")
        return settled, reserved

    def reserve(self, *, session_id: str, role: str, requested_usd: float,
                allow_partial: bool = False) -> dict[str, Any]:
        if (not isinstance(requested_usd, (int, float)) or isinstance(requested_usd, bool) or
                not math.isfinite(float(requested_usd)) or float(requested_usd) <= 0):
            raise ValueError("turn model budget must be a finite positive USD amount")
        with self._locked():
            ledger = self._read()
            settled, held = self._totals(ledger)
            remaining = max(0.0, self.limit_usd - settled - held)
            if role == 'recorder':
                remaining = max(0.0, remaining-recovery_floor(self.output,self.limit_usd))
            requested = float(requested_usd)
            if allow_partial and remaining > 1e-9:
                requested = min(requested, remaining)
            if remaining + 1e-9 < requested:
                raise AgentCostBudgetError(
                    f"run model-cost budget exhausted; ${settled + held:.6f} of "
                    f"${self.limit_usd:.6f} is spent or reserved, leaving "
                    f"${remaining:.6f}; a full ${requested:.6f} turn reservation is required")
            if len(ledger["entries"]) >= self.MAX_ENTRIES:
                raise AgentCostBudgetError("model-cost ledger reached its turn-entry limit")
            reservation_id = uuid.uuid4().hex
            entry = {"reservation_id": reservation_id,
                     "session_id": str(session_id)[:100], "role": str(role)[:80],
                     "status": "reserved", "requested_usd": requested,
                     "provider_limit_usd": requested, "reserved_usd": requested,
                     "actual_usd": None, "reserved_at": now()}
            ledger["entries"].append(entry)
            ledger["updated_at"] = now()
            atomic_json(self.path, ledger)
            return {"reservation_id": reservation_id, "allowed_usd": requested,
                    "spent_usd": settled, "reserved_usd": held + requested,
                    "remaining_usd": max(0.0, self.limit_usd - settled - held - requested)}

    def extend(self, reservation_id: str, required_usd: float) -> float:
        """Grow a live local allowance atomically, never beyond the run ceiling."""
        if (not isinstance(required_usd, (int, float)) or isinstance(required_usd, bool) or
                not math.isfinite(required_usd) or required_usd <= 0):
            raise ValueError("required reservation must be finite and positive")
        with self._locked():
            ledger = self._read()
            entry = next((row for row in ledger["entries"]
                          if row.get("reservation_id") == reservation_id), None)
            if entry is None or entry.get("status") != "reserved":
                raise AgentCostBudgetError("only a live reservation can be extended")
            settled, held = self._totals(ledger)
            current = float(entry["reserved_usd"])
            free = max(0.0, self.limit_usd-settled-held)
            if entry.get('role') == 'recorder':
                free = max(0.0, free-recovery_floor(self.output,self.limit_usd))
            if required_usd > current + free + 1e-9:
                raise AgentCostBudgetError("run model-cost budget cannot reserve this request")
            entry["reserved_usd"] = max(current, required_usd)
            entry["provider_limit_usd"] = entry["reserved_usd"]
            entry.setdefault("adjustments", []).append({"at": now(),
                "from_usd": current, "to_usd": entry["reserved_usd"],
                "reason": "request ceiling exceeds local allowance"})
            ledger["updated_at"] = now()
            atomic_json(self.path, ledger)
            return float(entry["reserved_usd"])

    def settle(self, reservation_id: str, *, actual_usd: float | None,
               launched: bool, details: dict[str, Any] | None = None,
               request_bound_usd: float | None = None) -> dict[str, Any]:
        if (actual_usd is not None and
                (not isinstance(actual_usd, (int, float)) or isinstance(actual_usd, bool) or
                 not math.isfinite(float(actual_usd)) or float(actual_usd) < 0)):
            raise ValueError("provider-reported model cost must be finite and non-negative")
        with self._locked():
            ledger = self._read()
            rows = ledger["entries"]
            entry = next((row for row in rows if isinstance(row, dict) and
                          row.get("reservation_id") == reservation_id), None)
            if entry is None:
                raise AgentCostBudgetError("model-cost reservation was not found")
            if request_bound_usd is not None:
                if (self.cost_basis != "deepseek_official_estimate_v1" or
                        isinstance(request_bound_usd, bool) or
                        not isinstance(request_bound_usd, (float, int)) or
                        not math.isfinite(request_bound_usd) or request_bound_usd < 0 or
                        request_bound_usd > float(entry["reserved_usd"]) + 1e-9 or
                        not details or not details.get("pricing_ref")):
                    raise ValueError("request bound requires a safe priced-gateway receipt")
            already_finalized = False
            if entry.get("status") in {"settled", "cancelled", "unknown"}:
                if (entry.get("status") == "settled" and actual_usd is not None and
                        math.isclose(float(entry.get("actual_usd") or 0),
                                     float(actual_usd), rel_tol=0, abs_tol=1e-9)):
                    already_finalized = True
                elif entry.get("status") == "unknown" and actual_usd is None:
                    already_finalized = True
                elif entry.get("status") == "cancelled" and not launched:
                    already_finalized = True
                else:
                    raise AgentCostBudgetError("model-cost reservation was already finalized")
            if not already_finalized:
                if details is not None:
                    if not isinstance(details, dict) or len(json.dumps(details)) > 4096:
                        raise ValueError("model-cost settlement details are invalid")
                    entry["details"] = details
                if not launched:
                    entry.update(status="cancelled", reserved_usd=0.0, actual_usd=0.0,
                                 finished_at=now())
                elif actual_usd is None:
                    # Preserve unknown usage, but the trusted request gate can prove
                    # how much was actually admitted; unused turn allowance is not spend.
                    entry.update(status="unknown", actual_usd=None, finished_at=now(),
                                 reserved_usd=(float(request_bound_usd) if request_bound_usd
                                     is not None else entry["reserved_usd"]))
                else:
                    entry.update(status="settled", actual_usd=float(actual_usd),
                                 reserved_usd=0.0, finished_at=now())
                ledger["updated_at"] = now()
                atomic_json(self.path, ledger)
        return self.snapshot()

    def reconcile_unknown(self, reservation_id: str, *, actual_usd: float,
                          evidence: str) -> dict[str, Any]:
        """Replace an unknown-usage hold with a provider-verified cost.

        Recovery is explicit so a routine interrupted call cannot free its reservation.
        ``evidence`` is a short receipt reference/description, not receipt contents or secrets.
        """
        if (not isinstance(actual_usd, (int, float)) or isinstance(actual_usd, bool) or
                not math.isfinite(float(actual_usd)) or float(actual_usd) < 0):
            raise ValueError("reconciled provider cost must be finite and non-negative")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 240:
            raise ValueError("reconciliation requires a short provider-usage evidence reference")
        evidence = evidence.strip()
        with self._locked():
            ledger = self._read()
            entry = next((row for row in ledger["entries"] if isinstance(row, dict) and
                          row.get("reservation_id") == reservation_id), None)
            if entry is None:
                raise AgentCostBudgetError("model-cost reservation was not found")
            if entry.get("status") == "settled":
                same = (math.isclose(float(entry.get("actual_usd") or 0),
                                      float(actual_usd), rel_tol=0, abs_tol=1e-9) and
                        entry.get("reconciliation_evidence") == evidence)
                if not same:
                    raise AgentCostBudgetError("model-cost reservation was already reconciled")
            elif entry.get("status") != "unknown":
                raise AgentCostBudgetError("only unknown provider usage can be reconciled")
            else:
                entry.update(status="settled", actual_usd=float(actual_usd),
                             reserved_usd=0.0, reconciled_at=now(),
                             reconciliation_evidence=evidence)
                ledger["updated_at"] = now()
                atomic_json(self.path, ledger)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            ledger = self._read()
            settled, held = self._totals(ledger)
            return {"run_id": self.run_id, "limit_usd": self.limit_usd,
                    "cost_basis": self.cost_basis,
                    "spent_usd": settled, "reserved_usd": held,
                    "active_reserved_usd": sum(r['reserved_usd'] for r in ledger['entries'] if r['status']=='reserved'),
                    "unknown_reserved_usd": sum(r['reserved_usd'] for r in ledger['entries'] if r['status']=='unknown'),
                    "recovery_reserve_usd": recovery_floor(self.output,self.limit_usd),
                    "remaining_usd": max(0.0, self.limit_usd - settled - held),
                    "unknown_entries": sum(row.get("status") == "unknown"
                                            for row in ledger["entries"]
                                            if isinstance(row, dict)),
                    "entry_count": len(ledger["entries"])}

    def reconcile_receipts(self, *, apply: bool = False) -> dict:
        from .budget_receipts import reconcile
        return reconcile(self, apply=apply)
