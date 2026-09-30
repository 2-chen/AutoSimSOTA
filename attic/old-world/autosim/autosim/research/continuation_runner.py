"""Product-owned bounded research relaunch, independent of the SCO transport."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .common import atomic_json, digest, read_json
from .continuation import ContinuationRefused, ContinuationSupervisor
from .execution_activation import ExecutionActivation, verify_prepared_activation


class _CommittedReceipts(Sequence[Path]):
    """Re-scan at both supervisor checkpoints, including fast child commits."""

    def __init__(self, directory: Path, mirror: list[Path] | None):
        self.directory, self.mirror = directory, mirror

    def _snapshot(self) -> list[Path]:
        rows = sorted(self.directory.glob("*.json"))
        if self.mirror is not None:
            self.mirror[:] = rows
        return rows

    def __iter__(self):
        return iter(self._snapshot())

    def __len__(self):
        return len(self._snapshot())

    def __getitem__(self, index):
        return self._snapshot()[index]


def run_bounded_research(allocation_root: Path, *, manifest: dict, policy: dict,
                         deadline_epoch: float, env: Mapping[str, str] | None = None,
                         budget_check: Callable[[float], bool] | None = None,
                         progress_receipts: list[Path] | None = None) -> dict:
    """All commands remain the frozen research argv; repair text cannot supply one."""
    from .resume_boundary import inspect_resume_boundary, load_resume_receipt
    root = Path(allocation_root).absolute()
    source, run_root = root / "source", Path(manifest["run_root"])
    science = deepcopy(manifest["scientific_contract"])
    command = tuple(manifest["research_command"])
    limits = deepcopy(policy["continuation"])
    supervisor = ContinuationSupervisor(root / "continuation", deadline_epoch=deadline_epoch,
        max_restarts=limits["max_relaunches"] if limits["enabled"] else 0,
        max_no_progress=limits["max_no_progress"])
    manager = ExecutionActivation(root / "execution", source_root=source,
        source_files=read_json(root / "source_manifest.json"), scientific_contract=science,
        base_execution_revision=manifest["production_code_identity"], deadline_epoch=deadline_epoch)
    active = manager.register_baseline()
    environment = dict(os.environ if env is None else env)
    # A relaunched host must never inherit a different run's resume authority.
    for key in ("AUTOSIM_RESUME_RECEIPT", "AUTOSIM_RESUME_RECEIPT_SHA256",
                "AUTOSIM_EXECUTION_COMPATIBILITY", "AUTOSIM_EXECUTION_COMPATIBILITY_SHA256"):
        environment.pop(key, None)
    progress = _CommittedReceipts(run_root / "harness/artifact_commits", progress_receipts)
    boundary, attempts = None, []
    stage = "initialize"
    final_opened = False
    result = {"returncode": 2, "status": "not_started", "attempts": attempts}
    while True:
        try:
            verify_prepared_activation(active)
            process_env = {**environment, **active["env"],
                           "AUTOSIM_CONTINUATION_ATTEMPT": str(len(read_json(supervisor.path)["attempts"]) + 1)}
            compatibility = active.get("compatibility")
            if boundary is not None:
                boundary_path = Path(boundary["receipt_path"])
                boundary_sha = boundary["receipt_sha256"]
                # Use the persisted host envelope, never a mutable in-memory verdict.
                checked = load_resume_receipt(boundary_path, expected_sha256=boundary_sha,
                    run_root=run_root, scientific_contract=science,
                    execution_revision=active["execution_revision"])
                boundary = {**checked, "receipt_sha256": boundary_sha}
                process_env.update(AUTOSIM_RESUME_RECEIPT=str(boundary_path),
                                   AUTOSIM_RESUME_RECEIPT_SHA256=boundary_sha)
                if compatibility is not None:
                    compatibility = {**compatibility, "previous_workers_reaped": True}
                    compatibility_path = root / "continuation/active_compatibility.json"
                    atomic_json(compatibility_path, compatibility)
                    process_env.update(AUTOSIM_EXECUTION_COMPATIBILITY=str(compatibility_path),
                                       AUTOSIM_EXECUTION_COMPATIBILITY_SHA256=digest(compatibility_path))
            attempt = supervisor.run_once(command, cwd=source,
                execution_revision=active["execution_revision"], scientific_contract=science,
                stage=stage, final_opened=final_opened, allocation_reconciled=True,
                workers=boundary.get("workers", ()) if boundary else (),
                resume_receipt=boundary["resume_receipt"] if boundary else None,
                validation_receipt=active["validation_receipt"], compatibility=compatibility,
                rollback=active.get("rollback", False), progress_receipts=progress,
                run_state_path=run_root / "run_state.json", env=process_env, budget_check=budget_check)
            attempts.append(attempt)
            result.update(returncode=attempt.get("returncode") or (0 if attempt["status"] == "completed" else 2),
                          status=attempt["status"], execution_revision=active["execution_revision"])
            atomic_json(root / "continuation/result.json", result)
            if attempt["status"] == "completed" or not limits["enabled"]:
                break
            journal = read_json(supervisor.path)
            if attempt["status"] in {"budget_exhausted", "interrupted", "invalid_progress", "invalid_run_state"}:
                result.update(status="stopped", reason="prior attempt is terminal; no automatic replay")
                break
            if len(journal["attempts"]) >= journal["max_restarts"] + 1:
                result.update(status="stopped", reason="continuation restart limit exhausted")
                break
            if journal["no_progress_count"] >= journal["max_no_progress"]:
                result.update(status="stopped", reason="continuation made no new committed progress")
                break
            if budget_check is not None and not budget_check(30):
                result.update(status="stopped", reason="original allocation budget or cancellation")
                break
            boundary = inspect_resume_boundary(root, run_root, scientific_contract=science,
                execution_revision=active["execution_revision"], previous_attempt=attempt,
                deadline_epoch=deadline_epoch, probe_attempt_limit=policy["lifecycle"]["startup_attempts"],
                compatibility=compatibility)
            if not boundary["allowed"]:
                result.update(status="stopped", reason=boundary["reason"])
                break
            stage = boundary["resume_receipt"]["stage"]
            final_opened = bool(read_json(run_root / "run_state.json").get("final_confirmation_opened"))
            request_path = run_root / "repair_activation_request.json"
            consumed_path = root / "continuation/consumed_repairs.json"
            consumed = read_json(consumed_path) if consumed_path.is_file() else []
            if request_path.is_file() and digest(request_path) not in consumed:
                # A candidate may not restart the immutable probe contract under
                # another executor identity; it must preserve the old attempt ledger.
                if any((run_root / "device_probe").rglob("probe_contract.json")):
                    result.update(status="stopped", reason="probe revision migration requires a distinct admitted execution")
                    break
                request = read_json(request_path)
                active = manager.prepare_activation(request, from_revision=active["execution_revision"])
                atomic_json(consumed_path, [*consumed, digest(request_path)])
            elif active["execution_revision"] != manifest["production_code_identity"]:
                active = manager.prepare_rollback(from_revision=active["execution_revision"])
            if active["execution_revision"] != attempt["execution_revision"]:
                # Activation and rollback each need an envelope for the destination
                # revision. Editing the old receipt would break its hash and authority.
                boundary = inspect_resume_boundary(root, run_root, scientific_contract=science,
                    execution_revision=active["execution_revision"], previous_attempt=attempt,
                    deadline_epoch=deadline_epoch, probe_attempt_limit=policy["lifecycle"]["startup_attempts"],
                    compatibility=active.get("compatibility"))
                if not boundary["allowed"]:
                    result.update(status="stopped", reason=boundary["reason"])
                    break
        except (ContinuationRefused, RuntimeError, ValueError, OSError) as exc:
            result.update(status="stopped", reason=str(exc)[:500], error_type=type(exc).__name__)
            break
    atomic_json(root / "continuation/result.json", result)
    return result
