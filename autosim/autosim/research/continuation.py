"""Trusted outer supervision: bounded new processes, revision activation and rollback.

Never reload Python modules in a running research process. This host owns its
journal and receipts; repair candidates must not be allowed to write them.
"""
from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .artifact_commits import lifecycle_compatible
from .common import atomic_json, digest, exclusive, now, object_digest, read_json


class ContinuationRefused(RuntimeError):
    """The host cannot establish a safe, affordable continuation boundary."""


def process_identity(pid: int) -> dict:
    result = {"pid": int(pid), "host": os.uname().nodename}
    try:
        # Some sandboxes expose a host /proc inside a nested PID namespace.
        # Its /proc/<child pid> would describe an unrelated host process.
        if int(Path("/proc/self/stat").read_text().split()[0]) != os.getpid():
            result["start_ticks"] = None
            return result
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result.update(start_ticks=fields[19], process_state=fields[0], pgid=int(fields[2]))
    except (OSError, IndexError, ValueError):
        result["start_ticks"] = None
    return result


def process_gone(identity: dict) -> bool:
    """Unknown/remote identity is not proof of termination; never signal by stale PID."""
    if identity.get("host") != os.uname().nodename:
        return bool(identity.get("termination_verified") is True and identity.get("termination_evidence_sha256"))
    pid = identity.get("pid", identity.get("worker_pid"))
    if type(pid) is not int or pid <= 0:
        return False
    gone = False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        gone = True
    except PermissionError:
        return False
    if not gone:
        current = process_identity(pid)
        gone = (current.get("process_state") == "Z" or
                bool(identity.get("start_ticks") and current.get("start_ticks")
                     and current["start_ticks"] != identity["start_ticks"]))
    if not gone:
        return False
    # The recorded process group may still contain a live orphan after leader exit.
    pgid = identity.get("pgid", identity.get("worker_pgid"))
    if pgid:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if int(Path("/proc/self/stat").read_text().split()[0]) != os.getpid():
            return False
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                member = process_identity(int(entry.name))
                if member.get("pgid") == pgid and member.get("process_state") != "Z":
                    return False
    return True


class ContinuationSupervisor:
    def __init__(self, root: Path, *, deadline_epoch: float, max_restarts: int = 3,
                 max_no_progress: int = 2, clock: Callable[[], float] = time.time):
        if not math.isfinite(deadline_epoch) or max_restarts < 0 or max_no_progress < 1:
            raise ValueError("invalid continuation bounds")
        self.root = Path(root).absolute()
        self.path = self.root / "continuation.json"
        self.clock = clock
        with exclusive(self.root / "host.lock"):
            if self.path.exists():
                row = read_json(self.path)
                # A new host invocation can tighten, never renew, previous limits.
                row["deadline_epoch"] = min(row["deadline_epoch"], deadline_epoch)
                row["max_restarts"] = min(row["max_restarts"], max_restarts)
                row["max_no_progress"] = min(row["max_no_progress"], max_no_progress)
            else:
                row = {"schema_version": 1, "kind": "harness_continuation", "created_at": now(),
                       "deadline_epoch": deadline_epoch, "max_restarts": max_restarts,
                       "max_no_progress": max_no_progress, "no_progress_count": 0,
                       "attempts": [], "active_revision": None, "verified_revisions": [],
                       "final_confirmation_opened": False, "progress_commits": []}
            atomic_json(self.path, row)

    def _known_processes(self, attempt: dict | None) -> list[dict]:
        if not attempt:
            return []
        identities = [attempt[name] for name in ("process", "worker") if attempt.get(name)]
        receipt = self.root / f"attempt_{attempt['attempt']}_worker.json"
        if receipt.is_file() and not attempt.get("worker"):
            item = read_json(receipt)
            identities.append({"pid": item["worker_pid"], "pgid": item["worker_pgid"],
                               "host": attempt.get("process", {}).get("host")})
        return identities

    @staticmethod
    def _validation(receipt: dict | None, revision: str, scientific_hash: str, cwd: Path) -> None:
        if not receipt or receipt.get("passed") is not True or receipt.get("execution_revision") != revision:
            raise ContinuationRefused("new revision has no passing host validation receipt")
        if receipt.get("scientific_contract_sha256") != scientific_hash or not receipt.get("source_files"):
            raise ContinuationRefused("revision validation does not bind science and source files")
        for relative, expected in receipt["source_files"].items():
            path = (cwd / relative).resolve()
            if not path.is_relative_to(cwd.resolve()) or not path.is_file() or digest(path) != expected:
                raise ContinuationRefused("validated source changed or escaped revision directory")

    @staticmethod
    def _progress(paths: Sequence[Path], scientific_hash: str) -> list[str]:
        ids = []
        for path in paths:
            row = read_json(Path(path))
            if row.get("state") != "committed":
                continue
            expected = object_digest({k: v for k, v in row.items() if k != "commit_sha256"})
            if (row.get("commit_sha256") != expected
                    or row.get("scientific_contract_sha256") != scientific_hash):
                raise ContinuationRefused("progress requires intact trusted scientific artifact commits")
            ids.append(expected)
        return sorted(set(ids))

    def reconcile_launch(self, *, workers: Sequence[dict], termination_evidence: dict) -> dict:
        """Resolve a host crash using externally observed process/allocation termination.

        This never runs research and does not erase a charged attempt or its budget.
        """
        if termination_evidence.get("verified") is not True or not termination_evidence.get("evidence_sha256"):
            raise ContinuationRefused("launch reconciliation requires verified termination evidence")
        with exclusive(self.root / "host.lock"):
            row = read_json(self.path)
            if not row["attempts"]:
                raise ContinuationRefused("no launch to reconcile")
            last = row["attempts"][-1]
            identities = [*workers, *self._known_processes(last)]
            if any(not process_gone(identity) for identity in identities):
                raise ContinuationRefused("a prior process or process group is alive or unknown")
            if last["status"] not in ("intent", "running"):
                if last.get("cleanup_unconfirmed"):
                    last.update(cleanup_unconfirmed=False, termination_evidence=termination_evidence)
                    atomic_json(self.path, row)
                return row
            last.update(status="reconciled_terminated", termination_evidence=termination_evidence,
                        finished_at=now())
            row["no_progress_count"] += 1
            atomic_json(self.path, row)
            return row

    def run_once(self, command: Sequence[str], *, cwd: Path, execution_revision: str,
                 scientific_contract: dict, stage: str, final_opened: bool = False,
                 workers: Sequence[dict] = (), allocation_reconciled: bool = False,
                 resume_receipt: dict | None = None, validation_receipt: dict | None = None,
                 compatibility: dict | None = None, rollback: bool = False,
                 progress_receipts: Sequence[Path] = (), run_state_path: Path | None = None,
                 env: Mapping[str, str] | None = None, cleanup_seconds: float = 10,
                 budget_check: Callable[[float], bool] | None = None) -> dict:
        """Run exactly one bounded child; caller decides whether another attempt is needed.

        resume_receipt is host-issued and binds a verified safe boundary to the
        stage/scientific hash. In final, it must also certify no episode started.
        Commands/receipts must come from the host allowlist, not model text.
        """
        if not command or any(not isinstance(value, str) or not value for value in command):
            raise ValueError("continuation command must be a nonempty argv")
        if not 0 < cleanup_seconds <= 60:
            raise ValueError("invalid cleanup reserve")
        cwd = Path(cwd).absolute()
        scientific_hash = object_digest(scientific_contract)
        with exclusive(self.root / "host.lock"):
            row = read_json(self.path)
            previous = row["attempts"][-1] if row["attempts"] else None
            if row.get("scientific_contract_sha256", scientific_hash) != scientific_hash:
                raise ContinuationRefused("science changed: start a distinct experiment")
            sealed = row["final_confirmation_opened"] or final_opened
            if run_state_path is not None and Path(run_state_path).is_file():
                state = read_json(Path(run_state_path))
                sealed = sealed or bool(state.get("final_confirmation_opened"))
                if state.get("status") in {"completed", "completed_without_target_improvement"}:
                    raise ContinuationRefused("research already completed; do not replay it")
            if sealed and stage not in {"final_confirmation", "export_validation", "complete"}:
                raise ContinuationRefused("final opened: research stages may not be replayed")
            if not allocation_reconciled:
                raise ContinuationRefused("allocation and old workers have not been reconciled")
            identities = [*workers, *self._known_processes(previous)]
            if any(not process_gone(identity) for identity in identities):
                raise ContinuationRefused("a prior process or process group is alive or unknown")
            if previous and previous["status"] in {"intent", "running"}:
                raise ContinuationRefused("prior launch outcome unresolved; reconcile before retry")
            if previous and previous.get("cleanup_unconfirmed"):
                raise ContinuationRefused("prior process cleanup is unconfirmed")
            if previous:
                if previous["status"] in {"budget_exhausted", "interrupted", "completed", "invalid_progress", "invalid_run_state"}:
                    raise ContinuationRefused("prior attempt is terminal; no automatic replay")
                if not resume_receipt or not (resume_receipt.get("safe_boundary") is True
                        and resume_receipt.get("scientific_contract_sha256") == scientific_hash
                        and resume_receipt.get("stage") == stage):
                    raise ContinuationRefused("resume requires a verified compatible phase boundary")
            if sealed and stage == "final_confirmation" and not (
                    resume_receipt and resume_receipt.get("safe_boundary") is True
                    and resume_receipt.get("episode_started") is False
                    and resume_receipt.get("stage") == stage
                    and resume_receipt.get("scientific_contract_sha256") == scientific_hash):
                raise ContinuationRefused("final continuation requires certified pre-reset failure")
            if len(row["attempts"]) >= row["max_restarts"] + 1:
                raise ContinuationRefused("continuation restart limit exhausted")
            if row["no_progress_count"] >= row["max_no_progress"]:
                raise ContinuationRefused("continuation made no new committed progress")
            if row["deadline_epoch"] - self.clock() <= cleanup_seconds + 1:
                raise ContinuationRefused("absolute continuation budget exhausted")
            if budget_check is not None and not budget_check(cleanup_seconds + 1):
                raise ContinuationRefused("shared allocation/API budget refused continuation")
            old = row["active_revision"]
            if old != execution_revision:
                self._validation(validation_receipt, execution_revision, scientific_hash, cwd)
                if old and not lifecycle_compatible(compatibility, old, execution_revision, scientific_hash):
                    raise ContinuationRefused("activation changes revision without lifecycle compatibility")
                if rollback and execution_revision not in row["verified_revisions"]:
                    raise ContinuationRefused("rollback target was never validated")
            elif validation_receipt is not None:
                self._validation(validation_receipt, execution_revision, scientific_hash, cwd)
            before = self._progress([p for p in progress_receipts if Path(p).exists()], scientific_hash)
            row["progress_commits"] = sorted(set(row["progress_commits"]) | set(before))
            attempt = {"attempt": len(row["attempts"]) + 1, "status": "intent", "started_at": now(),
                       "execution_revision": execution_revision, "stage": stage,
                       "command_sha256": object_digest(list(command)), "cwd": str(cwd),
                       "operation": "rollback" if rollback else "activate" if old != execution_revision else "continue"}
            row.update(scientific_contract_sha256=scientific_hash, final_confirmation_opened=sealed)
            row["attempts"].append(attempt)
            atomic_json(self.path, row)
            process = None
            worker_receipt = self.root / f"attempt_{attempt['attempt']}_worker.json"
            try:
                with (self.root / f"attempt_{attempt['attempt']}.log").open("a") as log:
                    guarded = [sys.executable, str(Path(__file__).with_name("process_guard.py")),
                               str(os.getpid()), str(worker_receipt), *command]
                    process = subprocess.Popen(guarded, cwd=cwd, env=dict(env) if env is not None else None,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    attempt.update(status="running", process=process_identity(process.pid))
                    row["active_revision"] = execution_revision
                    if execution_revision not in row["verified_revisions"]:
                        row["verified_revisions"].append(execution_revision)
                    atomic_json(self.path, row)
                    while True:
                        remaining = row["deadline_epoch"] - self.clock() - cleanup_seconds
                        if remaining <= 0 or (budget_check is not None and not budget_check(cleanup_seconds)):
                            attempt["status"] = "budget_exhausted"
                            break
                        try:
                            code = process.wait(timeout=min(5, remaining))
                            attempt.update(status="completed" if code == 0 else "failed", returncode=code)
                            break
                        except subprocess.TimeoutExpired:
                            pass
            except (KeyboardInterrupt, SystemExit):
                attempt["status"] = "interrupted"
                raise
            except OSError as exc:
                attempt.update(status="failed_to_start", error_type=type(exc).__name__)
            finally:
                if process is not None:
                    groups = {process.pid}
                    if worker_receipt.is_file():
                        worker = read_json(worker_receipt)
                        groups.add(int(worker["worker_pgid"]))
                        attempt["worker"] = {**process_identity(int(worker["worker_pid"])),
                                             "pgid": int(worker["worker_pgid"])}
                    for pgid in groups:
                        try:
                            os.killpg(pgid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    try:
                        process.wait(timeout=cleanup_seconds / 2)
                    except subprocess.TimeoutExpired:
                        pass
                    for pgid in groups:
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    try:
                        process.wait(timeout=cleanup_seconds / 2)
                    except subprocess.TimeoutExpired:
                        attempt["cleanup_unconfirmed"] = True
                    attempt["returncode"] = process.returncode
                    if attempt.get("worker") and not process_gone(attempt["worker"]):
                        attempt["cleanup_unconfirmed"] = True
                try:
                    after = self._progress([p for p in progress_receipts if Path(p).exists()], scientific_hash)
                except (OSError, ValueError, ContinuationRefused) as exc:
                    after = []
                    attempt.update(status="invalid_progress", progress_error_type=type(exc).__name__)
                new = sorted(set(after) - set(row["progress_commits"]))
                row["progress_commits"] = sorted(set(row["progress_commits"]) | set(after))
                row["no_progress_count"] = 0 if new else row["no_progress_count"] + 1
                if run_state_path is not None and Path(run_state_path).is_file():
                    try:
                        row["final_confirmation_opened"] = sealed or bool(
                            read_json(Path(run_state_path)).get("final_confirmation_opened"))
                    except (OSError, ValueError) as exc:
                        # Unknown final state is sealed, never permission to research again.
                        row["final_confirmation_opened"] = True
                        attempt.update(status="invalid_run_state", state_error_type=type(exc).__name__)
                attempt.update(finished_at=now(), new_progress_commits=new)
                atomic_json(self.path, row)
            return attempt
