"""Detached, budgeted native stage jobs controlled by the research Scheduler.

The model chooses a verified stage and a soft window. A separate worker runs the existing
native executor; it never accepts model-supplied shell text or arbitrary host paths.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .budget import RunBudget
from .common import atomic_json, atomic_text, digest, now, object_digest, read_json, sanitize_model_text
from .process_executor import capture_process_identity, inspect_process_identity


_JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
_SECRET = re.compile(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|AUTH", re.I)


def _path(output: Path, job_id: str) -> Path:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError("unsafe job id")
    output = Path(output).resolve(strict=True)
    directory = output / "native_jobs" / job_id
    if directory.is_symlink():
        raise ValueError("job directory is a symlink")
    return directory


def _active(output: Path) -> list[str]:
    root = Path(output) / "native_jobs"
    active = []
    for directory in root.iterdir() if root.is_dir() else []:
        if not _JOB_ID.fullmatch(directory.name) or directory.is_symlink():
            continue
        if (directory / "result.json").is_file():
            continue
        if (directory / "request.json").is_file():
            # A dead worker without a result is an unresolved side effect, not a free slot.
            active.append(directory.name)
    return active


def active_jobs(output: Path) -> list[str]:
    return _active(output)


def source_identity(output: Path, repo: Path) -> str:
    """Bind pre-existing checkout files without treating new training outputs as source."""
    manifest_path = Path(output) / "workspace_snapshot.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return ""
    manifest = read_json(manifest_path)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 2 or
            manifest.get("destination") != str(Path(repo).resolve(strict=True))):
        raise ValueError("job source manifest does not identify this checkout")
    entries = manifest.get("source_entries")
    if (not isinstance(entries, list) or
            object_digest(entries) != manifest.get("source_tree_fingerprint")):
        raise ValueError("job source inventory is invalid")
    current = []
    for row in entries:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError("job source inventory has a malformed entry")
        relative = Path(str(row[1]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("job source inventory has an unsafe path")
        path = Path(repo) / relative
        if row[0] == "file":
            if not path.is_file() or path.is_symlink():
                raise ValueError("an inventoried source file disappeared or changed type")
            current.append((str(relative), digest(path)))
        elif row[0] == "symlink":
            if not path.is_symlink() or len(row) != 3 or os.readlink(path) != row[2]:
                raise ValueError("an inventoried source link changed")
        elif row[0] != "directory":
            raise ValueError("job source inventory has an unknown entry type")
    return object_digest({"base": manifest["source_tree_fingerprint"],
                          "existing_files": current,
                          "derived_stages": digest(Path(output) / "derived_stages.json")})


def submit(output: Path, *, repo: Path, stage: str, settings: dict[str, Any],
           requested_seconds: float, reason: str, run_id: str = "derived",
           resources: dict | None = None) -> dict[str, Any]:
    output = Path(output).resolve(strict=True)
    repo = Path(repo).resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", stage):
        raise ValueError("unsafe stage name")
    from .scheduling import policy, resource_profile
    scheduling = policy(output)
    if stage != "train" and not scheduling:
        raise ValueError("detached jobs currently admit verified training stages only")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", run_id) or run_id in {".", ".."}:
        raise ValueError("unsafe run id")
    if not isinstance(settings, dict) or len(json.dumps(settings, default=str)) > 8000:
        raise ValueError("job settings are malformed or too large")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError("job needs a concise reason")
    if (not isinstance(requested_seconds, (int, float)) or
            isinstance(requested_seconds, bool) or not math.isfinite(requested_seconds) or requested_seconds <= 0):
        raise ValueError("job window must be positive")
    budget = RunBudget.existing(output)
    if budget is None or requested_seconds > budget.remaining():
        raise ValueError("job window exceeds the existing hard run budget")
    # Only commands already accepted by the derivation can be submitted.
    verified = read_json(output / "derived_stages.json")
    if not isinstance(verified, dict) or stage not in verified or not verified[stage].get("source"):
        raise ValueError("native stage has no verified command")
    source_stamp = source_identity(output, repo)
    profile = resource_profile(resources, gpu=settings.get("device") != "cpu") if scheduling else None
    if profile and profile["gpu"] is False and str(settings.get("device", "auto")).startswith("cuda"):
        raise ValueError("CPU resource profile conflicts with CUDA settings")
    if profile and profile["gpu"] and settings.get("device") == "cpu":
        raise ValueError("GPU resource profile conflicts with CPU settings")
    root = output / "native_jobs"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "submission.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = _active(output)
        if len(active) >= int(scheduling.get("native_slots", 1)):
            raise ValueError("another native job has an unresolved outcome")
        if any(read_json(_path(output, job) / "request.json").get("stage") == stage for job in active):
            raise ValueError("this stage already has an unresolved job; its output namespace is exclusive")
        job_id = uuid.uuid4().hex
        directory = _path(output, job_id)
        directory.mkdir(parents=True)
        # Publish a pending request under the same lock, before another Scheduler can submit.
        atomic_json(directory / "request.json", {"job_id": job_id, "pending": True})
    started = time.time()
    request = {"schema_version": 1, "job_id": job_id, "run_id": run_id,
               "repo": str(repo), "stage": stage, "settings": settings,
               "source_identity": source_stamp,
               "requested_seconds": float(requested_seconds),
               "reason": reason.strip(), "submitted_at": now(),
               "submitted_epoch": started}
    if scheduling:
        request["resources"] = profile
        if request["resources"]["gpu"] is False:
            request["settings"] = {**settings, "device": "cpu"}
    atomic_json(directory / "request.json", request)
    atomic_json(directory / "control.json", {
        "schema_version": 1, "job_id": job_id, "revision": 0,
        "deadline_epoch": started + requested_seconds, "cancelled": False,
        "state": "queued" if scheduling else "running"})
    package_root = Path(__file__).resolve().parents[2]
    environment = {key: value for key, value in os.environ.items()
                   if not _SECRET.search(key)}
    environment["PYTHONPATH"] = str(package_root)
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(
            [sys.executable, "-m", "autosim.research.native_jobs", "worker",
             "--output", str(output), "--job-id", job_id],
            cwd=repo, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        identity = capture_process_identity(worker, run_id=run_id,
                                            attempt_id=job_id, argv=worker.args)
        atomic_json(directory / "worker_identity.json", identity)
    except (OSError, ValueError):
        # The worker may already have finished. Its result or an unresolved job record
        # remains authoritative; never submit a second job merely because PID capture raced.
        pass
    return {"job_id": job_id, "status": "submitted", "stage": stage,
            "deadline_epoch": started + requested_seconds,
            "job_ref": f"native_jobs/{job_id}/request.json"}


def status(output: Path, job_id: str) -> dict[str, Any]:
    directory = _path(output, job_id)
    request = read_json(directory / "request.json")
    control = read_json(directory / "control.json")
    result_path = directory / "result.json"
    result = read_json(result_path) if result_path.is_file() else None
    identity_path = directory / "worker_identity.json"
    identity = (inspect_process_identity(read_json(identity_path))
                if identity_path.is_file() else {"status": "starting"})
    state = (str(result.get("status") or "unknown") if isinstance(result, dict) else
             ("queued" if control.get("state") == "queued" else "running")
             if identity.get("status") in {"matching_running", "starting"} else
             "outcome_unknown")
    progress: dict[str, Any] = {}
    attempts = Path(output) / "research" / str(request.get("run_id") or "derived") / "attempts"
    if attempts.is_dir():
        candidates = sorted(attempts.glob("*/receipt.json"),
                            key=lambda item: item.stat().st_mtime, reverse=True)[:50]
        for receipt_path in candidates:
            try:
                receipt = read_json(receipt_path)
            except (OSError, ValueError, TypeError):
                continue
            if (isinstance(receipt, dict) and
                    receipt.get("controller_decision_id") == job_id and
                    receipt.get("node_id") == request.get("stage")):
                attempt_id = str(receipt.get("attempt_id") or "")
                log = (Path(output) / "research" /
                       str(request.get("run_id") or "derived") /
                       str(request.get("stage")) / "attempts" / attempt_id / "output.log")
                progress = {"attempt_id": attempt_id,
                            "attempt_status": receipt.get("status"),
                            "receipt_ref": f"research/{request.get('run_id')}/attempts/"
                                           f"{attempt_id}/receipt.json"}
                if log.is_file() and not log.is_symlink():
                    stat = log.stat()
                    with log.open("rb") as source:
                        source.seek(max(0, stat.st_size - 3000))
                        tail = source.read(3000)
                    progress.update(last_output_epoch=stat.st_mtime,
                                    seconds_since_output=max(0.0, time.time() - stat.st_mtime),
                                    output_bytes=stat.st_size,
                                    output_tail=sanitize_model_text(tail.decode(
                                        "utf-8", "replace"))[-3000:])
                break
    return {"job_id": job_id, "stage": request.get("stage"), "status": state,
            "resources": request.get("resources"), "wait_reason": control.get("wait_reason") if state == "queued" else None,
            "window_seconds": control.get("window_seconds", request.get("requested_seconds")),
            "deadline_epoch": None if state == "queued" else control.get("deadline_epoch"),
            "cancel_requested": control.get("cancelled") is True,
            "worker": identity.get("status"), "progress": progress, "result": result,
            "job_ref": f"native_jobs/{job_id}/request.json"}


def adjust(output: Path, job_id: str, *, seconds_from_now: float,
           reason: str) -> dict[str, Any]:
    if (not isinstance(seconds_from_now, (int, float)) or
            isinstance(seconds_from_now, bool) or not math.isfinite(seconds_from_now) or seconds_from_now <= 0):
        raise ValueError("new local window must be positive")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError("budget adjustment needs a concise reason")
    current = status(output, job_id)
    if current["status"] not in {"running", "queued"} or current["cancel_requested"]:
        raise ValueError("only a running uncancelled job can be adjusted")
    budget = RunBudget.existing(output)
    if budget is None or seconds_from_now > budget.remaining():
        raise ValueError("requested local window exceeds hard run budget")
    directory = _path(output, job_id)
    path = directory / "control.json"
    with (directory / "control.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        control = read_json(path)
        if control.get("cancelled") is True:
            raise ValueError("job was cancelled before adjustment")
        if control.get("gpu_lease_id"):
            from .task_budget import TaskGPUBudget
            TaskGPUBudget(output).extend(control["gpu_lease_id"], seconds_from_now)
        control.update(revision=int(control.get("revision", 0)) + 1,
                       deadline_epoch=time.time() + seconds_from_now,
                       window_seconds=seconds_from_now,
                       adjustment_reason=reason.strip(), updated_at=now())
        atomic_json(path, control)
    return status(output, job_id)


def cancel(output: Path, job_id: str, *, reason: str) -> dict[str, Any]:
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError("cancellation needs a concise reason")
    current = status(output, job_id)
    if current["status"] not in {"running", "queued", "outcome_unknown"}:
        return current
    directory = _path(output, job_id)
    path = directory / "control.json"
    with (directory / "control.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        control = read_json(path)
        control.update(revision=int(control.get("revision", 0)) + 1,
                       cancelled=True, cancel_reason=reason.strip(), updated_at=now())
        atomic_json(path, control)
    return status(output, job_id)


def _queue_turn(output: Path, request: dict) -> bool:
    """Per-run priority with aging, separate CPU and GPU queues to permit backfill."""
    waiting = []
    for identity in _active(output):
        path = _path(output, identity)
        row = read_json(path / "request.json")
        control = read_json(path / "control.json") if (path / "control.json").is_file() else {}
        if control.get("state") != "queued" or control.get("cancelled"):
            continue
        profile = row.get("resources") or {}
        if profile.get("gpu") != (request.get("resources") or {}).get("gpu"):
            continue
        score = profile.get("priority", 0) + (time.time()-row["submitted_epoch"])/300
        waiting.append((-score, row["submitted_epoch"], row["job_id"]))
    return not waiting or min(waiting)[2] == request["job_id"]


def _admit(output: Path, directory: Path, request: dict):
    from .scheduling import HostLease, note, policy
    from .compute_decision import decide
    from .devices import NoCompatibleDevice
    budget = RunBudget.existing(output)
    started = time.monotonic()
    lease = HostLease(request["job_id"], request["resources"], limits=policy(output))
    while budget and budget.remaining() > 0:
        control = read_json(directory / "control.json")
        if control.get("cancelled"):
            raise InterruptedError("queued native job cancelled before execution")
        ready = _queue_turn(output, request)
        wait_reason = "priority_queue" if not ready else "host_capacity"
        if ready and request["resources"]["gpu"]:
            try:
                decision = decide(require_gpu=True)
                ready = decision.on_gpu
            except NoCompatibleDevice as exc:
                if not exc.evidence.get("busy_devices") and "busy" not in str(exc):
                    raise  # Missing hardware/visibility is not an indefinitely busy queue.
                ready = False
                wait_reason = "gpu_busy"
        if ready and lease.acquire():
            with (directory / "control.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                control = read_json(directory / "control.json")
                if control.get("cancelled"):
                    lease.release()
                    raise InterruptedError("queued native job cancelled before execution")
                control.update(state="running", started_epoch=time.time(),
                    deadline_epoch=time.time()+min(control.get("window_seconds", request["requested_seconds"]), budget.remaining()))
                atomic_json(directory / "control.json", control)
            note(output, kind="resource_wait", identity=request["job_id"],
                 seconds=time.monotonic()-started, status="admitted")
            return lease
        with (directory / "control.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            control = read_json(directory / "control.json")
            control.update(wait_reason=wait_reason, queue_checked_at=now())
            atomic_json(directory / "control.json", control)
        time.sleep(min(policy(output).get("resource_poll_seconds", 2), budget.remaining()))
    raise TimeoutError("hard run deadline reached while waiting for resources")


def _worker(output: Path, job_id: str) -> int:
    directory = _path(output, job_id)
    request = read_json(directory / "request.json")
    host_lease = None
    original_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set()
    try:
        if request.get("resources"):
            host_lease = _admit(output, directory, request)
            if host_lease.cores and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, set(host_lease.cores))
            count = str(request["resources"]["cpu"])
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[key] = count
        from .prepare import Preparation
        preparation = Preparation(
            repo=Path(request["repo"]), output=output, client=None,
            scouting=Path(output) / "scouting", run_id=request["run_id"],
            base_settings=request["settings"])
        if preparation.interpreter is None or request["stage"] not in preparation.stages:
            raise ValueError("verified interpreter or native stage was not recoverable")
        research, _ = preparation._research_controller()
        research.controller_decision_id = job_id
        research.job_control_path = directory / "control.json"
        research.job_resources = request.get("resources")
        if request.get("source_identity") and source_identity(
                output, Path(request["repo"])) != request["source_identity"]:
            raise ValueError("source changed between job submission and native start")
        while True:
            result = research.run_stage(
                request["stage"], settings=request["settings"],
                timeout=max(0.1, preparation.budget.remaining()
                            if preparation.budget else request["requested_seconds"]))
            evidence = (result.get("resource_block") or {}).get("evidence") or {}
            transient = (result.get("termination_reason") == "gpu_lease_busy" or
                result.get("termination_reason") == "gpu_unavailable" and evidence.get("busy_devices"))
            if not request.get("resources") or result.get("ran") is not False or not transient:
                break
            # Strong evidence says no native process launched. Queue rather than fail the
            # experiment when a neighboring worker won the device after preflight.
            if host_lease:
                host_lease.release()
                if hasattr(os, "sched_setaffinity"):
                    os.sched_setaffinity(0, original_affinity)
            with (directory / "control.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                control = read_json(directory / "control.json")
                control.update(state="queued", wait_reason="gpu_lease_race",
                               prestart_attempt_id=result.get("attempt_id"))
                atomic_json(directory / "control.json", control)
            time.sleep(min(2, preparation.budget.remaining()))
            host_lease = _admit(output, directory, request)
            if host_lease.cores and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, set(host_lease.cores))
        after_source = source_identity(output, Path(request["repo"]))
        if request.get("source_identity") and after_source != request["source_identity"]:
            result = {**result, "status": "failed", "why":
                      "pre-existing source or derived command changed during native job"}
        state = ("completed" if result.get("status") == "completed" else
                 "cancelled" if result.get("termination_reason") == "cancelled" else
                 "failed")
        atomic_json(directory / "result.json", {
            "job_id": job_id, "status": state, "stage_result": result,
            "source_identity_after": after_source,
            "finished_at": now()})
        exit_code = 0 if state == "completed" else 1
    except Exception as exc:  # noqa: BLE001 - worker must persist a terminal failure.
        failure = {
            "job_id": job_id, "status": "cancelled" if isinstance(exc, InterruptedError) else "failed", "error":
            f"{type(exc).__name__}: {exc}"[:800], "finished_at": now()}
        atomic_json(directory / "result.json", failure)
        # Setup/admission failures happen before a native process receipt exists. Seal
        # their complete traceback under the stable job ID rather than losing it in a
        # short worker status. Scheduler and read-only specialists can use read_evidence.
        from .evidence_store import capture_attempt_evidence
        log = directory / "failure.log"
        atomic_text(log, traceback.format_exc())
        sealed = capture_attempt_evidence(output, attempt_id=job_id, log=log,
            receipt_ref=f"native_jobs/{job_id}/result.json", status=failure["status"],
            returncode=None, termination_reason="worker_precondition_failure")
        atomic_json(directory / "result.json", {**failure,
            "evidence_id": sealed["evidence_id"], "evidence_ref": sealed["evidence_ref"]})
        exit_code = 1
    finally:
        if host_lease is not None:
            host_lease.release()
    marker_path = Path(output) / "native_jobs" / "controller_finished.json"
    marker = read_json(marker_path) if marker_path.is_file() else {}
    if (isinstance(marker, dict) and marker.get("state") == "finished" and not _active(output)
            and not (Path(output) / "task_gpu_budget.json").is_file()):
        try:
            from .repository_budget import RepositoryBudget
            project_root = Path(__file__).resolve().parents[3]
            RepositoryBudget(project_root / "autoresearch_runs" / "resource_budgets",
                             repo=Path(request["repo"])).finish(output)
        except (OSError, ValueError):
            # The terminal result is preserved; budget reconciliation remains conservative.
            pass
    return exit_code


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    return _worker(args.output, args.job_id)


if __name__ == "__main__":
    raise SystemExit(_main())
