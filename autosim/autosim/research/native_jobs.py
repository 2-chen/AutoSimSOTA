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
import stat
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
        if (directory / "result.json").is_file() and not (directory / "result.json").is_symlink():
            try:
                result = read_json(directory / "result.json")
                if isinstance(result, dict) and result.get("job_id") == directory.name and result.get("status") in {"completed", "failed", "cancelled"}:
                    continue
            except (OSError, ValueError, TypeError):
                pass  # An unreadable result does not release an unknown side effect.
        if (directory / "request.json").is_file():
            # A dead worker without a result is an unresolved side effect, not a free slot.
            active.append(directory.name)
    return active


def active_jobs(output: Path) -> list[str]:
    return _active(output)


def _checkout_source(output: Path, repo: Path) -> dict:
    """The original source identity payload, independent of stage adapter updates."""
    manifest_path = Path(output) / "workspace_snapshot.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('checkout source manifest is missing')
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
    from .source_tracking import materialized_files
    original = {str(row[1]) for row in entries}
    materialized = [(str(name), digest(Path(repo) / name))
                    for name in materialized_files(Path(repo)) if str(name) not in original]
    return {"base": manifest["source_tree_fingerprint"], "existing_files": current,
            "materialized_source": materialized}


def checkout_identity(output: Path, repo: Path) -> str:
    """Exact code/binding context for a no-op edit; not a dependency independence proof."""
    source = _checkout_source(output, repo)
    manifest = read_json(Path(output) / 'workspace_snapshot.json')
    environment = Path(output) / 'environment.json'
    if environment.is_symlink():
        raise ValueError('unsafe edit environment identity')
    return object_digest({'source': source,
        'modes': [(name, stat.S_IMODE((Path(repo) / name).lstat().st_mode))
                  for name in sorted({str(row[1]) for row in manifest['source_entries']}
                                     | {name for name, _ in source['materialized_source']})],
        'bindings': manifest.get('resource_bindings', manifest.get('resources', [])),
        'environment': digest(environment) if environment.is_file() else None})


def source_identity(output: Path, repo: Path, *, stage_scope: list[str] | None = None) -> str:
    """Bind pre-existing checkout files without treating new training outputs as source."""
    manifest_path = Path(output) / 'workspace_snapshot.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return ''
    source = _checkout_source(output, repo)
    stages_path = Path(output) / 'derived_stages.json'
    if stages_path.is_symlink():
        raise ValueError('unsafe derived stage identity')
    if stage_scope is None:
        stages = digest(stages_path) if stages_path.is_file() else None
    else:
        if (not isinstance(stage_scope, list) or not 1 <= len(stage_scope) <= 64
                or any(not isinstance(s, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', s)
                       for s in stage_scope) or stage_scope != sorted(set(stage_scope))):
            raise ValueError('invalid native job stage scope')
        document = read_json(stages_path)
        if not isinstance(document, dict) or any(s not in document for s in stage_scope):
            raise ValueError('native job stage scope disappeared')
        stages = {'scope_version': 1, 'stages': {s: document[s] for s in stage_scope}}
    return object_digest({**source,
                          # Early source investigation precedes command derivation.
                          # Absence is part of identity, not a fabricated verified stage.
                          "derived_stages": stages})


def request_source_identity(output: Path, repo: Path, request: dict) -> str:
    """Old jobs retain the exact global contract; never silently migrate history.

    New jobs bind the executed builder only. All inventoried/materialized checkout
    source remains bound: this does not pretend to know Python's dynamic imports.
    Upstream data identities, argv, settings and policy receipts are checked separately.
    """
    if 'source_stage_scope' not in request:
        return source_identity(output, repo)
    scope = request['source_stage_scope']
    if scope != [request.get('stage')]:
        raise ValueError('native job stage identity does not match its operation')
    return source_identity(output, repo, stage_scope=scope)


def source_matches(output: Path, repo: Path, request: dict) -> bool:
    if not request.get('source_identity'):
        return False
    if request_source_identity(output, repo, request) == request['source_identity']:
        return True
    # Only explicit inspection can create an additive bridge while the ORIGINAL
    # full identity still matches. Never reconstruct old dependencies after drift.
    if 'source_stage_scope' in request:
        return False
    try:
        from .experiment_transaction import safe
        job = str(request.get('job_id') or '')
        _path(output, job)
        base = 'native_jobs/'+job+'/'
        path = safe(output, base+'source_binding.json')
        if not path.is_file() or path.stat().st_size>65536:
            return False
        bridge = read_json(path)
        if (bridge.get('schema_version') != 1 or bridge.get('job_id') != job
                or bridge.get('sha256') != object_digest({k:v for k,v in bridge.items() if k!='sha256'})
                or bridge.get('original_source_identity') != request['source_identity']
                or bridge.get('stage') != request.get('stage')
                or bridge.get('request_sha256') != digest(safe(output,base+'request.json'))
                or read_json(safe(output,base+'request.json')) != request
                or bridge.get('result_sha256') != digest(safe(output,base+'result.json'))
                or bridge.get('receipt_sha256') != digest(safe(output,bridge['receipt_ref']))):
            return False
        return bridge.get('scoped_source_identity') == source_identity(output,repo,stage_scope=[request['stage']])
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        return False


def bind_completed_source(output: Path, repo: Path, request: dict) -> dict:
    """Additive legacy handoff proof at an explicit inspect, before ANY drift.

    Raw request/result/native receipt are never edited. A missing original source
    stamp, changed checkout, unresolved job or unsealed native exit cannot migrate.
    """
    from .experiment_transaction import safe
    from .evidence_store import read_attempt_evidence
    if 'source_stage_scope' in request:
        return {'status':'already_scoped'}
    job = str(request.get('job_id') or '')
    _path(output,job)
    base = 'native_jobs/'+job+'/'
    target = safe(output,base+'source_binding.json')
    if target.exists():
        if not source_matches(output,repo,request):
            raise ValueError('legacy source bridge no longer verifies')
        return {'status':'previously_bound','ref':base+'source_binding.json'}
    if (request.get('repo') != str(Path(repo).resolve(strict=True))
            or not request.get('source_identity')
            or source_identity(output,repo) != request['source_identity']):
        raise ValueError('original global source identity is not intact; cannot bridge')
    request_path = safe(output,base+'request.json')
    result_path = safe(output,base+'result.json')
    result = read_json(result_path)
    native = result.get('stage_result') or {}
    attempt = str(native.get('attempt_id') or '')
    if (read_json(request_path) != request or result.get('job_id') != job
            or result.get('status') != 'completed'
            or result.get('source_identity_after') != request['source_identity']
            or not _JOB_ID.fullmatch(attempt) or native.get('stage') != request.get('stage')):
        raise ValueError('legacy job lacks stable completed source proof')
    run_id = request.get('run_id')
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',run_id) or run_id in {'.','..'}:
        raise ValueError('invalid source bridge run')
    relative = f'research/{run_id}/attempts/{attempt}/receipt.json'
    receipt_path = safe(output,relative)
    receipt = read_json(receipt_path)
    proof = read_attempt_evidence(output,attempt,limit=1)
    if (receipt.get('attempt_id') != attempt or receipt.get('run_id') != run_id
            or receipt.get('controller_decision_id') != job
            or receipt.get('node_id') != request.get('stage')
            or receipt.get('status') != 'completed' or type(receipt.get('returncode')) is not int
            or receipt['returncode'] != 0 or proof.get('status') != 'completed'
            or proof.get('receipt_ref') != relative or proof.get('returncode') != 0):
        raise ValueError('legacy source bridge native receipt is not sealed and completed')
    body = {'schema_version':1,'job_id':job,'stage':request['stage'],
        'original_source_identity':request['source_identity'],
        'request_sha256':digest(request_path),'result_sha256':digest(result_path),
        'receipt_ref':relative,'receipt_sha256':digest(receipt_path),
        'scoped_source_identity':source_identity(output,repo,stage_scope=[request['stage']]),
        'authority':'additive dependency proof at explicit inspect; no score, work or protocol certification'}
    if source_identity(output,repo) != request['source_identity']:
        raise ValueError('source drifted while bridging the original job identity')
    atomic_json(target,{**body,'sha256':object_digest(body)})
    return {'status':'bound','ref':base+'source_binding.json'}


def checkpoint_overrides(output: Path, stage: str, selection: dict | None) -> dict:
    """An explicit run-owned choice, revalidated before every native replay."""
    if selection is None:
        return {}
    if stage != 'evaluate' or not isinstance(selection,dict):
        raise ValueError('checkpoint selection requires an evaluate job')
    from .experiment_transaction import safe
    from .checkpoint_handoff import validate
    identity = selection.get('id')
    if (not isinstance(identity,str) or not re.fullmatch(r'[0-9a-f]{64}',identity) or
            selection.get('evidence_ref') != f'verification_checkpoint_handoffs/{identity}.json'):
        raise ValueError('checkpoint selection lacks its exact registered proof')
    path = safe(output, selection['evidence_ref'])
    if not path.is_file() or path.stat().st_size > 1048576:
        raise ValueError('checkpoint selection proof unavailable')
    expected = {k:v for k,v in selection.items() if k!='evidence_ref'}
    if (read_json(path) != expected or not selection.get('files') or
            identity != object_digest({'producer':selection.get('producer_evidence_ref'),
                                       'files':selection['files']})):
        raise ValueError('checkpoint selection proof changed')
    validate(output,selection)
    return {'checkpoint':selection['selected_path']}


def completed_evaluation(output: Path, *, repo: Path, run_id: str,
                         settings: dict, checkpoint_selection: dict | None,
                         source_stamp: str) -> str | None:
    """Reuse an exact completed detached diagnostic, never training/collection work."""
    from .evidence_store import read_attempt_evidence
    if checkpoint_selection is None:
        return None
    for path in sorted((output/'native_jobs').glob('*/request.json'))[-128:]:
        try:
            if path.is_symlink() or path.parent.is_symlink() or not _JOB_ID.fullmatch(path.parent.name):
                continue
            request = read_json(path)
            if (request.get('job_id') != path.parent.name or request.get('stage') != 'evaluate' or
                    request.get('run_id') != run_id or request.get('repo') != str(repo) or
                    request.get('source_identity') != source_stamp or
                    object_digest(request.get('settings')) != object_digest(settings) or
                    (request.get('checkpoint_selection') or {}).get('id') != checkpoint_selection['id']):
                continue
            result_path = path.with_name('result.json')
            if result_path.is_symlink():
                continue
            result = read_json(result_path)
            native = result.get('stage_result') or {}
            attempt = str(native.get('attempt_id') or '')
            if (result.get('job_id') != request['job_id'] or result.get('status') != 'completed' or
                    result.get('source_identity_after') != source_stamp or
                    not _JOB_ID.fullmatch(attempt) or native.get('returncode') != 0):
                continue
            proof = read_attempt_evidence(output, attempt, limit=1)
            if (proof.get('status') == 'completed' and proof.get('returncode') == 0 and
                    proof.get('receipt_ref') == f'research/{run_id}/attempts/{attempt}/receipt.json'):
                return request['job_id']
        except (OSError,ValueError,TypeError,KeyError,AttributeError):
            continue
    return None


def training_overrides(output: Path, repo: Path, request: dict) -> dict:
    """Revalidate detached data bindings and preserve labels at the real caller.

    Old requests without registered selections keep their original behavior. No
    byte identities or counters are supplied to the native argv/reader.
    """
    if request['stage'] != 'train':
        return {}
    bindings = request.get('data_bindings') or (request.get('candidate_binding') or {}).get('data_versions') or []
    if not bindings:
        return {}
    from .data_versions import PREFIX, resolve, training_metadata
    settings = request['settings']
    labels = training_metadata(bindings, settings)  # Validate shape before indexing.
    handles = {**settings, **{row['setting']: PREFIX + row['id'] for row in labels}}
    resolved, current = resolve(output, repo, handles)
    if resolved != settings or current != bindings:
        raise ValueError('detached training data bindings changed before native start')
    return {'selected_training_data': training_metadata(current, resolved)}


def submit(output: Path, *, repo: Path, stage: str, settings: dict[str, Any],
           requested_seconds: float, reason: str, run_id: str = "derived",
           resources: dict | None = None, candidate_binding: dict | None = None,
           checkpoint_selection: dict | None = None,
           collection_control_contract: dict | None = None,
           repeat_job_id: str = '') -> dict[str, Any]:
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
    remaining_wall = max(0,budget.remaining()) if budget is not None else 0
    if budget is None or requested_seconds > remaining_wall:
        raise ValueError('job window exceeds the existing hard run budget: '
            f'requested_seconds={requested_seconds:.3f}; remaining_wall_seconds={remaining_wall:.3f}. '
            'GPU availability is separate. Main/Resource must choose a fitting local window '
            'and allow for evaluation/recovery; no window is silently shortened or budget extended.')
    # Only commands already accepted by the derivation can be submitted.
    verified = read_json(output / "derived_stages.json")
    if not isinstance(verified, dict) or stage not in verified or not verified[stage].get("source"):
        raise ValueError("native stage has no verified command")
    if stage == 'collect':
        from .data_acquisition import verify_control_contract
        verify_control_contract(output=output, repo=repo, source=verified[stage]['source'],
                                settings=settings, contract=collection_control_contract)
        from .training_work import missing_formal_collection_work
        problem = missing_formal_collection_work(verified[stage]['source'], settings)
        if problem:
            raise ValueError(problem)
    data_bindings = []
    if stage == 'train':
        from .data_versions import resolve
        settings, data_bindings = resolve(output, repo, settings)
        if candidate_binding:
            data_bindings = candidate_binding.get('data_versions') or data_bindings
        training_overrides(output, repo, {'stage':stage, 'settings':settings,
                                         'data_bindings':data_bindings})
    checkpoint_overrides(output,stage,checkpoint_selection)
    stage_scope = [stage]
    source_stamp = source_identity(output, repo, stage_scope=stage_scope)
    profile = resource_profile(resources, gpu=settings.get("device") != "cpu") if scheduling else None
    if profile and profile["gpu"] is False and str(settings.get("device", "auto")).startswith("cuda"):
        raise ValueError("CPU resource profile conflicts with CUDA settings")
    if profile and profile["gpu"] and settings.get("device") == "cpu":
        raise ValueError("GPU resource profile conflicts with CPU settings")
    if (candidate_binding is not None and profile and profile['gpu'] is False
            and settings.get('device') != 'cpu'):
        raise ValueError('candidate CPU profile must be declared in audited settings')
    root = output / "native_jobs"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "submission.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if repeat_job_id and (stage != 'evaluate' or not _JOB_ID.fullmatch(repeat_job_id)):
            raise ValueError('repeat_job_id is an explicit completed-evaluation replicate only')
        cached = completed_evaluation(output, repo=repo, run_id=run_id, settings=settings,
            checkpoint_selection=checkpoint_selection, source_stamp=source_stamp) if stage == 'evaluate' else None
        if repeat_job_id and cached != repeat_job_id:
            raise ValueError('replicate must identify the exact completed evaluation with unchanged inputs')
        if cached and not repeat_job_id:
            return {**status(output,cached), 'reused_completed_diagnostic':True,
                    'reuse_scope':'same policy/source/settings; no new GPU work and not formal score adoption'}
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
               "source_stage_scope": stage_scope,
               "requested_seconds": float(requested_seconds),
               "reason": reason.strip(), "submitted_at": now(),
               "submitted_epoch": started}
    if candidate_binding is not None:
        request['candidate_binding'] = candidate_binding
    if data_bindings:
        request['data_bindings'] = data_bindings
    if repeat_job_id:
        request['replicate_of'] = repeat_job_id
    if checkpoint_selection is not None:
        request['checkpoint_selection'] = checkpoint_selection
    if collection_control_contract is not None:
        request['collection_control_contract'] = collection_control_contract
    if scheduling:
        request["resources"] = profile
        if request["resources"]["gpu"] is False:
            request["settings"] = {**settings, "device": "cpu"}
    try:
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
    except (OSError, ValueError) as exc:
        _seal_terminal(output, directory, {"job_id": job_id, "status": "failed",
            "launched": False, "error": sanitize_model_text(str(exc)),
            "termination_reason": "worker_launch_failed", "finished_at": now()},
            traceback.format_exc())
        raise
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
    try:
        request = read_json(directory / "request.json")
        control = read_json(directory / "control.json")
        if not isinstance(request, dict) or not isinstance(control, dict):
            raise ValueError("job records must be objects")
    except (OSError, ValueError, TypeError) as exc:
        return {"job_id": job_id, "status": "unreadable", "stage": None,
            "worker": "unverifiable", "progress": {}, "result": None,
            "ownership_retained": True, "reconciliation_required": True,
            "error": sanitize_model_text(str(exc))[:500],
            "job_ref": f"native_jobs/{job_id}/request.json"}
    result_path = directory / "result.json"
    try:
        result = read_json(result_path) if result_path.is_file() else None
        if result is not None and (not isinstance(result, dict) or result.get("job_id") != job_id
                or result.get("status") not in {"completed", "failed", "cancelled"}):
            raise ValueError("job result identity/status is invalid")
    except (OSError, ValueError, TypeError) as exc:
        return {"job_id": job_id, "status": "unreadable", "stage": request.get("stage"),
            "worker": "unverifiable", "progress": {}, "result": None,
            "ownership_retained": True, "reconciliation_required": True,
            "error": sanitize_model_text(str(exc))[:500],
            "job_ref": f"native_jobs/{job_id}/result.json"}
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
            "candidate_binding": request.get("candidate_binding"),
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
    if current["status"] == "unreadable":
        return reconcile(output, job_id, reason=reason)
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
    if current["status"] == "outcome_unknown":
        reconcile(output, job_id, reason=reason)
    return status(output, job_id)


def _seal_terminal(output: Path, directory: Path, result: dict, detail: str) -> None:
    from .evidence_store import capture_attempt_evidence
    log = directory / "reconciliation.log"
    atomic_text(log, sanitize_model_text(detail))
    sealed = capture_attempt_evidence(output, attempt_id=result["job_id"], log=log,
        receipt_ref=f"native_jobs/{result['job_id']}/result.json", status=result["status"],
        returncode=None, termination_reason=result["termination_reason"])
    atomic_json(directory / "result.json", {**result, **sealed})


def reconcile(output: Path, job_id: str, *, reason: str) -> dict:
    """Close dead ownership only after worker AND native children are proven absent.

    Unknown metrics are never adopted. Missing/partial process identities remain blocked,
    not permission to signal a bare PID or free a possibly executing source namespace.
    """
    from .scheduling import locked
    directory = _path(output, job_id)
    with locked(directory / "reconciliation.lock"):
        current = status(output, job_id)
        if current["status"] in {"completed", "cancelled", "failed"}:
            return current
        identity_path = directory / "worker_identity.json"
        if not identity_path.is_file() or identity_path.is_symlink():
            raise ValueError("worker identity absent; ownership cannot safely be released")
        observed = inspect_process_identity(read_json(identity_path))
        if observed.get("status") != "not_running":
            raise ValueError("worker or process-group outcome is not proven absent")
        request = read_json(directory / "request.json")
        run_id = request.get("run_id", "derived")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", run_id) or run_id in {".", ".."}:
            raise ValueError("job run identity is unreadable")
        children = []
        attempts = Path(output) / "research" / run_id / "attempts"
        for index, path in enumerate(attempts.glob("*/receipt.json")):
            if index >= 10000 or path.is_symlink() or not path.resolve().is_relative_to(Path(output).resolve()):
                raise ValueError("native receipt inventory cannot safely be reconciled")
            receipt = read_json(path)
            if receipt.get("controller_decision_id") != job_id:
                continue
            child = receipt.get("process_identity")
            if child:
                observation = inspect_process_identity(child)
                if observation.get("status") != "not_running":
                    raise ValueError("native child still runs or has unverifiable identity")
            elif receipt.get("status") not in {"completed", "failed", "blocked", "cancelled"}:
                raise ValueError("native child has unknown launch side effects")
            children.append(receipt.get("attempt_id"))
        result = {"job_id": job_id, "status": "cancelled", "finished_at": now(),
            "termination_reason": "dead_owner_reconciled", "reason": reason[:1000],
            "measurement_adopted": False, "outcome_scope": "unknown_output_discarded",
            "reconciled_child_attempt_ids": children}
        _seal_terminal(output, directory, result, json.dumps({**result, "worker": observed}))
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
        # A detached worker reconstructs Preparation without calling _run_cycle,
        # where its wall budget is normally initialized. The requested job window
        # is a mutable soft limit, never a replacement for the persisted hard cap.
        if preparation.budget is None:
            preparation.budget = RunBudget.existing(output)
        if preparation.budget is None:
            raise ValueError("detached native job requires an existing hard run budget")
        if preparation.interpreter is None or request["stage"] not in preparation.stages:
            raise ValueError("verified interpreter or native stage was not recoverable")
        if request["stage"] == "collect":
            from .data_acquisition import verify_control_contract
            verify_control_contract(output=output, repo=preparation.repo,
                source=preparation.stages["collect"], settings=request["settings"],
                contract=request.get("collection_control_contract"))
        research, _ = preparation._research_controller()
        research.controller_decision_id = job_id
        research.job_request_context = {'request_ref':str((directory/'request.json').relative_to(output)),
                                        'request_sha256':digest(directory/'request.json')}
        research.job_control_path = directory / "control.json"
        research.job_resources = request.get("resources")
        if request.get("source_identity") and request_source_identity(
                output, Path(request["repo"]), request) != request['source_identity']:
            raise ValueError("source changed between job submission and native start")
        while True:
            overrides = checkpoint_overrides(output,request['stage'],request.get('checkpoint_selection'))
            overrides.update(training_overrides(output, Path(request['repo']), request))
            result = research.run_stage(
                request["stage"], settings=request["settings"],
                timeout=max(0.1, preparation.budget.remaining()),
                **overrides)
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
        after_source = request_source_identity(output, Path(request["repo"]), request)
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
