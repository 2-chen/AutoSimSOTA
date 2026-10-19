"""Bounded, immutable observations of an owned live producer, never work certification.

An Agent chooses literal text to inspect; no benchmark-specific counter is parsed.
The original active log is not promoted to terminal evidence or modified.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .common import atomic_json, now, sanitize_model_text
from .evidence_store import capture_attempt_evidence
from .experiment_transaction import safe


_ID = re.compile(r'[0-9a-f]{32}\Z')
_RUN = re.compile(r'[A-Za-z0-9_.-]{1,80}\Z')
_PRODUCERS = {'train', 'collect', 'prepare_data'}


def _record(path):
    with path.open('rb') as source:
        data = source.read(131073)
    if len(data) > 131072:
        raise ValueError('progress ownership record exceeds size bound')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('progress ownership record must be an object')
    return value, hashlib.sha256(data).hexdigest()


def _affinity(receipt: dict, *, run_id: str, attempt_id: str) -> dict:
    """Observe the exact live native launcher, not the controller's own CPUs."""
    unknown = {'status': 'unknown', 'authority': 'no verified live native affinity observation'}
    proof = receipt.get('process_identity')
    if (not isinstance(proof, dict) or proof.get('run_id') != run_id or
            proof.get('attempt_id') != attempt_id or type(proof.get('pid')) is not int or
            proof['pid'] <= 0):
        return unknown
    from .process_executor import inspect_process_identity
    try:
        if inspect_process_identity(proof).get('status') != 'matching_running':
            return unknown
        cores = sorted(os.sched_getaffinity(proof['pid']))
        if not cores or inspect_process_identity(proof).get('status') != 'matching_running':
            return unknown
        return {'status': 'observed', 'logical_cpu_count': len(cores), 'cpu_ids': cores,
                'authority': 'native launcher affinity during observation; not CPU quota or throughput'}
    except (OSError, ValueError, AttributeError):
        return unknown


def capture(output: Path, job_id: str, *, query: str = '', tail_bytes: int = 131072,
            limit: int = 8000) -> dict:
    """Snapshot an owned producer tail, with an optional Agent-selected literal filter.

    Evaluation/confirmation logs are intentionally excluded. Returned observations are
    not dataset profiles, completed work audits, policy selections, metrics or scores.
    """
    if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
        raise ValueError('unsafe job id')
    if not isinstance(query, str) or len(query.encode('utf-8')) > 512:
        raise ValueError('query must be at most 512 UTF-8 bytes of literal text')
    if type(tail_bytes) is not int or not 1 <= tail_bytes <= 262144:
        raise ValueError('tail_bytes must be 1..262144')
    if type(limit) is not int or not 1 <= limit <= 12000:
        raise ValueError('limit must be 1..12000 bytes')
    root = Path(output).resolve(strict=True)
    request_ref = f'native_jobs/{job_id}/request.json'
    request, request_sha = _record(safe(root, request_ref))
    stage, run_id = request.get('stage'), request.get('run_id')
    if (request.get('job_id') != job_id or stage not in _PRODUCERS or
            not isinstance(run_id, str) or not _RUN.fullmatch(run_id) or
            run_id in {'.', '..'}):
        raise ValueError('progress snapshots admit owned train/collect/prepare_data jobs only')
    from .native_jobs import status
    view = status(root, job_id)
    if (view.get('job_id') != job_id or view.get('stage') != stage or
            view.get('status') != 'running' or view.get('worker') != 'matching_running'):
        raise ValueError('live producer ownership is not verified; inspect terminal or unresolved evidence instead')
    attempt = (view.get('progress') or {}).get('attempt_id')
    if not isinstance(attempt, str) or not _ID.fullmatch(attempt):
        raise ValueError('job has no owned native attempt')
    receipt_ref = f'research/{run_id}/attempts/{attempt}/receipt.json'
    receipt, receipt_sha = _record(safe(root, receipt_ref))
    if (receipt.get('attempt_id') != attempt or receipt.get('run_id') != run_id or
            receipt.get('controller_decision_id') != job_id or
            receipt.get('node_id') != stage or receipt.get('status') != 'running'):
        raise ValueError('native progress receipt does not identify this running producer')
    log_ref = f'research/{run_id}/{stage}/attempts/{attempt}/output.log'
    log_path = safe(root, log_ref)
    observed_started = now()
    with log_path.open('rb') as source:
        before = os.fstat(source.fileno())
        start = max(0, before.st_size - tail_bytes)
        source.seek(start - 1 if start else 0)
        clipped_first_line = start > 0 and source.read(1) != b'\n'
        source.seek(start)
        data = source.read(before.st_size - start)
        after = os.fstat(source.fileno())
    if ((before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or
            len(data) != before.st_size - start or after.st_size < before.st_size):
        raise ValueError('native log changed identity or was truncated during observation')
    current_log = safe(root, log_ref).stat()
    if (current_log.st_dev, current_log.st_ino) != (before.st_dev, before.st_ino):
        raise ValueError('native log was replaced during observation')
    # Appended bytes are not part of this observation. Revalidate ownership, not a
    # terminal return code; a producer can finish between these two reads.
    current, _ = _record(safe(root, receipt_ref))
    if any(current.get(k) != receipt.get(k) for k in
           ('attempt_id', 'run_id', 'controller_decision_id', 'node_id')):
        raise ValueError('native ownership changed during observation')
    if _record(safe(root, request_ref))[1] != request_sha:
        raise ValueError('job request changed during observation')
    text = data.decode('utf-8', 'replace')
    if clipped_first_line:
        # Do not offer a clipped first line as a complete progress statement.
        text = text.split('\n', 1)[1] if '\n' in text else ''
    if not data.endswith(b'\n'):
        text = text.rsplit('\n', 1)[0] if '\n' in text else ''
    lines = text.splitlines()
    selected = [line for line in lines if query in line] if query else lines
    encoded = '\n'.join(selected).encode('utf-8')
    projection = encoded[-limit:].decode('utf-8', 'replace')
    identity = uuid.uuid4().hex
    # Validate the shared evidence destination before writing any observation.
    safe(root, f'evidence/{identity}.json')
    base = f'native_job_progress_snapshots/{identity}/'
    snapshot_log = safe(root, base + 'snapshot.log')
    snapshot_log.parent.mkdir(parents=True, exist_ok=False)
    # A unique private directory and exclusive file creation prevent overwrite.
    # Seal the exact bytes, including partial UTF-8; decode only for display.
    with snapshot_log.open('xb') as saved:
        saved.write(data)
        saved.flush()
        os.fsync(saved.fileno())
    body = {'schema_version': 1, 'observation_id': identity, 'job_id': job_id,
        'native_attempt_id': attempt, 'stage': stage, 'status': 'observed',
        'native_operation_status': 'running_at_observation_start',
        'observed_started_at': observed_started, 'observed_finished_at': now(),
        'request_ref': request_ref, 'request_sha256': request_sha,
        'native_receipt_ref': receipt_ref, 'native_receipt_sha256_at_start': receipt_sha,
        'native_status_at_end': current.get('status'), 'source_log_ref': log_ref,
        'source_bytes_observed': before.st_size, 'source_byte_range': [start, start + len(data)],
        'source_slice_sha256': hashlib.sha256(data).hexdigest(),
        'transport_encoding': 'exact raw snapshot bytes; display uses UTF-8 with replacement',
        'query': query, 'matching_lines_in_bounded_slice': len(selected),
        'projection_truncated': len(encoded) > limit,
        'requested_resources': request.get('resources'),
        'native_cpu_affinity': _affinity(receipt, run_id=run_id, attempt_id=attempt),
        'authority': 'bounded live log observation only; not terminal work, dataset quality, metric or score'}
    receipt_path = safe(root, base + 'receipt.json')
    atomic_json(receipt_path, body)
    evidence = capture_attempt_evidence(root, attempt_id=identity, log=snapshot_log,
        receipt_ref=base+'receipt.json', status='observed', returncode=None,
        termination_reason='live_progress_snapshot')
    return {**body, **evidence, 'text': sanitize_model_text(projection),
            'next_action': 'read_evidence on observation_id for this immutable snapshot; '
                           'native producer completion still requires its original terminal receipt'}
