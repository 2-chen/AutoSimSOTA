"""Seal controller/command-construction failures without pretending a process ran."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from .common import atomic_json, atomic_text, digest, now, sanitize_model_payload
from .evidence_store import capture_attempt_evidence


def seal(output: Path, *, operation: str, exc: Exception,
         context: dict | None = None, attempt_id: str = '',
         native_operation_status: str = 'not_inferred') -> dict:
    identity = attempt_id or uuid.uuid4().hex
    if not re.fullmatch(r'[0-9a-f]{32}', identity):
        raise ValueError('unsafe operation failure identity')
    directory = output / 'agent' / 'operation_failures'
    inputs = output / 'agent' / 'operation_inputs'
    if any(p.is_symlink() for p in (output, output / 'agent', directory, inputs, output / 'evidence')):
        raise ValueError('operation failure evidence directory is unsafe')
    message = sanitize_model_payload({'message': f'{type(exc).__name__}: {exc}'})['message']
    record = sanitize_model_payload({'schema_version': 1, 'operation': operation,
        'at': now(), 'exception': type(exc).__name__, 'message': message,
        'context': context or {}, 'native_operation_status': native_operation_status})
    log = directory / f'{identity}.log'
    if log.exists() or log.is_symlink():
        raise ValueError('operation failure evidence already exists')
    # Model-visible evidence is a presentation, not executable replay input.
    # Retain bounded original context privately, never as a model-readable evidence log.
    original = {'schema_version': 1, 'operation': operation, 'context': context or {}, 'at': record['at']}
    try:
        encoded = json.dumps(original, ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError):
        encoded = None
    private = inputs / (identity + '.json')
    if encoded is not None and len(encoded) <= 256 * 1024:
        if private.exists() or private.is_symlink():
            raise ValueError('operation failure input already exists')
        atomic_json(private, original)
        private_fields = {'_local_input_ref': private.relative_to(output).as_posix(),
                          '_local_input_sha256': digest(private)}
    else:
        private_fields = {'_local_input_status': 'context_not_serializable_or_exceeds_256_KiB_no_replay_authorization'}
    ref = f'agent/operation_failures/{identity}.json'
    atomic_text(log, json.dumps(record, ensure_ascii=False, sort_keys=True))
    sealed = capture_attempt_evidence(output, attempt_id=identity, log=log,
        receipt_ref=ref, status='operation_contract_failed', returncode=None,
        termination_reason='operation_contract_failed')
    atomic_json(output / ref, {**record, **sealed, **private_fields})
    return {**sealed, 'receipt_ref':ref, 'failure_domain': 'operation_contract',
            'native_operation_status': native_operation_status}
