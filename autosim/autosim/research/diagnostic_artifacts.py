"""Explicit, verified policy grants for CPU diagnostics; never mount the run root."""
import re
import os
from pathlib import Path

from .common import digest, read_json
from .experiment_bundle import artifact_identity
from .frozen_policy_trial import _safe


def _development_open(root: Path) -> bool:
    confirmation = _safe(root, 'confirmation.json')
    attempts = _safe(root, 'confirmation_attempts')
    return not confirmation.exists() and not any(attempts.glob('*.json'))


def _bounded_identity(policy: Path):
    entries, size = 0, 0
    if policy.is_dir():
        for parent, directories, files in os.walk(policy, followlinks=False):
            for name in (*directories, *files):
                path = Path(parent)/name
                entries += 1
                if entries > 4096 or path.is_symlink():
                    raise ValueError('diagnostic policy inventory is unsafe or too large')
                if path.is_file():
                    size += path.stat().st_size
                    if size > 4*1024**3:
                        raise ValueError('diagnostic policy exceeds the CPU inspection size limit')
    elif policy.is_file() and policy.stat().st_size > 4*1024**3:
        raise ValueError('diagnostic policy exceeds the CPU inspection size limit')
    return artifact_identity(policy)


def policy_binding(output: Path, request: dict) -> dict:
    from .receipt_verifier import verify_measurement
    output = Path(output).resolve(strict=True)
    if (not isinstance(request, dict) or set(request) != {'label', 'measurement_sha256'}
            or not isinstance(request['label'], str)
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', request['label'])
            or not isinstance(request['measurement_sha256'], str)
            or not re.fullmatch(r'[0-9a-f]{64}', request['measurement_sha256'])):
        raise ValueError('policy_request needs an exact development label and measurement_sha256')
    state_path = _safe(output, 'run_state.json')
    if not state_path.is_file() or state_path.stat().st_size > 8*1024**2:
        raise ValueError('diagnostic policy owner is unavailable')
    state = read_json(state_path)
    run_id = state.get('run_id') if isinstance(state, dict) else None
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
        raise ValueError('diagnostic policy owner is invalid')
    root = _safe(output, Path('research')/run_id)
    measurement = _safe(root, Path('measurements')/(request['label']+'.json'))
    if (not measurement.is_file() or measurement.stat().st_size > 2*1024**2
            or digest(measurement) != request['measurement_sha256']):
        raise ValueError('diagnostic policy measurement changed or unavailable')
    row = read_json(measurement)
    if (not isinstance(row, dict) or row.get('label') != request['label'] or row.get('ok') is not True
            or row.get('confirmation') or row.get('baseline_origin') != 'locally_trained'
            or not isinstance(row.get('guardrails') or {}, dict)
            or (row.get('guardrails') or {}).get('status') in {'unknown', 'violated'}):
        raise ValueError('only a scored local development policy can be granted')
    if not _development_open(root):
        raise ValueError('development policy diagnostics cannot resume after held-out exposure')
    artifact = row.get('policy_artifact') or {}
    if not isinstance(artifact, dict):
        raise ValueError('diagnostic policy artifact is invalid')
    raw = Path(str(artifact.get('path') or ''))
    if not raw.is_absolute() or not raw.is_relative_to(root):
        raise ValueError('diagnostic policy archive is not run-owned')
    policy = _safe(root, raw.relative_to(root))
    field, identity = _bounded_identity(policy)
    if verify_measurement(root, request['label']).get('status') != 'consistent':
        raise ValueError('diagnostic policy measurement evidence is inconsistent')
    if artifact.get(field) != identity or digest(measurement) != request['measurement_sha256']:
        raise ValueError('diagnostic policy identity changed')
    return {'run_id': run_id, 'request': dict(request), 'path': str(policy),
            'byte_identity_field': field, 'byte_identity': identity,
            'policy_sha256': artifact.get('sha256'),
            'environment_variable': 'AUTOSIM_DIAGNOSTIC_POLICY', 'read_only': True,
            'authority': 'verified run-owned development archive; CPU diagnostic only, not a new score'}


def binding_unchanged(output: Path, binding: dict) -> bool:
    try:
        output = Path(output).resolve(strict=True)
        if read_json(_safe(output, 'run_state.json')).get('run_id') != binding['run_id']:
            return False
        root = _safe(output, Path('research')/binding['run_id'])
        if not _development_open(root):
            return False
        measurement = _safe(root, Path('measurements')/(binding['request']['label']+'.json'))
        if digest(measurement) != binding['request']['measurement_sha256']:
            return False
        raw = Path(read_json(measurement)['policy_artifact']['path'])
        policy = _safe(root, raw.relative_to(root))
        return (('path' not in binding or str(policy) == binding['path']) and
                _bounded_identity(policy) == (binding['byte_identity_field'], binding['byte_identity']))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError):
        return False
