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


def data_binding(output: Path, request: dict) -> dict:
    """Grant one owned collection/processing artifact, never a host path or run root."""
    from .evidence_store import read_attempt_evidence
    output = Path(output).resolve(strict=True)
    if (not isinstance(request, dict) or set(request) - {'evidence_id','artifact_index','relative_path','scope','dataset_root_relative'}
            or not {'evidence_id','artifact_index'} <= set(request)
            or type(request['artifact_index']) is not int or not 0 <= request['artifact_index'] < 4096):
        raise ValueError('data producer grant needs a sealed evidence id and artifact index')
    run_id = read_json(_safe(output,'run_state.json')).get('run_id')
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',run_id):
        raise ValueError('data diagnostic run owner is invalid')
    # A Data worker may inspect collection products before the first baseline,
    # when the research directory does not exist yet. Resolve from the existing
    # run owner, not a future directory; still reject finalized development.
    confirmation = _safe(output, Path('research')/run_id/'confirmation.json')
    attempts = _safe(output, Path('research')/run_id/'confirmation_attempts')
    if confirmation.exists() or any(attempts.glob('*.json')):
        raise ValueError('new data diagnostics cannot reopen finalized development')
    owner = output
    scope = request.get('scope','this_run')
    if scope == 'budget_parent':
        from .run_successor import validate_inheritance
        validate_inheritance(output)
        owner = Path(read_json(_safe(output,'budget_inheritance.json'))['parent']).resolve(strict=True)
    elif scope != 'this_run':
        raise ValueError('data grant scope is not an authorized study ancestor')
    verified = read_attempt_evidence(owner,request['evidence_id'],limit=1)
    context = verified.get('native_context') or {}
    if (verified['status'] not in {'completed','failed','timed_out','cancelled','interrupted'}
            or context.get('stage') not in {'collect','prepare_data'}):
        raise ValueError('only sealed terminal collect/prepare_data artifacts may be inspected')
    receipt = _safe(owner, verified['receipt_ref'])
    row = read_json(receipt)
    if (row.get('attempt_id') != request['evidence_id'] or row.get('stage') != context['stage']
            or row.get('artifact') != verified['artifact'] or row.get('status') != verified['status']
            or any(row.get(k)!=context.get(k) for k in ('working_directory','output_directory'))):
        raise ValueError('data producer ownership/context changed')
    artifact = verified['artifact']
    candidates = artifact.get('candidate_paths') or artifact.get('examples') or []
    if not isinstance(candidates,list) or request['artifact_index'] >= len(candidates):
        raise ValueError('data artifact index is not offered by the sealed producer')
    raw = Path(candidates[request['artifact_index']])
    base = Path(context['working_directory'] if artifact.get('found_beside_the_command') else context['output_directory'])
    if not base.is_absolute() or not base.resolve().is_relative_to(owner):
        raise ValueError('data artifact owner is outside its study')
    rel = Path(request.get('relative_path',''))
    if raw.is_absolute():
        if not raw.is_relative_to(owner):
            raise ValueError('data artifact leaves its owner')
        data = raw / rel
    else:
        data = base / raw / rel
    if ('..' in raw.parts or rel.is_absolute() or '..' in rel.parts or
            data in (owner, base) or not data.is_relative_to(owner)):
        raise ValueError('data grant cannot expose the source/run root or an escaping path')
    data = _safe(owner,data.relative_to(owner))
    marker=None
    if 'dataset_root_relative' in request:
        root_ref=request['dataset_root_relative']
        if (not isinstance(root_ref,str) or not 1<=len(root_ref)<=512 or
                Path(root_ref).is_absolute() or '..' in Path(root_ref).parts or
                Path(root_ref) in (Path('.'),Path('')) or rel!=Path('.') or
                artifact.get('found_beside_the_command') or not data.is_file()):
            raise ValueError('marker dataset root must be an explicit confined producer-output directory; no relative_path combination')
        directory=_safe(owner,(base/Path(root_ref)).relative_to(owner))
        if (directory in (owner,base) or not directory.is_dir() or
                not directory.is_relative_to(base) or not data.is_relative_to(directory)):
            raise ValueError('marker dataset root must contain the selected marker inside this producer output')
        marker={'ref':data.relative_to(owner).as_posix(),'sha256':digest(data)}
        data=directory
    if not data.is_dir():
        raise ValueError('format data grant requires a dataset directory')
    files = list(data.rglob('*'))
    if (len(files)>10000 or any(p.is_symlink() for p in files)
            or sum(p.stat().st_size for p in files if p.is_file())>64*1024**3):
        raise ValueError('data diagnostic inventory exceeds the bounded dataset contract')
    field,identity=artifact_identity(data)
    return {'request':dict(request),'path':str(data),'byte_identity_field':field,'byte_identity':identity,
            **({'boundary_marker':marker} if marker else {}),
            'producer_receipt_sha256':digest(receipt),'environment_variable':'AUTOSIM_DIAGNOSTIC_DATA',
            'read_only':True,'authority':'one source-owned data artifact; not training-domain, expert or score certification'}


def data_binding_unchanged(output, binding):
    try:
        current=data_binding(output,binding['request'])
        return all(current.get(k)==binding.get(k) for k in
                   ('byte_identity_field','byte_identity','producer_receipt_sha256'))
    except (OSError,ValueError,TypeError,KeyError,RuntimeError):
        return False
