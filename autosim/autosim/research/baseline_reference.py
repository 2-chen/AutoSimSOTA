"""Published-reference provenance and performance are distinct from a local baseline."""
import math
import re
from pathlib import Path
from urllib.parse import urlparse

from .common import atomic_json, digest, now, object_digest, read_json


def source_evidence(repo: Path, refs: list) -> list[dict]:
    if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
        raise ValueError('reference requires 1..12 exact source citations')
    repo = Path(repo).resolve()
    result = []
    for row in refs:
        if not isinstance(row, dict):
            raise ValueError('source citations must be objects')
        rel = Path(row.get('ref') or '')
        path = repo/rel
        quote = row.get('quote')
        if (rel.is_absolute() or '..' in rel.parts or not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(repo) or path.stat().st_size > 2*1024**2
                or any(p.startswith('.env') or p in {'.git', '.aws', '.ssh'} for p in rel.parts)
                or path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}
                or any(x in path.name.lower() for x in ('credential', 'secret', 'id_rsa', 'id_ed25519'))
                or not isinstance(quote, str) or not quote or len(quote) > 4000
                or quote not in path.read_text(encoding='utf-8')):
            raise ValueError('reference source citation is missing, unsafe or not exact')
        result.append({'ref': rel.as_posix(), 'quote': quote, 'sha256': digest(path)})
    return result


def checkpoint_location(output: Path, repo: Path, ref: str) -> tuple[Path,Path]:
    """Resolve explicit run/checkout-relative refs; ambiguous or escaping refs fail closed.

    The two paths are the readable backing bytes and the controlled native mount view.
    Never search by basename or infer a host resource directory.
    """
    output,repo=Path(output).resolve(),Path(repo).resolve()
    if not isinstance(ref,str):
        raise ValueError('checkpoint_ref must be a relative string')
    rel = Path(ref)
    if not ref or rel.is_absolute() or '..' in rel.parts:
        raise ValueError('checkpoint_ref must be a confined run-relative path')
    from .workspace_resources import bindings_for
    bindings=bindings_for(repo)
    views=[output/rel]
    if repo.is_relative_to(output) and repo!=output:
        views.append(repo/rel)
    found=[]
    for native in views:
        path,boundary=native,output
        for binding in bindings:
            target=repo/binding['target']
            if native.is_relative_to(target):
                path=Path(binding['source'])/native.relative_to(target)
                boundary=Path(binding['source']).resolve()
                break
        if path.is_symlink():
            raise ValueError('checkpoint/resource is a symbolic link')
        if not path.exists():
            continue
        source=path.resolve(strict=True)
        if not source.is_relative_to(boundary):
            raise ValueError('checkpoint escaped its authorized boundary')
        found.append((source,native))
    if len(found)>1:
        raise ValueError('ambiguous checkpoint_ref; use an explicit run-relative checkout/... or artifact path')
    if not found:
        raise ValueError('checkpoint_ref does not resolve in the run or its authorized checkout; '
                         'inspect the bound resource and use an explicit relative reference')
    return found[0]


def resolve_checkpoint(output: Path, repo: Path, ref: str) -> Path:
    return checkpoint_location(output,repo,ref)[0]


def validate(repo: Path, reference: dict) -> dict:
    if not isinstance(reference, dict) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', str(reference.get('id') or '')):
        raise ValueError('reference requires a short path-safe id')
    kind = reference.get('kind')
    if kind not in {'released_checkpoint', 'unavailable'}:
        raise ValueError('reference kind must be released_checkpoint or unavailable')
    refs = source_evidence(repo, reference.get('source_refs'))
    reason = reference.get('reason')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError('reference requires a concise reason')
    if kind == 'released_checkpoint':
        episodes = reference.get('episodes')
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
            raise ValueError('reference requires the published positive episode count')
        url = urlparse(str(reference.get('release_url') or ''))
        if url.scheme != 'https' or not url.netloc or url.username or url.password:
            raise ValueError('release_url must be a public HTTPS provenance URL')
        if not isinstance(reference.get('revision'), str) or not reference['revision'].strip():
            raise ValueError('pin the published checkpoint revision')
        if not isinstance(reference.get('checkpoint_ref'), str) or not reference['checkpoint_ref']:
            raise ValueError('released checkpoint requires an explicit checkpoint_ref')
        value, tolerance = reference.get('expected_metric'), reference.get('tolerance')
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError('published metric and source-justified tolerance must be finite numbers')
    return {**reference, 'source_refs': refs}


def handoff(output: Path, reference_id: str) -> dict:
    held = view(output)
    if held.get('id') != reference_id or held.get('kind') != 'released_checkpoint':
        raise ValueError('choose the active reviewed released reference')
    path = output/'baseline_references'/(reference_id+'.json')
    row = read_json(path)
    frozen = row.get('frozen_checkpoint') or {}
    from .experiment_bundle import artifact_identity
    base = Path(frozen.get('path') or '')
    if base.is_symlink() or not base.resolve(strict=True).is_relative_to(output.resolve()):
        raise ValueError('reference checkpoint escaped its frozen run directory')
    field, identity = artifact_identity(base)
    if frozen.get(field) != identity:
        raise ValueError('frozen reference checkpoint bytes changed')
    from .checkpoint_handoff import file_digest
    files = [base] if base.is_file() else sorted(base.rglob('*'))
    manifest = []
    for file in files:
        if file.is_symlink() or not file.resolve().is_relative_to(base if base.is_dir() else base.parent):
            raise ValueError('reference bundle contains unsafe links')
        if file.is_file():
            manifest.append({'ref': file.relative_to(output).as_posix(),
                             'sha256': file_digest(file), 'bytes': file.stat().st_size})
        if len(manifest) > 256:
            raise ValueError('select the native policy bundle, not a whole resource tree')
    return {'selected_path': str(base), 'files': manifest,
            'evidence_ref': str(path.relative_to(output)), 'origin': 'reviewed_released_checkpoint',
            'authority': 'frozen published-reference claim; not locally trained or performance reproduced'}


def register(output: Path, reference: dict, review: dict, *, frozen_checkpoint: dict | None = None) -> dict:
    root = output/'baseline_references'
    if root.is_symlink():
        raise ValueError('unsafe reference registry')
    path = root/(reference['id']+'.json')
    if path.exists() or path.is_symlink():
        raise ValueError('reference id already recorded; preserve it and choose a new id')
    required = ['sources_supported', 'protocol_matches', 'tolerance_supported']
    required += ['release_identity_supported'] if reference['kind']=='released_checkpoint' else ['unavailability_supported']
    accepted = isinstance(review, dict) and all(review.get(k) is True for k in required)
    if accepted and reference['kind'] == 'released_checkpoint' and not frozen_checkpoint:
        raise ValueError('released reference requires frozen checkpoint bytes')
    row = {'id': reference['id'], 'at': now(), 'reference': reference, 'review': review,
           'frozen_checkpoint': frozen_checkpoint or {},
           'status': ('ready' if reference['kind']=='released_checkpoint' else 'unavailable') if accepted else 'rejected',
           'authority': 'reviewed provenance claim, not a measured performance reproduction'}
    atomic_json(path, row)
    if accepted:
        atomic_json(output/'baseline_reference.json', {'id': row['id'], 'record_sha256': digest(path)})
    return row


def view(output: Path) -> dict:
    pointer = output/'baseline_reference.json'
    if not pointer.is_file():
        return {'status': 'not_declared', 'meaning': 'official baseline has not been reproduced'}
    try:
        if pointer.is_symlink() or (output/'baseline_references').is_symlink():
            raise ValueError('linked reference pointer')
        held = read_json(pointer)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', str(held.get('id') or '')):
            raise ValueError('invalid reference id')
        path = output/'baseline_references'/(held['id']+'.json')
        if path.is_symlink() or digest(path) != held.get('record_sha256'):
            raise ValueError('reference record changed')
        row = read_json(path)
        result = row.get('evaluation') or {}
        spec = row['reference']
        if result:
            rel = Path(result['measurement_ref'])
            measured = output/rel
            if (rel.is_absolute() or '..' in rel.parts or measured.is_symlink()
                    or not measured.resolve().is_relative_to(output.resolve())
                    or object_digest(read_json(measured)) != result['measurement_sha256']):
                raise ValueError('reference measurement changed')
        return {'id': row['id'], 'status': row['status'],
                'kind': spec['kind'], 'release_url': spec.get('release_url'), 'revision': spec.get('revision'),
                'expected_metric': spec.get('expected_metric'), 'tolerance': spec.get('tolerance'),
                'measured_metric': result.get('metric_value'), 'samples': result.get('samples'),
                'measurement_ref': result.get('measurement_ref'),
                'reason': spec.get('reason'), 'record_ref': str(path.relative_to(output)),
                'meaning': 'local baseline.ok means valid measurement, not official reproduction'}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {'status': 'unverified', 'meaning': 'reference evidence needs repair'}


def complete(output: Path, reference_id: str, measurement: dict, measurement_ref: str) -> dict:
    held = view(output)
    if held.get('id') != reference_id or held['status'] != 'evaluating':
        raise ValueError('reference evaluation ownership changed')
    path = output/'baseline_references'/(reference_id+'.json')
    row = read_json(path)
    if measurement_ref != row.get('pending_measurement_ref'):
        raise ValueError('measurement is not owned by the pending reference evaluation')
    rel = Path(measurement_ref)
    measured = output/rel
    if (rel.is_absolute() or '..' in rel.parts or measured.is_symlink()
            or not measured.resolve().is_relative_to(output.resolve())
            or read_json(measured) != measurement):
        raise ValueError('reference needs its exact sealed native measurement')
    frozen = row['frozen_checkpoint']
    policy = measurement.get('policy_artifact') or {}
    from .experiment_bundle import artifact_identity
    reference_path = Path(frozen['path'])
    field, identity = artifact_identity(reference_path)
    loaded = Path(policy.get('path') or '')
    bound = (Path(policy.get('source') or '') == reference_path
             and frozen.get(field) == identity and not loaded.is_symlink()
             and loaded.resolve().is_relative_to(output.resolve())
             and artifact_identity(loaded) == (field, identity)) if policy.get('path') else False
    value = measurement.get('metric_value')
    valid = (bound and measurement.get('ok') is True and isinstance(value, (int, float))
             and not isinstance(value, bool) and math.isfinite(value)
             and (measurement.get('metric_reading') or {}).get('samples') == row['reference']['episodes']
             and all((measurement.get(k) or {}).get('status') == 'verified'
                     for k in ['policy_consumption', 'rollout_evidence', 'metric_lineage']))
    if valid:
        from .receipt_verifier import verify_measurement
        valid = verify_measurement(measured.parent.parent, measured.stem).get('status') == 'consistent'
    matches = valid and abs(value-row['reference']['expected_metric']) <= row['reference']['tolerance']
    row.update(status='reproduced' if matches else 'performance_mismatch' if valid else 'evaluation_failed',
               evaluation={'metric_value': value, 'samples': (measurement.get('metric_reading')or{}).get('samples'),
                           'measurement_ref': measurement_ref, 'measurement_sha256': object_digest(measurement),
                           'valid': valid, 'at': now()})
    row.setdefault('evaluations', []).append(dict(row['evaluation']))
    atomic_json(path, row)
    atomic_json(output/'baseline_reference.json', {'id': reference_id, 'record_sha256': digest(path)})
    return view(output)


def begin(output: Path, reference_id: str, *, measurement_ref: str, retry_reason: str = '') -> dict:
    if not isinstance(retry_reason, str) or len(retry_reason) > 1000:
        raise ValueError('retry reason must be concise')
    rel = Path(measurement_ref)
    if rel.is_absolute() or '..' in rel.parts or rel.suffix != '.json':
        raise ValueError('measurement_ref must be a confined JSON path')
    held = view(output)
    permitted = held.get('status') == 'ready' or (
        held.get('status') in {'performance_mismatch', 'evaluation_failed'} and retry_reason.strip())
    if held.get('id') != reference_id or not permitted:
        raise ValueError('reference is not ready; unknown outcome cannot be silently retried')
    path = output/'baseline_references'/(reference_id+'.json')
    row = read_json(path)
    row.update(status='evaluating', pending_measurement_ref=measurement_ref, retry_reason=retry_reason)
    atomic_json(path, row)
    atomic_json(output/'baseline_reference.json', {'id': reference_id, 'record_sha256': digest(path)})
    return row
