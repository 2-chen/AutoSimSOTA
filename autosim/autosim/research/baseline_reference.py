"""Published-reference provenance and performance are distinct from a local baseline."""
import math
import re
from pathlib import Path
from urllib.parse import urlparse

from .common import atomic_json, digest, now, object_digest, read_json


def _sealed_unstarted_reference(output: Path, measurement: dict) -> bool:
    """Legacy pre-launch failures may predate the released-reference origin stamp.

    A registry hash alone is not evidence that no learning ran. Require the original
    host-sealed protocol-guard receipt and its unchanged log, with the same label.
    This classifies history only; it never supplies or certifies a score.
    """
    if (measurement.get('ok') is not False or measurement.get('ran') is not False
            or measurement.get('native_operation_status') != 'not_started'
            or measurement.get('failure_domain') != 'operation_contract'
            or measurement.get('where') != 'comparison_protocol'
            or any(measurement.get(k) is not None for k in
                   ('train', 'evaluate', 'metric_value', 'success_rate', 'samples'))):
        return False
    identity = measurement.get('evidence_id')
    if not isinstance(identity, str) or not re.fullmatch(r'[0-9a-f]{32}', identity):
        return False
    receipt_ref = f'agent/operation_failures/{identity}.json'
    log_ref = f'agent/operation_failures/{identity}.log'
    if measurement.get('receipt_ref') != receipt_ref or measurement.get('log_ref') != log_ref:
        return False
    for ref in (receipt_ref, log_ref):
        path = output / ref
        if (not path.is_file() or path.stat().st_size > 2 * 1024**2
                or not path.resolve().is_relative_to(output.resolve())
                or any(p.is_symlink() for p in (path, *path.parents)
                       if p.is_relative_to(output))):
            return False
    receipt = read_json(output / receipt_ref)
    log = read_json(output / log_ref)
    return (receipt.get('operation') == log.get('operation') == 'measurement_protocol_guard'
            and receipt.get('native_operation_status') == log.get('native_operation_status') == 'not_started'
            and receipt.get('evidence_id') == identity
            and receipt.get('log_ref') == log_ref
            and receipt.get('log_sha256') == measurement.get('log_sha256') == digest(output / log_ref)
            and receipt.get('context') == log.get('context')
            and (receipt.get('context') or {}).get('label') == measurement.get('label'))


def registered_reference_measurement(output: Path, run_root: Path, path: Path) -> bool:
    """Identify a hash-registered evaluation-only reference, never certify its score.

    Reference measurements are independent of the local baseline/candidate controller.
    A label or self-declared origin alone must not bypass interrupted-study protection.
    Both unsuccessful original and recovered reference records remain immutable history.
    """
    try:
        output, run_root, path = Path(output), Path(run_root), Path(path)
        if (path.parent != run_root / 'measurements' or path.name == 'baseline.json'
                or not path.is_file() or path.stat().st_size > 64 * 1024**2
                or not path.resolve().is_relative_to(output.resolve())
                or any(p.is_symlink() for p in (path, *path.parents)
                       if p.is_relative_to(output))):
            return False
        measurement = read_json(path)
        if not isinstance(measurement, dict):
            return False
        origin = measurement.get('baseline_origin')
        if origin != 'released_reference' and not (
                origin is None and _sealed_unstarted_reference(output, measurement)):
            return False
        train = measurement.get('train') or {}
        if (not isinstance(train, dict) or any(train.get(k) is not None
                for k in ('attempt_id', 'evidence_id', 'artifact', 'returncode'))
                or train.get('reused') is True):
            return False
        ref = measurement.get('reference_protocol_ref')
        match = re.fullmatch(r'reference_protocols/([A-Za-z0-9_-]{1,64})\.json', str(ref))
        if not match:
            return False
        identity = match[1]
        protocol = run_root / ref
        registry = output / 'baseline_references' / (identity + '.json')
        for record, maximum in ((protocol, 2 * 1024**2), (registry, 2 * 1024**2)):
            if (not record.is_file() or record.stat().st_size > maximum
                    or not record.resolve().is_relative_to(output.resolve())
                    or any(p.is_symlink() for p in (record, *record.parents)
                           if p.is_relative_to(output))):
                return False
        row = read_json(registry)
        if (row.get('id') != identity or
                (row.get('reference') or {}).get('kind') != 'released_checkpoint'
                or not all((row.get('review') or {}).get(k) is True for k in
                           ('sources_supported', 'protocol_matches',
                            'tolerance_supported', 'release_identity_supported'))):
            return False
        history = row.get('evaluations') or []
        if not isinstance(history, list) or len(history) > 4096:
            return False
        history = [*history, row.get('evaluation')]
        relative = path.relative_to(output).as_posix()
        # Registry complete()/view() use canonical object identity, not file-format bytes.
        sha256 = object_digest(measurement)
        return any(isinstance(item, dict) and item.get('measurement_ref') == relative
                   and item.get('measurement_sha256') == sha256 for item in history)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


class SourceCitationError(ValueError):
    """A source transport/input failure, not a negative scientific review."""
    def __init__(self, code, *, index, ref, detail):
        self.details = {'code': code, 'citation_index': index + 1,
                        'source_ref': ref if isinstance(ref, str) else '', 'detail': detail}
        super().__init__(f'reference source citation {index + 1} [{code}]: {detail}')


def checkout_source(repo, ref, *, index=0):
    """Resolve file[:line[-line]] without weakening confinement or exact quoting."""
    def fail(code, detail):
        raise SourceCitationError(code, index=index, ref=ref, detail=detail)
    if not isinstance(ref, str) or not ref or len(ref) > 1024:
        fail('invalid_reference', 'use a checkout-relative file or an executor-issued source ID')
    if '[LOCAL_' in ref or '[OUTSIDE_' in ref:
        fail('display_placeholder', 'display placeholders are not executable references; '
             'capture checkout_source_request={ref: actual_relative_file[:line[-line]]} and cite its returned ID')
    match = re.fullmatch(r'(.+):([1-9][0-9]{0,8})(?:-([1-9][0-9]{0,8}))?', ref)
    name = match[1] if match else ref
    if ':' in name or '\\' in name:
        fail('invalid_reference', 'accepted syntax is relative/file[:positive_line[-end_line]]')
    rel = Path(name)
    root = Path(repo).resolve()
    path = root / rel
    if (rel.is_absolute() or '..' in rel.parts or
            any(p.startswith('.env') or p in {'.git', '.aws', '.ssh'} for p in rel.parts) or
            path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'} or
            any(x in path.name.lower() for x in ('credential', 'secret', 'id_rsa', 'id_ed25519')) or
            any(p.is_symlink() for p in [path, *path.parents] if p != root and p.is_relative_to(root)) or
            not path.resolve().is_relative_to(root)):
        fail('unsafe_source', 'source must be a non-secret, non-symlink file confined to this checkout')
    if not path.is_file():
        fail('source_missing', 'the cited checkout file does not exist; inspect its actual relative name')
    if path.stat().st_size > 2 * 1024**2:
        fail('source_too_large', 'source exceeds the 2 MiB file bound')
    before = digest(path)
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeError:
        fail('source_not_text', 'cite UTF-8 source/configuration, not a binary artifact')
    if before != digest(path):
        fail('source_changed', 'source changed while reading; recapture current bytes')
    selected, canonical = text, rel.as_posix()
    if match:
        start, end = int(match[2]), int(match[3] or match[2])
        lines = text.splitlines(keepends=True)
        if end < start or end > len(lines):
            fail('invalid_line_range', 'line range is reversed or outside the actual source file')
        selected = ''.join(lines[start-1:end])
        canonical += f':{start}' + (f'-{end}' if end != start else '')
    return path, selected, canonical, before


def source_evidence(repo: Path, refs: list, *, output: Path | None = None) -> list[dict]:
    if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
        raise SourceCitationError('invalid_sources', index=0, ref=None,
            detail='reference requires 1..12 exact source citations')
    repo = Path(repo).resolve()
    result = []
    for index, row in enumerate(refs):
        if not isinstance(row, dict):
            raise SourceCitationError('invalid_source_item', index=index, ref=None,
                detail='source citations must be objects with ref and quote')
        if isinstance(row.get('ref'),str) and row['ref'].startswith(('native-source:', 'checkout-source:')):
            if output is None:
                raise ValueError('installed source citations require a run-owned native source context')
            if row['ref'].startswith('native-source:'):
                from .native_source_citations import resolve
            else:
                from .checkout_source_citations import resolve
            try:
                result.append(resolve(output,repo,row['ref'],row.get('quote')))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise SourceCitationError('sealed_source_invalid', index=index, ref=row['ref'],
                    detail=str(exc)) from exc
            continue
        path, text, canonical, sha = checkout_source(repo, row.get('ref'), index=index)
        quote = row.get('quote')
        if not isinstance(quote, str) or not quote or len(quote) > 4000 or quote not in text:
            raise SourceCitationError('quote_not_exact', index=index, ref=row.get('ref'),
                detail='quote must occur verbatim in the cited file/line range; capture the exact source rather than guess')
        result.append({'ref': canonical, 'quote': quote, 'sha256': sha})
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


def freeze_reference(source: Path, destination: Path, *, max_bytes: int) -> dict:
    """Reconcile an interrupted reference copy without overwriting any bytes.

    This is scoped to published-reference registration; general experiment archive
    destinations still reject collisions. Reuse requires exact source identity,
    independent files (no hardlink to mutable source), and confined nonlink paths.
    """
    import fcntl
    from .experiment_bundle import artifact_identity, freeze_artifact
    source, destination = Path(source), Path(destination)
    if any(p.is_symlink() for p in [destination, *destination.parents]):
        raise ValueError('unsafe reference archive path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent/'copy.lock'
    if lock.is_symlink():
        raise ValueError('unsafe reference copy lock')
    with lock.open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not destination.exists():
            return freeze_artifact(source, destination, max_bytes=max_bytes)
        field, identity = artifact_identity(source)
        if artifact_identity(destination) != (field, identity):
            raise ValueError('existing reference archive conflicts with current source; not overwritten')
        files = sorted(p for p in source.rglob('*') if p.is_file()) if source.is_dir() else [source]
        total = sum(p.stat().st_size for p in files)
        if total > max_bytes:
            raise ValueError('reference archive exceeds copy limit')
        for path in files:
            archived = destination/path.relative_to(source) if source.is_dir() else destination
            if path.samefile(archived):
                raise ValueError('reference archive shares mutable source inode')
        return {'path':str(destination), 'source':str(source), 'bytes':total,
                'files':len(files), 'sha256':object_digest([
                    (str(p.relative_to(source)) if source.is_dir() else source.name, digest(p))
                    for p in files]),
                **({'content_sha256':identity} if field == 'content_sha256' else {}),
                'reconciled_existing_copy':True}


def resolve_review_citations(original: list, shown: list, citations: list) -> list:
    """Resolve only exact model-visible citations; never undo redaction heuristically."""
    from .source_citations import resolve
    return resolve(original, shown, citations)


def validate(repo: Path, reference: dict, *, output: Path | None = None) -> dict:
    if not isinstance(reference, dict) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', str(reference.get('id') or '')):
        raise ValueError('reference requires a short path-safe id')
    kind = reference.get('kind')
    if kind not in {'released_checkpoint', 'unavailable'}:
        raise ValueError('reference kind must be released_checkpoint or unavailable')
    refs = source_evidence(repo, reference.get('source_refs'), output=output)
    reason = reference.get('reason')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError('reference requires a nonempty reason of at most 1000 characters; '
                         'put source quotations in source_refs, not reason')
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
    from .policy_consumption import verified_episode_count
    episode_count=verified_episode_count(measurement)
    valid = (bound and measurement.get('ok') is True and isinstance(value, (int, float))
             and not isinstance(value, bool) and math.isfinite(value)
             and episode_count == row['reference']['episodes']
             and all((measurement.get(k) or {}).get('status') == 'verified'
                     for k in ['policy_consumption', 'rollout_evidence', 'metric_lineage']))
    if valid:
        from .receipt_verifier import verify_measurement
        valid = verify_measurement(measured.parent.parent, measured.stem).get('status') == 'consistent'
    matches = valid and abs(value-row['reference']['expected_metric']) <= row['reference']['tolerance']
    row.update(status='reproduced' if matches else 'performance_mismatch' if valid else 'evaluation_failed',
               evaluation={'metric_value': value, 'samples': episode_count,
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
