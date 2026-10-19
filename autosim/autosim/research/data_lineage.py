"""Small, native-format-neutral data cards backed by sealed producer/probe logs.

This is not a dataset converter. Agents discover the native reader and produce a
bounded aggregate profile; raw bytes remain in place and processed versions name
their immutable parents. Claims of quality/coverage are never inferred from size.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .baseline_reference import checkpoint_location
from .common import digest, read_json, object_digest
from .evidence_store import read_attempt_evidence
from .experiment_bundle import artifact_identity

PROFILE_MARKER = 'AUTOSIM_DATA_PROFILE '
KINDS = {'existing', 'official', 'collected', 'processed', 'synthetic'}
ID = re.compile(r'[A-Za-z0-9_-]{1,64}\Z')
COUNTS = {'episodes', 'transitions', 'samples', 'successful_episodes',
          'failed_episodes', 'invalid_episodes'}


def _text(value, label, limit=500):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'{label} must be concise nonempty text')
    return value


def _counts(value, allowed=None):
    if not isinstance(value, dict) or len(value) > 32:
        raise ValueError('profile counts must be bounded objects')
    for key, count in value.items():
        _text(key, 'count name', 64)
        if allowed is not None and key not in allowed:
            raise ValueError('unknown count unit')
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 10**15:
            raise ValueError('profile counts must be nonnegative integers')
    return dict(value)


def _record(output, ref, *, max_bytes=128 * 1024):
    rel = Path(str(ref or ''))
    path = output / rel
    if (not ref or rel.is_absolute() or '..' in rel.parts or path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(output.resolve())
            or path.stat().st_size > max_bytes):
        raise ValueError('unsafe data evidence reference')
    return path, read_json(path)


def parents(output, repo, ids):
    if (not isinstance(ids, list) or len(ids) > 16 or len(set(map(str, ids))) != len(ids)
            or any(not isinstance(i, str) or not ID.fullmatch(i) for i in ids)):
        raise ValueError('parents must be up to 16 distinct registered data version ids')
    result = []
    for ident in ids:
        path, row = _record(output, f'data_versions/{ident}.json')
        if (row.get('id') != ident or row.get('status') != 'reviewed_not_consumed'
                or not all((row.get('review') or {}).get(k) is True for k in
                    ('allowed_training_data','loader_connection_supported','sources_supported'))):
            raise ValueError('parent is not an accepted registered data version')
        source, _ = checkpoint_location(output, repo, row['native_ref'])
        if str(source) != row['source'] or artifact_identity(source) != (row['identity_field'], row['identity']):
            raise ValueError('parent data bytes changed')
        result.append({key: row[key] for key in ('id', 'source', 'native_ref', 'identity_field', 'identity')})
        result[-1].update(record_ref=str(path.relative_to(output)), record_sha256=digest(path))
    return result


def _sealed_profile(output, evidence_id, field, identity, *, producer=False, native=None,
                    partial_profile=None):
    verified = read_attempt_evidence(output, evidence_id, limit=1)
    path, evidence = _record(output, f'evidence/{evidence_id}.json')
    partial = producer and partial_profile is not None and verified['status'] in {
        'failed', 'timed_out', 'cancelled', 'interrupted'}
    if not partial and (verified['status'] != 'completed' or verified['returncode'] != 0):
        raise ValueError('data profile needs a completed successful native attempt')
    if producer:
        receipt_path, receipt = _record(output, verified['receipt_ref'])
        context = verified.get('native_context') or {}
        if (receipt.get('stage') not in {'collect', 'prepare_data'}
                or receipt.get('attempt_id') != evidence_id or receipt.get('status') != verified['status']
                or receipt.get('artifact') != verified.get('artifact')
                or any(context.get(k) != receipt.get(k) for k in
                       ('stage','output_directory','working_directory')) or not context):
            raise ValueError('producer must be a sealed collect/prepare_data attempt')
        artifact = verified.get('artifact') or {}
        root = Path(context['working_directory'] if artifact.get('found_beside_the_command')
                    else context['output_directory'])
        examples = artifact.get('candidate_paths') or artifact.get('examples') or []
        paths = [Path(e) if Path(e).is_absolute() else root / e for e in examples if isinstance(e,str)]
        contained=(any('..' not in p.parts and p.resolve().is_relative_to(output.resolve())
                           and (native == p or native.is_relative_to(p)) for p in paths))
        if not contained and partial_profile is not None:
            # A native entrypoint may declare a marker file, not its full dataset.
            # Require an explicit marker-root grant plus the trusted full manifest;
            # never infer a parent from a filename or certify a dataset from a log.
            _,witness=partial_profile
            probe_path,probe=_record(output,witness['receipt_ref'],max_bytes=2*1024**2)
            binding=probe.get('diagnostic_data_binding') or {}
            request=binding.get('request') or {}
            if (digest(probe_path)==witness.get('receipt_sha256') and
                    probe.get('diagnostic_data_binding_stable') is True and
                    request.get('evidence_id')==evidence_id and request.get('dataset_root_relative')
                    and request.get('scope','this_run')=='this_run'):
                from .diagnostic_artifacts import data_binding
                current=data_binding(output,request)
                contained=(current.get('boundary_marker') is not None and
                    current.get('read_only') is True and Path(current['path'])==native
                    and current['producer_receipt_sha256']==digest(receipt_path)
                    and current.get('byte_identity_field')==field and current.get('byte_identity')==identity
                    and all(current.get(k)==binding.get(k) for k in
                        ('boundary_marker','byte_identity_field','byte_identity','producer_receipt_sha256')))
        if (not root.is_absolute() or not root.resolve().is_relative_to(output.resolve())
                or not isinstance(examples, list) or not examples
                or not contained):
            raise ValueError('dataset is not inside the recorded producer artifact')
    if producer and partial_profile is not None:
        # Native collectors need not print our profile marker. A separate full
        # inspector can bind committed files, without changing the original status.
        row, recovery_witness = partial_profile
        if row['scope'] != 'full' or row.get('funnel', {}).get('format_valid_episodes', 0) < 1:
            raise ValueError('partial producer recovery requires a full nonempty committed manifest')
        witness = {'evidence_id': evidence_id, 'evidence_ref': str(path.relative_to(output)),
            'record_sha256': digest(path), 'log_sha256': evidence['log_sha256'],
            'receipt_ref': verified['receipt_ref'], 'receipt_sha256': digest(receipt_path),
            'producer_status': verified['status'], 'producer_returncode': verified['returncode'],
            'partial_output': partial, 'native_manifest': True, 'recovery': recovery_witness}
        return row, witness
    # read_attempt_evidence already verifies the entire log, not just the excerpt.
    log = output / evidence['log_ref']
    rows = []
    for line in log.read_text(errors='replace').splitlines():
        if line.startswith(PROFILE_MARKER):
            if len(line.encode()) > 16000:
                raise ValueError('data profile event exceeds 16 KiB')
            try:
                row = json.loads(line[len(PROFILE_MARKER):])
            except RecursionError as exc:
                raise ValueError('data profile nesting exceeds bounds') from exc
            if not isinstance(row, dict):
                raise ValueError('data profile must be an object')
            if row.get('identity_field') == field and row.get('identity') == identity:
                rows.append(row)
    if len(rows) != 1:
        raise ValueError('expected exactly one sealed profile for these data bytes')
    witness = {'evidence_id': evidence_id, 'evidence_ref': str(path.relative_to(output)),
               'record_sha256': digest(path), 'log_sha256': evidence['log_sha256']}
    if producer:
        witness.update(receipt_ref=verified['receipt_ref'],receipt_sha256=digest(receipt_path))
    return rows[0], witness


def _collection_request(output, witness):
    """Requested work is not attempted work; expose it only when receipt-bound."""
    receipt_path, receipt = _record(output, witness['receipt_ref'])
    job = receipt.get('controller_decision_id')
    if not isinstance(job, str) or not re.fullmatch(r'[0-9a-f]{32}', job):
        return {}, {}
    ref = f'native_jobs/{job}/request.json'
    if not (output/ref).is_file():
        return {}, {}
    path, request = _record(output, ref, max_bytes=2 * 1024**2)
    context = receipt.get('job_request_context') or {}
    if (request.get('stage') != 'collect' or request.get('job_id') != job or
            not (context.get('request_ref') == ref and context.get('request_sha256') == digest(path)
                 or receipt.get('settings_digest') == object_digest(request.get('settings') or {}))):
        return {}, {}
    binding = {'request_ref':ref, 'request_sha256':digest(path)}
    return request, binding


def _manifest_profile(output, evidence_id, source, field, identity, *, repo=None):
    """Bind trusted read-only format/episode observations to exact dataset bytes."""
    from .data_quality import validate_manifest
    from .data_health import PROFILE_GENERATORS
    verified = read_attempt_evidence(output, evidence_id, limit=1)
    record, evidence = _record(output, f'evidence/{evidence_id}.json')
    receipt_path, receipt = _record(output, verified['receipt_ref'], max_bytes=2 * 1024**2)
    native_source = receipt.get('data_profile_generator') == 'autosim_native_source_manifest_v1'
    if (verified['status'] != 'completed' or verified['returncode'] != 0
            or receipt.get('data_profile_generator') not in PROFILE_GENERATORS
            or receipt.get('gpu_access') is not False or receipt.get('repository_source_stable') is not True
            or receipt.get('native_context_stable') is False
            or ('diagnostic_data_binding' in receipt and receipt.get('diagnostic_data_binding_stable') is not True)
            or receipt.get('data_profile_status') or not (
                receipt.get('data_profile_request', {}).get('include_manifest') is True or
                'selection' in receipt.get('data_profile_request', {}) or
                'coverage_reference' in receipt.get('data_profile_request', {}) or native_source)):
        raise ValueError('episode manifest requires a completed trusted read-only native inspector')
    observed = receipt.get('data_profile') or {}
    manifest = validate_manifest(observed.get('episode_manifest'), root=source)
    if (manifest['identity_field'], manifest['identity']) != (field, identity):
        raise ValueError('episode manifest does not describe the registered bytes')
    if (manifest['scope'] != 'full' or observed.get('scope') != 'full' or observed.get('video_decode_scope') not in
            ({'full','not_applicable'} if native_source else {'full'})):
        raise ValueError('partial/sampled manifest cannot certify the complete committed population')
    rows = manifest['episodes']
    valid = sum(r['committed'] and r['format_valid'] is True for r in rows)
    native_counts = _counts(observed.get('counts', {}), COUNTS) if native_source else {
        'transitions':observed['counts']['parquet_frames']}
    if 'episodes' in native_counts and native_counts['episodes'] != len(rows):
        raise ValueError('native manifest and profiled episode population disagree')
    row = {'identity_field': field, 'identity': identity, 'scope': 'full',
        'format': observed['format'], 'training_only': None,
        'counts': {**native_counts, 'episodes': len(rows),
                   'invalid_episodes': len(rows) - valid},
        'issues': {'format_issues_reported': len(observed.get('issues') or [])},
        'coverage': {'known_source_families': len({r['source_family'] for r in rows if r['source_family'] is not None})},
        'funnel': {'committed_episodes': len(rows), 'format_valid_episodes': valid},
        'manifest': {'scope': 'full', 'sha256': object_digest(manifest),
            'unknown_source_families': sum(r['source_family'] is None for r in rows),
            'unknown_supervision': sum(r['supervision'] is None for r in rows)}}
    # The inspector does not know the benchmark domain. Independent registration
    # review, not format validation, establishes training-only scope and label semantics.
    witness = {'evidence_id': evidence_id, 'evidence_ref': str(record.relative_to(output)),
        'record_sha256': digest(record), 'log_sha256': evidence['log_sha256'],
        'receipt_ref': verified['receipt_ref'], 'receipt_sha256': digest(receipt_path)}
    if native_source:
        from .data_health import inspector_source
        inspector = receipt.get('data_profile_inspector') or {}
        if repo is None or inspector_source(repo, inspector.get('ref')) != inspector:
            raise ValueError('native inspector source changed or is not available to registration review')
        witness['inspector'] = inspector
    if receipt.get('data_selection'):
        row['selection'] = receipt['data_selection']
    return row, witness


def _profile(row, *, training_scope_unverified=False):
    allowed = {'identity_field', 'identity', 'scope', 'format', 'counts', 'issues',
               'coverage', 'parents', 'recipe', 'training_only', 'funnel', 'manifest', 'selection'}
    if set(row) - allowed or row.get('scope') not in {'full', 'sample', 'metadata'}:
        raise ValueError('data profile needs a supported scope and bounded fields')
    if row.get('training_only') is not True and not (
            training_scope_unverified and row.get('training_only') is None):
        raise ValueError('data profile must describe allowed training data, not held-out inputs')
    counts = _counts(row.get('counts', {}), COUNTS)
    if 'episodes' in counts and sum(counts.get(k, 0) for k in
            ('successful_episodes', 'failed_episodes', 'invalid_episodes')) > counts['episodes']:
        raise ValueError('disjoint episode outcome counts exceed episode count')
    result = {'scope': row['scope'], 'format': _text(row.get('format'), 'native format', 120),
            'counts': counts, 'issues': _counts(row.get('issues', {})),
            'coverage': _counts(row.get('coverage', {})),
            'authority': 'native-source-reviewed aggregate; sample scope is not a full-dataset count'}
    if 'funnel' in row:
        result['funnel'] = _counts(row['funnel'])
    if 'manifest' in row:
        if (not isinstance(row['manifest'], dict) or set(row['manifest']) != {
                'scope', 'sha256', 'unknown_source_families', 'unknown_supervision'}
                or row['manifest']['scope'] != 'full' or
                not re.fullmatch(r'[0-9a-f]{64}', str(row['manifest']['sha256']))):
            raise ValueError('invalid sealed manifest summary')
        _counts({k: row['manifest'][k] for k in ('unknown_source_families', 'unknown_supervision')})
        result['manifest'] = row['manifest']
    if 'selection' in row:
        from .data_quality import finite_json
        finite_json(row['selection'], maximum=16000)
        result['selection'] = row['selection']
    if training_scope_unverified:
        result['training_scope'] = 'requires_independent_registration_review'
    return result


def _recipe(value):
    if not isinstance(value, dict) or set(value) != {'operation', 'parameters', 'preserves_episode_groups'}:
        raise ValueError('processed data require a reproducible recipe, without editing raw inputs')
    _text(value['operation'], 'recipe operation', 120)
    params = value['parameters']
    if not isinstance(params, dict) or len(params) > 32 or value['preserves_episode_groups'] is not True:
        raise ValueError('recipe must preserve complete episode/parent groups')
    # No executable recipe, arbitrary files or unbounded per-episode rows in the registry.
    try:
        encoded = json.dumps(params, allow_nan=False)
    except (RecursionError,TypeError) as exc:
        raise ValueError('recipe parameters must be bounded JSON') from exc
    if len(encoded.encode()) > 4000:
        raise ValueError('recipe parameters exceed bounds; keep large selections in native artifacts')
    return value


def build(output, repo, spec, source, native, field, identity):
    if spec is None:
        return {'kind': 'unknown', 'parents': [], 'profile': {},
                'authority': 'legacy data registration; provenance and counts not established'}
    if (not isinstance(spec, dict) or set(spec) - {'kind', 'parents', 'producer_evidence_id',
            'profile_evidence_id', 'manifest_evidence_id', 'producer_scope',
            'license', 'description'} or spec.get('kind') not in KINDS):
        raise ValueError('asset requires a supported kind and known fields')
    kind = spec['kind']
    prior = parents(output, repo, spec.get('parents', []))
    if any(Path(row['source']) == source or source.is_relative_to(Path(row['source']))
           or Path(row['source']).is_relative_to(source) for row in prior):
        raise ValueError('derived data must use separate output; never overlap or modify raw parents')
    if kind == 'processed' and not prior:
        raise ValueError('processed data require immutable registered parents')
    producer_id = spec.get('producer_evidence_id')
    if kind in {'collected', 'processed', 'synthetic'} and not producer_id:
        raise ValueError('new data need actual completed producer evidence')
    profile_id = spec.get('profile_evidence_id') or producer_id
    manifest_id = spec.get('manifest_evidence_id')
    manifest_profile = _manifest_profile(output, manifest_id, source, field, identity, repo=repo) if manifest_id else None
    producer_output = output
    scope = spec.get('producer_scope', 'this_run')
    if scope not in {'this_run', 'budget_parent'}:
        raise ValueError('producer_scope must be this_run or the verified budget_parent')
    if scope == 'budget_parent':
        from .run_successor import validate_inheritance
        validate_inheritance(output)
        _, inheritance = _record(output, 'budget_inheritance.json')
        producer_output = Path(inheritance['parent']).resolve(strict=True)
        if kind != 'collected' or not manifest_profile:
            raise ValueError('parent-run collection requires a fresh full manifest probe and scope review')
        profile_id = None
    profile, witness, recipe = {}, {}, {}
    if producer_id:
        row, witness = _sealed_profile(producer_output, producer_id, field, identity, producer=True,
            native=source if scope == 'budget_parent' else native,
            partial_profile=manifest_profile if kind == 'collected' else None)
        witness['scope'] = scope
        request, request_binding = _collection_request(producer_output, witness)
        if request_binding:
            witness['job_request'] = request_binding
        if request.get('settings', {}).get('episodes') is not None:
            requested = request['settings']['episodes']
            if type(requested) is int and requested > 0:
                row.setdefault('funnel', {})['requested_episodes'] = requested
                if manifest_profile:
                    manifest_profile[0].setdefault('funnel', {})['requested_episodes'] = requested
        profile = _profile(row, training_scope_unverified=bool(witness.get('native_manifest')))
        expected = [{k: p[k] for k in ('id', 'identity_field', 'identity', 'record_sha256')} for p in prior]
        if row.get('parents', []) != expected:
            raise ValueError('producer profile does not bind exact immutable parents')
        if kind == 'processed':
            recipe = _recipe(row.get('recipe'))
    if manifest_profile:
        row, profile_witness = manifest_profile
        profile = _profile(row, training_scope_unverified=True)
    elif profile_id and profile_id != producer_id:
        row, profile_witness = _sealed_profile(output, profile_id, field, identity)
        profile = _profile(row)
    else:
        profile_witness = witness
    result = {'schema_version': 1, 'kind': kind, 'parents': prior,
              'profile': profile, 'producer': witness, 'profile_witness': profile_witness,
              'recipe': recipe, 'authority': 'lineage and native evidence; not certified quality or performance'}
    for key in ('license', 'description'):
        if key in spec:
            result[key] = _text(spec[key], key)
    return result


def verify(output, repo, asset, *, _seen=None, _depth=0):
    """Recheck small evidence identities and raw parents when actually adopting data."""
    if _depth > 16:
        raise ValueError('data lineage depth exceeds bounded 16 levels')
    _seen = set() if _seen is None else _seen
    prior = asset.get('parents') or []
    if repo is None:
        current = []
        for row in prior:
            path, recorded = _record(output, row['record_ref'])
            if (digest(path) != row['record_sha256']
                    or artifact_identity(Path(row['source'])) != (row['identity_field'], row['identity'])):
                raise ValueError('data parent registration or bytes changed during training')
            current.append({**{k: recorded[k] for k in
                ('id', 'source', 'native_ref', 'identity_field', 'identity')},
                'record_ref':row['record_ref'], 'record_sha256':digest(path)})
    else:
        current = parents(output, repo, [row['id'] for row in prior])
    if current != prior:
        raise ValueError('data parent registration or bytes changed')
    for row in prior:
        if row['id'] not in _seen:
            if len(_seen)>=64:
                raise ValueError('data lineage exceeds bounded 64 ancestors')
            _seen.add(row['id'])
            _, record = _record(output, row['record_ref'])
            verify(output, repo, record.get('asset') or {}, _seen=_seen, _depth=_depth+1)
    for witness in (asset.get('producer') or {}, asset.get('profile_witness') or {}):
        if witness:
            witness_output = output
            if witness.get('scope') == 'budget_parent':
                from .run_successor import validate_inheritance
                validate_inheritance(output)
                _, inherited = _record(output, 'budget_inheritance.json')
                witness_output = Path(inherited['parent'])
            path, _ = _record(witness_output, witness['evidence_ref'])
            if digest(path) != witness['record_sha256']:
                raise ValueError('data producer/profile evidence changed')
            read_attempt_evidence(witness_output, witness['evidence_id'], limit=1)
            if witness.get('receipt_ref'):
                receipt_path, _ = _record(witness_output, witness['receipt_ref'], max_bytes=2 * 1024**2)
                if digest(receipt_path)!=witness['receipt_sha256']:
                    raise ValueError('data producer receipt changed')
            if witness.get('job_request'):
                request_path, _ = _record(witness_output, witness['job_request']['request_ref'], max_bytes=2 * 1024**2)
                if digest(request_path) != witness['job_request']['request_sha256']:
                    raise ValueError('collection request/context changed')
            recovery = witness.get('recovery')
            if recovery:
                proof, _ = _record(output, recovery['evidence_ref'])
                receipt_path, _ = _record(output, recovery['receipt_ref'], max_bytes=2 * 1024**2)
                if (digest(proof) != recovery['record_sha256'] or
                        digest(receipt_path) != recovery['receipt_sha256']):
                    raise ValueError('partial collection recovery evidence changed')
                read_attempt_evidence(output, recovery['evidence_id'], limit=1)
            inspector = witness.get('inspector')
            if inspector and repo is not None:
                from .data_health import inspector_source
                if inspector_source(repo, inspector.get('ref')) != inspector:
                    raise ValueError('reviewed native manifest inspector changed')


def exposure(row, binding):
    value = row.get('exposure')
    if value is None:
        return {'status': 'basic_read_only', 'samples_read': row['samples_read']}
    if not isinstance(value, dict) or set(value) - {'batches', 'unique_samples', 'source_samples'}:
        raise ValueError('invalid aggregate training exposure')
    counts = _counts({k: v for k, v in value.items() if k != 'source_samples'}, {'batches', 'unique_samples'})
    if counts.get('batches', 1) < 1 or counts.get('unique_samples', 0) > row['samples_read']:
        raise ValueError('exposure counters contradict sample reads')
    full = (binding.get('asset') or {}).get('profile') or {}
    total = (full.get('counts') or {}).get('samples') if full.get('scope') == 'full' else None
    if total is not None and counts.get('unique_samples', 0) > total:
        raise ValueError('unique exposure exceeds profiled dataset samples')
    sources = value.get('source_samples')
    if sources is not None:
        sources = _counts(sources)
        allowed = {binding['id']} | {p['id'] for p in (binding.get('asset') or {}).get('parents', [])}
        if set(sources) - allowed or sum(sources.values()) != row['samples_read']:
            raise ValueError('source exposure must refer to selected data/parents and sum to actual reads')
    return {'status': 'aggregate_reported', 'samples_read': row['samples_read'], **counts,
            **({'source_samples': sources} if sources is not None else {}),
            'authority': 'sealed source-reviewed loader counters; not sufficient learning or quality'}
