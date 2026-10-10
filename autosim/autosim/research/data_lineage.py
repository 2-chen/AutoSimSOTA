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
from .common import digest, read_json
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


def _record(output, ref):
    rel = Path(str(ref or ''))
    path = output / rel
    if (not ref or rel.is_absolute() or '..' in rel.parts or path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(output.resolve())
            or path.stat().st_size > 128 * 1024):
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


def _sealed_profile(output, evidence_id, field, identity, *, producer=False, native=None):
    verified = read_attempt_evidence(output, evidence_id, limit=1)
    path, evidence = _record(output, f'evidence/{evidence_id}.json')
    if verified['status'] != 'completed' or verified['returncode'] != 0:
        raise ValueError('data profile needs a completed successful native attempt')
    if producer:
        receipt_path, receipt = _record(output, verified['receipt_ref'])
        context = verified.get('native_context') or {}
        if (receipt.get('stage') not in {'collect', 'prepare_data'}
                or receipt.get('attempt_id') != evidence_id or receipt.get('status') != 'completed'
                or receipt.get('artifact') != verified.get('artifact')
                or any(context.get(k) != receipt.get(k) for k in
                       ('stage','output_directory','working_directory')) or not context):
            raise ValueError('producer must be a sealed collect/prepare_data attempt')
        artifact = verified.get('artifact') or {}
        root = Path(context['working_directory'] if artifact.get('found_beside_the_command')
                    else context['output_directory'])
        examples = artifact.get('candidate_paths') or artifact.get('examples') or []
        paths = [Path(e) if Path(e).is_absolute() else root / e for e in examples if isinstance(e,str)]
        if (not root.is_absolute() or not root.resolve().is_relative_to(output.resolve())
                or not isinstance(examples, list) or not examples
                or not any('..' not in p.parts and p.resolve().is_relative_to(output.resolve())
                           and (native == p or native.is_relative_to(p)) for p in paths)):
            raise ValueError('dataset is not inside the recorded producer artifact')
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


def _profile(row):
    allowed = {'identity_field', 'identity', 'scope', 'format', 'counts', 'issues',
               'coverage', 'parents', 'recipe', 'training_only'}
    if set(row) - allowed or row.get('scope') not in {'full', 'sample', 'metadata'}:
        raise ValueError('data profile needs a supported scope and bounded fields')
    if row.get('training_only') is not True:
        raise ValueError('data profile must describe allowed training data, not held-out inputs')
    counts = _counts(row.get('counts', {}), COUNTS)
    if 'episodes' in counts and sum(counts.get(k, 0) for k in
            ('successful_episodes', 'failed_episodes', 'invalid_episodes')) > counts['episodes']:
        raise ValueError('disjoint episode outcome counts exceed episode count')
    return {'scope': row['scope'], 'format': _text(row.get('format'), 'native format', 120),
            'counts': counts, 'issues': _counts(row.get('issues', {})),
            'coverage': _counts(row.get('coverage', {})),
            'authority': 'native-source-reviewed aggregate; sample scope is not a full-dataset count'}


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
            'profile_evidence_id', 'license', 'description'} or spec.get('kind') not in KINDS):
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
    profile, witness, recipe = {}, {}, {}
    if producer_id:
        row, witness = _sealed_profile(output, producer_id, field, identity, producer=True, native=native)
        profile = _profile(row)
        expected = [{k: p[k] for k in ('id', 'identity_field', 'identity', 'record_sha256')} for p in prior]
        if row.get('parents', []) != expected:
            raise ValueError('producer profile does not bind exact immutable parents')
        if kind == 'processed':
            recipe = _recipe(row.get('recipe'))
    if profile_id and profile_id != producer_id:
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
            path, _ = _record(output, witness['evidence_ref'])
            if digest(path) != witness['record_sha256']:
                raise ValueError('data producer/profile evidence changed')
            read_attempt_evidence(output, witness['evidence_id'], limit=1)
            if witness.get('receipt_ref'):
                receipt_path, _ = _record(output, witness['receipt_ref'])
                if digest(receipt_path)!=witness['receipt_sha256']:
                    raise ValueError('data producer receipt changed')


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
