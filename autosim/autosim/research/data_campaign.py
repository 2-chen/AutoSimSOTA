"""Immutable development-data hypotheses and measured arms, not scientific approval.

The model chooses acquisition and selection. This registry links those decisions to
existing native producer, data-consumption and development-measurement contracts.
It never starts work, infers an expert from failed policy actions or promotes a score.
"""
from pathlib import Path
import math
import re

from .common import atomic_json, digest, now, object_digest, read_json
from .scheduling import locked


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise ValueError('expected a safe campaign/version/run id')
    return value


def _path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError('campaign evidence must not use symlinks')
    if not path.resolve().is_relative_to(root):
        raise ValueError('campaign evidence leaves the run')
    return path


def _read(root, relative):
    path = _path(root, relative)
    if not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError('missing or oversized campaign evidence')
    row = read_json(path)
    if not isinstance(row, dict):
        raise ValueError('expected an evidence object')
    return row, digest(path)


def _measurement(root, run_id, label):
    if not isinstance(label, str) or not re.fullmatch(r'baseline|round_[0-9]+', label):
        raise ValueError('only native baseline/round development measurements are allowed')
    ref = f'research/{_id(run_id)}/measurements/{label}.json'
    row, sha = _read(root, ref)
    # Native measurements describe *what* was scored with a string. A Boolean
    # fixture is not the runtime schema and must not reject every genuine arm.
    if (row.get('label') != label or row.get('ok') is not True
            or row.get('scored') is False
            or row.get('confirmation') or row.get('baseline_origin') == 'released_reference'):
        raise ValueError('a scored development measurement is required')
    from .receipt_verifier import verify_measurement
    if verify_measurement(Path(root)/'research'/run_id, label).get('status') != 'consistent':
        raise ValueError('development measurement receipts are not independently consistent')
    if any((row.get(k) or {}).get('status') != 'verified'
           for k in ('policy_consumption', 'rollout_evidence', 'metric_lineage')):
        raise ValueError('loaded policy, real rollout and metric identity must be verified')
    value = row.get('metric_value')
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('a finite measured score is required')
    return row, {'ref': ref, 'sha256': sha}


def _development_open(root, run_id):
    directory = _path(root, f'research/{_id(run_id)}/confirmation_attempts')
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('held-out exposure closes data campaign development')


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f'{name} must be a nonempty bounded explanation')
    return value


def register(root, run_id, campaign):
    _development_open(root, run_id)
    required = {'id', 'development_label', 'hypothesis', 'acquisition', 'selection', 'comparison'}
    if not isinstance(campaign, dict) or set(campaign) != required:
        raise ValueError('campaign requires exactly id,development_label,hypothesis,acquisition,selection,comparison')
    identity = _id(campaign['id'])
    measurement, witness = _measurement(root, run_id, campaign['development_label'])
    for name in ('hypothesis', 'acquisition', 'selection'):
        _text(campaign[name], name)
    comparison = campaign['comparison']
    required_comparison = {'arms', 'usable_episodes', 'training_recipe', 'seeds', 'compute_accounting'}
    if not isinstance(comparison, dict) or set(comparison) != required_comparison:
        raise ValueError('comparison requires arms,usable_episodes,training_recipe,seeds,compute_accounting')
    arms = comparison['arms']
    if (not isinstance(arms, list) or len(arms) != len(set(str(x) for x in arms))
            or not {'targeted', 'random'} <= set(str(x) for x in arms)
            or any(x not in ('targeted', 'random', 'unfiltered') for x in arms)):
        raise ValueError('prespecify targeted/random arms and optional unfiltered ablation')
    if type(comparison['usable_episodes']) is not int or not 1 <= comparison['usable_episodes'] <= 1000000:
        raise ValueError('usable_episodes must be a positive bounded integer')
    seeds = comparison['seeds']
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 32
            or any(type(x) is not int or not 0 <= x < 2**32 for x in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError('prespecify distinct training/collection seed identifiers')
    for name in ('training_recipe', 'compute_accounting'):
        _text(comparison[name], name)
    row = {**campaign, 'run_id': run_id, 'at': now(), 'schema_version': 1,
           'development_witness': witness,
           'starting_score': measurement['metric_value'],
           'status': 'hypothesis_registered',
           'authority': 'Agent hypothesis; not causal diagnosis, collection authorization or improvement'}
    _save(root, f'data_campaigns/{identity}/plan.json', row)
    return row


def _save(root, ref, row):
    path = _path(root, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(path.parent / 'registry.lock'):
        if path.exists() or path.is_symlink():
            raise ValueError('campaign plans and arm observations are immutable; use a new id/seed')
        for key in ('development_witness', 'measurement_witness'):
            if key in row:
                _check_witness(root, row[key])
        if 'measurement_witness' in row:
            for previous in path.parent.glob('*.json'):
                document, _ = _read(root, previous.relative_to(Path(root)))
                if document.get('measurement_witness') == row['measurement_witness']:
                    raise ValueError('the same measurement cannot stand for two campaign arms/seeds')
        atomic_json(path, row)


def _check_witness(root, witness):
    _, sha = _read(root, witness['ref'])
    if sha != witness['sha256']:
        raise ValueError('campaign evidence changed after registration')


def record_arm(root, repo, run_id, *, campaign_id, arm, seed, measurement_label,
               acquired_version_id, training_version_id):
    _development_open(root, run_id)
    identity = _id(campaign_id)
    plan, plan_sha = _read(root, f'data_campaigns/{identity}/plan.json')
    if plan.get('run_id') != run_id or arm not in plan['comparison']['arms']:
        raise ValueError('arm/run does not match the campaign')
    if type(seed) is not int or seed not in plan['comparison']['seeds']:
        raise ValueError('seed must have been prespecified')
    _check_witness(root, plan['development_witness'])
    measured, witness = _measurement(root, run_id, measurement_label)
    if measurement_label == plan['development_label']:
        raise ValueError('the starting measurement is not a new data intervention')
    for previous in _path(root, f'data_campaigns/{identity}').glob('*.json'):
        if previous.name == 'plan.json':
            continue
        document, _ = _read(root, previous.relative_to(root))
        if document.get('measurement_witness') == witness:
            raise ValueError('the same measurement cannot stand for two campaign arms/seeds')
    from .data_versions import resolve
    _, bindings = resolve(Path(root), Path(repo), {
        'acquired': 'data-version:' + _id(acquired_version_id),
        'training': 'data-version:' + _id(training_version_id)})
    acquired, training = bindings
    # Registered parent records are already recursively byte-verified by resolve.
    def has_parent(row, wanted, depth=0):
        if depth > 16:
            raise ValueError('campaign lineage is too deep')
        if row['id'] == wanted:
            return True
        for parent in (row.get('asset') or {}).get('parents', []):
            document, _ = _read(root, 'data_versions/' + _id(parent['id']) + '.json')
            if has_parent(document, wanted, depth + 1):
                return True
        return False
    if not has_parent(training, acquired['id']):
        raise ValueError('acquired data is not a parent of the measured training version')
    asset = acquired.get('asset') or {}
    if asset.get('kind') not in ('collected', 'processed', 'synthetic') or not asset.get('producer'):
        raise ValueError('new data requires sealed native collection/processing evidence')
    def collected_ancestor(row, depth=0):
        if depth > 16:
            raise ValueError('campaign lineage is too deep')
        metadata = row.get('asset') or {}
        producer = metadata.get('producer') or {}
        if metadata.get('kind') in ('collected', 'synthetic') and producer:
            from .evidence_store import read_attempt_evidence
            evidence = read_attempt_evidence(Path(root), producer['evidence_id'], limit=1)
            if (evidence.get('native_context') or {}).get('stage') == 'collect':
                return True
        return any(collected_ancestor(_read(root, 'data_versions/' + _id(parent['id']) + '.json')[0], depth+1)
                   for parent in metadata.get('parents', []))
    if not collected_ancestor(acquired):
        raise ValueError('acquisition arm needs a native collect ancestor; processing old data is a separate selection track')
    consumed = measured.get('data_consumption') or {}
    if consumed.get('status') != 'verified' or training['id'] not in consumed.get('versions', []):
        raise ValueError('the evaluated training must have consumed the registered version')
    profile = asset.get('profile') or {}
    count = (profile.get('counts') or {}).get('episodes') if profile.get('scope') == 'full' else None
    row = {'schema_version': 1, 'at': now(), 'campaign_id': identity, 'run_id': run_id,
           'arm': arm, 'seed': seed, 'measurement_label': measurement_label,
           'measurement_witness': witness, 'plan_sha256': plan_sha,
           'data_witnesses': [{'ref': b['record_ref'], 'sha256': b['record_sha256']} for b in bindings],
           'acquired_version_id': acquired['id'], 'training_version_id': training['id'],
           'reported_full_episodes': count, 'matches_requested_count': count == plan['comparison']['usable_episodes'],
           'metric_value': measured['metric_value'], 'status': 'consumed_and_measured',
           'authority': 'Observed development arm; seed/recipe equivalence, effective mixture and new-data count need independent audit'}
    _save(root, f'data_campaigns/{identity}/{arm}-{seed}.json', row)
    return row


def view(root, run_id):
    result = []
    try:
        directory = _path(root, 'data_campaigns')
    except ValueError:
        return [{'id': 'registry', 'status': 'evidence_unavailable', 'observations': []}]
    if not directory.exists():
        return result
    # Bounded read-only projection; no GPU, model call or dataset rehash on heartbeat.
    for path in sorted(directory.glob('*/plan.json'))[-8:]:
        try:
            plan, sha = _read(root, path.relative_to(root))
            if plan.get('run_id') != run_id:
                continue
            _check_witness(root, plan['development_witness'])
            observations = []
            for arm in plan['comparison']['arms']:
                for seed in plan['comparison']['seeds']:
                    ref = f'data_campaigns/{_id(plan["id"])}/{arm}-{seed}.json'
                    if not _path(root, ref).exists():
                        continue
                    row, _ = _read(root, ref)
                    if row.get('plan_sha256') != sha:
                        raise ValueError('campaign plan changed')
                    for witness in [row['measurement_witness'], *row['data_witnesses']]:
                        _check_witness(root, witness)
                    observations.append({**row, 'record_ref': ref})
            result.append({'id': plan['id'], 'status': plan['status'],
                           'hypothesis': plan['hypothesis'][:400],
                           'development_label': plan['development_label'],
                           'starting_score': plan['starting_score'],
                           'comparison': {**plan['comparison'],
                               'training_recipe': plan['comparison']['training_recipe'][:400],
                               'compute_accounting': plan['comparison']['compute_accounting'][:400]},
                           'plan_ref': str(path.relative_to(root)),
                           'observations': [{key: row[key] for key in (
                               'arm','seed','measurement_label','acquired_version_id','training_version_id',
                               'reported_full_episodes','matches_requested_count','metric_value','record_ref')}
                               for row in observations],
                           'fairness_status': 'requires_independent_audit'})
        except (OSError, ValueError, TypeError, KeyError):
            result.append({'id': path.parent.name, 'status': 'evidence_unavailable', 'observations': []})
    return result
