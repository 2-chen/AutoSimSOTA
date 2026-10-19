"""Immutable development-data hypotheses and measured arms, not scientific approval.

The model chooses acquisition and selection. This registry links those decisions to
existing native producer, data-consumption and development-measurement contracts.
It never starts work, infers an expert from failed policy actions or promotes a score.
"""
from pathlib import Path
import json
import hashlib
import math
import re

from .common import atomic_json, digest, now, object_digest, read_json
from .scheduling import locked

METHODS = {'fixed_official', 'fixed_continuation', 'random_quality', 'rule_targeted',
           'agent_targeted', 'targeted_quality', 'fdas'}
RULE_MARKER = 'AUTOSIM_FAILURE_STRATA '


def rule_allocation(strata, total):
    """Frozen failure-frequency baseline, not an LLM target selector.

    Strata/domain are prespecified. Counts must come from native development evidence;
    largest remainders give an exact finite allocation without silent expansion.
    """
    if type(total) is not int or not 1 <= total <= 1000000 or not isinstance(strata, list) or not 1 <= len(strata) <= 32:
        raise ValueError('rule allocation requires bounded strata and an explicit positive count')
    ids, failures = [], []
    for row in strata:
        if (not isinstance(row, dict) or set(row) != {'id', 'failures', 'evaluated'}
                or type(row['failures']) is not int or type(row['evaluated']) is not int
                or not 0 <= row['failures'] <= row['evaluated'] <= 1000000):
            raise ValueError('stratum needs actual nonnegative failure/evaluation counts')
        ids.append(_id(row['id'])); failures.append(row['failures'])
    if len(set(ids)) != len(ids) or not sum(failures):
        raise ValueError('unique strata with observed failures are required; zero evidence is not a target')
    denominator = sum(failures)
    counts = [total * n // denominator for n in failures]
    order = sorted(range(len(ids)), key=lambda i: (-(total * failures[i] % denominator), ids[i]))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return {'algorithm':'failure_frequency_v1', 'requested':total,
        'allocation':dict(zip(ids, counts)), 'authority':'deterministic allocation; native supply and actual sampler use still require verification'}


def native_rule(output, repo, spec, total):
    from .evidence_store import read_attempt_evidence
    from .baseline_reference import source_evidence
    if (not isinstance(spec, dict) or set(spec) != {'algorithm','strata_evidence_id','source_refs'}
            or spec['algorithm'] != 'failure_frequency_v1' or not spec['source_refs']):
        raise ValueError('rule comparison requires a frozen algorithm, native strata evidence and current source refs')
    if repo is None:
        raise ValueError('explicit workspace is required to verify rule sources')
    refs = source_evidence(Path(repo), spec['source_refs'], output=Path(output))
    verified = read_attempt_evidence(Path(output), spec['strata_evidence_id'], limit=1)
    if verified.get('status') != 'completed' or verified.get('returncode') != 0:
        raise ValueError('rule strata need a completed native diagnostic, not a model report')
    record, _ = _read(output, 'evidence/' + spec['strata_evidence_id'] + '.json')
    events = []
    with _path(output, record['log_ref']).open() as stream:
        for line in stream:
            if line.startswith(RULE_MARKER):
                if len(line.encode()) > 16000:
                    raise ValueError('native strata event exceeds bound')
                events.append(json.loads(line[len(RULE_MARKER):]))
    if len(events) != 1 or not isinstance(events[0], list):
        raise ValueError('exactly one native failure-strata event is required')
    return {**rule_allocation(events[0], total), 'strata_evidence_id':spec['strata_evidence_id'],
            'source_refs':refs, 'strata_evidence_sha256':digest(_path(output, 'evidence/'+spec['strata_evidence_id']+'.json'))}


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
    if not isinstance(label, str) or not re.fullmatch(
            r'baseline|round_[0-9]+(?:_handoff_correction|_eval_[0-9a-f]{12})?', label):
        raise ValueError('only native baseline/round development measurements are allowed')
    ref = f'research/{_id(run_id)}/measurements/{label}.json'
    row, sha = _read(root, ref)
    if '_eval_' in label:
        link=row.get('completed_evaluation_link') or {}
        job=link.get('job_id')
        if (not isinstance(job,str) or not re.fullmatch(r'[0-9a-f]{32}',job) or
                not label.endswith('_eval_'+job[:12]) or link.get('native_replayed') is not False):
            raise ValueError('linked evaluation label requires its exact completed-job proof')
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
    if label.endswith('_handoff_correction'):
        original_label = label.removesuffix('_handoff_correction')
        original_ref = f'research/{run_id}/measurements/{original_label}.json'
        original, original_sha = _read(root, original_ref)
        correction = row.get('handoff_correction') or {}
        attempt = (original.get('train') or {}).get('attempt_id')
        if (not isinstance(correction, dict) or
                original.get('label') != original_label or original.get('confirmation') or
                correction.get('status') != 'measured' or
                correction.get('original_measurement_ref') != original_ref or
                correction.get('original_measurement_sha256') != original_sha or
                correction.get('corrected_label') != label or
                correction.get('training_reused') is not True or
                correction.get('training_attempt_id') != attempt or
                not re.fullmatch(r'[a-f0-9]{32}', str(attempt or '')) or
                (row.get('train') or {}).get('attempt_id') != attempt or
                row.get('settings') != original.get('settings')):
            raise ValueError('corrected data measurement lacks its original same-producer/settings proof')
        receipt_ref = f'research/{run_id}/attempts/{attempt}/receipt.json'
        _, receipt_sha = _read(root, receipt_ref)
        state, _ = _read(root, f'research/{run_id}/attempts/{attempt}/handoff_correction.json')
        if (correction.get('training_receipt_sha256') != receipt_sha or
                state.get('status') != 'measured' or
                state.get('result_digest') != object_digest(row)):
            raise ValueError('corrected data measurement is not its sealed completed recovery')
        selections = []
        for measured in (original, row):
            proof = measured.get('data_selection')
            if proof is None:
                selections.append(None)
                continue
            if (not isinstance(proof, dict) or
                    not re.fullmatch(r'[a-f0-9]{64}', str(proof.get('id') or '')) or
                    proof.get('training_attempt_id') != attempt or
                    proof.get('record_ref') !=
                    f'research/{run_id}/data_selections/{proof["id"]}.json'):
                raise ValueError('corrected data selection must retain its original producer')
            body, body_sha = _read(root, proof['record_ref'])
            if (body_sha != proof.get('record_sha256') or object_digest(body) != proof['id'] or
                    body.get('label') != measured['label'] or
                    body.get('settings_sha256') != object_digest(measured.get('settings'))):
                raise ValueError('corrected data selection proof changed')
            selections.append(body.get('bindings'))
        if selections[0] != selections[1]:
            raise ValueError('correction cannot retrospectively select a different data version')
    return row, {'ref': ref, 'sha256': sha}


def _development_open(root, run_id):
    directory = _path(root, f'research/{_id(run_id)}/confirmation_attempts')
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('held-out exposure closes data campaign development')


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f'{name} must be a nonempty bounded explanation')
    return value


def register(root, run_id, campaign, *, repo=None):
    _development_open(root, run_id)
    required = {'id', 'development_label', 'hypothesis', 'acquisition', 'selection', 'comparison'}
    version = campaign.get('schema_version', 1) if isinstance(campaign, dict) else None
    if (not isinstance(campaign, dict) or set(campaign) - {'schema_version'} != required
            or type(version) is not int or version not in (1, 2)):
        raise ValueError('campaign requires exactly id,development_label,hypothesis,acquisition,selection,comparison')
    identity = _id(campaign['id'])
    measurement, witness = _measurement(root, run_id, campaign['development_label'])
    for name in ('hypothesis', 'acquisition', 'selection'):
        _text(campaign[name], name)
    comparison = campaign['comparison']
    required_comparison = {'arms', 'usable_episodes', 'training_recipe', 'seeds', 'compute_accounting'}
    if version == 2:
        required_comparison |= {'budget_axis','budget_limit','quality_floor','frozen_factors','rule'}
    if not isinstance(comparison, dict) or set(comparison) != required_comparison:
        actual=set(comparison) if isinstance(comparison,dict) else set()
        hint=('; v2 budget/quality fields require explicit campaign.schema_version=2; '
              'omitting schema_version selects legacy v1' if version==1 and actual &
              {'budget_axis','budget_limit','quality_floor','frozen_factors','rule'} else '')
        raise ValueError(f'comparison schema_version={version}: missing={sorted(required_comparison-actual)}, '
                         f'unsupported={sorted(actual-required_comparison)}'+hint)
    arms = comparison['arms']
    supported = METHODS if version == 2 else {'targeted', 'random', 'unfiltered'}
    essential = {'random_quality', 'agent_targeted'} if version == 2 else {'targeted', 'random'}
    if (not isinstance(arms, list) or len(arms) != len(set(str(x) for x in arms))
            or not essential <= set(str(x) for x in arms) or any(x not in supported for x in arms)):
        raise ValueError(f'comparison schema_version={version} requires arms {sorted(essential)}; '
                         f'allowed arms are {sorted(supported)}')
    if type(comparison['usable_episodes']) is not int or not 1 <= comparison['usable_episodes'] <= 1000000:
        raise ValueError('usable_episodes must be a positive bounded integer')
    seeds = comparison['seeds']
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 32
            or any(type(x) is not int or not 0 <= x < 2**32 for x in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError('prespecify distinct training/collection seed identifiers')
    for name in ('training_recipe', 'compute_accounting'):
        _text(comparison[name], name)
    rule = None
    if version == 2:
        axis, bound = comparison['budget_axis'], comparison['budget_limit']
        if axis not in {'accepted_episodes','attempts','acquisition_seconds','total_compute'}:
            raise ValueError('declare one finite acquisition/comparison budget axis')
        if axis == 'total_compute':
            if (not isinstance(bound, dict) or not bound or set(bound) -
                    {'gpu_seconds','cpu_seconds','wall_seconds','model_usd'} or
                    any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in bound.values())):
                raise ValueError('total_compute needs an explicit resource package; never add unlike units')
        elif (type(bound) not in (int, float) or not math.isfinite(bound) or bound <= 0
                or axis in {'accepted_episodes','attempts'} and type(bound) is not int):
            raise ValueError('comparison budget must be finite and use the declared count/time unit')
        if axis == 'accepted_episodes' and bound != comparison['usable_episodes']:
            raise ValueError('accepted episode limit and matched data count must agree')
        _text(comparison['quality_floor'], 'common quality floor')
        from .data_quality import finite_json
        if not isinstance(comparison['frozen_factors'], dict) or not comparison['frozen_factors']:
            raise ValueError('freeze explicit common initialization, learning work and evaluation factors')
        finite_json(comparison['frozen_factors'], maximum=8000)
        if 'rule_targeted' in arms:
            rule = native_rule(root, repo, comparison['rule'], comparison['usable_episodes'])
        elif comparison['rule'] is not None:
            raise ValueError('a rule specification without rule_targeted is ambiguous')
    row = {**campaign, 'run_id': run_id, 'at': now(), 'schema_version': version,
           'development_witness': witness,
           'starting_score': measurement['metric_value'],
           'status': 'hypothesis_registered',
           'authority': 'Agent hypothesis; not causal diagnosis, collection authorization or improvement'}
    if rule is not None:
        row['rule_allocation'] = rule
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


def record_boundary(root, run_id, *, campaign_id, arm, seed, boundary, review):
    """An independently reviewed inability is not a zero-score experiment."""
    _development_open(root, run_id)
    identity = _id(campaign_id)
    plan, sha = _read(root, f'data_campaigns/{identity}/plan.json')
    if (plan.get('schema_version') != 2 or plan.get('run_id') != run_id or
            arm not in plan['comparison']['arms'] or type(seed) is not int or seed not in plan['comparison']['seeds']):
        raise ValueError('boundary must refer to a prespecified v2 arm and seed')
    if (not isinstance(boundary, dict) or set(boundary) != {'status','reason','evidence_refs','source_refs'}
            or boundary['status'] not in {'unsupported','budget_limited'}
            or not all(review.get(key) is True for key in ('boundary_evidence_supported','no_safe_next_data_operation'))):
        raise ValueError('explicit native boundary and independent no-safe-next-operation review are required')
    _text(boundary['reason'], 'boundary reason')
    refs = boundary['evidence_refs']
    if not isinstance(refs, list) or not 1 <= len(refs) <= 4:
        raise ValueError('boundary needs a bounded sealed evidence list')
    from .evidence_store import read_attempt_evidence
    witnesses = []
    for ident in refs:
        read_attempt_evidence(Path(root), ident, limit=1)
        _, evidence_sha = _read(root, 'evidence/'+ident+'.json')
        witnesses.append({'ref':'evidence/'+ident+'.json','sha256':evidence_sha})
    row = {'schema_version':2,'at':now(),'campaign_id':identity,'run_id':run_id,
        'arm':arm,'seed':seed,'status':boundary['status'],'reason':boundary['reason'],
        'plan_sha256':sha,'boundary_witnesses':witnesses,'source_refs':boundary['source_refs'],
        'review':review,'metric_value':None,'training_version_id':None,'acquired_version_id':None,
        'reported_full_episodes':None,'matches_requested_count':False,
        'authority':'independently reviewed unresolved data boundary; not a completed successful experiment'}
    _save(root, f'data_campaigns/{identity}/{arm}-{seed}.json', row)
    return row


def pending(root, run_id, *, require_campaign=False):
    """Finalization cannot silently abandon a declared data experiment.

    Only v2 plans opt into this lifecycle. Old trajectories are never reopened/relabelled.
    All plans are checked, not only the last eight visible on a report page.
    """
    unresolved = []
    required_plans = 0
    plans = list((Path(root)/'data_campaigns').glob('*/plan.json'))
    if len(plans) > 256:
        raise ValueError('data obligations exceed bounded catalog; finalize none blindly')
    for path in plans:
        plan, sha = _read(root, path.relative_to(Path(root)))
        if plan.get('schema_version') != 2 or plan.get('run_id') != run_id:
            continue
        if require_campaign:
            comparison = plan.get('comparison') or {}
            arms, seeds = comparison.get('arms'), comparison.get('seeds')
            if (plan.get('status') != 'hypothesis_registered' or
                    _id(plan.get('id')) != path.parent.name or
                    not isinstance(arms, list) or
                    not {'random_quality','agent_targeted'} <= set(str(a) for a in arms) or
                    any(a not in METHODS for a in arms) or len(arms) != len(set(arms)) or
                    not isinstance(seeds, list) or not 1 <= len(seeds) <= 32 or
                    any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or
                    len(seeds) != len(set(seeds))):
                raise ValueError('required data comparison needs a valid frozen v2 random/targeted plan')
            _check_witness(root, plan['development_witness'])
        required_plans += 1
        for arm in plan['comparison']['arms']:
            for seed in plan['comparison']['seeds']:
                ref = f'data_campaigns/{plan["id"]}/{arm}-{seed}.json'
                outcome = _path(root, ref)
                if not outcome.exists():
                    unresolved.append({'campaign_id':plan['id'],'arm':arm,'seed':seed,'missing':'native data arm or reviewed boundary'})
                    continue
                row, _ = _read(root, ref)
                if require_campaign and (row.get('campaign_id') != plan['id'] or
                        row.get('run_id') != run_id or row.get('arm') != arm or
                        type(row.get('seed')) is not int or row['seed'] != seed):
                    raise ValueError('required data outcome differs from its frozen arm/run/seed')
                if row.get('plan_sha256') != sha:
                    raise ValueError('data obligation outcome no longer binds the frozen plan')
                if row.get('status') in {'unsupported','budget_limited'}:
                    if not all(row.get('review', {}).get(key) is True for key in
                               ('boundary_evidence_supported','no_safe_next_data_operation')):
                        raise ValueError('unreviewed data boundary cannot close obligations')
                    for witness in row['boundary_witnesses']:
                        _check_witness(root, witness)
                elif row.get('status') == 'consumed_and_measured':
                    _check_witness(root, row['measurement_witness'])
                    for witness in row.get('data_witnesses', []):
                        _check_witness(root, witness)
                else:
                    unresolved.append({'campaign_id':plan['id'],'arm':arm,'seed':seed,'missing':'accepted terminal evidence'})
    if require_campaign and not required_plans:
        unresolved.append({'missing':'required prespecified v2 data campaign',
                           'next_action':'register_data_campaign before acquisition/selection interventions; '
                                         'resolve its actual arms or independently reviewed boundaries'})
    return unresolved


def _acquired_version(root,repo,identity):
    """An ancestor asset needs its own provenance, not a fictitious loader slot."""
    from .baseline_reference import checkpoint_location
    from .experiment_bundle import artifact_identity
    from .data_lineage import verify
    ref='data_versions/'+_id(identity)+'.json'
    row,sha=_read(root,ref)
    if (row.get('id')!=identity or row.get('status')!='reviewed_not_consumed' or
            not all(row.get('review',{}).get(k) is True for k in
                ('allowed_training_data','loader_connection_supported','sources_supported'))):
        raise ValueError('acquired version lacks its original accepted registration')
    if ((row.get('asset') or {}).get('profile_witness',{}).get('inspector') and
            row.get('review',{}).get('manifest_observations_supported') is not True):
        raise ValueError('acquired native manifest observation review missing')
    source,_=checkpoint_location(Path(root),Path(repo),row.get('native_ref') or row['dataset_ref'])
    if str(source)!=row.get('source') or artifact_identity(source)!=(row.get('identity_field'),row.get('identity')):
        raise ValueError('acquired registered bytes changed')
    verify(Path(root),Path(repo),row.get('asset') or {})
    return {**row,'record_ref':ref,'record_sha256':sha}


def _arm_data_bindings(root,repo,run_id,measured,*,fixed,acquired_version_id,training_version_id):
    """Read existing assets; no registry write, version selection or new approval."""
    from .data_versions import resolve
    if fixed and acquired_version_id:
        raise ValueError('fixed-data controls cannot claim newly acquired episodes')
    if measured.get('data_selection') is not None:
        from .completed_evaluation_recovery import verified_training_data
        selected,actual_usage=verified_training_data(Path(root),run_id,measured)
        matches=[b for b in selected if b.get('id')==_id(training_version_id)]
        if not matches or actual_usage!=measured.get('data_consumption'):
            raise ValueError('requested arm differs from the original native training binding/consumption')
        training=matches[0]  # Its real native setting key remains unchanged.
        acquired=(_acquired_version(root,repo,acquired_version_id) if not fixed else None)
        bindings=([acquired] if acquired else [])+[training]
    else:
        # Legacy v1 records predate selected-version proofs. This compatibility
        # path cannot create a retrospective binding for newer measurements.
        selected = {'training': 'data-version:' + _id(training_version_id)}
        if not fixed:
            selected = {'acquired':'data-version:' + _id(acquired_version_id), **selected}
        _, bindings = resolve(Path(root), Path(repo), selected)
        training = bindings[-1]
        acquired = bindings[0] if not fixed else None
    return bindings,training,acquired


def record_arm(root, repo, run_id, *, campaign_id, arm, seed, measurement_label,
               acquired_version_id='', training_version_id):
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
    fixed = arm in {'fixed_official', 'fixed_continuation'}
    bindings,training,acquired=_arm_data_bindings(root,repo,run_id,measured,fixed=fixed,
        acquired_version_id=acquired_version_id,training_version_id=training_version_id)
    # Parent records are recursively byte-verified through the native asset proof.
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
    if not fixed and not has_parent(training, acquired['id']):
        raise ValueError('acquired data is not a parent of the measured training version')
    asset = (acquired or {}).get('asset') or {}
    if not fixed and (asset.get('kind') not in ('collected', 'processed', 'synthetic') or not asset.get('producer')):
        raise ValueError('new data requires sealed native collection/processing evidence')
    def collected_ancestor(row, depth=0):
        if depth > 16:
            raise ValueError('campaign lineage is too deep')
        metadata = row.get('asset') or {}
        producer = metadata.get('producer') or {}
        if metadata.get('kind') in ('collected', 'synthetic') and producer:
            if producer.get('scope') == 'budget_parent':
                raise ValueError('parent-run collection is a shared starting asset, not new acquisition in this campaign')
            from .evidence_store import read_attempt_evidence
            evidence = read_attempt_evidence(Path(root), producer['evidence_id'], limit=1)
            if (evidence.get('native_context') or {}).get('stage') == 'collect':
                return True
        return any(collected_ancestor(_read(root, 'data_versions/' + _id(parent['id']) + '.json')[0], depth+1)
                   for parent in metadata.get('parents', []))
    if not fixed and not collected_ancestor(acquired):
        raise ValueError('acquisition arm needs a native collect ancestor; processing old data is a separate selection track')
    consumed = measured.get('data_consumption') or {}
    if consumed.get('status') != 'verified' or training['id'] not in consumed.get('versions', []):
        raise ValueError('the evaluated training must have consumed the registered version')
    profile = asset.get('profile') or {}
    count = (profile.get('counts') or {}).get('episodes') if profile.get('scope') == 'full' else None
    if plan['schema_version'] == 2:
        # A selection preview does not rewrite a raw version. Do not report raw
        # invalid/unselected episode population as matched accepted training data.
        if ((profile.get('counts') or {}).get('invalid_episodes', 0) > 0 or
                (profile.get('funnel') or {}).get('format_valid_episodes', count) != count or
                (profile.get('selection') or {}).get('counts', {}).get('selected', count) != count):
            count = None
        actual = [row for row in consumed.get('exposure', []) if row.get('data_version_id') == training['id']]
        if not actual or any(row.get('consumption_stage') != 'training_batch' for row in actual):
            raise ValueError('v2 data comparison requires actual training_batch consumption, not prefetched sample reads')
    row = {'schema_version': plan['schema_version'], 'at': now(), 'campaign_id': identity, 'run_id': run_id,
           'arm': arm, 'seed': seed, 'measurement_label': measurement_label,
           'measurement_witness': witness, 'plan_sha256': plan_sha,
           'data_witnesses': [{'ref': b['record_ref'], 'sha256': b['record_sha256']} for b in bindings],
           'acquired_version_id': acquired['id'] if acquired else None, 'training_version_id': training['id'],
           'reported_full_episodes': count, 'matches_requested_count': count == plan['comparison']['usable_episodes'],
           'metric_value': measured['metric_value'], 'status': 'consumed_and_measured',
           'authority': 'Observed development arm; seed/recipe equivalence, effective mixture and new-data count need independent audit'}
    if plan['schema_version'] == 2:
        row.update(method=arm, budget_axis=plan['comparison']['budget_axis'],
            quality_floor=plan['comparison']['quality_floor'],
            count_status=('fixed_data_control' if fixed else 'matched' if row['matches_requested_count'] else 'incomplete_or_unknown'),
            comparison_ready=False,  # A registry is not independent scientific approval.
            acquisition_funnel=profile.get('funnel') or {},
            unknown_provenance=profile.get('manifest') or {},
            rule_allocation_ref=(f'data_campaigns/{identity}/plan.json#rule_allocation' if arm == 'rule_targeted' else None))
    _save(root, f'data_campaigns/{identity}/{arm}-{seed}.json', row)
    return row


def execution_protocols(root, run_id):
    """Full frozen declarations for LLM review, never an arm-compliance verdict."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError('unsafe data-study output root')
    root = root.resolve()
    directory = _path(root, 'data_campaigns')
    if not directory.exists():
        return []
    paths = sorted(directory.glob('*/plan.json'))
    if len(paths) > 256:
        raise ValueError('data-study protocol catalog exceeds its bounded scope')
    result = []
    for path in paths:
        checked = _path(root, path.relative_to(root))
        if checked.stat().st_size > 65536:
            raise ValueError('data-study plan exceeds bounded review')
        raw = checked.read_bytes()
        if len(raw) > 65536:
            raise ValueError('data-study plan exceeds bounded review')
        plan, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
        if not isinstance(plan, dict):
            raise ValueError('expected a registered data-study object')
        if plan.get('schema_version') != 2 or plan.get('run_id') != run_id:
            continue
        if (plan.get('status') != 'hypothesis_registered' or
                _id(plan.get('id')) != path.parent.name or
                not isinstance(plan.get('comparison'), dict)):
            raise ValueError('registered data-study protocol identity is invalid')
        _check_witness(root, plan['development_witness'])
        if digest(checked) != sha:
            raise ValueError('data-study plan changed during review projection')
        result.append({'id': plan['id'], 'plan_ref': path.relative_to(root).as_posix(),
            'plan_sha256': sha, 'hypothesis': plan.get('hypothesis'),
            'development_label': plan.get('development_label'),
            'comparison': plan['comparison'],
            'authority': 'frozen research requirements, not verified native configuration, '
                         'arm equivalence, permission to alter the plan or a performance claim'})
        if len(result) > 8 or len(json.dumps(result, ensure_ascii=False).encode()) > 32768:
            raise ValueError('complete data-study requirements exceed bounded review; '
                             'do not silently omit frozen factors or spend on an unchanged retry')
    return result


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
                    for witness in ([row['measurement_witness'], *row['data_witnesses']]
                                    if row.get('status') == 'consumed_and_measured' else row.get('boundary_witnesses', [])):
                        _check_witness(root, witness)
                    observations.append({**row, 'record_ref': ref})
            result.append({'id': plan['id'], 'status': plan['status'],
                           'schema_version':plan['schema_version'],
                           **({'rule_allocation':plan['rule_allocation']} if plan.get('rule_allocation') else {}),
                           'hypothesis': plan['hypothesis'][:400],
                           'development_label': plan['development_label'],
                           'starting_score': plan['starting_score'],
                           'comparison': {**plan['comparison'],
                               'training_recipe': plan['comparison']['training_recipe'][:400],
                               'compute_accounting': plan['comparison']['compute_accounting'][:400]},
                           'plan_ref': str(path.relative_to(root)),
                           'observations': [{key: row[key] for key in (
                               'arm','seed','measurement_label','acquired_version_id','training_version_id',
                               'reported_full_episodes','matches_requested_count','metric_value','record_ref',
                               'count_status','acquisition_funnel','unknown_provenance','status','reason',
                               'comparison_ready','authority') if key in row}
                               for row in observations],
                           'observation_authority': 'registered measurements are observations; '
                               'not independent verification of the frozen recipe, initialization or quality',
                           'fairness_status': 'requires_independent_audit'})
        except (OSError, ValueError, TypeError, KeyError):
            result.append({'id': path.parent.name, 'status': 'evidence_unavailable', 'observations': []})
    return result
