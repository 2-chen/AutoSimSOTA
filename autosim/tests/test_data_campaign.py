"""Campaign integration contracts; fixture metrics are not robot experiments."""
import json

import pytest

from autosim.research import data_campaign, data_versions, recorder, run_record
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.experiment_bundle import artifact_identity
from tests.test_data_lineage import processed, producer, REVIEW
from tests.test_data_versions import registered
from tests.test_prepare import preparation


@pytest.fixture(autouse=True)
def receipt_fixture(monkeypatch):
    # These registry fixtures do not implement native execution. Full receipt
    # fidelity is independently exercised by the native measurement integration.
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
                        lambda *args, **kwargs: {'status':'consistent'})


def test_native_scored_description_is_not_a_boolean_gate(tmp_path):
    row = score(tmp_path)
    row['scored'] = 'a checkpoint this run trained'
    atomic_json(tmp_path/'research/derived/measurements/baseline.json', row)
    assert data_campaign.register(tmp_path,'derived',plan())['status']=='hypothesis_registered'


def test_claimed_score_without_consistent_receipts_is_rejected(tmp_path,monkeypatch):
    score(tmp_path)
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
                        lambda *args, **kwargs: {'status':'unverifiable'})
    with pytest.raises(ValueError, match='independently consistent'):
        data_campaign.register(tmp_path,'derived',plan())


@pytest.mark.parametrize('change',[{'confirmation':True},{'baseline_origin':'released_reference'}])
def test_heldout_and_reference_scores_cannot_guide_acquisition(tmp_path, change):
    row = {**score(tmp_path), **change}
    atomic_json(tmp_path/'research/derived/measurements/baseline.json',row)
    with pytest.raises(ValueError, match='scored development'):
        data_campaign.register(tmp_path,'derived',plan())


def score(root, label='baseline', value=.2, versions=None):
    row = {'label': label, 'ok': True, 'scored': True, 'metric_value': value,
           'policy_consumption': {'status': 'verified'},
           'rollout_evidence': {'status': 'verified'}, 'metric_lineage': {'status': 'verified'},
           'data_consumption': {'status': 'verified' if versions else 'not_required', 'versions': versions or []}}
    atomic_json(root/f'research/derived/measurements/{label}.json', row)
    return row


def plan():
    return {'id': 'failure_slice', 'development_label': 'baseline',
            'hypothesis': '接近目标后的失误可能与覆盖不足有关，尚待验证。',
            'acquisition': '先验证原生专家在训练场景的成功率，再做定向和随机采集。',
            'selection': '保留完整轨迹，质量底线一致；原始数据不覆盖。',
            'comparison': {'arms': ['targeted', 'random'], 'usable_episodes': 2,
                           'training_recipe': '相同原生模型、官方数据、初始化和实际完成更新数。',
                           'seeds': [0, 1], 'compute_accounting': '包括失败采集、转换、训练、rollout及模型调用。'}}


def test_register_is_immutable_hypothesis_not_an_experiment(tmp_path):
    score(tmp_path)
    record = data_campaign.register(tmp_path, 'derived', plan())
    assert record['status'] == 'hypothesis_registered'
    assert not data_campaign.view(tmp_path, 'derived')[0]['observations']
    with pytest.raises(ValueError, match='immutable'):
        data_campaign.register(tmp_path, 'derived', plan())
    assert not (tmp_path/'jobs').exists()


@pytest.mark.parametrize('change', ['heldout', 'missing_score', 'unscored', 'wrong_label',
    'policy', 'rollout', 'metric', 'nan', 'unsafe_id', 'no_random', 'duplicate_arm',
    'bool_count', 'zero_count', 'duplicate_seed', 'bool_seed', 'unknown_field', 'oversize'])
def test_campaign_rejects_invalid_provenance_and_comparisons(tmp_path, change):
    row = score(tmp_path)
    proposal = plan()
    if change == 'heldout':
        atomic_json(tmp_path/'research/derived/confirmation_attempts/a.json', {'status': 'failed'})
    elif change == 'missing_score':
        proposal['development_label'] = 'round_100'
    elif change == 'unscored': row['scored'] = False
    elif change == 'wrong_label': row['label'] = 'other'
    elif change in {'policy', 'rollout', 'metric'}:
        row[{'policy': 'policy_consumption', 'rollout': 'rollout_evidence', 'metric': 'metric_lineage'}[change]]['status'] = 'unverified'
    elif change == 'nan': row['metric_value'] = float('nan')
    elif change == 'unsafe_id': proposal['id'] = '../outside'
    elif change == 'no_random': proposal['comparison']['arms'] = ['targeted']
    elif change == 'duplicate_arm': proposal['comparison']['arms'].append('random')
    elif change == 'bool_count': proposal['comparison']['usable_episodes'] = True
    elif change == 'zero_count': proposal['comparison']['usable_episodes'] = 0
    elif change == 'duplicate_seed': proposal['comparison']['seeds'] = [0, 0]
    elif change == 'bool_seed': proposal['comparison']['seeds'] = [True]
    elif change == 'unknown_field': proposal['test_failures'] = []
    else: proposal['hypothesis'] = 'x'*2001
    if change == 'nan':
        # Malformed external input bypasses the production writer's own finite guard.
        (tmp_path/'research/derived/measurements/baseline.json').write_text(json.dumps(row))
    else:
        atomic_json(tmp_path/'research/derived/measurements/baseline.json', row)
    with pytest.raises((ValueError, OSError)):
        data_campaign.register(tmp_path, 'derived', proposal)
    assert not (tmp_path/'data_campaigns/failure_slice/plan.json').exists()


def arm_fixture(tmp_path):
    repo, output, raw, _ = registered(tmp_path)
    ident, _ = producer(output, raw)
    parent = data_versions.register(output, repo, dict(id='collected',dataset_ref='data.json',
        source_refs=[],reason='fixture native collector',asset=dict(kind='collected',producer_evidence_id=ident)), REVIEW)
    filtered = output/'selected.json'
    filtered.write_text('[1,3]')
    field, value = artifact_identity(filtered)
    profile = dict(identity_field=field,identity=value,scope='full',format='fixture episodes',
        training_only=True,counts=dict(episodes=2,samples=2),
        parents=[dict(id='collected',identity_field=parent['identity_field'],identity=parent['identity'],
                      record_sha256=digest(output/'data_versions/collected.json'))],
        recipe=dict(operation='select complete episodes',parameters=dict(selected_ids=[0,2]),
                    preserves_episode_groups=True))
    ident, _ = producer(output, filtered, stage='prepare_data', profile=profile, ident='c'*32)
    data_versions.register(output, repo, dict(id='new',dataset_ref='selected.json',source_refs=[],
        reason='fixture selection',asset=dict(kind='processed',parents=['collected'],producer_evidence_id=ident)), REVIEW)
    score(output)
    data_campaign.register(output, 'derived', plan())
    score(output, 'round_1', .4, ['new'])
    kwargs = dict(campaign_id='failure_slice', arm='targeted', seed=0,
                  measurement_label='round_1', acquired_version_id='new', training_version_id='new')
    return repo, output, raw, filtered, kwargs


def test_arm_links_native_producer_actual_consumption_and_score(tmp_path):
    repo, output, raw, _, kwargs = arm_fixture(tmp_path)
    before = raw.read_bytes()
    result = data_campaign.record_arm(output, repo, 'derived', **kwargs)
    assert result['matches_requested_count'] and result['metric_value'] == .4
    assert raw.read_bytes() == before
    view = data_campaign.view(output, 'derived')[0]
    assert view['fairness_status'] == 'requires_independent_audit'
    assert len(view['observations']) == 1
    with pytest.raises(ValueError):
        data_campaign.record_arm(output, repo, 'derived', **kwargs)
    with pytest.raises(ValueError, match='two campaign arms'):
        data_campaign.record_arm(output, repo, 'derived', **{**kwargs, 'arm': 'random'})


@pytest.mark.parametrize('change', ['heldout', 'different_run', 'unknown_arm', 'unknown_seed',
    'baseline_reused', 'not_consumed', 'wrong_version', 'changed_measurement', 'changed_data',
    'wrong_parent', 'changed_parent', 'missing_producer'])
def test_arm_does_not_turn_intent_into_observed_improvement(tmp_path, change):
    repo, output, raw, filtered, kwargs = arm_fixture(tmp_path)
    run_id = 'derived'
    if change == 'heldout': atomic_json(output/'research/derived/confirmation_attempts/a.json', {})
    elif change == 'different_run': run_id = 'other'
    elif change == 'unknown_arm': kwargs['arm'] = 'invented'
    elif change == 'unknown_seed': kwargs['seed'] = 7
    elif change == 'baseline_reused': kwargs['measurement_label'] = 'baseline'
    elif change == 'not_consumed': score(output, 'round_1', .4)
    elif change == 'wrong_version': score(output, 'round_1', .4, ['extra'])
    elif change == 'changed_measurement': score(output, value=.8)
    elif change == 'changed_data': filtered.write_text('[2]')
    elif change == 'wrong_parent': kwargs['training_version_id'] = 'extra'
    elif change == 'changed_parent': raw.write_text('[9]')
    else:
        card = read_json(output/'data_versions/new.json')
        card['asset'].pop('producer')
        atomic_json(output/'data_versions/new.json', card)
    with pytest.raises((ValueError, OSError, KeyError)):
        data_campaign.record_arm(output, repo, run_id, **kwargs)


def test_projection_withholds_tampered_measurement(tmp_path):
    repo, output, _, _, kwargs = arm_fixture(tmp_path)
    data_campaign.record_arm(output, repo, 'derived', **kwargs)
    score(output, 'round_1', .99, ['new'])
    assert data_campaign.view(output, 'derived')[0]['status'] == 'evidence_unavailable'


def test_processing_existing_data_is_not_new_acquisition(tmp_path):
    repo, output, _, _, _ = processed(tmp_path)
    score(output)
    data_campaign.register(output, 'derived', plan())
    score(output, 'round_1', .4, ['new'])
    with pytest.raises(ValueError, match='collect ancestor'):
        data_campaign.record_arm(output, repo, 'derived', campaign_id='failure_slice',
            arm='targeted', seed=0, measurement_label='round_1',
            acquired_version_id='new', training_version_id='new')


def test_symlink_cannot_redirect_registry(tmp_path):
    score(tmp_path)
    external = tmp_path/'elsewhere'
    external.mkdir()
    (tmp_path/'data_campaigns').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        data_campaign.register(tmp_path, 'derived', plan())
    assert not list(external.iterdir())


def test_main_operations_and_chinese_run_panel(tmp_path):
    real = preparation(tmp_path)
    score(real.output)
    result = real._step_register_data_campaign(campaign=plan())
    assert result['outcome'] == 'registered'
    assert real._data_campaign_view()[0]['id'] == 'failure_slice'
    view = run_record.build_report_view(real.output, 'derived', status='running')
    snapshot = recorder.make_snapshot(real.output, view, {'actions': [], 'plan': {}})
    assert recorder._evidence(snapshot)['data_campaigns']
    document = recorder.render(real.output, snapshot, {})
    assert '从失败到数据的实验' in document and '等量随机采集' in document
    assert '未产出' in document and '待执行' in document
    assert '0.000' not in '\n'.join(recorder.campaign_panel(real.output, snapshot))


def test_campaign_chart_uses_only_observed_scores(tmp_path):
    score(tmp_path)
    data_campaign.register(tmp_path, 'derived', plan())
    campaign = data_campaign.view(tmp_path, 'derived')[0]
    campaign['observations'] = [dict(arm=arm,seed=0,reported_full_episodes=2,
        training_version_id=arm,metric_value=value,record_ref='fixture.json')
        for arm, value in [('targeted', .4), ('random', .3)]]
    panel = recorder.campaign_panel(tmp_path, {'data_campaigns': [campaign], 'revision': 'a'*32})
    assert any('![数据实验对比]' in line for line in panel)
    assert len(list((tmp_path/'report/charts').glob('*.svg'))) == 1
