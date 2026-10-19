"""General native-control and data-research lifecycle contracts."""
import copy
import hashlib
import json

import pytest

from autosim.research import data_acquisition, data_campaign, data_versions, recorder
from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.evidence_store import capture_attempt_evidence
from tests.test_data_campaign import plan, score, arm_fixture
from tests.test_prepare import preparation

BUILDER = """def command(inputs):
    return ['collector', str(inputs['episodes']), str(inputs['seed']),
            str(inputs['collection_controls']['scene.range'])]
"""


def settings():
    return dict(episodes=3,seed=1,controls={'scene.range':[-.1,.1]},
                control_source_refs=[{'ref':'collector.py','quote':'reset reads the training scene range'}])


def test_controls_reach_native_settings_without_changing_evaluation():
    value = data_acquisition.collection_inputs(settings(), BUILDER)
    assert value == {'episodes':3,'seed':1,'collection_controls':{'scene.range':[-.1,.1]}}
    assert 'control_source_refs' not in value


def test_control_input_reaches_actual_builder_and_never_train_or_evaluator(tmp_path):
    from tests.test_derived_research import research
    from autosim.research.execution_derive import blank_inputs
    real = research(tmp_path)
    settings_ = data_acquisition.collection_inputs(settings(), BUILDER)
    function = {}; exec(BUILDER, function)
    actual = real._inputs('collect', settings=settings_)
    assert function['command'](actual)[-1] == '[-0.1, 0.1]'
    assert 'collection_controls' not in actual['settings']
    assert real._inputs('evaluate', settings=settings_)['collection_controls'] == {}
    assert real._inputs('train', settings=settings_)['collection_controls'] == {}
    assert blank_inputs()['collection_controls'] == {}


@pytest.mark.parametrize('fault', ['unused','missing_refs','absolute','nan','unknown_field','bool_episodes'])
def test_unbound_controls_refused_before_execution_or_model_cost(fault):
    value = settings(); source = BUILDER
    if fault == 'unused': source = "def command(inputs): return ['collector', str(inputs['episodes'])]"
    elif fault == 'missing_refs': value.pop('control_source_refs')
    elif fault == 'absolute': value['controls']={'/private/root':1}
    elif fault == 'nan': value['controls']={'scene.range':float('nan')}
    elif fault == 'unknown_field': value['test_seeds']=[0]
    else:value['episodes']=True
    with pytest.raises(ValueError):data_acquisition.collection_inputs(value,source)


def test_source_control_contract_pins_values_builder_and_current_source(tmp_path):
    from autosim.research.baseline_reference import source_evidence
    repo=tmp_path/'repo';repo.mkdir();output=tmp_path/'out';output.mkdir()
    (repo/'collector.py').write_text('reset reads the training scene range\n')
    refs=source_evidence(repo,settings()['control_source_refs'],output=output)
    contract=dict(controls=settings()['controls'],source_refs=refs,
        builder_sha256=hashlib.sha256(BUILDER.encode()).hexdigest(),
        controls_sha256=object_digest(settings()['controls']),
        review={k:True for k in ('allowed_training_controls','routing_supported','label_supply_supported')})
    inputs=data_acquisition.collection_inputs(settings(),BUILDER)
    data_acquisition.verify_control_contract(output=output,repo=repo,source=BUILDER,settings=inputs,contract=contract)
    for changed in ({**inputs,'collection_controls':{'scene.range':[2,3]}},inputs):
        if changed is inputs:(repo/'collector.py').write_text('source changed')
        with pytest.raises(ValueError):
            data_acquisition.verify_control_contract(output=output,repo=repo,source=BUILDER,settings=changed,contract=contract)


def v2():
    proposal=plan();proposal['schema_version']=2
    proposal['comparison'].update(arms=['fixed_continuation','random_quality','agent_targeted'],
        budget_axis='accepted_episodes',budget_limit=2,quality_floor='same native-valid complete episodes',
        frozen_factors={'initialization':'same','updates':10,'batch':2,'evaluation':'same'},rule=None)
    return proposal


@pytest.fixture(autouse=True)
def receipts(monkeypatch):
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',lambda *a,**k:{'status':'consistent'})


def test_v2_frozen_plan_supports_methods_and_keeps_budget_axes_distinct(tmp_path):
    score(tmp_path)
    row=data_campaign.register(tmp_path,'derived',v2())
    assert row['schema_version']==2
    assert len(data_campaign.pending(tmp_path,'derived'))==6
    changed=v2();changed['id']='compute';changed['comparison'].update(
        budget_axis='total_compute',budget_limit={'gpu_seconds':300,'model_usd':1})
    assert data_campaign.register(tmp_path,'derived',changed)['comparison']['budget_limit']['model_usd']==1


@pytest.mark.parametrize('change', ['conflated_count','bool_limit','unscoped_compute','missing_factors','rule_without_proof'])
def test_v2_rejects_ambiguous_or_unverified_comparisons(tmp_path,change):
    score(tmp_path);value=v2()
    if change=='conflated_count':value['comparison']['budget_limit']=3
    elif change=='bool_limit':value['comparison']['budget_limit']=True
    elif change=='unscoped_compute':value['comparison'].update(budget_axis='total_compute',budget_limit=100)
    elif change=='missing_factors':value['comparison']['frozen_factors']={}
    else:value['comparison']['arms'].append('rule_targeted')
    with pytest.raises(ValueError):data_campaign.register(tmp_path,'derived',value)


def test_rule_counts_are_exact_finite_and_not_chosen_by_an_llm():
    strata=[dict(id='a',failures=3,evaluated=5),dict(id='b',failures=1,evaluated=5)]
    assert data_campaign.rule_allocation(strata,8)['allocation']=={'a':6,'b':2}
    assert sum(data_campaign.rule_allocation(strata,3)['allocation'].values())==3
    with pytest.raises(ValueError):data_campaign.rule_allocation([dict(id='a',failures=0,evaluated=2)],3)
    with pytest.raises(ValueError):data_campaign.rule_allocation([dict(id='a',failures=True,evaluated=2)],3)


def test_registered_rule_allocation_requires_native_sealed_failure_counts(tmp_path):
    repo=tmp_path/'checkout';repo.mkdir();(repo/'stats.py').write_text('counts = native_development_failures\n')
    score(tmp_path)
    ident='d'*32;log=tmp_path/'native.log'
    log.write_text(data_campaign.RULE_MARKER+json.dumps([dict(id='a',failures=3,evaluated=5),dict(id='b',failures=1,evaluated=5)]))
    capture_attempt_evidence(tmp_path,attempt_id=ident,log=log,receipt_ref='native.json',
                            status='completed',returncode=0,termination_reason='cpu_native_diagnostic')
    value=v2();value['comparison']['arms'].append('rule_targeted')
    value['comparison']['rule']=dict(algorithm='failure_frequency_v1',strata_evidence_id=ident,
        source_refs=[dict(ref='stats.py',quote='counts = native_development_failures')])
    row=data_campaign.register(tmp_path,'derived',value,repo=repo)
    assert sum(row['rule_allocation']['allocation'].values())==2
    assert row['rule_allocation']['strata_evidence_id']==ident


def test_fixed_data_control_is_not_relabelled_new_acquisition(tmp_path):
    repo,output,_,_,kwargs=arm_fixture(tmp_path)
    value=v2();value['id']='v2'
    data_campaign.register(output,'derived',value,repo=repo)
    measured=score(output,'round_2',.3,['new'])
    measured['data_consumption']['exposure']=[dict(data_version_id='new',samples_read=4,consumption_stage='training_batch')]
    atomic_json(output/'research/derived/measurements/round_2.json',measured)
    row=data_campaign.record_arm(output,repo,'derived',campaign_id='v2',arm='fixed_continuation',seed=0,
        measurement_label='round_2',training_version_id='new')
    assert row['acquired_version_id'] is None and row['count_status']=='fixed_data_control'
    assert row['reported_full_episodes'] is None


def test_prefetch_only_witness_does_not_complete_a_v2_data_comparison(tmp_path):
    repo,output,_,_,kwargs=arm_fixture(tmp_path)
    value=v2();value['id']='v2';data_campaign.register(output,'derived',value,repo=repo)
    with pytest.raises(ValueError,match='training_batch'):
        data_campaign.record_arm(output,repo,'derived',campaign_id='v2',arm='agent_targeted',seed=0,
            measurement_label='round_1',acquired_version_id='new',training_version_id='new')


def test_raw_invalid_population_is_not_matched_accepted_data(tmp_path):
    repo,output,_,_,kwargs=arm_fixture(tmp_path)
    path=output/'data_versions/new.json';card=read_json(path)
    card['asset']['profile']['counts']['invalid_episodes']=1
    atomic_json(path,card)
    value=v2();value['id']='v2';data_campaign.register(output,'derived',value,repo=repo)
    measured=score(output,'round_2',.3,['new'])
    measured['data_consumption']['exposure']=[dict(data_version_id='new',samples_read=4,consumption_stage='training_batch')]
    atomic_json(output/'research/derived/measurements/round_2.json',measured)
    row=data_campaign.record_arm(output,repo,'derived',campaign_id='v2',arm='agent_targeted',seed=0,
        measurement_label='round_2',acquired_version_id='new',training_version_id='new')
    assert row['reported_full_episodes'] is None and not row['matches_requested_count']
    assert row['count_status']=='incomplete_or_unknown'


def test_unfinished_declared_data_work_cannot_be_closed_by_confirmation(tmp_path):
    real=preparation(tmp_path);real.base_settings['_confirm']=True
    score(real.output);data_campaign.register(real.output,'derived',v2(),repo=real.repo)
    result=real._step_confirm_best()
    assert result['repair_owner']=='data' and result['failure_category']=='data_campaign_obligations'
    assert not (real.output/'research/derived/confirmation_attempts').exists()


def test_reviewed_boundary_is_not_a_zero_score_and_keeps_original_evidence(tmp_path):
    score(tmp_path);data_campaign.register(tmp_path,'derived',v2())
    ident='c'*32;log=tmp_path/'native.log';log.write_text('native source lacks requested state-expert access')
    capture_attempt_evidence(tmp_path,attempt_id=ident,log=log,receipt_ref='native.json',status='completed',
        returncode=0,termination_reason='cpu_native_diagnostic')
    boundary=dict(status='unsupported',reason='source-backed acquisition capability boundary',
        evidence_refs=[ident],source_refs=[dict(ref='task.py',quote='no state expert')])
    before=digest(tmp_path/f'evidence/{ident}.json')
    row=data_campaign.record_boundary(tmp_path,'derived',campaign_id='failure_slice',arm='agent_targeted',seed=0,
        boundary=boundary,review=dict(boundary_evidence_supported=True,no_safe_next_data_operation=True))
    assert row['metric_value'] is None and len(data_campaign.pending(tmp_path,'derived'))==5
    assert digest(tmp_path/f'evidence/{ident}.json')==before
    campaign=data_campaign.view(tmp_path,'derived')[0]
    panel='\n'.join(recorder.campaign_panel(tmp_path,{'revision':'a'*32,'data_campaigns':[campaign]}))
    assert '能力边界' in panel and '不是零分' in panel
    assert '-campaign.svg' not in panel
