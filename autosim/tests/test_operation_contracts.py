from types import SimpleNamespace

import pytest

from autosim.research.operation_contracts import schema, validate
from tests.test_prepare import preparation


def test_handoff_modes_share_an_exposed_cross_field_contract(tmp_path):
    real = preparation(tmp_path)
    contract = schema(real._step_correct_policy_handoff)
    assert 'completed EVALUATE' in contract['description']
    assert 'training attempt' in contract['properties']['job_id']['description']
    assert 'if' in contract and 'then' in contract
    for body in [
            {'label':'round_1','checkpoint_ref':'producer/complete_bundle'},
            {'label':'round_1','checkpoint_ref':'producer/complete_bundle','job_id':''},
            {'label':'round_1','job_id':'a'*32},
            {'label':'round_1','job_id':'a'*32,'checkpoint_ref':''},
            {'label':'round_1'}]:
        validate(contract, body)
    with pytest.raises(ValueError, match='checkpoint_ref'):
        validate(contract, {'label':'round_1','checkpoint_ref':'producer/complete_bundle','job_id':'a'*32})


def test_handoff_conflicting_modes_fail_before_research_controller(tmp_path, monkeypatch):
    real = preparation(tmp_path)
    monkeypatch.setattr(real, '_research_controller',
        lambda: pytest.fail('invalid argument combination must not prepare a controller'))
    with pytest.raises(ValueError, match='mutually exclusive'):
        real._step_correct_policy_handoff(label='round_1', checkpoint_ref='producer/bundle', job_id='a'*32)


def test_campaign_description_exposes_current_version_and_witness_limits():
    from autosim.research.prepare import OPERATIONS
    text=OPERATIONS['register_data_campaign']
    assert 'schema_version:2' in text and 'budget_axis' in text and 'quality_floor' in text
    assert 'omitting schema_version selects legacy v1' in text
    assert 'sample_read is NOT training_batch' in text


def test_boundary_contract_requires_real_source_and_bounded_evidence_ids(tmp_path):
    real=preparation(tmp_path)
    held=schema(real._step_record_data_campaign_arm)
    body={'campaign_id':'study','arm':'random_quality','seed':1000,'boundary':{
        'status':'budget_limited','reason':'bounded source/native evidence',
        'evidence_refs':['a'*32],'source_refs':[{'ref':'train.py:1','quote':'train'}]}}
    validate(held,body)
    body['boundary']['source_refs']=[]
    with pytest.raises(ValueError,match='source_refs'):validate(held,body)
    body['boundary']['source_refs']=[{'ref':'train.py:1','quote':'train'}]
    body['boundary']['evidence_refs']=['native_jobs/not-evidence']
    with pytest.raises(ValueError,match='evidence_refs'):validate(held,body)


def test_boundary_refusal_identifies_exact_missing_evidence_before_any_model(tmp_path,monkeypatch):
    from autosim.research.prepare import Preparation
    monkeypatch.setattr('autosim.research.baseline_reference.source_evidence',lambda *a,**kw:[])
    def read(output,identity,**kwargs):
        if identity=='a'*32:return {'status':'completed'}
        raise ValueError('evidence record is missing or unsafe')
    monkeypatch.setattr('autosim.research.evidence_store.read_attempt_evidence',read)
    real=SimpleNamespace(repo=tmp_path,output=tmp_path,run_id='derived',
                         _business_refusal=lambda **kwargs:kwargs)
    result=Preparation._step_record_data_campaign_arm(real,campaign_id='study',
        arm='random_quality',seed=1000,boundary={'status':'budget_limited','reason':'fixture',
            'evidence_refs':['a'*32,'b'*32],'source_refs':[{'ref':'train.py','quote':'train'}]})
    assert 'evidence_refs[1]='+'b'*32 in result['because']
    assert result['context']['scientific_review_started'] is False


def test_native_job_description_discloses_enforced_defaults_and_frozen_probe_scope():
    from autosim.research.prepare import OPERATIONS
    text=OPERATIONS['submit_native_job']
    assert '2 logical CPUs and 4096 MiB' in text and 'CPU affinity/thread caps' in text
    assert 'original episodes and evaluation seed' in text
    assert 'even for a diagnostic job' in text and 'ask Resource' in text


def test_monitor_distinguishes_new_native_failures_from_missing_postconditions():
    from autosim.research.prepare import MONITOR_SYSTEM
    assert 'CURRENT failed process/evidence ID' in MONITOR_SYSTEM
    assert 'not proof of its cause' in MONITOR_SYSTEM
    assert 'do not invent a stack' in MONITOR_SYSTEM and 'short string' in MONITOR_SYSTEM


def test_environment_reason_limit_is_exposed_before_paid_repair_or_execution(tmp_path):
    real=preparation(tmp_path)
    contract=schema(real._step_build_the_environment)
    with pytest.raises(ValueError,match='environment_selection_reason.*maximum is 2000'):
        validate(contract,{'environment_selection_reason':'x'*2677,'max_operations':2})
    validate(contract,{'environment_selection_reason':'continue the accepted cursor','max_operations':2})


def test_public_schema_is_generated_from_handler_and_rejects_type_errors(tmp_path):
    real=preparation(tmp_path)
    held=schema(real._step_submit_native_job)
    assert 'idea_label' in held['properties']
    assert {'checkpoint_ref','evaluation_settings'} <= set(held['properties'])
    validate(held,dict(stage='evaluate',window_seconds=100,reason='native probe',
        checkpoint_ref='producer/policy',evaluation_settings={'episodes':7,'seed':19}))
    with pytest.raises(ValueError):validate(held,dict(stage='train',window_seconds=True,reason='test'))
    with pytest.raises(ValueError):validate(held,dict(stage='train',window_seconds=4,reason='x'*1001))
    validate(held,dict(stage='train',window_seconds=4,reason='test',idea_label='audited'))


@pytest.mark.parametrize('settings',[{'episodes':True},{'episodes':0},{'seed':-1},{'metric':'override'},{}])
def test_detached_eval_nested_contract_rejects_invalid_protocol_controls(tmp_path,settings):
    real=preparation(tmp_path)
    with pytest.raises(ValueError):
        validate(schema(real._step_submit_native_job),dict(stage='evaluate',window_seconds=100,
            reason='development probe',checkpoint_ref='producer/policy',evaluation_settings=settings))


@pytest.mark.parametrize('field,value,expected',[
    ('id','unsafe/path','reference.id'),('source_refs',[{'ref':'a.py'}],'source_refs[0]'),
    ('episodes',True,'episodes'),('tolerance',-1,'tolerance'),('kind','invented','kind')])
def test_nested_reference_contract_identifies_the_exact_bad_field(tmp_path,field,value,expected):
    real=preparation(tmp_path)
    reference={'id':'official','kind':'released_checkpoint','reason':'native release',
        'source_refs':[{'ref':'a.py','quote':'native release'}], 'checkpoint_ref':'policy',
        'release_url':'https://example.org/release','revision':'commit','episodes':100,
        'expected_metric':0.5,'tolerance':0.05}
    validate(schema(real._step_configure_baseline_reference),{'reference':reference})
    reference[field]=value
    with pytest.raises(ValueError) as error:
        validate(schema(real._step_configure_baseline_reference),{'reference':reference})
    assert expected in str(error.value)


def test_nested_reference_requires_release_fields_but_unavailable_does_not(tmp_path):
    real=preparation(tmp_path);contract=schema(real._step_configure_baseline_reference)
    row={'id':'ref','kind':'unavailable','reason':'not locally present',
         'source_refs':[{'ref':'README.md','quote':'release'}]}
    validate(contract,{'reference':row})
    row['kind']='released_checkpoint'
    with pytest.raises(ValueError,match='checkpoint_ref'):validate(contract,{'reference':row})


def test_native_collection_contract_exposes_positive_target(tmp_path):
    real = preparation(tmp_path)
    contract = schema(real._step_submit_native_job)
    base = {'stage': 'collect', 'window_seconds': 100, 'reason': 'registered acquisition'}
    validate(contract, {**base, 'collection_settings': {'episodes': 7, 'seed': 19}})
    for invalid in ({}, {'episodes': 0}, {'episodes': True}, {'episodes': 7, 'seed': -1}):
        with pytest.raises(ValueError):
            validate(contract, {**base, 'collection_settings': invalid})


def test_native_progress_contract_is_optional_and_bounded(tmp_path):
    real = preparation(tmp_path)
    contract = schema(real._step_inspect_native_job)
    validate(contract, {'job_id': 'a'*32})
    validate(contract, {'job_id': 'a'*32, 'log_query': 'native work count'})
    with pytest.raises(ValueError):
        validate(contract, {'job_id': 'a'*32, 'log_query': 'x'*129})


def test_wrong_handle_kind_is_rejected_locally(tmp_path):
    real=preparation(tmp_path)
    with pytest.raises(ValueError,match='typed handle'):
        validate(schema(real._step_inspect_native_job),{'job_id':'a'*64})


def test_nested_data_version_rejects_extra_keys_and_malformed_citations(tmp_path):
    real=preparation(tmp_path);contract=schema(real._step_register_data_version)
    version={'id':'data','dataset_ref':'data/raw','reason':'new expert episodes',
             'source_refs':[{'ref':'train.py','quote':'load(data)'}]}
    validate(contract,{'version':version})
    version['extra']=True
    with pytest.raises(ValueError,match='unexpected fields'):validate(contract,{'version':version})


def test_invalid_job_arguments_do_not_enter_native_handler(tmp_path,monkeypatch):
    real=preparation(tmp_path)
    real.client.supports_main_agent=True
    result=real._do_operation('submit_native_job',stage='train',window_seconds=True,reason='bad')
    assert result['failure_category']=='operation_arguments_invalid'
    assert not (real.output/'native_jobs').exists()


def test_handoff_label_reaches_the_actual_handler_without_a_second_allowlist(
        tmp_path, monkeypatch):
    from autosim.research import handoff_correction
    real = preparation(tmp_path)
    sentinel = object()
    monkeypatch.setattr(real, '_research_controller', lambda: (sentinel, None))
    seen = []

    def correct(research, *, label):
        assert research is sentinel
        seen.append(label)
        return {'ok': True, 'metric_value': 0.2,
                'handoff_correction': {'training_reused': True}}

    monkeypatch.setattr(handoff_correction, 'correct', correct)
    result = real.do('correct_policy_handoff', label='round_4')
    assert seen == ['round_4']
    assert result['outcome'] == 'measurement_recorded'
    assert result['training_reused'] is True


def test_failed_data_consumption_after_handoff_reaches_data_with_sealed_failure_not_successful_train(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from autosim.research import handoff_correction
    from autosim.research.common import atomic_json
    from autosim.research.evidence_store import read_attempt_evidence
    real=preparation(tmp_path)
    root=real.output/'research/derived'
    failure={'label':'round_4_handoff_correction','ok':False,'where':'training_data',
        'metric_value':None,'why':'non-monotonic or duplicate loader counts',
        'train':{'attempt_id':'a'*32,'evidence_id':'a'*32,'returncode':0,'status':'completed'},
        'data_consumption':{'status':'unverified','why':'non-monotonic or duplicate loader counts',
                            'repair_owner':'data','next_operation':'research_task'},
        'handoff_correction':{'training_reused':True}}
    path=root/'measurements/round_4_handoff_correction.json';atomic_json(path,failure)
    before=path.read_bytes()
    research=SimpleNamespace(output=real.output,run_root=root,run_id='derived')
    monkeypatch.setattr(real,'_research_controller',lambda:(research,None))
    monkeypatch.setattr(handoff_correction,'correct',lambda *a,**k:failure)
    result=real.do('correct_policy_handoff',label='round_4')
    assert result['outcome']=='not recovered' and result['metric_value'] is None
    assert result['repair_owner']=='data' and result['failure_category']=='training_data_consumption_unverified'
    assert result['evidence_id']!='a'*32 and result['training_reused'] is True
    proof=read_attempt_evidence(real.output,result['evidence_id'])
    assert proof['status']=='operation_contract_failed'
    assert 'non-monotonic or duplicate loader counts' in proof['text']
    assert 'completed_producer_preserved' in proof['text']
    assert result['native_operation_status']=='not_started' and result['retryable'] is False
    assert path.read_bytes()==before and failure['train']['returncode']==0


@pytest.mark.parametrize('ok,where,usage',[(True,'training_data',{'status':'unverified'}),
    (False,'evaluate',{'status':'unverified'}),(False,'training_data',{'status':'unknown'}),
    (False,'training_data',None)])
def test_data_failure_handoff_does_not_reclassify_valid_zero_or_unknowns(tmp_path,ok,where,usage):
    real=preparation(tmp_path)
    assert real._data_consumer_failure(object(),{'ok':ok,'where':where,
        'metric_value':0.0,'data_consumption':usage},operation='test')=={}
    assert not (real.output/'agent/operation_failures').exists()


def test_corrected_evaluation_failure_reaches_fix_with_native_evidence(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from autosim.research import handoff_correction
    from autosim.research.common import atomic_json
    from autosim.research.evidence_store import capture_attempt_evidence
    real=preparation(tmp_path)
    root=real.output/'research/derived';attempt='e'*32
    ref=f'research/derived/attempts/{attempt}/receipt.json'
    receipt={'run_id':'derived','attempt_id':attempt,'stage':'evaluate',
             'status':'failed','returncode':139,'termination_reason':'nonzero_exit'}
    atomic_json(real.output/ref,receipt)
    log=root/f'evaluate/attempts/{attempt}/output.log';log.parent.mkdir(parents=True)
    log.write_text('Fatal Python error: Segmentation fault\n')
    capture_attempt_evidence(real.output,attempt_id=attempt,log=log,receipt_ref=ref,
        status='failed',returncode=139,termination_reason='nonzero_exit')
    failure={'label':'round_4_handoff_correction','ok':False,'where':'evaluate',
        'metric_value':None,'why':'native evaluator emitted no policy-load event',
        'train':{'attempt_id':'a'*32,'returncode':0,'status':'completed'},
        'evaluate':receipt,'handoff_correction':{'training_reused':True}}
    path=root/'measurements/round_4_handoff_correction.json';atomic_json(path,failure)
    before=path.read_bytes()
    research=SimpleNamespace(output=real.output,run_id='derived',run_root=root)
    monkeypatch.setattr(real,'_research_controller',lambda:(research,None))
    monkeypatch.setattr(handoff_correction,'correct',lambda *a,**k:failure)
    result=real.do('correct_policy_handoff',label='round_4')
    assert result['repair_owner']=='fix'
    assert result['failure_category']=='native_evaluation_failed'
    assert result['evidence_id']==attempt and result['native_operation_status']=='failed'
    assert result['evaluation_attempt_id']==attempt and result['training_reused']
    assert '139' in result['because'] and 'no policy-load event' in result['because']
    assert result['retryable'] is False and result['metric_value'] is None
    assert path.read_bytes()==before


@pytest.mark.parametrize('change',[{'ok':True},{'where':'training_data'},
    {'evaluate':{'attempt_id':'../escape'}},{'evaluate':{'attempt_id':'f'*32}},
    {'evaluate':{'attempt_id':'e'*32,'status':'running','returncode':None}}])
def test_unverified_native_evaluation_never_routes_as_a_sealed_failure(tmp_path,change):
    real=preparation(tmp_path)
    row={'ok':False,'where':'evaluate','evaluate':{'attempt_id':'e'*32,
        'status':'failed','returncode':139},**change}
    assert real._native_evaluation_failure(SimpleNamespace(run_id='derived'),row)=={}


def test_storage_arguments_are_not_silently_lost_before_validation(tmp_path, monkeypatch):
    real = preparation(tmp_path)
    seen = []

    def inspect_storage(*, evidence_id='', offset=0, limit=24,
                        registry_offset=0, registry_limit=24):
        seen.append((evidence_id, offset, limit, registry_offset, registry_limit))
        return {'outcome': 'observed'}

    monkeypatch.setattr(real, '_step_inspect_storage', inspect_storage)
    assert real.do('inspect_storage', evidence_id='a'*32, offset=2, limit=8,
                   registry_offset=3, registry_limit=5)['outcome'] == 'observed'
    assert seen == [('a'*32, 2, 8, 3, 5)]


@pytest.mark.parametrize('arguments', [dict(label='round_4', surprise=True),
                                      dict(label=4), {}])
def test_invalid_handoff_arguments_do_not_execute(tmp_path, monkeypatch, arguments):
    real = preparation(tmp_path)

    def forbidden():
        pytest.fail('invalid request reached the research controller')

    monkeypatch.setattr(real, '_research_controller', forbidden)
    result = real.do('correct_policy_handoff', **arguments)
    assert result['failure_category'] == 'operation_arguments_invalid'


@pytest.mark.parametrize('arguments', [{}, {'label':4}, {'label':'round_4','unexpected':True}])
def test_choose_repairs_arguments_using_same_handler_contract(tmp_path, monkeypatch, arguments):
    real = preparation(tmp_path, choices=[
        {'do':'correct_policy_handoff', 'why':'inspect completed artifact', 'arguments':arguments},
        {'do':'correct_policy_handoff', 'why':'repair rejected label', 'arguments':{'label':'round_4'}}])
    monkeypatch.setattr(real, 'state', lambda: {'available':['correct_policy_handoff'],
                                              'state_revision':0})
    choice = real.choose()
    assert choice['arguments'] == {'label':'round_4'}
    assert len(real.client.seen) == 2
    assert 'argument contract' in real.client.seen[1]
    assert 'no action ran' in real.client.seen[1]


def _idea():
    return dict(label='candidate', granularity='code', mechanism='Test native feature',
                change={'file': 'train.py', 'find': 'N = 1', 'replace': 'N = 2'},
                risk='medium', crosses='none', why='Observed training source',
                evidence=['train.py:1'], touches=['train.py'])


@pytest.mark.parametrize('defect', ['risk', 'fields', 'same_file', 'missing_source', 'touches',
                                 'crosses-prose', 'crosses-list', 'crosses-invented'])
def test_choose_repairs_nested_idea_before_execution(tmp_path, monkeypatch, defect):
    from copy import deepcopy
    (tmp_path/'train.py').write_text('N = 1\nM = 1\n')
    good = _idea()
    bad = deepcopy(good)
    if defect == 'risk':
        bad['risk'] = 'medium because uncertain'
    elif defect == 'fields':
        del bad['evidence']
    elif defect == 'same_file':
        bad['change'] = {'patches': [good['change'],
            {'file':'train.py', 'find':'M = 1', 'replace':'M = 2'}]}
    elif defect == 'missing_source':
        bad['change']['find'] = 'N = 999'
    elif defect == 'crosses-prose':
        bad['crosses'] = 'none (observation-only; no evaluator changes)'
    elif defect == 'crosses-list':
        bad['crosses'] = ['DATA_CONSUMPTION_WITNESS']
    elif defect == 'crosses-invented':
        bad['crosses'] = 'DATA_CONSUMPTION_WITNESS'
    else:
        bad['touches'] = ['other.py']
    real = preparation(tmp_path, choices=[
        {'do':'propose_research_idea', 'why':'propose', 'arguments':{'idea':bad}},
        {'do':'propose_research_idea', 'why':'correct syntax', 'arguments':{'idea':good}}])
    monkeypatch.setattr(real, 'state', lambda: {
        'available':['propose_research_idea'], 'state_revision':0})
    monkeypatch.setattr(real, '_research_controller',
        lambda: pytest.fail('format checking must not create/audit/execute research'))
    result = real.choose()
    assert result['arguments'] == {'idea':good}
    assert len(real.client.seen) == 2
    assert 'argument contract' in real.client.seen[1]
    assert 'repair only the rejected fields' in real.client.seen[1]
    assert (tmp_path/'train.py').read_text() == 'N = 1\nM = 1\n'


@pytest.mark.parametrize('field', ['risk', 'granularity'])
def test_unhashable_idea_enum_is_a_format_error(field):
    from autosim.research.ideas import validate_main_proposal
    candidate = _idea()
    candidate[field] = ['medium']
    with pytest.raises(ValueError, match=field):
        validate_main_proposal(candidate)


@pytest.mark.parametrize('crosses', ['none (no evaluator changes)', ['R1'],
                                   'DATA_CONSUMPTION_WITNESS', '', None])
def test_invalid_redline_syntax_never_records_or_audits_an_idea(tmp_path,crosses):
    from types import SimpleNamespace
    from autosim.research.derived_research import DerivedResearch
    (tmp_path/'train.py').write_text('N = 1\n')
    candidate=_idea();candidate['crosses']=crosses
    real=SimpleNamespace(repo=tmp_path)
    # No library, protocol or audit methods: invalid input must return before
    # touching any of them or creating an immutable rejection under this label.
    result=DerivedResearch.propose_research_idea(real,candidate)
    assert result['status']=='invalid' and 'crosses' in result['why']
    assert sorted(p.name for p in tmp_path.iterdir())==['train.py']


def test_valid_redline_declaration_still_reaches_unchanged_negative_audit(tmp_path):
    from autosim.research.ideas import Idea,validate_main_proposal
    from autosim.research.redlines import audit,from_declaration
    (tmp_path/'train.py').write_text('N = 1\n')
    candidate=_idea();candidate['crosses']='R1'
    validate_main_proposal(candidate,repo=tmp_path)
    result=audit(Idea(**candidate),from_declaration({},repo=tmp_path))
    assert result.verdict=='rejected'
