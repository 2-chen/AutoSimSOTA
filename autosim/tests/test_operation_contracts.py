import pytest

from autosim.research.operation_contracts import schema, validate
from tests.test_prepare import preparation


def test_public_schema_is_generated_from_handler_and_rejects_type_errors(tmp_path):
    real=preparation(tmp_path)
    held=schema(real._step_submit_native_job)
    assert 'idea_label' in held['properties']
    with pytest.raises(ValueError):validate(held,dict(stage='train',window_seconds=True,reason='test'))
    with pytest.raises(ValueError):validate(held,dict(stage='train',window_seconds=4,reason='x'*1001))
    validate(held,dict(stage='train',window_seconds=4,reason='test',idea_label='audited'))


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


@pytest.mark.parametrize('defect', ['risk', 'fields', 'same_file', 'missing_source', 'touches'])
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
