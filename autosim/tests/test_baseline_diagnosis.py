import pytest

from autosim.research import baseline_diagnosis as triage
from autosim.research.common import atomic_json, read_json
from autosim.research.evidence_store import capture_attempt_evidence


def fixture(tmp_path):
    root = tmp_path/'run'; repo = root/'checkout'; repo.mkdir(parents=True)
    (repo/'train.py').write_text('dataset = load(root)\nfor batch in loader: optimizer.step()\n')
    train, evaluate = 'a'*32, 'b'*32
    for identity in (train, evaluate):
        log = root/'logs'/(identity+'.log'); log.parent.mkdir(exist_ok=True)
        log.write_text('native completed; updates=1024; observed successes=0\n')
        capture_attempt_evidence(root, attempt_id=identity, log=log,
            receipt_ref=f'research/derived/attempts/{identity}/receipt.json',
            status='completed', returncode=0, termination_reason='completed')
    measured = root/'research/derived/measurements/baseline.json'
    atomic_json(measured, {'ok':True,'metric_value':0,'metric':{'name':'success_rate'},
        'metric_reading':{'samples':100},'train':{'ran':True,'attempt_id':train},
        'evaluate':{'evidence_id':evaluate}})
    current = triage.view(root,'derived')
    diagnosis = {'subject_sha256':current['subject_sha256'],'decision':'proceed_local',
        'checks':{'training':'1024 actual updates is a small local study, not released-recipe fidelity.',
            'data':'Loader consumes external data; actual batches/population remain unknown. Register existing data before claiming a data intervention.',
            'policy':'Inspect exact loaded policy and normalization/action mapping, not newest checkpoint.',
            'evaluation':'Native 100-episode development score is retained; no official or SOTA claim.'},
        'source_refs':[{'ref':'train.py','quote':'for batch in loader: optimizer.step()'}],
        'evidence_ids':[train,evaluate],'next_action':'Delegate actual loader profiling to Data; compare published same-family recipe.',
        'reason':'Continue scoped local optimization while preserving uncertainties and original zero score.'}
    return root,repo,measured,diagnosis


def test_triage_is_bound_to_actual_evidence_and_never_rewrites_score(tmp_path):
    root,repo,measured,diagnosis = fixture(tmp_path)
    before = measured.read_bytes()
    current = triage.view(root,'derived')
    assert current['status'] == 'required'
    assert set(current['subject']['problems']) == {'training_population_or_batch_consumption_unverified',
                                                  'all_development_episodes_at_metric_floor'}
    current = triage.record(root,repo,'derived',diagnosis)
    assert current['status'] == 'triaged_local_only'
    assert measured.read_bytes() == before
    record = read_json(root/current['diagnosis_ref'])
    assert 'not verified' in record['authority']
    assert list((root/current['diagnosis_ref']).parent.joinpath('history').glob('*.json'))
    diagnosis['decision'] = 'investigate'
    assert triage.record(root,repo,'derived',diagnosis)['status'] == 'required'
    assert len(list((root/current['diagnosis_ref']).parent.joinpath('history').glob('*.json'))) == 2


@pytest.mark.parametrize('changed_source', [False, True])
def test_triage_resolves_run_owned_source_handles_without_losing_native_context(tmp_path, changed_source):
    from autosim.research.checkout_source_citations import capture
    root, repo, measured, diagnosis = fixture(tmp_path)
    source = capture(root, repo, {'ref':'train.py'})
    diagnosis['source_refs'] = [{'ref':source['ref'], 'quote':source['quote']}]
    before = measured.read_bytes()
    if changed_source:
        (repo/'train.py').write_text('changed native loader')
        with pytest.raises(ValueError, match='sealed_source_invalid'):
            triage.record(root, repo, 'derived', diagnosis)
    else:
        assert triage.record(root, repo, 'derived', diagnosis)['status'] == 'triaged_local_only'
    assert measured.read_bytes() == before


@pytest.mark.parametrize('fault',['subject','missing_native','foreign_native','native_bytes','source_quote','shape'])
def test_unproven_or_stale_triage_cannot_clear_handoff(tmp_path,fault):
    root,repo,measured,diagnosis = fixture(tmp_path)
    if fault == 'subject': diagnosis['subject_sha256'] = 'd'*64
    elif fault == 'missing_native': diagnosis['evidence_ids'].pop()
    elif fault == 'foreign_native': diagnosis['evidence_ids'].append('d'*32)
    elif fault == 'native_bytes': (root/'logs'/('a'*32+'.log')).write_text('changed')
    elif fault == 'source_quote': diagnosis['source_refs'][0]['quote'] = 'imaginary loader'
    else: diagnosis['checks']['data'] = ''
    before = measured.read_bytes()
    with pytest.raises(ValueError): triage.record(root,repo,'derived',diagnosis)
    assert triage.view(root,'derived')['status'] == 'required'
    assert before == measured.read_bytes()


def test_changed_measurement_or_diagnosis_reopens_triage(tmp_path):
    root,repo,measured,diagnosis = fixture(tmp_path)
    current = triage.record(root,repo,'derived',diagnosis)
    path = root/current['diagnosis_ref']; record = read_json(path)
    record['diagnosis']['reason'] = 'tampered'; atomic_json(path,record)
    assert triage.view(root,'derived')['status'] == 'required'
    triage.record(root,repo,'derived',diagnosis)
    row = read_json(measured);row['settings'] = {'steps':8192};atomic_json(measured,row)
    assert triage.view(root,'derived')['status'] == 'required'


def test_prestart_reference_failure_exposes_sealed_evidence(tmp_path,monkeypatch):
    from autosim.research.operation_failure import seal
    root=tmp_path/'run';root.mkdir()
    proof=seal(root,operation='measurement_protocol_guard',exc=ValueError('frozen protocol changed'),
        native_operation_status='not_started')
    path=root/'research/derived/measurements/reference_blocked.json'
    atomic_json(path,{'ok':False,'ran':False,'why':'frozen protocol changed',**proof})
    monkeypatch.setattr('autosim.research.baseline_reference.view',lambda *a:{
        'id':'published','status':'evaluation_failed',
        'measurement_ref':'research/derived/measurements/reference_blocked.json',
        'record_ref':'baseline_references/published.json'})
    before=path.read_bytes()
    current=triage.view(root,'derived')
    assert current['status']=='required'
    assert current['subject']['evidence_ids']==[proof['evidence_id']]
    assert proof['evidence_ref'] in current['subject']['evidence_refs']
    assert path.read_bytes()==before


def test_missing_historical_proof_is_not_an_impossible_diagnosis_action(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    controller=preparation(tmp_path)
    controller.client.supports_main_agent=True
    measurement=controller.output/'research/derived/measurements/reference_legacy.json'
    atomic_json(measurement,{'ok':False,'why':'legacy prestart refusal','evaluate':{'ran':False}})
    monkeypatch.setattr('autosim.research.baseline_reference.view',lambda *a:{
        'id':'published','status':'evaluation_failed',
        'measurement_ref':'research/derived/measurements/reference_legacy.json',
        'record_ref':'baseline_references/published.json'})
    before=measurement.read_bytes()
    current=triage.view(controller.output,'derived')
    assert current['status']=='required' and current['diagnosis_recordable'] is False
    assert 'sealed_failure_evidence_missing' in current['subject']['problems']
    assert 'Do not fabricate' in current['evidence_gap']
    assert 'record_baseline_diagnosis' not in controller._available_operations()
    assert measurement.read_bytes()==before


def test_zero_reward_is_not_inferred_to_mean_zero_success(tmp_path):
    root,_,measured,_ = fixture(tmp_path)
    row = read_json(measured);row['metric'] = {'name':'return'}
    row['train'] = {'ran':False};atomic_json(measured,row)
    assert triage.view(root,'derived')['status'] == 'not_required'


def test_main_loop_refuses_blind_repeat_and_offers_scoped_triage(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    real = preparation(tmp_path/'controller')
    root,repo,measured,diagnosis = fixture(tmp_path/'baseline')
    real.output=root;real.repo=repo;real.run_id='derived';real.client.supports_main_agent=True
    atomic_json(root/'research/derived/controller_session.json',{'status':'paused',
        'run_id':'derived','repository':str(repo),'base_settings':{},'history':[]})
    monkeypatch.setattr(real,'_research_loop_ready',lambda:True)
    monkeypatch.setattr(real,'_research_controller',lambda:(object(),{}))
    monkeypatch.setattr(real,'_unscored_baseline_recovery_status',lambda:{})
    monkeypatch.setattr(real,'_research_options',lambda *a:{'status':'empty','items':[]})
    answer = real._step_run_the_loop(idea_label='data_trial')
    assert answer['failure_category'] == 'baseline_diagnosis_required'
    assert answer['retryable'] and answer['native_operation_status'] == 'not_started'
    assert answer['evidence_ids'] == diagnosis['evidence_ids']
    assert 'record_baseline_diagnosis' in real._available_operations({'status':'paused'})
    assert 'run_the_loop' not in real._available_operations({'status':'paused'})
    assert real._step_record_baseline_diagnosis(diagnosis=diagnosis)['diagnosis_status'] == 'triaged_local_only'
    assert 'record_baseline_diagnosis' not in real._available_operations({'status':'paused'})


def test_data_absence_is_unknown_only_when_training_really_ran(tmp_path):
    from autosim.research.data_versions import verify_consumption
    assert verify_consumption(tmp_path,[],{'ran':True})['status'] == 'unknown'
    assert verify_consumption(tmp_path,[],{'ran':False,'origin':'released_reference'})['status'] == 'not_applicable'
    assert verify_consumption(tmp_path,[],{'attempt_id':'a'*32,'returncode':0,'reused':True})['status'] == 'unknown'


def test_legacy_reused_training_and_custom_metric_still_require_triage(tmp_path):
    root,_,measured,_ = fixture(tmp_path)
    row = read_json(measured)
    row['train'] = {'attempt_id':'a'*32,'returncode':0,'reused':True}
    row['metric'] = {'name':'custom_success','direction':'maximize','minimum':0,'maximum':1}
    row['metric_reading']['episode_values'] = [0]*100
    atomic_json(measured,row)
    current = triage.view(root,'derived')
    assert current['status'] == 'required'
    assert len(current['subject']['problems']) == 2


@pytest.mark.parametrize('tampered',[False,True])
def test_new_fix_owned_operations_do_not_depend_on_stage_allowlist(tmp_path,monkeypatch,tampered):
    import json
    from contextlib import nullcontext
    from tests.test_prepare import preparation
    from autosim.research.operation_failure import seal
    from autosim.research.prepare import _sealed_fix_handoff
    real=preparation(tmp_path);real.client.supports_agent_fix=True
    real.client.as_role=lambda role:nullcontext()
    calls=[]
    def diagnosis(system,user,**kwargs):
        calls.append((json.loads(user),kwargs))
        return json.dumps({'assessment':'repair_attempted','summary':'Proposed source-backed repair; no edit performed.',
                           'changes':[],'evidence_refs':[]}),{}
    real.client.chat_with_metadata=diagnosis
    evidence=seal(real.output,operation='previously_unknown_research_operation',
                  exc=RuntimeError('sealed native binding error'))
    action={'step':'previously_unknown_research_operation','outcome':'new failure shape',
            'repair_owner':'fix','failure_category':'binding_failed','evidence_id':evidence['evidence_id']}
    assert _sealed_fix_handoff(action)
    if tampered:
        (real.output/read_json(real.output/evidence['evidence_ref'])['log_ref']).write_text('changed')
    observed=real._fix_failed_action(action_id='c'*16,action=action)
    if tampered:
        assert observed is None and not calls
    else:
        assert calls[0][1]['read_only'] is True
        assert calls[0][0]['failed_action']['evidence_id']==evidence['evidence_id']
        assert observed['assessment']=='uncertain'  # A read-only report cannot certify a repair.
