from autosim.research.common import atomic_json
from autosim.research.experiment_view import view
from autosim.research.recorder import render


def test_sample_read_granularity_survives_compact_projection(tmp_path):
    root=tmp_path/'research/derived'
    atomic_json(root/'measurements/round_1.json',{'label':'round_1','ok':False,
        'data_consumption':{'status':'verified','exposure':[{'data_version_id':'extra',
            'status':'basic_read_only','consumption_stage':'sample_read','samples_read':24}]}})
    row=view(tmp_path,'derived')['candidates'][0]
    assert row['data_consumption']=='verified'
    scope=row['data_consumption_scope']
    assert scope['reported_stages']==['sample_read']
    assert scope['training_batch_witness']=='not_reported'
    assert scope['recorded_exposure'][0]['samples_read']==24
    assert 'sparse lower bounds' in scope['authority']


def test_batch_projection_is_bounded_and_never_infers_missing_evidence():
    from autosim.research.experiment_view import _consumption_scope
    assert _consumption_scope({})['training_batch_witness']=='unknown'
    row={'data_consumption':{'exposure':[{'consumption_stage':'training_batch'}]*10}}
    scope=_consumption_scope(row)
    assert scope['training_batch_witness']=='reported'
    assert len(scope['recorded_exposure'])==8 and scope['exposures_omitted']==2
    row['data_consumption']['exposure'].append({'consumption_stage':'sample_read'})
    assert _consumption_scope(row)['training_batch_witness']=='not_reported'


def setup_job(root):
    job=root/'native_jobs'/('a'*32)
    atomic_json(job/'request.json',dict(job_id='a'*32,run_id='derived',
        stage='train',
        candidate_binding=dict(round=1,idea_label='data coverage')))
    atomic_json(job/'result.json',dict(job_id='a'*32,status='completed',stage_result=dict(attempt_id='b'*32)))
    return job


def test_completed_training_is_not_scored_until_receipt_verified(tmp_path,monkeypatch):
    setup_job(tmp_path)
    assert view(tmp_path,'derived')['experiments'][0]['status']=='completed'
    atomic_json(tmp_path/'research/derived/measurements/round_1.json',
        dict(label='round_1',ok=True,metric_value=.8,train=dict(attempt_id='b'*32)))
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
        lambda *a:dict(status='inconsistent'))
    row=view(tmp_path,'derived')['experiments'][0]
    assert row['status']=='unscored' and row['metric_value'] is None
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
        lambda *a:dict(status='consistent'))
    assert view(tmp_path,'derived')['experiments'][0]['metric_value']==.8


def test_symlink_result_is_not_projected(tmp_path):
    job=setup_job(tmp_path)
    (job/'result.json').unlink()
    outside=tmp_path/'outside';outside.write_text('{}')
    (job/'result.json').symlink_to(outside)
    assert view(tmp_path,'derived')['experiments']==[]


def test_source_changed_completed_training_is_not_recommended_as_adoptable(tmp_path,monkeypatch):
    import json
    job=setup_job(tmp_path)
    repo=tmp_path/'checkout';repo.mkdir()
    request=json.loads((job/'request.json').read_text());request['repo']=str(repo)
    atomic_json(job/'request.json',request)
    monkeypatch.setattr('autosim.research.native_jobs.source_matches',lambda *a:False)
    projected=view(tmp_path,'derived')
    row=projected['experiments'][0]
    assert row['native_status']=='completed'
    assert row['status']=='adoption_blocked'
    assert row['next_operation'] is None
    assert row['adoption_admission']['source_matches'] is False
    assert projected['metrics']['completed_training_adoption_blocked']==1
    assert projected['metrics']['completed_training_pending_score']==0
    monkeypatch.setattr('autosim.research.native_jobs.source_matches',lambda *a:True)
    restored=view(tmp_path,'derived')['experiments'][0]
    assert restored['status']=='completed' and restored['next_operation']=='inspect_native_job'


def test_inline_formal_training_pending_score_is_counted_without_a_native_job(tmp_path):
    attempt='d'*32
    root=tmp_path/'research/derived'
    atomic_json(root/'attempts'/attempt/'receipt.json',dict(attempt_id=attempt,run_id='derived',
        stage='train',status='completed',returncode=0))
    atomic_json(root/'measurements/baseline.json',dict(label='baseline',ok=False,where='evaluate',
        train=dict(attempt_id=attempt),evaluate=dict(attempt_id='e'*32)))
    projected=view(tmp_path,'derived')
    assert projected['metrics']['completed_training_pending_score']==1
    assert projected['metrics']['completed_training_jobs']==1
    assert projected['metrics']['valid_scored_development']==0
    assert projected['candidates'][0]['training_completed'] is True


def test_inline_and_job_views_deduplicate_the_same_completed_training(tmp_path):
    setup_job(tmp_path)
    root=tmp_path/'research/derived'
    atomic_json(root/'attempts'/('b'*32)/'receipt.json',dict(attempt_id='b'*32,run_id='derived',
        stage='train',status='completed',returncode=0))
    atomic_json(root/'measurements/baseline.json',dict(label='baseline',ok=False,where='evaluate',
        train=dict(attempt_id='b'*32)))
    assert view(tmp_path,'derived')['metrics']['completed_training_pending_score']==1


def test_hypothesis_is_annotation_not_receipt_authority(tmp_path, monkeypatch):
    root = tmp_path/'research/derived'
    atomic_json(root/'controller_session.json', {'run_id':'derived', 'history':[
        {'label':'round_1', 'hypothesis':'Longer action chunks improve success',
         'mechanism':'fewer policy queries', 'verdict':'keep'}]})
    atomic_json(root/'measurements/round_1.json', {'label':'round_1', 'ok':True, 'metric_value':.8})
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
        lambda *a: {'status':'inconsistent', 'issues':['loaded policy differs from requested policy']})
    row = view(tmp_path,'derived')['candidates'][0]
    assert row['research_note']['hypothesis'].startswith('Longer')
    assert row['research_note']['verdict'] == 'keep'
    assert row['status'] == 'unverified' and row['metric_value'] is None
    assert row['verification']['issues'] == ['loaded policy differs from requested policy']
    assert row['interpretation'].startswith('not_scientific_negative')


def test_real_zero_is_visible_but_unexecuted_zero_is_not(tmp_path, monkeypatch):
    root = tmp_path/'research/derived'
    atomic_json(root/'measurements/round_1.json', {'label':'round_1', 'ok':True, 'metric_value':0})
    atomic_json(root/'measurements/round_2.json', {'label':'round_2', 'ok':False, 'metric_value':0})
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
        lambda *a: {'status':'consistent'})
    rows = view(tmp_path,'derived')['candidates']
    assert rows[0]['metric_value'] == 0 and rows[0]['status'] == 'scored'
    assert rows[1]['metric_value'] is None and rows[1]['status'] == 'unscored'


def test_current_round_is_not_hidden_by_lexical_order(tmp_path):
    import os
    root = tmp_path/'research/derived/measurements'
    for i in range(1, 106):
        path = root/f'round_{i}.json'
        atomic_json(path, {'label':f'round_{i}', 'ok':False})
        os.utime(path, ns=(1_000_000_000 + i, 1_000_000_000 + i))
    rows = view(tmp_path, 'derived')['candidates']
    assert rows[-1]['label'] == 'round_105'
    assert rows[0]['label'] == 'round_76'


def test_reflection_is_bounded_and_keeps_no_implicit_verdict(tmp_path):
    root = tmp_path/'research/derived'
    atomic_json(root/'controller_session.json', {'run_id':'derived', 'history':[
        {'label':'round_1', 'hypothesis':'h'*10000, 'mechanism':'m'*10000}]})
    atomic_json(root/'measurements/round_1.json', {'label':'round_1', 'ok':False})
    note = view(tmp_path, 'derived')['candidates'][0]['research_note']
    assert len(note['hypothesis']) == 240 and len(note['mechanism']) == 160
    assert 'verdict' not in note
