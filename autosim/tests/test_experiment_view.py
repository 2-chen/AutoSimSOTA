from autosim.research.common import atomic_json
from autosim.research.experiment_view import view
from autosim.research.recorder import render


def setup_job(root):
    job=root/'native_jobs'/('a'*32)
    atomic_json(job/'request.json',dict(job_id='a'*32,run_id='derived',
        candidate_binding=dict(round=1,idea_label='data coverage')))
    atomic_json(job/'result.json',dict(status='completed',stage_result=dict(attempt_id='b'*32)))
    return job


def test_completed_training_is_not_scored_until_receipt_verified(tmp_path,monkeypatch):
    setup_job(tmp_path)
    assert view(tmp_path,'derived')['experiments'][0]['status']=='completed'
    atomic_json(tmp_path/'research/derived/measurements/round_1.json',
        dict(ok=True,metric_value=.8,train=dict(attempt_id='b'*32)))
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
