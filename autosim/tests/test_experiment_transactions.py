from types import SimpleNamespace

import pytest

from autosim.research.common import atomic_json, read_json
from autosim.research.experiment_transaction import track, stage, finish, view
from autosim.research.experiment_view import view as experiments
from tests.test_derived_research import research


def controller(tmp_path):
    root = tmp_path/'research/derived'
    root.mkdir(parents=True)
    return SimpleNamespace(run_root=root, output=tmp_path, run_id='derived', _note=lambda *a:None)


def test_interrupted_index_retains_original_attempt_and_never_executes(tmp_path):
    real = controller(tmp_path)
    with pytest.raises(RuntimeError, match='native fault'):
        with track(real,label='baseline',settings={}):
            stage(real,'train',result={'attempt_id':'a'*32,'status':'completed','returncode':0})
            stage(real,'evaluate')
            raise RuntimeError('native fault')
    row = view(tmp_path,'derived')['transactions'][0]
    assert row['status']=='interrupted'
    assert row['stages'][0]['attempt_id']=='a'*32
    assert 'ownership' in row['recovery']['guidance'] or 'receipts' in row['recovery']['guidance']
    assert real._experiment_journal is None
    assert not (real.run_root/'measurements').exists()


def test_returned_measurement_is_indexed_not_certified(tmp_path):
    real = controller(tmp_path)
    result = {'label':'baseline','ok':True,'metric_value':.5}
    with track(real,label='baseline',settings={}) as journal:
        atomic_json(real.run_root/'measurements/baseline.json', result)
        finish(real,journal,result)
    row = view(tmp_path,'derived')['transactions'][0]
    assert row['phase']=='measurement_returned' and 'metric_value' not in row
    assert experiments(tmp_path,'derived')['candidates'][0]['status']=='unverified'
    atomic_json(real.run_root/'measurements/baseline.json', {**result,'metric_value':.9})
    assert view(tmp_path,'derived')['index_errors']


def test_symlinked_index_and_escaped_run_are_rejected(tmp_path):
    real = controller(tmp_path)
    (real.run_root/'experiment_transactions').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='unsafe'):
        view(tmp_path,'../derived')
    with track(real,label='baseline',settings={}) as journal:
        finish(real,journal,{'ok':False})
    assert not (tmp_path/(journal['id']+'.json')).exists()


def test_completed_baseline_training_is_visible_without_candidate_binding(tmp_path):
    job = tmp_path/'native_jobs'/('a'*32)
    atomic_json(job/'request.json',{'job_id':'a'*32,'run_id':'derived','stage':'train'})
    atomic_json(job/'result.json',{'job_id':'a'*32,'status':'completed',
        'stage_result':{'attempt_id':'b'*32}})
    result = experiments(tmp_path,'derived')
    assert result['metrics']['completed_training_pending_score']==1
    assert result['experiments'][0]['next_operation']=='inspect_native_job'
    assert not (tmp_path/'research').exists()


def test_native_cpu_measurement_flows_through_the_real_transaction(tmp_path):
    real = research(tmp_path)
    result = real.measure(settings={},label='baseline')
    assert result['ok']
    rows = view(real.output,'derived')['transactions']
    assert len(rows)==1
    assert [row['stage'] for row in rows[0]['stages']]==['train','evaluate']
    assert rows[0]['measurement_ref']=='research/derived/measurements/baseline.json'
    from autosim.research.receipt_verifier import verify_measurement
    assert verify_measurement(real.run_root,'baseline')['status']=='consistent'
    assert experiments(real.output,'derived')['metrics']['valid_scored_development']==1


def test_transaction_failure_cannot_discard_native_score(tmp_path, monkeypatch):
    real = research(tmp_path)
    def no_index(*a):
        raise OSError('index unavailable')
    monkeypatch.setattr('autosim.research.experiment_transaction._save',no_index)
    result = real.measure(settings={},label='baseline')
    assert result['ok']
    assert read_json(real.run_root/'measurements/baseline.json')['ok']
