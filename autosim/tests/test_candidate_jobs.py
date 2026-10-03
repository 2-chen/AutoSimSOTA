from types import SimpleNamespace

import pytest

from autosim.research.candidate_jobs import allocation, adopt
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.ideas import Idea, CLEARED
from tests.test_derived_research import research


def candidate(tmp_path):
    real = research(tmp_path)
    real.sources['train'] = real.sources['train'].replace('print("succ: 0.50")',
        'print("global_step=2"); print("succ: 0.50")')
    real.backend.sources['train'] = real.sources['train']
    from autosim.research.declarative_backend import checked_function
    real.backend._functions['train'] = checked_function(real.sources['train'],'stage_argv_train')
    real.require_training_progress = True
    real.run(rounds=2, yield_after_action=True)
    idea = Idea(label='more training', granularity='param', risk='low',
                mechanism='increase learning work', change={'train.n_epochs':2},
                why='test a paid longer training candidate',status=CLEARED)
    real.library.add(idea)
    return real, idea


def test_allocation_is_bound_to_original_baseline_and_round(tmp_path):
    real, idea = candidate(tmp_path)
    held = allocation(real,idea.label)
    assert held['round']==1 and held['settings']['train.n_epochs']==2
    assert held['baseline_sha256']==digest(real.run_root/'measurements/baseline.json')
    assert read_json(real.run_root/'controller_session.json')['base_settings']=={}


@pytest.mark.parametrize('local_budget',[None,{'train.n_epochs':3}])
def test_paid_candidate_adopts_without_retraining(tmp_path,monkeypatch,local_budget):
    real,idea = candidate(tmp_path)
    held = allocation(real,idea.label,local_budget)
    trained=real.run_stage('train',settings=held['settings'])
    job_id='a'*32
    atomic_json(real.output/'native_jobs'/job_id/'request.json',dict(repo=str(real.repo),
        run_id=real.run_id,stage='train',settings=held['settings'],candidate_binding=held,
        source_identity='source'))
    monkeypatch.setattr('autosim.research.native_jobs.source_identity',lambda *a:'source')
    monkeypatch.setattr('autosim.research.native_jobs.status',lambda *a:dict(status='completed',
        result=dict(stage_result=trained)))
    binding=adopt(real,job_id,idea.label)
    before=digest(real.run_root/'measurements/baseline.json')
    calls=[]
    original=real.run_stage
    def observed(stage,**kwargs):
        calls.append(stage)
        return original(stage,**kwargs)
    monkeypatch.setattr(real,'run_stage',observed)
    report=real.run(rounds=2,yield_after_action=True,selected_idea_label=idea.label,
        main_controller_owns_selection=True,candidate_job=binding)
    assert calls==['evaluate']
    assert report['rounds'][-1]['status']=='measured'
    assert digest(real.run_root/'measurements/baseline.json')==before


def test_candidate_rejects_heldout_stale_round_or_unbound_job(tmp_path,monkeypatch):
    real,idea=candidate(tmp_path)
    job_id='b'*32
    atomic_json(real.output/'native_jobs'/job_id/'request.json',{})
    monkeypatch.setattr('autosim.research.native_jobs.status',lambda *a:dict(status='completed'))
    with pytest.raises(ValueError):adopt(real,job_id,idea.label)
    atomic_json(real.run_root/'confirmation_attempts/one.json',{})
    with pytest.raises(ValueError):allocation(real,idea.label)
