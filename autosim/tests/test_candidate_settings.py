import pytest

from autosim.research.candidate_settings import apply
from autosim.research.common import digest, read_json
from autosim.research.ideas import Idea, CLEARED
from tests.test_candidate_jobs import candidate


def test_explicit_candidate_work_preserves_baseline_and_is_recorded(tmp_path):
    real,idea=candidate(tmp_path)
    baseline=digest(real.run_root/'measurements/baseline.json')
    report=real.run(rounds=2,yield_after_action=True,selected_idea_label=idea.label,
        main_controller_owns_selection=True,candidate_training_settings={'train.n_epochs':3})
    assert report['rounds'][-1]['varied']['train.n_epochs']==3
    assert digest(real.run_root/'measurements/baseline.json')==baseline
    assert read_json(real.run_root/'controller_session.json')['base_settings']=={}


@pytest.mark.parametrize('value',[True,0,10000000,'8192'])
def test_candidate_local_work_rejects_wrong_type_range_or_unbound_slot(tmp_path,value):
    real,_=candidate(tmp_path)
    with pytest.raises(ValueError):apply(real,{}, {'steps':value})
    with pytest.raises(ValueError):apply(real,{}, {'not_declared':value})


def test_candidate_work_cannot_change_frozen_eval_protocol(tmp_path,monkeypatch):
    real,_=candidate(tmp_path)
    monkeypatch.setattr(real,'_comparison_protocol_violation',lambda *a,**k:'changed protocol')
    with pytest.raises(ValueError,match='protocol'):apply(real,{}, {'train.n_epochs':2})


def test_public_candidate_operation_passes_only_local_budget(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    real,idea=candidate(tmp_path)
    obj=preparation(tmp_path);obj.declaration={'benchmark':'toy'}
    monkeypatch.setattr(obj,'_research_loop_ready',lambda:True)
    monkeypatch.setattr(obj,'_research_controller',lambda:(real,{}))
    before=digest(real.run_root/'measurements/baseline.json')
    obj._step_run_the_loop(idea_label=idea.label,training_settings={'train.n_epochs':3})
    session=read_json(real.run_root/'controller_session.json')
    assert session['base_settings']=={}
    assert session['history'][-1]['varied']['train.n_epochs']==3
    assert digest(real.run_root/'measurements/baseline.json')==before


def test_missing_candidate_work_is_rejected_before_source_or_cursor_changes(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    real,idea=candidate(tmp_path)
    obj=preparation(tmp_path);obj.declaration={'benchmark':'toy'}
    monkeypatch.setattr(obj,'_research_loop_ready',lambda:True)
    monkeypatch.setattr(obj,'_research_controller',lambda:(real,{}))
    # Simulate the source-backed caller contract used by the recovered legacy baseline.
    real.sources['train']='def stage_argv_train(i):\n    return [str(i["steps"])]\n'
    monkeypatch.setattr(real,'_inputs',lambda *a,settings,**k:dict(steps=settings.get('steps',0)))
    before=digest(real.run_root/'controller_session.json')
    answer=obj._step_run_the_loop(idea_label=idea.label)
    assert answer['native_operation_status']=='not_started'
    assert answer['failure_category']=='candidate_input_required'
    assert 'explicit positive' in answer['because']
    assert digest(real.run_root/'controller_session.json')==before


def test_training_start_rejection_keeps_exact_native_cause(tmp_path,monkeypatch):
    real,_=candidate(tmp_path)
    cause='formal learning work missing; use an explicit native update budget'
    monkeypatch.setattr(real,'run_stage',lambda *a,**k:dict(stage='train',ran=False,status='blocked',
        returncode=None,why=cause,termination_reason='formal_training_work_required'))
    result=real.measure(settings={},label='guard_failure')
    assert result['why']==cause and result['train']['why']==cause
    assert result['termination_reason']=='formal_training_work_required'
