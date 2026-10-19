import pytest

from autosim.research.candidate_settings import apply
from autosim.research.common import digest, read_json
from autosim.research.ideas import Idea, CLEARED
from tests.test_candidate_jobs import candidate


def seeded_research(tmp_path):
    from tests.test_derived_research import research
    from autosim.research.declarative_backend import checked_function
    real=research(tmp_path)
    for stage in ('train','evaluate'):
        # A repository default must not overwrite either phase's caller input.
        real.backend.parameters[stage]={'--seed':{'value':777},
            **{axis.name:{'value':axis.default} for axis in real.space.training}}
        source=real.sources[stage].replace('print("succ: 0.50")',
            'print("global_step=2"); print("succ: 0.50")')
        source=source.replace('    return argv',
            '    argv = argv + ["--seed", str(i["seed"])]\n    return argv')
        real.sources[stage]=real.backend.sources[stage]=source
        real.backend._functions[stage]=checked_function(source,f'stage_argv_{stage}')
    real.require_training_progress=True
    real.run(rounds=2,settings={'seed':20261013},yield_after_action=True)
    idea=Idea(label='new training replicate',granularity='param',change={'train.n_epochs':2},
              mechanism='different optimizer randomness',status=CLEARED)
    real.library.add(idea)
    return real,idea


@pytest.mark.parametrize('seed',[0,1000])
def test_training_seed_reaches_only_native_train_and_preserves_frozen_evaluation(tmp_path,seed):
    from autosim.research.candidate_settings import TRAINING_SEED
    real,idea=seeded_research(tmp_path)
    protocol=digest(real.run_root/'comparison_protocol.json')
    baseline=digest(real.run_root/'measurements/baseline.json')
    report=real.run(rounds=2,settings={'seed':20261013},yield_after_action=True,
        selected_idea_label=idea.label,main_controller_owns_selection=True,
        candidate_training_settings={'seed':seed})
    label=report['rounds'][-1]['label']
    measured=read_json(real.run_root/'measurements'/f'{label}.json')
    assert measured['ok'] and measured['settings']['seed']==20261013
    assert measured['settings'][TRAINING_SEED]==seed
    for stage,expected in [('train',seed),('evaluate',20261013)]:
        attempt=measured[stage]['attempt_id']
        receipt=read_json(real.run_root/'attempts'/attempt/'receipt.json')
        assert receipt['requested_seed']==expected
        assert receipt['argv'][receipt['argv'].index('--seed')+1]==str(expected)
        assert not any(TRAINING_SEED in arg for arg in receipt['argv'])
    assert digest(real.run_root/'comparison_protocol.json')==protocol
    assert digest(real.run_root/'measurements/baseline.json')==baseline
    assert 'seed' in real._comparison_protocol_violation(
        {**measured['settings'],'seed':999},target='evaluate')


def test_detached_seed_allocation_retains_eval_seed_and_adopts_paid_work(tmp_path,monkeypatch):
    from autosim.research.candidate_jobs import allocation,adopt
    from autosim.research.common import atomic_json
    real,idea=seeded_research(tmp_path)
    held=allocation(real,idea.label,{'seed':1000})
    assert held['settings']['seed']==20261013
    trained=real.run_stage('train',settings=held['settings'])
    assert trained['requested_seed']==1000
    job_id='d'*32
    atomic_json(real.output/'native_jobs'/job_id/'request.json',dict(repo=str(real.repo),
        run_id=real.run_id,stage='train',settings=held['settings'],candidate_binding=held,
        source_identity='source'))
    monkeypatch.setattr('autosim.research.native_jobs.source_identity',lambda *a:'source')
    monkeypatch.setattr('autosim.research.native_jobs.status',lambda *a:dict(status='completed',
        result=dict(stage_result=trained)))
    binding=adopt(real,job_id,idea.label)
    calls=[];native=real.run_stage
    def record(stage,**kwargs):
        calls.append(stage);return native(stage,**kwargs)
    monkeypatch.setattr(real,'run_stage',record)
    report=real.run(rounds=2,settings={'seed':20261013},yield_after_action=True,
        selected_idea_label=idea.label,main_controller_owns_selection=True,candidate_job=binding)
    assert calls==['evaluate'] and report['rounds'][-1]['status']=='measured'


def test_detached_candidate_review_receives_phase_scoped_seed(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research.candidate_settings import TRAINING_SEED
    engine,idea=seeded_research(tmp_path)
    real=preparation(tmp_path)
    real.stages=engine.sources
    real.base_settings={'seed':20261013}
    monkeypatch.setattr(real,'_research_controller',lambda:(engine,{}))
    reviewed=[]
    def review(research,label,**kwargs):
        reviewed.append(kwargs['selected_training_settings'])
        return {'outcome':'not attempted','because':'review-only regression'}
    monkeypatch.setattr(real,'_review_candidate_execution',review)
    result=real._step_submit_native_job(stage='train',window_seconds=30,
        reason='independent training replicate',idea_label=idea.label,
        training_settings={'seed':1000})
    assert result['because']=='review-only regression'
    assert reviewed==[{'seed':20261013,TRAINING_SEED:1000}]
    assert real.base_settings=={'seed':20261013}


@pytest.mark.parametrize('value',[True,-1,'1000'])
def test_scoped_training_seed_rejects_invalid_values_without_changing_protocol(tmp_path,value):
    real,idea=seeded_research(tmp_path)
    protocol=digest(real.run_root/'comparison_protocol.json')
    with pytest.raises(ValueError): apply(real,{'seed':20261013},{'seed':value})
    assert digest(real.run_root/'comparison_protocol.json')==protocol


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
