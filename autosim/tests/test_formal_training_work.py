import json
from pathlib import Path

import pytest

from autosim.research.training_work import missing_formal_work
from autosim.research.development_evidence import summaries
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.ideas import Idea, execution_compatibility


@pytest.mark.parametrize('value',[None,0,-1,True,1.0,1.5,'1024',float('inf')])
def test_formal_work_requires_explicit_integral_updates(value):
    assert missing_formal_work('def stage_argv_train(i):\n    return [str(i["steps"])]',{'steps':value})


def test_epoch_units_and_native_defaults_are_not_replaced_by_probe_steps():
    assert not missing_formal_work('def stage_argv_train(i):\n    return [str(i["settings"]["epochs"])]',{'steps':0})
    assert not missing_formal_work('def stage_argv_train(i):\n    return [str(i.get("steps"))]',{'steps':8192})


@pytest.mark.parametrize('fault',['','changed','outside','confirmation'])
def test_development_diagnostics_verify_bytes_and_do_not_export_episode_rows(tmp_path,fault):
    root=tmp_path/'research';result=root/'experiments/baseline/result.json'
    atomic_json(result,{'episodes':[{'episode_index':17,'episode_seed':41,'success':False,
        'failure_stage':'under_threshold','depth_m':0.001,'raw_obs':[1,2,3]}]*100})
    row={'label':'baseline','ok':True,'metric_value':0.0,'metric_reading':{'samples':100},
         'result_artifact':{'path':str(result),'content_sha256':digest(result),
                            'sha256':'a manifest identity, not the byte hash'}}
    if fault=='changed':result.write_text('{"episodes":[]}')
    if fault=='outside':row['result_artifact']['path']=str(tmp_path/'outside.json');atomic_json(tmp_path/'outside.json',{'episodes':[]})
    if fault=='confirmation':row['label']='confirmation'
    target=root/'measurements'/('confirmation.json' if fault=='confirmation' else 'baseline.json')
    atomic_json(target,row)
    got=summaries(root)
    if fault=='confirmation':assert not got;return
    assert got[0]['samples']==100
    if fault:assert not got[0].get('diagnostic_artifact_verified');return
    assert got[0]['failure_stage_counts']=={'under_threshold':100}
    assert got[0]['diagnostic_numeric_aggregates']=={'depth_m':{'count':100,'min':0.001,'max':0.001,'mean':pytest.approx(0.001)}}
    assert 'episode_index' not in json.dumps(got) and 'raw_obs' not in json.dumps(got)


@pytest.mark.parametrize('find',['<the optimizer block>','missing','unique','duplicate'])
def test_source_candidate_requires_literal_unique_patch_before_a_round(tmp_path,find):
    (tmp_path/'producer.py').write_text('unique\nduplicate\nduplicate\n')
    idea=Idea(label='candidate',granularity='code',change={'file':'producer.py','find':find,'replace':'new'})
    ok,_=execution_compatibility(idea,stage_parameters={},repo=tmp_path)
    assert ok==(find=='unique')


def test_missing_formal_work_never_launches_a_native_process(tmp_path):
    from tests.test_derived_research import research
    from autosim.research.declarative_backend import DeclarativeBackend
    real=research(tmp_path)
    real.backend.sources['train'] = ('def stage_argv_train(i):\n'
        '    return [i["python"], "-c", "print(123)", str(i["steps"])]\n')
    real.backend=DeclarativeBackend(repo=tmp_path,answer=real.backend.answer,
        sources=real.backend.sources,parameters={})
    result=real.run_stage('train',settings={})
    assert result['ran'] is False and result['termination_reason']=='formal_training_work_required'
    assert not list((real.run_root/'attempts').glob('*/receipt.json'))


@pytest.mark.parametrize('declared_axis',[True,False])
def test_main_training_settings_are_persisted_and_cannot_rewrite_baseline(tmp_path,monkeypatch,declared_axis):
    from tests.test_prepare import preparation
    from autosim.research.adapter_protocol import Axis, OptimizationSpace
    real=preparation(tmp_path);real.declaration={'benchmark':'toy'}
    monkeypatch.setattr(real,'_research_loop_ready',lambda:True)
    seen=[]
    class Controller:
        run_root=real.output/'research'/real.run_id
        execution_graph=None
        space=OptimizationSpace(training=(Axis('steps','integer','updates',low=1,high=200000,default=100000),)
                                if declared_axis else ())
        sources={'train':'def stage_argv_train(i):\n    return [str(i["steps"])]'}
        def _inputs(self,stage,*,settings):return {'steps':settings.get('steps',0)}
        def run(self,**kwargs):
            seen.append(kwargs)
            atomic_json(self.run_root/'controller_session.json',{'run_id':real.run_id,
                'repository':str(real.repo),'status':'paused','base_settings':kwargs['settings'],
                'rounds':kwargs['rounds']})
            atomic_json(self.run_root/'research_report.json',{'run_id':real.run_id,
                'repo':str(real.repo),'run_status':'paused'})
            return {'rounds':[],'run_status':'paused','objective':{}}
    monkeypatch.setattr(real,'_research_controller',lambda:(Controller(),{}))
    assert real._step_run_the_loop()['outcome']=='not attempted' and not seen
    assert real._step_run_the_loop(training_settings={'steps':0})['outcome']=='not attempted'
    real._step_run_the_loop(training_settings={'steps':8192})
    assert seen[-1]['settings']=={'steps':8192}
    persisted=read_json(Controller.run_root/'controller_session.json')
    atomic_json(Controller.run_root/'controller_session.json',{**persisted,'rounds':5})
    real._step_run_the_loop(idea_label='audited performance idea')
    assert len(seen)==2 and seen[-1]['settings']=={'steps':8192}
    assert seen[-1]['rounds']==5  # Persisted Agent allocation takes priority over CLI defaults.
    refused=real._step_run_the_loop(idea_label='audited performance idea',training_settings={'steps':16384})
    assert refused['outcome']=='not attempted' and len(seen)==2


@pytest.mark.parametrize('fault',['array','bad_settings','directory_symlink','experiment_symlink'])
def test_development_summary_tolerates_malformed_or_escaping_evidence(tmp_path,fault):
    root=tmp_path/'research'
    outside=tmp_path/'outside'
    result=root/'experiments/result.json'
    atomic_json(result,{'episodes':[{'failure_stage':'failure','depth_m':1}]})
    row={'ok':True,'metric_value':0,'result_artifact':{'path':str(result),'content_sha256':digest(result)}}
    if fault=='array':row=[]
    if fault=='bad_settings':row['settings']=['not a mapping']
    if fault=='directory_symlink':
        atomic_json(outside/'baseline.json',row)
        (root/'measurements').symlink_to(outside,target_is_directory=True)
    else:
        atomic_json(root/'measurements/baseline.json',row)
    if fault=='experiment_symlink':
        external=outside/'experiments'
        external.mkdir(parents=True)
        (root/'experiments/result.json').rename(external/'result.json')
        (root/'experiments').rmdir()
        (root/'experiments').symlink_to(external,target_is_directory=True)
    got=summaries(root)
    if fault=='experiment_symlink':
        assert got and not got[0].get('diagnostic_artifact_verified')
    else:
        assert got==[]


def test_long_job_accepts_explicit_learning_work_without_rewriting_baseline(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tests.test_prepare import preparation
    from autosim.research.adapter_protocol import Axis, OptimizationSpace
    real=preparation(tmp_path)
    real.stages={'train':{},'collect':{}}
    real.base_settings={'steps':1024}
    real.budget=SimpleNamespace(limit=3600)
    atomic_json(real.output/'task_gpu_budget.json',{})
    research=SimpleNamespace(space=OptimizationSpace(training=(Axis('steps','integer','updates',low=1,high=200000),)),
        sources={'train':'def stage_argv_train(i):\n    return [str(i["steps"])]'},
        _inputs=lambda stage,settings:{'steps':settings.get('steps',0)})
    monkeypatch.setattr(real,'_research_controller',lambda:(research,{}))
    calls=[]
    def submit(*args,**kwargs):
        calls.append(kwargs)
        return {'job_ref':'native_jobs/test/request.json','job_id':'test'}
    monkeypatch.setattr('autosim.research.native_jobs.submit',submit)
    result=real._step_submit_native_job(stage='train',window_seconds=100,reason='adequate learning',
        training_settings={'steps':8192})
    assert result['outcome']=='submitted' and calls[-1]['settings']=={'steps':8192}
    assert real.base_settings=={'steps':1024}
    refused=real._step_submit_native_job(stage='train',window_seconds=100,reason='bad',training_settings={'steps':0})
    assert refused['outcome']=='not attempted' and len(calls)==1
    refused=real._step_submit_native_job(stage='collect',window_seconds=100,reason='bad',training_settings={'steps':8192})
    assert refused['outcome']=='not attempted' and len(calls)==1


def test_unknown_operation_argument_is_contract_error_before_native_execution(tmp_path):
    from types import SimpleNamespace
    from tests.test_prepare import preparation
    real=preparation(tmp_path)
    real.client=SimpleNamespace(supports_main_agent=True)
    result=real.do('submit_native_job',stage='train',window_seconds=100,reason='test',imaginary_option=True)
    assert result['outcome']=='not attempted'
    assert result['failure_category']=='operation_arguments_invalid'
    assert 'training_settings' in result['allowed_arguments']
    assert not list((real.output/'native_jobs').glob('*/request.json'))


def test_public_loop_dispatch_preserves_explicit_training_settings(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    real=preparation(tmp_path)
    seen=[]
    monkeypatch.setattr(real,'state',lambda:{'research_progress':{},'research_options':{}})
    monkeypatch.setattr(real,'_unscored_baseline_recovery_status',lambda:{})
    monkeypatch.setattr(real,'_publish_main_context',lambda facts:None)
    def run_the_loop(**kwargs):
        seen.append(kwargs)
        return {'outcome':'done'}
    monkeypatch.setattr(real,'_step_run_the_loop',run_the_loop)
    result=real.do('run_the_loop',training_settings={'steps':8192})
    assert result['outcome']=='done' and seen==[{'training_settings':{'steps':8192}}]
