import json
from pathlib import Path

import pytest

from autosim.research.training_work import missing_formal_work
from autosim.research.development_evidence import summaries
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.ideas import Idea, execution_compatibility


@pytest.mark.parametrize('fault', ['', 'valid', 'changed_log', 'changed_selection',
    'changed_card', 'wrong_producer', 'wrong_scope', 'wrong_environment', 'wrong_settings'])
def test_sealed_reader_failure_is_visible_without_recertifying_or_mutating(tmp_path, fault):
    from autosim.research import data_versions as versions
    from autosim.research.development_evidence import reader_failure_diagnostics
    from autosim.research.evidence_store import capture_attempt_evidence
    from autosim.research.common import object_digest
    output = tmp_path/'out'; repo = output/'checkout'; repo.mkdir(parents=True)
    data = output/'extra.json'; data.write_text('[1,2]')
    versions.register(output, repo,
        dict(id='extra', dataset_ref='extra.json', source_refs=[], reason='fixture'),
        dict(allowed_training_data=True, loader_connection_supported=True, sources_supported=True))
    settings, bindings = versions.resolve(output, repo, {'dataset':'data-version:extra', 'steps':1000})
    attempt = 'd'*32; root = output/'research/derived'
    proof = {**versions.seal_selection(output, 'derived', 'round_16', settings, bindings),
             'training_attempt_id':attempt}
    environment = {'AUTOSIM_TRAIN_SELECTION_JSON':json.dumps(versions.training_metadata(bindings, settings))}
    train = dict(attempt_id=attempt, status='completed', returncode=0, environment=environment)
    row = dict(label='round_16', ok=False, where='policy_artifact', train=train,
               settings=settings, data_selection=proof)
    native = dict(data_version_id='extra' if fault=='valid' else None, setting='dataset',
                  identity_field=bindings[0]['identity_field'], identity=bindings[0]['identity'], samples_read=8)
    log = output/'train.log'; log.write_text(versions.MARKER+json.dumps(native)+'\n')
    receipt_ref = f'research/derived/attempts/{attempt}/receipt.json'
    receipt = dict(**train, run_id='derived', stage='train', settings_digest=object_digest(settings))
    if fault=='wrong_scope':receipt['stage']='evaluate'
    if fault=='wrong_environment':receipt['environment']={}
    if fault=='wrong_settings':receipt['settings_digest']='0'*64
    atomic_json(output/receipt_ref, receipt)
    capture_attempt_evidence(output, attempt_id=attempt, log=log, receipt_ref=receipt_ref,
        status='completed', returncode=0, termination_reason='normal_exit')
    if fault=='wrong_producer':row['data_selection']['training_attempt_id']='a'*32
    atomic_json(root/'measurements/round_16.json', row)
    # Later successful corrections or unbound trials are not reader failures.
    # Selecting the last three files BEFORE filtering used to hide round_16.
    for index in range(20,25):
        atomic_json(root/f'measurements/round_{index}.json',
                    {'label':f'round_{index}','ok':False,'where':'policy_artifact',
                     'train':train,'data_selection':None})
    atomic_json(root/'measurements/round_25_handoff_correction.json',
                {'label':'round_25_handoff_correction','ok':True})
    if fault=='changed_log':log.write_text('changed')
    if fault=='changed_selection':atomic_json(output/proof['record_ref'], {})
    if fault=='changed_card':atomic_json(output/'data_versions/extra.json', {})
    before = {str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    got = reader_failure_diagnostics(root)
    if fault:
        assert got == []
    else:
        assert len(got)==1 and got[0]['status']=='sealed_reader_events_invalid'
        assert got[0]['training_attempt_id']==attempt
        assert 'no valid data-consumption witness' in got[0]['why']
        assert 'bundle-only' in got[0]['guidance']
        assert 'metric_value' not in got[0] and 'samples_read' not in got[0]
    assert summaries(root)==[]  # Scored evidence remains a separate, unchanged contract.
    assert before == {str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}


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


def test_idea_evidence_distinguishes_requested_training_and_scored_intermediate(tmp_path):
    from tests.test_handoff_correction import fixture
    root, _ = fixture(tmp_path)
    before = (root/'measurements/round_4.json').read_bytes()
    rows = summaries(root)
    item = next(row for row in rows if row['label']=='round_4')
    assert item['training_settings']['steps'] == 8192
    assert item['training_handoff']['selected_completed_work'] == 1000
    assert item['training_handoff']['other_counter_max'] == 8192
    assert 'not proof' in item['training_settings_meaning']
    assert 'terminal training policy score' in item['interpretation_warning']
    assert item['metric_value'] == 0
    assert (root/'measurements/round_4.json').read_bytes() == before


@pytest.mark.parametrize('consumption,expected', [
    (None, 'unknown'), ('invalid receipt', 'unknown'),
    ({'status': 'not_required'}, 'not_required'),
    ({'status': 'verified', 'private_samples': ['DO_NOT_EXPORT']}, 'verified'),
    ({'status': 'invented'}, 'unknown'),
])
def test_development_evidence_reports_data_consumption_without_inventing_collection(
        tmp_path, consumption, expected):
    root = tmp_path/'research'
    path = root/'measurements/baseline.json'
    atomic_json(path, {'label': 'baseline', 'ok': True, 'metric_value': 0,
                       'data_consumption': consumption})
    before = path.read_bytes()
    evidence = summaries(root)
    assert evidence[0]['data_consumption_status'] == expected
    assert 'private_samples' not in json.dumps(evidence)
    assert 'DO_NOT_EXPORT' not in json.dumps(evidence)
    assert path.read_bytes() == before


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
        sources={'train':'def stage_argv_train(i):\n    return [str(i["steps"])]',
                 'evaluate':'def stage_argv_evaluate(i):\n    return [str(i["episodes"])]'}
        def _inputs(self,stage,*,settings):
            return {'steps':settings.get('steps',0), 'episodes':settings.get('episodes',1)}
        def run(self,**kwargs):
            seen.append(kwargs)
            atomic_json(self.run_root/'controller_session.json',{'run_id':real.run_id,
                'repository':str(real.repo),'status':'paused','base_settings':kwargs['settings'],
                'rounds':kwargs['rounds']})
            atomic_json(self.run_root/'research_report.json',{'run_id':real.run_id,
                'repo':str(real.repo),'run_status':'paused'})
            return {'rounds':[],'run_status':'paused','objective':{}}
    monkeypatch.setattr(real,'_research_controller',lambda:(Controller(),{}))
    refused = real._step_run_the_loop(evaluation_settings={'episodes':100,'seed':0})
    assert refused['outcome']=='not attempted' and not seen
    assert refused['failure_category']=='formal_training_input_required'
    assert refused['native_operation_status']=='not_started'
    assert refused['repair_owner']=='main_agent'
    assert refused['evidence_id'] and refused['evidence_ref']
    assert 'evaluation_settings alone' in refused['repair_handoff']['next_action']
    sealed=read_json(real.output/refused['evidence_ref'])
    assert sealed['evidence_id']==refused['evidence_id']
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
    result=real.do('run_the_loop',training_settings={'steps':8192},
                   evaluation_settings={'episodes':50,'seed':0})
    assert result['outcome']=='done' and seen==[{'training_settings':{'steps':8192},
                                               'evaluation_settings':{'episodes':50,'seed':0}}]


@pytest.mark.parametrize('value',[None,0,-1,True,1.0,'50'])
def test_formal_evaluation_work_requires_positive_integral_episodes(value):
    from autosim.research.training_work import missing_formal_evaluation_work
    source='def score(i):\n    return [str(i.get("episodes"))]'
    assert missing_formal_evaluation_work(source,{'episodes':value})
    assert not missing_formal_evaluation_work(source,{'episodes':50})
    assert not missing_formal_evaluation_work('def score(i):\n    return ["native-config"]',{})


def evaluation_controller(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research.adapter_protocol import OptimizationSpace
    real=preparation(tmp_path)
    monkeypatch.setattr(real,'_research_loop_ready',lambda:True)
    monkeypatch.setattr(real,'_score_target',lambda:'evaluate')
    seen=[]
    class Controller:
        run_root=real.output/'research'/real.run_id
        execution_graph=None
        space=OptimizationSpace()
        sources={'evaluate':'def score(i):\n    return [str(i["episodes"]), str(i["seed"])]'}
        def _inputs(self,stage,*,settings):return {'episodes':settings.get('eval.n_eval',settings.get('episodes',0))}
        def run(self,**kwargs):
            seen.append(kwargs)
            return {'rounds':[],'run_status':'paused','objective':{}}
    monkeypatch.setattr(real,'_research_controller',lambda:(Controller(),{}))
    return real,Controller,seen


def test_evaluation_work_is_chosen_before_any_formal_training(tmp_path,monkeypatch):
    real,_,seen=evaluation_controller(tmp_path,monkeypatch)
    result=real._step_run_the_loop()
    assert result['failure_category']=='evaluation_input_required' and not seen
    assert result['native_operation_status']=='not_started'
    real._step_run_the_loop(evaluation_settings={'episodes':50,'seed':0})
    assert seen[-1]['settings']=={'episodes':50,'seed':0}
    assert real.base_settings=={}  # Never mutate original CLI settings.


@pytest.mark.parametrize('settings',[{},[],{'episodes':0},{'episodes':True},
    {'episodes':'50'},{'episodes':1.5},{'seed':-1},{'horizon':400},{'metric':'fake'},
    {'task':'easier'},{'episodes':50,'threshold':0}])
def test_evaluation_settings_refuse_invalid_or_scientific_rule_controls(tmp_path,monkeypatch,settings):
    real,_,seen=evaluation_controller(tmp_path,monkeypatch)
    assert real._step_run_the_loop(evaluation_settings=settings)['outcome']=='not attempted'
    assert not seen


@pytest.mark.parametrize('explicit',[{'episodes':10},{'eval.n_eval':10},{'seed':5}])
def test_evaluation_settings_preserve_explicit_user_settings(tmp_path,monkeypatch,explicit):
    real,_,seen=evaluation_controller(tmp_path,monkeypatch)
    real.base_settings=explicit
    result=real._step_run_the_loop(evaluation_settings={'episodes':50,'seed':0})
    assert 'conflicts' in result['because'] and not seen


def test_evaluation_settings_cannot_rewrite_protocol_even_without_session(tmp_path,monkeypatch):
    real,controller,seen=evaluation_controller(tmp_path,monkeypatch)
    atomic_json(controller.run_root/'comparison_protocol.json',{'frozen':True})
    result=real._step_run_the_loop(evaluation_settings={'episodes':50})
    assert 'frozen' in result['because'] and not seen


def test_evaluation_count_cannot_be_silently_ignored_by_native_configuration(tmp_path,monkeypatch):
    real,controller,seen=evaluation_controller(tmp_path,monkeypatch)
    controller.sources={'evaluate':'def score(i):\n    return ["fixed-config"]'}
    result=real._step_run_the_loop(evaluation_settings={'episodes':50})
    assert 'does not consume' in result['because'] and not seen


def test_evaluation_alias_uses_only_dedicated_input_channel(tmp_path):
    from tests.test_derived_research import research
    real=research(tmp_path)
    inputs=real._inputs('evaluate',settings={'eval.n_eval':50,'seed':0})
    assert inputs['episodes']==50 and 'eval.n_eval' not in inputs['settings']


def test_evaluation_guidance_keeps_main_prompt_template_renderable():
    from autosim.research.prepare import STEP_SYSTEM, OPERATIONS
    prompt=STEP_SYSTEM.format(operations=json.dumps(OPERATIONS))
    assert 'evaluation_settings={episodes: positive integer' in prompt


def test_requested_confirmation_is_declared_before_any_baseline_or_native_work(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tests.test_prepare import preparation
    real=preparation(tmp_path)
    real.client=SimpleNamespace(supports_main_agent=True)
    real.base_settings={'_confirm':True}
    real.declaration={'benchmark':'toy'}
    monkeypatch.setattr(real,'_research_controller',lambda:pytest.fail('no formal work before reservation'))
    result=real._step_run_the_loop()
    assert result['failure_category']=='confirmation_protocol_required'
    assert result['native_operation_status']=='not_started'
    # Missing reservation is repaired through the reviewed pre-search operation,
    # not repeated whole-repository discovery. The native-work guard stays intact.
    assert 'configure_confirmation_protocol' in result['because']
    assert real._declaration_refresh_is_justified() is False
    assert real._declaration_refresh_is_justified(research_progress={'status':'paused'}) is False
