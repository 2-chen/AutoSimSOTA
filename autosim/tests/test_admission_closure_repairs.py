"""Controller closure: native readiness, review evidence, pause, and producer reuse."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosim.research.common import atomic_json, digest, read_json
from autosim.research.evidence_store import read_attempt_evidence
from autosim.research.main_agent import memory_view, validate_plan
from autosim.research.operation_contracts import schema, validate
from autosim.research.runtime_freshness import controller_context, controller_identity
from tests.test_prepare import preparation


def test_confirmation_is_early_admission_not_missing_native_capability(tmp_path,monkeypatch):
    real = preparation(tmp_path)
    real.client.supports_main_agent=True
    real.base_settings['_confirm']=True
    real.declaration={'benchmark':'Toy'}
    real.interpreter=Path('/usr/bin/python3')
    real.stages={'evaluate':'verified'}
    monkeypatch.setattr(real,'_metric_bound',lambda:True)
    monkeypatch.setattr(real,'_native_identity_schema_view',lambda *a:{'status':'compatible'})
    state = real.state()
    assert state['to_measure']['native_ready'] is True
    assert state['to_measure']['ready'] is False
    assert state['to_measure']['repair_owner']=='objective'
    assert 'run_the_loop' not in state['available']
    assert 'configure_confirmation_protocol' in state['available']
    assert not real._research_loop_ready()
    refused=real._step_run_the_loop()
    assert refused['repair_handoff']['owner']=='objective'
    proof=read_attempt_evidence(real.output,refused['evidence_id'])
    assert proof['returncode'] is None
    assert json.loads(proof['text'])['native_operation_status']=='not_started'


def test_rejected_scientific_review_keeps_failed_flags_reason_and_owner(tmp_path):
    real=preparation(tmp_path)
    (tmp_path/'evaluate.py').write_text('return constant_state\n')
    real.client.choices=[{'native_evaluation_input_supported':True,'disjoint_conditions_supported':False,
        'preserves_policy_and_metric':True,'source_connection_supported':True,
        'why':'reset never consumes the evaluation seed',
        'citations':[{'ref':'evaluate.py','quote':'return constant_state'}]}]
    result=real._step_configure_confirmation_protocol(confirmation={'seed':101},
        source_refs=[{'ref':'evaluate.py','quote':'return constant_state'}],reason='native reset review')
    proof=read_attempt_evidence(real.output,result['evidence_id'],limit=12000)
    context=json.loads(proof['text'])['context']
    assert context['failed_checks']==['disjoint_conditions_supported']
    assert context['review']['why']=='reset never consumes the evaluation seed'
    assert result['repair_handoff']['evidence_id']==result['evidence_id']
    assert result['repair_owner']=='objective' and proof['returncode'] is None
    assert not (real.output/'confirmation_reservation.json').exists()
    current=controller_context(real.output,[{'step':'configure_confirmation_protocol',**result}])
    assert current['previous_action_failures'][0]['validity']=='observed_in_current_runtime'


@pytest.mark.parametrize('scope',['pause','framework'])
def test_agent_pause_cannot_bypass_one_independent_review(tmp_path,scope):
    from autosim.research.framework_stop_review import review
    calls=[]
    client=SimpleNamespace(supports_main_agent=True,as_role=lambda role:nullcontext(),timeout=5,
        chat_with_metadata=lambda *a,**kw:(calls.append(kw) or json.dumps({'approved':False,
            'why':'inspect the current native reset connection before declaring it impossible'}),{}))
    controller=SimpleNamespace(output=tmp_path,repo=tmp_path,budget=None,client=client,
        last_action={},state=lambda:{'available':['configure_confirmation_protocol']})
    choice={'do':'stop','arguments':{'stop_scope':scope},'why':'historical attempts exhausted'}
    first=review(controller,choice)
    assert first['decision_failure']['kind']=='framework_stop_unproven'
    second=review(controller,choice)
    assert 'decision_failure' not in second and second['arguments']['stop_scope']=='pause'
    assert len(calls)==1


def test_hard_wall_boundary_never_buys_stop_review(tmp_path):
    from autosim.research.framework_stop_review import review
    controller=SimpleNamespace(client=SimpleNamespace(supports_main_agent=True),
        budget=SimpleNamespace(remaining=lambda:0))
    choice={'do':'stop','arguments':{'stop_scope':'pause'}}
    assert review(controller,choice)==choice


def test_plan_write_projection_roundtrips_without_authority_and_reports_bounds(tmp_path):
    plan={'objective':'connect paid policy to native rollout','hypotheses':[],
        'open_questions':[],'next_actions':['inspect producer'],'evidence_refs':[],
        'data_strategy':{'route':'existing','why':'reuse','targets':[],
            'producer_to_loader':[],'evidence_refs':[],'next_probe':'inspect actual batches'}}
    held=validate_plan(plan)
    writable=memory_view({'plan':held})['writable_plan']
    assert validate_plan(writable)==held
    real=preparation(tmp_path)
    validate(schema(real._step_update_research_plan),{'plan':writable})
    with pytest.raises(ValueError,match='authority'):
        validate_plan(held)
    oversized={**plan,'next_actions':['中'*1400]*5}
    with pytest.raises(ValueError,match='UTF-8 bytes'):
        validate(schema(real._step_update_research_plan),{'plan':oversized})


def test_native_source_quote_is_executor_capture_not_model_print(tmp_path,monkeypatch):
    from autosim.research import native_source_citations as sources
    from autosim.research.baseline_reference import source_evidence
    repo=tmp_path/'checkout';repo.mkdir()
    prefix=tmp_path/'environment';path=prefix/'lib/reset.py';path.parent.mkdir(parents=True)
    path.write_text('def reset(rng):\n    return rng.uniform()\n')
    context={'identity':'selected-native-context','interpreter':str(prefix/'bin/python')}
    monkeypatch.setattr(sources,'load_context',lambda *a:context)
    captured=sources.capture(tmp_path,repo,{'root_index':0,'path':'lib/reset.py'})
    assert read_attempt_evidence(tmp_path,captured['evidence_id'])['text']==path.read_text()
    refs=[{'ref':captured['ref'],'quote':'return rng.uniform()'}]
    assert source_evidence(repo,refs,output=tmp_path)[0]['sha256']==digest(path)
    for bad in ('../secret.py','/etc/file.py','lib/secret.py'):
        with pytest.raises(ValueError):sources.capture(tmp_path,repo,{'root_index':0,'path':bad})
    with pytest.raises(ValueError):
        source_evidence(repo,[{'ref':'native-source:'+'a'*32,'quote':'fake printed source'}],output=tmp_path)
    original=path.read_text()
    path.write_text(original+'# changed\n')
    with pytest.raises(ValueError,match='changed'):
        source_evidence(repo,refs,output=tmp_path)
    path.write_text(original)
    context['identity']='new-environment'
    with pytest.raises(ValueError,match='environment changed'):
        source_evidence(repo,refs,output=tmp_path)


def test_initial_job_adoption_allows_consumer_only_inputs_without_retraining(tmp_path,monkeypatch):
    from tests.test_derived_research import ambiguous_policy_research
    from autosim.research import native_jobs
    research=ambiguous_policy_research(tmp_path)
    job_id='b'*32;research.controller_decision_id=job_id
    producer_settings={'work_tag':'original'}
    trained=research.run_stage('train',settings=producer_settings)
    atomic_json(research.output/'native_jobs'/job_id/'request.json',{
        'stage':'train','repo':str(research.repo),'run_id':research.run_id,
        'settings':producer_settings,'source_identity':'c'*64})
    monkeypatch.setattr(native_jobs,'status',lambda *a:{'status':'completed','result':{'stage_result':trained}})
    monkeypatch.setattr(native_jobs,'source_identity',lambda *a:'c'*64)
    real=preparation(tmp_path)
    real.declaration={'benchmark':'synthetic'};real.interpreter=research.interpreter
    real.stages={'train':'verified','evaluate':'verified'}
    real.base_settings={'episodes':2,'seed':0}
    monkeypatch.setattr(real,'_research_loop_ready',lambda:True)
    monkeypatch.setattr(real,'_research_controller',lambda:(research,{}))
    native=research.run_stage
    def no_retraining(stage,**kwargs):
        assert stage!='train','completed paid producer was executed twice'
        return native(stage,**kwargs)
    monkeypatch.setattr(research,'run_stage',no_retraining)
    original=(research.run_root/'attempts'/trained['attempt_id']/'receipt.json').read_bytes()
    result=real._step_run_the_loop(job_id=job_id)
    measured=read_json(research.run_root/'measurements/baseline.json')
    assert measured['ok'] and measured['metric_value']==0.75
    assert measured['train']['attempt_id']==trained['attempt_id']
    assert measured['train']['reused']
    assert (research.run_root/'attempts'/trained['attempt_id']/'receipt.json').read_bytes()==original


def test_native_source_is_available_through_actual_tool_and_reserved_protocol(tmp_path):
    from tests.test_main_agent import make
    from autosim.research.agent_runtime import _McpExecutor
    from autosim.research.native_context import publish_context
    from autosim.research import confirmation_reservation
    real=make(tmp_path)
    prefix=real.output/'environment';python=prefix/'bin/python';python.parent.mkdir(parents=True)
    python.write_text('native-source-only-fixture')
    sdk=prefix/'lib/sdk/reset.py';sdk.parent.mkdir(parents=True)
    sdk.write_text('def reset(stream):\n    return stream.uniform()\n')
    atomic_json(real.output/'workspace_snapshot.json',{
        'source':str(real.repo),'destination':str(real.repo),'isolated':True,
        'schema_version':2,'source_entries':[]})
    publish_context(real.output,real.repo,python,{})
    executor=_McpExecutor(workspace=real.repo,output=real.output,allow_commands=False,allow_native=True)
    captured=executor.native_probe({'purpose':'inspect installed reset source',
        'source_request':{'root_index':0,'path':'lib/sdk/reset.py'}})
    refs=[{'ref':captured['ref'],'quote':'return stream.uniform()'}]
    assert executor.read_evidence({'evidence_id':captured['evidence_id']})['returncode'] is None
    proposal={'confirmation':{'seed':101},'reason':'source reviewed native stream','protocol_keys':[],
        'source_refs':refs}
    review={'native_evaluation_input_supported':True,'disjoint_conditions_supported':True,
        'preserves_policy_and_metric':True,'source_connection_supported':True,'citations':refs}
    confirmation_reservation.record(real.output,real.repo,real.run_id,proposal,review)
    assert confirmation_reservation.apply(real.output,real.repo,real.run_id,{'benchmark':'Toy'})[
        'research_goal']['confirmation']=={'seed':101}
    sdk.write_text('def reset(stream):\n    return constant_state\n')
    with pytest.raises(ValueError,match='changed'):
        confirmation_reservation.apply(real.output,real.repo,real.run_id,{'benchmark':'Toy'})


def test_changed_consumed_training_work_is_not_reusable(tmp_path):
    from tests.test_derived_research import research as make_research
    research=make_research(tmp_path)
    job_id='c'*32;research.controller_decision_id=job_id
    # A real builder input that changes the producer command, not a key blacklist.
    from autosim.research.declarative_backend import DeclarativeBackend
    changed={**research.backend.sources,
        'train':'def stage_argv_train(i):\n    return [i["python"], "-c", "pass", str(i["seed"])]\n'}
    research.backend=DeclarativeBackend(repo=research.repo,answer=research.backend.answer,
        sources=changed,parameters=research.backend.parameters)
    trained=research.run_stage('train',settings={'seed':0})
    atomic_json(research.output/'native_jobs'/job_id/'request.json',{
        'stage':'train','repo':str(research.repo),'run_id':research.run_id,'settings':{'seed':0}})
    receipt=read_json(research.run_root/'attempts'/trained['attempt_id']/'receipt.json')
    assert not research._training_settings_match(receipt,{'seed':1,'episodes':100})
