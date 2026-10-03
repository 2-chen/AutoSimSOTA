"""Environment consumers precede artifact proof, without weakening scoring provenance."""
import json

import pytest

from autosim.research import provision
from autosim.research.common import read_json
from tests.test_main_agent import make as preparation
from tests.test_environment_overlay import fixture, make as overlay, provision_fixture
from tests.test_environment_and_budget_alignment import Planner


CHOICE={'stages':['train','evaluate'],'asset_keys':[],
        'why':'native producer and evaluator; inspect external SDK after environment exists'}


def test_scheduler_source_reference_guidance_renders_as_actual_template():
    from autosim.research.prepare import STEP_SYSTEM
    rendered=STEP_SYSTEM.format(operations='fixture operations')
    assert 'citations {ref,quote}' in rendered
    assert 'objects may use path/file/source/ref' in rendered


def graph(obj):
    obj.declaration={'benchmark':'toy','assets':{},'task_contract':{'policy_representation':'artifact'}}
    obj.execution={'stages':{
        'train':{'available':True,'entrypoint':'train.py','artifact':'model.pt'},
        'evaluate':{'available':True,'entrypoint':'eval.py','invocation':'python eval.py --checkpoint model.pt'}}}
    (obj.repo/'train.py').write_text('from external_sdk import save_policy\n')
    (obj.repo/'eval.py').write_text('from external_sdk import load_policy\n')


def test_environment_graph_does_not_buy_or_require_future_artifact_proof(tmp_path,monkeypatch):
    obj=preparation(tmp_path);graph(obj)
    obj.client=Planner(obj.output,[CHOICE])
    monkeypatch.setattr(obj,'_review_policy_handoff',lambda *a:pytest.fail('not an environment prerequisite'))
    answer=obj._select_execution_path(require_handoff=False)
    assert answer['handoff_review']['compatible'] is None
    assert answer['handoff_review']['status']=='pending_evidence'
    state=obj.state()['workflow_selection']
    assert state['input_digest']==answer['input_digest']
    assert state['status']=='selected_pending_handoff'
    assert obj._select_execution_path(require_handoff=False)==answer
    assert len(obj.client.payloads)==1
    (obj.repo/'eval.py').write_text('from external_sdk import different_policy\n')
    assert obj.state()['workflow_selection']['status']=='selected_for_different_inputs'


def test_provisional_environment_graph_is_not_reused_as_strict_handoff_proof(tmp_path,monkeypatch):
    obj=preparation(tmp_path);graph(obj)
    obj.client=Planner(obj.output,[CHOICE]*4)
    obj._select_execution_path(require_handoff=False)
    monkeypatch.setattr(obj,'_review_policy_handoff',lambda *a:{'compatible':None,
        'status':'unknown','why':'external SDK implementation not yet read'})
    with pytest.raises(provision.EnvironmentPlanError) as held:
        obj._select_execution_path(require_handoff=True)
    assert held.value.planning_failure['repair_owner']=='workflow_planner'
    assert (obj.output/held.value.planning_failure['evidence_ref']).is_file()
    assert read_json(obj.output/'selected_path.json')['handoff_review']['compatible'] is None


def test_bad_graph_is_typed_workflow_failure_not_native_install_or_checkout_bug(tmp_path,monkeypatch):
    obj=preparation(tmp_path);graph(obj)
    obj.client=Planner(obj.output,[{'stages':['train'],'asset_keys':[],'why':'missing score'}]*3)
    monkeypatch.setattr(provision,'build',lambda *a,**k:pytest.fail('invalid graph cannot execute'))
    result=obj.do('build_the_environment')
    assert result['failure_domain']=='framework_plan'
    assert result['repair_owner']=='workflow_planner'
    assert result['evidence_id'] and (obj.output/result['evidence_refs'][0]).is_file()


def test_workflow_planning_rejection_does_not_launch_checkout_fix(tmp_path,monkeypatch):
    obj=preparation(tmp_path);graph(obj)
    obj.client=Planner(obj.output,[{'stages':['train'],'asset_keys':[],'why':'missing score'}]*3)
    choices=iter([{'do':'build_the_environment','why':'verify consumers','arguments':{}},
                  {'do':'stop','why':'invalid graph requires planner correction',
                   'arguments':{'stop_scope':'framework'}}])
    monkeypatch.setattr(obj,'choose',lambda:next(choices))
    monkeypatch.setattr(obj,'_fix_failed_action',lambda **kw:pytest.fail('checkout Fix cannot repair a planning rejection'))
    monkeypatch.setattr(obj,'_monitor_failed_action',lambda **kw:pytest.fail('typed planner feedback needs no Monitor'))
    result=obj._run_cycle(max_steps=2)
    assert result['steps'][0]['failure_domain']=='framework_plan'
    assert result['steps'][0]['repair_owner']=='workflow_planner'


def test_review_reason_changes_do_not_invalidate_actual_source_evidence(tmp_path):
    obj=preparation(tmp_path);graph(obj)
    obj.client=Planner(obj.output,[{'compatible':True,'status':'compatible','why':'matching native SDK format'}])
    rows=obj.execution['stages']
    obj._review_policy_handoff(rows,['train','evaluate'],'evaluate','first phrasing')
    assert obj._review_policy_handoff(rows,['train','evaluate'],'evaluate','second phrasing')['reused']
    assert len(obj.client.payloads)==1


def test_missing_source_evidence_is_unknown_not_proved_incompatible(tmp_path):
    obj=preparation(tmp_path);graph(obj)
    obj.client=Planner(obj.output,[{'compatible':False,'why':'external library not available in excerpt'}])
    review=obj._review_policy_handoff(obj.execution['stages'],['train','evaluate'],'evaluate','inspect SDK')
    assert review['status']=='unknown' and review['compatible'] is None


def test_invalid_checkout_identity_is_visible_without_crashing_recovery_state(tmp_path):
    from autosim.research.common import atomic_json
    obj=preparation(tmp_path);graph(obj)
    atomic_json(obj.output/'workspace_snapshot.json',{'schema_version':2,'destination':'wrong'})
    view=obj.state()['workflow_selection']
    assert view['status']=='invalid_source_identity'
    assert view['identity_error'] and view['input_digest']==''


def test_onboarding_keeps_free_document_without_paid_recorder(tmp_path,monkeypatch):
    from autosim.research import recorder
    obj=preparation(tmp_path);graph(obj)
    obj.interpreter=None
    obj.steps=[{'step':'build_the_environment','outcome':'raised','because':'not verified yet'}]
    writers=[]
    native_refresh=recorder.refresh
    def record_writer(*args,**kwargs):
        writers.append(kwargs.get('client'))
        return native_refresh(*args,**kwargs)
    monkeypatch.setattr(recorder,'refresh',record_writer)
    obj._refresh_document(status='running',current='build_the_environment')
    assert writers==[None]
    assert (obj.output/'RUN.md').is_file()


def test_real_cpu_consumer_can_pass_before_any_trained_checkpoint_exists(fixture,monkeypatch):
    base,site,output,repo,version=fixture
    env=overlay(fixture)
    (repo/'consumer.py').write_text('import dependency,task\nassert dependency.VALUE==73\nassert task.VALUE=="current"\n')
    (repo/'train.py').write_text('import dependency,task\n')
    (repo/'evaluate.py').write_text('import dependency,task\n')
    (repo/'environment-data').mkdir()
    (repo/'environment-data/sample.txt').write_text('synthetic consumer fixture, not benchmark data')
    native_run=provision.run
    row,plan=provision_fixture(fixture,monkeypatch)
    from autosim.research import environment_pool
    monkeypatch.setattr(environment_pool,'catalog',lambda *a,**kw:[row])
    def consumer_only(command,**kwargs):
        assert ('consumer.py' in command or 'environment-data' in command) and 'install' not in command
        return native_run(command,**kwargs)
    monkeypatch.setattr(provision,'run',consumer_only)
    plan.update(python='existing',base_environment_mode='existing',commands=[],
        probes=['{python} {repo}/consumer.py'], assets=[{'what':'fixture input',
            'where':'{repo}/environment-data','produced_by':'already present'}])
    plan.pop('base_environment_id')
    from autosim.research.prepare import Preparation
    from autosim.research.compute_decision import ComputeDecision
    obj=Preparation(repo=repo,output=output,scouting=output/'scouting',
                    client=Planner(output,[CHOICE]))
    obj.declaration={'benchmark':'toy','assets':{},'task_contract':{'policy_representation':'artifact'}}
    obj.execution={'stages':{'train':{'available':True,'entrypoint':'train.py','artifact':'model.pt'},
        'evaluate':{'available':True,'entrypoint':'evaluate.py','invocation':'python evaluate.py --checkpoint model.pt'}}}
    obj.interpreter=None
    obj.decision=ComputeDecision(device='cpu',device_index=0,environment={},evidence={})
    from autosim.research.common import atomic_json
    atomic_json(output/'environment.json',{'interpreter':env['interpreter'],'verdict':{'passed':False}})
    monkeypatch.setattr(obj,'_review_policy_handoff',lambda *a:pytest.fail('future weights cannot block consumers'))
    result=obj.do('build_the_environment',max_operations=1)
    # Cooperative mode may yield after the real probe; then finalize from its cursor.
    if result['outcome']=='checkpoint':
        result=obj.do('build_the_environment',max_operations=1)
    assert result['outcome']=='done', result
    assert read_json(output/'environment.json')['verdict']['passed']
    assert not (repo/'model.pt').exists()
    assert obj.interpreter and 'derive_a_command' in obj._available_operations()
    assert obj.state()['environment']['probe_contract_faults']==[]


def test_only_exact_generated_asset_checks_get_supplemental_scope(tmp_path):
    context={'stage_paths':{'train':{'entrypoint':'train.py','artifact':'model.pt'}}}
    assets=[{'where':'data','what':'input','produced_by':'already present'}]
    check=provision.asset_probe('data')
    assert provision.consumer_or_asset_violation(check,context,assets)==''
    assert provision.probe_stage_violation(check,context)
    assert provision.consumer_or_asset_violation(check+'; true',context,assets)
    assert provision.consumer_or_asset_violation('echo ready',context,assets)
    future=provision.asset_probe('model.pt')
    assert provision.consumer_or_asset_violation(future,context,[{'where':'model.pt'}])


def test_sealed_review_paths_stay_distinct_before_display_redaction(tmp_path):
    from autosim.research.common import atomic_text
    from autosim.research.evidence_store import capture_attempt_evidence,read_attempt_evidence
    from autosim.research.execution_paths import ExecutionPaths
    output=tmp_path/'run';repo=output/'checkout';repo.mkdir(parents=True)
    prefix=output/'env';prefix.mkdir()
    paths=ExecutionPaths(repo,output)
    alias=paths.bind_run_reference('env')
    log=output/'failure.log'
    atomic_text(log,f"{prefix}/bin/python: can't open file '{repo}/scripts/train.py'\nAPI_KEY=secret-value\n")
    evidence='e'*32
    capture_attempt_evidence(output,attempt_id=evidence,log=log,receipt_ref='receipt.json',
        status='failed',returncode=2,termination_reason='fixture')
    text=read_attempt_evidence(output,evidence,path_encoder=lambda text:
        paths.encode_text(text).replace(str(repo),'{repo}'))['text']
    assert alias+'/bin/python' in text and '{repo}/scripts/train.py' in text
    assert str(tmp_path) not in text and 'secret-value' not in text
    from autosim.research.common import sanitize_model_payload
    final=sanitize_model_payload(sanitize_model_payload({'text':text}))['text']
    assert alias+'/bin/python' in final and '{repo}/scripts/train.py' in final
    assert '[LOCAL_PATH]' not in final


def test_only_known_execution_handle_suffixes_survive_path_projection():
    from autosim.research.common import sanitize_model_payload_text
    text=sanitize_model_payload_text('{repo}/scripts/train.py {run_path_1}/bin/python '
        '{unknown}/opt/private.py /opt/private.py {repo}/../private.py API_KEY=secret-value')
    assert '{repo}/scripts/train.py' in text and '{run_path_1}/bin/python' in text
    assert '{unknown}[LOCAL_PATH]' in text
    assert '/opt/private.py' not in text and '../private.py' not in text
    assert 'secret-value' not in text


def test_same_selected_overlay_continues_cursor_without_replanning(fixture,monkeypatch):
    base,site,output,repo,version=fixture
    row,planned=provision_fixture(fixture,monkeypatch)
    planned['probes']=["{python} -c 'import dependency,task'", "{python} -c 'import dependency'"]
    kwargs=dict(client=None,prefix=output/'env',output=output,manifests={},assets={},
        max_operations=1,base_environment_id=row['id'],base_environment_mode='overlay',
        source_binding_ids=[],environment_selection_reason='reuse compatible base')
    first=provision.build(repo,**kwargs)
    assert first['status']=='yielded'
    interpreter=first['interpreter']
    monkeypatch.setattr(provision,'plan',lambda *a,**kw:pytest.fail('same selection must not buy another plan'))
    second=provision.build(repo,**{**kwargs,'environment_selection_reason':'continue current cursor, new wording'})
    assert second['verdict']['passed'], second
    assert second['interpreter']==interpreter


def test_explicitly_cleared_probe_failure_is_not_restored_from_history(fixture,monkeypatch):
    from autosim.research.common import atomic_json
    base,site,output,repo,version=fixture
    row,planned=provision_fixture(fixture,monkeypatch)
    planned['probes']=["{python} -c 'import dependency,task'", "{python} -c 'import dependency'"]
    kwargs=dict(client=None,prefix=output/'env',output=output,manifests={},assets={},
        max_operations=1,base_environment_id=row['id'],base_environment_mode='overlay',
        source_binding_ids=[],environment_selection_reason='reuse compatible base')
    first=provision.build(repo,**kwargs)
    assert first['status']=='yielded'
    cursor=read_json(output/'provision_cursor.json')
    assert cursor['failed_probe']=={} and cursor['next_probe']==1
    transcript=read_json(output/'transcript.json')
    transcript['rows'].append({'kind':'probe','probe':planned['probes'][1],
        'ok':False,'evidence_id':'f'*32,'excerpt':'historical native fault, cleared by repair'})
    atomic_json(output/'transcript.json',transcript)
    monkeypatch.setattr(provision,'resume',lambda *a,**k:pytest.fail('cleared historical failure must not call Fix'))
    second=provision.build(repo,**kwargs)
    assert second['verdict']['passed'], second


def test_borrowed_runtime_libraries_are_shared_and_cannot_be_forged(fixture):
    from pathlib import Path
    import subprocess
    from autosim.research import environment_overlay as module
    from autosim.research.native_context import publish_context,load_context
    from autosim.research.common import atomic_json,object_digest,run_local_environment
    base,site,output,repo,version=fixture
    env=overlay(fixture)
    libraries=module.runtime_library_dirs(Path(env['interpreter']).parent.parent,output,repo)
    if not libraries:
        pytest.skip('system Python uses system runtime; no external native library closure')
    row=publish_context(output,repo,Path(env['interpreter']),{})
    assert row['runtime_library_dirs']==libraries
    context=run_local_environment(output,{'PATH':'/usr/bin:/bin'})
    assert context['LD_LIBRARY_PATH'].split(':')==libraries
    code=("import ctypes,pathlib; ctypes.CDLL('libstdc++.so.6'); "
          "loaded=pathlib.Path('/proc/self/maps').read_text(); "
          f"assert any(p in loaded for p in {libraries!r}); print('native runtime closure ok')")
    result=subprocess.run([env['interpreter'],'-c',code],env=context,capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='native runtime closure ok'
    from autosim.research.agent_runtime import _McpExecutor
    atomic_json(output/'agent_fixture.json',{'workspace':str(repo)})
    diagnostic=_McpExecutor(workspace=repo,output=output).native_probe(
        {'code':code,'purpose':'verify identical read-only native loader closure'})
    assert diagnostic['returncode']==0, diagnostic
    assert diagnostic['stdout'].strip()=='native runtime closure ok'
    row['runtime_library_dirs']=['/tmp/unapproved-library']
    row['identity']=object_digest({k:v for k,v in row.items() if k!='identity'})
    atomic_json(output/'native_context.json',row)
    with pytest.raises(ValueError,match='loader bindings'):
        load_context(output,repo)
    with pytest.raises(ValueError,match='loader bindings'):
        run_local_environment(output,{})
