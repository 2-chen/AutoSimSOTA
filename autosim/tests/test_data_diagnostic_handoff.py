"""Readonly workers must hand native evidence back through the same stable ID."""
import json
import sys
import shutil
from pathlib import Path

import pytest

from autosim.research import agent_runtime as runtime
from autosim.research.common import atomic_json, read_json, digest
from autosim.research.diagnostic_artifacts import data_binding, data_binding_unchanged
from autosim.research.evidence_store import capture_attempt_evidence, read_attempt_evidence
from autosim.research.process_executor import ProcessAttempt
from tests.test_agent_runtime import _fixture


def produced(tmp_path,*,marker=False,status='timed_out'):
    output,workspace=_fixture(tmp_path)
    atomic_json(output/'run_state.json',{'run_id':'derived'})
    data=output/'native_output'/'dataset';data.mkdir(parents=True)
    (data/'metadata.json').write_text('{"episodes":2}')
    ident='a'*32;ref='producer/receipt.json'
    artifact={'examples':['dataset/metadata.json' if marker else 'dataset']}
    context={'stage':'collect','working_directory':str(workspace),'output_directory':str(data.parent)}
    atomic_json(output/ref,{'attempt_id':ident,'stage':'collect','status':status,'artifact':artifact,**context})
    log=output/'producer/native.log';log.write_text('completed two episodes before timeout')
    capture_attempt_evidence(output,attempt_id=ident,log=log,receipt_ref=ref,status=status,
        returncode=0 if status=='completed' else None,
        termination_reason='normal_exit' if status=='completed' else 'deadline',artifact=artifact,native_context=context)
    return output,workspace,data,{'evidence_id':ident,'artifact_index':0}


def test_data_grant_binds_only_owned_artifact_and_preserves_failed_producer(tmp_path):
    output,_,data,request=produced(tmp_path)
    failed=output/'producer/receipt.json';before=digest(failed)
    grant=data_binding(output,request)
    assert Path(grant['path'])==data and grant['read_only'] is True
    assert data_binding_unchanged(output,grant)
    assert digest(failed)==before and read_json(failed)['status']=='timed_out'
    (data/'metadata.json').write_text('changed')
    assert not data_binding_unchanged(output,grant)


@pytest.mark.parametrize('fault',['host_path','escape','bool_index','unknown_index','parent_without_inheritance','heldout','wrong_stage'])
def test_data_grants_never_expose_unowned_or_finalized_material(tmp_path,fault):
    output,_,_,request=produced(tmp_path)
    if fault=='host_path':request['path']='/home/secret'
    elif fault=='escape':request['relative_path']='../..'
    elif fault=='bool_index':request['artifact_index']=True
    elif fault=='unknown_index':request['artifact_index']=5
    elif fault=='parent_without_inheritance':request['scope']='budget_parent'
    elif fault=='heldout':atomic_json(output/'research/derived/confirmation_attempts/a.json',{})
    else:
        p=output/'producer/receipt.json';row=read_json(p);row['stage']='evaluate';atomic_json(p,row)
    with pytest.raises((ValueError,OSError)):data_binding(output,request)


def test_worker_cpu_probe_has_one_global_evidence_id_and_worker_owned_files(tmp_path,monkeypatch):
    root,_,_,_=produced(tmp_path)
    worker=root/'agent_workers'/'w';workspace=worker/'checkout';workspace.mkdir(parents=True)
    atomic_json(worker/'agent_fixture.json',{'schema_version':1,'workspace':str(workspace)})
    atomic_json(worker/'worker_parent.json',{'output':str(root),'read_only':True})
    monkeypatch.setattr('autosim.research.native_context.load_context',lambda *a,**k:
        {'identity':'native-context','interpreter':sys.executable})
    class Stream:
        def fileno(self):return 0
        def close(self):pass
    monkeypatch.setattr(runtime,'_sandbox_argv',lambda *a,**k: (['native-fixture'],Stream()))
    monkeypatch.setattr(runtime,'run_process',lambda *a,**k:
        ProcessAttempt(launched=True,returncode=0,stdout='native-data-observation'))
    executor=runtime._McpExecutor(output=worker,workspace=workspace,allow_commands=False,allow_native=True)
    receipt=executor.native_probe({'code':'print("native-data-observation")','purpose':'inspect training data'})
    ident=receipt['evidence_id']
    assert not (worker/'evidence'/f'{ident}.json').exists()
    assert (root/'evidence'/f'{ident}.json').is_file()
    page=read_attempt_evidence(root,ident)
    assert page['receipt_ref'].startswith('agent_workers/w/native_diagnostics/')
    assert 'native-data-observation' in executor.read_evidence({'evidence_id':ident})['text']


def test_model_receives_manifest_summary_while_complete_native_evidence_is_retained(tmp_path,monkeypatch):
    from tests.test_data_health import dataset
    from autosim.research.data_health import lerobot_v2_profile
    root,_,_,_=produced(tmp_path)
    worker=root/'agent_workers'/'w';workspace=worker/'checkout';workspace.mkdir(parents=True)
    atomic_json(worker/'agent_fixture.json',{'schema_version':1,'workspace':str(workspace)})
    atomic_json(worker/'worker_parent.json',{'output':str(root),'read_only':True})
    data=dataset(workspace)
    profile=lerobot_v2_profile(data,include_manifest=True)
    monkeypatch.setattr('autosim.research.native_context.load_context',lambda *a,**k:
        {'identity':'native-context','interpreter':sys.executable})
    class Stream:
        def fileno(self):return 0
        def close(self):pass
    monkeypatch.setattr(runtime,'_sandbox_argv',lambda *a,**k: (['native-fixture'],Stream()))
    monkeypatch.setattr(runtime,'run_process',lambda *a,**k:
        ProcessAttempt(launched=True,returncode=0,stdout=json.dumps(profile)))
    executor=runtime._McpExecutor(output=worker,workspace=workspace,allow_commands=False,allow_native=True)
    result=executor.native_probe({'purpose':'native manifest','data_profile_request':
        {'format':'lerobot_v2.1','root':'data','include_manifest':True,'selection':{'require_split':None}}})
    assert 'episode_manifest' not in result['data_profile'] and not result['stdout']
    assert result['data_profile']['episode_manifest_summary']['episodes']==2
    assert 'selected_episode_ids' not in result['data_selection']
    evidence=read_attempt_evidence(root,result['evidence_id'])
    stored=read_json(root/evidence['receipt_ref'])
    assert len(stored['data_profile']['episode_manifest']['episodes'])==2
    assert stored['data_selection']['selected_episode_ids']==['0','1']


def test_data_alias_is_a_readonly_namespace_mount_with_no_gpu_grant(tmp_path,monkeypatch):
    output,workspace,data,request=produced(tmp_path)
    monkeypatch.setattr(runtime.shutil,'which',lambda name:'/usr/bin/bwrap')
    native={'interpreter':sys.executable,'dependency_roots':[],'paths':{},'library_dirs':[]}
    command,stream=runtime._sandbox_argv([sys.executable,'-c','pass'],output=output,workspace=workspace,
        cwd=workspace,native=native,diagnostic_data=data_binding(output,request))
    try:
        at=command.index(str(data));assert command[at-1]=='--ro-bind' and command[at+1]=='/tmp/autosim_data'
        assert 'AUTOSIM_DIAGNOSTIC_DATA' in command
        assert '/dev/nvidia0' not in command
        assert command[command.index('CUDA_VISIBLE_DEVICES')+1]==''
    finally:stream.close()


@pytest.mark.skipif(shutil.which('bwrap') is None,reason='bubblewrap unavailable')
def test_real_data_grant_reads_artifact_but_cannot_write_it_or_access_run_root(tmp_path):
    from autosim.research.process_executor import run_process
    output,workspace,data,request=produced(tmp_path)
    command,stream=runtime._sandbox_argv(['/usr/bin/python3','-c',
        "import os; from pathlib import Path; p=Path(os.environ['AUTOSIM_DIAGNOSTIC_DATA']); "
        "print((p/'metadata.json').read_text()); print(Path('/dev/nvidia0').exists()); "
        f"print(Path({str(output/'run_state.json')!r}).exists()); (p/'metadata.json').write_text('unauthorized')"],
        output=output,workspace=workspace,cwd=workspace,
        native={'interpreter':'/usr/bin/python3','dependency_roots':[],'paths':{},'library_dirs':[]},
        diagnostic_data=data_binding(output,request))
    try:
        result=run_process(command,cwd=workspace,env={'PATH':'/usr/bin:/bin'},timeout=10,
                           pass_fds=(stream.fileno(),))
    finally:stream.close()
    assert result.returncode != 0
    assert result.stdout.splitlines() == ['{"episodes":2}','False','False']
    assert 'Read-only file system' in result.stderr
    assert (data/'metadata.json').read_text()=='{"episodes":2}'
def marker_produced(tmp_path,*,status='timed_out'):
    output,workspace,data,request=produced(tmp_path,marker=True,status=status)
    return output,workspace,data,{**request,'dataset_root_relative':'dataset'}


def test_explicit_marker_directory_grant_preserves_existing_data_and_producer(tmp_path):
    output,_,data,request=marker_produced(tmp_path)
    paths=[output/'producer/receipt.json',data/'metadata.json'];before={p:digest(p) for p in paths}
    grant=data_binding(output,request)
    assert Path(grant['path'])==data and grant['read_only'] is True
    assert grant['boundary_marker']['sha256']==digest(data/'metadata.json')
    assert data_binding_unchanged(output,grant)
    assert all(digest(p)==sha for p,sha in before.items())
    (data/'metadata.json').write_text('changed')
    assert not data_binding_unchanged(output,grant)


@pytest.mark.parametrize('fault',['sibling','run_root','escape','absolute','combined_relative',
    'linked_root','linked_marker','working_directory_artifact','nonstring','wrong_owner'])
def test_marker_grant_never_broadens_to_unowned_roots_or_skips_identity(tmp_path,fault):
    output,_,data,request=marker_produced(tmp_path)
    if fault=='sibling':
        (data.parent/'sibling').mkdir();request['dataset_root_relative']='sibling'
    elif fault=='run_root':request['dataset_root_relative']='.'
    elif fault=='escape':request['dataset_root_relative']='../dataset'
    elif fault=='absolute':request['dataset_root_relative']=str(data)
    elif fault=='combined_relative':request['relative_path']='metadata.json'
    elif fault=='linked_root':
        (data.parent/'linked').symlink_to(data,target_is_directory=True);request['dataset_root_relative']='linked'
    elif fault=='linked_marker':
        p=data/'metadata.json';p.rename(data/'original.json');p.symlink_to(data/'original.json')
    elif fault=='working_directory_artifact':
        row=read_json(output/'producer/receipt.json');row['artifact']['found_beside_the_command']=True
        atomic_json(output/'producer/receipt.json',row)
    elif fault=='nonstring':request['dataset_root_relative']=True
    else:
        row=read_json(output/'producer/receipt.json');row['output_directory']=str(output/'unowned')
        atomic_json(output/'producer/receipt.json',row)
    with pytest.raises((ValueError,OSError)):data_binding(output,request)


@pytest.mark.parametrize('status',['completed','timed_out'])
def test_marker_producer_registration_requires_full_bound_manifest_not_marker_only(tmp_path,status):
    from autosim.research.data_lineage import _sealed_profile
    from autosim.research.common import object_digest
    output,_,data,request=marker_produced(tmp_path,status=status)
    grant=data_binding(output,request)
    binding={k:v for k,v in grant.items() if k!='path'}
    path=output/'native_diagnostics/profile.json'
    atomic_json(path,{'diagnostic_data_binding':binding,'diagnostic_data_binding_stable':True})
    profile={'scope':'full','funnel':{'format_valid_episodes':2}}
    witness={'receipt_ref':path.relative_to(output).as_posix(),'receipt_sha256':digest(path)}
    field,identity=grant['byte_identity_field'],grant['byte_identity']
    with pytest.raises(ValueError):
        _sealed_profile(output,request['evidence_id'],field,identity,producer=True,native=data)
    result,proof=_sealed_profile(output,request['evidence_id'],field,identity,
        producer=True,native=data,partial_profile=(profile,witness))
    assert result==profile and proof['recovery']==witness and proof['partial_output']==(status!='completed')
    bad=read_json(path);bad['diagnostic_data_binding']['request']['artifact_index']=1;atomic_json(path,bad)
    with pytest.raises(ValueError):
        _sealed_profile(output,request['evidence_id'],field,identity,producer=True,native=data,
            partial_profile=(profile,{**witness,'receipt_sha256':digest(path)}))
