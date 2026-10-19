import hashlib
import json
from pathlib import Path

import pytest

from autosim.research import producer_source_history as history
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.evidence_store import capture_attempt_evidence, read_attempt_evidence


def fixture(tmp_path, *, status='completed', returncode=0):
    output, repo = tmp_path/'run', tmp_path/'run/checkout'
    repo.mkdir(parents=True)
    name='policy/train.py'
    (repo/'policy').mkdir()
    original='def train():\n    return 1\n'
    after=original.replace('return 1','return 2')
    (repo/name).write_text(original)
    atomic_json(output/'workspace_snapshot.json', {'destination':str(repo),
        'source_entries':[['file',name,420,hashlib.sha256(original.encode()).hexdigest()]]})
    root=output/'research/derived'
    producer='a'*32
    receipt={'attempt_id':producer,'run_id':'derived','stage':'train',
             'status':status,'returncode':returncode}
    ref='research/derived/attempts/'+producer+'/receipt.json'
    atomic_json(output/ref, receipt)
    log=root/'train.log';log.write_text('complete\n')
    capture_attempt_evidence(output,attempt_id=producer,log=log,receipt_ref=ref,
        status=status,returncode=returncode,termination_reason='normal_exit' if returncode==0 else 'nonzero_exit')
    atomic_json(root/'controller_session.json',{'repository':str(repo)})
    atomic_json(root/'measurements/round_1.json',{'label':'round_1','ok':False,
        'status':'unscored','where':'policy_artifact','train':{'attempt_id':producer}})
    old_sha,new_sha=[hashlib.sha256(x.encode()).hexdigest() for x in (original,after)]
    blobs=root/'snapshots/blobs';blobs.mkdir(parents=True)
    (blobs/old_sha).write_text(original);(blobs/new_sha).write_text(after)
    atomic_json(root/'snapshots/snapshots.json',{'snapshots':[
        {'name':'before-round-1','files':{name:old_sha}},
        {'name':'round-1','files':{name:new_sha}}]})
    atomic_json(root/'code_changes/round-1.json',{'status':'applied','round':1,
        'snapshot':'before-round-1','patches':[{'file':name,'find':'return 1','replace':'return 2'}]})
    request={'ref':name,'producer_evidence_id':producer,'view':'patch'}
    return output,repo,request,old_sha,new_sha


def all_hashes(root):
    return {p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}


def test_pure_archive_read_survives_current_source_drift_without_installing(tmp_path):
    out,repo,request,old,new=fixture(tmp_path)
    (repo/'policy/train.py').write_text('unrelated new current source\n')
    before=all_hashes(out)
    row=history.load_view(out,repo,request)
    assert json.loads(row['text'])==[{'file':'policy/train.py','find':'return 1','replace':'return 2'}]
    assert row['postimage_sha256']==new and row['before_sha256']==old
    assert row['installed_status']=='not_asserted'
    assert 'candidate archive only' in row['execution_source_binding']
    assert all_hashes(out)==before


def test_source_range_reads_postimage_not_live_file(tmp_path):
    out,repo,request,_,_=fixture(tmp_path)
    row=history.load_view(out,repo,{**request,'view':'source','ref':'policy/train.py:2'})
    assert row['text']=='    return 2\n'
    assert row['line_range']=={'first':2,'last':2,'total':2,'next_line':None}
    assert (repo/'policy/train.py').read_text().endswith('return 1\n')


def test_capture_only_publishes_readonly_evidence_not_current_checkout_handle(tmp_path):
    out,repo,request,_,_=fixture(tmp_path)
    before=(repo/'policy/train.py').read_bytes()
    row=history.capture(out,repo,request)
    proof=read_attempt_evidence(out,row['evidence_id'])
    assert proof['status']=='historical_source_captured'
    assert json.loads(proof['text'])[0]['replace']=='return 2'
    assert row['historical_source_ref'].startswith('producer-source:')
    assert not row['historical_source_ref'].startswith('checkout-source:')
    assert 'does not apply' in row['guidance']
    assert (repo/'policy/train.py').read_bytes()==before


@pytest.mark.parametrize('fault', ['blob','patch','round','duplicate','producer','manifest','checkout','evaluation'])
def test_mutated_or_unbound_history_fails_closed(tmp_path,fault):
    out,repo,request,_,new=fixture(tmp_path)
    root=out/'research/derived'
    if fault=='blob':
        (root/'snapshots/blobs'/new).write_text('changed')
    elif fault in ('patch','round'):
        p=root/'code_changes/round-1.json';d=read_json(p)
        if fault=='patch':d['patches'][0]['replace']='return 9'
        else:d['round']=2
        atomic_json(p,d)
    elif fault=='duplicate':
        p=root/'snapshots/snapshots.json';d=read_json(p)
        d['snapshots'].append(d['snapshots'][-1]);atomic_json(p,d)
    elif fault=='producer':
        request['producer_evidence_id']='b'*32
    elif fault=='manifest':
        p=out/'workspace_snapshot.json';d=read_json(p);d['source_entries']=[];atomic_json(p,d)
    elif fault=='checkout':
        atomic_json(root/'controller_session.json',{'repository':str(tmp_path/'other')})
    elif fault=='evaluation':
        p=root/'attempts'/('a'*32)/'receipt.json';d=read_json(p);d['stage']='evaluate';atomic_json(p,d)
    with pytest.raises((ValueError,FileNotFoundError)):
        history.load_view(out,repo,request)


@pytest.mark.parametrize('ref', ['/etc/passwd','../secret.py','policy/../secret.py',
    'C:\\private.py','policy/train.py:0','policy/train.bin','policy/.env'])
def test_no_arbitrary_files_binary_resources_or_unsafe_refs(tmp_path,ref):
    out,repo,request,_,_=fixture(tmp_path)
    with pytest.raises(ValueError):
        history.load_view(out,repo,{**request,'ref':ref})


def test_symlink_blob_never_becomes_source_authority(tmp_path):
    out,repo,request,_,new=fixture(tmp_path)
    p=out/'research/derived/snapshots/blobs'/new
    outside=tmp_path/'external';outside.write_bytes(p.read_bytes());p.unlink();p.symlink_to(outside)
    with pytest.raises(ValueError):history.load_view(out,repo,request)


def test_line_range_cannot_mean_partial_patch_and_missing_fields_do_not_expand_access(tmp_path):
    out,repo,request,_,_=fixture(tmp_path)
    for changed in ({**request,'ref':'policy/train.py:1-2'},
                    {**request,'other_path':'anything'},
                    {'ref':'policy/train.py','view':'patch'}):
        with pytest.raises(ValueError):history.load_view(out,repo,changed)


def test_failed_native_producer_is_history_not_success(tmp_path):
    out,repo,request,_,_=fixture(tmp_path,status='failed',returncode=1)
    row=history.load_view(out,repo,request)
    assert row['producer_status']=='failed'
    assert row['installed_status']=='not_asserted'
    assert 'not current source' in row['authority']


def test_bound_resource_cannot_be_read_as_archived_source(tmp_path,monkeypatch):
    from autosim.research import workspace_resources
    out,repo,request,_,_=fixture(tmp_path)
    monkeypatch.setattr(workspace_resources,'bindings_for',lambda _: [{'target':'policy'}])
    with pytest.raises(ValueError,match='resource binding'):history.load_view(out,repo,request)


def test_existing_checkout_tool_dispatches_archive_read_and_schema_is_bounded(tmp_path):
    from autosim.research.checkout_source_citations import capture
    from autosim.research.agent_runtime import _native_tool_schema
    out,repo,request,_,_=fixture(tmp_path)
    row=capture(out,repo,request)
    assert row['historical_source_ref'].startswith('producer-source:')
    schema=_native_tool_schema(source_only=True)['inputSchema']['properties']['checkout_source_request']
    assert schema['additionalProperties'] is False
    assert schema['properties']['producer_evidence_id']['pattern']=='^[0-9a-f]{32}$'
    assert schema['properties']['view']['enum']==['source','patch']


def executor_for(out,repo,monkeypatch):
    from autosim.research.agent_runtime import _McpExecutor
    executor=_McpExecutor.__new__(_McpExecutor)
    executor.output,executor.workspace=out,repo
    executor.native_source_only=True
    executor.allow_commands,executor.allow_native=False,True
    executor.calls=0;executor.tool_seconds=0
    monkeypatch.setattr(executor,'_research_owner_workspace',lambda:(out,repo))
    return executor


def test_actual_source_only_mcp_route_does_not_execute_diagnostics(tmp_path,monkeypatch):
    out,repo,request,_,_=fixture(tmp_path)
    executor=executor_for(out,repo,monkeypatch)
    row=executor.native_probe({'purpose':'compare known historical reader',
                              'checkout_source_request':request})
    assert row['status']=='historical_source_captured'
    assert row['producer_evidence_id']==request['producer_evidence_id']
    assert executor.calls==1
    with pytest.raises(ValueError,match='source-only'):
        executor.native_probe({'purpose':'not authorized','checkout_source_request':request,'code':'print(1)'})


def test_unpublished_worker_cannot_gain_private_owner_archive_access(tmp_path,monkeypatch):
    from autosim.research import harness_repair
    out,repo,request,_,_=fixture(tmp_path)
    executor=executor_for(out,repo,monkeypatch)
    owner=out
    worker=out/'agent_workers/worker';worker.mkdir(parents=True)
    executor.output=worker
    monkeypatch.setattr(harness_repair,'native_source_access',lambda *a:False)
    with pytest.raises(ValueError,match='published owner source access'):
        executor.native_probe({'purpose':'not a private-run grant','checkout_source_request':request})


def test_paged_history_does_not_reexpose_a_credential_hidden_in_excerpt(tmp_path):
    out,repo,request,old,new=fixture(tmp_path)
    root=out/'research/derived'
    secret='sk-'+('a'*40)
    replacement='return 2 # api_key="'+secret+'"'
    text='def train():\n    '+replacement+'\n'
    sha=hashlib.sha256(text.encode()).hexdigest()
    (root/'snapshots/blobs'/sha).write_text(text)
    p=root/'snapshots/snapshots.json';index=read_json(p)
    index['snapshots'][-1]['files']['policy/train.py']=sha;atomic_json(p,index)
    p=root/'code_changes/round-1.json';change=read_json(p)
    change['patches'][0]['replace']=replacement;atomic_json(p,change)
    row=history.capture(out,repo,request)
    assert row['display_redacted'] is True
    assert secret not in row['excerpt']
    assert secret not in read_attempt_evidence(out,row['evidence_id'])['text']
    assert history.load_view(out,repo,request)['postimage_sha256']==sha
