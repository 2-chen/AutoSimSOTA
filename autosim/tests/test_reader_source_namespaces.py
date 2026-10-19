"""The task snapshot and framework catalog never alias or grant each other."""
import json

import pytest

from autosim.research.agent_tasks import snapshot, ReaderSourceScopeError
from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.harness_repair import status
from autosim.research.agent_runtime import _McpExecutor, handle_mcp_message


def workspace(tmp_path):
    out=tmp_path/'run';repo=out/'checkout';repo.mkdir(parents=True)
    train=repo/'policy/train.py';train.parent.mkdir();train.write_text('def train(): pass\n')
    entries=[['file','policy/train.py',train.stat().st_size,digest(train),0o644]]
    atomic_json(out/'workspace_snapshot.json',{'schema_version':2,'destination':str(repo),
        'source_entries':entries,'source_tree_fingerprint':object_digest(entries)})
    atomic_json(out/'derived_stages.json',{})
    return out,repo


def test_framework_catalog_cannot_be_copied_as_benchmark_source(tmp_path):
    out,repo=workspace(tmp_path)
    with pytest.raises(ReaderSourceScopeError) as caught:
        snapshot(out,repo,'bad',source_paths=['research/data_versions.py','policy/train.py'])
    text=str(caught.value)
    assert 'research/data_versions.py' in text and 'framework_source_request' in text
    assert 'BENCHMARK' in text
    assert not (out/'agent_workers/bad/checkout/research/data_versions.py').exists()


def test_narrow_benchmark_snapshot_and_separate_framework_read_both_work(tmp_path):
    out,repo=workspace(tmp_path)
    child,checkout=snapshot(out,repo,'data',source_paths=['policy/train.py'])
    assert (checkout/'policy/train.py').read_bytes()==(repo/'policy/train.py').read_bytes()
    assert read_json(child/'workspace_snapshot.json')['source_paths']==['policy/train.py']
    executor=_McpExecutor(workspace=checkout,output=child,allow_commands=False,allow_native=True)
    response=handle_mcp_message({'jsonrpc':'2.0','id':1,'method':'tools/call',
        'params':{'name':'inspect_native_environment','arguments':{
            'purpose':'Read public framework input contract, no native process',
            'framework_source_request':{'path':'research/training_work.py','query':'def consumes_input','limit':22}}}},executor)
    assert not response['result'].get('isError')
    row=json.loads(response['result']['content'][0]['text'])
    assert 'def consumes_input' in row['excerpt'] and row['evidence_id']
    assert not row['candidate_edits_allowed'] and not row['live_activation_allowed']
    assert not (checkout/'research/training_work.py').exists()
    assert not (out/'processes').exists() and not (child/'processes').exists()


def test_same_named_benchmark_file_stays_native_not_an_implicit_framework_alias(tmp_path):
    out,repo=workspace(tmp_path)
    path=repo/'research/data_versions.py';path.parent.mkdir();path.write_text('native = "native repo"\n')
    manifest=read_json(out/'workspace_snapshot.json')
    manifest['source_entries'].append(['file','research/data_versions.py',path.stat().st_size,digest(path),0o644])
    manifest['source_tree_fingerprint']=object_digest(manifest['source_entries'])
    atomic_json(out/'workspace_snapshot.json',manifest)
    _,checkout=snapshot(out,repo,'native',source_paths=['research/data_versions.py'])
    assert (checkout/'research/data_versions.py').read_bytes()==path.read_bytes()


def test_registered_framework_fix_can_read_original_native_source_without_edit_scope(tmp_path):
    from pathlib import Path
    from autosim.research import harness_repair
    out, repo = workspace(tmp_path)
    identity = 'a' * 32
    worker, checkout = snapshot(out, Path(harness_repair.__file__).resolve().parents[1],
                                'harness_' + identity, empty=True)
    proposal = out / 'harness_repairs' / identity / 'proposal.json'
    record = {'id': identity, 'status': 'proposing', 'activation_allowed': False,
              'worker_ref': worker.relative_to(out).as_posix()}
    atomic_json(proposal, record)
    assert harness_repair.native_source_access(out, worker)
    executor = _McpExecutor(workspace=checkout, output=worker, allow_commands=False,
                            allow_native=True, native_source_only=True)
    row = executor.native_probe({'purpose': 'verify native consumer while diagnosing harness',
        'checkout_source_request': {'ref': 'policy/train.py:1'}})
    assert row['ref'].startswith('checkout-source:') and row['evidence_id']
    assert 'def train' in row['quote']
    assert not (checkout / 'policy/train.py').exists()
    assert not (out / 'processes').exists()
    assert (repo / 'policy/train.py').read_text() == 'def train(): pass\n'
    atomic_json(proposal, {**record, 'status': 'candidate_only'})
    assert not harness_repair.native_source_access(out, worker)
    with pytest.raises((ValueError, OSError)):
        executor.native_probe({'purpose': 'completed proposal is not an enduring grant',
            'checkout_source_request': {'ref': 'policy/train.py:1'}})


def test_framework_worker_name_alone_grants_no_extra_native_read(tmp_path):
    from pathlib import Path
    from autosim.research import harness_repair
    out, _ = workspace(tmp_path)
    worker, checkout = snapshot(out, Path(harness_repair.__file__).resolve().parents[1],
                                'harness_' + 'b' * 32, empty=True)
    assert not harness_repair.native_source_access(out, worker)
    executor = _McpExecutor(workspace=checkout, output=worker, allow_commands=False, allow_native=True)
    with pytest.raises((ValueError, OSError)):
        executor.native_probe({'purpose': 'unregistered proposal',
            'checkout_source_request': {'ref': 'policy/train.py:1'}})


def test_catalog_guidance_scopes_each_operation_and_permissions(tmp_path):
    row=status(tmp_path)
    assert 'ONLY propose_harness_repair' in row['source_selection_guidance']
    assert 'BENCHMARK' in row['source_selection_guidance']
    assert 'framework_source_request' in row['source_read_guidance']
    assert all(item['namespace']=='framework_package' for item in row['source_catalog'])
    for path in ['research/training_work.py','research/decision.py','research/adapter_protocol.py']:
        item=next(i for i in row['source_catalog'] if i['path']==path)
        assert not item['candidate_edits_allowed'] and not item['live_activation_allowed']


def test_snapshot_preflight_refusal_reaches_main_with_stable_arguments(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    obj=preparation(tmp_path);obj.output.mkdir(exist_ok=True)
    monkeypatch.setattr(obj,'_decision_facts',lambda:{})
    def refused(*a,**kw):raise ReaderSourceScopeError('wrong namespace: research/data_versions.py')
    monkeypatch.setattr('autosim.research.agent_tasks.submit',refused)
    result=obj._step_submit_research_task(role='data',task='inspect consumer',
        expected_result='source evidence',source_paths=['research/data_versions.py'])
    assert result['failure_category']=='reader_source_scope' and result['repair_owner']=='main_agent'
    assert result['native_operation_status']=='not_started' and result['evidence_id']
    private=read_json(obj.output/'agent/operation_inputs'/f'{result["evidence_id"]}.json')
    assert private['context']['source_paths']==['research/data_versions.py']
    assert 'framework_source_request' in result['retry_when']
    assert not (obj.output/'agent_tasks').exists()
@pytest.mark.parametrize('relative', ['research/prelaunch_recovery.py', 'research/recorder.py'])
def test_preparation_and_report_recovery_sources_are_readable_not_hot_editable(tmp_path, relative):
    from autosim.research import harness_repair
    result = harness_repair.read_source(tmp_path, {'path': relative, 'limit': 20})
    assert result['candidate_edits_allowed'] is False
    assert result['live_activation_allowed'] is False
    assert relative in harness_repair.READABLE and relative not in harness_repair.ALLOWED
