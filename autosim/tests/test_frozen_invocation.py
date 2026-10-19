"""Diagnostic verification is not permission to replace a frozen judge adapter."""
import sys
from pathlib import Path

import pytest

from autosim.research.common import atomic_json, digest, read_json
from autosim.research import frozen_invocation as fi
from autosim.research.compute_decision import ComputeDecision
from tests.test_prepare import preparation


def setup(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir();out=tmp_path/'run';out.mkdir()
    judge=repo/'judge.py';judge.write_text('SUCCESS_RULE = "fixed"\n')
    row={'available':True,'entrypoint':'judge.py','parameters':[],
         'environment':{'FIXED':'original'}}
    source='def stage_argv_score(i):\n    return [i["python"],i["repo"]+"/judge.py"]\n'
    expected={'source':source,'parameters':{},'row':row}
    atomic_json(out/'execution.json',{'stages':{'score':row}})
    root=out/'research/derived'
    atomic_json(root/'comparison_protocol.json',{'schema_version':1,'target':'score',
        'evaluator_argv_source':source,'evaluator_parameters':{},'evaluator_stage':row})
    atomic_json(root/'protocol_frozen.json',{'hashes':{str(judge):digest(judge)}})
    atomic_json(out/'derived_stages.json',{'score':expected,'train':{'source':'unchanged'}})
    protocol_sha=digest(root/'comparison_protocol.json')
    # Unit fixture for the independent verifier; actual native baseline verification
    # is separately required by restore_legacy and performed in the real repair.
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
        lambda *a,**kw:{'status':'consistent' if digest(root/'comparison_protocol.json')==protocol_sha
                       else 'inconsistent','issues':[]})
    return repo,out,expected,judge


@pytest.mark.parametrize('field',['source','parameters','row'])
def test_frozen_score_entry_rejects_each_descriptor_change(tmp_path,monkeypatch,field):
    _,out,expected,_=setup(tmp_path,monkeypatch)
    new={**expected,field:('different' if field=='source' else {'different':'value'})}
    assert fi.differences(out,'derived','score',new)==[field]
    assert fi.differences(out,'derived','score',expected)==[]
    assert fi.differences(out,'derived','train',new)==[]


def test_successful_fixture_diagnostic_is_retained_not_installed(tmp_path,monkeypatch):
    repo,out,expected,_=setup(tmp_path,monkeypatch)
    obj=preparation(tmp_path/'preparation');obj.repo=repo;obj.output=out
    obj.interpreter=Path(sys.executable)
    obj.execution=read_json(out/'execution.json')
    obj.stages={'score':expected['source']};obj.parameters={'score':{}}
    obj.verified={'score':expected['row']}
    obj.decision=ComputeDecision(device='cpu',device_index=0,environment={},why='fixture')
    monkeypatch.setattr(obj,'_score_target',lambda:'score')
    monkeypatch.setattr('autosim.research.prepare.execution_derive.checkpoint_for_verification',
        lambda *a,**k:{'path':''})
    different=expected['source']+'# harmless-looking rewrite still is not the frozen adapter\n'
    monkeypatch.setattr('autosim.research.prepare.execution_derive.make_runnable',
        lambda *a,**k:(different,{},[],expected['row']))
    before=(out/'derived_stages.json').read_bytes()
    result=obj._step_derive_a_command(stage='score')
    assert result['failure_category']=='frozen_evaluator_invocation'
    assert result['native_operation_status']=='diagnostic_completed_not_installed'
    assert result['evidence_id'] and result['repair_owner']=='main_agent'
    assert obj.stages['score']==expected['source']
    assert (out/'derived_stages.json').read_bytes()==before
    records=list((out/'diagnostic_invocations').glob('*.json'))
    assert len(records)==1 and read_json(records[0])['candidate']['source']==different
    assert 'ORIGINAL' in result['retry_when']


def test_exact_frozen_adapter_can_be_reverified_without_replacement_drift(tmp_path,monkeypatch):
    repo,out,expected,_=setup(tmp_path,monkeypatch)
    obj=preparation(tmp_path/'preparation');obj.repo=repo;obj.output=out
    obj.interpreter=Path(sys.executable);obj.execution=read_json(out/'execution.json')
    obj.stages={'score':expected['source']};obj.parameters={'score':{}}
    obj.verified={'score':expected['row']}
    obj.decision=ComputeDecision(device='cpu',device_index=0,environment={},why='fixture')
    monkeypatch.setattr(obj,'_score_target',lambda:'score')
    monkeypatch.setattr('autosim.research.prepare.execution_derive.checkpoint_for_verification',
        lambda *a,**k:{'path':''})
    monkeypatch.setattr('autosim.research.prepare.execution_derive.make_runnable',
        lambda *a,**k:(expected['source'],{},[],expected['row']))
    assert obj._step_derive_a_command(stage='score')['outcome']=='done'
    assert fi.differences(out,'derived','score',read_json(out/'derived_stages.json')['score'])==[]


def test_rejected_diagnostic_cannot_mutate_active_nested_rows_in_memory(tmp_path,monkeypatch):
    repo,out,expected,_=setup(tmp_path,monkeypatch)
    obj=preparation(tmp_path/'preparation');obj.repo=repo;obj.output=out
    obj.interpreter=Path(sys.executable);obj.execution=read_json(out/'execution.json')
    obj.stages={'score':expected['source']};obj.parameters={'score':{}}
    obj.verified={'score':expected['row']}
    obj.decision=ComputeDecision(device='cpu',device_index=0,environment={},why='fixture')
    monkeypatch.setattr(obj,'_score_target',lambda:'score')
    monkeypatch.setattr('autosim.research.prepare.execution_derive.checkpoint_for_verification',
        lambda *a,**k:{'path':''})
    def mutate(client,stage,row,**kw):
        row['environment']['FIXED']='diagnostic-only-change'
        return expected['source'],{},[],row
    monkeypatch.setattr('autosim.research.prepare.execution_derive.make_runnable',mutate)
    result=obj._step_derive_a_command(stage='score')
    assert result['failure_category']=='frozen_evaluator_invocation'
    assert obj.verified['score']['environment']['FIXED']=='original'
    assert obj.execution['stages']['score']['environment']['FIXED']=='original'


def test_restore_quarantines_displaced_entry_and_preserves_every_other_record(tmp_path,monkeypatch):
    repo,out,expected,_=setup(tmp_path,monkeypatch)
    p=out/'derived_stages.json';row=read_json(p)
    row['score']={**expected,'source':expected['source']+'# drift\n'};atomic_json(p,row)
    protected=[out/'execution.json',out/'research/derived/comparison_protocol.json',
        out/'research/derived/protocol_frozen.json',repo/'judge.py']
    before={p:p.read_bytes() for p in protected}
    result=fi.restore_legacy(out,repo,'derived','score',expected_current_sha256=digest(p),reason='unit maintenance')
    assert result['status']=='exact_frozen_adapter_restored'
    assert read_json(p)=={**row,'score':expected}
    assert read_json(out/result['record_ref'])['before']==row['score']
    assert read_json(out/result['record_ref'])['status']=='restoration_prepared'
    assert read_json(out/result['applied_ref'])['after_record_sha256']==digest(p)
    assert all(p.read_bytes()==value for p,value in before.items())
    assert not (out/'native_jobs').exists()


@pytest.mark.parametrize('fault',['current_hash','protected_bytes','protocol_bytes','active_job','live_model'])
def test_restore_never_overrides_changed_proofs_or_live_owners(tmp_path,monkeypatch,fault):
    repo,out,expected,judge=setup(tmp_path,monkeypatch)
    p=out/'derived_stages.json';row=read_json(p);row['score']['source']+='\n# drift\n';atomic_json(p,row)
    wanted=digest(p)
    if fault=='current_hash':wanted='0'*64
    elif fault=='protected_bytes':judge.write_text('CHANGED_JUDGE=True\n')
    elif fault=='protocol_bytes':
        f=out/'research/derived/comparison_protocol.json';r=read_json(f);r['evaluator_argv_source']+='\n';atomic_json(f,r)
    elif fault=='active_job':
        monkeypatch.setattr('autosim.research.native_jobs.active_jobs',lambda *a:[{'status':'running'}])
    else:
        atomic_json(out/'agent/processes/a.json',{'process_identity':{'pid':123}})
        monkeypatch.setattr('autosim.research.process_executor.inspect_process_identity',
            lambda *a:{'status':'matching_running'})
    before=p.read_bytes()
    with pytest.raises((ValueError,RuntimeError)):
        fi.restore_legacy(out,repo,'derived','score',expected_current_sha256=wanted,reason='unit maintenance')
    assert p.read_bytes()==before


def test_protocol_race_during_verification_never_restores_changed_protocol(tmp_path,monkeypatch):
    repo,out,expected,_=setup(tmp_path,monkeypatch)
    p=out/'derived_stages.json';row=read_json(p);row['score']['source']+='\n# drift\n';atomic_json(p,row)
    protocol=out/'research/derived/comparison_protocol.json'
    def race(*a,**kw):
        r=read_json(protocol);r['evaluator_argv_source']+='\n# concurrent rewrite\n';atomic_json(protocol,r)
        return {'status':'consistent','issues':[]}
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',race)
    before=p.read_bytes()
    with pytest.raises(ValueError,match='protocol changed'):
        fi.restore_legacy(out,repo,'derived','score',expected_current_sha256=digest(p),reason='unit maintenance')
    assert p.read_bytes()==before
