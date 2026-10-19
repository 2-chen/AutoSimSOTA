"""A narrower original launcher resumes Main's frozen, unscored recovery chain."""
from copy import deepcopy
from pathlib import Path
import pytest

from autosim.research.common import atomic_json, object_digest, read_json
from autosim.research.action_receipt import ActionReceipt,write_action_receipt
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.derive_and_run import validate_paused_research_resume
from tests.test_baseline_score_recovery import fixture


def owned_fixture(tmp_path,monkeypatch, fault=''):
    real,_,_,native,train=fixture(tmp_path,monkeypatch);root=real.run_root
    m=read_json(root/'measurements/baseline.json');tid=m['train']['attempt_id']
    trained=read_json(train);trained.update(evidence_id=tid,controller_decision_id='a'*16);atomic_json(train,trained)
    m['train']['evidence_id']=tid
    failed_id='c'*32;failed_ref=f'research/derived/attempts/{failed_id}/receipt.json'
    original=deepcopy(m);original['evaluate']={'attempt_id':failed_id,'returncode':1}
    log=root/'evaluate/attempts'/failed_id/'output.log';log.parent.mkdir(parents=True);log.write_text('synthetic load failure')
    first={**read_json(native),'attempt_id':failed_id,'run_id':'derived','stage':'evaluate',
        'controller_decision_id':'a'*16,'status':'failed','returncode':1,
        'termination_reason':'nonzero_exit','log':str(log),'ran':True}
    atomic_json(real.output/failed_ref,first)
    capture_attempt_evidence(real.output,attempt_id=failed_id,log=log,receipt_ref=failed_ref,
        status='failed',returncode=1,termination_reason='nonzero_exit')
    backup_ref=f'attempts/{tid}/baseline_unscored_{failed_id}.json';atomic_json(root/backup_ref,original)
    actual=read_json(native);actual.update(run_id='derived',stage='evaluate',controller_decision_id='b'*16,ran=True)
    atomic_json(native,actual)
    capture_attempt_evidence(real.output,attempt_id=actual['attempt_id'],log=Path(actual['log']),
        receipt_ref=native.relative_to(real.output).as_posix(),status='completed',returncode=0,termination_reason='normal_exit')
    m['recovery']={'kind':'repair_development_consumer_handoff','training_reused':True,
        'training_attempt_id':tid,'failure_evidence_id':failed_id,'original_measurement_ref':backup_ref,
        'superseded_evaluation_attempt_id':failed_id}
    atomic_json(root/'measurements/baseline.json',m)
    journal={'status':'failed','training_attempt_id':tid,'training_reused':True,
        'recovery_kind':'repair_development_consumer_handoff',
        'original_measurement_sha256':object_digest(original),'measurement_sha256':object_digest(m),
        'evaluation_attempt_id':actual['attempt_id'],'superseded_evaluation_attempt_id':failed_id}
    journal_path=root/f'attempts/{tid}/baseline_handoff_recovery_{failed_id}.json';atomic_json(journal_path,journal)
    train_ref=train.relative_to(real.output).as_posix()
    for owner,action,status,children,refs in [
        ('a'*16,'run_the_loop','yielded',[tid,failed_id],[train_ref,failed_ref]),
        ('b'*16,'recover_unscored_baseline','not recovered',[],[])]:
        if fault=='original_parent_children' and owner=='a'*16:children=[tid]
        if fault=='owner_parent_action' and owner=='b'*16:action='derive_a_command'
        write_action_receipt(real.output,ActionReceipt(action_id=owner,decision_id=owner,
            run_id='derived',repository=str(real.repo),action=action,state_revision=1,
            started_at='2026-10-19T00:00:00+00:00',finished_at='2026-10-19T00:01:00+00:00',
            status=status,reason='synthetic ownership',child_attempt_ids=children,evidence_refs=refs))
    sp=root/'controller_session.json';s=read_json(sp)
    s.update(schema_version=1,rounds=2,repository=str(real.repo),
        input_component_sha256={'settings':object_digest(m['settings'])});atomic_json(sp,s)
    if fault in {'journal_original','journal_new','journal_settings'}:
        key={'journal_original':'original_measurement_sha256','journal_new':'measurement_sha256',
             'journal_settings':'settings_digest'}[fault];journal[key]='0'*64;atomic_json(journal_path,journal)
    elif fault=='owner_parent_hash':
        p=real.output/'action_receipts'/('b'*16+'.json');a=read_json(p);a['reason']='changed';atomic_json(p,a)
    elif fault=='native_owner':actual['controller_decision_id']='f'*16;atomic_json(native,actual)
    elif fault=='finished_returncode':actual['returncode']=2;atomic_json(native,actual)
    elif fault=='seal_log_changed':Path(actual['log']).write_text('changed after seal')
    elif fault=='session_hash':s['input_component_sha256']['settings']='0'*64;atomic_json(sp,s)
    elif fault=='heldout':atomic_json(root/'confirmation_attempts/exposed.json',{})
    elif fault=='journal_symlink':
        moved=journal_path.with_suffix('.original');journal_path.rename(moved);journal_path.symlink_to(moved)
    return real


@pytest.mark.parametrize('fault',['','constraint','journal_original','journal_new','journal_settings',
    'owner_parent_action','owner_parent_hash','native_owner','original_parent_children',
    'finished_returncode','seal_log_changed','session_hash','heldout','journal_symlink'])
def test_original_launcher_recovers_control_only_with_intact_ownership_chain(tmp_path,monkeypatch,fault):
    real=owned_fixture(tmp_path,monkeypatch,fault)
    paths=[real.run_root/'measurements/baseline.json',real.run_root/'controller_session.json']
    before={p:p.read_bytes() for p in paths}
    repository=Path(read_json(real.output/'workspace_snapshot.json')['source'])
    requested={'episodes':3} if fault=='constraint' else {}
    if fault:
        with pytest.raises(ValueError,match='different frozen rounds/settings'):
            validate_paused_research_resume(real.output,rounds=2,settings=requested,repository=repository)
    else:
        assert validate_paused_research_resume(real.output,rounds=2,settings=requested,repository=repository)==2
    assert all(p.read_bytes()==data for p,data in before.items())


def test_resume_accepts_second_sealed_recovery_without_resetting_original_inputs(tmp_path,monkeypatch):
    real=owned_fixture(tmp_path,monkeypatch);root=real.run_root
    old=read_json(root/'measurements/baseline.json');failure=old['evaluate']['attempt_id']
    tid=old['train']['attempt_id'];identity='e'*32;owner='e'*16
    backup_ref=f'attempts/{tid}/baseline_unscored_{failure}.json';atomic_json(root/backup_ref,old)
    prior=read_json(root/f'attempts/{failure}/receipt.json')
    log=root/'evaluate/attempts'/identity/'output.log';log.parent.mkdir(parents=True);log.write_text(Path(prior['log']).read_text())
    native={**prior,'attempt_id':identity,'controller_decision_id':owner,'log':str(log)}
    relative=f'research/derived/attempts/{identity}/receipt.json';atomic_json(real.output/relative,native)
    capture_attempt_evidence(real.output,attempt_id=identity,log=log,receipt_ref=relative,
        status='completed',returncode=0,termination_reason='normal_exit')
    current={**deepcopy(old),'evaluate':{'attempt_id':identity,'returncode':0},
        'recovery':{'kind':'repair_development_consumer_handoff','training_reused':True,
            'training_attempt_id':tid,'failure_evidence_id':failure,'original_measurement_ref':backup_ref,
            'superseded_evaluation_attempt_id':failure}}
    atomic_json(root/'measurements/baseline.json',current)
    atomic_json(root/f'attempts/{tid}/baseline_handoff_recovery_{failure}.json',{
        'status':'failed','training_attempt_id':tid,'training_reused':True,
        'recovery_kind':'repair_development_consumer_handoff',
        'original_measurement_sha256':object_digest(old),'measurement_sha256':object_digest(current),
        'evaluation_attempt_id':identity,'superseded_evaluation_attempt_id':failure})
    write_action_receipt(real.output,ActionReceipt(action_id=owner,decision_id=owner,run_id='derived',
        repository=str(real.repo),action='recover_unscored_baseline',state_revision=2,
        started_at='2026-10-19T00:02:00+00:00',finished_at='2026-10-19T00:03:00+00:00',
        status='not recovered',reason='second linked recovery',child_attempt_ids=[],evidence_refs=[]))
    repo=Path(read_json(real.output/'workspace_snapshot.json')['source'])
    assert validate_paused_research_resume(real.output,rounds=2,settings={},repository=repo)==2


def test_resume_accepts_original_main_owned_completed_missing_result(tmp_path,monkeypatch):
    real=owned_fixture(tmp_path,monkeypatch);root=real.run_root
    m=read_json(root/'measurements/baseline.json');identity=m['evaluate']['attempt_id']
    m.pop('recovery');atomic_json(root/'measurements/baseline.json',m)
    p=root/f'attempts/{identity}/receipt.json';r=read_json(p);r['controller_decision_id']='a'*16;atomic_json(p,r)
    from autosim.research.action_receipt import read_action_receipt
    action=read_action_receipt(real.output,'action_receipts/'+('a'*16)+'.json',
                              run_id='derived',repository=real.repo)
    action['child_attempt_ids'].append(identity)
    action['evidence_refs'].append(p.relative_to(real.output).as_posix())
    action.pop('receipt_sha256');action.pop('schema_version')
    (real.output/'action_receipts'/('a'*16+'.json')).unlink()
    write_action_receipt(real.output,ActionReceipt(**action))
    repo=Path(read_json(real.output/'workspace_snapshot.json')['source'])
    assert validate_paused_research_resume(real.output,rounds=2,settings={},repository=repo)==2
