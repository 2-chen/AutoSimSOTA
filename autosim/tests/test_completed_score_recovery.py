"""Candidate CPU result recovery never replays evaluation or rewrites history."""
import pytest

from autosim.research import completed_score_recovery as recovery
from autosim.research.common import atomic_json,digest,read_json
from autosim.research.receipt_verifier import verify_measurement
from tests.test_baseline_score_recovery import fixture as local_fixture
from tests.test_reference_score_recovery import LocatorReviewer


def fixture(tmp_path,monkeypatch,refuse=False):
    real,revision,artifact,native,train=local_fixture(tmp_path,monkeypatch)
    old=read_json(real.run_root/'measurements/baseline.json')
    old['label']='round_2'
    atomic_json(real.run_root/'measurements/round_2.json',old)
    real.client=LocatorReviewer(refuse=refuse)
    return real,revision,artifact,native,train


def test_completed_candidate_recovers_on_cpu_and_is_idempotent(tmp_path,monkeypatch):
    real,revision,artifact,native,train=fixture(tmp_path,monkeypatch)
    protected=[native,train,artifact,real.run_root/'controller_session.json',
        real.run_root/'research_report.json',real.run_root/'comparison_protocol.json',
        real.run_root/'measurements/baseline.json',real.run_root/'measurements/round_2.json']
    before={p:digest(p) for p in protected}
    assert recovery.offered(real.run_root)[0]['required_argument']=='artifact_revision'
    result=recovery.record(real,label='round_2',artifact_revision=revision)
    assert result['ok'] and result['metric_value']==.5
    assert result['scoring_recovery']['native_replayed'] is False
    assert verify_measurement(real.run_root,result['label'])['status']=='consistent'
    assert all(digest(p)==v for p,v in before.items())
    assert len(real.client.calls)==1
    again=recovery.record(real,label='round_2',artifact_revision=revision)
    assert again==result and len(real.client.calls)==1


@pytest.mark.parametrize('fault',['running','pending','active_patch','heldout','failed_native',
    'metric_present','policy_changed','result_changed','outside_locator','independent_refusal'])
def test_cpu_recovery_refuses_unsafe_or_changed_subjects(tmp_path,monkeypatch,fault):
    real,revision,artifact,native,train=fixture(tmp_path,monkeypatch,refuse=fault=='independent_refusal')
    original=real.run_root/'measurements/round_2.json';before=digest(original)
    session=read_json(real.run_root/'controller_session.json')
    if fault=='running':session['status']='running'
    elif fault=='pending':session['pending_action']={'round':2}
    elif fault=='active_patch':real._held={'x.py':object()}
    elif fault=='heldout':atomic_json(real.run_root/'confirmation_attempts/a.json',{})
    elif fault=='failed_native':
        row=read_json(native);row['returncode']=1;atomic_json(native,row)
    elif fault=='metric_present':
        old=read_json(original);old['metric_value']=.9;atomic_json(original,old);before=digest(original)
    elif fault=='policy_changed':
        path=read_json(original)['policy_artifact']['source']
        from pathlib import Path
        Path(path).write_bytes(b'different policy')
    elif fault=='result_changed':artifact.write_text('{"episodes":[]}')
    elif fault=='outside_locator':revision={**revision,'pattern':'../score.json'}
    atomic_json(real.run_root/'controller_session.json',session)
    with pytest.raises((ValueError,OSError,TypeError)):
        recovery.record(real,label='round_2',artifact_revision=revision)
    assert digest(original)==before
    assert not any((real.run_root/'measurements').glob('score_recovered_*.json'))


def test_correct_handoff_cpu_mode_does_not_call_native_correction(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research import handoff_correction
    real=preparation(tmp_path)
    marker=object();real._research_controller=lambda:(marker,{})
    monkeypatch.setattr(handoff_correction,'correct',lambda *a,**k:pytest.fail('no native replay'))
    seen=[]
    def record(research,**kwargs):
        seen.append((research,kwargs))
        return {'label':'score_recovered_'+'a'*16,'metric_value':.7,
            'scoring_recovery':{'attachment_ref':'metric_recoveries/fixture.json'}}
    monkeypatch.setattr(recovery,'record',record)
    revision={'pattern':'result.json','reason':'source-backed locator','source_refs':[]}
    result=real._step_correct_policy_handoff(label='round_2',artifact_revision=revision)
    assert seen==[(marker,{'label':'round_2','artifact_revision':revision})]
    assert result['outcome']=='measurement_recorded' and result['native_replayed'] is False
    with pytest.raises(ValueError,match='mutually exclusive'):
        real._step_correct_policy_handoff(label='round_2',checkpoint_ref='policy',artifact_revision=revision)


def test_cpu_recovery_preserves_frozen_policy_parent_identity(tmp_path,monkeypatch):
    from pathlib import Path
    from autosim.research.baseline_score_recovery import record as recover_baseline
    from autosim.research.evidence_store import capture_attempt_evidence
    real,revision,artifact,native,train=fixture(tmp_path,monkeypatch)
    parent=recover_baseline(real,revision=revision,client=real.client)
    parent_path=real.run_root/'measurements/baseline.json'
    old=read_json(real.run_root/'measurements/round_2.json')
    receipt=read_json(native);attempt='d'*32
    receipt['attempt_id']=attempt
    log=real.run_root/'evaluate/attempts'/attempt/'output.log';log.parent.mkdir(parents=True)
    log.write_bytes(Path(receipt['log']).read_bytes());receipt['log']=str(log)
    native=real.run_root/'attempts'/attempt/'receipt.json';atomic_json(native,receipt)
    capture_attempt_evidence(real.output,attempt_id=attempt,log=log,
        receipt_ref=native.relative_to(real.output).as_posix(),status='completed',
        returncode=0,termination_reason='normal_exit')
    old['evaluate']={'attempt_id':attempt,'returncode':0}
    old['train']={**old['train'],'reused':True}
    old['policy_reuse']={'mode':'evaluation_only_local_frozen_policy','label':'baseline',
        'measurement_sha256':digest(parent_path),'training_attempt_id':parent['train']['attempt_id'],
        'policy_sha256':parent['policy_artifact']['sha256'],
        'training_receipt_sha256':digest(train),'no_new_training_or_data_claim':True}
    atomic_json(real.run_root/'measurements/round_2.json',old)
    result=recovery.record(real,label='round_2',artifact_revision=revision)
    assert result['policy_reuse']==old['policy_reuse']
    verdict=verify_measurement(real.run_root,result['label'])
    assert verdict['status']=='consistent',verdict
    assert verdict['checks']['frozen_policy_identity']


def test_recovery_label_never_bypasses_adoption_receipt_verifier(tmp_path,monkeypatch):
    from tests.test_measurement_adoption import fixture as adoption_fixture
    from autosim.research import measurement_adoption as adoption
    real,original=adoption_fixture(tmp_path)
    old=read_json(real.run_root/'measurements'/f"{original['label']}.json")
    old.pop('handoff_correction')
    old['label']='score_recovered_'+'e'*16
    old['scoring_recovery']={'original_measurement_ref':'research/derived/measurements/round_2.json',
        'native_replayed':False,'attachment_ref':'metric_recoveries/'+'f'*64+'.json',
        'attachment_sha256':'f'*64}
    path=real.run_root/'measurements'/f"{old['label']}.json";atomic_json(path,old)
    request=next(x for x in adoption.offered(real.run_root) if x['label']==old['label'])
    before=digest(real.run_root/'snapshots/snapshots.json')
    with pytest.raises(ValueError,match='revalidation failed'):
        adoption.adopt(real,**{k:request[k] for k in ('label','expected_best','measurement_sha256')})
    assert digest(real.run_root/'snapshots/snapshots.json')==before


def test_cpu_mode_does_not_require_a_new_native_verification_generation(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research import workflow_patch
    real=preparation(tmp_path)
    real.stages={'evaluate':'def stage_argv_evaluate(i): return [i["python"]]'}
    monkeypatch.setattr(workflow_patch,'view',lambda *a,**k:[])
    monkeypatch.setattr(workflow_patch,'generation',lambda *a:'new-generation')
    def _step_correct_policy_handoff(*,label:str,artifact_revision:dict|None=None):
        return {'outcome':'cpu_only','native_replayed':False}
    monkeypatch.setattr(real,'_step_correct_policy_handoff',_step_correct_policy_handoff)
    result=real._do_operation('correct_policy_handoff',label='round_2',
        artifact_revision={'pattern':'score.json','reason':'source-supported output',
            'source_refs':[{'ref':'writer.py','quote':'write_score()'}]})
    assert result['outcome']=='cpu_only',result
