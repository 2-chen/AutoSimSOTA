"""No GPU/provider/native launch: synthetic completed local train/eval evidence."""
from pathlib import Path
import pytest

from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.baseline_score_recovery import record, available
from autosim.research.receipt_verifier import verify_measurement
from tests.test_reference_score_recovery import narrow_fixture, LocatorReviewer


def fixture(tmp_path,monkeypatch):
    real,ref,native,artifact,revision=narrow_fixture(tmp_path,monkeypatch)
    root=real.run_root;old=read_json(real.output/ref)
    protocol=root/'comparison_protocol.json'
    atomic_json(protocol,read_json(root/old.pop('reference_protocol_ref')))
    receipt=read_json(native);receipt.pop('reference_protocol_ref')
    receipt['comparison_protocol_sha256']=digest(protocol);atomic_json(native,receipt)
    capture_attempt_evidence(real.output,attempt_id=receipt['attempt_id'],log=Path(receipt['log']),
        receipt_ref=native.relative_to(real.output).as_posix(),status='completed',
        returncode=0,termination_reason='normal_exit')
    identity='b'*32;directory=root/'train/native_outputs'/identity
    source=directory/'checkpoints/terminal/policy.bin';source.parent.mkdir(parents=True)
    source.write_bytes(Path(old['policy_artifact']['path']).read_bytes())
    old['policy_artifact']['source']=str(source)
    counter=directory/'checkpoints/terminal/training_step.json';atomic_json(counter,{'step':2048})
    audit=real.output/'training_progress_audits'/('c'*64+'.json');atomic_json(audit,{'synthetic':True})
    train=dict(attempt_id=identity,run_id=real.run_id,node_id='train',stage='train',
        status='completed',returncode=0,termination_reason='normal_exit',output_directory=str(directory),
        settings_digest=object_digest(old['settings']),comparison_protocol_sha256=digest(protocol),
        training_progress={'status':'observed','audit_ref':audit.relative_to(real.output).as_posix(),
            'evidence':{'native_counter':{'path':counter.relative_to(directory).as_posix(),
                'field':['step'],'value':2048},'document_sha256':digest(counter)}})
    train_path=root/'attempts'/identity/'receipt.json';atomic_json(train_path,train)
    log=root/'train/attempts'/identity/'output.log';log.parent.mkdir(parents=True);log.write_text('completed 2048 updates')
    capture_attempt_evidence(real.output,attempt_id=identity,log=log,
        receipt_ref=train_path.relative_to(real.output).as_posix(),status='completed',returncode=0,
        termination_reason='normal_exit')
    old.update(label='baseline',baseline_origin='locally_trained',
        train={'attempt_id':identity,'returncode':0})
    (real.output/ref).unlink();atomic_json(root/'measurements/baseline.json',old)
    row={'label':'baseline','round':0,'settings':old['settings'],'measured':False,
         'metric_value':None,'why_not':'result locator missing'}
    atomic_json(root/'controller_session.json',{'run_id':real.run_id,'status':'paused',
        'action':'','history':[row],'next_round':1,'base_settings':old['settings']})
    atomic_json(root/'research_report.json',{'run_id':real.run_id,'run_status':'paused','rounds':[row],'next_round':1})
    return real,revision,artifact,native,train_path


def test_cpu_baseline_recovery_preserves_native_history_and_updates_controller(tmp_path,monkeypatch):
    real,revision,artifact,native,train=fixture(tmp_path,monkeypatch)
    root=real.run_root;old=read_json(root/'measurements/baseline.json')
    original={p:digest(p) for p in (native,train,root/'comparison_protocol.json',artifact)}
    reviewer=LocatorReviewer();result=record(real,revision=revision,client=reviewer)
    assert result['label']=='baseline' and result['ok'] and result['metric_value']==.5
    assert result['scoring_recovery']['native_replayed'] is False
    assert verify_measurement(root,'baseline')['status']=='consistent'
    state=read_json(root/'baseline_score_recovery.json')
    assert state['status']=='completed' and read_json(root/state['original_ref'])==old
    session=read_json(root/'controller_session.json')
    assert session['baseline_rate']==.5 and session['history'][0]['measured'] is True
    assert session['next_round']==1 and len(session['history'])==1
    assert read_json(root/'research_report.json')['rounds']==session['history']
    assert all(digest(p)==sha for p,sha in original.items())
    assert len(reviewer.calls)==1
    record(real,client=reviewer)  # Forward-idempotent; no review or native replay.
    assert len(reviewer.calls)==1


def test_scalar_summary_metric_uses_verified_rollout_count_not_metric_record_count(tmp_path,monkeypatch):
    from dataclasses import replace
    from autosim.research.policy_consumption import verified_episode_count
    real,revision,artifact,native,train=fixture(tmp_path,monkeypatch);root=real.run_root
    real.metric_spec=replace(real.metric_spec,json_key='summary.rate',json_value_key='',
        episode_id_column='',aggregation='')
    metric=real.metric_spec.as_dict()
    m=read_json(root/'measurements/baseline.json');m['metric']=metric;atomic_json(root/'measurements/baseline.json',m)
    p=root/'comparison_protocol.json';protocol=read_json(p);protocol['metric']=metric;atomic_json(p,protocol)
    for path in (native,train):
        receipt=read_json(path);receipt['comparison_protocol_sha256']=digest(p);atomic_json(path,receipt)
    binding=read_json(real.output/'metric_binding.json');binding['primary_metric']=metric
    atomic_json(real.output/'metric_binding.json',binding)
    atomic_json(artifact,{'summary':{'rate':.5}})
    log=Path(read_json(native)['log']);log.write_text(log.read_text())
    capture_attempt_evidence(real.output,attempt_id=read_json(native)['attempt_id'],log=log,
        receipt_ref=native.relative_to(real.output).as_posix(),status='completed',returncode=0,
        termination_reason='normal_exit')
    result=record(real,revision=revision,client=LocatorReviewer())
    assert result['ok'] and result['metric_value']==.5
    assert result['metric_reading'].get('samples') is None
    assert verified_episode_count(result)==2
    assert verify_measurement(root,'baseline')['status']=='consistent'
    result['rollout_evidence']['episode_ids']=['forged-0','forged-1']
    atomic_json(root/'measurements/baseline.json',result)
    assert verify_measurement(root,'baseline')['status']!='consistent'


def test_cpu_baseline_score_commit_resumes_after_publication_failure(tmp_path,monkeypatch):
    real,revision,_,_,_=fixture(tmp_path,monkeypatch)
    original=real._commit_recovered_baseline
    monkeypatch.setattr(real,'_commit_recovered_baseline',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('simulated crash')))
    reviewer=LocatorReviewer()
    with pytest.raises(RuntimeError,match='simulated crash'):record(real,revision=revision,client=reviewer)
    assert available(real.run_root)['kind']=='resume_cpu_baseline_score_commit'
    monkeypatch.setattr(real,'_commit_recovered_baseline',original)
    result=record(real,client=reviewer)
    assert result['ok'] and verify_measurement(real.run_root,'baseline')['status']=='consistent'
    assert len(reviewer.calls)==1


def test_cpu_baseline_score_commit_resumes_after_controller_publication(tmp_path,monkeypatch):
    real,revision,_,_,_=fixture(tmp_path,monkeypatch)
    original=real._commit_recovered_baseline
    def crash_after(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('after controller commit')
    monkeypatch.setattr(real,'_commit_recovered_baseline',crash_after)
    reviewer=LocatorReviewer()
    with pytest.raises(RuntimeError,match='after controller'):record(real,revision=revision,client=reviewer)
    monkeypatch.setattr(real,'_commit_recovered_baseline',original)
    record(real,client=reviewer)
    assert read_json(real.run_root/'baseline_score_recovery.json')['status']=='completed'
    assert len(reviewer.calls)==1


@pytest.mark.parametrize('fault',['history_changed','baseline_replaced','source_archive_changed','result_changed'])
def test_pending_commit_refuses_unrelated_mutation(tmp_path,monkeypatch,fault):
    real,revision,_,_,_=fixture(tmp_path,monkeypatch)
    original=real._commit_recovered_baseline
    monkeypatch.setattr(real,'_commit_recovered_baseline',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError):record(real,revision=revision,client=LocatorReviewer())
    monkeypatch.setattr(real,'_commit_recovered_baseline',original)
    root=real.run_root;state=read_json(root/'baseline_score_recovery.json')
    if fault=='history_changed':
        path=root/'controller_session.json';x=read_json(path);x['next_round']=2;atomic_json(path,x)
    elif fault=='baseline_replaced':
        path=root/'measurements/baseline.json';x=read_json(path);x['metric_value']=.99;atomic_json(path,x)
    elif fault=='source_archive_changed':
        path=root/state['original_ref'];x=read_json(path);x['settings']={};atomic_json(path,x)
    else:
        path=root/state['recovered_ref'];x=read_json(path);x['metric_value']=.99;atomic_json(path,x)
    before=digest(root/'measurements/baseline.json')
    with pytest.raises(ValueError):record(real)
    assert digest(root/'measurements/baseline.json')==before


def test_real_preparation_handler_exposes_and_executes_cpu_only_contract(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research.operation_contracts import schema,validate
    real,revision,_,_,_=fixture(tmp_path/'research',monkeypatch)
    host=preparation(tmp_path/'host');host.output=real.output;host.run_id=real.run_id
    host.client=LocatorReviewer()
    monkeypatch.setattr(host,'_research_controller',lambda:(real,{}))
    contract=schema(host._step_recover_unscored_baseline)
    validate(contract,{'artifact_revision':revision})
    with pytest.raises(ValueError,match='checkpoint_ref'):
        validate(contract,{'artifact_revision':revision,'checkpoint_ref':'producer/model'})
    assert host._unscored_baseline_recovery_status()['kind']=='completed_baseline_result_locator'
    result=host._step_recover_unscored_baseline(artifact_revision=revision)
    assert result['outcome']=='done' and result['native_replayed'] is False
    assert verify_measurement(real.run_root,'baseline')['status']=='consistent'


@pytest.mark.parametrize('fault',['','changed_log','wrong_status'])
def test_completed_result_failure_reaches_fix_with_original_native_identity(tmp_path,monkeypatch,fault):
    from tests.test_prepare import preparation
    real,_,_,native,_=fixture(tmp_path/'research',monkeypatch)
    host=preparation(tmp_path/'host');host.output=real.output;host.run_id=real.run_id
    root=real.run_root;m=read_json(root/'measurements/baseline.json')
    r=read_json(native);r.update(run_id=real.run_id,stage='evaluate');atomic_json(native,r)
    m['evaluate'].update(status='completed',returncode=0)
    if fault=='changed_log':Path(r['log']).write_text('changed after completion')
    elif fault=='wrong_status':m['evaluate']['status']='running'
    before=digest(native)
    packet=host._native_evaluation_failure(real,m)
    assert digest(native)==before
    if fault:
        assert packet=={}
    else:
        assert packet['repair_owner']=='fix' and packet['native_operation_status']=='completed'
        assert packet['failure_category']=='completed_native_scoring_failed'
        assert packet['evaluation_attempt_id']==r['attempt_id'] and packet['retryable'] is False
        assert native.relative_to(real.output).as_posix() in packet['evidence_refs']
        envelope=read_json(real.output/packet['receipt_ref'])
        assert envelope['context']['native_attempt_id']==r['attempt_id']


@pytest.mark.parametrize('fault',['counter_changed','producer_changed','policy_changed',
    'heldout','later_round','review_refused','wrong_metric','incomplete_rollout'])
def test_cpu_baseline_recovery_refuses_invalid_or_changed_evidence(tmp_path,monkeypatch,fault):
    real,revision,artifact,native,train=fixture(tmp_path,monkeypatch)
    root=real.run_root;before=digest(root/'measurements/baseline.json')
    if fault=='counter_changed':
        atomic_json(Path(read_json(train)['output_directory'])/'checkpoints/terminal/training_step.json',{'step':4096})
    elif fault=='producer_changed':
        row=read_json(train);row['status']='failed';atomic_json(train,row)
    elif fault=='policy_changed':
        Path(read_json(root/'measurements/baseline.json')['policy_artifact']['source']).write_bytes(b'other policy')
    elif fault=='heldout':atomic_json(root/'confirmation_attempts/one.json',{})
    elif fault=='later_round':
        s=read_json(root/'controller_session.json');s['history'].append({'label':'round_1','measured':True});atomic_json(root/'controller_session.json',s)
    elif fault=='wrong_metric':atomic_json(artifact,{'episodes':[{'episode_id':'e0','success':False},{'episode_id':'e1','success':False}]})
    elif fault=='incomplete_rollout':atomic_json(artifact,{'episodes':[{'episode_id':'e0','success':True}]})
    with pytest.raises((ValueError,OSError)):
        record(real,revision=revision,client=LocatorReviewer(refuse=fault=='review_refused'))
    assert digest(root/'measurements/baseline.json')==before
    assert not (root/'baseline_score_recovery.json').exists()
