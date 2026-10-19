"""Real CPU producer/consumer replay; no benchmark-specific paths or paid model."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosim.research import checkpoint_handoff as ch, measurement_recovery as mr
from autosim.research.common import atomic_json, read_json, digest
from autosim.research.declarative_backend import DeclarativeBackend
from autosim.research.derived_research import DerivedResearch
from autosim.research.compute_decision import ComputeDecision
from autosim.research.adapter_protocol import OptimizationSpace
from autosim.research import objective


class Review:
    supports_native_progress_audit = True
    model = 'cpu-fixture'

    def chat_with_metadata(self, system, user, **kwargs):
        payload = json.loads(user)
        document = next(row for row in payload['numeric_documents'] if row['path']=='progress.json')
        return json.dumps({'verdict':'verified', 'counter':{'path':document['path'],
            'field':['completed'], 'value':17}, 'source_evidence':[
                {'source_id':payload['sources'][0]['id'], 'quote':'completed += 1'}],
            'counter_selection':{'kind':'terminal_completed','reviewed_documents':['progress.json'],
                                 'why':'fixture producer records its completed loop'},
            'why':'CPU fixture only, not a real policy-learning claim'}), {}


def failed_bundle(tmp_path):
    output = tmp_path/'run'
    repo = output/'checkout'
    repo.mkdir(parents=True)
    (repo/'produce.py').write_text('''import sys,json
from pathlib import Path
root=Path(sys.argv[1]); bundle=root/'arbitrary_native_bundle'; bundle.mkdir(parents=True)
completed=0
for index in range(17):
    completed += 1
(bundle/'learned.bin').write_bytes(b'fixture policy')
(bundle/'native_config.json').write_text(json.dumps({'architecture':'toy'}))
(root/'progress.json').write_text(json.dumps({'completed':completed}))
''')
    (repo/'consume.py').write_text('''import sys,json
from pathlib import Path
bundle=Path(sys.argv[1])
assert bundle.is_dir(), 'native consumer requires a full bundle directory'
assert json.loads((bundle/'native_config.json').read_text())['architecture']=='toy'
assert (bundle/'learned.bin').read_bytes()==b'fixture policy'
print('success_rate: 0.6')
''')
    stages = {name:f'def stage_argv_{name}(i):\n    return [i["python"], i["repo"]+"/{script}", i["{arg}"]]'
        for name,script,arg in [('train','produce.py','output'),('evaluate','consume.py','checkpoint')]}
    answer = {'stages':{'train':{'available':True,'entrypoint':'produce.py',
                               'invocation':'python produce.py','artifact':'arbitrary_native_bundle/learned.bin'},
        'evaluate':{'available':True,'entrypoint':'consume.py','invocation':'python consume.py','artifact':''}}}
    real = DerivedResearch(repo=repo,output=output,run_id='derived',
        backend=DeclarativeBackend(repo=repo,answer=answer,sources=stages,parameters={}),
        stages=stages,space=OptimizationSpace(),client=Review(),interpreter=Path(sys.executable),
        require_training_progress=True,
        compute=ComputeDecision(device='cpu',device_index=0,environment={},why='CPU fixture'))
    real.controller_decision_id='a'*16
    failed = real.measure(settings={},label='baseline')
    assert failed['ok'] is False and failed['where']=='evaluate', failed
    history = [{'label':'baseline','round':0,'measured':False,'metric_value':None,'settings':{}}]
    real.rubric = objective.Rubric([objective.Check(name='measurement',question='Measured?',weight=1)])
    real.rubric_built = True
    real._advance_best('baseline',score=None,scale='success_rate',why='unscored fixture')
    atomic_json(real.run_root/'controller_session.json',{'run_id':'derived','status':'paused',
        'rounds':2,'next_round':1,'history':history,'current_measurement_label':'baseline'})
    atomic_json(real.run_root/'research_report.json',{'run_id':'derived','run_status':'paused',
        'rounds':history,'next_round':1})
    return real, failed


def test_formal_producer_survives_failed_consumer_and_original_replay_reuses_training(tmp_path,monkeypatch):
    real, failed = failed_bundle(tmp_path)
    packet = mr.offered(real.output,real.run_id)[0]
    assert packet['training_attempt_id']==failed['train']['attempt_id']
    assert packet['evidence_id']==failed['evaluate']['attempt_id']
    producers = ch.catalog(real.output)['producers']
    assert producers[0]['origin']=='formal_measurement'
    bundle = Path(producers[0]['output_ref'])/'arbitrary_native_bundle'
    _, choice = mr.select_for_recovery(real.output,'derived','baseline',bundle.as_posix())
    original = (real.run_root/'measurements/baseline.json').read_bytes()
    training_receipt = real.run_root/'attempts'/packet['training_attempt_id']/'receipt.json'
    training_sha = digest(training_receipt)
    native = real.run_stage
    calls = []
    def evaluate_only(stage,**kw):
        calls.append(stage)
        assert stage=='evaluate'
        return native(stage,**kw)
    monkeypatch.setattr(real,'run_stage',evaluate_only)
    result = real.recover_unscored_baseline(checkpoint_handoff=choice)
    assert result['ok'] and result['metric_value']==.6, result
    assert calls==['evaluate']
    assert result['train']['attempt_id']==packet['training_attempt_id'] and result['train']['reused']
    assert digest(training_receipt)==training_sha
    backup = real.run_root/result['recovery']['original_measurement_ref']
    assert backup.read_bytes()==original
    assert read_json(real.run_root/'controller_session.json')['history'][0]['measured']
    assert mr.offered(real.output,'derived')==[]


@pytest.mark.parametrize('fault',['counter','source','native_log','held_out','wrong_train','rc_zero','inflight'])
def test_changed_evidence_or_heldout_never_authorizes_consumer_recovery(tmp_path,fault):
    real, failed = failed_bundle(tmp_path)
    packet = mr.offered(real.output,'derived')[0]
    if fault=='counter':
        producer=ch.formal_producers(real.output)[0]
        atomic_json(real.output/producer['output_ref']/'progress.json',{'completed':999})
    elif fault=='source':(real.repo/'produce.py').write_text('print("changed")')
    elif fault=='native_log':
        evidence=read_json(real.output/'evidence'/f"{packet['evidence_id']}.json")
        (real.output/evidence['log_ref']).write_text('changed native evidence')
    elif fault=='held_out':atomic_json(real.run_root/'confirmation_attempts/x.json',{})
    elif fault=='wrong_train':
        row=read_json(real.run_root/'measurements/baseline.json');row['train']['attempt_id']='f'*32
        atomic_json(real.run_root/'measurements/baseline.json',row)
    else:
        path=real.run_root/'attempts'/packet['evaluation_attempt_id']/'receipt.json'
        row=read_json(path)
        if fault=='rc_zero':row['returncode']=0
        else:row['status']='running'
        atomic_json(path,row)
    assert mr.offered(real.output,'derived')==[]


def test_recovery_cannot_substitute_verification_producer(tmp_path,monkeypatch):
    real,_=failed_bundle(tmp_path)
    monkeypatch.setattr(ch,'select',lambda *a,**kw:{'producer_evidence_ref':'evidence/'+('e'*32)+'.json'})
    with pytest.raises(ValueError,match='exact formal training producer'):
        mr.select_for_recovery(real.output,'derived','baseline','verification_outputs/probe/bundle')


def test_single_file_needs_boundary_review_when_native_verification_used_a_directory(tmp_path,monkeypatch):
    from tests.test_derived_research import research
    real=research(tmp_path)
    monkeypatch.setattr(mr,'verified_consumer_kind',lambda *a:{'kind':'directory'})
    result=real.measure(settings={},label='baseline')
    assert result['ok'] is False and result['where']=='policy_artifact'
    assert result['train']['returncode']==0
    assert result['artifact_selection']['status']=='native_boundary_review_required'
    assert not result.get('evaluate')
    from autosim.research.evidence_store import read_attempt_evidence
    proof=read_attempt_evidence(real.output,result['policy_handoff_failure']['evidence_id'])
    assert proof['status']=='operation_contract_failed' and proof['returncode'] is None
    assert result['train']['evidence_id'] != proof['evidence_id']


def test_recovery_catalog_before_workspace_creation_is_empty(tmp_path):
    assert mr.offered(tmp_path/'not-created','derived')==[]


def test_fix_returns_validated_operation_proposal_and_unchanged_diagnosis_is_reused(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from autosim.research.prepare import Preparation
    real,_=failed_bundle(tmp_path)
    packet=mr.offered(real.output,'derived')[0]
    checkpoint_ref=packet['training_output_ref']+'/arbitrary_native_bundle'
    class FixClient:
        supports_agent_fix=True
        supports_main_agent=True
        calls=0
        @contextmanager
        def as_role(self,role):
            assert role=='fix'
            yield self
        def chat_with_metadata(self,system,user,**kwargs):
            self.calls+=1
            payload=json.loads(user)
            assert kwargs['read_only'] is True
            assert payload['development_recoveries'][0]['training_attempt_id']==packet['training_attempt_id']
            assert payload['verified_train_artifacts']['producers'][0]['origin']=='formal_measurement'
            return json.dumps({'assessment':'repair_proposed',
                'summary':'Use the source-supported complete bundle of the existing formal producer',
                'changes':[],'evidence_refs':[packet['measurement_ref']],
                'proposed_action':{'step':'recover_unscored_baseline',
                                   'arguments':{'checkpoint_ref':checkpoint_ref}}}),{}
    client=FixClient()
    prep=Preparation(repo=real.repo,output=real.output,client=client,scouting=real.output/'scouting')
    monkeypatch.setattr(prep,'_decision_facts',lambda:{
        'research_progress':{'development_recoveries':[packet]},
        'verified_train_artifacts':{'producers':ch.formal_producers(real.output)}})
    monkeypatch.setattr(prep,'_persist_run_state',lambda **kw:None)
    monkeypatch.setattr(prep,'_refresh_document',lambda **kw:None)
    result=prep._step_diagnose_development_failure(label='baseline')
    assert result['outcome']=='assessed'
    assert result['fix_proposal']['arguments']=={'checkpoint_ref':checkpoint_ref}
    assert client.calls==1
    assert prep._step_diagnose_development_failure(label='baseline')['outcome']=='already assessed'
    assert client.calls==1


def test_bundle_selection_cannot_drop_the_unique_trained_payload(tmp_path):
    real,_=failed_bundle(tmp_path)
    producer=ch.formal_producers(real.output)[0]
    unrelated=real.output/producer['output_ref']/'other_container'
    unrelated.mkdir();(unrelated/'state.bin').write_bytes(b'not the trained policy')
    with pytest.raises(ValueError,match='retain the unique payload'):
        mr.select_for_recovery(real.output,'derived','baseline',unrelated.relative_to(real.output).as_posix())


def test_bundle_selection_rejects_mutated_formal_payload(tmp_path):
    real,_=failed_bundle(tmp_path)
    producer=ch.formal_producers(real.output)[0]
    bundle=real.output/producer['output_ref']/'arbitrary_native_bundle'
    (bundle/'learned.bin').write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='changed after its original archival'):
        mr.select_for_recovery(real.output,'derived','baseline',bundle.relative_to(real.output).as_posix())


@pytest.mark.parametrize('legacy_compact',[False,True])
def test_recovered_bundle_can_be_frozen_reused_without_losing_its_boundary(tmp_path,monkeypatch,legacy_compact):
    from autosim.research import frozen_policy_trial as trial
    real,_=failed_bundle(tmp_path)
    producer=ch.formal_producers(real.output)[0]
    ref=producer['output_ref']+'/arbitrary_native_bundle'
    _, choice=mr.select_for_recovery(real.output,'derived','baseline',ref)
    recovered=real.recover_unscored_baseline(checkpoint_handoff=choice)
    assert recovered['ok']
    path=real.run_root/'measurements/baseline.json'
    assert recovered['train']['checkpoint_handoff_ref']==choice['evidence_ref']
    if legacy_compact:
        # The real historical baseline only retained the exact proof in recovery.
        recovered['train'].pop('checkpoint_handoff_ref')
        atomic_json(path,recovered)
    request={'label':'baseline','measurement_sha256':digest(path)}
    original=path.read_bytes()
    selected=trial.load(real,request)
    assert selected['trained']['explicit_checkpoint_handoff']['id']==choice['id']
    native=real.run_stage;calls=[]
    def evaluation_only(stage,**kw):
        calls.append(stage)
        assert stage=='evaluate'
        return native(stage,**kw)
    monkeypatch.setattr(real,'run_stage',evaluation_only)
    result=real.measure(settings={},label='reuse_after_handoff',_frozen_policy_request=request)
    assert result['ok'] and result['metric_value']==.6, result
    assert calls==['evaluate']
    assert result['policy_artifact']==recovered['policy_artifact']
    assert path.read_bytes()==original
    assert result['train']['checkpoint_handoff_ref']==choice['evidence_ref']


@pytest.mark.parametrize('fault',['missing_proof','producer','manifest','source','conflicting_refs'])
def test_recovered_policy_proof_must_bind_the_exact_original_producer(tmp_path,fault):
    from autosim.research import frozen_policy_trial as trial
    real,_=failed_bundle(tmp_path)
    producer=ch.formal_producers(real.output)[0]
    _,choice=mr.select_for_recovery(real.output,'derived','baseline',
        producer['output_ref']+'/arbitrary_native_bundle')
    recovered=real.recover_unscored_baseline(checkpoint_handoff=choice)
    assert recovered['ok']
    path=real.run_root/'measurements/baseline.json'
    proof=real.output/choice['evidence_ref']
    if fault=='missing_proof':proof.unlink()
    elif fault in {'producer','manifest'}:
        row=read_json(proof)
        if fault=='producer':row['producer_evidence_ref']='evidence/'+('f'*32)+'.json'
        else:row['files'][0]['sha256']='f'*64
        atomic_json(proof,row)
    elif fault=='source':
        recovered['policy_artifact']['source']=str(real.output/'another-producer')
        atomic_json(path,recovered)
    else:
        recovered['train']['checkpoint_handoff_ref']='verification_checkpoint_handoffs/'+('e'*64)+'.json'
        atomic_json(path,recovered)
    with pytest.raises((ValueError,OSError)):
        trial.load(real,{'label':'baseline','measurement_sha256':digest(path)})


@pytest.mark.parametrize('fault',[None,'settings','parent','evaluation','held_out'])
def test_launcher_resumes_sealed_main_consumer_failure_without_rewriting_inputs(tmp_path,fault):
    from autosim.research.derive_and_run import validate_paused_research_resume
    from autosim.research.action_receipt import ActionReceipt,write_action_receipt
    from autosim.research.common import object_digest
    real,failed=failed_bundle(tmp_path)
    # The launcher's task is a subset of Main's original explicit learning controls.
    settings=read_json(real.run_root/'measurements/baseline.json')['settings']
    request={}
    assert settings=={}
    settings={'steps':17,'episodes':20,'seed':0}
    path=real.run_root/'measurements/baseline.json'
    row=read_json(path);row['settings']=settings;atomic_json(path,row)
    ids=[failed[stage]['attempt_id'] for stage in ('train','evaluate')]
    refs=[f'research/derived/attempts/{identity}/receipt.json' for identity in ids]
    for ref in refs:
        native=read_json(real.output/ref);native['settings_digest']=object_digest(settings)
        atomic_json(real.output/ref,native)
    session_path=real.run_root/'controller_session.json'
    session=read_json(session_path)
    session.update(schema_version=1,repository=str(real.repo),base_settings=settings,
                   input_component_sha256={'settings':object_digest(settings)})
    atomic_json(session_path,session)
    write_action_receipt(real.output,ActionReceipt(action_id='a'*16,decision_id='a'*16,
        run_id='derived',repository=str(real.repo),action='run_the_loop',state_revision=1,
        started_at='2026-10-13T00:00:00+00:00',finished_at='2026-10-13T00:01:00+00:00',
        status='yielded',reason='paid baseline consumer failed',
        child_attempt_ids=[] if fault=='parent' else ids,evidence_refs=refs))
    if fault=='settings':request={'steps':99}
    if fault=='evaluation':
        ref=real.output/refs[1];native=read_json(ref);native['status']='running';atomic_json(ref,native)
    if fault=='held_out':atomic_json(real.run_root/'confirmation_attempts/seen.json',{})
    original=session_path.read_bytes()
    if fault:
        with pytest.raises(ValueError,match='different frozen rounds/settings'):
            validate_paused_research_resume(real.output,rounds=2,settings=request,repository=real.repo)
    else:
        assert validate_paused_research_resume(real.output,rounds=2,settings=request,repository=real.repo)==2
    assert session_path.read_bytes()==original
