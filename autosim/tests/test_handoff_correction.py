from pathlib import Path
from types import SimpleNamespace

import pytest

from autosim.research.common import atomic_json, read_json
from autosim.research import handoff_correction as hc


def fixture(tmp_path):
    root = tmp_path/'research/derived'
    attempt = 'a'*32
    output = root/'train/native_outputs'/attempt
    policy = output/'snapshot/policy'
    policy.mkdir(parents=True)
    atomic_json(output/'snapshot/state.json', {'work':1000})
    atomic_json(output/'finished/state.json', {'work':8192})
    folder = root/'attempts'/attempt
    atomic_json(folder/'receipt.json', {'attempt_id':attempt,'status':'completed','returncode':0,
        'output_directory':str(output),'started_epoch':0,'training_progress':{
        'status':'observed','evidence':{'native_counter':{
            'path':'snapshot/state.json','field':['work'],'value':1000}}}})
    atomic_json(folder/'artifact_selection.json', {'attempt_id':attempt,'status':'selected',
        'selected_path':str(policy),'answer':{'selection_basis':'verified_completed_work_container'}})
    atomic_json(root/'measurements/round_4.json', {'ok':True,'label':'round_4',
        'metric_value':0,'train':{'attempt_id':attempt},'settings':{'steps':8192},
        'policy_artifact':{'source':str(policy)}})
    return root, folder


@pytest.mark.parametrize('fault', ['', 'native_best', 'failed_train', 'wrong_source', 'already_corrected'])
def test_only_incomplete_receipt_bound_scored_handoffs_are_offered(tmp_path, fault):
    root, folder = fixture(tmp_path)
    receipt = read_json(folder/'receipt.json')
    measurement = read_json(root/'measurements/round_4.json')
    if fault == 'native_best':
        receipt['training_progress']['evidence']['counter_selection'] = {'kind':'native_best'}
    if fault == 'failed_train': receipt['status'] = 'failed'
    if fault == 'wrong_source': measurement['policy_artifact']['source'] = str(tmp_path/'other')
    if fault == 'already_corrected':
        atomic_json(root/'measurements/round_4_handoff_correction.json', {'ok':True})
    atomic_json(folder/'receipt.json', receipt)
    atomic_json(root/'measurements/round_4.json', measurement)
    rows = hc.eligible(root)
    assert bool(rows) == (not fault)
    if rows:
        assert rows[0]['selected_completed_work'] == 1000
        assert rows[0]['other_counter_max'] == 8192


def test_correction_requests_reuse_and_preserves_original_score_and_selection(tmp_path):
    root, folder = fixture(tmp_path)
    originals = {p:p.read_bytes() for p in [root/'measurements/round_4.json', folder/'artifact_selection.json']}
    calls = []
    def measure(**kwargs):
        calls.append(kwargs)
        assert kwargs['_reuse_training_attempt_id'] == 'a'*32
        assert kwargs['settings'] == {'steps':8192}
        assert kwargs['label'] == 'round_4_handoff_correction'
        return {'ok':True, 'metric_value':0.2, 'label':kwargs['label']}
    research = SimpleNamespace(run_root=root, output=tmp_path, measure=measure,
                               _append_local_event=lambda event: None)
    result = hc.correct(research, label='round_4')
    assert result['handoff_correction']['training_reused']
    assert len(calls) == 1
    assert all(path.read_bytes() == data for path,data in originals.items())
    assert read_json(root/'measurements/round_4_handoff_correction.json')['metric_value'] == 0.2
    assert hc.correct(research, label='round_4')['ok'] is False


def test_path_injection_is_not_a_correction_target(tmp_path):
    root, _ = fixture(tmp_path)
    research = SimpleNamespace(run_root=root)
    assert hc.correct(research, label='../round_4')['ok'] is False


def test_graph_correction_does_not_silently_execute_training(tmp_path):
    research = SimpleNamespace(execution_graph={'score_target':'evaluate'})
    assert hc.correct(research,label='round_4')['ok'] is False


def test_unknown_previous_correction_is_not_replayed(tmp_path):
    root, folder = fixture(tmp_path)
    atomic_json(folder/'handoff_correction.json',{'status':'revalidating'})
    research = SimpleNamespace(run_root=root,output=tmp_path)
    result = hc.correct(research,label='round_4')
    assert result['ok'] is False and 'unknown' in result['why']


def test_relative_native_selection_is_resolved_against_attempt_output(tmp_path):
    root, folder = fixture(tmp_path)
    selection = read_json(folder/'artifact_selection.json')
    selection['selected_path'] = 'snapshot/policy'
    atomic_json(folder/'artifact_selection.json', selection)
    assert hc.eligible(root)[0]['label'] == 'round_4'


def test_native_correction_evaluates_without_retraining_and_preserves_old_decision(tmp_path, monkeypatch):
    from tests.test_derived_research import ambiguous_policy_research
    real = ambiguous_policy_research(tmp_path)
    original = real.measure(settings={},label='round_4')
    assert original['ok']
    attempt = original['train']['attempt_id']
    folder = real.run_root/'attempts'/attempt
    path = folder/'receipt.json'
    receipt = read_json(path)
    output = Path(receipt['output_directory'])
    atomic_json(output/'intermediate.json',{'work':4})
    atomic_json(output/'terminal.json',{'work':17})
    receipt['training_progress'] = {'status':'observed','evidence':{'native_counter':{
        'path':'intermediate.json','field':['work'],'value':4}}}
    atomic_json(path,receipt)
    selection_path = folder/'artifact_selection.json'
    selected = read_json(selection_path)
    selected['answer']['selection_basis'] = 'verified_completed_work_container'
    atomic_json(selection_path,selected)
    untouched = {p:p.read_bytes() for p in [selection_path,real.run_root/'measurements/round_4.json']}
    monkeypatch.setattr(real,'_review_detached_training_progress',lambda receipt,**kw:receipt)
    monkeypatch.setattr(real,'_policy_work_evidence',lambda *a:({'candidate_2':{'completed_work':17}},[],'new-proof'))
    old = real.run_stage
    calls = []
    def evaluate_only(stage,**kwargs):
        calls.append(stage)
        assert stage == 'evaluate'
        return old(stage,**kwargs)
    monkeypatch.setattr(real,'run_stage',evaluate_only)
    result = hc.correct(real,label='round_4')
    assert result['ok'] and calls == ['evaluate']
    assert all(path.read_bytes() == data for path,data in untouched.items())
    corrected = list(folder.glob('artifact_selection_correction_*.json'))
    assert len(corrected) == 1
    assert read_json(corrected[0])['corrects_selection_ref']


def test_failed_reaudit_returns_exact_reason_and_audit_ref_without_training(tmp_path, monkeypatch):
    from tests.test_derived_research import ambiguous_policy_research
    real = ambiguous_policy_research(tmp_path)
    original = real.measure(settings={}, label='round_4')
    attempt = original['train']['attempt_id']

    def review(receipt, *, path, allow_model, failure_details):
        failure_details.update(status='unknown',
            reason='ValueError: incomplete competing counter review',
            audit_ref='training_progress_audits/audit.json')
        return None

    monkeypatch.setattr(real, '_review_detached_training_progress', review)
    monkeypatch.setattr(real, 'run_stage', lambda *a, **kw: pytest.fail('rejection launched a stage'))
    result = real._measure(settings={}, label='round_4_handoff_correction',
        reuse_training_attempt_id=attempt, selection_correction_ref='prior-selection.json')
    assert not result['ok']
    assert 'incomplete competing counter review' in result['why']
    assert result['evidence_refs'] == ['training_progress_audits/audit.json']


@pytest.mark.parametrize('fault', ['', 'failed_train', 'evaluated', 'zero_work', 'wrong_attempt'])
def test_completed_unscored_candidate_is_offered_without_claiming_a_score(tmp_path, fault):
    root, folder = fixture(tmp_path)
    receipt = read_json(folder/'receipt.json')
    receipt.update(termination_reason='normal_exit', training_progress={'status':'unknown'})
    selected = {'attempt_id':'a'*32, 'status':'abstained'}
    measurement = {'ok':False,'status':'unscored','where':'policy_artifact','label':'round_4',
                   'train':{'attempt_id':'a'*32},'settings':{'steps':8192}}
    if fault=='failed_train': receipt['returncode']=1
    if fault=='evaluated': measurement['evaluate']={'attempt_id':'b'*32}
    if fault=='zero_work': measurement['zero_work_reported']=True
    if fault=='wrong_attempt': selected['attempt_id']='b'*32
    atomic_json(folder/'receipt.json',receipt)
    atomic_json(folder/'artifact_selection.json',selected)
    atomic_json(root/'measurements/round_4.json',measurement)
    rows=hc.eligible(root)
    assert bool(rows)==(not fault)
    if not fault:
        originals={p:p.read_bytes() for p in [folder/'receipt.json',folder/'artifact_selection.json',
                                             root/'measurements/round_4.json']}
        calls=[]
        def measure(**kw):
            calls.append(kw)
            assert kw['_audit_training_progress'] is True
            assert '_selection_correction_ref' not in kw
            return {'ok':True,'label':kw['label'],'metric_value':0.1}
        research=SimpleNamespace(run_root=root,output=tmp_path,measure=measure,
                                 _append_local_event=lambda row:None)
        assert hc.correct(research,label='round_4')['ok']
        assert len(calls)==1 and all(p.read_bytes()==data for p,data in originals.items())


@pytest.mark.parametrize('fault', ['', 'source_drift', 'corrupt_blob', 'missing_snapshot'])
def test_recovery_reinstates_exact_candidate_sources_and_rolls_back(tmp_path,fault):
    from autosim.research.snapshot import Snapshots
    from autosim.research.codepatch import Patch,apply
    repo=tmp_path/'checkout';repo.mkdir()
    source=repo/'trainer.py';source.write_text('native_train()\n')
    root=tmp_path/'research/derived'
    snapshots=Snapshots(root/'snapshots')
    snapshots.capture(['trainer.py'],repo=repo,name='before-round-4')
    patch=Patch(file='trainer.py',find='native_train()',replace='native_train(); save_manifest()')
    apply(patch,repo=repo)
    snapshots.capture(['trainer.py'],repo=repo,name='round-4')
    source.write_text('native_train()\n')
    atomic_json(root/'code_changes/round-4.json',{'status':'applied','snapshot':'before-round-4',
                                              'patches':[patch.as_dict()]})
    if fault=='source_drift': source.write_text('unrelated_edit()\n')
    if fault=='corrupt_blob':
        (snapshots.blobs/snapshots.get('before-round-4').files['trainer.py']).write_text('tampered')
    if fault=='missing_snapshot': snapshots.rows=[]
    prior=source.read_bytes()
    research=SimpleNamespace(repo=repo,run_root=root,snapshots=snapshots,_held={})
    if fault:
        with pytest.raises((ValueError,OSError)):
            with hc.candidate_sources(research,'round_4'): pytest.fail('unsafe recovery admitted')
    else:
        with hc.candidate_sources(research,'round_4'):
            assert source.read_text()=='native_train(); save_manifest()\n'
            assert research._held['trainer.py'][0]==patch
    assert source.read_bytes()==prior and research._held=={}


def test_eligibility_omits_source_drift_without_mutating_checkout(tmp_path):
    from autosim.research.snapshot import Snapshots
    from autosim.research.codepatch import Patch,apply
    root,_=fixture(tmp_path)
    repo=tmp_path/'checkout';repo.mkdir()
    source=repo/'trainer.py';source.write_text('native_train()\n')
    snapshots=Snapshots(root/'snapshots')
    snapshots.capture(['trainer.py'],repo=repo,name='before-round-4')
    patch=Patch(file='trainer.py',find='native_train()',replace='native_train(); save_manifest()')
    apply(patch,repo=repo)
    snapshots.capture(['trainer.py'],repo=repo,name='round-4')
    atomic_json(root/'code_changes/round-4.json',{'status':'applied','snapshot':'before-round-4',
                                              'patches':[patch.as_dict()]})
    atomic_json(root/'controller_session.json',{'repository':str(repo)})
    source.write_text('native_train()\n')
    assert hc.eligible(root)
    assert source.read_text()=='native_train()\n'
    source.write_text('unrelated_change()\n')
    assert hc.eligible(root)==[]
    assert source.read_text()=='unrelated_change()\n'


def test_completed_unscored_reaudit_failure_reaches_controller_with_stable_evidence(tmp_path,monkeypatch):
    from tests.test_derived_research import ambiguous_policy_research
    real=ambiguous_policy_research(tmp_path)
    original=real.measure(settings={},label='round_4')
    attempt=original['train']['attempt_id']
    path=real.run_root/'attempts'/attempt/'receipt.json'
    receipt=read_json(path);receipt['training_progress']={'status':'unknown'}
    atomic_json(path,receipt)
    def review(receipt,*,path,allow_model,failure_details):
        assert allow_model
        failure_details.update(status='unknown',reason='Supervisor malformed evidence',
                               audit_ref='training_progress_audits/stable.json')
        return None
    monkeypatch.setattr(real,'_review_detached_training_progress',review)
    monkeypatch.setattr(real,'run_stage',lambda *a,**kw:pytest.fail('re-audit failure launched native work'))
    result=real.measure(settings={},label='round_4_handoff_correction',
                        _reuse_training_attempt_id=attempt,_audit_training_progress=True)
    assert not result['ok'] and 'Supervisor malformed evidence' in result['why']
    assert result['evidence_refs']==['training_progress_audits/stable.json']


def test_artifact_selection_switches_from_fix_to_readonly_supervisor(tmp_path):
    from contextlib import contextmanager
    from tests.test_derived_research import ambiguous_policy_research,ArtifactSelectionClient
    real=ambiguous_policy_research(tmp_path)
    class Client(ArtifactSelectionClient):
        role='fix'
        @contextmanager
        def as_role(self,role):
            old=self.role;self.role=role
            try: yield
            finally: self.role=old
        def chat_with_metadata(self,*args,**kw):
            assert self.role=='supervisor'
            assert kw['decision_only'] and kw['read_only']
            assert not kw['auto_skills'] and not kw['include_research_context'] and not kw['resume_session']
            return super().chat_with_metadata(*args,**kw)
    real.client=Client()
    result=real.measure(settings={},label='round_4')
    assert result['ok'] and real.client.role=='fix'


def test_unknown_progress_defers_selection_without_calling_model(tmp_path,monkeypatch):
    from tests.test_derived_research import ambiguous_policy_research
    real=ambiguous_policy_research(tmp_path)
    record=real.run_stage('train',settings={})
    record['training_progress']={'status':'unknown'}
    monkeypatch.setattr(real.client,'chat_with_metadata',lambda *a,**kw:pytest.fail('empty proof paid resolver'))
    selected=real._select_policy_artifact(record)
    assert selected['artifact_selection']['status']=='awaiting_progress_audit'
    ref=real.output/selected['artifact_selection']['selection_ref']
    assert read_json(ref)['status']=='unavailable'


@pytest.mark.parametrize('first',['quote_once','abstain','transport'])
def test_artifact_reply_has_one_bounded_delivery_repair_and_sealed_request(tmp_path,first):
    import json
    from tests.test_derived_research import ambiguous_policy_research,ArtifactSelectionClient
    real=ambiguous_policy_research(tmp_path)
    timeouts=[]
    class Client(ArtifactSelectionClient):
        def chat_with_metadata(self,system,user,**kw):
            timeouts.append(kw['timeout'])
            if len(timeouts)==1 and first=='transport': raise OSError('unavailable provider')
            text,provider=super().chat_with_metadata(system,user,**kw)
            answer=json.loads(text)
            if len(timeouts)==1:
                if first=='quote_once': answer['evidence'][0]['quote']='invented quote'
                if first=='abstain': answer.update(decision='abstain',why='insufficient evidence')
            else:
                assert json.loads(user)['format_feedback']
                assert kw['timeout']<timeouts[0]
            return json.dumps(answer),{'turn_id':str(len(timeouts))}
    real.client=Client()
    result=real.measure(settings={},label='round_4')
    assert len(timeouts)==(2 if first=='quote_once' else 1)
    assert bool(result['ok'])==(first=='quote_once')
    selection=read_json(real.output/result['train']['artifact_selection']['selection_ref'])
    assert selection['request_contract']=='source_quote_transport_v5'
    assert read_json(real.output/selection['request_ref'])['payload']['candidate_ids']
    assert len(selection['responses'])==(0 if first=='transport' else len(timeouts))


def test_corrected_request_contract_does_not_overwrite_old_negative_selection(tmp_path):
    from tests.test_derived_research import ambiguous_policy_research,ArtifactSelectionClient
    real=ambiguous_policy_research(tmp_path,selector_mode='invalid_quote')
    record=real.run_stage('train',settings={})
    failed=real._select_policy_artifact(record)
    original_ref=failed['artifact_selection']['selection_ref']
    original=real.output/original_ref
    old=read_json(original);old.pop('request_contract')
    atomic_json(original,old) # Simulate a legacy immutable receipt before upgrade.
    held=original.read_bytes()
    real.client=ArtifactSelectionClient()
    selected=real._select_policy_artifact(record,revalidate_abstained=True)
    assert selected['artifact_selection']['status']=='selected'
    assert selected['artifact_selection']['selection_ref']!=original_ref
    assert original.read_bytes()==held and real.client.calls==1
