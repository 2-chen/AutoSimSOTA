"""Synthetic sealed native records; no benchmark, provider API, or GPU use."""
import json
from pathlib import Path

import pytest

from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.experiment_bundle import freeze_artifact
from autosim.research.metric_contract import MetricSpec
from autosim.research.policy_consumption import verify_policy_consumption, verify_rollout_evidence
from autosim.research.reference_score_recovery import recover, scoring_receipt
from autosim.research.receipt_verifier import verify_measurement
from tests.test_derived_research import research


def fixture(tmp_path, monkeypatch):
    real = research(tmp_path)
    from autosim.research.workspace_snapshot import create
    source = tmp_path/'source';source.mkdir()
    (source/'x.py').write_text('native_evaluator = True\n')
    (source/'README.md').write_text('Published reference 0.5 over 2 episodes.')
    repo = real.output/'checkout'
    create(source,repo)
    real.repo=repo;real.backend.repo=repo
    source_state=real._capture_workspace_source_state('reference_failed')
    spec = MetricSpec.from_declaration({'research_goal': {'primary_metric': {
        'name':'success_rate','direction':'maximize','source':'json','artifact_root':'output',
        'artifact_pattern':'results/old/score.json','json_key':'episodes',
        'json_value_key':'success','episode_id_column':'episode_id','aggregation':'mean'}}})
    real.metric_spec = spec
    real.backend.stages['evaluate']['artifact'] = 'results/**'
    specimen = real.output/'verification_outputs/results/old/score.json'
    atomic_json(specimen, {'episodes':[{'episode_id':'smoke','success':False}]})
    atomic_json(real.output/'metric_binding.json', {'primary_metric': {
        'name':'success_rate','direction':'maximize','source':'json','artifact_root':'output',
        'artifact_pattern':spec.artifact_pattern,'json_key':'episodes','json_value_key':'success',
        'episode_id_column':'episode_id','aggregation':'mean'}, 'evidence':'native source',
        'verified_artifact':{'path':str(specimen),'sha256':digest(specimen),
            'root':'output','pattern':spec.artifact_pattern}})
    settings = {'episodes':2}
    protocol = real.run_root/'reference_protocols/public.json'
    atomic_json(protocol, {'target':'evaluate','settings':settings,'metric':spec.as_dict(),
                          'evaluator_stage':{'artifact':'results/**'}})
    source = real.output/'original-policy.bin'; source.write_bytes(b'synthetic policy')
    policy = freeze_artifact(source, real.run_root/'policy.bin', max_bytes=1024)
    identity = policy['content_sha256']; attempt = 'a'*32
    output = real.run_root/'evaluate/native_outputs'/attempt
    artifact = output/'results/new/score.json'
    atomic_json(artifact, {'episodes':[{'episode_id':'e0','success':True},
                                     {'episode_id':'e1','success':False}]})
    (artifact.parent/'demo.mp4').write_bytes(b'synthetic media')
    log = real.run_root/'evaluate/attempts'/attempt/'output.log'
    log.parent.mkdir(parents=True)
    text = 'AUTOSIM_POLICY_LOADED '+json.dumps({'path':policy['path'],'content_sha256':identity})+'\n'
    for episode in ['e0','e1']:
        text += 'AUTOSIM_ROLLOUT_COMPLETED '+json.dumps({'episode_id':episode,'policy_sha256':identity})+'\n'
    text += 'AUTOSIM_METRIC_REPORTED '+json.dumps({'episode_ids':['e0','e1'],'policy_sha256':identity,'value':0.5})+'\n'
    log.write_text(text)
    receipt = dict(attempt_id=attempt, node_id='evaluate',status='completed',returncode=0,
        termination_reason='normal_exit',ran=True,log=str(log),output_directory=str(output),
        started_epoch=artifact.stat().st_mtime-1,settings_digest=object_digest(settings),
        reference_protocol_ref='reference_protocols/public.json',comparison_protocol_sha256=digest(protocol),
        postconditions=[{'id':'metric_result_artifact','passed':False,'required_for_score':True}])
    native_path = real.run_root/'attempts'/attempt/'receipt.json'
    atomic_json(native_path, receipt)
    capture_attempt_evidence(real.output, attempt_id=attempt,log=log,
        receipt_ref=native_path.relative_to(real.output).as_posix(),status='completed',
        returncode=0,termination_reason='normal_exit')
    old = dict(label='reference_failed',ok=False,where='evaluate',why='result missing',
        settings=settings,metric=spec.as_dict(),metric_value=None,metric_utility=None,
        baseline_origin='released_reference',reference_protocol_ref='reference_protocols/public.json',
        policy_artifact=policy,policy_consumption=verify_policy_consumption(log,Path(policy['path'])),
        rollout_evidence=verify_rollout_evidence(log,policy_sha256=identity),
        metric_lineage={'status':'not_required'},metric_artifact_evidence={'status':'missing'},
        source_state=source_state,evaluate={'attempt_id':attempt,'returncode':0},
        result_artifact={},train={})
    path = real.run_root/'measurements/reference_failed.json';atomic_json(path,old)
    monkeypatch.setattr(real,'run_stage',lambda *a,**kw:pytest.fail('recovery must never launch native work'))
    return real, path.relative_to(real.output).as_posix(), native_path, artifact


class LocatorReviewer:
    def __init__(self, refuse=False):
        self.calls=[];self.refuse=refuse
    def chat_with_metadata(self, system, user, **kwargs):
        self.calls.append((system,user,kwargs))
        return json.dumps({'source_output_connection_supported':not self.refuse,
            'original_attempt_scope_supported':True,'native_semantics_preserved':True,
            'frozen_metric_mapping_preserved':True,'why':'Synthetic source review',
            'citations':[{'source_id':'review-source-1'}]}),{}


def narrow_fixture(tmp_path,monkeypatch):
    real,ref,native,artifact=fixture(tmp_path,monkeypatch)
    real.backend.stages['evaluate']['artifact']='results/old/score.json'
    protocol=real.run_root/'reference_protocols/public.json';p=read_json(protocol)
    p['evaluator_stage']['artifact']='results/old/score.json';atomic_json(protocol,p)
    receipt=read_json(native);receipt['comparison_protocol_sha256']=digest(protocol);atomic_json(native,receipt)
    capture_attempt_evidence(real.output,attempt_id=receipt['attempt_id'],log=Path(receipt['log']),
        receipt_ref=native.relative_to(real.output).as_posix(),status='completed',returncode=0,
        termination_reason='normal_exit')
    revision={'pattern':'results/new/score.json','reason':'writer output tag differs from the old specimen',
              'source_refs':[{'ref':'x.py','quote':'native_evaluator = True'}]}
    return real,ref,native,artifact,revision


def test_reviewed_locator_recovers_same_native_result_without_changing_frozen_metric(tmp_path,monkeypatch):
    real,ref,native,artifact,revision=narrow_fixture(tmp_path,monkeypatch)
    old_bytes=(real.output/ref).read_bytes();native_bytes=native.read_bytes()
    protocol_bytes=(real.run_root/'reference_protocols/public.json').read_bytes()
    reviewer=LocatorReviewer()
    result,newref=recover(real,reference_id='public',original_ref=ref,locator_revision=revision,client=reviewer)
    assert result['ok'] is True and result['metric_value']==.5
    assert result['metric']==read_json(real.output/ref)['metric']
    assert result['metric_artifact_evidence']['locator_pattern']==revision['pattern']
    assert result['metric_artifact_evidence']['pattern']=='results/old/score.json'
    assert (real.output/ref).read_bytes()==old_bytes and native.read_bytes()==native_bytes
    assert (real.run_root/'reference_protocols/public.json').read_bytes()==protocol_bytes
    assert verify_measurement(real.run_root,Path(newref).stem)['status']=='consistent'
    assert len(reviewer.calls)==1 and reviewer.calls[0][2]['read_only'] is True
    assert '0.5' not in reviewer.calls[0][1]  # review did not choose by score
    attachment=read_json(real.run_root/result['scoring_recovery']['attachment_ref'])
    assert attachment['locator_review']['ref'].startswith('reference_locator_reviews/')


@pytest.mark.parametrize('fault',['review_refused','missing_reviewer','unsafe_pattern','fake_source',
                                  'duplicate','written_after_exit','changed_during_review'])
def test_reviewed_locator_refuses_bad_scope_or_evidence(tmp_path,monkeypatch,fault):
    import os
    real,ref,native,artifact,revision=narrow_fixture(tmp_path,monkeypatch)
    reviewer=LocatorReviewer(refuse=fault=='review_refused')
    if fault=='missing_reviewer':reviewer=None
    elif fault=='unsafe_pattern':revision['pattern']='../score.json'
    elif fault=='fake_source':revision['source_refs'][0]['quote']='not present'
    elif fault=='duplicate':
        revision['pattern']='results/*/score.json'
        atomic_json(artifact.parent.parent/'other/score.json',read_json(artifact))
    elif fault=='written_after_exit':
        end=Path(read_json(native)['log']).stat().st_mtime
        os.utime(artifact,(end+10,end+10))
    elif fault=='changed_during_review':
        class Racing(LocatorReviewer):
            def chat_with_metadata(self,*a,**kw):
                answer=super().chat_with_metadata(*a,**kw)
                atomic_json(artifact,{'changed':True})
                return answer
        reviewer=Racing()
    before=(real.output/ref).read_bytes();receipt=native.read_bytes()
    with pytest.raises(ValueError):
        recover(real,reference_id='public',original_ref=ref,locator_revision=revision,client=reviewer)
    assert (real.output/ref).read_bytes()==before and native.read_bytes()==receipt
    assert len(list((real.run_root/'measurements').glob('*.json')))==1


def test_locator_certificate_tamper_invalidates_recovered_measurement(tmp_path,monkeypatch):
    real,ref,native,artifact,revision=narrow_fixture(tmp_path,monkeypatch)
    result,newref=recover(real,reference_id='public',original_ref=ref,
                          locator_revision=revision,client=LocatorReviewer())
    attachment=read_json(real.run_root/result['scoring_recovery']['attachment_ref'])
    review=real.run_root/attachment['locator_review']['ref'];value=read_json(review)
    value['review']['native_semantics_preserved']=False;atomic_json(review,value)
    with pytest.raises(ValueError):scoring_receipt(real.run_root,result,read_json(native))
    assert verify_measurement(real.run_root,Path(newref).stem)['status']!='consistent'


@pytest.mark.parametrize('fault',['metric_value','episode_count'])
def test_locator_approval_cannot_approve_wrong_native_metric_or_episodes(tmp_path,monkeypatch,fault):
    import os
    real,ref,native,artifact,revision=narrow_fixture(tmp_path,monkeypatch)
    oldmtime=artifact.stat().st_mtime
    rows=[{'episode_id':'e0','success':True},{'episode_id':'e1','success':True}]
    if fault=='episode_count':rows=rows[:1]
    atomic_json(artifact,{'episodes':rows});os.utime(artifact,(oldmtime,oldmtime))
    before=(real.output/ref).read_bytes();receipt=native.read_bytes()
    with pytest.raises(ValueError,match='lineage'):
        recover(real,reference_id='public',original_ref=ref,locator_revision=revision,client=LocatorReviewer())
    assert (real.output/ref).read_bytes()==before and native.read_bytes()==receipt
    assert len(list((real.run_root/'measurements').glob('*.json')))==1


def test_additive_recovery_preserves_native_and_failed_records(tmp_path, monkeypatch):
    real, ref, native, artifact = fixture(tmp_path,monkeypatch)
    old_bytes = (real.output/ref).read_bytes(); native_bytes=native.read_bytes()
    result,newref = recover(real,reference_id='public',original_ref=ref)
    assert result['ok'] and result['metric_value']==0.5
    assert result['metric_reading']['samples']==2
    assert result['scoring_recovery']['native_replayed'] is False
    assert (real.output/ref).read_bytes()==old_bytes and native.read_bytes()==native_bytes
    assert verify_measurement(real.run_root,Path(newref).stem)['status']=='consistent'
    assert getattr(real,'_reference_protocol_id','')==''


def test_source_identity_contains_snapshot_label_but_recovery_verifies_original(tmp_path,monkeypatch):
    real,ref,native,_=fixture(tmp_path,monkeypatch)
    original=read_json(real.output/ref)['source_state']
    renamed=real._capture_workspace_source_state('another-label-same-bytes')
    assert renamed['overlay_files']==original['overlay_files']
    assert renamed['source_tree_fingerprint']==original['source_tree_fingerprint']
    assert renamed['identity_sha256']!=original['identity_sha256']
    result,newref=recover(real,reference_id='public',original_ref=ref)
    assert result['source_state']==original
    assert verify_measurement(real.run_root,Path(newref).stem)['status']=='consistent'


@pytest.mark.parametrize('fault',['metric','protocol','source','native_exit','native_log','other_postcondition','duplicate','symlink','different_log','different_receipt'])
def test_recovery_refuses_unrelated_or_changed_evidence(tmp_path,monkeypatch,fault):
    real,ref,native,artifact=fixture(tmp_path,monkeypatch)
    receipt=read_json(native)
    if fault=='metric':real.metric_spec=MetricSpec(name='other')
    elif fault=='protocol':(real.run_root/'reference_protocols/public.json').write_text('{}')
    elif fault=='source':(real.repo/'x.py').write_text('changed_source = True\n')
    elif fault=='native_exit':receipt['returncode']=1;atomic_json(native,receipt)
    elif fault=='native_log':Path(receipt['log']).write_text('changed')
    elif fault=='different_log':
        other=Path(receipt['log']).with_name('other.log');other.write_text(Path(receipt['log']).read_text())
        receipt['log']=str(other);atomic_json(native,receipt)
    elif fault=='different_receipt':
        p=real.output/'evidence'/('a'*32+'.json');sealed=read_json(p)
        sealed['receipt_ref']='unrelated/receipt.json';atomic_json(p,sealed)
    elif fault=='other_postcondition':
        receipt['postconditions'].append({'id':'security','passed':False});atomic_json(native,receipt)
    elif fault=='duplicate':atomic_json(artifact.parent.parent/'another/score.json',read_json(artifact))
    elif fault=='symlink':
        moved=artifact.parent.parent.parent/'elsewhere';artifact.parent.rename(moved)
        artifact.parent.symlink_to(moved,target_is_directory=True)
    with pytest.raises((ValueError,OSError)):
        recover(real,reference_id='public',original_ref=ref)
    for p in (real.run_root/'measurements').glob('reference_recovered_*.json'):
        assert read_json(p)['ok'] is False


@pytest.mark.parametrize('fault',['attachment','original','native','settings','metric','value','policy','unsafe_ref'])
def test_independent_verifier_rejects_scoring_supplement_tampering(tmp_path,monkeypatch,fault):
    real,ref,native,_=fixture(tmp_path,monkeypatch)
    result,newref=recover(real,reference_id='public',original_ref=ref)
    if fault=='attachment':(real.run_root/result['scoring_recovery']['attachment_ref']).write_text('{}')
    elif fault=='original':(real.output/ref).write_text('{}')
    elif fault=='native':native.write_text('{}')
    elif fault=='settings':result['settings']={'episodes':1}
    elif fault=='metric':result['metric']['json_key']='invented'
    elif fault=='value':result['metric_value']=1
    elif fault=='policy':Path(result['policy_artifact']['path']).write_bytes(b'changed')
    elif fault=='unsafe_ref':result['scoring_recovery']['attachment_ref']='../other.json'
    atomic_json(real.output/newref,result)
    assert verify_measurement(real.run_root,Path(newref).stem)['status']!='consistent'


def test_scoring_supplement_is_readonly_for_fix():
    from autosim.research.harness_repair import READABLE,ALLOWED
    assert 'research/reference_score_recovery.py' in READABLE
    assert 'research/reference_score_recovery.py' not in ALLOWED


def test_main_recovery_operation_is_not_baseline_adoption(tmp_path,monkeypatch):
    from autosim.research import baseline_reference as refs
    from tests.test_main_agent import make
    real,original,native,_=fixture(tmp_path/'native',monkeypatch)
    controller=make(tmp_path/'controller')
    controller.output=real.output;controller.repo=real.repo;controller.run_id=real.run_id
    monkeypatch.setattr(controller,'_research_controller',lambda:(real,{}))
    (real.repo/'README.md').write_text('Published reference 0.5 over 2 episodes.')
    spec=refs.validate(real.repo,dict(id='public',kind='released_checkpoint',
        reason='synthetic reference for recovery contracts',
        source_refs=[{'ref':'README.md','quote':'Published reference 0.5 over 2 episodes.'}],
        checkpoint_ref='original-policy.bin',release_url='https://example.org/model',revision='abc',
        episodes=2,expected_metric=0.5,tolerance=0.01))
    row=read_json(real.output/original)
    frozen=row['policy_artifact']
    # The synthetic frozen policy was copied from original-policy.bin.
    # Registry ownership must instead point to the policy's original source;
    # complete() checks this separately from receipt consistency.
    refs.register(real.output,spec,dict(sources_supported=True,protocol_matches=True,
        tolerance_supported=True,release_identity_supported=True),
        frozen_checkpoint={'path':frozen['source'],
            'content_sha256':frozen['content_sha256']})
    refs.begin(real.output,'public',measurement_ref=original)
    assert refs.complete(real.output,'public',row,original)['status']=='evaluation_failed'
    result=controller._step_recover_reference_score(reference_id='public',
        reason='repair missing timestamp artifact binding without replay')
    assert result['outcome']=='recovered' and result['reference_status']=='reproduced',result
    assert result['metric_value']==0.5 and result['native_replayed'] is False
    assert read_json(real.output/original)['ok'] is False
    assert not (real.run_root/'measurements/baseline.json').exists()
    assert not (real.run_root/'comparison_protocol.json').exists()
@pytest.mark.parametrize('ref_mode',['matching','wrong_ref','unknown_field','negative_verdict'])
def test_redundant_review_reference_is_exact_and_cannot_override_a_negative_verdict(tmp_path,monkeypatch,ref_mode):
    real,ref,native,artifact,revision=narrow_fixture(tmp_path,monkeypatch)
    class WithRef(LocatorReviewer):
        def chat_with_metadata(self,system,user,**kwargs):
            answer,meta=super().chat_with_metadata(system,user,**kwargs)
            row=json.loads(answer);source=json.loads(user)['review_sources'][0]
            row['citations']=[{'source_id':source['source_id'],
                'ref':'wrong.py' if ref_mode=='wrong_ref' else source['ref']}]
            if ref_mode=='unknown_field':row['citations'][0]['unsupported']='ignored?'
            if ref_mode=='negative_verdict':row['source_output_connection_supported']=False
            return json.dumps(row),meta
    if ref_mode=='matching':
        result,newref=recover(real,reference_id='public',original_ref=ref,locator_revision=revision,client=WithRef())
        assert result['ok'] and verify_measurement(real.run_root,Path(newref).stem)['status']=='consistent'
    else:
        from autosim.research.reference_score_recovery import LocatorReviewRefused
        with pytest.raises(LocatorReviewRefused) as caught:
            recover(real,reference_id='public',original_ref=ref,locator_revision=revision,client=WithRef())
        if ref_mode=='negative_verdict':
            review=read_json(real.run_root/caught.value.review_ref)
            assert review['review_error']=='' and review['status']=='rejected'
            assert review['review']['source_output_connection_supported'] is False
