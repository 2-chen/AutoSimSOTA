from pathlib import Path
from types import SimpleNamespace

import pytest

from autosim.research import baseline_reference as refs
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.experiment_bundle import freeze_artifact
from autosim.research.round_extension import extend
from autosim.research.derive_and_run import validate_paused_research_resume
from tests.test_derived_research import research


def test_authorized_resource_ref_has_distinct_backing_and_native_paths(tmp_path,monkeypatch):
    output=tmp_path/'run';repo=output/'checkout';repo.mkdir(parents=True)
    source=tmp_path/'resources';source.mkdir();(source/'model.bin').write_bytes(b'weights')
    (repo/'models').mkdir()
    monkeypatch.setattr('autosim.research.workspace_resources.bindings_for',
        lambda *a:[dict(target='models',source=str(source))])
    for ref in ['models/model.bin','checkout/models/model.bin']:
        backing,native=refs.checkpoint_location(output,repo,ref)
        assert backing==source/'model.bin' and native==repo/'models/model.bin'
    (output/'models').mkdir();(output/'models/model.bin').write_bytes(b'other')
    with pytest.raises(ValueError,match='ambiguous'):
        refs.resolve_checkpoint(output,repo,'models/model.bin')
    assert refs.resolve_checkpoint(output,repo,'checkout/models/model.bin')==source/'model.bin'


def test_resource_ref_does_not_allow_symlink_escape(tmp_path,monkeypatch):
    output=tmp_path/'run';repo=output/'checkout';repo.mkdir(parents=True)
    source=tmp_path/'resources';source.mkdir()
    outside=tmp_path/'other';outside.mkdir();(outside/'model.bin').write_bytes(b'private')
    (source/'escape').symlink_to(outside,target_is_directory=True)
    monkeypatch.setattr('autosim.research.workspace_resources.bindings_for',
        lambda *a:[dict(target='models',source=str(source))])
    with pytest.raises(ValueError,match='boundary'):
        refs.resolve_checkpoint(output,repo,'models/escape/model.bin')


def test_cli_resume_uses_applied_round_allocation_without_reset(tmp_path):
    real=research(tmp_path)
    real.run(rounds=2,yield_after_action=True)
    real.budget=SimpleNamespace(remaining=lambda:1000)
    receipt=extend(real,additional_rounds=2,expected_rounds=2,reason='next candidate batch')
    assert validate_paused_research_resume(real.output,rounds=2,settings={},repository=real.repo)==4
    extend(real,additional_rounds=2,expected_rounds=4,reason='one more batch')
    assert validate_paused_research_resume(real.output,rounds=2,settings={},repository=real.repo)==6
    with pytest.raises(ValueError,match='different frozen'):
        validate_paused_research_resume(real.output,rounds=3,settings={})
    path=real.output/receipt['receipt_ref']
    row=read_json(path);row['status']='prepared';atomic_json(path,row)
    with pytest.raises(ValueError,match='extension chain'):
        validate_paused_research_resume(real.output,rounds=2,settings={})


def test_cli_extension_receipt_cannot_hide_baseline_or_settings_change(tmp_path):
    real=research(tmp_path)
    real.run(rounds=2,yield_after_action=True)
    real.budget=SimpleNamespace(remaining=lambda:1000)
    extend(real,additional_rounds=2,expected_rounds=2,reason='next candidate batch')
    with pytest.raises(ValueError,match='different frozen'):
        validate_paused_research_resume(real.output,rounds=2,settings={'steps':999})
    (real.run_root/'measurements/baseline.json').write_text('{}')
    with pytest.raises(ValueError,match='extension chain'):
        validate_paused_research_resume(real.output,rounds=2,settings={})


def reference(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    (repo/'README.md').write_text('Published policy: success 0.51 over 100 episodes, revision abc.')
    output = tmp_path/'out'
    output.mkdir()
    weight = output/'release.bin'
    weight.write_bytes(b'released policy')
    spec = refs.validate(repo, dict(id='published', kind='released_checkpoint',
        source_refs=[dict(ref='README.md', quote='success 0.51 over 100 episodes')],
        reason='Verify native published reference before claiming reproduction',
        release_url='https://example.org/model', revision='abc', checkpoint_ref='release.bin',
        episodes=100, expected_metric=0.51, tolerance=0.05))
    frozen = freeze_artifact(weight, output/'baseline_references/published/policy', max_bytes=1000)
    review = dict(sources_supported=True, protocol_matches=True, tolerance_supported=True,
                  release_identity_supported=True)
    refs.register(output, spec, review, frozen_checkpoint=frozen)
    return repo, output, spec, frozen


def measurement(output, frozen, value=0.51, samples=100):
    copied = freeze_artifact(Path(frozen['path']), output/'measured/policy', max_bytes=1000)
    row = dict(ok=True, metric_value=value, metric_reading=dict(samples=samples),
               policy_artifact=copied, **{k:dict(status='verified') for k in
               ('policy_consumption','rollout_evidence','metric_lineage')})
    rel = 'measurements/reference.json'
    atomic_json(output/rel, row)
    refs.begin(output, 'published', measurement_ref=rel)
    return row, rel


@pytest.mark.parametrize('value,samples,status', [(0.51,100,'reproduced'),
    (0.0,100,'performance_mismatch'), (0.51,1,'evaluation_failed')])
def test_reference_performance_is_not_measurement_validity(tmp_path, monkeypatch, value, samples, status):
    # Isolate classification; receipt consistency has its own real-process tests.
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
                        lambda *args:dict(status='consistent'))
    _, output, _, frozen = reference(tmp_path)
    row, rel = measurement(output, frozen, value, samples)
    assert refs.complete(output, 'published', row, rel)['status'] == status
    atomic_json(output/rel, {**row,'metric_value':1.0})
    assert refs.view(output)['status'] == 'unverified'


def test_unrelated_policy_cannot_reproduce_reference(tmp_path):
    _, output, _, frozen = reference(tmp_path)
    row, rel = measurement(output, frozen)
    row['policy_artifact']['source'] = str(output/'unrelated')
    atomic_json(output/rel, row)
    assert refs.complete(output, 'published', row, rel)['status'] == 'evaluation_failed'


def test_reference_requires_exact_public_source_and_immutable_bytes(tmp_path):
    repo, output, _, frozen = reference(tmp_path)
    with pytest.raises(ValueError):
        refs.source_evidence(repo, ['README.md'])
    with pytest.raises(ValueError):
        refs.source_evidence(repo, [dict(ref='README.md',quote='not present')])
    assert refs.handoff(output,'published')['selected_path'] == frozen['path']
    Path(frozen['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        refs.handoff(output,'published')


def test_extend_preserves_baseline_history_protocol_and_budget(tmp_path):
    real = research(tmp_path)
    real.run(rounds=1, yield_after_action=True)
    real.run(rounds=1, yield_after_action=True)
    root = real.run_root
    session_path = root/'controller_session.json'
    before = read_json(session_path)
    baseline = digest(root/'measurements/baseline.json')
    protocol = digest(root/'comparison_protocol.json')
    real.budget = SimpleNamespace(remaining=lambda:1000)
    result = extend(real, additional_rounds=2, expected_rounds=1, reason='Continue within total budget')
    after = read_json(session_path)
    assert result['new_rounds'] == after['rounds'] == 3
    assert after['history'] == before['history']
    assert after['base_settings'] == before['base_settings']
    assert after['next_round'] == before['next_round']
    assert digest(root/'measurements/baseline.json') == baseline
    assert digest(root/'comparison_protocol.json') == protocol
    assert extend(real, additional_rounds=2, expected_rounds=1,
                  reason='Continue within total budget')['id'] == result['id']
    real.budget = None
    real.run(rounds=3, yield_after_action=True)
    assert digest(root/'measurements/baseline.json') == baseline
    assert read_json(session_path)['rounds'] == 3


def test_extension_forbidden_after_heldout_or_without_budget(tmp_path):
    real = research(tmp_path)
    real.run(rounds=1, yield_after_action=True)
    real.budget = SimpleNamespace(remaining=lambda:0)
    with pytest.raises(RuntimeError):
        extend(real, additional_rounds=1, expected_rounds=1, reason='No funds')
    real.budget = SimpleNamespace(remaining=lambda:1000)
    atomic_json(real.run_root/'confirmation_attempts/one.json', {})
    with pytest.raises(RuntimeError):
        extend(real, additional_rounds=1, expected_rounds=1, reason='No test leakage')


def test_released_reference_evaluation_never_invokes_trainer(tmp_path, monkeypatch):
    real = research(tmp_path)
    weight = tmp_path/'published.pth'
    weight.write_bytes(b'released checkpoint')
    called = []
    original = real.run_stage
    def observed(stage, **kwargs):
        called.append(stage)
        return original(stage, **kwargs)
    monkeypatch.setattr(real, 'run_stage', observed)
    result = real.measure(settings={}, label='reference_test', _reference_checkpoint=weight)
    assert 'train' not in called
    assert called == ['evaluate']
    assert result['baseline_origin'] == 'released_reference'
    assert result['policy_artifact']['source'] == str(weight)


def install_reference(real):
    (real.repo/'README.md').write_text('Published same-family policy: success 0.50 over 100 episodes, revision abc.')
    weight = real.output/'published.pth';weight.write_bytes(b'published native weights')
    spec = refs.validate(real.repo, dict(id='published',kind='released_checkpoint',
        source_refs=[dict(ref='README.md',quote='success 0.50 over 100 episodes')],
        reason='Independent published-reference diagnosis',release_url='https://example.org/model',
        revision='abc',checkpoint_ref='published.pth',episodes=100,expected_metric=0.5,tolerance=0.01))
    frozen = freeze_artifact(weight,real.output/'baseline_references/published/policy',max_bytes=1000)
    refs.register(real.output,spec,dict(sources_supported=True,protocol_matches=True,
        tolerance_supported=True,release_identity_supported=True),frozen_checkpoint=frozen)
    return Path(frozen['path'])


@pytest.mark.parametrize('local_first',[False,True])
def test_real_cpu_reference_evaluation_has_separate_protocol_and_no_training(tmp_path,monkeypatch,local_first):
    real=research(tmp_path)
    weight=install_reference(real)
    local=real.run_root/'comparison_protocol.json'
    if local_first:
        assert not real._comparison_protocol_violation({'episodes':10,'seed':0},target='evaluate',freeze=True)
    old=local.read_bytes() if local.exists() else None
    calls=[];native=real.run_stage
    def observed(stage,**kwargs):
        calls.append(stage)
        return native(stage,**kwargs)
    monkeypatch.setattr(real,'run_stage',observed)
    result=real.measure(settings={'episodes':100,'seed':0},label='reference_native',
        _reference_checkpoint=weight,_reference_protocol_id='published')
    assert result['ok'] and calls == ['evaluate']
    assert local.read_bytes()==old if old is not None else not local.exists()
    protocol=real.run_root/result['reference_protocol_ref']
    assert read_json(protocol)['settings']['episodes']==100
    attempt=result['evaluate']['attempt_id']
    receipt=read_json(real.run_root/'attempts'/attempt/'receipt.json')
    assert receipt['reference_protocol_ref']==result['reference_protocol_ref']
    assert receipt['comparison_protocol_sha256']==digest(protocol)
    from autosim.research.receipt_verifier import verify_measurement
    checked=verify_measurement(real.run_root,'reference_native')
    # This toy only exercises protocol/process integrity, not real policy identity.
    assert checked['checks']['comparison_protocol']
    assert checked['checks']['comparison_protocol_bytes']
    assert not getattr(real,'_reference_protocol_id','')
    if not local_first:
        assert not real._comparison_protocol_violation({'episodes':10,'seed':0},target='evaluate',freeze=True)


def test_reference_scope_still_refuses_local_evaluator_and_heldout_drift(tmp_path):
    real=research(tmp_path)
    weight=install_reference(real)
    assert not real._comparison_protocol_violation({'episodes':10,'seed':0},target='evaluate',freeze=True)
    old=(real.run_root/'comparison_protocol.json').read_bytes()
    real.backend.sources['evaluate']+='\n# changed evaluator\n'
    result=real.measure(settings={'episodes':100,'seed':0},label='reference_drift',
        _reference_checkpoint=weight,_reference_protocol_id='published')
    assert not result['ok'] and 'frozen local evaluator' in result['why']
    assert (real.run_root/'comparison_protocol.json').read_bytes()==old
    assert not list((real.run_root/'attempts').glob('*/receipt.json'))
    assert not getattr(real,'_reference_protocol_id','')


def test_reference_prelaunch_refusal_is_terminal_not_permanently_evaluating(tmp_path):
    real=research(tmp_path)
    weight=install_reference(real)
    real.execution_graph={'score_target':'evaluate'}
    label='reference_graph_boundary';rel=f'research/{real.run_id}/measurements/{label}.json'
    refs.begin(real.output,'published',measurement_ref=rel)
    result=real.measure(settings={'episodes':100},label=label,
        _reference_checkpoint=weight,_reference_protocol_id='published')
    assert not result['ok'] and result['where']=='reference_graph_binding'
    assert read_json(real.output/rel)==result
    assert refs.complete(real.output,'published',result,rel)['status']=='evaluation_failed'
    assert not (real.run_root/'comparison_protocol.json').exists()


def test_reference_exception_restores_local_protocol_scope(tmp_path,monkeypatch):
    real=research(tmp_path);weight=install_reference(real)
    def raised(**kwargs):
        assert real._active_protocol_path().parent.name=='reference_protocols'
        raise RuntimeError('native harness exception')
    monkeypatch.setattr(real,'_measure',raised)
    with pytest.raises(RuntimeError,match='native harness'):
        real.measure(settings={'episodes':100},label='reference_exception',
            _reference_checkpoint=weight,_reference_protocol_id='published')
    assert real._active_protocol_path()==real.run_root/'comparison_protocol.json'


def test_preparation_reference_uses_reviewed_samples_not_local_or_verification_defaults(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    controller=preparation(tmp_path/'controller');real=research(tmp_path/'native')
    install_reference(real)
    controller.output=real.output;controller.repo=real.repo;controller.run_id=real.run_id
    controller.base_settings={'episodes':2,'eval.n_eval':2,'seed':0}
    monkeypatch.setattr(controller,'_research_controller',lambda:(real,{}))
    called=[]
    def measured(**kwargs):
        called.append(kwargs)
        row={'ok':False,'why':'test terminal failure','settings':kwargs['settings']}
        atomic_json(real.run_root/'measurements'/(kwargs['label']+'.json'),row)
        return row
    monkeypatch.setattr(real,'measure',measured)
    result=controller._step_evaluate_baseline_reference(reference_id='published')
    assert result['reference_status']=='evaluation_failed'
    assert called[0]['settings']['episodes']==called[0]['settings']['eval.n_eval']==100
    assert called[0]['_reference_protocol_id']=='published'
    assert controller.base_settings['episodes']==2


@pytest.mark.parametrize('receipt_created',[False,True])
def test_reference_exception_has_fix_evidence_and_never_replays_unknown_native_work(tmp_path,monkeypatch,receipt_created):
    from tests.test_prepare import preparation
    from autosim.research.evidence_store import read_attempt_evidence
    controller=preparation(tmp_path/'controller');real=research(tmp_path/'native')
    install_reference(real)
    controller.output=real.output;controller.repo=real.repo;controller.run_id=real.run_id
    monkeypatch.setattr(controller,'_research_controller',lambda:(real,{}))
    calls=[]
    def failed(**kwargs):
        calls.append(kwargs)
        if receipt_created:
            atomic_json(real.run_root/'attempts'/('a'*32)/'receipt.json',{'status':'running'})
        raise RuntimeError('reference harness setup failed')
    monkeypatch.setattr(real,'measure',failed)
    result=controller._step_evaluate_baseline_reference(reference_id='published')
    assert result['failure_category']=='reference_operation_failed' and result['repair_owner']=='fix'
    assert 'reference harness setup failed' in read_attempt_evidence(real.output,result['evidence_id'])['text']
    assert refs.view(real.output)['status']==('evaluating' if receipt_created else 'evaluation_failed')
    if receipt_created:
        again=controller._step_evaluate_baseline_reference(reference_id='published',retry_reason='inspect unknown outcome')
        assert again['failure_category']=='reference_operation_failed'
        assert len(calls)==1


def test_reference_cannot_accept_unsealed_receipt(tmp_path):
    _, output, _, frozen = reference(tmp_path)
    row, rel = measurement(output, frozen)
    assert refs.complete(output, 'published', row, rel)['status'] == 'evaluation_failed'


def test_public_extension_operation_preserves_input_arguments(tmp_path, monkeypatch):
    from tests.test_prepare import preparation
    real = preparation(tmp_path)
    seen = []
    monkeypatch.setattr(real, '_step_extend_research_rounds',
        lambda **kwargs: seen.append(kwargs) or {'outcome':'extended'})
    args = dict(additional_rounds=3, expected_rounds=2, reason='Budget remains')
    assert real._do_operation('extend_research_rounds', **args)['outcome']=='extended'
    assert seen == [args]
