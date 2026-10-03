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
