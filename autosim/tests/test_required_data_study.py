"""Explicit prospective data intent cannot be closed by leaving the registry empty."""
import pytest

from autosim.research import data_campaign as dc, research_brief as rb
from autosim.research.common import atomic_json, read_json
from tests.test_data_campaign import plan, score
from tests.test_data_controls_and_obligations import v2
from tests.test_prepare import preparation


def required(tmp_path):
    source = tmp_path/'data-brief.md'; source.write_text('完成真实随机与定向数据对照，不只刷参数分数。')
    rb.persist(tmp_path/'out', source, require_data_comparison=True)
    real = preparation(tmp_path); real.base_settings['_confirm'] = True
    return real, source


def test_explicit_requirements_are_frozen_and_inherited_without_keyword_inference(tmp_path):
    real, source = required(tmp_path)
    before = (real.output/'research_brief.json').read_bytes()
    assert rb.view(real.output)['requirements'] == {'data_comparison':True}
    rb.persist(real.output, None)
    rb.persist(real.output, source, require_data_comparison=True)
    assert (real.output/'research_brief.json').read_bytes() == before
    with pytest.raises(ValueError, match='frozen'):
        rb.persist(real.output, None, require_data_comparison=False)
    other = tmp_path/'legacy';rb.persist(other, source)
    assert 'requirements' not in rb.view(other)  # Same prose is not an inferred hard rule.
    with pytest.raises(ValueError, match='frozen'):
        rb.persist(other, source, require_data_comparison=True)


def test_requirement_without_a_new_intent_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='research-brief'):
        rb.persist(tmp_path/'empty', None, require_data_comparison=True)


def test_required_empty_registry_refuses_confirmation_before_any_native_work(tmp_path):
    real, _ = required(tmp_path)
    result = real._step_confirm_best()
    assert result['outcome'] == 'not attempted'
    assert result['failure_category'] == 'data_campaign_obligations'
    assert result['repair_owner'] == 'data' and result['native_operation_status'] == 'not_started'
    assert result['evidence_id']
    assert not (real.output/'research/derived/confirmation_attempts').exists()
    assert dc.pending(real.output, real.run_id) == []  # Legacy callers retain old semantics.
    assert dc.pending(real.output, real.run_id, require_campaign=True)


def test_legacy_campaign_does_not_satisfy_new_data_requirement(tmp_path, monkeypatch):
    real, _ = required(tmp_path)
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',lambda *a,**k:{'status':'consistent'})
    score(real.output);dc.register(real.output,real.run_id,plan())
    result = real._step_confirm_best()
    assert result['failure_category'] == 'data_campaign_obligations'
    assert not (real.output/'research/derived/confirmation_attempts').exists()


def test_registered_v2_plan_still_requires_its_actual_arms(tmp_path, monkeypatch):
    real, _ = required(tmp_path)
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',lambda *a,**k:{'status':'consistent'})
    score(real.output);dc.register(real.output,real.run_id,v2())
    pending = dc.pending(real.output,real.run_id,require_campaign=True)
    assert len(pending) == 6 and all(row.get('arm') for row in pending)
    assert real._step_confirm_best()['failure_category'] == 'data_campaign_obligations'


def test_changed_requirements_never_default_to_optional_at_confirmation(tmp_path):
    real, _ = required(tmp_path)
    path = real.output/'research_brief.json';row = read_json(path)
    row['requirements']['data_comparison'] = False;atomic_json(path,row)
    with pytest.raises(ValueError, match='requirements identity'):
        rb.view(real.output)
    result = real._step_confirm_best()
    assert result['failure_category'] == 'data_campaign_obligations'
    assert result['native_operation_status'] == 'not_started'


@pytest.mark.parametrize('fault', ['no_arms', 'no_seeds', 'different_start'])
def test_required_campaign_cannot_be_emptied_or_detached_after_registration(tmp_path, monkeypatch, fault):
    real, _ = required(tmp_path)
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',lambda *a,**k:{'status':'consistent'})
    score(real.output);dc.register(real.output,real.run_id,v2())
    path = real.output/'data_campaigns/failure_slice/plan.json';row = read_json(path)
    if fault == 'no_arms':row['comparison']['arms'] = []
    elif fault == 'no_seeds':row['comparison']['seeds'] = []
    else:score(real.output, value=.9)
    atomic_json(path,row)
    result = real._step_confirm_best()
    assert result['failure_category'] == 'data_campaign_obligations'
    assert result['native_operation_status'] == 'not_started'
