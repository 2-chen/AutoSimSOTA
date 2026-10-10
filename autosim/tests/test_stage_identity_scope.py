"""Data production capabilities must not silently change the scoring contract."""
from types import SimpleNamespace

import pytest

from autosim.research.common import atomic_json, object_digest, read_json
from autosim.research.research_state import ResearchStateError
from autosim.research.round_extension import extend, resume_rounds
from tests.test_derived_research import research


def test_new_data_capability_can_be_added_after_baseline_without_retraining(tmp_path):
    real = research(tmp_path)
    real.run(rounds=2, yield_after_action=True, main_controller_owns_selection=True)
    baseline = real.run_root/'measurements/baseline.json'
    before = baseline.read_bytes()
    session = read_json(real.run_root/'controller_session.json')
    attempts = sorted((real.run_root/'attempts').glob('*/receipt.json'))
    resumed = research(tmp_path, stages=('train', 'evaluate', 'collect'))
    assert resumed._research_session_identity(rounds=2, settings=session['base_settings']) == session['input_identity']
    # Identity accepts the capability; it must NOT bypass Main's audited selection.
    with pytest.raises(ResearchStateError, match='main controller must select'):
        resumed.run(rounds=2, yield_after_action=True, main_controller_owns_selection=True)
    assert baseline.read_bytes() == before
    assert read_json(real.run_root/'controller_session.json')['input_identity'] == session['input_identity']
    assert sorted((real.run_root/'attempts').glob('*/receipt.json')) == attempts
    assert resumed.available('collect')


def pause(tmp_path, stages=('train', 'evaluate')):
    real = research(tmp_path, stages=stages)
    real.run(rounds=2, yield_after_action=True, main_controller_owns_selection=True)
    return real, read_json(real.run_root/'controller_session.json')


def legacy(real, session):
    """Construct the pre-scope schema, retaining its exact original input digest."""
    inputs = real._research_session_inputs(rounds=session['rounds'], settings=session['base_settings'])
    inputs['stages'] = dict(real.sources)
    session = {**session, 'input_identity': object_digest(inputs),
               'input_component_sha256': {k: object_digest(v) for k, v in inputs.items()}}
    session.pop('input_stage_scope', None)
    atomic_json(real.run_root/'controller_session.json', session)
    return session


@pytest.mark.parametrize('stage', ['collect', 'prepare_data'])
def test_independent_producer_changes_do_not_change_new_scoring_identity(tmp_path, stage):
    real, session = pause(tmp_path, stages=('train', 'evaluate', stage))
    assert session['input_stage_scope']['frozen_stages'] == ['evaluate', 'train']
    real.sources[stage] += '\n# separately derived producer revision\n'
    assert real._research_session_identity(rounds=2, settings=session['base_settings']) == session['input_identity']
    del real.sources[stage]
    assert real._research_session_identity(rounds=2, settings=session['base_settings']) == session['input_identity']


@pytest.mark.parametrize('stage', ['train', 'evaluate'])
def test_scoring_command_change_still_rejected_before_native_work(tmp_path, stage):
    real, session = pause(tmp_path)
    attempts = sorted((real.run_root/'attempts').glob('*/receipt.json'))
    real.sources[stage] += '\n# changed command source\n'
    with pytest.raises(ResearchStateError, match='inputs changed.*stages'):
        real.run(rounds=2, yield_after_action=True, main_controller_owns_selection=True)
    assert read_json(real.run_root/'controller_session.json') == session
    assert sorted((real.run_root/'attempts').glob('*/receipt.json')) == attempts


def test_unknown_new_stage_cannot_be_hidden_outside_scoring_scope(tmp_path):
    real, _ = pause(tmp_path)
    real.sources['postprocess'] = 'def stage_argv_postprocess(i): return []'
    with pytest.raises(ResearchStateError, match='scoring inputs cannot be omitted'):
        real._research_session_stage_sources()


@pytest.mark.parametrize('scope', [False, {}, {'schema_version': 2, 'frozen_stages': ['evaluate']},
    {'schema_version': 2, 'frozen_stages': ['evaluate', 'train', 'train']},
    {'schema_version': 2, 'frozen_stages': ['evaluate', 'train', 'absent']},
    {'schema_version': 2.0, 'frozen_stages': ['evaluate', 'train']}])
def test_invalid_scope_is_not_a_protocol_escape(tmp_path, scope):
    real, session = pause(tmp_path)
    atomic_json(real.run_root/'controller_session.json', {**session, 'input_stage_scope': scope})
    with pytest.raises(ResearchStateError, match='stage scope'):
        real._research_session_stage_sources()


@pytest.mark.parametrize('added', [('collect',), ('prepare_data',), ('collect', 'prepare_data')])
def test_legacy_projection_requires_reproducing_original_component_digest(tmp_path, added):
    real, session = pause(tmp_path)
    session = legacy(real, session)
    resumed = research(tmp_path, stages=('train', 'evaluate', *added))
    assert resumed._research_session_identity(rounds=2, settings=session['base_settings']) == session['input_identity']
    assert read_json(real.run_root/'controller_session.json') == session
    resumed.sources['evaluate'] += '\n# cannot conceal changed scorer\n'
    assert resumed._research_session_identity(rounds=2, settings=session['base_settings']) != session['input_identity']


def test_legacy_existing_collector_remains_bound_when_new_preparer_added(tmp_path):
    real, session = pause(tmp_path, stages=('train', 'evaluate', 'collect'))
    session = legacy(real, session)
    resumed = research(tmp_path, stages=('train', 'evaluate', 'collect', 'prepare_data'))
    assert resumed._research_session_identity(rounds=2, settings=session['base_settings']) == session['input_identity']
    resumed.sources['collect'] += '\n# originally bound source changed\n'
    assert resumed._research_session_identity(rounds=2, settings=session['base_settings']) != session['input_identity']
    del resumed.sources['collect']
    assert resumed._research_session_identity(rounds=2, settings=session['base_settings']) != session['input_identity']


def test_legacy_without_component_proof_cannot_add_capability(tmp_path):
    real, session = pause(tmp_path)
    session = legacy(real, session)
    session.pop('input_component_sha256')
    atomic_json(real.run_root/'controller_session.json', session)
    resumed = research(tmp_path, stages=('train', 'evaluate', 'collect'))
    with pytest.raises(ResearchStateError, match='no component digests'):
        resumed.run(rounds=2, yield_after_action=True, main_controller_owns_selection=True)


def test_execution_graph_keeps_producers_bound(tmp_path):
    real = research(tmp_path, stages=('train', 'evaluate', 'collect', 'prepare_data'))
    real.execution_graph = {'nodes': ['collect', 'prepare_data', 'train', 'evaluate']}
    assert real._research_session_stage_sources() == real.sources


def test_source_removal_and_symlink_state_rejected(tmp_path):
    real, session = pause(tmp_path)
    del real.sources['train']
    with pytest.raises(ResearchStateError, match='stage scope'):
        real._research_session_stage_sources()
    path = real.run_root/'controller_session.json'
    backup = real.run_root/'controller_backup.json'
    path.rename(backup)
    path.symlink_to(backup)
    with pytest.raises(ResearchStateError, match='unsafe research'):
        real._research_session_stage_sources()


def test_new_capability_preserves_applied_round_extension_chain(tmp_path):
    real, session = pause(tmp_path)
    legacy(real, session)
    real.budget = SimpleNamespace(remaining=lambda: 1000)
    extend(real, additional_rounds=2, expected_rounds=2, reason='More measured candidates')
    resumed = research(tmp_path, stages=('train', 'evaluate', 'collect'))
    session = read_json(real.run_root/'controller_session.json')
    assert resume_rounds(real.run_root, session, 2) == 4
    assert resumed._research_session_identity(rounds=4, settings=session['base_settings']) == session['input_identity']
    resumed.budget = SimpleNamespace(remaining=lambda: 1000)
    extend(resumed, additional_rounds=1, expected_rounds=4, reason='One further candidate')
    assert resume_rounds(real.run_root, read_json(real.run_root/'controller_session.json'), 2) == 5
