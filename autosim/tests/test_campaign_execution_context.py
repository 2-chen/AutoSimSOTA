import json
from types import SimpleNamespace

import pytest

from autosim.research import candidate_execution_review as review
from autosim.research import data_campaign
from autosim.research.common import atomic_json, digest, object_digest, read_json, sanitize_model_payload
from autosim.research.ideas import Idea


def frozen(root, *, identity='study', run_id='derived', schema=2):
    root.mkdir(parents=True, exist_ok=True)
    anchor = root / 'anchor.json'
    atomic_json(anchor, {'development': True})
    plan = {'schema_version': schema, 'id': identity, 'run_id': run_id,
        'status': 'hypothesis_registered', 'hypothesis': 'compare acquisitions',
        'development_label': 'baseline',
        'development_witness': {'ref': 'anchor.json', 'sha256': digest(anchor)},
        'comparison': {'training_recipe': 'context ' * 100 + 'SAME_FROZEN_INITIALIZATION',
            'frozen_factors': {'initialization': 'same frozen directory',
                              'custom_factor': {'must_be_preserved': [1, 2, 3]}},
            'seeds': [11, 12], 'arms': ['random_quality', 'agent_targeted'],
            'quality_floor': 'same production-time supervision and completed episodes'}}
    path = root / 'data_campaigns' / identity / 'plan.json'
    atomic_json(path, plan)
    return path, plan


def packet_fixture(tmp_path, monkeypatch):
    root, repo = tmp_path / 'run', tmp_path / 'repo'
    repo.mkdir()
    (repo / 'train.py').write_text('def train():\n    return 1\n')
    idea = Idea(label='observer', granularity='code', risk='low',
        mechanism='observe samples', change={'file': 'train.py',
            'find': 'return 1', 'replace': 'return 2'})
    stages = {'train': 'def train(i):\n    return ["python", "train.py"]\n'}
    monkeypatch.setattr(review, 'native_observations', lambda *a, **k: [])
    return root, repo, idea, stages


def test_complete_frozen_protocol_is_source_bound_not_a_verdict(tmp_path):
    path, plan = frozen(tmp_path)
    before = path.read_bytes()
    rows = data_campaign.execution_protocols(tmp_path, 'derived')
    assert len(rows) == 1
    assert rows[0]['comparison'] == plan['comparison']
    assert 'SAME_FROZEN_INITIALIZATION' in rows[0]['comparison']['training_recipe']
    assert rows[0]['plan_sha256'] == digest(path)
    assert 'not verified native configuration' in rows[0]['authority']
    assert path.read_bytes() == before


def test_only_current_v2_studies_are_projected(tmp_path):
    frozen(tmp_path, identity='current')
    frozen(tmp_path, identity='legacy', schema=1)
    frozen(tmp_path, identity='other', run_id='elsewhere')
    assert [r['id'] for r in data_campaign.execution_protocols(tmp_path, 'derived')] == ['current']


@pytest.mark.parametrize('mutation', ['witness', 'identity', 'status', 'symlink'])
def test_unsafe_or_changed_protocol_fails_closed(tmp_path, mutation):
    path, plan = frozen(tmp_path)
    if mutation == 'witness':
        atomic_json(tmp_path / 'anchor.json', {'changed': True})
    elif mutation == 'symlink':
        outside = tmp_path / 'other.json'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        plan['id' if mutation == 'identity' else 'status'] = 'different'
        atomic_json(path, plan)
    with pytest.raises(ValueError):
        data_campaign.execution_protocols(tmp_path, 'derived')


@pytest.mark.parametrize('mode', ['many', 'text'])
def test_protocol_bounds_do_not_silently_drop_factors(tmp_path, mode):
    path, plan = frozen(tmp_path)
    if mode == 'many':
        for i in range(8):
            frozen(tmp_path, identity=f'extra_{i}')
    else:
        plan['comparison']['frozen_factors']['important'] = 'x' * 33000
        atomic_json(path, plan)
    with pytest.raises(ValueError, match='bounded'):
        data_campaign.execution_protocols(tmp_path, 'derived')


def test_absent_campaigns_do_not_forbid_ordinary_research(tmp_path):
    assert data_campaign.execution_protocols(tmp_path, 'derived') == []


def test_decision_context_is_current_bounded_copy_not_heartbeat(tmp_path):
    action = {'step': 'run_the_loop', 'research_decision': {
        'question': 'Does the completed batch consume the selected data?',
        'hypothesis': 'the actual optimizer wrapper emits a batch event',
        'expected_outputs': ['sealed witness'], 'postconditions': ['training_batch'],
        'stop_condition': 'missing witness', 'state_revision': 1}}
    context = review.decision_context(action)
    action['research_decision']['state_revision'] = 2
    assert context == review.decision_context(action)
    action['research_decision']['postconditions'].append('changed')
    assert context['postconditions'] == ['training_batch']
    assert review.decision_context({}) is None
    with pytest.raises(ValueError, match='bounded'):
        review.decision_context({'research_decision': {'question': 'x' * 33000}})


def test_packet_carries_current_goal_and_full_plan_without_choosing_inputs(tmp_path, monkeypatch):
    root, repo, idea, stages = packet_fixture(tmp_path, monkeypatch)
    _, plan = frozen(root)
    selected = {'dataset': 'data-version:observed', 'steps': 1000}
    context = review.decision_context({'step': 'run_the_loop', 'research_decision': {
        'question': 'data arm', 'postconditions': ['actual optimizer batch']}})
    packet = review.packet_for(root, repo, stages, idea,
        research_root=root / 'research/derived', selected_training_settings=selected,
        research_decision_context=context)
    assert packet['frozen_data_studies'][0]['comparison'] == plan['comparison']
    assert packet['current_research_decision'] == context
    sources = {r['id']: r for r in packet['sources']}
    assert 'SAME_FROZEN_INITIALIZATION' in sources['frozen_data_study_requirements']['text']
    assert 'actual optimizer batch' in sources['current_research_decision']['text']
    assert packet['selected_training_settings'] == sanitize_model_payload(selected, local_roots=(repo, root))
    assert selected == {'dataset': 'data-version:observed', 'steps': 1000}
    assert 'initialization' not in selected  # the executor must not invent a scientific setting
    assert packet['review_contract_version'] == 13


def test_changed_goal_or_frozen_factor_invalidates_review_identity(tmp_path, monkeypatch):
    root, repo, idea, stages = packet_fixture(tmp_path, monkeypatch)
    path, plan = frozen(root)
    kwargs = {'research_root': root / 'research/derived',
              'research_decision_context': {'postconditions': ['sample_read']}}
    first = review.packet_for(root, repo, stages, idea, **kwargs)
    kwargs['research_decision_context'] = {'postconditions': ['training_batch']}
    second = review.packet_for(root, repo, stages, idea, **kwargs)
    assert object_digest(first) != object_digest(second)
    plan['comparison']['frozen_factors']['custom_factor'] = {'changed': True}
    atomic_json(path, plan)
    third = review.packet_for(root, repo, stages, idea, **kwargs)
    assert object_digest(second) != object_digest(third)


def test_bounded_trial_rechecks_exact_same_main_context(tmp_path, monkeypatch):
    from autosim.research import candidate_validation_trial as trial
    root, repo, idea, stages = packet_fixture(tmp_path, monkeypatch)
    idea.status = 'cleared'
    frozen(root)
    context = {'postconditions': ['batch and frozen recipe']}
    expected = review.packet_for(root, repo, stages, idea,
        research_root=root / 'research/derived', selected_training_settings={'steps': 3},
        research_decision_context=context)
    seen = []
    def cached(output, packet):
        seen.append(packet)
        return {'status': 'reviewed', 'answer': {'verdict': 'uncertain'},
            'record_ref': 'review.json', 'material_sha256': object_digest(packet),
            'record_sha256': 'a' * 64}
    monkeypatch.setattr(review, 'cached', cached)
    research = SimpleNamespace(library=SimpleNamespace(get=lambda label: idea),
        run_root=root / 'research/derived', sources=stages)
    result = trial.authorize(research, output=root, repo=repo, stages=stages,
        idea_label=idea.label, settings={'steps': 3}, decision_id='a' * 16,
        request={'reason': 'check unresolved native binding',
                 'postcondition': 'batch is seen', 'stage_window_seconds': 30},
        admission=None, training_settings={'steps': 3}, research_decision_context=context)
    assert object_digest(seen[0]) == object_digest(expected)
    assert read_json(root / result['authorization_ref'])['review_material_sha256'] == object_digest(expected)


def test_preparation_passes_actual_main_postconditions_to_review(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    root, repo, idea, stages = packet_fixture(tmp_path, monkeypatch)
    owner = Preparation.__new__(Preparation)
    owner.output, owner.repo, owner.stages = root, repo, stages
    owner.client = SimpleNamespace(supports_candidate_execution_review=True)
    owner.current_action = {'step': 'run_the_loop', 'research_decision': {
        'question': 'Does THIS trial fulfill the frozen arm?',
        'postconditions': ['completed training_batch', 'same initialization'],
        'input_evidence': ['a' * 32]}}
    seen = []
    def independent(output, repository, builders, selected, client, **kwargs):
        seen.append(kwargs)
        return {'answer': {'verdict': 'applicable'}}
    monkeypatch.setattr(review, 'review', independent)
    research = SimpleNamespace(library=SimpleNamespace(get=lambda label: idea),
        run_root=root / 'research/derived')
    assert owner._review_candidate_execution(research, idea.label,
        selected_training_settings={'steps': 3}) is None
    assert seen[0]['research_decision_context'] == review.decision_context(owner.current_action)
    assert seen[0]['additional_evidence_refs'] == ['a' * 32]


def test_plan_mutation_during_projection_is_not_a_signed_old_requirement(tmp_path, monkeypatch):
    path, plan = frozen(tmp_path)
    real_check = data_campaign._check_witness
    def changed(root, witness):
        real_check(root, witness)
        plan['comparison']['frozen_factors']['initialization'] = 'changed while reading'
        atomic_json(path, plan)
    monkeypatch.setattr(data_campaign, '_check_witness', changed)
    with pytest.raises(ValueError, match='changed during'):
        data_campaign.execution_protocols(tmp_path, 'derived')
