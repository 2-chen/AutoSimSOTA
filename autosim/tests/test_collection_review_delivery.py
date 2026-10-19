"""A collection review's delivery repair cannot mint scientific permission."""
import json

import pytest

from autosim.research.data_acquisition import review_collection_controls
from autosim.research.source_citations import ReviewCitationError


CHECKS = ('allowed_training_controls', 'routing_supported', 'label_supply_supported')
SOURCE = "def collect(i): return ['collect', str(i['collection_controls']['mode'])]"


def fixture(tmp_path, replies):
    repo = tmp_path / 'repo'; repo.mkdir()
    output = tmp_path / 'run'; output.mkdir()
    (repo / 'collect.py').write_text('reset reads legal training mode\n')
    settings = {'controls': {'mode': 'expert'}, 'control_source_refs': [
        {'ref': 'collect.py', 'quote': 'reset reads legal training mode'}]}

    class Client:
        supports_bounded_generation = True

        def __init__(self):
            self.calls = []

        def chat_with_metadata(self, system, user, **options):
            self.calls.append((system, json.loads(user), options))
            reply = replies[len(self.calls) - 1]
            if callable(reply):
                reply = reply(repo)
            return json.dumps(reply), {}

    client = Client()
    kwargs = dict(output=output, repo=repo, client=client, source=SOURCE,
                  declaration={}, settings=settings)
    return client, kwargs


def review(**extra):
    return {**dict.fromkeys(CHECKS, True), 'why': 'independent source assessment',
            'citations': [{'source_id': 'review-source-1', 'quote': 'training mode'}], **extra}


def test_one_bounded_citation_repair_preserves_original_assessment(tmp_path):
    initial = review()
    repair = {'citations': [{'source_id': 'review-source-1'}]}
    client, kwargs = fixture(tmp_path, [initial, repair])
    contract = review_collection_controls(**kwargs)
    assert len(client.calls) == 2
    assert all(contract['review'][key] is True for key in CHECKS)
    assert contract['review']['why'] == initial['why']
    proof = contract['review']['citation_format_repair']
    assert proof['initial_review'] == initial and proof['reply'] == repair
    assert contract['review']['citations'] == contract['source_refs']
    system, packet, options = client.calls[1]
    assert 'FORMAT only' in system
    assert packet['prior_review'] == initial
    assert options['read_only'] and options['include_research_context'] is False
    assert options['max_tokens'] <= 600
    assert options['decision_only'] and options['thinking'] is False
    assert options['auto_skills'] is False and options['resume_session'] is False
    assert contract['effect_status'] == 'needs_native_observation'


def test_real_role_client_forwards_format_only_output_bound_and_no_tools(tmp_path, monkeypatch):
    from tests.test_agent_client import client_for

    client = client_for(tmp_path)
    (client.workspace / 'collect.py').write_text('reset reads legal training mode\n')
    calls = []
    replies = [review(), {'citations': [{'source_id': 'review-source-1'}]}]

    def run(**kwargs):
        calls.append(kwargs)
        return {'status': 'completed', 'final_text': json.dumps(replies[len(calls) - 1])}

    monkeypatch.setattr('autosim.research.agent_runtime.run_coding_agent', run)
    review_collection_controls(output=client.output, repo=client.workspace, client=client,
        source=SOURCE, declaration={}, settings={'controls': {'mode': 'expert'},
            'control_source_refs': [{'ref': 'collect.py', 'quote': 'reset reads legal training mode'}]})
    assert len(calls) == 2
    assert calls[0]['generation_policy'] is None  # Scientific reasoning remains unrestricted here.
    assert calls[1]['generation_policy'] == {'max_output_tokens': 600, 'thinking': False}
    assert calls[1]['decision_only'] is True and calls[1]['auto_skills'] is False
    assert calls[1]['resume'] is False


def test_valid_id_only_review_does_not_buy_a_format_repair(tmp_path):
    client, kwargs = fixture(tmp_path, [review(citations=[{'source_id': 'review-source-1'}])])
    assert review_collection_controls(**kwargs)['review']['routing_supported']
    assert len(client.calls) == 1


def test_control_review_scopes_intervention_without_preapproving_or_hiding_dependencies(tmp_path):
    client, kwargs = fixture(tmp_path, [review(citations=[{'source_id': 'review-source-1'}])])
    kwargs['declaration'] = {'optimization_space': {'collection': [
        {'name': 'mode', 'values': ['expert', 'correction']},
        {'name': 'unmodified_axis', 'default': 7}]}}
    contract = review_collection_controls(**kwargs)
    system, packet, _ = client.calls[0]
    assert packet['requested_control_keys'] == ['mode']
    assert packet['collection_controls'] == {'mode': 'expert'}
    assert packet['declaration'] == kwargs['declaration']
    assert 'necessary validity dependencies' in system
    assert 'name the exact dependency' in system
    assert 'do not deny solely' in system.lower()
    assert 'do not loosen protected evaluation' in system.lower()
    assert contract['review_scope'] == packet['review_scope']
    assert contract['effect_status'] == 'needs_native_observation'


@pytest.mark.parametrize('check', CHECKS)
def test_scientific_denial_dominates_bad_citation_and_is_not_reconsidered(tmp_path, check):
    client, kwargs = fixture(tmp_path, [review(**{check: False})])
    with pytest.raises(ValueError, match='control review did not pass'):
        review_collection_controls(**kwargs)
    assert len(client.calls) == 1


@pytest.mark.parametrize('repair', [
    {'citations': [{'source_id': 'invented'}]},
    {'citations': [{'source_id': 'review-source-1', 'quote': 'training mode'}]},
    {'citations': []},
    {'citations': [{'source_id': 'review-source-1'}], 'routing_supported': True},
    {'citations': [{'source_id': 'review-source-1'}], 'why': 'new assessment'},
    [], None, 'not an object',
])
def test_bad_or_permission_changing_repair_fails_after_exactly_two_calls(tmp_path, repair):
    client, kwargs = fixture(tmp_path, [review(), repair])
    with pytest.raises(ReviewCitationError):
        review_collection_controls(**kwargs)
    assert len(client.calls) == 2


def test_source_change_during_format_repair_is_still_rejected(tmp_path):
    def mutate(repo):
        (repo / 'collect.py').write_text('changed native route\n')
        return {'citations': [{'source_id': 'review-source-1'}]}
    client, kwargs = fixture(tmp_path, [review(), mutate])
    with pytest.raises(ValueError):
        review_collection_controls(**kwargs)
    assert len(client.calls) == 2


def test_exhausted_review_delivery_is_owned_by_supervisor_not_native_collector(tmp_path, monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research.common import read_json

    real = preparation(tmp_path)
    real.stages = {'collect': SOURCE}

    def fail(**kwargs):
        raise ReviewCitationError('second citation reply still invalid')

    monkeypatch.setattr('autosim.research.data_acquisition.review_collection_controls', fail)
    result = real._step_submit_native_job(stage='collect', window_seconds=60,
        reason='toy acquisition', collection_settings={'controls': {'mode': 'expert'},
            'control_source_refs': [{'ref': 'collect.py', 'quote': 'mode'}]})
    assert result['failure_category'] == 'review_citation_invalid'
    assert result['repair_owner'] == 'supervisor'
    assert result['native_operation_status'] == 'not_started'
    proof = read_json(real.output / 'agent/operation_failures' / (result['evidence_id'] + '.json'))
    assert proof['context']['collection_settings']['controls'] == {'mode': 'expert'}
    assert not (real.output / 'native_jobs').exists()


def test_main_receives_actual_collector_key_reads_and_current_review_recovery_contract(tmp_path):
    from tests.test_prepare import preparation

    real = preparation(tmp_path)
    real.stages = {'collect': SOURCE}
    facts = real._decision_facts()['executor_input_contracts']['data_acquisition']
    assert facts['builder_input_reads'] == {'episodes': False, 'seed': False,
                                           'collection_controls': True}
    review = facts['independent_control_review']
    assert review['citation_format_repair_limit'] == 1
    assert review['scientific_denial_precedes_citation_format'] is True
    assert review['exhausted_delivery_owner'] == 'supervisor'
    assert 'not a native' in review['boundary_rule']
    assert 'not automatically resolved' in facts['control_path_semantics']
    assert 'output-relative and checkout-relative' in facts['control_path_semantics']


def test_collection_source_input_failure_keeps_precise_item_and_capture_repair(tmp_path, monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research.baseline_reference import SourceCitationError
    from autosim.research.common import read_json

    real = preparation(tmp_path)
    real.stages = {'collect': SOURCE}

    def fail(**kwargs):
        raise SourceCitationError('quote_not_exact', index=6, ref='collect.py:2',
                                  detail='capture exact source rather than display placeholders')

    monkeypatch.setattr('autosim.research.data_acquisition.review_collection_controls', fail)
    result = real._step_submit_native_job(stage='collect', window_seconds=60,
        reason='toy acquisition', collection_settings={'controls': {'mode': 'expert'},
            'control_source_refs': [{'ref': 'collect.py:2', 'quote': '[LOCAL_RESOURCE]'}]})
    assert result['failure_category'] == 'source_citation_invalid'
    assert result['repair_owner'] == 'main_agent'
    assert result['native_operation_status'] == 'not_started'
    proof = read_json(real.output / 'agent/operation_failures' / (result['evidence_id'] + '.json'))
    assert proof['context']['source_validation']['citation_index'] == 7
    assert proof['context']['source_validation']['code'] == 'quote_not_exact'
    assert proof['context']['scientific_review_status'] == 'not_inferred_from_source_transport_failure'
    assert 'checkout_source_request' in result['retry_when']
    assert not (real.output / 'native_jobs').exists()
