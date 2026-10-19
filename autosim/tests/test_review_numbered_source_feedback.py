"""Exact source hints repair quotation transport, not a scientific judgment."""
import copy

import pytest

from autosim.research import candidate_execution_review as review


@pytest.mark.parametrize('verdict', ['applicable', 'inapplicable', 'uncertain'])
def test_numbered_pages_offer_literal_component_lines_without_accepting_bad_quote(verdict):
    packet = {'sources': [{'id': 'framework_contract_test', 'text':
        "277:         result[key] = str(native)\n278:         bindings.append(row)"}]}
    answer = {'verdict': verdict, 'reason': 'Current binding semantics need inspection.',
        'source_evidence': [{'source_id': 'framework_contract_test', 'quote':
        '        result[key] = str(native)\n        bindings.append(row)'}]}
    original = copy.deepcopy(answer)
    with pytest.raises(ValueError, match='exact substring'):
        review.validate(answer, packet)
    feedback = review.citation_feedback(answer, packet)
    assert {'source_id': 'framework_contract_test', 'quote': 'result[key] = str(native)'} in feedback['exact_quote_suggestions']
    assert answer == original
    assert 'verdict' not in feedback and 'answer' not in feedback
    for row in feedback['exact_quote_suggestions']:
        assert row['quote'] in packet['sources'][0]['text']


def test_second_response_retains_all_required_categories_even_when_first_quoted_them():
    packet = {'sources': [
        {'id':'candidate_change', 'text':'replace the producer call'},
        {'id':'workflow_train', 'text':'def train(i):\n    return call(i)'},
        {'id':'candidate_source_0', 'text':'def consume(i):\n    return i'},
        {'id':'selected_training_inputs', 'text':'    "dataset": "data-version:v1"'}],
        'native_references':[{'source_id':'candidate_source_0'}],
        'execution_plan':{'will_train':True}}
    answer = {'verdict':'applicable', 'reason':'The selected producer can observe the data.',
        'source_evidence':[{'source_id':row['id'], 'quote':row['text']} for row in packet['sources'][:3]]}
    answer['source_evidence'].append({'source_id':'selected_training_inputs',
        'quote':'      "dataset": "data-version:v1"'})
    with pytest.raises(ValueError, match='exact substring'):
        review.validate(answer, packet)
    suggestions = review.citation_feedback(answer, packet)['exact_quote_suggestions']
    assert {'candidate_change','workflow_train','candidate_source_0','selected_training_inputs'} <= {
        row['source_id'] for row in suggestions}
    # Suggestions are inert; the original answer remains invalid.
    with pytest.raises(ValueError, match='exact substring'):
        review.validate(answer, packet)


def test_matching_one_line_does_not_create_a_match_for_invented_multiline_evidence():
    packet = {'sources':[{'id':'actual', 'text':'11: known_line()\n12: other_line()'}]}
    answer = {'verdict':'uncertain', 'source_evidence':[
        {'source_id':'actual', 'quote':'known_line()\ninvented_fact()'}]}
    suggestions = review.citation_feedback(answer, packet)['exact_quote_suggestions']
    assert suggestions == [{'source_id':'actual','quote':'known_line()'}]
    assert all('invented_fact' not in row['quote'] for row in suggestions)
