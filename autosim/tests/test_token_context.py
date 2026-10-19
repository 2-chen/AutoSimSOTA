import json

import pytest

from autosim.research.token_context import compact_json, project_context, project_request, project_tool_result, request_json
from autosim.research.evidence_store import read_attempt_evidence


def test_context_paging_is_readable_and_preserves_protocol_errors(tmp_path):
    source = "source-" * 2000
    history = [{"round": i, "text": "measured-result-" * 6} for i in range(100)]
    context = {"history": history, "source": source,
               "comparison_protocol": {"rules": list(range(100))},
               "failure": source, "budget": {"remaining": 2}}
    result, metrics = project_context(context, "{}", tmp_path)
    assert result["comparison_protocol"] == context["comparison_protocol"]
    assert result["failure"] == source
    assert result["budget"] == context["budget"]
    assert result["history"]["preview"] == history[-6:]
    ref = result["history"]["evidence_id"]
    chunks, offset = [], 0
    while True:
        evidence = read_attempt_evidence(tmp_path, ref, offset=offset)
        chunks.append(evidence["text"])
        offset = evidence["next_offset"]
        if offset == evidence["log_bytes"]:
            break
    assert json.loads("".join(chunks)) == history
    assert metrics["paged_leaves"] == 2
    assert metrics["projected_context_bytes"] < metrics["original_context_bytes"]
    assert project_context(context, "{}", tmp_path)[0] == result


def test_deduplication_is_exact_and_toolless_context_is_not_paged(tmp_path):
    value = list(range(100))
    result, metrics = project_context({"history": value, "budget": 10},
        json.dumps({"context": {"budget": 10}}), tmp_path, pageable=False)
    assert result == {"history": value}
    assert metrics["deduplicated_fields"] == 1
    assert metrics["paged_leaves"] == 0
    # Conflicting request examples must not suppress the authoritative snapshot.
    result, _ = project_context({"budget": 10},
        '{"budget":10,"example":{"budget":0}}', tmp_path)
    assert result["budget"] == 10
    assert compact_json({"b": 2, "a": 1}) == compact_json({"a": 1, "b": 2})


def test_paging_rejects_symlink_parent_before_write(tmp_path):
    outside = tmp_path / "other"
    outside.mkdir()
    (tmp_path / "agent").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        project_context({"history": ["entry" * 50] * 100}, "{}", tmp_path)
    assert not list(outside.iterdir())


def test_large_unprotected_map_pages_with_stable_readable_identity(tmp_path):
    value = {f'file_{i:04}.py': 'abcdef0123456789' * 8 for i in range(400)}
    context = {'file_inventory': value, 'budget': {'remaining_usd': 2},
               'evaluation_protocol': {'instructions': 'frozen-' * 2000}}
    projected, metrics = project_context(context, '{}', tmp_path)
    page = projected['file_inventory']
    assert page['context_paged'] and len(page['preview']) == 8
    assert projected['budget'] == context['budget']
    assert projected['evaluation_protocol'] == context['evaluation_protocol']
    assert metrics['projected_context_bytes'] < metrics['original_context_bytes'] / 2
    assert project_context(context, '{}', tmp_path)[0] == projected
    blob = next((tmp_path / 'agent/context_blobs').glob('*.json'))
    assert json.loads(blob.read_text()) == value
    blob.write_text('{}')
    with pytest.raises(ValueError, match='collision'):
        project_context(context, '{}', tmp_path)


def test_request_pages_evidence_not_constraints_and_toolless_keeps_full_packet(tmp_path):
    request = {'source_excerpt': 'body-' * 8000,
               'instruction': 'important-' * 1000, 'failure': 'traceback-' * 1000,
               'protocol': {'test_access': False}, 'budget': {'left': 1}}
    raw = json.dumps(request)
    projected, metrics = project_request(raw, tmp_path, pageable=True)
    parsed = json.loads(projected)
    assert parsed['source_excerpt']['context_paged']
    assert parsed['failure'] == request['failure']
    assert parsed['instruction'] == request['instruction']
    assert parsed['budget'] == request['budget'] and parsed['protocol'] == request['protocol']
    assert metrics['request_paged_leaves'] == 1
    assert metrics['projected_request_bytes'] < metrics['original_request_bytes']
    complete, metrics = project_request(raw, tmp_path, pageable=False)
    assert json.loads(complete) == request and metrics['request_paged_leaves'] == 0
    assert project_request('free-form instructions ' * 1000, tmp_path, pageable=True)[0] == 'free-form instructions ' * 1000


def test_map_containing_nested_constraints_is_not_paged_wholesale(tmp_path):
    items = {f'entry{i}': {'value': 'value-' * 60} for i in range(80)}
    items['critical'] = {'contract': {'never_change': 'scoring'}}
    projected, _ = project_context({'catalog': items}, '{}', tmp_path)
    assert projected['catalog']['critical'] == items['critical']
    assert 'context_paged' not in projected['catalog']


def test_tool_success_offloads_stdout_but_not_errors_or_failed_operations(tmp_path):
    value = {'status': 'completed', 'returncode': 0, 'stdout': 'source-' * 10000,
             'stderr': 'important diagnostic-' * 1000, 'attempt_id': 'original-attempt',
             'protocol': {'frozen': True}}
    result = project_tool_result(value, tmp_path)
    assert result['stdout']['context_paged']
    assert result['stderr'] == value['stderr'] and result['attempt_id'] == value['attempt_id']
    assert result['protocol'] == value['protocol']
    blob = next((tmp_path / 'agent/context_blobs').glob('*.json'))
    assert json.loads(blob.read_text()) == value['stdout']
    assert project_tool_result({**value, 'status': 'failed'}, tmp_path)['stdout'] == value['stdout']


def test_static_method_body_precedes_event_changes_without_copying_or_losing_fields(tmp_path):
    first = {'methods': [{'id': 'chosen', 'method': 'immutable body'}],
             'instruction': 'write from current evidence', 'event_revision': 'first',
             'evidence': {'budget': 2}}
    second = {**first, 'event_revision': 'second', 'evidence': {'budget': 1}}
    rendered = [project_request(json.dumps(row), tmp_path, pageable=False)[0]
                for row in (first, second)]
    assert json.loads(rendered[0]) == first and json.loads(rendered[1]) == second
    assert rendered[0].split('"event_revision"')[0] == rendered[1].split('"event_revision"')[0]
    assert rendered[0].count('immutable body') == 1
    assert request_json({'b': 2, 'a': 1}) == request_json({'a': 1, 'b': 2})


@pytest.mark.parametrize('count', [1,4,10])
def test_few_large_advisory_records_have_small_readable_preview_without_recursion(tmp_path,count):
    rows = [{'id':f'idea-{i}','description':'detail-'*5000,
             'changes':[{'before':'x'*8000,'after':'y'*8000}]} for i in range(count)]
    result, metrics = project_context({'candidates':rows},'{}',tmp_path)
    page = result['candidates']
    assert page['context_paged'] is True
    assert len(compact_json(page).encode()) < 6000
    assert metrics['paged_leaves'] == 1
    assert len(page['preview']) <= 3
    assert 'read full evidence' in compact_json(page['preview'])
    ref=page['evidence_id']
    blob=tmp_path/f'agent/context_blobs/{ref}.json'
    assert json.loads(blob.read_text()) == rows
    read_attempt_evidence(tmp_path,ref,limit=1)


def test_new_small_list_paging_never_hides_nested_protocol_or_budget(tmp_path):
    rows=[{'text':'x'*10000,'contract':{'frozen':True},'budget':{'left':2}}]
    result,_=project_context({'candidates':rows},'{}',tmp_path)
    assert result['candidates'][0]['contract'] == rows[0]['contract']
    assert result['candidates'][0]['budget'] == rows[0]['budget']
    assert isinstance(result['candidates'],list)
    complete,_=project_context({'candidates':[{'text':'x'*10000}]},'{}',tmp_path,pageable=False)
    assert complete == {'candidates':[{'text':'x'*10000}]}
