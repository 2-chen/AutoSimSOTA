"""Actual Agent citation shapes round-trip without weakening source or review guards."""
import json
from pathlib import Path

import pytest

from autosim.research.baseline_reference import SourceCitationError, source_evidence
from autosim.research.checkout_source_citations import capture, resolve
from autosim.research.common import atomic_json, read_json, sanitize_model_payload
from autosim.research.agent_runtime import _McpExecutor, _native_tool_schema, handle_mcp_message
from tests.test_prepare import preparation


def workspace(tmp_path):
    output = tmp_path / 'run'; repo = output / 'checkout'; repo.mkdir(parents=True)
    (repo / 'configs').mkdir()
    path = repo / 'configs/task.json'
    path.write_text('{\n  "reset": "independent_bank[seed]",\n  "asset_path": "/home/private/assets"\n}\n')
    atomic_json(output / 'workspace_snapshot.json', {'schema_version': 2,
        'source': str(repo), 'destination': str(repo), 'isolated': True,
        'source_entries': [['file', 'configs/task.json']]})
    return output, repo, path


@pytest.mark.parametrize('ref', ['evaluate.py', 'evaluate.py:2', 'evaluate.py:2-3'])
def test_plain_and_line_citations_revalidate_consistently(tmp_path, ref):
    (tmp_path / 'evaluate.py').write_text('def reset(seed):\n    state = initial_bank[seed]\n    return state\n')
    rows = source_evidence(tmp_path, [{'ref': ref, 'quote': 'state = initial_bank[seed]'}])
    assert rows[0]['ref'] == ref
    assert source_evidence(tmp_path, rows) == rows


@pytest.mark.parametrize('ref,code', [
    ('evaluate.py:0', 'invalid_reference'), ('evaluate.py:3-2', 'invalid_line_range'),
    ('evaluate.py:2-100', 'invalid_line_range'), ('../evaluate.py:2', 'unsafe_source'),
    ('/tmp/evaluate.py:2', 'unsafe_source'), ('[LOCAL_RESOURCE]:2', 'display_placeholder'),
    ('missing.py:2', 'source_missing'), ('evaluate.py:1', 'quote_not_exact')])
def test_error_identifies_bad_item_and_preserves_line_confinement(tmp_path, ref, code):
    (tmp_path / 'evaluate.py').write_text('def reset(seed):\n    return initial_bank[seed]\n')
    with pytest.raises(SourceCitationError) as caught:
        source_evidence(tmp_path, [{'ref': 'evaluate.py', 'quote': 'def reset'},
                                  {'ref': ref, 'quote': 'return initial_bank[seed]'}])
    assert caught.value.details['citation_index'] == 2
    assert caught.value.details['code'] == code


def test_symlink_parent_is_not_source_authority(tmp_path):
    repo = tmp_path / 'repo'; repo.mkdir(); other = tmp_path / 'outside'; other.mkdir()
    (other / 'eval.py').write_text('return initial_bank[seed]')
    (repo / 'link').symlink_to(other, target_is_directory=True)
    with pytest.raises(SourceCitationError, match='unsafe_source'):
        source_evidence(repo, [{'ref': 'link/eval.py', 'quote': 'initial_bank'}])


def test_config_handle_and_redacted_quote_round_trip_to_checked_original(tmp_path):
    output, repo, _ = workspace(tmp_path)
    selected = capture(output, repo, {'ref': 'configs/task.json:2-3'})
    shown = sanitize_model_payload({'ref': selected['ref'], 'quote': selected['quote']}, local_roots=(repo, output))
    assert shown['ref'].startswith('checkout-source:')
    assert '/home/private' not in json.dumps(selected)
    result = source_evidence(repo, [shown], output=output)
    assert '/home/private/assets' in result[0]['quote']
    assert source_evidence(repo, result, output=output) == result
    assert sanitize_model_payload({'ref': 'configs/task.json'})['ref'] == '[LOCAL_RESOURCE]'


@pytest.mark.parametrize('change', ['source', 'snapshot', 'metadata', 'wrong_quote', 'other_repo'])
def test_config_handle_never_authorizes_changed_or_unrelated_bytes(tmp_path, change):
    output, repo, path = workspace(tmp_path)
    selected = capture(output, repo, {'ref': 'configs/task.json:2'})
    identity = selected['ref'].split(':')[1]; other_repo = repo
    if change == 'source': path.write_text(path.read_text() + '\nchanged\n')
    elif change == 'snapshot': (output / 'checkout_source_citations' / (identity + '.source')).write_text('wrong')
    elif change == 'metadata':
        p = output / 'checkout_source_citations' / (identity + '.json')
        row = read_json(p); row['source_ref'] = '../outside.json'; atomic_json(p, row)
    elif change == 'wrong_quote': selected['quote'] = 'independent_bank'
    elif change == 'other_repo': other_repo = tmp_path / 'unrelated'; other_repo.mkdir()
    with pytest.raises(ValueError): resolve(output, other_repo, selected['ref'], selected['quote'])


def test_source_only_mcp_capture_requires_no_interpreter_and_no_execution(tmp_path):
    output, repo, _ = workspace(tmp_path)
    executor = _McpExecutor(workspace=repo, output=output, allow_commands=False, allow_native=True)
    response = handle_mcp_message({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
        'params': {'name': 'inspect_native_environment', 'arguments': {
            'purpose': 'read public reset config', 'checkout_source_request': {'ref': 'configs/task.json:2'}}}}, executor)
    assert not response['result']['isError']
    selected = json.loads(response['result']['content'][0]['text'])
    assert source_evidence(repo, [{'ref': selected['ref'], 'quote': selected['quote']}], output=output)
    assert not (output / 'native_context.json').exists()
    assert not (output / 'task_gpu_budget.json').exists()
    with pytest.raises(ValueError, match='cannot execute'):
        executor.native_probe({'purpose': 'unsafe', 'checkout_source_request': {'ref': 'configs/task.json'}, 'code': 'pass'})
    assert not any(k in _native_tool_schema()['inputSchema'] for k in ['anyOf', 'oneOf', 'allOf'])


def test_capture_denies_bound_resources_untracked_outputs_and_secrets(tmp_path, monkeypatch):
    output, repo, _ = workspace(tmp_path)
    monkeypatch.setattr('autosim.research.workspace_resources.bindings_for',
        lambda *a: [{'target': 'configs', 'source': '/not-used'}])
    with pytest.raises(ValueError, match='bound dataset'):
        capture(output, repo, {'ref': 'configs/task.json:2'})
    monkeypatch.setattr('autosim.research.workspace_resources.bindings_for', lambda *a: [])
    (repo / 'generated.json').write_text('{}')
    with pytest.raises(ValueError, match='source manifest'):
        capture(output, repo, {'ref': 'generated.json'})
    (repo / 'secret.json').write_text('{}')
    with pytest.raises(ValueError, match='unsafe_source'):
        capture(output, repo, {'ref': 'secret.json'})


def test_invalid_citation_is_main_input_handoff_without_buying_scientific_review(tmp_path):
    real = preparation(tmp_path)
    result = real._step_configure_confirmation_protocol(confirmation={'seed': 101},
        source_refs=[{'ref': '[LOCAL_RESOURCE]:101-109', 'quote': 'reset'}], reason='native split')
    assert result['failure_category'] == 'source_citation_invalid'
    assert result['repair_owner'] == 'scheduler' and result['retryable']
    assert real.client.seen == []
    proof = read_json(real.output / 'agent/operation_failures' / (result['evidence_id'] + '.json'))
    assert proof['context']['source_validation']['code'] == 'display_placeholder'
    assert proof['context']['scientific_review_started'] is False
    assert proof['native_operation_status'] == 'not_started'
    assert not (real.output / 'confirmation_reservation.json').exists()


def test_redacted_source_handle_reaches_independent_review_then_preserves_proposal(tmp_path):
    real = preparation(tmp_path)
    (real.repo / 'config.json').write_text('{"reset": "initial_bank[seed]"}\n')
    selected = capture(real.output, real.repo, {'ref': 'config.json'})
    seen = []
    def reviewer(system, user, **options):
        packet = json.loads(user); seen.append(packet); row = packet['review_sources'][0]
        return json.dumps(dict(native_evaluation_input_supported=True, disjoint_conditions_supported=True,
            preserves_policy_and_metric=True, source_connection_supported=True,
            citations=[{'source_id': row['source_id'], 'quote': row['quote']}], why='toy independent bank')), {}
    real.client.chat_with_metadata = reviewer
    result = real._step_configure_confirmation_protocol(confirmation={'seed': 101},
        source_refs=[{'ref': selected['ref'], 'quote': selected['quote']}], reason='native split')
    assert result['outcome'] == 'registered' and len(seen) == 1
    assert read_json(real.output / 'confirmation_reservation.json')['confirmation'] == {'seed': 101}
    (real.repo / 'config.json').write_text('{"reset": "constant_state"}\n')
    from autosim.research.confirmation_reservation import apply
    with pytest.raises(ValueError): apply(real.output, real.repo, real.run_id, {})


def test_failed_scientific_review_still_goes_to_objective(tmp_path):
    real = preparation(tmp_path)
    (real.repo / 'eval.py').write_text('return constant_state\n')
    real.client.choices = [dict(native_evaluation_input_supported=True, disjoint_conditions_supported=False,
        preserves_policy_and_metric=True, source_connection_supported=True,
        citations=[{'ref': 'eval.py:1', 'quote': 'return constant_state'}])]
    result = real._step_configure_confirmation_protocol(confirmation={'seed': 101},
        source_refs=[{'ref': 'eval.py:1', 'quote': 'return constant_state'}], reason='native split')
    assert result['failure_category'] == 'confirmation_protocol_required' and result['repair_owner'] == 'objective'
    assert not (real.output / 'confirmation_reservation.json').exists()


def test_fix_can_inspect_but_not_patch_source_and_privacy_primitives():
    from autosim.research.harness_repair import ALLOWED, READABLE
    required = {'research/baseline_reference.py', 'research/confirmation_reservation.py',
                'research/common.py', 'research/checkout_source_citations.py',
                'research/source_continuity.py'}
    assert required <= READABLE and not required & ALLOWED


def test_reviewer_can_select_only_supplied_id_without_retyping_private_quote(tmp_path):
    from autosim.research.source_citations import project, resolve, ReviewCitationError
    original = [{'ref': 'configs/task.json', 'quote': 'asset_path="/home/private/assets"', 'sha256': 'a' * 64}]
    _, shown = project({}, original)
    assert resolve(original, shown, [{'source_id': 'review-source-1'}]) == original
    for row in [{'source_id': 'invented'}, {'source_id': 'review-source-1', 'quote': 'asset_path'},
                {'source_id': 'review-source-1', 'quote': original[0]['quote']}]:
        with pytest.raises(ReviewCitationError): resolve(original, shown, [row])


def test_review_copy_error_is_not_an_environment_or_scientific_failure(tmp_path):
    real = preparation(tmp_path); (real.repo / 'eval.py').write_text('return initial_bank[seed]\n')
    real.client.choices = [dict(native_evaluation_input_supported=True, disjoint_conditions_supported=True,
        preserves_policy_and_metric=True, source_connection_supported=True,
        citations=[{'source_id': 'review-source-1', 'quote': 'initial_bank'}])]
    result = real._step_configure_confirmation_protocol(confirmation={'seed': 101},
        source_refs=[{'ref': 'eval.py', 'quote': 'return initial_bank[seed]'}], reason='native split')
    assert result['failure_category'] == 'review_citation_invalid'
    assert result['repair_owner'] == 'supervisor' and result['retryable']
    assert not (real.output / 'confirmation_reservation.json').exists()


def test_failure_input_is_preserved_privately_not_reconstructed_from_display(tmp_path):
    from autosim.research.operation_failure import seal
    context = {'proposal': {'source_refs': [{'ref': 'configs/task.json:101-109', 'quote': 'reset()'}]}}
    failure = seal(tmp_path, operation='configure_confirmation_protocol', exc=ValueError('bad source'), context=context)
    receipt = read_json(tmp_path / failure['receipt_ref'])
    assert receipt['context']['proposal']['source_refs'][0]['ref'] == '[LOCAL_RESOURCE]:101-109'
    private = read_json(tmp_path / receipt['_local_input_ref'])
    assert private['context'] == context
    assert not any(k.startswith('_local_') for k in sanitize_model_payload(receipt))
    assert (tmp_path / receipt['_local_input_ref']).stat().st_mode & 0o077 == 0


@pytest.mark.parametrize('context', [{'path': Path('/not-executed')}, {'data': 'x' * (257 * 1024)}])
def test_private_input_unavailable_does_not_break_failure_sealing(tmp_path, context):
    from autosim.research.operation_failure import seal
    failure = seal(tmp_path, operation='diagnose', exc=ValueError('test'), context=context)
    receipt = read_json(tmp_path / failure['receipt_ref'])
    assert '_local_input_ref' not in receipt
    assert 'no_replay_authorization' in receipt['_local_input_status']


@pytest.mark.parametrize('status', ['running', 'paused'])
def test_report_does_not_turn_historical_runtime_fault_into_current_stop(tmp_path, status):
    from autosim.research import recorder, run_record
    atomic_json(tmp_path / 'agent/runtime_failure.json', {'role': 'scheduler',
        'message': 'old callback failure', 'at': '2026-10-11T03:28:00Z',
        'evidence_ref': 'evidence/' + 'a' * 32 + '.json'})
    atomic_json(tmp_path / 'runtime_blocker.json', {'recovery': {'status': 'blocked', 'reason': 'old fault'}})
    snapshot = recorder.make_snapshot(tmp_path, run_record.build_report_view(tmp_path, 'derived', status=status),
                                      {'actions': [], 'plan': {}})
    document = recorder.render(tmp_path, snapshot, {})
    assert 'old callback failure' in document and '完整错误证据' in document
    if status == 'running':
        assert '历史框架 / 运行故障记录' in document and '历史恢复判断' in document
        assert '不能据此认定当前已停止' in document
    else:
        assert '### 框架 / 运行故障与恢复' in document and '恢复状态：' in document
