"""Regressions for CPU/GPU grants and stale SDK tool assumptions, not benchmark rules."""
import json
import os
import sys
from pathlib import Path

import pytest

from autosim.research import provision as pv, agent_runtime as runtime
from autosim.research.compute_decision import ComputeDecision
from autosim.research.common import atomic_json, read_json
from autosim.research.agent_roles import role_profile


def gpu():
    return ComputeDecision('cuda', 0, evidence={'selected_device': {'uuid': 'GPU-fixture'}})


def test_selected_gpu_does_not_grant_cpu_or_generated_asset_probes():
    selected = gpu()
    for command in ('import entrypoint', pv.asset_probe('{repo}/input_data')):
        assert pv.operation_compute(command, planned={}, transcript=[], compute=selected) is None
    plan = {'resource_requests': [{'command': 'cuda computation', 'resource': 'gpu'}]}
    assert pv.operation_compute('cuda computation', planned=plan, transcript=[], compute=selected) is selected
    assert pv.operation_compute('different command', planned=plan, transcript=[], compute=selected) is None
    override = [{'resource_requests': [{'command': 'cuda computation', 'resource': 'cpu'}]}]
    assert pv.operation_compute('cuda computation', planned=plan, transcript=override, compute=selected) is None
    unavailable = pv.operation_compute('cuda computation', planned=plan, transcript=[], compute=None)
    assert unavailable.device == 'unavailable'


def test_busy_gpu_refusal_preserves_requested_resource_and_never_launches(tmp_path, monkeypatch):
    from autosim.research.devices import NoCompatibleDevice
    from autosim.research import compute_decision, environment_pool, installation_recovery
    def busy(*args, **kwargs):
        raise NoCompatibleDevice('selected GPU became busy before stage start')
    monkeypatch.setattr(compute_decision, 'recheck_gpu', busy)
    monkeypatch.setattr(pv, 'bounded_run', lambda *a, **k: pytest.fail('must not launch'))
    monkeypatch.setattr(environment_pool, 'publish_wheels', lambda *a, **k: None)
    monkeypatch.setattr(installation_recovery, 'inventory', lambda *a, **k: {})
    result = pv.run('echo fixture', env=dict(os.environ), cwd=tmp_path,
                    timeout=1, output=tmp_path / 'probe.log', compute=gpu())
    assert result['resource'] == result['requested_resource'] == 'gpu'
    assert result['gpu_access'] is False and result['launched'] is False
    assert result['failure_kind'] == 'resource_busy' and result['repair_owner'] == 'resource'
    assert result['evidence_id'] and pv.resource_retryable(result)
    assert not pv.resource_retryable({**result, 'launched': True})
    assert not pv.resource_retryable({**result, 'failure_kind': 'platform'})
    assert not pv.resource_retryable({**result, 'retryable': False})


def test_readonly_init_keeps_native_source_tool_but_no_shell_or_edits(tmp_path):
    profile = role_profile('init', read_only=True)
    args = runtime._role_cli_args(profile, workspace=tmp_path, output=tmp_path,
        python='/usr/bin/python3', model='fixture', max_budget_usd=1)
    config = json.loads(args[args.index('--mcp-config')+1])['mcpServers']['autosim_exec']['args']
    assert '--read-only' in config and '--native-readonly' in config
    assert 'mcp__autosim_exec__inspect_native_environment' in args
    assert 'mcp__autosim_exec__run_command' not in args
    assert profile.builtin_tools == ('Read', 'Glob', 'Grep')


def test_source_catalog_and_capture_work_without_commands_or_gpu(tmp_path):
    from autosim.research.native_context import publish_context
    output = tmp_path / 'run'; repo = output / 'checkout'; repo.mkdir(parents=True)
    prefix = output / 'env'; python = prefix / 'bin/python'; python.parent.mkdir(parents=True)
    python.write_text('not executed')
    atomic_json(output / 'workspace_snapshot.json', {'source': str(repo), 'destination': str(repo),
        'isolated': True, 'schema_version': 2, 'source_entries': []})
    source = prefix / 'lib/sdk/reset.py'; source.parent.mkdir(parents=True)
    source.write_text('def reset(seed):\n    return seed\n')
    (prefix / 'secrets.py').write_text('private')
    (prefix / 'escape.py').symlink_to(tmp_path / 'outside.py')
    publish_context(output, repo, python, {})
    executor = runtime._McpExecutor(workspace=repo, output=output, allow_commands=False, allow_native=True)
    discovery = executor.native_probe({'purpose': 'find source roots', 'source_catalog': True})
    assert 'lib/' in discovery['roots'][0]['entries']
    assert 'secrets.py' not in discovery['roots'][0]['entries']
    assert 'escape.py' not in discovery['roots'][0]['entries']
    captured = executor.native_probe({'purpose': 'read exact SDK reset source',
        'source_request': {'root_index': 0, 'path': 'lib/sdk/reset.py'}})
    assert captured['ref'].startswith('native-source:') and captured['excerpt'] == source.read_text()
    listed = runtime.handle_mcp_message({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}, executor)
    assert any(t['name'] == 'inspect_native_environment' for t in listed['result']['tools'])
    with pytest.raises(ValueError):
        executor.native_probe({'purpose': 'unsafe mix', 'source_catalog': True, 'code': 'print(1)'})
    with pytest.raises(ValueError):
        executor.native_probe({'purpose': 'escape', 'source_request': {'root_index': 0, 'path': '../outside.py'}})


def test_role_session_rotates_only_when_capabilities_change(tmp_path):
    from tests.test_agent_client import client_for
    from autosim.research.runtime_freshness import controller_identity
    client = client_for(tmp_path); root = client.output / 'agent'; key = 'a' * 32
    atomic_json(root / 'roles/init.json', {'role': 'init', 'run_id': client.run_id, 'session_key': key})
    session = {'schema_version': 1, 'status': 'completed',
               'controller_runtime_id': controller_identity(),
               'capability_identity': runtime.capability_identity(role_profile('init', read_only=True))}
    atomic_json(root / f'sessions/{key}.json', session)
    assert client._resume_role('init', read_only=True)
    assert not client._resume_role('init', read_only=False)
    assert read_json(root / f'sessions/{key}.json') == session
    assert client.total_budget_usd == 2
    atomic_json(root / f'sessions/{key}.json', {**session, 'controller_runtime_id': 'old-runtime'})
    assert not client._resume_role('init', read_only=True)
    atomic_json(root / f'sessions/{key}.json', {'schema_version': 1, 'status': 'completed'})
    assert not client._resume_role('init', read_only=True)


def test_cli_advertised_tools_are_evidence_not_model_assertion():
    event = {'type': 'system', 'subtype': 'init', 'session_id': 'fixture', 'tools': [
        'Read', 'mcp__autosim_exec__inspect_native_environment', '/private/path', 'Read']}
    projected = runtime._project_claude_event(event)[0]
    assert projected['advertised_tools'] == ['Read', 'mcp__autosim_exec__inspect_native_environment']
    assert runtime._project_claude_event({**event, 'tools': None})[0]['advertised_tools'] is None


def test_native_tool_transport_schema_is_not_silently_dropped_by_cli():
    schema = runtime._native_tool_schema()['inputSchema']
    assert schema['type'] == 'object'
    assert not {'anyOf', 'oneOf', 'allOf'} & schema.keys()
    assert {'code', 'source_request', 'source_catalog'} <= schema['properties'].keys()


def test_relaxed_transport_schema_does_not_allow_empty_or_invalid_native_request(tmp_path):
    from tests.test_agent_runtime import _fixture
    output, repo = _fixture(tmp_path)
    executor = runtime._McpExecutor(workspace=repo, output=output, allow_commands=False, allow_native=True)
    with pytest.raises(ValueError, match='bounded code'):
        executor.native_probe({'purpose': 'not an executable or source request'})
    with pytest.raises(ValueError):
        executor.native_probe({'purpose': 'invalid', 'source_catalog': False})


def test_context_publication_cannot_claim_missing_cli_tool_is_available(tmp_path):
    from autosim.research.native_context import inspection_availability
    atomic_json(tmp_path/'native_context.json', {})
    tool = 'mcp__autosim_exec__inspect_native_environment'
    assert not inspection_availability(tmp_path)['available']
    atomic_json(tmp_path/'agent/tool_capabilities.json', {'role': 'init',
        'configured_tools': [tool], 'advertised_tools': ['Read']})
    view = inspection_availability(tmp_path)
    assert view['context_published'] and not view['available'] and view['cli_tool_status'] == 'missing_from_cli'
    atomic_json(tmp_path/'agent/tool_capabilities.json', {'role': 'init',
        'configured_tools': [tool], 'advertised_tools': [tool]})
    assert inspection_availability(tmp_path)['available']


def test_framework_source_catalog_retains_selectable_refs_after_privacy_projection(tmp_path):
    from autosim.research.harness_repair import status
    from autosim.research.common import sanitize_model_payload
    projected = sanitize_model_payload(status(tmp_path))
    catalog = {row['source_ref']: row for row in projected['source_catalog']}
    assert catalog['research/provision.py']['candidate_edits_allowed'] is True
    assert catalog['research/native_execution.py']['candidate_edits_allowed'] is False
    assert catalog['research/agent_roles.py']['candidate_edits_allowed'] is False
    assert catalog['research/provision.py']['path'] == '[LOCAL_PATH]'


def test_real_stream_init_callback_records_capabilities_without_shadowed_clock(tmp_path, monkeypatch):
    from tests.test_agent_runtime import _fixture
    from autosim.research.process_executor import ProcessAttempt
    output, repo = _fixture(tmp_path)
    monkeypatch.setattr('autosim.llm_client.load_credential_file', lambda **k: {'source': 'fixture'})
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'fixture-key')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'fixture-model')
    def stream(command, **kwargs):
        kwargs['on_stdout_line'](json.dumps({'type': 'system', 'subtype': 'init',
            'session_id': 'fixture', 'tools': ['Read', 'mcp__autosim_exec__inspect_native_environment']}))
        kwargs['on_stdout_line'](json.dumps({'type': 'result', 'total_cost_usd': 0,
            'result': '{}', 'duration_ms': 1, 'usage': {}}))
        return ProcessAttempt(launched=True, returncode=0, containment_mode='fixture')
    monkeypatch.setattr(runtime, 'run_process_stream', stream)
    result = runtime.run_coding_agent(workspace=repo, output=output, prompt='fixture',
        max_budget_usd=.05, max_total_budget_usd=.05, cli='/usr/bin/claude', role='init',
        read_only=True, auto_skills=False)
    assert result['status'] == 'completed'
    row = read_json(output/'agent/tool_capabilities.json')
    from datetime import datetime, timezone
    assert row['role'] == 'init' and datetime.fromisoformat(row['at']).utcoffset() == timezone.utc.utcoffset(None)
    assert 'mcp__autosim_exec__inspect_native_environment' in row['advertised_tools']
    assert row['capability_identity'] == runtime.capability_identity(role_profile('init', read_only=True))


def test_gpu_contention_resumes_original_probe_without_installer_fix(tmp_path, monkeypatch):
    from autosim.research.evidence_store import capture_attempt_evidence
    repo = tmp_path / 'checkout'; repo.mkdir()
    output = tmp_path / 'run'; output.mkdir()
    probe = '{python} -c "import math"'
    calls = []
    def check(template, **kwargs):
        calls.append(template)
        if len(calls) == 1:
            log = output / 'busy.log'; log.write_text('selected GPU busy before launch')
            proof = capture_attempt_evidence(output, attempt_id='a'*32, log=log,
                receipt_ref='fixture.json', status='failed', returncode=None,
                termination_reason='resource_busy')
            return {**proof, 'command': template, 'ok': False, 'failure_kind': 'resource_busy',
                'requested_resource': 'gpu', 'resource': 'gpu', 'gpu_access': False,
                'launched': False, 'retryable': True, 'seconds': 0, 'excerpt': 'busy'}
        return {'command': template, 'ok': True, 'returncode': 0, 'seconds': .1, 'excerpt': 'imported'}
    monkeypatch.setattr(pv, 'probe_environment', check)
    monkeypatch.setattr(pv, 'platform_facts', lambda: {})
    monkeypatch.setattr(pv, 'resume', lambda *a, **k: pytest.fail('contention is not installer repair'))
    monkeypatch.setattr(pv, 'save_recipe', lambda *a, **k: None)
    monkeypatch.setattr(pv, 'load_recipe', lambda *a, **k: None)
    monkeypatch.setattr(pv, 'plan', lambda *a, **k: {'python': sys.executable,
        'commands': [], 'probes': [probe], 'assets': [], 'resource_requests': [
            {'command': probe, 'resource': 'gpu', 'why': 'synthetic GPU admission fixture'}]})
    kwargs = dict(client=None, prefix=output/'env', output=output, python=sys.executable,
                  manifests={}, assets={}, max_operations=1, compute=gpu())
    first = pv.build(repo, **kwargs)
    assert first['recovery_status'] == 'resource_wait'
    assert read_json(output/'provision_cursor.json')['next_probe'] == 0
    second = pv.build(repo, **kwargs)
    assert second['verdict']['passed'] is True
    assert calls == [probe, probe]
    assert any(row.get('kind') == 'resource_revalidation' for row in read_json(output/'transcript.json')['rows'])
