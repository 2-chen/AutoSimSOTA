"""Real reader handoffs use public source and read-only native context, not host grants."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from autosim.research import agent_runtime as runtime
from autosim.research.agent_roles import role_profile
from autosim.research.agent_tasks import snapshot
from autosim.research.common import atomic_json
from autosim.research.native_context import publish_context
from autosim.research.native_source_citations import resolve as resolve_native
from autosim.research.checkout_source_citations import resolve as resolve_checkout
from autosim.research.workspace_snapshot import create


def reader(tmp_path, *, real_python=False):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'loader.py').write_text('VALUE = "delegated"\n')
    (source / 'private_not_selected.py').write_text('VALUE = "not delegated"\n')
    output = tmp_path / 'run'
    repo = output / 'checkout'
    create(source, repo)
    prefix = output / 'env'
    if real_python:
        subprocess.run(['/usr/bin/python3', '-m', 'venv', '--copies', '--without-pip', str(prefix)], check=True)
    else:
        (prefix / 'bin').mkdir(parents=True)
        (prefix / 'bin/python').write_text('not executed')
    (prefix / 'public_sdk.py').write_text('def reset(seed):\n    return seed\n')
    (prefix / 'protected_marker').write_text('keep')
    publish_context(output, repo, prefix / 'bin/python', {})
    atomic_json(output / 'private_accounting.json', {'not': 'a grant'})
    worker, checkout = snapshot(output, repo, 'a' * 32, source_paths=['loader.py'])
    return output, repo, prefix, worker, checkout


def test_objective_native_access_is_source_only_in_cli_and_actual_mcp(tmp_path):
    output, repo, _, _, _ = reader(tmp_path)
    profile = role_profile('objective', read_only=True)
    assert profile.can_inspect_native and profile.native_source_only
    assert not profile.can_edit_checkout and not profile.can_execute_diagnostics
    args = runtime._role_cli_args(profile, workspace=repo, output=output,
        python='/usr/bin/python3', model='fixture', max_budget_usd=1)
    config = json.loads(args[args.index('--mcp-config') + 1])['mcpServers']['autosim_exec']['args']
    assert '--read-only' in config and '--native-source-only' in config
    assert 'mcp__autosim_exec__run_command' not in args
    executor = runtime._McpExecutor(output=output, workspace=repo,
        allow_commands=False, allow_native=True, native_source_only=True)
    listing = runtime.handle_mcp_message({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}, executor)
    tool = next(t for t in listing['result']['tools'] if t['name'] == 'inspect_native_environment')
    assert set(tool['inputSchema']['properties']) == {
        'purpose', 'timeout_seconds', 'source_catalog', 'source_request', 'checkout_source_request',
        'framework_source_request'}
    assert executor.native_probe({'purpose': 'public source roots', 'source_catalog': True})['status'] == 'source_catalog'
    for extra in ({'code': 'print(1)'}, {'source_catalog': True, 'code': 'print(1)'},
                  {'policy_request': {'label': 'baseline', 'measurement_sha256': 'a' * 64}},
                  {'data_profile_request': {'format': 'lerobot_v2.1', 'root': 'data'}},
                  {'source_catalog': True, 'source_request': {'root_index': 0, 'path': 'public_sdk.py'}}):
        with pytest.raises(ValueError, match='source-only'):
            executor.native_probe({'purpose': 'not granted', **extra})


def test_reader_source_ids_resolve_in_main_without_copying_native_environment(tmp_path):
    output, repo, _, worker, checkout = reader(tmp_path)
    assert not (worker / 'native_context.json').exists()
    executor = runtime._McpExecutor(output=worker, workspace=checkout,
        allow_commands=False, allow_native=True, native_source_only=True)
    native = executor.native_probe({'purpose': 'fresh reset source',
        'source_request': {'root_index': 0, 'path': 'public_sdk.py'}})
    resolved = resolve_native(output, repo, native['ref'], 'return seed')
    assert resolved['quote'] == 'return seed'
    assert 'return seed' in executor.read_evidence({'evidence_id': native['evidence_id']})['text']
    checkout_source = executor.native_probe({'purpose': 'selected loader source',
        'checkout_source_request': {'ref': 'loader.py:1'}})
    assert resolve_checkout(output, repo, checkout_source['ref'], checkout_source['quote'])['quote'] == 'VALUE = "delegated"\n'
    assert not (worker / 'native_source_citations').exists()
    assert not (worker / 'env').exists()
    with pytest.raises((ValueError, OSError)):
        executor.native_probe({'purpose': 'outside source scope',
            'checkout_source_request': {'ref': 'private_not_selected.py:1'}})
    (repo / 'loader.py').write_text('VALUE = "changed"\n')
    with pytest.raises(ValueError, match='no longer matches'):
        executor.native_probe({'purpose': 'stale delegated file',
            'checkout_source_request': {'ref': 'loader.py:1'}})


def test_owner_lookup_uses_the_same_marker_bound_as_the_runtime(tmp_path):
    output, _, _, worker, checkout = reader(tmp_path)
    path = output / 'workspace_snapshot.json'
    manifest = json.loads(path.read_text())
    manifest['fixture_padding'] = 'x' * (200 * 1024)
    atomic_json(path, manifest)
    assert path.stat().st_size > 128 * 1024
    executor = runtime._McpExecutor(output=worker, workspace=checkout,
        allow_commands=False, allow_native=True, native_source_only=True)
    assert executor.native_probe({'purpose': 'public dependency names', 'source_catalog': True})['status'] == 'source_catalog'


@pytest.mark.skipif(shutil.which('bwrap') is None, reason='bubblewrap unavailable')
def test_real_data_reader_borrows_only_readonly_prefix_and_its_source_snapshot(tmp_path):
    output, repo, prefix, worker, checkout = reader(tmp_path, real_python=True)
    executor = runtime._McpExecutor(output=worker, workspace=checkout,
        allow_commands=False, allow_native=True)
    code = ("from pathlib import Path; import os; "
            f"p=Path({str(repo)!r}); print((p/'loader.py').read_text().strip()); "
            "print((p/'private_not_selected.py').exists()); "
            f"print(Path({str(output / 'private_accounting.json')!r}).exists()); "
            "print(Path('/dev/nvidia0').exists()); "
            f"Path({str(prefix / 'protected_marker')!r}).write_text('not allowed')")
    result = executor.native_probe({'purpose': 'real readonly reader boundary', 'code': code})
    assert result['returncode'] != 0
    assert result['stdout'].splitlines() == ['VALUE = "delegated"', 'False', 'False', 'False']
    assert 'Read-only file system' in result['stderr']
    assert result['gpu_access'] is False and result['native_context_stable'] is True
    assert (prefix / 'protected_marker').read_text() == 'keep'
    assert (output / 'evidence' / (result['evidence_id'] + '.json')).is_file()
    assert not (worker / 'env').exists()
