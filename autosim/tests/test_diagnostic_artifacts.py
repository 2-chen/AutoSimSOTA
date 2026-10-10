from pathlib import Path

import pytest

from autosim.research.common import atomic_json, digest
from autosim.research.diagnostic_artifacts import policy_binding, binding_unchanged
from tests.test_frozen_policy_trial import fixture


def owned_fixture(tmp_path, directory=False):
    real, row, request = fixture(tmp_path, directory=directory)
    atomic_json(real.output/'run_state.json', {'run_id': real.run_id})
    return real, row, request


@pytest.mark.parametrize('directory', [False, True])
def test_exact_scored_archive_grant_and_later_identity(directory, tmp_path):
    real, row, request = owned_fixture(tmp_path, directory)
    grant = policy_binding(real.output, request)
    assert grant['path'] == row['policy_artifact']['path']
    assert grant['read_only'] and binding_unchanged(real.output, grant)
    public = {k:v for k,v in grant.items() if k != 'path'}
    assert binding_unchanged(real.output, public)
    target = Path(grant['path'])/ 'model.bin' if directory else Path(grant['path'])
    target.write_text('changed policy bytes')
    assert not binding_unchanged(real.output, public)
    with pytest.raises(ValueError):
        policy_binding(real.output, request)


@pytest.mark.parametrize('fault', ['bad_label','extra','hash','owner','heldout','row','symlink','released','heldout_symlink'])
def test_grants_refuse_unverified_requests_and_unsafe_owner(tmp_path, fault):
    real, row, request = owned_fixture(tmp_path)
    if fault == 'bad_label': request['label'] = '../baseline'
    elif fault == 'extra': request['path'] = row['policy_artifact']['path']
    elif fault == 'hash': request['measurement_sha256'] = 'a'*64
    elif fault == 'owner': atomic_json(real.output/'run_state.json', [])
    elif fault == 'heldout': atomic_json(real.run_root/'confirmation_attempts/seen.json', {})
    elif fault == 'heldout_symlink':
        (real.run_root/'confirmation_attempts').symlink_to(tmp_path/'missing-heldout', target_is_directory=True)
    elif fault == 'symlink':
        path = Path(row['policy_artifact']['path'])
        other = path.with_name('other')
        path.rename(other)
        path.symlink_to(other)
    else:
        row = [] if fault == 'row' else {**row, 'baseline_origin':'released'}
        atomic_json(real.run_root/'measurements/baseline.json', row)
        request['measurement_sha256'] = digest(real.run_root/'measurements/baseline.json')
    with pytest.raises(ValueError):
        policy_binding(real.output, request)


def test_cpu_grant_mounts_only_archive_and_preserves_readonly_boundary(tmp_path):
    import subprocess
    from autosim.research.agent_runtime import _McpExecutor
    from autosim.research.native_context import publish_context
    from autosim.research.common import run_local_environment
    real, row, request = owned_fixture(tmp_path, True)
    repo = real.output/'checkout'
    repo.mkdir()
    atomic_json(real.output/'workspace_snapshot.json', {'destination':str(repo)})
    subprocess.run(['/usr/bin/python3','-m','venv','--without-pip',str(real.output/'env')], check=True)
    publish_context(real.output, repo, real.output/'env/bin/python', run_local_environment(real.output, {}))
    executor = _McpExecutor(output=real.output, workspace=repo, allow_commands=False, allow_native=True)
    code = """import os, pathlib
p = pathlib.Path(os.environ['AUTOSIM_DIAGNOSTIC_POLICY'])
assert (p/'model.bin').read_text() == 'two'
assert not pathlib.Path('/dev/nvidia0').exists()
try: (p/'model.bin').write_text('corrupt')
except OSError: print('archive-readonly')
else: raise RuntimeError('archive writable')
print('actual-scored-policy-read')
"""
    receipt = executor.native_probe({'code':code, 'purpose':'load exact scored policy', 'policy_request':request})
    assert receipt['status'] == 'completed', receipt
    assert receipt['diagnostic_policy_binding_stable'] is True
    assert 'path' not in receipt['diagnostic_policy_binding']
    assert 'actual-scored-policy-read' in receipt['stdout'] and 'archive-readonly' in receipt['stdout']
    assert (Path(row['policy_artifact']['path'])/'model.bin').read_text() == 'two'
    assert executor.calls == 1 and executor.tool_seconds > 0


def test_granted_diagnostic_not_readmitted_after_heldout_exposure(tmp_path):
    real, row, request = owned_fixture(tmp_path)
    grant = policy_binding(real.output, request)
    assert binding_unchanged(real.output, grant)
    atomic_json(real.run_root/'confirmation_attempts/seen.json', {})
    assert not binding_unchanged(real.output, grant)


def test_invalid_grant_still_charges_tool_call_and_elapsed_time(tmp_path):
    from autosim.research.agent_runtime import _McpExecutor
    repo = tmp_path/'checkout'
    repo.mkdir()
    atomic_json(tmp_path/'workspace_snapshot.json', {'destination':str(repo)})
    executor = _McpExecutor(output=tmp_path, workspace=repo, allow_commands=False, allow_native=True)
    with pytest.raises((ValueError, OSError)):
        executor.native_probe({'code':'print(1)', 'purpose':'missing owner', 'policy_request':{}})
    assert executor.calls == 1 and executor.tool_seconds > 0
