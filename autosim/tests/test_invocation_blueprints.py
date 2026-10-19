import copy
import sys

import pytest

from autosim.research.common import atomic_json, object_digest, read_json
from autosim.research import execution_interface, invocation_blueprints as blueprints
from tests.test_execution_interface import interface


def fixture(tmp_path):
    output, repo, proof = interface(tmp_path)
    entry = {'source': "def stage_argv_train(i):\n return [i['python'],i['repo']+'/train.py']\n",
             'parameters': {}, 'row': {'entrypoint': 'train.py', 'available': True,
                 'environment': {'NATIVE_DRIVER': 'headless', 'PYTHON_BIN': sys.executable},
                 'working_directory': '{repo}', 'artifact': 'checkpoint/**'}}
    atomic_json(output/'derived_stages.json', {'train': entry})
    log = read_json(output/'derivation_attempts/train.json')
    atomic_json(output/'derivation_attempts/train.json', {**log, 'arguments': {'task': 'native-task'}})
    execution_interface.publish(output, repo, 'train')
    capsule = blueprints.remember(output, repo, 'train', entry,
                                 interpreter=sys.executable, task='native-task')
    return output, repo, entry, capsule, proof


def choose(output, original_repo, entry, capsule, **changes):
    kwargs = dict(repo=original_repo, stage='train', interpreter=sys.executable,
                  task='native-task', current_row=entry['row'])
    kwargs.update(changes)
    return blueprints.load(output, capsule['ref'], **kwargs)


def test_source_edit_retires_approval_but_not_explicit_runtime_blueprint(tmp_path):
    output, repo, entry, capsule, _ = fixture(tmp_path)
    before = (output/capsule['ref']).read_bytes()
    (repo/'train.py').write_text('print("new trainer")\n')
    atomic_json(output/'derived_stages.json', {})
    with pytest.raises((ValueError, KeyError)):
        execution_interface.reuse(output, repo, 'train')
    loaded = choose(output, repo, entry, capsule)
    assert loaded['entry']['row']['environment'] == entry['row']['environment']
    assert loaded['status'] == 'unverified'
    assert 'NEW native verification' in loaded['authority']
    assert read_json(output/'derived_stages.json') == {}
    assert (output/capsule['ref']).read_bytes() == before
    assert blueprints.catalog(output)[0]['selection'].startswith('Main chooses')


@pytest.mark.parametrize('change', ['repository', 'task', 'interpreter', 'stage',
                                    'entrypoint', 'environment', 'bindings', 'proof', 'capsule', 'interface'])
def test_blueprint_cannot_transfer_other_authority(tmp_path, change):
    output, repo, entry, capsule, proof = fixture(tmp_path)
    kwargs = {}
    if change == 'repository':
        other = tmp_path/'different'; other.mkdir(); kwargs['repo'] = other
    elif change in {'task', 'interpreter', 'stage'}: kwargs[change] = 'other'
    elif change == 'entrypoint': kwargs['current_row'] = {'entrypoint': 'different.py'}
    elif change == 'environment': atomic_json(output/'environment.json', {'interpreter': 'other'})
    elif change == 'bindings':
        doc = read_json(output/'workspace_snapshot.json')
        atomic_json(output/'workspace_snapshot.json', {**doc, 'resource_bindings': [{'target': 'other'}]})
    elif change == 'proof': (output/proof['log_ref']).write_text('tampered native log')
    elif change == 'capsule':
        doc = read_json(output/capsule['ref']); doc['entry']['row']['environment']['NATIVE_DRIVER'] = 'changed'
        atomic_json(output/capsule['ref'], doc)
    else:
        atomic_json(output/capsule['interface_ref'], {'id': capsule['interface_id']})
    with pytest.raises((ValueError, KeyError)):
        choose(output, repo, entry, capsule, **kwargs)


def test_archive_migration_uses_original_task_and_matching_provenance(tmp_path):
    output, repo, entry, capsule, _ = fixture(tmp_path)
    archive = output/'superseded_stages'/'old.json'
    atomic_json(archive, {'kept': {'train': entry}})
    atomic_json(output/'derived_stages.json', {})
    assert blueprints.import_archives(output, repo, interpreter=sys.executable) == [capsule['ref']]
    log = read_json(output/'derivation_attempts/train.json')
    atomic_json(output/'derivation_attempts/train.json', {**log, 'arguments': {'task': 'different'}})
    assert blueprints.import_archives(output, repo, interpreter=sys.executable) == []
    assert read_json(output/'derived_stages.json') == {}


@pytest.mark.parametrize('location', ['directory', 'capsule', 'interface'])
def test_symlink_blueprints_and_evidence_are_rejected(tmp_path, location):
    output, repo, entry, capsule, _ = fixture(tmp_path)
    target = output/capsule['ref'] if location != 'interface' else output/capsule['interface_ref']
    if location == 'directory': target = target.parent
    relocated = tmp_path/('relocated-'+location)
    target.rename(relocated)
    target.symlink_to(relocated, target_is_directory=location == 'directory')
    with pytest.raises(ValueError): choose(output, repo, entry, capsule)
    if location == 'directory': assert blueprints.catalog(output) == []


def test_remember_refuses_unpublished_adapter_and_wrong_task(tmp_path):
    output, repo, entry, _, _ = fixture(tmp_path)
    changed = copy.deepcopy(entry); changed['row']['environment']['NATIVE_DRIVER'] = 'different'
    with pytest.raises(ValueError):
        blueprints.remember(output, repo, 'train', changed, interpreter=sys.executable, task='native-task')
    with pytest.raises(ValueError):
        blueprints.remember(output, repo, 'train', entry, interpreter=sys.executable, task='different-task')


def test_broken_optional_catalog_entry_does_not_crash_controller(tmp_path):
    output, _, _, capsule, _ = fixture(tmp_path)
    (output/'invocation_blueprints'/'broken.json').symlink_to(tmp_path/'absent')
    assert blueprints.catalog(output)[0]['id'] == capsule['id']


def test_main_projection_preserves_selectable_blueprint_id(tmp_path):
    from autosim.research.prepare import _controller_model_value
    output, repo, _, capsule, _ = fixture(tmp_path)
    projected = _controller_model_value(
        {'historical_invocation_blueprints': blueprints.catalog(output)},
        local_roots=(repo, output))['historical_invocation_blueprints'][0]
    assert projected['id'] == capsule['id']
    assert projected['proposed_operation']['arguments']['blueprint_ref'] == capsule['id']


def test_fix_can_inspect_blueprint_authority_but_not_edit_it():
    from autosim.research.harness_repair import READABLE, ALLOWED
    for ref in ['research/invocation_blueprints.py', 'research/execution_interface.py']:
        assert ref in READABLE
        assert ref not in ALLOWED


def test_preparation_uses_blueprint_only_on_main_selection_and_still_verifies(tmp_path, monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research import prepare as module
    output, repo, entry, capsule, _ = fixture(tmp_path)
    real = preparation(tmp_path/'controller'); real.output = output; real.repo = repo
    real.interpreter = sys.executable
    real.execution = {'stages': {'train': {'available': True, 'entrypoint': 'train.py'}}}
    monkeypatch.setattr(real, '_declared_task', lambda: 'native-task')
    monkeypatch.setattr(module.execution_derive, 'checkpoint_for_verification', lambda *a, **k: {'path': ''})
    seen = []
    def verify(*args, **kwargs):
        seen.append((args[2], kwargs.get('invocation_blueprint')))
        return None, {}, [{'status': 'rejected', 'error': 'fixture remains unverified'}], args[2]
    monkeypatch.setattr(module.execution_derive, 'make_runnable', verify)
    real._step_derive_a_command(stage='train')
    assert seen[-1][1] is None and 'environment' not in seen[-1][0]
    real._step_derive_a_command(stage='train', blueprint_ref=capsule['id'])
    assert seen[-1][0]['environment'] == entry['row']['environment']
    assert seen[-1][1]['ref'] == capsule['ref']
    assert 'train' not in real.stages
    assert real._step_derive_a_command(stage='train', blueprint_ref=capsule['id'], reuse_verified=True)['outcome'] == 'rejected'
    assert real._step_derive_a_command(stage='train', blueprint_ref=capsule['id'], draft_ref='other')['outcome'] == 'rejected'
    assert len(seen) == 2
