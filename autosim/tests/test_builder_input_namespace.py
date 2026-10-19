"""A native trainer read is not a generated caller-builder read."""
from types import SimpleNamespace

from autosim.research.candidate_settings import canonical_dataset_input
from autosim.research.harness_repair import read_source, ALLOWED
from autosim.research.runtime_freshness import controller_context


def test_native_get_call_does_not_establish_builder_input():
    native = 'selection.get("dataset")'
    no_slot = SimpleNamespace(sources={'train':'def train(i): return ["python", "train.py"]'})
    slot = SimpleNamespace(sources={'train':'def train(i): return ["python", "train.py", i["dataset"]]'})
    assert native
    assert not canonical_dataset_input(no_slot,'dataset','data-version:registered_v1')
    assert canonical_dataset_input(slot,'dataset','data-version:registered_v1')
    assert not canonical_dataset_input(slot,'dataset_root','data-version:registered_v1')


def test_current_contract_names_actual_caller_and_never_injects_native_environment(tmp_path):
    item = controller_context(tmp_path,[])['current_admission_contracts']['candidate_settings']
    assert item['revision']==4 and 'BUILDER source' in item['rule']
    assert 'TRAINING randomness only' in item['rule']
    assert 'settings.seed still governs frozen evaluation' in item['rule']
    assert 'not injected' in item['rule'] and 'neither approve' in item['rule']
    assert not list(tmp_path.iterdir())


def test_public_axis_source_is_read_only_not_a_new_patch_permission(tmp_path):
    row=read_source(tmp_path,{'path':'research/adapter_protocol.py','query':'def accepts','limit':35})
    assert 'def accepts' in row['excerpt']
    assert not row['candidate_edits_allowed'] and not row['live_activation_allowed']
    assert 'research/adapter_protocol.py' not in ALLOWED


def test_loaded_builder_is_readable_but_not_an_experiment_or_environment_writer(tmp_path):
    source = 'def train(i):\n    return ["python", "native_train.py", i["dataset"]]'
    item = controller_context(tmp_path, [], train_source=source)[
        'current_admission_contracts']['candidate_settings']['current_train_builder']
    assert item['text']==source and item['complete']
    assert item['namespace']=='run_stage_argv_builder_not_native_trainer'
    assert 'not native execution' in item['authority']
    assert not list(tmp_path.iterdir())


def test_source_bound_omission_never_masquerades_as_a_complete_caller(tmp_path):
    source = '# '+('large public source '*1000)
    item = controller_context(tmp_path, [], train_source=source)[
        'current_admission_contracts']['candidate_settings']['current_train_builder']
    assert not item['complete'] and 'text' not in item
    assert 'Do not infer missing arguments' in item['reading_guidance']
    absent = controller_context(tmp_path, [])['current_admission_contracts']['candidate_settings']
    assert 'current_train_builder' not in absent


def test_private_host_paths_and_keys_are_not_exposed_in_caller_source(tmp_path):
    source = 'token="sk-testsecret123456789012345678901234"\nroot="/home/private/research/dataset.hdf5"'
    item = controller_context(tmp_path, [], train_source=source)[
        'current_admission_contracts']['candidate_settings']['current_train_builder']
    assert '/home/private' not in item['text']
    assert 'sk-testsecret123456789012345678901234' not in item['text']
