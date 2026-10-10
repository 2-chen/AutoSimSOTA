"""Canonical data inputs must reach the loader, not become invented CLI flags."""
import json

import pytest

from autosim.research import data_versions
from autosim.research.declarative_backend import DeclarativeBackend
from autosim.research.evidence_store import read_attempt_evidence
from autosim.research.common import read_json
from autosim.research.ideas import Idea
from tests.test_derived_research import research


def test_explicit_canonical_dataset_reaches_top_input_without_duplicate_flag(tmp_path):
    real = research(tmp_path)
    path = str(tmp_path/'reviewed_new_data')
    inputs = real._inputs('train', settings={'dataset': path, 'train.lr': 0.001})
    assert inputs['dataset'] == path
    assert inputs['settings'] == {'train.lr': 0.001}


def test_collected_dataset_override_still_wins_over_training_setting(tmp_path):
    real = research(tmp_path)
    inputs = real._inputs('train', settings={'dataset': 'selected_data'}, dataset='collected_data')
    assert inputs['dataset'] == 'collected_data'
    assert 'dataset' not in inputs['settings']
    assert real._inputs('train', settings={'dataset': None}, dataset='collected_data')['dataset'] == 'collected_data'
    with pytest.raises(ValueError, match='dataset input must be a string'):
        real._inputs('train', settings={'dataset': 'chosen'}, dataset=None)


def test_absent_dataset_keeps_derived_default_and_native_names_are_not_guessed(tmp_path):
    real = research(tmp_path)
    inputs = real._inputs('train', settings={'folder': 'native_folder', 'dataset_root': 'native_root'})
    assert inputs['dataset'] == ''
    assert inputs['settings'] == {'folder': 'native_folder', 'dataset_root': 'native_root'}


def test_generic_flag_emitter_uses_its_own_native_data_option_only_once(tmp_path):
    real = research(tmp_path)
    source = '''def stage_argv_train(i):
    argv = [i["python"], "-c", "pass", "--native-data-folder", i["dataset"]]
    for key in sorted(i["settings"]):
        argv = argv + ["--" + key, str(i["settings"][key])]
    return argv
'''
    backend = DeclarativeBackend(repo=real.repo, answer=real.backend.answer,
                                 sources={'train': source}, parameters={})
    argv = backend.argv('train', real._inputs('train', settings={'dataset': 'chosen', 'lr': 0.001}))
    assert argv.count('--native-data-folder') == 1
    assert argv[argv.index('--native-data-folder')+1] == 'chosen'
    assert '--dataset' not in argv
    assert argv[-2:] == ['--lr', '0.001']


@pytest.mark.parametrize('selected,expected', [({}, 'verified_default'),
    ({'dataset': ''}, 'verified_default'), ({'dataset': 'chosen'}, 'chosen')])
def test_unknown_caller_dataset_does_not_erase_verified_stage_default(tmp_path, selected, expected):
    real = research(tmp_path)
    backend = DeclarativeBackend(repo=real.repo, answer=real.backend.answer,
        sources={'train': 'def stage_argv_train(i):\n    return [i["python"], "-c", "pass", i["dataset"]]\n'},
        parameters={'train': {'dataset': {'value': 'verified_default'}}})
    argv = backend.argv('train', real._inputs('train', settings=selected))
    assert argv[-1] == expected


@pytest.mark.parametrize('name,default', [('checkpoint', 'unselected_policy'), ('steps', 1234)])
def test_dataset_fallback_does_not_relax_weight_or_work_controls(tmp_path, name, default):
    real = research(tmp_path)
    source = f'def stage_argv_train(i):\n    return [i["python"], "-c", "pass", str(i[{name!r}])]\n'
    backend = DeclarativeBackend(repo=real.repo, answer=real.backend.answer,
        sources={'train': source}, parameters={'train': {name: {'value': default}}})
    argv = backend.argv('train', real._inputs('train', settings={}))
    assert argv[-1] == ('' if name == 'checkpoint' else '0')


@pytest.mark.parametrize('value', [None, True, 3, {'path': 'guessed'}])
def test_invalid_canonical_input_is_rejected_instead_of_falling_back(tmp_path, value):
    real = research(tmp_path)
    with pytest.raises(ValueError, match='dataset input must be a string'):
        real._inputs('train', settings={'dataset': value})


@pytest.mark.parametrize('settings', [{'dataset': None}, {'dataset': True}, {'device_index': 'not_integer'}])
def test_invalid_stage_input_returns_sealed_known_failure_not_unknown_live_action(tmp_path, settings):
    real = research(tmp_path)
    result = real.measure(settings=settings, label='invalid_input')
    assert result['ok'] is False and result['where'] == 'train'
    assert result['train']['status'] == 'blocked'
    assert result['train']['native_operation_status'] == 'not_started'
    assert read_attempt_evidence(real.output, result['train']['evidence_id'])['status'] == 'operation_contract_failed'
    assert not (real.run_root/'evaluate').exists()


def test_candidate_input_rejection_commits_idle_checkpoint_and_preserves_baseline(tmp_path, monkeypatch):
    real = research(tmp_path)
    real.library.add(Idea(label='more', granularity='param', change={'train.n_epochs': 2},
                          risk='low', crosses='none', why='CPU fixture hypothesis'))
    real.run(rounds=2, yield_after_action=True)
    baseline = real.run_root/'measurements/baseline.json'
    before = baseline.read_bytes()
    original = real._inputs

    def bad_candidate_input(stage, *, settings, **overrides):
        if stage == 'train' and settings.get('train.n_epochs') == 2:
            settings = {**settings, 'dataset': None}  # Simulated normalization fault.
        return original(stage, settings=settings, **overrides)

    monkeypatch.setattr(real, '_inputs', bad_candidate_input)
    real.library.audit(real.red_lines())
    report = real.run(rounds=2, yield_after_action=True, selected_idea_label='more',
                      main_controller_owns_selection=True)
    session = read_json(real.run_root/'controller_session.json')
    assert report['run_status'] == session['status'] == 'paused'
    assert session['pending_action'] is None and session['next_round'] == 2
    assert baseline.read_bytes() == before
    failed = read_json(real.run_root/'measurements/round_1.json')
    assert failed['ok'] is False and failed['train']['command_started'] is False
    assert read_attempt_evidence(real.output, failed['train']['evidence_id'])['status'] == 'operation_contract_failed'


@pytest.mark.parametrize('witness', [True, False])
def test_selected_data_handle_reaches_real_cpu_fixture_loader_with_consumption_gate(tmp_path, witness):
    # CPU contract fixture, NOT RoboSyn collection/training or a benchmark score.
    real = research(tmp_path)
    data = real.output/'data.json'
    data.write_text('[1,2,3]')
    row = data_versions.register(real.output, real.repo,
        dict(id='extra', dataset_ref='data.json', source_refs=[], reason='CPU fixture'),
        dict(allowed_training_data=True, loader_connection_supported=True, sources_supported=True))
    proof = dict(data_version_id='extra', setting='dataset',
                 identity_field=row['identity_field'], identity=row['identity'], samples_read=3)
    code = (
        "import pathlib,sys,json; data=json.loads(pathlib.Path(sys.argv[1]).read_text()); "
        "assert data==[1,2,3]; out=pathlib.Path(sys.argv[2])/'models'; "
        "out.mkdir(parents=True,exist_ok=True); (out/'model.pth').write_text(str(data)); "
        + (f"print({(data_versions.MARKER+json.dumps(proof))!r})" if witness else "print('loader read 3 samples')"))
    source = ("def stage_argv_train(i):\n"
              f"    return [i['python'], '-c', {code!r}, i['dataset'], i['output']]\n")
    real.sources['train'] = source
    # Production rebuilds this backend after deriving a command; compiled argv
    # functions are immutable within a backend instance, not hot-reloaded maps.
    real.backend = DeclarativeBackend(repo=real.repo, answer=real.backend.answer,
                                      sources=real.sources, parameters={})
    result = real.measure(settings={'dataset': 'data-version:extra'}, label='data_candidate')
    if witness:
        assert result['ok'] is True
        assert result['data_consumption']['status'] == 'verified'
    else:
        assert result['ok'] is False and result['where'] == 'training_data'
        assert result['metric_value'] is None
        assert not (real.run_root/'evaluate').exists()


@pytest.mark.parametrize('frozen', [{}, {'seed': 5}])
def test_empty_frozen_eval_settings_are_not_replaced_by_training_settings(tmp_path, frozen):
    from tests.test_derived_research import research
    from autosim.research.common import read_json
    real = research(tmp_path)
    result = real.measure(settings={**frozen, 'steps': 3}, label='baseline')
    assert result['ok']
    assert real._frozen_search_settings() == frozen
    arm = read_json(real.run_root / 'selection_record.json')['arms'][-1]
    assert arm['settings'] == frozen and 'steps' not in arm['settings']
