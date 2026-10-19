"""System comparisons are distinct from data-arm studies and never grant execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from tools import system_study as study


def draft():
    from autosim.research.common import read_json
    return read_json(Path(__file__).resolve().parents[2] / 'docs/paper/SYSTEM_EXPERIMENT_MATRIX.json')


def formal(tmp_path):
    manifest = draft()
    manifest['phase'] = 'formal'
    for task in manifest['tasks']:
        task['native_task'] = task['native_task'] or 'source-selected-task'
        task['admission'] = 'fixture admission report; not real capability evidence'
    for system in manifest['systems']:
        system['model_configuration'] = 'same controlled model, fixture'
        system['capability_manifest'] = 'same lawful resources, fixture'
    manifest['protocol'] = {key: value or 'explicit fixture protocol, not a real experiment'
                            for key, value in manifest['protocol'].items()}
    manifest['bindings'] = {cell: {'output': str(tmp_path / cell),
        'repository': str(tmp_path / cell / 'checkout'), 'run_id': 'derived'}
        for cell in study.cells(manifest)}
    return manifest


def test_draft_has_system_repetitions_not_data_seeds_or_scores():
    manifest = draft()
    assert study.validate(manifest) == manifest
    report = study.inspect(manifest)
    assert report['planned_cells'] == 18
    assert len(report['unadmitted_tasks']) == 2
    assert report['unresolved_protocol_fields']
    assert all(row['status'] == 'unallocated' for row in report['rows'])
    assert not report['formal_paper_eligible']
    assert not any('metric_value' in row for row in report['rows'])
    assert 'accepted_new_episodes' not in manifest['protocol']


@pytest.mark.parametrize('fault', ['null_protocol', 'null_admission', 'one_replica',
    'duplicate_replica', 'bool_replica', 'missing_control', 'duplicate_output',
    'nested_output', 'escape_checkout', 'source_as_checkout', 'unknown_binding'])
def test_formal_rejects_ambiguous_or_false_repetition(tmp_path, fault):
    manifest = formal(tmp_path)
    bindings = list(manifest['bindings'].values())
    if fault == 'null_protocol': manifest['protocol']['resource_package'] = None
    if fault == 'null_admission': manifest['tasks'][0]['admission'] = None
    if fault == 'one_replica': manifest['research_replicas'] = [0]
    if fault == 'duplicate_replica': manifest['research_replicas'] = [0, 0, 1]
    if fault == 'bool_replica': manifest['research_replicas'] = [False, 1, 2]
    if fault == 'missing_control': manifest['systems'][1]['kind'] = 'ablation'
    if fault in {'duplicate_output', 'nested_output'}:
        bindings[1]['output'] = bindings[0]['output'] + ('/child' if fault == 'nested_output' else '')
        bindings[1]['repository'] = bindings[1]['output'] + '/checkout'
    if fault == 'escape_checkout': bindings[0]['repository'] = bindings[0]['output'] + '/../outside'
    if fault == 'source_as_checkout': bindings[0]['repository'] = bindings[0]['output']
    if fault == 'unknown_binding': manifest['bindings']['unplanned'] = bindings[0]
    with pytest.raises(ValueError): study.validate(manifest)


def test_freeze_is_prospective_immutable_and_detects_runtime_drift(tmp_path, monkeypatch):
    manifest = formal(tmp_path / 'runs')
    monkeypatch.setattr(study, 'identity', lambda project: 'frozen-implementation')
    snapshot = study.freeze(manifest, tmp_path / 'freeze')
    report = study.inspect(manifest, frozen=snapshot)
    assert report['implementation_matches_freeze'] is True
    assert all(row['status'] == 'not_started' for row in report['rows'])
    assert not any(Path(b['output']).exists() for b in manifest['bindings'].values())
    assert not report['formal_paper_eligible']
    changed = deepcopy(manifest); changed['protocol']['resource_package'] = 'changed budget'
    with pytest.raises(ValueError): study.inspect(changed, frozen=snapshot)
    monkeypatch.setattr(study, 'identity', lambda project: 'new-implementation')
    assert study.inspect(manifest, frozen=snapshot)['implementation_matches_freeze'] is False


@pytest.mark.parametrize('fault', ['initialized', 'protocol_inside_trial', 'symlink'])
def test_freeze_cannot_relabel_development_or_initialize_a_trial(tmp_path, fault):
    manifest = formal(tmp_path / 'runs')
    first = Path(next(iter(manifest['bindings'].values()))['output'])
    destination = tmp_path / 'freeze'
    if fault == 'initialized': first.mkdir(parents=True)
    if fault == 'protocol_inside_trial': destination = first / 'frozen'
    if fault == 'symlink':
        first.parent.mkdir(parents=True); target = tmp_path / 'target'; target.mkdir()
        first.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError): study.freeze(manifest, destination)
    assert not (destination / 'protocol.json').exists()


def test_inspect_never_reads_development_scores_or_held_out(tmp_path):
    manifest = formal(tmp_path / 'runs')
    first = Path(next(iter(manifest['bindings'].values()))['output']); first.mkdir(parents=True)
    (first / 'research').write_text('not a directory: no scientific evidence scan is permitted')
    report = study.inspect(manifest)
    assert report['rows'][0]['status'] == 'initialized_unreviewed'
    assert report['pending_result_audit']
    assert not report['formal_paper_eligible']
