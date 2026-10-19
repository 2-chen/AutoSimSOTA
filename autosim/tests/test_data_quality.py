"""Selection/counter/partial-output invariants, not robot performance claims."""
import copy
import json

import pytest

from autosim.research import data_quality, data_versions
from autosim.research.common import digest, object_digest
from autosim.research.experiment_bundle import artifact_identity
from autosim.research.evidence_store import capture_attempt_evidence
from tests.test_data_versions import registered


def manifest(tmp_path):
    root = tmp_path / 'episodes'
    root.mkdir()
    episodes = []
    for i in range(3):
        path = root / f'{i}.json'
        path.write_text(json.dumps({'action': i}))
        episodes.append({'id': str(i), 'committed': True, 'files': {path.name: digest(path)},
            'content_sha256': digest(path), 'format_valid': True, 'source_family': 'family',
            'source_instance': str(i), 'split': 'train', 'supervision': 'reset_expert',
            'conditions': {'pose': {'value': i / 10, 'unit': 'metres', 'source_ref': 'native-reset'}}})
    field, identity = artifact_identity(root)
    return root, dict(schema_version=1, identity_field=field, identity=identity, scope='full', episodes=episodes)


def target():
    return dict(field='pose', op='between', value=[.05, .25], unit='metres')


def test_native_selection_is_reversible_and_condition_based(tmp_path):
    root, rows = manifest(tmp_path)
    before = artifact_identity(root)
    data_quality.validate_manifest(rows, root=root)
    result = data_quality.selection(rows, target=target(), require_supervision='reset_expert')
    assert result['selected_episode_ids'] == ['1', '2']
    assert result['counts']['target_observed'] == 3
    assert result['counts']['target_matched'] == 2
    assert result['rejected'] == {'outside_target': 1}
    assert result['manifest_sha256'] == object_digest(rows)
    assert artifact_identity(root) == before


def test_unknown_conditions_and_provenance_are_not_invented(tmp_path):
    _, rows = manifest(tmp_path)
    rows['episodes'][1]['conditions'] = {}
    rows['episodes'][2]['conditions']['pose']['unit'] = 'millimetres'
    result = data_quality.selection(rows, target=target())
    assert result['selected_episode_ids'] == []
    assert result['counts']['target_observed'] == 1
    assert result['rejected']['target_condition_unobserved'] == 2
    rows['episodes'][0]['source_instance'] = None
    assert data_quality.selection(rows, deduplicate='source_instance')['rejected']['source_instance_unverified'] == 1


def test_shared_container_is_not_automatically_a_duplicate_episode(tmp_path):
    _, rows = manifest(tmp_path)
    for row in rows['episodes']:
        row.pop('content_sha256')
    selected = data_quality.selection(rows)
    assert selected['selected_episode_ids'] == []
    assert selected['rejected'] == {'episode_content_identity_unverified': 3}
    assert len(data_quality.selection(rows, deduplicate='none')['selected_episode_ids']) == 3


def test_shared_native_container_is_not_rehashed_for_every_episode(tmp_path,monkeypatch):
    root,rows=manifest(tmp_path)
    for row in rows['episodes']:row['files']=dict(rows['episodes'][0]['files'])
    calls=[];actual=data_quality.digest
    monkeypatch.setattr(data_quality,'digest',lambda path:calls.append(str(path)) or actual(path))
    data_quality.validate_manifest(rows,root=root)
    assert len(calls)==1
    rows['episodes'][1]['files']['0.json']='0'*64
    with pytest.raises(ValueError):data_quality.validate_manifest(rows,root=root)


def test_exact_duplicates_and_same_source_instances_are_separate(tmp_path):
    _, rows = manifest(tmp_path)
    rows['episodes'][1]['content_sha256'] = rows['episodes'][0]['content_sha256']
    assert data_quality.selection(rows)['rejected'] == {'duplicate_content': 1}
    rows['episodes'][1]['source_instance'] = rows['episodes'][0]['source_instance']
    assert data_quality.selection(rows, deduplicate='source_instance')['rejected'] == {'duplicate_source_instance': 1}


@pytest.mark.parametrize('fault', ['changed_file', 'escaped_file', 'duplicate_id', 'bad_scope', 'bool_format', 'bool_schema'])
def test_manifest_rejects_changed_or_ambiguous_committed_bytes(tmp_path, fault):
    root, rows = manifest(tmp_path)
    if fault == 'changed_file': (root/'0.json').write_text('changed')
    elif fault == 'escaped_file': rows['episodes'][0]['files'] = {'../other': '0'*64}
    elif fault == 'duplicate_id': rows['episodes'][1]['id'] = '0'
    elif fault == 'bad_scope': rows['scope'] = 'certified'
    elif fault == 'bool_format': rows['episodes'][0]['format_valid'] = 1
    else: rows['schema_version'] = True
    with pytest.raises(ValueError): data_quality.validate_manifest(rows, root=root)


@pytest.mark.parametrize('value', [dict(field='x', op='eval', value='1+1', unit='m'),
    dict(field='x', op='between', value=[2, 1], unit='m'),
    dict(field='x', op='gt', value=True, unit='m'),
    dict(field='x', op='eq', value=float('nan'), unit='m')])
def test_predicates_cannot_execute_or_hide_invalid_numbers(value):
    with pytest.raises(ValueError): data_quality.predicate(value)


def test_heldout_uncommitted_and_unverified_labels_do_not_enter_selection(tmp_path):
    _, rows = manifest(tmp_path)
    rows['episodes'][0]['split'] = 'test'
    rows['episodes'][1]['committed'] = False
    rows['episodes'][2]['supervision'] = None
    result = data_quality.selection(rows, require_supervision='reset_expert')
    assert result['selected_episode_ids'] == []
    assert sum(result['rejected'].values()) == 3


def training_witness(tmp_path, events):
    repo, output, _, _ = registered(tmp_path)
    _, bindings = data_versions.resolve(output, repo, dict(dataset_root='data-version:extra'))
    b = bindings[0]
    default = dict(data_version_id=b['id'], setting=b['setting'],
                   identity_field=b['identity_field'], identity=b['identity'])
    log = output/'training.log'
    log.write_text('\n'.join(data_versions.MARKER+json.dumps({**default, **event}) for event in events))
    attempt = 'c'*32
    capture_attempt_evidence(output, attempt_id=attempt, log=log, receipt_ref='train.json',
                            status='completed', returncode=0, termination_reason='normal_exit')
    return data_versions.verify_consumption(output, bindings, dict(attempt_id=attempt))


def test_incremental_witness_counts_final_reads_not_sum_of_progress(tmp_path):
    proof = training_witness(tmp_path, [dict(samples_read=i) for i in (1, 2, 7)])
    assert proof['status'] == 'verified'
    assert proof['exposure'][0]['samples_read'] == 7
    assert proof['exposure'][0]['witness_events'] == 3


def test_multi_reader_counter_identity_and_unique_exposure(tmp_path):
    rows = [dict(reader_id=reader, samples_read=count, exposure={'unique_samples': count})
            for reader, count in [('worker:0', 1), ('worker:1', 2), ('worker:0', 3), ('worker:1', 4)]]
    proof = training_witness(tmp_path, rows)
    exposure = proof['exposure'][0]
    assert proof['status'] == 'verified' and exposure['samples_read'] == 7
    assert exposure['reader_streams'] == 2
    assert 'unique_samples' not in exposure  # overlaps across readers are unknown


def test_prefetched_reads_are_not_added_to_actual_training_batches(tmp_path):
    proof = training_witness(tmp_path, [dict(samples_read=100),
        dict(samples_read=8, consumption_stage='training_batch')])
    assert proof['status'] == 'verified'
    assert proof['exposure'][0]['samples_read'] == 8
    assert proof['exposure'][0]['consumption_stage'] == 'training_batch'


@pytest.mark.parametrize('events', [[dict(samples_read=3), dict(samples_read=2)],
    [dict(samples_read=2), dict(samples_read=2)],
    [dict(samples_read=3, reader_id='../host')],
    [dict(samples_read=3, setting='different_root')]])
def test_counter_ambiguity_returns_repairable_data_failure(tmp_path, events):
    proof = training_witness(tmp_path, events)
    assert proof['status'] == 'unverified'
    assert proof['repair_owner'] == 'data'


def test_control_feedback_distinguishes_observed_change_noop_and_unknown(tmp_path):
    _, before = manifest(tmp_path)
    after = copy.deepcopy(before)
    same = data_quality.coverage_change(before, after, target())
    assert same['status'] == 'no_observed_coverage_change'
    after['episodes'][0]['conditions']['pose']['value'] = .1
    changed = data_quality.coverage_change(before, after, target())
    assert changed['status'] == 'observed_coverage_change'
    assert changed['before']['matched'] == 2 and changed['after']['matched'] == 3
    after['episodes'][1]['conditions'] = {}
    missing = data_quality.coverage_change(before, after, target())
    assert missing['status'] == 'needs_native_observation'
    assert missing['after']['unknown'] == 1
