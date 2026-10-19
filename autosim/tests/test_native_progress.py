import hashlib
from pathlib import Path

import pytest

from autosim.research.common import atomic_json, read_json
from autosim.research.evidence_store import read_attempt_evidence
from autosim.research.native_progress import capture


JOB, ATTEMPT = 'a'*32, 'b'*32


def fixture(tmp_path, monkeypatch, stage='collect'):
    root = tmp_path/'run'; root.mkdir()
    request = root/f'native_jobs/{JOB}/request.json'
    atomic_json(request, {'job_id': JOB, 'run_id': 'derived', 'stage': stage,
        'resources': {'cpu': 2, 'memory_mib': 4096, 'gpu': True}})
    receipt = root/f'research/derived/attempts/{ATTEMPT}/receipt.json'
    atomic_json(receipt, {'attempt_id': ATTEMPT, 'run_id': 'derived',
        'node_id': stage, 'controller_decision_id': JOB, 'status': 'running'})
    log = root/f'research/derived/{stage}/attempts/{ATTEMPT}/output.log'
    log.parent.mkdir(parents=True)
    log.write_bytes(b'frames 10/s\nacquired 7 of 20\nframes 20/s\nacquired 8 of 20\n')
    view = {'job_id': JOB, 'stage': stage, 'status': 'running',
        'worker': 'matching_running', 'progress': {'attempt_id': ATTEMPT}}
    monkeypatch.setattr('autosim.research.native_jobs.status', lambda *a: view)
    return root, request, receipt, log, view


@pytest.mark.parametrize('stage', ['train', 'collect', 'prepare_data'])
def test_agent_selects_literal_projection_not_executor_counter(tmp_path, monkeypatch, stage):
    root, request, receipt, log, view = fixture(tmp_path, monkeypatch, stage)
    before = (request.read_bytes(), receipt.read_bytes(), log.read_bytes())
    result = capture(root, JOB, query='acquired')
    assert result['text'] == 'acquired 7 of 20\nacquired 8 of 20'
    assert result['matching_lines_in_bounded_slice'] == 2
    assert result['requested_resources']['cpu'] == 2
    assert result['status'] == 'observed' and result['native_status_at_end'] == 'running'
    proof = read_attempt_evidence(root, result['observation_id'])
    assert proof['status'] == 'observed' and proof['returncode'] is None
    assert proof['artifact'] == {} and proof['metric_artifact'] == {}
    assert result['source_slice_sha256'] == result['log_sha256']
    assert before == (request.read_bytes(), receipt.read_bytes(), log.read_bytes())


def test_snapshot_remains_readable_after_producer_log_changes(tmp_path, monkeypatch):
    root, _, _, log, _ = fixture(tmp_path, monkeypatch)
    result = capture(root, JOB, query='acquired')
    log.write_bytes(b'later content\n')
    proof = read_attempt_evidence(root, result['observation_id'])
    assert 'acquired 8 of 20' in proof['text']
    assert 'later content' not in proof['text']


def test_raw_non_utf8_bytes_are_sealed_without_reencoding(tmp_path, monkeypatch):
    root, _, _, log, _ = fixture(tmp_path, monkeypatch)
    raw = b'work\xff\n'; log.write_bytes(raw)
    result = capture(root, JOB)
    assert result['log_sha256'] == hashlib.sha256(raw).hexdigest()
    assert '\ufffd' in read_attempt_evidence(root, result['observation_id'])['text']


def test_bounded_tail_omits_partial_lines_and_marks_projection_limits(tmp_path, monkeypatch):
    root, _, _, log, _ = fixture(tmp_path, monkeypatch)
    log.write_bytes(b'prefix\nacquired 7\nacquired 8\npartial acquired 9')
    result = capture(root, JOB, query='acquired', tail_bytes=35, limit=5)
    assert result['matching_lines_in_bounded_slice'] == 1
    assert result['text'] == 'red 8' and result['projection_truncated'] is True
    assert result['source_byte_range'][1] == log.stat().st_size


@pytest.mark.parametrize('stage', ['evaluate', 'confirmation', 'custom_heldout'])
def test_evaluation_or_unknown_stage_never_exposes_snapshot(tmp_path, monkeypatch, stage):
    root, *_ = fixture(tmp_path, monkeypatch, stage)
    with pytest.raises(ValueError, match='train/collect/prepare_data'):
        capture(root, JOB)
    assert not (root/'native_job_progress_snapshots').exists()


@pytest.mark.parametrize('fault', ['foreign_job', 'wrong_stage', 'finished', 'unknown_worker',
    'log_symlink', 'parent_symlink', 'unsafe_run'])
def test_progress_requires_exact_owned_running_receipt(tmp_path, monkeypatch, fault):
    root, request, receipt, log, view = fixture(tmp_path, monkeypatch)
    row = read_json(receipt)
    if fault == 'foreign_job': row['controller_decision_id'] = 'c'*32
    if fault == 'wrong_stage': row['node_id'] = 'evaluate'
    if fault == 'finished': row['status'] = 'completed'
    if fault == 'unknown_worker': view['worker'] = 'unverifiable'
    if fault == 'unsafe_run':
        spec = read_json(request); spec['run_id'] = '..'; atomic_json(request, spec)
    if fault == 'log_symlink':
        outside = tmp_path/'outside.log'; outside.write_bytes(b'private\n')
        log.unlink(); log.symlink_to(outside)
    if fault == 'parent_symlink':
        other = root/'renamed'; log.parent.rename(other); log.parent.symlink_to(other)
    atomic_json(receipt, row)
    with pytest.raises(ValueError): capture(root, JOB)
    assert not (root/'native_job_progress_snapshots').exists()


@pytest.mark.parametrize('arguments', [{'tail_bytes': True}, {'limit': 0},
    {'tail_bytes': 262145}, {'query': 'x'*513}])
def test_progress_reads_have_explicit_io_and_response_bounds(tmp_path, monkeypatch, arguments):
    root, *_ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError): capture(root, JOB, **arguments)


def test_literal_query_is_not_executed_as_regex(tmp_path, monkeypatch):
    root, *_ = fixture(tmp_path, monkeypatch)
    result = capture(root, JOB, query='.*')
    assert result['matching_lines_in_bounded_slice'] == 0 and result['text'] == ''


def test_evidence_destination_cannot_escape_via_parent_link(tmp_path, monkeypatch):
    root, *_ = fixture(tmp_path, monkeypatch)
    outside = tmp_path/'other'; outside.mkdir()
    (root/'evidence').symlink_to(outside)
    with pytest.raises(ValueError): capture(root, JOB)
    assert list(outside.iterdir()) == []
    assert not (root/'native_job_progress_snapshots').exists()


def test_tail_starting_on_line_boundary_keeps_first_complete_line(tmp_path, monkeypatch):
    root, _, _, log, _ = fixture(tmp_path, monkeypatch)
    log.write_bytes(b'prefix\nacquired 8\n')
    result = capture(root, JOB, query='acquired', tail_bytes=len(b'acquired 8\n'))
    assert result['text'] == 'acquired 8'


@pytest.mark.parametrize('process_stays_owned', [True, False])
def test_actual_affinity_requires_native_process_identity_before_and_after(
        tmp_path, monkeypatch, process_stays_owned):
    root, _, receipt, _, _ = fixture(tmp_path, monkeypatch)
    row = read_json(receipt)
    row['process_identity'] = {'run_id': 'derived', 'attempt_id': ATTEMPT, 'pid': 12345}
    atomic_json(receipt, row)
    checks = iter(['matching_running', 'matching_running' if process_stays_owned else 'not_running'])
    monkeypatch.setattr('autosim.research.process_executor.inspect_process_identity',
                        lambda _: {'status': next(checks)})
    monkeypatch.setattr('autosim.research.native_progress.os.sched_getaffinity', lambda _: {4, 5})
    result = capture(root, JOB)
    if process_stays_owned:
        assert result['native_cpu_affinity']['logical_cpu_count'] == 2
        assert result['native_cpu_affinity']['cpu_ids'] == [4, 5]
    else:
        assert result['native_cpu_affinity']['status'] == 'unknown'
        assert 'logical_cpu_count' not in result['native_cpu_affinity']


def test_inspect_operation_exposes_agent_chosen_query_and_snapshot_id(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    root, _, receipt, _, view = fixture(tmp_path, monkeypatch)
    view['job_ref'] = f'native_jobs/{JOB}/request.json'
    before = receipt.read_bytes()
    prep = Preparation(repo=tmp_path, output=root, client=None, scouting=root/'scouting')
    result = prep._step_inspect_native_job(job_id=JOB, log_query='acquired')
    observed = result['progress_observation']
    assert result['outcome'] == 'observed'
    assert observed['text'] == 'acquired 7 of 20\nacquired 8 of 20'
    assert observed['evidence_ref'] in result['evidence_refs']
    assert read_attempt_evidence(root, observed['observation_id'])['status'] == 'observed'
    assert receipt.read_bytes() == before


@pytest.mark.parametrize('capture_fails', [False, True])
def test_resource_event_attaches_snapshot_without_model_or_native_mutation(
        tmp_path, monkeypatch, capture_fails):
    import time
    from autosim.research.prepare import Preparation
    root, _, receipt, _, view = fixture(tmp_path, monkeypatch)
    prep = Preparation(repo=tmp_path, output=root, client=None, scouting=root/'scouting')
    before = receipt.read_bytes()
    jobs = [{**view, 'window_seconds': 1000, 'deadline_epoch': time.time()+400,
             'job_ref': f'native_jobs/{JOB}/request.json'}]
    monkeypatch.setattr(prep, '_native_job_view', lambda: jobs)
    if capture_fails:
        def unavailable(*a, **k): raise ValueError('source observation unavailable')
        monkeypatch.setattr('autosim.research.native_progress.capture', unavailable)
    result = prep._step_wait_for_jobs()
    assert result['outcome'] == 'event'
    observed = result['resource_review']['progress_observation']
    if capture_fails:
        assert observed['status'] == 'unavailable'
        assert 'observation_id' not in observed
    else:
        assert observed['evidence_ref'] in result['evidence_refs']
        assert read_attempt_evidence(root, observed['observation_id'])['status'] == 'observed'
    assert receipt.read_bytes() == before
