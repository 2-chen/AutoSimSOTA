from autosim.research.common import atomic_json
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.monitor_findings import view


def finding(root, identity='a'*32, native='b'*32):
    folder = root/'monitor_findings'
    folder.mkdir(exist_ok=True)
    log = folder/f'{identity}.log'
    log.write_text('operator-only synthetic boundary reproduction')
    ref = f'monitor_findings/{identity}.json'
    sealed = capture_attempt_evidence(root, attempt_id=identity, log=log,
        receipt_ref=ref, status='operator_diagnostic', returncode=0,
        termination_reason='reproduction', artifact={'native_attempt_id':native})
    atomic_json(root/'research/derived/attempts'/native/'receipt.json',
                {'run_id':'derived','attempt_id':native})
    record = {'schema_version':1,'run_id':'derived','origin':'operator_diagnostic',
              'native_attempt_id':native, 'summary':'inspect before retry', **sealed}
    atomic_json(root/ref, record)
    return folder/f'{identity}.json', log, record


def test_separate_operator_reproduction_is_bounded_advisory(tmp_path):
    finding(tmp_path)
    rows = view(tmp_path, 'derived')
    assert len(rows) == 1
    assert rows[0]['origin'] == 'operator_diagnostic'
    assert 'not native outcome' in rows[0]['authority']
    assert rows[0]['excerpt'] == 'operator-only synthetic boundary reproduction'
    assert view(tmp_path, 'other-run') == []


def test_changed_log_and_wrong_native_binding_are_not_offered(tmp_path):
    path, log, record = finding(tmp_path)
    record['native_attempt_id'] = 'c'*32
    atomic_json(tmp_path/'research/derived/attempts'/('c'*32)/'receipt.json',
                {'run_id':'derived','attempt_id':'c'*32})
    atomic_json(path, record)
    assert view(tmp_path, 'derived') == []
    record['native_attempt_id'] = 'b'*32
    atomic_json(path, record)
    log.write_text('tampered')
    assert view(tmp_path, 'derived') == []


def test_unsafe_and_malformed_findings_do_not_break_main_state(tmp_path):
    path, log, record = finding(tmp_path)
    path.unlink()
    path.symlink_to(log)
    assert view(tmp_path, 'derived') == []
    path.unlink()
    atomic_json(path, ['not-a-record'])
    assert view(tmp_path, 'derived') == []
    atomic_json(path, {**record,'native_attempt_id':'../outside'})
    assert view(tmp_path, 'derived') == []
    path.unlink()
    directory=path.parent
    directory.rename(tmp_path/'hidden')
    directory.symlink_to(tmp_path/'hidden', target_is_directory=True)
    assert view(tmp_path, 'derived') == []


def test_latest_diagnostics_use_time_not_random_uuid_order(tmp_path):
    import os
    for index in range(9):
        identity = f'{15-index:x}'*32
        path, _, _ = finding(tmp_path, identity=identity)
        os.utime(path, ns=(1000+index, 1000+index))
    assert [r['evidence_id'] for r in view(tmp_path, 'derived')] == ['9'*32,'8'*32,'7'*32]


def test_complete_native_receipt_can_include_large_execution_context(tmp_path):
    finding(tmp_path)
    receipt = tmp_path/'research/derived/attempts'/('b'*32)/'receipt.json'
    atomic_json(receipt, {'run_id':'derived', 'attempt_id':'b'*32,
                         'native_context':'recorded context ' * 16000})
    assert len(view(tmp_path, 'derived')) == 1
    atomic_json(receipt, {'run_id':'derived', 'attempt_id':'b'*32,
                         'native_context':'x' * (8*1024*1024)})
    assert view(tmp_path, 'derived') == []


def test_internal_native_receipt_ancestor_alias_is_not_an_actual_attempt_binding(tmp_path):
    finding(tmp_path)
    attempt = tmp_path/'research/derived/attempts'/('b'*32)
    relocated = tmp_path/'relocated_native_receipt'
    attempt.rename(relocated)
    attempt.symlink_to(relocated, target_is_directory=True)
    assert view(tmp_path, 'derived') == []


def test_diagnostic_survives_native_completion_without_becoming_a_measurement(tmp_path):
    finding(tmp_path)
    receipt = tmp_path/'research/derived/attempts'/('b'*32)/'receipt.json'
    for status in ('running', 'completed'):
        atomic_json(receipt, {'run_id':'derived', 'attempt_id':'b'*32, 'status':status})
        rows = view(tmp_path, 'derived')
        assert len(rows) == 1
        assert rows[0]['origin'] == 'operator_diagnostic'
        assert 'not native outcome, score or permission' in rows[0]['authority']
