"""Pre-native recovery is established by owned receipts, not exception names."""
from copy import deepcopy

import pytest

from autosim.research.common import atomic_json, object_digest, read_json
from autosim.research.prelaunch_recovery import inspect, reconcile
from tests.test_prelaunch_contract_recovery import legacy_session


def failed_candidate(tmp_path, *, patched, reason):
    real, session, steps = legacy_session(tmp_path)
    pending = session['pending_action']
    pending.update(phase='preparing_idea', source_transaction_protocol=1)
    pending['idea']['granularity'] = 'code' if patched else 'param'
    atomic_json(real.run_root / 'controller_session.json', session)
    if patched:
        atomic_json(real.run_root / 'code_changes/round-1.json', {
            'round': 1, 'idea_label': pending['idea']['label'], 'status': 'applied'})
    receipt_path = real.output / steps[0]['receipt_ref']
    receipt = read_json(receipt_path)
    receipt['reason'] = reason
    receipt['receipt_sha256'] = object_digest({
        k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    atomic_json(receipt_path, receipt)
    steps[0]['receipt_sha256'] = receipt['receipt_sha256']
    return real, session, steps, receipt


@pytest.mark.parametrize('patched', [False, True])
@pytest.mark.parametrize('reason', [
    'OSError: failed to commit candidate preparation state',
    'ValueError: exact consumer review cannot approve this input',
    'AgentRuntimeClientError: provider unavailable during independent review',
    'RuntimeError: candidate snapshot preparation failed',
])
def test_owned_pre_native_failures_are_reconcilable_regardless_of_exception_type(
        tmp_path, patched, reason):
    real, session, steps, _ = failed_candidate(tmp_path, patched=patched, reason=reason)
    old_history = deepcopy(session['history'])
    native_before = sorted((real.run_root / 'attempts').glob('*/receipt.json'))
    transaction = real.run_root / 'code_changes/round-1.json'
    transaction_before = transaction.read_bytes() if patched else None
    evidence = inspect(real.output, real.repo, real.run_id, steps)
    assert evidence and evidence['reason'] == reason
    result = reconcile(real, steps)
    assert result['outcome'] == 'reconciled' and result['native_replayed'] is False
    after = read_json(real.run_root / 'controller_session.json')
    assert after['history'] == old_history and after['next_round'] == session['next_round']
    assert sorted((real.run_root / 'attempts').glob('*/receipt.json')) == native_before
    assert after['status'] == ('interrupted' if patched else 'paused')
    if patched:
        assert transaction.read_bytes() == transaction_before
        assert after['pending_action'] == session['pending_action']
    else:
        assert after['pending_action'] is None


@pytest.mark.parametrize('patched', [False, True])
@pytest.mark.parametrize('fault', [
    'wrong_parent_digest', 'unknown_parent', 'wrong_protocol', 'parent_process',
    'unresolved_child', 'native_attempt', 'measurement', 'active_job',
])
def test_exception_text_never_substitutes_for_unlaunched_ownership_proof(
        tmp_path, monkeypatch, patched, fault):
    real, session, steps, receipt = failed_candidate(
        tmp_path, patched=patched, reason='OSError: preparation commit failed')
    receipt_path = real.output / steps[0]['receipt_ref']
    if fault == 'wrong_parent_digest':
        steps[0]['receipt_sha256'] = '0' * 64
    if fault == 'unknown_parent':
        steps[0]['outcome'] = 'running'
    if fault == 'wrong_protocol':
        session['pending_action']['source_transaction_protocol'] = 0
        atomic_json(real.run_root / 'controller_session.json', session)
    if fault == 'parent_process':
        receipt['process_identity'] = {'pid': 123}
    if fault == 'unresolved_child':
        receipt['child_attempt_ids'] = ['e' * 32]
    if fault in {'parent_process', 'unresolved_child'}:
        receipt['receipt_sha256'] = object_digest({
            k: v for k, v in receipt.items() if k != 'receipt_sha256'})
        atomic_json(receipt_path, receipt)
        steps[0]['receipt_sha256'] = receipt['receipt_sha256']
    if fault == 'native_attempt':
        atomic_json(real.run_root / 'attempts' / ('c' * 32) / 'receipt.json', {
            'started_at': session['pending_action']['updated_at']})
    if fault == 'measurement':
        atomic_json(real.run_root / 'measurements/round_1.json', {'ok': False})
    if fault == 'active_job':
        monkeypatch.setattr('autosim.research.native_jobs.active_jobs', lambda _: ['unknown'])
    before = (real.run_root / 'controller_session.json').read_bytes()
    assert not inspect(real.output, real.repo, real.run_id, steps)
    assert reconcile(real, steps)['outcome'] == 'blocked'
    assert (real.run_root / 'controller_session.json').read_bytes() == before


@pytest.mark.parametrize('phase', ['manifest_prepare', 'manifest_commit', 'snapshot', 'consumer'])
def test_real_candidate_preparation_fault_returns_to_owned_reconciliation(
        tmp_path, monkeypatch, phase):
    from types import SimpleNamespace
    from autosim.research import derived_research
    from autosim.research.action_receipt import ActionReceipt, write_action_receipt
    from autosim.research.common import now
    from autosim.research.ideas import Idea
    from autosim.research.prepare import Preparation
    from tests.test_derived_research import research

    real = research(tmp_path)
    monkeypatch.setattr(real, '_summarize', lambda: None)
    monkeypatch.setattr(real, 'capture_demo', lambda **kw: {})
    real.run(rounds=2, yield_after_action=True)
    loader = real.repo / 'loader.py'
    original_source = 'def load(root):\n    return reader(root)\n'
    loader.write_text(original_source)
    idea = Idea(label='reader', granularity='code', risk='low', crosses='none',
                change={'file': 'loader.py', 'find': 'reader(root)',
                        'replace': 'correct_reader(root)'})
    real.library.add(idea)
    real.library.audit(real.red_lines())
    baseline = (real.run_root / 'measurements/baseline.json').read_bytes()
    native_before = sorted((real.run_root / 'attempts').glob('*/receipt.json'))
    real_atomic = derived_research.atomic_json
    transaction = real.run_root / 'code_changes/round-1.json'
    if phase in {'manifest_prepare', 'manifest_commit'}:
        wanted = 'applying' if phase == 'manifest_prepare' else 'applied'
        def failing_atomic(path, value):
            if path == transaction and value.get('status') == wanted:
                raise OSError('injected candidate transaction ' + phase)
            return real_atomic(path, value)
        monkeypatch.setattr(derived_research, 'atomic_json', failing_atomic)
    elif phase == 'snapshot':
        real_capture = real.snapshots.capture
        def failing_capture(*args, **kwargs):
            if kwargs.get('name') == 'round-1':
                raise OSError('injected candidate post-patch snapshot failure')
            return real_capture(*args, **kwargs)
        monkeypatch.setattr(real.snapshots, 'capture', failing_capture)
    else:
        def failing_consumer(*args, **kwargs):
            raise ValueError('injected current consumer pre-native failure')
        monkeypatch.setattr('autosim.research.data_consumer_review.resolve_candidate', failing_consumer)
    started = now()
    with pytest.raises((OSError, ValueError)) as failure:
        real.run(rounds=2, yield_after_action=True, selected_idea_label='reader',
                 main_controller_owns_selection=True)
    ref, receipt = write_action_receipt(real.output, ActionReceipt(
        action_id='f' * 16, decision_id='f' * 16, run_id=real.run_id,
        repository=str(real.repo), action='run_the_loop', state_revision=1,
        started_at=started, finished_at=now(), status='raised',
        reason=f'{type(failure.value).__name__}: {failure.value}',
        arguments={'idea_label': 'reader'}))
    steps = [{'step': 'run_the_loop', 'outcome': 'raised',
              'arguments': {'idea_label': 'reader'}, 'receipt_ref': ref,
              'receipt_sha256': receipt['receipt_sha256']}]
    assert inspect(real.output, real.repo, real.run_id, steps)
    assert reconcile(real, steps)['outcome'] == 'reconciled'
    session = read_json(real.run_root / 'controller_session.json')
    assert session['status'] == ('paused' if phase == 'manifest_prepare' else 'interrupted')
    assert (real.run_root / 'measurements/baseline.json').read_bytes() == baseline
    assert sorted((real.run_root / 'attempts').glob('*/receipt.json')) == native_before
    if phase == 'manifest_prepare':
        assert loader.read_text() == original_source and not transaction.exists()
    else:
        # Reconciliation does not edit source. The existing explicit rollback is
        # separately source-verified, even when commit failed after actual writes.
        assert 'correct_reader' in loader.read_text()
        rollback = Preparation._rollback_interrupted_code_change(
            SimpleNamespace(output=real.output, repo=real.repo, run_id=real.run_id),
            round_index=1, idea_label='reader')
        assert rollback['status'] == 'rolled_back'
        assert loader.read_text() == original_source
