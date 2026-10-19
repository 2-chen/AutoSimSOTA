"""Short-loop contracts with real files/receipts, no paid model or simulator calls."""
import json
import time

import pytest

from autosim.research import prompt_library
from autosim.research.common import atomic_json, object_digest, read_json
from autosim.research.evidence_store import read_attempt_evidence
from autosim.research.main_agent import memory_view, validate_plan
from autosim.research.prepare import OPERATIONS
from tests.test_main_agent import PLAN, REPORT, make


def test_short_prompt_keeps_redlines_schema_and_readable_complete_manual(tmp_path):
    short = prompt_library.scheduler_request_prompt(operations='CURRENT', permitted={'stop'},
        operation_names=OPERATIONS, on_demand=True)
    full = prompt_library.load_prompt('scheduler').format(operations='CURRENT')
    assert len(short) < len(full) / 3
    assert short.count('CURRENT') == 1
    for clause in ('state_revision', 'held-out', 'method_review', 'data-study', 'read_evidence'):
        assert clause in short
    catalog = prompt_library.scheduler_guidance(tmp_path, operation_names=OPERATIONS)
    assert catalog == prompt_library.scheduler_guidance(tmp_path, operation_names=OPERATIONS)
    reconstructed = ''
    for row in catalog['pages']:
        chunks, offset = [], 0
        while True:
            page = read_attempt_evidence(tmp_path, row['evidence_id'], offset=offset, limit=1000)
            chunks.append(page['text'])
            offset = page['next_offset']
            if offset == page['log_bytes']:
                break
        reconstructed += ''.join(chunks)
    # Sanitization may obscure paths, but no procedure is selected/removed by keywords.
    assert 'Historical invocations after source edits' in reconstructed
    assert 'Data-centric research campaigns' in reconstructed
    assert 'Decision response contract' in reconstructed
    assert 'read_instruction' in catalog


def test_guidance_cannot_be_overwritten_or_written_through_symlink(tmp_path):
    catalog = prompt_library.scheduler_guidance(tmp_path, operation_names=OPERATIONS)
    identity = catalog['pages'][0]['sha256']
    target = tmp_path / 'agent/guidance' / (identity + '.md')
    target.write_text('changed instructions')
    with pytest.raises(ValueError, match='changed'):
        prompt_library.scheduler_guidance(tmp_path, operation_names=OPERATIONS)
    other = tmp_path / 'other'; other.mkdir()
    unsafe = tmp_path / 'unsafe'; unsafe.mkdir()
    (unsafe / 'agent').symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match='unsafe'):
        prompt_library.scheduler_guidance(unsafe, operation_names=OPERATIONS)
    assert not list(other.iterdir())


def test_decision_memory_survives_restart_and_remains_advisory(tmp_path):
    real = make(tmp_path)
    memory = {'bottleneck': 'Unknown control units', 'observations': ['receipt:A shows clipping'],
        'lessons': ['Normalization was verified under config X'],
        'failed_directions': ['More data failed with old loader; revisit after loader repair'],
        'next_experiment': 'Compare commanded and executed action before another training'}
    plan = {**PLAN, 'decision_memory': memory}
    assert real.do('update_research_plan', plan=plan)['outcome'] == 'recorded'
    restored = make(tmp_path)
    view = memory_view(restored.main_agent)
    assert view['writable_plan']['decision_memory'] == memory
    assert 'not_verified' in view['plan']['decision_memory']['authority']
    from autosim.research.operation_contracts import PLAN as plan_schema, validate
    validate(plan_schema, plan)
    with pytest.raises(ValueError):
        validate_plan({**plan, 'decision_memory': {**memory, 'authority': 'verified'}})
    with pytest.raises(ValueError):
        validate_plan({**plan, 'decision_memory': {**memory, 'lessons': ['x' * 1501]}})


@pytest.mark.parametrize('change', ['none', 'source', 'mode', 'environment', 'binding', 'unknown_worker', 'old_worker'])
def test_noop_edit_restores_only_exact_context_after_verified_worker_exit(tmp_path, monkeypatch, change):
    from autosim.research.common import digest
    real = make(tmp_path)
    source = real.repo / 'policy.py'; source.write_text('value = 1\n')
    entries = [['file', 'policy.py', source.stat().st_size, digest(source), 0o644]]
    snapshot = {'schema_version': 2, 'destination': str(real.repo), 'source_entries': entries,
                'source_tree_fingerprint': object_digest(entries)}
    atomic_json(real.output / 'workspace_snapshot.json', snapshot)
    atomic_json(real.output / 'environment.json', {'interpreter': 'owned-fixture'})
    real.stages = {'train': 'old accepted source'}
    real.parameters = {'train': {}}
    real.verified = {'train': {'entrypoint': 'policy.py'}}
    kept = {'train': {'source': real.stages['train'], 'row': real.verified['train']}}
    atomic_json(real.output / 'derived_stages.json', kept)
    def turn(*args, **kwargs):
        assert not real.stages  # approvals remain suspended while a writer owns the checkout
        if change == 'source': source.write_text('value = 2\n')
        if change == 'mode': source.chmod(0o700)
        if change == 'environment': atomic_json(real.output / 'environment.json', {'interpreter': 'changed'})
        if change == 'binding': atomic_json(real.output / 'workspace_snapshot.json', {**snapshot, 'resource_bindings': [{'name': 'new'}]})
        atomic_json(real.output / 'agent/session.json', {'schema_version': 1,
            'status': 'running' if change == 'unknown_worker' else 'completed',
            'role': 'fix', 'workspace_identity': str(real.repo),
            'started_at': 0 if change == 'old_worker' else time.time(), 'process_identity': {'pid': 1}})
        return json.dumps(REPORT), {}
    monkeypatch.setattr(real.client, 'chat_with_metadata', turn)
    monkeypatch.setattr('autosim.research.prepare.inspect_process_identity', lambda _: {'status': 'not_running'})
    result = real.do('research_task', role='fix', task='Check one native connection', expected_result='Evidence and patch only if needed')
    assert result['outcome'] == 'reported'
    assert bool(real.stages) == (change == 'none')
    assert read_json(real.output / 'derived_stages.json') == (kept if change == 'none' else {})
    assert bool(make(tmp_path).stages) == (change == 'none')


def test_source_citation_refusal_yields_without_stranding_session_or_losing_baseline(tmp_path, monkeypatch):
    from tests.test_derived_research import research
    from autosim.research.ideas import Idea
    from autosim.research.baseline_reference import SourceCitationError
    real = research(tmp_path)
    real.run(rounds=2, yield_after_action=True)
    baseline = (real.run_root / 'measurements/baseline.json').read_bytes()
    real.library.add(Idea(label='candidate', granularity='param', change={'train.n_epochs': 2}, risk='low', crosses='none', why='test'))
    real.library.audit(real.red_lines())
    before = len(list((real.run_root / 'attempts').glob('*/receipt.json')))
    def fail(*args, **kwargs):
        raise SourceCitationError('sealed_source_invalid', index=0, ref='checkout-source:old', detail='refresh actual evidence')
    monkeypatch.setattr(real, '_prepare_idea', fail)
    result = real.run(rounds=2, yield_after_action=True, selected_idea_label='candidate', main_controller_owns_selection=True)
    session = read_json(real.run_root / 'controller_session.json')
    assert session['status'] == result['run_status'] == 'paused'
    assert session['pending_action'] is None and session['next_round'] == 1
    assert 'refresh actual evidence' in read_attempt_evidence(real.output, result['source_preflight_failure']['evidence_id'])['text']
    assert before == len(list((real.run_root / 'attempts').glob('*/receipt.json')))
    assert (real.run_root / 'measurements/baseline.json').read_bytes() == baseline


@pytest.mark.parametrize('fault', [None, 'manifest', 'unknown_protocol', 'native_attempt'])
def test_legacy_pre_patch_reconciliation_requires_positive_unlaunched_boundary(tmp_path, fault):
    from tests.test_prelaunch_contract_recovery import legacy_session
    from autosim.research.prelaunch_recovery import inspect, reconcile
    real, session, steps = legacy_session(tmp_path)
    pending = session['pending_action']
    pending.update(phase='preparing_idea', source_transaction_protocol=1)
    pending.pop('settings')
    if fault == 'unknown_protocol': pending.pop('source_transaction_protocol')
    atomic_json(real.run_root / 'controller_session.json', session)
    receipt_path = real.output / steps[0]['receipt_ref']
    receipt = read_json(receipt_path)
    receipt['reason'] = 'SourceCitationError: stale source quote needs refresh'
    receipt['receipt_sha256'] = object_digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    atomic_json(receipt_path, receipt)
    steps[0]['receipt_sha256'] = receipt['receipt_sha256']
    if fault == 'manifest': atomic_json(real.run_root / 'code_changes/round-1.json', {'status': 'applying'})
    if fault == 'native_attempt': atomic_json(real.run_root / 'attempts' / ('c' * 32) / 'receipt.json', {'started_at': '2026-10-05T09:39:23+00:00'})
    original = (real.run_root / 'controller_session.json').read_bytes()
    assert bool(inspect(real.output, real.repo, real.run_id, steps)) == (fault is None)
    result = reconcile(real, steps)
    assert result['outcome'] == ('reconciled' if fault is None else 'blocked')
    if fault:
        assert (real.run_root / 'controller_session.json').read_bytes() == original
    else:
        after = read_json(real.run_root / 'controller_session.json')
        assert after['status'] == 'paused' and after['history'] == session['history']
        assert after['next_round'] == session['next_round']
        assert 'citation refresh required' in result['prelaunch_reconciliation']['native_revalidation']


def test_recording_failure_preserves_score_and_next_candidate(tmp_path, monkeypatch):
    from tests.test_derived_research import research
    from autosim.research.ideas import Idea
    real = research(tmp_path)
    attempts = []
    def fail(**kwargs):
        attempts.append(kwargs['trigger'])
        raise OSError('renderer unavailable after real metric commit')
    monkeypatch.setattr(real, '_capture_demo', fail)
    report = real.run(rounds=2, yield_after_action=True)
    assert report['run_status'] == 'paused' and attempts
    baseline = read_json(real.run_root / 'measurements/baseline.json')
    assert baseline['ok'] is True
    real.library.add(Idea(label='more', granularity='param', change={'train.n_epochs': 2}, risk='low', crosses='none', why='test continuation'))
    real.library.audit(real.red_lines())
    result = real.run(rounds=2, yield_after_action=True, selected_idea_label='more', main_controller_owns_selection=True)
    assert result['run_status'] == 'paused'
    assert read_json(real.run_root / 'measurements/round_1.json')['ok'] is True
    assert read_json(real.run_root / 'measurements/baseline.json') == baseline


def test_data_boundary_uses_existing_budget_owner_not_removed_main_export():
    import inspect
    from autosim.research.prepare import Preparation
    code = inspect.getsource(Preparation._step_record_data_campaign_arm)
    assert 'from .runtime_freshness import current_model_budget' in code
    assert 'from .main_agent import current_model_budget' not in code


def test_boundary_handoff_reaches_budget_and_independent_review(tmp_path, monkeypatch):
    from tests.test_prepare import preparation
    real = preparation(tmp_path)
    source = real.repo / 'loader.py'; source.write_text('training_only = True\n')
    real.client.choices = [{'boundary_evidence_supported': True,
        'no_safe_next_data_operation': True, 'why': 'fixture review only',
        'citations': [{'source_id': 'review-source-1'}]}]
    monkeypatch.setattr(real, '_available_operations', lambda: ['stop'])
    monkeypatch.setattr(real, '_task_gpu_budget_view', lambda: {'available_gpu_seconds': 10})
    calls = []
    def budget(*args):
        calls.append(args)
        return {'status': 'verified_current', 'remaining_usd': 1}
    monkeypatch.setattr('autosim.research.runtime_freshness.current_model_budget', budget)
    monkeypatch.setattr('autosim.research.data_campaign.record_boundary', lambda *a, **kw: {'metric_value': None})
    result = real._step_record_data_campaign_arm(campaign_id='test', arm='agent_targeted', seed=1,
        boundary={'source_refs': [{'ref': 'loader.py', 'quote': 'training_only = True'}], 'evidence_refs': []})
    assert result['outcome'] == 'recorded' and result['metric_value'] is None
    assert calls == [(real.output, real.run_id)]
    packet = json.loads(real.client.seen[-1])
    contracts = packet['controller_runtime']['current_admission_contracts']
    assert 'selected_training_data' in contracts['training_selection_transport']['rule']
    assert packet['controller_runtime']['controller_runtime_id']


def test_public_input_contract_survives_actual_main_privacy_and_paging(tmp_path):
    from autosim.research.common import sanitize_model_payload
    from autosim.research.token_context import project_request
    real = make(tmp_path)
    facts = real._decision_facts()
    facts['private_resource_probe'] = {'dataset':'/home/private/data.hdf5'}
    projected = sanitize_model_payload(facts, local_roots=(real.repo, real.output))
    request, _ = project_request(json.dumps(projected),real.output,pageable=True)
    shown = json.loads(request)
    contract = shown['executor_input_contracts']['dataset_selection']
    assert 'selected_training_data' in contract['selection_metadata']
    assert contract['candidate_setting'] == 'dataset'
    assert contract['value_format'] == 'data-version:<registered_id>'
    assert '/home/private' not in request
    assert shown['private_resource_probe']['dataset'] == '[LOCAL_RESOURCE_IDENTITY_OMITTED]'


@pytest.mark.parametrize('fault', [None, 'active_native', 'native_during_parent', 'receipt', 'measurement'])
def test_applied_patch_recovery_preserves_later_independent_work(tmp_path, monkeypatch, fault):
    from tests.test_prelaunch_contract_recovery import legacy_session
    from autosim.research.prelaunch_recovery import inspect, reconcile
    real, session, steps = legacy_session(tmp_path)
    pending = session['pending_action']
    pending.update(phase='preparing_idea', source_transaction_protocol=1)
    pending['idea']['granularity'] = 'code'
    atomic_json(real.run_root / 'controller_session.json', session)
    patch_path = real.run_root / 'code_changes/round-1.json'
    transaction = {'round': 1, 'idea_label': 'workers', 'status': 'applied'}
    atomic_json(patch_path, transaction)
    receipt_path = real.output / steps[0]['receipt_ref']
    receipt = read_json(receipt_path)
    receipt['reason'] = 'SourceCitationError: stale loader citation after patch'
    receipt['receipt_sha256'] = object_digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    atomic_json(receipt_path, receipt)
    steps[0]['receipt_sha256'] = receipt['receipt_sha256']
    if fault == 'receipt': steps[0]['receipt_sha256'] = '0' * 64
    later = real.run_root / 'attempts' / ('d' * 32) / 'receipt.json'
    atomic_json(later, {'started_at': ('2026-10-05T09:39:23+00:00' if fault == 'native_during_parent'
                                     else '2026-10-05T10:00:00+00:00'), 'status': 'completed'})
    if fault == 'active_native': monkeypatch.setattr('autosim.research.native_jobs.active_jobs', lambda _: ['unresolved'])
    if fault == 'measurement': atomic_json(real.run_root / 'measurements/round_1.json', {'ok': True})
    before = (patch_path.read_bytes(), later.read_bytes())
    found = inspect(real.output, real.repo, real.run_id, steps)
    assert bool(found) == (fault is None)
    if fault is None:
        assert found['status'] == 'failed_candidate_checkpoint_reconcilable'
        recovery = reconcile(real, steps)
        assert recovery['outcome'] == 'reconciled' and recovery['resume_scope'] == 'preparation'
        after = read_json(real.run_root / 'controller_session.json')
        assert after['status'] == 'interrupted' and after['pending_action'] == pending
        assert after['history'] == session['history']
        assert before == (patch_path.read_bytes(), later.read_bytes())
        from autosim.research.prepare import Preparation
        assert Preparation._interrupted_candidate_discard_available({
            'status': after['status'], 'interruption': after['interruption'],
            'reconciliation': after['reconciliation'],
            'interruption_boundary': after['interruption']['boundary']})


def test_optional_narration_failure_does_not_change_native_measurement(tmp_path, monkeypatch):
    from tests.test_derived_research import research
    real = research(tmp_path)
    real.run(rounds=2, yield_after_action=True)
    path = real.run_root / 'measurements/baseline.json'
    before = path.read_bytes()
    def fail(*args, **kwargs): raise OSError('report write failed')
    monkeypatch.setattr('autosim.research.derived_research.run_record.generate', fail)
    real._summarize()
    assert path.read_bytes() == before


def test_cleanup_waits_for_asynchronous_kill_without_ignoring_real_workers(monkeypatch):
    from types import SimpleNamespace
    from autosim.research import process_executor as executor
    sequence = iter([[123], [123], []])
    monkeypatch.setattr(executor, '_live_group_members', lambda _: next(sequence))
    monkeypatch.setattr(executor, 'terminate_group', lambda _: None)
    orphaned, audit = executor._audit_group_cleanup(SimpleNamespace(pid=123), 'process_group_only')
    assert orphaned and audit['status'] == 'terminated_orphans'
    assert audit['remaining_pids'] == []
    monkeypatch.setattr(executor, '_live_group_members', lambda _: [123])
    orphaned, audit = executor._audit_group_cleanup(SimpleNamespace(pid=123), 'process_group_only')
    assert orphaned and audit['status'] == 'remaining' and audit['remaining_pids'] == [123]


def test_reconciliation_yields_back_to_main_without_an_operator_restart(tmp_path, monkeypatch):
    from types import SimpleNamespace
    real = make(tmp_path)
    real.budget = SimpleNamespace(remaining=lambda: 1000)
    monkeypatch.setattr(real, '_supervision_fingerprint', lambda: 'fixture-state')
    monkeypatch.setattr(real, '_refresh_document', lambda **kw: None)
    monkeypatch.setattr(real, 'state', lambda: {})
    calls = []
    def cycle(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            real.steps.append({'step': 'reconcile_interrupted_action', 'outcome': 'reconciled',
                               'resume_scope': 'preparation'})
            return {'status': 'paused'}
        real.steps.append({'step': 'stop', 'outcome': 'stopped'})
        return {'status': 'completed'}
    monkeypatch.setattr(real, '_run_cycle', cycle)
    result = real.run(max_steps=1, max_relaunch=1)
    assert len(calls) == 2 and result['status'] == 'completed'
    assert [r['step'] for r in real.steps] == ['reconcile_interrupted_action', 'stop']
