import json
from pathlib import Path

import pytest

from autosim.research import native_jobs, run_successor
from autosim.research.agent_budget import AgentCostLedger
from autosim.research.budget import RunBudget
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.task_budget import TaskGPUBudget


@pytest.fixture
def parent(tmp_path, monkeypatch):
    root = tmp_path / 'pilot'
    root.mkdir()
    atomic_json(root / 'native_jobs/controller_finished.json', {'state': 'finished'})
    atomic_json(root / 'budget_scope.json', {'scope': 'task', 'schema_version': 1})
    TaskGPUBudget(root).initialize(tmp_path / 'repo')
    gpu = TaskGPUBudget(root)
    gpu.reserve('training', 'GPU-1', 60)
    gpu.finish('training', 20)
    RunBudget(root, wall_seconds=5000)
    ledger = AgentCostLedger(root, run_id='derived', limit_usd=60)
    held = ledger.reserve(session_id='missing-receipt', role='scheduler', requested_usd=4)
    assert held
    monkeypatch.setattr(run_successor, 'service_state', lambda unit: {
        'MainPID': '0', 'ActiveState': 'inactive'})
    monkeypatch.setattr(native_jobs, 'active_jobs', lambda root: [])
    return root


def test_successor_preserves_all_accounting_but_no_scientific_history(parent):
    before = {p: digest(parent / p) for p in (
        'budget.json', 'task_gpu_budget.json', 'agent/cost_ledger.json')}
    atomic_json(parent / 'research/derived/measurements/baseline.json', {'metric_value': .5})
    child = parent.parent / 'new_study'
    receipt = run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='authorized restart')
    run_successor.validate_inheritance(child)
    assert read_json(child / 'agent/cost_ledger.json') == read_json(parent / 'agent/cost_ledger.json')
    assert RunBudget.existing(child).started_epoch == RunBudget.existing(parent).started_epoch
    assert TaskGPUBudget(child).snapshot()['settled_gpu_seconds'] == 20
    assert TaskGPUBudget(child).snapshot()['cap_gpu_seconds'] == 86400
    TaskGPUBudget(child).initialize(parent.parent / 'repo')
    assert not (child / 'research').exists()
    assert receipt['formal_paper_eligible'] is False
    assert before == {p: digest(parent / p) for p in before}
    with pytest.raises(ValueError, match='retired'):
        run_successor.assert_not_retired(parent)
    # Parent changes cannot silently produce two spenders from a stale snapshot.
    doc = read_json(parent / 'agent/cost_ledger.json')
    doc['updated_at'] = 'new writer'
    atomic_json(parent / 'agent/cost_ledger.json', doc)
    with pytest.raises(ValueError, match='accounting changed'):
        run_successor.validate_inheritance(child)


def authorized_parent(parent):
    from autosim.research.budget_amendment import extend
    atomic_json(parent / 'agent/runtime_spec.json', {
        'run_id': 'derived', 'workspace_identity': str(parent / 'checkout'),
        'turn_budget_usd': 4, 'total_budget_usd': 60})
    grant = extend(parent, model_usd=0, wall_seconds=0, gpu_seconds=10000,
        operator_gpu_ceiling_seconds=96400, reason='explicit GPU renewal',
        preserve_stopped_holds=True)
    return grant


def test_successor_preserves_renewal_chain_and_can_initialize(parent):
    from autosim.research.budget_amendment import authorizes_gpu_cap, extend
    authorized_parent(parent)
    child = parent.parent / 'renewed_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='continue study')
    assert authorizes_gpu_cap(child, 96400)
    TaskGPUBudget(child).initialize(parent.parent / 'repo')
    run_successor.assert_pristine_bootstrap(child)
    assert read_json(child / 'agent/runtime_spec.json')['workspace_identity'] == str(child / 'checkout')
    # A later child-only wall grant must still verify the inherited GPU renewal.
    extend(child, model_usd=0, wall_seconds=3600, reason='child wall addition',
           preserve_stopped_holds=True)
    assert authorizes_gpu_cap(child, 96400)
    run_successor.validate_inheritance(child)


def test_successor_carries_model_prefix_rewrite_proofs(parent):
    from autosim.research.budget_amendment import authorizes_gpu_cap
    authorized_parent(parent)
    ledger = AgentCostLedger(parent, run_id='derived', limit_usd=60)
    held = read_json(ledger.path)['entries'][0]['reservation_id']
    ledger.settle(held, actual_usd=1, launched=True)
    assert list((parent / 'agent/ledger_history').rglob('*.json'))
    assert authorizes_gpu_cap(parent, 96400)
    child = parent.parent / 'prefix_proof_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='continue')
    assert authorizes_gpu_cap(child, 96400)
    run_successor.assert_pristine_bootstrap(child)
    assert read_json(child / 'agent/cost_ledger.json') == read_json(parent / 'agent/cost_ledger.json')


@pytest.mark.parametrize('fault', ['missing', 'extra', 'changed', 'link'])
def test_pristine_successor_rejects_changed_authorization_proofs(parent, fault):
    grant = authorized_parent(parent)
    child = parent.parent / 'proof_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='continue')
    path = child / 'budget_amendments' / (grant['id'] + '.json')
    if fault == 'missing':
        path.unlink()
    elif fault == 'extra':
        atomic_json(path.parent / 'extra.json', {'not_authorized': True})
    elif fault == 'changed':
        atomic_json(path, {'id': grant['id'], 'status': 'applied'})
    else:
        path.unlink()
        path.symlink_to(parent / 'budget_amendments' / (grant['id'] + '.json'))
    with pytest.raises(ValueError):
        run_successor.assert_pristine_bootstrap(child)


def test_successor_refuses_unauthorized_parent_gpu_cap_before_retiring(parent):
    doc = read_json(parent / 'task_gpu_budget.json')
    doc['cap_seconds'] = 96400
    atomic_json(parent / 'task_gpu_budget.json', doc)
    child = parent.parent / 'invalid_cap'
    with pytest.raises(ValueError, match='authorization'):
        run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='continue')
    assert not child.exists()
    assert not (parent / 'successor_run.json').exists()


def historical_stage(parent):
    from autosim.research.common import object_digest
    from autosim.research.evidence_store import capture_attempt_evidence
    entry = {'source': 'def stage_argv_evaluate(i):\n return [i["python"], "native_eval.py", str(i["episodes"])]\n',
             'parameters': {}, 'row': {'entrypoint': 'native_eval.py',
                 'environment': {'NATIVE_DISPLAY_MODE': 'source_backed_headless'}}}
    atomic_json(parent / 'derived_stages.json', {'evaluate': entry})
    ref = 'derivation_attempts/evaluate.json'
    atomic_json(parent / ref, {'arguments': {'task': 'old_task'}})
    ident = 'b' * 32
    receipt_ref = f'derivation_native/{ident}/receipt.json'
    atomic_json(parent / receipt_ref, {'status': 'completed', 'returncode': 0})
    log = parent / f'derivation_native/{ident}/native.log'
    log.write_text('native command verification completed\n')
    capture_attempt_evidence(parent, attempt_id=ident, log=log, receipt_ref=receipt_ref,
        status='completed', returncode=0, termination_reason='completed')
    body = {'stage': 'evaluate', 'stage_sha256': object_digest(entry),
            'native_evidence_ref': f'evidence/{ident}.json',
            'derivation_ref': ref, 'derivation_sha256': digest(parent / ref),
            'verification_scope': 'completed_stage'}
    interface_id = object_digest(body)
    atomic_json(parent / f'execution_interfaces/evaluate/{interface_id}.json',
                {**body, 'id': interface_id})
    return entry, interface_id


def test_predecessor_reference_preserves_runtime_memory_without_scientific_transfer(parent):
    entry, identity = historical_stage(parent)
    atomic_json(parent / 'research/derived/confirmation.json', {'private_result': 'NEVER_EXPOSE'})
    child = parent.parent / 'new_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='authorized inheritance')
    before = digest(child / 'agent/cost_ledger.json')
    record = run_successor.preparation_reference(child, 'evaluate')
    assert record['status'] == 'historical_reference_requires_new_verification'
    assert record['entry'] == entry and record['interface_id'] == identity
    assert 'NEVER_EXPOSE' not in json.dumps(record)
    assert not (child / 'derived_stages.json').exists()
    assert not (child / 'execution_interfaces').exists()
    assert not (child / 'research').exists()
    assert digest(child / 'agent/cost_ledger.json') == before
    assert run_successor.preparation_reference(child, 'collect')['status'] == 'not_available'


@pytest.mark.parametrize('fault', ['source', 'interface', 'log', 'derivation', 'symlink', 'accounting'])
def test_predecessor_reference_refuses_unbound_or_tampered_history(parent, fault):
    entry, identity = historical_stage(parent)
    child = parent.parent / 'new_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='authorized inheritance')
    if fault == 'source':
        entry['source'] = 'def stage_argv_evaluate(i):\n return ["different"]\n'
        atomic_json(parent / 'derived_stages.json', {'evaluate': entry})
    elif fault == 'interface':
        atomic_json(parent / f'execution_interfaces/evaluate/{identity}.json', {'id': identity})
    elif fault == 'log':
        (parent / f'derivation_native/{"b" * 32}/native.log').write_text('changed')
    elif fault == 'derivation':
        atomic_json(parent / 'derivation_attempts/evaluate.json', {'changed': True})
    elif fault == 'symlink':
        p = parent / 'derived_stages.json'
        p.unlink()
        p.symlink_to(parent / 'budget.json')
    else:
        atomic_json(parent / 'budget.json', {'changed': True})
    record = run_successor.preparation_reference(child, 'evaluate')
    assert record['status'] == 'not_available'
    assert 'entry' not in record


@pytest.mark.parametrize('live', ['controller', 'native', 'gpu'])
def test_successor_refuses_live_ownership(parent, monkeypatch, live):
    if live == 'controller':
        monkeypatch.setattr(run_successor, 'service_state', lambda unit: {'MainPID': '123', 'ActiveState': 'active'})
    elif live == 'native':
        monkeypatch.setattr(native_jobs, 'active_jobs', lambda root: ['one'])
    else:
        TaskGPUBudget(parent).reserve('live', 'GPU-1', 60)
    child = parent.parent / 'new_study'
    with pytest.raises(ValueError):
        run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='authorized restart')
    assert not child.exists()
    assert not (parent / 'successor_run.json').exists()


def test_partial_successor_cannot_launch(tmp_path):
    root = tmp_path / 'child'
    root.mkdir()
    atomic_json(root / 'successor_initialization.json', {'status': 'pending'})
    with pytest.raises(ValueError, match='incomplete'):
        run_successor.validate_inheritance(root)


def test_pristine_successor_can_declare_first_intent_without_resetting_budget(parent):
    from autosim.research.research_brief import persist, view
    child = parent.parent / 'new_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='authorized')
    source = parent.parent / 'brief.md'
    source.write_text('Declare before any evaluator or model activity.')
    before = digest(child / 'agent/cost_ledger.json')
    persist(child, source)
    assert view(child)['status'] == 'frozen'
    assert digest(child / 'agent/cost_ledger.json') == before
    source.write_text('changed')
    with pytest.raises(ValueError, match='frozen'):
        persist(child, source)


@pytest.mark.parametrize('marker', ['run_state.json', 'research', 'native_jobs',
                                  'verification_outputs', 'agent/session', 'runtime_preflight.json'])
def test_successor_with_any_execution_cannot_retrofit_intent(parent, marker):
    from autosim.research.research_brief import persist
    child = parent.parent / 'new_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='authorized')
    (child / marker).mkdir(parents=True)
    source = parent.parent / 'brief.md'
    source.write_text('intent')
    with pytest.raises(ValueError):
        persist(child, source)
    assert not (child / 'research_brief.json').exists()


def test_successor_with_new_model_usage_cannot_retrofit_intent(parent):
    from autosim.research.research_brief import persist
    child = parent.parent / 'new_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='authorized')
    doc = read_json(child / 'agent/cost_ledger.json')
    ledger = AgentCostLedger(child, run_id=doc['run_id'], limit_usd=doc['limit_usd'])
    ledger.reserve(session_id='new-session', role='scheduler', requested_usd=1)
    source = parent.parent / 'brief.md'
    source.write_text('intent')
    with pytest.raises(ValueError, match='accounting changed'):
        persist(child, source)


def test_source_bootstrap_only_tracked_changes_not_data_or_scores(parent):
    child = parent.parent / 'new_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='authorized restart')
    old_code, new_code = parent / 'checkout', child / 'checkout'
    old_code.mkdir(); new_code.mkdir()
    (new_code / 'train.py').write_text('old source\n')
    row = ['file', 'train.py', (new_code / 'train.py').stat().st_size,
           digest(new_code / 'train.py'), 0o644]
    for root in (parent, child):
        atomic_json(root / 'workspace_snapshot.json', {'source': str(parent.parent / 'repo'), 'source_entries': [row]})
    (old_code / 'train.py').write_text('repaired source\n')
    (old_code / 'weights.bin').write_bytes(b'not code')
    result = run_successor.bootstrap_source(parent, child)
    assert (new_code / 'train.py').read_text() == 'repaired source\n'
    assert result['files'][0]['reused_sha256'] == digest(old_code / 'train.py')
    assert not (new_code / 'weights.bin').exists()


@pytest.mark.parametrize('bad', ['', 'abi', 'hook', 'escape'])
def test_local_dependency_delta_is_bounded_and_abi_matched(parent, bad):
    child = parent.parent / 'new_study'
    run_successor.prepare(parent, child, parent_unit='autosim-test.service', reason='authorized restart')
    base, target = parent / 'environment', child / 'env'
    for prefix in (base, target):
        prefix.mkdir()
        (prefix / 'pyvenv.cfg').write_text('version = 3.10.1\n')
    if bad == 'abi':
        (target / 'pyvenv.cfg').write_text('version = 3.11.1\n')
    site = base / 'lib/python3.10/site-packages'
    info = site / 'delta-1.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text('Name: delta\nVersion: 1\n')
    (site / 'delta.py').write_text('VALUE = 1\n')
    records = 'delta.py,,\ndelta-1.dist-info/METADATA,,\ndelta-1.dist-info/RECORD,,\n'
    if bad == 'hook':
        (site / 'bad.pth').write_text('import unsafe\n')
        records += 'bad.pth,,\n'
    if bad == 'escape':
        records += '../../../../outside,,\n'
    (info / 'RECORD').write_text(records)
    if bad:
        with pytest.raises(ValueError):
            run_successor.bootstrap_dependency_delta(base, target, output=child)
        assert not (target / 'lib/python3.10/site-packages/delta.py').exists()
    else:
        result = run_successor.bootstrap_dependency_delta(base, target, output=child)
        assert (target / 'lib/python3.10/site-packages/delta.py').read_text() == 'VALUE = 1\n'
        assert result['distributions'] == [{'name': 'delta', 'version': '1'}]
