import json
import pytest

from autosim.research.common import atomic_json, digest
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research import confirmation_execution_recovery as recovery


def study(tmp_path):
    output = tmp_path / 'run'; root = output / 'research/derived'
    root.mkdir(parents=True)
    identity = 'a' * 32
    log = output / 'native.log'
    log.write_text('PRIVATE_SEED=998877\nTraceback (most recent call last):\n'
                   '  File "setup.py", line 7, in initialize\n'
                   'TypeError: secret success_rate=0.99\n')
    receipt = output / 'native.json'
    atomic_json(receipt, {'attempt_id': identity, 'ran': True, 'returncode': 1})
    capture_attempt_evidence(output, attempt_id=identity, log=log, receipt_ref='native.json',
                            status='failed', returncode=1, termination_reason='nonzero_exit')
    row = {'ok': False, 'candidate_label': 'round_2', 'settings': {'seed': 998877},
           'metric_value': None, 'baseline_metric_value': None,
           'baseline_evaluate': {'attempt_id': identity}, 'attempt_id': None,
           'held_out_exposure': 'exposed'}
    atomic_json(root / 'confirmation.json', row)
    atomic_json(root / 'measurements/confirmation.json', row)
    atomic_json(root / 'confirmation_attempts/a.json', row)
    atomic_json(root / 'comparison_protocol.json', {'frozen': True})
    atomic_json(root / 'protocol_frozen.json', {'hashes': {}})
    probe = 'b' * 32
    atomic_json(output / 'cpu.json', {'gpu_access': False, 'returncode': 0,
        'repository_source_stable': True})
    (output / 'cpu.log').write_text('original import/resource connection verified on CPU')
    capture_attempt_evidence(output, attempt_id=probe, log=output / 'cpu.log',
        receipt_ref='cpu.json', status='completed', returncode=0,
        termination_reason='cpu_native_diagnostic')
    return output, root, probe


def positive(packet):
    assert packet['native_failures'] and packet['repair_probe']
    return {'verdict': 'setup_only', 'setup_only_positive_evidence': True,
            'no_rollout_or_metric_exposure': True,
            'repair_probe_covers_original_failure': True,
            'frozen_policy_protocol_preserved': True,
            'positive_evidence': 'source control flow and sealed CPU reproduction establish initialization failure',
            'source_refs': [{'ref': 'setup.py:7', 'quote': 'initialize()'}]}


def test_independent_review_is_additive_and_bound_to_every_receipt(tmp_path):
    output, root, probe = study(tmp_path)
    paths = list(output.rglob('*.json'))
    before = {p: digest(p) for p in paths}
    assert recovery.approved(root, output) is False
    assert recovery.review(root, output, diagnostic_evidence_id=probe, reviewer=positive)['approved']
    assert recovery.approved(root, output)
    assert before == {p: digest(p) for p in paths}
    row = json.loads((root / 'confirmation.json').read_text())
    atomic_json(root / 'confirmation_attempts/new.json', row)
    assert recovery.approved(root, output) is False


def test_fix_packet_preserves_trace_locations_without_metrics_or_reserved_inputs(tmp_path):
    output, root, _ = study(tmp_path)
    safe = recovery.engineering_summary(root, output)
    text = json.dumps(safe)
    assert '998877' not in text and '0.99' not in text and 'secret' not in text
    assert safe['exception_types'] == ['TypeError']
    frame = safe['traceback_frames'][0]
    assert frame['file'] == 'setup.py' and frame['line'] == 7 and frame['function'] == 'initialize'
    assert frame['source_ref'] == 'setup.py:7'


def test_cpp_assertion_is_engineering_evidence_without_exposing_native_addresses(tmp_path):
    output, root, _ = study(tmp_path)
    path = output / 'native.log'
    path.write_text('Assertion at : /sdk/core/NativeString.h:94 :\nassert failed!\n'
                    ' #0 :sdk::Material::Create(char const*) at 0x123456\n')
    evidence = json.loads((output / 'evidence' / ('a' * 32 + '.json')).read_text())
    evidence.update(log_bytes=path.stat().st_size, log_sha256=digest(path))
    atomic_json(output / 'evidence' / ('a' * 32 + '.json'), evidence)
    safe = recovery.engineering_summary(root, output)
    assert safe['exception_types'] == ['NativeAssertion']
    assert safe['traceback_frames'][0]['function'] == 'sdk::Material::Create'
    assert '0x123456' not in json.dumps(safe)


@pytest.mark.parametrize('fault', ['gpu', 'unstable', 'log_change', 'scored', 'successful_native'])
def test_launch_or_missing_score_never_grants_retry(tmp_path, fault):
    output, root, probe = study(tmp_path)
    if fault in ('gpu', 'unstable'):
        path = output / 'cpu.json'; row = json.loads(path.read_text())
        row['gpu_access' if fault == 'gpu' else 'repository_source_stable'] = fault == 'gpu'
        atomic_json(path, row)
    elif fault == 'log_change':
        (output / 'native.log').write_text('different failure')
    elif fault == 'scored':
        row = json.loads((root / 'confirmation_attempts/a.json').read_text())
        row['metric_value'] = .8
        atomic_json(root / 'confirmation_attempts/a.json', row)
    else:
        atomic_json(output / 'native.json', {'attempt_id': 'a' * 32, 'ran': True, 'returncode': 0})
    with pytest.raises(ValueError):
        recovery.review(root, output, diagnostic_evidence_id=probe, reviewer=positive)
    assert not recovery.approved(root, output)


def test_supervisor_abstention_keeps_gate_closed(tmp_path):
    output, root, probe = study(tmp_path)
    row = recovery.review(root, output, diagnostic_evidence_id=probe,
                          reviewer=lambda _: {'verdict': 'unknown'})
    assert not row['approved'] and not recovery.approved(root, output)


def test_missing_metric_descriptor_is_not_confused_with_produced_metrics(tmp_path):
    output, root, _ = study(tmp_path)
    path = output / 'evidence' / ('a' * 32 + '.json')
    value = json.loads(path.read_text())
    value['metric_artifact'] = {'status': 'missing', 'matched': 0, 'candidate_paths': []}
    atomic_json(path, value)
    assert recovery.engineering_summary(root, output)['exception_types'] == ['TypeError']
    assert not recovery.approved(root, output)
    value['metric_artifact'] = {'status': 'matched', 'matched': 1, 'candidate_paths': ['score.json']}
    atomic_json(path, value)
    with pytest.raises(ValueError):
        recovery.packet(root, output)
    # Engineering diagnosis remains possible after exposure, without exposing scores.
    assert recovery.engineering_summary(root, output)['exception_types'] == ['TypeError']


def test_protected_source_change_prevents_review_and_invalidates_approval(tmp_path):
    output, root, probe = study(tmp_path)
    source = output / 'checkout/eval.py'; source.parent.mkdir()
    source.write_text('initialize()\nrollout()\n')
    atomic_json(root / 'protocol_frozen.json', {'hashes': {str(source): digest(source)}})
    assert recovery.review(root, output, diagnostic_evidence_id=probe, reviewer=positive)['approved']
    source.write_text('different_evaluator()\n')
    assert not recovery.approved(root, output)
    with pytest.raises(ValueError, match='protected source'):
        recovery.review(root, output, diagnostic_evidence_id=probe, reviewer=positive)


def test_preparation_diagnosis_routes_the_real_traceback_to_fix_without_replay(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    from tests.test_prepare import _Client
    from autosim.research.evidence_store import read_attempt_evidence
    output, root, _ = study(tmp_path)
    repo = output / 'checkout'; repo.mkdir()
    real = Preparation(repo=repo, output=output, client=_Client([]),
                       scouting=tmp_path / 'scouting')
    monkeypatch.setattr(recovery, 'probe_connections', lambda *a, **k: {'scope': 'CPU connections only'})
    class Engine:
        snapshots = None
    real._research_controller = lambda: (Engine(), {})
    row = real._step_diagnose_confirmation_failure()
    assert row['repair_owner'] == 'fix' and not row['retryable']
    text = read_attempt_evidence(output, row['evidence_id'])['text']
    assert 'TypeError' in text and 'initialize' in text
    assert '998877' not in text and '0.99' not in text
    assert real._step_diagnose_confirmation_failure()['outcome'] == 'already diagnosed'


def test_confirmation_failure_terminal_status_is_not_completed(tmp_path, monkeypatch):
    from tests.test_prepare import preparation
    real = preparation(tmp_path)
    real.base_settings['_confirm'] = True
    real.steps = [{'step': 'run_the_loop', 'outcome': 'done', 'verification_level': 'L2'},
                  {'step': 'stop', 'outcome': 'stopped', 'because': 'engineering recovery pending'}]
    original = real.state
    monkeypatch.setattr(real, 'state', lambda: {**original(), 'research_progress': {
        'status': 'completed', 'confirmation': {'status': 'attempted_and_failed'}}})
    monkeypatch.setattr(real, '_review_final_study', lambda: pytest.fail('no completed-study review'))
    assert real._run_cycle(max_steps=0)['status'] == 'confirmation_failed'


def test_fix_hook_accepts_confirmation_engineering_without_gpu_or_edit_authority():
    from autosim.research.prepare import _sealed_fix_handoff
    assert _sealed_fix_handoff({'repair_owner': 'fix', 'failure_category': 'confirmation_engineering',
                                'evidence_id': 'a' * 32, 'retryable': False})


def test_stable_diagnosis_envelope_resolves_only_its_sealed_cpu_edge(tmp_path):
    from autosim.research.operation_failure import seal
    output, root, probe = study(tmp_path)
    subject = recovery.packet(root, output)['subject_sha256']
    parent = seal(output, operation='confirm_best_engineering', exc=ValueError('setup failure'),
                  context={'subject_sha256': subject, 'connection_probe': {'evidence_id': probe}})
    assert recovery.review(root, output, diagnostic_evidence_id=parent['evidence_id'],
                           reviewer=positive)['approved']
    wrong = seal(output, operation='confirm_best_engineering', exc=ValueError('another failure'),
                 context={'subject_sha256': '0' * 64, 'connection_probe': {'evidence_id': probe}})
    with pytest.raises(ValueError, match='does not bind'):
        recovery.review(root, output, diagnostic_evidence_id=wrong['evidence_id'], reviewer=positive)


def test_explicit_reason_bound_and_cache_preserve_real_independent_response(tmp_path):
    output, root, probe = study(tmp_path)
    def detailed(packet):
        return {**positive(packet), 'positive_evidence': 'source evidence ' * 240}
    row = recovery.review(root, output, diagnostic_evidence_id=probe, reviewer=detailed)
    assert row['approved']  # 3600 chars: must not silently turn setup_only into abstention
    cached = recovery.cached_review(root, output, diagnostic_evidence_id=probe)
    assert cached and cached[0]['verdict'] == 'setup_only'
    assert recovery.cached_review(root, output, diagnostic_evidence_id='c' * 32) is None
    (output / 'cpu.log').write_text('changed CPU evidence')
    assert recovery.cached_review(root, output, diagnostic_evidence_id=probe) is None
