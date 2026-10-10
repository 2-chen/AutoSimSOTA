import json

import pytest

from autosim.research.common import atomic_json
from autosim.research.development_evidence import summaries
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.native_runtime_summary import stage_runtime


def fixture(tmp_path):
    output = tmp_path/'run'
    root = output/'research/derived'
    attempt = 'a' * 32
    receipt = root/'attempts'/attempt/'receipt.json'
    native = {'run_id': 'derived', 'attempt_id': attempt, 'node_id': 'train',
              'status': 'completed', 'returncode': 0, 'termination_reason': 'normal_exit',
              'seconds': 8379.7, 'argv': ['--password=NEVER_EXPORT']}
    atomic_json(receipt, native)
    log = root/'train/attempts'/attempt/'output.log'
    log.parent.mkdir(parents=True)
    log.write_text('native result\n')
    capture_attempt_evidence(output, attempt_id=attempt, log=log,
        receipt_ref=receipt.relative_to(output).as_posix(), status='completed',
        returncode=0, termination_reason='normal_exit')
    measurement = {'label': 'baseline', 'ok': True, 'metric_value': .02,
                   'settings': {'steps': 8192}, 'train': {'attempt_id': attempt,
                   'evidence_id': attempt, 'returncode': 0, 'reused': False}}
    atomic_json(root/'measurements/baseline.json', measurement)
    return root, receipt, native, log, measurement


def test_runtime_is_bounded_context_not_cost_or_convergence(tmp_path):
    root, receipt, native, log, row = fixture(tmp_path)
    before = receipt.read_bytes()
    result = stage_runtime(root, row, 'train')
    assert result['seconds'] == 8379.7
    assert result['reused'] is False
    assert 'not GPU charges' in result['meaning']
    evidence = summaries(root)
    assert evidence[0]['native_stage_runtime']['train'] == result
    assert evidence[0]['metric_value'] == .02
    assert 'NEVER_EXPORT' not in json.dumps(evidence)
    assert receipt.read_bytes() == before


@pytest.mark.parametrize('seconds', [None, True, 0, -1, '8379', float('nan'), float('inf')])
def test_invalid_seconds_are_unknown(tmp_path, seconds):
    root, receipt, native, log, row = fixture(tmp_path)
    # Deliberately malformed external JSON; the normal writer rejects NaN/Inf.
    receipt.write_text(json.dumps({**native, 'seconds': seconds}))
    assert stage_runtime(root, row, 'train') is None


@pytest.mark.parametrize('key,value', [
    ('run_id', 'other'), ('attempt_id', 'b'*32), ('node_id', 'evaluate'),
    ('status', 'running'), ('returncode', True), ('returncode', 1),
    ('termination_reason', 'timeout'),
])
def test_mismatched_or_failed_native_receipt_is_not_a_completed_cost(tmp_path, key, value):
    root, receipt, native, log, row = fixture(tmp_path)
    atomic_json(receipt, {**native, key: value})
    assert stage_runtime(root, row, 'train') is None


@pytest.mark.parametrize('fault', ['missing_seal', 'changed_log', 'wrong_reference',
    'wrong_evidence', 'unlinked_attempt', 'receipt_symlink', 'ancestor_symlink',
    'log_symlink', 'oversized_log'])
def test_only_exact_sealed_attempt_is_exposed(tmp_path, fault):
    root, receipt, native, log, row = fixture(tmp_path)
    output = root.parent.parent
    seal = output/'evidence'/f'{native["attempt_id"]}.json'
    if fault == 'missing_seal':
        seal.unlink()
    elif fault == 'changed_log':
        log.write_text('NATIVE result\n')  # Same byte count; checksum must reject it.
    elif fault == 'wrong_reference':
        document = json.loads(seal.read_text())
        atomic_json(seal, {**document, 'receipt_ref': 'other/receipt.json'})
    elif fault == 'wrong_evidence':
        row['train']['evidence_id'] = 'b' * 32
    elif fault == 'unlinked_attempt':
        row['train']['attempt_id'] = '../outside'
    elif fault == 'receipt_symlink':
        other = receipt.with_name('other.json')
        receipt.rename(other)
        receipt.symlink_to(other)
    elif fault == 'ancestor_symlink':
        original = receipt.parent
        other = original.with_name('b'*32)
        original.rename(other)
        original.symlink_to(other, target_is_directory=True)
    elif fault == 'log_symlink':
        other = log.with_name('other.log')
        log.rename(other)
        log.symlink_to(other)
    elif fault == 'oversized_log':
        document = json.loads(seal.read_text())
        atomic_json(seal, {**document, 'log_bytes': 2*1024*1024+1})
    assert stage_runtime(root, row, 'train') is None


def test_reuse_does_not_claim_new_work_and_confirmation_not_projected(tmp_path):
    root, receipt, native, log, row = fixture(tmp_path)
    row['train']['reused'] = True
    assert stage_runtime(root, row, 'train')['reused'] is True
    assert stage_runtime(root, row, 'confirmation') is None
    assert stage_runtime(root, row, 'collect') is None
    atomic_json(root/'measurements/confirmation.json', {**row, 'label': 'confirmation'})
    assert [r['label'] for r in summaries(root)] == ['baseline']


def test_evaluation_elapsed_is_also_exposed(tmp_path):
    root, receipt, native, log, row = fixture(tmp_path)
    atomic_json(receipt, {**native, 'node_id': 'evaluate', 'seconds': 243.5})
    row['evaluate'] = row.pop('train')
    atomic_json(root/'measurements/baseline.json', row)
    result = summaries(root)[0]['native_stage_runtime']
    assert set(result) == {'evaluate'}
    assert result['evaluate']['seconds'] == 243.5


@pytest.mark.parametrize('key,value', [('returncode', True), ('returncode', 1),
    ('status', 'failed'), ('termination_reason', 'timeout'), ('evidence_id', 'b'*32)])
def test_sealed_outcome_must_match_successful_receipt(tmp_path, key, value):
    root, receipt, native, log, row = fixture(tmp_path)
    seal = root.parent.parent/'evidence'/f'{native["attempt_id"]}.json'
    document = json.loads(seal.read_text())
    atomic_json(seal, {**document, key: value})
    assert stage_runtime(root, row, 'train') is None
