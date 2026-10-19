import json
import pytest

from autosim.research.common import atomic_json, digest
from autosim.research.confirmation_recovery import review, reviewed_refusal, lifecycle


def legacy(tmp_path):
    root = tmp_path / 'research' / 'derived'
    row = {'schema_version': 1, 'label': 'confirmation', 'candidate_label': 'round_2',
           'settings': {'seed': 998877}, 'why': 'content-addressed source identity mismatch',
           'ok': False, 'metric_value': None, 'baseline_metric_value': None,
           'baseline_evaluate': {}, 'attempt_id': None, 'comparison': {}}
    result = {k: row[k] for k in ('label', 'candidate_label', 'settings', 'why', 'ok', 'metric_value')}
    result.update(ran=False, where='source_identity')
    atomic_json(root / 'confirmation.json', row)
    atomic_json(root / 'confirmation_attempts' / ('a' * 32 + '.json'), row)
    atomic_json(root / 'measurements' / 'confirmation.json', result)
    return root, row, result


def test_legacy_review_is_additive_idempotent_private_and_invalidates_on_change(tmp_path):
    root, row, _ = legacy(tmp_path)
    paths = list(root.rglob('*.json'))
    hashes = {p: digest(p) for p in paths}
    assert reviewed_refusal(root) is None
    result = review(root, tmp_path, authorization='user approved original study continuation')
    assert result['held_out_exposure'] == result['native_operation_status'] == 'not_started'
    assert reviewed_refusal(root) == result
    assert review(root, tmp_path, authorization='same request') == result
    assert hashes == {p: digest(p) for p in paths}
    log = tmp_path / 'agent' / 'operation_failures' / (result['evidence_id'] + '.log')
    assert '998877' not in log.read_text()
    atomic_json(root / 'confirmation_attempts' / ('b' * 32 + '.json'), row)
    assert reviewed_refusal(root) is None  # history changes invalidate independent review


@pytest.mark.parametrize('change', ['ran', 'where', 'native', 'value', 'unknown', 'disagree', 'history'])
def test_legacy_review_never_infers_unexposed_from_missing_ids(tmp_path, change):
    root, row, result = legacy(tmp_path)
    if change == 'ran':
        result.pop('ran')
    elif change == 'where':
        result['where'] = 'evaluate'
    elif change == 'native':
        result['evaluate'] = {'attempt_id': 'c' * 32}
    elif change == 'value':
        result['metric_value'] = 0.99
    elif change == 'unknown':
        row['held_out_exposure'] = 'unknown'
    elif change == 'disagree':
        result['candidate_label'] = 'round_3'
    elif change == 'history':
        prior = {**row, 'attempt_id': 'd' * 32}
        atomic_json(root / 'confirmation_attempts' / ('b' * 32 + '.json'), prior)
    atomic_json(root / 'confirmation.json', row)
    atomic_json(root / 'measurements' / 'confirmation.json', result)
    with pytest.raises(ValueError):
        review(root, tmp_path, authorization='user approved continuation')
    assert reviewed_refusal(root) is None
    assert not (root / 'confirmation_preflight_recovery.json').exists()


def test_unknown_exposure_overrides_stale_engineering_handoff(tmp_path):
    root, row, _ = legacy(tmp_path)
    row.update(held_out_exposure='unknown', engineering_failure={
        'held_out_exposure': 'not_started', 'native_operation_status': 'not_started',
        'evidence_id': 'a' * 32})
    atomic_json(root / 'confirmation.json', row)
    assert lifecycle(root) == {'status': 'attempted_and_failed', 'retryable': False}
