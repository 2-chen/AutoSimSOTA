"""Additive operator review of old, explicitly unexecuted confirmation refusals.

No attempt, measurement, protocol or score is rewritten. Absence of an ID alone
is not proof: an authoritative controller result must say ran=false at a known
identity preflight, and all immutable attempts must agree. Unknown/exposed cases
remain closed. This is not an Agent tool or permission to resume development.
"""
from pathlib import Path

from .common import digest, immutable_json, now, object_digest, read_json


def _originals(root: Path) -> tuple[dict, dict]:
    records = {}
    def load(path):
        if path.is_symlink() or path.parent.is_symlink() or path.stat().st_size > 256 * 1024:
            raise ValueError('unsafe or oversized legacy confirmation record')
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError('invalid legacy confirmation record')
        records[path.relative_to(root).as_posix()] = digest(path)
        return value

    row = load(root / 'confirmation.json')
    result = load(root / 'measurements' / 'confirmation.json')
    if (result.get('ran') is not False or result.get('ok') is not False or
            result.get('where') not in {'source_identity', 'baseline_source_identity',
                                        'policy_identity', 'baseline_identity'}):
        raise ValueError('legacy result does not prove an identity preflight refusal')
    for key in ('label', 'candidate_label', 'settings', 'why'):
        if row.get(key) != result.get(key):
            raise ValueError('legacy result and confirmation projection disagree')
    def unexposed(value):
        if (value.get('ok') is not False or value.get('metric_value') is not None or
                value.get('baseline_metric_value') is not None or
                value.get('held_out_exposure') not in (None, 'not_started') or
                value.get('attempt_id') or value.get('baseline_evaluate') or
                value.get('evaluate') or value.get('comparison') or
                value.get('confirmation_comparison')):
            raise ValueError('legacy confirmation is exposed or exposure is unknown')
    unexposed(row)
    unexposed(result)
    directory = root / 'confirmation_attempts'
    if directory.is_symlink():
        raise ValueError('unsafe legacy attempt directory')
    attempts = sorted(directory.glob('*.json'))
    if not attempts or len(attempts) > 64:
        raise ValueError('legacy review needs a bounded immutable attempt history')
    for path in attempts:
        prior = load(path)
        unexposed(prior)
        if prior != row:
            raise ValueError('immutable confirmation attempt differs; independent review required')
    return records, {'where': result['where'], 'because': result['why']}


def reviewed_refusal(root: Path) -> dict | None:
    """Read-only projection; stale/missing proof never opens the retry gate."""
    try:
        path = root / 'confirmation_preflight_recovery.json'
        if path.is_symlink() or path.parent.is_symlink() or path.stat().st_size > 256 * 1024:
            return None
        recovery = read_json(path)
        records, _ = _originals(root)
        if (recovery.get('schema_version') != 1 or
                recovery.get('review_authority') != 'operator_engineering' or
                not recovery.get('authorization') or recovery.get('originals') != records or
                recovery.get('id') != object_digest(records)):
            return None
        failure = recovery.get('engineering_failure') or {}
        if (failure.get('held_out_exposure') != 'not_started' or
                failure.get('native_operation_status') != 'not_started' or
                not failure.get('evidence_id')):
            return None
        return failure
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def lifecycle(root: Path) -> dict | None:
    """Read authoritative confirmation lifecycle, never values, settings or outcomes."""
    path = root / 'confirmation.json'
    if not path.exists() and not path.is_symlink():
        return None
    try:
        if path.is_symlink() or path.parent.is_symlink() or path.stat().st_size > 256 * 1024:
            raise ValueError('unsafe confirmation record')
        row = read_json(path)
        if not isinstance(row, dict):
            raise ValueError('confirmation is not an object')
        if row.get('ok') is True:
            return {'status': 'taken'}
        from .confirmation_execution_recovery import approved
        if approved(root, root.parent.parent):
            return {'status': 'engineering_recovery_approved', 'retryable': True}
        native = bool(row.get('attempt_id') or (row.get('baseline_evaluate') or {}).get('attempt_id'))
        engineering = row.get('engineering_failure') or {}
        preflight = row.get('held_out_exposure') == 'not_started'
        if not engineering and not native and 'held_out_exposure' not in row:
            engineering = reviewed_refusal(root) or {}
            preflight = bool(engineering)
        if (not native and preflight and engineering and
                engineering.get('held_out_exposure') == 'not_started' and
                engineering.get('native_operation_status') == 'not_started'):
            return {'status': 'admission_refused', 'retryable': True,
                    'engineering_failure': engineering}
        return {'status': 'attempted_and_failed', 'retryable': False}
    except (OSError, ValueError, TypeError, AttributeError):
        return {'status': 'unreadable', 'retryable': False}


def review(root: Path, output: Path, *, authorization: str) -> dict:
    """Explicit independent engineering review; save proof without changing history."""
    if not isinstance(authorization, str) or not authorization.strip() or len(authorization) > 1000:
        raise ValueError('explicit bounded operator authorization is required')
    previous = reviewed_refusal(root)
    if previous:
        return previous
    records, safe = _originals(root)
    from .operation_failure import seal
    failure = {**seal(output, operation='confirm_best', exc=ValueError(safe['because']),
                     context={'where': safe['where'], 'legacy_review_id': object_digest(records)},
                     native_operation_status='not_started'),
               **safe, 'held_out_exposure': 'not_started', 'retryable': True,
               'repair_owner': 'fix', 'failure_category': 'confirmation_admission'}
    immutable_json(root / 'confirmation_preflight_recovery.json', {
        'schema_version': 1, 'id': object_digest(records), 'at': now(),
        'review_authority': 'operator_engineering', 'authorization': authorization,
        'originals': records, 'engineering_failure': failure,
        'authority': 'engineering preflight review only; no score or development reopening'})
    return failure
