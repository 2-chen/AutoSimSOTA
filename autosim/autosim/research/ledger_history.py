"""Write-ahead proofs for accounting changes inside an authorized ledger prefix.

Budget grants remain immutable. A changed prefix is accepted only by walking
executor-owned before/after proofs back to its exact original digest. Appended
turns are not historical rewrites; missing, corrupt or unproved changes fail closed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from itertools import product
import math
from pathlib import Path
import re

from .common import atomic_json, immutable_json, object_digest, read_json

_SHA = re.compile(r'[0-9a-f]{64}\Z')
_FIELDS = {
    'extend': {'reserved_usd', 'provider_limit_usd', 'adjustments'},
    'settle': {'status', 'actual_usd', 'reserved_usd', 'details', 'finished_at'},
    'reconcile_unknown': {'status', 'actual_usd', 'reserved_usd',
                          'reconciled_at', 'reconciliation_evidence'},
    'receipt_reconciliation': {'status', 'actual_usd', 'reserved_usd', 'reconciled_at',
                               'reconciliation_evidence', 'reconciliation_receipt_sha256'},
}


def _amount(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _safe(root, relative):
    path = Path(root) / relative
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('unsafe ledger history path')
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('ledger history escaped run')
    return path


def _read(root, relative):
    path = _safe(root, relative)
    if not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError('missing or oversized ledger history proof')
    row = read_json(path)
    if not isinstance(row, dict):
        raise ValueError('ledger history proof must be an object')
    return row


def _valid_change(before, after, operation):
    fields = _FIELDS.get(operation)
    if not fields or not isinstance(before, dict) or not isinstance(after, dict):
        return False
    if ({k:v for k,v in before.items() if k not in fields}
            != {k:v for k,v in after.items() if k not in fields}):
        return False
    if not before.get('reservation_id') or not _amount(before.get('reserved_usd')):
        return False
    if not _amount(after.get('reserved_usd')):
        return False
    if operation == 'extend':
        return (before.get('status') == after.get('status') == 'reserved'
            and after['reserved_usd'] >= before['reserved_usd']
            and after.get('provider_limit_usd') == after['reserved_usd']
            and isinstance(after.get('adjustments'), list)
            and after['adjustments'][:-1] == before.get('adjustments', []))
    if before.get('status') not in ({'unknown'} if operation == 'reconcile_unknown'
                                    else {'reserved'} if operation == 'settle'
                                    else {'reserved', 'unknown'}):
        return False
    if after.get('status') == 'unknown':
        valid = after.get('actual_usd') is None and 0 <= after['reserved_usd'] <= before['reserved_usd']
    elif after.get('status') == 'settled':
        valid = _amount(after.get('actual_usd')) and after['reserved_usd'] == 0
    elif operation == 'settle' and after.get('status') == 'cancelled':
        valid = after.get('actual_usd') == 0 and after['reserved_usd'] == 0
    else:
        return False
    if operation == 'receipt_reconciliation':
        valid = (valid and isinstance(after.get('reconciliation_evidence'), str)
            and bool(_SHA.fullmatch(str(after.get('reconciliation_receipt_sha256', '')))))
    return bool(valid)


def _record(output, document, before, after, operation):
    changes = [{'index': i, 'before': old, 'after': new}
               for i, (old, new) in enumerate(zip(before, after)) if old != new]
    if not changes or len(before) != len(after) or not all(
            _valid_change(c['before'], c['after'], operation) for c in changes):
        raise ValueError('unproved accounting transition')
    record = {'schema_version': 1, 'run_id': document['run_id'],
        'cost_basis': document.get('cost_basis', 'cli_reported'),
        'prefix_count': len(before), 'before_sha256': object_digest(before),
        'after_sha256': object_digest(after), 'operation': operation, 'changes': changes}
    record['sha256'] = object_digest(record)
    relative = f'agent/ledger_history/{len(before)}/{record["after_sha256"]}.json'
    immutable_json(_safe(output, relative), record)
    return relative


def persist(ledger, before_document, document, *, operation):
    """Called under the ledger lock; proof is durable BEFORE the ledger mutation."""
    before, after = before_document['entries'], document['entries']
    if len(before) != len(after):
        raise ValueError('accounting update cannot insert or delete ledger entries')
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    if not changed:
        atomic_json(ledger.path, document)
        return
    directory = _safe(ledger.output, 'budget_amendments')
    paths = sorted(directory.glob('*.json'))
    if len(paths) > 256:
        raise ValueError('too many budget amendments for bounded history audit')
    counts = set()
    for path in paths:
        grant = _read(ledger.output, path.relative_to(ledger.output))
        if grant.get('status') != 'applied':
            continue
        count = grant.get('preserved', {}).get('model_entry_count')
        if type(count) is not int or not 0 <= count <= len(before):
            raise ValueError('invalid historical ledger prefix')
        if changed[0] < count:
            counts.add(count)
    for count in sorted(counts):
        _record(ledger.output, document, before[:count], after[:count], operation)
    atomic_json(ledger.path, document)


def matches(output, document, *, count, sha256):
    """No disk writes: rewind exact current rows to the authorized history anchor."""
    rows = deepcopy(document['entries'][:count])
    seen = set()
    while object_digest(rows) != sha256:
        current = object_digest(rows)
        if current in seen or len(seen) >= 1024:
            return False
        seen.add(current)
        relative = f'agent/ledger_history/{count}/{current}.json'
        record = _read(output, relative)
        if (record.get('schema_version') != 1 or record.get('run_id') != document['run_id']
                or record.get('cost_basis') != document.get('cost_basis', 'cli_reported')
                or record.get('prefix_count') != count or record.get('after_sha256') != current
                or record.get('sha256') != object_digest({k:v for k,v in record.items() if k != 'sha256'})
                or not _SHA.fullmatch(str(record.get('before_sha256', '')))):
            return False
        changes = record.get('changes')
        if not isinstance(changes, list) or not changes or len(changes) > count:
            return False
        indexes = set()
        for change in changes:
            if not isinstance(change, dict):
                return False
            i = change.get('index')
            if (type(i) is not int or not 0 <= i < count or i in indexes
                    or rows[i] != change.get('after') or not _valid_change(
                        change.get('before'), change.get('after'), record.get('operation'))):
                return False
            indexes.add(i)
            rows[i] = change['before']
        if object_digest(rows) != record['before_sha256']:
            return False
    return True


def recover_legacy_reconciliation(ledger, amendment_id):
    """Explicit stopped-run migration, NOT a relaxed authorization fallback.

    Older receipt reconciliations lacked before/after proofs. Search ONLY the old
    full-allowance reserved/unknown state of <=8 receipt-backed rows. Persist a
    proof only if the ENTIRE reconstructed prefix matches the existing grant hash.
    Never change the grant, ledger, historical charges, usage unknowns or limits.
    """
    if not isinstance(amendment_id, str) or not _SHA.fullmatch(amendment_id):
        raise ValueError('unsafe amendment identity')
    from .budget_receipts import reconcile
    report = reconcile(ledger, apply=False)
    audited = {r['reservation_id'] for r in report['results']
               if r['status'] == 'unknown_bound_verified'}
    with ledger._locked():
        document = ledger._read()
        grant = _read(ledger.output, f'budget_amendments/{amendment_id}.json')
        if grant.get('id') != amendment_id or grant.get('status') != 'applied':
            raise ValueError('applied original grant required')
        count = grant['preserved']['model_entry_count']
        if type(count) is not int or not 0 <= count <= len(document['entries']):
            raise ValueError('invalid original grant prefix')
        current = document['entries'][:count]
        target = grant['preserved']['model_entries_digest']
        granted_at = datetime.fromisoformat(grant['at'])
        try:
            if matches(ledger.output, document, count=count, sha256=target):
                return {'status': 'already_verified', 'amendment_id': amendment_id}
        except (OSError, ValueError, TypeError, KeyError):
            pass
        candidates = []
        for i, row in enumerate(current):
            if (row.get('reservation_id') in audited and row.get('status') == 'unknown'
                    and row.get('reconciled_at') and row.get('reconciliation_evidence')
                    and row.get('reconciliation_receipt_sha256')
                    and datetime.fromisoformat(row['reconciled_at']) >= granted_at):
                old = {k:v for k,v in row.items() if k not in
                    {'reconciled_at', 'reconciliation_evidence', 'reconciliation_receipt_sha256'}}
                old.update(reserved_usd=row['provider_limit_usd'], actual_usd=None)
                if not _amount(old['reserved_usd']) or old['reserved_usd'] < row['reserved_usd']:
                    raise ValueError('legacy reservation allowance is invalid')
                candidates.append((i, old))
        if not 1 <= len(candidates) <= 8:
            raise ValueError('legacy changes cannot be recovered within the exact bounded receipt scope')
        for statuses in product(('reserved', 'unknown'), repeat=len(candidates)):
            original = deepcopy(current)
            for (i, old), status in zip(candidates, statuses):
                original[i] = {**old, 'status': status}
            if object_digest(original) == target:
                # Recheck receipt bytes under the lock. Row/receipt linkage is bound by the
                # original reconciliation digest, not a model-provided assertion.
                for i, _ in candidates:
                    row = current[i]
                    price = _read(ledger.output, row['reconciliation_evidence'])
                    if price.get('kind') == 'request_gateway_journal_v1':
                        from .budget_receipts import known_gateway_cost
                        price = {**price, 'official_estimate_usd': known_gateway_cost(price.get('gateway'))}
                    if object_digest(price) != row['reconciliation_receipt_sha256']:
                        raise ValueError('legacy reconciliation receipt changed')
                ref = _record(ledger.output, document, original, current, 'receipt_reconciliation')
                return {'status': 'exact_legacy_prefix_proved', 'amendment_id': amendment_id,
                        'proof_ref': ref, 'reconstructed_rows': len(candidates),
                        'ledger_changed': False, 'authorized_history_sha256': target}
        raise ValueError('no exact original prefix; preserve accounting and request explicit investigation')
