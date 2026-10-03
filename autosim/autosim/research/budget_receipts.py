"""Reconcile abandoned model holds from executor receipts, never from process death alone."""
from pathlib import Path
import math
import re

from .common import atomic_json, now, object_digest, read_json
from .process_executor import inspect_process_identity


def _read(root, path):
    path = Path(path)
    if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or
            not path.is_file() or path.stat().st_size > 2*1024**2):
        raise ValueError('unsafe or missing receipt')
    row = read_json(path)
    if not isinstance(row, dict):
        raise ValueError('receipt must be an object')
    return row


def _amount(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value >= 0


def known_gateway_cost(snapshot):
    """A trusted local gate can also prove that no provider request was forwarded."""
    if not isinstance(snapshot,dict) or snapshot.get('unknown') is not False or not _amount(snapshot.get('cost_usd')):
        return None
    count = snapshot.get('requests')
    if not isinstance(count,int) or isinstance(count,bool) or count < 0:
        return None
    if count > 0 and snapshot.get('bound_valid',True) is True:
        return float(snapshot['cost_usd'])
    if (count == 0 and snapshot.get('bound_valid') is True and snapshot['cost_usd']==0 and
            snapshot.get('accounting_ceiling_usd')==0 and snapshot.get('held_usd')==0):
        return 0.0
    return None


def reconcile(ledger, *, apply=False):
    """Idempotent audit/apply. Unknown or live/unverifiable turns retain their holds.

    No provider calls, process termination, guessing or reinterpretation of CLI prices.
    Root-relative evidence is stored so nested Recorder receipts remain discoverable.
    """
    root = ledger.output
    with ledger._locked():
        document = ledger._read()
        ledger._totals(document)
        before = object_digest(document)
        pending = {r['reservation_id']:r for r in document['entries']
                   if r['status'] in {'reserved','unknown'}}
        sessions = {}
        paths = list(root.glob('agent/sessions/*.json'))
        paths += list(root.glob('agent_tasks/*/agent/sessions/*.json'))
        paths += list(root.glob('agent_workers/*/agent/sessions/*.json'))
        for path in sorted(paths)[:4000]:
            try:
                session = _read(root,path)
                identity = session.get('budget_reservation_id')
                if identity in pending:
                    sessions.setdefault(identity,[]).append((path,session))
            except (OSError,ValueError,TypeError):
                continue
        results = []
        changed = False
        for identity, entry in pending.items():
            result = {'reservation_id':identity,'role':entry['role'],
                      'held_usd':entry['reserved_usd'],'status':'retained'}
            try:
                matches = sessions.get(identity,[])
                if len(matches) != 1:
                    raise ValueError('unique reservation-owned session receipt unavailable')
                path, session = matches[0]
                owner = path.parent.parent.parent
                if (session.get('run_id') != ledger.run_id or session.get('role') != entry['role'] or
                        session.get('session_id') != entry['session_id'] or
                        session.get('cost_basis','cli_reported') != ledger.cost_basis or
                        Path(session.get('budget_output') or owner).resolve() != root.resolve()):
                    raise ValueError('session ownership or frozen accounting basis differs')
                turn = session.get('process_attempt_id','')
                if not isinstance(turn,str) or not re.fullmatch('[0-9a-f]{32}',turn):
                    raise ValueError('session has no stable executor attempt ID')
                process = _read(root,owner/'agent/processes'/f'{turn}.json')
                if process.get('attempt_id') != turn or process.get('run_id') != ledger.run_id:
                    raise ValueError('process receipt identity differs')
                process_id = process.get('process_identity')
                if isinstance(process_id,dict):
                    observed = inspect_process_identity(process_id)
                    if observed.get('status') != 'not_running':
                        raise ValueError('process remains live or cannot be verified absent')
                elif process.get('status') != 'not_launched':
                    raise ValueError('missing process identity is not evidence of zero usage')
                pricing_path = owner/'agent/pricing'/f'{turn}.json'
                price = _read(root,pricing_path)
                from .deepseek_pricing import PRICE_CARD_ID
                if (ledger.cost_basis != 'deepseek_official_estimate_v1' or
                        price.get('cost_basis') != ledger.cost_basis or price.get('turn_id') != turn or
                        price.get('schema_version') != 1 or price.get('price_card') != PRICE_CARD_ID):
                    raise ValueError('trusted frozen pricing receipt unavailable')
                gateway = price.get('gateway') or {}
                actual = price.get('official_estimate_usd')
                if actual is None and known_gateway_cost(gateway) == 0 and gateway.get('requests') == 0:
                    actual = 0.0
                ref = str(pricing_path.relative_to(root))
                if (_amount(actual) and isinstance(gateway,dict) and
                        known_gateway_cost(gateway) is not None and
                        math.isclose(actual,gateway['cost_usd'],rel_tol=0,abs_tol=1e-9) and
                        (gateway.get('requests',0)>0 or known_gateway_cost(gateway)==0)):
                    update = {'status':'settled','actual_usd':float(actual),'reserved_usd':0.0}
                    result.update(status='settled_from_receipt',actual_usd=actual,evidence_ref=ref)
                else:
                    bound = gateway.get('accounting_ceiling_usd') if isinstance(gateway,dict) else None
                    if (not isinstance(gateway,dict) or gateway.get('unknown') is not True or
                            gateway.get('bound_valid') is not True or not _amount(bound) or
                            not 0 < bound <= entry['reserved_usd']):
                        raise ValueError('usage unknown and no verified tighter request bound')
                    update = {'status':'unknown','actual_usd':None,'reserved_usd':float(bound)}
                    result.update(status='unknown_bound_verified',held_usd=bound,evidence_ref=ref)
                if apply and any(entry.get(k) != v for k,v in update.items()):
                    entry.update(**update,reconciled_at=now(),reconciliation_evidence=ref,
                                 reconciliation_receipt_sha256=object_digest(price))
                    changed = True
            except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
                result['reason'] = str(exc)[:240]
            results.append(result)
        if apply and changed:
            document['updated_at'] = now()
            atomic_json(ledger.path,document)
        report = {'schema_version':1,'created_at':now(),'applied':bool(apply),
                  'ledger_before_sha256':before,'ledger_after_sha256':object_digest(document),
                  'results':results,'policy':'receipt-backed only; process death never clears unknown usage'}
    if apply:
        atomic_json(root/'agent/budget_reconciliation.json',report)
    return report
