"""Amortized receipt-based reconciliation; never release unknown cost by timeout."""
import time
import math
from pathlib import Path

from .common import atomic_json, read_json


def due(output: Path, *, interval: float = 60) -> dict:
    output = Path(output)
    path = output/'agent/usage_reconciliation_status.json'
    if path.is_symlink() or path.parent.is_symlink():
        return {'status':'blocked','reason':'unsafe reconciliation record'}
    if path.is_file():
        try:
            previous = read_json(path)
            age=time.time()-float(previous.get('checked_epoch',0))
            if math.isfinite(age) and 0 <= age < interval:
                return previous
        except (OSError,ValueError,TypeError,AttributeError):
            pass
    ledger_path = output/'agent/cost_ledger.json'
    if ledger_path.is_symlink():
        return {'status':'blocked','reason':'unsafe ledger'}
    if not ledger_path.is_file():
        return {'status':'not_configured'}
    try:
        from .agent_budget import AgentCostLedger
        from .budget_receipts import reconcile
        raw = read_json(ledger_path)
        ledger = AgentCostLedger(output,run_id=raw['run_id'],limit_usd=raw['limit_usd'],
                                 cost_basis=raw.get('cost_basis','cli_reported'))
        report = reconcile(ledger,apply=True)
        result = {'status':'checked','checked_epoch':time.time(),
                  'settled_from_receipt':sum(r['status']=='settled_from_receipt' for r in report['results']),
                  'unresolved_retained':sum(r['status']=='retained' for r in report['results']),
                  'report_ref':'agent/budget_reconciliation.json',
                  'meaning':'only owned terminal evidence settles holds; unresolved usage remains reserved'}
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        result = {'status':'retained','checked_epoch':time.time(),'reason':str(exc)[:500]}
    atomic_json(path,result)
    return result
