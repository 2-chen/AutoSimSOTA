from autosim.research.common import read_json
from autosim.research.usage_reconciliation import due
from tests.test_environment_and_budget_alignment import priced_hold


def test_periodic_reconciliation_settles_only_trusted_receipt(tmp_path,monkeypatch):
    ledger,_,_=priced_hold(tmp_path,monkeypatch,unknown=True)
    result=due(tmp_path)
    assert result['status']=='checked'
    entry=read_json(ledger.path)['entries'][0]
    assert entry['status']=='settled' and entry['actual_usd']==.07
    monkeypatch.setattr('autosim.research.budget_receipts.reconcile',
        lambda *a,**k: (_ for _ in ()).throw(AssertionError('cached')))
    assert due(tmp_path)==result


def test_malformed_cache_does_not_crash_or_release_unknown(tmp_path,monkeypatch):
    ledger,price,_=priced_hold(tmp_path,monkeypatch,unknown=True)
    price.unlink()
    (tmp_path/'agent/usage_reconciliation_status.json').write_text('not json')
    assert due(tmp_path)['status']=='checked'
    entry=read_json(ledger.path)['entries'][0]
    assert entry['status']=='unknown' and entry['reserved_usd']==3


def test_unsafe_ledger_is_not_read(tmp_path):
    (tmp_path/'agent').mkdir()
    (tmp_path/'outside').write_text('{}')
    (tmp_path/'agent/cost_ledger.json').symlink_to(tmp_path/'outside')
    assert due(tmp_path)['status']=='blocked'
