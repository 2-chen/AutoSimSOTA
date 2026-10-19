from autosim.research.common import atomic_json
from autosim.research.token_usage import usage_report


def test_token_report_separates_unknown_from_zero_and_groups_roles(tmp_path):
    atomic_json(tmp_path / "agent/processes/a.json", {"attempt_id": "a", "role": "fix"})
    atomic_json(tmp_path / "agent/pricing/a.json", {"turn_id": "a", "gateway": {
        "receipts": [{"usage": {"input_hit_tokens": 80, "input_miss_tokens": 20,
                                 "output_tokens": 5}}, {"status": "unknown"}]}})
    report = usage_report(tmp_path)
    assert report["totals"]["cache_hit_ratio"] == .8
    assert report["roles"]["fix"]["requests_without_usage"] == 1
    assert report["unknown_usage_is_not_zero_cost"]
    atomic_json(tmp_path / "agent_workers/w/agent/processes/b.json",
                {"attempt_id": "b", "role": "recorder"})
    atomic_json(tmp_path / "agent_workers/w/agent/pricing/b.json", {"turn_id": "b",
        "gateway": {"receipts": [{"usage": {"input_hit_tokens": 0,
            "input_miss_tokens": 100, "output_tokens": 2}}]}})
    report = usage_report(tmp_path)
    assert report["worker_roots_included"] == 1
    assert report["totals"]["cache_hit_ratio"] == .4
    assert report["roles"]["recorder"]["input_miss_tokens"] == 100


def test_stored_costs_unknown_holds_and_prefix_observability_are_not_repriced(tmp_path):
    atomic_json(tmp_path / 'agent/processes/a.json', {'attempt_id': 'a', 'role': 'scheduler'})
    atomic_json(tmp_path / 'agent/pricing/a.json', {'turn_id': 'a', 'gateway': {'receipts': [
        {'status': 'priced', 'cost_usd': .123, 'price_card': 'historic-card',
         'usage': {'input_hit_tokens': 80, 'input_miss_tokens': 20, 'output_tokens': 10},
         'request_profile': {'system_sha256': 'same', 'envelope_sha256': 'envelope'}},
        {'status': 'priced', 'cost_usd': .077,
         'usage': {'input_hit_tokens': 180, 'input_miss_tokens': 20, 'output_tokens': 30},
         'request_profile': {'system_sha256': 'same', 'envelope_sha256': 'envelope'}},
        {'status': 'unknown', 'reserved_usd': .4}]}})
    report = usage_report(tmp_path)
    totals = report['totals']
    assert totals['known_cost_usd'] == .2 and totals['unknown_reserved_usd'] == .4
    assert totals['requests_without_cost'] == 1
    assert totals['input_tokens_per_request'] == {'mean': 150, 'max': 200, 'p95': 200}
    assert totals['request_profiles']['distinct_hashes']['system_sha256'] == 1
    assert totals['request_profiles']['missing_requests'] == 1
    assert totals['known_cost_share'] == 1
    assert report['not_full_inherited_ledger_or_provider_invoice']
    assert report['basis'] == 'stored_gateway_receipts_only_no_repricing'
