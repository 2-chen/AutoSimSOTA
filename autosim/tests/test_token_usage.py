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
