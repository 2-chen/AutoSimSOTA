import datetime as dt
import http.client
import io
import json
from urllib.parse import urlparse

import pytest

from autosim.research.deepseek_gateway import (DeepSeekGatewayError, DeepSeekTurnGateway,
                                               TurnCostGate, _has_nontext)
from autosim.research.deepseek_pricing import (normalized_usage, official_cost_usd,
                                               peak_at, request_cost_ceiling_usd)


PEAK = dt.datetime(2026, 9, 29, 3, 0, tzinfo=dt.timezone.utc)
OFF_PEAK = dt.datetime(2026, 9, 29, 5, 0, tzinfo=dt.timezone.utc)


def test_local_gate_can_grow_under_persistent_total_and_records_actual_denial(tmp_path):
    from autosim.research.agent_budget import AgentCostLedger
    ledger = AgentCostLedger(tmp_path, run_id="adaptive", limit_usd=.5)
    reservation = ledger.reserve(session_id="s", role="fix", requested_usd=.01)
    gate = TurnCostGate(limit_usd=.01, model="deepseek-flash",
        extend_budget=lambda amount: ledger.extend(reservation["reservation_id"], amount))
    request = gate.reserve(model="deepseek-flash", max_tokens=100, at=PEAK,
                           input_bytes=800_000)
    assert gate.snapshot()["limit_usd"] > .25
    assert gate.snapshot()["denied_count"] == 0
    gate.settle(request, {"input_tokens": 20, "output_tokens": 10})
    with pytest.raises(DeepSeekGatewayError):
        gate.reserve(model="deepseek-flash", max_tokens=384_000, at=PEAK,
                     input_bytes=1_000_000)
    denial = gate.snapshot()["denials"][-1]
    assert denial["category"] == "run_model_budget"
    assert denial["input_bytes"] == 1_000_000


def test_flash_peak_and_offpeak_rates_match_pinned_official_card():
    usage = {"input_tokens": 1_000_000, "cache_read_input_tokens": 0,
             "output_tokens": 0}
    assert peak_at(PEAK) and not peak_at(OFF_PEAK)
    assert official_cost_usd("deepseek-flash", usage, at=PEAK)[0] == pytest.approx(0.30)
    assert official_cost_usd("deepseek-flash", usage, at=OFF_PEAK)[0] == pytest.approx(0.15)
    cached = {"input_tokens": 100, "cache_read_input_tokens": 900,
              "output_tokens": 500}
    amount, counts, tier = official_cost_usd("deepseek-flash", cached, at=PEAK)
    assert amount == pytest.approx((100 * .30 + 900 * .006 + 500 * 1.2) / 1e6)
    assert counts["input_hit_tokens"] == 900 and tier == "peak"


def test_provider_usage_shapes_and_unknowns_fail_closed():
    assert normalized_usage({"prompt_cache_miss_tokens": 5,
                             "prompt_cache_hit_tokens": 8,
                             "completion_tokens": 3}) == {
        "input_miss_tokens": 5, "input_hit_tokens": 8, "output_tokens": 3}
    with pytest.raises(ValueError, match="output tokens"):
        normalized_usage({"input_tokens": 1})
    with pytest.raises(ValueError, match="model"):
        official_cost_usd("claude-sonnet", {"input_tokens": 1, "output_tokens": 1}, at=PEAK)
    with pytest.raises(ValueError, match="expired"):
        request_cost_ceiling_usd("deepseek-flash", 100,
                                 at=dt.datetime(2026, 10, 11, tzinfo=dt.timezone.utc))


def test_request_gate_reserves_before_forwarding_and_retains_unknown_ceiling():
    ceiling = request_cost_ceiling_usd("deepseek-flash", 100, at=PEAK)
    gate = TurnCostGate(limit_usd=ceiling * 1.1, model="deepseek-flash")
    first = gate.reserve(model="deepseek-flash", max_tokens=100, at=PEAK)
    with pytest.raises(DeepSeekGatewayError, match="cannot reserve"):
        gate.reserve(model="deepseek-flash", max_tokens=100, at=PEAK)
    gate.settle(first, {"input_tokens": 100, "output_tokens": 20})
    assert gate.snapshot()["cost_usd"] > 0
    second = gate.reserve(model="deepseek-flash", max_tokens=100, at=PEAK)
    gate.settle(second, None)
    assert gate.snapshot()["unknown"]
    with pytest.raises(DeepSeekGatewayError, match="cannot reserve"):
        gate.reserve(model="deepseek-flash", max_tokens=100, at=PEAK)


def test_nested_tool_schema_type_does_not_crash_text_guard():
    assert not _has_nontext({"tools": [{"input_schema": {
        "type": {"oneOf": [{"type": "object"}]}}}]})
    assert _has_nontext({"content": [{"type": "image", "source": "x"}]})


def test_loopback_gateway_forwards_with_upstream_key_and_accounts_usage(monkeypatch):
    from autosim.research import deepseek_gateway

    seen = {}

    class FakeResponse(io.BytesIO):
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_upstream(request, **_):
        seen["url"] = request.full_url
        seen["key"] = request.get_header("X-api-key")
        seen["body"] = json.loads(request.data)
        return FakeResponse(json.dumps({
            "usage": {"input_tokens": 100, "cache_read_input_tokens": 20,
                      "output_tokens": 10}, "content": [{"type": "text", "text": "ok"}]
        }).encode())

    monkeypatch.setattr(deepseek_gateway.urllib.request, "urlopen", fake_upstream)
    with DeepSeekTurnGateway(upstream_base_url="https://api.deepseek.com/anthropic",
                             upstream_key="upstream-secret", model="deepseek-flash",
                             limit_usd=.05) as gateway:
        import time
        original_settle = gateway.gate.settle
        def delayed_settle(*args, **kwargs):
            time.sleep(.03)
            return original_settle(*args, **kwargs)
        monkeypatch.setattr(gateway.gate, 'settle', delayed_settle)
        url = urlparse(gateway.base_url)
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        body = json.dumps({"model": "deepseek-flash", "max_tokens": 100,
                           "messages": [{"role": "user", "content": "hello"}]})
        conn.request("POST", "/v1/messages?beta=fixture", body=body,
                     headers={"Authorization": "Bearer " + gateway.local_token,
                              "Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["content"][0]["text"] == "ok"
        conn.close()
        snapshot = gateway.gate.snapshot()
        assert snapshot["requests"] == 1 and not snapshot["unknown"]
        assert snapshot["cost_usd"] > 0
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        oversized = json.dumps({"model": "deepseek-flash", "max_tokens": 384_000,
                                "messages": [{"role": "user", "content": "hello"}]})
        conn.request("POST", "/v1/messages", body=oversized,
                     headers={"Authorization": "Bearer " + gateway.local_token,
                              "Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 400  # deterministic local refusal, not retryable 429
        response.read()
        conn.close()
        assert gateway.gate.snapshot()["denied_count"] == 1
        assert gateway.gate.snapshot()["requests"] == 1  # not forwarded upstream
    assert seen["url"] == "https://api.deepseek.com/anthropic/v1/messages?beta=fixture"
    assert seen["key"] == "upstream-secret"
    assert seen["body"]["model"] == "deepseek-flash"
