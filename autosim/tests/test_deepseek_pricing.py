import datetime as dt
import http.client
import io
import json
from urllib.parse import urlparse

import pytest

from autosim.research.deepseek_gateway import (DeepSeekGatewayError, DeepSeekTurnGateway,
                                               TurnCostGate, _has_nontext)
from autosim.research.deepseek_pricing import (normalized_usage, official_cost_usd,
                                               peak_at, request_cost_ceiling_usd, PRICE_CARD_EXPIRES)


PEAK = dt.datetime(2026, 9, 29, 3, 0, tzinfo=dt.timezone.utc)
OFF_PEAK = dt.datetime(2026, 9, 29, 5, 0, tzinfo=dt.timezone.utc)


def test_gate_journals_before_forward_and_after_usage_without_request_payload(tmp_path):
    from autosim.research.common import atomic_json, read_json
    path = tmp_path / 'journal.json'
    gate = TurnCostGate(limit_usd=1, model='deepseek-flash',
                        checkpoint=lambda value: atomic_json(path, value))
    zero = read_json(path)
    assert zero['model'] == 'deepseek-flash'
    assert zero['unknown'] is False and zero['requests'] == 0
    request = gate.reserve(model='deepseek-flash', max_tokens=100, at=PEAK, input_bytes=1000)
    inflight = read_json(path)
    assert inflight['unknown'] is True and inflight['held_usd'] > 0
    gate.settle(request, {'input_tokens': 100, 'output_tokens': 10})
    settled = read_json(path)
    assert settled['unknown'] is False and settled['cost_usd'] > 0
    assert settled['held_usd'] == 0 and settled['requests'] == 1
    assert settled['receipts'][0]['requested_model'] == 'deepseek-flash'
    assert 'input_bytes' not in path.read_text()  # only accounting, not prompts/keys


def test_gate_cannot_forward_after_durable_reservation_write_fails():
    calls = []
    def checkpoint(value):
        calls.append(value)
        if value['held_usd'] > 0:
            raise OSError('fixture full disk')
    gate = TurnCostGate(limit_usd=1, model='deepseek-flash', checkpoint=checkpoint)
    with pytest.raises(DeepSeekGatewayError, match='durable'):
        gate.reserve(model='deepseek-flash', max_tokens=100, at=PEAK, input_bytes=1000)
    assert gate.snapshot()['bound_valid'] is False
    with pytest.raises(DeepSeekGatewayError):
        gate.reserve(model='deepseek-flash', max_tokens=100, at=PEAK, input_bytes=1000)


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
                                 at=PRICE_CARD_EXPIRES+dt.timedelta(seconds=1))


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


@pytest.mark.parametrize("model", ["deepseek-v4-pro", "deepseek-pro", "deepseek-reasoner"])
def test_gateway_never_forwards_cli_model_fallbacks(monkeypatch, model):
    from autosim.research import deepseek_gateway
    monkeypatch.setattr(deepseek_gateway.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("paid request must not start"))
    with DeepSeekTurnGateway(upstream_base_url="https://api.deepseek.com/anthropic",
                             upstream_key="fixture-key", model="deepseek-flash",
                             limit_usd=.05) as gateway:
        url = urlparse(gateway.base_url)
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        body = json.dumps({"model": model, "max_tokens": 100,
                           "messages": [{"role": "user", "content": "hello"}]})
        conn.request("POST", "/v1/messages", body=body,
                     headers={"Authorization": "Bearer " + gateway.local_token,
                              "Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 400
        response.read()
        conn.close()
        snapshot = gateway.snapshot()
        assert snapshot["model"] == "deepseek-flash"
        assert snapshot["requests"] == 0
        assert snapshot["accounting_ceiling_usd"] == 0


@pytest.mark.parametrize('generation_policy', [None, {'max_output_tokens':64,'thinking':'disabled'}])
def test_loopback_gateway_forwards_with_upstream_key_and_accounts_usage(monkeypatch,generation_policy):
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
                      "output_tokens": 10}, "stop_reason":"end_turn",
            "content": [{"type": "text", "text": "ok"}]
        }).encode())

    monkeypatch.setattr(deepseek_gateway.urllib.request, "urlopen", fake_upstream)
    with DeepSeekTurnGateway(upstream_base_url="https://api.deepseek.com/anthropic",
                             upstream_key="upstream-secret", model="deepseek-flash",
                             limit_usd=.05,generation_policy=generation_policy) as gateway:
        import time
        original_reserve=gateway.gate.reserve
        def capture_reserve(**kwargs):
            seen['reservation']=kwargs
            return original_reserve(**kwargs)
        monkeypatch.setattr(gateway.gate,'reserve',capture_reserve)
        original_settle = gateway.gate.settle
        def delayed_settle(*args, **kwargs):
            time.sleep(.03)
            return original_settle(*args, **kwargs)
        monkeypatch.setattr(gateway.gate, 'settle', delayed_settle)
        url = urlparse(gateway.base_url)
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        body = json.dumps({"model": "deepseek-flash", "max_tokens": 100,
                           "messages": [{"role": "user", "content": "hello"}],
                           "thinking":{"type":"enabled","budget_tokens":90},
                           "output_config":{"effort":"high","format":{"type":"json_schema"}}})
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
        diagnostics=snapshot['receipts'][0]['diagnostics']
        assert diagnostics['requested_max_output_tokens']==100
        assert diagnostics['effective_max_output_tokens']==(64 if generation_policy else 100)
        assert diagnostics['provider_stop_reason']=='end_turn'
        assert seen['body']['max_tokens']==(64 if generation_policy else 100)
        assert seen['body']['messages']==[{'role':'user','content':'hello'}]
        assert seen['body']['thinking']==({'type':'disabled'} if generation_policy else
                                          {'type':'enabled','budget_tokens':90})
        assert ('effort' in seen['body']['output_config']) is (generation_policy is None)
        assert seen['body']['output_config']['format']=={'type':'json_schema'}
        assert gateway.snapshot()['generation_policy']==(generation_policy or {})
        assert seen['reservation']['max_tokens']==seen['body']['max_tokens']
        assert seen['reservation']['input_bytes']==len(json.dumps(seen['body'],ensure_ascii=False).encode())
        assert seen['url']=='https://api.deepseek.com/anthropic/v1/messages?beta=fixture'
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        oversized = json.dumps({"model": "deepseek-flash", "max_tokens": 384_000,
                                "messages": [{"role": "user", "content": "hello"}]})
        conn.request("POST", "/v1/messages", body=oversized,
                     headers={"Authorization": "Bearer " + gateway.local_token,
                              "Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == (200 if generation_policy else 400)
        response.read()
        conn.close()
        assert gateway.gate.snapshot()["denied_count"] == (0 if generation_policy else 1)
        assert gateway.gate.snapshot()["requests"] == (2 if generation_policy else 1)
    assert seen["url"] == ("https://api.deepseek.com/anthropic/v1/messages" +
                            ("" if generation_policy else "?beta=fixture"))
    assert seen["key"] == "upstream-secret"
    assert seen["body"]["model"] == "deepseek-flash"


@pytest.mark.parametrize('policy',[{'max_output_tokens':True},{'max_output_tokens':0},
    {'max_output_tokens':1.5},{'max_output_tokens':65537},{'thinking':'enabled'},
    {'temperature':0},[]])
def test_invalid_review_generation_policy_refused_before_gateway(policy):
    from autosim.research.deepseek_gateway import validate_generation_policy
    with pytest.raises(ValueError):
        validate_generation_policy(policy)


def test_review_controls_do_not_expand_output_or_touch_history():
    from autosim.research.deepseek_gateway import apply_generation_policy
    request={'max_tokens':32,'messages':[{'content':[{'type':'thinking','thinking':'old'}]}],
             'tools':[{'name':'x'}],'cache_control':{'type':'ephemeral'}}
    before=json.loads(json.dumps(request))
    result=apply_generation_policy(request,{'max_output_tokens':1800,'thinking':'disabled'})
    assert result['max_tokens']==32 and request==before
    assert result['messages']==before['messages'] and result['tools']==before['tools']
    assert result['cache_control']==before['cache_control']
