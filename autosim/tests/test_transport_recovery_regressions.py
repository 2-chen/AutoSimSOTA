"""Regression coverage for real LIBERO v2 transport/accounting/installer faults."""
import datetime as dt
import json
from pathlib import Path
import pytest

from autosim.research.deepseek_gateway import TurnCostGate, DeepSeekGatewayError
from autosim.research.agent_budget import AgentCostLedger

AT = dt.datetime(2026, 9, 30, 12, tzinfo=dt.timezone.utc)


def test_unknown_request_does_not_lock_future_priced_requests():
    gate = TurnCostGate(limit_usd=4, model="deepseek-flash")
    first = gate.reserve(model="deepseek-flash", max_tokens=100, at=AT, input_bytes=1000)
    gate.settle(first, None, diagnostics={"http_status": 503, "error_type": "HTTPError"})
    second = gate.reserve(model="deepseek-flash", max_tokens=100, at=AT, input_bytes=1000)
    gate.settle(second, {"input_tokens": 20, "output_tokens": 10})
    snapshot = gate.snapshot()
    assert snapshot["unknown"] and snapshot["cost_usd"] > 0
    assert snapshot["accounting_ceiling_usd"] == pytest.approx(
        snapshot["cost_usd"] + snapshot["unknown_reserved_usd"])
    assert snapshot["receipts"][0]["diagnostics"]["http_status"] == 503
    assert snapshot["accounting_ceiling_usd"] < .01


def test_broken_stream_keeps_full_request_bound_then_next_request_succeeds(monkeypatch):
    import http.client
    import io
    from urllib.parse import urlparse
    from autosim.research import deepseek_gateway
    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Type": "text/event-stream"}
        def __enter__(self): return self
        def __exit__(self, *_): self.close()
    initial = b'data:{"type":"message_start","message":{"usage":{"input_tokens":20,"output_tokens":0}}}\n\n'
    endings = [b"", b'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":100}}\n\ndata: {"type":"message_stop"}\n\n']
    monkeypatch.setattr(deepseek_gateway.urllib.request, "urlopen",
        lambda *_a, **_kw: Response(initial + endings.pop(0)))
    with deepseek_gateway.DeepSeekTurnGateway(upstream_base_url="https://api.deepseek.com/anthropic",
            upstream_key="fixture", model="deepseek-flash", limit_usd=4) as gateway:
        url = urlparse(gateway.base_url)
        for _ in range(2):
            connection = http.client.HTTPConnection(url.hostname, url.port, timeout=3)
            connection.request("POST", "/v1/messages", body=json.dumps({"model":"deepseek-flash",
                "max_tokens":100,"stream":True,"messages":[{"role":"user","content":"fixture"}]}),
                headers={"Authorization":"Bearer " + gateway.local_token})
            response = connection.getresponse()
            assert response.status == 200
            response.read()
            connection.close()
    result = gateway.snapshot()
    assert result["requests"] == 2 and result["unknown"]
    assert result["receipts"][0]["diagnostics"]["error_type"] == "IncompleteStream"
    assert result["receipts"][1]["status"] == "priced"
    assert result['receipts'][1]['diagnostics']['provider_stop_reason']=='max_tokens'
    assert result['receipts'][1]['diagnostics']['effective_max_output_tokens']==100


def test_request_bound_releases_only_unforwarded_allowance(tmp_path):
    ledger = AgentCostLedger(tmp_path, run_id="fixture", limit_usd=5,
        cost_basis="deepseek_official_estimate_v1")
    reservation = ledger.reserve(session_id="s", role="scheduler", requested_usd=4)
    state = ledger.settle(reservation["reservation_id"], actual_usd=None, launched=True,
        request_bound_usd=.2, details={"pricing_ref": "agent/pricing/fixture.json"})
    assert state["unknown_entries"] == 1
    assert state["reserved_usd"] == .2 and state["remaining_usd"] == 4.8
    ledger.reserve(session_id="s2", role="fix", requested_usd=4)


@pytest.mark.parametrize("bound", [-1, float("nan"), 5, True])
def test_invalid_request_bounds_cannot_release_budget(tmp_path, bound):
    ledger = AgentCostLedger(tmp_path, run_id="fixture", limit_usd=5,
        cost_basis="deepseek_official_estimate_v1")
    reservation = ledger.reserve(session_id="s", role="scheduler", requested_usd=4)
    with pytest.raises(ValueError):
        ledger.settle(reservation["reservation_id"], actual_usd=None, launched=True,
            request_bound_usd=bound, details={"pricing_ref": "receipt"})
    assert ledger.snapshot()["reserved_usd"] == 4


def test_provider_usage_above_ceiling_still_fails_closed():
    gate = TurnCostGate(limit_usd=4, model="deepseek-flash")
    request = gate.reserve(model="deepseek-flash", max_tokens=1, at=AT, input_bytes=1)
    gate.settle(request, {"input_tokens": 1000000, "output_tokens": 10000})
    assert not gate.snapshot()["bound_valid"]
    with pytest.raises(DeepSeekGatewayError, match="unsafe"):
        gate.reserve(model="deepseek-flash", max_tokens=1, at=AT, input_bytes=1)


def test_background_installer_discovery_without_interactive_path(tmp_path, monkeypatch):
    import shutil
    from autosim.research.provision import conda_executable
    monkeypatch.delenv("AUTOSIM_CONDA_EXECUTABLE", raising=False)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    binary = tmp_path / "miniconda3/bin/conda"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    assert conda_executable() == str(binary)


def test_transport_decision_retry_does_not_call_other_roles(tmp_path):
    from autosim.research.prepare import Preparation
    class Client:
        supports_main_agent = True
        model = "fixture"
        def as_role(self, role):
            raise AssertionError("do not call the same failed transport through Monitor")
    repo = tmp_path / "checkout"
    repo.mkdir()
    controller = Preparation(repo=repo, output=tmp_path, client=Client(), scouting=tmp_path / "scouting")
    failure = {"failure_category": "provider_transport", "failure_fingerprint": "same"}
    assert controller._handoff_scheduler_decision_failure(failure=failure)
    assert controller._handoff_scheduler_decision_failure(failure=failure)
    assert controller._handoff_scheduler_decision_failure(failure=failure)
    assert not controller._handoff_scheduler_decision_failure(failure=failure)
    assert failure["recovery"]["status"] == "blocked"


@pytest.mark.parametrize('category',['provider_payment_required','provider_access_blocked'])
def test_terminal_provider_refusal_does_not_retry_or_ask_another_role(tmp_path,category):
    from autosim.research.prepare import Preparation
    class Client:
        supports_main_agent=True
        model='fixture'
        def as_role(self,role):raise AssertionError('blocked endpoint must not be retried')
    repo=tmp_path/'checkout';repo.mkdir()
    controller=Preparation(repo=repo,output=tmp_path,client=Client(),scouting=tmp_path/'scouting')
    failure={'failure_category':category,'failure_fingerprint':'terminal'}
    assert not controller._handoff_scheduler_decision_failure(failure=failure)
    assert failure['recovery']['status']=='external_action_required'
    assert not (tmp_path/'agent/runtime_recovery_gate.json').exists()


@pytest.mark.parametrize('status,expected',[(402,'provider_payment_required'),
    (401,'provider_access_blocked'),(403,'provider_access_blocked'),(429,''),(503,''),('402',''),(True,'')])
def test_provider_http_status_classification_is_typed_not_text(status,expected):
    from autosim.research.agent_runtime import blocked_provider_category
    assert blocked_provider_category({'receipts':[{'diagnostics':{'http_status':status}}]},None)==expected
    assert blocked_provider_category(None,{'is_error':True,'api_error_status':status})==expected
    assert blocked_provider_category(None,{'is_error':False,'api_error_status':status})==''
    assert blocked_provider_category(None,{'is_error':True,'result':'API Error: 402'})==''
