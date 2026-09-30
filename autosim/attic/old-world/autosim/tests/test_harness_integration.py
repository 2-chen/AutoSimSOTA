"""Operational API advice cannot enlarge an admitted recovery or weaken identity."""
import json
from unittest.mock import patch

import pytest

from autosim.research.common import atomic_json, digest
from autosim.research.harness_config import load_harness_policy
from autosim.research.harness_acceptance import inventory_verdict
from autosim.research.infrastructure_recovery import choose_probe_recovery
from autosim.research.compute_agent import ComputeAgent


@pytest.mark.parametrize("value", [
    {"lifecycle": {"startup_attempts": 4}},
    {"lifecycle": {"minimum_overlap_seconds": float("nan")}},
    {"repair": {"max_candidates": True}},
    {"validation": {"inject_failure_worker": 0}},
    {"validation": {"probe_repetitions": 2}},
    {"continuation": {"max_no_progress": 3}},
    {"validation": {"stage": "force_final"}},
    {"lifecycle": {"change_physics": True}},
])
def test_frozen_policy_rejects_unsafe_or_ambiguous_knobs(tmp_path, value):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_harness_policy(path)


def recovery(tmp_path, proposal, *, remaining=300):
    env = tmp_path / "fake.env"
    env.write_text("fixture; never a real credential")
    def answer(kind, snapshot, **kwargs):
        assert kind == "native_startup_recovery"
        assert "private" not in json.dumps(snapshot)
        return {"request_id": "test-request", "content": json.dumps({
            "incident_id": snapshot["incident_id"], **proposal})}
    with patch("autosim.research.infrastructure_recovery.client_from_file") as client, \
         patch("autosim.research.infrastructure_recovery.ComputeAgent") as agent:
        agent.return_value.request.side_effect = answer
        result = choose_probe_recovery(incident={"primary_failure": {"signal": 11},
            "private": "must not leave the process"}, output=tmp_path,
            current_construction_concurrency=4, max_workers=4,
            remaining_seconds=remaining, env_file=env)
        return result, agent.return_value.request.call_count, client.call_count


def test_api_can_reduce_construction_and_receipt_prevents_repeated_call(tmp_path):
    proposal = {"action": "restart_generation", "construction_concurrency": 1, "hypothesis": "construction contention"}
    first, calls, _ = recovery(tmp_path, proposal)
    assert first["api_used"] and first["next_construction_concurrency"] == 1 and calls == 1
    second, calls, _ = recovery(tmp_path, proposal)
    assert first == second and calls == 0


def test_api_stop_is_preserved(tmp_path):
    result, _, _ = recovery(tmp_path, {"action": "stop", "construction_concurrency": 4, "hypothesis": "unknown failure"})
    assert result["action"] == "stop" and result["api_used"]


@pytest.mark.parametrize("proposal", [
    {"action": "restart_generation", "construction_concurrency": 8, "hypothesis": "more devices"},
    {"action": "restart_generation", "construction_concurrency": True, "hypothesis": "boolean"},
    {"action": "reset_budget", "construction_concurrency": 1, "hypothesis": "reset"},
    {"action": "restart_generation", "construction_concurrency": 1, "hypothesis": "change", "seed": 2},
])
def test_invalid_api_advice_cannot_change_admitted_limits(tmp_path, proposal):
    result, calls, _ = recovery(tmp_path, proposal)
    assert calls == 1 and not result["api_used"]
    assert result["next_construction_concurrency"] == 4 and result["fresh_cache"] is True


def test_insufficient_original_time_does_not_start_api(tmp_path):
    result, calls, clients = recovery(tmp_path, {}, remaining=184)
    assert calls == clients == 0 and not result["api_used"]


def test_inventory_binds_uuid_and_host_outside_worker_self_report(tmp_path):
    inventory = {"host": "node", "gpus": [{"index": 0, "uuid": "GPU-actual"}], "allowed": [0]}
    path = tmp_path / "inventory.json"
    atomic_json(path, inventory)
    document = {"requested_devices": ["GPU-actual"],
        "concurrency_evidence": {"workers": [{"host": "node"}]},
        "inventory_evidence": {"artifact": str(path), "sha256": digest(path), "host": "node"}}
    outer = tmp_path / "allocation_inventory.json"
    atomic_json(outer, inventory)
    assert not inventory_verdict(document, tmp_path / "device_probe.json", outer)
    atomic_json(outer, {**inventory, "host": "different-node"})
    assert inventory_verdict(document, tmp_path / "device_probe.json", outer)
    atomic_json(outer, inventory)
    document["requested_devices"] = ["GPU-invented"]
    assert inventory_verdict(document, tmp_path / "device_probe.json", outer)


def test_shared_api_callers_cannot_expand_original_limits_even_after_cached_read(tmp_path):
    class Client:
        model, base_url = "fixture", "https://invalid.example"
        calls = 0
        def chat_with_metadata(self, *args, **kwargs):
            self.calls += 1
            return "{}", {"usage": {"total_tokens": 10}}
    client = Client()
    ComputeAgent(tmp_path, client, max_calls=8, max_tokens=80000).request("first", {}, system="fixture")
    ComputeAgent(tmp_path, client, max_calls=1, max_tokens=80000).request("first", {}, system="fixture")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        ComputeAgent(tmp_path, client, max_calls=8, max_tokens=80000).request("new", {}, system="fixture")
    assert client.calls == 1
