import pytest

from autosim.research.compute_decision import ComputeDecision, decide, recheck_gpu
from autosim.research.devices import NoCompatibleDevice


def gpu(index, *, used=0, utilization=0.0, pids=None):
    return {"index": index, "torch_index": index, "uuid": f"GPU-test-{index}",
            "model": "NVIDIA Test GPU", "memory_total_mib": 16000,
            "memory_used_mib": used, "utilization_gpu_pct": utilization,
            "compute_pids": pids or []}


def report(*gpus):
    return {"gpus": list(gpus), "allowed": [row["index"] for row in gpus],
            "outer": {}, "excluded": [], "host": "test-node",
            "cpu": {"effective_cpus": 4}, "memory": {"available_mib": 10000}}


def test_single_busy_visible_gpu_is_refused(monkeypatch):
    monkeypatch.setattr("autosim.research.compute_decision.discover",
                        lambda **_: report(gpu(0, used=9000, utilization=57.0,
                                                pids=[4312])))

    with pytest.raises(NoCompatibleDevice, match="all visible GPUs are busy") as caught:
        decide()

    assert caught.value.evidence["busy_devices"][0]["uuid"] == "GPU-test-0"
    assert "9000 MiB" in caught.value.evidence["busy_devices"][0]["reason"]


def test_explicit_cuda_preference_is_not_silently_changed_to_cpu(monkeypatch):
    monkeypatch.setattr("autosim.research.compute_decision.discover",
                        lambda **_: {**report(), "gpus": [], "allowed": []})

    with pytest.raises(NoCompatibleDevice, match="no GPU"):
        decide(prefer="cuda")


def test_hardware_seen_by_nvidia_smi_but_not_cuda_is_unavailable_not_cpu(monkeypatch):
    physical = gpu(0)
    discovery = {**report(physical), "allowed": [],
                 "excluded": [{"index": 0, "reason": "not visible to CUDA"}]}
    monkeypatch.setattr("autosim.research.compute_decision.discover",
                        lambda **_: discovery)

    decision = decide()

    assert decision.device == "unavailable"
    assert decision.evidence["resource_unavailable"] is True
    assert "no card is visible to CUDA" in decision.why
    assert decide(prefer="cpu").device == "cpu"


def test_device_query_error_is_not_evidence_that_the_machine_has_no_gpu(monkeypatch):
    monkeypatch.setattr(
        "autosim.research.compute_decision.discover",
        lambda **_: {**report(), "errors": {"gpu_query": "nvidia-smi rc=9"}})

    assert decide().device == "unavailable"
    assert decide(prefer="cpu").device == "cpu"


def test_discovery_exception_requires_explicit_cpu_fallback(monkeypatch):
    def fail(**_kwargs):
        raise OSError("device table unavailable")

    monkeypatch.setattr("autosim.research.compute_decision.discover", fail)

    assert decide().device == "unavailable"
    assert decide(prefer="cpu").device == "cpu"


def test_idle_gpu_is_selected_while_another_gpu_is_busy(monkeypatch):
    monkeypatch.setattr("autosim.research.compute_decision.discover",
                        lambda **_: report(gpu(0, used=9000, utilization=30.0), gpu(1)))

    decision = decide()

    assert decision.device == "cuda:0"  # pinned process ordinal, not the physical index
    assert decision.evidence["selected_device"]["uuid"] == "GPU-test-1"
    assert decision.evidence["selected_device"]["index"] == 1
    assert decision.evidence["busy_devices"][0]["index"] == 0
    assert decision.environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_small_idle_desktop_allocation_does_not_block_training(monkeypatch):
    monkeypatch.setattr("autosim.research.compute_decision.discover",
                        lambda **_: report(gpu(0, used=512, pids=[77])))

    decision = decide()

    assert decision.on_gpu
    assert decision.evidence["selected_device"]["uuid"] == "GPU-test-0"


def test_recheck_refuses_the_same_gpu_if_it_becomes_busy():
    decision = ComputeDecision(
        device="cuda:0", device_index=0,
        evidence={"selected_device": {"index": 0, "uuid": "GPU-test-0"}})

    with pytest.raises(NoCompatibleDevice, match="became busy"):
        recheck_gpu(decision, discoverer=lambda: report(
            gpu(0, used=8192, utilization=41.0, pids=[99])))


def test_recheck_waits_briefly_for_stale_utilization_after_a_completed_stage():
    decision = ComputeDecision(
        device="cuda:0", device_index=0,
        evidence={"selected_device": {"index": 0, "uuid": "GPU-test-0"}})
    observations = iter((report(gpu(0, utilization=3.0)), report(gpu(0))))
    calls = []

    def discoverer():
        calls.append(True)
        return next(observations)

    fresh = recheck_gpu(decision, discoverer=discoverer, settle_seconds=0.5)

    assert len(calls) == 2
    assert fresh["observed_device"]["utilization_gpu_pct"] == 0.0
    assert 0 < fresh["settle_wait_seconds"] <= 0.5
