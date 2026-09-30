import pytest


@pytest.fixture(autouse=True)
def preparation_contracts_use_explicit_cpu(monkeypatch):
    """The orchestration contract suite is hardware-independent by design."""
    from autosim.research import prepare
    from autosim.research.compute_decision import ComputeDecision

    monkeypatch.setattr(
        prepare, "decide",
        lambda **_: ComputeDecision(device="cpu", device_index=0,
                                    environment={"CUDA_VISIBLE_DEVICES": ""},
                                    why="the test explicitly uses CPU"))
