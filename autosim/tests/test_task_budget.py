import json
import pytest
from autosim.research.task_budget import TaskGPUBudget


def test_new_studies_independent_resume_shared_and_crash_keeps_hold(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first, second = TaskGPUBudget(a), TaskGPUBudget(b)
    for budget in (first, second):
        budget.initialize(tmp_path / "same_repo", cap_seconds=100)
    first.reserve("train", "GPU-test")
    with pytest.raises(ValueError, match="exhausted or reserved"):
        TaskGPUBudget(a).reserve("resume", "GPU-test")
    assert second.reserve("train", "GPU-test")["reserved_seconds"] == 100
    first.finish("train", 12)
    first.finish("train", 12)  # idempotent, no double billing
    assert TaskGPUBudget(a).reserve("eval", "GPU-test")["reserved_seconds"] == 88
    assert json.loads(first.path.read_text())["charged_seconds"] == 12


def test_gpu_budget_scope_and_cap_cannot_be_changed_on_resume(tmp_path):
    budget = TaskGPUBudget(tmp_path)
    budget.initialize(tmp_path / "repo")
    with pytest.raises(ValueError, match="cannot change"):
        budget.initialize(tmp_path / "other")
    with pytest.raises(ValueError):
        budget.initialize(tmp_path / "repo", cap_seconds=float("nan"))


def test_model_wait_does_not_charge_gpu(tmp_path, monkeypatch):
    from autosim.research import task_budget
    clock = [1.0]
    monkeypatch.setattr(task_budget.time, "time", lambda: clock[0])
    budget = TaskGPUBudget(tmp_path)
    budget.initialize(tmp_path / "repo")
    clock[0] += 10000
    assert budget.reserve("first_gpu_work", "GPU-test")["reserved_seconds"] == 86400
