"""Wall-clock budget is persisted across resumption and blocks new work."""

import pytest

from autosim.research import budget


def test_wall_budget_survives_resume_and_cannot_be_silently_changed(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(budget.time, "time", lambda: clock[0])
    first = budget.RunBudget(tmp_path, wall_seconds=10)
    clock[0] = 104.0
    assert first.remaining() == 6
    resumed = budget.RunBudget.existing(tmp_path)
    assert resumed is not None and resumed.remaining() == 6
    with pytest.raises(ValueError, match="cannot silently change"):
        budget.RunBudget(tmp_path, wall_seconds=20)
    clock[0] = 111.0
    assert resumed.record()["status"] == "exhausted"
    assert resumed.remaining() == 0


def test_model_request_is_capped_by_run_deadline(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(budget.time, "time", lambda: clock[0])
    held = budget.RunBudget(tmp_path, wall_seconds=10)

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            assert kwargs["timeout"] == 6
            assert kwargs["retries"] == 0
            return "{}", {}

    client = budget.BudgetedClient(Client(), held)
    clock[0] = 104.0
    assert client.chat_with_metadata("", "", timeout=300)[0] == "{}"
    clock[0] = 111.0
    with pytest.raises(TimeoutError, match="budget exhausted"):
        client.chat_with_metadata("", "", timeout=300)
