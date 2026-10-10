"""Data specialist selection is explicit, bounded and never a native execution."""
import json

import pytest

from autosim.research import data_agent


ASSIGNMENT = {"role": "data", "task": "Trace collector to batch", "expected_result": "probe"}


class Selector:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def chat_with_metadata(self, system, user, **kwargs):
        self.calls.append((system, json.loads(user), kwargs))
        return json.dumps(self.answer), {}


@pytest.fixture
def methods(monkeypatch):
    reads = []
    monkeypatch.setattr(data_agent, "skills_catalog", lambda **kw: {"entries": [
        {"id": "suggested", "recommended_by_keyword": True},
        {"id": "chosen", "recommended_by_keyword": False}]})
    def read(ids, **kwargs):
        reads.append(ids)
        return {"selection": [{"id": identity, "body_sha256": "a" * 64} for identity in ids],
                "skills": [{"method": "BODY_FOR_" + identity} for identity in ids]}
    monkeypatch.setattr(data_agent, "read_selected_skills", read)
    return reads


def test_data_selects_unrecommended_method_with_reason_before_loading_body(methods):
    client = Selector({"skill_reads": [{"id": "chosen", "why": "Need actual loader exposure"}]})
    result = data_agent.select_methods(client, ASSIGNMENT, {}, timeout=40)
    assert methods == [["chosen"]]
    system, payload, options = client.calls[0]
    assert "delegated Data Agent" in system and "main research Agent" not in system
    assert "BODY_FOR" not in json.dumps(payload)
    assert options["read_only"] and options["auto_skills"] is False
    assert options["timeout"] == 40 and len(client.calls) == 1
    selection = result["method_library"]["selection"][0]
    assert selection["selected_by"] == "data_agent"
    assert selection["selection_reason"] == "Need actual loader exposure"
    assert result["method_library"]["skills"][0]["method"] == "BODY_FOR_chosen"


@pytest.mark.parametrize("choices", [None, "chosen", [0], [{"id": "missing", "why": "why"}],
    [{"id": "chosen", "why": ""}], [{"id": "chosen", "why": "x" * 401}],
    [{"id": "chosen", "why": "why"}] * 4])
def test_invalid_data_skill_choices_do_not_load_guessed_bodies_or_repeat_model(methods, choices):
    client = Selector({"skill_reads": choices})
    result = data_agent.select_methods(client, ASSIGNMENT, {}, timeout=1)
    assert methods == [[]] and len(client.calls) == 1
    assert result["skill_selection_issues"]


def test_empty_data_selection_is_legal_and_duplicate_ids_are_not_read_twice(methods):
    result = data_agent.select_methods(Selector({"skill_reads": []}), ASSIGNMENT, {}, timeout=1)
    assert not result["skill_selection_issues"]
    result = data_agent.select_methods(Selector({"skill_reads": [
        {"id": "chosen", "why": "reason"}, {"id": "chosen", "why": "other"}]}),
        ASSIGNMENT, {}, timeout=1)
    assert methods[-1] == ["chosen"] and result["skill_selection_issues"]


def test_data_skill_selection_uses_original_local_window(monkeypatch):
    monkeypatch.setattr(data_agent.time, "monotonic", lambda: 125)
    assert data_agent.remaining_window(100, 40) == 15
    with pytest.raises(TimeoutError, match="local window exhausted"):
        data_agent.remaining_window(100, 25)
