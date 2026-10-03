import json

import pytest

from autosim.research.token_context import compact_json, project_context
from autosim.research.evidence_store import read_attempt_evidence


def test_context_paging_is_readable_and_preserves_protocol_errors(tmp_path):
    source = "source-" * 2000
    history = [{"round": i, "text": "measured-result-" * 6} for i in range(100)]
    context = {"history": history, "source": source,
               "comparison_protocol": {"rules": list(range(100))},
               "failure": source, "budget": {"remaining": 2}}
    result, metrics = project_context(context, "{}", tmp_path)
    assert result["comparison_protocol"] == context["comparison_protocol"]
    assert result["failure"] == source
    assert result["budget"] == context["budget"]
    assert result["history"]["preview"] == history[-6:]
    ref = result["history"]["evidence_id"]
    chunks, offset = [], 0
    while True:
        evidence = read_attempt_evidence(tmp_path, ref, offset=offset)
        chunks.append(evidence["text"])
        offset = evidence["next_offset"]
        if offset == evidence["log_bytes"]:
            break
    assert json.loads("".join(chunks)) == history
    assert metrics["paged_leaves"] == 2
    assert metrics["projected_context_bytes"] < metrics["original_context_bytes"]
    assert project_context(context, "{}", tmp_path)[0] == result


def test_deduplication_is_exact_and_toolless_context_is_not_paged(tmp_path):
    value = list(range(100))
    result, metrics = project_context({"history": value, "budget": 10},
        json.dumps({"context": {"budget": 10}}), tmp_path, pageable=False)
    assert result == {"history": value}
    assert metrics["deduplicated_fields"] == 1
    assert metrics["paged_leaves"] == 0
    # Conflicting request examples must not suppress the authoritative snapshot.
    result, _ = project_context({"budget": 10},
        '{"budget":10,"example":{"budget":0}}', tmp_path)
    assert result["budget"] == 10
    assert compact_json({"b": 2, "a": 1}) == compact_json({"a": 1, "b": 2})


def test_paging_rejects_symlink_parent_before_write(tmp_path):
    outside = tmp_path / "other"
    outside.mkdir()
    (tmp_path / "agent").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        project_context({"history": ["entry" * 50] * 100}, "{}", tmp_path)
    assert not list(outside.iterdir())
