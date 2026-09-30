import json

import pytest

from autosim.research.common import atomic_json
from autosim.research.supervisor import build_audit_packet, validate_response


def test_audit_packet_omits_confirmation_values_and_uses_logical_refs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "output"
    research = output / "research" / "run-a"
    research.mkdir(parents=True)
    atomic_json(research / "research_report.json", {
        "run_id": "run-a", "repo": str(repo), "run_status": "completed",
        "objective": {"metric": "success"}, "rounds": [],
        "confirmation": {"status": "available_not_taken", "metric_value": 0.987,
                         "comparison": {"verdict": "candidate_better"}},
    })
    atomic_json(research / "comparison_protocol.json", {
        "frozen": True, "settings": {"seed": 7}, "confirmation": {"seed": 991},
    })
    atomic_json(research / "rubric.json", {"primary_metric": "success"})
    atomic_json(research / "ideas.json", {"ideas": []})
    atomic_json(research / "snapshots.json", {"snapshots": [], "best": "baseline"})
    (research / "measurements").mkdir()
    atomic_json(research / "measurements" / "confirmation.json", {
        "label": "confirmation", "ok": True, "confirmation": True,
        "metric_value": 0.987, "settings": {"held_out_seed": 991},
        "comparison": {"verdict": "candidate_better"},
    })

    packet, refs = build_audit_packet(output=output, repo=repo, run_id="run-a")

    serialized = json.dumps(packet, sort_keys=True)
    assert "0.987" not in serialized
    assert "991" not in serialized
    assert "candidate_better" not in serialized
    assert packet["frozen_protocol"] == {"frozen": True, "settings": {"seed": 7}}
    assert packet["confirmation"] == {"recorded": True, "status": "available_not_taken"}
    assert packet["confirmation_receipts"]
    assert "metric_value" not in packet["confirmation_receipts"][0]
    assert packet["evidence_refs"] == refs
    assert "research_report" in refs
    assert not any("/" in ref for ref in refs)
    assert "videos" not in packet and "demonstrations" not in packet


def test_completed_report_identity_must_match(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "output"
    research = output / "research" / "run-a"
    research.mkdir(parents=True)
    atomic_json(research / "research_report.json", {
        "run_id": "other-run", "repo": str(repo), "run_status": "completed"})

    with pytest.raises(ValueError, match="identity"):
        build_audit_packet(output=output, repo=repo, run_id="run-a")


def test_claimed_real_requires_a_consistently_verified_cited_measurement():
    packet = {"measurements": [{"receipt_verification": {"status": "consistent"}}]}
    refs = ["research_report", "measurement:baseline"]
    answer = {"verdict": "real", "summary": "The measurement is auditable.",
              "findings": [], "evidence_refs": ["measurement:baseline"]}

    review = validate_response(answer, allowed_refs=refs, packet=packet)

    assert review["verdict"] == "real"
    assert review["deterministic_measurement_check"]["status"] == "real"


def test_real_is_downgraded_without_verified_measurements_or_citations():
    packet = {"measurements": []}
    answer = {"verdict": "real", "summary": "Looks valid.", "findings": [],
              "evidence_refs": []}

    review = validate_response(answer, allowed_refs=["research_report"], packet=packet)

    assert review["verdict"] == "uncertain"
    assert review["deterministic_measurement_check"]["status"] == "uncertain"


def test_contradictory_receipt_forces_invalid_and_unknown_reference_is_rejected():
    packet = {"measurements": [{"receipt_verification": {"status": "inconsistent"}}]}
    answer = {"verdict": "real", "summary": "The evidence is sound.", "findings": [],
              "evidence_refs": ["measurement:baseline", "/private/output/measurement.json"]}

    with pytest.raises(ValueError, match="evidence"):
        validate_response(answer, allowed_refs=["measurement:baseline"], packet=packet)
    answer["evidence_refs"] = ["measurement:baseline"]
    review = validate_response(answer, allowed_refs=["measurement:baseline"], packet=packet)
    assert review["verdict"] == "invalid"
