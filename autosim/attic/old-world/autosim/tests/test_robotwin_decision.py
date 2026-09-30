import pytest

from autosim.research.robotwin_decision import decide, validate


def evidence(clean=0.8, randomized=0.2):
    def row(identifier, score):
        return {"evidence_id": identifier,
                "metrics": {"summary": {"success_rate": score}},
                "failure_categories": {}, "skipped_seed_reasons": {}}
    return {"demo_clean": row("clean", clean), "demo_randomized": row("random", randomized)}


@pytest.mark.parametrize("controller", ["fixed", "random", "heuristic"])
def test_controls_share_valid_schema(controller, tmp_path):
    proposal, provenance = decide(controller, evidence(), {}, seed=1000, project_root=tmp_path)
    assert validate(proposal, proposal["evidence_id"]) == proposal
    assert provenance["external_request"] is False


def test_heuristic_targets_observed_randomized_gap(tmp_path):
    proposal, _ = decide("heuristic", evidence(), {}, seed=1, project_root=tmp_path)
    assert proposal["collection"]["setting"] == "demo_randomized"


def test_api_egress_requires_explicit_permission(tmp_path):
    with pytest.raises(PermissionError, match="allow-api-egress"):
        decide("api", evidence(), {}, seed=1, project_root=tmp_path)
