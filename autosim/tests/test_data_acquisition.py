import copy
import json

import pytest

from autosim.research.data_acquisition import acquisition_proposal, failure_case
from autosim.research.main_agent import validate_report, memory_view
from tests.test_main_agent import REPORT, Client, make


def proposal():
    return {"target": "Increase legal training pose coverage near development failures",
        "sampling_unit": "full_episode",
        "supervision": {"kind": "reset_expert", "domain": "declared training domain",
                        "takeover": "not_required", "evidence_refs": ["expert-source"]},
        "native_binding": {"recipe_ref": "collect-recipe", "source_refs": ["task-source"],
            "controls": [{"path": "scene.pose_range", "value": [-.1, .1], "unit": "metres",
                          "legal_domain": "within native allowed range", "consumer_ref": "reset-source"}]},
        "target_predicate": "native recorded poses in the declared region",
        "budget": {"axis": "attempts", "limit": 4, "yield_estimate": "unknown; small probe first",
                   "downstream_reserve": "training and evaluation must still fit"},
        "data_context": "complete episodes with native action labels",
        "comparison": "same expert/Q0/recipe, random and rule targets",
        "falsification": ["actual target coverage unchanged"], "fallback": "inspect missing routing",
        "evidence_refs": ["development-evidence"]}


def test_plan_is_advisory_and_never_marks_native_readiness():
    result = acquisition_proposal(proposal())
    assert result["status"] == "needs_native_probe"
    assert "authorization" in result["authority"]


def test_reset_expert_cannot_be_promoted_to_midtrajectory_supervision():
    value = proposal()
    value["sampling_unit"] = "recovery_segment"
    result = acquisition_proposal(value)
    assert result["status"] == "needs_evidence"
    assert any("takeover" in x for x in result["unmet_prerequisites"])


@pytest.mark.parametrize("fault", ["bool_budget", "nan", "fractional_attempts", "host_path", "suffix_as_episode", "extra_authority"])
def test_malformed_or_ambiguous_control_and_budget_is_refused(fault):
    value = proposal()
    if fault == "bool_budget": value["budget"]["limit"] = True
    if fault == "nan": value["budget"]["limit"] = float("nan")
    if fault == "fractional_attempts": value["budget"]["limit"] = 1.5
    if fault == "host_path": value["native_binding"]["controls"][0]["path"] = "../source/config.yml"
    if fault == "suffix_as_episode":
        value["sampling_unit"] = "recovery_segment"
        value["budget"]["axis"] = "accepted_episodes"
    if fault == "extra_authority": value["status"] = "ready"
    with pytest.raises(ValueError): acquisition_proposal(value)


def test_failure_observations_and_hypotheses_remain_separate():
    result = failure_case({"development_ref": "baseline", "evidence_refs": ["native-log"],
        "observations": ["object slipped"], "hypotheses": ["coverage may be insufficient"],
        "unknowns": ["expert takeover reliability"], "diagnostic_probe": "inspect consumer semantics"})
    assert result["observations"] != result["hypotheses"]
    assert "not_causal_verdict" in result["authority"]


def test_total_compute_proposal_keeps_resource_dimensions_separate():
    value = proposal()
    value['budget'].update(axis='total_compute', limit={'gpu_seconds':30, 'model_usd':.2})
    assert acquisition_proposal(value)['budget']['limit']['model_usd'] == .2
    value['budget']['limit'] = 30
    with pytest.raises(ValueError): acquisition_proposal(value)


def test_full_report_survives_data_role_and_bounded_main_handoff(tmp_path):
    class DataClient(Client):
        def chat_with_metadata(self, system, user, **kwargs):
            if "assignment" in json.loads(user):
                return json.dumps({**REPORT, "acquisition_proposals": [proposal()]}), {}
            return super().chat_with_metadata(system, user, **kwargs)

    real = make(tmp_path, DataClient())
    result = real._step_research_task(role="data", task="choose a data target",
        expected_result="source-grounded proposal", mode="inspect")
    handoff = memory_view(real.main_agent)["data_agent_handoff"]
    assert handoff["report"]["acquisition_proposals"][0]["status"] == "needs_native_probe"
    from autosim.research.evidence_store import read_attempt_evidence
    page = read_attempt_evidence(real.output, handoff["report_evidence_id"])
    assert "scene.pose_range" in page["text"]
    assert (real.output / handoff["report_ref"]).is_file()
    assert result["verification_level"] is None
    assert not (real.output / "native_jobs").exists()
