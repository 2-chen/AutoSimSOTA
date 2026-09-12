from autosim.research.stage1_summary import render


def test_render_distinguishes_execution_from_hypothesis():
    result = {
        "execution_complete": True,
        "result_valid": True,
        "hypothesis_supported": False,
        "findings": {
            "bounded_expert_yield": {"full_random": {
                "accepted": 1, "attempts": 2, "rate": .5, "wilson_95": [.1, .9]}},
            "learning_probe_mean_success": {"parent": .5, "proportional": .45,
                                             "correction_10pct": .4875},
            "correction_10pct_minus_proportional": .0375,
            "correction_10pct_minus_parent": -.0125,
        },
        "decision": {"status": "stop_current_candidate_before_stage_2",
                     "reason": "frozen gate failed", "prohibited_follow_up": "do not extend"},
        "budget": {"conservative_all_process_wall_hours": 1.5},
        "test_suite": {"passed": 1, "failures": 0, "errors": 0},
    }
    text = render(result)
    assert "Execution complete: **True**" in text
    assert "Current mechanism hypothesis supported: **False**" in text
