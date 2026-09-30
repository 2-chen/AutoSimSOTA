from autosim.research.score_push import CAMPAIGN, OFFSETS


def test_score_push_task_scope_and_gate_budgets():
    assert "click_bell" not in CAMPAIGN
    assert set(CAMPAIGN) == {
        "handle_basket", "water_pouring", "manipulate_pipette", "items_handover",
        "drawer_open_place", "mixer_operating", "table_rearrangement",
        "sample_loading", "item_assembly",
    }
    for task, row in CAMPAIGN.items():
        if not row.get("requires_expert_repair"):
            assert row["cap_attempts"] == 100
            assert row["production_target"] > 0
            assert row["production_attempts"] >= row["production_target"]
    assert len(set(OFFSETS.values())) == len(OFFSETS)
