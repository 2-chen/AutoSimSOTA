from autosim.research.score_push_pipeline import SCREEN_ARMS


def test_screen_has_missing_training_control_and_two_data_arms():
    assert SCREEN_ARMS == ("official_continue", "mixed_proportional", "mixed_20pct")
