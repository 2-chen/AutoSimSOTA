from autosim.research.score_push_fallback import FINAL_UPDATES, MIN_GAIN, SCREEN_UPDATES, TRAINING_SEED


def test_fallback_is_fixed_budget_official_data_only_comparison():
    assert SCREEN_UPDATES == 20_000
    assert FINAL_UPDATES == 80_000
    assert TRAINING_SEED == 1000
    assert MIN_GAIN == .03
