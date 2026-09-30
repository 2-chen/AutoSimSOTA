from autosim.research.score_push_final import (
    FINAL_UPDATES,
    MIN_CONFIRMATION_GAIN,
    MIN_DEVELOPMENT_GAIN,
    TRAINING_SEED,
)


def test_final_stage_budget_and_thresholds_are_frozen():
    assert FINAL_UPDATES == 80_000
    assert TRAINING_SEED == 1000
    assert MIN_DEVELOPMENT_GAIN == .03
    assert MIN_CONFIRMATION_GAIN == .03
