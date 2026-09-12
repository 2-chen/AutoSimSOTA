from autosim.research.score_push_recovery import MAX_STARTUP_ATTEMPTS, NATIVE_STARTUP_CODES


def test_collection_recovery_is_native_startup_only_and_bounded():
    assert NATIVE_STARTUP_CODES == {-11, -6}
    assert MAX_STARTUP_ATTEMPTS == 3
