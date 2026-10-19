import pytest

from autosim.research.context_pressure import rotation_decision


@pytest.mark.parametrize('hit,miss,out,status,rotate', [
    (150000, 10000, 2000, 'priced', True),
    (95000, 5000, 2000, 'priced', False),
    (115000, 5000, 2000, 'priced', True),
    (137216, 1378, 6773, 'priced', True),
    (230000, 10000, 2000, 'priced', True),
    (150000, 15000, 2000, 'priced', True),
    (110000, 20000, 2000, 'priced', True),
    (150000, 10000, 9000, 'priced', True),
    (150000, 10000, 2000, 'unknown', True),
    (90000, 10000, 2000, 'priced', False),
])
def test_only_bounded_known_warm_history_can_cross_soft_rotation_threshold(hit, miss, out, status, rotate):
    result = rotation_decision({'status': status, 'usage': {
        'input_hit_tokens': hit, 'input_miss_tokens': miss, 'output_tokens': out}})
    assert result['rotate'] is rotate
    assert result['effective_limit_tokens'] <= 120000


@pytest.mark.parametrize('bad', [-1, True, '120000'])
def test_malformed_usage_rotates_without_inventing_cache_hits(bad):
    assert rotation_decision({'usage': {'input_hit_tokens': bad}})['rotate']
