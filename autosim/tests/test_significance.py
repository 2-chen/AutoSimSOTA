"""The failure these hold down is a loop that reports its best draw as an improvement.

A research loop measures candidates, keeps the highest number, and calls the difference an
improvement. Every step of that is arithmetically correct and the conclusion does not follow:
the highest of several noisy measurements is above the noise by construction, and a benchmark
run twice on the same policy does not print the same number. So the tests below are less
about arithmetic -- the arithmetic is short -- than about what the module refuses to say.

Two defects get their own tests because both are the shape this repository keeps producing: a
check that passes on a broken state, and a guard that kills what it guards. The first is an
interval that excludes zero because there was one observation; the second is a comparison
that raises on the input it exists to describe.
"""

import pytest

from autosim.research.significance import (
    DIRECTIONAL_BELOW, across, compare, spread_between_runs)


# --- what it refuses to say ------------------------------------------------------------

def test_the_same_policy_measured_twice_is_not_an_improvement():
    """The central case. These two lists come from one distribution, so a loop that kept the
    higher of them would report a gain that does not exist."""
    first = [1, 1, 0, 1, 0, 0, 1, 0] * 5
    second = [1, 0, 1, 0, 0, 1, 1, 0] * 5
    verdict = compare(first, second, kind="binary", paired=True)
    assert verdict.verdict == "no_difference"
    assert verdict.interval[0] < 0 < verdict.interval[1], "an interval that excludes zero"
    assert "crosses zero" in verdict.why


def test_a_small_sample_is_a_direction_and_not_a_result():
    """Six against one at n=4 is what an early round looks like, and the interval at n=4 is
    wide enough to contain almost anything. Reporting it as better is the failure."""
    verdict = compare([0, 1, 0, 0], [1, 1, 1, 1], kind="binary", paired=True)
    assert verdict.difference > 0.5
    assert verdict.verdict == "directional_only"
    assert "smoke test" in verdict.why


def test_a_real_difference_at_a_real_sample_size_is_believed():
    """The ladder has to reach the top or it is only a refusal machine: a policy that fails
    everywhere against one that succeeds everywhere, over enough paired episodes."""
    verdict = compare([0.0] * 20, [1.0] * 20, kind="binary", paired=True)
    assert verdict.verdict == "better"
    assert verdict.interval == (1.0, 1.0)
    assert verdict.p_value is not None and verdict.p_value < 0.05


def test_the_verdict_ladder_is_ordered_and_nothing_skips_a_rung():
    """`directional_only` outranks `no_difference` even when the sample difference looks
    decisive, because the count is what limits the claim -- and a small sample whose interval
    happens to exclude zero must not be promoted past it."""
    small = compare([0.0] * 3, [1.0] * 3, kind="binary", paired=True)
    assert small.verdict == "directional_only"
    wide = compare([0, 1] * 20, [1, 0] * 20, kind="binary", paired=True)
    assert wide.verdict == "no_difference"


# --- the shape of the answer ------------------------------------------------------------

def test_every_comparison_carries_what_it_does_not_say():
    """The numbers get quoted later, in a document, away from the code that produced them --
    so the limit on them travels with them rather than staying in a comment here."""
    verdict = compare([1.0] * 12, [1.0] * 10 + [0.0] * 2, kind="binary", paired=True)
    assert verdict.limitation
    assert "not a claim about" in verdict.limitation
    assert "the episodes that were run" in verdict.limitation
    assert verdict.as_dict()["limitation"] == verdict.limitation


def test_an_unpaired_difference_says_the_episodes_were_not_the_same():
    verdict = compare([0.0] * 20, [1.0] * 20, kind="binary", paired=False)
    assert verdict.verdict == "better"
    assert "not the same episodes" in verdict.limitation


def test_the_same_input_gives_the_same_interval_twice():
    """An interval that changes between two calls is an interval no record can quote."""
    args = ([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    assert (compare(*args).interval == compare(*args).interval)


# --- the inputs it exists to describe, rather than reject -------------------------------

def test_a_binary_comparison_of_non_outcomes_says_so_instead_of_testing_for_anything():
    """A metric contract that hands a success rate to a binary comparison is a fault upstream,
    and the answer to it is a sentence -- not a significance test computed on 0.5."""
    verdict = compare([0.5, 0.5], [0.6, 0.6], kind="binary", paired=True)
    assert verdict.verdict == "insufficient"
    assert "not a success or a failure" in verdict.why
    assert verdict.interval is None, "no interval is reported for input that cannot have one"


def test_a_paired_comparison_of_unequal_lists_explains_rather_than_raising():
    verdict = compare([1.0, 0.0], [1.0, 0.0, 1.0], kind="binary", paired=True)
    assert verdict.verdict == "insufficient"
    assert "one reading per episode on both sides" in verdict.why


def test_an_empty_side_is_answered_not_crashed():
    assert compare([], [], kind="binary").verdict == "insufficient"
    assert compare([], [1.0], kind="continuous").verdict == "insufficient"


def test_a_continuous_metric_is_not_forced_through_a_binary_test():
    """Many benchmarks report a return, a distance or a completion fraction. Comparing those
    with McNemar would reject every value that is not 0 or 1 -- which is all of them."""
    verdict = compare([1.0, 1.2, 0.9, 1.1] * 4, [2.0, 2.2, 1.9, 2.1] * 4,
                      kind="continuous", paired=True)
    assert verdict.verdict == "better"
    assert verdict.p_value is None, "a bootstrap interval is not a p-value"
    assert "bootstrap" in verdict.test


# --- the scale a difference has to beat -------------------------------------------------

def test_running_the_same_thing_twice_has_a_spread_or_says_it_does_not():
    """The number a loop's improvement has to beat, and the only place it can come from: two
    runs of one condition."""
    one = spread_between_runs([0.51])
    assert one["range"] is None and "no spread" in one["limitation"]
    two = spread_between_runs([0.51, 0.62])
    assert two["range"] == pytest.approx(0.11)
    assert "0.1100" in two["limitation"]


def test_a_single_reading_has_no_interval_to_report():
    """A degenerate interval around one number reads as certainty, which is the opposite of
    what one reading is."""
    single = across([0.5])
    assert single["n"] == 1
    assert "nothing" not in str(single["interval"])
    assert single["limitation"]


def test_the_threshold_is_a_stated_number_rather_than_a_magic_one():
    """A reader has to be able to find the sample size at which this stops calling a
    comparison directional, and change it deliberately."""
    assert DIRECTIONAL_BELOW >= 2
    below = compare([0.0] * (DIRECTIONAL_BELOW - 1), [1.0] * (DIRECTIONAL_BELOW - 1))
    assert below.verdict == "directional_only"
    at = compare([0.0] * DIRECTIONAL_BELOW, [1.0] * DIRECTIONAL_BELOW)
    assert at.verdict == "better"
