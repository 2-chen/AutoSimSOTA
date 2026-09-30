"""Holding down the sentence a run has to be able to say about its own best number.

A loop that reports "0.58 against a 0.41 baseline" has reported its incumbent and not what it
did to get it. The incumbent was chosen *because* it was the highest of the arms measured, so
the number is a maximum, and the reader is the one who has to know that. These tests are about
the record saying it without being asked.

The other half is the count itself. An arm that failed is still an arm the loop took: it had
the same cost and its failure is what the next round was chosen from. A record that counts
only the arms that produced numbers understates the search, and understating the search is
exactly the direction that makes a result look better.
"""

import json

from autosim.research.selection import note, rate_of, summary


def _run(tmp_path):
    return tmp_path / "run"


def test_the_best_number_arrives_with_what_it_is_the_maximum_of(tmp_path):
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.41, metric_utility=0.41, baseline=True, settings={"episodes": 40})
    note(root, protocol_sha256="a" * 64, label="candidate-3", started=True,
         metric_value=0.58, metric_utility=0.58, settings={"episodes": 40})
    result = summary(root)
    assert result["best"]["label"] == "candidate-3"
    assert result["best_is_the_maximum_of"] == 2
    assert "best of 2 arms" in result["reading"]
    assert "maximum of draws" in result["reading"]


def test_a_failed_arm_is_counted_as_a_draw_the_loop_took(tmp_path):
    """The understating direction. Five arms were started and two produced numbers; a record
    that says "best of two" has described a search of five as a search of two."""
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.41, metric_utility=0.41, baseline=True, settings={"episodes": 40})
    for index in range(4):
        note(root, protocol_sha256="a" * 64, label=f"candidate-{index}", started=True,
             metric_value=None, metric_utility=None, settings={"episodes": 40})
    note(root, protocol_sha256="a" * 64, label="candidate-4", started=True,
         metric_value=0.52, metric_utility=0.52, settings={"episodes": 40})
    result = summary(root)
    assert result["attempted"] == 6
    assert result["scored"] == 2
    assert result["failed"] == 4
    assert result["best_is_the_maximum_of"] == 2
    assert "4 further arms were started and produced no number" in result["reading"]


def test_one_arm_is_a_measurement_and_says_so(tmp_path):
    """The record must not call every result a maximum. A run that measured once has measured
    once, and telling its reader to distrust it as a best-of-many is its own kind of wrong."""
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="only", started=True,
         metric_value=0.41, metric_utility=0.41, baseline=True, settings={"episodes": 40})
    result = summary(root)
    assert result["best_is_the_maximum_of"] == 1
    assert "a measurement rather than a maximum" in result["reading"]
    assert "maximum of draws" not in result["reading"]


def test_the_difference_against_the_baseline_carries_an_interval(tmp_path):
    """The whole reason the count is worth recording: with it, the difference can be weighed,
    and 0.58 against 0.41 over forty episodes reads very differently from 0.52 against 0.48."""
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.4, metric_utility=0.4, baseline=True, settings={"episodes": 40},
         actual_episodes=40, successes=16)
    note(root, protocol_sha256="a" * 64, label="candidate", started=True,
         metric_value=0.575, metric_utility=0.575, settings={"episodes": 40},
         actual_episodes=40, successes=23)
    comparison = summary(root)["comparison"]
    assert comparison["verdict"] == "no_difference"
    low, high = comparison["interval"]
    assert low < 0 < high
    assert f"{low:+.3f} to {high:+.3f}" in comparison["why"], "the interval is quoted"
    assert comparison["counts_reconstructed"] is False
    assert "native completed episode" in comparison["counts_from"]


def test_a_difference_larger_than_the_interval_is_reported_as_one(tmp_path):
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.10, metric_utility=0.10, baseline=True, settings={"episodes": 100},
         actual_episodes=100, successes=10)
    note(root, protocol_sha256="a" * 64, label="candidate", started=True,
         metric_value=0.80, metric_utility=0.80, settings={"episodes": 100},
         actual_episodes=100, successes=80)
    comparison = summary(root)["comparison"]
    assert comparison["verdict"] == "better"
    assert comparison["interval"][0] > 0
    assert comparison["counts_reconstructed"] is False, "80 of 100 divides evenly"


def test_no_denominator_means_no_interval_rather_than_a_guessed_one(tmp_path):
    """An interval built on an assumed episode count is a fabricated interval, and it would
    be indistinguishable in the record from one built on the real count."""
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.41, metric_utility=0.41, baseline=True, settings={})
    note(root, protocol_sha256="a" * 64, label="candidate", started=True,
         metric_value=0.58, metric_utility=0.58, settings={})
    comparison = summary(root)["comparison"]
    assert comparison["verdict"] == "not_established"
    assert "native completed episode counts" in comparison["why"]
    assert "interval" not in comparison


def test_a_run_that_measured_nothing_says_that_rather_than_reporting_zero(tmp_path):
    """The shape this repository keeps producing: a number reported where none was measured.
    A run whose every arm died has findings, and its result is not 0.0."""
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="candidate", started=True,
         metric_value=None, metric_utility=None, settings={"episodes": 40})
    result = summary(root)
    assert result["verdict"] == "nothing_measured"
    assert result["best"] is None
    assert "no result to report" in result["reading"]
    assert "failures" in result["reading"]


def test_arms_reading_identically_is_stated_as_the_ambiguity_it_is(tmp_path):
    """Several arms at exactly the same number is either a benchmark that cannot separate
    them or a loop re-measuring one unchanged policy. Both readings matter and neither is an
    improvement, so the record says the ambiguity instead of picking."""
    root = _run(tmp_path)
    for index in range(4):
        note(root, protocol_sha256="a" * 64, label=f"candidate-{index}", started=True,
             metric_value=0.31, metric_utility=0.31, settings={"episodes": 40})
    result = summary(root)
    assert result["identical_reading"] is True
    assert "cannot distinguish from the arms being the same" in result["reading"]


def test_the_baseline_is_compared_against_itself_and_not_reported_as_a_gain(tmp_path):
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.41, metric_utility=0.41, baseline=True, settings={"episodes": 40})
    result = summary(root)
    assert result["comparison"]["verdict"] == "not_established"
    assert "nothing to compare" in result["comparison"]["why"]


def test_the_record_survives_a_restart_because_it_is_the_file_that_persists(tmp_path):
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="first", started=True,
         metric_value=0.41, metric_utility=0.41, baseline=True, settings={"episodes": 40})
    reopened = summary(root)
    assert reopened["attempted"] == 1
    note(root, protocol_sha256="a" * 64, label="second", started=True,
         metric_value=0.50, metric_utility=0.50, settings={"episodes": 40})
    assert summary(root)["attempted"] == 2


def test_a_corrupt_record_is_treated_as_absent_rather_than_crashing_the_run(tmp_path):
    """The record is written after the measurement, so losing it costs a run its accounting
    and must not cost it the measurement it just took."""
    root = _run(tmp_path)
    (root).mkdir(parents=True, exist_ok=True)
    (root / "selection_record.json").write_text("{not json", encoding="utf-8")
    result = summary(root)
    assert result["attempted"] == 0
    assert result["verdict"] == "nothing_measured"
    note(root, protocol_sha256="a" * 64, label="after", started=True,
         metric_value=0.4, metric_utility=0.4, settings={"episodes": 40})
    assert summary(root)["attempted"] == 1


def test_the_protocol_hash_travels_with_the_arms_so_two_searches_are_not_pooled(tmp_path):
    """Arms measured under a different protocol were measured against a different benchmark.
    Keeping the hash in the record is what lets a reader see which search a number belongs
    to instead of taking the file's word that they were the same."""
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="under-a", started=True,
         metric_value=0.4, metric_utility=0.4, settings={"episodes": 40})
    result = summary(root)
    assert result["protocol_sha256"] == "a" * 64
    assert json.loads((root / "selection_record.json").read_text())["arms"][0]["label"] \
        == "under-a"


def test_a_rate_with_a_denominator_gets_its_interval_and_one_without_gets_none(tmp_path):
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="with", started=True, metric_value=0.5,
         metric_utility=0.5, settings={"n_eval_episodes": 20},
         actual_episodes=20, successes=10)
    note(root, protocol_sha256="a" * 64, label="without", started=True, metric_value=0.5,
         metric_utility=0.5, settings={})
    arms = summary(root)["arms"]
    with_count = next(arm for arm in arms if arm["label"] == "with")
    without = next(arm for arm in arms if arm["label"] == "without")
    reading = rate_of(with_count)
    assert reading["n"] == 20
    assert reading["interval"][0] < 0.5 < reading["interval"][1]
    assert rate_of(without) is None


def test_requested_episode_count_cannot_create_an_interval(tmp_path):
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.25, metric_utility=0.25, baseline=True,
         settings={"episodes": 100})
    note(root, protocol_sha256="a" * 64, label="candidate", started=True,
         metric_value=0.75, metric_utility=0.75, settings={"episodes": 100})
    result = summary(root)
    assert result["comparison"]["verdict"] == "not_established"
    assert rate_of(result["best"]) is None


def test_a_continuous_metric_is_not_forced_into_a_proportion(tmp_path):
    """A return of 1.7 is not a rate and rounding it into one would invent episodes."""
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="return", started=True, metric_value=1.7,
         metric_utility=1.7, settings={"episodes": 40})
    assert rate_of(summary(root)["best"]) is None


def test_a_tie_goes_to_the_baseline_whichever_order_the_arms_arrived_in(tmp_path):
    """An arm reading exactly what the run started from has not improved on it. Left to
    insertion order the winner of a tie is whichever arm was recorded first, so a candidate
    measured before the baseline would be reported as a gain over it."""
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="candidate", started=True,
         metric_value=0.41, metric_utility=0.41, settings={"episodes": 40})
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.41, metric_utility=0.41, baseline=True, settings={"episodes": 40})
    assert summary(root)["best"]["label"] == "baseline"


def test_an_arm_recovered_from_its_measurement_file_still_counts(tmp_path):
    """A run that predates the record -- or lost it -- still took its arms and their files are
    still on disk. Reading only the record made an eight-arm run report `attempted: 0`, and
    understating the search is the flattering direction: it is the count that decides whether
    the best number is a measurement or a maximum of draws.
    """
    root = _run(tmp_path)
    (root / "measurements").mkdir(parents=True)
    for label, value in (("baseline", 0.41), ("round_0", None), ("round_1", None),
                         ("round_2", 0.55)):
        (root / "measurements" / f"{label}.json").write_text(json.dumps({
            "label": label, "metric_value": value,
            "metric_utility": value, "settings": {"episodes": 40}}), encoding="utf-8")
    result = summary(root)
    assert result["attempted"] == 4
    assert result["scored"] == 2
    assert result["failed"] == 2
    assert result["counted_from"] == ["measurement files"]
    assert result["best_is_the_maximum_of"] == 2
    assert result["baseline"]["label"] == "baseline"


def test_a_label_the_record_knows_is_not_counted_twice_from_its_file(tmp_path):
    """The record may hold several entries for one label where the file holds only the last,
    so the record wins where it speaks. Counting both would inflate the search instead."""
    root = _run(tmp_path)
    (root / "measurements").mkdir(parents=True)
    (root / "measurements" / "round_0.json").write_text(json.dumps({
        "label": "round_0", "metric_value": 0.5, "metric_utility": 0.5,
        "settings": {}}), encoding="utf-8")
    note(root, protocol_sha256="a" * 64, label="round_0", started=True,
         metric_value=0.5, metric_utility=0.5, settings={})
    note(root, protocol_sha256="a" * 64, label="round_0", started=True,
         metric_value=0.6, metric_utility=0.6, settings={})
    result = summary(root)
    assert result["attempted"] == 2, "the file's single arm does not re-add the label"
    assert result["counted_from"] == ["selection record"]


def test_the_sources_the_count_came_from_travel_with_it(tmp_path):
    """A reader has to be able to tell an exact count from a reconstructed one."""
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.4, metric_utility=0.4, baseline=True, settings={"episodes": 40})
    (root / "measurements").mkdir(parents=True, exist_ok=True)
    (root / "measurements" / "orphan.json").write_text(json.dumps({
        "label": "orphan", "metric_value": 0.3, "metric_utility": 0.3,
        "settings": {}}), encoding="utf-8")
    result = summary(root)
    assert result["attempted"] == 2
    assert result["counted_from"] == ["selection record", "measurement files"]


def test_a_corrupt_measurement_file_does_not_take_the_count_with_it(tmp_path):
    """One unreadable arm file is one arm's worth of uncertainty, not the run's whole count."""
    root = _run(tmp_path)
    (root / "measurements").mkdir(parents=True)
    (root / "measurements" / "good.json").write_text(json.dumps({
        "label": "good", "metric_value": 0.4, "metric_utility": 0.4,
        "settings": {"episodes": 10}}), encoding="utf-8")
    (root / "measurements" / "broken.json").write_text("{nope", encoding="utf-8")
    assert summary(root)["attempted"] == 1


def test_arms_measured_under_different_settings_are_not_a_best_of_anything(tmp_path):
    """"Best of N" presumes the N were measured the same way. Where they were not, calling
    one of them the best of the others is a comparison the run never fixed -- and the count
    of distinct setting sets is the cheapest way to find that out.

    This is not hypothetical: the RoboTwin run in this repository has eight arms measured
    under two different setting sets, and nothing in its report said so.
    """
    root = _run(tmp_path)
    note(root, protocol_sha256="a" * 64, label="baseline", started=True,
         metric_value=0.41, metric_utility=0.41, baseline=True,
         settings={"episodes": 40, "train.n_epochs": 1})
    note(root, protocol_sha256="a" * 64, label="round_0", started=True,
         metric_value=0.58, metric_utility=0.58,
         settings={"episodes": 40, "train.n_epochs": 1, "num_epochs": 6000})
    result = summary(root)
    assert result["distinct_settings"] == 2
    assert result["one_protocol"] is False
    assert "2 distinct setting sets" in result["reading"]
    assert "not comparable" in result["reading"]


def test_a_frozen_protocol_over_one_setting_set_is_one_protocol(tmp_path):
    root = _run(tmp_path)
    for label, value in (("baseline", 0.41), ("round_0", 0.58)):
        note(root, protocol_sha256="a" * 64, label=label, started=True, metric_value=value,
             metric_utility=value, baseline=label == "baseline",
             settings={"episodes": 40, "train.n_epochs": 1})
    result = summary(root)
    assert result["one_protocol"] is True
    assert result["distinct_settings"] == 1
    assert "not all measured under one frozen protocol" not in result["reading"]


def test_arms_with_no_frozen_protocol_say_the_count_is_of_arms_not_draws(tmp_path):
    """Eight arms taken by hand across restarts is a real thing a run directory contains, and
    it is not a best-of-eight: nothing fixed that they were measured the same way."""
    root = _run(tmp_path)
    for label, value in (("baseline", 0.41), ("round_0", 0.58)):
        note(root, protocol_sha256="", label=label, started=True, metric_value=value,
             metric_utility=value, baseline=label == "baseline", settings={"episodes": 40})
    result = summary(root)
    assert result["one_protocol"] is False
    assert "no protocol was frozen for this run" in result["reading"]
    assert "arms taken rather than of comparable draws" in result["reading"]
