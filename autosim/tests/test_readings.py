"""A number a program printed, and the line it came from.

The failure these hold down is not a wrong reading. It is a *right* reading that cannot be
checked, and a reduction nobody was told about: a program reporting one success rate per task
prints them separated by `|`, and the mean of them was stored as a bare float -- with nothing
saying it was a mean, over how many tasks, or what the per-task values were. That float is
what every candidate is ranked by.
"""

from autosim.research.readings import (named_numbers, success_rate, success_reading,
                                       training_progress, evaluation_progress)


def test_a_single_reading_carries_its_line():
    reading = success_reading("epoch 3\nsucc: 0.62\n")
    assert reading["value"] == 0.62
    assert reading["averaged"] is False
    assert "succ: 0.62" in reading["read_from"]


def test_a_row_of_per_task_rates_says_it_was_reduced():
    """The mean of three tasks and one task scoring the mean are different findings about a
    policy, and a bare float cannot tell a reader which they are holding."""
    reading = success_reading("[All task succ.] 0.10 | 0.20 | 0.30 |")
    assert abs(reading["value"] - 0.2) < 1e-9
    assert reading["averaged"] is True
    assert reading["values"] == [0.1, 0.2, 0.3]
    assert "0.10 | 0.20 | 0.30" in reading["read_from"]


def test_the_last_reading_is_the_one_kept():
    """A program that reports every epoch has said what it settled on, and that is the last
    statement of it."""
    reading = success_reading("succ: 0.10\nsucc: 0.62\n")
    assert reading["value"] == 0.62


def test_nothing_labelled_is_not_a_zero():
    """A program that ends with a bare `print(test_loss, success_rate)` has printed two
    unlabelled numbers. Reporting that as nothing read is the honest answer; reporting it as
    zero is not."""
    for said in ("", "0.42 0.61", "loss: 1.2"):
        reading = success_reading(said)
        assert reading["value"] is None, said
        assert reading["read_from"] == ""
    assert success_rate("") is None


def test_the_float_accessor_is_the_same_reading():
    """Every caller that wants only the number keeps working, and gets the same one."""
    for said in ("succ: 0.62", "[All task succ.] 0.10 | 0.20 | 0.30 |", "", "0.42 0.61"):
        assert success_rate(said) == success_reading(said)["value"], said


def test_success_reading_rejects_cross_line_and_ambiguous_units():
    assert success_rate("success rate:\nepisodes: 80") is None
    assert success_rate("success rate: 75") is None
    assert success_rate("success rate: 75%") == 0.75
    assert success_rate("success_rate = 0.625") == 0.625
    assert success_rate("best succ: 1.0\nloss: 0.2") is None
    assert success_rate("succ: 0.5e309") is None


def test_named_numbers_is_still_unfiltered():
    """Which parts of what a program said matter is the caller's question, and a reader that
    has already answered it cannot be asked again -- so `named_numbers` answers a different
    one: what the program put a name to."""
    assert named_numbers("loss: 5.36 succ: 0.62") == {"loss": 5.36, "succ": 0.62}
    assert named_numbers("line 1119 cuda:0") == {"line": 1119.0, "cuda": 0.0}
    assert named_numbers("") == {}


def test_training_progress_distinguishes_updates_from_new_checkpoint_only():
    assert training_progress("model saved to final_ckpt.pt")["status"] == "unknown"
    assert training_progress("args.num_iterations=0\nmodel saved")["status"] == "zero"
    assert training_progress("\r0it [00:00, ?it/s]\r0it [00:00, ?it/s]")["status"] == "zero"
    assert training_progress("\r  0%| | 0/1 [00:00<?, ?it/s]"
                             "\r100%|█| 1/1 [00:01<00:00, 1.00it/s]")["status"] == "observed"
    slow_simulator = ("success_once: 0.00, return: 3.56: "
                      "0%| | 1/1024 [00:03<56:16, 3.30s/it]")
    assert training_progress(slow_simulator) == {
        "status": "observed", "evidence": {"tqdm_completed": 1, "tqdm_total": 1024}}
    assert training_progress("0/2 [00:00<?, ?it/s]") == {
        "status": "zero", "evidence": {"tqdm_completed": 0, "tqdm_total": 2}}
    assert training_progress("Epoch: 1, global_step=640")["status"] == "observed"
    assert training_progress("SPS: 12") == {
        "status": "observed", "evidence": {"steps_per_second": 12.0}}
    assert training_progress("SPS: 0")["status"] == "unknown"


def test_evaluation_progress_uses_completed_not_requested_episodes():
    assert evaluation_progress("Evaluated 8 steps resulting in 0 episodes") == {
        "status": "zero", "completed_episodes": 0,
        "evidence": "Evaluated 8 steps resulting in 0 episodes"}
    assert evaluation_progress("Evaluated 400 steps resulting in 8 episodes")[
        "completed_episodes"] == 8
    assert evaluation_progress("completed_episodes: 2")["status"] == "observed"
    assert evaluation_progress("episodes requested: 2")["status"] == "unknown"
