"""The loop works on ideas, audited before they run, and can change the benchmark's source.

The behaviour these tests hold down is the one the six gaps were closed for. Before them the
loop could name values from a declared space and nothing else, so a benchmark whose own code
stopped its evaluation from running could be diagnosed correctly -- three times, in one
RoboTwin run -- and then the loop ended. Every diagnosis was accurate and every one of them
was answered with a stop.
"""

import json
import sys
from pathlib import Path

from autosim.research.adapter_protocol import Axis, OptimizationSpace
from autosim.research.compute_decision import ComputeDecision
from autosim.research.declarative_backend import DeclarativeBackend
from autosim.research.derived_research import DerivedResearch
from autosim.research.ideas import CLEARED, REJECTED, Idea

#: A benchmark step that reads the name the data does not have. RoboTwin's shape: the path is
#: computed one way and read another, and no setting that the declared space can hold makes
#: the file appear under the name the loader asks for.
BROKEN = '''\
import pathlib
here = pathlib.Path(__file__).parent
text = (here / "logs" / "train.txt").read_text()
print("succ: 0.50" if text.strip() else "succ: 0.00")
'''

#: The same file with the name corrected: what the checkout looks like after the fix.
WORKING = BROKEN.replace('"train.txt"', '"train_data.txt"')


# Made fresh per test rather than held as a module constant. An `Idea` carries its status,
# and `IdeaLibrary.add` stores the object it is given -- so a constant reused across tests
# arrives already `worked` or `rejected` from the test before, `audit` skips it because it is
# no longer new, and the run quietly has no ideas at all.
def fix_idea() -> Idea:
    return Idea(label="the loader opens a name the data does not have",
                granularity="code", risk="low",
                mechanism="the step reads logs/train.txt and the data is logs/train_data.txt",
                why="the file the loader opens is not the file the benchmark writes",
                change={"file": "report.py", "find": '"train.txt"',
                        "replace": '"train_data.txt"'},
                touches=["report.py"])


#: Reads the same nothing, and fixes nothing: what a change that did not work looks like.
def useless_idea() -> Idea:
    return Idea(label="a change that does not help", granularity="code", risk="low",
                mechanism="reorders two statements that do not matter",
                change={"file": "report.py", "find": "import pathlib",
                        "replace": "import pathlib  # reordered"},
                touches=["report.py"])


#: It would edit the evaluation entry point. Refused before it runs.
def crosses_a_line_idea() -> Idea:
    return Idea(label="edit the evaluator so the stage finishes",
                granularity="code", risk="high",
                mechanism="change the evaluation script to return early",
                change={"file": "evaluate.py", "find": "a", "replace": "b"},
                touches=["evaluate.py"])


def benchmark(tmp_path: Path, *, report: str = BROKEN) -> Path:
    """A checkout with one step, its data under a name the step does not ask for."""
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "train_data.txt").write_text("episodes\n", encoding="utf-8")
    (tmp_path / "report.py").write_text(report, encoding="utf-8")
    (tmp_path / "evaluate.py").write_text("print('succ: 0.00')\n", encoding="utf-8")
    return tmp_path


def argv_pointing_at(tmp_path: Path) -> dict:
    """A trainer that works and an evaluation that does not, which is the RoboTwin shape."""
    return {
        "train": ('def stage_argv_train(i):\n'
                  '    return [i["python"], "-c", "import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True);(p/\'model.pth\').write_text(\'trained\')", i["output"]]\n'),
        "evaluate": ('def stage_argv_evaluate(i):\n'
                     f'    return [i["python"], {str(tmp_path / "report.py")!r}]\n')}


def loop(tmp_path: Path) -> DerivedResearch:
    sources = argv_pointing_at(tmp_path)
    answer = {"stages": {stage: {"available": True, "entrypoint": "report.py",
                                 "invocation": "python report.py",
                                 "artifact": "model.pth" if stage == "train" else ""}
                         for stage in ("train", "evaluate")}}
    return DerivedResearch(
        repo=tmp_path, output=tmp_path / "out",
        backend=DeclarativeBackend(repo=tmp_path, answer=answer, sources=sources,
                                   parameters={}),
        interpreter=Path(sys.executable),
        space=OptimizationSpace(training=(
            Axis("train.n_epochs", "integer", "epochs", low=1, high=100, default=1),)),
        client=None, stages=sources, run_id="ideas",
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="the test chose it"))


def test_a_code_idea_makes_a_stage_run_that_could_not_run(tmp_path):
    """The gap, end to end: the evaluation produces no number, and the run fixes that.

    Nothing in the declared space could have. The number does not come out because the
    benchmark's own step opens a name the data does not have, and a value is something a step
    reads once it runs.
    """
    repo = benchmark(tmp_path)
    real = loop(repo)
    real.library.add(fix_idea())
    report = real.run(rounds=2)
    assert report["rounds"][0]["measured"] is False
    row = report["rounds"][1]
    assert row["status"] == "measured", report["rounds"]
    assert row["success_rate"] == 0.5
    assert row["undone"] is False
    assert "train_data.txt" in (repo / "report.py").read_text(encoding="utf-8")


def test_a_change_that_does_not_help_is_taken_back(tmp_path):
    """A change left in place is a checkout the next round runs on top of.

    Then the round after that runs on both, the run's later failures are attributed to the
    wrong thing, and any number it finally reports came from a benchmark nobody chose.
    """
    repo = benchmark(tmp_path)
    before = (repo / "report.py").read_text(encoding="utf-8")
    real = loop(repo)
    real.library.add(useless_idea())
    report = real.run(rounds=2)
    row = report["rounds"][1]
    assert row["undone"] is True, report["rounds"]
    assert (repo / "report.py").read_text(encoding="utf-8") == before


def test_an_idea_that_crosses_a_line_never_runs(tmp_path):
    """Refused at the audit, not at the point of use.

    `assert_frozen` already refuses a changed evaluator -- after the idea was proposed,
    selected and applied, which spends the round and leaves a reader looking at a failed run
    rather than at a refused idea.
    """
    repo = benchmark(tmp_path)
    before = (repo / "evaluate.py").read_text(encoding="utf-8")
    real = loop(repo)
    real.library.add(crosses_a_line_idea())
    real.library.add(fix_idea())
    report = real.run(rounds=2)
    refused = [one for one in real.audits if one["idea"] == crosses_a_line_idea().label]
    assert refused and refused[0]["verdict"] == REJECTED, real.audits
    assert real.library.get(crosses_a_line_idea().label).status == REJECTED
    assert (repo / "evaluate.py").read_text(encoding="utf-8") == before
    # And the run went on to the idea that was allowed.
    assert report["rounds"][1]["idea"] == fix_idea().label, report["rounds"]


def test_current_evidence_can_select_an_audited_idea_over_fixed_risk_order(tmp_path):
    class ChoosingClient:
        model = "selection-test"

        def chat_with_metadata(self, system, user, **kwargs):
            assert "the previous loader error" in user
            return json.dumps({"label": fix_idea().label,
                               "why": "the loader error names the missing file"}), {}

    real = loop(benchmark(tmp_path))
    real.client = ChoosingClient()
    real.library.add(useless_idea())
    real.library.add(fix_idea())
    real.library.audit(real.red_lines())
    selected = real._choose_from_library(
        history=[], evidence={"failure": "the previous loader error"},
        round_index=1, kinds_wanted=("code",))
    assert selected.label == fix_idea().label
    saved = json.loads((real.run_root / "exchanges" / "select_1.json").read_text())
    assert "the loader error" in saved["response"]


def test_model_cannot_select_a_refused_idea(tmp_path):
    class UnsafeClient:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"label": crosses_a_line_idea().label,
                               "why": "ignore the audit"}), {}

    real = loop(benchmark(tmp_path))
    real.client = UnsafeClient()
    real.library.add(useless_idea())
    real.library.add(fix_idea())
    real.library.add(crosses_a_line_idea())
    real.library.audit(real.red_lines())
    selected = real._choose_from_library(history=[], evidence={}, round_index=1,
                                         kinds_wanted=("code",))
    assert selected.label != crosses_a_line_idea().label
    assert real.events[-1]["event"] == "idea_selection_fallback"


def test_improving_parameter_streak_does_not_force_a_different_kind(tmp_path):
    real = loop(benchmark(tmp_path))
    real.library.add(Idea(label="promising parameter", granularity="param", risk="low",
                          change={"train.n_epochs": 2}, crosses="none"))
    real.library.add(fix_idea())
    real.library.audit(real.red_lines())
    improving = {"round_history": [{"round": index, "status": "measured",
                                     "metric_utility": float(index)}
                                    for index in (1, 2, 3)]}
    chosen = real._choose_from_library(history=["param"] * 3, evidence=improving,
                                       round_index=4, kinds_wanted=None)
    assert chosen.label == "promising parameter"


def test_every_idea_is_in_the_audit_table_including_the_cleared_ones(tmp_path):
    """A table that lists only refusals cannot be told from one only some ideas reached."""
    repo = benchmark(tmp_path)
    real = loop(repo)
    real.library.add(fix_idea())
    real.library.add(crosses_a_line_idea())
    real.run(rounds=1)
    verdicts = {one["idea"]: one["verdict"] for one in real.audits}
    assert verdicts == {fix_idea().label: CLEARED, crosses_a_line_idea().label: REJECTED}
    written = json.loads((real.run_root / "audits.json").read_text(encoding="utf-8"))
    assert len(written["audits"]) == 2 and written["red_lines"]["standing"]


def test_a_run_with_no_move_says_which_kind_of_no_move_it_was(tmp_path):
    """A run that has genuinely run out is a different finding from one that cannot start."""
    repo = benchmark(tmp_path)
    real = loop(repo)
    report = real.run(rounds=3)
    assert "no baseline could be measured" in report["stopped_because"]
    assert report["rounds"][1]["status"] == "no valid proposal"


def test_the_run_keeps_going_when_no_number_comes_out(tmp_path):
    """Stopping was the loop answering an accurate diagnosis with an ending."""
    repo = benchmark(tmp_path)
    real = loop(repo)
    real.library.add(useless_idea())
    report = real.run(rounds=1)
    # No baseline, and a round was still attempted -- with the only kind of change that
    # could have produced one.
    assert report["rounds"][1]["kind"] == "code", report["rounds"]
    assert "no_baseline" in [one.get("event") for one in real.events]


def test_progress_is_reported_and_is_a_scale_of_its_own(tmp_path):
    """A run that cannot measure yet still has a best state, and it is not the other scale."""
    repo = benchmark(tmp_path)
    real = loop(repo)
    real.library.add(fix_idea())
    report = real.run(rounds=2)
    objective = report["objective"]
    assert 0.0 <= objective["score"] <= 1.0
    assert any(one["state"] == "not reached" for one in objective["checks"])
    best = report["best"]
    assert best is not None
    # What was reached is a measured number, so `_best` points at the measured state and not
    # at a state that merely got further toward producing one.
    assert best["scale"] == "success_rate" and best["score"] == 0.5
    assert "train_data.txt" in (repo / "report.py").read_text(encoding="utf-8")


def test_a_parameter_idea_is_still_validated_against_the_space(tmp_path):
    """The declared space is still what says which names exist.

    An idea that says `train.loss_scale` when the benchmark spells it `train.loss_coef` is a
    proposal the declaration refuses, and it is refused here rather than arriving at a stage
    as an argument nothing reads.
    """
    repo = benchmark(tmp_path / "ok", report=WORKING)
    real = loop(repo)
    real.library.add(Idea(label="a name the space does not have", granularity="param",
                          mechanism="raise the epoch count",
                          change={"train.epoch_count": 4}))
    report = real.run(rounds=2)
    assert report["rounds"][0]["measured"] is True
    assert report["rounds"][1]["status"] == "the idea does not fit the declared space"
    assert real.library.get("a name the space does not have").status != CLEARED
    # And the round after that did not end the run: there was nothing left to try, and the
    # reason is a finding rather than an exception out of the middle of the loop.
    assert report["rounds"][2]["status"] == "no idea to try", report["rounds"]


class _RepairingClient:
    """A model that answers with the text it was shown, which is the whole point of the step."""

    model = "a-test-client"

    def __init__(self, find, replace):
        self.find, self.replace = find, replace
        self.seen: list[str] = []

    def chat_with_metadata(self, system, user, **kwargs):
        self.seen.append(user)
        return json.dumps({"change": {"file": "report.py", "find": self.find,
                                      "replace": self.replace},
                           "why": "copied the line out of the file I was shown"}), {}


def test_a_change_that_names_text_the_file_does_not_have_is_repaired(tmp_path):
    """The model that wrote the change had not been shown the file.

    Being refused for that is not the idea's fault and not a finding about the benchmark --
    it is the gap `execution_derive`'s inspection loop closes for `argv`, one level up. What
    it is not is a reason to let the model guess again.
    """
    repo = benchmark(tmp_path)
    real = loop(repo)
    real.client = _RepairingClient(find="(here / \"logs\" / \"train.txt\")",
                                   replace="(here / \"logs\" / \"train_data.txt\")")
    wrong = fix_idea()
    wrong.change = {"file": "report.py", "find": 'text = (here / "logs/train.txt").read_text()',
                    "replace": 'text = (here / "logs/train_data.txt").read_text()'}
    real.library.add(wrong)
    report = real.run(rounds=1)
    assert report["rounds"][1]["status"] == "measured", report["rounds"]
    # And the file it was shown was in the prompt it answered, not merely assumed to be. The
    # client is asked more than once -- the rubric's decomposition comes first -- so this is
    # about the repair having happened at all, not about which call it was.
    assert any("read_text()" in one for one in real.client.seen), real.client.seen


def test_an_idea_that_cannot_be_made_to_work_says_so(tmp_path):
    """`give_up` rather than a `find` the model has not seen in the file."""

    class _GivesUp:
        model = "a-test-client"

        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"give_up": "the file has no such line"}), {}

    repo = benchmark(tmp_path)
    real = loop(repo)
    real.client = _GivesUp()
    hopeless = fix_idea()
    hopeless.change = {"file": "report.py",
                       "find": "the line this idea assumed was there",
                       "replace": "something else"}
    real.library.add(hopeless)
    report = real.run(rounds=1)
    # Nothing was applied, nothing was measured, and no exception came out of the middle.
    assert report["rounds"][1]["status"] == "the change could not be applied", report["rounds"]
    assert (repo / "report.py").read_text(encoding="utf-8").startswith("import pathlib")
    assert real.touched == set()


def test_an_algo_idea_is_not_asked_for_a_file_patch(tmp_path):
    """`algo` names what the run does; `code` names a file. Sending both to the patch path
    asked an algo idea for a `find` it had never been given, and the round was spent on a
    refusal that said nothing about the benchmark -- four of seven rounds, on RoboTwin."""
    repo = benchmark(tmp_path, report=WORKING)
    real = loop(repo)
    real.library.add(Idea(
        label="collect first and train on the result", granularity="algo",
        mechanism="train longer before evaluating", why="the trainer stops too early",
        change={"collection": {}, "training": {"train.n_epochs": 3}}))
    report = real.run(rounds=1)
    row = report["rounds"][1]
    assert row["kind"] == "algo"
    # Not refused for lacking a file, and not refused for lacking the protocol's envelope
    # either. It ran, under what it asked for.
    assert row["status"] == "measured", report["rounds"]
    assert row["varied"]["train.n_epochs"] == 3, row


def test_a_change_that_does_not_fit_the_space_is_repaired_not_reshaped(tmp_path):
    """The framework's answer to a spelling the system cannot read: show the model the facts.

    The rule this replaces moved a section out of a section, joined a stage name to an axis,
    and un-doubled a repeated key -- three rules for three spellings its author had seen. This
    covers the ones nobody has, because what the model corrects itself against is the declared
    sections and axes, not a list of accepted forms.
    """

    class _RepairsAgainstTheSpace:
        model = "a-test-client"

        def __init__(self):
            self.seen: list[str] = []

        def chat_with_metadata(self, system, user, **kwargs):
            self.seen.append(user)
            return json.dumps({"change": {"training": {"train.n_epochs": 3}},
                               "why": "the axis is named train.n_epochs and lives in training"}), {}

    repo = benchmark(tmp_path, report=WORKING)
    real = loop(repo)
    real.client = _RepairsAgainstTheSpace()
    real.library.add(Idea(label="nested the wrong way", granularity="algo",
                          mechanism="raise the epoch count", why="the default is a stub",
                          change={"training": {"task_selection": {"train.n_epochs": 3}}}))
    report = real.run(rounds=1)
    assert report["rounds"][1]["status"] == "measured", report["rounds"]
    assert report["rounds"][1]["varied"]["train.n_epochs"] == 3
    # The model was shown the sections and axes it was writing against, not a rule about
    # nesting -- that is the whole difference. The client answers more than one question
    # during a round, so this is about the repair having been given the facts at all.
    assert any("the_sections_this_benchmark_declares" in one for one in real.client.seen)
    assert any("train.n_epochs" in one for one in real.client.seen)


def test_a_change_the_model_cannot_fix_is_recorded_as_refused(tmp_path):
    """And when it says it cannot be made to fit, that is a finding rather than a crash."""

    class _GivesUp:
        model = "a-test-client"

        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"give_up": "the space has no axis for what this idea needs"}), {}

    repo = benchmark(tmp_path, report=WORKING)
    real = loop(repo)
    real.client = _GivesUp()
    real.library.add(Idea(label="an axis the space does not have", granularity="param",
                          mechanism="read the seeds from the benchmark's own bank",
                          change={"train.seed_bank": "task_config/seeds.json"}))
    report = real.run(rounds=2)
    row = report["rounds"][1]
    assert row["status"] == "the idea does not fit the declared space", row
    assert "no axis for what this idea needs" in row["repair"], row
    assert row["change"]["train.seed_bank"] == "task_config/seeds.json"


def test_an_algo_idea_that_names_an_axis_the_space_does_not_have_is_refused(tmp_path):
    """The space is what says which names exist, for an `algo` idea as much as a `param` one.

    The envelope is supplied by the loop and not by the model -- `decision`, `proposal_id`
    and `expected_validation` are the protocol's, not the idea's -- so what is left for the
    space to check is the part that is the change.
    """
    repo = benchmark(tmp_path, report=WORKING)
    real = loop(repo)
    real.library.add(Idea(label="an axis that is not declared", granularity="algo",
                          mechanism="read the seeds from the benchmark's own bank",
                          change={"collection": {"seed_bank": "task_config/seeds.json"}}))
    report = real.run(rounds=2)
    assert report["rounds"][0]["measured"] is True
    assert report["rounds"][1]["status"] == "the idea does not fit the declared space"
    assert "seed_bank" in report["rounds"][1]["why_not"]


def test_a_default_in_the_envelope_is_not_a_request_for_data(tmp_path):
    """The envelope carries every required axis at its declared default, because the space
    check needs them. Reading the request off the validated proposal therefore made every
    idea a request for data -- and on RoboTwin four rounds of eight died asking a benchmark
    with no collector to collect."""
    repo = benchmark(tmp_path, report=WORKING)
    sources = argv_pointing_at(repo)
    answer = {"stages": {"train": {"available": True, "entrypoint": "t.sh", "artifact": "model.pth"},
                         "evaluate": {"available": True, "entrypoint": "report.py",
                                      "artifact": ""}}}
    real = DerivedResearch(
        repo=repo, output=repo / "out",
        backend=DeclarativeBackend(repo=repo, answer=answer, sources=sources, parameters={}),
        interpreter=Path(sys.executable),
        space=OptimizationSpace(
            training=(Axis("train.n_epochs", "integer", "epochs", low=1, high=100, default=1),),
            collection=(Axis("collect_data", "choice", "collect before training",
                             values=("true", "false"), default="true"),)),
        client=None, stages=sources, run_id="defaults",
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="the test chose it"))
    real.library.add(Idea(label="train a bit longer", granularity="algo",
                          mechanism="raise the epoch count", why="the default is a stub",
                          change={"training": {"train.n_epochs": 3}}))
    report = real.run(rounds=1)
    row = report["rounds"][1]
    # `collect` is unavailable here, and this idea never asked to collect, so the round ran
    # -- rather than being spent on a request the envelope had filled in on its behalf.
    assert row["asked_for_and_did_not_get"] == [], row
    assert row["status"] == "measured", report["rounds"]


def test_the_loop_leaves_the_checkout_as_it_found_it_after_a_failed_round(tmp_path):
    """Every file a failed change touched is back, not just the last one."""
    repo = benchmark(tmp_path)
    before = (repo / "report.py").read_text(encoding="utf-8")
    real = loop(repo)
    real.library.add(useless_idea())
    real.run(rounds=1)
    assert real.touched == {"report.py"}
    assert (repo / "report.py").read_text(encoding="utf-8") == before
