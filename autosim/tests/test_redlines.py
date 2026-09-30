"""What a run may not do, checked before anything runs rather than while it is running.

The failure this exists for is not hypothetical: a loop that can change both the thing being
measured and the way it is measured will find improvements, and they will not be improvements.
The hard part of automating research is that such a run reports a better number *and* a clean
record -- so the refusal has to be cheap, early, and explainable.
"""

from dataclasses import dataclass, field
from pathlib import Path

from autosim.research import redlines


@dataclass
class Idea:
    label: str = "an idea"
    mechanism: str = ""
    change: str = ""
    why: str = ""
    evidence: str = ""
    touches: list = field(default_factory=list)
    #: The idea's own answer to "which line would this cross", asked for when it is written.
    #: `"none"` is an answer; saying nothing is not, and is refused for that.
    crosses: str = "none"


LINES = redlines.RedLines(benchmark="Toy")


def test_an_idea_that_edits_the_evaluator_is_refused_before_it_runs():
    """`assert_frozen` refuses a changed evaluator at the moment it is used -- after the idea
    has been proposed, selected and applied. What a reader sees then is a failed run, not a
    refused idea, and the round is spent either way."""
    verdict = redlines.audit(Idea(label="widen the tolerance",
                                  touches=["scripts/eval_policy.py"]), LINES)
    assert verdict.verdict == redlines.REJECTED
    assert verdict.line == "R2"
    assert "eval_policy.py" in verdict.because and "eval*.py" in verdict.because
    # And the reason says why the judgement is not the run's to make.
    assert "not a judgement this run may make about itself" in verdict.because


def test_filename_guards_do_not_freeze_trainer_output_beneath_verify_directory(tmp_path):
    trainer_output = tmp_path / "runs" / "policy-verify" / "final_ckpt.pt"
    trainer_output.parent.mkdir(parents=True)
    trainer_output.write_bytes(b"checkpoint")
    evaluator = tmp_path / "scripts" / "eval_policy.py"
    evaluator.parent.mkdir()
    evaluator.write_text("# evaluator\n", encoding="utf-8")

    hashes = redlines.protected_hashes(tmp_path, LINES)
    assert str(evaluator) in hashes
    assert str(trainer_output) not in hashes
    assert redlines.audit(Idea(touches=["runs/policy-verify/final_ckpt.pt"]),
                          LINES).verdict == redlines.CLEARED
    assert redlines.audit(Idea(touches=["scripts/eval_policy.py"]),
                          LINES).verdict == redlines.REJECTED


def test_declaration_relative_path_still_protects_only_that_file(tmp_path):
    specific = redlines.RedLines(protected=("scripts/judge.py",))
    assert redlines.audit(Idea(touches=["scripts/judge.py"]),
                          specific).verdict == redlines.REJECTED
    assert redlines.audit(Idea(touches=["other/judge.py"]),
                          specific).verdict == redlines.CLEARED


def test_parameter_idea_does_not_edit_the_file_that_declares_its_flag():
    from autosim.research.ideas import Idea as ResearchIdea

    parameter = ResearchIdea(label="change learning rate", granularity="param",
                             change={"learning_rate": 1e-5},
                             touches=[])
    assert redlines.audit(parameter, LINES).verdict == redlines.CLEARED

    # Explicitly naming a write target remains forbidden regardless of the idea's label.
    parameter.touches = ["scripts/eval_train.py"]
    assert redlines.audit(parameter, LINES).verdict == redlines.REJECTED
    parameter.touches = []
    parameter.change = {"file": "scripts/eval_train.py", "learning_rate": 1e-5}
    assert redlines.audit(parameter, LINES).verdict == redlines.REJECTED
    code = ResearchIdea(label="edit evaluator", granularity="code",
                        touches=["scripts/eval_train.py"])
    assert redlines.audit(code, LINES).verdict == redlines.REJECTED


def test_a_multi_file_idea_cannot_hide_an_evaluator_in_its_change():
    candidate = Idea(label="hidden evaluator", touches=["train.py"], crosses="none")
    candidate.change = {"patches": [{"file": "train.py"},
                                    {"file": "evaluate.py"}]}
    verdict = redlines.audit(candidate, LINES)
    assert verdict.verdict == redlines.REJECTED
    assert "evaluate.py" in verdict.because


def test_the_lines_are_read_from_what_the_idea_says_about_itself():
    """The idea answers which line it would cross, the way it answers its risk and its
    evidence. It is not inferred from the words of the description: a phrase table is a model
    of how a model phrases things, and this one was wrong in both directions."""
    for line in ("R1", "R3", "R4", "R5", "R6"):
        verdict = redlines.audit(Idea(label="x", mechanism="something", crosses=line), LINES)
        assert verdict.verdict == redlines.REJECTED, line
        assert verdict.line == line, (line, verdict.line)
        # The reason quotes the line's own statement of what it forbids, not a guess at the
        # idea's intent.
        assert line in verdict.because or "it says of itself" in verdict.because


def test_an_idea_that_names_a_line_the_benchmark_does_not_have_is_refused():
    """An answer nobody can read is not an answer: a line id that is not one of this run's is
    refused, and the refusal lists the ones that are."""
    verdict = redlines.audit(Idea(label="x", mechanism="m", crosses="R99"), LINES)
    assert verdict.verdict == redlines.REJECTED
    assert "R99" in verdict.because and "R1" in verdict.because


def test_an_idea_that_says_nothing_about_the_lines_is_not_cleared():
    """`"none"` is an answer; saying nothing is not. Clearing an idea that did not answer
    would make silence the way past the audit."""
    verdict = redlines.audit(Idea(label="x", mechanism="m", crosses=""), LINES)
    assert verdict.verdict == redlines.REJECTED
    assert "does not say whether" in verdict.because


def test_an_honest_idea_is_cleared_and_the_verdict_says_what_was_checked():
    """Cleared has to be as explicit as rejected: a table that lists only refusals cannot be
    told from a table only some ideas reached."""
    verdict = redlines.audit(Idea(label="more augmentation",
                                  mechanism="raise the sampling mass for the targeted profile",
                                  change="targeted_sampling_mass: 0.5 -> 0.75",
                                  touches=["configs/train.yaml"], crosses="none"), LINES)
    assert verdict.verdict == redlines.CLEARED and verdict.line == ""
    assert "by its own account" in verdict.because


def test_the_benchmark_s_own_line_is_derived_from_what_it_declares():
    """R7 is not written here because it cannot be: it is whatever this benchmark makes
    non-negotiable, and it is found by reading the repository. What the declaration names as
    the evaluation entry point becomes a protected path -- so a benchmark that moves its
    evaluator moves its protection with it."""
    declaration = {
        "benchmark": "Toy",
        "capabilities": {"native_evaluation": {"status": "declared",
                                               "entrypoint": "XPolicyLab/eval.sh"},
                         "training": {"status": "declared", "entrypoint": "train.sh"}},
        "evidence": "the README says the seeds come from the benchmark's own bank",
    }
    lines = redlines.from_declaration(declaration, repo=Path("/tmp"))
    assert "XPolicyLab/eval.sh" in lines.protected_paths()
    # The trainer is not protected: it is upstream of the boundary, which is where
    # optimisation belongs.
    assert "train.sh" not in lines.protected_paths()
    assert any(one[0] == "R7" for one in lines.particular)
    assert redlines.audit(Idea(label="patch it", touches=["XPolicyLab/eval.sh"]),
                          lines).verdict == redlines.REJECTED
    # And the evidence travels as a note, because it is what a reader weighs the lines by.
    assert "own bank" in lines.notes


def test_the_prompt_form_carries_the_why_and_not_only_the_rule():
    """A list of prohibitions without reasons is a list a run will find a reading of. The why
    is what lets a reader tell whether a new constraint belongs or an old one is being
    stretched."""
    said = redlines.RedLines(benchmark="Toy").states_itself()
    assert "R1" in said and "R5" in said
    assert said.count("Because") == len(redlines.STANDING)


def test_the_audit_table_shows_every_idea_including_the_cleared_ones():
    audits = [redlines.audit(Idea(label="a", touches=["evaluate.py"]), LINES),
              redlines.audit(Idea(label="b", mechanism="raise the mass", crosses="none"),
                             LINES)]
    rendered = redlines.table(audits)
    assert "**rejected**" in rendered and "**cleared**" in rendered
    assert "1 条通过审计，1 条被拒绝" in rendered
    assert "没有跑过" in rendered


def test_a_verdict_and_a_status_are_the_same_word():
    """Two vocabularies for one decision, and they have to agree.

    Written in two cases they were two different strings, so `idea.status != redlines.CLEARED`
    was true for a cleared idea -- never true, never raising. A leap that had been cleared was
    treated as refused, and nothing anywhere said so.
    """
    from autosim.research import ideas

    assert redlines.CLEARED == ideas.CLEARED and redlines.REJECTED == ideas.REJECTED
