"""Deriving how a benchmark runs, without a per-benchmark backend file.

No model is called. What is tested is the part that is a check rather than a judgement: the
answer must be complete enough to act on, and every path it names must exist. Neither
settles whether an entry point is the *right* one -- a repository with six policy families
has six trainers that all exist -- and the report is written so that this is not mistaken
for having been settled.
"""

import json
from pathlib import Path

import pytest

from autosim.research.execution_derive import STAGES, problems_in

PROJECT = Path(__file__).resolve().parents[1]


def answer(**stages):
    value = {"stages": {name: {"available": False, "why": "not in this fixture"}
                        for name in STAGES}, "reasoning": "read the docs"}
    value["stages"].update(stages)
    return value


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "train.py").write_text("# trainer\n", encoding="utf-8")
    (tmp_path / "scripts" / "eval.py").write_text("# evaluator\n", encoding="utf-8")
    return tmp_path


def test_every_stage_must_be_answered(repo):
    value = {"stages": {"train": {"available": False, "why": "none"}}, "reasoning": "r"}
    faults = problems_in(value, repo)
    assert sum("must be answered" in fault for fault in faults) == len(STAGES) - 1


def test_an_unavailable_stage_must_say_why(repo):
    faults = problems_in(answer(collect={"available": False, "why": "  "}), repo)
    assert any("stages.collect is reported unavailable and must say why" in f for f in faults)


def test_an_available_stage_must_name_an_entry_point_that_exists(repo):
    faults = problems_in(answer(train={"available": True, "entrypoint": "scripts/trainer.py",
                                       "invocation": "python ...", "artifact": "a checkpoint"}),
                         repo)
    assert any("does not exist" in fault for fault in faults)


def test_an_available_stage_must_say_what_success_looks_like(repo):
    """Without an artifact there is nothing for the execution check to look for."""
    faults = problems_in(answer(train={"available": True, "entrypoint": "scripts/train.py",
                                       "invocation": "python scripts/train.py",
                                       "artifact": ""}), repo)
    assert any("artifact is required" in fault for fault in faults)


def test_a_directory_is_reported_as_a_weaker_answer(repo):
    """Where to look is not what to run, and the difference is kept visible."""
    faults = problems_in(answer(train={"available": True, "entrypoint": "scripts",
                                       "invocation": "?", "artifact": "?"}), repo)
    assert any("is a directory, not a file to run" in fault for fault in faults)


def test_a_complete_answer_about_real_files_passes(repo):
    assert problems_in(answer(
        train={"available": True, "entrypoint": "scripts/train.py",
               "invocation": "python scripts/train.py --data <dir>",
               "artifact": "checkpoints/<steps>/pretrained_model/"},
        evaluate={"available": True, "entrypoint": "scripts/eval.py",
                  "invocation": "python scripts/eval.py --policy <ckpt>",
                  "artifact": "evaluation_metrics.json"},
    ), repo) == []


def test_an_invented_stage_is_rejected(repo):
    value = answer()
    value["stages"]["finetune"] = {"available": False, "why": "no"}
    assert any("no such entry" in fault for fault in problems_in(value, repo))


def test_the_answer_carries_that_it_is_unverified(tmp_path):
    """A claim about which entry point is right is not settled by a path existing.

    The distinction has to travel with the answer: read as settled, "train is
    policy/pi0/finetune.sh" is indistinguishable from a verified fact.
    """
    from autosim.research import execution_derive
    repo = tmp_path / "Bench"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "eval.py").write_text("# eval\n", encoding="utf-8")

    stages = {}
    for name in STAGES:
        if name == "evaluate":
            stages[name] = {"available": True, "entrypoint": "scripts/eval.py",
                            "invocation": "python scripts/eval.py",
                            "artifact": "metrics.json"}
        else:
            stages[name] = {"available": False, "why": "not in this fixture"}

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            if "files_to_read" in system:
                return json.dumps({"files_to_read": ["scripts/eval.py"]}), {}
            return json.dumps({"stages": stages, "reasoning": "read the header"}), {}

    result = execution_derive.derive(repo, client=Client())
    assert result["evidence"] == "static_only"
    assert "no stage has been executed" in result["unverified"]
    assert result["stages"]["evaluate"]["entrypoint"] == "scripts/eval.py"
