"""Executing a benchmark through a derived answer.

No benchmark is run. What is tested is that the backend refuses to claim more than it
checked: a stage whose derivation named no artifact is unverifiable rather than passed, and
a command that exits zero having written nothing it promised has not run.
"""

import json
from pathlib import Path

import pytest

from autosim.research.declarative_backend import DeclarativeBackend, argv_name

SOURCE = ('def {name}(i):\n'
          '    return [i["python"], i["repo"] + "/run.py", "--task", i["task"],\n'
          '            "--out", i["output"], "--steps", str(i["steps"])]\n')


def backend(tmp_path, *, artifact="*_done.json", stage="train"):
    source = SOURCE.format(name=argv_name(stage))
    return DeclarativeBackend(
        repo=tmp_path,
        answer={"stages": {stage: {"available": True, "entrypoint": "run.py",
                                   "invocation": "python run.py", "artifact": artifact,
                                   "parameters": [{"name": "setting", "value": "random",
                                                   "evidence": "read it"}]}}},
        sources={stage: source},
        parameters={stage: {"setting": {"value": "random"}}})


def test_the_command_comes_from_the_generated_function(tmp_path):
    argv = backend(tmp_path).argv("train", {"python": "/p", "task": "t", "output": "/o",
                                            "steps": 5})
    assert argv == ["/p", str(tmp_path) + "/run.py", "--task", "t", "--out", "/o",
                    "--steps", "5"]


def test_a_declared_parameter_reaches_the_function(tmp_path):
    source = ('def {name}(i):\n    return [i["setting"]]\n').format(name=argv_name("train"))
    value = DeclarativeBackend(repo=tmp_path,
                               answer={"stages": {"train": {"available": True, "entrypoint": "r",
                                                            "invocation": "i", "artifact": "o"}}},
                               sources={"train": source},
                               parameters={"train": {"setting": {"value": "random"}}})
    assert value.argv("train", {}) == ["random"]


def test_a_stage_with_no_derived_command_says_why(tmp_path):
    value = backend(tmp_path)
    value.stages["collect"] = {"available": False, "why": "no automated producer here"}
    with pytest.raises(ValueError, match="no automated producer here"):
        value.argv("collect", {})


def test_an_artifact_that_did_not_appear_is_not_a_pass(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    check = backend(tmp_path).check_artifact("train", out)
    assert check["checked"] and check["matched"] == 0


def test_a_glob_is_what_makes_an_artifact_findable(tmp_path):
    out = tmp_path / "out"
    (out / "nested").mkdir(parents=True)
    (out / "nested" / "run_done.json").write_text("{}", encoding="utf-8")
    check = backend(tmp_path, artifact="**/*_done.json").check_artifact("train", out)
    assert check["matched"] == 1 and check["examples"] == ["nested/run_done.json"]


def test_a_stage_that_promised_nothing_is_unverifiable_rather_than_passed(tmp_path):
    """The derivation is allowed to be vague; the backend may not call that success."""
    out = tmp_path / "out"
    out.mkdir()
    check = backend(tmp_path, artifact="").check_artifact("train", out)
    assert check["checked"] is False
    assert "named no artifact" in check["why"]


def test_an_artifact_that_exists_without_the_process_succeeding_is_not_a_pass(tmp_path):
    """Both have to hold; a file left by an earlier attempt is not this run's evidence."""
    class Runner:
        def run(self, argv, output, timeout):
            output.mkdir(parents=True, exist_ok=True)
            (output.parent / "run_done.json").write_text("{}", encoding="utf-8")
            return {"status": "failed", "returncode": 2}

    out = tmp_path / "out"
    result = backend(tmp_path).run_stage("train", Runner(),
                                         inputs={"python": "/p", "task": "t", "output": str(out),
                                                 "steps": 1}, output=out)
    assert result["returncode"] == 2
    assert result["verified"] is True  # the artifact is there; the process is not
    # Which is why both fields are reported rather than folded into one verdict: the
    # caller has to see that the file exists and the run failed.
    assert result["artifact"]["matched"] == 1
