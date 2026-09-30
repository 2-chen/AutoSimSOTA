"""Deriving how a benchmark runs, without a per-benchmark backend file.

No model is called. What is tested is the part that is a check rather than a judgement: the
answer must be complete enough to act on, and every path it names must exist. Neither
settles whether an entry point is the *right* one -- a repository with six policy families
has six trainers that all exist -- and the report is written so that this is not mistaken
for having been settled.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autosim.research import execution_derive
from autosim.research.execution_derive import STAGES, problems_in

PROJECT = Path(__file__).resolve().parents[1]


def test_checkpoint_probe_preserves_exact_discovered_artifact(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"model")
    declared = execution_derive.checkpoint_for_verification(
        {}, declaration={"assets": {"checkpoint": {"path": str(checkpoint)}}})
    assert declared["path"] == str(checkpoint)
    surveyed = execution_derive.checkpoint_for_verification(
        {"model_artifacts": [{"path": str(checkpoint), "found_under": str(tmp_path)}]})
    assert surveyed["path"] == str(checkpoint)


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
                                       "invocation": "python ...", "artifact": "checkpoints/*/model.json"}),
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


def test_evaluation_side_effect_is_not_a_separate_collection_stage(repo):
    """A second artifact glob cannot turn the evaluator's output into a collector."""
    value = answer(
        evaluate={"available": True, "entrypoint": "scripts/eval.py",
                  "invocation": "python scripts/eval.py --evaluate --save-trajectory",
                  "artifact": "runs/*/metrics.json"},
        collect={"available": True, "entrypoint": "scripts/eval.py",
                 "invocation": "python scripts/eval.py --evaluate --save-trajectory",
                 "artifact": "runs/*/trajectory.h5",
                 "evidence": "evaluation branch records its rollout"})

    faults = problems_in(value, repo)

    assert any("duplicates the evaluate command and context" in fault for fault in faults)


def test_independent_collection_mode_needs_and_accepts_source_evidence(repo):
    """One entrypoint can expose both modes, but they must be separately justified."""
    value = answer(
        evaluate={"available": True, "entrypoint": "scripts/eval.py",
                  "invocation": "python scripts/eval.py --evaluate --checkpoint={checkpoint}",
                  "artifact": "runs/*/metrics.json"},
        collect={"available": True, "entrypoint": "scripts/eval.py",
                 "invocation": "python scripts/eval.py --collect --episodes=100",
                 "artifact": "datasets/*.h5",
                 "evidence": "scripts/eval.py has a distinct collect branch that writes "
                            "datasets/*.h5 for the trainer"})

    assert problems_in(value, repo) == []
    value["stages"]["collect"].pop("evidence")
    assert any("stages.collect.evidence is required" in fault
               for fault in problems_in(value, repo))


def test_an_invented_stage_is_rejected(repo):
    value = answer()
    value["stages"]["finetune"] = {"available": False, "why": "no"}
    assert any("no such entry" in fault for fault in problems_in(value, repo))


def test_evidenced_extra_stage_and_explicit_graph_are_accepted(repo):
    (repo / "scripts" / "rollout.py").write_text("# native rollout\n", encoding="utf-8")
    value = answer(
        rollout={"available": True, "role": "collect", "why_required":
                 "online RL produces experience before updating", "evidence":
                 "scripts/rollout.py contains the native rollout entrypoint",
                 "entrypoint": "scripts/rollout.py", "invocation": "python scripts/rollout.py",
                 "artifact": "rollouts.json"})
    value["execution_graph"] = {"nodes": [{"id": "rollout", "role": "collect"}]}
    assert problems_in(value, repo) == []
    value["execution_graph"]["score_target"] = "rollout"
    assert any("score_target" in fault for fault in problems_in(value, repo))
    value["execution_graph"] = {"nodes": [{"id": "not_derived", "role": "evaluate"}]}
    assert any("no available stage" in fault for fault in problems_in(value, repo))


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


def test_incomplete_survey_is_retried_compactly_and_keeps_provider_finish_reason(repo):
    calls = []
    complete = answer(evaluate={"available": True, "entrypoint": "scripts/eval.py",
                                "invocation": "python scripts/eval.py",
                                "artifact": "metrics.json"})

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            calls.append({"system": system, "user": user, **kwargs})
            if "files_to_read" in system:
                return json.dumps({"files_to_read": ["scripts/eval.py"]}), {}
            if sum("stages" in call["system"] and "files_to_read" not in call["system"]
                   for call in calls) == 1:
                return json.dumps({"stages": {"evaluate": complete["stages"]["evaluate"]}}), {
                    "finish_reason": "length", "usage": {"completion_tokens": 4000}}
            return json.dumps(complete), {"finish_reason": "stop",
                                          "usage": {"completion_tokens": 900}}

    result = execution_derive.derive(repo, client=Client())
    survey_calls = [call for call in calls if "files_to_read" not in call["system"]]
    assert survey_calls[0]["max_tokens"] == 8000
    assert "stages.collect must be answered" in survey_calls[1]["user"]
    assert result["attempts"][0]["response_metadata"]["finish_reason"] == "length"
    assert result["attempts"][1]["response_metadata"]["finish_reason"] == "stop"
    assert result["stages"]["evaluate"]["available"] is True


def test_malformed_unbound_optional_stage_does_not_discard_valid_core_stages(tmp_path):
    from autosim.research.execution_derive import _discard_malformed_optional_stages

    value = {"stages": {"evaluate": {"available": True,
                                     "entrypoint": "eval.py", "invocation": "python eval.py",
                                     "artifact": "score.json"},
                        "record_demo": {"available": True, "artifact": "video.mp4"}}}
    (tmp_path / "eval.py").write_text("", encoding="utf-8")
    cleaned, discarded = _discard_malformed_optional_stages(value)
    assert discarded == ["record_demo"]
    assert problems_in(cleaned, tmp_path) == [
        "stages.prepare_data must be answered, including as unavailable",
        "stages.train must be answered, including as unavailable",
        "stages.collect must be answered, including as unavailable",
    ]
    value["execution_graph"] = {"nodes": [{"id": "record_demo", "role": "collect"}]}
    untouched, discarded = _discard_malformed_optional_stages(value)
    assert discarded == [] and "record_demo" in untouched["stages"]


def test_a_declared_parameter_must_say_where_its_value_came_from(repo):
    """The field exists so that an inferred value is visible as inferred.

    This is where a benchmark with several policy implementations gets disambiguated, and
    the disambiguation is a claim about the repository -- which can be checked, or at least
    read, but only if the evidence travels with the value.
    """
    faults = problems_in(answer(train={
        "available": True, "entrypoint": "scripts/train.py",
        "invocation": "python scripts/train.py --policy <name>", "artifact": "checkpoints/*/model.json",
        "parameters": [{"name": "--policy", "value": "act", "evidence": "  "}]}), repo)
    assert any("parameters[0].evidence is required" in fault for fault in faults)


def test_a_declared_parameter_must_have_a_value(repo):
    faults = problems_in(answer(train={
        "available": True, "entrypoint": "scripts/train.py",
        "invocation": "python scripts/train.py --policy <name>", "artifact": "checkpoints/*/model.json",
        "parameters": [{"name": "--policy", "value": "", "evidence": "the shipped checkpoints"}]}),
        repo)
    assert any("parameters[0].value is required" in fault for fault in faults)


def test_an_evidenced_parameter_is_accepted(repo):
    assert problems_in(answer(train={
        "available": True, "entrypoint": "scripts/train.py",
        "invocation": "python scripts/train.py --policy <name>", "artifact": "checkpoints/*/model.json",
        "parameters": [{"name": "--policy", "value": "act",
                        "evidence": "every shipped checkpoint is named ACT_sim_*"}]}),
        repo) == []


def test_a_stage_without_parameters_is_fine(repo):
    """Not every invocation needs a value the caller cannot supply."""
    assert problems_in(answer(evaluate={
        "available": True, "entrypoint": "scripts/eval.py", "invocation": "python scripts/eval.py",
        "artifact": "metrics.json"}), repo) == []


# -- the feedback a failed command produces -------------------------------------------------

def test_the_error_excerpt_finds_the_cause_not_the_tail():
    """A version banner printed on import must not be mistaken for the failure.

    This is not hypothetical: four attempts regenerated the same wrong command because the
    feedback was the last four lines of a program that announces itself on import.
    """
    from autosim.research.execution_derive import error_excerpt
    output = ("17:40:29 PyTorch version 2.7.1+cu128 available.\n"
              "17:40:29 Polars version 1.31.0 available.\n"
              "usage: eval_policy.py [-h] --config CONFIG [--overrides ...]\n"
              "eval_policy.py: error: unrecognized arguments: --checkpoint /x --task y\n")
    excerpt = error_excerpt(output)
    assert "unrecognized arguments" in excerpt
    assert "usage:" in excerpt


def test_the_error_excerpt_keeps_a_traceback_from_its_start():
    from autosim.research.execution_derive import error_excerpt
    output = "".join(f"line {i}\n" for i in range(40))
    output += ("Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n"
               "ModuleNotFoundError: No module named 'jax'\n")
    excerpt = error_excerpt(output)
    assert "ModuleNotFoundError" in excerpt and "Traceback" in excerpt


def test_the_error_excerpt_reaches_the_cause_of_a_nested_traceback():
    """A re-raised exception is at the *end*, past frames of library internals.

    Keeping a traceback from its start is right until the start is all a framework's own
    frames: hydra re-raises, so the first fourteen lines say only that something was
    re-raised, and the exception that names the problem is thirty lines further down. The
    model read exactly that and answered, correctly, that the output did not identify the
    cause -- and no change to the command came out of the round.
    """
    from autosim.research.execution_derive import error_excerpt
    output = "\n".join(
        ["[info] policy has 13.2 GFLOPs", "Traceback (most recent call last):"]
        + [f'  File "hydra/_internal/utils.py", line {i}, in run_and_report'
           for i in range(1, 30)]
        + ["LexerNoViableAltException: output_dir=/tmp/a b",
           "Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace."])
    excerpt = error_excerpt(output)
    assert "LexerNoViableAltException" in excerpt
    assert "Traceback (most recent call last)" in excerpt
    assert "..." in excerpt, "the gap between the two ends must be visible"


def test_the_error_excerpt_falls_back_to_the_tail_when_nothing_announces_itself():
    from autosim.research.execution_derive import error_excerpt
    assert error_excerpt("a\nb\nlast thing said") == "a\nb\nlast thing said"
    assert error_excerpt("") == "(no output)"


def test_a_parameter_value_must_be_a_literal_not_a_description(repo):
    """How an unsure answer looks, and why it is worth refusing.

    A command substitutes the value. Given "pi0 (directory policy/pi0), also pi0.5
    (policy/pi05)" the generated function tried to parse it -- with a method call the
    validator refuses -- instead of using it. Saying the value is unsettled is a legitimate
    answer and a different field.
    """
    faults = problems_in(answer(train={
        "available": True, "entrypoint": "scripts/train.py",
        "invocation": "python scripts/train.py --policy <name>",
        "artifact": "checkpoints/*/model.json",
        "parameters": [{"name": "--policy",
                        "value": "act, or possibly dp depending on the config",
                        "evidence": "the docs list several"}]}), repo)
    assert any("is a description, not a value" in fault for fault in faults)


def test_a_literal_parameter_value_is_accepted(repo):
    assert problems_in(answer(train={
        "available": True, "entrypoint": "scripts/train.py",
        "invocation": "python scripts/train.py --policy <name>",
        "artifact": "checkpoints/*/model.json",
        "parameters": [{"name": "--policy", "value": "act",
                        "evidence": "every shipped checkpoint is named ACT_sim_*"}]}),
        repo) == []


# -- an invocation is more than its arguments ----------------------------------------------

def test_a_stage_declares_where_it_runs_and_what_it_runs_with(repo):
    """Two things an invocation carries that are not arguments.

    A package whose top level has no `__init__.py` cannot be imported from outside its
    parent however right the arguments are, and a renderer needs telling to run headless.
    """
    from autosim.research.declarative_backend import (invocation_directory,
                                                      invocation_environment)
    row = {"working_directory": "{repo}", "environment": {"PYTHONPATH": "{repo}",
                                                          "MUJOCO_GL": "egl"}}
    assert invocation_environment(row, repo=repo) == {"PYTHONPATH": str(repo), "MUJOCO_GL": "egl"}
    assert invocation_directory(row, repo=repo, default=repo / "x") == repo
    assert invocation_directory({}, repo=repo, default=repo / "x") == repo / "x"
    assert invocation_directory({"working_directory": "scripts"}, repo=repo,
                                default=repo) == repo / "scripts"


def test_a_declared_directory_has_to_exist(repo):
    faults = problems_in(answer(train={
        "available": True, "entrypoint": "scripts/train.py",
        "invocation": "python scripts/train.py", "artifact": "out.json",
        "working_directory": "not/a/place"}), repo)
    assert any("working_directory does not exist" in fault for fault in faults)


def test_a_placeholder_directory_is_left_to_the_caller(repo):
    """`{repo}` names the checkout, which the reader of the declaration does not know."""
    assert problems_in(answer(train={
        "available": True, "entrypoint": "scripts/train.py",
        "invocation": "python scripts/train.py", "artifact": "out.json",
        "working_directory": "{repo}"}), repo) == []


def test_the_environment_must_be_a_map(repo):
    faults = problems_in(answer(train={
        "available": True, "entrypoint": "scripts/train.py",
        "invocation": "python scripts/train.py", "artifact": "out.json",
        "environment": ["PYTHONPATH=."]}), repo)
    assert any("must be a map of variable to value" in fault for fault in faults)


def test_a_revision_can_change_where_a_stage_runs(repo):
    """The argv is a function of the invocation, so a failure the invocation caused cannot
    be repaired by generating the argv again -- and the loop spins, which it did: five
    attempts at an import error, each regenerating a command whose arguments were never it."""
    import json as _json
    from autosim.research import execution_derive

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return _json.dumps({"working_directory": "{repo}",
                                "environment": {"PYTHONPATH": "{repo}", "MUJOCO_GL": "egl"},
                                "why": "the traceback is an import error, not an argument"}), {}

    change = execution_derive.revise_invocation(
        Client(), "evaluate", answer()["stages"]["train"], repo=repo, argv=["python", "x.py"],
        failure="ModuleNotFoundError: No module named 'x'")
    assert change["environment"] == {"PYTHONPATH": "{repo}", "MUJOCO_GL": "egl"}
    assert "import error" in change["why"]


def test_a_revision_can_say_the_invocation_is_not_the_problem(repo):
    """Which is the answer that stops the loop guessing, and the one a reader needs."""
    import json as _json
    from autosim.research import execution_derive

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return _json.dumps({"not_an_invocation_problem":
                                "the script dereferences a variable it never assigns"}), {}

    change = execution_derive.revise_invocation(
        Client(), "train", answer()["stages"]["train"], repo=repo, argv=["python", "x.py"],
        failure="UnboundLocalError: task_i_dataset")
    assert "never assigns" in change["obstacle"]


# -- reading a failure, and what the loop does about it --------------------------------------

def test_the_failure_reaches_the_model_as_the_program_wrote_it():
    """No verdict appended, and no keyword deciding what the failure *is*.

    There used to be two classifiers here -- `is_contract_error` and `is_environment_error`,
    seventeen substrings between them -- and the second one fired only on
    `no such file or directory: '/home`, so the same failure was an invocation problem on
    this machine and a value problem on any other. The conclusion they produced was appended
    to the output and read by the model as though the program had said it.

    What the model gets now is the output. It is the same reader that has to act on it.
    """
    from autosim.research import execution_derive
    for gone in ("is_contract_error", "is_environment_error"):
        assert not hasattr(execution_derive, gone), gone


def test_a_failing_command_is_retried_a_bounded_number_of_times(tmp_path):
    """A command whose shape cannot fix the failure is not worth five shapes -- and deciding
    that is the outer loop's, from what the diagnosis says, not this loop's from a substring.

    What this loop owes is a bound and a record: every attempt with its output, so the
    diagnosis has something to read and a reader can see what was tried.
    """
    import sys as _sys

    from autosim.research import execution_derive

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"source": "def stage_argv_train(i):\n    return [i['python']]\n",
                               "reasoning": "r", "shape_was_accepted": False}), {}

    source, parameters, log = execution_derive.generate_argv(
        Client(), "train", entrypoint="main.py", invocation="python main.py",
        repository_files=[],
        verify=lambda argv: {"ok": False,
                             "error": "ModuleNotFoundError: No module named 'example'"},
        inputs_for_verify={"python": _sys.executable}, attempts=3)
    assert source is None
    assert len(log) == 3
    # Every attempt carries the program's own output, which is what the diagnosis reads.
    assert all(entry["said"] for entry in log)
    assert all("No module named" in entry["said"] for entry in log)


def test_evaluator_must_use_the_caller_checkpoint_when_its_invocation_names_one():
    from autosim.research import execution_derive

    class Client:
        def __init__(self):
            self.calls = 0

        def chat_with_metadata(self, system, user, **kwargs):
            self.calls += 1
            policy = "'/default/model.pt'" if self.calls == 1 else "i['checkpoint']"
            return json.dumps({"source": "def stage_argv_evaluate(i):\n"
                                         "    return [i['python'], '--checkpoint', "
                                         + policy + "]\n"}), {}

    client = Client()
    seen = []
    source, _, log = execution_derive.generate_argv(
        client, "evaluate", entrypoint="eval.py",
        invocation="python eval.py --checkpoint PATH", repository_files=[],
        verify=lambda argv: seen.append(argv) or {"ok": True},
        inputs_for_verify={"python": sys.executable,
                           "checkpoint": "/frozen/candidate.pt"}, attempts=2)
    assert source is not None and client.calls == 2
    assert len(seen) == 1 and "/frozen/candidate.pt" in seen[0]
    assert "ignored the caller's checkpoint" in log[0]["error"]


def test_verification_caller_checkpoint_overrides_declared_checkpoint_parameter():
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"source": "def stage_argv_evaluate(i):\n"
                                         "    return [i['python'], '--checkpoint', i['checkpoint']]\n",
                               "reasoning": "use the caller policy"}), {}

    seen = []
    source, _, _ = execution_derive.generate_argv(
        Client(), "evaluate", entrypoint="eval.py",
        invocation="python eval.py --checkpoint PATH", repository_files=[],
        declared_parameters={"--checkpoint": "/default/or/template.pt"},
        verify=lambda argv: seen.append(argv) or {"ok": True},
        inputs_for_verify={"python": sys.executable,
                           "checkpoint": "/frozen/candidate.pt"}, attempts=1)
    assert source is not None
    assert seen == [[sys.executable, "--checkpoint", "/frozen/candidate.pt"]]


def test_training_verification_requires_a_bound_step_budget():
    class Client:
        def __init__(self):
            self.calls = 0

        def chat_with_metadata(self, system, user, **kwargs):
            self.calls += 1
            source = ("def stage_argv_train(i):\n    return [i['python'], '--steps', '2000000']\n"
                      if self.calls == 1 else
                      "def stage_argv_train(i):\n    return [i['python'], '--steps', str(i['steps'])]\n")
            return json.dumps({"source": source}), {}

    client = Client()
    seen = []
    source, _, log = execution_derive.generate_argv(
        client, "train", entrypoint="train.py", invocation="python train.py --steps 2000000",
        repository_files=[], verify=lambda argv: seen.append(argv) or {"ok": True},
        inputs_for_verify={"python": sys.executable, "steps": 1}, attempts=2,
        require_step_control=True)
    assert source is not None and client.calls == 2
    assert seen == [[sys.executable, "--steps", "1"]]
    assert "ignores the caller's steps" in log[0]["error"]


def test_repository_full_run_budget_is_not_frozen_as_a_training_parameter():
    class Client:
        def __init__(self):
            self.prompts = []

        def chat_with_metadata(self, system, user, **kwargs):
            self.prompts.append(json.loads(user))
            return json.dumps({
                "source": ("def stage_argv_train(i):\n"
                           "    return [i['python'], i['repo'] + '/train.py', "
                           "'--total_timesteps=' + str(i['steps'])]\n")}), {}

    client = Client()
    seen = []
    source, _, log = execution_derive.generate_argv(
        client, "train", entrypoint="train.py",
        invocation="python train.py --total_timesteps=2000000",
        repository_files=[], declared_parameters={"--total_timesteps": "2000000",
                                                    "--env_id": "PushCube-v1"},
        verify=lambda argv: seen.append(argv) or {"ok": True},
        inputs_for_verify={"python": sys.executable, "repo": "/checkout", "steps": 1024},
        attempts=1, require_step_control=True)
    assert source is not None and log[-1]["status"] == "accepted"
    assert "--total_timesteps" not in client.prompts[0][
        "values_already_settled_for_this_repository"]
    assert client.prompts[0]["values_already_settled_for_this_repository"][
        "--env_id"] == "PushCube-v1"
    assert seen == [[sys.executable, "/checkout/train.py", "--total_timesteps=1024"]]


def test_total_work_control_detection_excludes_rollout_and_episode_shape():
    is_total_work = execution_derive.is_total_work_control
    assert is_total_work("--total_timesteps")
    assert is_total_work("--max-updates")
    assert is_total_work("num_epochs")
    assert not is_total_work("--num_envs")
    assert not is_total_work("--num_steps")
    assert not is_total_work("--max_episode_steps")


def test_native_rollout_counters_give_a_bounded_one_update_hint():
    guidance = execution_derive.bounded_training_batch_guidance(
        {"batch_size": 3200, "num_envs": 64, "num_steps": 50}, steps=1024)
    assert "64 × 50 = 3200" in guidance
    assert "num_envs ≤ 20" in guidance
    assert execution_derive.bounded_training_batch_guidance(
        {"batch_size": 3200, "num_envs": 64, "num_steps": 49}, steps=1024) == ""
    inferred = execution_derive.bounded_training_batch_guidance(
        {"batch_size": 40960, "num_envs": 2048}, steps=1024,
        argv=["python", "ppo.py", "--num-steps=20"])
    assert "2048 × 20 = 40960" in inferred
    assert "num_envs ≤ 51" in inferred


def test_redacted_artifact_revision_preserves_local_source_declaration():
    original = {"artifact": "runs/*/final_ckpt.pt", "invocation": "python train.py"}
    changed = execution_derive.preserve_local_artifact(
        original, {"artifact": "runs/**[WITHHELD_ASSET_PATH]",
                   "invocation": "python train.py --env-id PushCube-v1"})
    assert changed["artifact"] == original["artifact"]
    assert changed["invocation"] == "python train.py --env-id PushCube-v1"


def test_make_runnable_stops_after_three_same_failure_classes(tmp_path, monkeypatch):
    drafts = []
    revisions = []

    def generate(*args, **kwargs):
        drafts.append(kwargs.get("invocation"))
        return None, {}, [{"status": "rejected",
                           "error": "RuntimeError: trainer exited zero; num_iterations=0"}]

    def diagnose(*args, **kwargs):
        return {"finding": "native training made zero updates"}

    def revise(*args, **kwargs):
        revisions.append(len(revisions))
        return {"invocation": f"python train.py --attempt={len(revisions)}",
                "why": "bounded diagnostic revision"}

    monkeypatch.setattr(execution_derive, "generate_argv", generate)
    monkeypatch.setattr(execution_derive, "diagnose", diagnose)
    monkeypatch.setattr(execution_derive, "revise_invocation", revise)
    events = []
    source, _, log, _ = execution_derive.make_runnable(
        object(), "train", {"entrypoint": "train.py", "invocation": "python train.py",
                            "artifact": "", "parameters": []}, repo=tmp_path,
        repository_files=[], inputs_for_verify={"python": sys.executable}, rounds=40,
        attempts=8, on_event=lambda stage, entries: events.extend(entries))
    assert source is None and len(drafts) == 3 and len(revisions) == 2
    assert any("same failure repeated" in str(event.get("status")) for event in events)
    assert any("same stage-verification failure" in str(event.get("error"))
               for event in events)
    assert any("same stage-verification failure" in row.get("error", "") for row in log)


def test_training_verification_binds_exact_steps_placeholder_parameter():
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({
                "source": ("def stage_argv_train(i):\n"
                           "    return [i['python'], '--total_timesteps=' + "
                           "str(i['total_timesteps'])]\n"),
                "parameters": {"--total_timesteps": "i['steps']"},
            }), {}

    client = Client()
    seen = []
    source, _, log = execution_derive.generate_argv(
        client, "train", entrypoint="train.py",
        invocation="python train.py --total_timesteps={steps}",
        repository_files=[], verify=lambda argv: seen.append(argv) or {"ok": True},
        inputs_for_verify={"python": sys.executable, "steps": 1024}, attempts=1,
        require_step_control=True)
    assert source is not None
    assert seen == [[sys.executable, "--total_timesteps=1024"]]
    assert len(log) == 1 and log[0]["status"] == "accepted"


def test_steps_placeholder_binding_is_exact_and_does_not_interpret_prose():
    bind = execution_derive.bind_step_budget_placeholders
    assert bind(["--total_steps={steps}", "--iterations", 'i["steps"]'], 1024) == [
        "--total_steps=1024", "--iterations", "1024"]
    assert bind(["--total_steps=budget from i['steps']"], 1024) == [
        "--total_steps=budget from i['steps']"]


def test_step_budget_cannot_be_satisfied_by_changing_parallelism():
    problem = execution_derive.step_budget_problem(
        ["python", "train.py", "--total_timesteps", "2_000_000", "--num_envs", "1024"],
        ["python", "train.py", "--total_timesteps", "2_000_000", "--num_envs", "1025"],
        steps=1024)
    assert "without changing its native total-work limit" in problem
    assert execution_derive.step_budget_problem(
        ["python", "train.py", "--total_timesteps=1024", "--num_envs=8"],
        ["python", "train.py", "--total_timesteps=1025", "--num_envs=8"],
        steps=1024) == ""
    assert "exceeds" in execution_derive.step_budget_problem(
        ["python", "train.py", "--total_timesteps=2_000_000"],
        ["python", "train.py", "--total_timesteps=2_000_001"], steps=1024)
    assert "repeats" in execution_derive.step_budget_problem(
        ["python", "train.py", "--total_timesteps=2_000_000", "--total_timesteps=1024"],
        ["python", "train.py", "--total_timesteps=2_000_000", "--total_timesteps=1025"],
        steps=1024)


def test_coarse_native_epoch_budget_can_be_derived_from_caller_steps():
    assert execution_derive.step_budget_problem(
        ["python", "train.py", "train.n_epochs=1"],
        ["python", "train.py", "train.n_epochs=2"], steps=1024) == ""
    assert "must increase" in execution_derive.step_budget_problem(
        ["python", "train.py", "train.n_epochs=2"],
        ["python", "train.py", "train.n_epochs=1"], steps=1024)

    class Client:
        def chat_with_metadata(self, *_args, **_kwargs):
            return json.dumps({"source": "def stage_argv_train(i):\n"
                                         "    epochs = max(1, i['steps'] // 1024)\n"
                                         "    return [i['python'], 'train.py', "
                                         "'train.n_epochs=' + str(epochs)]\n",
                               "reasoning": "native work unit is an epoch"}), {}

    seen = []
    source, _, log = execution_derive.generate_argv(
        Client(), "train", entrypoint="train.py", invocation="python train.py",
        repository_files=[], verify=lambda argv: seen.append(argv) or {"ok": True},
        inputs_for_verify={"python": sys.executable, "steps": 1024}, attempts=1,
        require_step_control=True)
    assert source is not None and log[0]["status"] == "accepted"
    assert seen == [[sys.executable, "train.py", "train.n_epochs=1"]]
def test_the_data_a_missing_path_was_pointing_at_is_looked_for(tmp_path):
    """LIBERO's loader joins `{folder}/{problem}/{task}.hdf5` against its own checkout, which
    ships no dataset; the data is in a sibling directory. Nothing in the repository says so,
    and the answer is on the disk -- so the loop surveys rather than guesses."""
    from autosim.research.execution_derive import data_near
    checkout = tmp_path / "test" / "LIBERO"
    (checkout / "libero").mkdir(parents=True)
    (checkout / "setup.py").write_text("#\n", encoding="utf-8")
    data = tmp_path / "test" / "datasets" / "libero" / "libero_10"
    data.mkdir(parents=True)
    (data / "task_demo.hdf5").write_bytes(b"\x00")
    (tmp_path / "test" / "notes.txt").write_text("not data\n", encoding="utf-8")
    found = data_near(checkout)
    assert str(tmp_path / "test" / "datasets") in found
    assert found[str(tmp_path / "test" / "datasets")]["files"] == 1
    assert "libero_10/task_demo.hdf5" in found[str(tmp_path / "test" / "datasets")]["examples"][0]
    assert str(checkout) not in found, "the checkout itself is not 'near the checkout'"


def test_the_reviser_is_shown_the_changes_it_already_made(tmp_path):
    """A revision that made things worse has to be visible, or it cannot be taken back.

    One added `PYTHONNOUSERSITE=1` as a precaution and hid the matplotlib that LIBERO was
    importing from user site-packages; without the history the next revision had no way to
    know the change was its own.
    """
    from autosim.research import execution_derive
    seen = {}

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            seen["payload"] = json.loads(user)
            return json.dumps({"not_an_invocation_problem": "the dataset is absent"}), {}

    execution_derive.revise_invocation(
        Client(), "train", {"entrypoint": "main.py", "invocation": "python main.py"},
        repo=tmp_path, argv=["python", "main.py"], failure="FileNotFoundError",
        attempted=[{"changed": {"environment": {"PYTHONNOUSERSITE": "1"}},
                    "because": "make the run hermetic"}])
    assert seen["payload"]["changes_already_tried_and_what_they_were_for"] == [
        {"changed": {"environment": {"PYTHONNOUSERSITE": "1"}}, "because": "make the run hermetic"}]
    assert "data_directories_near_the_checkout" in seen["payload"]


def test_a_stage_that_cannot_be_ruled_out_is_not_answered_with_the_word_omit(tmp_path):
    """The prompt said "otherwise omit", and a model answered with the literal word, which
    the caller then read as an obstacle named "omit"."""
    from autosim.research import execution_derive
    assert 'otherwise omit' not in execution_derive.REVISE_SYSTEM
    assert '"not_an_invocation_problem": "<null unless' in execution_derive.REVISE_SYSTEM

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"not_an_invocation_problem": None,
                               "working_directory": "{repo}",
                               "environment": {"PYTHONPATH": "{repo}"},
                               "why": "the package root is the parent"}), {}

    change = execution_derive.revise_invocation(
        Client(), "train", {"entrypoint": "main.py", "invocation": "python main.py"},
        repo=tmp_path, argv=["python", "main.py"], failure="ModuleNotFoundError")
    assert change == {"working_directory": "{repo}", "environment": {"PYTHONPATH": "{repo}"},
                      "why": "the package root is the parent"}
    assert "obstacle" not in change


def test_a_command_defect_the_model_diagnoses_is_handed_back_to_the_writer(tmp_path):
    """"The command is wrong, not where it runs" is a finding about the command.

    The loop ended on it, so the model's own diagnosis -- "the dataset path is redirected via
    a parameter" -- was discarded by the one part of the system that could act on it. It now
    goes back into the next generation as guidance, and the parameter that was plumbed
    through for it is actually set, which a previous edit had left unassigned.
    """
    from autosim.research import execution_derive
    seen = []

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            payload = json.loads(user)
            if "what_the_last_round_established" in payload:
                seen.append(payload["what_the_last_round_established"])
            if "not_an_invocation_problem" in system or "as_derived" in payload:
                return json.dumps({"working_directory": "",
                                   "not_an_invocation_problem":
                                       "the dataset is not in the checkout; the loader's "
                                       "folder override must point at it"}), {}
            # A real command that really fails: `make_runnable` builds its own verifier, so
            # the only honest way to test the loop is to let it run something.
            return json.dumps({"source": "def stage_argv_train(i):\n"
                                         "    return [i['python'], '-c', "
                                         "'raise SystemExit(1)']\n",
                               "reasoning": "r"}), {}

    import sys as _sys
    source, parameters, log, row = execution_derive.make_runnable(
        Client(), "train", {"entrypoint": "main.py", "invocation": "python main.py",
                            "parameters": []},
        repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": _sys.executable}, rounds=3, attempts=1)
    assert source is None
    assert seen, "the diagnosis never reached the next generation"
    assert "folder override must point at it" in seen[-1], seen


def test_a_field_the_reviser_left_null_does_not_overwrite_the_stage(tmp_path):
    """Null means "unchanged", and merging it meant the row's invocation became None.

    The prompt offers `invocation` and `parameters` as "if that was also wrong", so the
    natural answer when they are right is `null`. The next draft was then built from the
    string "None", which no amount of reading the final command would explain.
    """
    from autosim.research import execution_derive

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"working_directory": "{repo}",
                               "environment": {"PYTHONPATH": "{repo}"},
                               "invocation": None, "parameters": None,
                               "why": "the checkout root was not on the search path"}), {}

    row = {"entrypoint": "libero/lifelong/main.py",
           "invocation": "python libero/lifelong/main.py benchmark_name=LIBERO_10"}
    change = execution_derive.revise_invocation(
        Client(), "train", row, repo=tmp_path, argv=["python", "main.py"],
        failure="ModuleNotFoundError: No module named 'libero'")
    assert "invocation" not in change and "parameters" not in change
    assert change["environment"] == {"PYTHONPATH": "{repo}"}


def test_parameters_may_come_back_in_the_shape_the_derivation_uses(tmp_path):
    """A model shown a list of `{name, value, evidence}` records answers in that shape."""
    from autosim.research import execution_derive

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({
                "working_directory": "{repo}", "environment": {},
                "parameters": [{"name": "folder", "value": "/data/libero",
                                "evidence": "the only directory holding .hdf5"},
                               {"name": "--task_id", "value": "0", "evidence": "it is the "
                                "first task and the assertion bounds it by task_id"}],
                "why": "the loader's folder default points inside the checkout"}), {}

    change = execution_derive.revise_invocation(
        Client(), "train", {"entrypoint": "main.py", "invocation": "python main.py"},
        repo=tmp_path, argv=["python", "main.py"], failure="FileNotFoundError")
    assert change["parameters"] == {"folder": "/data/libero", "--task_id": "0"}


def test_a_revision_that_changes_nothing_hands_its_reasoning_to_the_writer(tmp_path):
    """"The invocation is already right" is a finding about the command.

    LIBERO ended here: the invocation it had was the correct one, the generated argv appended
    an override hydra refuses, and the reviser said exactly that -- "stop passing output_dir"
    -- in a round where nothing about the invocation needed to change. The loop discarded it
    because the round's *change* was empty.
    """
    from autosim.research import execution_derive
    seen = []

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            payload = json.loads(user)
            if "what_the_last_round_established" in payload:
                seen.append(payload["what_the_last_round_established"])
            if "as_derived" in payload:
                return json.dumps({
                    "working_directory": "{repo}", "environment": {"PYTHONPATH": "{repo}"},
                    "invocation": "python main.py benchmark_name=LIBERO_10",
                    "why": "the invocation is right; the argv appends an override hydra "
                           "refuses because output_dir is not a key in the config"}), {}
            return json.dumps({"source": "def stage_argv_train(i):\n"
                                         "    return [i['python'], '-c', "
                                         "'raise SystemExit(1)']\n",
                               "reasoning": "r"}), {}

    import sys as _sys
    row = {"entrypoint": "main.py", "invocation": "python main.py benchmark_name=LIBERO_10",
           "working_directory": "{repo}", "environment": {"PYTHONPATH": "{repo}"},
           "parameters": []}
    source, parameters, log, row = execution_derive.make_runnable(
        Client(), "train", row, repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": _sys.executable}, rounds=4, attempts=1)
    assert source is None
    assert seen, "the reasoning never reached the next generation"
    assert "not a key in the config" in seen[-1]


def test_a_draft_that_cannot_be_executed_is_rejected_not_fatal(tmp_path):
    """A draft can drop the interpreter, and then argv[0] is the script.

    That raises `PermissionError` from `subprocess`, which escaped the verifier and ended the
    run -- a derivation one round from finishing, killed by one bad draft. Nothing a draft
    contains may be able to do that.
    """
    from autosim.research import execution_derive
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            if "as_derived" in user:
                return json.dumps({"working_directory": "{repo}",
                                   "not_an_invocation_problem":
                                       "the draft is malformed, not the invocation"}), {}
            return json.dumps({"source": "def stage_argv_train(i):\n"
                                         "    return [i['repo'] + '/main.py']\n",
                               "reasoning": "no interpreter"}), {}

    import sys as _sys
    # No assertion about the outcome beyond this: the point is that it returns at all.
    source, parameters, log, row = execution_derive.make_runnable(
        Client(), "train", {"entrypoint": "main.py", "invocation": "python main.py",
                            "parameters": []},
        repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": _sys.executable, "repo": str(tmp_path)},
        rounds=2, attempts=1)
    # Which of the two rejections it lands in is a judgement about the failure; what matters
    # here is that there is one, that it names the cause, and that the loop came back.
    assert source is None
    assert log and log[0]["status"] in ("rejected", "not the command's fault")
    assert "PermissionError" in log[0]["error"], log[0]["error"]
    assert log[0]["argv"] == [str(tmp_path / "main.py")]


def test_train_command_verification_rejects_explicit_zero_updates(tmp_path):
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            if "as_derived" in user:
                return json.dumps({"working_directory": "{repo}",
                                   "not_an_invocation_problem": "zero planned updates"}), {}
            return json.dumps({"source": "def stage_argv_train(i):\n"
                                         "    return [i['python'], '-c', "
                                         "'print(\"args.num_iterations=0 "
                                         "args.batch_size=40960 args.num_envs=2048\")', "
                                         "'--steps=' + str(i['steps'])]\n",
                               "reasoning": "smoke"}), {}

    source, _, log, _ = execution_derive.make_runnable(
        Client(), "train", {"entrypoint": "train.py", "invocation": "python train.py",
                            "parameters": []},
        repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": sys.executable, "repo": str(tmp_path), "steps": 1024},
        rounds=1, attempts=1, require_step_control=True)
    assert source is None
    assert any("positive training progress was not verified" in str(row.get("error"))
               for row in log)
    assert any("batch_size" in str(row.get("error")) for row in log)


def test_evaluate_command_verification_rejects_explicit_zero_episodes(tmp_path):
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            if "as_derived" in user:
                return json.dumps({"working_directory": "{repo}",
                                   "not_an_invocation_problem":
                                       "the evaluator accepted its arguments but ran no episodes"}), {}
            return json.dumps({"source": "def stage_argv_evaluate(i):\n"
                                         "    return [i['python'], '-c', "
                                         "'print(\\\"Evaluated 8 steps resulting in 0 episodes\\\")']\n",
                               "reasoning": "bounded evaluator smoke"}), {}

    source, _, log, _ = execution_derive.make_runnable(
        Client(), "evaluate", {"entrypoint": "eval.py", "invocation": "python eval.py",
                               "parameters": []},
        repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": sys.executable, "repo": str(tmp_path), "episodes": 1},
        rounds=1, attempts=1, require_evaluation_progress=True)
    assert source is None
    assert any("zero completed episodes" in str(row.get("error")) for row in log)


def test_identical_rejected_argv_gets_at_most_one_runtime_retry():
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"source": "def stage_argv_train(i):\n"
                                         "    return [i['python'], '-c', 'raise RuntimeError()']\n",
                               "reasoning": "same command"}), {}

    executions = []

    def verify(argv):
        executions.append(argv)
        return {"ok": False, "error": "RuntimeError: native failure"}

    source, _, log = execution_derive.generate_argv(
        Client(), "train", entrypoint="train.py", invocation="python train.py",
        repository_files=[], verify=verify,
        inputs_for_verify={"python": sys.executable}, attempts=5)
    assert source is None
    assert len(executions) == 2
    assert len(log) == 5
    assert "Exact argv already failed twice" in log[-1]["error"]


def test_exhausted_wall_budget_stops_command_derivation_rounds(tmp_path):
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            raise TimeoutError("run wall-clock budget exhausted before model request")

    events = []
    source, _, log, _ = execution_derive.make_runnable(
        Client(), "train", {"entrypoint": "train.py", "invocation": "python train.py",
                            "parameters": []}, repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": sys.executable}, rounds=40, attempts=1,
        on_event=lambda stage, rows: events.extend(rows))
    assert source is None
    assert len(events) == 1
    assert "wall-clock budget exhausted" in log[0]["error"]


def test_a_response_that_is_not_json_is_a_rejected_draft_not_a_crash(tmp_path):
    """The handler for a bad response runs when something has already gone wrong.

    It named the parsed source, which is not bound yet when the response is not JSON at all,
    so the handler raised `UnboundLocalError` and ended the run. An LLM that answers with a
    paragraph is an ordinary event in a loop that runs for hours; it may not be fatal.
    """
    from autosim.research import execution_derive

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return "I'm sorry, I can't help with that.", {}

    source, parameters, log = execution_derive.generate_argv(
        Client(), "train", entrypoint="main.py", invocation="python main.py",
        repository_files=[], verify=None, inputs_for_verify={}, attempts=1)
    assert source is None
    assert log and log[0]["status"] == "rejected"
    assert "no JSON object" in log[0]["error"]
    assert log[0]["source"] == "I'm sorry, I can't help with that."


def test_a_round_that_raises_does_not_end_the_loop(tmp_path):
    """Four separate crashes came out of one function, and each took the whole run with it.

    A draft that dropped the interpreter; a survey that walked into a file it could not
    read; a handler that named a variable the failure had prevented from being bound; a
    model that answered with a paragraph. A derivation that has spent an hour getting four
    of five facts right should not lose all of them to a fifth that raised.
    """
    from autosim.research import execution_derive
    rounds = []

    class Client:
        def __init__(self):
            self.n = 0

        def chat_with_metadata(self, system, user, **kwargs):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("the transport went away")
            rounds.append(self.n)
            return json.dumps({"working_directory": "{repo}",
                               "not_an_invocation_problem": "nothing further"}), {}

    import sys as _sys
    source, parameters, log, row = execution_derive.make_runnable(
        Client(), "train", {"entrypoint": "main.py", "invocation": "python main.py",
                            "parameters": []},
        repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": _sys.executable}, rounds=3, attempts=1)
    assert source is None
    assert rounds, "the loop never reached a second round"


def test_a_program_that_asks_a_question_can_be_answered(tmp_path):
    """A benchmark that prompts on first import cannot be run unattended.

    LIBERO asks whether to store its datasets somewhere custom and writes a default
    configuration either way. The answer is not an argument, so the invocation carries it —
    and with none declared the input is closed rather than inherited, because an unattended
    stage that inherits a terminal waits at the question until its timeout.
    """
    from autosim.research import execution_derive
    from autosim.research.declarative_backend import invocation_stdin
    assert invocation_stdin({"stdin": "N\n"}) == "N\n"
    assert invocation_stdin({}) is None

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"source":
                               "def stage_argv_train(i):\n"
                               "    return [i['python'], '-c', "
                               "'import sys; print(sys.stdin.readline().strip())']\n",
                               "reasoning": "reads stdin"}), {}

    import sys as _sys
    source, parameters, log = execution_derive.generate_argv(
        Client(), "train", entrypoint="main.py", invocation="python main.py",
        repository_files=[], verify=None, inputs_for_verify={}, attempts=1)
    assert source is not None
    # With the prompt unanswered the program gets EOF instead of blocking forever.
    import subprocess

    from autosim.research.patch_validation import checked_function
    function = checked_function(source, "stage_argv_train")
    argv = function({"python": _sys.executable})
    done = subprocess.run(argv, text=True, capture_output=True, stdin=subprocess.DEVNULL)
    assert done.returncode == 0 and done.stdout.strip() == ""


def test_successful_stage_verification_persists_fresh_structured_sidecars(tmp_path):
    code = ("import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); "
            "(p/'trajectory.h5').write_bytes(b'native-output'); "
            "(p/'episodes.json').write_text(json.dumps({'episodes':["
            "{'episode_id':0,'success':True}]}))")

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"source": "def stage_argv_evaluate(i):\n"
                                        "    return [i['python'], '-c', " + repr(code) +
                                        ", i['output']]\n",
                               "reasoning": "use the evaluator's native result"}), {}

    import sys as _sys
    output = tmp_path / "verify-output"
    source, _, log, _ = execution_derive.make_runnable(
        Client(), "evaluate", {"entrypoint": "eval.py", "invocation": "python eval.py",
                                     "artifact": "trajectory.h5", "parameters": []},
        repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": _sys.executable, "output": str(output)},
        attempts=1, rounds=1)

    assert source is not None, log
    accepted = next(row for row in log if row.get("status") == "accepted")
    verified = accepted["verified_artifact"]
    candidate = next((row for row in verified["structured_candidates"]
                      if row["relative_path"] == "episodes.json"), None)
    assert candidate is not None, verified
    assert verified["matched"] == 1
    assert candidate["root"] == "output"
    assert candidate["sha256"]
    assert Path(candidate["path"]).is_relative_to(output)


def test_multiple_declared_outputs_keep_adjacent_episode_sidecar_from_policy_parent(tmp_path):
    checkpoint = tmp_path / "runs" / "ppo" / "final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen policy")
    code = ("import json,pathlib,sys; root=pathlib.Path(sys.argv[1]).parent/'test_videos'; "
            "root.mkdir(parents=True,exist_ok=True); "
            "[(root/f'{i}.mp4').write_bytes(b'video') for i in range(2)]; "
            "(root/'trajectory.json').write_text(json.dumps({'env_info':"
            "{'env_id':'PushCube-v1'},'episodes':[{'episode_id':0,'success':False}]}))")

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"source": "def stage_argv_evaluate(i):\n"
                                        "    return [i['python'], '-c', " + repr(code) +
                                        ", i['checkpoint']]\n",
                               "reasoning": "native evaluator emits an episode sidecar"}), {}

    output = tmp_path / "verify-output"
    source, _, log, _ = execution_derive.make_runnable(
        Client(), "evaluate", {"entrypoint": "eval.py", "invocation": "python eval.py",
                                     "artifact": "test_videos/*.mp4", "parameters": []},
        repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": sys.executable, "output": str(output),
                           "checkpoint": str(checkpoint)},
        attempts=1, rounds=1)

    assert source is not None, log
    accepted = next(row for row in log if row.get("status") == "accepted")
    verified = accepted["verified_artifact"]
    assert verified["matched"] == 2
    candidate = next(row for row in verified["structured_candidates"]
                     if row["relative_path"] == "test_videos/trajectory.json")
    assert candidate["root"] == "policy_parent"
    assert candidate["sha256"]


def test_the_file_the_program_named_is_looked_for_by_name(tmp_path):
    """The program has already said which file it wanted; that is a far better signal than a
    survey of names.

    LIBERO's trainer printed `unable to open file: name = '.../LIVING_ROOM_..._demo.hdf5'`,
    and a survey that reported the first three names in each nearby directory led the reviser
    to conclude -- from those three, correctly -- that the directory held a different
    benchmark's data. The file it was asking about was two levels further down.
    """
    from autosim.research.execution_derive import files_the_program_could_not_open
    data = tmp_path / "datasets" / "libero" / "libero_10"
    data.mkdir(parents=True)
    wanted = data / "LIVING_ROOM_SCENE2_put_the_soup_in_the_basket_demo.hdf5"
    wanted.write_bytes(b"\x00")
    failure = ("[error] [Errno 2] Unable to synchronously open file (unable to open file: "
               f"name = '{tmp_path}/libero/libero/../datasets/libero_10/"
               "LIVING_ROOM_SCENE2_put_the_soup_in_the_basket_demo.hdf5', errno = 2)")
    found = files_the_program_could_not_open(failure, roots=[tmp_path])
    assert list(found) == ["LIVING_ROOM_SCENE2_put_the_soup_in_the_basket_demo.hdf5"]
    assert found[list(found)[0]] == [str(wanted)]
    # A quoted word in prose is not a missing file.
    assert files_the_program_could_not_open("see 'the docs' for details",
                                            roots=[tmp_path]) == {}


def test_the_survey_says_which_subdirectory_holds_the_data(tmp_path):
    """File names alone do not tell one benchmark's directory from another's.

    `datasets/` reported three `demo_v15.hdf5` and read as a robomimic directory, when it
    holds every suite LIBERO ships a level further down. The immediate child is the fact
    that distinguishes them.
    """
    from autosim.research.execution_derive import data_near
    checkout = tmp_path / "test" / "LIBERO"
    checkout.mkdir(parents=True)
    (tmp_path / "test" / "datasets" / "robomimic" / "v1.5").mkdir(parents=True)
    (tmp_path / "test" / "datasets" / "robomimic" / "v1.5" / "demo_v15.hdf5").write_bytes(b"\x00")
    suite = tmp_path / "test" / "datasets" / "libero" / "libero_10"
    suite.mkdir(parents=True)
    (suite / "task_demo.hdf5").write_bytes(b"\x00")
    entry = data_near(checkout)[str(tmp_path / "test" / "datasets")]
    assert entry["subdirectories_holding_data"] == ["robomimic", "libero"]


def test_a_value_a_rejected_draft_established_is_kept(tmp_path):
    """Facts accumulate across attempts, because a command's rejection says nothing about
    the values it carried.

    Without this every round re-derived everything from nothing: LIBERO's dataset directory
    was established in one round, absent from the next round's command, and re-derived in the
    round after that -- which is what forty rounds of the same four guesses looks like.
    """
    from autosim.research import execution_derive
    seen = []

    class Client:
        def __init__(self):
            self.n = 0

        def chat_with_metadata(self, system, user, **kwargs):
            # The retry prompt is the original request with the rejection appended, so it is
            # not one JSON document. Read the first one and ignore what follows.
            payload = json.JSONDecoder().raw_decode(user.strip())[0]
            seen.append(dict(payload["values_already_settled_for_this_repository"]))
            self.n += 1
            return json.dumps({
                "source": "def stage_argv_train(i):\n"
                          "    return [i['python'], '-c', 'raise SystemExit(1)']\n",
                "reasoning": "wrong on purpose",
                # The first draft establishes the value; every later one repeats it.
                "parameters": {"folder": "/data/libero"}}), {}

    import sys as _sys
    source, parameters, log = execution_derive.generate_argv(
        Client(), "train", entrypoint="main.py", invocation="python main.py",
        repository_files=[], verify=lambda argv: {"ok": False, "error": "nope"},
        inputs_for_verify={"python": _sys.executable}, attempts=3)
    assert source is None
    assert parameters.get("folder") == "/data/libero"
    # The second attempt was told about it, so it did not have to invent it again.
    assert seen[1].get("folder") == "/data/libero", seen


def test_the_config_keys_the_program_printed_reach_the_next_draft(tmp_path):
    """The retry is where the key gets guessed, so the key list has to be there.

    A program that composes a configuration prints it before it refuses an override, and that
    print names every key it has. Quoted it is a hundred lines and does not fit a prompt; by
    name it is one line. LIBERO printed `folder` on every run while successive drafts reached
    for `data_root`, `data.dataset_path` and `data.data_folder`, none of which exist.
    """
    from autosim.research import execution_derive
    contexts = []

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            contexts.append(user)
            return json.dumps({"source": "def stage_argv_train(i):\n"
                                         "    return [i['python'], '-c', "
                                         "'raise SystemExit(1)']\n",
                               "reasoning": "wrong on purpose"}), {}

    import sys as _sys
    dump = "\n".join(["  'bddl_folder': None,", "  'benchmark_name': 'LIBERO_10',",
                      "  'folder': None,"] + [f"  'k{i}': 1," for i in range(10)])
    execution_derive.generate_argv(
        Client(), "train", entrypoint="main.py", invocation="python main.py",
        repository_files=[],
        verify=lambda argv: {"ok": False, "error": dump + "\nError: no such override"},
        inputs_for_verify={"python": _sys.executable}, attempts=2)
    assert len(contexts) >= 2, contexts
    assert "the_program_printed_these_config_keys" in contexts[1]
    assert "'folder'" in contexts[1] or '"folder"' in contexts[1]


def test_the_revised_environment_comes_back_with_the_command(tmp_path):
    """The source is not the invocation, and the driver used to keep only the source.

    A revision that adds `PYTHONPATH` is what got the program to import at all. The driver
    dropped the revised row, so every stage that verified ran afterwards without it and died
    at its first import -- `ModuleNotFoundError: No module named 'libero'`, from a command
    that had just been shown to work. The row is returned for that reason.
    """
    from autosim.research import execution_derive

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            if "as_derived" in user:
                return json.dumps({"working_directory": "{repo}",
                                   "environment": {"PYTHONPATH": "{repo}"},
                                   "why": "the checkout was not on the search path"}), {}
            return json.dumps({"source": "def stage_argv_train(i):\n"
                                         "    return [i['python'], '-c', "
                                         "'import sys; sys.exit(1)']\n",
                               "reasoning": "r"}), {}

    import sys as _sys
    source, parameters, log, row = execution_derive.make_runnable(
        Client(), "train", {"entrypoint": "main.py", "invocation": "python main.py",
                            "parameters": []},
        repo=tmp_path, repository_files=[],
        inputs_for_verify={"python": _sys.executable}, rounds=2, attempts=1)
    assert source is None
    assert row.get("environment") == {"PYTHONPATH": "{repo}"}, row
    assert row.get("working_directory") == "{repo}"


# -- facts only this machine knows, which the failure cannot list for itself ----------------

def test_a_python_interpreter_given_a_shell_script_is_refused_before_it_runs(tmp_path):
    """The command cannot work, and the failure it produces says something else.

    `python train.sh` dies with `SyntaxError: invalid syntax` on the script's third line, which
    reads as a broken script or a wrong environment -- so the loop revised the working
    directory and the environment, three times, for a command whose first element was wrong.
    RoboTwin's evaluate and train both lost their whole budget to it. The instruction not to do
    it was in the prompt already and was not followed; refusing to run it is the part that
    holds."""
    script = tmp_path / "train.sh"
    script.write_text("#!/bin/bash\nset -euo pipefail\n", encoding="utf-8")

    problems = execution_derive.argv_problems(["python", str(script), "a", "b"])
    assert len(problems) == 1
    assert "a shell script" in problems[0] and "`bash`" in problems[0]

    # Named by extension, even when the file is not readable here.
    assert execution_derive.argv_problems(["python3.10", "/gone/eval.sh"])

    # A shebang with no extension is the same fault.
    plain = tmp_path / "run"
    plain.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    assert execution_derive.argv_problems(["/usr/bin/python", str(plain)])

    # And the commands that are right are left alone.
    assert execution_derive.argv_problems(["bash", str(script), "a"]) == []
    assert execution_derive.argv_problems(["python", str(tmp_path / "train.py")]) == []
    assert execution_derive.argv_problems(["/usr/bin/ffmpeg", "-i", "x"]) == []
    assert execution_derive.argv_problems([]) == []


def test_conflicting_repeated_native_options_are_rejected_before_execution():
    argv = ["python", "train.py", "--total_timesteps=1024", "--learning_rate", "0.0003",
            "--total-timesteps=10000000", "--learning-rate=0.0001"]
    problems = execution_derive.argv_problems(argv)
    assert problems and "total-timesteps" in problems[0]
    assert "ambiguous" not in problems[0] or "effective" in problems[0]
    assert execution_derive.conflicting_option_problem(
        ["python", "train.py", "--seed=1", "--seed", "1"]) == ""


def test_dynamic_setting_conflict_is_explained_as_a_builder_override(tmp_path):
    """A sparse research setting must replace a baked-in command default, not duplicate it.

    The first draft keeps the invocation's `--num_envs=16` and appends the candidate's
    `--num-envs=512`. The static guard must tell the next draft exactly that this is a
    generated-command collision, so it removes the fixed copy instead of guessing at the
    benchmark's CLI syntax or accepting ambiguous last-wins behavior.
    """
    import sys as _sys

    from autosim.research import execution_derive

    class Client:
        def __init__(self):
            self.calls = 0
            self.prompts = []

        def chat_with_metadata(self, system, user, **kwargs):
            self.calls += 1
            self.prompts.append(user)
            if self.calls == 1:
                source = (
                    "def stage_argv_train(i):\n"
                    "    argv = [i['python'], 'train.py', '--num_envs=16', "
                    "'--num-envs=512']\n"
                    "    return argv\n")
            else:
                source = (
                    "def stage_argv_train(i):\n"
                    "    argv = [i['python'], 'train.py', '--num-envs=512']\n"
                    "    return argv\n")
            return json.dumps({"source": source, "shape_was_accepted": True}), {}

    client = Client()
    seen = []
    source, _, log = execution_derive.generate_argv(
        client, "train", entrypoint="train.py", invocation="python train.py --num_envs=16",
        repository_files=[], verify=lambda argv: seen.append(argv) or {"ok": True},
        inputs_for_verify={"python": _sys.executable},
        attempts=2)

    assert source is not None and client.calls == 2
    assert len(seen) == 1
    assert seen[0].count("--num-envs=512") == 1
    assert "--num_envs=16" not in seen[0]
    assert "command-builder conflict" in client.prompts[1]
    assert "Remove the fixed copy from the base argv" in client.prompts[1]
    assert "conflicting values" in log[0]["error"]


def test_sparse_setting_replaces_generated_default_and_duplicate_override():
    from autosim.research import execution_derive

    argv = ["python", "train.py", "--num_envs=8", "--num-envs", "512",
            "--num-steps", "50", "--seed=1"]
    actual = execution_derive.coalesce_dynamic_overrides(argv, {"num_envs": 1024})
    assert actual.count("--num-envs") == 1
    assert actual[-2:] == ["--num-envs", "1024"]
    assert "--num_envs=8" not in actual and "512" not in actual
    assert "--num-steps" in actual and "50" in actual
    assert "--seed=1" in actual


def test_generated_command_applies_sparse_settings_over_its_hardcoded_default():
    import sys as _sys

    from autosim.research import execution_derive

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            source = (
                "def stage_argv_train(i):\n"
                "    argv = [i['python'], 'train.py', '--num_envs=8']\n"
                "    for key in sorted(i['settings']):\n"
                "        argv = argv + ['--' + key + '=' + str(i['settings'][key])]\n"
                "    return argv\n")
            return json.dumps({"source": source, "shape_was_accepted": True}), {}

    seen = []
    source, _, log = execution_derive.generate_argv(
        Client(), "train", entrypoint="train.py", invocation="python train.py --num_envs=8",
        repository_files=[], verify=lambda argv: seen.append(argv) or {"ok": True},
        inputs_for_verify={"python": _sys.executable, "settings": {"num_envs": 512}},
        attempts=1)

    assert source is not None and not log[-1].get("error")
    assert seen == [[_sys.executable, "train.py", "--num_envs=512"]]


def test_anchored_accepted_shape_still_uses_corrected_settled_values():
    import sys as _sys

    from autosim.research import execution_derive

    class Client:
        def __init__(self):
            self.calls = 0

        def chat_with_metadata(self, system, user, **kwargs):
            self.calls += 1
            source = (
                "def stage_argv_train(i):\n"
                "    return [i['python'], 'train.py', '--num_envs=256', "
                "'--total_timesteps=' + str(i['steps'])]\n")
            payload = {"source": source, "shape_was_accepted": True}
            if self.calls == 2:
                payload["parameters"] = {"--num_envs": "8"}
            return json.dumps(payload), {}

    client = Client()
    seen = []

    def verify(argv):
        seen.append(argv)
        if len(seen) == 1:
            return {"ok": False, "error": "trainer exited zero; num_iterations=0"}
        return {"ok": True}

    source, parameters, _ = execution_derive.generate_argv(
        client, "train", entrypoint="train.py", invocation="python train.py --num_envs=256",
        repository_files=[], declared_parameters={"--num_envs": "256"},
        verify=verify, inputs_for_verify={"python": _sys.executable, "steps": 1024},
        attempts=2, require_step_control=True)

    assert source is not None and client.calls == 2
    assert parameters["--num_envs"] == "8"
    assert len(seen) == 2
    assert all(argv[:2] == [_sys.executable, "train.py"] for argv in seen)
    assert "--num_envs=256" in seen[0] and "--total_timesteps=1024" in seen[0]
    assert "--num_envs=8" in seen[1] and "--num_envs=256" not in seen[1]
    assert "--total_timesteps=1024" in seen[1]


def test_sparse_settings_keep_config_equals_syntax_and_leave_unmatched_keys_alone():
    from autosim.research import execution_derive

    argv = ["python", "train.py", "learning_rate=0.0003", "task=PushCube-v1"]
    actual = execution_derive.coalesce_dynamic_overrides(
        argv, {"learning_rate": 0.0001, "new_setting": "x"})
    assert actual == ["python", "train.py", "task=PushCube-v1", "learning_rate=0.0001"]
    checkpoint = ["python", "eval.py", "--checkpoint=/frozen/policy.pt"]
    assert execution_derive.coalesce_dynamic_overrides(
        checkpoint, {"--checkpoint": "/another/policy.pt"}) == checkpoint


# -- looking, instead of being handed a list of conventions --------------------------------

def test_an_inspection_reads_checkout_files_and_asks_the_machine_in_a_sandbox(
        tmp_path, monkeypatch):
    """Three shapes, because there are three questions a failed command makes you ask.

    This is what replaced three hand-written helpers -- the environment names on this machine,
    the keys of a table beside the entry point, where a bundled binary lives. Each of those
    was a substitute for looking, and none generalises: the next repository keeps its facts in
    a file nobody wrote a helper for.
    """
    from autosim.research import execution_derive

    repo = tmp_path / "repo"
    (repo / "policy" / "ACT").mkdir(parents=True)
    (repo / "policy" / "ACT" / "TASK_CONFIGS.json").write_text(
        json.dumps({"demo_clean-beat_block_hammer": {"num_episodes": 50}}), encoding="utf-8")
    (repo / "policy" / "ACT" / "notes.txt").write_text("hello\n" * 5, encoding="utf-8")

    # A path relative to the checkout.
    found = execution_derive.inspect({"look_at": "policy/ACT/TASK_CONFIGS.json"},
                                     repo=repo, directory=repo, environment=dict(os.environ))
    assert "demo_clean-beat_block_hammer" in found["content"]
    assert found["path"] == "policy/ACT/TASK_CONFIGS.json"

    # A path relative to where the stage runs: a reviser that writes a bare filename means
    # the one beside the entry point.
    beside = execution_derive.inspect({"look_at": "notes.txt"}, repo=repo,
                                      directory=repo / "policy" / "ACT",
                                      environment=dict(os.environ))
    assert beside["path"] == "policy/ACT/notes.txt" and beside["lines_in_preview"] == 6

    # A listing.
    listed = execution_derive.inspect({"look_at_dir": "policy/ACT"}, repo=repo, directory=repo,
                                      environment=dict(os.environ))
    assert "TASK_CONFIGS.json" in listed["entries"]

    # `{repo}` is substituted wherever it appears, because that is how every field of the
    # payload spells the checkout and a reviser writes the same thing back. A literal
    # `{repo}` in a path resolves to a directory that does not exist -- it asked, was told the
    # file was not there, asked again with the same string, and spent its budget.
    spelled = execution_derive.inspect({"look_at": "{repo}/policy/ACT/TASK_CONFIGS.json"},
                                       repo=repo, directory=repo,
                                       environment=dict(os.environ))
    assert "demo_clean-beat_block_hammer" in spelled["content"]

    # And the machine, which is the only way to answer some of them.
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "42", "")

    monkeypatch.setattr(execution_derive, "bounded_run", run)
    ran = execution_derive.inspect(
        {"run": "printf 42"}, repo=repo, directory=repo,
        environment={**os.environ, "PATH": "/usr/bin:/bin", "DEEPSEEK_API_KEY": "do-not-pass",
                     "PYTHONPATH": "/home/wbc/private/site-packages"})
    assert ran["returncode"] == 0 and "42" in ran["said"]
    assert ran["cwd"] == "."
    argv, kwargs = calls[0]
    assert kwargs["shell"] is False
    assert "--unshare-net" in argv and "--unshare-pid" in argv
    assert "--ro-bind" in argv and "--bind" not in argv
    assert "DEEPSEEK_API_KEY" not in kwargs["env"]
    assert "PYTHONPATH" not in kwargs["env"]
    assert str(repo) not in argv[argv.index("--") + 1:]
    monkeypatch.setattr(execution_derive, "bounded_run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 1, "",
                                                     "bwrap: network namespace unavailable"))
    blocked = execution_derive.inspect({"run": "printf 42"}, repo=repo, directory=repo,
                                       environment={"PATH": "/usr/bin:/bin"})
    assert "sandbox is unavailable" in blocked["error"] and "was not run" in blocked["error"]


def test_a_file_that_is_not_there_says_so_rather_than_raising(tmp_path):
    """An inspection that dies fails exactly when the reviser is already looking at a
    failure -- and the caller would lose the revision it was about to make."""
    from autosim.research import execution_derive

    missing = execution_derive.inspect({"look_at": "nope.txt"}, repo=tmp_path,
                                       directory=tmp_path, environment=dict(os.environ))
    assert "error" in missing and "No such file" in missing["error"]
    assert execution_derive.inspect({"look_at_dir": "nope"}, repo=tmp_path,
                                    directory=tmp_path,
                                    environment=dict(os.environ))["error"]
    # And a request that is none of the three shapes says which three there are.
    assert "look_at" in execution_derive.inspect({}, repo=tmp_path, directory=tmp_path,
                                                 environment=dict(os.environ))["error"]


def test_inspections_refuse_paths_outside_checkout_and_private_assets(tmp_path):
    from autosim.research import execution_derive

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "weights").mkdir()
    (repo / "weights" / "policy.pt").write_bytes(b"weight bytes")
    (repo / "private-link").symlink_to("/etc/passwd")
    for target in ("/etc/passwd", "private-link", "weights/policy.pt"):
        result = execution_derive.inspect({"look_at": target}, repo=repo, directory=repo,
                                          environment={})
        assert "error" in result


def test_inspection_text_preview_redacts_and_refuses_binary_source_lookalikes(tmp_path):
    from autosim.research import execution_derive

    repo = tmp_path / "checkout"
    repo.mkdir()
    source = repo / "scripts" / "inspect.sh"
    source.parent.mkdir()
    source.write_text(
        '#!/bin/bash\nAPI_TOKEN = "PRIVATE_INSPECTION_TOKEN"\n'
        'DATA_ROOT = "/home/private-user/demos"\n', encoding="utf-8")
    visible = execution_derive.inspect({"look_at": "scripts/inspect.sh"}, repo=repo,
                                       directory=repo, environment={})
    assert "#!/bin/bash" in visible["content"]
    assert "PRIVATE_INSPECTION_TOKEN" not in visible["content"]
    assert "/home/private-user" not in visible["content"]

    disguised = repo / "scripts" / "binary.py"
    disguised.write_bytes(b"\x00PRIVATE_BINARY_DEMO\xff")
    withheld = execution_derive.inspect({"look_at": "scripts/binary.py"}, repo=repo,
                                        directory=repo, environment={})
    assert "error" in withheld
    assert "binary" in withheld["error"]
    assert "PRIVATE_BINARY_DEMO" not in str(withheld)


def test_diagnostic_prompt_redacts_credentials_host_paths_and_private_artifacts(tmp_path):
    from autosim.research import execution_derive

    repo = tmp_path / "checkout"
    repo.mkdir()
    row = {
        "entrypoint": "train.py",
        "invocation": "python {repo}/train.py --checkpoint={repo}/weights/best.pt",
        "environment": {"API_TOKEN": "not-for-the-model", "PYTHONPATH": "{repo}"},
    }
    payload = execution_derive._told_about(
        row, "train", ["python", str(Path.home() / "datasets" / "secret.hdf5")],
        f"failed opening {Path.home()}/datasets/secret.hdf5 API_KEY=secret-value",
        repo=repo, attempted=[{"what": "{repo}/weights/best.pt"}], recalled={},
        searched="", may_look=True)
    assert str(Path.home()) not in payload
    assert "not-for-the-model" not in payload and "secret-value" not in payload
    assert "best.pt" not in payload and "secret.hdf5" not in payload
    assert "API_TOKEN" not in payload
    assert "API_KEY=[REDACTED]" in payload
    assert '"value": "{repo}"' in payload


def test_an_inspection_will_not_change_anything(tmp_path, monkeypatch):
    """Shell writes and paths outside the checkout are refused before sandbox launch."""
    from autosim.research import execution_derive

    ran = []
    monkeypatch.setattr(execution_derive, "bounded_run",
                        lambda *a, **k: ran.append(a) or type("R", (), {
                            "stdout": "", "stderr": "", "returncode": 0})())
    refused = execution_derive.inspect({"run": "rm -rf /tmp/x"}, repo=tmp_path,
                                       directory=tmp_path, environment=dict(os.environ))
    assert not ran, "the command was run anyway"
    assert "changes things" in refused["error"]
    # A command that only looks is not refused.
    execution_derive.inspect({"run": "ls -la"}, repo=tmp_path, directory=tmp_path,
                             environment=dict(os.environ))
    assert ran
    refused_shell = execution_derive.inspect({"run": "printf 1; cat .env"}, repo=tmp_path,
                                             directory=tmp_path, environment={})
    assert "shell syntax" in refused_shell["error"]
    refused_private = execution_derive.inspect({"run": "cat /etc/passwd"}, repo=tmp_path,
                                               directory=tmp_path, environment={})
    assert "inside the checkout" in refused_private["error"]
    refused_python = execution_derive.inspect({"run": "python -c 'print(1)'"}, repo=tmp_path,
                                              directory=tmp_path, environment={})
    assert "limited to version" in refused_python["error"]


def test_real_inspection_namespace_is_readonly_and_masks_host_home(tmp_path):
    """Exercise bubblewrap itself where the host allows network namespaces.

    Some managed shells prohibit creating a network namespace even though the host supports
    bubblewrap. In that context the product must fail closed; the same test is run in an
    approved local context to cover the successful OS-sandbox path.
    """
    import shutil

    from autosim.research.common import bounded_run, inspection_argv, inspection_mountpoint

    if not shutil.which("bwrap") or not shutil.which("findmnt"):
        pytest.skip("bubblewrap/findmnt unavailable")
    repo = tmp_path / "checkout"
    repo.mkdir()
    mountpoint = inspection_mountpoint(repo)

    def run(argv):
        return bounded_run(inspection_argv(argv, repo=repo, directory=repo,
                                           environment={"PATH": "/usr/bin:/bin"}),
                           cwd=repo, env={"PATH": "/usr/bin:/bin"}, timeout=20)

    mounted = run([shutil.which("findmnt"), "-n", "-o", "OPTIONS", str(mountpoint)])
    if mounted.returncode != 0 and "bwrap:" in ((mounted.stderr or "") + (mounted.stdout or "")):
        pytest.skip("host does not permit the required bubblewrap network namespace")
    assert mounted.returncode == 0
    assert "ro" in (mounted.stdout or "").strip().split(",")

    write_target = repo / "must-not-be-created"
    attempted_write = run(["/usr/bin/touch", str(write_target)])
    assert attempted_write.returncode != 0 and not write_target.exists(), \
        "inspection was able to write into the checkout"

    home_visible = run(["/usr/bin/test", "-e", str(Path.home())])
    assert home_visible.returncode == 1, "host home was visible inside inspection namespace"
    routes = run(["/usr/bin/cat", "/proc/net/route"])
    assert routes.returncode == 0
    default_routes = [line for line in (routes.stdout or "").splitlines()[1:]
                      if len(line.split()) > 1 and line.split()[1] == "00000000"]
    assert not default_routes, "network namespace exposes a default route"


def test_the_diagnostician_looks_before_it_answers(tmp_path):
    """The whole point: a fixed payload of pre-computed facts becomes a loop the model drives.

    The first reply is a request to look; the second is the finding. What the request returned
    is appended to what the diagnostician can see, which is what makes the second reply
    answerable -- a name it looked up rather than one it guessed.
    """
    from autosim.research import execution_derive

    (tmp_path / "TASK_CONFIGS.json").write_text(
        json.dumps({"autosim_705e29a58f-beat_block_hammer-aloha_agilex-joint": {}}),
        encoding="utf-8")
    seen: list[str] = []

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            seen.append(user)
            if len(seen) == 1:
                return json.dumps({"look_at": "TASK_CONFIGS.json"}), {}
            return json.dumps({"finding": "the key is not one the table has",
                               "evidence": "TASK_CONFIGS.json lists its keys"}), {}

    found = execution_derive.diagnose(
        Client(), "train", {"entrypoint": "train.sh", "invocation": "bash train.sh",
                            "working_directory": "{repo}"},
        repo=tmp_path, argv=["bash", "train.sh"], failure="KeyError: 'beat_block_hammer'")

    assert found["finding"] == "the key is not one the table has"
    # The second prompt carries what the first request returned.
    assert "WHAT YOU ASKED TO SEE" in seen[1]
    assert "autosim_705e29a58f-beat_block_hammer-aloha_agilex-joint" in seen[1]


def test_what_is_wrong_and_what_to_do_are_two_questions(tmp_path):
    """They were one prompt, and it grew to nine thousand characters carrying eleven concerns
    and stopped answering either well: its replies degraded from a diagnosis to "Let me look at
    what actually failed". AutoSOTA's whole agent architecture exists for that observed
    failure, and one role per prompt is the fix.

    The split is visible in what each will accept: a diagnosis that proposes a fix is refused,
    and a revision that asks to look is not a revision.
    """
    from autosim.research import execution_derive

    class Diagnoser:
        def chat_with_metadata(self, system, user, **kwargs):
            if "look_at" in kwargs or "what_you_may_look_at" not in user:
                raise AssertionError("the diagnostician was not told it may look")
            return json.dumps({"working_directory": "{repo}"}), {}   # a fix, not a finding

    assert execution_derive.diagnose(
        Diagnoser(), "train", {"entrypoint": "t.sh", "invocation": "bash t.sh",
                               "working_directory": "{repo}"},
        repo=tmp_path, argv=["bash", "t.sh"], failure="boom") == {}

    # And the reviser is given the finding, is never offered an inspection, and answers.
    seen: list[str] = []

    class Reviser:
        def chat_with_metadata(self, system, user, **kwargs):
            seen.append(user)
            return json.dumps({"working_directory": "{repo}",
                               "environment": {"PYTHONPATH": "{repo}"},
                               "why": "the finding said the import never resolved"}), {}

    change = execution_derive.revise_invocation(
        Reviser(), "train", {"entrypoint": "t.sh", "invocation": "bash t.sh",
                             "working_directory": "{repo}"},
        repo=tmp_path, argv=["bash", "t.sh"], failure="boom",
        finding="the package root is not on the search path")
    assert change["environment"] == {"PYTHONPATH": "{repo}"}
    assert "WHAT IS WRONG" in seen[0] and "the package root is not on the search path" in seen[0]
    assert "what_you_may_look_at" not in seen[0], "the reviser was offered an inspection"


def test_the_diagnostician_runs_every_inspection_it_asked_for(tmp_path):
    """Both of them, in the order it asked, and both in what it is shown next."""
    from autosim.research import execution_derive

    (tmp_path / "policy").mkdir()
    (tmp_path / "policy" / "train.sh").write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
    seen: list[str] = []

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            seen.append(user)
            if len(seen) == 1:
                return ('{"look_at_dir": "policy"}\n\n{"look_at": "policy/train.sh"}'), {}
            return json.dumps({"finding": "the script is a shell script"}), {}

    found = execution_derive.diagnose(
        Client(), "train", {"entrypoint": "policy/train.sh", "invocation": "bash train.sh",
                            "working_directory": "{repo}"},
        repo=tmp_path, argv=["bash", "train.sh"], failure="boom")
    assert found["finding"] == "the script is a shell script"
    assert "train.sh" in seen[1]                      # the directory listing
    assert "#!/bin/bash" in seen[1]                   # and the file it holds


def test_the_reviser_cannot_look_forever(tmp_path):
    """Each inspection is a round trip and a stage's budget is finite. A reviser that only
    looks never converges, and the loop has to end with a reason rather than a spin."""
    from autosim.research import execution_derive

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"look_at_dir": "."}), {}

    change = execution_derive.revise_invocation(
        Client(), "train", {"entrypoint": "train.sh", "invocation": "bash train.sh",
                            "working_directory": "{repo}"},
        repo=tmp_path, argv=["bash", "train.sh"], failure="boom", attempts=3)
    assert change is None


def test_the_payload_says_what_the_checkout_actually_is(tmp_path):
    """`placeholders` described `{repo}` -- "the checkout's absolute path" -- without saying
    which path. A reviser writing an absolute one had to combine the parts itself, and one
    combined them wrong: `.../AutoSimSOTA/RoboTwin/RoboTwin/scripts/...`, one directory too
    deep, seven inspections into "no such file" for a file at the checkout root.

    A value is checkable by the reader. A description of a value is a puzzle.
    """
    from autosim.research import execution_derive

    seen: list[str] = []

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            seen.append(user)
            return json.dumps({"working_directory": "{repo}", "why": "w"}), {}

    execution_derive.revise_invocation(
        Client(), "train", {"entrypoint": "train.sh", "invocation": "bash train.sh",
                            "working_directory": "{repo}"},
        repo=tmp_path, argv=["bash", "train.sh"], failure="boom")
    assert str(tmp_path) not in seen[0]
    assert '"value": "{repo}"' in seen[0]
    assert "the checkout, and the root" in seen[0]


def test_a_reply_holding_two_objects_is_two_objects():
    """From the first `{` to the last `}` is right only when the reply holds exactly one.

    A reviser asked to look at a directory and then at the file inside it wrote both, in one
    reply, separated by a blank line. Read as a single object that is `{a}\\n\\n{b}`, which is
    not JSON, so the first half ran, the second half was dropped, and the failure that came
    back said the model had answered with nothing. Every caller of `_object` in this module
    was one two-object reply away from the same silent nothing.
    """
    from autosim.research.execution_derive import _object, _objects

    two = '{"look_at_dir": "/a"}\n\n{"look_at": "/a/b.sh"}'
    assert _objects(two) == [{"look_at_dir": "/a"}, {"look_at": "/a/b.sh"}]
    assert _object(two) == {"look_at_dir": "/a"}

    # Prose between them is the common shape, not the exception.
    assert _objects('I will look.\n{"run": "ls"}\nAnd then {"look_at": "/x"}.') == [
        {"run": "ls"}, {"look_at": "/x"}]
    # A brace inside a string is not a brace.
    assert _objects('{"a": "}}{{"}') == [{"a": "}}{{"}]
    # An object that does not parse is skipped rather than swallowing the one after it.
    assert _objects('{not json}\n{"ok": 1}') == [{"ok": 1}]
    assert _objects("no json here") == []
    with pytest.raises(ValueError, match="no JSON object"):
        _object("no json here")


def test_a_finding_about_the_repository_is_answered_with_staging(tmp_path):
    """A diagnosis that names a property of the repository is not a reason to stop.

    It is what AutoSOTA calls protocol-preserving repository repair, and the loop had no way
    to act on one: a diagnosis said "the real key names are in TASK_CONFIGS.json" -- naming
    the file it had just read -- and was filed as `no_change_can_fix_this`, so the stage ended
    with the answer in its own record and nothing acting on it.
    """
    from autosim.research import execution_derive

    seen: list[str] = []

    class Reviser:
        def chat_with_metadata(self, system, user, **kwargs):
            seen.append(user)
            return json.dumps({"working_directory": "{repo}",
                               "staging": ["ln -sfn {repo}/data/a {repo}/data/b"],
                               "why": "the repository assumes a path it does not have"}), {}

    change = execution_derive.revise_invocation(
        Reviser(), "train", {"entrypoint": "t.sh", "invocation": "bash t.sh",
                             "working_directory": "{repo}"},
        repo=tmp_path, argv=["bash", "t.sh"], failure="boom",
        finding="the script computes a path whose depth cannot be satisfied from where it runs",
        about="the repository")
    assert change["staging"] == ["ln -sfn {repo}/data/a {repo}/data/b"]
    assert "ABOUT THE REPOSITORY" in seen[0]
    assert "must be safe to run twice" in seen[0]

    # And a command-level finding is not told to stage anything.
    seen.clear()
    execution_derive.revise_invocation(
        Reviser(), "train", {"entrypoint": "t.sh", "invocation": "bash t.sh",
                             "working_directory": "{repo}"},
        repo=tmp_path, argv=["bash", "t.sh"], failure="boom",
        finding="the package root is not on the search path", about="the command")
    assert "ABOUT THE REPOSITORY" not in seen[0]


def test_staging_is_resolved_and_runs_before_the_stage(tmp_path):
    """`{repo}` is substituted, and the commands run in the stage's own directory -- a symlink
    made anywhere else makes the wrong thing true."""
    from autosim.research.declarative_backend import invocation_staging

    assert invocation_staging({"staging": ["ln -sfn {repo}/a {repo}/b", "  "]}) == [
        "ln -sfn {repo}/a {repo}/b"]
    assert invocation_staging({}) == []
    assert invocation_staging({"staging": "ln -sfn {repo}/a {repo}/b"}) == [
        "ln -sfn {repo}/a {repo}/b"]


def test_an_unset_variable_expands_to_nothing_and_does_not_crash_the_loop(tmp_path):
    """`{repo}/XPolicyLab:${PYTHONPATH}` is an ordinary thing to write, and `PYTHONPATH` is
    absent on a machine that never needed it. Refusing the reference -- which this did, to
    avoid "a value that is nearly right" -- raised `ValueError` while the loop was building
    the invocation, ending the whole stage and discarding every round that had succeeded.

    A guard that takes down the thing it guards is worse than the fault it prevents. The
    record is what is kept instead: the value is written down, and the caller is told which
    names were unset."""
    from autosim.research.declarative_backend import expand, invocation_environment

    unset: list[str] = []
    assert expand("/a/XPolicyLab:${PYTHONPATH}", base={"PATH": "/bin"}, unset=unset) == \
        "/a/XPolicyLab:"
    assert unset == ["PYTHONPATH"]

    # Set, and it is used.
    unset.clear()
    assert expand("/a:${PYTHONPATH}", base={"PYTHONPATH": "/b"}, unset=unset) == "/a:/b"
    assert unset == []

    # And through the field, which is where it crashed.
    stage = {"environment": {"PYTHONPATH": "{repo}/XPolicyLab:${PYTHONPATH}"}}
    collected: list[str] = []
    resolved = invocation_environment(stage, repo=tmp_path, base={"PATH": "/bin"},
                                      unset=collected)
    assert resolved["PYTHONPATH"] == f"{tmp_path}/XPolicyLab:"
    assert collected == ["PYTHONPATH"]


def test_nothing_a_round_does_can_end_the_loop(tmp_path):
    """The loop says so in a comment and the comment was only true of half the round.

    `generate_argv` was inside a `try` and everything after it -- the diagnosis, the revision,
    the memory lookup, the monitor -- was not. Four separate crashes have come out of this one
    function and each discarded the rounds that had succeeded along with the rounds that had
    not started; a fifth arrived today, when an environment value naming an unset variable
    raised `ValueError` while the invocation was being built.
    """
    from autosim.research import execution_derive

    class Broken:
        """Answers nothing, at every step, in a different way each time."""

        def __init__(self):
            self.calls = 0

        def chat_with_metadata(self, system, user, **kwargs):
            self.calls += 1
            if self.calls % 3 == 0:
                raise TimeoutError("the provider did not answer")
            if self.calls % 3 == 1:
                raise KeyboardInterrupt            # not an Exception; must still not escape
            return "a paragraph, not JSON", {}

    # `KeyboardInterrupt` is a `BaseException`, and a loop that swallows one is a loop that
    # cannot be stopped. Everything else must be survivable.
    class Timeouts:
        def chat_with_metadata(self, system, user, **kwargs):
            raise TimeoutError("the provider did not answer")

    source, parameters, log, row = execution_derive.make_runnable(
        Timeouts(), "train", {"entrypoint": "t.sh", "invocation": "bash t.sh",
                              "parameters": []},
        repo=tmp_path, repository_files=[], inputs_for_verify={}, rounds=2, attempts=1,
        base_environment=dict(os.environ))
    assert source is None
    assert log and all(row.get("status") for row in log)


def test_a_revision_starts_from_the_staging_that_is_already_there(tmp_path):
    """A field a stage can carry is a field its correction has to start from.

    RoboTwin's training stage reached a finding that read "the staging script itself must be
    corrected to discover by content and to name the destination files the way the loader
    opens them" -- and the reviser was never shown the staging script, so the one thing the
    finding named was the one thing it could not see. The correction could not be made, and
    the loop spent its rounds re-deriving from nothing.
    """
    from autosim.research import execution_derive

    seen: list[tuple[str, str]] = []

    class Reviser:
        def chat_with_metadata(self, system, user, **kwargs):
            seen.append((system, user))
            return json.dumps({"working_directory": "{repo}",
                               "staging": ["ln -sfn {repo}/a/episode0.hdf5 {repo}/b/"],
                               "why": "the guard compared a filename form the data does not use"}), {}

    change = execution_derive.revise_invocation(
        Reviser(), "train",
        {"entrypoint": "t.sh", "invocation": "bash t.sh", "working_directory": "{repo}",
         "staging": ["ln -sfn {repo}/a/episode_0.hdf5 {repo}/b/"]},
        repo=tmp_path, argv=["bash", "t.sh"], failure="boom",
        finding="the staging guard compares a filename form the data does not use",
        about="the repository")
    assert change["staging"] == ["ln -sfn {repo}/a/episode0.hdf5 {repo}/b/"]
    system, user = seen[0]
    assert "episode_0.hdf5" in user, "the reviser was not shown the staging it is correcting"
    # And it is told what to do with it: a field given replaces the value it had, so a
    # correction is the whole list rather than an addition to one it cannot see.
    assert "The staging this stage already has is in `as_derived`" in system


def test_a_staging_value_that_is_not_commands_is_refused_where_it_is_written(tmp_path):
    """A malformed staging list is a rejected draft, not a stage that silently stages
    nothing -- which is what an empty or prose value would be."""
    from autosim.research import execution_derive

    class Reviser:
        def __init__(self, staging):
            self.staging = staging

        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({"working_directory": "{repo}", "staging": self.staging,
                               "why": "w"}), {}

    for bad in ({"a": "b"}, ["  "], [1, 2]):
        assert execution_derive.revise_invocation(
            Reviser(bad), "train", {"entrypoint": "t.sh", "invocation": "bash t.sh",
                                    "working_directory": "{repo}"},
            repo=tmp_path, argv=["bash", "t.sh"], failure="boom", attempts=1) is None, bad

    # A single command as a bare string is accepted, because that is how a model spells one.
    assert execution_derive.revise_invocation(
        Reviser("ln -sfn {repo}/a {repo}/b"), "train",
        {"entrypoint": "t.sh", "invocation": "bash t.sh", "working_directory": "{repo}"},
        repo=tmp_path, argv=["bash", "t.sh"], failure="boom")["staging"] == \
        ["ln -sfn {repo}/a {repo}/b"]
