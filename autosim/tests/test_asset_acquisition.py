"""What a benchmark needs and does not ship, as a step rather than as an assumption.

Most embodied-simulation benchmarks need something that is not in their repository: scene
assets, demonstrations, a simulator's binary distribution, a released checkpoint. The build
asked only how to install the repository's dependencies, so the answer to "where does the data
come from" was never sought -- and on a machine without the data already, the build succeeded
and the first training step failed.
"""

import subprocess
from pathlib import Path

from autosim.research.provision import (asset_commands, asset_probe, asset_section,
                                        plan_problems, stage_command_violation)


def run(command: str) -> int:
    return subprocess.run(["bash", "-lc", command], capture_output=True).returncode


def test_a_plan_names_what_it_needs_and_where_it_comes_from():
    good = {"python": "3.10", "commands": ["x"], "probes": ["y"],
            "assets": [{"what": "demonstrations", "where": "/data/demos",
                        "produced_by": "bash scripts/download.sh",
                        "evidence": "README.md, Installation"}]}
    assert plan_problems(good) == []
    # Absent is a real answer: a benchmark that needs nothing outside its checkout gets no
    # asset list at all.
    assert plan_problems({"python": "3.10", "commands": ["x"], "probes": ["y"]}) == []
    # Present but incomplete is not.
    faults = plan_problems({**good, "assets": [{"what": "demonstrations"}]})
    assert len(faults) == 2 and all("assets[0]" in fault for fault in faults)


def test_unknown_asset_source_is_not_executed_as_a_shell_command():
    plan = {"python": "3.10", "commands": ["true"], "probes": ["true"],
            "assets": [{"what": "checkpoint", "where": "/tmp/checkpoints",
                        "produced_by": "not determinable from the checkout"}]}
    assert any("produced_by" in fault for fault in plan_problems(plan))


def test_asset_where_rejects_prose_and_parenthetical_explanations():
    plan = {"python": "3.10", "commands": ["true"], "probes": ["true"],
            "assets": [{"what": "weights", "where": "the site-packages of Python",
                        "produced_by": "already present"}]}
    assert any("where must be a path" in fault for fault in plan_problems(plan))
    plan["assets"][0]["where"] = "~/.cache/demos (the default download directory)"
    assert any("where must be a path" in fault for fault in plan_problems(plan))
    plan["assets"][0]["where"] = "runs/<timestamp>/model.pt"
    assert any("unresolved placeholder" in fault for fault in plan_problems(plan))
    plan["assets"][0]["where"] = "runs/model.pt"
    plan["assets"][0]["produced_by"] = "produced by train stage: python train.py"
    assert any("produced_by" in fault for fault in plan_problems(plan))


def test_environment_plan_cannot_execute_selected_research_stages():
    context = {"stage_paths": {
        "prepare_data": {"entrypoint": "pkg/trajectory/replay_trajectory.py"},
        "train": {"entrypoint": "examples/baselines/bc/bc.py"}}}
    assert "selected train" in stage_command_violation("{python} bc.py --total-iters 10000",
                                                       context)
    assert "selected prepare_data" in stage_command_violation(
        "{python} -m pkg.trajectory.replay_trajectory --save-traj", context)
    assert stage_command_violation("{pip} install -e {repo}", context) == ""
    plan = {"python": "3.10", "commands": ["{python} bc.py --total-iters 10000"],
            "probes": ["{python} bc.py --help"], "assets": [{
                "what": "converted data", "where": "/data/converted.h5",
                "produced_by": "{python} -m pkg.trajectory.replay_trajectory --save-traj"}]}
    faults = plan_problems(plan, context)
    assert any("commands[0]" in fault and "execution graph" in fault for fault in faults)
    assert any("assets[0].produced_by" in fault for fault in faults)


def test_environment_plan_cannot_require_future_stage_output_as_an_asset_or_probe():
    context = {"stage_paths": {"train": {"entrypoint": "examples/ppo.py",
                                          "artifact": "runs/*/final_ckpt.pt"}},
               "selected_asset_keys": []}
    plan = {"python": "3.10", "commands": ["{pip} install -e {repo}"],
            "probes": ["{python} -c 'import torch'",
                       "test -e {repo}/runs/x/final_ckpt.pt"],
            "assets": [{"what": "checkpoint from train",
                        "where": "{repo}/runs/x/final_ckpt.pt",
                        "produced_by": "already present"}]}
    faults = plan_problems(plan, context)
    assert any("no external prerequisite assets" in fault for fault in faults)
    assert any("probes[1]" in fault and "before that stage runs" in fault
               for fault in faults)
    assert any("assets[0].where" in fault and "before that stage runs" in fault
               for fault in faults)


def test_selected_stage_help_is_allowed_as_an_environment_probe():
    context = {"stage_paths": {"train": {"entrypoint": "examples/ppo.py",
                                          "artifact": "runs/*/final_ckpt.pt"}},
               "selected_asset_keys": []}
    plan = {"python": "3.10", "commands": ["{pip} install -e {repo}"],
            "probes": ["cd {repo}/examples && {python} ppo.py --help"], "assets": []}
    assert plan_problems(plan, context) == []
    plan["probes"] = ["{python} ppo.py --help; {python} ppo.py --total_timesteps 2000000"]
    assert any("probes[0]" in fault for fault in plan_problems(plan, context))


def test_asset_acquisition_must_target_the_isolated_run_not_host_home(tmp_path):
    context = {"selected_asset_keys": ["dataset"],
               "writable_output": str(tmp_path / "run")}
    value = {"python": "3.10", "commands": ["{pip} install -e {repo}"],
             "probes": ["{python} -c 'import sim'"],
             "assets": [{"what": "demos", "where": "/home/example/.sim/demos/data.h5",
                         "produced_by": "{python} -m sim.download -o /home/example/.sim"}]}
    assert any("outside the run's writable output" in fault
               for fault in plan_problems(value, context))
    value["assets"][0]["where"] = str(tmp_path / "run" / "demos" / "data.h5")
    assert not any("outside the run's writable output" in fault
                   for fault in plan_problems(value, context))


def test_python_commands_must_use_the_selected_interpreter():
    base = {"python": "/tmp/run/env/bin/python", "commands": ["python -m pip install x"],
            "probes": ["{python} -c 'pass'"]}
    assert any("PATH-dependent" in fault for fault in plan_problems(base))
    base["commands"] = ["{python} -m pip install x"]
    base["assets"] = [{"what": "data", "where": "/tmp/data",
                       "produced_by": "python -m download_data"}]
    assert any("PATH-dependent" in fault for fault in plan_problems(base))


def test_what_is_already_here_is_not_downloaded_again(tmp_path):
    """`already present` is the model saying the survey found it. Eight gigabytes is eight
    gigabytes, and on a machine without a network it is a failure rather than a delay."""
    assets = [{"what": "demos", "where": "/data/d", "produced_by": "already present"},
              {"what": "checkpoint", "where": "/ckpt", "produced_by": "wget https://x/w"}]
    assert asset_commands(assets) == ["wget https://x/w"]
    assert asset_commands([{"produced_by": "  Already Present  ", "where": "/x"}]) == []


def test_a_probe_for_an_asset_fails_on_a_directory_that_is_empty(tmp_path):
    """The shape an interrupted download leaves.

    `test -e` accepts it, so a build would report an environment ready to run against a
    directory with nothing in it -- the same failure this module exists to catch, one level
    over.
    """
    wanted = tmp_path / "assets"
    wanted.mkdir()                                     # exists, and holds nothing
    assert run(asset_probe(str(wanted))) != 0

    (wanted / "scene.xml").write_text("<mujoco/>")
    assert run(asset_probe(str(wanted))) == 0

    # A file is tested by having bytes in it, and an empty one is not an asset.
    empty = tmp_path / "weights.bin"
    empty.write_text("")
    assert run(asset_probe(str(empty))) != 0
    empty.write_text("weights")
    assert run(asset_probe(str(empty))) == 0

    # And a path that is not there at all fails, which is the ordinary case.
    assert run(asset_probe(str(tmp_path / "absent"))) != 0


def test_the_plan_is_told_what_the_survey_already_found(tmp_path):
    """Without this the only answer to "is the data already here" is "unknown", and an
    unknown asset is one a model fetches again, or invents a URL for."""
    report = {"datasets": [{"path": "/data/libero/libero_10/a.hdf5", "bytes": 6}],
              "manifest_excerpts": {"README.md": "# Benchmark\n\nRun `bash download.sh`."}}
    section = asset_section(report)
    assert section["data_the_repository_ships_or_this_machine_already_has"] == \
        [{"asset_id": "local_dataset_1", "format": ".hdf5", "present": True}]
    assert section["_local_asset_aliases"] == {
        "local_dataset_1": "/data/libero/libero_10/a.hdf5"}
    assert "download.sh" in section["readme"]
    # A build asked without a survey says nothing rather than guessing.
    assert asset_section(None) == {}


# -- an interpreter that already exists ----------------------------------------------------

def test_a_plan_may_name_an_interpreter_instead_of_a_version(tmp_path):
    """Not every benchmark's environment is one this machine builds.

    A simulator distributed as a binary -- Omniverse, an engine that ships its own Python --
    is used where it was installed. `{python}` and `{pip}` have to resolve to it, and the
    stages after the build have to be handed the same answer.
    """
    from autosim.research.provision import interpreter_is_given, values_for

    assert interpreter_is_given("3.10") is False
    assert interpreter_is_given("") is False
    assert interpreter_is_given(str(tmp_path / "absent" / "python")) is False

    interpreter = tmp_path / "isaac" / "python.sh"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n")
    (interpreter.parent / "pip").write_text("#!/bin/sh\n")
    assert interpreter_is_given(str(interpreter)) is True

    values = values_for(tmp_path / "env", tmp_path, tmp_path / "work", "conda",
                        str(interpreter))
    assert values["python"] == str(interpreter)
    assert values["pip"] == str(interpreter.parent / "pip")

    # A version still means "build one here", which is the path it always was.
    built = values_for(tmp_path / "env", tmp_path, tmp_path / "work", "conda", "3.10")
    assert built["python"] == str(tmp_path / "env/bin/python")


def test_the_interpreter_the_build_used_is_what_the_stages_are_given(tmp_path):
    """`env_python` used to compute `<output>/env/bin/python` and nothing else, so a build
    whose interpreter was named rather than created produced nothing any later stage could
    run -- and the build would have reported success."""
    import json

    from autosim.research.provision import env_python

    interpreter = tmp_path / "isaac/python.sh"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n")
    (tmp_path / "environment.json").write_text(json.dumps(
        {"verdict": {"passed": True}, "python": str(interpreter),
         "interpreter": str(interpreter)}))

    assert env_python(tmp_path) == interpreter

    # And a build that created its own still resolves the way it did.
    built = tmp_path / "again"
    (built / "env/bin").mkdir(parents=True)
    (built / "env/bin/python").write_text("")
    (built / "environment.json").write_text(json.dumps({"verdict": {"passed": True}}))
    assert env_python(built) == built / "env/bin/python"

    # A build that did not pass hands over nothing, whichever way it was built.
    (built / "environment.json").write_text(json.dumps({"verdict": {"passed": False}}))
    assert env_python(built) is None
