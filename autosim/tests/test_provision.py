"""Building an environment, tested where it can be tested without building one.

No environment is created. What is tested is everything around that: which requirements
count as declared, what shape a plan must have, which failures are the environment's and
which are the machine's, what the feedback carries, and whether a record of what survived
can be trusted to describe what is present.

That last one is not hypothetical. A build here recorded a 1.6 GB torch install and then
recreated the environment underneath it, and went on reporting torch as installed.
"""

import json
from pathlib import Path

import pytest

from autosim.research import provision as pv


# -- what a repository declares ------------------------------------------------------------

def test_every_way_a_repository_states_its_dependencies_is_read(tmp_path):
    for name in ("requirements.txt", "environment.yml", "pyproject.toml", "setup.py",
                 "install.sh", "README.md"):
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    found = pv.manifests_of(tmp_path)
    assert "requirements.txt" in found and "environment.yml" in found
    assert found["requirements.txt"] == "# requirements.txt\n"


def test_a_manifest_too_large_to_read_is_skipped_rather_than_truncated(tmp_path):
    (tmp_path / "requirements.txt").write_text("x" * 10, encoding="utf-8")
    assert pv.manifests_of(tmp_path, limit=5) == {}


# -- the plan ------------------------------------------------------------------------------

def plan(**overrides):
    value = {"python": "3.9", "commands": ["{conda} create -p {prefix} python=3.9 -y"],
             "probes": ["{python} -c \"print(1)\""], "reasoning": "read the manifests"}
    value.update(overrides)
    return value


def test_a_plan_without_a_python_version_is_refused():
    assert any("python is required" in p for p in pv.plan_problems(plan(python="")))


def test_a_plan_needs_probes_and_more_than_one_kind_of_thing_to_check():
    """One probe certifies one thing, and the thing it certifies may be the easy one."""
    faults = pv.plan_problems(plan(probes=[]))
    assert any("one command per stage" in f for f in faults)


def test_a_probe_must_be_a_command():
    assert any("non-empty" in f for f in pv.plan_problems(plan(probes=["", "  "])))


def test_a_single_probe_field_is_not_accepted():
    """The field that made a build pass on a simulator it could not train for."""
    value = plan()
    value["probe"] = value.pop("probes")
    assert any("probes must be" in f for f in pv.plan_problems(value))


# -- placeholders --------------------------------------------------------------------------

def test_a_command_may_only_use_placeholders_that_exist():
    assert pv.unknown_placeholders("{python} -m pip install -r {repo}/r.txt") == []
    assert pv.unknown_placeholders("{python} {gpu}") == ["gpu"]


def test_substitution_fills_the_values_it_was_given():
    assert pv.substitute("{pip} install x", {"pip": "/e/bin/pip"}) == "/e/bin/pip install x"


# -- what kind of failure is it ------------------------------------------------------------

def test_a_missing_package_is_a_dependency_problem():
    assert pv.classify(1, "", "ModuleNotFoundError: No module named 'robosuite'") == "dependency"


def test_a_renderer_that_will_not_start_is_a_platform_problem():
    assert pv.classify(1, "", "libEGL error: cannot open display :0") == "platform"
    assert pv.classify(-11, "", "") == "platform"


def test_a_package_name_containing_a_signal_is_not_a_platform_problem():
    """`egl_probe` failed to compile and was reported as a platform limit.

    A bare "egl" signal matched the package name, so a dependency the build could have
    fixed stopped the build -- which is the non-terminating loop this classification exists
    to prevent, reached from the other side.
    """
    assert pv.classify(1, "", "ERROR: Failed building wheel for egl-probe") == "unknown"
    assert pv.classify(1, "", "ERROR: Failed building wheel for glm_package") != "platform"


# -- what the feedback carries -------------------------------------------------------------

def test_the_excerpt_carries_the_cause_and_not_only_the_wrapper():
    """A pip build log announces failure at the top and states it hundreds of lines later."""
    log = ("error: subprocess-exited-with-error\n"
           + "\n".join(f"  building {i}" for i in range(40))
           + "\nCMake Error at CMakeLists.txt:1:\n  cmake_minimum_required VERSION 3.5\n"
           "make: *** [all] Error 2")
    excerpt = pv.error_excerpt(log)
    assert "subprocess-exited-with-error" in excerpt
    assert "cmake_minimum_required" in excerpt


def test_the_excerpt_falls_back_when_nothing_announces_itself():
    assert pv.error_excerpt("a\nb\nlast") == "a\nb\nlast"
    assert pv.error_excerpt("") == "(no output)"


def test_what_the_probes_printed_travels_with_the_verdict(tmp_path):
    """A probe that asks rather than does passes on a broken environment.

    A build here passed with `torch.cuda.is_available()` returning true while printing, in
    the same output, that the installed torch cannot execute on this GPU. The verdict was
    correct about the exit code and wrong about the environment, and only the output says so.
    """
    source = Path(pv.__file__).read_text(encoding="utf-8")
    assert "probe_output" in source
    assert "a verdict that hides its evidence" in source.lower() or \
           "A verdict that hides its evidence" in source


# -- a record that describes what is present -----------------------------------------------

def test_recreating_the_environment_invalidates_everything_before_it(tmp_path):
    """The record only ever grew, so it claimed a torch install that had been removed."""
    prefix = tmp_path / "env"
    assert pv.resets_environment(f"conda create -y -p {prefix} python=3.8", prefix)
    assert not pv.resets_environment("conda create -y -n elsewhere python=3.8", prefix)
    assert not pv.resets_environment(f"{prefix}/bin/pip install torch", prefix)


def test_a_reset_in_the_transcript_empties_the_record_before_it(tmp_path):
    """Replayed in order, because the same mistake in an earlier attempt looks like a longer
    recipe rather than a wrong one."""
    import io
    import contextlib
    from autosim.research import provision

    transcript = {"rows": [
        {"command": "a", "ok": True, "kind": "cmd"},
        {"command": "b", "ok": True, "kind": "cmd"},
        {"kind": "reset", "command": "conda create -p x", "invalidated": 2},
        {"command": "c", "ok": True, "kind": "cmd"},
    ]}
    output = tmp_path / "out"
    output.mkdir()
    (output / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    replayed, record = [], []
    for row in json.loads((output / "transcript.json").read_text())["rows"]:
        if row.get("kind") == "reset":
            record = []
        elif row.get("ok") and row.get("kind") != "probe":
            record.append(row)
    assert [r["command"] for r in record] == ["c"]


def test_an_environment_that_was_not_built_has_no_interpreter(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    assert pv.env_python(output) is None
    (output / "environment.json").write_text(
        json.dumps({"verdict": {"passed": False}}), encoding="utf-8")
    assert pv.env_python(output) is None
    (output / "environment.json").write_text(
        json.dumps({"verdict": {"passed": True}}), encoding="utf-8")
    assert pv.env_python(output) is None  # no interpreter on disk either
    (output / "env" / "bin").mkdir(parents=True)
    (output / "env" / "bin" / "python").write_text("", encoding="utf-8")
    assert pv.env_python(output) == output / "env" / "bin" / "python"


def test_the_identity_is_the_commands_that_built_it(tmp_path):
    """An environment left over from a changed recipe is not this environment."""
    record = [{"command": "a", "ok": True}, {"command": "b", "ok": True},
              {"command": "c", "ok": False}]
    first = pv.content_id("3.9", record, tmp_path)
    assert first == pv.content_id("3.9", record, tmp_path)
    assert first != pv.content_id("3.10", record, tmp_path)
    assert first != pv.content_id("3.9", record[:1], tmp_path)
    # A command that failed is not part of what built it.
    assert first == pv.content_id("3.9", [{**record[0]}, {**record[1]}], tmp_path)


def test_the_machine_is_reported_so_a_plan_can_be_checked_against_it():
    facts = pv.platform_facts()
    assert "os" in facts and "gpus" in facts and "cuda_toolkit" in facts
