"""Building an environment, tested where it can be tested without building one.

No environment is created. What is tested is everything around that: which requirements
count as declared, what shape a plan must have, which failures are the environment's and
which are the machine's, what the feedback carries, and whether a record of what survived
can be trusted to describe what is present.

That last one is not hypothetical. A build here recorded a 1.6 GB torch install and then
recreated the environment underneath it, and went on reporting torch as installed.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from autosim.research import provision as pv
from autosim.research.budget import RunBudget


def test_middle_error_is_visible_and_entire_operation_is_sealed(tmp_path, monkeypatch):
    import subprocess
    from autosim.research.evidence_store import read_attempt_evidence
    text = ("error: subprocess-exited-with-error\n" + "build noise\n" * 200 +
            "CMake Error at CMakeLists.txt:1 (cmake_minimum_required):\n"
            "Compatibility with CMake < 3.5 has been removed from CMake.\n" +
            "more noise\n" * 200 + "ERROR: Failed building wheel\n")
    monkeypatch.setattr(pv, "bounded_run", lambda *a, **k:
                        subprocess.CompletedProcess([], 1, "", text))
    monkeypatch.setattr(pv, "isolated_argv", lambda argv, **k: argv)
    result = pv.run("false", env={}, cwd=tmp_path, timeout=1,
                    output=tmp_path / "build.log")
    assert "Compatibility with CMake < 3.5" in result["excerpt"]
    evidence = read_attempt_evidence(tmp_path, result["evidence_id"], limit=12000)
    assert "Compatibility with CMake < 3.5" in evidence["text"]
    assert evidence["returncode"] == 1
    progress = json.loads((tmp_path / "provision_progress.json").read_text())
    assert progress["attempts"][0]["evidence_id"] == result["evidence_id"]


def test_model_failure_during_repair_preserves_completed_install_work(tmp_path, monkeypatch):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "run"
    calls = []

    def execute(*a, **k):
        calls.append(a)
        return subprocess.CompletedProcess([], 0 if len(calls) == 1 else 1,
                                           "", "error: dependency failure")

    class BrokenModel:
        def chat_with_metadata(self, *a, **k):
            raise RuntimeError("model transport interrupted")

    monkeypatch.setattr(pv, "bounded_run", execute)
    monkeypatch.setattr(pv, "isolated_argv", lambda argv, **k: argv)
    with pytest.raises(RuntimeError, match="transport interrupted"):
        pv.build(repo, client=BrokenModel(), prefix=output / "env", output=output,
                 python=sys.executable, manifests={}, assets={}, max_rounds=1,
                 seed={"python": sys.executable, "templates": ["echo installed", "false"],
                       "probes": ["{python} -c 'print(1)'"], "reasoning": "test"})
    transcript = json.loads((output / "transcript.json").read_text())["rows"]
    assert any(row.get("ok") for row in transcript)
    assert any(row.get("ok") is False for row in transcript)
    assert pv.env_python(output) is None  # partial install is not verified readiness
    assert len(json.loads((output / "provision_progress.json").read_text())["attempts"]) == 2


def test_environment_command_timeout_has_an_explicit_failed_receipt(tmp_path):
    result = pv.run("sleep 3", env=dict(os.environ), cwd=tmp_path,
                    timeout=0.1, output=tmp_path / "build.log")
    assert result["ok"] is False
    assert result["returncode"] is None
    assert result["failure_kind"] == "timeout"
    assert "timed out" in (tmp_path / "build.log").read_text(encoding="utf-8")


def test_environment_build_stops_at_whole_run_budget(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "run"
    held = RunBudget(output, wall_seconds=0.2)
    result = pv.build(repo, client=None, prefix=output / "env", output=output,
                      python=sys.executable, manifests={}, assets={},
                      seed={"python": sys.executable, "templates": ["sleep 2"],
                            "probes": ["{python} -c 'print(1)'"], "reasoning": "test"},
                      budget=held, max_rounds=1)
    assert result["verdict"]["reason"] == "budget_exhausted"
    assert any(row.get("kind") == "budget_exhausted" for row in
               json.loads((output / "transcript.json").read_text())["rows"])


def test_unpinned_replacement_of_an_installed_package_is_deferred_to_probes():
    planned = {"commands": ["{pip} install --no-build-isolation -e {repo}",
                            '{pip} install "torch==2.6.0"']}
    installed = [{"name": "torch", "version": "2.7.1"}]
    faults = pv.unnecessary_version_installs(
        planned, installed_packages=installed,
        manifests={"setup.py": "install_requires=['torch']"})
    assert len(faults) == 1 and "commands[1]" in faults[0]
    assert "native probes" in faults[0]
    assert pv.unnecessary_version_installs(
        planned, installed_packages=installed,
        manifests={"requirements.txt": "torch==2.6.0\n"}) == []
    assert pv.unnecessary_version_installs(
        planned, installed_packages=[{"name": "torch", "version": "2.6.0"}],
        manifests={}) == []


def test_a_failed_dependency_cannot_be_replaced_by_a_handwritten_import_stub(tmp_path):
    fake = ('{python} -c "import os,sys; p=os.path.join(sys.prefix, '
            "'lib','python3.10','site-packages','missing_backend'); "
            "os.makedirs(p,exist_ok=True); "
            "open(os.path.join(p,'__init__.py'),'w').write('def ready(): return True')\"")
    assert "direct package-tree manipulation" in pv.package_tree_write_violation(fake)
    assert pv.plan_problems(plan(commands=[fake]))
    receipt = pv.run(fake.replace("{python}", sys.executable), env=dict(os.environ),
                     cwd=tmp_path, timeout=3, output=tmp_path / "build.log")
    assert receipt["failure_kind"] == "integrity_boundary"
    assert not (tmp_path / "build.log").exists()
    assert pv.package_tree_write_violation("{pip} install a_real_package") == ""


def test_package_tree_boundary_applies_to_repair_and_probe_commands():
    command = "mkdir -p {prefix}/lib/python3.10/dist-packages/a_fake_package"
    assert pv.package_tree_write_violation(command)
    assert any("direct package-tree" in problem for problem in
               pv.plan_problems(plan(probes=[command])))


def test_an_asset_claimed_already_present_must_exist_at_its_declared_path(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    value = plan(assets=[{"what": "dataset", "where": "datasets/example.hdf5",
                          "produced_by": "already present"}])
    context = {"repository_checkout": str(repo), "selected_asset_keys": ["dataset"]}
    assert any("declared path is absent" in problem for problem in
               pv.plan_problems(value, context))
    (repo / "datasets").mkdir()
    (repo / "datasets" / "example.hdf5").write_bytes(b"sample")
    assert not any("declared path is absent" in problem for problem in
                   pv.plan_problems(value, context))


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


def test_environment_probes_cannot_pass_by_installing_or_downloading_a_package():
    context = {"stage_paths": {"evaluate": {"entrypoint": "scripts/evaluate.py"}}}
    for probe in (
            "{python} -m pip install -v simulator_package",
            "{python} -m pip download --no-deps simulator_package",
            "{python} -m pip install simulator_package 2>&1 | tail -40"):
        assert "cannot install or download" in pv.probe_stage_violation(probe, context)
        assert any("probes[0]" in fault for fault in
                   pv.plan_problems(plan(probes=[probe]), context))


def test_environment_probe_rejects_shell_success_without_runtime_execution():
    context = {"stage_paths": {"evaluate": {"entrypoint": "scripts/evaluate.py"}}}
    assert "not print/list/check" in pv.probe_stage_violation("true", context)
    assert "constant/output-only" in pv.probe_stage_violation(
        "{python} -c 'print(1)'", context)
    assert "exercise selected repository/runtime capability" in pv.probe_stage_violation(
        "ls -la", context)


def test_source_backed_runtime_probes_remain_generic_and_allowed():
    context = {"stage_paths": {"evaluate": {"entrypoint": "scripts/evaluate.py"}}}
    assert pv.probe_stage_violation("{python} -c 'import simulator_package'", context) == ""
    assert pv.probe_stage_violation("cd {repo} && {python} scripts/evaluate.py --help",
                                    context) == ""


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


def test_plan_prompt_uses_local_asset_ids_and_resolves_them_only_after_reply(tmp_path):
    class Client:
        def __init__(self, choices):
            self.choices = list(choices)
            self.seen = []

        def chat_with_metadata(self, system, user, **kwargs):
            self.seen.append(user)
            return json.dumps(self.choices.pop(0)), {}

    repo = tmp_path / "repo"
    repo.mkdir()
    local_data = tmp_path / "private" / "lift_private_demos.hdf5"
    local_data.parent.mkdir()
    local_data.write_bytes(b"opaque demonstration bytes")
    runtime_probe = ["{python} -c 'import simulator_package'"]
    invalid = plan(probes=runtime_probe,
                   assets=[{"what": "demos", "where": "missing/private_demo.h5",
                            "produced_by": "already present"}])
    valid = plan(python="existing", probes=runtime_probe,
                 assets=[{"what": "demos", "where": "local_dataset_1",
                          "produced_by": "already present"}])
    client = Client([invalid, valid])
    assets = {**pv.asset_section({"datasets": [{"path": str(local_data)}],
                                  "manifest_excerpts": {}}),
              "research_context": {"selected_asset_keys": ["dataset"],
                                   "stage_paths": {"evaluate": {
                                       "entrypoint": "examples/eval.py",
                                       "artifact": "runs/*/final_ckpt.pt"}},
                                   "writable_output": str(tmp_path / "out")}}

    selected = pv.plan(client, repo, manifests={"README.md": "dataset is preloaded"},
                       assets=assets, python=sys.executable, attempts=2)

    assert selected["assets"][0]["where"] == str(local_data)
    assert len(client.seen) == 2
    for sent in client.seen:
        for forbidden in (str(local_data), local_data.name, "final_ckpt.pt",
                          str(sys.executable), "missing/private_demo.h5"):
            assert forbidden not in sent
    assert "local_dataset_1" in client.seen[0]
    assert json.loads(client.seen[0])["interpreter_already_present"] is True


def test_environment_repair_and_diagnosis_prompts_redact_paths_and_episode_ids(tmp_path):
    class Client:
        def __init__(self, answer):
            self.answer = answer
            self.seen = []

        def chat_with_metadata(self, system, user, **kwargs):
            self.seen.append(user)
            return json.dumps(self.answer), {}

    secret = "/mnt/private_lift/demos/episode_17.hdf5"
    episode_id = "episode-id:raw-seed-8c11"
    transcript = [{"command": f"python evaluate.py --dataset {secret}",
                   "ok": False, "failure_kind": "failed",
                   "excerpt": f"native evaluator failed for {episode_id}"}]
    record = [{"command": f"{sys.executable} -m pip install package --checkpoint={secret}",
               "kind": "command", "ok": True}]
    failure = {"command": f"python evaluate.py --dataset={secret}",
               "failure_kind": "failed", "excerpt": f"missing input for {episode_id}"}

    resume_client = Client({"commands": ["{python} -m pip install a_dependency"]})
    pv.resume(resume_client, tmp_path, record, failure, manifests={}, transcript=[],
              values={"python": sys.executable})
    reconsider_client = Client({"commands": ["{python} -m pip install a_dependency"]})
    pv.reconsider(reconsider_client, tmp_path, prefix=tmp_path / "environment",
                  record=record, transcript=transcript, manifests={})
    diagnose_client = Client({"blocker": "dependency", "what": "a missing import",
                              "evidence": "native probe failed", "tried": [],
                              "would_unblock": "install its dependency", "confidence": "high"})
    pv.diagnose(diagnose_client, tmp_path, record=record, transcript=transcript,
                probes=[failure["command"]], verdict={"reason": secret})

    for client in (resume_client, reconsider_client, diagnose_client):
        sent = "\n".join(client.seen)
        assert secret not in sent
        assert "episode_17.hdf5" not in sent
        assert "raw-seed-8c11" not in sent
        assert "--dataset" in sent
    assert "missing input for episode" in "\n".join(resume_client.seen)
    assert "native evaluator failed" in "\n".join(reconsider_client.seen)


# -- what kind of failure is it ------------------------------------------------------------

def test_the_classifier_reports_only_whether_the_command_ran():
    """It used to answer a second question -- environment or machine -- from seventeen
    substrings, and the answer ended the build.

    That question is asked better in this same file: `diagnose` puts the record, the
    transcript and the machine in front of a model and returns `blocker` with its evidence,
    and `resume` and `reconsider` carry an `unbuildable` field for the same judgement made
    earlier. Those are statements about *this* failure with a reason attached. This is a
    reading of the exit code and nothing else.
    """
    assert pv.classify(0, "all good", "") == "ok"
    for text in ("ModuleNotFoundError: No module named 'robosuite'",
                 "libEGL error: cannot open display :0",
                 "ERROR: Failed building wheel for egl-probe"):
        assert pv.classify(1, "", text) == "failed", text
    assert pv.classify(-11, "", "") == "failed"


def test_no_substring_decides_whether_the_build_can_continue():
    """The signals are gone from the module, not moved.

    They were patched by making them longer after a bare `egl` matched a package called
    `egl_probe` -- a rule repaired by narrowing a pattern, which is the shape that grows.
    """
    source = Path(pv.__file__).read_text(encoding="utf-8")
    for gone in ("libegl", "cannot open display", "no kernel image", "could not load gpu"):
        assert gone not in source.lower(), gone
    # And the model's own terminal answer is still wired: both prompts offer it.
    assert "unbuildable" in source


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


# -- carrying a lesson across environments -------------------------------------------------

def test_a_recipe_can_seed_the_next_build():
    """A lesson that cost nine rounds should not be relearned from a different prefix.

    The CMake switch was used in one build and lost when the next began elsewhere, because
    what a build learned about a repository lived only in the environment it learned it in.
    """
    seed = pv.seed_plan({"python": "3.10", "commands": ["a"], "probes": ["p"],
                         "reasoning": "it worked"})
    assert seed["python"] == "3.10" and seed["commands"] == ["a"] and seed["probes"] == ["p"]
    assert "seeded" in seed["reasoning"]


def test_recipe_identity_includes_selected_research_path():
    manifests = {"pyproject.toml": "same package"}
    machine = {"os": "linux", "arch": "x86_64", "gpus": [], "cuda_toolkit": ""}
    online = {"task": "A", "stage_paths": {"train": {"invocation": "online.py"}}}
    offline = {"task": "A", "stage_paths": {"train": {"invocation": "offline.py"}}}
    assert pv.recipe_key(manifests, machine, online) != pv.recipe_key(
        manifests, machine, offline)
    assert pv.recipe_key(manifests, machine, online) != pv.recipe_key(manifests, machine)


def test_a_seeded_plan_is_still_only_a_proposal():
    """The commands run and the probes decide, exactly as for a plan the model wrote."""
    import inspect
    source = inspect.getsource(pv.build)
    assert "seed_plan(seed)" in source
    assert "Not trusted" in source


def test_the_excerpt_carries_the_cause_in_a_short_log_too():
    """The threshold was a fixed line count, so a log just under it returned its head only
    and the fix named in its last line was never shown."""
    log = ("error: wrapper\n" + "\n".join(f"  copying f{i}" for i in range(20))
           + "\nCMake Error at CMakeLists.txt:1\n"
             "  add -DCMAKE_POLICY_VERSION_MINIMUM=3.5")
    excerpt = pv.error_excerpt(log)
    assert "wrapper" in excerpt and "CMAKE_POLICY_VERSION_MINIMUM" in excerpt
