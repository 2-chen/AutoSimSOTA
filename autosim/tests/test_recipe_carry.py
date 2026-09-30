"""A lesson learned building one environment, kept where the next build can find it.

`seed_plan`'s docstring said it: "a lesson that cost nine rounds should not have to be
relearned because the next build started from a different prefix." The file was written beside
the environment and read by nothing -- and even read, its commands were absolute paths, so the
lesson did not contain anything that survived the prefix it was learned in.
"""

import json
from pathlib import Path

from autosim.research.provision import (load_recipe, recipe_cache_dir, recipe_key, save_recipe,
                                        seed_plan, substitute)

MANIFESTS = {"requirements.txt": "mujoco==2.3.7\nrobosuite==1.4.0\n"}
MACHINE = {"gpus": [{"name": "RTX 5090", "compute_cap": "12.0"}], "cuda_toolkit": "12.8",
           "os": "Linux", "arch": "x86_64"}


def test_a_recipe_kept_for_these_dependencies_is_found_again(tmp_path):
    recipe = {"python": "3.10", "templates": ["{conda} create -y -p {prefix} python=3.10"],
              "probes": ["{python} -c 'import mujoco'"], "source": "libero"}
    save_recipe(recipe, manifests=MANIFESTS, machine=MACHINE, cache=tmp_path)
    assert load_recipe(manifests=MANIFESTS, machine=MACHINE, cache=tmp_path) == recipe
    # Nothing is claimed for a repository that declares something else.
    assert load_recipe(manifests={"requirements.txt": "torch\n"}, machine=MACHINE,
                       cache=tmp_path) is None
    # Nor for a machine of a different shape: a plan is a function of both.
    elsewhere = {**MACHINE, "cuda_toolkit": "11.3"}
    assert load_recipe(manifests=MANIFESTS, machine=elsewhere, cache=tmp_path) is None


def test_the_key_does_not_depend_on_where_the_checkout_is(tmp_path):
    """Or a lesson would only be reusable by the directory it was learned in."""
    here = recipe_key(MANIFESTS, MACHINE)
    assert here == recipe_key(dict(MANIFESTS), dict(MACHINE))
    assert here != recipe_key({"requirements.txt": "mujoco==3.13.0\n"}, MACHINE)


def test_the_portable_form_keeps_its_placeholders_and_so_fits_another_prefix(tmp_path):
    """The whole reason `templates` exists beside `commands`.

    A recipe of absolute paths substitutes nothing into itself: the next build would install
    into wherever the last one happened to live.
    """
    recipe = {"python": "3.10",
              "commands": ["/old/prefix/bin/pip install mujoco==2.3.7"],
              "templates": ["{pip} install mujoco==2.3.7"], "probes": []}
    planned = seed_plan(recipe)
    assert planned["commands"] == ["{pip} install mujoco==2.3.7"]
    values = {"pip": "/new/prefix/bin/pip"}
    assert substitute(planned["commands"][0], values) == "/new/prefix/bin/pip install mujoco==2.3.7"

    # A recipe written before templates existed still runs; it simply cannot be re-pointed.
    old = seed_plan({"python": "3.10", "commands": ["/old/prefix/bin/pip install mujoco"],
                     "probes": []})
    assert old["commands"] == ["/old/prefix/bin/pip install mujoco"]


def test_a_build_that_established_something_keeps_it(tmp_path, monkeypatch):
    """`save_recipe` is called by `_finish`, and this is that call, in the small."""
    monkeypatch.setenv("AUTOSIM_RECIPE_CACHE", str(tmp_path / "cache"))
    assert recipe_cache_dir() == tmp_path / "cache"
    save_recipe({"python": "3.10", "templates": ["{pip} install mujoco"], "probes": []},
                manifests=MANIFESTS, machine=MACHINE)
    assert (tmp_path / "cache").is_dir()
    assert load_recipe(manifests=MANIFESTS, machine=MACHINE) is not None


def test_a_build_around_an_interpreter_it_already_has_still_writes_a_recipe(tmp_path,
                                                                             monkeypatch):
    """The path where the plan names an existing interpreter instead of a version to build.

    Not a hypothetical: this is how a benchmark whose environment already exists on the
    machine gets provisioned, and nothing had exercised it. `_finish` read `manifests` and
    `plan_assets` out of its own `locals()` -- names that belong to `build`, so `manifests`
    was undefined and the recipe recorded `assets: []` whatever the plan had asked for. Every
    build that reached that line raised `NameError`, and the one that did not would have
    written down the wrong assets.
    """
    from autosim.research import provision

    # `_finish` saves the recipe into the machine-wide cache. Redirected, or this test writes
    # into the cache of whoever ran it -- which it did.
    monkeypatch.setenv("AUTOSIM_RECIPE_CACHE", str(tmp_path / "cache"))
    output = tmp_path / "provisioning"
    output.mkdir()
    recipe_path = provision._finish(
        output, tmp_path / "repo", str(tmp_path / "repo" / "bin" / "python"),
        record=[{"kind": "command", "command": "python -c 'import torch'",
                 "template": "{python} -c 'import torch'"}],
        transcript=[], probes=["torch"],
        verdict={"passed": True, "reason": "passed"},
        interpreter=tmp_path / "repo" / "bin" / "python",
        manifests={"requirements.txt": "torch\n", "README.md": "# x\n"},
        plan_assets=[{"what": "checkpoint", "where": "/models/act", "produced_by": "present"}])
    import json
    recipe = json.loads((output / "recipe.json").read_text(encoding="utf-8"))
    assert recipe["manifests"] == ["README.md", "requirements.txt"]
    assert recipe["assets"] == [{"what": "checkpoint", "where": "/models/act",
                                 "produced_by": "present"}]
    # And the interpreter is the one it was told to use, not one it would have built.
    assert recipe["python"].endswith("bin/python")


def test_no_name_is_read_out_of_a_caller_s_locals():
    """`locals().get(name)` reaching for a variable that belongs to another function returns
    `None` and looks like an answer. It did here twice: `manifests` raised `NameError` on
    every build that got that far, and `plan_assets` silently recorded `assets: []` for plans
    that had asked for assets. Both are now parameters."""
    from autosim.research import provision

    source = Path(provision.__file__).read_text(encoding="utf-8")
    assert "locals().get" not in source


def test_a_build_given_an_interpreter_still_plans_and_still_probes(tmp_path, monkeypatch):
    """The whole point of the stage is that a build which did nothing cannot report that it
    passed. It could: a given interpreter skipped the plan, the installs, the assets and every
    probe, and fell through to `passed` on an empty record."""
    from autosim.research import provision

    monkeypatch.setenv("AUTOSIM_RECIPE_CACHE", str(tmp_path / "cache"))
    repo = tmp_path / "repo"
    (repo).mkdir()
    (repo / "requirements.txt").write_text("torch\n", encoding="utf-8")
    interpreter = tmp_path / "given" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    prefix = tmp_path / "provisioning" / "env"

    ran: list[str] = []

    def fake_run(command, **kwargs):
        ran.append(command)
        return {"command": command, "ok": True, "returncode": 0, "seconds": 0.0,
                "failure_kind": None, "excerpt": ""}

    monkeypatch.setattr(provision, "run", fake_run)
    monkeypatch.setattr(provision, "platform_facts", lambda: {"gpus": [], "cuda_toolkit": None,
                                                              "os": "Linux", "arch": "x86_64"})

    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return json.dumps({
                "python": "3.10",
                "commands": ["{conda} create -p {prefix} python=3.10",
                             "{pip} install -r {repo}/requirements.txt"],
                "probes": ['{python} -c "import torch"'],
                "reasoning": "test"}), {}

    result = provision.build(repo, client=Client(), prefix=prefix,
                             output=tmp_path / "provisioning", python=str(interpreter))

    # The plan was made and its installs ran.
    assert any("install -r" in command for command in ran), ran
    # The environment creation did not: the caller named an interpreter that exists.
    assert not any("conda create" in command for command in ran), ran
    # And the probe ran, which is what makes `passed` mean anything.
    assert any("import torch" in command for command in ran), ran
    assert result["verdict"]["passed"] is True
    assert result["interpreter"] == str(interpreter)


def test_an_asset_where_that_is_really_two_paths_is_refused_when_the_plan_is_written():
    """The model wrote the two directories a dataset lives in joined by the word " and ".

    That string passes every other check here, becomes a probe for a path that cannot exist,
    and cannot be recovered from -- the RoboTwin build spent its entire round budget on it and
    ended with `rounds exhausted`, having verified everything else about the environment.

    A validator is the right place because the message goes back to the model, which can fix
    it in one turn. The probe was catching the same fault and could not say what to write.
    """
    from autosim.research.provision import plan_problems

    def plan(where):
        return {"python": "3.10", "commands": ["x"], "probes": ["y"],
                "assets": [{"what": "the episodes", "where": where,
                            "produced_by": "already present"}]}

    faults = plan_problems(plan("data/a/data/ and data/b/data/"))
    assert len(faults) == 1
    assert "exactly one path" in faults[0]
    assert "its own entry in `assets`" in faults[0]
    assert " and " in faults[0]                     # names what it saw

    # And the paths that are one path still pass, including awkward ones. A validator that
    # refuses a directory with a space in its name costs a valid plan.
    for fine in ("/data/a", "/mnt/my data/v1", "/a,b/c", "data/episode_0000001.hdf5"):
        assert plan_problems(plan(fine)) == [], fine
