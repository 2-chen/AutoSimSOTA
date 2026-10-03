"""Reusable packages/bases are accelerators, not simulator readiness certificates."""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from autosim.research import environment_pool as pool
from autosim.research import provision as pv
from autosim.research.common import atomic_json, isolated_argv, run_local_environment


def prefix_at(path, *, name="torch", version="2.7.1", editable=False):
    (path / "bin").mkdir(parents=True)
    (path / "bin" / "python").write_text("fixture interpreter; never execute")
    (path / "conda-meta").mkdir()
    info = path / "lib/python3.10/site-packages" / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    if editable:
        (info / "direct_url.json").write_text(json.dumps({"url": "file:///private/repo",
                                                        "dir_info": {"editable": True}}))
    return path


def cached_wheel(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("autosim_pool_fixture.py", "VALUE = 'offline cache reused'\n")
        archive.writestr("autosim_pool_fixture-0.0.1.dist-info/WHEEL",
                         "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("autosim_pool_fixture-0.0.1.dist-info/METADATA",
                         "Metadata-Version: 2.1\nName: autosim-pool-fixture\nVersion: 0.0.1\n")
        archive.writestr("autosim_pool_fixture-0.0.1.dist-info/RECORD", "")
    return path


def test_inventory_reads_metadata_without_running_interpreter(tmp_path):
    source = prefix_at(tmp_path / "operator-env")
    rows = pool.discover(tmp_path / "store", prefixes=[source])
    assert rows[0]["cloneable"] and rows[0]["packages"]["torch"] == "2.7.1"
    short = pool.short_catalog(rows)
    assert "prefix" not in short[0]
    assert str(source) not in json.dumps(short)
    assert short[0]["readiness"] == "unverified"


def test_editable_and_inherited_environments_are_not_cloneable(tmp_path):
    source = prefix_at(tmp_path / "editable", editable=True)
    assert not pool.describe(source)["cloneable"]
    plain = prefix_at(tmp_path / "inherited")
    (plain / "lib/python3.10/site-packages/parent.pth").write_text("/outside/site-packages\n")
    assert not pool.describe(plain)["cloneable"]


def test_python_version_and_conda_build_provenance_are_not_false_rejections(tmp_path):
    source = prefix_at(tmp_path / "base")
    direct = source / "lib/python3.10/site-packages/torch-2.7.1.dist-info/direct_url.json"
    direct.write_text(json.dumps({"url": "file:///deleted/build/torch", "dir_info": {}}))
    atomic_json(source / "conda-meta/python.json", {"name": "python", "version": "3.10.19"})
    atomic_json(source / "conda-meta/torch.json", {
        "files": [direct.relative_to(source).as_posix()]})
    (source / "lib/python3.1/site-packages").mkdir(parents=True)
    row = pool.describe(source)
    assert row["python"] == "3.10" and row["cloneable"]
    assert row["packages"] == {"torch": "2.7.1"}
    (source / "conda-meta/torch.json").unlink()
    assert "repository-local installed package torch" in pool.describe(source)["limitations"]


def test_pool_cannot_be_inside_run_or_source(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        pool.configure(tmp_path / "run", tmp_path / "run/cache")
    with pytest.raises(ValueError, match="source repository"):
        pool.configure(tmp_path / "run", tmp_path / "source/cache",
                       source_repository=tmp_path / "source")


def test_package_cache_is_shared_but_installs_and_home_remain_local(tmp_path):
    first, second = tmp_path / "r1", tmp_path / "r2"
    store = tmp_path / "store"
    pool.configure(first, store)
    wheel = cached_wheel(first / "home/.cache/pip/http-v2/download.body")
    result = pool.publish_wheels(first)
    assert result["wheels"] and result["bytes"] == wheel.stat().st_size
    pool.configure(second, store)
    env = run_local_environment(second)
    assert env["PIP_FIND_LINKS"].startswith("file:")
    assert str(second) in env["PIP_CACHE_DIR"] and str(second) in env["CONDA_PKGS_DIRS"]
    assert pool.publish_wheels(first)["bytes"] == 0
    target = next(store.glob("wheels/*/*.whl"))
    target.chmod(0o644)
    target.write_bytes(b"corrupted")
    assert pool.wheel_links(second) == []


def test_conda_clone_uses_copy_and_read_only_source(tmp_path, monkeypatch):
    source = prefix_at(tmp_path / "base")
    output = tmp_path / "run"
    output.mkdir()
    captured = {}

    def execute(command, **kwargs):
        captured.update(kwargs)
        captured["command"] = command
        shutil.copytree(source, output / "env")
        return {"ok": True}

    monkeypatch.setattr(pv, "run", execute)
    result = pool.clone(source, destination=output / "env", output=output, repo=output)
    assert result["ok"]
    assert "--offline --copy --clone" in captured["command"]
    assert captured["read_only_roots"] == (source,)
    assert os.stat(source / "bin/python").st_ino != os.stat(output / "env/bin/python").st_ino


def test_conda_cache_sequence_uses_comma_not_path_separator(tmp_path, monkeypatch):
    source = prefix_at(tmp_path / "base")
    cache = tmp_path / "packages"
    extracted = cache / "python-package"
    extracted.mkdir(parents=True)
    atomic_json(source / "conda-meta/python.json", {"link": {"source": str(extracted)}})
    output = tmp_path / "run"
    output.mkdir()
    captured = {}

    def execute(command, **kwargs):
        captured.update(kwargs)
        shutil.copytree(source, output / "env")
        return {"ok": True}

    monkeypatch.setattr(pv, "run", execute)
    assert pool.clone(source, destination=output / "env", output=output, repo=output)["ok"]
    assert captured["env"]["CONDA_PKGS_DIRS"].split(",") == [
        str(output / "home/.cache/conda"), str(cache)]
    assert captured["read_only_roots"] == (source, cache)


def test_clone_validation_has_its_own_failed_evidence(tmp_path, monkeypatch):
    source = prefix_at(tmp_path / "base")
    output = tmp_path / "run"
    output.mkdir()
    original = {"ok": True, "evidence_id": "a" * 32}

    def execute(command, **kwargs):
        prefix_at(output / "env", version="9.0")
        return original

    monkeypatch.setattr(pv, "run", execute)
    result = pool.clone(source, destination=output / "env", output=output, repo=output)
    assert not result["ok"] and result["phase"] == "after_execution"
    assert result["failure_kind"] == "clone_verification"
    assert original["ok"]  # A successful command is not rewritten to be a failure.
    assert result["prior_evidence_id"] == original["evidence_id"]
    assert result["evidence_id"] != original["evidence_id"]
    assert (output / result["evidence_ref"]).is_file()


def test_snapshot_capacity_failure_is_optional_and_keeps_source(tmp_path, monkeypatch):
    output = tmp_path / "run"
    pool.configure(output, tmp_path / "store")
    source = prefix_at(output / "env")
    before = pool.tree_digest(source)
    monkeypatch.setattr(pool, "MAX_STORE_BYTES", 1)
    result = pool.publish_snapshot(output, interpreter=source / "bin/python",
                                  manifests={}, machine={})
    assert result["status"] == "capacity_skipped"
    assert pool.tree_digest(source) == before
    assert not list((tmp_path / "store/snapshots").iterdir())


def test_clone_refuses_existing_nonempty_prefix(tmp_path):
    source = prefix_at(tmp_path / "base")
    output = tmp_path / "run"
    destination = prefix_at(output / "env", version="old")
    before = pool.tree_digest(destination)
    with pytest.raises(ValueError, match="overwrite"):
        pool.clone(source, destination=destination, output=output, repo=output)
    assert pool.tree_digest(destination) == before


def test_agent_selects_opaque_base_id_with_reason_and_matching_python(tmp_path):
    row = pool.short_catalog([pool.describe(prefix_at(tmp_path / "base"))])[0]

    class Agent:
        def chat_with_metadata(self, system, payload, **kwargs):
            assert "environment_candidates" in payload
            return json.dumps({"python": "3.10", "commands": ["echo incremental install"],
                "probes": ["{python} -c 'print(1)'"], "reasoning": "verify capabilities",
                "base_environment_id": row["id"],
                "environment_selection_reason": "matching Python and compatible torch"}), {}

    plan = pv.plan(Agent(), tmp_path, manifests={}, assets={"environment_candidates": [row]})
    assert plan["base_environment_id"] == row["id"]


def test_agent_selected_clone_skips_creation_but_runs_install_and_probes(tmp_path, monkeypatch):
    output = tmp_path / "run"
    repo = output / "checkout"
    repo.mkdir(parents=True)
    pool.configure(output, tmp_path / "store")
    source = prefix_at(tmp_path / "base")
    row = pool.describe(source)
    calls = []

    class Agent:
        def chat_with_metadata(self, system, payload, **kwargs):
            assert row["id"] in payload and str(source) not in payload
            return json.dumps({"python": "3.10", "commands": [
                "{conda} create --yes --prefix {prefix} python=3.10",
                "{python} -m pip install incremental-package"],
                "probes": ["{python} -c 'print(1)'"], "reasoning": "verify native capabilities",
                "base_environment_id": row["id"],
                "environment_selection_reason": "matching Python and torch"}), {}

    def clone(origin, *, destination, **kwargs):
        calls.append("clone")
        shutil.copytree(origin, destination)
        return {"ok": True}

    def run(command, **kwargs):
        calls.append(command)
        return {"ok": True, "command": command, "seconds": 0, "excerpt": "installed"}

    def probe(command, **kwargs):
        calls.append("probe")
        assert kwargs["values"]["python"] == str(output / "env/bin/python")
        return {"ok": True, "seconds": 0, "excerpt": "native probe executed"}

    monkeypatch.setattr(pool, "catalog", lambda *a: [row])
    monkeypatch.setattr(pool, "clone", clone)
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    monkeypatch.setattr(pv, "load_recipe", lambda **k: None)
    monkeypatch.setattr(pv, "run", run)
    monkeypatch.setattr(pv, "probe_environment", probe)
    monkeypatch.setattr(pv, "_finish", lambda *a, **k: {"verdict": a[6]})
    result = pv.build(repo, client=Agent(), prefix=output / "env", output=output,
                      manifests={}, assets={}, max_rounds=1)
    assert result["verdict"]["passed"]
    assert calls[0] == "clone" and calls[-1] == "probe"
    assert len(calls) == 3 and "pip install incremental-package" in calls[1]
    selected = json.loads((output / "environment_selection.json").read_text())
    assert selected["result"]["ok"] and selected["reason"] == "matching Python and torch"


def test_snapshot_contains_no_repository_or_task_assets_and_requires_reprobe(tmp_path, monkeypatch):
    actual_clone = pool.clone
    output = tmp_path / "run"
    pool.configure(output, tmp_path / "store")
    source = prefix_at(output / "env")

    def copy_clone(origin, *, destination, **kwargs):
        shutil.copytree(origin, destination)
        return {"ok": True}

    monkeypatch.setattr(pool, "clone", copy_clone)
    monkeypatch.setattr(pool, "discover", lambda *_, **kw: [])
    manifests, machine = {"requirements.txt": "torch==2.7.1"}, {"os": "Linux"}
    result = pool.publish_snapshot(output, interpreter=source / "bin/python",
                                   manifests=manifests, machine=machine)
    assert result["status"] == "published"
    catalog = pool.catalog(output, manifests, machine)
    assert catalog[0]["dependency_match"] is True
    assert catalog[0]["readiness"] == "requires_current_repository_probes"
    (Path(catalog[0]["prefix"]) / "bin/python").write_text("mutated")
    with pytest.raises(ValueError, match="bytes changed"):
        actual_clone(Path(catalog[0]["prefix"]), destination=tmp_path / "newenv",
                   output=output, repo=output, expected_tree=catalog[0]["tree_sha256"])
    assert pool.catalog(output, manifests, machine) == []


def test_shared_store_is_read_only_even_when_under_tmp(tmp_path):
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap unavailable")
    output = tmp_path / "run"
    repo = output / "checkout"
    repo.mkdir(parents=True)
    atomic_json(output / "workspace_snapshot.json", {})
    pool.configure(output, tmp_path / "store")
    target = tmp_path / "store/protected"
    target.write_text("original")
    argv = [sys.executable, "-c", "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).write_text('modified')", str(target)]
    result = subprocess.run(isolated_argv(argv, output=output, repo=repo),
                            capture_output=True, timeout=10)
    assert result.returncode != 0 and target.read_text() == "original"


def test_real_offline_pip_consumes_shared_wheel_without_mutating_store(tmp_path):
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap unavailable")
    first, output = tmp_path / "producer", tmp_path / "consumer"
    store = tmp_path / "store"
    pool.configure(first, store)
    cached_wheel(first / "home/.cache/pip/wheels/fixture.whl")
    pool.publish_wheels(first)
    pool.configure(output, store)
    repo = output / "checkout"
    repo.mkdir()
    subprocess.run([sys.executable, "-m", "venv", str(output / "env")], check=True, timeout=60)
    target = next(store.glob("wheels/*/*.whl"))
    before = pool.digest(target)
    argv = [str(output / "env/bin/python"), "-m", "pip", "install", "--no-index",
            "autosim-pool-fixture==0.0.1"]
    installed = subprocess.run(isolated_argv(argv, output=output, repo=repo),
        env=run_local_environment(output), capture_output=True, text=True, timeout=60)
    assert installed.returncode == 0, installed.stderr
    checked = subprocess.check_output([str(output / "env/bin/python"), "-c",
        "import autosim_pool_fixture; print(autosim_pool_fixture.VALUE)"], text=True)
    assert "offline cache reused" in checked
    assert pool.digest(target) == before
