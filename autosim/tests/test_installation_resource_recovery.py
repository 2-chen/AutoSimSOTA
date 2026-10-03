"""Failed downloads remain actionable evidence, not an automatic impossibility claim."""
import json
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from autosim.research import environment_pool as pool, provision as pv, recorder, run_record
from autosim.research.common import atomic_json, run_local_environment
from autosim.research.installation_recovery import inventory, review_boundary
from autosim.research.prepare import Preparation


def wheel(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("recovery_fixture.py", "VALUE = 73\n")
        for name, content in {
            "WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            "METADATA": "Metadata-Version: 2.1\nName: recovery-fixture\nVersion: 1.0\n",
            "RECORD": ""}.items():
            archive.writestr("recovery_fixture-1.0.dist-info/" + name, content)
    return path


@pytest.mark.parametrize("timed_out", [False, True])
def test_failed_install_publishes_download_and_next_operation_refreshes_view(tmp_path, monkeypatch, timed_out):
    output = tmp_path / "run"
    repo = output / "checkout"
    repo.mkdir(parents=True)
    pool.configure(output, tmp_path / "store")
    original_env = run_local_environment(output)
    assert "PIP_FIND_LINKS" not in original_env
    seen = []
    def execute(argv, **kwargs):
        seen.append(kwargs["env"])
        if len(seen) == 1:
            wheel(output / "home/.cache/pip/http-v2/already-downloaded.body")
            if timed_out:
                raise subprocess.TimeoutExpired(argv, 10, output="downloaded wheel; source timed out")
            return subprocess.CompletedProcess(argv, 1, "downloaded wheel", "connection timeout")
        return subprocess.CompletedProcess(argv, 0, "ok", "")
    @contextmanager
    def resource_window(*args, **kwargs):
        yield 10, False, {}
    monkeypatch.setattr(pv, "isolated_argv", lambda argv, **kwargs: argv)
    monkeypatch.setattr(pv, "bounded_run", execute)
    monkeypatch.setattr("autosim.research.native_execution.resource_window", resource_window)
    first = pv.run("echo failed", env=original_env, cwd=repo, timeout=10, output=output / "build.log")
    assert first["ok"] is False and first["evidence_id"]
    target = next((tmp_path / "store").glob("wheels/*/*.whl"))
    assert target.name == "recovery_fixture-1.0-py3-none-any.whl"
    second = pv.run("echo retry", env=original_env, cwd=repo, timeout=10, output=output / "build.log")
    assert second["ok"] and target.as_uri() in seen[-1]["PIP_FIND_LINKS"]
    assert seen[-1]["PIP_DEFAULT_TIMEOUT"] == "20" and seen[-1]["PIP_RETRIES"] == "1"
    # Genuine offline pip consumption of the published wheel, not just a link assertion.
    install = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
        "--target", str(output / "fixture-site"), str(target)], capture_output=True, text=True)
    assert install.returncode == 0, install.stderr
    probe = subprocess.run([sys.executable, "-c", "import recovery_fixture; assert recovery_fixture.VALUE == 73"],
        env={**original_env, "PYTHONPATH": str(output / "fixture-site")}, capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr


def test_inventory_finds_body_even_without_shared_pool_and_excludes_symlinks(tmp_path):
    wheel(tmp_path / "home/.cache/pip/http-v2/cache.body")
    (tmp_path / "home/.cache/pip/escape.body").symlink_to(tmp_path / "home/.cache/pip/http-v2/cache.body")
    result = inventory(tmp_path)
    assert len(result["local_wheels"]) == 1
    assert result["local_wheels"][0]["cache_ref"].endswith("cache.body")
    assert str(tmp_path) not in json.dumps(result)
    assert inventory(tmp_path)["digest"] == result["digest"]
    assert inventory(tmp_path, limit=0)["scan_complete"] is False


def test_state_inventory_is_read_only_and_stable(tmp_path):
    output = tmp_path / "not-created"
    first = inventory(output, persist=False)
    assert first == inventory(output, persist=False)
    assert not output.exists()
    wheel(tmp_path / "home/.cache/pip/cache.body")
    assert inventory(tmp_path) == inventory(tmp_path)


def test_old_export_index_and_stale_view_are_refreshed_without_redownload(tmp_path):
    output = tmp_path / "run"
    store = tmp_path / "store"
    pool.configure(output, store)
    cached = wheel(output / "home/.cache/pip/cache.body")
    pool.publish_wheels(output)
    atomic_json(output / "package_cache_view.json", {"root": str(store), "links": []})
    # Old publication succeeded but the run view stayed empty.
    atomic_json(output / "package_cache_export_index.json", {
        "cache.body": [cached.stat().st_size, cached.stat().st_mtime_ns]})
    assert pool.publish_wheels(output)["bytes"] == 0
    assert "recovery_fixture" in run_local_environment(output)["PIP_FIND_LINKS"]
    atomic_json(output / "package_cache_view.json", {"root": str(store), "links": []})
    assert pool.publish_wheels(output)["bytes"] == 0
    assert "recovery_fixture" in run_local_environment(output)["PIP_FIND_LINKS"]


def test_report_exposes_recovery_leads_without_claiming_readiness(tmp_path):
    wheel(tmp_path / "home/.cache/pip/cache.body")
    inventory(tmp_path)
    view = run_record.build_report_view(tmp_path, "derived", status="adaptation_unresolved")
    snap = recorder.make_snapshot(tmp_path, view, {"actions": []})
    assert recorder._evidence(snap)["installation_recovery_inventory"]["local_wheels"]
    result = recorder.render(tmp_path, snap, {})
    assert "本轮缓存发现 1 个 wheel" in result
    assert "不是已安装、兼容或仿真就绪" in result


def test_boundary_cannot_ignore_local_wheel_or_incomplete_scan(tmp_path):
    wheel(tmp_path / "home/.cache/pip/http-v2/cache.body")
    resources = inventory(tmp_path)
    with pytest.raises(ValueError, match="resource_assessment"):
        review_boundary(object(), resources=resources, reason="source timed out", assessment=None, failure={})
    with pytest.raises(ValueError, match="resource_assessment"):
        review_boundary(object(), resources=inventory(tmp_path, limit=0), reason="absent", assessment={}, failure={})


@pytest.mark.parametrize("approved", [False, True])
def test_boundary_requires_read_only_objective_and_current_inventory(tmp_path, approved):
    wheel(tmp_path / "home/.cache/pip/cache.body")
    resources = inventory(tmp_path)
    assessment = {"inventory_digest": resources["digest"], "local_artifacts": "native probe showed unsupported tags",
                  "environment_reuse": "available clones incompatible", "alternative_sources": "documented sources checked"}
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            assert kwargs["read_only"] is True
            assert json.loads(user)["resources"]["local_wheels"]
            return json.dumps({"approved": approved, "reason": "independent evidence assessment"}), {}
    if approved:
        assert review_boundary(Client(), resources=resources, reason="cannot run", assessment=assessment, failure={})["approved"]
    else:
        with pytest.raises(ValueError, match="independently approved"):
            review_boundary(Client(), resources=resources, reason="cannot run", assessment=assessment, failure={})


def test_fix_gets_cache_inventory_and_unsupported_unbuildable_is_rejected(tmp_path, monkeypatch):
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    wheel(output / "home/.cache/pip/cache.body")
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            data = json.JSONDecoder().raw_decode(user)[0]
            assert data["installation_recovery_inventory"]["local_wheels"]
            if "rejected" not in user:
                return json.dumps({"unbuildable": "online source unavailable"}), {}
            return json.dumps({"commands": ["echo inspect cached wheel"]}), {}
    client = Client(); client.output = output
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    transcript = []
    commands, _ = pv.resume(client, repo, [], {"command": "pip install dependency", "failure_kind": "timeout"},
        manifests={}, transcript=transcript, values={}, attempts=2)
    assert commands == ["echo inspect cached wheel"]
    assert not any(row["kind"] == "unbuildable" for row in transcript)


def test_main_controller_cannot_bypass_resource_boundary_review(tmp_path, monkeypatch):
    wheel(tmp_path / "home/.cache/pip/cache.body")
    monkeypatch.setattr(pv, "latest_failure", lambda *args: {"evidence_id": "a" * 32})
    state = SimpleNamespace(output=tmp_path, interpreter=None, budget=None, client=object())
    state._installation_recovery_inventory = lambda: inventory(tmp_path)
    rejected = Preparation._review_environment_stop(state, {"do": "stop", "why": "no network", "arguments": {}})
    assert rejected["decision_failure"]["kind"] == "resource_boundary_unproven"
    assert Preparation._review_environment_stop(state, {"do": "build_the_environment"})["do"] == "build_the_environment"


def test_report_marks_historical_plan_error_as_crossed_not_current(tmp_path):
    atomic_json(tmp_path / "provision_progress.json", {"attempts": [{"ok": False, "seconds": 12}]})
    view = run_record.build_report_view(tmp_path, "derived", status="adaptation_unresolved")
    snap = recorder.make_snapshot(tmp_path, view, {"actions": [
        {"step": "build_the_environment", "failure_domain": "framework_plan", "because": "bad plan"},
        {"step": "retry_failed_action", "replayed_step": "build_the_environment", "outcome": "checkpoint failure"}]})
    result = recorder.render(tmp_path, snap, {})
    assert "后续已进入原生安装" in result
    assert "尚未执行原生安装" not in result
    assert "不代表完整环境就绪" in result
