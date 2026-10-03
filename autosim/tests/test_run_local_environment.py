from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

import autosim.research.common as common
from autosim.research.common import isolated_argv, run_local_environment
from autosim.research.process_executor import run_process


def test_implicit_home_and_cache_writes_use_the_run_directory(tmp_path):
    outside = tmp_path / "operator-home"
    output = tmp_path / "run"
    env = run_local_environment(output, {"HOME": str(outside), "TOKEN": "kept"})
    assert env["HOME"] == str(output / "home")
    assert env["XDG_CACHE_HOME"] == str(output / "home" / ".cache")
    assert env["HF_HOME"] == str(output / "home" / ".cache" / "huggingface")
    assert env["TOKEN"] == "kept"
    assert Path(env["XDG_CACHE_HOME"]).is_dir()
    assert not outside.exists()


def test_native_home_has_short_alias_but_preserves_actual_cache_bytes(tmp_path):
    import json
    output = tmp_path / ('long_run_' * 12)
    repo = output / 'checkout'
    repo.mkdir(parents=True)
    env = run_local_environment(output, {})
    home = output / 'home'
    (home / '.cache' / 'witness.txt').write_text('existing cache')
    code = ("import os,json;from pathlib import Path;"
            "h=Path.home(); c=Path(os.environ['XDG_CACHE_HOME']);"
            "print(json.dumps({'home':str(h),'cached':(c/'witness.txt').read_text()}));"
            "(c/'native_write.txt').write_text('same backing home')")
    command = isolated_argv([sys.executable, '-c', code], output=output, repo=repo,
                            native_environment=env, allow_gpu=False)
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert len(data['home'].encode()) < 32 and data['cached'] == 'existing cache'
    assert (home / '.cache/native_write.txt').read_text() == 'same backing home'
    assert not (Path(data['home']) / '.cache/native_write.txt').exists()
    assert list((output / 'native_home_mappings').glob('*.json'))


def test_native_alias_cannot_grant_external_home_or_restricted_write_access(tmp_path):
    output = tmp_path / 'run'
    repo = output / 'checkout'
    repo.mkdir(parents=True)
    with pytest.raises(ValueError, match='confined'):
        isolated_argv(['true'], output=output, repo=repo,
                      native_environment={'HOME': str(tmp_path / 'external')})
    env = run_local_environment(output, {})
    with pytest.raises(ValueError, match='widen'):
        isolated_argv(['true'], output=output, repo=repo,
                      native_environment=env, writable_paths=(repo,))


def test_isolated_copy_can_write_its_run_but_not_a_sibling():
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap is not available")
    project = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="autosim-boundary-", dir=project) as temporary:
        root = Path(temporary)
        output = root / "run"
        repo = output / "checkout"
        repo.mkdir(parents=True)
        (output / "workspace_snapshot.json").write_text("{}", encoding="utf-8")
        outside = root / "sibling.txt"
        allowed = output / "allowed.txt"
        command = [sys.executable, "-c",
                   "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('x')",
                   str(outside)]
        refused = subprocess.run(isolated_argv(command, output=output, repo=repo),
                                 cwd=repo, capture_output=True, text=True, timeout=10)
        assert refused.returncode != 0
        assert not outside.exists()
        command[-1] = str(allowed)
        accepted = subprocess.run(isolated_argv(command, output=output, repo=repo),
                                  cwd=repo, capture_output=True, text=True, timeout=10)
        assert accepted.returncode == 0, accepted.stderr
        assert allowed.read_text(encoding="utf-8") == "x"


def test_no_bubblewrap_direct_run_requires_explicit_weak_isolation_opt_in(
        tmp_path, monkeypatch):
    monkeypatch.setattr(common.shutil, "which", lambda _name: None)
    repo = tmp_path / "repo"
    output = tmp_path / "run"
    repo.mkdir()
    output.mkdir()
    argv = [sys.executable, "-c", "pass"]

    monkeypatch.delenv("AUTOSIM_ALLOW_PROCESS_GROUP_ONLY", raising=False)
    with pytest.raises(ValueError, match="refusing weak process-group-only execution"):
        isolated_argv(argv, output=output, repo=repo)

    with pytest.raises(ValueError, match="requires a bubblewrap PID namespace"):
        isolated_argv(argv, output=output, repo=repo, require_pid_namespace=True)

    monkeypatch.setenv("AUTOSIM_ALLOW_PROCESS_GROUP_ONLY", "1")
    command = isolated_argv(argv, output=output, repo=repo)
    assert command == argv
    result = run_process(command, cwd=repo, env={}, timeout=5)

    assert result.returncode == 0
    assert result.containment_mode == "process_group_only"


def test_no_bubblewrap_refuses_isolated_copy_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(common.shutil, "which", lambda _name: None)
    output = tmp_path / "run"
    repo = output / "checkout"
    repo.mkdir(parents=True)
    (output / "workspace_snapshot.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="requires bubblewrap"):
        isolated_argv([sys.executable, "-c", "raise SystemExit(99)"],
                      output=output, repo=repo)
