import shutil
import subprocess

import pytest

from autosim.research.common import isolated_argv, inspection_argv
from autosim.research.workspace_resources import (bindings_for, mount_args, parse_binding,
                                                  resource_view)
from autosim.research.workspace_snapshot import create, existing


def workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("original")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "train.py"], check=True)
    data = source / "large-data"
    data.mkdir()
    (data / "input.txt").write_text("dataset")
    checkout = tmp_path / "run" / "checkout"
    return source, data, checkout


def attach(source, data, checkout):
    return create(source, checkout, tracked_only=True, max_bytes=8,
                  resources=[{"source": str(data), "target": "data"}])


def test_explicit_resources_are_not_copied_or_linked(tmp_path):
    source, data, checkout = workspace(tmp_path)
    manifest = attach(source, data, checkout)
    assert manifest["bytes"] == 8
    assert list((checkout / "data").iterdir()) == []
    assert not (checkout / "data").is_symlink()
    assert "large-data/" in manifest["omitted_candidates"]
    assert resource_view(checkout)["bindings"][0]["access"] == "read_only"
    (checkout / "train.py").write_text("edited")
    assert (source / "train.py").read_text() == "original"
    assert bindings_for(checkout) == manifest["resource_bindings"]
    assert existing(checkout, source=source)["resource_bindings"]
    from autosim.research.survey import survey, summarise_for_model
    report = summarise_for_model(survey(checkout))
    assert report["workspace_resources"]["bindings"][0]["target"] == "data"
    assert str(data) not in str(report["workspace_resources"])


@pytest.mark.parametrize("target", ["../escape", "/tmp/escape", "", ".git/config", "train.py"])
def test_unsafe_or_code_overlay_targets_refused(tmp_path, target):
    source, data, checkout = workspace(tmp_path)
    with pytest.raises(ValueError):
        create(source, checkout, tracked_only=True,
               resources=[{"source": str(data), "target": target}])
    assert not checkout.exists()


def test_missing_replaced_and_modified_resources_fail_closed(tmp_path):
    source, data, checkout = workspace(tmp_path)
    attach(source, data, checkout)
    moved = data.with_name("saved-data")
    data.rename(moved)
    with pytest.raises(FileNotFoundError):
        mount_args(checkout)
    data.mkdir()
    with pytest.raises(ValueError, match="identity changed"):
        mount_args(checkout)
    data.rmdir()
    moved.rename(data)
    (checkout / "data" / "unexpected-output").write_text("keep this")
    with pytest.raises(ValueError, match="placeholder"):
        mount_args(checkout)


def test_symlinked_mount_target_refused(tmp_path):
    source, data, checkout = workspace(tmp_path)
    attach(source, data, checkout)
    (checkout / "data").rmdir()
    (checkout / "data").symlink_to(data)
    with pytest.raises(ValueError, match="symlink"):
        mount_args(checkout)


def test_file_resource_and_reconstruction_identity(tmp_path):
    source, data, checkout = workspace(tmp_path)
    manifest = create(source, checkout, tracked_only=True,
                      resources=[parse_binding(f"weights.bin={data / 'input.txt'}")])
    assert (checkout / "weights.bin").stat().st_size == 0
    rebuilt = create(source, tmp_path / "other" / "checkout", tracked_only=True,
                     resources=manifest["resource_bindings"])
    assert rebuilt["source_tree_fingerprint"] == manifest["source_tree_fingerprint"]


def test_explicit_tracked_resource_is_excluded_from_code_copy(tmp_path):
    source, data, checkout = workspace(tmp_path)
    (data / "large.bin").write_bytes(b"a" * 1024)
    subprocess.run(["git", "-C", str(source), "add", "large-data"], check=True)
    manifest = create(source, checkout, tracked_only=True, max_bytes=8,
                      resources=[{"target": "large-data", "source": str(data)}])
    assert manifest["bytes"] == 8
    assert manifest["resource_source_exclusions"] == ["large-data"]
    assert not list((checkout / "large-data").iterdir())
    assert bindings_for(checkout)


def test_resource_overlap_and_repository_exposure_refused(tmp_path):
    source, data, checkout = workspace(tmp_path)
    for specs in ([{"target": "source", "source": str(source)}],
                  [{"target": "data", "source": str(data)},
                   {"target": "data/sub", "source": str(data)}]):
        with pytest.raises(ValueError):
            create(source, checkout, tracked_only=True, resources=specs)


@pytest.mark.parametrize("mode", ["native", "inspection", "agent"])
def test_real_sandbox_reads_resource_but_cannot_modify_it(tmp_path, mode):
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap not installed")
    source, data, checkout = workspace(tmp_path)
    attach(source, data, checkout)
    script = (
        "from pathlib import Path; "
        "assert Path('data/input.txt').read_text() == 'dataset'; "
        "\ntry: Path('data/input.txt').write_text('corrupt')"
        "\nexcept OSError: pass"
        "\nelse: raise AssertionError('resource writable')"
    )
    # System Python remains visible inside diagnostics' masked home directory.
    argv = ["/usr/bin/python3", "-c", script]
    filter_stream = None
    if mode == "native":
        script += (f"\nfor path in {list(map(str, [source / 'train.py', data / 'input.txt']))!r}:"
                   "\n try: Path(path).write_text('corrupt')"
                   "\n except OSError: pass"
                   "\n else: raise AssertionError('original path writable')"
                   "\nPath('new-output').write_text('allowed')")
        argv[-1] = script
        command = isolated_argv(argv, output=checkout.parent, repo=checkout)
    elif mode == "inspection":
        command = inspection_argv(argv, repo=checkout, directory=checkout, environment={})
    else:
        from autosim.research.agent_runtime import _sandbox_argv
        command, filter_stream = _sandbox_argv(argv, output=checkout.parent,
                                              workspace=checkout, cwd=checkout)
    try:
        done = subprocess.run(command, cwd=checkout, capture_output=True, text=True,
                              timeout=15, pass_fds=((filter_stream.fileno(),)
                                                    if filter_stream else ()))
    finally:
        if filter_stream:
            filter_stream.close()
    assert done.returncode == 0, done.stderr
    assert (source / "train.py").read_text() == "original"
    assert (data / "input.txt").read_text() == "dataset"
