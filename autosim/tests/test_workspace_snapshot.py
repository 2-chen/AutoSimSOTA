"""A research patch must be able to change a copy without changing its source."""

import json
import stat
import subprocess

import pytest

from autosim.research.snapshot import Snapshots
from autosim.research.workspace_snapshot import (apply_state, capture_state, create,
                                                 existing, verify_state)


def test_copy_is_independent_and_has_source_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("value = 1\n", encoding="utf-8")
    (source / "sub").mkdir()
    (source / "sub" / "config.yaml").write_text("batch: 8\n", encoding="utf-8")
    (source / "train_link.py").symlink_to("train.py")
    destination = tmp_path / "run" / "checkout"
    manifest = create(source, destination, max_bytes=1024)
    assert manifest["files"] == 2 and manifest["symlinks"] == 1
    assert existing(destination, source=source)["source_tree_fingerprint"] == manifest[
        "source_tree_fingerprint"]
    (destination / "train.py").write_text("value = 2\n", encoding="utf-8")
    assert (source / "train.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (destination / "train_link.py").read_text(encoding="utf-8") == "value = 2\n"
    saved = json.loads((destination.parent / "workspace_snapshot.json").read_text())
    assert saved["bytes"] > 0
    assert any(entry[0] == "symlink" for entry in saved["source_entries"])


def test_snapshot_refuses_external_symlink_and_does_not_leave_partial_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "escape").symlink_to(tmp_path)
    destination = tmp_path / "run" / "checkout"
    with pytest.raises(ValueError, match="external directory symlink"):
        create(source, destination)
    assert not destination.exists()


def test_copy_rebases_absolute_internal_symlink_into_the_isolated_checkout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "controller.py").write_text("value = 1\n", encoding="utf-8")
    (source / "controller_link.py").symlink_to(source / "controller.py")
    destination = tmp_path / "run" / "checkout"
    manifest = create(source, destination)

    link = destination / "controller_link.py"
    assert link.resolve().is_relative_to(destination)
    assert link.read_text() == "value = 1\n"
    recorded = next(row for row in manifest["source_entries"]
                    if row[0] == "symlink")
    assert recorded[2] == link.readlink().as_posix()


def test_snapshot_refuses_excess_size_and_recursive_target(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"12345")
    with pytest.raises(ValueError, match="byte limit"):
        create(source, tmp_path / "run" / "checkout", max_bytes=4)
    with pytest.raises(ValueError, match="disjoint"):
        create(source, source / "run" / "checkout")


def test_resume_cannot_rebind_a_copy_to_another_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.py").write_text("x = 1\n", encoding="utf-8")
    destination = tmp_path / "run" / "checkout"
    create(source, destination)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="does not match"):
        existing(destination, source=other)


def test_tracked_copy_keeps_current_source_edit_and_omits_untracked_asset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    script = source / "train.py"
    script.write_text("version = 1\n")
    subprocess.run(["git", "-C", str(source), "add", "train.py"], check=True)
    script.write_text("version = 2\n")
    (source / "large_dataset.bin").write_bytes(b"x" * 128)
    destination = tmp_path / "run" / "checkout"
    manifest = create(source, destination, tracked_only=True, max_bytes=32)
    assert (destination / "train.py").read_text() == "version = 2\n"
    assert not (destination / "large_dataset.bin").exists()
    assert manifest["copy_mode"] == "tracked_worktree"
    assert manifest["untracked_assets_omitted"] is True
    assert manifest["bytes"] < 32


def test_source_state_freezes_and_verifies_the_exact_overlay(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "controller.py").write_text("version = 1\n", encoding="utf-8")
    (source / "remove.py").write_text("obsolete = True\n", encoding="utf-8")
    destination = tmp_path / "run" / "checkout"
    manifest = create(source, destination)
    snapshots = Snapshots(tmp_path / "run" / "research" / "snapshots")

    (destination / "controller.py").write_text("version = 2\n", encoding="utf-8")
    (destination / "remove.py").unlink()
    (destination / "new.py").write_text("new = True\n", encoding="utf-8")
    state = capture_state(destination, manifest, snapshots, name="source-state-round-1")

    assert set(state["overlay_files"]) == {"controller.py", "remove.py", "new.py"}
    assert verify_state(destination, manifest, state, snapshots)
    assert snapshots.restore(state["source_snapshot"], repo=destination) == []

    (destination / "controller.py").write_text("version = 3\n", encoding="utf-8")
    assert not verify_state(destination, manifest, state, snapshots)


def test_source_state_refuses_changed_symlink_and_does_not_claim_restorability(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.py").write_text("one\n", encoding="utf-8")
    (source / "link.py").symlink_to("one.py")
    destination = tmp_path / "run" / "checkout"
    manifest = create(source, destination)
    snapshots = Snapshots(tmp_path / "run" / "research" / "snapshots")
    (destination / "link.py").unlink()
    (destination / "link.py").symlink_to("missing.py")

    with pytest.raises(ValueError, match="cannot preserve symlink change"):
        capture_state(destination, manifest, snapshots, name="source-state-baseline")


def test_source_state_reconstructs_file_modes_and_directory_additions_and_deletions(tmp_path):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / "controller.py").write_text("version = 1\n", encoding="utf-8")
    (source / "obsolete").mkdir()
    (source / "obsolete" / "gone.py").write_text("gone\n", encoding="utf-8")
    destination = tmp_path / "run" / "checkout"
    manifest = create(source, destination)
    snapshots = Snapshots(tmp_path / "run" / "research" / "snapshots")

    controller = destination / "scripts" / "controller.py"
    controller.chmod(0o755)
    (destination / "scripts").chmod(0o700)
    (destination / "obsolete" / "gone.py").unlink()
    (destination / "obsolete").rmdir()
    (destination / "new" / "nested").mkdir(parents=True)
    (destination / "new" / "nested" / "candidate.py").write_text(
        "version = 2\n", encoding="utf-8")
    (destination / "new" / "nested" / "candidate.py").chmod(0o755)
    state = capture_state(destination, manifest, snapshots, name="source-state-round-1")

    fresh = tmp_path / "fresh" / "checkout"
    fresh_manifest = create(source, fresh)
    apply_state(fresh, state, snapshots)
    assert verify_state(fresh, fresh_manifest, state, snapshots)
    assert stat.S_IMODE((fresh / "scripts" / "controller.py").stat().st_mode) == 0o755
    assert stat.S_IMODE((fresh / "scripts").stat().st_mode) == 0o700
    assert stat.S_IMODE((fresh / "new" / "nested" / "candidate.py").stat().st_mode) == 0o755
    assert not (fresh / "obsolete").exists()
