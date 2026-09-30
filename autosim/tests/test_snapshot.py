"""Undo, and the best state kept.

Two things become necessary the moment a run can change the benchmark's code. A change that
does not work has to leave nothing behind, or the next round runs on top of it. And what a run
reports has to be the best state it reached, not the last one -- a loop whose later rounds go
worse would otherwise report the worse number as its finding.
"""

from pathlib import Path

import pytest

from autosim.research import snapshot
from autosim.research.codepatch import Patch, apply_many


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("one\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("two\n", encoding="utf-8")
    return tmp_path


def test_restoring_reports_what_actually_moved(tmp_path):
    """Not everything in the snapshot: a restore that lists every file it looked at cannot be
    told from one that changed nothing, and the next decision depends on knowing which."""
    repo = _repo(tmp_path)
    states = snapshot.Snapshots(tmp_path / "snaps")
    states.capture(["src/a.py", "src/b.py"], repo=repo, name="before")
    (repo / "src" / "a.py").write_text("changed\n", encoding="utf-8")

    moved = states.restore("before", repo=repo)
    assert moved == ["src/a.py"]
    assert (repo / "src" / "a.py").read_text(encoding="utf-8") == "one\n"
    # Nothing left to do, and it says so.
    assert states.restore("before", repo=repo) == []


def test_a_file_that_was_absent_is_restored_to_absent(tmp_path):
    """A snapshot that silently omits a file cannot restore the state it claims to describe."""
    repo = _repo(tmp_path)
    states = snapshot.Snapshots(tmp_path / "snaps")
    states.capture(["src/a.py", "src/new.py"], repo=repo, name="before")
    (repo / "src" / "new.py").write_text("created by an idea\n", encoding="utf-8")
    assert states.restore("before", repo=repo) == ["src/new.py"]
    assert not (repo / "src" / "new.py").exists()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_snapshot_restore_refuses_missing_or_corrupt_blob_before_any_write(tmp_path, damage):
    repo = _repo(tmp_path)
    states = snapshot.Snapshots(tmp_path / "snaps")
    captured = states.capture(["src/a.py", "src/b.py"], repo=repo, name="before")
    (repo / "src" / "a.py").write_text("changed a\n", encoding="utf-8")
    (repo / "src" / "b.py").write_text("changed b\n", encoding="utf-8")
    blob = states.blobs / captured.files["src/b.py"]
    if damage == "missing":
        blob.unlink()
    else:
        blob.write_text("not the saved bytes", encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        states.restore("before", repo=repo)
    assert (repo / "src" / "a.py").read_text(encoding="utf-8") == "changed a\n"
    assert (repo / "src" / "b.py").read_text(encoding="utf-8") == "changed b\n"


def test_two_snapshots_share_the_bytes_they_share(tmp_path):
    """A loop that tries ten variations of one file should not hold ten copies of it to be able
    to go back."""
    repo = _repo(tmp_path)
    states = snapshot.Snapshots(tmp_path / "snaps")
    states.capture(["src/a.py"], repo=repo, name="one")
    (repo / "src" / "a.py").write_text("two\n", encoding="utf-8")
    states.capture(["src/a.py"], repo=repo, name="two")
    assert len(list((tmp_path / "snaps" / "blobs").iterdir())) == 2
    # And capturing the same state again does not add a third.
    states.capture(["src/a.py"], repo=repo, name="three")
    assert len(list((tmp_path / "snaps" / "blobs").iterdir())) == 2


def test_capture_repairs_a_corrupt_content_addressed_blob(tmp_path):
    repo = _repo(tmp_path)
    states = snapshot.Snapshots(tmp_path / "snaps")
    captured = states.capture(["src/a.py"], repo=repo, name="first")
    blob = states.blobs / captured.files["src/a.py"]
    blob.write_text("partial", encoding="utf-8")
    states.capture(["src/a.py"], repo=repo, name="second")
    assert blob.read_text(encoding="utf-8") == "one\n"


def test_the_best_advances_only_when_it_is_better(tmp_path):
    """A tag that advances on every capture makes the last attempt the answer."""
    repo = _repo(tmp_path)
    states = snapshot.Snapshots(tmp_path / "snaps")
    states.capture(["src/a.py"], repo=repo, name="first", score=0.31)
    assert states.advance("first", score=0.31) is True
    assert states.best().name == "first"

    states.capture(["src/a.py"], repo=repo, name="worse", score=0.12)
    assert states.advance("worse", score=0.12) is False
    assert states.best().name == "first", "the run's result became its last attempt"

    states.capture(["src/a.py"], repo=repo, name="better", score=0.42)
    assert states.advance("better", score=0.42) is True
    assert states.best().name == "better" and states.best().score == 0.42

    # A state that produced no number is not better than one that produced a number.
    states.capture(["src/a.py"], repo=repo, name="unmeasured", score=None)
    assert states.advance("unmeasured", score=None) is False
    assert states.best().name == "better"


def test_the_history_survives_being_reopened(tmp_path):
    repo = _repo(tmp_path)
    states = snapshot.Snapshots(tmp_path / "snaps")
    states.capture(["src/a.py"], repo=repo, name="one", score=0.2, why="the baseline")
    # Capturing records a state; advancing says it is the best one. Two acts, because a state
    # that has been taken is not thereby the answer -- and the failure this separates is a tag
    # that follows every capture and ends up pointing at whatever ran last.
    assert states.advance("one", score=0.2) is True
    reopened = snapshot.Snapshots(tmp_path / "snaps")
    assert [one.name for one in reopened.rows] == ["one"]
    assert reopened.best().name == "one" and reopened.best().why == "the baseline"
    assert reopened.history()[0]["score"] == 0.2


def test_a_snapshot_of_nothing_is_harmless(tmp_path):
    states = snapshot.Snapshots(tmp_path / "snaps")
    assert states.restore("never-taken", repo=tmp_path) == []
    assert states.best() is None
    assert states.advance("never-taken", score=0.5) is False


def test_snapshot_never_reads_or_restores_outside_the_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    outside = tmp_path / "outside.py"
    outside.write_text("unchanged", encoding="utf-8")
    states = snapshot.Snapshots(tmp_path / "snaps")
    with pytest.raises(ValueError, match="escapes repository"):
        states.capture(["../outside.py"], repo=repo, name="invalid")
    states.rows.append(snapshot.Snapshot(name="malicious", at="now",
                                         files={"../outside.py": ""}))
    with pytest.raises(ValueError, match="escapes repository"):
        states.restore("malicious", repo=repo)
    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_guarded_patch_rollback_handles_partial_multi_file_apply_and_repeats(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo = _repo(repo_root)
    states = snapshot.Snapshots(tmp_path / "snaps")
    patches = [Patch("src/a.py", "one", "changed_one"),
               Patch("src/b.py", "two", "changed_two")]
    states.capture([one.file for one in patches], repo=repo, name="before-candidate")
    apply_many(patches[:1], repo=repo)
    # A crash after one atomic file replacement leaves a mixture of old and new files.
    assert states.restore_patched("before-candidate", patches, repo=repo) == ["src/a.py"]
    assert (repo / "src/a.py").read_text(encoding="utf-8") == "one\n"
    assert (repo / "src/b.py").read_text(encoding="utf-8") == "two\n"
    assert states.restore_patched("before-candidate", patches, repo=repo) == []


def test_guarded_patch_rollback_refuses_unrelated_edits_before_any_write(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo = _repo(repo_root)
    states = snapshot.Snapshots(tmp_path / "snaps")
    patches = [Patch("src/a.py", "one", "changed_one"),
               Patch("src/b.py", "two", "changed_two")]
    states.capture([one.file for one in patches], repo=repo, name="before-candidate")
    apply_many(patches, repo=repo)
    (repo / "src/b.py").write_text("human edit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="diverged"):
        states.restore_patched("before-candidate", patches, repo=repo)
    assert (repo / "src/a.py").read_text(encoding="utf-8") == "changed_one\n"
    assert (repo / "src/b.py").read_text(encoding="utf-8") == "human edit\n"


def test_guarded_patch_rollback_refuses_symlink_and_invalid_snapshot_before_writing(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo = _repo(repo_root)
    states = snapshot.Snapshots(tmp_path / "snaps")
    patches = [Patch("src/a.py", "one", "changed_one"),
               Patch("src/b.py", "two", "changed_two")]
    captured = states.capture([one.file for one in patches], repo=repo,
                              name="before-candidate")
    apply_many(patches, repo=repo)
    outside = tmp_path / "outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    (repo / "src/b.py").unlink()
    (repo / "src/b.py").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        states.restore_patched("before-candidate", patches, repo=repo)
    assert (repo / "src/a.py").read_text(encoding="utf-8") == "changed_one\n"
    assert outside.read_text(encoding="utf-8") == "outside\n"

    (repo / "src/b.py").unlink()
    (repo / "src/b.py").write_text("changed_two\n", encoding="utf-8")
    (states.blobs / captured.files["src/b.py"]).write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        states.restore_patched("before-candidate", patches, repo=repo)
    assert (repo / "src/a.py").read_text(encoding="utf-8") == "changed_one\n"

    captured.files["src/b.py"] = "../../outside.py"
    with pytest.raises(ValueError, match="snapshot has no file bytes"):
        states.restore_patched("before-candidate", patches, repo=repo)
    assert outside.read_text(encoding="utf-8") == "outside\n"
