"""Changing the benchmark's own source, safely enough to be worth doing.

Most real improvements are here and not in a configuration value -- AutoSOTA puts 51% of its
improvements in algorithmic change against 33% in tuning -- and this loop could not reach it
at all: a fault in the benchmark's own code could be diagnosed correctly and never fixed.
RoboTwin produced three such diagnoses in one run and stopped on each.

What makes it safe is not this module. The red lines refuse it before it is applied and the
snapshot puts it back if it fails. What this owes them is a patch that is what it says it is:
one place, exactly, and reversible exactly.
"""

from pathlib import Path

import pytest

from autosim.research import codepatch


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "loader.py").write_text(
        "PATH = 'a/b'\n\ndef load():\n    return open(PATH)\n", encoding="utf-8")
    return tmp_path


def test_the_text_to_replace_has_to_occur_exactly_once(tmp_path):
    """Twice, and the change would be applied to one of two identical places and the run would
    not know which. The fix is a longer `find`, and saying so is cheaper than discovering it by
    running."""
    repo = _repo(tmp_path)
    (repo / "lib" / "twice.py").write_text("x = 1\nx = 1\n", encoding="utf-8")

    twice = codepatch.Patch(file="lib/twice.py", find="x = 1", replace="x = 2")
    problems = codepatch.check(twice, repo=repo)
    assert len(problems) == 1 and "occurs 2 times" in problems[0] and "unique" in problems[0]

    absent = codepatch.Patch(file="lib/loader.py", find="PATH = 'q'", replace="PATH = 'z'")
    assert "does not occur" in codepatch.check(absent, repo=repo)[0]

    missing = codepatch.Patch(file="lib/nope.py", find="a", replace="b")
    assert "is not a file in this checkout" in codepatch.check(missing, repo=repo)[0]

    itself = codepatch.Patch(file="lib/loader.py", find="PATH = 'a/b'", replace="PATH = 'a/b'")
    assert "with itself" in codepatch.check(itself, repo=repo)[0]


def test_a_python_file_that_would_no_longer_parse_is_refused_before_it_is_written(tmp_path):
    """Every fault here would otherwise be found by running, which is the more expensive way to
    find all of them."""
    repo = _repo(tmp_path)
    bad = codepatch.Patch(file="lib/loader.py", find="return open(PATH)",
                          replace="return open(PATH")
    problems = codepatch.check(bad, repo=repo)
    assert len(problems) == 1 and "no longer parse" in problems[0]
    # And the file is untouched, because check refuses before anything is written.
    assert "return open(PATH)" in (repo / "lib" / "loader.py").read_text(encoding="utf-8")


def test_applying_returns_the_text_that_was_there_and_reverting_is_exact(tmp_path):
    """Not replacing in the other direction: a `replace` that occurs elsewhere in the file
    would be undone in the wrong place."""
    repo = _repo(tmp_path)
    patch = codepatch.Patch(
        file="lib/loader.py",
        find="PATH = 'a/b'",
        replace="PATH = str(Path(__file__).parent.parent / 'data')  # the loader's own dir")
    before = codepatch.apply(patch, repo=repo)
    after = (repo / "lib" / "loader.py").read_text(encoding="utf-8")
    assert "the loader's own dir" in after and "a/b" not in after

    codepatch.revert(patch, before, repo=repo)
    assert (repo / "lib" / "loader.py").read_text(encoding="utf-8") == before


def test_reverting_refuses_to_overwrite_a_later_source_edit(tmp_path):
    repo = _repo(tmp_path)
    patch = codepatch.Patch(file="lib/loader.py", find="PATH = 'a/b'",
                            replace="PATH = 'c/d'")
    before = codepatch.apply(patch, repo=repo)
    target = repo / "lib" / "loader.py"
    target.write_text(target.read_text(encoding="utf-8") + "# human edit\n",
                      encoding="utf-8")

    with pytest.raises(ValueError, match="diverged"):
        codepatch.revert(patch, before, repo=repo)
    assert "# human edit" in target.read_text(encoding="utf-8")


def test_a_patch_that_was_not_checked_cannot_be_applied_by_accident(tmp_path):
    """A caller that applies without checking has decided the faults do not matter, and it
    should have to say so by catching rather than by not looking."""
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="does not occur"):
        codepatch.apply(codepatch.Patch(file="lib/loader.py", find="nope", replace="y"),
                        repo=repo)


def test_a_rewrite_dressed_as_a_patch_is_refused(tmp_path):
    """The reason for the size limit is that the review a patch gets is reading a diff."""
    repo = _repo(tmp_path)
    huge = codepatch.Patch(file="lib/loader.py", find="x" * (codepatch.MAX_FIND + 1),
                           replace="y")
    assert "a rewrite rather than a patch" in codepatch.check(huge, repo=repo)[0]


def test_patch_cannot_escape_checkout_even_through_symlink(tmp_path):
    (tmp_path / "repo").mkdir()
    repo = _repo(tmp_path / "repo")
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    (repo / "linked.py").symlink_to(outside)
    for name in ("../outside.py", str(outside), "linked.py"):
        patch = codepatch.Patch(file=name, find="value = 1", replace="value = 2")
        assert "escapes this checkout" in codepatch.check(patch, repo=repo)[0]
    assert outside.read_text(encoding="utf-8") == "value = 1\n"


def test_a_change_that_is_not_a_patch_yields_no_patch():
    assert codepatch.Patch.from_change(None) is None
    assert codepatch.Patch.from_change({"file": "a.py"}) is None
    assert codepatch.Patch.from_change({"file": "", "find": "a", "replace": "b"}) is None
    made = codepatch.Patch.from_change({"file": "a.py", "find": "a", "replace": "b"})
    assert made.file == "a.py" and made.as_dict()["find"] == "a"


def test_the_diff_is_what_a_reader_reviews(tmp_path):
    patch = codepatch.Patch(file="lib/loader.py", find="PATH = 'a/b'", replace="PATH = 'c/d'")
    rendered = codepatch.diff(patch)
    assert rendered.startswith("--- lib/loader.py")
    assert "- PATH = 'a/b'" in rendered and "+ PATH = 'c/d'" in rendered


def test_multi_file_patch_is_checked_and_applied_as_one_transaction(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    other = repo / "lib" / "policy.py"
    other.write_text("GAIN = 1\n", encoding="utf-8")
    patches = codepatch.patches_from_change({"patches": [
        {"file": "lib/loader.py", "find": "PATH = 'a/b'", "replace": "PATH = 'c/d'"},
        {"file": "lib/policy.py", "find": "GAIN = 1", "replace": "GAIN = 2"}]})
    assert len(patches) == 2
    before = codepatch.apply_many(patches, repo=repo)
    assert "c/d" in (repo / "lib" / "loader.py").read_text(encoding="utf-8")
    assert "GAIN = 2" in other.read_text(encoding="utf-8")
    for patch in patches:
        codepatch.revert(patch, before[patch.file], repo=repo)

    original = codepatch.apply

    def fail_second(patch, *, repo):
        if patch.file == "lib/policy.py":
            raise OSError("disk full")
        return original(patch, repo=repo)

    monkeypatch.setattr(codepatch, "apply", fail_second)
    with pytest.raises(OSError, match="disk full"):
        codepatch.apply_many(patches, repo=repo)
    assert "a/b" in (repo / "lib" / "loader.py").read_text(encoding="utf-8")
    assert "GAIN = 1" in other.read_text(encoding="utf-8")
