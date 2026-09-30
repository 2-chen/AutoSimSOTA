"""Whose data a survey reports, when several benchmarks are installed side by side.

The parent directory is searched because a benchmark's assets are routinely kept beside its
code rather than inside it. On any machine with more than one benchmark installed -- which is
the normal case, and this one -- "beside it" is where the others are, and the survey reported
a union. Measured on four checkouts in one directory: `robomimic`, which ships 29 of its own
demonstrations, was shown twenty-five files belonging to LIBERO, every one found under the
shared parent, and the contract reader then read another benchmark's HDF5 as this one's.

None of this is about a benchmark. It is about a budget being spent by the wrong directory.
"""

from pathlib import Path

from autosim.research import survey as module


def _plant(root: Path, name: str, count: int, *, megabytes: int) -> list[Path]:
    """A dataset directory of `count` files, each `megabytes` big."""
    directory = root / "datasets" / name
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for index in range(count):
        path = directory / f"episode_{index}.hdf5"
        path.write_bytes(b"\0" * (megabytes * 1_000_000))
        out.append(path)
    return out


def _checkout(parent: Path, name: str) -> Path:
    repo = parent / name
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    return repo


def test_a_checkout_is_shown_its_own_assets_before_the_neighbours(tmp_path):
    """The parent is searched for assets. It should not be searched *instead of* the
    checkout, and a neighbour with bigger files should not be able to fill the list."""
    parent = tmp_path / "benchmarks"
    mine = _checkout(parent, "alpha")
    _checkout(parent, "beta")
    _plant(parent, "beta", 12, megabytes=3)          # the neighbour's, and larger
    own = _plant(mine, "alpha", 4, megabytes=1)      # this one's, and smaller

    rows = module._datasets(module._asset_roots(mine, ()), max_depth=6, max_candidates=40)
    first_four = {Path(row["path"]) for row in rows[:4]}
    assert first_four & set(own), "none of this checkout's own assets reached the reader"


def test_a_neighbour_cannot_spend_the_whole_allowance(tmp_path):
    """One directory filling the budget is what the per-subtree cap exists for, and under the
    shared parent the top-level directory *is* every benchmark at once."""
    parent = tmp_path / "benchmarks"
    mine = _checkout(parent, "alpha")
    _checkout(parent, "beta")
    _plant(parent, "beta", 40, megabytes=3)
    _plant(mine, "alpha", 6, megabytes=1)

    rows = module._datasets(module._asset_roots(mine, ()), max_depth=6, max_candidates=40)
    from_alpha = [row for row in rows if "/alpha/" in row["path"]]
    assert from_alpha, "the neighbour took every slot"


def test_the_reader_sees_more_than_one_directory(tmp_path):
    """`summarise` shows the first ten, so the order decides what the reader is told. A list
    that is ten of one subtree, in a directory holding four, is a list about one of them."""
    parent = tmp_path / "benchmarks"
    mine = _checkout(parent, "alpha")
    for name in ("beta", "gamma", "delta"):
        _checkout(parent, name)
        _plant(parent, name, 8, megabytes=2)
    _plant(mine, "alpha", 8, megabytes=1)

    rows = module._datasets(module._asset_roots(mine, ()), max_depth=6, max_candidates=40)
    tops = {row["path"].split("/datasets/")[1].split("/")[0] for row in rows[:4]
            if "/datasets/" in row["path"]}
    assert len(tops) > 1, f"the first four all came from {tops}"


def test_what_each_asset_was_found_under_travels_with_it(tmp_path):
    """The field that makes the difference visible downstream. A reader that cannot tell an
    asset inside the checkout from one beside it cannot tell whose it is."""
    parent = tmp_path / "benchmarks"
    mine = _checkout(parent, "alpha")
    _checkout(parent, "beta")
    _plant(parent, "beta", 4, megabytes=2)

    rows = module._datasets(module._asset_roots(mine, ()), max_depth=6, max_candidates=40)
    assert rows
    assert all("found_under" in row for row in rows)
