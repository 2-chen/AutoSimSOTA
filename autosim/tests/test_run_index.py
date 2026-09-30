"""The offline run index treats a nested research record as part of its parent run."""

from autosim.research.run_index import find_runs, runs_index


def test_one_prepared_run_is_indexed_once(tmp_path):
    root = tmp_path / "runs"
    parent = root / "one"
    child = parent / "research" / "derived"
    child.mkdir(parents=True)
    (parent / "RUN.md").write_text("# run\n", encoding="utf-8")
    (child / "RUN.md").write_text("# nested\n", encoding="utf-8")
    assert find_runs(root) == [parent]
    assert [row["directory"] for row in runs_index(root)["runs"]] == ["one"]
