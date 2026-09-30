"""What the survey is allowed to miss, and the one thing it must not.

The survey decides what the scout can know. Everything downstream -- the declaration, the
optimisation space, which stages exist -- is a conclusion drawn from this report, so a
directory the survey never opened is a fact the system cannot have.

**The failure this file is about happened four times.** A walk with a finite budget spent it
all inside whichever large directory sorted first, and a capital letter sorts before every
lower-case one, so that was always a vendored library: on RoboTwin, `XPolicyLab/`, a policy
library shipped inside the benchmark's checkout. 119 of the 120 signal sources the scan
reached were inside it; `envs/`, `script/` and `task_config/` were never opened. The
declaration that came out had an **empty optimisation space**, and nothing in it said why.

It was corrected one walk at a time -- the file listing, then the dataset ranking, then the
signal scan -- each time after the next walk turned out to have the same fault. The tests
below are written against the property rather than against any one of those walks, because a
test that named the walk that broke last time is how the fourth one survived.
"""

import json
from pathlib import Path

from autosim.research.common import sanitize_model_text
from autosim.research import survey as sv


def _checkout(root: Path, *, vendored_files: int = 400) -> Path:
    """A benchmark whose entry points are shallow and which vendors a large library.

    Named with a capital letter, because that is what made this fail: `XPolicyLab` sorts
    before `envs/` and the walk was alphabetical.
    """
    (root / "XPolicyLab").mkdir(parents=True)
    (root / "envs").mkdir()
    (root / "script").mkdir()
    (root / "task_config").mkdir()
    for index in range(vendored_files):
        # A real vendored policy library mentions these too, so it produces hits -- which is
        # what makes the test about whether the budget was *spread* rather than *redirected*.
        (root / "XPolicyLab" / f"module_{index:04d}.py").write_text(
            "# a vendored policy library\noptimizer = None\ncheckpoint = None\n",
            encoding="utf-8")
    (root / "envs" / "beat_block_hammer.py").write_text(
        "# collect episodes\ndef evaluate(policy):\n    return is_success()\n", encoding="utf-8")
    (root / "script" / "collect_data.py").write_text(
        "# collect_data: save_episode to_hdf5\n", encoding="utf-8")
    (root / "task_config" / "beat.yml").write_text("task: beat\n", encoding="utf-8")
    (root / "README.md").write_text("# A benchmark\n", encoding="utf-8")
    return root


def test_a_vendored_library_does_not_consume_the_budget(tmp_path):
    """The property the whole file exists for: the benchmark's own shallow directories are
    reached, whatever the checkout also contains."""
    repo = _checkout(tmp_path / "bench")
    report = sv.survey(repo, max_scan_files=40)
    reached = {row["path"] for row in report["signal_sources"]}
    assert any(one.startswith("envs/") for one in reached), reached
    assert any(one.startswith("script/") for one in reached), reached
    # And the vendored library is still read: it is a fact about the checkout, not something
    # to hide. What changed is that it no longer gets the whole budget -- all four top-level
    # directories are read, in proportion to how many of them there are.
    assert any(one.startswith("XPolicyLab/") for one in reached), reached


def test_the_directory_list_puts_the_shallow_ones_first(tmp_path):
    """Alphabetical order here is the same accident as alphabetical order in the walk:
    `XPolicyLab/...` fills the front of the list and `envs/` and `task_config/` fall off the
    end of the cap, so a reader looking for the benchmark's own directories finds the
    vendored library's and no sign that anything is missing."""
    repo = _checkout(tmp_path / "bench")
    for index in range(300):
        (repo / "XPolicyLab" / f"deep_{index}").mkdir()
    report = sv.survey(repo)
    listed = report["tree"]["directories"]
    own = [one for one in listed if not one.startswith("XPolicyLab")]
    assert {"envs", "script", "task_config"} <= set(own), own[:20]
    # Depth order, so the checkout's own top-level directories come before anything nested.
    assert listed.index("envs") < listed.index("XPolicyLab/deep_0")


def test_the_report_says_how_many_directories_it_found(tmp_path):
    """A list that was cut to four hundred and a list that happens to hold four hundred read
    the same. The count makes the difference visible."""
    repo = _checkout(tmp_path / "bench", vendored_files=5)
    report = sv.survey(repo)
    assert report["tree"]["directory_count"] >= len(report["tree"]["directories"])
    assert report["tree"]["directory_count"] >= 4


def test_the_walk_is_breadth_first_and_skips_what_it_always_skipped(tmp_path):
    """The helper directly: shallow before deep, and `.git`-style caches still out."""
    root = tmp_path / "tree"
    (root / "shallow").mkdir(parents=True)
    (root / "deep" / "deeper").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "shallow" / "a.txt").write_text("x", encoding="utf-8")
    (root / "deep" / "deeper" / "b.txt").write_text("x", encoding="utf-8")
    (root / ".git" / "config").write_text("x", encoding="utf-8")
    seen = [(directory.relative_to(root).as_posix(), files)
            for directory, _, files in sv._walk_by_depth(root)]
    depths = [0 if one == "." else one.count("/") + 1 for one, _ in seen]
    assert depths == sorted(depths), seen
    assert all(one != ".git" for one, _ in seen)
    assert ("shallow", ["a.txt"]) in seen


def test_a_directory_that_cannot_be_read_does_not_end_the_walk(tmp_path, monkeypatch):
    """A survey that dies on one permission error reports nothing about the rest of the
    checkout, which is the worst possible moment to report nothing."""
    repo = _checkout(tmp_path / "bench", vendored_files=2)
    real = sv.os.scandir

    def refuse(path):
        if Path(path).name == "XPolicyLab":
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(sv.os, "scandir", refuse)
    report = sv.survey(repo, max_scan_files=40)
    reached = {row["path"] for row in report["signal_sources"]}
    assert any(one.startswith("envs/") for one in reached), reached


def test_a_depth_limit_is_still_honoured(tmp_path):
    """The helper replaces `os.walk`'s depth handling, so it has to keep it."""
    root = tmp_path / "tree"
    (root / "a" / "b" / "c").mkdir(parents=True)
    (root / "a" / "b" / "c" / "deep.txt").write_text("x", encoding="utf-8")
    seen = {directory.relative_to(root).as_posix()
            for directory, _, _ in sv._walk_by_depth(root, max_depth=1)}
    assert "a" in seen and "a/b" not in seen


def test_scout_json_preview_withholds_record_values_and_array_counts(tmp_path):
    """A file selected by LLM triage may be a trajectory, so JSON values stay local."""
    path = tmp_path / "trajectory.json"
    path.write_text(
        '{"episodes":[{"episode_id":"PRIVATE_EPISODE_ID",'
        '"observation":[1.234567,9.876543],"success":true}],'
        '"credential":"PRIVATE_API_SECRET"}', encoding="utf-8")

    preview = sv._peek(str(path), repo=tmp_path)

    rendered = str(preview)
    assert preview["values_redacted"] is True
    assert preview["structure"]["fields"]["episodes"] == {
        "type": "array", "item_shapes": [{"type": "object", "fields": {
            "episode_id": {"type": "string"},
            "observation": {"type": "array", "item_shapes": [{"type": "number"}]},
            "success": {"type": "boolean"}}}]}
    assert "PRIVATE_EPISODE_ID" not in rendered
    assert "PRIVATE_API_SECRET" not in rendered
    assert "1.234567" not in rendered
    assert "\"length\"" not in rendered


def test_scout_malformed_json_fails_closed_without_raw_fallback(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"episode":"PRIVATE_SAMPLE", bad}', encoding="utf-8")

    preview = sv._peek(str(path), repo=tmp_path)

    assert preview["readable"] is False
    assert "PRIVATE_SAMPLE" not in str(preview)
    assert "raw contents withheld" in preview["why"]


def test_llm_file_peek_refuses_paths_outside_checkout_and_binary_artifacts(tmp_path):
    repo = tmp_path / "benchmark"
    repo.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("PRIVATE_OUTSIDE_CONTENT", encoding="utf-8")
    weights = repo / "checkpoints" / "policy.pt"
    weights.parent.mkdir()
    weights.write_text("PRIVATE_CHECKPOINT_CONTENT", encoding="utf-8")

    escaped = sv._peek(str(outside), repo=repo)
    checkpoint = sv._peek(str(weights), repo=repo)

    assert escaped["readable"] is False
    assert "PRIVATE_OUTSIDE_CONTENT" not in str(escaped)
    assert escaped["path"] == "[outside-checkout-path]"
    assert "exists" not in escaped
    assert checkpoint["readable"] is False
    assert "PRIVATE_CHECKPOINT_CONTENT" not in str(checkpoint)
    assert checkpoint["path"] == "[private-artifact]"


def test_llm_file_peek_fails_closed_on_binary_disguised_as_source(tmp_path):
    repo = tmp_path / "benchmark"
    repo.mkdir()
    source = repo / "scripts" / "fixture.py"
    source.parent.mkdir()
    source.write_bytes(b"\x00PRIVATE_BINARY_TRAJECTORY\xff")

    preview = sv._peek(str(source), repo=repo)

    assert preview["readable"] is False
    assert "PRIVATE_BINARY_TRAJECTORY" not in str(preview)
    assert "binary" in preview["why"]


def test_llm_file_peek_redacts_credentials_and_machine_local_paths(tmp_path):
    repo = tmp_path / "benchmark"
    source = repo / "scripts" / "inspect.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'DEEPSEEK_API_KEY = "PRIVATE_API_KEY_VALUE"\n'
        'DATA_ROOT = "/home/private-user/private-demos"\n', encoding="utf-8")

    preview = sv._peek(str(source), repo=repo)

    assert preview["readable"] is True
    assert "PRIVATE_API_KEY_VALUE" not in preview["head"]
    assert "/home/private-user" not in preview["head"]
    assert "PRIVATE_PATH" not in preview["head"]
    assert "[REDACTED]" in preview["head"]


def test_model_text_sanitizer_handles_quoted_json_keys_and_local_path_schemes():
    text = ('{"DEEPSEEK_API_KEY": "PRIVATE_JSON_KEY", '
            '"data_root": "file:///mnt/private/demonstrations", '
            '"cache": "\\\\private-host\\share\\secret"}')

    safe = sanitize_model_text(text)

    assert "PRIVATE_JSON_KEY" not in safe
    assert "file:///mnt/private/demonstrations" not in safe
    assert "\\\\private-host\\share" not in safe


def test_model_survey_projection_withholds_external_paths_and_artifact_values(tmp_path):
    repo = tmp_path / "benchmark"
    repo.mkdir()
    external = tmp_path / "private-storage" / "demo_123.hdf5"
    checkpoint = tmp_path / "private-storage" / "secret_policy.pt"
    report = {
        "repo": str(repo),
        "project_manifests": {
            "setup.py": ('DEEPSEEK_API_KEY = "PRIVATE_MANIFEST_SECRET"\n'
                         'DATA_ROOT = "' + str(tmp_path / "private-storage") + '"\n')},
        "tree": {"extensions": {".py": 1}, "truncated": False,
                 "seen_files": ["train.py", "demos/episode_123.json", str(external)],
                 "largest_files": [{"path": "train.py", "bytes": 12},
                                   {"path": str(external), "bytes": 4096}]},
        "signal_totals": {"training": 1},
        "signal_sources": [{"path": "train.py", "signals": ["training"]},
                           {"path": str(external), "signals": ["collection"]}],
        "datasets": [{"path": str(external), "suffix": ".hdf5", "bytes": 4096}],
        "model_artifacts": [{"path": str(checkpoint), "suffix": ".pt", "bytes": 99}],
        "asset_provenance": {"inside_repo": 0, "outside_repo": 2,
                             "repository_mentions": {str(external): ["PRIVATE_ATTR_VALUE"]}},
        "structured_roots": [{"directory": str(external.parent),
                               "manifests": ["episodes.jsonl"]}],
        "dataset_structure": [{"path": str(external), "suffix": ".hdf5", "bytes": 4096,
                               "root_attrs": {"token": "PRIVATE_ATTR_VALUE"},
                               "groups": {"demo_123": {
                                   "attrs": {"source_path": str(external),
                                             "credential": "PRIVATE_HDF5_SECRET"},
                                   "count": 3, "examples": ["episode_123"],
                                   "records": [{"episode_id": "PRIVATE_EPISODE_ID",
                                                "observation": [1.2345]}]}}}],
        "recorded_provenance": {"recorded_paths": [str(external)],
                                "resolve_inside_repo": []},
        "searched_roots": [str(repo), str(external.parent)],
        "budget": {"max_depth": 4, "max_entries": 100,
                   "max_scan_files": 20, "max_bytes_per_file": 1000},
    }

    safe = sv.summarise_for_model(report)
    rendered = json.dumps(safe, ensure_ascii=False)

    for private in (str(tmp_path), "demo_123.hdf5", "secret_policy.pt",
                    "PRIVATE_MANIFEST_SECRET", "PRIVATE_ATTR_VALUE",
                    "PRIVATE_HDF5_SECRET", "PRIVATE_EPISODE_ID", "episode_123"):
        assert private not in rendered
    assert "train.py" in rendered
    assert "record_shapes" in rendered or "groups" in rendered
