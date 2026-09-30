from autosim.research.runtime import Runtime


def test_probe_uses_standalone_repository_when_clean_copy_is_absent(tmp_path):
    repo = tmp_path / "RoboSynChallenge"
    repo.mkdir()
    runtime = Runtime(tmp_path, tmp_path / "probe")
    assert runtime.eval_repo == repo
    clean = tmp_path / "RoboSynChallenge_eval_clean"
    clean.mkdir()
    assert runtime.eval_repo == clean


def test_probe_preserves_explicit_evaluation_repository(tmp_path):
    explicit = tmp_path / "exported_repo"
    runtime = Runtime(tmp_path, tmp_path / "probe", eval_repo_path=explicit)
    assert runtime.eval_repo == explicit
