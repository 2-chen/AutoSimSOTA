from pathlib import Path

import pytest

from autosim.research import agent_tasks
from autosim.research.common import atomic_json, read_json
from autosim.research.native_jobs import source_identity
from autosim.research.budget import RunBudget


def checkout(tmp_path):
    from tests.test_scheduling import inventory
    repo = tmp_path / "checkout"; repo.mkdir()
    inventory(tmp_path, repo)
    dependency = repo / "DownloadedDependency/library.py"
    dependency.parent.mkdir()
    dependency.write_text("NATIVE_API_VERSION = 1\n")
    return repo, dependency


def test_reader_can_scope_public_source_materialized_after_initial_copy(tmp_path):
    repo, dependency = checkout(tmp_path)
    root, copy = agent_tasks.snapshot(tmp_path, repo, "a"*32,
        source_paths=["DownloadedDependency/library.py"])
    assert (copy / "DownloadedDependency/library.py").read_text() == dependency.read_text()
    manifest = read_json(root / "workspace_snapshot.json")
    assert manifest["reader_source_files"] == ["DownloadedDependency/library.py"]
    assert manifest["reader_scope_identity"]
    assert not (copy / "train.py").exists()


def test_materialized_scope_does_not_expose_external_links_or_credentials(tmp_path):
    repo, _ = checkout(tmp_path)
    (repo / "outside.py").symlink_to(tmp_path / "secret.py")
    with pytest.raises(ValueError, match="unsafe"):
        agent_tasks.snapshot(tmp_path, repo, "b"*32, source_paths=["outside.py"])
    (repo / ".env").write_text("DO_NOT_SEND")
    with pytest.raises(ValueError, match="unsafe"):
        agent_tasks.snapshot(tmp_path, repo, "c"*32, source_paths=[".env"])
    (repo / "DownloadedDependency/weights.pt").write_bytes(b"not source")
    with pytest.raises(ValueError, match="no public files"):
        agent_tasks.snapshot(tmp_path, repo, "d"*32, source_paths=["DownloadedDependency/weights.pt"])


def test_completed_reader_is_stale_if_downloaded_dependency_changes(tmp_path):
    repo, dependency = checkout(tmp_path)
    root, _ = agent_tasks.snapshot(tmp_path, repo, "e"*32,
        source_paths=["DownloadedDependency/library.py"])
    manifest = read_json(root / "workspace_snapshot.json")
    atomic_json(tmp_path / "agent_tasks" / ("e"*32) / "task.json", {
        "id": "e"*32, "status": "completed", "source_identity": source_identity(tmp_path, repo),
        "reader_scope_identity": manifest["reader_scope_identity"],
        "reader_source_files": manifest["reader_source_files"]})
    dependency.write_text("NATIVE_API_VERSION = 2\n")
    assert agent_tasks.collect(tmp_path, repo)[0]["stale"] is True


@pytest.mark.parametrize("scope,expected_cycles", [("preparation", 2), ("research", 1)])
def test_refresh_continues_only_reconciled_preparation_not_research(tmp_path, monkeypatch, scope, expected_cycles):
    from tests.test_main_agent import make
    prep = make(tmp_path)
    RunBudget(prep.output, wall_seconds=30)
    prep.budget = RunBudget.existing(prep.output)
    calls = []
    def cycle(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            prep.steps.append({"step": "reconcile_interrupted_action", "resume_scope": scope})
            return {"status": "paused"}
        prep.steps.append({"step": "stop", "outcome": "done"})
        return {"status": "completed"}
    monkeypatch.setattr(prep, "_run_cycle", cycle)
    report = prep.run(max_steps=1, max_relaunch=1)
    assert len(calls) == expected_cycles
    assert report["status"] == ("completed" if scope == "preparation" else "paused")
