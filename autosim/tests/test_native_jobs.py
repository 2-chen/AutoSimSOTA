import json
import shutil
import sys
import time

import pytest

from autosim.research.budget import RunBudget
from autosim.research.native_jobs import adjust, cancel, source_identity, status, submit
from autosim.research.common import digest, object_digest


def test_job_source_identity_tracks_existing_code_not_new_training_output(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    source = repo / "train.py"
    source.write_text("print('train')\n", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    (output / "derived_stages.json").write_text("{}", encoding="utf-8")
    entries = [["file", "train.py", source.stat().st_size, digest(source), 0o644]]
    (output / "workspace_snapshot.json").write_text(json.dumps({
        "schema_version": 2, "destination": str(repo),
        "source_entries": entries, "source_tree_fingerprint": object_digest(entries)}),
        encoding="utf-8")
    first = source_identity(output, repo)
    (repo / "checkpoint.pt").write_bytes(b"new training product")
    assert source_identity(output, repo) == first
    source.write_text("print('changed')\n", encoding="utf-8")
    assert source_identity(output, repo) != first


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="native isolation unavailable")
def test_detached_native_job_can_be_adjusted_and_cancelled(tmp_path):
    repo = tmp_path / "toy_sim"
    repo.mkdir()
    (repo / "README.md").write_text("synthetic trainer", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    RunBudget(output, wall_seconds=12)
    (output / "environment.json").write_text(json.dumps({
        "verdict": {"passed": True}, "interpreter": sys.executable}), encoding="utf-8")
    code = "import time; print('epoch=1', flush=True); time.sleep(8)"
    source = ("def stage_argv_train(i):\n"
              f"    return [i['python'], '-c', {code!r}]\n")
    (output / "derived_stages.json").write_text(json.dumps({
        "train": {"source": source, "parameters": {}}}), encoding="utf-8")
    (output / "execution.json").write_text(json.dumps({"stages": {
        "train": {"available": True, "entrypoint": "train.py",
                  "invocation": "python train.py", "artifact": ""}}}), encoding="utf-8")
    scout = output / "scouting" / "toy_sim_prepared"
    scout.mkdir(parents=True)
    declaration = {
        "benchmark": "toy_sim", "evidence": "synthetic test", "repo_markers": ["README.md"],
        "tasks": {"kind": "module_registry", "pattern": "", "names": ["ToyTask"]},
        "task_contract": {"primary_metric": {"name": "success_rate",
                     "direction": "maximize", "unit": "fraction", "source": "log"}},
        "assets": {}, "capabilities": {},
        "optimization_space": {"collection": [], "training": [{
            "name": "epochs", "kind": "integer", "description": "epochs",
            "low": 1, "high": 2, "default": 1}]}}
    (scout / "declaration.json").write_text(
        json.dumps({"declaration": declaration}), encoding="utf-8")
    job = submit(output, repo=repo, stage="train", settings={"device": "cpu"},
                 requested_seconds=4, reason="test native long job")
    job_id = job["job_id"]
    try:
        adjusted = adjust(output, job_id, seconds_from_now=5,
                          reason="first epoch has started")
        assert adjusted["deadline_epoch"] > job["deadline_epoch"]
        cancellation = cancel(output, job_id, reason="fixture has enough evidence")
        assert cancellation["cancel_requested"]
        for _ in range(100):
            current = status(output, job_id)
            if current["status"] in {"cancelled", "failed", "completed"}:
                break
            time.sleep(.1)
        assert current["status"] == "cancelled", current
        assert current["result"]["stage_result"]["termination_reason"] == "cancelled"
    finally:
        cancel(output, job_id, reason="test cleanup")
