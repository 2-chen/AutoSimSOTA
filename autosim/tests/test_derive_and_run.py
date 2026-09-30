import json
from pathlib import Path

import pytest

from autosim.research.derive_and_run import main, validate_paused_research_resume


def _session(output: Path, *, status="paused", rounds=2, settings=None,
             repository=None):
    path = output / "research" / "derived" / "controller_session.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "run_id": "derived",
        "repository": str(repository or output.parent / "benchmark"),
        "status": status,
        "rounds": rounds,
        "base_settings": ({"task": "PushCube-v1"} if settings is None else settings),
    }), encoding="utf-8")
    return path


def test_paused_session_accepts_exact_frozen_inputs(tmp_path):
    _session(tmp_path)

    validate_paused_research_resume(
        tmp_path, rounds=2, settings={"task": "PushCube-v1"})


@pytest.mark.parametrize("rounds,settings", [
    (1, {"task": "PushCube-v1"}),
    (2, {"task": "PushCube-v1", "steps": 2_000_000}),
])
def test_paused_session_rejects_protocol_drift_before_controller_action(
        tmp_path, rounds, settings):
    _session(tmp_path)

    with pytest.raises(ValueError, match="different frozen rounds/settings"):
        validate_paused_research_resume(tmp_path, rounds=rounds, settings=settings)


def test_cli_rejects_round_mismatch_before_creating_budget_or_action_receipt(tmp_path):
    output = tmp_path / "run"
    session_path = _session(output)
    original_session = session_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "benchmark"), str(output), "1",
              '{"task":"PushCube-v1"}'])

    assert error.value.code == 2
    assert session_path.read_bytes() == original_session
    assert not (output / "budget.json").exists()
    assert not (output / "action_receipts").exists()


def test_autosota_framework_requires_an_isolated_checkout_before_reserving_run(
        tmp_path):
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "benchmark"), str(output),
              "--framework", "autosota_sim_v1", "--in-place"])

    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("full_copy", [False, True])
def test_formal_research_entry_constructs_role_runtime_on_isolated_checkout(
        tmp_path, monkeypatch, full_copy):
    from autosim.research.agent_client import RoleAwareAgentClient

    source = tmp_path / "benchmark"
    source.mkdir()
    (source / "README.md").write_text("synthetic no-GPU fixture", encoding="utf-8")
    import subprocess
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    data = source / "data"
    data.mkdir()
    (data / "input.txt").write_text("not source code")
    output = tmp_path / "run"
    captured = {}

    class FakeRepositoryBudget:
        def __init__(self, *_args, **_kwargs):
            pass

        def reserve(self, *_args, **_kwargs):
            return None

        def finish(self, *_args, **_kwargs):
            return None

    class FakePreparation:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, *, max_steps, max_relaunch):
            captured["max_steps"] = max_steps
            captured["max_relaunch"] = max_relaunch
            return {"status": "completed"}

    monkeypatch.setattr("autosim.llm_client.load_credential_file",
                        lambda **_: {"model": "test-model", "source": "test"})
    monkeypatch.setattr("autosim.research.repository_budget.RepositoryBudget",
                        FakeRepositoryBudget)
    monkeypatch.setattr("autosim.research.prepare.Preparation", FakePreparation)
    monkeypatch.setattr("autosim.research.runtime_preflight.check_runtime",
                        lambda: {"status": "ready", "checks": []})

    result = main([str(source), str(output), "--framework", "autosota_sim_v1",
                   "--max-actions", "3",
                   *(["--full-copy"] if full_copy else
                     ["--resource", f"inputs={data}"])])

    assert result == 0
    assert isinstance(captured["client"], RoleAwareAgentClient)
    assert captured["repo"] == (output / "checkout").resolve()
    assert captured["scouting"] == (output / "scouting").resolve()
    assert captured["max_steps"] == 3
    assert captured["max_relaunch"] == 4
    manifest = json.loads((output / "workspace_snapshot.json").read_text())
    assert manifest["copy_mode"] == ("full_worktree" if full_copy else "tracked_worktree")
    assert (output / "checkout" / "data" / "input.txt").exists() is full_copy
    if not full_copy:
        assert manifest["resource_bindings"][0]["target"] == "inputs"
    spec = json.loads((output / "agent" / "runtime_spec.json").read_text())
    assert spec["framework"] == "autosota_sim_v1"
    assert spec["run_id"] == "derived"
    assert spec["workspace_identity"] == str((output / "checkout").resolve())
    assert spec["turn_budget_usd"] == 4.0
    assert spec["total_budget_usd"] == 60.0
    assert json.loads((output / "budget.json").read_text())["wall_seconds"] == 172800.0


def test_default_isolation_refuses_nested_output_before_writing_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(SystemExit):
        main([str(source), str(source / "output")])
    assert list(source.iterdir()) == []


@pytest.mark.parametrize("status", ["paused", "running", "interrupted", "completed"])
def test_existing_research_session_cannot_be_rederived_in_place(tmp_path, status):
    _session(tmp_path, status=status)

    with pytest.raises(ValueError, match="cannot be re-derived in place"):
        validate_paused_research_resume(
            tmp_path, rounds=2, settings={"task": "PushCube-v1"}, rederive=True)


def test_paused_session_rejects_different_repository(tmp_path):
    _session(tmp_path)

    with pytest.raises(ValueError, match="repository identity differs"):
        validate_paused_research_resume(
            tmp_path, rounds=2, settings={"task": "PushCube-v1"},
            repository=tmp_path / "another-benchmark")


def test_isolated_paused_session_matches_its_manifest_identity(tmp_path):
    output = tmp_path / "run"
    source = tmp_path / "benchmark"
    checkout = output / "checkout"
    checkout.mkdir(parents=True)
    _session(output, repository=checkout)
    (output / "workspace_snapshot.json").write_text(json.dumps({
        "source": str(source), "destination": str(checkout),
    }), encoding="utf-8")

    validate_paused_research_resume(
        output, rounds=2, settings={"task": "PushCube-v1"}, repository=source)


def test_nonpaused_session_remains_with_reconciliation_controller(tmp_path):
    _session(tmp_path, status="interrupted")

    validate_paused_research_resume(tmp_path, rounds=1, settings={})


def test_missing_session_is_a_new_run(tmp_path):
    validate_paused_research_resume(tmp_path, rounds=2, settings={})
