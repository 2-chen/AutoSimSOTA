import json
from pathlib import Path

import pytest

from autosim.research.derive_and_run import main, validate_paused_research_resume


def _session(output: Path, *, status="paused", rounds=2, settings=None,
             repository=None):
    path = output / "research" / "derived" / "controller_session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
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


@pytest.mark.parametrize('fault',['','changed_constraint','unowned','tampered_settings'])
def test_launcher_preserves_verified_main_baseline_settings_without_requiring_agent_arguments(tmp_path,fault):
    from tests.test_derived_research import research
    real=research(tmp_path)
    real.controller_decision_id='a'*16 if fault!='unowned' else ''
    settings={'task':'Lift','steps':20000,'episodes':50,'seed':0}
    baseline=real.baseline(settings=settings)
    assert baseline['ok'] is True
    _session(real.output,settings={**settings,**({'steps':1} if fault=='tampered_settings' else {})},
             repository=tmp_path)
    before=(real.run_root/'controller_session.json').read_bytes()
    requested={'task':'different' if fault=='changed_constraint' else 'Lift'}
    if fault:
        with pytest.raises(ValueError,match='different frozen rounds/settings'):
            validate_paused_research_resume(real.output,rounds=2,settings=requested,repository=tmp_path)
    else:
        assert validate_paused_research_resume(real.output,rounds=2,settings=requested,repository=tmp_path)==2
    assert (real.run_root/'controller_session.json').read_bytes()==before


@pytest.mark.parametrize('fault', ['', 'constraint', 'session_settings', 'session_hash',
    'parent_hash', 'parent_action', 'parent_child', 'native_settings', 'native_stage',
    'evaluation_exposed', 'linked_parent'])
def test_launcher_can_resume_main_owned_paid_unscored_baseline_without_adoption(tmp_path, fault):
    from tests.test_derived_research import research
    from autosim.research.common import atomic_json, object_digest, read_json
    from autosim.research.action_receipt import ActionReceipt, write_action_receipt
    real = research(tmp_path)
    real.require_training_progress = True
    real.controller_decision_id = 'a'*16
    real.backend.stages['train']['artifact'] = 'models/**'
    real.backend.sources['train'] = real.backend.sources['train'].replace(
        'print("succ: 0.50")', 'print("global_step=17")')
    settings = {'task':'Lift', 'steps':17, 'episodes':50, 'seed':0}
    baseline = real.baseline(settings=settings)
    assert baseline['ok'] is False and baseline['where'] == 'policy_artifact'
    assert not baseline.get('evaluate')
    train = baseline['train']
    attempt = train['attempt_id']
    relative = f'research/derived/attempts/{attempt}/receipt.json'
    session_path = _session(real.output, settings=settings, repository=tmp_path)
    session = read_json(session_path)
    if fault == 'session_settings':
        session['base_settings']['steps'] = 1
    session['input_component_sha256'] = {'settings':object_digest(session['base_settings'])}
    if fault == 'session_hash':
        session['input_component_sha256']['settings'] = '0'*64
    atomic_json(session_path, session)
    ref, _ = write_action_receipt(real.output, ActionReceipt(
        action_id='a'*16, decision_id='a'*16, run_id='derived', repository=str(tmp_path),
        action='derive_a_command' if fault == 'parent_action' else 'run_the_loop',
        state_revision=1, started_at='2026-10-11T00:00:00+00:00',
        finished_at='2026-10-11T00:01:00+00:00', status='yielded', reason='Unscored paid baseline',
        child_attempt_ids=[] if fault == 'parent_child' else [attempt], evidence_refs=[relative]))
    if fault == 'parent_hash':
        parent = read_json(real.output/ref)
        parent['reason'] = 'modified without resealing'
        atomic_json(real.output/ref, parent)
    if fault == 'linked_parent':
        original = real.output/ref
        original.rename(original.with_name('original.json'))
        original.symlink_to('original.json')
    if fault in {'native_settings', 'native_stage'}:
        receipt = read_json(real.output/relative)
        if fault == 'native_settings':
            receipt['settings_digest'] = '0'*64
        else:
            receipt['node_id'] = 'evaluate'
        atomic_json(real.output/relative, receipt)
    if fault == 'evaluation_exposed':
        baseline['evaluate'] = {'attempt_id':'f'*32}
        atomic_json(real.run_root/'measurements/baseline.json', baseline)
    requested = {'task':'Other' if fault == 'constraint' else 'Lift'}
    before_session = session_path.read_bytes()
    before_baseline = (real.run_root/'measurements/baseline.json').read_bytes()
    if fault:
        with pytest.raises(ValueError, match='different frozen rounds/settings'):
            validate_paused_research_resume(real.output, rounds=2, settings=requested, repository=tmp_path)
    else:
        assert validate_paused_research_resume(real.output, rounds=2, settings=requested,
            repository=tmp_path) == 2
    assert session_path.read_bytes() == before_session
    assert (real.run_root/'measurements/baseline.json').read_bytes() == before_baseline


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

    output.mkdir()
    (output / "STARTUP.md").write_text("# 旧运行启动受阻\n")

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
    assert "启动检查通过" in (output / "STARTUP.md").read_text()
    assert "受阻" not in (output / "STARTUP.md").read_text()
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
