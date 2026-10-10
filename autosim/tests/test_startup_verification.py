import json
import subprocess
from types import SimpleNamespace

import pytest

from autosim.research import execution_derive, startup_verification as startup
from autosim.research.common import atomic_json, digest
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.main_agent import (
    retain_handoff, memory_view, validate_budget_strategy, validate_plan,
    task_system_for_role, validate_task,
)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "run"
    repo = root / "checkout"
    repo.mkdir(parents=True)
    code = repo / "train.py"
    code.write_text("batch = next(loader)\noptimizer.step()\nprint('completed update 1')\n")
    identity = "a" * 32
    log = root / "native.log"
    log.write_text("completed update 1\n")
    argv = ["python", str(code)]
    receipt = {"id": identity, "stage": "train", "argv": argv,
               "capture_error": None, "orphaned_children": False}
    atomic_json(root / "receipt.json", receipt)
    capture_attempt_evidence(root, attempt_id=identity, log=log,
                             receipt_ref="receipt.json", status="timed_out", returncode=-15,
                             termination_reason="timed_out")
    source = {"id": "source-0", "path": code, "sha256": digest(code),
              "content": code.read_text(), "excerpt": code.read_text()}
    monkeypatch.setattr(startup, "sources", lambda *args, **kwargs: [source])
    answer = {"verdict": "verified", "why": "loader to optimizer update observed",
              "log_quote": "completed update 1",
              "source_evidence": [{"source_id": "source-0", "quote": "optimizer.step()"}]}
    class Client:
        supports_native_progress_audit = True
        def chat_with_metadata(self, system, user, **kw):
            self.kw = kw
            return json.dumps(answer), {}
    return SimpleNamespace(root=root, repo=repo, argv=argv, log=log, code=code,
                           answer=answer, client=Client(), receipt=receipt,
                           ref=f"evidence/{identity}.json")


def run(e):
    return startup.review(e.client, repo=e.repo, root=e.root, argv=e.argv,
                          evidence_ref=e.ref, timeout=20)


def test_contained_timeout_can_verify_startup_but_never_completion(evidence):
    result = run(evidence)
    assert result["status"] == "verified"
    assert result["completion_verified"] is False
    assert result["artifact_verified"] is False
    assert evidence.client.kw["read_only"] is True
    assert evidence.client.kw["auto_skills"] is False
    assert (evidence.root / result["review_ref"]).is_file()


@pytest.mark.parametrize("change", ["log_tamper", "fake_source", "fake_log", "abstain",
                                    "storage", "orphan", "capture", "wrong_argv", "no_auditor"])
def test_startup_review_abstains_on_invalid_or_insufficient_evidence(evidence, change):
    if change == "log_tamper": evidence.log.write_text("forged update\n")
    if change == "fake_source": evidence.answer["source_evidence"][0]["quote"] = "fake_step()"
    if change == "fake_log": evidence.answer["log_quote"] = "completed update 999"
    if change == "abstain": evidence.answer["verdict"] = "unverified"
    if change == "storage": evidence.receipt["termination_reason"] = "storage_pressure"
    if change == "orphan": evidence.receipt["orphaned_children"] = True
    if change == "capture": evidence.receipt["capture_error"] = {"type": "IOError"}
    if change == "wrong_argv": evidence.receipt["argv"] = ["different"]
    if change == "no_auditor": evidence.client.supports_native_progress_audit = False
    atomic_json(evidence.root / "receipt.json", evidence.receipt)
    assert run(evidence)["status"] == "unverified"


def test_model_error_preserves_window_as_unverified(evidence):
    def fail(*args, **kwargs): raise TimeoutError("provider time limit")
    evidence.client.chat_with_metadata = fail
    assert "provider time limit" in run(evidence)["reason"]


def test_source_trace_resolves_relative_entrypoint_and_package_imports(tmp_path):
    from autosim.research.training_progress_audit import sources
    repo = tmp_path / 'checkout'
    package = repo / 'native'
    package.mkdir(parents=True)
    (repo / 'train.py').write_text('from native import worker\n')
    (package / '__init__.py').write_text('from .worker import learn\n')
    (package / 'worker.py').write_text('def learn():\n    optimizer.step()\n')
    code = sources(repo, tmp_path, ['python', 'train.py'], cwd=repo)
    assert {'train.py', '__init__.py', 'worker.py'} <= {row['path'].name for row in code}


def test_resource_selects_skills_explicitly_with_its_own_identity(monkeypatch):
    from autosim.research import data_agent
    monkeypatch.setattr(data_agent, 'skills_catalog', lambda **kw: {'entries': [{'id': 'budget'}]})
    monkeypatch.setattr(data_agent, 'read_selected_skills', lambda ids, **kw: {
        'selection': [{'id': identity} for identity in ids], 'skills': []})
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            assert 'Resource Agent' in system
            assert kwargs['read_only'] and not kwargs['auto_skills']
            return json.dumps({'skill_reads': [{'id': 'budget', 'why': 'job time bound'}]}), {}
    result = data_agent.select_methods(Client(), {'role': 'resource', 'task': 'allocate'}, {}, timeout=10)
    assert result['method_library']['selection'][0]['selected_by'] == 'resource_agent'


def test_delegated_resource_role_persists_budget_report(tmp_path):
    from tests.test_main_agent import make
    real = make(tmp_path)
    outcome = real.do('research_task', role='resource', task='Plan local GPU and model windows',
                      expected_result='Budget allocation grounded in native receipts', mode='inspect')
    assert outcome['outcome'] == 'reported'
    handoff = real.main_agent['resource_agent_handoff']
    assert handoff['role'] == 'resource'
    assert 'resource_skill_selection' in handoff
    assert real.client.role == 'scheduler'


@pytest.mark.parametrize("reviewed,score_stage", [(True, False), (False, False), (True, True)])
def test_make_runnable_handles_window_without_replaying_invocation(tmp_path, monkeypatch, reviewed, score_stage):
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "train.py").write_text("# native training\n")
    from autosim.research import stage_verification
    monkeypatch.setattr(stage_verification, "resource_context", lambda repo: {})
    monkeypatch.setattr(stage_verification, "recent_attempts", lambda root: [])
    launches = []
    def native(argv, **kw):
        launches.append(argv)
        exc = subprocess.TimeoutExpired(argv, kw["timeout"], output="completed update 1")
        exc.evidence_ref = "evidence/" + "a"*32 + ".json"
        raise exc
    monkeypatch.setattr(stage_verification, "run_native", native)
    monkeypatch.setattr(startup, "review", lambda *args, **kwargs: {
        "status": "verified" if reviewed else "unverified", "completion_verified": False})
    def generate(*args, verify, inputs_for_verify, **kw):
        source = "def stage_argv_train(i):\n    return [i['python'], i['repo'] + '/train.py']\n"
        outcome = verify(["python", str(repo / "train.py")])
        assert outcome["ok"]
        return source, {}, [{"status": "accepted", **outcome}]
    monkeypatch.setattr(execution_derive, "generate_argv", generate)
    source, _, log, row = execution_derive.make_runnable(None, "train", {
        "entrypoint": "train.py", "invocation": "python train.py", "parameters": []},
        repo=repo, repository_files=["train.py"], inputs_for_verify={
            "repo": str(repo), "python": "python", "output": str(tmp_path / "verify")},
        verification_window=lambda seconds: None, require_step_control=True,
        require_evaluation_progress=score_stage, verification_timeout=2400, rounds=3, attempts=3)
    assert len(launches) == 1
    if reviewed and not score_stage:
        assert source and row["verification_scope"] == "training_startup_only"
    else:
        assert source is None
        assert log[-1]["failure_kind"] == "verification_window_exhausted"
        assert log[-1]["repair_owner"] == "resource"


def test_startup_cannot_bind_metric_or_change_total_budget(tmp_path):
    from tests.test_main_agent import make
    real = make(tmp_path)
    real.stages['train'] = 'native command'
    real._score_target = lambda: 'train'
    atomic_json(real.output / 'derivation_attempts/train.json', {'attempts': [
        {'status': 'accepted', 'verification_scope': 'training_startup_only', 'said': 'success 1.0'}]})
    assert real._step_bind_metric()['outcome'] == 'not bound'
    assert not real.client.calls
    total = {'limit': 600, 'started': 123}
    atomic_json(real.output / 'budget.json', total)
    strategy = {'why': 'measured window', 'allocations': ['job A: 1200 seconds'],
                'evidence_refs': ['native receipt'], 'revisit_condition': 'next update',
                'next_action': 'submit_native_job'}
    plan = {'objective': 'train', 'hypotheses': [], 'open_questions': [],
            'next_actions': [], 'evidence_refs': [], 'budget_strategy': strategy}
    assert real._step_update_research_plan(plan=plan)['outcome'] == 'recorded'
    assert json.loads((real.output / 'budget.json').read_text()) == total
    assert json.loads((real.output / 'budget_strategy.json').read_text())['allocations'] == strategy['allocations']


def test_resource_advice_is_pinned_bounded_and_not_total_budget_authority():
    report = {"summary": "budget " + "x"*4000, "recommended_next_actions": [],
              "evidence_refs": [], "findings": [], "risks": []}
    memory = retain_handoff({}, {"role": "resource", "task": "allocation", "report": report})
    for _ in range(15): memory = retain_handoff(memory, {"role": "monitor", "report": report})
    view = memory_view(memory)
    assert view["resource_agent_handoff"]["role"] == "resource"
    assert len(json.dumps(view["resource_agent_handoff"])) < 2200
    assert "local budget ownership" in task_system_for_role("resource")
    with pytest.raises(ValueError, match="read-only"):
        validate_task({"role": "resource", "task": "allocate", "expected_result": "advice", "mode": "edit"})
    strategy = {"why": "measured progress", "allocations": ["job A: 3600 local seconds"],
                "evidence_refs": ["native receipt"], "revisit_condition": "next checkpoint", "next_action": "adjust_native_job"}
    result = validate_budget_strategy(strategy)
    assert result["authority"] == "resource_allocation_advice_not_hard_budget_authorization"
    plan = {"objective": "train", "hypotheses": [], "open_questions": [],
            "next_actions": [], "evidence_refs": [], "budget_strategy": strategy}
    assert validate_plan(plan)["budget_strategy"] == result
    with pytest.raises(ValueError): validate_budget_strategy({**strategy, "total_usd": 999})
