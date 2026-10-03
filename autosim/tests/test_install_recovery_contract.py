"""Installation recovery regression shapes, including read-only RoboSyn manifests."""
import json
import sys
from pathlib import Path
import pytest
from autosim.research import provision as pv, recovery_contract
from autosim.research.execution_paths import ExecutionPaths
from autosim.research.common import sanitize_model_payload


def test_parent_dependency_paths_survive_model_redaction_without_host_paths(tmp_path):
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    paths = ExecutionPaths(repo, output)
    manifests = paths.manifests({"policy/act/pyproject.toml":
        'local = {path="../.."}\nengine = {path="../../../Engine", editable=true}'})
    payload = json.dumps(sanitize_model_payload({"manifests":manifests, "paths":paths.catalog()}, local_roots=(repo,)))
    assert "[OUTSIDE_PATH]" not in payload
    assert str(tmp_path) not in payload
    assert "Engine" in payload
    restored = paths.decode('install "{run_path_1}" "{run_path_2}"')
    assert str(repo) in restored and str(output / "Engine") in restored
    with pytest.raises(ValueError): paths.decode("install [OUTSIDE_PATH]")
    with pytest.raises(ValueError): paths.decode("install {run_path_99}")
    with pytest.raises(ValueError): paths.decode("install {run_path_1}/../../outside")


def test_outside_run_and_symlink_escape_cannot_become_execution_aliases(tmp_path):
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    paths = ExecutionPaths(repo, output)
    assert "[UNBOUND_PATH]" in paths.manifests({"pyproject.toml":'path="../../outside"'})["pyproject.toml"]
    assert not paths.catalog()
    paths.manifests({"pyproject.toml":'path="../Engine"'})
    (output / "Engine").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escaped"): paths.decode("{run_path_1}")


def test_planner_restores_alias_before_adopting_executable_commands(tmp_path, monkeypatch):
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            context = json.loads(user)
            assert context["execution_path_aliases"][0]["run_relative_target"] == "Engine"
            return json.dumps({"python":"3.10", "commands":['{pip} install "{run_path_1}"'],
                               "probes":["{python} -c 'import engine'"]}), {}
    client = Client(); client.output = output
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    monkeypatch.setattr(pv, "plan_problems", lambda *args: [])
    plan = pv.plan(client, repo, manifests={"pyproject.toml":'path="../Engine"'})
    assert plan["commands"] == ['{pip} install "' + str(output / "Engine") + '"']


def test_install_replacement_requires_matching_failure_and_source_quotes(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('name = "local-package"\n')
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            assert "INSTALLATION" in system
            assert kwargs["read_only"] is True
            return json.dumps({"approved":True, "reason":"local source is the required package",
                "citations":[{"file":"pyproject.toml","quote":'name = "local-package"'}]}), {}
    claim = {"same_capability":"install the actual local package", "source_refs":["pyproject.toml"],
             "failure_evidence_id":"a"*32}
    failure = {"evidence_id":"a"*32, "command":"pip install absent-name"}
    assert pv.review_install_replacement(Client(), tmp_path, failure, ["pip install -e ."], claim)["approved"]
    assert not pv.review_install_replacement(Client(), tmp_path, failure, [], {**claim,"failure_evidence_id":"wrong"})["approved"]
    assert not pv.review_install_replacement(Client(), tmp_path, failure, [], {**claim,"source_refs":["../outside"]})["approved"]


@pytest.mark.parametrize("approved", [True, False])
def test_cooperative_build_only_retires_original_after_approved_replacement(tmp_path, monkeypatch, approved):
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    seen = []
    def execute(command, **kwargs):
        seen.append(command)
        return {"command":command,"ok":False,"returncode":1,"seconds":0.01,
                "failure_kind":"failed","evidence_id":"a"*32,"attempt_id":"a"*32}
    def resume(*args, **kwargs):
        kwargs["transcript"].append({"kind":"resume", "commands":["echo corrected"],
            "replacement_review":{"approved":approved}})
        return ["echo corrected"], None
    monkeypatch.setattr(pv, "run", execute)
    monkeypatch.setattr(pv, "resume", resume)
    result = pv.build(repo, client=object(), prefix=output / "env", output=output,
        python=sys.executable, manifests={}, assets={}, max_rounds=1, max_operations=1,
        seed={"python":sys.executable,"templates":["false"], "probes":["{python} -c 'import json'"]})
    cursor = json.loads((output / "provision_cursor.json").read_text())
    assert cursor["pending"] == (["echo corrected"] if approved else ["echo corrected", "false"])
    assert cursor["probes"] == ["{python} -c 'import json'"]
    assert seen == ["false"]
    assert result["verdict"]["passed"] is False


def test_native_install_failure_is_pending_then_explicitly_unresolved_not_cleared():
    failed = {"step":"build_the_environment","outcome":"checkpoint failure", "receipt_ref":"one",
        "evidence_id":"a"*32,"failure_domain":"native_install", "repair_owner":"environment_executor",
        "native_operation":{"template":"pip install absent-name"}}
    view = recovery_contract.view([failed], {})
    assert view["pending"]["owner"] == "environment_executor"
    failed_again = {**failed, "receipt_ref":"two", "evidence_id":"b"*32}
    view = recovery_contract.view([failed, failed_again], {})
    assert view["status"] == "unresolved_requires_new_evidence"
    assert view["unresolved_failure"]["evidence_id"] == "b"*32
    progress = {"step":"build_the_environment", "outcome":"checkpoint"}
    assert recovery_contract.view([failed,progress], {})["unresolved_failure"]
    done = {"step":"build_the_environment", "outcome":"done"}
    assert not recovery_contract.view([failed,done], {})["unresolved_failure"]


def test_native_checkpoint_handoff_keeps_stable_failure_id_and_recovery_owner(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    class Client: model = "fixture"
    repo = tmp_path / "checkout"; repo.mkdir()
    controller = Preparation(repo=repo, output=tmp_path / "out", client=Client(), scouting=tmp_path / "scouting")
    controller.declaration = {"benchmark":"fixture"}
    monkeypatch.setattr(controller, "state", lambda: {"workspace_resources":{}})
    monkeypatch.setattr(controller, "_surveyed_runnable_stages", lambda: {"evaluate":{"entrypoint":"eval.py"}})
    monkeypatch.setattr(controller, "_select_execution_path", lambda **_: {"stages":["evaluate"],"asset_keys":[],"why":"fixture"})
    attempt = {"ok":False,"evidence_id":"a"*32,"attempt_id":"a"*32,
               "evidence_ref":"evidence/" + "a"*32 + ".json", "template":"pip install missing",
               "returncode":1,"failure_kind":"failed"}
    monkeypatch.setattr(pv, "build", lambda *args, **kwargs: {"status":"yielded","latest_attempt":attempt})
    result = controller._step_build_the_environment()
    assert result["outcome"] == "checkpoint failure"
    assert result["failure_domain"] == "native_install"
    assert result["repair_owner"] == "environment_executor"
    assert result["evidence_id"] == attempt["evidence_id"]
    assert attempt["evidence_ref"] in result["evidence_refs"]
    controller.steps = [{**result,"step":"build_the_environment","receipt_ref":"parent"}]
    assert controller._repair_retry_available()
    assert controller._recovery_transaction()["status"] == "revalidation_required"
    # A new operation failure inside revalidation is not an automatic exhausted old retry.
    controller.steps.append({**result, "step":"retry_failed_action", "replayed_step":"build_the_environment",
        "replayed_arguments":{"max_operations":3}, "receipt_ref":"second",
        "native_operation":{"template":"different setup operation"}})
    assert controller._repair_retry_available()
    called = []
    monkeypatch.setattr(controller, "do", lambda step, **args: called.append((step,args)) or {"outcome":"checkpoint"})
    controller._step_retry_failed_action()
    assert called == [("build_the_environment",{"max_operations":3})]


def test_resume_replacement_requires_review_and_records_parent_evidence(tmp_path):
    (tmp_path / "pyproject.toml").write_text('name = "actual-local-package"\n')
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            if "INSTALLATION" in system:
                return json.dumps({"approved":True,"reason":"correct local source",
                    "citations":[{"file":"pyproject.toml","quote":'name = "actual-local-package"'}]}), {}
            return json.dumps({"repair_mode":"replace_operation","commands":["{pip} install -e {repo}"],
                "install_replacement_evidence":{"same_capability":"install required local package",
                    "source_refs":["pyproject.toml"],"failure_evidence_id":"a"*32}}), {}
    transcript = []
    commands, probes = pv.resume(Client(), tmp_path, [], {"command":"pip install absent","evidence_id":"a"*32},
        manifests={}, transcript=transcript, values={})
    assert commands == ["{pip} install -e {repo}"] and probes is None
    assert transcript[-1]["replacement_review"]["approved"]
    assert transcript[-1]["parent_evidence_id"] == "a"*32


def test_bad_reviewer_quote_cannot_retire_original_install(tmp_path):
    (tmp_path / "pyproject.toml").write_text('name = "actual"\n')
    class Client:
        def chat_with_metadata(self, *args, **kwargs):
            return json.dumps({"approved":True,"reason":"invented",
                "citations":[{"file":"pyproject.toml","quote":"not in the source"}]}), {}
    claim = {"same_capability":"same dependency","source_refs":["pyproject.toml"],"failure_evidence_id":"a"*32}
    assert not pv.review_install_replacement(Client(), tmp_path, {"evidence_id":"a"*32}, ["echo done"], claim)["approved"]


def test_real_robosyn_manifest_paths_map_to_isolated_run_not_source_host(tmp_path):
    source = Path('/home/wbc/下载/autoresearch/AutoSimSOTA/autoresearch_runs/robosyn_recovery_anchor_20260930_v1/checkout/policy/act/pyproject.toml')
    if not source.is_file(): pytest.skip("optional read-only real RoboSyn manifest")
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    paths = ExecutionPaths(repo, output)
    manifests = paths.manifests({"policy/act/pyproject.toml":source.read_text()})
    assert "[OUTSIDE_PATH]" not in json.dumps(sanitize_model_payload(manifests))
    assert set(paths.paths.values()) == {repo, output / "EmbodiChain"}
