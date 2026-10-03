"""Recovery routing/receipts, without claiming a native benchmark run."""
from autosim.research import recovery_contract, recorder
from autosim.research.prepare import Preparation


def planning_failure(ref="action_receipts/first.json"):
    return {"step": "build_the_environment", "outcome": "raised",
            "failure_domain": "framework_plan", "repair_owner": "environment_planner",
            "receipt_ref": ref, "evidence_id": "a" * 32,
            "arguments": {"max_operations": 1}}


def test_planning_failure_has_correct_owner_and_bounded_revalidation():
    first = planning_failure()
    assert recovery_contract.pending([first], {})["owner"] == "environment_planner"
    second = {**planning_failure("second"), "step": "retry_failed_action",
              "replayed_step": "build_the_environment", "outcome": "reverification_failed",
              "recovery_parent_ref": first["receipt_ref"]}
    assert not recovery_contract.pending([first, second], {})
    assert not recovery_contract.pending([first, planning_failure("second")], {})
    assert not recovery_contract.pending([first, {"step":"build_the_environment", "outcome":"done"}], {})


def test_missing_evidence_and_candidate_failure_are_not_replay_permissions():
    fault = planning_failure(); fault.pop("evidence_id")
    assert not recovery_contract.pending([fault], {})
    assert not recovery_contract.pending([{"step":"run_the_loop", "outcome":"failed"}], {})


def test_concrete_fix_is_retired_by_original_operation_success():
    failed = {"step":"derive_a_command", "outcome":"failed", "receipt_ref":"original",
              "arguments":{"stage":"evaluate"}}
    fix = {"status":"assessed", "assessment":"repair_attempted",
           "fix_attempt_id":"fix", "failure_receipt_ref":"original"}
    assert recovery_contract.pending([failed], fix)
    different = {"step":"derive_a_command", "outcome":"done", "arguments":{"stage":"train"}}
    assert recovery_contract.pending([failed, different], fix)
    same = {**different, "arguments":{"stage":"evaluate"}}
    assert not recovery_contract.pending([failed, same], fix)


def test_planner_retry_replays_exact_arguments_and_preserves_failure_identity(tmp_path, monkeypatch):
    class Client: model = "fixture"
    controller = Preparation(repo=tmp_path, output=tmp_path / "out", client=Client(), scouting=tmp_path / "scouting")
    controller.steps = [planning_failure()]
    calls = []
    def perform(step, **args):
        calls.append((step, args))
        return {"outcome":"raised", "because":"new invalid proposal", "failure_domain":"framework_plan",
                "repair_owner":"environment_planner", "evidence_id":"b" * 32,
                "evidence_refs":["evidence/" + "b" * 32 + ".json"]}
    monkeypatch.setattr(controller, "do", perform)
    result = controller._step_retry_failed_action()
    assert calls == [("build_the_environment", {"max_operations":1})]
    assert result["outcome"] == "reverification_failed"
    assert result["evidence_id"] == "b" * 32
    assert result["recovery_parent_ref"] == controller.steps[0]["receipt_ref"]
    assert result["replayed_arguments"] == {"max_operations":1}
    assert result["repair_owner"] == "environment_planner"
    assert "new invalid proposal" in result["because"]


def test_main_agent_cannot_stop_before_one_admitted_revalidation(tmp_path, monkeypatch):
    class Client:
        model = "fixture"
        supports_main_agent = True
    controller = Preparation(repo=tmp_path, output=tmp_path / "out", client=Client(), scouting=tmp_path / "scouting")
    controller.declaration = {"benchmark":"fixture"}
    controller.execution = {"stages":{"evaluate":{"available":True,"entrypoint":"eval.py"}}}
    controller.steps = [planning_failure()]
    assert "retry_failed_action" in controller._available_operations({}, {})
    assert "stop" not in controller._available_operations({}, {})
    controller.steps.append(planning_failure("second"))
    assert "stop" in controller._available_operations({}, {})


def test_incremental_setup_progress_is_not_failed_revalidation(tmp_path, monkeypatch):
    class Client: model = "fixture"
    controller = Preparation(repo=tmp_path, output=tmp_path / "out", client=Client(), scouting=tmp_path / "scouting")
    controller.steps = [planning_failure()]
    monkeypatch.setattr(controller, "do", lambda *args, **kwargs: {"outcome":"checkpoint", "because":"one installation step done"})
    assert controller._step_retry_failed_action()["outcome"] == "revalidation_in_progress"


def test_run_render_exposes_revalidation_not_fake_score(tmp_path):
    first = planning_failure()
    retry = {"step":"retry_failed_action", "outcome":"reverified", "original_outcome":"done",
             "replayed_step":"build_the_environment", "recovery_parent_ref":first["receipt_ref"]}
    snapshot = {"status":"running", "measurements":[], "actions":[first,retry],
                "revision":"fixture", "event_revision":"fixture",
                "budget":{}, "run_id":"fixture", "verified_media":[],
                "recovery_transaction":recovery_contract.view([first,retry], {})}
    text = recorder.render(tmp_path, snapshot, {})
    assert "后续已复验通过" in text
    assert "原操作复验通过" in text
    assert "不代表正式指标提升" in text


def test_preparation_live_context_preserves_error_routing(tmp_path, monkeypatch):
    class Client: model = "fixture"
    controller = Preparation(repo=tmp_path, output=tmp_path / "out", client=Client(), scouting=tmp_path / "scouting")
    controller.steps = [planning_failure()]
    captured = []
    real_refresh = recorder.refresh
    def refresh(root, view, **kwargs):
        captured.append(kwargs["context"])
        return real_refresh(root, view, **{**kwargs, "client":None})
    monkeypatch.setattr(recorder, "refresh", refresh)
    controller._refresh_document(status="running", current="fixture")
    assert captured
    assert captured[0]["actions"][0]["failure_domain"] == "framework_plan"
    assert captured[0]["recovery_transaction"]["pending"]["owner"] == "environment_planner"
