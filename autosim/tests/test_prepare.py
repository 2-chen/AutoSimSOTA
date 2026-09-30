"""One loop for the five things that have to be true, and what it does when one is not.

The defects this replaces were all at the seams: a driver that read a document the previous
program had not written yet and raised `FileNotFoundError`; a build that reported success on
an empty record; a declaration chosen by newest mtime. None of them were about a benchmark and
none of them were inside any one step.
"""

import json
import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from autosim.research.prepare import OPERATIONS, STEP_SYSTEM, Preparation
from autosim.research.common import object_digest
from autosim.research.action_receipt import read_action_receipt


class _Client:
    """Answers the choose-step question with a scripted list, then stops."""

    model = "a-test-client"

    def __init__(self, choices):
        self.choices = list(choices)
        self.seen = []

    def chat_with_metadata(self, system, user, **kwargs):
        self.seen.append(user)
        step = self.choices.pop(0) if self.choices else {"do": "stop", "why": "nothing left"}
        return json.dumps(step), {}


def preparation(tmp_path, choices=()):
    return Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client(choices),
                       scouting=tmp_path / "scouting")


def test_preparation_steps_are_closed_and_named():
    """The set is closed and named, including the mandatory interruption recovery step."""
    assert set(OPERATIONS) == {"read_the_checkout", "declare", "build_the_environment",
                               "derive_a_command", "reconcile_interrupted_action",
                               "discard_interrupted_candidate",
                               "bind_metric", "recover_unscored_baseline", "run_the_loop",
                               "generate_research_ideas", "propose_research_idea",
                               "confirm_best", "stop", "research_task",
                               "update_research_plan", "retry_failed_action",
                               "submit_native_job", "inspect_native_job",
                               "adjust_native_job", "cancel_native_job", "review_report_demo",
                               "capture_environment_demo",
                               "submit_research_task", "cancel_research_task", "wait_for_jobs",
                               "configure_screening", "run_screening_trial", "inspect_screening"}


def test_runtime_infrastructure_failure_does_not_relaunch_or_call_fix(tmp_path, monkeypatch):
    from autosim.research.agent_client import AgentRuntimeClientError
    real = preparation(tmp_path)
    calls = []
    def denied():
        calls.append("choose")
        raise AgentRuntimeClientError("socket denied", status="infrastructure_blocked",
                                      failure_category="runtime_startup")
    monkeypatch.setattr(real, "choose", denied)
    monkeypatch.setattr(real, "_handoff_scheduler_decision_failure",
                        lambda **_: pytest.fail("runtime unavailable to all roles"))
    result = real.run(max_steps=8, max_relaunch=4)
    assert calls == ["choose"]
    assert result["status"] == "infrastructure_blocked"
    assert result["supervision"]["automatic_relaunches"] == 0


def test_fix_retry_replays_original_operation_once(tmp_path, monkeypatch):
    real = preparation(tmp_path)
    called = []
    monkeypatch.setattr(real, "_checkout_read_is_justified", lambda: True)
    monkeypatch.setattr(real, "_step_read_the_checkout", lambda: (
        called.append("survey") or {"outcome": "done", "because": "surveyed"}))
    real.steps = [{"step": "read_the_checkout", "outcome": "failed",
                   "receipt_ref": "action_receipts/" + "a" * 16 + ".json"}]
    real.fix_observation = {
        "status": "assessed", "assessment": "repair_attempted",
        "fix_attempt_id": "b" * 16,
        "failure_receipt_ref": "action_receipts/" + "a" * 16 + ".json"}
    result = real.do("retry_failed_action")
    assert result["outcome"] == "reverified"
    assert result["replayed_step"] == "read_the_checkout"
    assert called == ["survey"]
    real.steps.append({"step": "retry_failed_action", "fix_attempt_id": "b" * 16})
    assert real.do("retry_failed_action")["outcome"] == "not attempted"


def test_completed_detached_train_is_passed_to_initial_baseline_only(tmp_path, monkeypatch):
    from autosim.research import native_jobs

    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.interpreter = Path(sys.executable)
    real.stages = {"train": "verified", "evaluate": "verified"}
    monkeypatch.setattr(real, "_research_loop_ready", lambda: True)
    attempt_id = "a" * 32
    job_id = "b" * 32
    request = real.output / "native_jobs" / job_id / "request.json"
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps({"repo": str(real.repo), "run_id": real.run_id,
                                   "settings": {}, "source_identity": "frozen"}),
                       encoding="utf-8")
    monkeypatch.setattr(native_jobs, "source_identity", lambda *_: "frozen")
    monkeypatch.setattr(native_jobs, "status", lambda *_: {
        "status": "completed", "result": {"stage_result": {
            "stage": "train", "attempt_id": attempt_id}}})

    class Research:
        run_root = real.output / "research" / real.run_id
        execution_graph = None

        @staticmethod
        def _base_settings(settings):
            return settings

        @staticmethod
        def _verified_training_receipt(attempt, settings):
            return {"verified": True} if attempt == attempt_id and settings == {} else None

        def run(self, **kwargs):
            seen.append(kwargs)
            return {"rounds": [], "run_status": "completed", "objective": {}}

    seen = []
    monkeypatch.setattr(real, "_research_controller", lambda: (Research(), {}))
    result = real._step_run_the_loop(job_id=job_id)
    assert result["outcome"] == "no number came out"
    assert seen[0]["baseline_training_attempt_id"] == attempt_id
    assert real._step_run_the_loop(job_id="x" * 32)["outcome"] == "not attempted"


def test_candidate_fix_is_read_only_and_cannot_be_replayed_as_preparation(tmp_path):
    class Client(_Client):
        supports_agent_fix = True

        @contextmanager
        def as_role(self, role):
            assert role == "fix"
            yield

        def chat_with_metadata(self, system, user, **kwargs):
            self.seen.append((system, user, kwargs))
            return json.dumps({"assessment": "no_safe_repair",
                               "summary": "The original candidate was rolled back.",
                               "changes": [], "evidence_refs": []}), {}

    client = Client([])
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=client,
                       scouting=tmp_path / "run" / "scouting")
    action = {"step": "run_the_loop", "outcome": "yielded",
              "candidate_failure": {"idea_label": "bad_candidate"},
              "evidence_id": "a" * 32,
              "receipt_ref": "action_receipts/" + "b" * 16 + ".json"}
    observation = real._fix_failed_action(action_id="b" * 16, action=action)
    assert observation["assessment"] == "no_safe_repair"
    assert client.seen[-1][2]["read_only"] is True
    real.steps.append({"step": "run_the_loop", "receipt_ref": action["receipt_ref"]})
    real.fix_observation = {**observation, "assessment": "repair_attempted"}
    assert real._repair_retry_available() is False


def test_formal_run_local_declaration_store_rejects_symlink_escape(tmp_path, monkeypatch):
    from autosim.research import prepare as prepare_module

    output = tmp_path / "run"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "scouting").symlink_to(outside, target_is_directory=True)
    called = []
    monkeypatch.setattr(prepare_module.scout, "run",
                        lambda *args, **kwargs: called.append((args, kwargs)))
    real = Preparation(repo=tmp_path, output=output, client=_Client([]),
                       scouting=output / "scouting")

    result = real._step_declare()

    assert result["outcome"] == "blocked"
    assert "must not be a symlink" in result["because"]
    assert called == []


def test_action_model_budget_exhaustion_is_terminal_and_not_internal_error(
        tmp_path, monkeypatch):
    from autosim.research.agent_client import AgentRuntimeClientError

    client = _Client([{"do": "read_the_checkout", "why": "test the budget boundary"}])
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=client,
                       scouting=tmp_path / "run" / "scouting")

    def exhausted():
        raise AgentRuntimeClientError(
            "turn budget is exhausted", status="budget_exhausted",
            failure_category="provider_turn_budget",
            run_budget={"limit_usd": 2.0, "spent_usd": 2.0, "remaining_usd": 0.0})

    monkeypatch.setattr(real, "_step_read_the_checkout", exhausted)

    report = real.run(max_steps=3)

    assert report["status"] == "budget_exhausted"
    assert len(client.seen) == 1
    assert report["steps"][-1]["outcome"] == "exhausted"
    assert report["steps"][-1]["failure_category"] == "provider_turn_budget"
    assert report["steps"][-1]["run_budget"]["remaining_usd"] == 0.0


def test_scheduler_turn_budget_exhaustion_is_reported_without_another_choice(tmp_path):
    from autosim.research.agent_client import AgentRuntimeClientError

    class ExhaustedClient:
        model = "budget-fixture"

        def __init__(self):
            self.calls = 0

        def chat_with_metadata(self, *_args, **_kwargs):
            self.calls += 1
            raise AgentRuntimeClientError(
                "run model budget exhausted", status="budget_exhausted",
                failure_category="run_model_budget",
                run_budget={"limit_usd": 2.0, "spent_usd": 2.0, "remaining_usd": 0.0})

    client = ExhaustedClient()
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=client,
                       scouting=tmp_path / "run" / "scouting")

    report = real.run(max_steps=3)

    assert report["status"] == "budget_exhausted"
    assert report["steps"][-1]["outcome"] == "exhausted"
    assert report["steps"][-1]["failure_category"] == "run_model_budget"
    assert client.calls == 1


def test_autosota_monitor_assesses_failed_action_and_scheduler_retains_control(
        tmp_path, monkeypatch):
    class RoleClient:
        model = "role-fixture"

        def __init__(self):
            self.role = "scheduler"
            self.roles = []
            self.scheduler_payloads = []

        @contextmanager
        def as_role(self, role):
            previous = self.role
            self.role = role
            self.roles.append(role)
            try:
                yield
            finally:
                self.role = previous

        def chat_with_metadata(self, *_args, **_kwargs):
            if self.role == "scheduler":
                self.scheduler_payloads.append(_args[1])
                if len(self.scheduler_payloads) == 1:
                    return json.dumps({"do": "read_the_checkout",
                                       "why": "inspect the source before other work"}), {}
                return json.dumps({"do": "stop", "why": "the current evidence is sufficient"}), {}
            if self.role == "monitor":
                return json.dumps({"progress": "blocked",
                                   "summary": "The selected survey did not produce stage evidence.",
                                   "guidance": "Review the source-backed failure and consider a "
                                               "different investigation route.",
                                   "evidence_refs": ["README.md", "execution.json",
                                                     "../outside.txt", "/etc/passwd"]}), {}
            raise AssertionError(f"unexpected model role: {self.role}")

    client = RoleClient()
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=client,
                       scouting=tmp_path / "run" / "scouting")
    monkeypatch.setattr(real, "_step_read_the_checkout",
                        lambda: {"outcome": "blocked",
                                 "because": "the read-only source survey failed"})

    report = real.run(max_steps=2)

    assert report["status"] == "adaptation_unresolved"
    assert client.roles == ["scheduler", "resource", "monitor", "scheduler"]
    failed_step = next(row for row in report["steps"] if row["step"] == "read_the_checkout")
    assert failed_step["monitor"]["status"] == "assessed"
    assert failed_step["monitor"]["progress"] == "blocked"
    assert report["state"]["monitor_observation"]["guidance"].startswith("Review the source")
    assert report["state"]["monitor_observation"]["evidence_refs"] == [
        "README.md", "execution.json"]
    assert report["state"]["available"]
    scheduler_after_monitor = json.loads(client.scheduler_payloads[-1])
    assert scheduler_after_monitor["monitor_observation"]["progress"] == "blocked"
    run_doc = (tmp_path / "run" / "RUN.md").read_text(encoding="utf-8")
    assert "## AgentMonitor" in run_doc
    assert "Scheduler guidance (advisory)" in run_doc
    assert "AgentMonitor `blocked`" in run_doc
    assert "../outside.txt" not in run_doc
    assert "/etc/passwd" not in run_doc
    state = json.loads((tmp_path / "run" / "run_state.json").read_text(encoding="utf-8"))
    assert state["phases"]["monitor"]["latest_observation"]["action_id"] == \
        failed_step["receipt_ref"].split("/")[-1].removesuffix(".json")


def test_recoverable_preparation_failure_handoffs_monitor_then_fix_then_scheduler(
        tmp_path, monkeypatch):
    class RoleClient:
        model = "role-fixture"
        supports_agent_fix = True

        def __init__(self):
            self.role = "scheduler"
            self.roles = []
            self.scheduler_payloads = []

        @contextmanager
        def as_role(self, role):
            previous = self.role
            self.role = role
            self.roles.append(role)
            try:
                yield
            finally:
                self.role = previous

        def chat_with_metadata(self, _system, user, **_kwargs):
            if self.role == "scheduler":
                self.scheduler_payloads.append(user)
                if len(self.scheduler_payloads) == 1:
                    return json.dumps({"do": "read_the_checkout",
                                       "why": "survey before declaration"}), {}
                return json.dumps({"do": "stop", "why": "review the repair before resuming"}), {}
            if self.role == "monitor":
                return json.dumps({"progress": "stalled",
                                   "summary": "The source survey raised before recording stages.",
                                   "guidance": "Ask Fix to inspect the failing path, then re-evaluate.",
                                   "evidence_refs": ["execution.json"]}), {}
            if self.role == "fix":
                assert "failure_receipt_ref" in user
                return json.dumps({"assessment": "repair_attempted",
                                   "summary": ("Inspected the failed survey path.\n"
                                               "## Injected | [open](https://invalid.test)"),
                                   "changes": ["Added a missing-file guard.\n"
                                               "| [open](https://invalid.test)"],
                                   "evidence_refs": ["execution.json"]}), {
                                       "role": "fix", "turn_id": "a" * 32,
                                       "process_ref": f"agent/processes/{'a' * 32}.json"}
            raise AssertionError(f"unexpected model role: {self.role}")

    client = RoleClient()
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=client,
                       scouting=tmp_path / "run" / "scouting")

    def failed_survey():
        raise RuntimeError("survey failed before stage evidence was written")

    monkeypatch.setattr(real, "_step_read_the_checkout", failed_survey)
    report = real.run(max_steps=3)

    assert report["status"] == "adaptation_unresolved"
    assert report["actions_this_cycle"] == 2
    assert client.roles == ["scheduler", "resource", "monitor", "fix", "scheduler"]
    failed = next(row for row in report["steps"] if row["step"] == "read_the_checkout")
    assert failed["fix"]["assessment"] == "repair_attempted"
    assert report["state"]["fix_observation"]["failure_receipt_ref"] == \
        failed["receipt_ref"]
    assert report["state"]["fix_observation"]["evidence_validation"] == \
        "model_report_not_independently_verified"
    assert report["state"]["fix_observation"]["runtime_evidence_refs"] == [
        f"agent/events.jsonl#turn_id={'a' * 32}", f"agent/processes/{'a' * 32}.json"]
    assert json.loads(client.scheduler_payloads[-1])["fix_observation"]["assessment"] == \
        "repair_attempted"
    assert "read_the_checkout" in \
        json.loads(client.scheduler_payloads[-1])["available"]
    event_state = json.loads((tmp_path / "run" / "run_state.json").read_text())
    assert event_state["phases"]["fix"]["latest_observation"]["failed_action_id"] == \
        failed["receipt_ref"].split("/")[-1].removesuffix(".json")
    run_doc = (tmp_path / "run" / "RUN.md").read_text(encoding="utf-8")
    assert "## AgentFix" in run_doc
    assert "not independent proof" in run_doc
    assert "## Injected" not in run_doc
    assert "[open](https://invalid.test)" not in run_doc


def test_fix_budget_exhaustion_is_terminal_without_another_scheduler_turn(
        tmp_path, monkeypatch):
    from autosim.research.agent_client import AgentRuntimeClientError

    class RoleClient:
        model = "role-fixture"
        supports_agent_fix = True

        def __init__(self):
            self.role = "scheduler"
            self.roles = []

        @contextmanager
        def as_role(self, role):
            previous = self.role
            self.role = role
            self.roles.append(role)
            try:
                yield
            finally:
                self.role = previous

        def chat_with_metadata(self, _system, _user, **_kwargs):
            if self.role == "scheduler":
                return json.dumps({"do": "read_the_checkout",
                                   "why": "inspect the failing source survey"}), {}
            if self.role == "monitor":
                return json.dumps({"progress": "uncertain",
                                   "summary": "The source survey failed.",
                                   "guidance": "Fix may inspect the recorded failure.",
                                   "evidence_refs": ["execution.json"]}), {}
            if self.role == "fix":
                raise AgentRuntimeClientError(
                    "run model budget exhausted", status="budget_exhausted",
                    failure_category="run_model_budget",
                    run_budget={"limit_usd": 2.0, "spent_usd": 2.0,
                                "remaining_usd": 0.0})
            raise AssertionError(f"unexpected role: {self.role}")

    client = RoleClient()
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=client,
                       scouting=tmp_path / "run" / "scouting")
    monkeypatch.setattr(real, "_step_read_the_checkout",
                        lambda: (_ for _ in ()).throw(RuntimeError("survey failed")))

    report = real.run(max_steps=4)

    assert report["status"] == "budget_exhausted"
    assert report["actions_this_cycle"] == 1
    assert client.roles == ["scheduler", "resource", "monitor", "fix"]
    assert report["state"]["fix_observation"]["status"] == "budget_exhausted"
    assert report["steps"][-1]["step"] == "read_the_checkout"
    assert report["steps"][-1]["fix"]["status"] == "budget_exhausted"


def test_monitor_budget_exhaustion_stops_before_another_scheduler_turn(tmp_path, monkeypatch):
    from autosim.research.agent_client import AgentRuntimeClientError

    class RoleClient:
        model = "role-fixture"

        def __init__(self):
            self.role = "scheduler"
            self.scheduler_calls = 0

        @contextmanager
        def as_role(self, role):
            previous = self.role
            self.role = role
            try:
                yield
            finally:
                self.role = previous

        def chat_with_metadata(self, *_args, **_kwargs):
            if self.role == "scheduler":
                self.scheduler_calls += 1
                return json.dumps({"do": "read_the_checkout",
                                   "why": "inspect the source"}), {}
            if self.role == "monitor":
                raise AgentRuntimeClientError(
                    "no full turn reservation remains", status="budget_exhausted",
                    failure_category="run_model_budget",
                    run_budget={"limit_usd": 1.0, "spent_usd": 0.8,
                                "remaining_usd": 0.2})
            raise AssertionError(f"unexpected role: {self.role}")

    client = RoleClient()
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=client,
                       scouting=tmp_path / "run" / "scouting")
    monkeypatch.setattr(real, "_step_read_the_checkout",
                        lambda: {"outcome": "blocked", "because": "fixture failure"})

    report = real.run(max_steps=3)

    assert report["status"] == "budget_exhausted"
    assert client.scheduler_calls == 1
    assert report["state"]["monitor_observation"]["status"] == "budget_exhausted"
    assert report["state"]["monitor_observation"]["run_budget"]["remaining_usd"] == 0.2


def test_scheduler_runtime_error_is_assessed_then_scheduler_retries(tmp_path):
    class RoleClient:
        model = "scheduler-recovery-fixture"

        def __init__(self):
            self.role = "scheduler"
            self.roles = []
            self.scheduler_calls = 0
            self.scheduler_payloads = []

        @contextmanager
        def as_role(self, role):
            previous = self.role
            self.role = role
            self.roles.append(role)
            try:
                yield
            finally:
                self.role = previous

        def chat_with_metadata(self, _system, user, **_kwargs):
            if self.role == "scheduler":
                self.scheduler_calls += 1
                self.scheduler_payloads.append(user)
                if self.scheduler_calls == 1:
                    raise RuntimeError("temporary provider connection reset")
                return json.dumps({"do": "stop", "why": "the retry can now assess state"}), {}
            if self.role == "monitor":
                return json.dumps({"progress": "progress",
                                   "summary": "The Scheduler request failed before any action ran.",
                                   "guidance": "Re-read durable state and decide again.",
                                   "evidence_refs": ["run_state.json"]}), {}
            raise AssertionError(f"unexpected role: {self.role}")

    client = RoleClient()
    real = preparation(tmp_path)
    real.client = client

    report = real.run(max_steps=2)

    assert client.roles == ["scheduler", "monitor", "scheduler"]
    assert [row["step"] for row in report["steps"]] == ["choose", "stop"]
    failed = report["steps"][0]
    assert failed["outcome"] == "could not be asked"
    assert "temporary provider connection reset" in failed["because"]
    assert failed["monitor"]["progress"] == "progress"
    assert report["state"]["monitor_observation"]["status"] == "assessed"
    assert json.loads(client.scheduler_payloads[-1])["monitor_observation"]["summary"].startswith(
        "The Scheduler request failed")
    event_log = json.loads((tmp_path / "out" / "run_events.json").read_text())
    assert any(row["phase"] == "monitor" and
               row["event"] == "failed_action_assessed" for row in event_log["rows"])


def test_rejected_scheduler_protocol_is_assessed_before_retry(tmp_path):
    class RoleClient:
        model = "scheduler-protocol-fixture"

        def __init__(self):
            self.role = "scheduler"
            self.roles = []
            self.scheduler_calls = 0

        @contextmanager
        def as_role(self, role):
            previous = self.role
            self.role = role
            self.roles.append(role)
            try:
                yield
            finally:
                self.role = previous

        def chat_with_metadata(self, _system, _user, **_kwargs):
            if self.role == "scheduler":
                self.scheduler_calls += 1
                if self.scheduler_calls <= 2:
                    return "not a JSON object", {}
                return json.dumps({"do": "stop", "why": "state was reconsidered"}), {}
            if self.role == "monitor":
                return json.dumps({"progress": "uncertain",
                                   "summary": "Two Scheduler replies failed the decision contract.",
                                   "guidance": "Use the persisted state and try a fresh decision.",
                                   "evidence_refs": ["run_state.json"]}), {}
            raise AssertionError(f"unexpected role: {self.role}")

    client = RoleClient()
    real = preparation(tmp_path)
    real.client = client

    report = real.run(max_steps=2)

    assert client.roles == ["scheduler", "scheduler", "monitor", "scheduler"]
    assert report["steps"][0]["outcome"] == "decision_rejected"
    assert report["steps"][0]["monitor"]["status"] == "assessed"
    assert report["steps"][-1]["outcome"] == "stopped"


def test_scheduler_budget_exhaustion_does_not_call_monitor_or_retry(tmp_path):
    from autosim.research.agent_client import AgentRuntimeClientError

    class RoleClient:
        model = "scheduler-budget-fixture"

        def __init__(self):
            self.role = "scheduler"
            self.roles = []

        @contextmanager
        def as_role(self, role):
            previous = self.role
            self.role = role
            self.roles.append(role)
            try:
                yield
            finally:
                self.role = previous

        def chat_with_metadata(self, *_args, **_kwargs):
            raise AgentRuntimeClientError(
                "no model budget remains", status="budget_exhausted",
                failure_category="run_model_budget",
                run_budget={"limit_usd": 1.0, "spent_usd": 1.0,
                            "remaining_usd": 0.0})

    client = RoleClient()
    real = preparation(tmp_path)
    real.client = client

    report = real.run(max_steps=4)

    assert report["status"] == "budget_exhausted"
    assert client.roles == ["scheduler"]
    assert report["steps"][-1]["outcome"] == "exhausted"


def test_autosota_supervisor_relaunches_a_bounded_scheduler_segment(tmp_path, monkeypatch):
    class RoleClient:
        model = "role-fixture"

        def __init__(self):
            self.role = "scheduler"
            self.calls = []

        @contextmanager
        def as_role(self, role):
            previous = self.role
            self.role = role
            try:
                yield
            finally:
                self.role = previous

        def chat_with_metadata(self, *_args, **_kwargs):
            self.calls.append(self.role)
            if self.role == "scheduler":
                if self.calls.count("scheduler") == 1:
                    return json.dumps({"do": "read_the_checkout",
                                       "why": "record the source evidence"}), {}
                return json.dumps({"do": "stop", "why": "the run is intentionally bounded"}), {}
            raise AssertionError(f"unexpected role: {self.role}")

    client = RoleClient()
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=client,
                       scouting=tmp_path / "run" / "scouting")
    monkeypatch.setattr(real, "_step_read_the_checkout",
                        lambda: {"outcome": "done", "because": "source facts persisted"})

    report = real.run(max_steps=1, max_relaunch=1)

    assert [row["step"] for row in report["steps"]] == ["read_the_checkout", "stop"]
    assert report["status"] == "adaptation_unresolved"
    assert report["supervision"] == {
        "scheduler_segments": 2, "automatic_relaunches": 1,
        "max_relaunch": 1, "stagnation_reviews": 0,
    }
    assert client.calls == ["scheduler", "scheduler"]


def test_completed_research_runs_one_read_only_supervisor_and_caches_by_packet(
        tmp_path, monkeypatch):
    from autosim.research import prepare as prepare_module

    class SupervisorClient:
        model = "review-fixture"

        def __init__(self):
            self.role = "scheduler"
            self.roles = []
            self.calls = 0
            self.prompts = []

        @contextmanager
        def as_role(self, role):
            prior = self.role
            self.role = role
            self.roles.append(role)
            try:
                yield
            finally:
                self.role = prior

        def chat_with_metadata(self, system, user, **kwargs):
            assert self.role == "supervisor"
            assert "held-out" in system.lower()
            assert "0.991" not in user
            assert kwargs["timeout"] > 0
            self.calls += 1
            self.prompts.append(user)
            return json.dumps({"verdict": "real", "summary": "The receipt is auditable.",
                               "findings": [],
                               "evidence_refs": ["measurement:baseline"]}), {
                                   "model": self.model, "role": self.role,
                                   "status": "completed", "total_cost_usd": 0.01}

    client = SupervisorClient()
    output = tmp_path / "run"
    real = Preparation(repo=tmp_path, output=output, client=client,
                       scouting=output / "scouting")
    packet = {"run_id": "run-a", "confirmation": {"recorded": True,
                                                       "metric_value": 0.991},
              "evidence_refs": ["research_report", "measurement:baseline"],
              "measurements": [{"label": "baseline", "metric_value": 0.4,
                                "receipt_verification": {"status": "consistent"}}]}
    monkeypatch.setattr(prepare_module.supervisor, "build_audit_packet",
                        lambda **_kwargs: (packet, packet["evidence_refs"]))

    first = real._review_final_study()
    second = real._review_final_study()

    assert first["verdict"] == "real"
    assert first["packet_sha256"] == second["packet_sha256"]
    assert first["review_sha256"] == second["review_sha256"]
    assert client.roles == ["supervisor"]
    assert client.calls == 1
    projection = json.loads((output / "evaluation_verdict.json").read_text())
    assert projection["review"]["verdict"] == "real"
    assert projection["reviewer"]["role"] == "supervisor"
    assert real.state()["evaluation_verdict"]["review_sha256"] == first["review_sha256"]
    state = json.loads((output / "run_state.json").read_text())
    assert state["phases"]["supervisor"]["status"] == "real"
    assert state["phases"]["supervisor"]["evaluation_verdict"]["verdict"] == "real"


def test_supervisor_failure_is_persisted_as_uncertain_not_execution_failure(
        tmp_path, monkeypatch):
    from autosim.research import prepare as prepare_module

    class FailedSupervisor:
        model = "review-fixture"

        def chat_with_metadata(self, *_args, **_kwargs):
            raise TimeoutError("provider did not return")

    output = tmp_path / "run"
    real = Preparation(repo=tmp_path, output=output, client=FailedSupervisor(),
                       scouting=output / "scouting")
    packet = {"evidence_refs": [], "measurements": []}
    monkeypatch.setattr(prepare_module.supervisor, "build_audit_packet",
                        lambda **_kwargs: (packet, []))

    review = real._review_final_study()

    assert review["verdict"] == "uncertain"
    assert "TimeoutError" in review["summary"]
    assert json.loads((output / "evaluation_verdict.json").read_text())["review"][
        "verdict"] == "uncertain"


def test_monitor_reviews_two_stagnant_segments_before_scheduler_resumes(tmp_path, monkeypatch):
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=_Client([]),
                       scouting=tmp_path / "run" / "scouting")
    segment_observations = []
    monitor_actions = []

    def bounded_segment(*, max_steps):
        segment_observations.append(real.monitor_observation)
        if len(segment_observations) == 3:
            return {"status": "paused", "steps": real.steps, "state": real.state()}
        return {"status": "action_limit", "steps": real.steps, "state": real.state()}

    def assess_stagnation(*, action_id, action):
        monitor_actions.append((action_id, action))
        observation = {"status": "assessed", "progress": "stalled",
                       "summary": "Two Scheduler segments made no decision-relevant change.",
                       "guidance": "Inspect a different source of evidence."}
        real.monitor_observation = observation
        return observation

    monkeypatch.setattr(real, "_run_cycle", bounded_segment)
    monkeypatch.setattr(real, "_monitor_failed_action", assess_stagnation)

    report = real.run(max_steps=1, max_relaunch=2)

    assert report["status"] == "paused"
    assert len(monitor_actions) == 1
    assert monitor_actions[0][1]["step"] == "scheduler_segment"
    assert monitor_actions[0][1]["outcome"] == "no_progress"
    assert segment_observations[0] == {}
    assert segment_observations[1] == {}
    assert segment_observations[2]["progress"] == "stalled"
    assert report["supervision"] == {
        "scheduler_segments": 3, "automatic_relaunches": 2,
        "max_relaunch": 2, "stagnation_reviews": 1,
    }


def test_l2_measurement_during_paused_research_is_not_reported_as_complete(
        tmp_path, monkeypatch):
    real = preparation(tmp_path, [{"do": "run_the_loop",
                                   "why": "continue the native research loop"}])
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    real.interpreter = Path(sys.executable)
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real._metric_bound = lambda: True

    def yielded_action():
        root = real.output / "research" / real.run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "research_report.json").write_text(json.dumps({
            "run_id": real.run_id, "repo": str(real.repo),
            "run_status": "paused", "next_round": 1, "planned_rounds": 2,
            "rounds": [{"status": "measured", "metric_value": 0.5}],
            "best": {"name": "baseline"},
        }), encoding="utf-8")
        return {"outcome": "yielded", "because": "the baseline was measured; next round pending",
                "verification_level": "L2"}

    monkeypatch.setattr(real, "_step_run_the_loop", yielded_action)

    report = real.run(max_steps=1, max_relaunch=0)

    assert report["status"] == "paused"
    assert report["steps"][0]["verification_level"] == "L2"
    assert report["state"]["research_progress"]["status"] == "paused"
    assert report["pause_reason"].startswith("automatic Scheduler relaunch limit reached")


@pytest.mark.parametrize(("failure", "expected_role"), [
    (None, "init"),
    ({"stage": "train", "outcome": "failed", "because": "native import failed",
      "failure_kind": "stage_failure", "evidence_ref": "train/attempts/a/output.log"},
     "fix"),
])
def test_environment_reverification_routes_native_stage_contradiction_to_fix(
        tmp_path, monkeypatch, failure, expected_role):
    from autosim.research import prepare as prepare_module

    class RoleClient(_Client):
        def __init__(self):
            super().__init__([])
            self.role = "scheduler"
            self.role_calls = []

        @contextmanager
        def as_role(self, role):
            previous = self.role
            self.role = role
            self.role_calls.append(role)
            try:
                yield
            finally:
                self.role = previous

    client = RoleClient()
    real = Preparation(repo=tmp_path, output=tmp_path / "run", client=client,
                       scouting=tmp_path / "run" / "scouting")
    monkeypatch.setattr(prepare_module, "_last_stage_failure_after_build",
                        lambda *_args: failure)
    monkeypatch.setattr(real, "_step_build_the_environment",
                        lambda: {"outcome": "done", "because": "probe completed"})

    result = real.do("build_the_environment")

    assert client.role_calls == [expected_role]
    if failure:
        assert result["delegated_role"] == "fix"
        assert result["fix_handoff"] == failure
    else:
        assert "fix_handoff" not in result


def test_controller_prompt_requires_exact_declared_axis_names_for_parameter_ideas():
    assert "Copy each axis name exactly as shown in `declared_space`" in STEP_SYSTEM
    assert "not add a `train.`/`training.` prefix" in STEP_SYSTEM
    assert "based on its section" in STEP_SYSTEM
    assert 'use `{{"change":{{"n_epochs":2}}' in STEP_SYSTEM


def test_controller_prompt_anchors_quantitative_ideas_to_actual_training_receipt():
    assert "last_observation.actual_training" in STEP_SYSTEM
    assert "what actually produced the measured" in STEP_SYSTEM
    assert "A surveyed invocation" in STEP_SYSTEM


def test_busy_gpu_is_persistable_and_blocks_probe_or_command_execution(tmp_path, monkeypatch):
    from autosim.research import prepare as prepare_module
    from autosim.research.devices import NoCompatibleDevice

    def busy(**_kwargs):
        raise NoCompatibleDevice("all visible GPUs are busy",
                                 evidence={"busy_devices": [{"index": 0}]})

    monkeypatch.setattr(prepare_module, "decide", busy)
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client([]),
                       scouting=tmp_path / "scouting")

    assert real.decision.device == "unavailable"
    assert real.decision.evidence["resource_unavailable"] is True
    assert real._step_build_the_environment()["outcome"] == "not attempted"  # CPU setup is allowed; declaration is missing.
    assert real._step_derive_a_command(stage="train")["outcome"] == "blocked"
    assert not (real.output / "environment.json").exists()
    assert not (real.output / "derivation_attempts" / "train.json").exists()


def test_the_state_is_read_from_the_records(tmp_path):
    """Every line is a reading, not a decision: what exists, what was refused, and why."""
    real = preparation(tmp_path)
    real.output.mkdir(parents=True)
    (real.output / "execution.json").write_text(json.dumps(
        {"stages": {"train": {"available": True}, "collect": {"available": False}}}),
        encoding="utf-8")
    (real.output / "derived_stages.json").write_text(json.dumps({"train": {"source": "x"}}),
                                                    encoding="utf-8")
    state = real.state()
    assert state["stages_the_checkout_has"] == {"train": "kept from an earlier run",
                                                "collect": "no command"}
    assert state["available"] == [name for name in OPERATIONS
                                      if name not in {"research_task", "update_research_plan", "review_report_demo", "capture_environment_demo",
                                                      "retry_failed_action", "submit_research_task",
                                                      "cancel_research_task", "wait_for_jobs",
                                                      "configure_screening", "run_screening_trial", "inspect_screening",
                                                      "submit_native_job", "inspect_native_job",
                                                      "adjust_native_job", "cancel_native_job",
                                                       "confirm_best", "recover_unscored_baseline",
                                                       "build_the_environment", "derive_a_command",
                                                       "bind_metric", "reconcile_interrupted_action",
                                                       "discard_interrupted_candidate",
                                                       "run_the_loop", "generate_research_ideas",
                                                   "propose_research_idea"}]


def test_action_availability_tracks_evidence_and_readiness(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.steps = [{"step": "declare", "outcome": "done"}]

    assert "declare" not in real.state()["available"]
    assert real.do("declare")["outcome"] == "not attempted"
    real.steps.append({"step": "read_the_checkout", "outcome": "done"})
    assert "declare" in real.state()["available"]
    real.steps.append({"step": "declare", "outcome": "done"})
    assert "declare" not in real.state()["available"]
    real.steps.append({"step": "build_the_environment", "outcome": "failed"})
    assert "declare" in real.state()["available"]

    unavailable = real.state()["available"]
    assert "bind_metric" not in unavailable
    assert "run_the_loop" not in unavailable
    assert "reconcile_interrupted_action" not in unavailable
    assert real.do("bind_metric")["outcome"] == "not attempted"

    real.execution = {"stages": {"evaluate": {"available": True}}}
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real.output.mkdir(parents=True)
    attempts = real.output / "derivation_attempts"
    attempts.mkdir()
    (attempts / "evaluate.json").write_text(json.dumps({"attempts": [{
        "status": "accepted", "said": "native_success=0.5"}]}), encoding="utf-8")
    assert "bind_metric" in real.state()["available"]
    assert "run_the_loop" not in real.state()["available"]
    (attempts / "evaluate.json").write_text(json.dumps({"attempts": [
        {"status": "accepted", "said": "native_success=0.5"},
        {"status": "rejected", "said": "the latest score attempt did not run"}]}),
        encoding="utf-8")
    assert "bind_metric" not in real.state()["available"]
    (attempts / "evaluate.json").write_text(json.dumps({"attempts": [
        {"status": "accepted", "said": "native_success=0.5"},
        {"status": "accepted", "said": "native_success=0.75"}]}), encoding="utf-8")
    assert "bind_metric" in real.state()["available"]

    real.interpreter = Path(sys.executable)
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    assert "run_the_loop" in real.state()["available"]


def test_benchmark_name_follows_nested_isolated_copy_provenance(tmp_path):
    """A second isolated copy is still ManiSkill, not the literal name `checkout`."""
    original = tmp_path / "sources" / "maniskill"
    first_run = tmp_path / "run_a"
    first = first_run / "checkout"
    second_run = tmp_path / "run_b"
    second = second_run / "checkout"
    for path in (original, first, second):
        path.mkdir(parents=True)
    (first_run / "workspace_snapshot.json").write_text(json.dumps({
        "source": str(original), "destination": str(first)}), encoding="utf-8")
    (second_run / "workspace_snapshot.json").write_text(json.dumps({
        "source": str(first), "destination": str(second)}), encoding="utf-8")
    real = Preparation(repo=second, output=second_run, client=_Client([]),
                       scouting=tmp_path / "scouting")

    assert real._benchmark_name() == "maniskill"
    real._refresh_document(status="created")
    document = (second_run / "RUN.md").read_text(encoding="utf-8")
    assert "# maniskill 研究进展" in document
    assert "Repository identity: `maniskill`" in document
    assert f"Source repository: `{original}`" in document
    assert f"Execution checkout: `{second}`" in document


def test_report_does_not_follow_snapshot_source_when_destination_mismatches(tmp_path):
    original = tmp_path / "sources" / "unrelated-benchmark"
    checkout = tmp_path / "run" / "checkout"
    wrong_destination = tmp_path / "other-run" / "checkout"
    original.mkdir(parents=True)
    checkout.mkdir(parents=True)
    (checkout.parent / "workspace_snapshot.json").write_text(json.dumps({
        "source": str(original), "destination": str(wrong_destination)}),
        encoding="utf-8")
    real = Preparation(repo=checkout, output=checkout.parent, client=_Client([]),
                       scouting=tmp_path / "scouting")

    assert real._source_repository() == checkout
    real._refresh_document(status="created")
    document = (checkout.parent / "RUN.md").read_text(encoding="utf-8")
    assert "Repository identity: `checkout`" in document
    assert f"Source repository: `{original}`" not in document
    assert f"Execution checkout: `{checkout}`" in document


def test_missing_declaration_hides_impossible_steps_and_caps_repeated_drafts(tmp_path):
    real = preparation(tmp_path)
    real.execution = {"stages": {"train": {"available": True}}}
    real.steps = [
        {"step": "read_the_checkout", "outcome": "done"},
        {"step": "declare", "outcome": "no usable declaration"},
        {"step": "declare", "outcome": "raised"},
    ]

    available = real.state()["available"]
    assert "build_the_environment" not in available
    assert "derive_a_command" not in available
    assert "bind_metric" not in available
    assert "declare" not in available
    assert "read_the_checkout" not in available
    assert "stop" in available


def test_state_offers_baseline_recovery_only_for_matching_receipt_and_selection(tmp_path):
    real = preparation(tmp_path)
    real.interpreter = Path(sys.executable)
    real.output.mkdir(parents=True)
    research_root = real.output / "research" / real.run_id
    attempt_id = "a" * 32
    attempt_dir = research_root / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    protocol = research_root / "comparison_protocol.json"
    protocol.write_text(json.dumps({"target": "evaluate"}), encoding="utf-8")
    settings = {"task": "PushCube-v1", "episodes": 2}
    receipt = {
        "run_id": real.run_id, "attempt_id": attempt_id, "node_id": "train",
        "stage": "train", "status": "completed", "returncode": 0,
        "termination_reason": "normal_exit", "settings_digest": object_digest(settings),
        "working_directory": str(tmp_path), "argv": [sys.executable, str(tmp_path / "ppo.py")],
        "artifact": {"checked": True, "matched": 2},
        "training_progress": {"status": "observed"},
    }
    (attempt_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    selection_ref = f"research/{real.run_id}/attempts/{attempt_id}/artifact_selection.json"
    (attempt_dir / "artifact_selection.json").write_text(json.dumps({
        "attempt_id": attempt_id, "status": "abstained"}), encoding="utf-8")
    measurement = {"label": "baseline", "ok": False, "where": "policy_artifact",
                   "status": "unscored", "settings": settings,
                   "training_outcome": {"attempt_id": attempt_id},
                   "artifact_selection": {"selection_ref": selection_ref}}
    (research_root / "measurements").mkdir()
    (research_root / "measurements" / "baseline.json").write_text(
        json.dumps(measurement), encoding="utf-8")
    history = [{"round": 0, "label": "baseline", "measured": False,
                "failure": {"stage": "policy_artifact"}}]
    (research_root / "controller_session.json").write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id, "repository": str(tmp_path),
        "status": "paused", "history": history, "next_round": 1}), encoding="utf-8")
    (research_root / "research_report.json").write_text(json.dumps({
        "run_id": real.run_id, "repo": str(tmp_path), "run_status": "paused",
        "rounds": history}), encoding="utf-8")

    state = real.state()

    assert state["research_progress"]["unscored_baseline_recovery"]["available"] is True
    assert "recover_unscored_baseline" in state["available"]
    assert "run_the_loop" not in state["available"]
    assert "already verified" in state["to_measure"]["because"]
    guarded = real.do("run_the_loop")
    assert guarded["outcome"] == "not attempted"
    assert "without retraining" in guarded["because"]

    # Terminal research progress must not hide a receipt-bound recovery when resuming the
    # outer controller with --keep-only. This is the v25 path: research stopped after an
    # unscored baseline and an unmeasured protocol-rejected round.
    session_path = research_root / "controller_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["status"] = "completed"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    report_path = research_root / "research_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["run_status"] = "completed"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    real.keep_only = True
    assert real.choose()["do"] == "recover_unscored_baseline"

    (attempt_dir / "receipt.json").write_text(json.dumps({**receipt,
        "settings_digest": "wrong"}), encoding="utf-8")
    blocked = real.state()
    assert blocked["research_progress"]["unscored_baseline_recovery"]["available"] is False
    assert "recover_unscored_baseline" not in blocked["available"]


def test_state_offers_eval_only_recovery_for_exact_prestart_gpu_refusal(tmp_path):
    from autosim.research.compute_decision import ComputeDecision
    from autosim.research.experiment_bundle import artifact_identity

    real = preparation(tmp_path)
    real.interpreter = Path(sys.executable)
    real.decision = ComputeDecision(
        device="cuda:0", device_index=0,
        evidence={"selected_device": {"index": 0, "uuid": "GPU-test-5090"}})
    root = real.output / "research" / real.run_id
    attempts = root / "attempts"
    train_id, eval_id = "a" * 32, "b" * 32
    train_dir = attempts / train_id
    eval_dir = attempts / eval_id
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    protocol = root / "comparison_protocol.json"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(json.dumps({"target": "evaluate"}), encoding="utf-8")
    settings = {"task": "PushCube-v1"}
    source_policy = tmp_path / "models" / "final.pt"
    source_policy.parent.mkdir()
    source_policy.write_bytes(b"fresh policy")
    checkpoint = root / "experiments" / "baseline" / train_id / "policy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(source_policy.read_bytes())
    identity_field, identity = artifact_identity(checkpoint)
    train_receipt = {
        "run_id": real.run_id, "attempt_id": train_id, "node_id": "train",
        "stage": "train", "status": "completed", "returncode": 0,
        "termination_reason": "normal_exit", "settings_digest": object_digest(settings),
        "comparison_protocol_sha256": __import__("hashlib").sha256(
            protocol.read_bytes()).hexdigest(),
        "working_directory": str(tmp_path), "argv": [sys.executable, str(tmp_path / "ppo.py")],
        "gpu_device_uuid": "GPU-test-5090",
        "artifact": {"checked": True, "matched": 1,
                     "candidate_paths": ["models/final.pt"]},
        "training_progress": {"status": "observed"},
    }
    (train_dir / "receipt.json").write_text(json.dumps(train_receipt), encoding="utf-8")
    blocked = {
        "run_id": real.run_id, "attempt_id": eval_id, "node_id": "evaluate",
        "status": "blocked", "ran": False, "command_started": False,
        "termination_reason": "gpu_unavailable",
        "settings_digest": object_digest(settings),
    }
    (eval_dir / "receipt.json").write_text(json.dumps(blocked), encoding="utf-8")
    measurement = {
        "label": "baseline", "ok": False, "where": "evaluate",
        "settings": settings,
        "train": {"attempt_id": train_id, "returncode": 0},
        "evaluate": {"attempt_id": eval_id, "returncode": None},
        "policy_artifact": {"path": str(checkpoint), "source": str(source_policy),
                            identity_field: identity},
    }
    (root / "measurements").mkdir(parents=True)
    (root / "measurements" / "baseline.json").write_text(
        json.dumps(measurement), encoding="utf-8")
    history = [{"round": 0, "label": "baseline", "measured": False,
                "failure": {"stage": "evaluate", "attempt_id": eval_id}}]
    (root / "controller_session.json").write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id, "repository": str(real.repo),
        "status": "paused", "history": history}), encoding="utf-8")
    (root / "research_report.json").write_text(json.dumps({
        "run_id": real.run_id, "repo": str(real.repo), "run_status": "paused",
        "rounds": history}), encoding="utf-8")

    recovery = real._unscored_baseline_recovery_status()
    assert recovery["available"] is True, recovery
    assert recovery["recovery_kind"] == "reevaluate_after_prestart_resource_block"
    assert recovery["attempt_id"] == train_id
    assert recovery["blocked_evaluation_attempt_id"] == eval_id
    assert "recover_unscored_baseline" in real.state()["available"]

    real.decision.evidence["selected_device"]["uuid"] = "GPU-another-card"
    mismatch = real._unscored_baseline_recovery_status()
    assert mismatch["available"] is False
    assert "physical GPU" in mismatch["why"]

    blocked["command_started"] = True
    (eval_dir / "receipt.json").write_text(json.dumps(blocked), encoding="utf-8")
    refused = real._unscored_baseline_recovery_status()
    assert refused["available"] is False
    assert "before process start" in refused["why"]

    blocked["command_started"] = False
    (eval_dir / "receipt.json").write_text(json.dumps(blocked), encoding="utf-8")
    real.decision.evidence["selected_device"]["uuid"] = "GPU-test-5090"
    original = dict(measurement)
    backup_path = train_dir / "baseline_unscored.json"
    backup_path.write_text(json.dumps(original), encoding="utf-8")
    failed_recovery = {
        "training_attempt_id": train_id, "training_reused": True,
        "kind": "reevaluate_after_prestart_resource_block",
        "superseded_evaluation_attempt_id": eval_id,
        "original_measurement_ref": f"attempts/{train_id}/baseline_unscored.json",
    }
    current = {
        "label": "baseline", "ok": False, "where": "policy_archive",
        "settings": settings,
        "archive_error": "FileExistsError: archive destination already exists",
        "why": "policy bytes could not be frozen before evaluation",
        "recovery": failed_recovery,
    }
    baseline_path = root / "measurements" / "baseline.json"
    baseline_path.write_text(json.dumps(current), encoding="utf-8")
    (train_dir / "baseline_recovery.json").write_text(json.dumps({
        "status": "failed", "recovery_kind":
        "reevaluate_after_prestart_resource_block", "training_attempt_id": train_id,
        "original_measurement_sha256": object_digest(original),
        "measurement_sha256": object_digest(current), "evaluation_attempt_id": None,
    }), encoding="utf-8")

    retry = real._unscored_baseline_recovery_status()
    assert retry["available"] is True, retry
    assert retry["recovery_attempt"] == 2
    assert retry["recovery_state_ref"].endswith("baseline_recovery_attempt_2.json")

    (train_dir / "baseline_recovery_attempt_2.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8")
    refused_repeat = real._unscored_baseline_recovery_status()
    assert refused_repeat["available"] is False
    assert "already exists" in refused_repeat["why"]


def test_controller_state_distinguishes_surveyed_candidates_from_verified_commands(tmp_path):
    real = preparation(tmp_path)
    real.interpreter_hint = Path(sys.executable)
    real.output.mkdir(parents=True)
    (real.output / "execution.json").write_text(json.dumps({"stages": {
        "train": {"available": True, "entrypoint": "examples/ppo.py",
                  "invocation": "python examples/ppo.py --train"},
        "evaluate": {"available": True, "entrypoint": "examples/ppo.py",
                     "invocation": "python examples/ppo.py --evaluate"},
    }}), encoding="utf-8")
    (real.output / "path_selection_attempts.json").write_text(json.dumps({
        "input_digest": "older-survey", "rows": [{"stages": ["train", "evaluate"],
            "why_rejected": "the previous survey paired ACT with PPO"}]}), encoding="utf-8")

    state = real.state()

    assert state["stages_the_checkout_has"] == {"train": "no command",
                                                "evaluate": "no command"}
    assert state["surveyed_stages"]["train"]["entrypoint"] == "examples/ppo.py"
    assert state["workflow_selection"]["status"] == "rejections_for_different_inputs"
    assert state["workflow_selection"]["train_score_same_entrypoint"] is True
    assert state["environment"]["interpreter"] == ""
    assert state["environment"]["base_python_hint"]["exists"] is True


def test_checkout_reread_requires_new_actionable_failure_evidence(tmp_path):
    real = preparation(tmp_path)
    real.steps = [
        {"step": "read_the_checkout", "outcome": "done"},
        {"step": "build_the_environment", "outcome": "raised",
         "because": "old workflow handoff rejected"},
        {"step": "read_the_checkout", "outcome": "done"},
        {"step": "derive_a_command", "outcome": "not attempted",
         "because": "no verified interpreter yet"},
    ]

    assert "read_the_checkout" not in real.state()["available"]
    assert real.do("read_the_checkout")["outcome"] == "not attempted"

    real.steps.append({"step": "build_the_environment", "outcome": "raised",
                       "because": "a new path selection failed"})
    assert "read_the_checkout" in real.state()["available"]


def test_provisioning_and_command_derivation_wait_for_a_runnable_survey_stage(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.interpreter = Path(sys.executable)
    real.execution = {"stages": {}}

    unavailable = real.state()["available"]
    assert "build_the_environment" not in unavailable
    assert "derive_a_command" not in unavailable

    real.execution = {"stages": {
        "evaluate": {"available": True, "entrypoint": "eval.py"}}}
    available = real.state()["available"]
    assert "build_the_environment" in available
    assert "derive_a_command" in available


def test_choose_stops_instead_of_executing_an_unavailable_operation(tmp_path, monkeypatch):
    invalid = {"do": "build_the_environment",
               "why": "the model ignored the available-actions list"}
    real = preparation(tmp_path, [invalid, invalid])
    real.declaration = {"benchmark": "toy"}
    real.execution = {"stages": {}}
    executed = []
    monkeypatch.setattr(real, "_step_build_the_environment",
                        lambda: executed.append("build"))

    report = real.run(max_steps=1)

    assert report["status"] == "adaptation_unresolved"
    assert executed == []
    assert report["steps"][-1]["step"] == "stop"
    assert "build_the_environment" in report["steps"][-1]["because"]
    assert "No action was executed" in report["steps"][-1]["because"]


def test_choose_gives_one_bounded_repair_for_an_unavailable_operation(tmp_path):
    client = _Client([
        {"do": "build_the_environment", "why": "no stage is available"},
        {"do": "read_the_checkout", "why": "survey the repository"},
    ])
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=client,
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "toy"}
    real.execution = {"stages": {}}

    choice = real.choose()

    assert choice["do"] == "read_the_checkout"
    assert "permitted_actions" in client.seen[1]


def test_choose_repairs_a_malformed_json_response_without_ending_the_run(tmp_path):
    class _MalformedThenValid:
        model = "format-fixture"

        def __init__(self):
            self.calls = []

        def chat_with_metadata(self, system, user, **kwargs):
            self.calls.append(user)
            if len(self.calls) == 1:
                return "The repository looks promising; let me inspect it first.", {}
            return json.dumps({"do": "read_the_checkout", "why": "inspect source",
                               "state_revision": 0}), {}

    client = _MalformedThenValid()
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=client,
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "synthetic"}
    real.execution = {"stages": {}}

    choice = real.choose()

    assert len(client.calls) == 2
    assert choice["do"] == "read_the_checkout"
    assert "no action ran" in client.calls[1]
    assert "Repair the response format only" in client.calls[1]


def test_choose_stops_with_protocol_reason_after_malformed_reply_and_one_repair(tmp_path):
    class _AlwaysMalformed:
        model = "format-fixture"

        def chat_with_metadata(self, system, user, **kwargs):
            return "not a JSON object", {}

    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_AlwaysMalformed(),
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "synthetic"}
    real.execution = {"stages": {}}

    choice = real.choose()

    assert choice["do"] == "stop"
    assert "controller decision was rejected twice" in choice["why"]
    assert "no action ran" in choice["why"]


def test_preparation_run_continues_after_repaired_decision_protocol_error(
        tmp_path, monkeypatch):
    class _RecoveryClient:
        model = "format-fixture"

        def __init__(self):
            self.calls = 0

        def chat_with_metadata(self, system, user, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "I will inspect the checkout now.", {}
            facts = json.loads(user)
            return json.dumps({"do": "read_the_checkout", "why": "inspect source",
                               "state_revision": facts["state_revision"]}), {}

    client = _RecoveryClient()
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=client,
                       scouting=tmp_path / "scouting")
    monkeypatch.setattr(real, "_step_read_the_checkout",
                        lambda **_: {"outcome": "done"})

    report = real.run(max_steps=1)

    assert client.calls == 2
    assert report["steps"][0]["step"] == "read_the_checkout"
    assert report["steps"][0]["outcome"] == "done"
    assert all(row.get("outcome") != "could not be asked" for row in report["steps"])


def test_controller_prompt_projects_local_artifacts_and_sanitizes_retry_echo(tmp_path):
    from autosim.research.prepare import _controller_model_value

    repo = tmp_path / "checkout"
    repo.mkdir()
    secret_checkpoint = str(tmp_path / "private" / "secret_policy.pt")
    secret_episode = "episode-9f88d3c1"
    client = _Client([
        {"do": "not_a_permitted_action", "why":
         f"retry {secret_checkpoint} runs/PushCube/final_ckpt.pt {secret_episode}",
         "arguments": {"checkpoint_path": secret_checkpoint,
                       "episode_ids": [secret_episode]}},
        {"do": "read_the_checkout", "why": "inspect the native source"},
    ])
    real = Preparation(repo=repo, output=tmp_path / "out", client=client,
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "synthetic-sim"}
    real.execution = {"stages": {
        "train": {"available": True, "entrypoint": "examples/ppo.py",
                  "invocation": "python examples/ppo.py --train --output runs/PushCube/",
                  "artifact": "runs/PushCube/final_ckpt.pt"},
        "evaluate": {"available": True, "entrypoint": "examples/ppo.py",
                      "invocation": ("python examples/ppo.py --evaluate --checkpoint="
                                     f"{secret_checkpoint} --demos ~/.maniskill/demos/push.h5"),
                      "artifact": "runs/PushCube/test_videos/episode_9.mp4"},
    }}

    choice = real.choose()

    assert choice["do"] == "read_the_checkout"
    initial = json.loads(client.seen[0])
    train = initial["surveyed_stages"]["train"]
    evaluate = initial["surveyed_stages"]["evaluate"]
    assert train["entrypoint"] == "examples/ppo.py"
    assert train["artifact"] == "[DECLARED_LOCAL_ARTIFACT]"
    assert "--checkpoint=" in evaluate["invocation"]
    assert "[LOCAL_RESOURCE]" in evaluate["invocation"]
    for sent in client.seen:
        for forbidden in (secret_checkpoint, "final_ckpt.pt", "episode_9.mp4",
                          "push.h5", secret_episode, "episode-9f88d3c1"):
            assert forbidden not in sent
    assert "permitted_actions" in client.seen[1]

    projected = _controller_model_value({
        "episode_keys": [secret_episode],
        "initial_state_hashes": {secret_episode: "state-hash"},
        "metric_value": 0.75,
    }, local_roots=(repo, tmp_path / "out"))
    assert projected["episode_keys"] == {"present": True, "values_omitted": True}
    assert projected["initial_state_hashes"] == "[LOCAL_RESOURCE_IDENTITY_OMITTED]"
    assert projected["metric_value"] == 0.75


def test_choose_captures_a_complete_version_bound_research_decision(tmp_path):
    response = {
        "state_revision": 0, "do": "read_the_checkout", "why": "inspect the source",
        "question": "which native score path exists?",
        "hypothesis": "the repository exposes a runnable evaluator",
        "evidence_refs": ["state.surveyed_stages"],
        "resource_limits": {"wall_seconds": 120},
        "expected_outputs": ["execution.json"],
        "postconditions": ["all required stages are answered"],
        "stop_condition": "stop if source inspection finds no native score path",
    }
    real = preparation(tmp_path, [response])
    real.declaration = {"benchmark": "toy"}

    choice = real.choose()

    assert choice["do"] == "read_the_checkout"
    contract = choice["research_decision"]
    assert contract["question"] == response["question"]
    assert contract["hypothesis"] == response["hypothesis"]
    assert contract["input_evidence"] == response["evidence_refs"]
    assert contract["contract_missing"] == []
    assert contract["method_review_missing"]
    assert all(row["verdict"] in {"orientation_only", "not_assessed"}
               for row in choice["method_selection"])
    prompt_payload = json.loads(real.client.seen[0])
    assert prompt_payload["method_library"]["index"] == []
    assert 1 <= len(prompt_payload["method_library"]["skills"]) <= 4
    assert '"method_library"' in real.client.seen[0]


def test_scheduler_decision_cites_each_coding_agent_turn_and_process_receipt(tmp_path):
    response = {
        "state_revision": 0, "do": "read_the_checkout", "why": "inspect source",
        "question": "which native score path exists?",
        "hypothesis": "the source contains an evaluator entrypoint",
        "evidence_refs": ["state.surveyed_stages"], "resource_limits": {},
        "expected_outputs": ["execution.json"], "postconditions": ["source inspected"],
        "stop_condition": "stop if no native evaluator is found",
    }

    class _TraceClient(_Client):
        def __init__(self):
            super().__init__(["not JSON", json.dumps(response)])
            self.roles = []
            self.requests = []

        @contextmanager
        def as_role(self, role):
            self.roles.append(role)
            yield

        def chat_with_metadata(self, system, user, **kwargs):
            call_index = len(self.requests)
            self.requests.append(dict(kwargs))
            content = self.choices.pop(0)
            turn_id = f"{call_index + 1:032x}"
            return content, {
                "role": "scheduler", "turn_id": turn_id,
                "process_ref": f"agent/processes/{turn_id}.json",
                "decision_attempt_id": kwargs["decision_attempt_id"], "event_count": 3,
            }

    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_TraceClient(),
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "toy"}

    choice = real.choose()
    contract = choice["research_decision"]
    attempt_id = real.client.requests[0]["decision_attempt_id"]

    assert choice["do"] == "read_the_checkout"
    assert real.client.roles == ["scheduler", "scheduler"]
    assert len(contract["coding_agent_trace"]) == 2
    assert all(item["event_count"] == 3 for item in contract["coding_agent_trace"])
    assert contract["runtime_evidence_refs"][0] == (
        f"agent/events.jsonl#decision_attempt_id={attempt_id}")
    assert real.client.requests[1]["decision_attempt_id"] == attempt_id

    real._record_controller_decision("read_the_checkout", choice, {})
    used = real.decisions.rows[-1]["used"]
    for trace in contract["coding_agent_trace"]:
        assert f"agent/events.jsonl#turn_id={trace['turn_id']}" in used
        assert trace["process_ref"] in used


def test_main_controller_records_selected_skill_applicability_and_declines(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "unseen-sim"}
    candidates = [item for item in real._controller_methods(real.state())["selection"]
                  if item["id"] != "autosimsota.using-the-method-library"]
    assert candidates
    response = {
        "state_revision": 0, "do": "read_the_checkout", "why": "inspect native source",
        "question": "which score path is source-supported?",
        "hypothesis": "one native evaluate path can be verified",
        "evidence_refs": ["state.surveyed_stages"], "resource_limits": {},
        "expected_outputs": ["execution.json"], "postconditions": ["source paths resolve"],
        "stop_condition": "stop if no native path exists",
        "method_review": [
            {"id": item["id"], "verdict": "decline",
             "why": "the listed precondition is not established by this state",
             "evidence_refs": ["state.surveyed_stages"]}
            for item in candidates],
    }
    real.client.choices = [response]

    choice = real.choose()

    assert choice["research_decision"]["method_review_missing"] == []
    assert all(item["verdict"] == "decline" for item in choice["method_selection"]
               if item["id"] != "autosimsota.using-the-method-library")
    assert all(item["version"] and item["body_sha256"]
               for item in choice["method_selection"])
    decision_id = real._record_controller_decision(
        choice["do"], choice, choice["arguments"])
    row = next(item for item in real.decisions.rows if item["id"] == decision_id)
    recorded = row["research_decision"]["method_selection"]
    assert len(recorded) == len(choice["method_selection"])
    assert all(item["verdict_source"] in {"model", "kernel_default"} for item in recorded)
    assert row["used"] == ["run_state.json#phases.preparation.state"]


def test_main_agent_selects_skill_from_short_catalog_before_reading_body(tmp_path):
    chosen = "autosimsota.building-an-environment-that-runs"
    decision = {
        "state_revision": 0, "do": "read_the_checkout", "why": "inspect native source",
        "question": "which score path is runnable?", "hypothesis": "an evaluator exists",
        "evidence_refs": ["state.surveyed_stages"], "resource_limits": {},
        "expected_outputs": ["execution.json"],
        "postconditions": ["source paths resolve"],
        "stop_condition": "stop if no native path exists",
        "method_review": [{"id": chosen, "verdict": "decline",
                           "why": "environment building is premature before source survey",
                           "evidence_refs": ["state.surveyed_stages"]}],
    }

    class MainClient(_Client):
        supports_main_agent = True

        @contextmanager
        def as_role(self, role):
            assert role == "scheduler"
            yield

    client = MainClient([
        {"skill_reads": [{"id": chosen,
                          "why": "I need the environment method before choosing a build"}]},
        decision,
    ])
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=client,
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "toy"}

    choice = real.choose()

    assert choice["do"] == "read_the_checkout"
    assert len(client.seen) == 2
    first = json.loads(client.seen[0])
    assert "skill_catalog" in first and "method_library" not in first
    assert all("method" not in row and "source" not in row
               for row in first["skill_catalog"]["entries"])
    second = json.loads(client.seen[1])
    methods = second["method_library"]
    assert [row["id"] for row in methods["skills"]] == [chosen]
    assert methods["skills"][0]["method"]
    assert choice["method_selection"][0]["selected_by"] == "main_agent"
    assert "environment method" in choice["method_selection"][0]["selection_reason"]
    assert choice["research_decision"]["method_review_missing"] == []
    decision_id = real._record_controller_decision(choice["do"], choice, {})
    recorded = next(row for row in real.decisions.rows if row["id"] == decision_id)
    assert recorded["research_decision"]["method_selection"][0]["selection_reason"] == \
        choice["method_selection"][0]["selection_reason"]


def test_main_agent_invalid_skill_selection_never_reads_an_unlisted_body(tmp_path):
    class MainClient(_Client):
        supports_main_agent = True

        @contextmanager
        def as_role(self, role):
            yield

    client = MainClient([
        {"skill_reads": [{"id": "autosimsota.robosynchallenge-measurements",
                          "why": "this is not in the LIBERO catalog"}]},
        {"state_revision": 0, "do": "read_the_checkout", "why": "inspect source",
         "question": "what can run?", "hypothesis": "source has an evaluator",
         "evidence_refs": [], "resource_limits": {}, "expected_outputs": [],
         "postconditions": [], "stop_condition": "stop if no evaluator exists"},
    ])
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=client,
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "LIBERO"}

    choice = real.choose()

    assert json.loads(client.seen[1])["method_library"]["skills"] == []
    assert choice["method_selection"] == []
    assert "invalid, duplicate, or unexplained" in \
        choice["research_decision"]["skill_selection_issues"][0]


def test_choose_retries_a_decision_bound_to_an_old_state_revision(tmp_path):
    stale = {"state_revision": 4, "do": "read_the_checkout", "why": "stale"}
    fresh = {"state_revision": 0, "do": "read_the_checkout", "why": "fresh",
             "question": "what is native?", "hypothesis": "an evaluator exists",
             "evidence_refs": [], "resource_limits": {}, "expected_outputs": [],
             "postconditions": [], "stop_condition": "stop if no evaluator is found"}
    client = _Client([stale, fresh])
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=client,
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "toy"}

    choice = real.choose()

    assert choice["do"] == "read_the_checkout"
    assert len(client.seen) == 2
    assert "state_revision must equal 0" in client.seen[1]


def test_choose_discards_a_decision_if_state_changes_while_model_thinks(tmp_path):
    class _MutatingClient:
        model = "test"

        def __init__(self, store):
            self.store = store

        def chat_with_metadata(self, *args, **kwargs):
            self.store.record("preparation", "concurrent_state_change", status="running")
            return json.dumps({"state_revision": 0, "do": "read_the_checkout",
                               "why": "the survey is needed"}), {}

    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=None,
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "toy"}
    real.client = _MutatingClient(real.state_store)

    choice = real.choose()

    assert choice["do"] == "stop"
    assert choice["decision_by"] == "tool"
    assert "state changed during decision" in choice["why"]


def test_choose_accepts_a_decision_when_only_agent_runtime_telemetry_changes(tmp_path):
    class _TelemetryClient:
        model = "test"

        def __init__(self, store):
            self.store = store

        def chat_with_metadata(self, *args, **kwargs):
            self.store.record("agent_runtime", "agent_text_delta", status="running",
                              details={"role": "scheduler"}, decision_relevant=False)
            return json.dumps({"state_revision": 0, "do": "read_the_checkout",
                               "why": "inspect the source"}), {}

    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=None,
                       scouting=tmp_path / "scouting")
    real.declaration = {"benchmark": "toy"}
    real.client = _TelemetryClient(real.state_store)

    choice = real.choose()

    assert choice["do"] == "read_the_checkout"
    assert choice["decision_by"] == "model"
    assert choice["research_decision"]["state_revision"] == 0
    assert real.state_store.load()["state_revision"] == 1
    assert real.state_store.load()["decision_revision"] == 0


def test_keep_only_uses_records_without_a_model_call(tmp_path):
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client([]),
                       scouting=tmp_path / "scouting", keep_only=True)
    real.declaration = {"benchmark": "synthetic", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real.interpreter = Path(sys.executable)
    assert real.choose()["do"] == "run_the_loop"


def test_main_controller_receives_research_summary_after_nested_run(tmp_path, monkeypatch):
    real = preparation(tmp_path, [
        {"do": "run_the_loop", "why": "the native evaluator is ready"},
        {"do": "stop", "why": "the persisted report has been reviewed"},
    ])
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    real.interpreter = Path(sys.executable)
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real._metric_bound = lambda: True

    def one_research_action():
        root = real.output / "research" / real.run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "research_report.json").write_text(json.dumps({
            "run_id": real.run_id, "repo": str(real.repo),
            "rounds": [{"status": "measured", "metric_value": 0.5}],
            "best": {"name": "baseline"},
            "confirmation": {"status": "not_declared"},
        }), encoding="utf-8")
        return {"outcome": "done", "because": "one bounded research action completed",
                "verification_level": "L1"}

    monkeypatch.setattr(real, "_step_run_the_loop", one_research_action)
    report = real.run(max_steps=3)

    assert [row["step"] for row in report["steps"]] == ["run_the_loop", "stop"]
    second_decision = json.loads(real.client.seen[1])
    progress = second_decision["research_progress"]
    assert {key: progress[key] for key in (
        "status", "round_count", "measured_rounds", "best_candidate", "confirmation",
        "report_ref")} == {
        "status": "completed", "round_count": 1, "measured_rounds": 1,
        "best_candidate": "baseline",
        "confirmation": {"status": "not_declared", "recorded": False},
        "report_ref": "state.research_progress.report"}
    assert progress["unscored_baseline_recovery"]["available"] is False
    assert "run_the_loop" not in second_decision["available"]


def test_preparation_controller_can_continue_after_a_configured_action_cap(
        tmp_path, monkeypatch):
    first = preparation(tmp_path, [{"do": "read_the_checkout", "why": "record one fact"}])
    monkeypatch.setattr(first, "_step_read_the_checkout",
                        lambda: {"outcome": "done", "because": "one fact persisted"})

    paused = first.run(max_steps=1)

    assert len(paused["steps"]) == 1
    assert paused["steps"][0]["step"] == "read_the_checkout"
    resumed = preparation(tmp_path, [{"do": "stop", "why": "the saved fact was reviewed"}])
    completed = resumed.run(max_steps=2)

    assert [row["step"] for row in completed["steps"]] == ["read_the_checkout", "stop"]


def test_completed_research_is_not_launched_twice(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real.interpreter = Path(sys.executable)
    real._metric_bound = lambda: True
    root = real.output / "research" / real.run_id
    root.mkdir(parents=True)
    (root / "research_report.json").write_text(json.dumps({
        "run_id": real.run_id, "repo": str(real.repo), "rounds": [],
    }), encoding="utf-8")

    result = real.do("run_the_loop")

    assert result["outcome"] == "already done"
    assert "rather than repeating training" in result["because"]


def test_paused_research_state_allows_only_a_safe_next_action(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}, "optimization_space": {
        "collection": [], "training": [{"name": "train.learning_rate", "kind": "number",
                                           "description": "learning rate", "low": 0,
                                           "high": 1, "default": 0.001}]}}
    real.interpreter = Path(sys.executable)
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real._metric_bound = lambda: True
    root = real.output / "research" / real.run_id
    root.mkdir(parents=True)
    report = {"run_id": real.run_id, "repo": str(real.repo), "run_status": "paused",
              "rounds": [{"round": 0, "label": "baseline", "measured": True}],
              "next_round": 1, "planned_rounds": 3, "best": {"name": "baseline"},
              "confirmation": {"status": "not_declared"}}
    (root / "research_report.json").write_text(json.dumps(report), encoding="utf-8")
    session = {"schema_version": 1, "run_id": real.run_id,
               "repository": str(real.repo), "status": "paused", "rounds": 3,
               "next_round": 1, "history": report["rounds"]}
    session_path = root / "controller_session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    from autosim.research.ideas import CLEARED, Idea, IdeaLibrary
    library = IdeaLibrary(root / "ideas.json")
    library.add(Idea(label="an audited candidate", granularity="algo",
                     change={"training": {"train.learning_rate": 0.001}}, risk="low",
                     crosses="none", why="exercise paused main-controller availability",
                     status=CLEARED))

    progress = real.state()["research_progress"]
    assert progress["status"] == "paused"
    assert progress["next_round"] == 1
    assert "run_the_loop" in real.state()["available"]

    # A stale paused report must not authorize a second worker while the session says that an
    # action's outcome is unknown.
    session["status"] = "running"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    active = real.state()
    assert active["research_progress"]["status"] == "running"
    assert "run_the_loop" not in active["available"]


def test_paused_session_never_refreshes_its_frozen_declaration(tmp_path):
    from autosim.research.ideas import CLEARED, Idea, IdeaLibrary

    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}, "optimization_space": {
        "collection": [], "training": [{"name": "total_iters", "kind": "integer",
            "description": "optimizer iterations", "low": 1, "high": 100, "default": 10}]}}
    real.interpreter = Path(sys.executable)
    real.stages = {"train": "def stage_argv_train(i): return [i['python']]",
                   "evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real.parameters = {"train": {"--num_envs": {"value": 16}}}
    real._metric_bound = lambda: True
    root = real.output / "research" / real.run_id
    root.mkdir(parents=True)
    history = [{"round": 0, "label": "baseline", "metric_value": 0.5,
                "success_rate": 0.5}]
    (root / "controller_session.json").write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id, "repository": str(real.repo),
        "status": "paused", "rounds": 2, "next_round": 1, "history": history,
    }), encoding="utf-8")
    (root / "research_report.json").write_text(json.dumps({
        "run_id": real.run_id, "repo": str(real.repo), "run_status": "paused",
        "rounds": history, "next_round": 1, "planned_rounds": 2,
        "best": {"name": "baseline"}, "confirmation": {"status": "not_declared"},
    }), encoding="utf-8")
    IdeaLibrary(root / "ideas.json").add(Idea(
        label="trainer flag not declared", granularity="param", change={"num_envs": 32},
        status=CLEARED, risk="low", crosses="none", why="stage-compatible test idea"))

    state = real.state()
    assert state["research_options"]["items"] == []
    assert "declared optimization space" in state["research_options"][
        "excluded_incompatible"][0]["because"]
    assert "declare" not in state["available"]
    assert "generate_research_ideas" in state["available"]
    result = real.do("declare")
    assert result["outcome"] == "not attempted"
    assert "stage/declaration mismatch" in result["because"]


@pytest.mark.parametrize(
    ("controller_action", "safe_to_resume"),
    [("baseline", True), ("round_2", False)],
)
def test_reconciled_interrupted_research_exposes_only_supported_resume(
        tmp_path, controller_action, safe_to_resume):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    real.interpreter = Path(sys.executable)
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real._metric_bound = lambda: True
    root = real.output / "research" / real.run_id
    root.mkdir(parents=True)
    interruption = {"controller_action": controller_action, "stage": "train",
                    "attempt_id": "c" * 32, "evidence_status": "interrupted",
                    "outcome_known": False}
    pending_action = ({"round": 2, "phase": "measurement_starting",
                       "idea": {"label": "test idea", "granularity": "param"}}
                      if controller_action != "baseline" else None)
    session = {"schema_version": 1, "run_id": real.run_id,
               "repository": str(real.repo), "status": "interrupted",
               "action": controller_action, "rounds": 3, "next_round": 2,
               "history": [], "interruption": interruption,
               "pending_action": pending_action,
               "reconciliation": {"status": "reconciled", "attempt_id": "c" * 32}}
    (root / "controller_session.json").write_text(json.dumps(session), encoding="utf-8")
    (root / "research_report.json").write_text(json.dumps({
        "run_id": real.run_id, "repo": str(real.repo), "run_status": "interrupted",
        "rounds": [], "best": None,
    }), encoding="utf-8")

    state = real.state()
    assert state["research_progress"]["status"] == "interrupted"
    assert state["research_progress"]["safe_to_resume"] is safe_to_resume
    assert ("run_the_loop" in state["available"]) is safe_to_resume
    if not safe_to_resume:
        assert state["research_progress"]["pending_action"]["idea"]["label"] == "test idea"


def test_paused_research_progress_surfaces_failure_and_receipt_reference(tmp_path):
    real = preparation(tmp_path)
    root = real.output / "research" / real.run_id
    root.mkdir(parents=True)
    history = [
        {"round": 0, "label": "baseline", "status": "measured", "measured": True},
        {"round": 1, "label": "round_1", "status": "nothing was measured",
         "why_not": "evaluate: native evaluator exited with code 1",
         "failure": {"stage": "evaluate", "reason": "native evaluator exited with code 1",
                     "ran": True, "returncode": 1, "attempt_id": "attempt-1",
                     "evidence": {"receipt": "attempts/attempt-1/receipt.json",
                                  "log": "evaluate/attempts/attempt-1/output.log"}}},
    ]
    (root / "research_report.json").write_text(json.dumps({
        "run_id": real.run_id, "repo": str(real.repo), "run_status": "paused",
        "rounds": history, "next_round": 2, "planned_rounds": 3,
        "best": {"name": "baseline"}}), encoding="utf-8")
    (root / "controller_session.json").write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id, "repository": str(real.repo),
        "status": "paused", "rounds": 3, "next_round": 2, "history": history}),
        encoding="utf-8")

    progress = real.state()["research_progress"]
    assert progress["status"] == "paused"
    assert progress["last_observation"]["label"] == "round_1"
    assert progress["last_observation"]["why_not"] == (
        "evaluate: native evaluator exited with code 1")
    assert progress["last_observation"]["failure"]["attempt_id"] == "attempt-1"
    assert progress["last_observation"]["measurement_ref"] == (
        f"research/{real.run_id}/measurements/round_1.json")


def test_research_observation_exposes_safe_actual_training_parameters(tmp_path):
    real = preparation(tmp_path)
    root = real.output / "research" / real.run_id
    attempt_id = "a" * 32
    history = [{"round": 0, "label": "baseline", "status": "measured",
                "measured": True, "metric_name": "eval_success_once_mean",
                "metric_value": 0.0,
                "settings": {"task": "PushCube-v1", "checkpoint": "/private/model.pt"}}]
    root.mkdir(parents=True)
    (root / "controller_session.json").write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id, "repository": str(real.repo),
        "status": "paused", "rounds": 2, "next_round": 1, "history": history}),
        encoding="utf-8")
    (root / "measurements").mkdir()
    (root / "measurements" / "baseline.json").write_text(json.dumps({
        "label": "baseline", "metric": {"name": "eval_success_once_mean"},
        "metric_value": 0.0,
        "train": {"attempt_id": attempt_id, "reused": False}}), encoding="utf-8")
    receipt_dir = root / "attempts" / attempt_id
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "receipt.json").write_text(json.dumps({
        "attempt_id": attempt_id, "node_id": "train", "status": "completed",
        "returncode": 0,
        "argv": ["python", "train.py", "--total_timesteps", "1024",
                 "--num_envs=32", "--checkpoint=/private/model.pt",
                 "--api-key=sk-12345678901234567890",
                 "--access_key=never-ship-this-value",
                 "--misc=sk-12345678901234567890"]}), encoding="utf-8")

    observation = real.state()["research_progress"]["last_observation"]

    assert observation["metric_value"] == 0.0
    assert observation["metric_name"] == "eval_success_once_mean"
    assert observation["settings"] == [{"name": "task", "value": "PushCube-v1"}]
    assert observation["actual_training"]["attempt_id"] == attempt_id
    assert observation["actual_training"]["receipt_ref"].endswith(
        f"attempts/{attempt_id}/receipt.json")
    assert observation["actual_training"]["status"] == "completed"
    parameters = observation["actual_training"]["parameters"]
    assert {"name": "--total_timesteps", "value": "1024"} in parameters
    assert {"name": "--num_envs", "value": "32"} in parameters
    assert not any("checkpoint" in row["name"] or "api-key" in row["name"] or
                   "access_key" in row["name"]
                   for row in parameters)
    assert "12345678901234567890" not in json.dumps(observation)
    assert "never-ship-this-value" not in json.dumps(observation)
    assert "/private/model.pt" not in json.dumps(observation)


def test_main_controller_hands_off_between_baseline_and_candidate_without_duplicate_work(
        tmp_path):
    from autosim.research.adapter_protocol import Axis, OptimizationSpace
    from autosim.research.compute_decision import ComputeDecision
    from autosim.research.declarative_backend import DeclarativeBackend
    from autosim.research.derived_research import DerivedResearch

    run_id = "controller-bounded"
    choices = [
        {"do": "run_the_loop", "why": "take the frozen baseline measurement"},
        {"do": "propose_research_idea", "why":
         "the baseline receipt supports a small, testable training change",
         "arguments": {"idea": {
             "label": "increase training epochs",
             "granularity": "param",
             "mechanism": "more optimizer epochs may improve the trained policy",
             "change": {"train.n_epochs": 2},
             "risk": "low",
             "crosses": "none",
             "why": "compare one declared training setting against the frozen baseline",
             "evidence": ["research/controller-bounded/measurements/baseline.json"],
             "touches": [],
         }}},
        {"do": "run_the_loop", "why": "continue from the persisted next round",
         "arguments": {"idea_label": "increase training epochs"}},
        {"do": "stop", "why": "the bounded research plan is complete"},
    ]
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client(choices),
                       scouting=tmp_path / "scouting", run_id=run_id,
                       base_settings={"_rounds": 1})
    real.declaration = {"benchmark": "synthetic", "task_contract": {
        "primary_metric": {"name": "success_rate", "direction": "maximize",
                           "unit": "fraction", "source": "log"}},
        "optimization_space": {"collection": [], "training": [{
            "name": "train.n_epochs", "kind": "integer", "description": "epochs",
            "low": 1, "high": 10, "default": 1}]}}
    real.interpreter = Path(sys.executable)
    stage_code = ("import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                  "(p/'models').mkdir(parents=True,exist_ok=True); "
                  "(p/'models'/'model.pth').write_text('ok'); print('succ: 0.50')")
    real.stages = {
        stage: (f"def stage_argv_{stage}(i):\n"
                f"    code = {stage_code!r}\n"
                "    return [i['python'], '-c', code, i['output']]\n")
        for stage in ("train", "evaluate")
    }
    real.execution = {"stages": {stage: {"available": True}
                                 for stage in real.stages}}
    real.verified = {stage: {"verified": True} for stage in real.stages}
    real._metric_bound = lambda: True
    answer = {"stages": {stage: {"available": True, "entrypoint": "native.py",
                                 "invocation": "python native.py",
                                 "artifact": "models/*.pth" if stage == "train" else ""}
                         for stage in real.stages}}
    backend = DeclarativeBackend(repo=tmp_path, answer=answer,
                                 sources=real.stages, parameters={})
    engine = DerivedResearch(
        repo=tmp_path, output=real.output, backend=backend,
        interpreter=Path(sys.executable), space=OptimizationSpace(training=(
            Axis("train.n_epochs", "integer", "epochs", low=1, high=10, default=1),)),
        client=None, stages=real.stages, run_id=run_id,
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="synthetic orchestration test"),
        benchmark="synthetic")
    engine.choose_idea = lambda **_: (_ for _ in ()).throw(
        AssertionError("inner research engine must not choose the candidate"))
    def research_controller():
        engine.controller_decision_id = str(
            real.current_action.get("decision_id") or "")
        return engine, {}

    real._research_controller = research_controller

    report = real.run(max_steps=5)
    seen = [json.loads(payload) for payload in real.client.seen]
    assert [row["step"] for row in report["steps"]] == [
        "run_the_loop", "propose_research_idea", "run_the_loop", "stop"]
    assert seen[1]["research_progress"]["status"] == "paused"
    assert seen[1]["research_progress"]["next_round"] == 1
    assert seen[1]["research_options"]["items"] == []
    assert "run_the_loop" not in seen[1]["available"]
    assert "propose_research_idea" in seen[1]["available"]
    assert "generate_research_ideas" in seen[1]["available"]
    assert [one["label"] for one in seen[2]["research_options"]["items"]] == [
        "increase training epochs"]
    assert "run_the_loop" in seen[2]["available"]
    assert seen[3]["research_progress"]["status"] == "completed"
    assert "run_the_loop" not in seen[3]["available"]
    receipts = [json.loads(path.read_text(encoding="utf-8"))
                for path in (engine.run_root / "attempts").glob("*/receipt.json")]
    assert sum(row.get("node_id") == "train" for row in receipts) == 2
    decision_rows = json.loads((real.output / "decisions.json").read_text(
        encoding="utf-8"))["rows"]
    research_decisions = {row["id"]: row for row in decision_rows
                          if row.get("kind") == "decision" and
                          (row.get("research_decision") or {}).get("action") ==
                          "run_the_loop"}
    assert len(research_decisions) == 2
    for receipt_path in (engine.run_root / "attempts").glob("*/receipt.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        parent_id = receipt["controller_decision_id"]
        assert parent_id in research_decisions
        outcome = research_decisions[parent_id]["outcome"]["what"]
        receipt_ref = str(receipt_path.resolve().relative_to(real.output.resolve()))
        assert receipt_ref in outcome["evidence_refs"]
        assert receipt["attempt_id"] in outcome["attempt_ids"]
    assert report["steps"][0]["outcome"] == "yielded"
    assert report["steps"][2]["arguments"] == {
        "idea_label": "increase training epochs"}


def test_paused_research_rejects_unreviewed_idea_label_without_mutating_session(tmp_path):
    real = preparation(tmp_path)
    root = real.output / "research" / real.run_id
    root.mkdir(parents=True)
    session_path = root / "controller_session.json"
    session_path.write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id,
        "repository": str(real.repo), "status": "paused", "rounds": 1,
        "next_round": 1, "history": [{"round": 0, "label": "baseline",
                                        "metric_value": 0.5, "success_rate": 0.5}],
        "kinds": [], "current_measurement_label": "baseline",
    }), encoding="utf-8")

    state = real.state()
    assert state["research_options"]["status"] == "empty"
    assert "run_the_loop" not in state["available"]
    assert "propose_research_idea" in state["available"]
    result = real.do("run_the_loop", idea_label="not audited")
    assert result["outcome"] == "not attempted"
    assert "currently audited" in result["because"]
    persisted = json.loads(session_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "paused"


def test_fresh_idea_batch_is_suggestion_only_and_does_not_run_a_stage(tmp_path):
    from autosim.research.ideas import CLEARED, Idea, IdeaLibrary

    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy", "optimization_space": {
        "collection": [], "training": [{"name": "train.learning_rate", "kind": "number",
                                           "description": "learning rate", "low": 0,
                                           "high": 1, "default": 0.001}]}}
    root = real.output / "research" / real.run_id
    root.mkdir(parents=True)
    session_path = root / "controller_session.json"
    session_path.write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id,
        "repository": str(real.repo), "status": "paused", "rounds": 1,
        "next_round": 1, "history": [{"round": 0, "label": "baseline",
                                        "metric_value": 0.5, "success_rate": 0.5}],
        "kinds": [], "current_measurement_label": "baseline",
    }), encoding="utf-8")

    class _IdeaGenerator:
        def prepare_idea_options(self, *, extend, generate_if_empty):
            assert extend is True
            assert generate_if_empty is False
            idea = Idea(label="new suggestion", granularity="param",
                        change={"train.learning_rate": 0.001}, risk="low",
                        crosses="none", why="test generated suggestion",
                        status=CLEARED)
            library = IdeaLibrary(root / "ideas.json")
            library.add(idea)
            return [idea.as_dict()]

    real._research_controller = lambda: (_IdeaGenerator(), {})
    before = real.state()
    assert "generate_research_ideas" in before["available"]
    assert "run_the_loop" not in before["available"]

    result = real.do("generate_research_ideas")

    assert result["outcome"] == "ideas available"
    assert real.state()["research_options"]["items"][0]["label"] == "new suggestion"
    assert not (root / "attempts").exists()
    assert json.loads(session_path.read_text(encoding="utf-8"))["status"] == "paused"


def test_confirm_best_requires_explicit_request_and_never_trains(tmp_path):
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client([]),
                       scouting=tmp_path / "scouting", base_settings={"_confirm": True})
    root = real.output / "research" / real.run_id
    root.mkdir(parents=True)
    report_path = root / "research_report.json"
    report_path.write_text(json.dumps({
        "run_id": real.run_id, "repo": str(real.repo),
        "confirmation": {"status": "available_not_taken"},
    }), encoding="utf-8")

    class _ConfirmationOnly:
        def __init__(self):
            self.calls = []

        def confirmation_state(self):
            return {"status": "available_not_taken"} if not self.calls else {
                "status": "taken", "metric_value": 0.75}

        def confirm(self):
            self.calls.append("evaluate")
            return {"ok": True, "metric_value": 0.75,
                    "evaluate": {"attempt_id": "eval-only"}}

    engine = _ConfirmationOnly()
    real._research_controller = lambda: (engine, {})

    result = real.do("confirm_best")

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["outcome"] == "measurement_recorded"
    assert engine.calls == ["evaluate"]
    assert result["training_performed"] is False
    assert "metric_value" not in result
    assert "metric_value" not in saved["confirmation_action"]
    assert saved["confirmation_action"]["training_performed"] is False
    assert saved["confirmation"]["status"] == "taken"
    controller_confirmation = real.state()["research_progress"]["confirmation"]
    assert controller_confirmation == {"status": "taken", "recorded": True}
    assert "0.75" not in json.dumps(controller_confirmation)


def test_metric_binding_needs_a_label_in_verified_output_and_source(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real.verified = {"evaluate": {"entrypoint": "eval.py"}}
    (tmp_path / "eval.py").write_text(
        'print(f"eval_success_once_mean={episode_success.mean()}")\n', encoding="utf-8")
    attempts = real.output / "derivation_attempts"
    attempts.mkdir(parents=True)
    (attempts / "evaluate.json").write_text(json.dumps({"attempts": [
        {"status": "accepted", "said": "eval_success_once_mean=0.25"}]}),
        encoding="utf-8")
    real.client = _Client([])
    real.client.choices = [{"primary_metric": {"name": "eval_success_once_mean",
                                               "direction": "maximize", "unit": "fraction",
                                               "source": "log", "min": 0, "max": 1},
                            "evidence": "source averages episode success"}]
    assert real.do("bind_metric")["outcome"] == "done"
    assert real._metric_bound()
    assert (real.output / "metric_binding.json").is_file()

    real.declaration = {"benchmark": "toy"}
    real.client.choices = [{"primary_metric": {"name": "eval_success_at_end_mean",
                                               "direction": "maximize", "unit": "fraction",
                                               "source": "log"},
                            "evidence": "different label"}]
    assert real.do("bind_metric")["outcome"] == "not bound"


def test_metric_binding_uses_only_fresh_verified_json_schema_and_pins_bytes(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real.verified = {"evaluate": {"entrypoint": "eval.py"}}
    (tmp_path / "eval.py").write_text(
        "records = native_evaluator.run_episodes()\n"
        "episodes = [{'episode_id':row.id,'success':row.success} for row in records]\n"
        "write_json('episode_records.json', {'episodes':episodes})\n",
        encoding="utf-8")
    artifact = tmp_path / "episode_records.json"
    artifact.write_text(json.dumps({"episodes": [
        {"episode_id": 0, "success": False},
        {"episode_id": 1, "success": True}]}), encoding="utf-8")
    attempts = real.output / "derivation_attempts"
    attempts.mkdir(parents=True)
    (attempts / "evaluate.json").write_text(json.dumps({"attempts": [{
        "status": "accepted", "attempt": 2, "said": "eval_success_once_mean=0.5",
        "verified_artifact": {
            "attempt_started": artifact.stat().st_mtime - 1,
            "working_directory": str(tmp_path),
            "output_directory": str(real.output / "v"),
            "structured_candidates": [{
                "path": str(artifact), "root": "working_directory",
                "relative_path": artifact.name, "suffix": ".json",
                "size_bytes": artifact.stat().st_size,
                "mtime": artifact.stat().st_mtime,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]}}]}),
        encoding="utf-8")
    real.client.choices = [{"primary_metric": {
        "name": "eval_success_once_mean", "direction": "maximize", "unit": "fraction",
        "source": "json", "artifact_candidate": "candidate_1",
        "json_key": "episodes",
        "json_value_key": "success", "episode_id_column": "episode_id",
        "aggregation": "mean", "min_samples": 2},
        "evidence": "the native source maps each completed rollout record's boolean success"}]

    result = real.do("bind_metric")

    assert result["outcome"] == "done", result
    bound = json.loads((real.output / "metric_binding.json").read_text())
    assert bound["verified_artifact"]["sha256"] == hashlib.sha256(
        artifact.read_bytes()).hexdigest()
    sent = json.loads(real.client.seen[-1])
    assert "true" not in json.dumps(sent).lower()
    assert "false" not in json.dumps(sent).lower()
    assert sent["fresh_structured_candidates"][0]["candidate_id"] == "candidate_1"
    for forbidden in ("episode_records.json", hashlib.sha256(
            artifact.read_bytes()).hexdigest(), "relative_path"):
        assert forbidden not in json.dumps(sent)
    assert sent["fresh_structured_candidates"][0]["schema_only_no_result_values"][
        "fields"]["episodes"]["item_shapes"][0]["fields"]["success"] == {
            "type": "boolean"}


def test_existing_log_metric_can_be_upgraded_from_fresh_policy_sidecar(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy", "task_contract": {"task": "PushCube-v1"},
                        "research_goal": {"primary_metric": {
                            "name": "eval_success_once_mean", "direction": "maximize",
                            "unit": "fraction", "source": "log"}}}
    real.execution = {"score_target": "evaluate", "stages": {
        "evaluate": {"available": True, "entrypoint": "eval.py"}}}
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real.verified = {"evaluate": {"entrypoint": "eval.py"}}
    (tmp_path / "eval.py").write_text(
        "records = native_evaluator.run_episodes()\n"
        "write_json('test_videos/trajectory.json', {'env_info': env_info, "
        "'episodes': records})\n", encoding="utf-8")
    policy_parent = real.output / "experiments" / "baseline" / "train-attempt"
    artifact = policy_parent / "test_videos" / "trajectory.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"env_info": {"env_id": "PushCube-v1"},
                                    "episodes": [
                                        {"episode_id": 0, "episode_seed": 12,
                                         "success": False},
                                        {"episode_id": 1, "episode_seed": 13,
                                         "success": True}]}), encoding="utf-8")
    candidates = [{"path": str(artifact), "root": "policy_parent",
                   "relative_path": "test_videos/trajectory.json", "suffix": ".json",
                   "size_bytes": artifact.stat().st_size,
                   "mtime": artifact.stat().st_mtime,
                   "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]
    attempts = real.output / "derivation_attempts"
    attempts.mkdir(parents=True)
    (attempts / "evaluate.json").write_text(json.dumps({"attempts": [{
        "status": "accepted", "attempt": 1, "said": "eval_success_once_mean=0.5",
        "verified_artifact": {"attempt_started": artifact.stat().st_mtime - 1,
                              "working_directory": str(tmp_path),
                              "output_directory": str(real.output / "v"),
                              "policy_parent": str(policy_parent),
                              "structured_candidates": candidates}}]}), encoding="utf-8")
    real.client.choices = [{"primary_metric": {
        "name": "eval_success_once_mean", "direction": "maximize", "unit": "fraction",
        "source": "log"},
        "evidence": "the evaluator reports an exact aggregate label"}, {"primary_metric": {
        "name": "eval_success_once_mean", "direction": "maximize", "unit": "fraction",
        "source": "json", "artifact_candidate": "candidate_1",
        "json_key": "episodes",
        "json_value_key": "success", "episode_id_column": "episode_id",
        "task_column": "env_info.env_id", "aggregation": "mean", "min_samples": 2},
        "evidence": "the evaluator writes completed native task episode records"}]

    assert "bind_metric" in real._available_operations()
    result = real.do("bind_metric")

    assert result["outcome"] == "done", result
    assert len(real.client.seen) == 2
    bound = json.loads((real.output / "metric_binding.json").read_text())
    assert bound["primary_metric"]["source"] == "json"
    assert bound["reviewed_structured_candidates_sha256"] == object_digest(candidates)
    assert real._structured_metric_upgrade_available() is False
    sent = json.loads(real.client.seen[-1])
    assert "true" not in json.dumps(sent).lower()
    assert "false" not in json.dumps(sent).lower()


def test_failed_metric_binding_requires_fresh_score_evidence(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.interpreter = Path(sys.executable)
    real.execution = {"score_target": "evaluate", "stages": {
        "evaluate": {"available": True}}}
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    attempts = real.output / "derivation_attempts"
    attempts.mkdir(parents=True)
    (attempts / "evaluate.json").write_text(json.dumps({"attempts": [
        {"status": "accepted", "said": "Evaluated 8 steps resulting in 0 episodes"}]}),
        encoding="utf-8")
    real.steps = [{"step": "bind_metric", "outcome": "not bound",
                   "because": "source/output did not prove a metric"}]

    state = real.state()
    assert state["metric_binding"]["retry_blocked"] is True
    assert "derive_a_command" in state["available"]
    assert "bind_metric" not in state["available"]
    assert "0 episodes" in state["metric_binding"]["verified_output_excerpt"]
    assert real.do("bind_metric")["outcome"] == "not attempted"

    real.steps.append({"step": "derive_a_command", "outcome": "done",
                       "arguments": {"stage": "evaluate"}})
    (attempts / "evaluate.json").write_text(json.dumps({"attempts": [
        {"status": "accepted", "said": "eval_success_once_mean=0.5"}]}),
        encoding="utf-8")
    refreshed = real.state()
    assert refreshed["metric_binding"]["retry_blocked"] is False
    assert "bind_metric" in refreshed["available"]


def test_evaluator_waits_for_a_real_training_checkpoint(tmp_path):
    real = preparation(tmp_path)
    real.interpreter = Path(sys.executable)
    real.declaration = {"benchmark": "toy"}
    real.execution = {"stages": {
        "train": {"available": True, "entrypoint": "train.py"},
        "evaluate": {"available": True, "entrypoint": "eval.py",
                     "invocation": "python eval.py --checkpoint=<trained policy>"}}}
    assert "derive train first" in real.state()["score_dependency"]
    result = real.do("derive_a_command", stage="evaluate")
    assert result["outcome"] == "not attempted"
    assert "checkpoint" in result["because"]


def test_a_step_with_a_missing_precondition_says_which_one(tmp_path):
    """`not attempted` and `why` are the answer. A step that reports only "failed" leaves a
    reader unable to tell a missing precondition from a broken one."""
    real = preparation(tmp_path)
    result = real.do("build_the_environment")
    assert result["outcome"] == "not attempted"
    assert "no declaration yet" in result["because"]
    assert "declaration" in result["because"]


def test_a_step_that_raises_is_a_step_that_failed(tmp_path, monkeypatch):
    """A failing step that raised would take the run down from inside the loop that was
    supposed to be deciding what to do about it."""
    from autosim.research import prepare as module

    def explodes(*args, **kwargs):
        raise RuntimeError("the checkout is not a git repository")

    monkeypatch.setattr(module.execution_derive, "run", explodes)
    result = preparation(tmp_path).do("read_the_checkout")
    assert result["outcome"] == "raised"
    assert "RuntimeError" in result["because"]


def test_an_unknown_step_is_refused_not_attempted(tmp_path):
    result = preparation(tmp_path).do("do_something_clever")
    assert result["outcome"] == "no such step"
    assert "not one of the steps" in result["because"]


def test_scheduler_cannot_add_a_stage_argument_to_a_repository_survey(tmp_path, monkeypatch):
    from autosim.research import prepare as module

    seen = []

    def surveyed(repo, **kwargs):
        seen.append(kwargs)
        return {"stages": {"train": {"available": True}}}

    monkeypatch.setattr(module.execution_derive, "run", surveyed)
    result = preparation(tmp_path).do("read_the_checkout", stage="train")
    assert result["outcome"] == "done"
    assert len(seen) == 1


def test_the_loop_takes_the_steps_the_model_chose(tmp_path, monkeypatch):
    from autosim.research import prepare as module

    calls = []
    monkeypatch.setattr(module.execution_derive, "run",
                        lambda repo, *, client, output, context=None: calls.append("read") or
                        {"stages": {"train": {"available": True}}})
    real = preparation(tmp_path, [
        {"do": "read_the_checkout", "why": "nothing is known yet"},
        {"do": "stop", "why": "the checkout has one stage and no environment can be built"},
    ])
    report = real.run()
    assert calls == ["read"]
    assert [row["step"] for row in report["steps"]] == ["read_the_checkout", "stop"]
    assert report["steps"][1]["outcome"] == "stopped"
    assert "one stage" in report["steps"][1]["because"]


def test_the_loop_goes_on_after_a_step_fails(tmp_path, monkeypatch):
    """The next decision is made from the state, and the state now includes the failure."""
    from autosim.research import prepare as module

    monkeypatch.setattr(module.execution_derive, "run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unreadable")))
    real = preparation(tmp_path, [
        {"do": "read_the_checkout", "why": "start there"},
        {"do": "run_the_loop", "why": "try the loop anyway"},
        {"do": "stop", "why": "no verified research path exists"},
    ])
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    real.interpreter = Path(sys.executable)
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real._metric_bound = lambda: True
    steps = [row["step"] for row in real.run()["steps"]]
    # The controller now regains control after research, instead of treating a nested loop as
    # the terminal owner of the run.
    assert steps == ["read_the_checkout", "run_the_loop", "stop"]
    assert real.steps[0]["outcome"] == "raised"
    assert "unreadable" in real.steps[0]["because"]


def test_stop_is_hidden_while_a_paused_run_has_actionable_research(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.interpreter = Path(sys.executable)
    real.stages = {"train": "train-source", "evaluate": "evaluate-source"}
    real._declaration_refresh_is_justified = lambda **_: False
    real._metric_bound = lambda: True
    real._structured_metric_upgrade_available = lambda: False
    real._verified_score_output_available = lambda: True
    real._checkout_read_is_justified = lambda: False
    real._surveyed_runnable_stages = lambda: {"train": {}, "evaluate": {}}
    real._research_loop_ready = lambda: True
    progress = {"status": "paused", "unscored_baseline_recovery": {"available": False}}
    options = {"status": "available", "items": [{"label": "candidate"}],
               "generation_attempted": False}

    available = real._available_operations(progress, options)

    assert "run_the_loop" in available
    assert "stop" not in available


def test_stop_remains_available_when_paused_research_has_no_action(tmp_path):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.interpreter = Path(sys.executable)
    real.stages = {"train": "train-source", "evaluate": "evaluate-source"}
    real._declaration_refresh_is_justified = lambda **_: False
    real._metric_bound = lambda: True
    real._structured_metric_upgrade_available = lambda: False
    real._verified_score_output_available = lambda: True
    real._checkout_read_is_justified = lambda: False
    real._surveyed_runnable_stages = lambda: {"train": {}, "evaluate": {}}
    real._research_loop_ready = lambda: True
    progress = {"status": "paused", "unscored_baseline_recovery": {"available": False}}
    options = {"status": "empty", "items": [], "generation_attempted": True}

    assert "stop" in real._available_operations(progress, options)


def test_the_loop_is_bounded(tmp_path):
    """Repeatedly requesting a survey with no new evidence cannot consume the step budget."""
    real = preparation(tmp_path, [{"do": "read_the_checkout", "why": "again"}] * 50)
    report = real.run(max_steps=3)
    assert [row["step"] for row in report["steps"]] == ["read_the_checkout", "stop"]
    assert "controller decision was rejected twice" in report["steps"][-1]["because"]


def test_a_model_that_cannot_be_asked_ends_the_loop_and_says_so(tmp_path):
    class _Broken:
        model = "x"

        def chat_with_metadata(self, *a, **k):
            raise RuntimeError("no network")

    real = preparation(tmp_path)
    real.client = _Broken()
    report = real.run()
    assert report["steps"][-1]["outcome"] == "could not be asked"
    assert "no network" in report["steps"][-1]["because"]


def test_the_record_is_written_where_a_reader_looks(tmp_path):
    real = preparation(tmp_path, [{"do": "stop", "why": "nothing to do"}])
    real.run()
    written = json.loads((real.output / "preparation_derived.json").read_text(encoding="utf-8"))
    assert written["steps"][0]["step"] == "stop"
    assert "state" in written and "stages_with_commands" in written
    decisions = json.loads((real.output / "decisions.json").read_text(encoding="utf-8"))
    assert len(decisions["rows"]) == 1
    row = decisions["rows"][0]
    assert row["by"] == "model" and row["state_revision"] > 0
    assert row["outcome"]["what"]["outcome"] == "stopped"
    assert row["research_decision"]["action"] == "stop"
    assert "hypothesis" in row["research_decision"]["contract_missing"]
    assert "enforced" in row["research_decision"]["resource_limits"]
    assert real.last_decision["decision_id"] == row["id"]
    assert "[Controller decisions](decisions.json)" in (
        real.output / "RUN.md").read_text(encoding="utf-8")


def test_controller_action_receipt_is_linked_and_verifiable(tmp_path, monkeypatch):
    real = preparation(tmp_path, [{"do": "read_the_checkout", "why": "inspect the checkout"}])
    monkeypatch.setattr(real, "do", lambda step, **_kwargs: {
        "outcome": "done", "because": "the checkout inventory was recorded",
        "evidence_refs": ["execution.json"], "attempt_ids": ["attempt-1"],
        "verification_level": "L1"})

    report = real.run(max_steps=1)

    step = report["steps"][0]
    assert step["step"] == "read_the_checkout"
    assert step["receipt_ref"] == f"action_receipts/{real.last_action['decision_id']}.json"
    receipt = read_action_receipt(
        real.output, step["receipt_ref"], run_id=real.run_id, repository=real.repo)
    assert receipt["receipt_sha256"] == step["receipt_sha256"]
    assert receipt["decision_id"] == real.last_action["decision_id"]
    assert receipt["child_attempt_ids"] == ["attempt-1"]
    assert receipt["postcondition_verification"] == "not_independently_verified"
    document = (real.output / "RUN.md").read_text(encoding="utf-8")
    assert f"[receipt]({step['receipt_ref']})" in document
    assert f"Latest action receipt: [{step['receipt_ref']}]" in document


def test_controller_receipt_links_new_process_identity_for_its_action(tmp_path, monkeypatch):
    real = preparation(tmp_path, [{"do": "read_the_checkout", "why": "inspect sources"}])
    process_id = "a" * 32
    unrelated_id = "b" * 32

    def do_with_process_records(step, **_kwargs):
        process_dir = real.output / "processes"
        process_dir.mkdir(parents=True, exist_ok=True)
        decision_id = real.current_action["decision_id"]
        for attempt_id, record_run, record_decision in (
                (process_id, real.run_id, decision_id),
                (unrelated_id, "another-run", decision_id)):
            (process_dir / f"{attempt_id}.json").write_text(json.dumps({
                "schema_version": 1, "run_id": record_run,
                "attempt_id": attempt_id, "status": "running",
                "action": {"step": step, "decision_id": record_decision},
                "process_identity": {"run_id": record_run,
                                     "attempt_id": attempt_id},
            }), encoding="utf-8")
        return {"outcome": "done", "because": "surveyed",
                "evidence_refs": ["execution.json"]}

    monkeypatch.setattr(real, "do", do_with_process_records)
    report = real.run(max_steps=1)

    step_result = report["steps"][0]
    expected_ref = f"processes/{process_id}.json"
    assert step_result["attempt_ids"] == [process_id]
    assert expected_ref in step_result["evidence_refs"]
    receipt = read_action_receipt(
        real.output, step_result["receipt_ref"], run_id=real.run_id, repository=real.repo)
    assert receipt["child_attempt_ids"] == [process_id]
    assert expected_ref in receipt["evidence_refs"]


def test_run_document_keeps_every_step_row(tmp_path):
    real = preparation(tmp_path)
    real.steps = [
        {"step": "read_the_checkout", "outcome": "done",
         "because": "the checkout was inventoried"},
        {"step": "declare", "outcome": "done",
         "because": "the task contract was recorded"},
    ]

    real._refresh_document(status="running")

    document = (real.output / "RUN.md").read_text(encoding="utf-8")
    assert "| read_the_checkout | done | the checkout was inventoried |" in document
    assert "| declare | done | the task contract was recorded |" in document


def test_corrupt_decision_log_fails_closed_without_overwriting_it(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    corrupt = out / "decisions.json"
    corrupt.write_text("not json", encoding="utf-8")
    client = _Client([{"do": "read_the_checkout", "why": "inspect"}])
    real = Preparation(repo=tmp_path, output=out, client=client,
                       scouting=tmp_path / "scouting")

    report = real.run(max_steps=1)

    assert report["status"] == "internal_error"
    assert "decision log is not trustworthy" in report["state_persistence_error"]
    assert client.seen == []
    assert corrupt.read_text(encoding="utf-8") == "not json"


def test_run_document_and_feasibility_exist_before_the_first_model_call(tmp_path):
    out = tmp_path / "out"

    class _InspectingClient:
        model = "test"

        def chat_with_metadata(self, *args, **kwargs):
            assert "Status: running" in (out / "RUN.md").read_text(encoding="utf-8")
            view = json.loads((out / "report" / "view.json").read_text(encoding="utf-8"))
            assert view["status"] == "running"
            assert view["current_action"] == "choose"
            assert view["source_consistency"] == "stable"
            state = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
            assert state["schema_version"] == 1
            assert state["phase"] == "preparation"
            assert state["status"] == "running"
            assert state["state_revision"] >= 2
            assert state["current_action"]["step"] == "choose"
            assert json.loads((out / "feasibility.json").read_text(encoding="utf-8"))[
                "status"] == "running"
            return '{"do":"stop","why":"no assets have been supplied"}', {}

    real = Preparation(repo=tmp_path, output=out, client=_InspectingClient(),
                       scouting=tmp_path / "scouting")
    report = real.run()
    assert report["status"] == "adaptation_unresolved"
    assert json.loads((out / "feasibility.json").read_text(encoding="utf-8"))[
        "blocking_conditions"]
    state = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "adaptation_unresolved"
    assert state["state_revision"] >= 3
    assert state["last_decision"]["step"] == "stop"
    assert state["last_action"]["outcome"] == "stopped"


def test_failed_action_start_checkpoint_prevents_repository_operation(tmp_path, monkeypatch):
    """A failed/ambiguous durable start record must fail closed before side effects."""
    real = preparation(tmp_path, [{"do": "read_the_checkout", "why": "inspect source"}])
    calls = []
    from autosim.research import prepare as module
    monkeypatch.setattr(module.execution_derive, "run",
                        lambda *args, **kwargs: calls.append("repository read") or {})
    original_record = real.state_store.record

    def fail_action_start(phase, event, *, phase_state=None, **kwargs):
        action = (phase_state or {}).get("current_action") or {}
        if action.get("step") == "read_the_checkout":
            raise OSError("injected event append failure")
        return original_record(phase, event, phase_state=phase_state, **kwargs)

    monkeypatch.setattr(real.state_store, "record", fail_action_start)
    report = real.run(max_steps=1)

    assert report["status"] == "internal_error"
    assert "injected event append failure" in report["state_persistence_error"]
    assert calls == []
    assert real.do("read_the_checkout")["outcome"] == "blocked"
    assert calls == []


def test_research_journal_failure_aborts_preparation_instead_of_retrying(tmp_path, monkeypatch):
    from autosim.research.research_state import ResearchStatePersistenceError

    real = preparation(tmp_path, [
        {"do": "run_the_loop", "why": "start the bounded research loop"},
        {"do": "read_the_checkout", "why": "must never be requested after journal loss"},
    ])
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    real.interpreter = Path(sys.executable)
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real._metric_bound = lambda: True

    def fail_research_journal():
        raise ResearchStatePersistenceError("injected research event append failure")

    monkeypatch.setattr(real, "_step_run_the_loop", fail_research_journal)
    report = real.run(max_steps=4)

    assert report["status"] == "internal_error"
    assert "injected research event append failure" in real.state_persistence_error
    assert len(real.client.seen) == 1
    assert report["steps"][-1]["step"] == "run_state"


def test_rejected_research_contract_closes_outer_action_without_poisoning_run_state(
        tmp_path, monkeypatch):
    from autosim.research.research_state import ResearchStateError

    real = preparation(tmp_path, [
        {"do": "run_the_loop", "why": "resume the frozen candidate session"},
    ])
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    real.interpreter = Path(sys.executable)
    real.stages = {"evaluate": "def stage_argv_evaluate(i): return [i['python']]"}
    real._metric_bound = lambda: True

    class _Research:
        run_root = real.output / "research" / real.run_id
        state_persistence_error = ""

        def run(self, **_kwargs):
            raise ResearchStateError(
                "research inputs changed since the paused action; start a new run")

    monkeypatch.setattr(real, "_research_controller", lambda: (
        _Research(), {"stages": {"evaluate": {"available": True}}}))
    report = real.run(max_steps=1)

    persisted = json.loads((real.output / "run_state.json").read_text(encoding="utf-8"))
    phase = persisted["phases"]["preparation"]
    assert report["status"] == "adaptation_unresolved"
    assert report["steps"][-1]["outcome"] == "rejected"
    assert real.state_persistence_error == ""
    assert phase["current_action"] is None
    assert phase["last_action"]["outcome"] == "rejected"
    assert phase["last_action"]["receipt_ref"].startswith("action_receipts/")
    events = json.loads((real.output / "run_events.json").read_text(encoding="utf-8"))
    assert events["rows"][-1]["event"] == "action_completed"
    assert events["rows"][-1]["details"]["action"]["outcome"] == "rejected"


def test_corrupt_resumed_budget_leaves_a_readable_failure_report(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "budget.json").write_text("{broken", encoding="utf-8")
    real = preparation(tmp_path)
    report = real.run()
    assert report["status"] == "internal_error"
    assert "budget record unusable" in (out / "RUN.md").read_text(encoding="utf-8")
    assert json.loads((out / "feasibility.json").read_text(encoding="utf-8"))[
        "status"] == "internal_error"


def test_the_task_is_read_from_the_declaration_not_written_here(tmp_path):
    """A driver that spells one benchmark's task name is the thing this file exists not to be."""
    real = preparation(tmp_path)
    real.declaration = {"optimization_space": {"task_selection": [
        {"name": "task_name", "default": "the_task_this_checkout_is_for"}]}}
    assert real._declared_task() == "the_task_this_checkout_is_for"
    real.declaration = {"optimization_space": {"training": [{"name": "epochs"}]}}
    assert real._declared_task() == ""


def test_explicit_task_setting_overrides_the_declaration_for_command_verification(tmp_path):
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client([]),
                       scouting=tmp_path / "scouting",
                       base_settings={"task": "a_user_selected_task"})
    real.declaration = {"optimization_space": {"task_selection": [
        {"name": "task_name", "default": "a_different_task"}]}}
    assert real._declared_task() == "a_user_selected_task"


def test_existing_interpreter_hint_is_passed_to_provision_without_skipping_it(
        tmp_path, monkeypatch):
    from autosim.research import prepare as module

    seen = {}
    monkeypatch.setattr(module.provision, "build", lambda *args, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(module.provision, "env_python", lambda output: None)
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client([
        {"stages": ["evaluate"], "asset_keys": [], "why": "native scorer only"}]),
                       scouting=tmp_path / "scouting", interpreter_hint=Path(sys.executable))
    real.declaration = {"assets": {}}
    real.execution = {"stages": {"evaluate": {"available": True,
                                                "entrypoint": "eval.py"}}}
    real.do("build_the_environment")
    assert seen["python"] == str(Path(sys.executable).absolute())


def test_environment_failure_evidence_reopens_probe_verification(tmp_path, monkeypatch):
    from autosim.research import prepare as module
    from autosim.research.compute_decision import ComputeDecision

    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy", "assets": {}}
    real.execution = {"stages": {"evaluate": {
        "available": True, "entrypoint": "scripts/evaluate.py",
        "invocation": "python scripts/evaluate.py"}}}
    real.interpreter = Path(sys.executable)
    real.decision = ComputeDecision(device="cpu", device_index=0, environment={}, evidence={})
    real.steps = [
        {"step": "build_the_environment", "outcome": "done"},
        {"step": "derive_a_command", "outcome": "no runnable command",
         "arguments": {"stage": "evaluate"}, "because": "stage import failed"},
    ]
    real.output.mkdir(parents=True)
    (real.output / "environment.json").write_text(json.dumps({
        "interpreter": sys.executable,
        "probes": ["{python} -m pip install simulator_package 2>&1 | tail -20"],
        "record": [{"kind": "probe", "probe": "pip install simulator_package",
                    "ok": True, "excerpt": "Requirement already satisfied"}],
        "verdict": {"passed": True, "reason": "all probes passed"},
    }), encoding="utf-8")
    attempts = real.output / "derivation_attempts"
    attempts.mkdir()
    (attempts / "evaluate.json").write_text(json.dumps({"attempts": [
        {"status": "rejected", "error": "ModuleNotFoundError: No module named runtime_dep"}
    ]}), encoding="utf-8")
    real._select_execution_path = lambda: {"stages": ["evaluate"], "asset_keys": [],
                                           "why": "selected native evaluation"}
    seen = {}
    monkeypatch.setattr(module.provision, "build",
                        lambda *args, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(module.provision, "env_python", lambda _output: Path(sys.executable))

    state = real.state()["environment"]
    assert state["verification_status"] == "stale_probe_contract"
    assert state["probe_contract_faults"]
    assert state["stage_failure_after_verification"]["stage"] == "evaluate"

    result = real.do("build_the_environment")

    assert result["outcome"] == "done"
    assert result["verification_reopened"] is True
    assert seen["python"] == sys.executable
    assert seen["assets"]["observed_stage_failure"]["stage"] == "evaluate"
    assert "ModuleNotFoundError" in seen["assets"]["observed_stage_failure"][
        "failure_excerpt"]


def test_native_research_stage_failure_reopens_environment_claim_once(tmp_path):
    from autosim.research.prepare import _last_stage_failure_after_build

    output = tmp_path / "out"
    root = output / "research" / "derived"
    root.mkdir(parents=True)
    (root / "controller_session.json").write_text(json.dumps({
        "run_id": "derived", "status": "paused", "history": [{
            "round": 0, "status": "failed", "failure": {
                "stage": "evaluate", "reason": "ModuleNotFoundError: runtime_dep",
                "evidence": "attempts/evaluate.json"}}],
    }), encoding="utf-8")
    steps = [
        {"step": "build_the_environment", "outcome": "done"},
        {"step": "run_the_loop", "outcome": "baseline failed"},
    ]

    failure = _last_stage_failure_after_build(steps, output, "derived")

    assert failure["stage"] == "evaluate"
    assert failure["failure_kind"] == "research_stage_failure"
    assert "ModuleNotFoundError" in failure["because"]
    assert failure["evidence_ref"] == "attempts/evaluate.json"
    assert _last_stage_failure_after_build(
        [*steps, {"step": "build_the_environment", "outcome": "done"}],
        output, "derived") is None


def test_unrebutted_passing_environment_is_still_reused(tmp_path, monkeypatch):
    from autosim.research import prepare as module
    from autosim.research.compute_decision import ComputeDecision

    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy", "assets": {}}
    real.execution = {"stages": {"evaluate": {
        "available": True, "entrypoint": "scripts/evaluate.py"}}}
    real.interpreter = Path(sys.executable)
    real.decision = ComputeDecision(device="cpu", device_index=0, environment={}, evidence={})
    real.output.mkdir(parents=True)
    (real.output / "environment.json").write_text(json.dumps({
        "interpreter": sys.executable,
        "probes": ["{python} -c 'import simulator_package'"],
        "verdict": {"passed": True, "reason": "native import verified"},
    }), encoding="utf-8")
    monkeypatch.setattr(module.provision, "build",
                        lambda *_args, **_kwargs: pytest.fail("environment was needlessly rebuilt"))

    result = real._step_build_the_environment()

    assert result["outcome"] == "already done"


def test_environment_build_receives_only_the_selected_workflow(tmp_path, monkeypatch):
    from autosim.research import prepare as module

    seen = {}
    monkeypatch.setattr(module.provision, "build", lambda *args, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(module.provision, "env_python", lambda output: Path(sys.executable))
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client([
        {"stages": ["train", "evaluate"], "asset_keys": ["checkpoint"],
         "why": "online trainer consumes simulator state, not demonstrations"},
        {"compatible": True, "why": "the source save/load formats match"}]),
                       scouting=tmp_path / "scouting")
    real.declaration = {"assets": {"dataset": {"path": None},
                                   "checkpoint": {"path": None}},
                        "task_contract": {"policy_representation": "artifact"}}
    real.execution = {"stages": {
        "train": {"available": True, "entrypoint": "train.py", "invocation": "python train.py"},
        "evaluate": {"available": True, "entrypoint": "eval.py",
                     "invocation": "python eval.py --checkpoint PATH"},
        "prepare_data": {"available": True, "entrypoint": "convert.py"}}}
    assert real.do("build_the_environment")["outcome"] == "done"
    assert seen["assets"]["research_context"]["available_stages"] == ["train", "evaluate"]
    assert "dataset" not in seen["assets"]
    assert "checkpoint" not in seen["assets"]
    assert "prepare_data" not in seen["assets"]["research_context"]["stage_paths"]


def test_selected_external_asset_cannot_be_a_selected_future_stage_output(
        tmp_path, monkeypatch):
    from autosim.research import prepare as module

    seen = {}
    monkeypatch.setattr(module.provision, "build", lambda *args, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(module.provision, "env_python", lambda output: Path(sys.executable))
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client([
        {"stages": ["prepare_data", "train", "evaluate"], "asset_keys": ["dataset"],
         "why": "convert before training"},
        {"stages": ["train", "evaluate"], "asset_keys": ["dataset"],
         "why": "use the already native dataset without conversion"},
    ]), scouting=tmp_path / "scouting")
    real.declaration = {"assets": {"dataset": {"path": "test/data/test_v15.hdf5"}}}
    real.execution = {"stages": {
        "prepare_data": {"available": True, "entrypoint": "convert.py",
                         "artifact": "test_v15.hdf5"},
        "train": {"available": True, "entrypoint": "ppo.py"},
        "evaluate": {"available": True, "entrypoint": "ppo.py"}}}
    assert real.do("build_the_environment")["outcome"] == "done"
    assert seen["assets"]["research_context"]["available_stages"] == [
        "train", "evaluate"]
    attempts = json.loads((real.output / "path_selection_attempts.json").read_text())["rows"]
    assert "asset dataset is not external" in attempts[0]["why_rejected"]


def test_path_selector_reasks_after_a_missing_score_stage(tmp_path):
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=_Client([
        {"stages": ["train"], "asset_keys": [], "why": "first draft"},
        {"stages": ["train", "evaluate"], "asset_keys": [], "why": "complete path"},
    ]), scouting=tmp_path / "scouting")
    real.declaration = {"assets": {}}
    real.execution = {"stages": {
        "train": {"available": True, "entrypoint": "ppo.py"},
        "evaluate": {"available": True, "entrypoint": "ppo.py"}}}
    selected = real._select_execution_path()
    assert selected["stages"] == ["train", "evaluate"]
    attempts = json.loads((real.output / "path_selection_attempts.json").read_text())["rows"]
    assert "score target" in attempts[0]["why_rejected"]


def test_path_selection_prompt_keeps_workflow_semantics_but_not_local_paths(tmp_path):
    checkpoint = str(tmp_path / "weights" / "private_policy.pt")
    dataset = str(tmp_path / "demos" / "lift_low_dim.hdf5")
    client = _Client([{"stages": ["train", "evaluate"], "asset_keys": ["dataset"],
                       "why": "train produces a policy consumed by evaluate"}])
    real = Preparation(repo=tmp_path, output=tmp_path / "out", client=client,
                       scouting=tmp_path / "scouting")
    real.declaration = {"assets": {"dataset": {"path": dataset, "required": True}},
                        "task_contract": {"policy_representation": "artifact"}}
    real.execution = {"stages": {
        "train": {"available": True, "entrypoint": "examples/ppo.py",
                  "invocation": "python examples/ppo.py --save private_policy.pt",
                  "artifact": "runs/*/private_policy.pt"},
        "evaluate": {"available": True, "entrypoint": "examples/ppo.py",
                     "invocation": f"python examples/ppo.py --checkpoint {checkpoint}"},
    }}
    real._shipped_checkpoint = lambda: checkpoint

    selected = real._select_execution_path()

    assert selected["stages"] == ["train", "evaluate"]
    sent = json.loads(client.seen[0])
    assert sent["task"] is not None
    assert sent["shipped_checkpoint"] is True
    assert sent["stages"]["train"]["entrypoint"] == "examples/ppo.py"
    assert sent["stages"]["train"]["artifact"] == "[DECLARED_LOCAL_ARTIFACT]"
    assert "--checkpoint" in sent["stages"]["evaluate"]["invocation"]
    serialized = json.dumps(sent)
    for forbidden in (checkpoint, dataset, "private_policy.pt", "lift_low_dim.hdf5"):
        assert forbidden not in serialized


def test_selection_rejects_unproven_cross_family_checkpoint_handoff(tmp_path):
    (tmp_path / "bc.py").write_text("# BC policy save\n", encoding="utf-8")
    (tmp_path / "other_eval.py").write_text("# different policy load\n",
                                             encoding="utf-8")
    choice = {"stages": ["train", "evaluate"], "asset_keys": [],
              "why": "the BC checkpoint is not loadable by this evaluator"}
    review = {"compatible": False,
              "why": "trainer saves BC weights; evaluator loads another policy architecture"}
    real = Preparation(repo=tmp_path, output=tmp_path / "out",
                       client=_Client([choice, review] * 3),
                       scouting=tmp_path / "scouting")
    real.declaration = {"assets": {}, "task_contract": {"policy_representation": "artifact"}}
    real.execution = {"stages": {
        "train": {"available": True, "entrypoint": "bc.py", "artifact": "model.pt"},
        "evaluate": {"available": True, "entrypoint": "other_eval.py",
                     "invocation": "python other_eval.py --checkpoint model.pt"}}}
    with pytest.raises(ValueError, match="no source-supported train-to-score"):
        real._select_execution_path()
    assert not (real.output / "selected_path.json").exists()
    attempts = json.loads((real.output / "path_selection_attempts.json").read_text())
    assert len(attempts["rows"]) == 3
    assert "model.pt" not in "\n".join(real.client.seen)
    assert "--checkpoint" in real.client.seen[1]
    assert "BC policy save" in real.client.seen[1]


def test_score_path_rejects_collect_trajectories_as_a_missing_policy_checkpoint(
        tmp_path, monkeypatch):
    """A rollout/video output cannot stand in for the policy consumed by evaluation."""
    choice = {"stages": ["collect", "evaluate"], "asset_keys": [],
              "why": "collect trajectories before scoring"}
    real = Preparation(repo=tmp_path, output=tmp_path / "out",
                       client=_Client([choice, choice, choice]),
                       scouting=tmp_path / "scouting")
    real.declaration = {"assets": {}, "task_contract": {
        "policy_representation": "artifact"}}
    real.execution = {"stages": {
        "collect": {"available": True, "entrypoint": "collect.py",
                    "invocation": "python collect.py --evaluate",
                    "artifact": "trajectories/*.h5"},
        "evaluate": {"available": True, "entrypoint": "eval.py",
                     "invocation": "python eval.py --checkpoint model.pt"}}}
    monkeypatch.setattr(real, "_shipped_checkpoint", lambda: "")

    with pytest.raises(ValueError, match="no source-supported train-to-score"):
        real._select_execution_path()
    attempts = json.loads((real.output / "path_selection_attempts.json").read_text())
    assert len(attempts["rows"]) == 3
    assert all("trajectory-producing collect stage" in row["why_rejected"]
               for row in attempts["rows"])


def test_resurvey_receives_the_rejected_handoff_as_context(tmp_path, monkeypatch):
    from autosim.research import prepare as module

    real = preparation(tmp_path)
    real.output.mkdir(parents=True)
    (real.output / "path_selection_attempts.json").write_text(json.dumps({"rows": [
        {"stages": ["train", "evaluate"], "why_rejected": "checkpoint format differs"}]}),
        encoding="utf-8")
    seen = {}
    monkeypatch.setattr(module.execution_derive, "run",
                        lambda *args, **kwargs: seen.update(kwargs) or {"stages": {}})
    real._step_read_the_checkout()
    assert seen["context"]["rejected_handoffs"][0]["why_rejected"] == \
        "checkpoint format differs"


def test_deriving_a_stage_that_does_not_exist_names_the_ones_that_do(tmp_path):
    real = preparation(tmp_path)
    real.interpreter = Path("/usr/bin/python3")
    real.execution = {"stages": {"train": {"available": True}}}
    result = real.do("derive_a_command", stage="evaluate")
    assert result["outcome"] == "no such stage"
    assert "['train']" in result["because"]


def test_preparation_routes_a_custom_score_graph_to_a_baseline_measurement(tmp_path):
    real = preparation(tmp_path)
    source = ("def stage_argv_score(i):\n"
              "    return [i['python'], '-c', "
              "'import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
              "p.mkdir(parents=True,exist_ok=True); "
              "(p/\"result.txt\").write_text(\"scored\"); "
              "print(\"mean_reward: 3.0\")', i['output']]\n")
    real.declaration = {"benchmark": "synthetic",
                        "optimization_space": {"collection": [], "training": []},
                        "task_contract": {"policy_representation": "source"},
                        "research_goal": {"primary_metric": {"name": "mean_reward",
                                                             "direction": "maximize"}}}
    real.interpreter = Path(sys.executable)
    real.stages = {"score": source}
    real.base_settings = {"_rounds": 0}
    real.execution = {"stages": {"score": {"available": True,
                                           "entrypoint": "native.py",
                                           "artifact": "result.txt"}},
                      "execution_graph": {"score_target": "score",
                                          "nodes": [{"id": "score", "role": "evaluate"}]}}
    result = real.do("run_the_loop")
    assert result["outcome"] == "done", result
    assert result["metric_value"] == 3.0
    assert result["research_attempted"] is False
    assert result["independently_confirmed"] is False
    measured = json.loads((real.output / "research" / "derived" / "measurements" /
                           "baseline.json").read_text())
    assert measured["ok"] is True


def test_the_run_says_what_it_is_doing_while_it_does_it(tmp_path, monkeypatch):
    """A preparation takes as long as it takes -- a build solves a conda environment, a
    derivation runs a benchmark forty times -- and until this existed the whole thing printed
    one JSON document when it finished. A run that is working and a run that is stuck looked
    the same.
    """
    import io
    from autosim.research import prepare as module

    monkeypatch.setattr(module.execution_derive, "run",
                        lambda repo, *, client, output: {"stages": {"train": {"available": True}}})
    real = preparation(tmp_path, [
        {"do": "read_the_checkout", "why": "nothing is known yet"},
        {"do": "stop", "why": "nothing further"},
    ])
    real.out = io.StringIO()
    real.run()
    said = real.out.getvalue()
    assert "preparing" in said
    assert "starting from" in said
    # What it chose, why, and what came of it -- while it happened, not afterwards.
    assert "read_the_checkout" in said and "nothing is known yet" in said
    assert "stop —" in said


def test_the_state_says_what_is_true_now_not_how_it_got_there(tmp_path):
    """217 entries of which 210 were failures, and one line saying the stage has a command.
    The scheduler read the history and stopped, saying the derivation had not converged --
    while the line beside it said it had.

    A scheduler decides from what is true now. The history is kept -- every attempt is written
    to `derivation_attempts/` -- but it is not what the next decision is made from.
    """
    from autosim.research.prepare import _latest_per_step

    steps = ([{"step": "derive:evaluate", "outcome": "rejected",
               "because": f"ValueError: attempt {n}"} for n in range(210)]
             + [{"step": "derive:evaluate", "outcome": "accepted", "because": ""}]
             + [{"step": "read_the_checkout", "outcome": "done", "because": "four stages"}])
    rows = _latest_per_step(steps)
    assert len(rows) == 2
    evaluate = next(one for one in rows if one["step"] == "derive:evaluate")
    assert evaluate["outcome"] == "accepted"
    assert evaluate["attempts"] == 211
    # The reason travels with the outcome, not with whichever attempt last had one.
    assert "attempt 209" not in evaluate["because"]


def test_the_state_says_whether_the_run_is_ready_to_measure(tmp_path):
    """A scheduler that has to infer "therefore I may run the loop" from a list of facts will
    instead go and fix whichever fact looks broken.

    RoboTwin's did exactly that: two usable declarations on file, an interpreter, four stages
    with commands -- and it spent its steps re-declaring, because `declare` was the only step
    that had failed. The failure was untidiness, not a blocker, and nothing in the state said
    which was which.
    """
    real = preparation(tmp_path)
    assert real.state()["to_measure"]["ready"] is False
    assert "missing" in real.state()["to_measure"]["because"]

    real.declaration = {"benchmark": "x", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}}
    real.stages = {"train": "def stage_argv_train(i): ..."}
    assert real.state()["to_measure"]["ready"] is False
    assert "evaluate score command" in real.state()["to_measure"]["because"]
    real.stages["evaluate"] = "def stage_argv_evaluate(i): ..."
    real.interpreter = Path(sys.executable)
    ready = real.state()["to_measure"]
    assert ready["ready"] is True
    assert "can start" in ready["because"]
    assert "training is optional" in ready["what_it_would_do"]


def test_successful_derivation_is_saved_used_and_reloaded(tmp_path, monkeypatch):
    from autosim.research import prepare as module

    real = preparation(tmp_path)
    real.interpreter = Path("/usr/bin/python3")
    real.declaration = {"benchmark": "toy", "task_contract": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "unit": "fraction",
        "source": "log"}}, "optimization_space": {
        "collection": [], "training": [{"name": "steps", "kind": "integer",
                                           "description": "training steps", "low": 1,
                                           "high": 2, "default": 1}]}}
    real.execution = {"stages": {"train": {"available": True, "entrypoint": "train.py"}}}
    source = "def stage_argv_train(i):\n    return [i['python'], '-c', 'print(1)']\n"
    settled_row = {"available": True, "entrypoint": "train.py",
                   "working_directory": "{repo}", "environment": {"MARKER": "yes"},
                   "artifact": "model.pt"}
    monkeypatch.setattr(module.execution_derive, "make_runnable",
                        lambda *a, **k: (source, {}, [], settled_row))
    monkeypatch.setattr(module.execution_derive, "checkpoint_for_verification",
                        lambda *a, **k: {"path": ""})
    result = real.do("derive_a_command", stage="train")
    assert result["outcome"] == "done"
    kept = json.loads((real.output / "derived_stages.json").read_text(encoding="utf-8"))
    assert kept["train"]["row"]["environment"]["MARKER"] == "yes"

    seen = []

    run_calls = []

    def checked_run(research, **kwargs):
        run_calls.append(kwargs)
        seen.append(research.backend.stages["train"]["environment"]["MARKER"])
        return {"rounds": [{"success_rate": 0.5}], "objective": {"score": 0.5}}

    monkeypatch.setattr(module.DerivedResearch, "run", checked_run)
    assert real.do("run_the_loop")["outcome"] == "not attempted"
    real.stages["evaluate"] = source.replace("stage_argv_train", "stage_argv_evaluate")
    real.verified["evaluate"] = settled_row
    loop_result = real.do("run_the_loop")
    assert loop_result["outcome"] == "done", loop_result
    assert seen == ["yes"]
    assert run_calls[-1]["yield_after_action"] is True
    assert run_calls[-1]["max_rounds_per_action"] == 1
    resumed = Preparation(repo=tmp_path, output=real.output, client=_Client([]),
                          scouting=tmp_path / "scouting")
    assert resumed.stages["train"] == source
    assert resumed.verified["train"]["environment"]["MARKER"] == "yes"


def test_selected_train_producer_is_required_before_research_loop(tmp_path, monkeypatch):
    real = preparation(tmp_path)
    real.declaration = {"benchmark": "toy"}
    real.interpreter = Path("/usr/bin/python3")
    real.stages["evaluate"] = "def stage_argv_evaluate(i): return []"
    monkeypatch.setattr(real, "_metric_bound", lambda: True)
    real.output.mkdir(parents=True, exist_ok=True)
    (real.output / "selected_path.json").write_text(
        json.dumps({"stages": ["train", "evaluate"]}), encoding="utf-8")

    assert real._selected_train_command_missing()
    assert not real._research_loop_ready()
    assert real._step_run_the_loop()["outcome"] == "not attempted"

    real.stages["train"] = "def stage_argv_train(i): return []"
    assert not real._selected_train_command_missing()
    assert real._research_loop_ready()


def test_failed_derivation_keeps_attempts_but_not_a_fake_command(tmp_path, monkeypatch):
    from autosim.research import prepare as module

    real = preparation(tmp_path)
    real.interpreter = Path("/usr/bin/python3")
    real.execution = {"stages": {"train": {"available": True, "entrypoint": "train.py"}}}
    monkeypatch.setattr(module.execution_derive, "make_runnable",
                        lambda *a, **k: (None, {}, [{"status": "rejected",
                                                    "error": "missing asset"}], {}))
    monkeypatch.setattr(module.execution_derive, "checkpoint_for_verification",
                        lambda *a, **k: {"path": ""})
    result = real.do("derive_a_command", stage="train")
    assert result["outcome"] == "no runnable command"
    assert "train" not in real.stages
    assert not (real.output / "derived_stages.json").exists()
    assert (real.output / "derivation_attempts" / "train.json").is_file()
