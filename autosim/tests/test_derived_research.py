"""The research protocol, tested on a benchmark that is a stub.

No simulation. What is tested is the part that must be general: which stages exist is read
rather than assumed, a benchmark without collection is researched through the families it
has, a proposal travels as settings rather than as a command line, and the loop ends for a
stated reason rather than by running out of rounds.
"""

import json
import sys
from pathlib import Path

import pytest

from autosim.research.adapter_protocol import Axis, OptimizationSpace
from autosim.research import common
from autosim.research.compute_decision import ComputeDecision
from autosim.research.common import object_digest
from autosim.research.declarative_backend import DeclarativeBackend
from autosim.research.derived_research import DerivedResearch, _source_quote_present
from autosim.research.ideas import Idea
from autosim.research.metric_contract import MetricSpec
from autosim.research import objective, selection


#: A stage function for a benchmark whose settings are spelled as `--name value`, and whose
#: only output is one line of the shape benchmarks print. Generated rather than written by
#: hand because that is what the runner is handed in production.
ARGV = """\
def stage_argv_{stage}(i):
    argv = [i["python"], "-c", 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); (p/"models").mkdir(parents=True,exist_ok=True); (p/"models"/"model.pth").write_text("checkpoint"); print("succ: 0.50")', i["output"]]
    for key in sorted(i["settings"]):
        argv = argv + ["--" + key, str(i["settings"][key])]
    return argv
"""


def sources_for(stages) -> dict:
    return {stage: ARGV.format(stage=stage) for stage in stages}


def space() -> OptimizationSpace:
    return OptimizationSpace(training=(
        Axis("train.n_epochs", "integer", "epochs", low=1, high=100, default=1),
        Axis("train.loss_scale", "number", "loss multiplier", low=0.0, high=10.0, default=1.0)))


def research(tmp_path, *, stages=("train", "evaluate"), run_id="derived") -> DerivedResearch:
    answer = {"stages": {stage: {"available": True, "entrypoint": "x.py",
                                 "invocation": "python x.py", "artifact": "models/*.pth"}
                         for stage in stages}}
    return DerivedResearch(
        repo=tmp_path, output=tmp_path / "out",
        backend=DeclarativeBackend(repo=tmp_path, answer=answer,
                                   sources=sources_for(stages), parameters={}),
        interpreter=Path(sys.executable), space=space(), client=None,
        stages=sources_for(stages), run_id=run_id,
        # Supplied, so the suite does not depend on what cards this machine happens to have.
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="the test chose it"))


class ArtifactSelectionClient:
    """A deterministic stand-in for source-grounded checkpoint selection."""

    model = "artifact-selection-test-double"

    def __init__(self, mode="select"):
        self.mode = mode
        self.payload = None
        self.calls = 0

    def chat_with_metadata(self, system, user, **kwargs):
        self.calls += 1
        self.payload = json.loads(user)
        candidate_id = "candidate_2"
        quote = 'NATIVE_EVAL_POLICY = "candidate_2"'
        if self.mode == "invalid_candidate":
            candidate_id = "candidate_999"
        if self.mode == "invalid_quote":
            quote = "the evaluator picks the latest checkpoint"
        answer = {
            "decision": "select",
            "candidate_id": candidate_id,
            "native_rule": "The native evaluator loads the configured policy path.",
            "evidence": [{"source_path": "ppo.py", "quote": quote,
                          "why": "This is the repository's declared native policy path."}],
            "candidate_trace": "Native evaluator checkpoint: candidate_2",
            "why": "",
        }
        return json.dumps(answer), {"provider": "test", "model": self.model}


def ambiguous_policy_research(tmp_path, *, selector_mode="select"):
    real = research(tmp_path)
    (tmp_path / "ppo.py").write_text(
        'NATIVE_EVAL_POLICY = "models/two.pth"\n', encoding="utf-8")
    train_code = (
        "import pathlib,sys; p=pathlib.Path(sys.argv[1])/'models'; "
        "p.mkdir(parents=True,exist_ok=True); "
        "(p/'one.pth').write_text('one'); (p/'two.pth').write_text('two'); "
        "print('global_step=4'); "
        "print('Native evaluator checkpoint: models/two.pth')")
    eval_code = (
        "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
        "assert p.read_text() == 'two', f'unexpected policy bytes: {p}'; "
        "print('success_rate: 0.75')")
    sources = {
        "train": ("def stage_argv_train(i):\n"
                  f"    return [i['python'], '-c', {train_code!r}, i['output']]\n"),
        "evaluate": ("def stage_argv_evaluate(i):\n"
                     f"    return [i['python'], '-c', {eval_code!r}, i['checkpoint']]\n"),
    }
    answer = {"stages": {
        "train": {"available": True, "entrypoint": "ppo.py",
                  "invocation": "python ppo.py", "artifact": "models/*.pth"},
        "evaluate": {"available": True, "entrypoint": "ppo.py",
                     "invocation": "python ppo.py", "artifact": ""},
    }}
    real.backend = DeclarativeBackend(repo=tmp_path, answer=answer,
                                      sources=sources, parameters={})
    real.sources = sources
    real.client = ArtifactSelectionClient(selector_mode)
    real.require_training_progress = True
    return real


def test_which_stages_exist_is_read_not_assumed(tmp_path):
    """A benchmark with no way to produce trajectories is researched without that stage.

    LIBERO is the case: `collect` is unavailable because its only collector drives the robot
    by hand, and the loop works from the families that remain.
    """
    real = research(tmp_path, stages=("train", "evaluate"))
    assert (real.run_root / "RUN.md").is_file()
    assert real.available("train") and real.available("evaluate")
    assert not real.available("collect") and not real.available("prepare_data")
    described = real.describe()
    assert described["collect"]["available"] is False
    assert "produce new trajectories" in described["collect"]["role"]


def test_bounded_research_yields_after_baseline_and_resumes_without_retraining_it(tmp_path):
    real = research(tmp_path, run_id="bounded")
    real.library.add(Idea(label="slightly more training", granularity="param",
                          change={"train.n_epochs": 2}, risk="low", crosses="none",
                          why="test one bounded candidate action"))

    first = real.run(rounds=1, yield_after_action=True, max_rounds_per_action=1)
    assert first["run_status"] == "paused"
    assert first["next_round"] == 1
    assert (real.run_root / "measurements" / "baseline.json").is_file()
    assert (json.loads((real.run_root / "controller_session.json").read_text())[
        "current_measurement_label"] == "baseline")
    attempts_after_baseline = len(list((real.run_root / "attempts").glob("*/receipt.json")))
    assert attempts_after_baseline == 2  # one train and one native evaluation

    # A fresh controller instance takes over from the durable session and performs only the
    # candidate measurement. The baseline train/evaluate pair must not be repeated.
    resumed = research(tmp_path, run_id="bounded")
    second = resumed.run(rounds=1, yield_after_action=True, max_rounds_per_action=1)
    assert second["run_status"] == "completed"
    assert len(second["rounds"]) == 2
    assert second["rounds"][0]["label"] == "baseline"
    assert second["rounds"][1]["label"] == "round_1"
    assert len(list((resumed.run_root / "attempts").glob("*/receipt.json"))) == 4
    session = json.loads((resumed.run_root / "controller_session.json").read_text())
    assert session["status"] == "completed"
    finalizations = session["round_finalizations"]
    assert len(finalizations) == 1
    finalization = finalizations[0]
    assert finalization["status"] == "committed"
    assert finalization["round"] == 1
    assert finalization["transaction_id"] == second["rounds"][1]["finalization_id"]
    assert finalization["history_row_sha256"] == object_digest(second["rounds"][1])
    assert all(row["status"] in {"completed", "skipped"}
               for row in finalization["steps"].values())
    proof = DerivedResearch.verify_committed_candidate_finalization(
        run_root=resumed.run_root, run_id=resumed.run_id,
        repo=resumed.repo, round_index=1)
    assert proof["status"] == "verified_committed"
    assert proof["transaction_id"] == finalization["transaction_id"]
    session_path = resumed.run_root / "controller_session.json"
    tampered = json.loads(session_path.read_text(encoding="utf-8"))
    event_evidence = tampered["round_finalizations"][0]["steps"]["round_event"][
        "evidence"]
    event_evidence["summary"]["tampered"] = True
    session_path.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = DerivedResearch.verify_committed_candidate_finalization(
        run_root=resumed.run_root, run_id=resumed.run_id,
        repo=resumed.repo, round_index=1)
    assert rejected["status"] == "unverified"
    assert "summary evidence hash" in rejected["because"]


def test_paused_research_reports_changed_input_component_without_starting_an_attempt(
        tmp_path):
    from autosim.research.research_state import ResearchStateError

    real = research(tmp_path, run_id="input-component-mismatch")
    first = real.run(rounds=2, yield_after_action=True, max_rounds_per_action=1)
    assert first["run_status"] == "paused"
    session_path = real.run_root / "controller_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["input_component_sha256"]["rounds"]
    assert session["input_component_sha256"]["settings"]
    attempts_before = sorted(path.parent.name for path in
                             (real.run_root / "attempts").glob("*/receipt.json"))

    resumed = research(tmp_path, run_id="input-component-mismatch")
    with pytest.raises(ResearchStateError, match=r"\(rounds\)"):
        resumed.run(rounds=1, yield_after_action=True, max_rounds_per_action=1)

    after = json.loads(session_path.read_text(encoding="utf-8"))
    attempts_after = sorted(path.parent.name for path in
                            (resumed.run_root / "attempts").glob("*/receipt.json"))
    assert after["status"] == "paused"
    assert after["input_identity"] == session["input_identity"]
    assert attempts_after == attempts_before


def test_main_controlled_research_rejects_bad_label_before_unpausing(tmp_path):
    from autosim.research.research_state import ResearchStateError

    real = research(tmp_path, run_id="main-owned-selection")
    real.run(rounds=1, yield_after_action=True, max_rounds_per_action=1,
             main_controller_owns_selection=True)
    session_path = real.run_root / "controller_session.json"
    assert json.loads(session_path.read_text(encoding="utf-8"))["status"] == "paused"
    real.choose_idea = lambda **_: (_ for _ in ()).throw(
        AssertionError("main-controlled mode must not call the inner selector"))

    with pytest.raises(ResearchStateError, match="not in the current audited"):
        real.run(rounds=1, yield_after_action=True, max_rounds_per_action=1,
                 main_controller_owns_selection=True,
                 selected_idea_label="not-a-cleared-idea")

    assert json.loads(session_path.read_text(encoding="utf-8"))["status"] == "paused"


def test_failed_candidate_yields_failure_receipt_to_the_outer_controller(tmp_path):
    real = research(tmp_path, run_id="bounded-failure")
    real.library.add(Idea(label="larger training run", granularity="param",
                          change={"train.n_epochs": 2}, risk="low", crosses="none",
                          why="the synthetic evaluator will reject this candidate"))
    trainer = """\
def stage_argv_train(i):
    code = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); (p/'models').mkdir(parents=True,exist_ok=True); (p/'models'/'model.pth').write_text('candidate' if sys.argv[2] == '2' else 'baseline')"
    return [i["python"], "-c", code, i["output"],
            str(i["settings"].get("train.n_epochs", 1))]
"""
    evaluator = """\
def stage_argv_evaluate(i):
    code = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.read_text() == 'baseline'; print('succ: 0.50')"
    return [i["python"], "-c", code, i["checkpoint"]]
"""
    sources = {**real.sources, "train": trainer, "evaluate": evaluator}
    real.sources = sources
    real.backend = DeclarativeBackend(repo=real.repo, answer=real.backend.answer,
                                     sources=sources,
                                     parameters=real.backend.parameters)
    baseline = real.run(rounds=2, yield_after_action=True)
    assert baseline["run_status"] == "paused"

    resumed = research(tmp_path, run_id="bounded-failure")
    sources = {**resumed.sources, "train": trainer, "evaluate": evaluator}
    resumed.sources = sources
    resumed.backend = DeclarativeBackend(repo=resumed.repo, answer=resumed.backend.answer,
                                         sources=sources,
                                         parameters=resumed.backend.parameters)
    report = resumed.run(rounds=2, yield_after_action=True, max_rounds_per_action=1)
    assert report["run_status"] == "paused"
    assert report["next_round"] == 2
    candidate = report["rounds"][-1]
    assert candidate["status"] == "nothing was measured"
    assert candidate["failure"]["stage"] == "evaluate"
    assert candidate["failure"]["returncode"] == 1
    receipt_ref = candidate["failure"]["evidence"]["receipt"]
    receipt = json.loads((resumed.run_root / receipt_ref).read_text(encoding="utf-8"))
    assert receipt["node_id"] == "evaluate"
    assert receipt["status"] == "failed"
    assert receipt["returncode"] == 1


def test_interrupted_running_session_fails_closed_without_retraining(tmp_path):
    real = research(tmp_path, run_id="interrupted")
    first = real.run(rounds=1, yield_after_action=True)
    assert first["run_status"] == "paused"
    session_path = real.run_root / "controller_session.json"
    session = json.loads(session_path.read_text())
    session.update(status="running", action="round_1")
    session_path.write_text(json.dumps(session), encoding="utf-8")
    attempts_before = len(list((real.run_root / "attempts").glob("*/receipt.json")))

    resumed = research(tmp_path, run_id="interrupted")
    from autosim.research.research_state import ResearchStateError
    with pytest.raises(ResearchStateError, match="unknown outcome"):
        resumed.run(rounds=1, yield_after_action=True)
    assert len(list((real.run_root / "attempts").glob("*/receipt.json"))) == attempts_before


def test_reconciled_interrupted_baseline_reuses_verified_measurement(tmp_path):
    real = research(tmp_path, run_id="interrupted-baseline")
    real.library.add(Idea(label="one bounded change", granularity="param",
                          change={"train.n_epochs": 2}, risk="low", crosses="none",
                          why="resume after a baseline completed before controller checkpoint"))
    first = real.run(rounds=1, yield_after_action=True)
    assert first["run_status"] == "paused"
    attempts_before = len(list((real.run_root / "attempts").glob("*/receipt.json")))

    session_path = real.run_root / "controller_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session.update(status="interrupted", action="baseline", history=[],
                   current_measurement_label="",
                   interruption={"controller_action": "baseline", "stage": "evaluate",
                                 "attempt_id": "a" * 32,
                                 "evidence_ref": "attempts/" + "a" * 32 + "/receipt.json",
                                 "evidence_status": "interrupted", "outcome_known": False},
                   reconciliation={"status": "reconciled", "attempt_id": "a" * 32})
    session_path.write_text(json.dumps(session), encoding="utf-8")

    resumed = research(tmp_path, run_id="interrupted-baseline")
    from autosim.research.research_state import ResearchStateError
    with pytest.raises(ResearchStateError, match="explicit controller decision"):
        resumed.run(rounds=1, yield_after_action=True)

    completed = resumed.run(rounds=1, yield_after_action=True, resume_interrupted=True)
    assert completed["run_status"] == "completed"
    assert completed["rounds"][0]["label"] == "baseline"
    assert completed["rounds"][1]["label"] == "round_1"
    # Receipt verification recovers the baseline measurement; only the candidate train/eval
    # pair is new.
    assert len(list((resumed.run_root / "attempts").glob("*/receipt.json"))) == (
        attempts_before + 2)


def test_reconciled_interrupted_candidate_is_not_replayed(tmp_path):
    real = research(tmp_path, run_id="interrupted-candidate")
    first = real.run(rounds=1, yield_after_action=True)
    assert first["run_status"] == "paused"
    attempts_before = len(list((real.run_root / "attempts").glob("*/receipt.json")))
    session_path = real.run_root / "controller_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session.update(status="interrupted", action="round_1",
                   interruption={"controller_action": "round_1", "stage": "train",
                                 "attempt_id": "b" * 32,
                                 "evidence_ref": "attempts/" + "b" * 32 + "/receipt.json",
                                 "evidence_status": "interrupted", "outcome_known": False},
                   reconciliation={"status": "reconciled", "attempt_id": "b" * 32})
    session_path.write_text(json.dumps(session), encoding="utf-8")

    resumed = research(tmp_path, run_id="interrupted-candidate")
    from autosim.research.research_state import ResearchStateError
    with pytest.raises(ResearchStateError, match="candidate action was interrupted"):
        resumed.run(rounds=1, yield_after_action=True, resume_interrupted=True)
    assert len(list((resumed.run_root / "attempts").glob("*/receipt.json"))) == attempts_before


def test_candidate_pending_decision_is_durable_before_measurement(tmp_path, monkeypatch):
    real = research(tmp_path, run_id="pending-candidate")
    real.library.add(Idea(label="increase training", granularity="param",
                          change={"train.n_epochs": 2}, risk="low", crosses="none",
                          why="durability regression"))
    first = real.run(rounds=1, yield_after_action=True)
    assert first["run_status"] == "paused"

    resumed = research(tmp_path, run_id="pending-candidate")
    original_measure = resumed.measure

    def interrupted_measure(*, settings, label, dataset=None):
        if label == "round_1":
            raise RuntimeError("simulated controller interruption before measurement")
        return original_measure(settings=settings, label=label, dataset=dataset)

    monkeypatch.setattr(resumed, "measure", interrupted_measure)
    with pytest.raises(RuntimeError, match="simulated controller interruption"):
        resumed.run(rounds=1, yield_after_action=True, max_rounds_per_action=1)

    session = json.loads((resumed.run_root / "controller_session.json").read_text())
    pending = session["pending_action"]
    assert session["status"] == "running"
    assert session["action"] == "round_1"
    assert pending["round"] == 1
    assert pending["phase"] == "measurement_starting"
    assert pending["idea"]["label"] == "increase training"
    assert pending["settings"]["train.n_epochs"] == 2
    assert not (resumed.run_root / "measurements" / "round_1.json").exists()


def test_finalizing_session_resumes_report_without_another_benchmark_action(tmp_path):
    real = research(tmp_path, run_id="finalizing")
    report = real.run(rounds=1, yield_after_action=True)
    session_path = real.run_root / "controller_session.json"
    session = json.loads(session_path.read_text())
    session.update(status="finalizing", action="select_best_confirm_and_export",
                   next_round=2)
    session_path.write_text(json.dumps(session), encoding="utf-8")
    attempts_before = len(list((real.run_root / "attempts").glob("*/receipt.json")))

    resumed = research(tmp_path, run_id="finalizing")
    completed = resumed.run(rounds=1, yield_after_action=True)
    assert completed["run_status"] == "completed"
    assert len(list((resumed.run_root / "attempts").glob("*/receipt.json"))) == attempts_before
    assert report["run_status"] == "paused"


def test_an_unavailable_stage_says_so_rather_than_failing(tmp_path):
    result = research(tmp_path).run_stage("collect", settings={})
    assert result["ran"] is False and "unavailable" in result["why"]


def test_gpu_task_occupancy_is_settled_even_when_stage_raises(tmp_path, monkeypatch):
    from autosim.research import derived_research as module
    from autosim.research.task_budget import TaskGPUBudget
    real = research(tmp_path)
    ledger = TaskGPUBudget(real.output)
    ledger.initialize(real.repo, cap_seconds=100)
    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module, "recheck_gpu", lambda *a, **k: None)
    real.compute = ComputeDecision(device="cuda", device_index=0, environment={},
                                   why="fixture", evidence={"selected_device": {"uuid": "GPU-fixture"}})
    released = []
    class Lease:
        def __init__(self, *a, **k): pass
        def acquire(self): return self
        def release(self): released.append(True)
    monkeypatch.setattr(module, "GPUResourceLease", Lease)
    def fail(stage, **kwargs):
        assert kwargs["_gpu_deadline"] == 1100
        clock[0] += 12
        raise RuntimeError("stage failed")
    monkeypatch.setattr(real, "_run_stage", fail)
    with pytest.raises(RuntimeError, match="stage failed"):
        real.run_stage("train", settings={})
    record = json.loads(ledger.path.read_text())
    assert record["charged_seconds"] == 12
    assert released == [True]
    assert all(row["status"] == "finished" for row in record["leases"].values())


def test_gpu_deadline_clamps_native_process_timeout(tmp_path, monkeypatch):
    import time
    from autosim.research import derived_research as module
    from autosim.research.process_executor import ProcessAttempt
    real = research(tmp_path)
    seen = []
    def process(*args, **kwargs):
        seen.append(kwargs["timeout"])
        return ProcessAttempt(False, None, error=PermissionError("fixture refusal"))
    monkeypatch.setattr(module, "run_process", process)
    monkeypatch.setattr(module, "isolated_argv", lambda argv, **kw: argv)
    real._run_stage("train", settings={}, timeout=1000, _gpu_deadline=time.monotonic() + 5)
    assert len(seen) == 1 and 0 < seen[0] <= 5


def test_benchmark_stage_refuses_weak_containment_even_with_operator_opt_in(
        tmp_path, monkeypatch):
    monkeypatch.setattr(common.shutil, "which", lambda _name: None)
    monkeypatch.setenv("AUTOSIM_ALLOW_PROCESS_GROUP_ONLY", "1")
    real = research(tmp_path, run_id="no-pid-namespace")

    result = real.run_stage("train", settings={})

    assert result["ran"] is False
    assert result["containment_mode"] == "unavailable"
    assert result["termination_reason"] == "launch_error"
    assert "requires a bubblewrap PID namespace" in result["said"]
    assert result["process_identity"] is None
    saved = json.loads((real.run_root / "attempts" / result["attempt_id"] /
                        "receipt.json").read_text(encoding="utf-8"))
    assert saved["ran"] is False
    assert saved["containment_mode"] == "unavailable"
    assert saved["containment_requirement"] == "pid_namespace"


def test_expired_persisted_budget_blocks_a_new_stage(tmp_path, monkeypatch):
    from autosim.research import budget

    clock = [100.0]
    monkeypatch.setattr(budget.time, "time", lambda: clock[0])
    budget.RunBudget(tmp_path / "out", wall_seconds=5)
    clock[0] = 106.0
    real = research(tmp_path)
    result = real.run_stage("train", settings={})
    assert result["status"] == "blocked" and result["ran"] is False
    assert "budget exhausted" in result["why"]
    assert not (real.run_root / "train" / "output.log").exists()


def test_gpu_resource_refusal_is_a_durable_block_not_a_benchmark_failure(tmp_path):
    from autosim.research.devices import NoCompatibleDevice

    real = research(tmp_path, run_id="busy-gpu")

    def refuse(*_args, **_kwargs):
        raise NoCompatibleDevice("all visible GPUs are busy",
                                 evidence={"busy_devices": [{"index": 0,
                                                              "memory_used_mib": 9000}]})

    real.compute_for = refuse
    result = real.run_stage("train", settings={})

    assert result["status"] == "blocked"
    assert result["ran"] is False
    assert result["returncode"] is None
    receipt = json.loads((real.run_root / "attempts" / result["attempt_id"] /
                          "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["command_started"] is False
    assert receipt["termination_reason"] == "gpu_unavailable"
    assert receipt["resource_block"]["evidence"]["busy_devices"][0][
        "memory_used_mib"] == 9000
    assert "process_identity" not in receipt
    assert not (real.stage_directory("train") / "output.log").exists()


def test_unavailable_execution_context_cannot_silently_run_a_stage_on_cpu(tmp_path):
    real = research(tmp_path, run_id="gpu-not-visible")
    real.compute = ComputeDecision(
        device="unavailable", device_index=0,
        why="physical GPU is not visible to CUDA",
        evidence={"resource_unavailable": True, "gpus_seen": 1})

    result = real.run_stage("train", settings={})

    assert result["status"] == "blocked" and result["ran"] is False
    assert "not visible to CUDA" in result["why"]
    receipt = json.loads((real.run_root / "attempts" / result["attempt_id"] /
                          "receipt.json").read_text(encoding="utf-8"))
    assert receipt["termination_reason"] == "gpu_unavailable"
    assert receipt["command_started"] is False


def test_each_stage_attempt_has_a_durable_receipt_and_log(tmp_path):
    real = research(tmp_path)
    first = real.run_stage("train", settings={})
    second = real.run_stage("train", settings={})
    assert first["attempt_id"] != second["attempt_id"]
    for result in (first, second):
        receipt = json.loads((real.run_root / "attempts" / result["attempt_id"] /
                              "receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "completed"
        assert receipt["returncode"] == 0
        assert Path(receipt["log"]).is_file()
        assert receipt["process_identity"]["run_id"] == real.run_id
        assert receipt["process_identity"]["attempt_id"] == result["attempt_id"]
        assert receipt["process_identity"]["pid"] > 0
        assert receipt["telemetry"]["status"] == "completed"
        assert receipt["telemetry"]["sample_count"] >= 1
        assert (real.output / receipt["telemetry"]["chart_ref"]).is_file()
    assert real.stage_directory("train").joinpath("output.log").is_file()
    shared = json.loads((real.output / "run_events.json").read_text(encoding="utf-8"))
    started = [row for row in shared["rows"] if row["event"] == "stage_process_started"]
    assert len(started) == 2
    assert all(row["details"]["process_identity"]["pgid"] ==
               row["details"]["process_identity"]["pid"] for row in started)


def test_training_telemetry_updates_during_stage_and_projects_to_top_level(
        tmp_path, monkeypatch):
    from autosim.research import derived_research as module, run_record

    real = research(tmp_path)
    real.backend.sources["train"] = (
        "def stage_argv_train(i):\n"
        "    return [i['python'], '-c', \"import time; "
        "print('global_step=1 train_loss=1.0', flush=True); "
        "time.sleep(0.18); "
        "print('global_step=2 train_loss=0.5', flush=True); "
        "time.sleep(0.18)\"]\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)

    original_context = module._live_stage_document
    poll_updates = []

    class CountingWriter(module.telemetry.LiveTelemetryWriter):
        def poll(self):
            state = super().poll()
            poll_updates.append(state["sample_count"])
            return state

    monkeypatch.setattr(module.telemetry, "LiveTelemetryWriter", CountingWriter)

    def fast_context(*args, **kwargs):
        return original_context(*args, telemetry_poll_seconds=0.03,
                                heartbeat_seconds=0.06, **kwargs)

    monkeypatch.setattr(module, "_live_stage_document", fast_context)
    real.output.mkdir(parents=True, exist_ok=True)
    (real.output / "RUN.md").write_text(
        "# Synthetic run\n\n<!-- AUTOSIM_RESEARCH_START -->\n"
        "<!-- AUTOSIM_RESEARCH_END -->\n", encoding="utf-8")

    result = real.run_stage("train", settings={})
    receipt = json.loads((real.run_root / "attempts" / result["attempt_id"] /
                          "receipt.json").read_text(encoding="utf-8"))
    view = run_record.build_report_view(real.output, real.run_id)
    report = (real.output / "RUN.md").read_text(encoding="utf-8")

    assert receipt["telemetry"]["status"] == "completed"
    assert receipt["telemetry"]["sample_count"] == 2
    assert 1 in poll_updates and 2 in poll_updates
    assert view["telemetry_attempts"][0]["status"] == "completed"
    assert view["telemetry_attempts"][0]["metrics"][0]["last_step"] == 2
    assert "Native training observations" in report
    assert f"report/telemetry/{result['attempt_id']}.svg" in report
    assert "official score" in report


def test_telemetry_observer_failure_does_not_fail_training_or_scoring(tmp_path, monkeypatch):
    from autosim.research import derived_research as module

    real = research(tmp_path)
    original = module.telemetry.LiveTelemetryWriter

    class FailingObserver(original):
        def poll(self):
            raise RuntimeError("synthetic observer failure")

    monkeypatch.setattr(module.telemetry, "LiveTelemetryWriter", FailingObserver)

    result = real.run_stage("train", settings={})

    assert result["status"] == "completed"
    assert result["returncode"] == 0
    receipt = json.loads((real.run_root / "attempts" / result["attempt_id"] /
                          "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["telemetry"]["status"] == "completed"
    assert "observer_RuntimeError" in receipt["telemetry"]["errors"]


def test_telemetry_finalize_failure_is_reported_without_failing_training(
        tmp_path, monkeypatch):
    from autosim.research import derived_research as module

    real = research(tmp_path)
    original = module.telemetry.LiveTelemetryWriter

    class FailingFinalize(original):
        def finalize(self, status):
            raise OSError("synthetic report storage failure")

    monkeypatch.setattr(module.telemetry, "LiveTelemetryWriter", FailingFinalize)

    result = real.run_stage("train", settings={})

    assert result["status"] == "completed"
    receipt = json.loads((real.run_root / "attempts" / result["attempt_id"] /
                          "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["telemetry"]["status"] == "unavailable"
    assert receipt["telemetry"]["errors"] == ["finalize_OSError"]


def test_stage_is_not_launched_when_its_start_event_cannot_be_committed(tmp_path, monkeypatch):
    from autosim.research import derived_research as module
    from autosim.research.research_state import ResearchStateError

    real = research(tmp_path)
    launched = []
    monkeypatch.setattr(module, "run_process",
                        lambda *args, **kwargs: launched.append(True))
    original_record = real.state_store.record

    def fail_start_event(phase, event, **kwargs):
        if event == "stage_started":
            raise OSError("injected stage-start journal failure")
        return original_record(phase, event, **kwargs)

    monkeypatch.setattr(real.state_store, "record", fail_start_event)
    with pytest.raises(ResearchStateError, match="stage-start journal failure"):
        real.run_stage("train", settings={})

    receipts = list((real.run_root / "attempts").glob("*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["termination_reason"] == "run_state_unavailable"
    assert launched == []
    assert real.state_persistence_error


def test_failed_process_identity_event_kills_worker_and_receipt_recovers_identity(
        tmp_path, monkeypatch):
    from autosim.research.research_state import ResearchStateError
    from autosim.research.process_executor import inspect_process_identity
    from autosim.research.prepare import Preparation

    real = research(tmp_path)
    source = ('def stage_argv_train(i):\n'
              '    return [i["python"], "-c", "import time; time.sleep(30)"]\n')
    real.sources["train"] = source
    real.backend.sources["train"] = source
    original_record = real.state_store.record

    def fail_identity_event(phase, event, **kwargs):
        if event == "stage_process_started":
            raise OSError("injected process-identity journal failure")
        return original_record(phase, event, **kwargs)

    monkeypatch.setattr(real.state_store, "record", fail_identity_event)
    with pytest.raises(ResearchStateError, match="process-identity journal failure"):
        real.run_stage("train", settings={})

    receipt_path = next((real.run_root / "attempts").glob("*/receipt.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = receipt["process_identity"]
    assert receipt["status"] == "failed"
    assert inspect_process_identity(identity)["status"] == "not_running"

    restarted = Preparation(repo=tmp_path, output=real.output, client=object(),
                            scouting=tmp_path / "scouting")
    recovery = restarted.state()["recovery_after_interruption"]
    assert recovery["action"]["step"] == "train"
    assert recovery["process_status"]["status"] == "not_running"
    result = restarted.do("reconcile_interrupted_action")
    assert result["outcome"] == "interrupted; outcome unknown"
    assert result["process_status"] == "not_running"


def test_failed_staging_clears_the_live_markdown_status(tmp_path):
    real = research(tmp_path)
    real.backend.stages["train"]["staging"] = ["false"]
    result = real.run_stage("train", settings={})
    assert result["ran"] is False
    assert "AUTOSIM_LIVE_START" not in (real.run_root / "RUN.md").read_text(
        encoding="utf-8")


def test_a_proposal_travels_as_settings_and_reaches_the_command(tmp_path):
    """The controller varies names from the declared space; the argv function spells them."""
    real = research(tmp_path)
    inputs = real._inputs("evaluate", settings={"train.loss_scale": 2.0})
    assert real.backend.argv("evaluate", inputs)[-2:] == ["--train.loss_scale", "2.0"]


def test_protocol_controls_are_not_forwarded_as_hyperparameter_flags(tmp_path):
    real = research(tmp_path)
    inputs = real._inputs("train", settings={"task": "PickCube-v1", "steps": 8192,
                                           "episodes": 2, "seed": 3,
                                           "train.learning_rate": 0.001})
    assert inputs["task"] == "PickCube-v1"
    assert inputs["steps"] == 8192
    assert inputs["episodes"] == 2
    assert inputs["seed"] == 3
    assert inputs["settings"] == {"train.learning_rate": 0.001}


def test_missing_training_budget_does_not_use_verification_budget(tmp_path):
    real = research(tmp_path)
    assert real._inputs("train", settings={})["steps"] == 0
    assert real._inputs("train", settings={'train.n_epochs':2})['steps'] == 0
    assert real._inputs("train", settings={'steps':8192})['steps'] == 8192
    # An explicit zero remains explicit so the native-progress guard can reject it; it is
    # not silently rewritten into a run the caller did not request.
    assert real._inputs("train", settings={"steps": 0})["steps"] == 0


def test_parameter_idea_can_rescue_an_executable_stage_before_first_score(tmp_path):
    real = research(tmp_path)
    real.backend.sources["train"] = (
        "def stage_argv_train(i):\n"
        "    if i['settings'].get('train.n_epochs', 8) > 2:\n"
        "        return [i['python'], '-c', 'raise MemoryError(\"OOM\")']\n"
        "    return [i['python'], '-c', 'import pathlib,sys; p=pathlib.Path(sys.argv[1])/\"models\"; p.mkdir(parents=True,exist_ok=True); (p/\"model.pth\").write_text(\"trained\")', i['output']]\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    real.library.add(Idea(label="smaller training batch", granularity="param",
                          change={"train.n_epochs": 1}, risk="low", crosses="none",
                          why="the current training invocation is too large"))
    report = real.run(rounds=1)
    assert report["rounds"][0]["measured"] is False
    assert report["rounds"][1]["status"] == "measured", report["rounds"]


def test_measurement_archives_the_policy_before_a_later_round_overwrites_it(tmp_path):
    real = research(tmp_path)
    first = real.measure(settings={}, label="baseline")
    assert first["ok"] is True and first["restorable_policy"] is True
    archive = Path(first["policy_artifact"]["path"])
    assert first["evaluated_policy_path"] == str(archive)
    assert archive.read_text(encoding="utf-8") == "checkpoint"
    receipt = json.loads((real.run_root/'attempts'/first['train']['attempt_id']/'receipt.json').read_text())
    stage_policy = Path(receipt["output_directory"]) / "models" / "model.pth"
    stage_policy.write_text("overwritten", encoding="utf-8")
    assert archive.read_text(encoding="utf-8") == "checkpoint"


@pytest.mark.parametrize("non_ascii", [False, True])
def test_native_producer_gets_absent_output_leaf_on_every_attempt(tmp_path, monkeypatch, non_ascii):
    repo = tmp_path / "中文仓库" if non_ascii else tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("AUTOSIM_STAGE_ROOT", str(tmp_path / "ascii-alias"))
    real = research(repo)
    code = ("import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "assert not p.exists(), 'refusing to overwrite existing native output'; "
            "(p/'models').mkdir(parents=True); "
            "(p/'models'/'model.pth').write_text('checkpoint')")
    real.backend.sources["train"] = (
        "def stage_argv_train(i):\n"
        f"    return [i['python'], '-c', {code!r}, i['output']]\n")
    real.backend = DeclarativeBackend(repo=repo, answer=real.backend.answer,
        sources=real.backend.sources, parameters={})
    first = real.run_stage("train", settings={})
    second = real.run_stage("train", settings={})
    assert first["status"] == second["status"] == "completed"
    a, b = Path(first["output_directory"]), Path(second["output_directory"])
    assert a != b
    assert (a/'models'/'model.pth').read_text() == 'checkpoint'
    assert (b/'models'/'model.pth').read_text() == 'checkpoint'
    for record, root in ((first, a), (second, b)):
        receipt = json.loads((real.run_root/'attempts'/record['attempt_id']/'receipt.json').read_text())
        assert receipt['output_directory'] == str(root.resolve())
        log = real.stage_directory('train')/'attempts'/record['attempt_id']/'output.log'
        assert log.is_file() and not log.resolve().is_relative_to(root.resolve())


@pytest.mark.parametrize('fault', ['', 'duplicate', 'changed_specimen', 'changed_producer', 'outside_producer', 'different_name', 'stale', 'symlink', 'changed_metric', 'exact_ambiguous'])
def test_bound_metric_relocation_requires_original_bytes_and_frozen_unique_producer(tmp_path, fault):
    from autosim.research.common import atomic_json, digest
    real = research(tmp_path)
    producer = 'results/**/*.json'
    original_pattern = 'results/old-timestamp/score.json'
    spec = MetricSpec.from_declaration({'research_goal': {'primary_metric': {
        'name': 'return', 'direction': 'maximize', 'source': 'json',
        'json_key': 'reward', 'artifact_root': 'output',
        'artifact_pattern': original_pattern}}})
    real.metric_spec = spec
    real.backend.stages['evaluate']['artifact'] = producer
    specimen = real.output/'verification_outputs'/original_pattern
    specimen.parent.mkdir(parents=True)
    specimen.write_text('{"reward": 0}')
    declaration = {'name': 'return', 'direction': 'maximize', 'source': 'json',
                   'json_key': 'reward', 'artifact_root': 'output',
                   'artifact_pattern': original_pattern}
    binding = {'primary_metric': declaration, 'evidence': 'source emits this scored product',
               'verified_artifact': {'path': str(specimen), 'sha256': digest(specimen),
                                     'root': 'output', 'pattern': original_pattern}}
    atomic_json(real.output/'metric_binding.json', binding)
    protocol = {'metric': spec.as_dict(), 'evaluator_stage': {'artifact': producer}}
    atomic_json(real.run_root/'comparison_protocol.json', protocol)
    output = real.run_root/'evaluate'/'native_outputs'/'attempt'
    new = output/'results'/'new-timestamp'/'score.json'
    new.parent.mkdir(parents=True)
    new.write_text('{"reward": 1}')
    started = new.stat().st_mtime - 1
    if fault == 'duplicate':
        other = output/'results'/'another'/'score.json'
        other.parent.mkdir(); other.write_text('{"reward": 2}')
    elif fault == 'changed_specimen':
        specimen.write_text('{"reward": 3}')
    elif fault == 'changed_producer':
        real.backend.stages['evaluate']['artifact'] = '**/*.json'
    elif fault == 'outside_producer':
        new.rename(output/'score.json')
    elif fault == 'different_name':
        new.rename(new.with_name('other.json'))
    elif fault == 'stale':
        started = new.stat().st_mtime + 1
    elif fault == 'symlink':
        new.unlink(); new.symlink_to(specimen)
    elif fault == 'changed_metric':
        protocol['metric']['json_key'] = 'other'
        atomic_json(real.run_root/'comparison_protocol.json', protocol)
    elif fault == 'exact_ambiguous':
        # Exact existing products do not fall through to producer-based relocation.
        exact = output/original_pattern
        exact.parent.mkdir(parents=True); exact.write_text('{"reward": 4}')
    result = real._resolve_bound_metric_artifact(spec, roots={'output': output},
        started_at=started, allowed_roots=[real.run_root])
    if not fault:
        assert result['status'] == 'matched'
        assert result['path'] == str(new.resolve())
        assert result['pattern'] == original_pattern
        assert result['producer_pattern'] == producer
        assert result['metric_mapping_unchanged'] is True
        assert spec.read(said='', artifact=new)['value'] == 1
    elif fault == 'exact_ambiguous':
        assert result['status'] == 'matched'
        assert 'producer_pattern' not in result
        assert result['path'] == str(exact.resolve())
    else:
        assert result['status'] != 'matched'


@pytest.mark.parametrize('native_exit', [0, 1])
@pytest.mark.parametrize('verdict', ['observed', 'unknown'])
def test_formal_quiet_trainer_reuses_sealed_source_backed_progress_audit(tmp_path, monkeypatch, native_exit, verdict):
    from types import SimpleNamespace
    from autosim.research.evidence_store import read_attempt_evidence
    real = research(tmp_path)
    real.require_training_progress = True
    real.client = SimpleNamespace(supports_native_progress_audit=True)
    code = ("import pathlib,sys; p=pathlib.Path(sys.argv[1])/'models'; "
            "p.mkdir(parents=True); (p/'model.pth').write_text('checkpoint'); "
            f"sys.exit({native_exit})")
    real.backend.sources['train'] = (
        'def stage_argv_train(i):\n'
        f"    return [i['python'], '-c', {code!r}, i['output']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
        sources=real.backend.sources, parameters={})
    calls = []
    def audit(client, **kwargs):
        calls.append(kwargs)
        proof = read_attempt_evidence(real.output, Path(kwargs['native_evidence_ref']).stem, limit=1)
        assert proof['status'] == 'completed'
        return {'status': verdict, 'audit_ref': 'training_progress_audits/receipt.json'}
    monkeypatch.setattr('autosim.research.training_progress_audit.audit', audit)
    record = real.run_stage('train', settings={})
    assert len(calls) == (1 if native_exit == 0 else 0)
    assert record['training_progress']['status'] == (verdict if native_exit == 0 else 'unknown')
    if calls:
        assert calls[0]['output'] == Path(record['output_directory'])
        receipt = json.loads((real.run_root/'attempts'/record['attempt_id']/'receipt.json').read_text())
        assert receipt['training_progress']['audit_ref'] == 'training_progress_audits/receipt.json'


@pytest.mark.parametrize('fault', ['', 'counter', 'audit', 'receipt', 'log'])
def test_parent_reviews_detached_progress_without_rewriting_native_receipt(tmp_path, monkeypatch, fault):
    from autosim.research import training_progress_audit as pa
    from autosim.research.common import atomic_json, digest
    real = research(tmp_path)
    real.require_training_progress = True
    real._comparison_protocol_violation({}, target='evaluate', freeze=True)
    code = ("import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); "
            "(p/'models').mkdir(parents=True); (p/'models'/'model.pth').write_text('checkpoint'); "
            "(p/'work.json').write_text(json.dumps({'done': {'updates': 17}}))")
    real.backend.sources['train'] = 'def stage_argv_train(i):\n' + f"    return [i['python'], '-c', {code!r}, i['output']]\n"
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources, parameters={})
    record = real.run_stage('train', settings={})
    attempt = record['attempt_id']
    receipt_path = real.run_root/'attempts'/attempt/'receipt.json'
    original_bytes = receipt_path.read_bytes()
    assert record['training_progress']['status'] == 'unknown'
    assert real._verified_training_receipt(attempt, {}) is None
    producer = tmp_path/'producer.py'
    producer.write_text('optimizer.step()\nstep += 1\nsave(step)\n')
    source = {'id': 'source-0', 'path': producer, 'sha256': digest(producer),
              'content': producer.read_text(), 'excerpt': producer.read_text()}
    monkeypatch.setattr(pa, 'sources', lambda *args: [source])
    class Client:
        supports_native_progress_audit = True
        calls = 0
        def chat_with_metadata(self, *args, **kwargs):
            self.calls += 1
            return json.dumps({'verdict': 'verified',
                'counter': {'path': 'work.json', 'field': ['done', 'updates'], 'value': 17},
                'source_evidence': [{'source_id': 'source-0', 'quote': producer.read_text()}],
                'why': 'completed native updates'}), {}
    real.client = Client()
    reviewed = real._verified_training_receipt(attempt, {}, audit_progress=True)
    assert reviewed is not None and reviewed['training_progress']['status'] == 'observed'
    assert receipt_path.read_bytes() == original_bytes
    assert real._verified_training_receipt(attempt, {}) is not None
    assert real.client.calls == 1
    if fault == 'counter':
        (Path(record['output_directory'])/'work.json').write_text('{"done": {"updates": 18}}')
    elif fault == 'audit':
        audit_path = real.output/reviewed['training_progress']['audit_ref']
        v=json.loads(audit_path.read_text()); v['extra']='changed'; atomic_json(audit_path, v)
    elif fault == 'receipt':
        v=json.loads(receipt_path.read_text()); v['extra']='changed'; atomic_json(receipt_path, v)
    elif fault == 'log':
        log=real.stage_directory('train')/'attempts'/attempt/'output.log'; log.write_text('changed')
    if fault:
        assert real._verified_training_receipt(attempt, {}) is None
        assert real.client.calls == 1


@pytest.mark.parametrize('fault', ['', 'unbound', 'changed_proof'])
def test_quiet_policy_selection_accepts_only_verified_metadata_association(tmp_path, monkeypatch, fault):
    real=ambiguous_policy_research(tmp_path)
    record=real.run_stage('train',settings={})
    record['said']='quiet native trainer'
    count=[0]
    def proof(*args):
        count[0]+=1
        return ({'candidate_2':{'completed_work':17}} if fault!='unbound' else {}, [],
                'changed' if fault=='changed_proof' and count[0]>1 else 'proof')
    monkeypatch.setattr(real,'_policy_work_evidence',proof)
    original=real.client.chat_with_metadata
    def choose(system,user,**kwargs):
        content,metadata=original(system,user,**kwargs)
        answer=json.loads(content);answer['selection_basis']='verified_completed_work_container'
        answer['candidate_trace']=''
        return json.dumps(answer),metadata
    real.client.chat_with_metadata=choose
    selected=real._select_policy_artifact(record)
    assert selected['artifact_selection']['status']==('selected' if not fault else 'abstained')


@pytest.mark.parametrize('fault', ['', 'counter', 'source'])
def test_policy_work_association_is_native_counter_container_not_filename(tmp_path, monkeypatch, fault):
    from autosim.research import training_progress_audit as pa
    from autosim.research.common import digest
    real=research(tmp_path);real.require_training_progress=True
    real._comparison_protocol_violation({},target='evaluate',freeze=True)
    real.backend.stages['train']['artifact']='*/policy_bundle'
    code=("import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); "
          "(p/'areaA'/'policy_bundle').mkdir(parents=True); "
          "(p/'areaB'/'policy_bundle').mkdir(parents=True); "
          "(p/'areaAlias').symlink_to('areaA',target_is_directory=True); "
          "(p/'areaA'/'proof').mkdir(); "
          "(p/'areaA'/'proof'/'work.json').write_text(json.dumps({'done':{'updates':17}}))")
    real.backend.sources['train']='def stage_argv_train(i):\n'+f"    return [i['python'],'-c',{code!r},i['output']]\n"
    real.backend=DeclarativeBackend(repo=tmp_path,answer=real.backend.answer,sources=real.backend.sources,parameters={})
    record=real.run_stage('train',settings={})
    producer=tmp_path/'producer.py';producer.write_text('optimizer.step()\nstep += 1\nsave_checkpoint(model, step)\n')
    row={'id':'source-0','path':producer,'sha256':digest(producer),'content':producer.read_text(),'excerpt':producer.read_text()}
    monkeypatch.setattr(pa,'sources',lambda *args:[row])
    class Client:
        supports_native_progress_audit=True
        def chat_with_metadata(self,*args,**kwargs):
            return json.dumps({'verdict':'verified','counter':{'path':'areaA/proof/work.json',
                'field':['done','updates'],'value':17},'source_evidence':[{
                'source_id':'source-0','quote':producer.read_text()}]}),{}
    real.client=Client()
    reviewed=real._verified_training_receipt(record['attempt_id'],{},audit_progress=True)
    assert reviewed is not None
    if fault=='counter':(Path(record['output_directory'])/'areaA/proof/work.json').write_text('{"done":{"updates":18}}')
    if fault=='source':producer.write_text('different producer')
    bindings,sources,proof=real._policy_work_evidence(reviewed,{
        'candidate_1':'areaA/policy_bundle','candidate_2':'areaB/policy_bundle',
        'candidate_3':'areaAlias/policy_bundle'})
    if fault:
        assert not bindings and not proof
    else:
        assert set(bindings)=={'candidate_1','candidate_3'}
        assert bindings['candidate_1']['completed_work']==17
        assert bindings['candidate_1']['equivalent_candidate_ids']==['candidate_1','candidate_3']
        assert bindings['candidate_3']['equivalent_candidate_ids']==['candidate_1','candidate_3']
        assert sources[0]['path']=='native_source_source-0' and proof


@pytest.mark.parametrize('fault', ['', 'changed_inputs', 'later_score', 'settings'])
def test_failed_baseline_can_evaluate_exact_completed_replacement_without_retraining(tmp_path, monkeypatch, fault):
    real = research(tmp_path)
    real.require_training_progress = True
    code = ("import pathlib,sys; marker=pathlib.Path(sys.argv[2])/'first-attempt'; "
            "seen=marker.exists(); marker.write_text('seen'); "
            "sys.exit(1) if not seen else None; "
            "p=pathlib.Path(sys.argv[1])/'models'; p.mkdir(parents=True); "
            "(p/'model.pth').write_text('checkpoint'); print('global_step=17')")
    source = 'def stage_argv_train(i):\n'+f"    return [i['python'], '-c', {code!r}, i['output'], i['repo']]\n"
    real.sources['train'] = source
    real.backend.sources['train'] = source
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources, parameters={})
    report=real.run(rounds=1, yield_after_action=True)
    assert report['run_status']=='paused'
    baseline_path=real.run_root/'measurements/baseline.json'
    original=json.loads(baseline_path.read_text())
    assert original['where']=='train' and not original['ok']
    replacement=real.run_stage('train', settings={'train.n_epochs': 2} if fault=='settings' else {})
    assert replacement['status']=='completed' and replacement['training_progress']['status']=='observed'
    if fault=='changed_inputs':
        real.sources['train'] += '\n'
    elif fault=='later_score':
        session_path=real.run_root/'controller_session.json'; s=json.loads(session_path.read_text())
        s['history'].append({'label':'round_1', 'measured':True, 'metric_value':0.75})
        session_path.write_text(json.dumps(s))
    old=real.run_stage
    calls=[]
    def score_only(stage, **kwargs):
        calls.append(stage)
        assert stage=='evaluate', 'completed replacement must never be retrained'
        return old(stage, **kwargs)
    monkeypatch.setattr(real, 'run_stage', score_only)
    result=real.recover_unscored_baseline(replacement_training_attempt_id=replacement['attempt_id'])
    if fault:
        assert result['ok'] is False and not calls
        assert json.loads(baseline_path.read_text())==original
    else:
        assert result['ok'] is True and calls==['evaluate']
        assert result['recovery']['training_reused'] is True
        assert result['recovery']['superseded_training_attempt_id']==original['attempt_id']
        backup=real.run_root/result['recovery']['original_measurement_ref']
        assert json.loads(backup.read_text())==original


@pytest.mark.parametrize('fault', ['', 'candidate', 'source', 'same_proof', 'scored', 'evaluated', 'receipt', 'used'])
@pytest.mark.parametrize('prior_additive', [False, True])
def test_baseline_selection_retry_requires_new_additive_evidence_once(tmp_path, monkeypatch, fault, prior_additive):
    from autosim.research.common import atomic_json
    real=ambiguous_policy_research(tmp_path)
    record=real.run_stage('train',settings={})
    paths,_=real._selection_candidates(record)
    sources,_=real._policy_selection_sources('train')
    hashes={row['path']:row['sha256'] for row in sources}
    directory=real.run_root/'attempts'/record['attempt_id']
    selection_path=directory/'artifact_selection.json'
    prior={'status':'abstained','candidate_paths_sha256':object_digest(paths),
           'source_hashes':hashes, 'completed_work_proof_sha256':''}
    measurement={'where':'policy_artifact','status':'unscored','ok':False,'settings':{},
                 'recovery':{'training_attempt_id':record['attempt_id']},
                 'artifact_selection':{'selection_ref':str(selection_path.relative_to(real.output))}}
    if fault=='candidate':prior['candidate_paths_sha256']='changed'
    if fault=='source':prior['source_hashes']={**hashes,'missing.py':'changed'}
    if fault=='same_proof':prior['completed_work_proof_sha256']='new-proof'
    if fault=='scored':measurement['ok']=True
    if fault=='evaluated':measurement['evaluate']={'attempt_id':'executed'}
    atomic_json(selection_path,prior)
    recovery_path=directory/'baseline_recovery.json'
    if prior_additive:
        backup=directory/('baseline_unscored_additive_'+('a'*64)+'.json')
        atomic_json(backup,{'old_failure':True})
        measurement['recovery']['original_measurement_ref']=str(backup.relative_to(real.run_root))
        recovery_path=directory/('baseline_recovery_additive_'+('a'*64)+'.json')
    atomic_json(recovery_path,{'status':'failed',
        'training_attempt_id':record['attempt_id'],'measurement_sha256':object_digest(measurement)})
    monkeypatch.setattr(real,'_verified_training_receipt',lambda *args:None if fault=='receipt' else record)
    monkeypatch.setattr(real,'_policy_work_evidence',lambda *args:({'candidate_2':{}},[], 'new-proof'))
    before=selection_path.read_bytes()
    plan=real.baseline_selection_retry_plan(measurement)
    if fault=='used':
        assert plan['available']
        atomic_json(Path(plan['state_path']),{'status':'running'})
        plan=real.baseline_selection_retry_plan(measurement)
    assert plan['available']==(not fault)
    assert selection_path.read_bytes()==before
    if not fault:
        assert Path(plan['backup_path']).parent==directory
        assert not Path(plan['state_path']).exists()
        assert real.baseline_selection_retry_plan(measurement)==plan


def test_quiet_selection_adds_evidence_without_overwriting_abstention(tmp_path, monkeypatch):
    real=ambiguous_policy_research(tmp_path)
    record=real.run_stage('train',settings={});record['said']='quiet trainer'
    first=real._select_policy_artifact(dict(record))
    assert first['artifact_selection']['status']=='abstained'
    old_path=real.output/first['artifact_selection']['selection_ref'];old_bytes=old_path.read_bytes()
    monkeypatch.setattr(real,'_policy_work_evidence',lambda *args:({'candidate_2':{'completed_work':17}},[], 'proof'))
    original=real.client.chat_with_metadata
    def choose(*args,**kwargs):
        content,metadata=original(*args,**kwargs);answer=json.loads(content)
        answer['selection_basis']='verified_completed_work_container';answer['candidate_trace']=''
        return json.dumps(answer),metadata
    real.client.chat_with_metadata=choose
    selected=real._select_policy_artifact(dict(record),revalidate_abstained=True)
    assert selected['artifact_selection']['status']=='selected'
    assert selected['artifact_selection']['selection_ref']!=first['artifact_selection']['selection_ref']
    assert old_path.read_bytes()==old_bytes
    calls=real.client.calls
    again=real._select_policy_artifact(dict(record),revalidate_abstained=True)
    assert again['artifact_selection']['status']=='selected' and real.client.calls==calls


def test_comparison_protocol_rejects_eval_changes_before_training(tmp_path):
    real = research(tmp_path)
    baseline = real.measure(settings={"task": "Lift", "seed": 11,
                                      "episodes": 20, "train.n_epochs": 1},
                            label="baseline")
    assert baseline["ok"] is True
    protocol = json.loads((real.run_root / "comparison_protocol.json").read_text())
    assert protocol["settings"] == {"task": "Lift", "seed": 11, "episodes": 20}
    assert real.measure(settings={"task": "Lift", "seed": 11,
                                  "episodes": 20, "train.n_epochs": 2},
                        label="candidate")["ok"] is True
    before = len(list((real.run_root / "attempts").iterdir()))
    for key, value in (("task", "Can"), ("seed", 12), ("episodes", 2),
                       ("eval.n_eval", 2), ("horizon", 5)):
        settings = {"task": "Lift", "seed": 11, "episodes": 20,
                    "train.n_epochs": 3, key: value}
        refused = real.measure(settings=settings, label=f"changed_{key.replace('.', '_')}")
        assert refused["where"] == "comparison_protocol"
        assert refused["ran"] is False
    assert len(list((real.run_root / "attempts").iterdir())) == before


def test_declared_custom_protocol_key_is_immutable(tmp_path):
    real = research(tmp_path)
    real.declaration = {"research_goal": {"protocol_keys": ["rollout_length"]}}
    assert real.measure(settings={"rollout_length": 100}, label="baseline")["ok"]
    refused = real.measure(settings={"rollout_length": 10}, label="shortened")
    assert refused["where"] == "comparison_protocol"


def test_corrupted_frozen_comparison_protocol_fails_closed(tmp_path):
    real = research(tmp_path)
    assert real.measure(settings={}, label="baseline")["ok"]
    (real.run_root / "comparison_protocol.json").write_text("[]", encoding="utf-8")
    refused = real.measure(settings={}, label="candidate")
    assert refused["where"] == "comparison_protocol" and refused["ran"] is False


def test_derived_evaluator_parameter_change_is_protocol_change(tmp_path):
    real = research(tmp_path)
    real.backend.parameters["evaluate"] = {"task_id": {"value": "Lift"}}
    assert real.measure(settings={}, label="baseline")["ok"]
    real.backend.parameters["evaluate"]["task_id"]["value"] = "Can"
    refused = real.measure(settings={}, label="changed_task")
    assert refused["where"] == "comparison_protocol" and refused["ran"] is False


def test_evaluator_receives_archived_policy_not_mutable_training_path(tmp_path):
    real = research(tmp_path)
    real.backend.answer["stages"]["evaluate"]["artifact"] = ""
    real.backend.sources["evaluate"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', "
        "'import pathlib,sys; print(\"succ: 0.5\" if "
        "pathlib.Path(sys.argv[1]).read_text()==\"checkpoint\" else \"succ: 0.0\")', "
        "i['checkpoint']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                      sources=real.backend.sources, parameters={})
    measured = real.measure(settings={}, label="baseline")
    assert measured["ok"] is True and measured["metric_value"] == 0.5
    archived = measured["policy_artifact"]["path"]
    receipt = json.loads((real.run_root / "attempts" /
                          measured["evaluate"]["attempt_id"] / "receipt.json").read_text())
    assert archived in receipt["argv"]
    assert str(real.stage_directory("train") / "models" / "model.pth") not in receipt["argv"]


def test_policy_copy_limit_refuses_unrepeatable_score_before_evaluation(tmp_path, monkeypatch):
    real = research(tmp_path)
    monkeypatch.setenv("AUTOSIM_ARTIFACT_COPY_LIMIT_BYTES", "1")
    failed = real.measure(settings={}, label="baseline")
    assert failed["ok"] is False and failed["where"] == "policy_archive"
    assert "above copy limit" in failed["archive_error"]
    assert not (real.run_root / "evaluate" / "output.log").exists()


def test_research_can_measure_a_continuous_primary_metric(tmp_path):
    real = research(tmp_path)
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "mean_reward", "direction": "maximize", "unit": "return"}}})
    real.backend.sources["evaluate"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', 'import pathlib,sys; p=pathlib.Path(sys.argv[1])/\"models\"; p.mkdir(parents=True,exist_ok=True); (p/\"result.pth\").write_text(\"result\"); print(\"mean_reward: -3.5\")', i['output']]\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    result = real.measure(settings={}, label="reward_baseline")
    assert result["ok"] is True, result
    assert result["metric_value"] == -3.5
    assert result["success_rate"] is None
    assert real._was_measured(result) is True
    report = real.run(rounds=0)
    assert report["rounds"][0]["measured"] is True
    assert report["best"]["scale"] == "metric:mean_reward"


def test_native_json_result_can_supply_the_primary_metric(tmp_path):
    real = research(tmp_path)
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "return", "direction": "maximize", "source": "json",
        "json_key": "summary.reward", "unit": "reward"}}})
    real.backend.answer["stages"]["evaluate"]["artifact"] = "result.json"
    real.backend.sources["evaluate"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', 'import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True); (p/\"result.json\").write_text(json.dumps({\"summary\":{\"reward\":2.25}}))', i['output']]\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    result = real.measure(settings={}, label="json_reward")
    assert result["ok"] is True, result
    assert result["metric_value"] == 2.25
    assert "result.json#summary.reward" in result["metric_reading"]["read_from"]
    assert Path(result["result_artifact"]["path"]).is_file()
    assert Path(result["result_artifact"]["path"]).read_text() == (
        '{"summary": {"reward": 2.25}}')
    assert "success_reading" not in result


def test_json_episode_sidecar_flows_from_eval_attempt_into_verified_measurement(tmp_path):
    from autosim.research.receipt_verifier import verify_measurement

    real = research(tmp_path)
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "eval_success_once_mean", "direction": "maximize", "unit": "fraction",
        "source": "json", "artifact_root": "output",
        "artifact_pattern": "episodes.json", "json_key": "episodes",
        "json_value_key": "success", "aggregation": "mean", "min_samples": 2,
        "episode_id_column": "episode_id", "task_column": "task"}}})
    real.backend.answer["stages"]["evaluate"]["artifact"] = "models/*.pth"
    code = ("import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "(p/'models').mkdir(parents=True,exist_ok=True); "
            "(p/'models'/'evaluation.pth').write_text('native-eval-artifact'); "
            "(p/'episodes.json').write_text(json.dumps({'episodes':["
            "{'task':'lift','episode_id':0,'success':False},"
            "{'task':'lift','episode_id':1,'success':True}]}))")
    real.backend.sources["evaluate"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', " + repr(code) + ", i['output']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                      sources=real.backend.sources, parameters={})

    measured = real.measure(settings={"episodes": 200}, label="json_episodes")

    assert measured["ok"] is True, measured
    assert measured["metric_value"] == 0.5
    assert measured["metric_reading"]["episodes_completed"] == 2
    assert measured["metric_reading"]["episode_keys"] == ["lift\x1f0", "lift\x1f1"]
    assert measured["metric_artifact_evidence"]["status"] == "matched"
    assert measured["metric_artifact_evidence"]["sha256"] == \
        measured["result_artifact"]["content_sha256"]
    assert verify_measurement(real.run_root, "json_episodes")["status"] == "consistent"


def test_json_episode_sidecar_next_to_frozen_policy_is_receipt_bound(tmp_path):
    from autosim.research.receipt_verifier import verify_measurement

    real = research(tmp_path)
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "eval_success_once_mean", "direction": "maximize", "unit": "fraction",
        "source": "json", "artifact_root": "policy_parent",
        "artifact_pattern": "test_videos/trajectory.json", "json_key": "episodes",
        "json_value_key": "success", "aggregation": "mean", "min_samples": 2,
        "episode_id_column": "episode_id", "task_column": "env_info.env_id"}}})
    code = ("import json,pathlib,sys; root=pathlib.Path(sys.argv[1]).parent/'test_videos'; "
            "root.mkdir(parents=True,exist_ok=True); "
            "(root/'trajectory.json').write_text(json.dumps({'env_info':"
            "{'env_id':'PushCube-v1'},'episodes':["
            "{'episode_id':0,'episode_seed':12,'success':False},"
            "{'episode_id':1,'episode_seed':13,'success':True}]}))")
    real.backend.sources["evaluate"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', " + repr(code) + ", i['checkpoint']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                      sources=real.backend.sources, parameters={})

    measured = real.measure(settings={"task": "PushCube-v1"}, label="policy_sidecar")

    assert measured["ok"] is True, measured
    assert measured["metric_value"] == 0.5
    assert measured["metric_reading"]["episodes_completed"] == 2
    assert measured["metric_reading"]["episode_keys"] == [
        "PushCube-v1\x1f0", "PushCube-v1\x1f1"]
    assert measured["metric_artifact_evidence"]["status"] == "matched"
    assert verify_measurement(real.run_root, "policy_sidecar")["status"] == "consistent"


def test_native_episode_csv_counts_completed_rows_not_requested_rows(tmp_path):
    from autosim.research.receipt_verifier import verify_measurement

    real = research(tmp_path)
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "source": "csv",
        "aggregation": "mean", "episode_id_column": "episode_id",
        "task_column": "task"}}})
    real.backend.answer["stages"]["evaluate"]["artifact"] = "episodes.csv"
    code = ("import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True); (p/'episodes.csv').write_text("
            "'task,episode_id,success_rate\\nlift,0,1\\nlift,1,0\\n')")
    real.backend.sources["evaluate"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', " + repr(code) + ", i['output']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                      sources=real.backend.sources,
                                      parameters=real.backend.parameters)
    result = real.measure(settings={"task": "lift", "episodes": 100}, label="baseline")
    assert result["ok"] is True, result
    assert result["metric_value"] == 0.5
    assert result["metric_reading"]["episodes_completed"] == 2
    arm = selection.summary(real.run_root)["arms"][0]
    assert arm["episodes"] == 2 and arm["successes"] == 1
    assert verify_measurement(real.run_root, "baseline")["status"] == "consistent"


def test_a_measurement_that_needs_a_stage_that_cannot_run_is_not_started(tmp_path):
    """A benchmark whose evaluator is missing must not report a number nobody measured -- and
    must not spend the training half discovering that.

    It used to: the loop started `train`, waited for it, and then recorded that `evaluate` had
    not run. On RoboTwin that is six thousand epochs and hours of GPU, spent to produce a
    record saying no number came out. A stage that cannot run is a finding, and a finding does
    not need a day of compute to be true.
    """
    real = research(tmp_path, stages=("train",))
    measured = real.measure(settings={"train.n_epochs": 1}, label="try")
    assert measured["ok"] is False and measured["success_rate"] is None
    assert measured["ran"] is False
    assert "evaluate" in measured["why"]
    # Nothing was started, so nothing was written where a stage writes.
    assert not (real.run_root / "train").exists()
    # And the reason is in the event stream, where a reader of the run finds it.
    assert any(row.get("event") == "measurement_not_started" for row in real.events)


def test_the_loop_runs_a_baseline_then_a_round_and_stops_when_told(tmp_path):
    real = research(tmp_path)
    real.propose = lambda evidence, evidence_id, round_index=0: {"decision": "stop",
                                                  "hypothesis": "the lever is exhausted"}
    report = real.run(rounds=3)
    assert report["rounds"][0]["label"] == "baseline"
    assert report["rounds"][0]["success_rate"] == 0.5
    assert report["rounds"][-1]["status"] == "controller stopped"
    assert len(report["rounds"]) == 2
    shared = json.loads((real.output / "run_events.json").read_text(encoding="utf-8"))
    events = shared["rows"]
    assert any(row["event"] == "stage_started" and row["details"]["stage"] == "train"
               for row in events)
    assert any(row["event"] == "stage_completed" and row["details"]["stage"] == "evaluate"
               for row in events)
    state = json.loads((real.output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["current_action"] is None
    assert state["last_action"]["step"] == "research_complete"


def test_the_loop_stops_when_no_valid_proposal_comes_back(tmp_path):
    """A controller that will not propose against this declaration is a finding about the
    declaration, not a transient fault, so the loop stops and says which of the two it was."""
    real = research(tmp_path)
    real.propose = lambda evidence, evidence_id, round_index=0: None
    report = real.run(rounds=3)
    assert report["rounds"][-1]["status"] == "no valid proposal"
    assert len(report["rounds"]) == 2        # baseline, plus the round that could not run


def test_a_round_records_what_it_varied_and_what_came_of_it(tmp_path):
    real = research(tmp_path)
    seen = []

    def propose(evidence, evidence_id, round_index=0):
        seen.append(evidence)
        if len(seen) > 1:
            return {"decision": "stop", "hypothesis": "enough"}
        return {"decision": "experiment", "hypothesis": "more epochs help",
                "proposal_id": "p", "expected_validation": "succ rises",
                "evidence_id": evidence_id, "training": {"train.n_epochs": 4}}

    real.propose = propose
    report = real.run(rounds=3)
    measured = report["rounds"][1]
    assert measured["status"] == "measured"
    assert measured["varied"] == {"train.n_epochs": 4}
    assert measured["success_rate"] == 0.5
    # The second proposal is made against the first round's result, which is the whole point
    # of running a protocol rather than a sweep.
    assert seen[1]["baseline"] == 0.5
    assert seen[1]["round_history"][-1]["round"] == 1


def test_the_report_says_which_stages_the_benchmark_had(tmp_path):
    real = research(tmp_path, stages=("train",))
    real.propose = lambda evidence, evidence_id, round_index=0: None
    assert real.run(rounds=1)["available_stages"] == ["train"]


def test_the_number_is_read_from_the_last_labelled_thing_printed(tmp_path):
    """The runner does not re-implement scoring; it reads the benchmark's own number.

    The shapes are LIBERO's, taken from its source rather than imagined: the per-epoch line
    carries three success-flavoured numbers of which only one is the measurement, and the
    run ends with a per-task table.
    """
    read = DerivedResearch._success_rate
    assert read("succ 0.62") == 0.62
    assert read("success rate 0.75") == 0.75
    assert read("[Task  0 succ.]  0.44 |") == 0.44
    assert read("[All task succ.]  0.10 | 0.20 | 0.30 |") == pytest.approx(0.2)
    # `best succ` is the best epoch so far and `AoC` is counterbalanced over all tasks; a
    # reader that averages either into the measurement reports a different experiment.
    epoch = "[info] Epoch:   1 | succ: 0.40 ± 0.00 | best succ: 0.90 | succ. AoC 0.10"
    assert read(epoch) == 0.40
    assert read("last succ: 0.30") == 0.30
    assert read("no number here") is None
    # The evaluator's final line, which is what `libero.lifelong.evaluate` ends with.
    assert read("Results are saved at /tmp/x\n0.2043 0.45") is None


def test_a_benchmark_that_prints_nothing_readable_reports_that(tmp_path):
    assert json.dumps({"success_rate": DerivedResearch._success_rate("nothing")}) == \
        '{"success_rate": null}'


def test_the_output_path_handed_to_a_stage_is_one_the_benchmark_can_express(tmp_path):
    """The system chooses where a stage writes, so it chooses a path that can be written.

    A checkout under a directory whose name is not ASCII -- which is where LIBERO lives --
    yields an output path hydra's override lexer refuses at the equals sign. The reviser read
    that traceback correctly and still could not act, because the offending value was the
    harness's own. The stage now gets plain characters and the visible location is a symlink.
    """
    awkward = tmp_path / "下载" / "run"
    real = research(awkward)
    told = real._inputs("train", settings={})["output"]
    assert all(ord(character) < 128 for character in told), told
    assert "下载" not in told
    # ... and the place a person would look still resolves to it.
    logical = real.run_root / "train"
    assert logical.exists()
    assert logical.resolve() == Path(told).resolve()


def test_an_ascii_output_path_is_left_alone(tmp_path):
    """No indirection where none is needed: a symlink is a cost, not a feature."""
    real = research(tmp_path / "plain")
    told = real._inputs("train", settings={})["output"]
    assert told == str(real.run_root / "train")
    assert not (real.run_root / "train").is_symlink()


def test_an_artifact_written_beside_the_command_is_found_and_only_if_new(tmp_path):
    """A benchmark that builds its own output path leaves the caller nothing to set.

    LIBERO's training tree is `./experiments/{benchmark}/{algo}/{policy}_seed{seed}/run_N`,
    computed from config values and the working directory, with no key for the caller to
    override. The artifact the stage promised therefore appears beside the command rather
    than under the output directory -- and the checkout already holds every earlier run's
    copy, so the search is bounded by modification time.
    """
    import os
    import time as _time
    real = research(tmp_path)
    old = tmp_path / "experiments" / "LIBERO_10" / "base" / "run_1" / "models"
    old.mkdir(parents=True)
    stale = old / "task0_model.pth"
    stale.write_bytes(b"yesterday")
    os.utime(stale, (0, 0))                      # a run that finished long ago

    backend = real.backend
    beside = backend.artifact_beside("train", real.backend.directory("train"),
                                     since=_time.time())
    assert beside["matched"] == 0, "an earlier run's checkpoint satisfied the check"

    fresh = old / "task1_model.pth"
    fresh.write_bytes(b"today")
    beside = backend.artifact_beside("train", real.backend.directory("train"),
                                     since=_time.time() - 5)
    assert beside["matched"] == 1
    assert beside["examples"] == ["experiments/LIBERO_10/base/run_1/models/task1_model.pth"]
    assert beside["found_beside_the_command"] is True


def test_a_loop_with_nothing_to_measure_stops_and_says_so(tmp_path):
    """Research starts from a measurement, and the report may not claim one that did not happen.

    LIBERO's report records a round with `"status": "measured"` in which no stage ran at all,
    because a stage with no return code passed the `returncode in (0, None)` check and the
    round was written down as though a number had come out. A baseline that cannot be taken
    is the finding; a proposal against it cannot be evaluated and should not be made.
    """
    real = research(tmp_path, stages=("evaluate",))     # no train: nothing can be measured
    asked = []
    real.propose = lambda evidence, evidence_id, round_index=0: asked.append(evidence) or {
        "decision": "experiment", "hypothesis": "h", "proposal_id": "p",
        "expected_validation": "v", "evidence_id": evidence_id, "training": {}}
    report = real.run(rounds=3)
    assert not asked, "the controller was spent on a round that could not be evaluated"
    assert "no baseline could be measured" in report["stopped_because"]
    baseline = report["rounds"][0]
    assert baseline["label"] == "baseline" and baseline["measured"] is False
    assert "train" in baseline["why_not"]


def test_detached_baseline_receipt_is_reused_without_retraining(tmp_path):
    real = research(tmp_path)
    attempt = "a" * 32
    seen = []
    real._verified_training_receipt = lambda ident, settings: (
        {"attempt_id": ident} if ident == attempt and settings == {} else None)
    real.baseline = lambda *, settings, training_attempt_id: (
        seen.append(training_attempt_id) or
        {"ok": False, "where": "evaluate", "why": "synthetic score failure",
         "settings": settings, "metric_value": None})
    real.run(rounds=0, baseline_training_attempt_id=attempt)
    assert seen == [attempt]


def test_a_completed_training_failure_keeps_its_actual_reason_and_evidence(tmp_path):
    real = research(tmp_path)
    failed = {
        "ok": False, "where": "train", "ran": True, "status": "completed",
        "returncode": 0, "termination_reason": "normal_exit", "attempt_id": "attempt-123",
        "why": "no unique, existing policy artifact belongs to this training attempt",
        "artifact": {"checked": True, "matched": 6, "pattern": "runs/*/ckpt_*.pt",
                     "examples": ["runs/0/ckpt_100.pt", "runs/0/ckpt_200.pt"]},
        "said": "global_step=1000",
    }
    real.baseline = lambda settings: failed

    report = real.run(rounds=0)

    baseline = report["rounds"][0]
    assert baseline["why_not"] == (
        "train: no unique, existing policy artifact belongs to this training attempt")
    assert baseline["failure"]["ran"] is True
    assert baseline["failure"]["returncode"] == 0
    assert baseline["failure"]["attempt_id"] == "attempt-123"
    assert baseline["failure"]["artifact"]["matched"] == 6
    assert baseline["failure"]["evidence"] == {
        "receipt": "attempts/attempt-123/receipt.json",
        "log": "train/attempts/attempt-123/output.log"}
    assert baseline["said"] == "global_step=1000"
    assert "did not run a command" not in baseline["why_not"]
    events = json.loads((real.run_root / "events.json").read_text())
    no_baseline = next(row for row in events["rows"] if row["event"] == "no_baseline")
    assert no_baseline["rows"][0]["why_not"] == baseline["why_not"]


def test_a_measured_round_is_recorded_as_measured(tmp_path):
    real = research(tmp_path)
    real.propose = lambda evidence, evidence_id, round_index=0: {
        "decision": "experiment", "hypothesis": "h", "proposal_id": "p",
        "expected_validation": "v", "evidence_id": evidence_id,
        "training": {"train.n_epochs": 2}}
    report = real.run(rounds=1)
    assert report["rounds"][0]["measured"] is True
    assert report["rounds"][1]["status"] == "measured"
    assert report["rounds"][1]["success_rate"] == 0.5


def test_the_runner_does_not_guess_where_the_data_is(tmp_path):
    """`dataset` used to be the checkout path, and the caller's inputs win over the
    derivation's settled values -- so a `folder` the system had discovered and recorded was
    silently replaced by a path with no data in it. LIBERO lost several rounds to that: the
    command looked right, named the right key, and pointed at the wrong directory."""
    real = research(tmp_path)
    assert real._inputs("train", settings={})["dataset"] == ""
    # And with nothing from the caller, a value the derivation settled is what the command
    # carries -- which is the whole point of not guessing.
    real.backend.parameters = {"train": {"folder": {"value": "/data/libero"}}}
    real.backend._functions["train"] = lambda i: [i["python"], "folder=" + str(i["folder"])]
    assert real.backend.argv("train", real._inputs("train", settings={}))[1] == \
        "folder=/data/libero"


def test_the_baseline_does_not_override_the_benchmarks_own_device(tmp_path):
    """A default that overrides the benchmark's is a different experiment.

    `device` was pinned to `"cpu"`. LIBERO's own config says `cuda`, and the baseline ran for
    over an hour on the CPU without finishing a run that takes about ninety seconds on the
    card -- which was the entire wall clock of the attempt, not a rounding error.
    """
    real = research(tmp_path)
    base = real._base_settings(None)
    assert "device" not in base, base
    assert real._inputs("train", settings=base)["device"] == ""
    # And the caller can still say, when it has a reason to.
    assert real._base_settings({"device": "cpu"})["device"] == "cpu"


def test_an_artifact_pattern_that_resolves_to_an_absolute_path_is_searched(tmp_path):
    """`Path.glob` refuses an absolute pattern, and the pattern can become one.

    LIBERO's tree is `./experiments/{benchmark}/...` and its placeholder was filled with an
    absolute path. The run that produced a real FWT of 0.488 finished, wrote `result.pt`, and
    then raised `NotImplementedError: Non-relative patterns are unsupported` -- the number
    existed and the system kept no record of it.
    """
    real = research(tmp_path)
    written = tmp_path / "experiments" / "LIBERO_10" / "base" / "run_3" / "models"
    written.mkdir(parents=True)
    (written / "task0_model.pth").write_bytes(b"x")
    real.backend.stages["train"] = {**real.backend.stages["train"],
                                    "artifact": str(written) + "/*.pth"}
    found = real._artifact_path("train", settings={})
    assert found is not None and found.name == "task0_model.pth"


def test_malformed_artifact_glob_is_a_missing_artifact_not_a_crash(tmp_path):
    real = research(tmp_path)
    real.backend.stages["train"] = {**real.backend.stages["train"],
                                    "artifact": "runs/**final_ckpt.pt"}
    assert real._artifact_path("train", settings={}) is None


def test_a_stage_that_ran_out_of_memory_is_tried_somewhere_else(tmp_path):
    """The failure knows which card, and nothing else does.

    A card that could not hold the run is not a card to choose again, so the device it failed
    on is excluded and the decision is remade. Returns None when the failure says nothing
    about the device -- which is most failures -- so a caller cannot loop on it.
    """
    from autosim.research.compute_decision import ComputeDecision, reconsider
    on_card = ComputeDecision(device="cuda:0", device_index=0,
                              why="chosen", evidence={"choices": []})
    assert reconsider(on_card, "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert reconsider(on_card, "AssertionError: load_task should be in [0, ..., 9]") is None
    # Already off the card: there is nothing further the failure implies.
    on_cpu = ComputeDecision(device="cpu", device_index=0, why="chosen")
    assert reconsider(on_cpu, "CUDA out of memory") is None
    assert reconsider(on_cpu, "no CUDA-capable device is detected") is None


def test_the_device_the_stage_declares_wins_over_the_machines_choice(tmp_path):
    """A benchmark that names a variable for its own reasons knows something the table does
    not -- and a caller that names a device has a reason too."""
    real = research(tmp_path)
    assert real.compute.device == "cpu"          # what the test supplied
    assert real._inputs("train", settings={}, device=real.compute.device,
                        device_index=real.compute.device_index)["device"] == "cpu"


def test_the_measurement_carries_a_signal_to_rank_on_cheaply(tmp_path):
    """A method that screens at low budget cannot be used against one number known at the end.

    LIBERO prints its loss every epoch -- 5.36 at epoch 0, -18.6 by epoch 50 -- so a
    controller told to rank candidates cheaply has something to rank on, and the system was
    keeping only the last success rate it recognised.
    """
    read = DerivedResearch._readings
    line = "[info] Epoch:   0 | train loss:  5.36 | time: 0.99"
    assert read(line)["loss"] == 5.36 and read(line)["epoch"] == 0.0
    # `best succ` is the best epoch so far, not this one: reading them as one quantity gives
    # the wrong number under the right name.
    both = "[info] Epoch:   1 | succ: 0.62 ± 0.00 | best succ: 0.90 | time: 1.2"
    assert read(both)["succ"] == 0.62
    assert read("nothing labelled here") == {}


def test_a_declared_axis_the_program_contradicts_is_reported(tmp_path):
    """The declaration is verified for the existence of its axes, not for their values.

    LIBERO's space offers `policy.policy_type ∈ (bc_rnn_policy, bc_transformer_policy,
    bc_vilt_policy)` -- the yaml file names, which is what a reader of the repository sees --
    while the field holds a *class* name and the registry is keyed by that. Setting it to any
    declared value raises `Policy class with name bc_transformer_policy not found in registry`:
    every value of that axis produces a command that cannot run, and nothing checked.

    A program that composes a configuration prints it. That print is the authority.
    """
    from autosim.research.adapter_protocol import Axis, OptimizationSpace
    from autosim.research.declaration import values_the_program_contradicts

    space = OptimizationSpace(training=(
        Axis("policy.policy_type", "choice", "the policy", group="",
             values=("bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy")),
        Axis("lifelong.algo", "choice", "the algorithm", group="",
             values=("Sequential", "ER", "EWC", "PackNet", "Multitask")),
        Axis("train.n_epochs", "integer", "epochs", low=1, high=100, default=50)))

    # The shape LIBERO actually prints: a nested dict, one field per line.
    # The shape LIBERO actually prints -- pprint wraps, so a nested field lands on its own
    # line. Written on one line it is not recognised, which is what the first version of this
    # test asserted and why it failed against correct code.
    # Verbatim from what LIBERO printed, indentation and all: the config is a deeply nested
    # dict and pprint puts the leaf on its own line, which is what the reader sees.
    printed = "\n".join([
        "  'lifelong': {'algo': 'EWC'},",
        "  'policy': { 'color_aug': {'augs': ['IdentityAug']},",
        "              'num_modes': 5}},",
        "              'policy_type': 'BCTransformerPolicy',",
        "              'temporal_position_encoding': { 'network': 'Sinusoidal'},",
        "  'n_epochs': 50,"])
    found = values_the_program_contradicts(space, printed)
    assert [row["axis"] for row in found] == ["policy.policy_type"], found
    assert found[0]["the_program_says"] == "BCTransformerPolicy"
    assert "bc_transformer_policy" in found[0]["declared_values"]

    # A value the axis does offer is not a disagreement, and an integer axis is not a choice.
    assert values_the_program_contradicts(space, "  'algo': 'ER',") == []
    assert values_the_program_contradicts(space, "  'n_epochs': 50,") == []

    # Output with no configuration in it yields nothing rather than guessing.
    assert values_the_program_contradicts(space, "just some output") == []


def test_a_run_with_no_declared_space_does_not_crash_on_the_check(tmp_path):
    """`space` may be None -- a controlled experiment driving the verified commands directly
    has no declared space, and nothing about that is a defect. Raising on it took down a
    measurement that had already screened two candidates successfully."""
    from autosim.research.declaration import values_the_program_contradicts
    assert values_the_program_contradicts(None, "  'folder': None,") == []
    real = research(tmp_path)
    real.space = None
    record = real.run_stage("train", settings={})
    assert record.get("ran") is True, record


def test_a_stage_that_times_out_keeps_what_it_produced(tmp_path):
    """A run that trained nine tasks and then hung has nine tasks' worth of evidence.

    Four runs printed losses and success rates for hours and then timed out, and the readings
    came back `{}` -- because the timeout path replaced the child's output with a note that it
    had stopped. The output is the record; a timeout is not a reason to discard it.
    """
    real = research(tmp_path)
    # A command that reports, then hangs: the shape of every run that timed out.
    real.backend.sources["train"] = (
        "def stage_argv_train(i):\n"
        "    return [i['python'], '-c', \"import sys, time; "
        "print('[info] Epoch: 3 | train loss: 1.25 | succ: 0.5'); "
        "sys.stdout.flush(); time.sleep(30)\"]\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    record = real.run_stage("train", settings={}, timeout=2)
    assert record["returncode"] is None and "did not finish" in record["said"]
    assert "train loss: 1.25" in record["said"], record["said"]
    assert real._readings(record["said"])["loss"] == 1.25


def test_timeout_and_old_artifact_cannot_create_a_measurement(tmp_path):
    real = research(tmp_path)
    old = real.stage_directory("train") / "models" / "model.pth"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("old checkpoint", encoding="utf-8")
    real.backend.sources["train"] = (
        "def stage_argv_train(i):\n"
        "    return [i['python'], '-c', 'print(\"succ: 0.99\")']\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    result = real.measure(settings={}, label="old_artifact")
    assert result["ok"] is False
    assert result["success_rate"] is None
    assert result["where"] == "train"
    assert result["artifact"]["matched"] == 0


def test_fresh_checkpoint_with_explicit_zero_training_iterations_is_not_a_measurement(tmp_path):
    real = research(tmp_path)
    real.backend.sources["train"] = (
        "def stage_argv_train(i):\n"
        "    return [i['python'], '-c', 'import pathlib,sys; "
        "p=pathlib.Path(sys.argv[1])/\"models\"; p.mkdir(parents=True,exist_ok=True); "
        "(p/\"model.pth\").write_text(\"untrained\"); "
        "print(\"args.num_iterations=0\")', i['output']]\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    result = real.measure(settings={"steps": 8192}, label="zero_updates")
    assert result["ok"] is False and result["where"] == "train"
    assert result["zero_work_reported"] == {"num_iterations": 0.0}
    assert not (real.run_root / "evaluate" / "output.log").exists()


def test_production_measurement_requires_positive_native_training_progress(tmp_path):
    real = research(tmp_path)
    real.require_training_progress = True
    failed = real.measure(settings={}, label="initialization_only")
    assert failed["ok"] is False and failed["where"] == "train"
    assert failed["training_progress"]["status"] == "unknown"
    assert not (real.run_root / "evaluate" / "output.log").exists()

    real.backend.sources["train"] = ARGV.format(stage="train").replace(
        'print("succ: 0.50")', 'print("global_step=1"); print("succ: 0.50")')
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    passed = real.measure(settings={}, label="updated")
    assert passed["ok"] is True
    assert passed["metric_value"] == 0.5


def test_log_metric_survives_an_unmatched_optional_video_glob(tmp_path):
    from autosim.research.receipt_verifier import verify_measurement

    real = research(tmp_path)
    real.backend.stages["evaluate"]["artifact"] = "videos/*.mp4"
    result = real.measure(settings={}, label="video_is_optional")
    assert result["ok"] is True and result["metric_value"] == 0.5
    assert verify_measurement(real.run_root, "video_is_optional")["status"] == "consistent"
    receipt = json.loads((real.run_root / "attempts" / result["evaluate"]["attempt_id"] /
                          "receipt.json").read_text(encoding="utf-8"))
    assert receipt["postconditions"][0]["required_for_score"] is False


def test_native_video_beside_frozen_policy_is_bound_to_scored_attempt(tmp_path):
    real = research(tmp_path)
    real.backend.stages["evaluate"]["artifact"] = "videos/*.mp4"
    real.backend.sources["evaluate"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', 'import pathlib,sys; "
        "p=pathlib.Path(sys.argv[1]).parent/\"test_videos\"; "
        "p.mkdir(parents=True,exist_ok=True); "
        "(p/\"0.mp4\").write_bytes(b\"real video bytes\"); "
        "print(\"succ: 0.50\")', i['checkpoint']]\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    result = real.measure(settings={}, label="baseline")
    assert result["ok"] is True
    receipt = json.loads((real.run_root / "attempts" / result["evaluate"]["attempt_id"] /
                          "receipt.json").read_text(encoding="utf-8"))
    assert len(receipt["media"]) == 1
    assert "test_videos/0.mp4" in receipt["media"][0]["path"]
    manifest = json.loads((real.run_root / "media" / "manifest.json").read_text())
    recording = next(row for row in manifest["recordings"]
                     if "test_videos/0.mp4" in row["path"])
    assert recording["attempt_id"] == result["evaluate"]["attempt_id"]
    assert recording["policy_sha256"] == result["policy_artifact"]["sha256"]
    assert recording["same_scored_evaluation"] is True


def test_ambiguous_checkpoint_glob_cannot_select_an_arbitrary_policy(tmp_path):
    real = research(tmp_path)
    real.backend.sources["train"] = (
        "def stage_argv_train(i):\n"
        "    return [i['python'], '-c', 'import pathlib,sys; p=pathlib.Path(sys.argv[1])/\"models\"; p.mkdir(parents=True,exist_ok=True); (p/\"one.pth\").write_text(\"one\"); (p/\"two.pth\").write_text(\"two\")', i['output']]\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    result = real.measure(settings={}, label="ambiguous")
    assert result["ok"] is False
    assert result["where"] == "policy_artifact"
    assert result["status"] == "unscored"
    assert result["training_outcome"] == {"status": "completed", "returncode": 0,
                                          "attempt_id": result["attempt_id"]}
    assert result["artifact"]["matched"] == 2
    assert not (real.run_root / "evaluate" / "output.log").exists()


def test_native_evidence_selects_and_freezes_the_policy_consumed_by_evaluator(tmp_path):
    real = ambiguous_policy_research(tmp_path)

    result = real.measure(settings={}, label="native-selection")

    assert result["ok"] is True, result
    assert result["metric_value"] == 0.75
    assert result["train"]["artifact"]["matched"] == 2
    assert result["train"]["artifact_selection"]["status"] == "selected"
    assert result["train"]["artifact_selection"]["selected_path"] == "models/two.pth"
    assert result["train"]["artifact"]["candidate_paths"] == [
        "models/one.pth", "models/two.pth"]
    assert Path(result["evaluated_policy_path"]).read_text(encoding="utf-8") == "two"

    train_id = result["train"]["artifact_selection"]["selection_ref"]
    selection_record = json.loads((real.output / train_id).read_text(encoding="utf-8"))
    assert selection_record["status"] == "selected"
    assert selection_record["candidate_paths_sha256"]
    assert selection_record["source_hashes"]["ppo.py"]
    assert selection_record["evidence"][0]["quote"] == (
        'NATIVE_EVAL_POLICY = "candidate_2"')
    # Produced checkpoint names and the artifact glob stay local; the model reasons over
    # opaque IDs and source/log excerpts with candidate references aliased.
    assert real.client.payload["candidate_ids"] == ["candidate_1", "candidate_2"]
    request_text = json.dumps(real.client.payload)
    assert "models/two.pth" not in request_text
    assert "models/one.pth" not in request_text
    assert "artifact_pattern" not in real.client.payload
    assert "invocation" not in real.client.payload
    assert "candidate_2" in real.client.payload["training_log_tail"]

    eval_receipt = json.loads((real.run_root / "attempts" /
                               result["evaluate"]["attempt_id"] /
                               "receipt.json").read_text(encoding="utf-8"))
    assert result["evaluated_policy_path"] in eval_receipt["argv"]
    training_attempt = Path(train_id).parent.name
    selection_exchange = real.run_root / "exchanges" / \
        f"artifact_selection_{training_attempt}.json"
    assert selection_exchange.is_file()


def test_benchmark_rubric_prompt_projects_measurement_artifacts_to_presence_only(tmp_path):
    class CaptureClient:
        def __init__(self):
            self.payload = None

        def chat_with_metadata(self, system, user, **kwargs):
            self.payload = json.loads(user)
            return json.dumps({"answers": [{"name": "native", "passed": True,
                                             "because": "recorded"}]}), {}

    real = research(tmp_path)
    real.client = CaptureClient()
    real.rubric = objective.Rubric([
        objective.Check(name="stages", question="Were native stages used?", weight=1.0,
                        children=[objective.Check(name="native", question="Native?",
                                                  weight=1.0)])])
    real._answer_the_run_s_own_questions(
        rounds=[], current={
            "ok": True, "metric_value": 0.5,
            "readings": {"episode_9834_private": 0.5},
            "artifact": {"pattern": "private-policy.pth", "path": "/tmp/private-policy.pth",
                         "candidate_paths": ["/tmp/private-policy.pth"]},
            "policy_artifact": {"path": "/tmp/frozen-policy.pth",
                                 "sha256": "private-policy-digest"},
            "result_artifact": {"path": "/tmp/result.json", "sha256": "private-result-digest"},
            "train": {"artifact": {"selected_path": "private-policy.pth",
                                    "candidate_paths": ["private-policy.pth"]}},
        })

    request = json.dumps(real.client.payload)
    assert "private-policy.pth" not in request
    assert "frozen-policy.pth" not in request
    assert "result.json" not in request
    assert "private-policy-digest" not in request
    assert "episode_9834_private" not in request
    assert (real.client.payload["what_the_run_has_recorded"]["the_last_measurement"]
            ["artifact_available"] is True)
    assert "readings" not in real.client.payload["what_the_run_has_recorded"][
        "the_last_measurement"]
    assert (real.client.payload["what_the_run_has_recorded"]["the_last_measurement"]
            ["metric_value"] == 0.5)


@pytest.mark.parametrize("mode,expected", [
    ("invalid_quote", "source citation is not an exact quote"),
    ("invalid_candidate", "exact opaque candidate ID"),
])
def test_invalid_native_selection_evidence_fails_closed(tmp_path, mode, expected):
    real = ambiguous_policy_research(tmp_path, selector_mode=mode)

    result = real.measure(settings={}, label=f"invalid-{mode}")

    assert result["ok"] is False
    assert result["where"] == "policy_artifact"
    assert result["training_outcome"]["status"] == "completed"
    assert result["artifact_selection"]["status"] == "abstained"
    assert expected in result["artifact_selection"]["why"]
    assert not (real.run_root / "evaluate" / "output.log").exists()


def test_source_quote_accepts_only_common_block_indentation_normalization():
    source = (
        "def save_policy(args):\n"
        "    if args.save_model:\n"
        "        model_path = f'checkpoints/{args.name}.pt'\n"
        "        torch.save(agent.state_dict(), model_path)\n")
    dedented = (
        "if args.save_model:\n"
        "    model_path = f'checkpoints/{args.name}.pt'\n"
        "    torch.save(agent.state_dict(), model_path)")

    assert _source_quote_present(source, dedented)
    assert not _source_quote_present(source, dedented.replace("state_dict", "weights"))
    assert not _source_quote_present(source, "if args.save_model:\\n    torch.save(agent, p)")


def test_legacy_ambiguous_receipt_candidates_are_reconstructed_only_when_consistent(
        tmp_path):
    real = ambiguous_policy_research(tmp_path)
    record = real.run_stage("train", settings={})
    later_directory = real.stage_directory("train") / "other-run"
    later_directory.mkdir()
    (later_directory / "late.pth").write_text("from another attempt", encoding="utf-8")
    legacy = {**record, "artifact": {
        key: value for key, value in record["artifact"].items()
        if key not in {"candidate_paths", "candidate_paths_truncated"}}}

    candidates, error = real._selection_candidates(legacy)

    assert error == ""
    assert candidates == ["models/one.pth", "models/two.pth"]

    extra = Path(record["output_directory"]) / "models" / "late.pth"
    extra.write_text("not from this attempt", encoding="utf-8")
    legacy_changed = {**record, "artifact": {
        key: value for key, value in record["artifact"].items()
        if key not in {"candidate_paths", "candidate_paths_truncated"}}}
    changed, error = real._selection_candidates(legacy_changed)
    assert changed == []
    assert "no longer match" in error


def test_recovered_legacy_selection_appends_a_new_record_instead_of_overwriting(
        tmp_path):
    real = ambiguous_policy_research(tmp_path)
    record = real.run_stage("train", settings={})
    late = Path(record["output_directory"]) / "models" / "late.pth"
    late.write_text("later file", encoding="utf-8")
    record["artifact"].pop("candidate_paths", None)
    record["artifact"].pop("candidate_paths_truncated", None)

    first = real._select_policy_artifact(record)
    first_ref = (first.get("artifact_selection") or {}).get("selection_ref")
    assert first["artifact_selection"]["status"] == "unavailable"
    first_path = real.output / first_ref
    first_bytes = first_path.read_bytes()

    late.unlink()
    recovered = real._select_policy_artifact(record)

    assert recovered["artifact_selection"]["status"] == "selected"
    second_ref = recovered["artifact_selection"]["selection_ref"]
    assert second_ref != first_ref
    assert first_path.read_bytes() == first_bytes
    assert json.loads(first_bytes)["status"] == "unavailable"
    assert json.loads((real.output / second_ref).read_text())["status"] == "selected"


def test_abstained_artifact_can_be_revalidated_from_original_response_without_requery(
        tmp_path):
    real = ambiguous_policy_research(tmp_path, selector_mode="invalid_quote")
    (tmp_path / "ppo.py").write_text(
        'def policy_path():\n    NATIVE_EVAL_POLICY = "models/two.pth"\n',
        encoding="utf-8")
    record = real.run_stage("train", settings={})
    first = real._select_policy_artifact(record)
    first_ref = first["artifact_selection"]["selection_ref"]
    first_path = real.output / first_ref
    assert first["artifact_selection"]["status"] == "abstained"

    # Simulate a response saved under an older validator that rejected a code excerpt only
    # because it omitted the function's common indentation. The source/candidates stay fixed.
    exchange_path = real.run_root / "exchanges" / \
        f"artifact_selection_{record['attempt_id']}.json"
    exchange = json.loads(exchange_path.read_text(encoding="utf-8"))
    answer = json.loads(exchange["response"])
    answer["evidence"][0]["quote"] = 'NATIVE_EVAL_POLICY = "models/two.pth"'
    content = json.dumps(answer)
    exchange["response"] = content
    exchange["response_sha256"] = object_digest(content)
    exchange_path.write_text(json.dumps(exchange), encoding="utf-8")
    persisted = json.loads(first_path.read_text(encoding="utf-8"))
    persisted["answer"] = answer
    persisted["response_sha256"] = object_digest(content)
    first_path.write_text(json.dumps(persisted), encoding="utf-8")
    historical_bytes = first_path.read_bytes()
    calls_before = real.client.calls

    recovered = real._select_policy_artifact(record, revalidate_abstained=True)

    assert recovered["artifact_selection"]["status"] == "selected"
    second_ref = recovered["artifact_selection"]["selection_ref"]
    assert second_ref != first_ref
    assert first_path.read_bytes() == historical_bytes
    assert real.client.calls == calls_before
    second = json.loads((real.output / second_ref).read_text(encoding="utf-8"))
    assert second["supersedes_selection_ref"] == first_ref
    assert second["revalidation"]["current_inputs_unchanged"] is True


def test_unscored_baseline_recovery_reuses_train_receipt_and_only_runs_evaluator(tmp_path):
    real = ambiguous_policy_research(tmp_path, selector_mode="invalid_quote")
    (tmp_path / "ppo.py").write_text(
        'def policy_path():\n    NATIVE_EVAL_POLICY = "models/two.pth"\n',
        encoding="utf-8")
    failed = real.measure(settings={}, label="baseline")
    assert failed["where"] == "policy_artifact"
    train_id = failed["training_outcome"]["attempt_id"]

    selection_ref = failed["artifact_selection"]["selection_ref"]
    selection_path = real.output / selection_ref
    selection_doc = json.loads(selection_path.read_text(encoding="utf-8"))
    exchange_path = real.run_root / "exchanges" / f"artifact_selection_{train_id}.json"
    exchange = json.loads(exchange_path.read_text(encoding="utf-8"))
    answer = json.loads(exchange["response"])
    answer["evidence"][0]["quote"] = 'NATIVE_EVAL_POLICY = "models/two.pth"'
    content = json.dumps(answer)
    exchange.update(response=content, response_sha256=object_digest(content))
    exchange_path.write_text(json.dumps(exchange), encoding="utf-8")
    selection_doc.update(answer=answer, response_sha256=object_digest(content))
    selection_path.write_text(json.dumps(selection_doc), encoding="utf-8")

    original_bytes = (real.run_root / "measurements" / "baseline.json").read_bytes()
    failure_row = {"round": 0, "label": "baseline", "metric_value": None,
                   "metric_utility": None, "metric_name": "success_rate",
                   "settings": {}, "measured": False,
                   "why_not": failed["why"],
                   "failure": real._failure_context(failed)["failure"]}
    real.run_root.mkdir(parents=True, exist_ok=True)
    real._advance_best("baseline", score=None, scale="success_rate",
                       why="original baseline")
    real._advance_best("baseline-progress", score=0.2, scale="progress",
                       why="reached the measurement stage")
    real.rubric = objective.Rubric([objective.Check(
        name="measurement", question="Did a measured value come out?", weight=1.0)])
    real.rubric_built = True
    session = {"schema_version": 1, "run_id": real.run_id,
               "repository": str(real.repo), "status": "paused", "rounds": 1,
               "next_round": 1, "current_measurement_label": "baseline",
               "history": [failure_row], "baseline_rate": None}
    report = {"schema_version": 1, "run_id": real.run_id,
              "repo": str(real.repo), "run_status": "paused", "rounds": [failure_row],
              "best": real.snapshots.best().as_dict(),
              "objective": real.rubric.as_dict(), "next_round": 1}
    (real.run_root / "controller_session.json").write_text(
        json.dumps(session), encoding="utf-8")
    (real.run_root / "research_report.json").write_text(
        json.dumps(report), encoding="utf-8")

    calls_before = real.client.calls
    recovered = real.recover_unscored_baseline()

    assert recovered["ok"] is True, recovered
    assert recovered["metric_value"] == 0.75
    assert recovered["recovery"]["training_reused"] is True
    assert recovered["train"]["attempt_id"] == train_id
    assert recovered["train"]["reused"] is True
    assert real.client.calls == calls_before
    receipts = {path.parent.name for path in
                (real.run_root / "attempts").glob("*/receipt.json")}
    assert receipts == {train_id, recovered["evaluate"]["attempt_id"]}
    assert (real.run_root / "attempts" / train_id / "baseline_unscored.json").is_file()
    assert json.loads((real.run_root / "attempts" / train_id /
                       "baseline_unscored.json").read_bytes())
    assert (real.run_root / "measurements" / "baseline.json").read_bytes() != original_bytes
    assert json.loads((real.run_root / "controller_session.json").read_text())[
        "history"][0]["measured"] is True
    assert json.loads((real.run_root / "research_report.json").read_text())[
        "rounds"][0]["recovered_from"]["training_reused"] is True


def test_resource_blocked_baseline_recovery_reuses_exact_train_policy(tmp_path, monkeypatch):
    from autosim.research.experiment_bundle import artifact_identity

    real = research(tmp_path, run_id="resource-recovery")
    settings = {"task": "PushCube-v1"}
    real.require_training_progress = True
    real.compute = ComputeDecision(
        device="cuda:0", device_index=0,
        evidence={"selected_device": {"index": 0, "uuid": "GPU-test-5090"}})
    protocol = real.run_root / "comparison_protocol.json"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(json.dumps({"target": "evaluate", "settings": settings}),
                        encoding="utf-8")
    train_id, blocked_eval_id = "a" * 32, "b" * 32
    source_policy = tmp_path / "models" / "final.pt"
    source_policy.parent.mkdir()
    source_policy.write_bytes(b"policy bytes from the completed baseline train")
    frozen_policy = (real.run_root / "experiments" / "baseline" / train_id / "policy.pt")
    frozen_policy.parent.mkdir(parents=True)
    frozen_policy.write_bytes(source_policy.read_bytes())
    identity_field, identity = artifact_identity(frozen_policy)
    train_dir = real.run_root / "attempts" / train_id
    blocked_eval_dir = real.run_root / "attempts" / blocked_eval_id
    train_dir.mkdir(parents=True)
    blocked_eval_dir.mkdir(parents=True)
    train_receipt = {
        "schema_version": 1, "run_id": real.run_id, "attempt_id": train_id,
        "node_id": "train", "stage": "train", "status": "completed",
        "returncode": 0, "termination_reason": "normal_exit",
        "settings_digest": object_digest(settings),
        "comparison_protocol_sha256": common.digest(protocol),
        "working_directory": str(tmp_path),
        "argv": [sys.executable, str(tmp_path / "ppo.py")],
        "gpu_device_uuid": "GPU-test-5090",
        "artifact": {"checked": True, "matched": 1,
                     "candidate_paths": ["models/final.pt"]},
        "training_progress": {"status": "observed"},
    }
    (train_dir / "receipt.json").write_text(json.dumps(train_receipt), encoding="utf-8")
    blocked_eval = {
        "schema_version": 1, "run_id": real.run_id,
        "attempt_id": blocked_eval_id, "node_id": "evaluate", "status": "blocked",
        "ran": False, "command_started": False,
        "termination_reason": "gpu_unavailable",
        "settings_digest": object_digest(settings),
    }
    (blocked_eval_dir / "receipt.json").write_text(json.dumps(blocked_eval),
                                                     encoding="utf-8")
    original = {
        "label": "baseline", "ok": False, "where": "evaluate",
        "why": "evaluate was refused before start because the card was briefly busy",
        "settings": settings,
        "train": {"attempt_id": train_id, "returncode": 0, "reused": False},
        "policy_artifact": {"path": str(frozen_policy), "source": str(source_policy),
                            identity_field: identity},
        "evaluate": {"attempt_id": blocked_eval_id, "returncode": None},
    }
    (real.run_root / "measurements").mkdir(parents=True)
    measurement_path = real.run_root / "measurements" / "baseline.json"
    measurement_path.write_text(json.dumps(original), encoding="utf-8")
    failure = {"round": 0, "label": "baseline", "metric_value": None,
               "metric_utility": None, "metric_name": "success_rate",
               "settings": settings, "measured": False,
               "why_not": original["why"],
               "failure": {"stage": "evaluate", "attempt_id": blocked_eval_id,
                           "termination_reason": "gpu_unavailable"}}
    real._advance_best("baseline", score=None, scale="success_rate",
                       why="original baseline failed before evaluation")
    real._advance_best("baseline-progress", score=0.2, scale="progress",
                       why="baseline training completed")
    real.rubric = objective.Rubric([objective.Check(
        name="measurement", question="Did a measured value come out?", weight=1.0)])
    real.rubric_built = True
    session = {"schema_version": 1, "run_id": real.run_id,
               "repository": str(real.repo), "status": "paused", "rounds": 1,
               "next_round": 1, "current_measurement_label": "baseline",
               "history": [failure], "baseline_rate": None}
    report = {"schema_version": 1, "run_id": real.run_id,
              "repo": str(real.repo), "run_status": "paused", "rounds": [failure],
              "best": real.snapshots.best().as_dict(),
              "objective": real.rubric.as_dict(), "next_round": 1}
    (real.run_root / "controller_session.json").write_text(json.dumps(session),
                                                            encoding="utf-8")
    (real.run_root / "research_report.json").write_text(json.dumps(report),
                                                       encoding="utf-8")

    calls = []
    recovery_calls = 0

    def evaluate_only(*, settings, label, _reuse_training_attempt_id,
                      _baseline_recovery, _reuse_policy_artifact=None):
        nonlocal recovery_calls
        recovery_calls += 1
        calls.append((_reuse_training_attempt_id, _baseline_recovery,
                      _reuse_policy_artifact))
        assert label == "baseline"
        assert _reuse_policy_artifact == original["policy_artifact"]
        if recovery_calls == 1:
            return {"label": "baseline", "ok": False, "where": "policy_archive",
                    "settings": settings,
                    "archive_error": "FileExistsError: archive destination already exists",
                    "why": "policy bytes could not be frozen before evaluation",
                    "train": {"attempt_id": train_id, "returncode": 0}}
        return {"label": "baseline", "ok": True, "settings": settings,
                "metric_value": 0.5, "metric_utility": 0.5, "success_rate": 0.5,
                "train": {"attempt_id": train_id, "reused": True},
                "policy_artifact": original["policy_artifact"],
                "evaluate": {"attempt_id": "c" * 32, "returncode": 0}}

    monkeypatch.setattr(real, "measure", evaluate_only)
    first = real.recover_unscored_baseline()

    assert first["ok"] is False
    assert first["where"] == "policy_archive"
    first_state_path = train_dir / "baseline_recovery.json"
    first_state_bytes = first_state_path.read_bytes()
    original_backup_bytes = (train_dir / "baseline_unscored.json").read_bytes()
    assert json.loads(first_state_bytes)["status"] == "failed"
    assert json.loads(first_state_bytes)["evaluation_attempt_id"] is None

    recovered = real.recover_unscored_baseline()

    assert recovered["ok"] is True, recovered
    assert recovered["recovery"]["kind"] == "reevaluate_after_prestart_resource_block"
    assert recovered["recovery"]["training_reused"] is True
    assert recovered["recovery"]["recovery_attempt"] == 2
    assert recovered["recovery"]["superseded_evaluation_attempt_id"] == blocked_eval_id
    assert len(calls) == 2
    assert all(call[:2] == (train_id, True) for call in calls)
    recovery_state = json.loads((train_dir / "baseline_recovery_attempt_2.json").read_text())
    assert recovery_state["status"] == "completed"
    assert recovery_state["recovery_attempt"] == 2
    assert recovery_state["recovery_kind"] == "reevaluate_after_prestart_resource_block"
    assert first_state_path.read_bytes() == first_state_bytes
    assert (train_dir / "baseline_unscored.json").read_bytes() == original_backup_bytes
    assert json.loads((train_dir / "baseline_unscored.json").read_text()) == original
    assert json.loads((real.run_root / "controller_session.json").read_text())[
        "history"][0]["recovered_from"]["training_reused"] is True


def test_transient_selector_transport_failure_can_be_retried_without_rewriting_history(
        tmp_path):
    real = ambiguous_policy_research(tmp_path)
    record = real.run_stage("train", settings={})

    class FlakySelector:
        model = "flaky-selector-test-double"

        def __init__(self):
            self.calls = 0
            self.delegate = ArtifactSelectionClient()

        def chat_with_metadata(self, system, user, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary transport failure")
            return self.delegate.chat_with_metadata(system, user, **kwargs)

    client = FlakySelector()
    real.client = client
    first = real._select_policy_artifact(record)
    first_ref = first["artifact_selection"]["selection_ref"]
    first_path = real.output / first_ref
    first_bytes = first_path.read_bytes()
    assert first["artifact_selection"]["status"] == "unavailable"

    second = real._select_policy_artifact(record, retry_unavailable=True)

    assert second["artifact_selection"]["status"] == "selected"
    assert second["artifact_selection"]["selection_ref"] != first_ref
    assert first_path.read_bytes() == first_bytes
    assert client.calls == 2


def test_actual_code_patch_file_is_audited_even_when_idea_omits_it(tmp_path):
    from types import SimpleNamespace
    real = research(tmp_path)
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("score = 0\n", encoding="utf-8")
    idea = SimpleNamespace(label="claim a better score", touches=[], crosses="none",
                           change={"file": "evaluate.py", "find": "score = 0",
                                   "replace": "score = 1"})
    result = real.code_change(idea)
    assert result["applied"] is False and result["red_line"] == "R2"
    assert evaluator.read_text(encoding="utf-8") == "score = 0\n"


def test_code_patch_persists_recovery_manifest_before_applying_source(tmp_path):
    source = tmp_path / "loader.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    real = research(tmp_path)
    idea = Idea(label="repair loader", granularity="code",
                change={"file": "loader.py", "find": "VALUE = 1",
                        "replace": "VALUE = 2"}, touches=["loader.py"])

    result = real.code_change(idea, transaction_round=4)

    manifest = json.loads((real.run_root / "code_changes" / "round-4.json").read_text(
        encoding="utf-8"))
    assert result["applied"] is True
    assert result["snapshot"] == "before-round-4"
    assert result["transaction_ref"] == "code_changes/round-4.json"
    assert manifest["status"] == "applied"
    assert manifest["idea_label"] == idea.label
    assert manifest["patches"] == [idea.change]
    snapshot = real.snapshots.get("before-round-4")
    assert snapshot is not None
    assert (real.snapshots.blobs / snapshot.files["loader.py"]).read_text(
        encoding="utf-8") == "VALUE = 1\n"
    assert source.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_candidate_measurement_checkpoint_separates_saved_result_from_round_commit(
        tmp_path, monkeypatch):
    real = research(tmp_path)
    session_path = real.run_root / "controller_session.json"
    session_path.write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id, "repository": str(real.repo),
        "status": "running", "action": "round_1",
        "pending_action": {"round": 1, "phase": "measurement_starting",
                           "idea": {"label": "candidate"},
                           "changed": {"files": []}},
    }), encoding="utf-8")
    result = {"label": "round_1", "ok": True, "metric_value": 0.5,
              "success_rate": 0.5, "settings": {}}
    measurement_path = real.run_root / "measurements" / "round_1.json"

    def fake_measure(**kwargs):
        measurement_path.parent.mkdir(parents=True, exist_ok=True)
        measurement_path.write_text(json.dumps(result), encoding="utf-8")
        return dict(result)

    monkeypatch.setattr(real, "_measure", fake_measure)

    assert real.measure(settings={}, label="round_1") == result
    session = json.loads(session_path.read_text(encoding="utf-8"))
    pending = session["pending_action"]
    assert session["status"] == "running" and session["action"] == "round_1"
    assert pending["phase"] == "measurement_recorded"
    assert pending["measurement_ref"] == "measurements/round_1.json"
    assert pending["measurement_ok"] is True and pending["metric_value"] == 0.5
    assert pending["measurement_sha256"]
    assert pending["result_sha256"] == object_digest(result)
    finalization = pending["round_finalization"]
    assert finalization["status"] == "in_progress"
    assert finalization["round"] == 1 and finalization["idea_label"] == "candidate"
    assert finalization["measurement_sha256"] == pending["measurement_sha256"]
    assert finalization["result_sha256"] == pending["result_sha256"]
    assert finalization["steps"]["demo_capture"]["status"] == "pending"
    # Reconciliation of this checkpoint remains evidence-only; it does not append a round.
    assert "history" not in session
    assert real._checkpoint_candidate_measurement(label="round_1", result=result) is True


def test_candidate_finalization_journals_unknown_step_without_replaying_it(tmp_path):
    from autosim.research.research_state import ResearchStateError

    real = research(tmp_path, run_id="finalization-fault")
    session_path = real.run_root / "controller_session.json"
    session_path.write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id, "repository": str(real.repo),
        "status": "running", "action": "round_1",
        "pending_action": {"round": 1, "phase": "measurement_starting",
                           "idea": {"label": "candidate"}},
    }), encoding="utf-8")
    result = {"label": "round_1", "ok": True, "metric_value": 0.5}
    measurement_path = real.run_root / "measurements" / "round_1.json"
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_path.write_text(json.dumps(result), encoding="utf-8")
    assert real._checkpoint_candidate_measurement(label="round_1", result=result)

    def interrupted_side_effect():
        raise RuntimeError("simulated crash after external effect may have started")

    with pytest.raises(RuntimeError, match="simulated crash"):
        real._run_candidate_finalization_step(
            label="round_1", step="demo_capture", action=interrupted_side_effect)

    session = json.loads(session_path.read_text(encoding="utf-8"))
    journal = session["pending_action"]["round_finalization"]
    assert journal["status"] == "unresolved"
    assert journal["steps"]["demo_capture"]["status"] == "unknown"
    assert journal["steps"]["demo_capture"]["evidence"]["exception"] == "RuntimeError"
    with pytest.raises(ResearchStateError, match="cannot start finalization step"):
        real._update_candidate_finalization(
            label="round_1", step="demo_capture", status="started")


def test_candidate_measurement_checkpoint_refuses_a_different_saved_result(tmp_path):
    from autosim.research.research_state import ResearchStateError

    real = research(tmp_path)
    session_path = real.run_root / "controller_session.json"
    session_path.write_text(json.dumps({
        "schema_version": 1, "run_id": real.run_id, "repository": str(real.repo),
        "status": "running", "action": "round_1",
        "pending_action": {"round": 1, "phase": "measurement_starting"},
    }), encoding="utf-8")
    measurement_path = real.run_root / "measurements" / "round_1.json"
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_path.write_text(json.dumps({"label": "round_1", "ok": True,
                                            "metric_value": 0.3}), encoding="utf-8")

    with pytest.raises(ResearchStateError, match="does not match"):
        real._checkpoint_candidate_measurement(
            label="round_1", result={"label": "round_1", "ok": True,
                                     "metric_value": 0.5})
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["pending_action"]["phase"] == "measurement_starting"


def test_evaluation_refuses_a_changed_frozen_protocol(tmp_path):
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("score = 0\n", encoding="utf-8")
    real = research(tmp_path)
    evaluator.write_text("score = 1\n", encoding="utf-8")
    result = real.run_stage("evaluate", settings={})
    assert result["status"] == "blocked"
    assert result["ran"] is False
    assert "frozen protocol changed" in result["why"]
    receipt = json.loads((real.run_root / "attempts" / result["attempt_id"] /
                          "receipt.json").read_text(encoding="utf-8"))
    assert receipt["termination_reason"] == "protocol_changed"


def test_evaluation_refuses_new_protected_file_after_protocol_freeze(tmp_path):
    real = research(tmp_path)
    (tmp_path / "evaluate_new.py").write_text("print('succ: 1.0')\n", encoding="utf-8")
    result = real.run_stage("evaluate", settings={})
    assert result["status"] == "blocked"
    assert "gained protected files" in result["why"]
    assert result["ran"] is False


def test_collector_timeout_is_not_success_even_with_a_log_number(tmp_path):
    real = research(tmp_path, stages=("collect",))
    real.backend.sources["collect"] = (
        "def stage_argv_collect(i):\n"
        "    return [i['python'], '-c', 'import time; print(\"succ: 1.0\", flush=True); time.sleep(10)']\n")
    real.backend = DeclarativeBackend(
        repo=tmp_path, answer=real.backend.answer, sources=real.backend.sources,
        parameters=real.backend.parameters)
    real.STAGE_TIMEOUT_SECONDS = 1
    result = real.collect({})
    assert result["ran"] is False
    assert result["returncode"] is None


def test_a_proposal_that_asks_for_data_it_cannot_get_says_so_and_stops(tmp_path):
    """A validated field that the loop ignores is a handle that does not exist.

    A proposal carries a `collection` half: the validator reads its `enabled` flag, the space
    declares collection axes, and the protocol's notes say that half decides what data exists.
    The loop read none of it, so the request was accepted, validated, and dropped in silence.
    Whether the system can act on it is a separate question from whether it may pretend it did.
    """
    real = research(tmp_path, stages=("train", "evaluate"))     # no collector derived
    rounds = []

    def propose(evidence, evidence_id, round_index=0):
        rounds.append(evidence)
        return {"decision": "experiment", "hypothesis": "more data helps",
                "proposal_id": "p", "expected_validation": "succ rises",
                "evidence_id": evidence_id,
                "training": {"train.n_epochs": 2},
                "collection": {"enabled": True, "episodes": 50}}

    real.propose = propose
    report = real.run(rounds=2)
    last = report["rounds"][-1]
    assert last["status"] == "nothing was measured", last
    assert last["asked_for_and_did_not_get"] == ["enabled", "episodes"]
    assert "collect unavailable" in last["why_not"]
    assert len(rounds) == 1, "the controller was asked again after a round that could not run"


def test_a_proposal_that_asks_for_data_gets_it_and_measures_on_it(tmp_path):
    """The other half of the same rule: when the benchmark can collect, the data is produced
    and the training that follows is measured on it -- not on the data it replaced."""
    import json

    answer = {"stages": {name: {"available": True, "entrypoint": "x.py",
                                "invocation": "python x.py", "artifact": "made.bin"}
                         for name in ("train", "evaluate", "collect")}}
    produced = tmp_path / "collected"
    produced.mkdir()
    (produced / "made.bin").write_text("new trajectories")
    # The dataset the caller supplied is handed to the program as an argument. Writing it into
    # the program's text does not work: `i` belongs to the argv function, which the program
    # the function builds cannot see.
    sources = {
        "train": "\n".join([
            "def stage_argv_train(i):",
            "    return [i['python'], '-c',",
                "            \"import pathlib,sys;p=pathlib.Path(sys.argv[2]);p.mkdir(parents=True,exist_ok=True);(p/'made.bin').write_text('model');print('succ: 0.50 dataset=' + sys.argv[1])\",",
                "            str(i['dataset']), str(i['output'])]",
            ""]),
        "evaluate": "\n".join([
            "def stage_argv_evaluate(i):",
                "    return [i['python'], '-c', \"import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True);(p/'made.bin').write_text('score');print('succ: 0.50')\", i['output']]",
            ""]),
        "collect": "\n".join([
            "def stage_argv_collect(i):",
                f"    return [i['python'], '-c', \"import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True);(p/'made.bin').write_text('data');print('collected')\", i['output']]",
            ""]),
    }
    real = DerivedResearch(
        repo=tmp_path, output=tmp_path / "out",
        backend=DeclarativeBackend(repo=tmp_path, answer=answer, sources=sources,
                                   parameters={}),
        interpreter=Path(sys.executable), space=space(), client=None, stages=sources,
        run_id="collected", compute=ComputeDecision(device="cpu", device_index=0,
                                                    environment={}, why="the test chose it"))
    real.propose = lambda evidence, evidence_id, round_index=0: {
        "decision": "experiment", "hypothesis": "more data helps", "proposal_id": "p",
        "expected_validation": "succ rises", "evidence_id": evidence_id,
        "training": {"train.n_epochs": 2},
        "collection": {"enabled": True, "episodes": 10}}
    report = real.run(rounds=1)
    row = report["rounds"][1]
    assert row["status"] == "measured", row
    # What the collector produced is what the trainer was given, which is the whole point of
    # collecting -- and it is recorded, so a reader can tell which data a round measured on.
    assert row["collected"].endswith("made.bin"), row
    collected_events = [e for e in real.events if e.get("event") == "collection"]
    assert collected_events and collected_events[-1]["ran"] is True, real.events


def test_a_proposal_that_asks_for_nothing_extra_says_nothing(tmp_path):
    """Silence about a field nobody set is not the same as silence about one that was."""
    real = research(tmp_path, stages=("train", "evaluate"))
    real.propose = lambda evidence, evidence_id, round_index=0: {
        "decision": "experiment", "hypothesis": "h", "proposal_id": "p",
        "expected_validation": "v", "evidence_id": evidence_id,
        "training": {"train.n_epochs": 2}}
    report = real.run(rounds=1)
    assert report["rounds"][1]["asked_for_and_did_not_get"] == []


def test_the_step_before_the_trainer_is_run_when_the_benchmark_has_one(tmp_path):
    """A stage the loop holds and never invokes is a stage the loop does not have.

    `prepare_data`'s role is "turn the benchmark's shipped demonstrations into the form its
    trainer reads", and `measure` went straight to `train`. RoboTwin's trainer reads a table
    that step generates: the loop failed on a missing key twice with the diagnosis right both
    times -- "the config that would hold it is a file, not a field, and the stage must be run
    so that key exists" -- and no invocation change could supply it.
    """
    import json as _json

    from autosim.research.derived_research import STAGE_ROLES

    assert "prepare_data" in STAGE_ROLES
    real = research(tmp_path, stages=("prepare_data", "train", "evaluate"))
    real.measure(settings={}, label="one")
    real.measure(settings={}, label="two")
    events = (real.run_root / "events.json")
    rows = _json.loads(events.read_text(encoding="utf-8"))["rows"] if events.is_file() else []
    # It ran, and it ran once: the conversion does not depend on what is being varied, so
    # doing it per measurement would re-do it for every arm.
    assert sum(1 for row in rows if row.get("event") == "prepared_data") == 1


def test_a_benchmark_without_that_step_is_not_asked_for_it(tmp_path):
    real = research(tmp_path, stages=("train", "evaluate"))
    real.measure(settings={}, label="one")
    assert getattr(real, "_prepared", False) is False


def test_an_evaluator_with_no_trainer_still_produces_a_number(tmp_path):
    """A benchmark that ships a policy can be scored without training one.

    Nothing calls `evaluate` outside `measure`, and `measure` insisted on `train` first -- so
    a benchmark with a runnable evaluator and a released checkpoint reported that it had no
    number, forever. What this buys is the measurement rung and a baseline to know there is
    something to beat. What it does not buy is a candidate: the weights being scored are the
    benchmark's, which is why the record says which of the two it scored.
    """
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    (shipped / "model.pth").write_bytes(b"weights")
    answer = {"stages": {"evaluate": {"available": True, "entrypoint": "x.py",
                                      "invocation": "python x.py", "artifact": ""}}}
    sources = {"evaluate": "def stage_argv_evaluate(i):\n"
                           "    return [i['python'], '-c', \"print('succ: 0.42')\"]\n"}
    real = DerivedResearch(
        repo=tmp_path, output=tmp_path / "out",
        backend=DeclarativeBackend(repo=tmp_path, answer=answer, sources=sources,
                                   parameters={}),
        interpreter=Path(sys.executable), space=space(), client=None, stages=sources,
        run_id="eval-only", checkpoint=str(shipped),
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="the test chose it"))
    report = real.run(rounds=1)
    baseline = report["rounds"][0]
    assert baseline["measured"] is True, report["rounds"]
    assert baseline["success_rate"] == 0.42
    record = json.loads((real.run_root / "measurements" / "baseline.json").read_text())
    assert "the checkpoint the benchmark ships" in record["scored"]


def test_with_no_shipped_checkpoint_an_evaluator_alone_measures_nothing(tmp_path):
    """The other half of the same rule: an evaluator with nothing to score is not a
    measurement, and saying so is the finding."""
    answer = {"stages": {"evaluate": {"available": True, "entrypoint": "x.py",
                                      "invocation": "python x.py", "artifact": ""}}}
    sources = {"evaluate": "def stage_argv_evaluate(i):\n"
                           "    return [i['python'], '-c', \"print('succ: 0.42')\"]\n"}
    real = DerivedResearch(
        repo=tmp_path, output=tmp_path / "out",
        backend=DeclarativeBackend(repo=tmp_path, answer=answer, sources=sources,
                                   parameters={}),
        interpreter=Path(sys.executable), space=space(), client=None, stages=sources,
        run_id="eval-only", checkpoint="",
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="the test chose it"))
    report = real.run(rounds=1)
    assert report["rounds"][0]["measured"] is False
    assert "train" in report["rounds"][0]["why_not"]


def test_declared_source_controller_can_be_evaluated_without_weights(tmp_path):
    """A planner or source-defined controller need not create a checkpoint to be scored."""
    answer = {"stages": {"evaluate": {"available": True, "entrypoint": "score.py",
                                      "invocation": "python score.py", "artifact": ""}}}
    sources = {"evaluate": "def stage_argv_evaluate(i):\n"
                           "    return [i['python'], '-c', \"print('mean_reward: 4.25')\"]\n"}
    declaration = {"task_contract": {"policy_representation": "source"},
                   "research_goal": {"primary_metric": {"name": "mean_reward",
                                                        "direction": "maximize"}}}
    real = DerivedResearch(
        repo=tmp_path, output=tmp_path / "out",
        backend=DeclarativeBackend(repo=tmp_path, answer=answer, sources=sources,
                                   parameters={}),
        interpreter=Path(sys.executable), space=space(), client=None, stages=sources,
        run_id="source-policy", declaration=declaration,
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="the test chose it"))
    report = real.run(rounds=0)
    baseline = report["rounds"][0]
    assert baseline["measured"] is True
    assert baseline["metric_value"] == 4.25
    record = json.loads((real.run_root / "measurements" / "baseline.json").read_text())
    assert record["restorable_policy"] is False
    assert "source-defined controller" in record["scored"]


@pytest.mark.parametrize("non_ascii", [False, True])
def test_real_evaluation_media_is_linked_only_to_its_scored_attempt(
        tmp_path, monkeypatch, non_ascii):
    from autosim.research import run_record

    repo = tmp_path / "中文目录" if non_ascii else tmp_path
    repo.mkdir(exist_ok=True)
    monkeypatch.setenv("AUTOSIM_STAGE_ROOT", str(tmp_path / "ascii-stage-alias"))
    answer = {"stages": {"evaluate": {"available": True, "entrypoint": "score.py",
                                      "invocation": "python score.py", "artifact": ""}}}
    # A valid tiny GIF is written by the synthetic evaluator itself, not pre-seeded in the
    # run. It has attempt provenance, but no invented task or episode identity.
    gif = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")
    code = ("import pathlib,sys; p=pathlib.Path(sys.argv[1])/'eval_result'; "
            "p.mkdir(parents=True,exist_ok=True); "
            f"(p/'rollout.gif').write_bytes({gif!r}); "
            "print('mean_reward: 4.25')")
    sources = {"evaluate": "def stage_argv_evaluate(i):\n"
                           f"    return [i['python'], '-c', {code!r}, i['output']]\n"}
    declaration = {"task_contract": {"policy_representation": "source"},
                   "research_goal": {"primary_metric": {"name": "mean_reward",
                                                        "direction": "maximize"}}}
    real = DerivedResearch(
        repo=repo, output=repo / "out",
        backend=DeclarativeBackend(repo=repo, answer=answer, sources=sources,
                                   parameters={}),
        interpreter=Path(sys.executable), space=space(), client=None, stages=sources,
        run_id="media-source", declaration=declaration,
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="the test chose it"))
    measured = real.measure(settings={"task": "Lift", "seed": 12}, label="baseline")
    assert measured["ok"] is True
    manifest = json.loads((real.run_root / "media" / "manifest.json").read_text())
    assert len(manifest["recordings"]) == 1
    row = manifest["recordings"][0]
    assert row["same_scored_evaluation"] is True
    assert row["attempt_id"] == measured["evaluate"]["attempt_id"]
    assert row["episode"] is None and row["policy_sha256"] is None
    assert row["task"] == "Lift" and row["seed"] == 12
    assert row["evaluation_attempt_id"] == measured["evaluate"]["attempt_id"]
    assert len(row["evaluation_receipt_sha256"]) == 64
    assert len(row["comparison_protocol_sha256"]) == 64
    assert row["episode_identity_status"] == \
        "aggregate_evaluation_media_not_bound_to_one_episode"
    assert row["media_capture_seconds"] >= 0
    assert row["capture_seconds"] >= row["stage_seconds"]

    # A later byte change breaks the lineage even though the filename is unchanged.
    receipt = json.loads((real.run_root/'attempts'/measured['evaluate']['attempt_id']/'receipt.json').read_text())
    (Path(receipt["output_directory"]) / "eval_result" / "rollout.gif").write_bytes(b"changed")
    run_record.generate(real.run_root)
    changed = json.loads((real.run_root / "media" / "manifest.json").read_text())
    assert changed["recordings"][0]["same_scored_evaluation"] is None


def test_a_stage_can_run_under_an_interpreter_of_its_own(tmp_path):
    """One interpreter per run is the assumption that stops this system on a benchmark whose
    stages do not share one environment.

    The shape is ordinary now: an entry point that is a shell script starting a policy server
    under one environment and a simulation under another, or a trainer and an evaluator
    written against different framework versions. Nothing about "which python" is a property
    of the run; it is a property of the command.
    """
    mine = tmp_path / "other/bin/python"
    mine.parent.mkdir(parents=True)
    mine.write_text("#!/bin/sh\n", encoding="utf-8")
    mine.chmod(0o755)
    answer = {"stages": {"evaluate": {"available": True, "entrypoint": "x.py",
                                      "invocation": "python x.py", "artifact": "",
                                      "interpreter": str(mine)}}}
    sources = {"evaluate": "def stage_argv_evaluate(i):\n    return [i['python']]\n"}
    backend = DeclarativeBackend(repo=tmp_path, answer=answer, sources=sources, parameters={})
    # The caller passes one interpreter for the whole run; the stage's own wins.
    argv = backend.argv("evaluate", {"python": sys.executable})
    assert argv == [str(mine)]


def test_a_stage_without_one_uses_the_runs_interpreter(tmp_path):
    answer = {"stages": {"evaluate": {"available": True, "entrypoint": "x.py",
                                      "invocation": "python x.py", "artifact": ""}}}
    sources = {"evaluate": "def stage_argv_evaluate(i):\n    return [i['python']]\n"}
    backend = DeclarativeBackend(repo=tmp_path, answer=answer, sources=sources, parameters={})
    assert backend.argv("evaluate", {"python": sys.executable}) == [sys.executable]


def test_every_measured_arm_is_counted_including_the_ones_that_failed(tmp_path):
    """A research loop reports the best of its arms and not how many it took, and the count is
    the part a reader needs: the best of three and the best of thirty are the same sentence
    with the same numbers and mean different things.

    The arms that failed count. A round whose trainer died took the round, cost the wall clock
    and is what the next idea was chosen from -- and a record that drops them describes a
    search smaller than the one that happened, which is the direction that flatters the result.
    """
    real = research(tmp_path)
    assert real.measure(settings={}, label="baseline")["ok"] is True
    assert real.measure(settings={"train.n_epochs": 2}, label="round_0")["ok"] is True
    # A round that cannot run at all: no command was derived for one half of the measurement.
    real.backend.sources.pop("evaluate")
    real.backend.answer["stages"]["evaluate"]["available"] = False
    assert real.measure(settings={"train.n_epochs": 3}, label="round_1")["ok"] is False
    record = selection.summary(real.run_root)
    assert record["attempted"] == 3
    assert record["scored"] == 2
    assert record["failed"] == 1
    assert record["best_is_the_maximum_of"] == 2
    assert record["baseline"]["label"] == "baseline"
    # This stub prints the same number for every arm, so the two scored arms tie -- and the
    # tie goes to the baseline. That is the conservative direction on purpose: a run whose
    # candidate reads exactly the same as what it started from has not improved on it, and
    # naming the candidate "best" would turn a tie into a gain in the run's own summary.
    assert record["best"]["label"] == "baseline"
    assert record["identical_reading"] is True


def test_the_run_reports_its_own_best_number_as_a_maximum_of_draws(tmp_path):
    """The sentence the record exists to make sayable, asserted where a reader would look for
    it: the run's result, not a side file."""
    real = research(tmp_path)
    real.measure(settings={}, label="baseline")
    report = real.run(rounds=1)
    record = selection.summary(real.run_root)
    assert record["best_is_the_maximum_of"] >= 1
    assert record["reading"].startswith(str(record["best"]["metric_value"]))
    assert "maximum of draws" in record["reading"]
    assert report["rounds"], "the run still finished, the accounting is not a brake"


def test_a_refused_arm_is_recorded_as_an_arm_that_produced_nothing(tmp_path):
    """A protocol violation is refused before anything runs, and it is still an arm the loop
    took: it chose that candidate, and a count that skips it is the count of a different run."""
    real = research(tmp_path)
    assert real.measure(settings={"task": "Lift", "seed": 1}, label="baseline")["ok"] is True
    refused = real.measure(settings={"task": "Can", "seed": 1}, label="round_0")
    assert refused["where"] == "comparison_protocol"
    record = selection.summary(real.run_root)
    assert record["attempted"] == 2 and record["scored"] == 1


def test_secondary_objectives_are_part_of_immutable_comparison_protocol(tmp_path):
    from autosim.research.metric_guardrails import specifications
    real = research(tmp_path)
    declaration = {"research_goal": {"guardrail_metrics": [{"name": "latency", "unit": "seconds",
        "direction": "minimize", "source": "log", "max_regression": .1}]}}
    real.guardrail_specs = specifications(declaration)
    assert not real._comparison_protocol_violation({}, target="evaluate", freeze=True)
    real.guardrail_specs[0]["max_regression"] = 100
    assert "immutable comparison protocol changed" in real._comparison_protocol_violation({}, target="evaluate", freeze=True)
