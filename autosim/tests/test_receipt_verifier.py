"""Receipt consistency is useful, but explicitly not a native confirmation run."""

import json
import sys
from pathlib import Path

from autosim.research.adapter_protocol import OptimizationSpace
from autosim.research.compute_decision import ComputeDecision
from autosim.research.declarative_backend import DeclarativeBackend
from autosim.research.derived_research import DerivedResearch
from autosim.research.metric_contract import MetricSpec
from autosim.research.receipt_verifier import verify_measurement


def research(tmp_path):
    answer = {"stages": {name: {"available": True, "entrypoint": "x.py",
                               "invocation": "python x.py", "artifact": "models/*.pth"}
                         for name in ("train", "evaluate")}}
    source = ("def stage_argv_{name}(i):\n"
              "    return [i['python'], '-c', "
              "'import pathlib,sys; p=pathlib.Path(sys.argv[1])/\"models\"; "
              "p.mkdir(parents=True,exist_ok=True); "
              "(p/\"model.pth\").write_text(\"checkpoint\"); "
              "print(\"succ: 0.50\")', i['output']]\n")
    sources = {name: source.format(name=name) for name in ("train", "evaluate")}
    return DerivedResearch(
        repo=tmp_path, output=tmp_path / "out",
        backend=DeclarativeBackend(repo=tmp_path, answer=answer,
                                   sources=sources, parameters={}),
        interpreter=Path(sys.executable), space=OptimizationSpace(), client=None,
        stages=sources, run_id="verify",
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="synthetic verifier test"))


def test_rechecks_score_from_evaluation_log_and_attempt(tmp_path):
    real = research(tmp_path)
    measured = real.measure(settings={}, label="baseline")
    assert measured["ok"] is True
    verdict = verify_measurement(real.run_root, "baseline")
    assert verdict["status"] == "consistent"
    assert verdict["reread_metric"] == 0.5
    assert "no independent native rerun" in verdict["limitation"]


def test_rejects_tampered_metric_and_nonzero_receipt(tmp_path):
    real = research(tmp_path)
    measured = real.measure(settings={}, label="baseline")
    path = real.run_root / "measurements" / "baseline.json"
    claimed = json.loads(path.read_text())
    claimed["metric_value"] = 0.9
    path.write_text(json.dumps(claimed))
    assert verify_measurement(real.run_root, "baseline")["checks"]["metric_reread"] is False
    path.write_text(json.dumps(measured))
    receipt_path = real.run_root / "attempts" / measured["evaluate"]["attempt_id"] / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["returncode"] = 1
    receipt_path.write_text(json.dumps(receipt))
    assert verify_measurement(real.run_root, "baseline")["checks"]["process_completed"] is False


def test_verifier_binds_settings_to_receipt_and_frozen_protocol(tmp_path):
    real = research(tmp_path)
    measured = real.measure(settings={"task": "Lift", "seed": 4, "episodes": 20},
                            label="baseline")
    assert measured["ok"]
    path = real.run_root / "measurements" / "baseline.json"
    changed = json.loads(path.read_text())
    changed["settings"]["episodes"] = 2
    path.write_text(json.dumps(changed))
    verdict = verify_measurement(real.run_root, "baseline")
    assert verdict["checks"]["receipt_settings"] is False
    assert verdict["checks"]["comparison_protocol"] is False


def test_verifier_rejects_frozen_protocol_byte_change(tmp_path):
    real = research(tmp_path)
    assert real.measure(settings={}, label="baseline")["ok"]
    path = real.run_root / "comparison_protocol.json"
    path.write_text(path.read_text() + " ", encoding="utf-8")
    verdict = verify_measurement(real.run_root, "baseline")
    assert verdict["checks"]["comparison_protocol_bytes"] is False


def test_missing_or_outside_log_is_not_verifiable(tmp_path):
    real = research(tmp_path)
    measured = real.measure(settings={}, label="baseline")
    receipt_path = real.run_root / "attempts" / measured["evaluate"]["attempt_id"] / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["log"] = str(tmp_path / "outside.log")
    receipt_path.write_text(json.dumps(receipt))
    verdict = verify_measurement(real.run_root, "baseline")
    assert verdict["status"] == "unverifiable"
    assert any("outside" in issue for issue in verdict["issues"])


def test_json_score_is_verified_from_frozen_result_bytes(tmp_path):
    real = research(tmp_path)
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "return", "direction": "maximize", "source": "json",
        "json_key": "summary.reward"}}})
    real.backend.answer["stages"]["evaluate"]["artifact"] = "result.json"
    real.backend.sources["evaluate"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', 'import pathlib,json,sys; "
        "pathlib.Path(sys.argv[1],\"result.json\").write_text(" 
        "json.dumps({\"summary\":{\"reward\":2.25}}))', i['output']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                      sources=real.backend.sources, parameters={})
    measured = real.measure(settings={}, label="json_reward")
    assert measured["ok"] is True
    assert verify_measurement(real.run_root, "json_reward")["status"] == "consistent"
    frozen = Path(measured["result_artifact"]["path"])
    frozen.write_text('{"summary":{"reward":99}}')
    verdict = verify_measurement(real.run_root, "json_reward")
    assert verdict["status"] == "inconsistent"
    assert verdict["checks"]["result_bytes"] is False
