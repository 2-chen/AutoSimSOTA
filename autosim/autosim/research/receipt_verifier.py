"""Read-only consistency check of a scored measurement against its process receipt.

This is deliberately narrower than independent benchmark confirmation. It re-reads the
original log without trusting the measurement's copied ``said`` field, but it does not
restart the simulator, validate the native evaluator, or establish statistical improvement.
"""

from __future__ import annotations

import math
import re
import argparse
import json
from pathlib import Path
from typing import Any

from .common import digest, object_digest, read_json
from .experiment_bundle import artifact_identity
from .metric_contract import MetricSpec
from .readings import tail_of
from .policy_consumption import (verify_policy_consumption, verify_rollout_evidence,
                                 verify_metric_lineage)


def _metric(row: dict[str, Any]) -> MetricSpec:
    declared = {"name": row["name"], "direction": row["direction"],
                "unit": row.get("unit"), "source": row.get("source"),
                "min": row.get("minimum"), "max": row.get("maximum"),
                "artifact_root": row.get("artifact_root"),
                "artifact_pattern": row.get("artifact_pattern"),
                "json_key": row.get("json_key"), "csv_column": row.get("csv_column"),
                "json_value_key": row.get("json_value_key"),
                "aggregation": row.get("aggregation"),
                "min_samples": row.get("min_samples", 1),
                "episode_id_column": row.get("episode_id_column"),
                "task_column": row.get("task_column"),
                "initial_state_hash_column": row.get("initial_state_hash_column")}
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": declared}})
    normalized = {**row, "initial_state_hash_column": row.get(
        "initial_state_hash_column") or ""}
    if spec.as_dict() != normalized:
        raise ValueError("metric contract differs from its validated representation")
    return spec


def verify_measurement(root: Path, label: str, *, _policy_lineage: tuple[str,...]=()) -> dict[str, Any]:
    """Return structured checks; ``consistent`` is never a performance claim."""
    root = Path(root).resolve()
    checks: dict[str, bool] = {}
    issues: list[str] = []
    if label in _policy_lineage or len(_policy_lineage)>=32:
        return {'label':label,'status':'unverifiable','checks':{},
                'issues':['frozen policy lineage cycle or bounded-depth limit reached']}

    def check(name: str, condition: bool, why: str) -> None:
        checks[name] = bool(condition)
        if not condition:
            issues.append(why)

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
        raise ValueError("measurement label must be a short path-safe identifier")
    path = root / "measurements" / f"{label}.json"
    try:
        measured = read_json(path)
    except (OSError, ValueError) as exc:
        return {"label": label, "status": "unverifiable", "checks": {},
                "issues": [f"measurement missing or invalid: {type(exc).__name__}"]}
    if not isinstance(measured, dict):
        return {"label": label, "status": "unverifiable", "checks": {},
                "issues": ["measurement is not an object"]}
    check("measurement_claims_valid", measured.get("ok") is True,
          "measurement does not claim a valid score")
    evaluation = measured.get("evaluate") or {}
    if not isinstance(evaluation, dict):
        evaluation = {}
    attempt = str(evaluation.get("attempt_id") or "")
    if not re.fullmatch(r"[a-f0-9]{32}", attempt):
        return {"label": label, "status": "unverifiable", "checks": checks,
                "issues": issues + ["evaluation attempt ID absent or malformed"]}
    try:
        receipt = read_json(root / "attempts" / attempt / "receipt.json")
    except (OSError, ValueError) as exc:
        return {"label": label, "status": "unverifiable", "checks": checks,
                "issues": issues + [f"evaluation receipt missing or invalid: {type(exc).__name__}"]}
    if not isinstance(receipt, dict):
        return {"label": label, "status": "unverifiable", "checks": checks,
                "issues": issues + ["evaluation receipt is not an object"]}
    expected_node = "evaluate"
    graph_run = measured.get("graph_run") or {}
    if graph_run:
        if not isinstance(graph_run, dict):
            return {"label": label, "status": "unverifiable", "checks": checks,
                    "issues": issues + ["graph run record is malformed"]}
        graph_attempt = str(graph_run.get("graph_attempt_id") or "")
        if not re.fullmatch(r"[a-f0-9]{32}", graph_attempt):
            return {"label": label, "status": "unverifiable", "checks": checks,
                    "issues": issues + ["graph attempt ID absent or malformed"]}
        try:
            from .execution_graph import ExecutionGraph
            graph = ExecutionGraph(read_json(root / "execution_graph_frozen.json"))
            graph_receipt = read_json(root / "graph_runs" / f"{graph_attempt}.json")
            expected_node = str(graph_run.get("target") or "")
            if expected_node not in graph.nodes or graph.nodes[expected_node].role != "evaluate":
                raise ValueError("graph target is not an evaluate node")
            recorded = next(row for row in graph_receipt.get("nodes") or []
                            if row.get("id") == expected_node)
        except (OSError, ValueError, TypeError, StopIteration) as exc:
            return {"label": label, "status": "unverifiable", "checks": checks,
                    "issues": issues + [f"graph provenance could not be reread: {exc}"]}
        check("graph_identity", graph_receipt.get("status") == "completed" and
              recorded.get("attempt_id") == attempt and
              graph_receipt.get("graph_attempt_id") == graph_attempt,
              "graph receipt does not bind this scored attempt")
    check("receipt_identity", receipt.get("attempt_id") == attempt and
          receipt.get("node_id") == expected_node,
          "receipt is not the named evaluate attempt")
    settings = measured.get("settings")
    check("receipt_settings", isinstance(settings, dict) and
          receipt.get("settings_digest") == object_digest(settings),
          "evaluation receipt is not bound to the measurement settings")
    try:
        protocol_path = root / "comparison_protocol.json"
        reference_ref = measured.get('reference_protocol_ref')
        if reference_ref is not None:
            rel = Path(reference_ref)
            if (measured.get('baseline_origin') != 'released_reference'
                    or measured.get('confirmation') or len(rel.parts) != 2
                    or rel.parts[0] != 'reference_protocols'
                    or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}\.json', rel.name)
                    or receipt.get('reference_protocol_ref') != reference_ref):
                raise ValueError('invalid reference protocol ownership')
            protocol_path = root/rel
            if protocol_path.is_symlink() or protocol_path.parent.is_symlink():
                raise ValueError('unsafe reference protocol')
        protocol = read_json(protocol_path)
        if not isinstance(protocol, dict):
            raise ValueError("comparison protocol is not an object")
        keys = protocol.get("protocol_keys") or []
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError("comparison protocol keys are invalid")
        exact = {"task", "tasks", "suite", "benchmark", "seed", "seeds",
                 "episodes", "n_episodes", "horizon", "max_episode_steps",
                 "initial_states", "eval_seed", *keys}
        selected = {key: value for key, value in settings.items()
                    if key in exact or key.startswith(("eval.", "evaluation."))}
        allowed_settings = ({**(protocol.get("settings") or {}),
                             **(protocol.get("confirmation") or {})}
                            if measured.get("confirmation") else protocol.get("settings"))
        check("comparison_protocol", protocol.get("target") == expected_node and
              allowed_settings == selected and
              protocol.get("metric") == measured.get("metric"),
              "measurement differs from the frozen comparison protocol")
        check("comparison_protocol_bytes",
              receipt.get("comparison_protocol_sha256") == digest(protocol_path),
              "comparison protocol bytes changed after the scored attempt")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        check("comparison_protocol", False,
              f"frozen comparison protocol is unavailable or malformed: {exc}")
        check("comparison_protocol_bytes", False,
              "frozen comparison protocol bytes could not be verified")
    check("process_completed", receipt.get("status") == "completed" and
          receipt.get("termination_reason") == "normal_exit" and
          receipt.get("returncode") == 0 and evaluation.get("returncode") == 0,
          "evaluation did not have a matching normal zero exit")
    archived_policy = measured.get("policy_artifact") or {}
    if isinstance(archived_policy, dict) and archived_policy.get("path"):
        policy_path = Path(str(archived_policy["path"]))
        if not policy_path.is_absolute():
            policy_path = root / policy_path
        policy_path = policy_path.resolve()
        try:
            if not policy_path.is_relative_to(root):
                raise ValueError("policy is outside this run")
            identity_field, actual_identity = artifact_identity(policy_path)
            policy_ok = archived_policy.get(identity_field) == actual_identity
        except (OSError, ValueError):
            policy_ok = False
        check("policy_identity", policy_ok,
              "the scored frozen policy is absent or its bytes changed")
    consumption = measured.get("policy_consumption") or {}
    if isinstance(consumption, dict) and consumption.get("status") == "verified":
        try:
            log_path = Path(str(receipt.get("log") or "")).resolve(strict=True)
            policy_path = Path(str(archived_policy.get("path") or "")).resolve(strict=True)
            proof = verify_policy_consumption(log_path, policy_path)
            check("native_policy_consumption", proof.get("status") == "verified",
                  "native policy loader evidence no longer matches frozen policy bytes")
            rollout = verify_rollout_evidence(
                log_path, policy_sha256=str(proof.get("sha256") or ""))
            check("native_rollout_identity", rollout.get("status") == "verified" and
                  (measured.get("rollout_evidence") or {}).get("status") == "verified",
                  "native completed-rollout evidence no longer matches loaded policy")
        except (OSError, ValueError, RuntimeError):
            check("native_policy_consumption", False,
                  "native policy loader evidence could not be independently reread")
    elif isinstance(consumption, dict) and consumption.get("status") not in {None, "not_required"}:
        check("native_policy_consumption", False,
              "measurement has no verified native policy-load event")
    conditions = receipt.get("postconditions") or []
    check("postconditions", isinstance(conditions, list) and
          all(isinstance(item, dict) and item.get("passed") is not False
              for item in conditions), "a receipt postcondition failed or is malformed")
    metric = measured.get("metric") or {}
    try:
        spec = _metric(metric)
    except (KeyError, TypeError, ValueError) as exc:
        return {"label": label, "attempt_id": attempt, "status": "unverifiable",
                "checks": checks, "issues": issues +
                [f"metric contract invalid: {type(exc).__name__}: {exc}"]}
    if spec.source == "log":
        log = Path(str(receipt.get("log") or ""))
        if not log.is_absolute():
            log = root / log
        log = log.resolve()
        if not log.is_relative_to(root) or not log.is_file():
            return {"label": label, "attempt_id": attempt, "status": "unverifiable",
                    "checks": checks, "issues": issues +
                    ["evaluation log is absent or outside the run directory"]}
        reading = spec.read_log(tail_of(log))
    else:
        archived = measured.get("result_artifact") or {}
        if not isinstance(archived, dict):
            archived = {}
        artifact = Path(str(archived.get("path") or ""))
        if not artifact.is_absolute():
            artifact = root / artifact
        artifact = artifact.resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            return {"label": label, "attempt_id": attempt, "status": "unverifiable",
                    "checks": checks, "issues": issues +
                    ["frozen result artifact is absent or outside the run directory"]}
        if artifact.stat().st_size > 64 * 1024**2:
            return {"label": label, "attempt_id": attempt, "status": "unverifiable",
                    "checks": checks, "issues": issues +
                    ["frozen result artifact exceeds verifier read limit"]}
        check("result_bytes", bool(archived.get("content_sha256")) and
              digest(artifact) == archived.get("content_sha256"),
              "frozen result artifact bytes changed since scoring")
        if spec.source in {"json", "csv"}:
            source_evidence = measured.get("metric_artifact_evidence") or {}
            receipt_evidence = receipt.get("metric_result_artifact") or {}
            same_source = (source_evidence.get("status") == "matched" and
                           receipt_evidence.get("status") == "matched" and
                           source_evidence.get("path") == receipt_evidence.get("path") and
                           source_evidence.get("sha256") == receipt_evidence.get("sha256") and
                           source_evidence.get("sha256") == archived.get("content_sha256") and
                           receipt_evidence.get("frozen_copy_sha256") ==
                           archived.get("content_sha256"))
            check("result_receipt_binding", same_source,
                  "frozen structured result is not bound to this evaluation receipt")
            if spec.artifact_pattern:
                expected_ref = str(artifact.resolve().relative_to(root))
                check("result_pattern_binding",
                      source_evidence.get("root") == spec.artifact_root and
                      source_evidence.get("pattern") == spec.artifact_pattern and
                      receipt_evidence.get("root") == spec.artifact_root and
                      receipt_evidence.get("pattern") == spec.artifact_pattern and
                      receipt_evidence.get("frozen_copy_ref") == expected_ref,
                      "structured result path differs from the declared metric contract")
        reading = spec.read(said="", artifact=artifact) if checks["result_bytes"] else {
            "value": None}
    claimed = measured.get("metric_value")
    if isinstance(consumption, dict) and consumption.get("status") == "verified":
        try:
            native_rollout = verify_rollout_evidence(
                log_path, policy_sha256=str(proof.get("sha256") or ""))
            native_lineage = verify_metric_lineage(
                log_path, policy_sha256=str(proof.get("sha256") or ""),
                episode_ids=native_rollout.get("episode_ids") or [],
                value=reading.get("value"))
            check("native_metric_lineage", native_lineage.get("status") == "verified" and
                  (measured.get("metric_lineage") or {}).get("status") == "verified",
                  "native metric did not bind the same completed episodes and policy")
        except (OSError, ValueError, RuntimeError, TypeError):
            check("native_metric_lineage", False,
                  "native metric lineage could not be independently reread")
    valid_number = isinstance(claimed, (int, float)) and not isinstance(claimed, bool)
    check("metric_reread", valid_number and math.isfinite(float(claimed)) and
          reading.get("value") == float(claimed),
          "the recorded native output does not reproduce the claimed metric")
    check("utility_reread", valid_number and math.isfinite(float(claimed)) and
          measured.get("metric_utility") == spec.utility(float(claimed)),
          "the claimed utility disagrees with the metric direction")
    if spec.episode_id_column:
        claimed_reading = measured.get("metric_reading") or {}
        check("native_episode_identity", isinstance(claimed_reading, dict) and
              claimed_reading.get("episode_keys") == reading.get("episode_keys") and
              claimed_reading.get("episodes_completed") ==
              reading.get("episodes_completed") and
              claimed_reading.get("episode_values") == reading.get("episode_values") and
              claimed_reading.get("initial_state_hashes") ==
              reading.get("initial_state_hashes") and
              claimed_reading.get("successes") == reading.get("successes"),
              "native task/episode identities or success totals differ from the result bytes")
    reuse=measured.get('policy_reuse')
    if reuse is not None:
        try:
            from .frozen_policy_trial import _safe
            if not isinstance(reuse,dict) or reuse.get('mode')!='evaluation_only_local_frozen_policy':
                raise ValueError('invalid frozen policy trial identity')
            parent_label=reuse['label']
            if not isinstance(parent_label,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',parent_label):
                raise ValueError('unsafe frozen policy parent label')
            parent_path=_safe(root,'measurements/'+parent_label+'.json')
            if parent_path.stat().st_size>2*1024*1024 or digest(parent_path)!=reuse['measurement_sha256']:
                raise ValueError('frozen policy parent measurement changed')
            parent=read_json(parent_path)
            parent_train=parent.get('train') or {}
            child_train=measured.get('train') or {}
            parent_policy=parent.get('policy_artifact') or {}
            train_id=parent_train.get('attempt_id')
            if (not isinstance(train_id,str) or not re.fullmatch(r'[0-9a-f]{32}',train_id)
                    or reuse.get('training_attempt_id')!=train_id
                    or child_train.get('attempt_id')!=train_id or child_train.get('reused') is not True
                    or measured.get('baseline_origin')!='locally_trained'
                    or parent.get('baseline_origin')!='locally_trained'
                    or measured.get('confirmation') or measured.get('dataset')
                    or measured.get('settings')!=parent.get('settings')
                    or measured.get('policy_artifact')!=parent_policy
                    or reuse.get('policy_sha256')!=parent_policy.get('sha256')
                    or reuse.get('no_new_training_or_data_claim') is not True
                    or (parent.get('evaluate') or {}).get('attempt_id')==attempt):
                raise ValueError('frozen policy trial does not bind the original local weights and settings')
            train_path=_safe(root,'attempts/'+train_id+'/receipt.json')
            if digest(train_path)!=reuse.get('training_receipt_sha256'):
                raise ValueError('frozen policy native training receipt changed')
            verdict=verify_measurement(root,parent_label,_policy_lineage=(*_policy_lineage,label))
            check('frozen_policy_parent',verdict.get('status')=='consistent',
                  'frozen policy parent independent verification failed')
            check('frozen_policy_identity',True,'')
        except (OSError,ValueError,TypeError,KeyError,RuntimeError,AttributeError) as exc:
            check('frozen_policy_identity',False,'frozen policy reuse identity: '+str(exc)[:300])
    return {"label": label, "attempt_id": attempt,
            "status": "consistent" if all(checks.values()) else "inconsistent",
            "checks": checks, "reread_metric": reading.get("value"), "issues": issues,
            "limitation": "Receipt and declared setting consistency only; no independent "
                          "native rerun, trustworthy origin proof for instrumentation, full "
                          "protocol semantics or statistical confirmation."}


def main() -> int:
    parser = argparse.ArgumentParser(description="Recheck a measurement against its run receipt")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("label")
    args = parser.parse_args()
    verdict = verify_measurement(args.run_root, args.label)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict["status"] == "consistent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
