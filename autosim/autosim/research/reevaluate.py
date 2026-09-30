"""Running a scored number again, in a new process, from the records alone.

Everything else this system checks about a measurement is a consistency check on paper.
`receipt_verifier` reads the receipt, the frozen protocol and the log and reports whether they
agree with one another -- and they do agree, because the same process wrote all of them. A
benchmark whose evaluator prints a number it did not compute, or whose stage ran the wrong
task because a parameter was read from the wrong place, produces a receipt that passes every
one of those checks. Its own docstring says so.

The only thing that distinguishes a measured number from a plausible one is running the
evaluation again. So this does: it reads the archived policy and the frozen settings out of
the run's records, rebuilds the evaluator's command from the same recipe the run used,
executes it **in a fresh output directory in a separate process**, reads the metric from the
new native output, and reports whether the number came back.

**The output is always fresh; source isolation depends on the original run.** The command is
rebuilt by the same recorded recipe. Runs with a source-base inventory and a matching
candidate snapshot also rebuild a fresh checkout from the verified original base and apply
the candidate's saved source overlay. Older runs lacking those records reuse their mutable
checkout and say so. In both cases, the evaluator gets a new output directory: a stale result
cannot masquerade as a successful re-run. The interpreter, installed packages and external
assets remain references, not frozen dependencies.

**Three outcomes, and they are not the same finding.** `reproduced` means the evaluator ran
and printed the number again. `not_reproduced` means it ran and printed something else --
which is a finding about the benchmark or the policy. `could_not_reevaluate` means it did not
run: the interpreter is gone, the environment no longer builds, the policy bytes changed
under us. Reporting the third as the second is how a deleted conda environment becomes
evidence that a result was fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .common import digest, object_digest, read_json
from .declarative_backend import DeclarativeBackend
from .experiment_bundle import artifact_identity
from .metric_contract import MetricSpec
from .snapshot import Snapshots

#: Files a run leaves that say how its stages are invoked. All three are read; a record whose
#: recipe is incomplete cannot be re-run and says which piece is missing rather than guessing.
RECIPE_FILES: tuple[str, ...] = ("derived_stages.json", "execution.json", "environment.json")


def _read(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _recipe_root(run_root: Path) -> Path:
    """Where the preparation records live, given where the run's records live.

    A run's own directory holds measurements and events; the commands that produced them are
    written one level up, by the preparation that derived them. Rather than take that as an
    argument the caller would have to know to pass, the layout is read: the directory holding
    `derived_stages.json` is the first one at or above the run root that has it.
    """
    for candidate in (run_root, *run_root.parents):
        if (candidate / "derived_stages.json").is_file():
            return candidate
        if candidate == candidate.parent:
            break
    return run_root.parent


def plan(run_root: Path, label: str) -> dict[str, Any]:
    """Everything a re-evaluation would need, and everything it would be missing.

    Pure: reads files, runs nothing. Split out so the decision to run -- and the reasons not
    to -- can be checked without a benchmark present, and so a reader can see what the
    re-evaluation would be *of* before spending the wall clock.
    """
    run_root = Path(run_root).resolve()
    out: dict[str, Any] = {"run_root": str(run_root), "label": label, "status": "refused",
                           "why": "", "gaps": [], "would_run": ""}

    measured = _read(run_root / "measurements" / f"{label}.json")
    if not isinstance(measured, dict):
        out["why"] = "no such measurement in this run"
        return out
    out["claimed"] = measured.get("metric_value")
    out["measured_ok"] = measured.get("ok") is True
    if not isinstance(out["claimed"], (int, float)) or isinstance(out["claimed"], bool):
        out["why"] = ("this measurement produced no number, so there is nothing to reproduce; "
                      "its result is the failure it recorded")
        return out

    metric_row = measured.get("metric")
    try:
        out["metric"] = MetricSpec.from_declaration(
            {"task_contract": {"primary_metric": metric_row}}).as_dict()
    except (KeyError, TypeError, ValueError) as exc:
        out["why"] = f"the measurement's metric contract cannot be rebuilt: {exc}"
        return out

    # The policy. Its bytes are the whole subject of the re-evaluation, so they are checked
    # before anything is started: re-running an evaluation against a policy that has since
    # been overwritten measures a different policy and would be reported as a failure to
    # reproduce.
    archived = measured.get("policy_artifact") or {}
    source_controller = str(measured.get("scored") or "").startswith(
        "a source-defined controller")
    source_state = (measured.get("source_state") if isinstance(
        measured.get("source_state"), dict) else {})
    source_policy = (measured.get("source_policy_artifact") if isinstance(
        measured.get("source_policy_artifact"), dict) else {})
    path = Path(str(archived.get("path") or ""))
    if not path.is_absolute():
        path = run_root / path
    if source_controller:
        expected = str(source_policy.get("identity_sha256") or "")
        if (source_policy.get("kind") != "source_tree" or not expected or
                source_policy.get("source_snapshot") != source_state.get("source_snapshot") or
                source_policy.get("identity_sha256") != source_state.get("identity_sha256")):
            out["why"] = ("the source-defined controller has no content-addressed source-tree "
                          "identity from measurement time")
            return out
        out["policy"] = {"kind": "source_tree", "expected": expected, "actual": expected,
                         "matches": True}
    else:
        out["policy"] = {"path": str(path),
                         "expected": (archived.get("content_sha256") if path.is_file() else
                                      archived.get("sha256")) or ""}
        if not (path.is_file() or path.is_dir()):
            out["why"] = "the frozen policy this measurement scored is no longer on disk"
            return out
        try:
            _, actual = artifact_identity(path)
        except ValueError as exc:
            out["why"] = str(exc)
            return out
        out["policy"]["actual"] = actual
        out["policy"]["matches"] = bool(out["policy"]["expected"]) and actual == \
            out["policy"]["expected"]
        if not out["policy"]["matches"]:
            out["why"] = ("the archived policy's bytes differ from what the measurement recorded, "
                          "so a re-run would score a different policy")
            return out

    # The settings. A frozen protocol is what makes a re-run the *same* evaluation rather than
    # a new one that happens to use the same command; without it the settings still exist in
    # the measurement, and that is a weaker thing to reproduce against.
    protocol = _read(run_root / "comparison_protocol.json")
    frozen = isinstance(protocol, dict) and isinstance(protocol.get("settings"), dict)
    out["settings"] = dict((protocol.get("settings") if frozen else measured.get("settings"))
                           or {})
    out["settings_frozen"] = bool(frozen)
    if not frozen:
        out["gaps"].append("no frozen comparison protocol is present, so the settings being "
                           "re-run are the ones this measurement recorded rather than ones "
                           "frozen before it ran")

    # Re-evaluation is a new process, but it must not silently change the compute path. New
    # measurements keep this beside the score attempt; older ones may still recover it from
    # their durable attempt receipt. A GPU ordinal without its physical UUID is insufficient
    # to prove that the same card can be selected again.
    evaluation = measured.get("evaluate") if isinstance(measured.get("evaluate"), dict) else {}
    attempt_id = str(evaluation.get("attempt_id") or "")
    stage_receipt = (_read(run_root / "attempts" / attempt_id / "receipt.json")
                     if attempt_id else None)
    stage_receipt = stage_receipt if isinstance(stage_receipt, dict) else {}
    out["evaluation_device"] = (evaluation.get("device") or stage_receipt.get("device") or
                                 measured.get("evaluation_device"))
    out["evaluation_gpu_uuid"] = (
        evaluation.get("gpu_device_uuid") or stage_receipt.get("gpu_device_uuid") or
        measured.get("evaluation_gpu_uuid"))
    compute_environment = (
        evaluation.get("compute_environment") or
        stage_receipt.get("compute_environment") or
        measured.get("evaluation_compute_environment") or {})
    out["evaluation_compute_environment"] = (compute_environment
                                             if isinstance(compute_environment, dict) else {})

    # The recipe: what makes the command rebuildable.
    root = _recipe_root(run_root)
    out["recipe_root"] = str(root)
    stages_doc = _read(root / "derived_stages.json")
    execution = _read(root / "execution.json")
    environment = _read(root / "environment.json")
    for name, doc in zip(RECIPE_FILES, (stages_doc, execution, environment)):
        if not isinstance(doc, dict):
            out["gaps"].append(f"{name} is missing or unreadable at {root}, so the evaluator's "
                               f"command cannot be rebuilt from the records")
    if out["gaps"] and not isinstance(stages_doc, dict):
        out["why"] = "the derivation records needed to rebuild the evaluator are not present"
        return out

    sources = {name: str(row.get("source") or "") for name, row in
               (stages_doc or {}).items() if isinstance(row, dict)}
    parameters = {name: (row.get("parameters") or {}) for name, row in
                  (stages_doc or {}).items() if isinstance(row, dict)}
    answer = {"stages": dict((execution or {}).get("stages") or {})}
    if isinstance((execution or {}).get("execution_graph"), dict):
        answer["execution_graph"] = execution["execution_graph"]
    # The row a stage was *verified* as, over what a later reading of the checkout said --
    # the same precedence the run used, because the verified row is the one carrying the
    # working directory and environment the command actually ran with.
    for stage, row in (stages_doc or {}).items():
        if isinstance(row, dict) and isinstance(row.get("row"), dict):
            answer["stages"][stage] = {**answer["stages"].get(stage, {}), **row["row"],
                                       "available": True}
    out["_sources"] = sources
    out["_parameters"] = parameters
    out["_answer"] = answer
    # The checkout the run read. A record whose checkout has moved cannot be re-run, and
    # saying that is better than running against whatever is at the old path now.
    repo = Path(str((execution or {}).get("repo") or ""))
    out["_repo"] = str(repo)
    workspace = _read(root / "workspace_snapshot.json")
    if (isinstance(workspace, dict) and workspace.get("destination") == str(repo) and
            isinstance(workspace.get("source_entries"), list) and
            workspace.get("source_tree_fingerprint")):
        if source_controller and not source_state:
            out["why"] = "source-defined policy measurements require an isolated source snapshot"
            return out
        if source_state:
            snapshot_name = str(source_state.get("source_snapshot") or "")
            overlay_files = source_state.get("overlay_files")
            expected_identity = object_digest({
                "schema_version": 1,
                "source_tree_fingerprint": workspace.get("source_tree_fingerprint"),
                "source_snapshot": snapshot_name,
                "overlay_files": overlay_files,
                "overlay_modes": source_state.get("overlay_modes") or {},
                "overlay_directories": source_state.get("overlay_directories") or {},
                "copy_mode": source_state.get("copy_mode", "full_worktree"),
                "untracked_assets_omitted": bool(
                    source_state.get("untracked_assets_omitted")),
            }) if isinstance(overlay_files, dict) and snapshot_name else ""
            if (workspace.get("schema_version") != 2 or
                    source_state.get("source_tree_fingerprint") !=
                    workspace.get("source_tree_fingerprint") or
                    source_state.get("identity_sha256") != expected_identity or
                    (source_controller and source_policy.get("identity_sha256") !=
                     expected_identity)):
                out["why"] = "the measured source-state identity does not match the workspace base"
                return out
        else:
            # Older checkpoint measurements used the best-state snapshot convention. They
            # remain readable, but new measurements bind the exact tree observed at scoring.
            snapshot_name = label.replace("round_", "round-", 1)
            overlay_files = None
        snapshots = Snapshots(run_root / "snapshots")
        snapshot = snapshots.get(snapshot_name) if snapshot_name else None
        if snapshot is None:
            out["why"] = ("this isolated measurement has no matching frozen source "
                          f"snapshot ({snapshot_name})")
            return out
        if source_state and dict(snapshot.files) != overlay_files:
            out["why"] = "the source overlay snapshot differs from its measurement identity"
            return out
        for key in snapshot.files.values():
            if key:
                blob = snapshots.blobs / key
                if not blob.is_file() or digest(blob) != key:
                    out["why"] = "the measured source overlay has a missing or corrupt snapshot blob"
                    return out
        out["source_isolation"] = "fresh_copy_planned"
        out["_workspace"] = workspace
        out["_source_snapshot"] = snapshot_name
        out["_source_state"] = source_state
        if not Path(str(workspace.get("source") or "")).is_dir():
            out["why"] = "the original source base needed for a fresh copy is unavailable"
            return out
    else:
        if source_controller:
            out["why"] = "source-defined policy re-evaluation requires the matching isolated workspace manifest"
            return out
        if not repo.is_dir():
            out["why"] = (f"the benchmark checkout this run read is not at {repo}, so the "
                          f"evaluator cannot be rebuilt against the source it was derived from")
            return out
        out["source_isolation"] = "mutable_checkout"
        out["gaps"].append("the run has no reconstructible source-base inventory; "
                           "re-evaluation would use its mutable original checkout")
    interpreter = str((environment or {}).get("interpreter")
                      or (environment or {}).get("python") or "")
    out["interpreter"] = interpreter
    target = str((protocol or {}).get("target") or
                 ((answer.get("execution_graph") or {}).get("score_target") or "evaluate"))
    out["target"] = target
    if not isinstance(answer["stages"].get(target), dict) and target not in sources:
        out["why"] = f"the records do not contain an evaluate stage ({target}) to re-run"
        return out
    if not (sources.get(target) or "").strip():
        out["why"] = (f"the records carry no source for the {target} command, so there is "
                      "nothing to rebuild it from")
        return out
    recorded_device = str(out["evaluation_device"] or "")
    if recorded_device != "cpu" and not recorded_device.startswith("cuda"):
        out["why"] = ("the measurement does not record its original evaluation device; "
                       "CPU/GPU equivalence cannot be assumed during re-evaluation")
        return out
    if recorded_device.startswith("cuda") and not out["evaluation_gpu_uuid"]:
        out["why"] = ("the measurement records a CUDA ordinal but no physical GPU UUID; "
                       "the same device cannot be selected safely")
        return out
    if (recorded_device.startswith("cuda") and
            "CUDA_VISIBLE_DEVICES" not in out["evaluation_compute_environment"]):
        out["why"] = ("the measurement does not record the environment needed to pin its "
                       "original GPU")
        return out

    out["status"] = "ready"
    policy_description = (f"source tree {out['policy']['expected']}" if source_controller else
                          f"frozen policy {path.name}")
    out["would_run"] = (
        f"rebuild `stage_argv_{target}` from the recipe at {root}, run it against the "
        f"{policy_description} at the settings above, with the output directory pointed "
        f"somewhere new, then read the metric from what this run's evaluator prints")
    return out


def public(decision: dict[str, Any]) -> dict[str, Any]:
    """A plan without the private keys, which are inputs to `reevaluate` rather than findings.

    Kept as a function rather than a `plan(..., public=True)` flag so the two audiences cannot
    drift: the report a reader sees and the value the runner consumes are the same dict, read
    two ways.
    """
    return {key: value for key, value in decision.items() if not key.startswith("_")}


def reevaluate(run_root: Path, label: str, *, output: Path, timeout: int | None = None
               ) -> dict[str, Any]:
    """Rebuild, run and compare. See the module docstring for what the three outcomes mean.

    The output directory must not exist: a re-evaluation that could read the run's own
    artifacts would report that the number is still on disk, which is not the question.
    """
    run_root = Path(run_root).resolve()
    output = Path(output).expanduser().resolve()
    decision = plan(run_root, label)
    result: dict[str, Any] = {
        "run_root": str(run_root), "label": label,
        "status": "refused", "claimed": decision.get("claimed"),
        "measured": None, "why": decision.get("why", ""),
        "settings": decision.get("settings") or {},
        "settings_frozen": bool(decision.get("settings_frozen")),
        "evaluation_device": decision.get("evaluation_device"),
        "policy_matches": (decision.get("policy") or {}).get("matches", False),
        "limitations": []}
    if decision["status"] != "ready":
        return result
    if output.exists():
        result["why"] = "the re-evaluation output directory already exists"
        return result

    sources, parameters = decision["_sources"], decision["_parameters"]
    answer = decision["_answer"]
    interpreter = decision["interpreter"]
    if not interpreter or not Path(interpreter).is_file():
        result["status"] = "could_not_reevaluate"
        result["why"] = (f"the interpreter the run used is gone ({interpreter or 'unrecorded'}"
                         f"), so the evaluator cannot be started; this is not evidence about "
                         f"the number")
        return result

    try:
        from .adapter_protocol import OptimizationSpace
        from .compute_decision import ComputeDecision
        from .derived_research import DerivedResearch

        output.mkdir(parents=True)
        repo = Path(decision["_repo"])
        if decision.get("_workspace"):
            from .workspace_snapshot import apply_state, create, verify_state

            workspace = decision["_workspace"]
            fresh = output / "checkout"
            copied = create(Path(workspace["source"]), fresh,
                            max_bytes=max(1, int(workspace["bytes"])),
                            resources=workspace.get("resource_bindings", []),
                            tracked_only=workspace.get("copy_mode") == "tracked_worktree")
            if copied["source_tree_fingerprint"] != workspace["source_tree_fingerprint"]:
                raise ValueError("the original source base changed since the run snapshot")
            source_state = decision.get("_source_state") or {}
            if source_state:
                apply_state(fresh, source_state, Snapshots(run_root / "snapshots"))
            else:
                Snapshots(run_root / "snapshots").restore(
                    decision["_source_snapshot"], repo=fresh)
            if source_state and not verify_state(
                    fresh, copied, source_state, Snapshots(run_root / "snapshots")):
                raise ValueError("the reconstructed source tree differs from the measured identity")
            original = str(repo)
            def rebase(value: Any) -> Any:
                if isinstance(value, str):
                    return value.replace(original, str(fresh))
                if isinstance(value, list):
                    return [rebase(one) for one in value]
                if isinstance(value, dict):
                    return {key: rebase(one) for key, one in value.items()}
                return value
            repo = fresh
            answer, sources, parameters = rebase(answer), rebase(sources), rebase(parameters)
            result["source_isolation"] = "fresh_copy_from_verified_base_and_snapshot"
        else:
            result["source_isolation"] = "mutable_checkout"
        backend = DeclarativeBackend(repo=repo, answer=answer, sources=sources,
                                     parameters=parameters)
        device = str(decision["evaluation_device"])
        environment = dict(decision.get("evaluation_compute_environment") or {})
        if device == "cpu":
            environment.setdefault("CUDA_VISIBLE_DEVICES", "")
            compute = ComputeDecision(device="cpu", device_index=0,
                                      environment=environment,
                                      why="reusing the measurement's recorded CPU path")
        else:
            if "CUDA_VISIBLE_DEVICES" not in environment:
                result["status"] = "could_not_reevaluate"
                result["why"] = ("the measurement does not record the environment needed "
                                 "to pin its original GPU")
                return result
            compute = ComputeDecision(
                device=device, device_index=int(device.split(":", 1)[1])
                if ":" in device else 0,
                environment=environment,
                why="reusing the measurement's recorded GPU path",
                evidence={"selected_device": {
                    "uuid": decision.get("evaluation_gpu_uuid")}})
        research = DerivedResearch(
            repo=repo, output=output, backend=backend, interpreter=Path(interpreter),
            space=OptimizationSpace(), client=None, stages=sources,
            run_id=f"reevaluation-{label}", declaration=dict(decision.get("declaration") or {}),
            compute=compute)
        # The evaluation alone. The policy is the archived one, so the training half of a
        # measurement is not repeated -- and could not be, since that would score a *new*
        # policy and answer a different question.
        ran = research.run_stage(decision["target"], settings=decision["settings"],
                                 timeout=timeout,
                                 checkpoint=(decision["policy"].get("path") or ""),
                                 experiment_dir=(decision["policy"].get("path") or ""),
                                 previous_artifact=(decision["policy"].get("path") or ""),
                                 _protocol_guard=True)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        result["status"] = "could_not_reevaluate"
        result["why"] = f"the evaluator could not be started: {type(exc).__name__}: {exc}"
        return result

    artifact_receipt = ran.get("artifact") or {}
    if (not ran.get("ran") or ran.get("returncode") != 0 or
            ran.get("status") != "completed" or
            (artifact_receipt.get("checked") and not artifact_receipt.get("matched"))):
        result["status"] = "could_not_reevaluate"
        result["why"] = (f"the evaluator did not complete (returncode "
                         f"{ran.get('returncode')}), so this run establishes nothing about "
                         f"the number either way")
        result["said"] = str(ran.get("said", ""))[-1200:]
        return result

    spec = MetricSpec.from_declaration(
        {"task_contract": {"primary_metric": decision["metric"]}})
    reading = spec.read(said=ran.get("said", ""),
                        artifact=research._recorded_artifact(ran))
    result["measured"] = reading.get("value")
    if reading.get("value") is None:
        result["status"] = "could_not_reevaluate"
        result["why"] = ("the evaluator exited zero and printed no value for the declared "
                         "metric, so there is nothing to compare")
        result["said"] = str(ran.get("said", ""))[-1200:]
        return result

    same = abs(float(reading["value"]) - float(decision["claimed"])) < 1e-9
    result["status"] = "reproduced" if same else "not_reproduced"
    result["why"] = (
        f"the evaluator was run again in a fresh directory against the frozen policy and "
        f"printed {reading['value']}, the same value the measurement recorded"
        if same else
        f"the evaluator was run again in a fresh directory against the frozen policy and "
        f"printed {reading['value']} where the measurement recorded {decision['claimed']}")
    result["said"] = str(ran.get("said", ""))[-1200:]
    result["attempt_id"] = ran.get("attempt_id")
    result["limitations"] = [
        "one re-run at one settings set. Two runs of the same evaluator can differ, so "
        "'reproduced' means this number is reachable again and not that it is the number the "
        "benchmark's test set would give.",
        "the recipe and the settings were read from this run's own records, so a mistake "
        "already baked into them is repeated here rather than caught.",
    ]
    if result.get("source_isolation") == "mutable_checkout":
        result["limitations"].append(
            "the source checkout was reused and may have changed since scoring")
    else:
        result["limitations"].append(
            "the source was reconstructed from a verified base and per-candidate overlay, "
            "but the interpreter, packages and external assets remain mutable")
    if not decision.get("settings_frozen"):
        result["limitations"].append(
            "no frozen protocol was present, so this re-ran the settings the measurement "
            "recorded rather than settings fixed before it ran.")
    for gap in decision.get("gaps") or []:
        result["limitations"].append(gap)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autosim recheck",
        description="Re-run one scored evaluation from a run's records, in a fresh directory")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true",
                        help="say what would be run and what is missing, and run nothing")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.plan_only:
        decision = public(plan(args.run_root, args.label))
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0 if decision["status"] == "ready" else 1
    verdict = reevaluate(args.run_root, args.label, output=args.output, timeout=args.timeout)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict["status"] == "reproduced" else 1


if __name__ == "__main__":
    raise SystemExit(main())
