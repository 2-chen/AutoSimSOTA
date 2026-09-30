"""Agent-declared fidelity trials; advisory ranking never changes formal best scores.

No repository names or guessed epoch switches. Code/algo ideas keep using the full
audited transaction; this isolated screening lane handles declared parameter ideas.
"""
from __future__ import annotations

import copy
import math
import re
import time
import uuid

from .common import atomic_json, immutable_json, now, object_digest, read_json
from .decisions import Decisions
from .native_jobs import active_jobs, source_identity
from .receipt_verifier import verify_measurement
from .scheduling import locked, note, policy


def _path(research, study_id):
    if not isinstance(study_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", study_id):
        raise ValueError("unsafe screening study ID")
    root = research.run_root / "screening" / study_id
    if root.is_symlink():
        raise ValueError("screening study is a symlink")
    return root


def configure(research, document: dict, base_settings: dict) -> dict:
    if not policy(research.output):
        raise ValueError("screening requires the opt-in scheduler")
    if not isinstance(document, dict) or set(document) != {
            "id", "budget_axis", "rungs", "eta", "min_peers", "evidence", "reason"}:
        raise ValueError("study needs id, budget_axis, rungs, eta, min_peers, evidence, reason")
    root = _path(research, document["id"])
    axis = research.space.axis("training", document["budget_axis"])
    if axis is None or axis.kind not in {"integer", "number"} or axis.group:
        raise ValueError("fidelity must bind a declared numeric native training axis")
    rungs = document["rungs"]
    if (not isinstance(rungs, list) or not 2 <= len(rungs) <= 8 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or v <= 0 or not axis.accepts(v) for v in rungs) or
            any(b <= a for a, b in zip(rungs, rungs[1:]))):
        raise ValueError("rungs must be strictly increasing positive declared axis values")
    if (type(document["eta"]) is not int or not 2 <= document["eta"] <= 8 or
            type(document["min_peers"]) is not int or not 2 <= document["min_peers"] <= 32):
        raise ValueError("invalid reduction or grace peer count")
    if (not isinstance(document["evidence"], list) or not document["evidence"] or
            any(not isinstance(ref, str) or not ref.strip() for ref in document["evidence"]) or
            not isinstance(document["reason"], str) or not document["reason"].strip()):
        raise ValueError("Agent must explain native budget control and fidelity relevance with evidence")
    target = str((research.execution_graph or {}).get("score_target") or "evaluate")
    violation = research._comparison_protocol_violation(base_settings, target=target, freeze=True)
    if violation:
        raise ValueError(violation)
    protocol = read_json(research.run_root / "comparison_protocol.json")
    if axis.name in protocol["settings"] or axis.name in protocol["protocol_keys"]:
        raise ValueError("screening may not vary an evaluation protocol axis")
    if base_settings.get(axis.name, axis.default) != rungs[-1]:
        raise ValueError("last rung must equal the formal baseline training budget")
    identity = source_identity(research.output, research.repo)
    if not identity:
        raise ValueError("screening requires an inventoried isolated source checkout")
    frozen = {**document, "schema_version": 1, "base_settings": dict(base_settings),
        "source_identity": identity, "metric": research.metric_spec.as_dict(), "protocol": protocol,
        "authority": "screening_only_not_formal_selection", "created_at": now()}
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / "study.lock"):
        path = root / "study.json"
        if path.is_file():
            old = read_json(path)
            if {k: v for k, v in old.items() if k != "created_at"} != {
                    k: v for k, v in frozen.items() if k != "created_at"}:
                raise ValueError("screening study is frozen; use a new ID")
            return old
        immutable_json(path, frozen)
    return frozen


def trial(research, *, study_id: str, idea_label: str, rung: int,
          window_seconds: float, reason: str) -> dict:
    if active_jobs(research.output):
        raise ValueError("screening cannot change experiment state during native jobs")
    if (isinstance(window_seconds, bool) or not isinstance(window_seconds, (int, float)) or
            not math.isfinite(window_seconds) or window_seconds <= 0 or
            research.budget is None or window_seconds > research.budget.remaining()):
        raise ValueError("screening window exceeds remaining hard wall budget")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("screening trial needs an evidence-backed reason")
    root = _path(research, study_id)
    study = read_json(root / "study.json")
    if type(rung) is not int or not 0 <= rung < len(study["rungs"]):
        raise ValueError("invalid fidelity rung")
    idea = research.library.get(idea_label)
    if idea is None or idea.status not in {"cleared", "tried", "worked"} or idea.granularity != "param":
        raise ValueError("fidelity trials need an audited parameter idea; code/algo use full transactions")
    from .decision import validate_proposal
    validate_proposal(research._envelope_for(idea, evidence_id="screening"),
                      space=research.space, evidence_id="screening")
    if any(k in research.space.sections() for k in idea.change):
        raise ValueError("screening requires a flat native training settings map")
    axis = study["budget_axis"]
    if axis in idea.change:
        raise ValueError("a candidate may not redefine the shared fidelity budget")
    identity = source_identity(research.output, research.repo)
    if identity != study["source_identity"] or research.metric_spec.as_dict() != study["metric"]:
        raise ValueError("source/metric changed; create a new screening study after repair")
    settings = {**study["base_settings"], **idea.change, axis: study["rungs"][rung]}
    target = str((research.execution_graph or {}).get("score_target") or "evaluate")
    violation = research._comparison_protocol_violation(settings, target=target)
    if violation:
        raise ValueError(violation)
    trial_id = uuid.uuid4().hex
    destination = root / "trials" / trial_id
    destination.mkdir(parents=True)
    label = "screen_" + trial_id
    worker = copy.copy(research)
    worker.run_root = destination
    worker.events = []
    worker.active_research_action = {}
    worker.decisions = Decisions(destination)
    worker.job_control_path = destination / "control.json"
    atomic_json(worker.job_control_path, {"revision": 0, "cancelled": False,
        "deadline_epoch": time.time() + window_seconds})
    immutable_json(destination / "comparison_protocol.json", study["protocol"])
    request = {"trial_id": trial_id, "idea_label": idea_label, "rung": rung,
        "settings": settings, "reason": reason, "source_identity": identity,
        "study_sha256": object_digest(study), "created_at": now(), "status": "running"}
    atomic_json(destination / "trial.json", request)
    started = time.monotonic()
    try:
        measured = worker._measure(settings=settings, label=label)
        verified = verify_measurement(destination, label)
        stable = source_identity(research.output, research.repo) == identity
        valid = bool(measured.get("ok") and verified["status"] == "consistent" and stable)
        result = {**request, "status": "completed" if valid else "failed",
            "metric_value": measured.get("metric_value") if valid else None,
            "measurement_ref": str((destination / "measurements" / (label + ".json")).relative_to(research.output)),
            "verification": verified, "source_stable": stable,
            "why": measured.get("why") or "", "finished_at": now(),
            "authority": "screening_only_not_formal_selection"}
    except Exception as exc:
        result = {**request, "status": "failed", "metric_value": None,
            "why": f"{type(exc).__name__}: {exc}"[:600], "finished_at": now()}
    atomic_json(destination / "trial.json", result)
    note(research.output, kind="screening_trial", identity=trial_id,
         seconds=time.monotonic()-started, status=result["status"], rung=rung)
    return result


def inspect(research, study_id: str) -> dict:
    root = _path(research, study_id)
    study = read_json(root / "study.json")
    rows = [read_json(p) for p in sorted((root / "trials").glob("*/trial.json"))]
    recommendations = []
    for rung in range(len(study["rungs"])-1):
        peers = {r["idea_label"]: r for r in rows if r.get("status") == "completed" and r["rung"] == rung}
        if len(peers) < study["min_peers"]:
            continue
        ordered = sorted(peers.values(), key=lambda r: r["metric_value"],
                         reverse=study["metric"]["direction"] == "maximize")
        for row in ordered[:max(1, math.ceil(len(ordered)/study["eta"]))]:
            recommendations.append({"idea_label": row["idea_label"], "from_rung": rung,
                                    "suggested_rung": rung+1})
    return {"study": study, "trials": rows[-32:], "promotion_recommendations": recommendations,
        "warning": "Advisory only. Compare same rung; full research and independent confirmation remain required."}
