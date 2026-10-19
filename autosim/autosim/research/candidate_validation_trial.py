"""Main-selected validation of source-plausible, not yet observed connections.

An uncertain relevance review is not a scientific veto. It is also not approval:
only an explicit, source-bound Main decision may spend a local stage window on
testing it. All normal native, policy, consumption and scoring checks still run.
"""

from contextlib import contextmanager
import math
import re

from .common import immutable_json, object_digest


def validate_request(request, *, decision_id, idea_label, training_settings,
                     job_id="", reuse_policy=None, evaluation_settings=None):
    if (not isinstance(request, dict) or set(request) != {
            "reason", "postcondition", "stage_window_seconds"}):
        raise ValueError("execution_uncertainty requires reason, postcondition and stage_window_seconds")
    for key in ("reason", "postcondition"):
        if (not isinstance(request[key], str) or not request[key].strip()
                or len(request[key]) > 1500):
            raise ValueError(f"execution_uncertainty.{key} must be nonempty prose, at most 1500 characters")
    window = request["stage_window_seconds"]
    try:
        valid_window = (not isinstance(window, bool) and isinstance(window, (int, float))
                        and math.isfinite(window) and window > 0)
    except OverflowError:
        valid_window = False
    if not valid_window:
        raise ValueError("stage_window_seconds must be finite and positive")
    if not isinstance(decision_id, str) or not re.fullmatch(r"[a-f0-9]{16}", decision_id):
        raise ValueError("bounded validation needs a recorded Main decision")
    if (not idea_label or not isinstance(training_settings, dict) or not training_settings
            or job_id or reuse_policy is not None or evaluation_settings is not None):
        raise ValueError("bounded validation needs an exact audited candidate and explicit training_settings, without job adoption, policy reuse or protocol changes")
    return dict(request)


def authorize(research, *, output, repo, stages, idea_label, settings,
              decision_id, request, admission, training_settings, additional_evidence_refs=(),
              research_decision_context=None):
    """Revalidate the exact cached review; never change or infer its verdict."""
    from .candidate_execution_review import cached, packet_for
    request = validate_request(request, decision_id=decision_id, idea_label=idea_label,
                               training_settings=training_settings)
    idea = research.library.get(idea_label)
    if idea is None or idea.status != "cleared":
        raise ValueError("bounded validation requires a currently cleared audited idea")
    if admission is not None and (
            admission.get("failure_category") != "candidate_execution_relevance"
            or (admission.get("candidate_execution_review") or {}).get("verdict") != "uncertain"):
        raise ValueError("bounded validation cannot bypass inapplicability or an unavailable review")
    packet = packet_for(output, repo, stages, idea, selected_training_settings=settings,
                        additional_evidence_refs=additional_evidence_refs,
                        research_root=getattr(research,'run_root',None),
                        **({'research_decision_context': research_decision_context}
                           if research_decision_context is not None else {}))
    reviewed = cached(output, packet)
    if (not reviewed or reviewed.get("status") != "reviewed"
            or reviewed["answer"]["verdict"] not in {"applicable", "uncertain"}):
        raise ValueError("a current, authenticated applicable or uncertain review is required")
    if admission is not None and reviewed["answer"] != admission["candidate_execution_review"]:
        raise ValueError("candidate review changed since admission")
    from .training_work import consumes_steps
    steps = training_settings.get("steps")
    if consumes_steps(research.sources.get("train", "")) and (
            isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0):
        raise ValueError("the native trainer consumes steps; choose positive training work explicitly")
    row = {"schema_version": 1, "decision_id": decision_id,
           "idea_label": idea_label, "requested_training_settings": settings,
           "explicit_training_work": training_settings,
           "review_ref": reviewed["record_ref"],
           "review_material_sha256": reviewed["material_sha256"],
           "review_record_sha256": reviewed["record_sha256"],
           "review_verdict": reviewed["answer"]["verdict"],
           "request": request,
           "authority": "Main-selected bounded validation, not reviewer approval, native success or a score"}
    identity = object_digest(row)
    directory = output / "candidate_execution_reviews" / "validation_trials"
    path = directory / f"{identity}.json"
    if any(p.is_symlink() for p in (directory.parent, directory, path)):
        raise ValueError("unsafe bounded-validation record destination")
    immutable_json(path, row)
    # Reuse the existing paged evidence namespace. A relative filename alone
    # would strand Main/Fix outside their read-only tool boundary.
    from .evidence_store import capture_attempt_evidence
    evidence = capture_attempt_evidence(output, attempt_id=identity[:32], log=path,
        receipt_ref=path.relative_to(output).as_posix(), status='authorization_only',
        returncode=None, termination_reason='main_selected_validation')
    return {"authorization_ref": path.relative_to(output).as_posix(),
            "authorization_sha256": identity, "review_ref": row["review_ref"],
            "evidence_id":evidence['evidence_id'], "evidence_ref":evidence['evidence_ref'],
            "review_verdict": row["review_verdict"], "authority": row["authority"]}


@contextmanager
def stage_window(research, request):
    """Local per-stage window, never a new whole-run budget or a reset."""
    if request is None:
        yield
        return
    previous = research.STAGE_TIMEOUT_SECONDS
    window = min(previous, request["stage_window_seconds"])
    if research.budget is not None:
        window = min(window, research.budget.remaining())
    if not math.isfinite(window) or window <= 0:
        raise ValueError("no whole-run wall time remains for bounded validation")
    research.STAGE_TIMEOUT_SECONDS = window
    try:
        yield
    finally:
        research.STAGE_TIMEOUT_SECONDS = previous
