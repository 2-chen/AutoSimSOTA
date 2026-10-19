"""Bounded Data-Agent hypotheses, not a second scheduler or collection authority.

Fields describe the proposed connection from failures to native controls. Source,
expert reliability, actual coverage and consumption must still be checked by the
existing native executor and independent reviewers. No code is evaluated here.
"""
from __future__ import annotations

import json
import math
from pathlib import PurePosixPath


def _text(value, field, limit=1000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{field} requires bounded nonempty text")
    return value.strip()


def _texts(value, field, *, limit=12):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{field} requires a bounded list")
    return [_text(item, field, 600) for item in value]


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} requires exactly {','.join(fields)}")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > 10000:
        raise ValueError(f"{name} exceeds 10000 bytes")
    return value


def failure_case(value):
    _object(value, ("development_ref", "evidence_refs", "observations", "hypotheses",
                    "unknowns", "diagnostic_probe"), "failure_case")
    return {"development_ref": _text(value["development_ref"], "development_ref", 300),
            **{key: _texts(value[key], key) for key in
               ("evidence_refs", "observations", "hypotheses", "unknowns")},
            "diagnostic_probe": _text(value["diagnostic_probe"], "diagnostic_probe"),
            "authority": "development_observation_and_hypothesis_not_causal_verdict"}


def acquisition_proposal(value):
    _object(value, ("target", "sampling_unit", "supervision", "native_binding",
                    "target_predicate", "budget", "data_context", "comparison",
                    "falsification", "fallback", "evidence_refs"), "acquisition_proposal")
    unit = value["sampling_unit"]
    if unit not in {"full_episode", "expert_state_label", "recovery_segment", "selected_episode"}:
        raise ValueError("unsupported acquisition sampling unit")
    supervision = _object(value["supervision"], ("kind", "domain", "takeover", "evidence_refs"), "supervision")
    if supervision["kind"] not in {"reset_expert", "state_expert", "demo_adaptation", "policy_success", "human_only", "unknown"}:
        raise ValueError("unsupported supervision kind")
    if supervision["takeover"] not in {"supported", "unknown", "unsupported", "not_required"}:
        raise ValueError("unsupported takeover claim")
    binding = _object(value["native_binding"], ("recipe_ref", "controls", "source_refs"), "native_binding")
    if not isinstance(binding["controls"], list) or len(binding["controls"]) > 12:
        raise ValueError("controls require a bounded list")
    controls = []
    for control in binding["controls"]:
        _object(control, ("path", "value", "unit", "legal_domain", "consumer_ref"), "control")
        path = _text(control["path"], "control.path", 300)
        if path.startswith(("/", "\\")) or ".." in PurePosixPath(path).parts or "\\" in path:
            raise ValueError("control path must be a scoped native field, not a host path")
        # Preserve arbitrary native parameter values, but never executable objects.
        if len(json.dumps(control["value"], allow_nan=False).encode()) > 2000:
            raise ValueError("control value exceeds bounded JSON")
        controls.append({**control, "path": path,
            **{k: _text(control[k], k, 600) for k in ("unit", "legal_domain", "consumer_ref")}})
    budget = _object(value["budget"], ("axis", "limit", "yield_estimate", "downstream_reserve"), "budget")
    if budget["axis"] not in {"acquisition_seconds", "attempts", "accepted_episodes", "total_compute"}:
        raise ValueError("budget axis is not a declared comparison unit")
    bound = budget["limit"]
    if budget['axis'] == 'total_compute':
        if (not isinstance(bound, dict) or not bound or set(bound) -
                {'gpu_seconds', 'cpu_seconds', 'wall_seconds', 'model_usd'} or
                any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in bound.values())):
            raise ValueError('total_compute requires a finite explicit resource package, not unlike units added together')
    elif (not isinstance(bound, (int, float)) or isinstance(bound, bool) or
            not math.isfinite(bound) or bound <= 0):
        raise ValueError("acquisition budget limit must be finite and positive")
    if budget["axis"] in {"attempts", "accepted_episodes"} and type(bound) is not int:
        raise ValueError("episode/attempt target must be an integer")
    if budget["axis"] == "accepted_episodes" and unit in {"expert_state_label", "recovery_segment"}:
        raise ValueError("state labels and suffixes are not complete accepted episodes")
    issues = []
    if not value["evidence_refs"]:
        issues.append("failure evidence is missing")
    if not supervision["evidence_refs"] or supervision["kind"] == "unknown":
        issues.append("correct-action supply needs evidence")
    if supervision["kind"] == "human_only":
        issues.append("human supervision is not autonomous supply")
    if unit in {"expert_state_label", "recovery_segment"} and (
            supervision["kind"] == "reset_expert" or supervision["takeover"] != "supported"):
        issues.append("reset-only or unverified takeover cannot supervise the proposed state")
    if not binding["source_refs"] or not binding["controls"]:
        issues.append("native target control needs source routing; consider a selection-only proposal")
    from .data_quality import predicate
    target_predicate = (predicate(value["target_predicate"]) if isinstance(value["target_predicate"], dict)
                        else _text(value["target_predicate"], "target_predicate"))
    return {"target": _text(value["target"], "target"), "sampling_unit": unit,
        "supervision": {**supervision, "domain": _text(supervision["domain"], "domain"),
                        "evidence_refs": _texts(supervision["evidence_refs"], "supervision.evidence_refs")},
        "native_binding": {"recipe_ref": _text(binding["recipe_ref"], "recipe_ref", 500),
            "controls": controls, "source_refs": _texts(binding["source_refs"], "source_refs")},
        "target_predicate": target_predicate,
        "budget": {**budget, "yield_estimate": _text(budget["yield_estimate"], "yield_estimate"),
                   "downstream_reserve": _text(budget["downstream_reserve"], "downstream_reserve")},
        "data_context": _text(value["data_context"], "data_context"),
        "comparison": _text(value["comparison"], "comparison"),
        "falsification": _texts(value["falsification"], "falsification"),
        "fallback": _text(value["fallback"], "fallback"),
        "evidence_refs": _texts(value["evidence_refs"], "evidence_refs"),
        "status": "needs_evidence" if issues else "needs_native_probe",
        "unmet_prerequisites": issues,
        "authority": "agent_acquisition_hypothesis_not_job_or_quality_authorization"}


def collection_inputs(value, source):
    """Scoped native controls, separate from evaluation and optimizer settings."""
    from .data_quality import controls
    from .training_work import consumes_input
    if (not isinstance(value, dict) or not value or
            set(value) - {"episodes", "seed", "controls", "control_source_refs"}):
        raise ValueError("collection_settings accepts episodes, seed, controls and control_source_refs")
    result = {}
    for key in ("episodes", "seed"):
        if key in value:
            if type(value[key]) is not int or value[key] < (1 if key == "episodes" else 0):
                raise ValueError("collection work/seed must be an explicit legal integer")
            if not consumes_input(source, key):
                raise ValueError(f"verified collect builder does not consume the requested {key} input")
            result[key] = value[key]
    if "controls" in value:
        result["collection_controls"] = controls(value["controls"])
        if not consumes_input(source, "collection_controls"):
            raise ValueError("collector builder does not consume collection_controls; derive the native route or use post-hoc selection")
        refs = value.get("control_source_refs")
        if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
            raise ValueError("native controls require current caller-to-consumer source evidence")
    elif "control_source_refs" in value:
        raise ValueError("control_source_refs without controls is not a collection contract")
    return result


def review_collection_controls(*, output, repo, client, source, declaration, settings):
    """Independent preflight checks legality/routing, NOT actual control effect."""
    import hashlib
    from .agent_client import role_scope
    from .baseline_reference import source_evidence
    from .common import object_digest, now
    from .execution_derive import _object
    from .source_citations import project, resolve, INSTRUCTION, ReviewCitationError
    values = collection_inputs(settings, source)["collection_controls"]
    refs = source_evidence(repo, settings["control_source_refs"], output=output)
    review_scope = 'explicit_controls_and_required_dependencies_not_entire_declared_space'
    packet, shown = project({"collection_controls": values,
        "requested_control_keys": sorted(values), "review_scope": review_scope,
        "collector_builder": source,
        "declaration": declaration, "source_refs": refs}, refs, local_roots=(repo, output))
    with role_scope(client, "supervisor"):
        answer, _ = client.chat_with_metadata(
            "Review only explicitly requested DEVELOPMENT TRAINING collection_controls and their "
            "necessary validity dependencies, not every axis/default in the declaration. "
            "The requested_control_keys identify the intervention. Other declared controls and "
            "unchanged native defaults are context, not additional proposed changes; do not deny "
            "solely because unrequested defaults lack citations. If such a default is necessary "
            "to validity, name the exact dependency and why it matters for this request. "
            "Require exact source support for the legal training domain, requested values/units/ranges, "
            "reset or state-expert label supply "
            "and the real caller-to-consumer route. CLI acceptance or a same-named field is insufficient. "
            "A first bounded source-reviewed test does not require a prior observed control effect. "
            "Do not claim observed effect before a native probe; do not loosen protected evaluation. "
            "Return JSON booleans allowed_training_controls, routing_supported, label_supply_supported, "
            "why and citations." + INSTRUCTION,
            json.dumps(packet, ensure_ascii=False), max_tokens=1800, read_only=True,
            include_research_context=False)
    review = _object(answer)
    if not all(review.get(key) is True for key in
               ("allowed_training_controls", "routing_supported", "label_supply_supported")):
        raise ValueError("native collection control review did not pass: " + str(review.get("why", "missing checks"))[:500])
    try:
        checked = resolve(refs, shown, review.get("citations"))
    except ReviewCitationError as exc:
        # A delivery defect is not a fresh scientific question for Main. Repair
        # only the citation selection, once, under the original independent
        # verdict. Never silently discard an incorrect model-supplied quote.
        initial_review = dict(review)
        with role_scope(client, "supervisor"):
            fixed, _ = client.chat_with_metadata(
                'Repair citation FORMAT only for your prior independent collection review. '
                'Return only JSON {"citations":[{"source_id":"review-source-N"}]}. '
                'Select only actually supplied IDs; do not change scientific checks or why, '
                'invent evidence, quote substrings, or claim execution. ' + INSTRUCTION,
                json.dumps({'prior_review': initial_review, 'review_sources': shown,
                            'citation_error': str(exc)}, ensure_ascii=False),
                max_tokens=600, read_only=True, include_research_context=False,
                **({'decision_only': True, 'thinking': False, 'auto_skills': False,
                    'resume_session': False}
                   if getattr(client, 'supports_bounded_generation', False) else {}))
        try:
            repaired = _object(fixed)
        except (TypeError, ValueError) as parse_error:
            raise ReviewCitationError('collection citation repair must deliver a JSON object') from parse_error
        if set(repaired) != {'citations'}:
            raise ReviewCitationError('collection citation repair may return only citations')
        checked = resolve(refs, shown, repaired.get('citations'))
        review['citation_format_repair'] = {
            'initial_review': initial_review, 'reply': repaired,
            'authority': 'citation_mapping_only_not_a_new_scientific_verdict'}
    review['citations'] = checked
    if source_evidence(repo, refs, output=output) != refs:
        raise ValueError("native control sources changed during review")
    return {"schema_version": 1, "controls": values, "source_refs": refs, "review": review,
        "review_scope": review_scope,
        "builder_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "controls_sha256": object_digest(values), "at": now(), "effect_status": "needs_native_observation"}


def verify_control_contract(*, output, repo, source, settings, contract):
    import hashlib
    from .common import object_digest
    from .baseline_reference import source_evidence
    from .data_quality import controls
    if "collection_controls" not in settings:
        if contract is not None:
            raise ValueError("orphan collection control contract")
        return
    values = controls(settings["collection_controls"])
    if (not isinstance(contract, dict) or contract.get("controls") != values
            or contract.get("controls_sha256") != object_digest(values)
            or contract.get("builder_sha256") != hashlib.sha256(source.encode()).hexdigest()
            or not all((contract.get("review") or {}).get(key) is True for key in
                       ("allowed_training_controls", "routing_supported", "label_supply_supported"))
            or not contract.get("source_refs")
            or source_evidence(repo, contract["source_refs"], output=output) != contract["source_refs"]):
        raise ValueError("missing or changed source-reviewed native collection controls")
