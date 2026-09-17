"""One decision layer for every benchmark.

The system previously carried a decision module per benchmark: each declared its
own proposal schema, wrote its own prompt, and validated its own fields. The two
copies drifted, and a third benchmark would have meant a third copy. What they
actually disagreed about was almost entirely *what can be varied* -- which is a
fact the adapter knows and the decision layer was guessing.

This module keeps no benchmark's vocabulary. It is handed an OptimizationSpace,
it builds the request from that space's axes, and it validates a proposal against
them. Ask it about `epochs` on a benchmark with no epoch axis and it will tell you
the field is unregistered, which is the correct answer and one it reached without
being told which benchmark it was looking at.

Nothing here enforces a *value* judgment -- that a particular setting is wise, or
that a hypothesis is plausible. Axes carry the range the benchmark implements, and
that is the only ceiling.
"""

from __future__ import annotations

import json
from typing import Any

from .adapter_protocol import Axis, OptimizationSpace

#: Fields every proposal carries regardless of benchmark. `evidence_id` binds a
#: proposal to the evidence it was made against, so a stale proposal cannot be
#: replayed against a different round's numbers.
_BASE_FIELDS = ("decision", "proposal_id", "evidence_id", "hypothesis", "expected_validation")

SYSTEM = (
    "You are the research decision module for a simulator manipulation benchmark. Return "
    "exactly one JSON object matching the supplied skeleton. The skeleton's values are a "
    "workable starting point, not the whole space: each field is described with the values "
    "or range this benchmark implements, and you may choose anywhere inside that. You may "
    "not change observations, the success judge, evaluation seeds, the evaluator, or the "
    "task's own definition -- those are what makes two rounds comparable. Treat evidence "
    "with stated limitations as evidence, not as proof of cause. Choose decision=stop when "
    "no further experiment is justified by the evidence you have; say why in the hypothesis."
)


def _axis_line(name: str, axis: Axis) -> str:
    if axis.kind == "choice":
        return f"{name}: one of {list(axis.values)} -- {axis.description}"
    return f"{name}: {axis.kind} in [{axis.low}, {axis.high}] -- {axis.description}"


def describe_space(space: OptimizationSpace) -> dict[str, list[str]]:
    """The space as prose lines, which is what a model reads better than nested dicts."""
    return {
        "collection": [_axis_line(a.name, a) for a in space.collection],
        "training": [_axis_line(a.name, a) for a in space.training],
    }


def skeleton(space: OptimizationSpace, *, evidence_id: str,
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The starting object shown to the controller: declared defaults only."""
    body = {
        "decision": "experiment",
        "proposal_id": "short_identifier",
        "evidence_id": evidence_id,
        "hypothesis": "a testable statement about what you expect and why",
        "collection": {a.name: a.default for a in space.collection},
        "training": {a.name: a.default for a in space.training},
        "expected_validation": "what the paired native evaluation should show",
    }
    if extra:
        body.update(extra)
    return body


def build_request(space: OptimizationSpace, *, evidence: dict[str, Any], evidence_id: str,
                  method_library: dict[str, Any] | None = None,
                  benchmark_notes: dict[str, Any] | None = None,
                  extra_skeleton: dict[str, Any] | None = None) -> tuple[str, str]:
    """The (system, user) pair for one proposal request."""
    payload: dict[str, Any] = {
        "evidence": evidence,
        "required_skeleton": skeleton(space, evidence_id=evidence_id, extra=extra_skeleton),
        "what_you_may_vary": describe_space(space),
    }
    if benchmark_notes:
        payload["benchmark_notes"] = benchmark_notes
    if method_library:
        # Reference, not policy: nothing validates against it and the controller may
        # contradict any entry. Kept out of the schema so editing it cannot change the
        # identity of a round's evidence.
        payload["method_library"] = method_library
    return SYSTEM, json.dumps(payload, ensure_ascii=False, sort_keys=True)


def sample_axes(space: OptimizationSpace, rng: Any,
                groups: tuple[str, ...] = ("collection", "training")) -> dict[str, Any]:
    """Draw a legal setting from the declared space.

    The matched-budget control arms have to sample from the *same* space the
    controller is offered, or a control can drift onto values the treatment could
    not have chosen. Deriving both from one declaration is what keeps that true
    when someone edits the declaration.
    """
    drawn: dict[str, Any] = {}
    for group in groups:
        values: dict[str, Any] = {}
        for axis in space.axes(group):
            if axis.kind == "choice":
                values[axis.name] = rng.choice(list(axis.values))
            elif axis.kind == "integer":
                values[axis.name] = rng.randint(int(axis.low), int(axis.high))
            else:
                values[axis.name] = rng.uniform(float(axis.low), float(axis.high))
        drawn[group] = values
    return drawn


def validate_proposal(raw: dict[str, Any], *, space: OptimizationSpace,
                      evidence_id: str, required_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate against the declared space. Raises ValueError with a usable message.

    The messages name the axis and what it accepts, because a rejected proposal is
    the controller's only feedback about why, and "schema mismatch" teaches it nothing.
    """
    if not isinstance(raw, dict):
        raise ValueError("proposal must be one JSON object")
    expected = set(_BASE_FIELDS) | {"collection", "training"} | set(required_context or {})
    missing, extra = sorted(expected - set(raw)), sorted(set(raw) - expected)
    if missing or extra:
        raise ValueError(f"proposal fields differ; missing={missing}; extra={extra}")
    if raw["decision"] not in {"experiment", "stop"}:
        raise ValueError("decision must be experiment or stop")
    if raw["evidence_id"] != evidence_id:
        raise ValueError("proposal does not cite the current evidence")
    for key in ("proposal_id", "hypothesis", "expected_validation"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    for key, value in (required_context or {}).items():
        if raw[key] != value:
            raise ValueError(f"{key} does not match the request context")

    for group in ("collection", "training"):
        declared = {a.name: a for a in space.axes(group)}
        supplied = raw[group]
        if not isinstance(supplied, dict):
            raise ValueError(f"{group} must be an object")
        unknown = sorted(set(supplied) - set(declared))
        if unknown:
            raise ValueError(f"{group} has no such axis here: {unknown}; "
                             f"available: {sorted(declared)}")
        absent = sorted(set(declared) - set(supplied))
        if absent:
            raise ValueError(f"{group} is missing required axes: {absent}")
        for name, value in supplied.items():
            axis = declared[name]
            if not axis.accepts(value):
                if axis.kind == "choice":
                    raise ValueError(f"{group}.{name}={value!r} is not one of {list(axis.values)}")
                raise ValueError(f"{group}.{name}={value!r} is outside [{axis.low}, {axis.high}]")

    if raw["decision"] == "stop":
        # Stopping must not also collect; otherwise "stop" is an experiment with a
        # misleading label.
        enabled = raw["collection"].get("enabled")
        if enabled not in (False, 0, None):
            raise ValueError("a stop proposal cannot also collect data")
    return raw
