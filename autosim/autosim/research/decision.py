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
from typing import Any, Sequence

from .adapter_protocol import Axis, OptimizationSpace

#: Fields every proposal carries regardless of benchmark. `evidence_id` binds a
#: proposal to the evidence it was made against, so a stale proposal cannot be
#: replayed against a different round's numbers.
_BASE_FIELDS = ("decision", "proposal_id", "hypothesis", "expected_validation")
#: The field carrying the evidence digest. A schema choice, not a policy: a
#: benchmark keeps whatever name its recorded proposals already use.
EVIDENCE_FIELD = "evidence_id"

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
    suffix = " (optional)" if axis.optional else ""
    return f"{name}: {axis.explain()} -- {axis.description}{suffix}"


def describe_space(space: OptimizationSpace) -> dict[str, list[str]]:
    """The space as prose lines, which is what a model reads better than nested dicts."""
    return {name: [_axis_line(a.name, a) for a in axes]
            for name, axes in space.sections().items()}


def skeleton(space: OptimizationSpace, *, evidence_id: str,
             extra: dict[str, Any] | None = None,
             evidence_field: str = EVIDENCE_FIELD) -> dict[str, Any]:
    """The starting object shown to the controller: declared defaults only."""
    # Built from the space's own skeleton rather than re-derived here: a second copy of
    # the shape is how the nesting a declaration asked for got silently dropped.
    body: dict[str, Any] = {
        "decision": "experiment",
        "proposal_id": "short_identifier",
        evidence_field: evidence_id,
        "hypothesis": "a testable statement about what you expect and why",
        "expected_validation": "what the paired native evaluation should show",
    }
    body.update(space.skeleton())
    if extra:
        body.update(extra)
    return body


def build_request(space: OptimizationSpace, *, evidence: dict[str, Any], evidence_id: str,
                  method_library: dict[str, Any] | None = None,
                  benchmark_notes: dict[str, Any] | None = None,
                  extra_skeleton: dict[str, Any] | None = None,
                  evidence_field: str = EVIDENCE_FIELD) -> tuple[str, str]:
    """The (system, user) pair for one proposal request."""
    payload: dict[str, Any] = {
        "evidence": evidence,
        "required_skeleton": skeleton(space, evidence_id=evidence_id, extra=extra_skeleton,
                                      evidence_field=evidence_field),
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
                groups: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Draw a legal setting from the declared space.

    The matched-budget control arms have to sample from the *same* space the
    controller is offered, or a control can drift onto values the treatment could
    not have chosen. Deriving both from one declaration is what keeps that true
    when someone edits the declaration.
    """
    drawn: dict[str, Any] = {}
    for section in (groups or tuple(space.sections())):
        values: dict[str, Any] = {}
        for axis in space.axes(section):
            if axis.kind == "choice":
                value = rng.choice(list(axis.values))
            elif axis.kind == "integer":
                value = rng.randint(int(axis.low), int(axis.high))
            elif axis.kind == "number":
                value = rng.uniform(float(axis.low), float(axis.high))
            else:
                # A structure has no range to draw from; its declared default is the
                # only value the control arms can be sure the benchmark can execute.
                value = axis.default
            if axis.group:
                values.setdefault(axis.group, {})[axis.name] = value
            else:
                values[axis.name] = value
        drawn[section] = values
    return drawn


def _check(section: str, name: str, axis: Axis, value: Any) -> None:
    """Reject a value the declaration does not accept, saying what it does accept."""
    if axis.accepts(value):
        return
    raise ValueError(f"{section}.{name} does not accept {value!r}; it takes {axis.explain()}")


def validate_proposal(raw: dict[str, Any], *, space: OptimizationSpace,
                      evidence_id: str, required_context: dict[str, Any] | None = None,
                      extra_fields: Sequence[str] = (),
                      evidence_field: str = EVIDENCE_FIELD) -> dict[str, Any]:
    """Validate against the declared space. Raises ValueError with a usable message.

    The messages name the axis and what it accepts, because a rejected proposal is
    the controller's only feedback about why, and "schema mismatch" teaches it nothing.
    """
    if not isinstance(raw, dict):
        raise ValueError("proposal must be one JSON object")
    expected = (set(_BASE_FIELDS) | {evidence_field} | set(space.sections())
                | set(required_context or {}) | set(extra_fields))
    missing, extra = sorted(expected - set(raw)), sorted(set(raw) - expected)
    if missing or extra:
        raise ValueError(f"proposal fields differ; missing={missing}; extra={extra}")
    if raw["decision"] not in {"experiment", "stop"}:
        raise ValueError("decision must be experiment or stop")
    if raw[evidence_field] != evidence_id:
        raise ValueError("proposal does not cite the current evidence")
    for key in ("proposal_id", "hypothesis", "expected_validation"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    for key, value in (required_context or {}).items():
        if raw[key] != value:
            raise ValueError(f"{key} does not match the request context")

    for section in space.sections():
        supplied = raw[section]
        if not isinstance(supplied, dict):
            raise ValueError(f"{section} must be an object")
        axes = space.axes(section)
        standalone = {a.name: a for a in axes if not a.group}
        grouped: dict[str, dict[str, Axis]] = {}
        for axis in axes:
            if axis.group:
                grouped.setdefault(axis.group, {})[axis.name] = axis

        allowed = set(standalone) | set(grouped)
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(f"{section} has no such axis here: {unknown}; "
                             f"available: {sorted(allowed)}")
        required = {a.name for a in axes if not a.optional and not a.group} | set(grouped)
        absent = sorted(required - set(supplied))
        if absent:
            raise ValueError(f"{section} is missing required entries: {absent}")

        for name, value in supplied.items():
            if name not in grouped:
                _check(section, name, standalone[name], value)
                continue
            if not isinstance(value, dict):
                raise ValueError(f"{section}.{name} must be an object")
            members = grouped[name]
            sub_unknown = sorted(set(value) - set(members))
            if sub_unknown:
                raise ValueError(f"{section}.{name} has no such axis: {sub_unknown}; "
                                 f"available: {sorted(members)}")
            sub_absent = sorted(k for k, a in members.items() if not a.optional and k not in value)
            if sub_absent:
                raise ValueError(f"{section}.{name} is missing required axes: {sub_absent}")
            for sub_name, sub_value in value.items():
                _check(f"{section}.{name}", sub_name, members[sub_name], sub_value)

    if raw["decision"] == "stop":
        # Stopping must not also collect; otherwise "stop" is an experiment with a
        # misleading label.
        enabled = raw["collection"].get("enabled")
        if enabled not in (False, 0, None):
            raise ValueError("a stop proposal cannot also collect data")
    return raw
