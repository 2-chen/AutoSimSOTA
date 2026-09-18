"""What the research loop should be, decided from what the benchmark can do.

The loop had a fixed shape: collect data, train on it, measure, keep the better policy. A
model chose values inside that shape -- which collection profile, how much sampling mass,
which loss -- but never the shape itself. That was fine while there was one benchmark, and
wrong as soon as there were two: the shape was built on a benchmark that ships a scripted
expert, and it silently made "an expert exists" a precondition for research anywhere.

The shape is not the method. The method is: find something the benchmark lets you vary,
vary it, and measure the result on a comparison that cannot be gamed. Collecting new
trajectories is one way to do that and needs an expert. Re-deciding which of the shipped
demonstrations to train on needs nothing but the demonstrations. So does the loss, the
augmentation, the action horizon, the number of steps. A benchmark with no expert is not a
benchmark that cannot be researched; it is one where a particular family of interventions
is unavailable, and the plan has to be built from the families that are.

So this module asks the model for a plan rather than assuming one, and then checks the
plan against two things it cannot talk its way around: the axes the adapter declared, and
the capabilities the declaration established. A family the benchmark cannot support is
caught here, before it becomes a round that fails at the collector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapter_protocol import OptimizationSpace
from .common import object_digest, redact

#: The intervention families the loop knows how to execute. Each needs a way to be carried
#: out and a way to be measured; naming them here is what lets a plan be checked against
#: the machinery rather than against prose.
FAMILIES: dict[str, dict[str, str]] = {
    "targeted_collection": {
        "does": "generate new trajectories aimed at a region of the scene or initial-state "
                "distribution the current policy handles badly",
        "needs": "new_trajectory_generation, and for aiming it, targeted_generation",
        "axes": "the collection section of the declared space",
    },
    "data_selection": {
        "does": "change which of the benchmark's own demonstrations are used, or how much "
                "weight each one carries, without generating anything",
        "needs": "official_dataset",
        "axes": "any axis in the declared space that selects or weights training samples",
    },
    "training_recipe": {
        "does": "change how the policy is fitted to the data it already has",
        "needs": "training",
        "axes": "the training section of the declared space",
    },
    "evaluation_resolution": {
        "does": "change how much evidence a claim is judged on, so a difference the current "
                "measurement cannot resolve becomes resolvable",
        "needs": "native_evaluation",
        "axes": "any axis in the declared space that sets how many episodes a round is "
                "judged on",
    },
}

#: Fields every plan carries.
_BASE = ("strategy_id", "reasoning", "intervention_families", "first_intervention",
         "measurement", "stop_condition")

SYSTEM = (
    "You are the research planner for a simulator manipulation benchmark. You are given "
    "what the benchmark can do and everything its adapter declared that can be varied, and "
    "you decide what this research loop is: which interventions are worth trying, which "
    "comes first, how the result will be measured, and what would make you stop.\n\n"
    "The families you may use are listed in `intervention_families_available`. For each, "
    "say whether this benchmark can support it and cite the axes in the declared space that "
    "would carry it out. A family whose needed capabilities are not available must be "
    "reported unavailable -- and that is not a dead end. It means the plan is built from "
    "the families that do work.\n\n"
    "Two things decide whether this is a plan or a wish.\n\n"
    "First, the axis you cite must exist. Every intervention is carried out by varying "
    "something the adapter declared; if nothing in the declared space implements an idea, "
    "the system cannot execute it, however good the idea is.\n\n"
    "Second, `first_intervention` is what the first round will actually do -- not the most "
    "interesting thing you might do eventually. It must name an available family and the "
    "axes that family will move. A first round that cannot run wastes the whole budget, so "
    "prefer the intervention that is both informative and certain to execute.\n\n"
    "Say what would falsify your first intervention. A plan whose first step cannot fail "
    "measures nothing.\n\n"
    "Return exactly one JSON object and nothing else."
)


def skeleton() -> dict[str, Any]:
    return {
        "strategy_id": "<short identifier for this plan>",
        "reasoning": "<why this plan suits what this benchmark can and cannot do>",
        "intervention_families": {
            name: {"available": True,
                   "why": "<what in the benchmark supports this, or why it does not>",
                   "axes": ["<names from the declared space that carry this out>"]}
            for name in FAMILIES
        },
        "first_intervention": {
            "family": "<one of the available families>",
            "axes": ["<names from the declared space the first round will move>"],
            "hypothesis": "<what you expect and why>",
            "falsified_by": "<the measurement that would show this was wrong>",
        },
        "measurement": "<what the benchmark's own evaluator will be asked, on which bank>",
        "stop_condition": "<what would make further rounds pointless>",
    }


def axis_names(space: OptimizationSpace) -> set[str]:
    """Every axis name the adapter declared, in either dotted or bare form.

    Both spellings are returned because a plan naming `training.steps` and one naming
    `steps` mean the same axis, and refusing the second would be a spelling test.
    """
    names: set[str] = set()
    for section, axes in space.sections().items():
        for axis in axes:
            names.add(axis.name)
            names.add(f"{section}.{axis.name}")
    return names


def problems_in(raw: dict[str, Any], space: OptimizationSpace,
                capabilities: dict[str, Any]) -> list[str]:
    """Faults in a plan, as messages. Empty means executable, not wise.

    Two questions, both answerable: does the plan move things the benchmark actually
    exposes, and does it ask for capabilities the declaration did not establish.
    """
    if not isinstance(raw, dict):
        return ["strategy must be one JSON object"]
    problems = [f"missing required key: {key}" for key in _BASE if key not in raw]
    if problems:
        return problems

    known = axis_names(space)
    families = raw["intervention_families"]
    if not isinstance(families, dict):
        return ["intervention_families must be an object"]
    unknown_families = sorted(set(families) - set(FAMILIES))
    if unknown_families:
        problems.append(f"intervention_families has no such entry: {unknown_families}; "
                        f"answer exactly {sorted(FAMILIES)}")

    for name, row in families.items():
        if name not in FAMILIES or not isinstance(row, dict):
            continue
        if not isinstance(row.get("available"), bool):
            problems.append(f"intervention_families.{name}.available must be true or false")
            continue
        if not str(row.get("why", "")).strip():
            problems.append(f"intervention_families.{name}.why is required either way")
        cited = row.get("axes") or []
        if row["available"] and not cited:
            problems.append(f"intervention_families.{name} claims to be available and "
                            f"cites no axis to carry it out")
        for axis in cited:
            if axis not in known:
                problems.append(f"intervention_families.{name} cites {axis!r}, which the "
                                f"benchmark does not expose; declared axes are {sorted(known)}")

    first = raw["first_intervention"]
    if not isinstance(first, dict):
        return problems + ["first_intervention must be an object"]
    family = first.get("family")
    if family not in FAMILIES:
        problems.append(f"first_intervention.family must be one of {sorted(FAMILIES)}")
    else:
        chosen = families.get(family) if isinstance(families.get(family), dict) else {}
        if not chosen.get("available"):
            problems.append(f"first_intervention uses {family}, which this plan reports "
                            f"as unavailable")
        for axis in first.get("axes") or []:
            if axis not in known:
                problems.append(f"first_intervention cites {axis!r}, which the benchmark "
                                f"does not expose")
        if not (first.get("axes") or []):
            problems.append("first_intervention must name the axes it will move")
    for key in ("hypothesis", "falsified_by"):
        if not str(first.get(key, "")).strip():
            problems.append(f"first_intervention.{key} is required; a step that cannot "
                            f"fail measures nothing")

    needs = {"targeted_collection": ("new_trajectory_generation", "targeted_generation"),
             "data_selection": ("official_dataset",),
             "training_recipe": ("training",),
             "evaluation_resolution": ("native_evaluation",)}
    for name, required in needs.items():
        row = families.get(name)
        if not isinstance(row, dict) or not row.get("available"):
            continue
        missing = [cap for cap in required
                   if str((capabilities.get(cap) or {}).get("status")) == "unsupported"]
        if missing:
            problems.append(f"intervention_families.{name} is claimed available but the "
                            f"declaration reports {missing} as unsupported here")

    for key in ("reasoning", "measurement", "stop_condition", "strategy_id"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            problems.append(f"{key} must be a non-empty string")
    return problems


def build_request(space: OptimizationSpace, capabilities: dict[str, Any], *,
                  benchmark: str, evidence: dict[str, Any] | None = None) -> tuple[str, str]:
    """The plan request: what can be varied, and what the benchmark can do."""
    payload = {
        "benchmark": benchmark,
        "intervention_families_available": FAMILIES,
        "capabilities_established": {
            name: {"status": row.get("status"), "note": row.get("limitation")}
            for name, row in capabilities.items()},
        "what_you_may_vary": {
            section: [{"name": axis.name, "description": axis.description,
                       "accepts": axis.explain()} for axis in axes]
            for section, axes in space.sections().items()},
        "required_skeleton": skeleton(),
    }
    if evidence:
        payload["benchmark_evidence"] = evidence
    return SYSTEM, json.dumps(payload, ensure_ascii=False, sort_keys=True)


def plan(space: OptimizationSpace, capabilities: dict[str, Any], client: Any, *,
         benchmark: str, evidence: dict[str, Any] | None = None, output: Path | None = None,
         attempts: int = 3) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ask for a plan and refuse to accept one the benchmark cannot execute."""
    system, user = build_request(space, capabilities, benchmark=benchmark, evidence=evidence)
    log: list[dict[str, Any]] = []
    for repair in range(attempts):
        current = user if repair == 0 else user + json.dumps({
            "rejected": log[-1]["error"],
            "instruction": "Return the full corrected plan. Cite only axes that appear in "
                           "what_you_may_vary, and mark a family available only if the "
                           "capability it needs is not reported unsupported."}, ensure_ascii=False)
        content, metadata = client.chat_with_metadata(system, current, max_tokens=4096,
                                                      timeout=180, thinking="disabled")
        if output is not None:
            (output / f"strategy_draft_{repair + 1}.json").write_text(
                json.dumps({"attempt": repair + 1, "content": content, "provider": metadata},
                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            value = json.loads(content.strip().removeprefix("```json")
                               .removesuffix("```").strip())
            faults = problems_in(value, space, capabilities)
            if faults:
                raise ValueError("; ".join(faults[:6]))
            log.append({"attempt": repair + 1, "status": "accepted",
                        "response_sha256": object_digest(content)})
            return value, log
        except (ValueError, json.JSONDecodeError) as exc:
            log.append({"attempt": repair + 1, "status": "rejected",
                        "error": redact(f"{type(exc).__name__}: {exc}"),
                        "response_sha256": object_digest(content)})
    raise ValueError(f"no executable plan: {log[-1].get('error')}")


def executable_first_step(plan_value: dict[str, Any]) -> dict[str, Any]:
    """The first round's intervention, as the loop needs to read it.

    The loop used to require a collection probe in round one, which made an expert a
    precondition for research. What it actually needs is that round one runs *the plan's*
    first intervention -- whatever family that is -- so that a benchmark without an expert
    is researched through the families it does have.
    """
    first = plan_value["first_intervention"]
    return {"family": first["family"], "axes": list(first["axes"]),
            "hypothesis": first["hypothesis"], "falsified_by": first["falsified_by"]}
