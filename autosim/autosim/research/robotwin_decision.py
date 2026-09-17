"""Budget-matched API and control decisions for the RoboTwin learner backend."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from autosim.llm_client import LLMClient

from .adapter_protocol import OptimizationSpace
from .common import atomic_json, object_digest, read_json, redact
from .decision import build_request, sample_axes, skeleton, validate_proposal
from .repository_autoresearch import load_deepseek_environment
from .robotwin_adapter import RoboTwinAdapter
from .skills import skills_reference




def validate(proposal: dict[str, Any], evidence_id: str,
             space: OptimizationSpace | None = None) -> dict[str, Any]:
    """Validate against the declared space. Kept as a thin wrapper for callers that
    have an evidence id but no space; the decision itself lives in `decision`."""
    outcome = validate_proposal(proposal, space=space or _default_space(), evidence_id=evidence_id)
    # RoboTwin's own coupling: asking for zero episodes while collecting is a
    # contradiction, and it is a fact about this benchmark's collection, not a general rule.
    collection = outcome["collection"]
    if bool(collection["enabled"]) != (int(collection["new_episodes"]) > 0):
        raise ValueError("collection enabled/count mismatch: enabled must track new_episodes > 0")
    return outcome


def _default_space() -> OptimizationSpace:
    """The declaration does not depend on which checkout is on disk."""
    return RoboTwinAdapter.declared_space()


def _proposal(controller: str, evidence: dict[str, Any], *, seed: int) -> dict[str, Any]:
    evidence_id = object_digest({key: row["evidence_id"] for key, row in sorted(evidence.items())})
    clean = evidence["demo_clean"]["metrics"]["summary"]["success_rate"]
    randomized = evidence["demo_randomized"]["metrics"]["summary"]["success_rate"]
    setting = "demo_randomized" if randomized <= clean else "demo_clean"
    params = {"epochs": 6000, "lr": 1e-5, "chunk_size": 50, "kl_weight": 10.0}
    hypothesis = "collecting official randomized demonstrations targets the measured robustness gap"
    if controller == "fixed":
        setting = "demo_randomized"
        hypothesis = "fixed control always adds the official randomized acquisition profile"
    elif controller == "random":
        # Sampled from the adapter's declaration, not from a second copy of the values:
        # a control has to draw from the same space the controller is offered, or it can
        # land on a setting the treatment was never allowed to choose.
        drawn = sample_axes(_default_space(), random.Random(seed))
        setting = drawn["collection"]["setting"]
        params = drawn["training"]
        hypothesis = "uniformly sampled legal data and learner settings form the random control"
    elif controller != "heuristic":
        raise ValueError("local controller must be fixed, random or heuristic")
    return validate({"decision": "experiment", "proposal_id": f"{controller}_{setting}_10",
                     "evidence_id": evidence_id, "hypothesis": hypothesis,
                     "collection": {"enabled": True, "setting": setting, "new_episodes": 10},
                     "training": params,
                     "expected_validation": "paired improvement on the same clean and randomized native seed banks"},
                    evidence_id)


def decide(controller: str, evidence: dict[str, Any], task_contract: dict[str, Any], *,
           seed: int, project_root: Path, allow_api_egress: bool = False) -> tuple[dict, dict]:
    if controller != "api":
        proposal = _proposal(controller, evidence, seed=seed)
        return proposal, {"controller": controller, "external_request": False}
    if not allow_api_egress:
        raise PermissionError("RoboTwin API decision requires explicit --allow-api-egress")
    environment = load_deepseek_environment(project_root=project_root)
    client = LLMClient()
    if not client.available:
        raise RuntimeError("DeepSeek credentials are unavailable in the project-local environment")
    evidence_id = object_digest({key: row["evidence_id"] for key, row in sorted(evidence.items())})
    # The schema, the value space and the prompt all come from the adapter's declaration.
    # This module supplies only what is specific to RoboTwin: the evidence payload and the
    # cross-axis coupling below.
    space = _default_space()
    system, request = build_request(
        space,
        evidence={
            "task_contract": task_contract,
            "current_evidence_id": evidence_id,
            "current_results": {setting: {"summary": row["metrics"]["summary"],
                                          "failure_categories": row["failure_categories"],
                                          "skipped_seed_reasons": row["skipped_seed_reasons"]}
                                for setting, row in evidence.items()},
            "information_boundary": "aggregate outcomes only; no trajectories, images, "
                                    "checkpoints or final seeds",
        },
        evidence_id=evidence_id,
        # Shared across benchmarks rather than copied per benchmark: a method learned on one
        # benchmark is exactly the thing that should travel, and two prompts drifting apart
        # is how one path quietly stops learning from the other's experience.
        method_library=skills_reference(benchmark=RoboTwinAdapter.benchmark),
    )
    prompts = [request]
    attempts, provider, response_hash = [], None, None
    for repair in range(3):
        response, provider = client.chat_with_metadata(system, prompts[-1], max_tokens=1200,
                                                       timeout=90, thinking="disabled")
        response_hash = object_digest(response)
        try:
            proposal = validate(json.loads(response.strip().removeprefix("```json").removesuffix("```").strip()),
                                evidence_id, space)
            attempts.append({"attempt": repair + 1, "status": "accepted", "response_sha256": response_hash})
            return proposal, {"controller": "api", "external_request": True,
                              "environment": environment, "provider": provider, "attempts": attempts,
                              "response_sha256": response_hash}
        except Exception as exc:
            error = redact(f"{type(exc).__name__}: {exc}")
            attempts.append({"attempt": repair + 1, "status": "rejected", "error": error,
                             "response_sha256": response_hash})
            prompts.append(json.dumps({"validation_error": error,
                                       "required_skeleton": skeleton(space, evidence_id=evidence_id),
                                       "instruction": "Return the full corrected object only."}, ensure_ascii=False))
    raise ValueError(f"DeepSeek RoboTwin proposal failed schema validation: {attempts[-1]['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["api", "fixed", "random", "heuristic"], required=True)
    parser.add_argument("--clean-evidence", type=Path, required=True)
    parser.add_argument("--randomized-evidence", type=Path, required=True)
    parser.add_argument("--task-contract", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--allow-api-egress", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {"demo_clean": read_json(args.clean_evidence),
                "demo_randomized": read_json(args.randomized_evidence)}
    proposal, provenance = decide(args.controller, evidence, read_json(args.task_contract), seed=args.seed,
                                  project_root=args.project_root,
                                  allow_api_egress=args.allow_api_egress)
    result = {"schema_version": 1, "proposal": proposal, "provenance": provenance}
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
