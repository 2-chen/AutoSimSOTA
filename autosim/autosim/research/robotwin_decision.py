"""Budget-matched API and control decisions for the RoboTwin learner backend."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from autosim.llm_client import LLMClient

from .common import atomic_json, object_digest, read_json, redact
from .repository_autoresearch import load_deepseek_environment


SETTINGS = {"demo_clean", "demo_randomized"}
LRS = {5e-6, 1e-5, 2e-5}
CHUNKS = {25, 50, 75}
KL_WEIGHTS = {1.0, 10.0}


def validate(proposal: dict[str, Any], evidence_id: str) -> dict[str, Any]:
    required = {"decision", "proposal_id", "evidence_id", "hypothesis", "collection",
                "training", "expected_validation"}
    if set(proposal) != required:
        raise ValueError(f"proposal top-level fields differ: {sorted(set(proposal) ^ required)}")
    if proposal["decision"] not in {"experiment", "stop"}:
        raise ValueError("decision must be experiment or stop")
    if proposal["evidence_id"] != evidence_id:
        raise ValueError("proposal does not reference current evidence")
    if not all(isinstance(proposal[key], str) and proposal[key].strip()
               for key in ("proposal_id", "hypothesis", "expected_validation")):
        raise ValueError("proposal text fields must be non-empty strings")
    collection = proposal["collection"]
    if set(collection) != {"enabled", "setting", "new_episodes"}:
        raise ValueError("collection fields differ")
    if not isinstance(collection["enabled"], bool) or collection["setting"] not in SETTINGS:
        raise ValueError("invalid collection setting")
    if collection["new_episodes"] not in {0, 10}:
        raise ValueError("new_episodes must be 0 or 10")
    if collection["enabled"] != (collection["new_episodes"] > 0):
        raise ValueError("collection enabled/count mismatch")
    training = proposal["training"]
    if set(training) != {"epochs", "lr", "chunk_size", "kl_weight"}:
        raise ValueError("training fields differ")
    if training["epochs"] not in {2000, 6000} or float(training["lr"]) not in LRS:
        raise ValueError("training epoch/lr outside registered space")
    if training["chunk_size"] not in CHUNKS or float(training["kl_weight"]) not in KL_WEIGHTS:
        raise ValueError("training chunk/KL outside registered space")
    if proposal["decision"] == "stop" and collection["enabled"]:
        raise ValueError("stop proposal cannot collect")
    return proposal


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
        rng = random.Random(seed)
        setting = rng.choice(sorted(SETTINGS))
        params = {"epochs": rng.choice([2000, 6000]), "lr": rng.choice(sorted(LRS)),
                  "chunk_size": rng.choice(sorted(CHUNKS)), "kl_weight": rng.choice(sorted(KL_WEIGHTS))}
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
    payload = {
        "task_contract": task_contract,
        "current_evidence_id": evidence_id,
        "current_results": {setting: {"summary": row["metrics"]["summary"],
                                      "failure_categories": row["failure_categories"],
                                      "skipped_seed_reasons": row["skipped_seed_reasons"]}
                            for setting, row in evidence.items()},
        "allowed_collection": {"setting": sorted(SETTINGS), "new_episodes": [0, 10]},
        "allowed_training": {"epochs": [2000, 6000], "lr": sorted(LRS),
                             "chunk_size": sorted(CHUNKS), "kl_weight": sorted(KL_WEIGHTS)},
        "information_boundary": "aggregate outcomes only; no trajectories, images, checkpoints or final seeds",
    }
    skeleton = {"decision": "experiment", "proposal_id": "short_identifier",
                "evidence_id": evidence_id, "hypothesis": "testable hypothesis",
                "collection": {"enabled": True, "setting": "demo_randomized", "new_episodes": 10},
                "training": {"epochs": 6000, "lr": 1e-5, "chunk_size": 50, "kl_weight": 10.0},
                "expected_validation": "paired native evaluation prediction"}
    system = ("You are a bounded robotics experiment planner. Return exactly one JSON object matching the "
              "provided skeleton. Choose only registered values. Do not modify the evaluator, task judge, "
              "policy observations, or request raw data. Acknowledge that aggregate failures do not prove cause.")
    prompts = [json.dumps({"evidence": payload, "required_skeleton": skeleton}, ensure_ascii=False)]
    attempts, provider, response_hash = [], None, None
    for repair in range(3):
        response, provider = client.chat_with_metadata(system, prompts[-1], max_tokens=1200,
                                                       timeout=90, thinking="disabled")
        response_hash = object_digest(response)
        try:
            proposal = validate(json.loads(response.strip().removeprefix("```json").removesuffix("```").strip()),
                                evidence_id)
            attempts.append({"attempt": repair + 1, "status": "accepted", "response_sha256": response_hash})
            return proposal, {"controller": "api", "external_request": True,
                              "environment": environment, "provider": provider, "attempts": attempts,
                              "response_sha256": response_hash}
        except Exception as exc:
            error = redact(f"{type(exc).__name__}: {exc}")
            attempts.append({"attempt": repair + 1, "status": "rejected", "error": error,
                             "response_sha256": response_hash})
            prompts.append(json.dumps({"validation_error": error, "required_skeleton": skeleton,
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
