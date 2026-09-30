"""Live API smoke with a wholly synthetic incident, never historical run data."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit

from autosim.research.common import atomic_json, read_json
from autosim.research.compute_agent import client_from_file
from autosim.research.recovery_agent import RecoveryAgent
from autosim.research.repository_autoresearch import RepositoryAutoResearch, TRAINING_SPACE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = client_from_file(args.env_file)
    url = urlsplit(client.base_url)
    if url.scheme != "https" or url.hostname != "api.deepseek.com" or url.username or url.password:
        raise ValueError("this smoke is scoped to the verified https://api.deepseek.com endpoint")
    snapshot = {"schema_version": 1, "scope": "candidate_development", "synthetic_fixture": True,
        "round": 1, "rounds_remaining": 1, "baseline_verified": True, "state_dim": 2,
        "shards": [{"shard": "synthetic_worker", "failed": True,
                    "failure_kind": "nonfinite_observation", "nonfinite_indices": [0, 1]}],
        "aggregate_metrics_valid": False, "quarantine_eligible": True,
        "previous_training_params": {},
        "allowed_training_params": {k: sorted(v) for k, v in TRAINING_SPACE.items()},
        "causal_limit": "Artificial NaN fixture. No private run data or measured performance."}
    atomic_json(args.output / "outbound_snapshot.json", snapshot)
    agent = RecoveryAgent(args.output / "agent", client, max_calls=2, max_tokens=30000)
    record = agent.decide(snapshot)
    first_budget = read_json(args.output / "agent/api/budget.json")
    cached = agent.decide(snapshot)
    assert record == cached and first_budget == read_json(args.output / "agent/api/budget.json")
    assert record["decision"]["action"] == "quarantine_candidate"
    params = record["decision"]["next_trial_training_params"]
    assert params, "recovery smoke must produce a next-trial intervention"
    bound = RepositoryAutoResearch._bind_recovery_params({"training": {"params": {}}},
                                                        {"recovery_training_constraints": params})
    assert bound["training"]["params"] == params
    atomic_json(args.output / "validation.json", {"status": "passed", "api_used": True,
        "historical_data_sent": False, "gpu_or_simulator_used": False,
        "decision": record, "cached_resume_no_extra_charge": True,
        "next_trial_parameters_bound": bound["training"]["params"], "budget": first_budget})
    print({"status": "passed", "model": client.model, "action": record["decision"]["action"],
           "usage": record["provider"].get("usage"), "next_trial_params": params})


if __name__ == "__main__":
    main()
