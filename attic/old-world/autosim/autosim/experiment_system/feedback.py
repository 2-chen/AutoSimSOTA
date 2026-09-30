"""Evidence-linked collection tickets and bounded, version-specific repairs."""
from dataclasses import asdict, dataclass
from pathlib import Path

from autosim.research.common import digest, object_digest, read_json


@dataclass(frozen=True)
class Repair:
    name: str
    failure_kind: str
    stage: str
    operation: str
    source_hashes: dict
    regression_evidence: dict

    def applicable(self, failure_kind, stage):
        if stage == "evaluate" or self.stage != stage or self.failure_kind != failure_kind:
            return False
        if self.operation not in {"retry_new_attempt", "use_official_data", "reduce_training_batch"}:
            return False
        if not self.source_hashes or not self.regression_evidence:
            return False
        if not all(Path(p).is_file() and digest(Path(p)) == h
                   for p, h in {**self.source_hashes, **self.regression_evidence}.items()):
            return False
        return all(read_json(Path(p)).get("passed") is True for p in self.regression_evidence)


def collection_ticket(analysis: dict, *, supported_profiles: set[str], budget: int) -> dict:
    if analysis.get("purpose") != "development":
        raise ValueError("only development evidence may guide collection")
    if budget < 1 or "full_random" not in supported_profiles:
        raise ValueError("positive collection budget and full-random fallback required")
    source = Path(analysis["source"])
    if not source.is_file():
        raise ValueError("analysis provenance is missing")
    failures = analysis.get("failures", [])
    supported = [f for f in failures if f.get("confidence") == "validated"
                 and f.get("collection_profile") in supported_profiles
                 and f.get("evidence_files")
                 and all(Path(p).is_file() and digest(Path(p)) == h for p, h in f["evidence_files"].items())]
    chosen = supported[0]["collection_profile"] if supported else "full_random"
    targeted = budget // 2 if chosen != "full_random" else 0
    result = {"schema_version": 1, "source": str(source), "source_sha256": digest(source),
              "purpose": "training_only", "target_successful_episodes": budget,
              "max_attempts": budget * 12, "random_coverage_episodes": budget - targeted,
              "targeted_episodes": targeted, "profile": chosen,
              "diagnosis_status": "supported_hypothesis" if supported else "insufficient_causal_evidence",
              "judge_mutation_allowed": False, "inference_privileged_state_allowed": False}
    result["ticket_id"] = object_digest(result)
    return result
