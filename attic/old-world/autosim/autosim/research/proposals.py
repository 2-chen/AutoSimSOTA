"""Bounded rule/HTTP-model proposals; never arbitrary shell or evaluator patches."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .common import object_digest, redact


PARAM_SPACE = {
    "optimizer_lr": [5e-6, 1e-5, 2e-5], "chunk_size": [25, 50, 75],
    "n_action_steps": [10, 25, 50], "kl_weight": [1.0, 10.0],
    "action_loss_profile": ["legacy_mask_mean", "valid_mean"],
    "image_augmentation_profile": ["none", "photometric_mild", "camera_geometry_mild"],
}
COLLECTION_PROFILES = {"full_random", "targeted_camera", "targeted_appearance", "targeted_clutter"}


@dataclass(frozen=True)
class Proposal:
    hypothesis: str
    evidence: str
    params: dict
    collection_profile: str = "full_random"
    source: str = "rule"

    def validate(self):
        if not self.hypothesis or not self.evidence:
            raise ValueError("a proposal needs a hypothesis and evidence provenance")
        if set(self.params) - set(PARAM_SPACE):
            raise ValueError("proposal contains unapproved parameter or mutation")
        for name, value in self.params.items():
            if value not in PARAM_SPACE[name]:
                raise ValueError(f"parameter outside allowed search space: {name}")
        if self.params.get("n_action_steps", 50) > self.params.get("chunk_size", 50):
            raise ValueError("execution window exceeds prediction window")
        if self.collection_profile not in COLLECTION_PROFILES:
            raise ValueError("collection profile not allowed")
        if self.source not in {"rule", "llm", "fixed_control"}:
            raise ValueError("unknown proposal provenance")
        return self

    @property
    def signature(self):
        return object_digest({"params": self.params, "collection_profile": self.collection_profile})

    def as_dict(self):
        return asdict(self)


def rule_proposals(analysis: dict, seen: set[str]) -> list[Proposal]:
    evidence = analysis["source"]
    categories = analysis.get("categories", {})
    candidates = [Proposal("Normalize action loss over valid labels; test whether tail padding dilutes supervision",
                           evidence, {"action_loss_profile": "valid_mean"})]
    if categories.get("incomplete_after_motion", 0) or categories.get("large_object_height_drop", 0):
        candidates.insert(0, Proposal("Motion starts but task is incomplete; test more frequent visual feedback",
                                     evidence, {"n_action_steps": 10}))
    if categories.get("little_object_motion", 0):
        candidates.append(Proposal("Low object motion has unknown cause; explore mild appearance regularization",
                                   evidence, {"image_augmentation_profile": "photometric_mild"}, "targeted_appearance"))
    candidates += [Proposal("Explore camera coverage and image regularization; visual causality remains unconfirmed",
                            evidence, {"image_augmentation_profile": "camera_geometry_mild"}, "targeted_camera"),
                   Proposal("Test a lower learning rate under the same data and update budget", evidence,
                            {"optimizer_lr": 5e-6}),
                   Proposal("Test a larger learning rate under the same data and update budget", evidence,
                            {"optimizer_lr": 2e-5})]
    return [p.validate() for p in candidates if p.signature not in seen]


def propose(analysis: dict, seen: set[str], *, allow_llm=False) -> tuple[Proposal | None, dict]:
    provider = {"requested_llm": allow_llm, "used": "rule"}
    if allow_llm:
        from autosim.llm_client import LLMClient

        client = LLMClient()
        if client.available:
            try:
                # Only aggregate development evidence; no private trajectories,
                # credentials, final seeds, or filesystem source context.
                response = client.chat(
                    "Return only JSON with hypothesis, params, collection_profile. Do not propose evaluator "
                    "changes. Explain uncertainty; motion statistics do not establish visual causality.",
                    json.dumps({"categories": analysis["categories"], "summary": analysis["summary"],
                                "allowed_parameters": PARAM_SPACE,
                                "allowed_collection_profiles": sorted(COLLECTION_PROFILES)}),
                    max_tokens=1200, timeout=60)
                data = json.loads(response)
                if set(data) != {"hypothesis", "params", "collection_profile"}:
                    raise ValueError("model response schema mismatch")
                proposal = Proposal(evidence=analysis["source"], source="llm", **data).validate()
                if proposal.signature in seen:
                    raise ValueError("model repeated a previously tested proposal")
                return proposal, {"requested_llm": True, "used": "llm", "model": client.model}
            except Exception as exc:
                provider["fallback_reason"] = redact(f"{type(exc).__name__}: {exc}")
        else:
            provider["fallback_reason"] = "no_explicit_model_credentials"
    candidates = rule_proposals(analysis, seen)
    return (candidates[0] if candidates else None), provider
