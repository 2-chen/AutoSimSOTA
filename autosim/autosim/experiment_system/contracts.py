"""Explicit semantics and evidence-bound capabilities, independent of benchmark."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from autosim.research.common import digest, object_digest


@dataclass(frozen=True)
class TaskContract:
    benchmark: str
    task: str
    observation: dict
    action: dict
    timing: dict
    evaluation: dict
    sources: dict[str, str]

    def validate(self):
        if not self.benchmark or not self.task or not self.sources:
            raise ValueError("task identity and pinned sources are required")
        for block, fields in ((self.observation, ("state_dim", "cameras")),
                              (self.action, ("dimension", "mode", "units", "order", "frame")),
                              (self.timing, ("max_actions", "recorded_hz", "control_hz")),
                              (self.evaluation, ("native_entry", "success_authority", "setting"))):
            if any(key not in block for key in fields):
                raise ValueError(f"incomplete semantic contract: {fields}")
        if self.observation["state_dim"] <= 0 or self.action["dimension"] <= 0:
            raise ValueError("nonpositive dimensions")
        if len(self.action["units"]) != self.action["dimension"]:
            raise ValueError("one unit per action dimension required")
        if not self.observation["cameras"] or self.timing["max_actions"] <= 0:
            raise ValueError("missing cameras or horizon")
        return self

    @property
    def signature(self):
        self.validate()
        return object_digest(asdict(self))

    def assert_sources(self):
        for path, expected in self.sources.items():
            if not Path(path).is_file() or digest(Path(path)) != expected:
                raise ValueError(f"source version changed: {path}")


@dataclass(frozen=True)
class Capability:
    name: str
    status: str  # declared, validated, unavailable, unknown
    contract_signature: str
    evidence: dict[str, str]
    detail: str

    def usable(self, contract: TaskContract):
        if self.status != "validated" or self.contract_signature != contract.signature or not self.evidence:
            return False
        contract.assert_sources()
        return all(Path(p).is_file() and digest(Path(p)) == h for p, h in self.evidence.items())


def select_workflow(contract: TaskContract, capabilities: list[Capability], policy: str) -> dict:
    contract.validate()
    available = {c.name for c in capabilities if c.usable(contract)}
    required = {"semantic_probe", "native_evaluation", f"train_reload:{policy}"}
    missing = sorted(required - available)
    if missing:
        return {"mode": "needs_validation", "missing": missing, "automatic_collection": False}
    if "expert_collection" in available:
        return {"mode": "collect_train_evaluate", "missing": [], "automatic_collection": True}
    if "official_data" in available:
        return {"mode": "official_data_only", "missing": [], "automatic_collection": False}
    return {"mode": "needs_data_adapter", "missing": ["usable_data_source"], "automatic_collection": False}


def assert_compatible(expected: TaskContract, observed: dict):
    """Same-shaped semantic mistakes fail instead of being silently reshaped."""
    for key in ("observation", "action", "timing"):
        if object_digest(observed.get(key)) != object_digest(getattr(expected, key)):
            raise ValueError(f"{key} semantics mismatch")


class BenchmarkPlugin(Protocol):
    def contract(self, task: str) -> TaskContract: ...
    def inventory(self, task: str) -> dict: ...
