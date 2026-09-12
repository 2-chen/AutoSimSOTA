"""Versioned contracts shared by repository AutoResearch components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


CapabilityStatus = Literal["declared", "verified", "unsupported", "failed", "unknown"]


@dataclass(frozen=True)
class CapabilityRecord:
    capability_id: str
    scope: str
    status: CapabilityStatus
    evidence: dict[str, Any] = field(default_factory=dict)
    limitation: str | None = None
    estimated_cost: dict[str, float] = field(default_factory=dict)
    probe_version: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    execution_complete: bool
    result_valid: bool
    performance_improved: bool
    hypothesis_supported: bool | None
    export_runtime_verified: bool
    reasons: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
