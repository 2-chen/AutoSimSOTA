"""A persisted wall-clock guard; unknown GPU/LLM consumption is never reported as zero."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .common import atomic_json, now, read_json


class RunBudget:
    def __init__(self, root: Path, *, wall_seconds: float | None = None):
        self.path = Path(root) / "budget.json"
        if self.path.is_file():
            held = read_json(self.path)
            if wall_seconds is not None and float(held["wall_seconds"]) != float(wall_seconds):
                raise ValueError("cannot silently change a resumed run's wall-clock budget")
            self.limit = float(held["wall_seconds"])
            self.started_epoch = float(held["started_epoch"])
        else:
            if wall_seconds is None or wall_seconds <= 0:
                raise ValueError("a new run budget needs a positive wall_seconds limit")
            self.limit = float(wall_seconds)
            self.started_epoch = time.time()
            self.record()

    @classmethod
    def existing(cls, root: Path) -> "RunBudget | None":
        return cls(root) if (Path(root) / "budget.json").is_file() else None

    def remaining(self) -> float:
        return max(0.0, self.limit - (time.time() - self.started_epoch))

    def record(self) -> dict[str, Any]:
        remaining = self.remaining()
        state: dict[str, Any] = {"schema_version": 1, "updated_at": now(),
                                 "pause_policy": "deadline_continues",
                                 "started_epoch": self.started_epoch,
                                 "wall_seconds": self.limit,
                                 "elapsed_wall_seconds": self.limit - remaining,
                                 "remaining_wall_seconds": remaining,
                                 "gpu_seconds": None, "llm_cost": None,
                                 "status": "exhausted" if remaining <= 0 else "active"}
        gpu_path = self.path.parent / "task_gpu_budget.json"
        if gpu_path.is_file():
            gpu = read_json(gpu_path)
            held = sum(row["reserved_seconds"] for row in gpu["leases"].values()
                       if row["status"] == "active")
            state.update(gpu_seconds=gpu["charged_seconds"] if not held else None,
                         settled_gpu_seconds=gpu["charged_seconds"],
                         reserved_gpu_seconds=held, gpu_cap_seconds=gpu["cap_seconds"],
                         gpu_accounting=gpu["accounting"], budget_scope="task")
        atomic_json(self.path, state)
        return state


class BudgetedClient:
    """Bound each model request by the persisted whole-run deadline.

    The provider's internal retries would each get a fresh timeout and could multiply the
    requested wall time, so bounded calls disable those retries. The outer research loop may
    make another decision only if the budget still permits it.
    """

    def __init__(self, client: Any, budget: RunBudget):
        self.client = client
        self.budget = budget

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def chat_with_metadata(self, system: str, user: str, **kwargs: Any) -> Any:
        left = self.budget.remaining()
        if left <= 0:
            self.budget.record()
            raise TimeoutError("run wall-clock budget exhausted before model request")
        requested = kwargs.get("timeout", left)
        kwargs["timeout"] = min(float(requested), left)
        kwargs["retries"] = 0
        try:
            return self.client.chat_with_metadata(system, user, **kwargs)
        finally:
            self.budget.record()
