"""Role-aware chat adapter backed by the persisted coding-agent runtime.

The adapter is intentionally thin: AutoSOTA roles share one configured model/runtime, while
each role gets its own resumable provider session and capability profile.  The outer research
controller still validates every decision and trusted executors still own benchmark scoring.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

from .common import atomic_json, now, sanitize_model_payload
from .deepseek_pricing import PRICE_CARD_ID


def run_model_budget_exhausted(exc: Exception) -> bool:
    """Only a run-level refusal or a verified empty run ledger is terminal."""
    if getattr(exc, "failure_category", "") == "run_model_budget":
        return True
    budget = getattr(exc, "run_budget", {}) or {}
    remaining = budget.get("remaining_usd")
    return (getattr(exc, "status", "") == "budget_exhausted" and
            isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and
            remaining <= 1e-9)


class AgentRuntimeClientError(RuntimeError):
    """A coding-agent turn did not produce a usable completed response."""

    def __init__(self, message: str, *, status: str = "unknown",
                 failure_category: str = "", run_budget: dict[str, Any] | None = None,
                 decision_attempt_id: str | None = None,
                 turn_id: str | None = None, process_ref: str | None = None,
                 event_count: int | None = None, runtime_failure: dict | None = None):
        super().__init__(message)
        self.status = status
        self.failure_category = failure_category
        self.run_budget = dict(run_budget or {})
        self.decision_attempt_id = decision_attempt_id
        self.turn_id = turn_id
        self.process_ref = process_ref
        self.event_count = event_count
        self.runtime_failure = runtime_failure or {}


class RoleAwareAgentClient:
    """Expose the LLMClient chat contract through one persistent coding-agent role session."""

    # Preparation may hand recoverable onboarding/command failures to AgentFix.  Keep this
    # explicit capability so older/fake role clients retain their existing contracts until
    # they opt into the same repair-and-audit semantics.
    supports_agent_fix = True
    supports_main_agent = True
    supports_recorder = True

    def __init__(self, *, workspace: Path, output: Path, run_id: str,
                 turn_budget_usd: float, total_budget_usd: float,
                 timeout: float = 300.0):
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        self.output = Path(output).expanduser().resolve(strict=True)
        if (self.workspace == self.output or
                not self.workspace.is_relative_to(self.output)):
            raise ValueError("agent runtime workspace must be nested under its run output")
        if not run_id.strip():
            raise ValueError("agent runtime run_id is required")
        if turn_budget_usd <= 0 or total_budget_usd <= 0 or timeout <= 0:
            raise ValueError("agent runtime budgets and timeout must be positive")
        self.run_id = run_id
        self.turn_budget_usd = float(turn_budget_usd)
        self.total_budget_usd = float(total_budget_usd)
        self.timeout = float(timeout)
        self.model = (os.environ.get("DEEPSEEK_MODEL") or
                      os.environ.get("AUTOSIM_LLM_MODEL") or "deepseek-flash")
        self._role = "scheduler"
        self._research_context: dict[str, Any] = {}
        self._freeze_runtime_spec()
        self.budget_output = self.output

    def fork_readonly(self, *, output: Path, workspace: Path, role: str):
        """Independent role/session state, but the original run's atomic cost ledger."""
        from .main_agent import READ_ONLY_ROLES
        if role not in READ_ONLY_ROLES | {"recorder", "fix"}:
            raise ValueError("isolated agent fork must be read-only")
        if not Path(output).resolve().is_relative_to(self.budget_output / "agent_workers"):
            raise ValueError("agent fork must live in the parent worker namespace")
        child = RoleAwareAgentClient(workspace=workspace, output=output, run_id=self.run_id,
            turn_budget_usd=self.turn_budget_usd, total_budget_usd=self.total_budget_usd,
            timeout=self.timeout)
        child.budget_output = self.budget_output
        child._role = role
        return child

    def set_research_context(self, context: dict[str, Any]) -> None:
        """Refresh cross-role context; provider session memory is not the source of truth."""
        self._research_context = sanitize_model_payload(
            context, local_roots=(self.workspace, self.output))

    def _freeze_runtime_spec(self) -> None:
        agent_root = self.output / "agent"
        if agent_root.is_symlink():
            raise ValueError("agent runtime directory is a symlink")
        agent_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = agent_root / "runtime_spec.json"
        if path.is_symlink():
            raise ValueError("agent runtime specification is a symlink")
        expected = {
            "schema_version": 1,
            "framework": "autosota_sim_v1",
            "runtime": "claude_code",
            "run_id": self.run_id,
            "workspace_identity": str(self.workspace),
            "model": self.model,
            "cost_basis": ("deepseek_official_estimate_v1" if self.model == "deepseek-flash"
                           else "cli_reported"),
            "price_card": PRICE_CARD_ID if self.model == "deepseek-flash" else None,
            "turn_budget_usd": self.turn_budget_usd,
            "total_budget_usd": self.total_budget_usd,
            "timeout_seconds": self.timeout,
        }
        if path.is_file():
            if path.stat().st_size > 64 * 1024:
                raise ValueError("agent runtime specification exceeds size limit")
            held = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(held, dict) or any(held.get(k) != v for k, v in expected.items()):
                raise ValueError("agent runtime inputs changed; resume with the frozen settings")
            return
        atomic_json(path, {**expected, "created_at": now()})

    @contextmanager
    def as_role(self, role: str) -> Iterator[None]:
        """Set the responsibility for one bounded call, restoring it on every exit path."""
        normalized = str(role).strip().lower()
        if not normalized or len(normalized) > 32:
            raise ValueError("agent role is malformed")
        from .agent_roles import role_profile
        role_profile(normalized)
        previous = self._role
        self._role = normalized
        try:
            yield
        finally:
            self._role = previous

    def _resume_role(self, role: str) -> bool:
        root = self.output / "agent"
        roles = root / "roles"
        pointer = roles / f"{role}.json"
        if root.is_symlink() or roles.is_symlink() or pointer.is_symlink():
            raise AgentRuntimeClientError("agent role-session path is unsafe")
        if not pointer.is_file():
            return False
        if pointer.stat().st_size > 16 * 1024:
            raise AgentRuntimeClientError("agent role-session pointer exceeds size limit")
        held = json.loads(pointer.read_text(encoding="utf-8"))
        if (not isinstance(held, dict) or held.get("role") != role or
                held.get("run_id") != self.run_id):
            raise AgentRuntimeClientError("agent role-session identity differs from this run")
        # Rehydrate from current controller facts instead of indefinitely accumulating
        # historical tool output. Old sessions and their receipts remain available.
        from .agent_runtime import _safe_session
        session = _safe_session(self.output, session_key=held.get("session_key"))
        relative = session.get("pricing_ref")
        if isinstance(relative, str) and relative.startswith("agent/pricing/"):
            path = self.output / relative
            if (path.is_file() and not path.is_symlink() and
                    path.resolve().is_relative_to(self.output) and path.stat().st_size < 1024*1024):
                pricing = json.loads(path.read_text(encoding="utf-8"))
                receipts = (pricing.get("gateway") or {}).get("receipts") or []
                usage = receipts[-1].get("usage") or {} if receipts else {}
                size = int(usage.get("input_miss_tokens") or 0) + int(usage.get("input_hit_tokens") or 0)
                if size >= 120_000:
                    atomic_json(root / "context_rotation.json", {
                        "role": role, "previous_session_key": held.get("session_key"),
                        "input_tokens": size, "reason": "rehydrate from current evidence",
                        "at": now()})
                    return False
        return True

    def chat_with_metadata(self, system: str, user: str, **kwargs: Any
                           ) -> tuple[str, dict[str, Any]]:
        """Run one real coding-agent turn and return its final text and safe usage metadata."""
        from .agent_runtime import run_coding_agent

        role = self._role
        prompt = (str(system).strip() + "\n\n## Current request and evidence\n" +
                  str(user).strip())
        if self._research_context and kwargs.get("include_research_context", True):
            prompt += ("\n\n## Shared research context (latest controller snapshot)\n" +
                       "Prior role-session recollections do not override this snapshot or "
                       "native evidence. Working plans and handoffs are model claims.\n" +
                       json.dumps(self._research_context, ensure_ascii=False, default=str))
        decision_attempt_id = kwargs.get("decision_attempt_id")
        try:
            result = run_coding_agent(
                workspace=self.workspace, output=self.output, prompt=prompt,
                run_id=self.run_id, timeout=min(self.timeout,
                                                float(kwargs.get("timeout", self.timeout))),
                max_budget_usd=self.turn_budget_usd,
                max_total_budget_usd=self.total_budget_usd,
                resume=self._resume_role(role), role=role,
                read_only=kwargs.get("read_only") is True or self.budget_output != self.output,
                auto_skills=kwargs.get("auto_skills", True) is True,
                decision_attempt_id=(str(decision_attempt_id)
                                     if decision_attempt_id is not None else None),
                **({"budget_output": self.budget_output}
                   if self.budget_output != self.output else {}))
        except Exception as exc:  # noqa: BLE001 - callers classify service/runtime failures.
            from .runtime_recovery import seal_failure
            from .agent_runtime import _worker_evidence_root
            fault = seal_failure(_worker_evidence_root(self.output), self.workspace, exc, role=role,
                                 decision_attempt_id=decision_attempt_id)
            raise AgentRuntimeClientError(
                f"{role} coding-agent turn failed before a result: "
                f"{type(exc).__name__}: {fault['message'][:600]}",
                status="failed", failure_category=fault["category"], runtime_failure=fault,
                decision_attempt_id=(str(decision_attempt_id)
                                     if decision_attempt_id is not None else None)) from exc
        status = str(result.get("status") or "unknown")
        content = str(result.get("final_text") or "").strip()
        metadata = {
            "available": True,
            "model": result.get("model") or self.model,
            "runtime": "claude_code",
            "role": role,
            "status": status,
            "session_id": result.get("session_id"),
            "turn_id": result.get("turn_id"),
            "process_ref": result.get("process_ref"),
            "decision_attempt_id": result.get("decision_attempt_id"),
            "event_count": result.get("event_count", result.get("events")),
            "total_cost_usd": result.get("total_cost_usd"),
            "usage": result.get("usage") or {},
            "run_budget": result.get("run_budget") or {},
            "returncode": result.get("returncode"),
            "failure_category": result.get("failure_category"),
        }
        if status != "completed":
            raise AgentRuntimeClientError(
                f"{role} coding-agent turn ended with status {status} "
                f"({result.get('failure_category') or 'no category'}); "
                f"{result.get('error') or 'no action is inferred'}",
                status=status,
                failure_category=str(result.get("failure_category") or ""),
                run_budget=metadata["run_budget"],
                decision_attempt_id=metadata["decision_attempt_id"],
                turn_id=metadata["turn_id"], process_ref=metadata["process_ref"],
                event_count=metadata["event_count"])
        if not content:
            raise AgentRuntimeClientError(
                f"{role} coding-agent turn completed without a final response")
        return content, metadata


@contextmanager
def role_scope(client: Any, role: str) -> Iterator[None]:
    """Switch only role-aware clients; legacy LLMClient behavior stays unchanged."""
    switch = getattr(client, "as_role", None)
    context = switch(role) if callable(switch) else nullcontext()
    with context:
        yield
