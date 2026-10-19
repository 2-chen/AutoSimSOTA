"""AutoSOTA role contracts and capability profiles for one shared coding runtime.

Roles are responsibilities, not a requirement to spawn eight model processes. The Scheduler
is the single research orchestrator; this table narrows each delegated turn's tools. Recorder
may narrate a supplied evidence snapshot, but has no filesystem or execution tools.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .prompt_library import load_prompt


@dataclass(frozen=True)
class RoleProfile:
    name: str
    title: str
    can_edit_checkout: bool
    can_execute_diagnostics: bool
    can_run_agent: bool
    responsibility: str

    @property
    def can_inspect_native(self) -> bool:
        return self.can_execute_diagnostics or self.name in {"scheduler", "fix", "init", "data", "objective"}

    @property
    def native_source_only(self) -> bool:
        return self.name == 'objective'

    @property
    def builtin_tools(self) -> tuple[str, ...]:
        if self.name == "recorder":
            return ()
        base = ("Read", "Glob", "Grep")
        if self.can_edit_checkout:
            return (*base, "Edit", "Write")
        return base

    @property
    def mcp_tools(self) -> tuple[str, ...]:
        if self.name == "recorder":
            return ()
        evidence = ("mcp__autosim_exec__read_evidence", "mcp__autosim_exec__search_public_sources",
                    "mcp__autosim_exec__read_public_source", "mcp__autosim_exec__inspect_workspace_resources")
        return (*evidence, "mcp__autosim_exec__run_command", "mcp__autosim_exec__inspect_native_environment") \
            if self.can_execute_diagnostics else (
                (*evidence, "mcp__autosim_exec__inspect_native_environment")
                if self.can_inspect_native else evidence)

    def instruction_for(self, *, decision_only: bool = False) -> str:
        if not decision_only:
            return self.instruction
        return (f"You are {self.title}. {self.responsibility} "
                "This turn has no tools, filesystem access or execution capability. "
                "Decide only from the supplied evidence; identify any missing check "
                "without claiming it ran. Follow the current request's response schema.")

    @property
    def instruction(self) -> str:
        # Caller schemas differ across decision, audit and narrative turns. Do not
        # impose a second generic output format in the runtime's appended contract.
        parts = [f"You are {self.title}. {self.responsibility}",
                 "Use the current request's response schema; do not invent evidence."]
        if self.name == "recorder":
            return " ".join(parts + [
                "You have no filesystem, network or execution tools. "
                "Use only the supplied development snapshot; request declarative "
                "charts/demos instead of trying to create them yourself."])
        if self.can_edit_checkout:
            parts.append("Edits are limited to the isolated checkout. Scoring, GPU "
                         "workloads, live environment installs and budget approval "
                         "remain with the trusted executor.")
        else:
            parts.append("You are read-only; do not claim to have applied a patch.")
        if self.can_execute_diagnostics:
            parts.append("Use bounded CPU diagnostics and /tmp/diagnostics for "
                         "scratch/diagnostic venvs, not the checkout or live prefix.")
        else:
            parts.append("Do not execute commands; cite existing receipts or "
                         "request a trusted probe.")
        if self.can_inspect_native:
            parts.append("inspect_native_environment is SOURCE-ONLY for this role: "
                         "read the published dependency catalog or capture exact public source. "
                         "No code, policy/data grants, GPU, installs or configuration writes. "
                         "Use the supplied confirmation_reservation for the original intent; "
                         "stale citations are not permission to replace its conditions."
                         if self.native_source_only else
                         "inspect_native_environment is the bounded read-only CPU "
                         "exception: use the actual selected interpreter and run-local "
                         "config. If context is unpublished, request preparation "
                         "instead of repeating inspection.")
        if self.name in {"init", "fix"}:
            parts.append("An EOF/config prompt is not proof of an impossible "
                         "environment. Propose missing installs through Scheduler, "
                         "not diagnostic prefix writes. Revalidate the failed native "
                         "capability after repair; an evidence-reviewed install route "
                         "may replace an obsolete command without weakening consumers "
                         "or the scientific protocol.")
        parts.append(load_prompt("runtime-tools"))
        return "\n\n".join(parts)


ROLE_PROFILES = {
    "data": RoleProfile("data", "Data Agent", False, False, True,
                         "Own the delegated data lifecycle: acquisition feasibility and targeted "
                         "supply, quality/coverage, conversion, actual loader consumption, lineage "
                         "and preservation. Revisit obsolete deferrals using current development "
                         "evidence; return actionable probes and unresolved blockers to Scheduler. "
                         "Never collect, train, delete data, alter evaluation or certify scores yourself."),
    "resource": RoleProfile("resource", "AgentResource", False, False, True,
                             "Own local GPU/wall/model/storage allocation, extension and stall "
                             "cancellation recommendations within hard caps; distinguish startup "
                             "from full training. Scheduler executes; never mutate ledger or scores. "
                             "Identify repository assets/dependencies and data supply: native "
                             "collector, expert/planner or policy, prerequisites, conversion "
                             "and actual trainer loader. Report source evidence and gaps."),
    "objective": RoleProfile("objective", "AgentObjective", False, False, True,
                              "Translate the user goal and native evaluator into a rubric and redlines."),
    "init": RoleProfile("init", "AgentInit", True, True, True,
                         "Prepare the reproducible baseline and verify producer-to-consumer "
                         "data paths; request bounded native collection/loader probes from "
                         "Scheduler without changing evaluation rules. Reuse unchanged source-backed "
                         "workflow and handoff memory; probe an existing compatible interpreter first, "
                         "then install only dependencies demonstrated missing or incompatible. "
                         "A pending handoff is not an environment failure: validate bounded native "
                         "consumers first, inspect external SDK save/load after it is available, "
                         "and require actual artifact loading and rollout before scoring. "
                         "Do not train or demand future checkpoint files during environment setup. "
                         "Separate bounded command verification from formal learning: hand off the "
                         "native learning units, source defaults and measured cost, never declare "
                         "a probe checkpoint converged or silently reuse probe work as the baseline."),
    "monitor": RoleProfile("monitor", "AgentMonitor", False, False, True,
                             "Assess progress, errors, time/resource budgets and whether to resume or stop."),
    "fix": RoleProfile("fix", "AgentFix", True, True, True,
                        "Diagnose a recorded failure, make a justified repair and CPU probe; "
                        "handoff to Scheduler for trusted native revalidation."),
    "ideator": RoleProfile("ideator", "AgentIdeator", False, False, True,
                            "Propose falsifiable optimization ideas from source and verified "
                            "results; consider an early data intervention when autonomous "
                            "collection is legal, feasible and usable by the trainer. Distinguish "
                            "metric-improving hypotheses from diagnostic or throughput work; give "
                            "a causal metric expectation. Use current formal development evidence, "
                            "not smoke scores. At a floor score check adequate training, expert/data "
                            "and deployment semantics; code patches must cite actual literal source, "
                            "not descriptions of a block to replace."),
    "scheduler": RoleProfile("scheduler", "AgentScheduler", True, True, True,
                              "Own the global research plan and unresolved questions; inspect "
                              "evidence, investigate with tools, assign specialist tasks and "
                              "review handoffs before selecting the next legal experiment."),
    "supervisor": RoleProfile("supervisor", "AgentSupervisor", False, False, True,
                               "Independently audit a candidate, protocol, receipts and claims; never modify the candidate."),
    "recorder": RoleProfile("recorder", "Recorder", False, False, True,
                             "Explain only the supplied development evidence in Chinese; "
                             "request declarative charts or demos. No tools, simulation, "
                             "scoring, budget changes or access to held-out evidence."),
}


def role_profile(value: str, *, read_only: bool = False) -> RoleProfile:
    try:
        profile = ROLE_PROFILES[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown AutoSOTA role: {value}") from exc
    if not profile.can_run_agent:
        raise ValueError("Recorder is a deterministic reporting service, not a coding-agent turn")
    return (replace(profile, can_edit_checkout=False, can_execute_diagnostics=False)
            if read_only else profile)
