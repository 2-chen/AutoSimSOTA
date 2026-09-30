"""AutoSOTA role contracts and capability profiles for one shared coding runtime.

Roles are responsibilities, not a requirement to spawn eight model processes. The Scheduler
is the single research orchestrator; this table narrows each delegated turn's tools. Recorder
may narrate a supplied evidence snapshot, but has no filesystem or execution tools.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RoleProfile:
    name: str
    title: str
    can_edit_checkout: bool
    can_execute_diagnostics: bool
    can_run_agent: bool
    responsibility: str

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
                    "mcp__autosim_exec__read_public_source")
        return (*evidence, "mcp__autosim_exec__run_command", "mcp__autosim_exec__inspect_native_environment") \
            if self.can_execute_diagnostics else (
                (*evidence, "mcp__autosim_exec__inspect_native_environment")
                if self.name == "scheduler" else evidence)

    @property
    def instruction(self) -> str:
        limits = []
        if not self.can_edit_checkout:
            limits.append("you are read-only and must not propose that you applied a patch")
        if not self.can_execute_diagnostics:
            limits.append("you cannot execute commands; cite existing receipts or request a probe")
            if self.name == "scheduler":
                limits.append("exception: inspect_native_environment performs bounded read-only CPU inspection of the actual selected environment; it cannot edit, use GPU or access network")
        constraint = "; ".join(limits) or (
            "you may patch only the isolated checkout and run bounded CPU diagnostics; "
            "scoring, GPU workloads and budget approval remain with the trusted executor")
        return (f"You are {self.title}. {self.responsibility} Act only within your role. "
                f"Permission boundary: {constraint}. Return a concise handoff with evidence, "
                "uncertainty and the next decision needed; do not invent missing evidence. "
                "Use /tmp/diagnostics for CPU diagnostic venvs and scratch files through "
                "the execution tool; never create them inside the source checkout. "
                "For noninteractive native setup, inspect repository config initialization, "
                "Use inspect_native_environment for the actual selected interpreter and "
                "run-local configuration, not system Python or a new diagnostic venv. "
                "For unknown public resources/methods, search_public_sources then "
                "read_public_source; cite sealed sources. Search hits are not verified "
                "papers/resources. Never submit secrets, private paths or raw data in queries. "
                "If retrieval fails, report network capability as unavailable, not that "
                "the resource does not exist. Binary downloads use a native acquisition plan. "
                "prepare run-local config and use the same config context in every probe "
                "and stage. An EOF/input prompt is not proof the repository cannot run. "
                "After repair revalidate the original failed operation, not a weaker substitute.")


ROLE_PROFILES = {
    "resource": RoleProfile("resource", "AgentResource", False, False, True,
                             "Identify repository assets/dependencies and data supply: native "
                             "collector, expert/planner or policy, prerequisites, conversion "
                             "and actual trainer loader. Report source evidence and gaps."),
    "objective": RoleProfile("objective", "AgentObjective", False, False, True,
                              "Translate the user goal and native evaluator into a rubric and redlines."),
    "init": RoleProfile("init", "AgentInit", True, True, True,
                         "Prepare the reproducible baseline and verify producer-to-consumer "
                         "data paths; request bounded native collection/loader probes from "
                         "Scheduler without changing evaluation rules."),
    "monitor": RoleProfile("monitor", "AgentMonitor", False, False, True,
                             "Assess progress, errors, time/resource budgets and whether to resume or stop."),
    "fix": RoleProfile("fix", "AgentFix", True, True, True,
                        "Diagnose a recorded failure, make a justified repair and CPU probe; "
                        "handoff to Scheduler for trusted native revalidation."),
    "ideator": RoleProfile("ideator", "AgentIdeator", False, False, True,
                            "Propose falsifiable optimization ideas from source and verified "
                            "results; consider an early data intervention when autonomous "
                            "collection is legal, feasible and usable by the trainer."),
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
