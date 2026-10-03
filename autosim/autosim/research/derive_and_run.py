"""Run a research loop on a benchmark nobody wrote a runner for.

Usage: `autosim research <repo> <output-dir> [rounds] [settings-json]`

There used to be two ways in. This file was one: a driver that assumed a declaration had
already been scouted, an environment already built and a checkout already read, checked that
each of those was true, and ended with `run provisioning first` when one was not. The other was
a tool script that did those things in a different order for one benchmark. Neither could do
the whole job, and the boundary between them was maintained by hand.

What is here now is an entry point and nothing else: it assembles the facts the preparation
loop needs -- where the checkout is, where the records go, which device this machine will run
on -- and hands off to `prepare.Preparation`, which takes the five steps in whatever order the
model can justify and records what it found.

The flags that remain are about *what to spend*, not about what to do: `--rederive` discards
what an earlier run kept, `--keep-only` runs the loop on what is already kept without deriving
anything new. Both used to be branches in a sequence; they are now instructions the loop is
given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import read_json

ROOT = Path(__file__).resolve().parents[3]


def validate_paused_research_resume(output: Path, *, rounds: int,
                                    settings: dict[str, Any],
                                    repository: Path | None = None,
                                    rederive: bool = False) -> int | None:
    """Reject protocol drift before the outer controller records or starts an action.

    The inner research engine validates its frozen identity, but discovering a mismatch only
    after `run_the_loop` begins leaves a stale outer `current_action` and a raised receipt.
    Paused sessions have enough durable state to reject changed rounds/settings up front;
    running/interrupted states remain with the controller's reconciliation path.
    """
    output = Path(output).expanduser().resolve()
    root = output / "research" / "derived"
    session_path = root / "controller_session.json"
    for parent in (output, output / "research", root):
        if parent.is_symlink():
            raise ValueError("paused research state is behind a symlink; refusing resume")
    if session_path.is_symlink():
        raise ValueError("paused research session is a symlink; refusing resume")
    if not session_path.is_file():
        return
    if (not session_path.resolve(strict=True).is_relative_to(output) or
            session_path.stat().st_size > 2 * 1024 * 1024):
        raise ValueError("research session path or size is unsafe; refusing resume")
    session = read_json(session_path)
    if not isinstance(session, dict) or session.get("schema_version") != 1:
        raise ValueError("research session identity/schema is invalid; refusing resume")
    if rederive:
        raise ValueError("an existing research session cannot be re-derived in place; "
                         "start a new output directory")
    if session.get("status") != "paused":
        return
    prior_settings = session.get("base_settings")
    if (session.get("run_id") != "derived" or
            not isinstance(session.get("rounds"), int) or
            not isinstance(prior_settings, dict)):
        raise ValueError("paused research session lacks frozen inputs; "
                         "start a new output directory")
    if repository is not None:
        expected_repo = Path(repository).expanduser().resolve()
        snapshot_path = output / "workspace_snapshot.json"
        if snapshot_path.is_symlink():
            raise ValueError("workspace snapshot is a symlink; refusing resume")
        if snapshot_path.is_file():
            if (not snapshot_path.resolve(strict=True).is_relative_to(output) or
                    snapshot_path.stat().st_size > 512 * 1024):
                raise ValueError("workspace snapshot path or size is unsafe; refusing resume")
            snapshot = read_json(snapshot_path)
            if not isinstance(snapshot, dict):
                raise ValueError("workspace snapshot is invalid; refusing resume")
            source_value = str(snapshot.get("source") or "").strip()
            destination_value = str(snapshot.get("destination") or "").strip()
            if (not source_value or not destination_value or
                    Path(source_value).expanduser().resolve() != expected_repo or
                    Path(destination_value).expanduser().resolve() !=
                    (output / "checkout").resolve()):
                raise ValueError("paused research session has a different isolated repository; "
                                 "start a new output directory")
            expected_repo = (output / "checkout").resolve()
        saved_repo = session.get("repository")
        if (not isinstance(saved_repo, str) or not saved_repo or
                Path(saved_repo).expanduser().resolve() != expected_repo):
            raise ValueError("paused research session repository identity differs; "
                             "resume with the original repository or start a new output")
    if prior_settings != settings:
        raise ValueError("paused research session has different frozen rounds/settings; "
                         "resume with the original inputs or start a new output directory")
    if session['rounds']!=rounds:
        from .round_extension import resume_rounds
        return resume_rounds(root,session,rounds)
    return rounds


def _run_local_interpreter(source_python: Path, output: Path, repo: Path | None = None) -> Path:
    """Borrow dependencies without inheriting stale editable imports/startup hooks."""
    source_python = Path(source_python).expanduser().absolute()
    if not source_python.is_file():
        raise ValueError(f"--interpreter does not exist: {source_python}")
    local_env = output / "env"
    from .environment_pool import describe
    from .environment_overlay import create, readonly_roots
    repo = repo or output/'checkout'
    if local_env.exists():
        manifest = local_env/'overlay.json'
        if not manifest.is_file() or read_json(manifest).get('base_prefix') != str(source_python.parent.parent):
            raise ValueError('legacy or different run-local environment; explicitly switch base instead of overwriting')
        readonly_roots(local_env, output, repo)
    else:
        create(describe(source_python.parent.parent), prefix=local_env, output=output,
               repo=repo, source_binding_ids=[])
    return local_env/'bin/python'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autosim research")
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("rounds", nargs="?", type=int, default=2)
    parser.add_argument("settings_json", nargs="?", default="{}")
    parser.add_argument("--keep-only", action="store_true")
    parser.add_argument("--rederive", action="store_true")
    parser.add_argument("--wall-seconds", type=float,
                        help="persisted wall deadline; new AutoSOTA runs default to 48 hours, "
                             "legacy runs to one hour; resume preserves the recorded limit")
    parser.add_argument("--environment-store", type=Path,
                        help="shared read-only wheel/base/snapshot store (outside this run)")
    parser.add_argument("--no-environment-reuse", action="store_true",
                        help="disable catalog, package cache and snapshot reuse for a fresh run")
    parser.add_argument('--publish-environment-snapshots', action='store_true', default=None,
                        help='opt in to large conda snapshot copies; default publishes small base/delta templates only')
    parser.add_argument("--max-actions", type=int, default=8,
                        help="maximum Scheduler actions per supervised segment; the AutoSOTA "
                             "Monitor resumes additional segments within the frozen budgets")
    parser.add_argument("--max-relaunch", type=int,
                        help="maximum automatic Scheduler segment relaunches; on the AutoSOTA "
                             "framework the default is rounds + 2")
    parser.add_argument("--confirm", action="store_true",
                        help="after the rounds, measure the best candidate once on the "
                             "held-out settings the declaration reserves")
    parser.add_argument("--interpreter", type=Path,
                        help="base Python for a run-local virtual environment; "
                             "does not skip dependency or simulator probes")
    isolation = parser.add_mutually_exclusive_group()
    isolation.add_argument("--isolated-copy", action="store_true", default=True,
                           help="default: run against a bounded independent source copy")
    isolation.add_argument("--in-place", action="store_false", dest="isolated_copy",
                           help="legacy only: explicitly allow modification of the source "
                                "repository; never permitted with autosota_sim_v1")
    copy_mode = parser.add_mutually_exclusive_group()
    copy_mode.add_argument("--tracked-copy", action="store_true",
                        help="with --isolated-copy, copy Git-tracked worktree files only; "
                             "untracked datasets and checkpoints remain external")
    copy_mode.add_argument("--full-copy", action="store_true",
                          help="explicitly copy untracked files too, bounded by copy limit")
    parser.add_argument("--resource", action="append", default=[],
                        metavar="RELATIVE_TARGET=/ABSOLUTE/SOURCE",
                        help="attach a file/directory read-only inside the isolated checkout; "
                             "repeat for multiple resources; never overlays copied code")
    parser.add_argument("--copy-limit-bytes", type=int, default=4 * 1024**3,
                        help="maximum source bytes copied with --isolated-copy")
    parser.add_argument("--framework", choices=("legacy_v1", "autosota_sim_v1"),
                        default="legacy_v1",
                        help="select the opt-in AutoSOTA coding-agent runtime; it requires "
                             "an isolated repository copy")
    parser.add_argument("--agent-turn-budget-usd", type=float, default=4.0,
                        help="initial local allowance; DeepSeek can expand it within the run ceiling")
    parser.add_argument("--agent-total-budget-usd", type=float, default=60.0,
                        help="persisted provider-cost ceiling shared by all roles in this run")
    parser.add_argument("--agent-timeout-seconds", type=float, default=300.0,
                        help="wall-clock limit for one coding-agent turn")
    parser.add_argument("--budget-scope", choices=("task", "repository"),
                        help="new runs default to independent task GPU budgets; old runs "
                             "retain repository-wide wall-time accounting")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rounds = args.rounds
    if (rounds < 0 or args.max_actions <= 0 or
            (args.max_relaunch is not None and args.max_relaunch < 0) or
            (args.wall_seconds is not None and args.wall_seconds <= 0) or
            args.copy_limit_bytes <= 0):
        parser.error("rounds/relaunch limits must be non-negative and action/wall/copy limits "
                     "must be positive")
    if (args.tracked_copy or args.full_copy or args.resource) and not args.isolated_copy:
        parser.error("copy modes and resources require --isolated-copy")
    if (args.agent_turn_budget_usd <= 0 or args.agent_total_budget_usd <= 0 or
            args.agent_timeout_seconds <= 0):
        parser.error("coding-agent turn/total budgets and timeout must be positive")
    if args.framework == "autosota_sim_v1" and not args.isolated_copy:
        parser.error("--framework autosota_sim_v1 requires --isolated-copy so the coding "
                     "agent cannot modify the supplied repository")
    if args.isolated_copy and (output.is_relative_to(repo) or repo.is_relative_to(output)):
        parser.error("isolated run output and source repository must be disjoint")
    # The settings the baseline is measured at. A benchmark's own defaults decide what a
    # measurement costs, so the caller choosing them is the caller choosing the experiment.
    settings = json.loads(args.settings_json)
    if not isinstance(settings, dict):
        parser.error("settings-json must be an object")
    if args.keep_only and args.rederive:
        print("--keep-only and --rederive contradict each other; pick one")
        return 2
    try:
        persisted_rounds = validate_paused_research_resume(output, rounds=rounds, settings=settings,
                                        repository=repo,
                                        rederive=args.rederive)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if persisted_rounds is not None:
        rounds=persisted_rounds
    from ..llm_client import LLMClient, PROJECT_ROOT, load_credential_file
    from .budget import RunBudget
    from .prepare import Preparation
    from .repository_budget import RepositoryBudget

    load_credential_file(project_root=PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    for amendment_path in (output / 'budget_amendments').glob('*.json'):
        if amendment_path.is_symlink() or read_json(amendment_path).get('status') != 'applied':
            parser.error('unfinished/unsafe explicit budget amendment; reconcile before resume')
    wall_seconds = args.wall_seconds
    if wall_seconds is None:
        old_budget = output / "budget.json"
        wall_seconds = (float(json.loads(old_budget.read_text())["wall_seconds"])
                        if old_budget.is_file() else
                        (172800.0 if args.framework == "autosota_sim_v1" else 3600.0))
    from .common import atomic_json
    from .task_budget import TaskGPUBudget
    scope_path = output / "budget_scope.json"
    saved_scope = (read_json(scope_path)["scope"] if scope_path.is_file() else
                   "repository" if (output / "budget.json").exists() else None)
    scope = args.budget_scope or saved_scope or "task"
    if saved_scope and scope != saved_scope:
        parser.error("budget scope is frozen for this run; start a new study output")
    atomic_json(scope_path, {"schema_version": 1, "scope": scope,
                            "wall_pause_policy": "deadline_continues"})
    repo_budget = (RepositoryBudget(ROOT / "autoresearch_runs" / "resource_budgets",
                                    repo=repo) if scope == "repository" else None)
    reserved = False
    try:
        if repo_budget is not None:
            repo_budget.reserve(output, wall_seconds=wall_seconds)
            reserved = True
        else:
            TaskGPUBudget(output).initialize(repo)
        RunBudget(output, wall_seconds=wall_seconds)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if reserved:
            repo_budget.finish(output)
        parser.error(f"repository budget refused this run: {exc}")
    try:
        from .common import atomic_json, now
        atomic_json(output / "native_jobs" / "controller_finished.json",
                    {"at": now(), "run_id": "derived", "state": "running"})
        if args.framework == "autosota_sim_v1":
            from .runtime_preflight import check_runtime
            from .common import atomic_text
            preflight = check_runtime()
            atomic_json(output / "runtime_preflight.json", preflight)
            if preflight["status"] != "ready":
                # Preserve a previous research report when resuming; the standalone
                # startup report remains accessible even without any working Agent.
                failed = [row["name"] for row in preflight["checks"]
                          if not row["ok"] and row.get("required", True)]
                message = ("# 研究运行启动受阻\n\n启动检查未通过：" + "、".join(failed) +
                           "。尚未启动 Agent 或实验。\n\n"
                           "请在允许本机计费代理监听、模型联网及进程隔离的环境中续跑。\n\n"
                           "[检查详情](runtime_preflight.json)\n")
                atomic_text(output / "STARTUP.md", message)
                if not (output / "RUN.md").exists():
                    atomic_text(output / "RUN.md", message)
                print(message)
                return 2
        interpreter_hint = None
        if args.rederive:
            for name in ("derived_stages.json", "environment.json", "execution.json"):
                (output / name).unlink(missing_ok=True)
        if (output / "workspace_snapshot.json").is_file() and not args.isolated_copy:
            parser.error("this run uses an isolated working copy; pass --isolated-copy to resume")
        if args.isolated_copy:
            from .workspace_snapshot import create, existing
            from .workspace_resources import parse_binding, validate_bindings
            working = output / "checkout"
            resources = [parse_binding(value) for value in args.resource]
            if working.exists():
                manifest = existing(working, source=repo)
                expected_mode = ("tracked_worktree" if args.tracked_copy else
                                 "full_worktree" if args.full_copy else
                                 manifest.get("copy_mode", "full_worktree"))
                if manifest.get("copy_mode", "full_worktree") != expected_mode:
                    parser.error("isolated copy mode differs from the saved working copy")
                if resources and validate_bindings(repo, working, resources) != manifest.get(
                        "resource_bindings", []):
                    parser.error("resource bindings differ from saved run; choose a new output")
            elif args.keep_only:
                parser.error("--keep-only --isolated-copy requires an existing working copy")
            else:
                if any((output / name).exists() for name in (
                        "declaration.json", "execution.json", "derived_stages.json")):
                    parser.error("existing direct-run records cannot be rebound to a fresh copy")
                create(repo, working, max_bytes=args.copy_limit_bytes,
                       tracked_only=not args.full_copy, resources=resources)
            repo = working
        if args.interpreter:
            try:
                interpreter_hint = _run_local_interpreter(args.interpreter, output, repo)
            except (OSError, ValueError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as exc:
                parser.error(f"run-local interpreter setup failed: {exc}")
        if args.framework == "autosota_sim_v1" or args.environment_store is not None:
            from .environment_pool import configure
            try:
                configure(output, args.environment_store,
                          enabled=not args.no_environment_reuse,
                          publish_snapshots=args.publish_environment_snapshots,
                          source_repository=(Path(read_json(output / "workspace_snapshot.json")["source"])
                              if (output / "workspace_snapshot.json").is_file() else repo))
            except (OSError, ValueError) as exc:
                parser.error(f"environment store configuration failed: {exc}")
        if args.framework == "autosota_sim_v1" and not (output / "run_state.json").exists():
            from .scheduling import configure as configure_scheduler
            configure_scheduler(output)
        client: Any = LLMClient()
        if args.framework == "autosota_sim_v1":
            from .agent_client import RoleAwareAgentClient
            try:
                client = RoleAwareAgentClient(
                    workspace=repo, output=output, run_id="derived",
                    turn_budget_usd=args.agent_turn_budget_usd,
                    total_budget_usd=args.agent_total_budget_usd,
                    timeout=args.agent_timeout_seconds)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                parser.error(f"AutoSOTA Runtime refused this run: {type(exc).__name__}: {exc}")
        # AutoSOTA runs own their Resource/Onboard evidence.  The legacy path shares a
        # project-level scout cache for compatibility, but a formal run must not consume a
        # declaration produced for another checkout or leave its only copy outside the run
        # that is expected to resume and export it.
        scouting = (output / "scouting" if args.framework == "autosota_sim_v1" else
                    ROOT / "autoresearch_runs" / "scouting")
        preparation = Preparation(repo=repo, output=output, client=client,
                                  scouting=scouting,
                                  base_settings={**settings, "_rounds": rounds,
                                                 "_confirm": bool(args.confirm)},
                                  keep_only=args.keep_only,
                                  wall_seconds=wall_seconds,
                                  interpreter_hint=interpreter_hint)
        if args.keep_only:
            # A missing stage is a finding; kept records are reused without rederivation.
            kept = output / "derived_stages.json"
            if kept.is_file():
                for stage, row in json.loads(kept.read_text(encoding="utf-8")).items():
                    preparation.stages[stage] = row.get("source") or ""
                    preparation.parameters[stage] = row.get("parameters") or {}
        max_relaunch = (args.max_relaunch if args.max_relaunch is not None else
                        rounds + 2 if args.framework == "autosota_sim_v1" else 0)
        report = preparation.run(max_steps=args.max_actions,
                                 max_relaunch=max_relaunch)
        print(json.dumps(report, ensure_ascii=False, indent=2)[:6000])
        return 0 if report["status"] == "completed" else 1
    finally:
        from .native_jobs import active_jobs
        from .common import atomic_json, now
        atomic_json(output / "native_jobs" / "controller_finished.json",
                    {"at": now(), "run_id": "derived", "state": "finished"})
        if repo_budget is not None and not active_jobs(output):
            repo_budget.finish(output)
