"""User-facing entry point for testing the coding-agent runtime on an isolated checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent_runtime import AgentRuntimeError, run_coding_agent
from .agent_roles import ROLE_PROFILES
from .workspace_snapshot import create as create_workspace_snapshot
from .common import read_json


def _validate_existing_snapshot(output: Path, source: Path) -> Path:
    marker = output / "workspace_snapshot.json"
    workspace = output / "checkout"
    if (marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 512 * 1024 or
            workspace.is_symlink() or not workspace.is_dir()):
        raise ValueError("resume requires the original isolated checkout and snapshot manifest")
    manifest = read_json(marker)
    if (not isinstance(manifest, dict) or
            Path(str(manifest.get("source") or "")).expanduser().resolve() != source or
            Path(str(manifest.get("destination") or "")).expanduser().resolve() != workspace.resolve()):
        raise ValueError("resume repository identity differs from the original agent session")
    return workspace.resolve(strict=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autosim agent",
        description="Run one coding-agent turn on a tracked-only isolated checkout; this does "
                    "not run benchmark scoring or claim research results.")
    parser.add_argument("repo", type=Path, help="repository to copy as an isolated checkout")
    parser.add_argument("output", type=Path, help="new run directory, reused only with --resume")
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--prompt", help="task description for this coding-agent turn")
    task.add_argument("--prompt-file", type=Path, help="UTF-8 task description file")
    parser.add_argument("--resume", action="store_true",
                        help="continue the saved coding-agent session and exact checkout")
    parser.add_argument("--include-untracked", action="store_true",
                        help="copy untracked worktree files too; sensitive-file audit still applies")
    parser.add_argument("--copy-limit-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--resource", action="append", default=[],
                        metavar="RELATIVE_TARGET=/ABSOLUTE/SOURCE",
                        help="explicit read-only resource attachment, not copied code")
    parser.add_argument("--timeout", type=float, default=600,
                        help="wall-clock limit for this coding-agent turn")
    parser.add_argument("--max-budget-usd", type=float, default=0.50,
                        help="per-turn model cost ceiling (DeepSeek: local official-rate estimate)")
    parser.add_argument("--max-total-budget-usd", type=float,
                        help="optional persistent total model-cost ceiling across resumed turns")
    parser.add_argument("--role", choices=[name for name, profile in ROLE_PROFILES.items()
                                           if profile.can_run_agent], default="scheduler",
                        help="AutoSOTA responsibility profile; does not imply a full workflow")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    source = args.repo.expanduser().resolve()
    raw_output = args.output.expanduser().absolute()
    if raw_output.is_symlink():
        parser.error("output directory may not be a symlink")
    output = raw_output.resolve()
    if not source.is_dir():
        parser.error("repository must be an existing directory")
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("isolated run output and source repository must be disjoint")
    if args.copy_limit_bytes <= 0 or args.timeout <= 0 or args.max_budget_usd <= 0:
        parser.error("copy, wall-time and API-cost limits must be positive")
    try:
        from .workspace_resources import parse_binding, bindings_for, validate_bindings
        resources = [parse_binding(value) for value in args.resource]
        if args.resume:
            workspace = _validate_existing_snapshot(output, source)
            saved = bindings_for(workspace)
            if resources and validate_bindings(source, workspace, resources) != saved:
                raise ValueError("resource bindings differ from saved run")
        else:
            if output.exists():
                parser.error("output already exists; use --resume or choose a fresh run directory")
            output.mkdir(parents=True)
            workspace = output / "checkout"
            create_workspace_snapshot(source, workspace,
                                     max_bytes=args.copy_limit_bytes,
                                     resources=resources,
                                     tracked_only=not args.include_untracked)
        prompt = (Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
                  if args.prompt_file else args.prompt)
        result = run_coding_agent(workspace=workspace, output=output, prompt=prompt,
                                  run_id=output.name, timeout=args.timeout,
                                  max_budget_usd=args.max_budget_usd, resume=args.resume,
                                  max_total_budget_usd=args.max_total_budget_usd,
                                  role=args.role)
    except (AgentRuntimeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"agent run refused or failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
