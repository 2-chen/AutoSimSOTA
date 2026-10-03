#!/usr/bin/env python3
"""The three things this system does, and nothing that names a benchmark.

**research** runs a research loop on a checkout. **scout** reads an unfamiliar checkout and
reports what it can do. **recheck** runs one already-scored measurement again, in a fresh
directory, from the run's own records. **agent** runs one coding-agent turn on a tracked-only
isolated checkout for runtime development; it does not score a benchmark. These verbs answer
separate questions: scout asks whether a benchmark can support a loop, research runs one, recheck
tests a measurement, and agent exposes the coding runtime without implying research evidence.

There used to be four more commands and a bare `autosim --repo ... --task ...` form. Each of
them described one benchmark -- its task names, its epoch counts, its adapter, its simulator --
in an argument parser, and each reached a parallel implementation of work this package already
does generally. A system that carries one benchmark's defaults in its own command line cannot
be pointed at a different one, and the defaults go stale without anything saying so.

What is left says nothing about any benchmark. Which stages exist, how they are invoked, and
what may vary are all read from the checkout by the code behind them.
"""

import sys


def main(argv: list[str] | None = None) -> int:
    """`argv` as an argument rather than only from `sys.argv`, so the dispatcher itself can be
    tested -- a verb that is wired up and a verb that is reachable are different claims."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        print("\n  autosim research <repo> <output-dir> [rounds] [settings-json] [--wall-seconds N]")
        print("  autosim scout    <repo> [--read-limit N]")
        print("  autosim recheck  <run-root> <label> --output <fresh-dir> [--plan-only]")
        print("  autosim agent    <repo> <output-dir> (--prompt TEXT | --prompt-file FILE) [--resume]")
        print("  autosim environments list | register --prefix <existing-env> [--store PATH]")
        print("  autosim environments verify --prefix <existing-env> --profile <family> --output <fresh-dir> [--gpu-seconds N]")
        return 0

    if argv[0] == "research":
        from .research.derive_and_run import main as research_main

        return research_main(argv[1:])
    if argv[0] == "environments":
        from .research.environment_pool import main as environments_main
        return environments_main(argv[1:])

    if argv[0] == "scout":
        from .research.scout import main as scout_main

        return scout_main(argv[1:])

    if argv[0] == "recheck":
        from .research.reevaluate import main as recheck_main

        return recheck_main(argv[1:])

    if argv[0] == "agent":
        from .research.agent_cli import main as agent_main

        return agent_main(argv[1:])

    print(f"unknown command: {argv[0]!r}", file=sys.stderr)
    print("  autosim research <repo> <output-dir> [rounds] [settings-json] [--wall-seconds N]")
    print("  autosim scout    <repo> [--read-limit N]")
    print("  autosim recheck  <run-root> <label> --output <fresh-dir> [--plan-only]")
    print("  autosim agent    <repo> <output-dir> (--prompt TEXT | --prompt-file FILE) [--resume]")
    return 2


if __name__ == "__main__":
    sys.exit(main())
