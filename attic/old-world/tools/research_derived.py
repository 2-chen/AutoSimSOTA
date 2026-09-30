"""Run a research loop on a benchmark nobody wrote a runner for.

Usage: `.venv/bin/python tools/research_derived.py <repo> <output-dir> [rounds]`

Assembles what was built about the benchmark -- the environment `provision` made, the stages
`execution_derive` found, the commands it generated and verified, and the optimisation space
`scout` declared -- into the loop that the two hand-written runners each implement once.

Nothing here is specific to a benchmark. Which stages exist is read; which settings may
vary is read; how they are spelled is the argv function's business.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autosim"))

from autosim.research import execution_derive, provision                  # noqa: E402
from autosim.research.survey import survey                                # noqa: E402
from autosim.research.execution_derive import blank_inputs                # noqa: E402
from autosim.research.declaration import (space_from,                     # noqa: E402
                                           usable_declarations)
from autosim.research.compute_decision import decide                      # noqa: E402
from autosim.research.declarative_backend import DeclarativeBackend       # noqa: E402
from autosim.research.derived_research import DerivedResearch             # noqa: E402


#: The parts of a compute decision that change what a command's verification established.
#: Not the whole environment: thread counts differ between runs on one machine and say
#: nothing about whether the command works.
_DEVICE_KEYS = ("CUDA_VISIBLE_DEVICES", "AUTOSIM_DEFAULT_CUDA_ORDINAL")


def _device_identity(decision) -> dict[str, str]:
    return {key: str(decision.environment.get(key, "")) for key in _DEVICE_KEYS}


def _same_machine(entry: dict, decision) -> bool:
    """Was this command verified under the device it is about to be run under?

    A record written before this distinction existed has no `verified_under`, and is treated
    as verified elsewhere -- which costs one re-derivation and is the honest default: the
    alternative is claiming an environment was tested that never was.
    """
    return entry.get("verified_under") == _device_identity(decision)


def _declared_task(declaration: dict) -> str:
    """The task the declaration names, so the verifier runs the benchmark's own choice.

    The value is read rather than written here: a driver that spells one benchmark's task
    name is the thing this file exists not to be.

    Every section is searched and not only `training`. RoboTwin declares no `training` axis
    ending in `benchmark_name` -- it has `task_name` under a `task_selection` section of its
    own -- so this returned the empty string and the verifier was handed no task at all.
    Matched by the shape of the name rather than by the section, because which section a
    benchmark puts the task in is the benchmark's business.
    """
    space = declaration.get("optimization_space") or {}
    for section, axes in space.items():
        for axis in axes or []:
            if not isinstance(axis, dict):
                continue
            name = str(axis.get("name", "")).lower()
            if name.endswith("benchmark_name") or name in ("task_name", "task"):
                if axis.get("default"):
                    return str(axis["default"])
    return ""


def main() -> int:
    # Flags are taken out before the positional arguments are read, so `--keep-only` can be
    # written wherever it is remembered rather than in the one position that happens not to
    # be a JSON document. Putting it in the settings slot raised a JSONDecodeError about a
    # character at line 1 column 1, which is a traceback about the wrong thing entirely.
    flags = {one for one in sys.argv[1:] if one.startswith("--")}
    positional = [one for one in sys.argv[1:] if not one.startswith("--")]
    if len(positional) < 2:
        print("usage: research_derived.py <repo> <output-dir> [rounds] [settings-json] "
              "[--keep-only] [--rederive]")
        return 2
    repo = Path(positional[0]).expanduser().resolve()
    output = Path(positional[1]).expanduser().resolve()
    rounds = int(positional[2]) if len(positional) > 2 else 1
    # The settings the baseline is measured at. A benchmark's own defaults decide what a
    # measurement costs: one epoch of LIBERO's lifelong loop is an hour and tells you
    # nothing, fifty are a day and tell you something. The caller choosing that is the
    # caller choosing the experiment, which is not the system's to guess.
    base = json.loads(positional[3]) if len(positional) > 3 else {}
    if "--keep-only" in flags and "--rederive" in flags:
        print("--keep-only and --rederive contradict each other; pick one")
        return 2

    interpreter = provision.env_python(output)
    if interpreter is None:
        print("no successful build here; run provisioning first")
        return 2

    execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    # Matched on the benchmark's own name as the declaration recorded it, and on the name
    # the checkout carries, rather than on how the directory happens to be spelled.
    wanted = repo.name.lower()
    candidates, rejected = usable_declarations(ROOT / "autoresearch_runs/scouting", wanted)
    for name, why in rejected:
        print(f"  skipping {name}: it does not build a usable space -- {why}", flush=True)
    if not candidates:
        print(f"no usable declaration names {repo.name}; run `autosim scout {repo}` first")
        return 2
    _, source_name, declaration = max(candidates, key=lambda row: row[0])
    print("declaration:", declaration.get("benchmark"), "from", source_name,
          f"({len(candidates)} candidate(s))", flush=True)

    from autosim.llm_client import LLMClient
    from autosim.research.repository_autoresearch import (PROJECT_ROOT,
                                                          load_deepseek_environment)
    load_deepseek_environment(project_root=PROJECT_ROOT)
    client = LLMClient()

    # What earlier runs established, kept. A verified command is the expensive product of
    # this whole file -- an hour of drafts and runs to find, and free to reuse -- and every
    # run used to start from nothing: `train` verified once in twelve attempts and the next
    # attempt threw it away. The environment is kept this way already; so is this.
    derived_path = output / "derived_stages.json"
    kept = json.loads(derived_path.read_text(encoding="utf-8")) if derived_path.is_file() else {}
    if "--rederive" in flags:
        kept = {}

    # Which device this runs on, decided from the machine once -- and handed to both the
    # verifier and the loop, so a command is never verified on one device and run on another.
    decision = decide()
    print("compute:", decision.device, "--", decision.why, flush=True)

    # A stage that consumes a checkpoint cannot be verified without one: the generated
    # function reads `i["checkpoint"]`, the verifier's slot for it was empty, and the program
    # then reported that the checkpoint it was handed does not exist -- so the loop revised
    # the one part of the command that was right. Offered here, once, from what the machine
    # actually has; the finding when there is none is about the benchmark's assets.
    asset_report = survey(repo)
    checkpoint = execution_derive.checkpoint_for_verification(asset_report,
                                                              declaration=declaration)
    print("checkpoint for verification:", checkpoint["path"] or "(none on this machine)",
          "--", checkpoint["note"], flush=True)
    verify_inputs = {**blank_inputs(), "python": str(interpreter), "repo": str(repo),
                     "task": _declared_task(declaration), "dataset": "",
                     "checkpoint": checkpoint["path"], "output": str(output / "v"),
                     "steps": 1, "episodes": 1, "device": decision.device,
                     "device_index": decision.device_index}

    stages, parameters = {}, {}
    for stage, row in sorted((execution.get("stages") or {}).items()):
        if not row.get("available"):
            continue
        if stage in kept and _same_machine(kept[stage], decision):
            stages[stage] = kept[stage]["source"]
            parameters[stage] = kept[stage]["parameters"]
            execution["stages"][stage] = kept[stage]["row"]
            print(f"  {stage}: command reused from a previous run", flush=True)
            continue
        if stage in kept:
            # A command verified pinned to card 1 has not been verified on card 0, and a
            # command verified unpinned has not been verified against a container that
            # exposes a subset. But the *command* is not in question -- the device is. So it
            # is re-run under the new environment, which is deterministic and costs one run,
            # rather than re-derived, which throws away a proven artifact and gambles on a
            # generation. That gamble is what lost a command that had already run a
            # twelve-hour measurement: forty rounds of failing to reproduce it.
            print(f"  {stage}: re-verifying under the current device", flush=True)
            ok, said = execution_derive.verify_kept(
                kept[stage]["source"], stage, kept[stage]["row"], repo=repo,
                inputs=verify_inputs,
                base_environment=decision.environment)
            if ok:
                stages[stage] = kept[stage]["source"]
                parameters[stage] = kept[stage]["parameters"]
                execution["stages"][stage] = kept[stage]["row"]
                kept[stage]["verified_under"] = _device_identity(decision)
                derived_path.write_text(json.dumps(kept, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
                print(f"  {stage}: still runs; kept", flush=True)
                continue
            print(f"  {stage}: it no longer runs under this device "
                  f"({said.strip().splitlines()[-1][:120] if said.strip() else 'no output'}); "
                  f"deriving again", flush=True)
        if "--keep-only" in flags:
            # Derive nothing; run the loop on what is already kept. Added because the loop and
            # the derivation are two different things and iterating on one meant paying for
            # the other: a stage that will not converge costs forty rounds of running a real
            # benchmark, and every attempt at the loop after it paid that again. A stage with
            # no command is not an error here -- a benchmark the loop works on without it is a
            # case the loop is supposed to handle.
            print(f"  {stage}: not derived (--keep-only)", flush=True)
            continue
        source, settled, log, row = execution_derive.make_runnable(
            client, stage, row, repo=repo, repository_files=execution.get("read") or [],
            inputs_for_verify=verify_inputs,
            base_environment=decision.environment,
            # A command the program refuses on its arguments fails in a second or two, and
            # the usage line it prints is the most informative thing in the loop. The
            # expensive outcome is acceptance, which times out rather than retrying.
            attempts=8,
            # Each round is a revision of the invocation plus five drafts, and
            # the revisions so far have each advanced the stage rather than
            # repeating: the path, then the flag names, then the data. A budget
            # of three ended the loop on the round that was about to work.
            rounds=40,
            on_event=lambda stage, entries: [print(
                f"  {stage} #{entry.get('attempt', '-')}: {entry.get('status')}\n"
                f"      {(entry.get('error') or '')[:900]}\n"
                # What the reviser looked at, before what it decided. Without this a run's
                # record shows the revision and not the grounds for it, and a reviser that
                # looked at the wrong thing is indistinguishable from one that did not look.
                + (f"      LOOKED AT: {entry.get('asked')}\n"
                   f"      FOUND: {entry.get('found')}\n"
                   if entry.get("asked") else "")
                + f"      DRAFT: {(entry.get('source') or '')[:1100]}", flush=True)
                for entry in entries])
        if source is None:
            # Why, not only that. A stage that could not be made to run has a reason, and the
            # reason is a finding about the benchmark -- `no runnable command` on its own is
            # the shape of report this system is built to avoid.
            why = next((str(row.get("error") or "") for row in reversed(log)
                        if row.get("status") == "no command can fix this"), "")
            print(f"  {stage}: no runnable command"
                  + (f"\n      {why}" if why else ""), flush=True)
            continue
        stages[stage] = source
        # The values the verifier ran with, so the loop runs the same command. Supplying
        # none is what produced a KeyError from inside a function nobody can see.
        parameters[stage] = {name: {"value": value} for name, value in (settled or {}).items()}
        # The working directory and environment as they now stand, which is what made the
        # command work. The source alone is not the invocation: a revision that added
        # `PYTHONPATH` is what got the program to import at all, and dropping it here meant
        # every verified stage failed at its first import when the loop ran it.
        execution["stages"][stage] = {**row,
                                      "parameters": [{"name": name, "value": value}
                                                     for name, value in (settled or {}).items()]}
        kept[stage] = {"source": source, "parameters": parameters[stage],
                       "row": execution["stages"][stage],
                       "verified_under": _device_identity(decision)}
        derived_path.write_text(json.dumps(kept, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print(f"  {stage}: command verified", flush=True)

    if not stages:
        print("no stage produced a runnable command")
        return 1

    backend = DeclarativeBackend(repo=repo, answer=execution, sources=stages,
                                  parameters=parameters)
    space = space_from(declaration)
    research = DerivedResearch(repo=repo, output=output, backend=backend,
                               interpreter=interpreter, space=space, client=client,
                               stages=stages, run_id="derived", compute=decision,
                               # So the controller is shown the methods that were measured
                               # on this benchmark and not only the general ones.
                               benchmark=str(declaration.get("benchmark") or ""),
                               # The declaration itself, so the red lines can carry the
                               # benchmark's own evaluator and the objective can ask its
                               # questions in the benchmark's terms. Without it the loop runs
                               # with the standing six lines and a bare spine -- which is a
                               # run that may not edit `evaluate.py` by name and does not know
                               # which file that is here.
                               declaration=declaration)
    print("stage availability:", json.dumps(research.describe(), ensure_ascii=False)[:400])
    print("baseline settings:", json.dumps(base, ensure_ascii=False), flush=True)
    report = research.run(rounds=rounds, settings=base)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
