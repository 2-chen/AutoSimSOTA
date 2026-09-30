"""Run a benchmark's stages from what the system derived, in the environment it built.

Usage: `.venv/bin/python tools/run_derived_stages.py <repo> <output-dir> [stage ...]`

The composition: a repository nobody wrote code for, an environment nobody installed by
hand, and stages nobody named -- turned into commands, verified by running them, and
reported with what each one actually did.

The verifier is the point. A command that reads correctly and names flags the entry point
does not accept is caught by running it and by nothing else: the program prints what it
accepts, in the second it takes to refuse. Where the failure is an import rather than an
argument, the command's environment is what has to change, and the derivation is asked to
declare it instead of the caller guessing.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autosim"))

from autosim.research import execution_derive, provision                # noqa: E402
from autosim.research.declarative_backend import (DeclarativeBackend,   # noqa: E402
                                                  invocation_directory,
                                                  invocation_environment)
from autosim.research.patch_validation import checked_function          # noqa: E402


def main() -> int:
    repo = Path(sys.argv[1]).expanduser().resolve()
    output = Path(sys.argv[2]).expanduser().resolve()
    wanted = sys.argv[3:]
    output.mkdir(parents=True, exist_ok=True)

    interpreter = provision.env_python(output)
    if interpreter is None:
        print(json.dumps({"error": "no successful build here", "output": str(output)}))
        return 2
    print("interpreter:", interpreter, flush=True)

    from autosim.llm_client import LLMClient
    from autosim.research.repository_autoresearch import (PROJECT_ROOT,
                                                          load_deepseek_environment)
    load_deepseek_environment(project_root=PROJECT_ROOT)
    client = LLMClient()

    document = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    stages = document.get("stages") or {}
    inputs = {"python": str(interpreter), "repo": str(repo), "task": "libero_10",
              "dataset": str(repo), "checkpoint": "", "output": str(output / "stage_out"),
              "steps": 1, "episodes": 1, "seed": 0, "device": "0", "setting": "random",
              "extra": {}}

    sources, parameters = {}, {}
    for stage, row in sorted(stages.items()):
        if not row.get("available") or (wanted and stage not in wanted):
            continue
        # Held in a dict rather than closed over, because a revision replaces the directory
        # and the environment and the verifier has to see the new ones. A default argument
        # captures its value at definition time, so the closure would keep testing the
        # invocation that had already been revised -- which is what it did.
        current = {"directory": invocation_directory(row, repo=repo, default=repo),
                   "environment": {**os.environ, **invocation_environment(row, repo=repo)}}

        def verify(argv):
            """Run it. The program is the only authority on what it accepts."""
            try:
                done = subprocess.run([str(a) for a in argv], text=True, capture_output=True,
                                      timeout=300, cwd=str(current["directory"]),
                                      env=current["environment"])
            except subprocess.TimeoutExpired:
                return {"ok": True}          # accepted, and it began working
            return {"ok": done.returncode == 0,
                    "error": (done.stderr or "") + "\n" + (done.stdout or "")}

        # A failure the invocation caused cannot be repaired by generating the command
        # again, and the loop would spin -- so when the command cannot be made to run, the
        # invocation is what gets revised and the command is generated once more from it.
        source, revised, log = None, {}, []
        for round_index in range(3):
            source, revised, log = execution_derive.generate_argv(
                client, stage, entrypoint=str(row.get("entrypoint")),
                invocation=str(row.get("invocation")),
                repository_files=document.get("read") or [],
                declared_parameters={p["name"]: p["value"] for p in (row.get("parameters") or [])
                                     if isinstance(p, dict) and "name" in p},
                verify=verify, inputs_for_verify=inputs, attempts=3)
            for attempt in log:
                print(f"  {stage} #{attempt['attempt']}: {attempt['status']} "
                      f"{(attempt.get('error') or '')[:200]}", flush=True)
            if source is not None:
                break
            argv_attempt = _last_argv(inputs, log)
            change = execution_derive.revise_invocation(
                client, stage, row, repo=repo, argv=argv_attempt,
                failure=str(log[-1].get("error") or ""))
            if change is None or change.get("obstacle"):
                print(f"  {stage}: not an invocation problem: "
                      f"{(change or {}).get('obstacle', 'no revision offered')}"[:300], flush=True)
                break
            print(f"  {stage}: revising the invocation -> {json.dumps(change, ensure_ascii=False)[:220]}",
                  flush=True)
            # A revision returns parameters as a name -> value map; the derivation carries
            # them as a list of records. Merged in the derivation's shape so the next round
            # reads what it expects.
            merged = {k: v for k, v in change.items() if k not in {"why", "parameters"}}
            if isinstance(change.get("parameters"), dict):
                existing = {p["name"]: p for p in (row.get("parameters") or [])
                            if isinstance(p, dict) and "name" in p}
                for name, value in change["parameters"].items():
                    existing[str(name)] = {**(existing.get(str(name)) or {}),
                                           "name": str(name), "value": value,
                                           "evidence": "revised after the command was run"}
                merged["parameters"] = list(existing.values())
            row = {**row, **merged}
            current["directory"] = invocation_directory(row, repo=repo, default=repo)
            current["environment"] = {**os.environ,
                                      **invocation_environment(row, repo=repo)}
        if source is None:
            continue
        sources[stage] = source
        settled = {p["name"]: p["value"] for p in (row.get("parameters") or [])
                   if isinstance(p, dict) and "name" in p}
        settled.update(revised or {})
        parameters[stage] = {k: {"value": v} for k, v in settled.items()}

    if not sources:
        print(json.dumps({"stages": {s: r.get("available") for s, r in stages.items()},
                          "note": "no stage produced a command that runs"}, ensure_ascii=False))
        return 1
    print("verified:", sorted(sources), flush=True)

    backend = DeclarativeBackend(repo=repo, answer=document, sources=sources,
                                 parameters=parameters)
    print("environment declared:", json.dumps(backend.environment(sorted(sources)[0])))
    print("directory declared  :", backend.directory(sorted(sources)[0]))
    results = []
    for stage in sorted(sources):
        function = checked_function(sources[stage], f"stage_argv_{stage}")
        supplied = dict(inputs)
        for key, value in _expand(parameters.get(stage) or {}).items():
            supplied.setdefault(key, value)
        argv = function(supplied)
        started = time.monotonic()
        try:
            done = subprocess.run([str(a) for a in argv], text=True, capture_output=True,
                                  timeout=1800, cwd=str(backend.directory(stage)),
                                  env={**os.environ, **backend.environment(stage)})
            outcome = {"returncode": done.returncode,
                       "said": ((done.stderr or "") + "\n" + (done.stdout or ""))[-1500:]}
        except subprocess.TimeoutExpired:
            outcome = {"returncode": None, "said": "did not finish within 1800s"}
        results.append({"stage": stage, "argv": [str(a) for a in argv],
                        "seconds": round(time.monotonic() - started, 1), **outcome})
        print(f"\n=== {stage}  rc={outcome['returncode']}  {results[-1]['seconds']}s", flush=True)
        print("    " + " ".join(str(a) for a in argv)[:500], flush=True)
        print("    " + outcome["said"][-800:].replace("\n", "\n    "), flush=True)

    (output / "stage_runs.json").write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _last_argv(inputs: dict, log: list) -> list:
    """What the command looked like when it was refused, for the revision to look at."""
    for row in reversed(log):
        entry = row.get("argv")
        if entry:
            return [str(a) for a in entry]
    return []


def _expand(parameters: dict) -> dict:
    from autosim.research.execution_derive import expand_parameters
    return expand_parameters({name: row.get("value") for name, row in parameters.items()})


if __name__ == "__main__":
    raise SystemExit(main())
