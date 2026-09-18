"""Run a benchmark's stages from what the system derived, with the environment it built.

The composition this demonstrates is the whole point of the last several stages: a
repository nobody wrote code for, an environment nobody installed by hand, and stages
nobody named -- turned into commands, run against the provisioned interpreter, and reported
with what each one actually did.

Usage: `.venv/bin/python tools/run_derived_stages.py <repo> <output-dir> [stage ...]`

Each stage is run to completion or to its timeout. A stage that fails is reported with the
program's own words, because a stage that did not run is a finding and not a crash.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autosim"))

from autosim.research import execution_derive, provision               # noqa: E402
from autosim.research.declarative_backend import DeclarativeBackend    # noqa: E402
from autosim.research.patch_validation import checked_function         # noqa: E402


def load_or_derive(repo: Path, output: Path, client) -> dict:
    """The derivation, from disk if it was made already."""
    cached = output / "execution.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    result = execution_derive.run(repo, client=client, output=output)
    return {"stages": result["stages"], "read": result["read"]}


def main() -> int:
    repo = Path(sys.argv[1]).expanduser().resolve()
    output = Path(sys.argv[2]).expanduser().resolve()
    wanted = sys.argv[3:]
    output.mkdir(parents=True, exist_ok=True)

    interpreter = provision.env_python(output)
    if interpreter is None:
        print(json.dumps({"error": "no successful build here; run provisioning first",
                          "output": str(output)}, ensure_ascii=False))
        return 2
    print("interpreter:", interpreter)

    from autosim.llm_client import LLMClient
    from autosim.research.repository_autoresearch import (PROJECT_ROOT,
                                                          load_deepseek_environment)
    load_deepseek_environment(project_root=PROJECT_ROOT)
    client = LLMClient()

    document = load_or_derive(repo, output, client)
    stages = document.get("stages") or {}

    sources: dict[str, str] = {}
    parameters: dict[str, dict] = {}
    for stage, row in stages.items():
        if not row.get("available"):
            continue
        if wanted and stage not in wanted:
            continue
        source = (document.get("sources") or {}).get(stage, {}).get("source")
        if not source:
            source, log = execution_derive.generate_argv(
                client, stage, entrypoint=str(row.get("entrypoint")),
                invocation=str(row.get("invocation")), repository_files=document.get("read") or [],
                declared_parameters={p["name"]: p["value"] for p in (row.get("parameters") or [])})
            if source is None:
                print(f"{stage}: no command could be generated: {log[-1].get('error', '')[:200]}")
                continue
        sources[stage] = source
        parameters[stage] = {p["name"]: p for p in (row.get("parameters") or [])}

    if not sources:
        print(json.dumps({"stages": {s: r.get("available") for s, r in stages.items()},
                          "note": "no stage had a runnable command"}, ensure_ascii=False))
        return 1

    backend = DeclarativeBackend(repo=repo, answer=document, sources=sources,
                                 parameters=parameters)
    print("will run:", sorted(sources))

    inputs = {"python": str(interpreter), "repo": str(repo), "task": None,
              "dataset": str(repo), "checkpoint": None, "output": str(output / "stage_out"),
              "steps": 1, "episodes": 1, "seed": 0, "device": "cpu", "setting": "random",
              "extra": {}}
    results = []
    for stage in sorted(sources):
        function = checked_function(sources[stage], f"stage_argv_{stage}")
        supplied = dict(inputs)
        supplied["repo"] = str(repo)
        for key, value in _expand(parameters.get(stage) or {}).items():
            supplied.setdefault(key, value)
        try:
            argv = function(supplied)
        except Exception as exc:
            results.append({"stage": stage, "built": False,
                            "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        started = time.monotonic()
        try:
            done = subprocess.run([str(a) for a in argv], text=True, capture_output=True,
                                  timeout=900, cwd=str(repo))
            outcome = {"returncode": done.returncode,
                       "said": (done.stderr or "")[-800:] or (done.stdout or "")[-800:]}
        except subprocess.TimeoutExpired:
            outcome = {"returncode": None, "said": "did not finish within 900s"}
        results.append({"stage": stage, "built": True, "argv": [str(a) for a in argv],
                        "seconds": round(time.monotonic() - started, 1), **outcome})
    print(json.dumps({"interpreter": str(interpreter), "results": results},
                     ensure_ascii=False, indent=1)[:6000])
    return 0


def _expand(parameters: dict) -> dict:
    from autosim.research.execution_derive import expand_parameters
    return expand_parameters({name: row.get("value") for name, row in parameters.items()})


if __name__ == "__main__":
    raise SystemExit(main())
