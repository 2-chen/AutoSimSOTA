"""Provision RoboTwin for the derived runner, reusing the interpreter that already works.

Run from the repository root: `.venv/bin/python tools/provision_robotwin.py`.

Unlike LIBERO, this machine already has a working RoboTwin environment: `.venv_robotwin`,
which the full pipeline used for a completed run. Building another one would spend hours
producing something that exists, so the plan names this interpreter instead of a version --
which is the case `provision.interpreter_is_given` exists for.

Two things are produced, and the second is easy to forget:

* the environment record (`environment.json`, `recipe.json`), so a later run on another
  machine knows what this one was configured to be;
* the stage derivation (`execution.json`), which says *which file is the trainer, which is
  the evaluator, which is the collector*. That is a separate model call and it is what
  `research_derived.py` reads; without it the loop has an environment and nothing to run in
  it.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autosim"))

from autosim.llm_client import LLMClient                                   # noqa: E402
from autosim.research import execution_derive, provision                   # noqa: E402
from autosim.research.repository_autoresearch import (                     # noqa: E402
    PROJECT_ROOT, load_deepseek_environment)

#: The checkout the declaration was made for -- RoboTwin 2.0, which vendors XPolicyLab.
#: The sibling `test/RoboTwin` is version 1.0 with a different layout and is not this one.
REPO = ROOT / "RoboTwin"
OUTPUT = PROJECT_ROOT / "autoresearch_runs" / "provisioning" / "robotwin"

#: The interpreter the full pipeline's completed RoboTwin run used, named rather than built.
INTERPRETER = ROOT / ".venv_robotwin" / "bin" / "python"


def main() -> int:
    load_deepseek_environment(project_root=PROJECT_ROOT)
    client = LLMClient()
    if not client.available:
        raise SystemExit("DEEPSEEK_API_KEY is not set; provisioning has no model to ask")
    if not INTERPRETER.exists():
        raise SystemExit(f"the interpreter this reuses is gone: {INTERPRETER}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"repo   {REPO}")
    print(f"output {OUTPUT}")
    print(f"python {INTERPRETER} (already present; the plan names it rather than building)",
          flush=True)

    # Stages first, because it is one model call and it decides whether there is anything to
    # provision *for*: a benchmark with no runnable stage wants a different conversation.
    execution_path = OUTPUT / "execution.json"
    if execution_path.is_file() and "--rederive" not in sys.argv:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        print(f"stages: reusing {execution_path} "
              f"({sorted((execution.get('stages') or {}))})", flush=True)
    else:
        print("stages: deriving (one model call)", flush=True)
        execution = execution_derive.run(REPO, client=client, output=OUTPUT)
        stages = execution.get("stages") or {}
        print(f"stages: {sorted(stages)}", flush=True)
        for name, row in sorted(stages.items()):
            print(f"  {name:10} available={row.get('available')} "
                  f"entrypoint={row.get('entrypoint')}", flush=True)

    print("environment: provisioning", flush=True)
    result = provision.build(REPO, client=client, prefix=OUTPUT / "env", output=OUTPUT,
                             python=str(INTERPRETER), max_rounds=20, step_timeout=3600)
    print("elapsed:", round((time.time() - started) / 60, 1), "min", flush=True)
    print("verdict:", json.dumps(result["verdict"], ensure_ascii=False)[:800], flush=True)
    print("interpreter recorded:", provision.env_python(OUTPUT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
