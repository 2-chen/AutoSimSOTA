"""Execute a benchmark through a derived answer, with no code written for that benchmark.

A backend is what turns a stage into a command and runs it. Until now each benchmark had
one written by hand: a file naming RoboSyn's trainer and evaluator, a second naming another
benchmark's. The derived answer already names the entry point, the invocation and the
artifact, so this reads all three and executes.

What it does not do is believe them. A derived entry point that exists and is the wrong one
passes every static check -- the repository that ships six policy families ships six
trainers -- so the artifact a stage promises is what the run is checked against. A stage
that exits zero and writes nothing it promised has not run, and saying so is the difference
between a stage that worked and a stage that looked like it did.

The failure this is built to catch has already happened once: the derivation named a pi0
trainer where the benchmark's own checkpoints are ACT, and nothing downstream could tell.

Two limits are worth stating rather than discovering. The generated function maps the
system's input vocabulary onto this benchmark's command line, and a benchmark needing a
value outside that vocabulary cannot be driven -- the function says so instead of inventing
it, and the derivation is asked to declare what it needs. And an artifact existing is not
the same as the artifact being right: a trainer that writes a checkpoint after failing to
read the dataset satisfies the check. Settling that needs the numbers, which is what
evaluation is for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import atomic_json, read_json, redact
from .patch_validation import checked_function


#: What a stage is told when it is asked for a command. The system's vocabulary: a stage
#: learns what it is being asked to do and says how this benchmark expresses it. Shared with
#: the generator, so a stage cannot be asked for a value the vocabulary does not carry.
def argv_name(stage: str) -> str:
    return f"stage_argv_{stage}"


class DeclarativeBackend:
    """A benchmark's execution, read from what the derivation found.

    ``answer`` is the object `execution_derive` produces and ``sources`` the generated argv
    function per stage. Nothing here is specific to a benchmark: the entry point, the
    command and the artifact all come from the answer, and the artifact is what gets checked.
    """

    def __init__(self, *, repo: Path, answer: dict[str, Any], sources: dict[str, str],
                 parameters: dict[str, dict[str, str]] | None = None):
        self.repo = Path(repo).expanduser().resolve()
        self.answer = answer
        self.stages = answer.get("stages") or {}
        self.parameters = parameters or {}
        self._functions: dict[str, Any] = {}
        self.sources = sources
        for stage, source in sources.items():
            self._functions[stage] = checked_function(source, argv_name(stage))

    # -- what a backend must offer ------------------------------------------------------

    def available(self, stage: str) -> bool:
        row = self.stages.get(stage) or {}
        return bool(row.get("available")) and stage in self._functions

    def argv(self, stage: str, inputs: dict[str, Any]) -> list[str]:
        """The command for one stage, from the generated function and the declared values."""
        if stage not in self._functions:
            raise ValueError(
                f"no command was derived for {stage}: "
                f"{redact(str((self.stages.get(stage) or {}).get('why', 'no entry point')))}")
        supplied = dict(inputs)
        supplied.setdefault("repo", str(self.repo))
        supplied.setdefault("extra", {})
        # Declared values are settled for this repository, so they are added rather than
        # asked for. A generated function may read them and may not need them, and it reads
        # them by whatever key it chose, so a parameter is reachable by every spelling of
        # its name rather than only the one the command line uses.
        from .execution_derive import expand_parameters
        for key, value in expand_parameters(
                {name: row.get("value") for name, row in
                 (self.parameters.get(stage) or {}).items()}).items():
            supplied.setdefault(key, value)
        argv = self._functions[stage](supplied)
        if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
            raise ValueError(f"{argv_name(stage)} did not return a list of strings")
        return argv

    def artifact_pattern(self, stage: str) -> str:
        return str((self.stages.get(stage) or {}).get("artifact", "")).strip()

    def check_artifact(self, stage: str, output: Path) -> dict[str, Any]:
        """Did the stage write what it said it would?

        A glob with no wildcard is a file; anything else is matched under the output. A
        stage that guarantees nothing findable is reported as unverifiable rather than
        failed, because the derivation is allowed to be vague and the honest answer is that
        nothing was checked.
        """
        pattern = self.artifact_pattern(stage)
        if not pattern:
            return {"checked": False, "why": "the derivation named no artifact"}
        output = Path(output)
        target = output / pattern
        matches = sorted(p for p in output.glob(pattern)) if any(
            char in pattern for char in "*?[") else ([target] if target.exists() else [])
        return {"checked": True, "pattern": pattern, "matched": len(matches),
                "examples": [str(p.relative_to(output)) for p in matches[:3]]}

    def run_stage(self, stage: str, runner: Any, *, inputs: dict[str, Any], output: Path,
                  timeout: int = 3600) -> dict[str, Any]:
        """Build the command, run it through the machine layer, and look for the artifact.

        The runner is the Runtime, narrowed to running a process: device binding, the
        compute ledger and the receipt belong to the machine, not to a benchmark.
        """
        output = Path(output)
        argv = self.argv(stage, inputs)
        record = runner.run(argv, output / "process", timeout)
        check = self.check_artifact(stage, output)
        result = {"stage": stage, "argv": argv, "process": record.get("status"),
                  "returncode": record.get("returncode"), "artifact": check,
                  "verified": bool(check.get("checked") and check.get("matched"))}
        atomic_json(output / "stage_result.json", result)
        return result

    def describe(self) -> dict[str, Any]:
        return {stage: {"available": self.available(stage),
                        "entrypoint": (self.stages.get(stage) or {}).get("entrypoint"),
                        "artifact": self.artifact_pattern(stage),
                        "parameters": sorted((self.parameters.get(stage) or {}))}
                for stage in self.stages}


def load(repo: Path, execution_path: Path, *,
         sources: dict[str, str] | None = None) -> DeclarativeBackend:
    """Build a backend from a derivation on disk and the argv functions generated for it."""
    document = read_json(Path(execution_path))
    stages = document.get("stages") or {}
    sources = sources or {stage: row["source"] for stage, row
                          in (document.get("sources") or {}).items()}
    parameters = {stage: {p["name"]: p for p in (row.get("parameters") or [])}
                  for stage, row in stages.items()}
    return DeclarativeBackend(repo=repo, answer=document, sources=sources,
                              parameters=parameters)
