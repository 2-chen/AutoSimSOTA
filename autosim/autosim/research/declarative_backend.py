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
import os
import re
from pathlib import Path
from typing import Any

from .common import atomic_json, read_json, redact
from .patch_validation import checked_function


ARTIFACT_CANDIDATE_LIMIT = 128


def artifact_pattern_problem(pattern: str) -> str:
    """Reject recursive wildcards that are not complete path components."""
    value = str(pattern or "").strip()
    malformed = next((part for part in Path(value).parts
                      if "**" in part and part != "**"), "")
    if malformed:
        return ("recursive wildcard `**` must be a complete path component; "
                f"invalid component: {malformed!r}")
    return ""


#: What a stage is told when it is asked for a command. The system's vocabulary: a stage
#: learns what it is being asked to do and says how this benchmark expresses it. Shared with
#: the generator, so a stage cannot be asked for a value the vocabulary does not carry.
def argv_name(stage: str) -> str:
    return f"stage_argv_{stage}"


def resolve(value: str, *, repo: Path) -> str:
    """A declared path or value, with `{repo}` filled in."""
    out = str(value).replace("{repo}", str(repo))
    return os.path.expanduser(out) if out.startswith("~") else out


#: `${NAME}` and `$NAME` inside a declared environment value, which is a shell value and reads
#: like one -- including when the name is not set, which for `PYTHONPATH` on a machine that
#: never needed it is the ordinary case and means the empty string.
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def expand(value: str, *, base: dict[str, str], unset: list[str] | None = None) -> str:
    """A declared environment value with the variables in it filled from `base`.

    A stage's environment is a shell environment, so `PATH: "{repo}/env/bin:${PATH}"` means
    what it looks like: put this directory in front of the one that is already there. It was
    passed through literally, so the variable arrived as the four characters `${PATH}` and the
    PATH became a string containing no real directory at all -- and the command then failed
    with `FileNotFoundError: 'bash'`, which is a long way from the thing that was wrong. A
    RoboTwin build spent twenty-four attempts on that message.

    **A name that is not set expands to nothing, because that is what a shell does**, and the
    variables this is most often used with are exactly the ones commonly unset: `PYTHONPATH`
    and `LD_LIBRARY_PATH` are absent on a machine that never needed them, so
    `"{repo}/XPolicyLab:${PYTHONPATH}"` is an ordinary thing to write and means "prepend to
    whatever is there".

    Refusing it -- which this did, to avoid a value that is nearly right -- **crashed the
    derivation it was guarding**: the expansion runs while the loop is building the invocation,
    so a `ValueError` there ends the whole stage and discards every round that had succeeded.
    A guard that takes down the thing it guards is worse than the fault it prevents, and this
    repository has produced one in three different forms now.

    What is kept instead is the record: the caller may pass a list and is told which names were
    unset, so a value that lost a component is written down where a reader sees it rather than
    being either silent or fatal.
    """
    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in base:
            if unset is not None and name not in unset:
                unset.append(name)
            return ""
        return str(base[name])

    return _VARIABLE.sub(replace, value)


def invocation_environment(stage_row: dict[str, Any], *, repo: Path,
                           base: dict[str, str] | None = None,
                           unset: list[str] | None = None) -> dict[str, str]:
    """The variables a stage declared, resolved.

    Carried separately from the argv because they are not arguments. A command whose
    arguments are right and whose search path is wrong fails at import, and what it prints
    is about the import rather than about the thing being asked of it -- which is how a
    derivation that had the right flags spent its corrections on the wrong problem.

    `unset` collects the names the value referred to that are not set here, so a caller can
    record a value that lost a component. It never raises: this runs while an invocation is
    being built, and an exception here ends the stage.
    """
    base = dict(os.environ if base is None else base)
    return {str(name): expand(resolve(str(value), repo=repo), base=base, unset=unset)
            for name, value in (stage_row.get("environment") or {}).items()}


def invocation_stdin(stage_row: dict[str, Any]) -> str | None:
    """What to feed the program's standard input, when it asks something.

    A benchmark that prompts on first import cannot be run unattended, and how it is answered
    is not an argument -- LIBERO asks on the very first import whether to put its datasets
    somewhere custom, and writes a default config either way. Feeding the answer is the
    difference between a stage that runs and one that dies at the prompt.

    When nothing is declared the input is closed rather than inherited. An unattended stage
    that inherits a terminal waits at the question until its timeout: two hours of a research
    loop spent on a keypress nobody is there to make.
    """
    declared = stage_row.get("stdin")
    return None if declared is None else str(declared)


def invocation_staging(stage_row: dict[str, Any]) -> list[str]:
    """Commands that make the repository's assumptions true, run before the stage.

    Some benchmarks cannot be run as they ship, and the reason is not the command. A script
    computes `../../../data/<task>/<setting>/data` and the data is somewhere else; a loader
    opens `policy_last.ckpt` and the checkpoint on the disk holds the same weights under
    another name; a config table is generated by a step that has not run. No choice of working
    directory, environment or argument fixes those, which is what a diagnosis here concludes
    and is right to conclude.

    AutoSOTA calls the answer **protocol-preserving repository repair**: the agent may
    "synthesize missing glue logic, repair file-system assumptions, or reconstruct non-core
    scripts, provided the original evaluation protocol, dataset split, and target setting stay
    unchanged". This is that, in the smallest form that is checkable: a list of commands, run
    before the stage, recorded.

    **What makes it safe is not a rule here.** It is that every staging command is in the
    run's record where a reader sees it, and that the files an evaluation depends on are
    already frozen and checked (`assert_frozen`) at the moment it runs -- so a staging step
    that changed the evaluator is caught by the mechanism that exists for it rather than by a
    second one written for this.
    """
    declared = stage_row.get("staging") or []
    if isinstance(declared, str):
        declared = [declared]
    return [str(one) for one in declared if str(one).strip()]


def invocation_directory(stage_row: dict[str, Any], *, repo: Path,
                         default: Path) -> Path:
    """Where the stage declared it runs, or the checkout when it said nothing."""
    declared = str(stage_row.get("working_directory", "") or "").strip()
    if not declared:
        return default
    path = Path(resolve(declared, repo=repo))
    return path if path.is_absolute() else Path(repo) / path


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
        # Three layers, and the order between them is the whole of this method. The
        # vocabulary goes down first so a function that reads a key the caller did not think
        # to pass gets an empty value rather than a `KeyError` from inside itself. The
        # declared values go over it, because they are settled facts about this repository
        # rather than defaults. The caller's inputs go on top, because the caller is what
        # decides. Getting the first two the wrong way round silently blanks every declared
        # value -- the function then runs with an empty benchmark name and says nothing.
        #
        # Declared values are added rather than asked for, and reachable by every spelling
        # of their name: a function reads them by whatever key it chose, which is usually
        # the bare word rather than the `--flag` the command line uses.
        from .execution_derive import (blank_inputs, bind_step_budget_placeholders,
                                       coalesce_dynamic_overrides, expand_parameters)
        declared = expand_parameters({name: row.get("value") for name, row in
                                     (self.parameters.get(stage) or {}).items()})
        supplied = {**blank_inputs(), **declared, **dict(inputs)}
        supplied["repo"] = supplied.get("repo") or str(self.repo)
        # The stage's own interpreter, when it declared one, over the caller's.
        #
        # There was one interpreter per run, and it is the assumption that stops this system
        # on a benchmark whose stages do not share one environment. The shape is common now:
        # an entry point is a shell script that starts a policy server under one environment
        # and a simulation under another, or a trainer and an evaluator that were written
        # against different framework versions. Nothing about "which python" is a property
        # of the run; it is a property of the command, established when the command was
        # verified -- so it is recorded beside the command rather than supplied to all of
        # them at once.
        own = str((self.stages.get(stage) or {}).get("interpreter") or "").strip()
        if own:
            supplied["python"] = own
        argv = self._functions[stage](supplied)
        if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
            raise ValueError(f"{argv_name(stage)} did not return a list of strings")
        # A derived function may declare a repository default and then append a sparse
        # research override. Resolve that pair here too, at the actual producer→consumer
        # boundary; validating only the one-time command probe leaves the later research
        # stage free to emit two conflicting values. Preserve precedence: repository facts,
        # then current research settings, then stage-specific extras.
        dynamic_overrides: dict[str, Any] = dict(declared)
        for input_name in ("settings", "extra"):
            rows = supplied.get(input_name)
            if isinstance(rows, dict):
                dynamic_overrides.update(rows)
        argv = coalesce_dynamic_overrides(argv, dynamic_overrides)
        if stage == "train":
            argv = bind_step_budget_placeholders(argv, supplied.get("steps"))
        from .execution_derive import conflicting_option_problem
        conflict = conflicting_option_problem(argv)
        if conflict:
            raise ValueError(f"{argv_name(stage)} produced an ambiguous command: {conflict}")
        return argv

    def artifact_pattern(self, stage: str) -> str:
        return str((self.stages.get(stage) or {}).get("artifact", "")).strip()

    def check_artifact(self, stage: str, output: Path, *, since: float | None = None) -> dict[str, Any]:
        """Did the stage write what it said it would?

        A glob with no wildcard is a file; anything else is matched under the output. A
        stage that guarantees nothing findable is reported as unverifiable rather than
        failed, because the derivation is allowed to be vague and the honest answer is that
        nothing was checked.
        """
        pattern = self.artifact_pattern(stage)
        if not pattern:
            return {"checked": False, "why": "the derivation named no artifact"}
        problem = artifact_pattern_problem(pattern)
        if problem:
            return {"checked": True, "pattern": pattern, "matched": 0, "examples": [],
                    "candidate_paths": [], "invalid_pattern": True, "why": problem}
        import glob
        output = Path(output)
        target = Path(pattern)
        if not target.is_absolute():
            target = output / target
        try:
            matches = sorted(Path(name) for name in glob.glob(str(target), recursive=True)) \
                if any(char in pattern for char in "*?[") else ([target] if target.exists() else [])
        except (NotImplementedError, ValueError) as exc:
            return {"checked": True, "pattern": pattern, "matched": 0, "examples": [],
                    "candidate_paths": [], "invalid_pattern": True,
                    "why": f"invalid artifact glob: {exc}"}
        if since is not None:
            matches = [p for p in matches if p.exists() and p.stat().st_mtime >= since]
        absolute_pattern = Path(pattern).is_absolute()
        candidates = [str(path) if absolute_pattern else str(path.relative_to(output))
                      for path in matches[:ARTIFACT_CANDIDATE_LIMIT]]
        return {"checked": True, "pattern": pattern, "matched": len(matches),
                "examples": candidates[:3], "candidate_paths": candidates,
                "candidate_paths_truncated": len(matches) > ARTIFACT_CANDIDATE_LIMIT,
                "freshness_checked": since is not None}

    def artifact_beside(self, stage: str, directory: Path, *, since: float) -> dict[str, Any]:
        """Did the stage write what it promised somewhere the caller could not name?

        A benchmark that builds its output path from config values and its own working
        directory gives the caller nothing to set, so the artifact lands where the command
        ran rather than under the output directory. This is the same check against the
        second place the relative path could resolve to, and it is bounded by modification
        time: the checkout already contains every previous run's output, and a check that
        yesterday's checkpoint can satisfy is not a check.
        """
        pattern = self.artifact_pattern(stage)
        if not pattern:
            return {"checked": False, "why": "the derivation named no artifact"}
        problem = artifact_pattern_problem(pattern)
        if problem:
            return {"checked": True, "pattern": pattern, "matched": 0, "examples": [],
                    "candidate_paths": [], "invalid_pattern": True, "why": problem}
        directory = Path(directory)
        if not Path(pattern).is_absolute() and ".." not in Path(pattern).parts:
            try:
                exact = [path for path in directory.glob(pattern)
                         if path.exists() and path.stat().st_mtime >= since]
            except (NotImplementedError, ValueError) as exc:
                return {"checked": True, "pattern": pattern, "matched": 0, "examples": [],
                        "candidate_paths": [], "invalid_pattern": True,
                        "why": f"invalid artifact glob: {exc}"}
            if exact:
                candidates = [str(path.relative_to(directory))
                              for path in sorted(exact)[:ARTIFACT_CANDIDATE_LIMIT]]
                return {"checked": True, "pattern": pattern, "matched": len(exact),
                        "examples": candidates[:3], "candidate_paths": candidates,
                        "candidate_paths_truncated":
                            len(exact) > ARTIFACT_CANDIDATE_LIMIT,
                        "found_beside_the_command": True}
        name = pattern.split("/")[-1]
        if not any(char in name for char in "*?["):
            return {"checked": False, "why": f"`{name}` names no way to find it elsewhere"}
        # Preserve a concrete immediate parent when searching an alternate output root.
        # Falling back from `runs/*/videos/*.mp4` to just `**/*.mp4` accepts a wholly
        # different `test_videos/` directory as proof of the declared glob. It is useful
        # as a discovery hint, not as confirmation that the declaration was right.
        parent = Path(pattern).parent.name
        suffix = (f"{parent}/{name}" if parent not in ("", ".") and
                  not any(char in parent for char in "*?[") else name)
        found = []
        try:
            for path in directory.glob(f"**/{suffix}"):
                try:
                    if path.exists() and path.stat().st_mtime >= since:
                        found.append(path)
                except OSError:
                    continue
        except (NotImplementedError, ValueError) as exc:
            return {"checked": True, "pattern": suffix, "matched": 0, "examples": [],
                    "candidate_paths": [], "invalid_pattern": True,
                    "why": f"invalid artifact glob: {exc}"}
        candidates = [str(path.relative_to(directory))
                      for path in sorted(found)[:ARTIFACT_CANDIDATE_LIMIT]]
        return {"checked": True, "pattern": suffix, "matched": len(found),
                "examples": candidates[:3], "candidate_paths": candidates,
                "candidate_paths_truncated": len(found) > ARTIFACT_CANDIDATE_LIMIT,
                "found_beside_the_command": True}

    def staging(self, stage: str) -> list[str]:
        """What has to be made true before this stage can run, resolved."""
        return [resolve(one, repo=self.repo)
                for one in invocation_staging(self.stages.get(stage) or {})]

    def environment(self, stage: str) -> dict[str, str]:
        return invocation_environment(self.stages.get(stage) or {}, repo=self.repo)

    def directory(self, stage: str) -> Path:
        return invocation_directory(self.stages.get(stage) or {}, repo=self.repo,
                                    default=self.repo)

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
