"""Build the environment a benchmark needs, by running commands and recording what worked.

Every other stage assumes something can run. This is the stage that makes that true, and
without it a benchmark the system can read, plan for and derive is still one it cannot
execute -- which is the state LIBERO has been in throughout.

The design comes from Repo2Run (NeurIPS 2025), which builds executable environments for
arbitrary repositories at 86% success, and the part worth taking is not its containers but
its ordering. Its ablation is unambiguous: removing the component that *records* the
commands that worked, in favour of asking a model to write the environment file directly,
costs 80.5% of successfully generated environments. The paper's explanation is that a model
writing the file cannot follow the history of events that actually occurred.

That is the same lesson this system learned a stage earlier. A generated command line
passed every static check and was refused by the program it was written for -- six flags
that did not exist, all of them plausible. Reading cannot tell you whether a dependency
resolves; running it can. So nothing here is written down until it has run.

Three properties, each load-bearing in the research:

* **Recorded, not generated.** The recipe is the sequence of commands that succeeded, in
  the order they succeeded. A command that fails is repaired or dropped, never recorded.
* **Rolled back by rebuilding.** Repo2Run snapshots with `docker commit`; without containers
  the equivalent is to resume from the recorded prefix, because only commands that succeeded
  are ever in it. An environment cannot be polluted by a step that was never kept.
* **Verified by doing the thing.** A package that imports is not an environment that runs a
  simulator. The probe is a real action, and what it reports decides whether the next stage
  has anything to work with.

What this cannot do is make an environment exist where the platform cannot host one. A
driver, a rendering backend and the system libraries underneath are not pip-installable, so
a failure there is reported as a platform limit rather than retried as a dependency problem
-- which is the difference between a build that stops with a reason and one that loops.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .common import atomic_json, now, object_digest, read_json, redact


#: Ways a benchmark states its dependencies. A project that ships a lock file has answered
#: the question; one that ships only a README has not, and the difference is worth seeing.
MANIFESTS = ("environment.yml", "environment.yaml", "conda.yml", "requirements.txt",
             "pyproject.toml", "setup.py", "setup.cfg", "install.sh", "Makefile",
             "Dockerfile", "README.md")

#: What a planned command may refer to. The system substitutes these; a command needing
#: anything else is asking for a fact the plan has not established.
PLACEHOLDERS = {
    "conda": "the conda executable to create the environment with",
    "prefix": "the absolute path of the environment being built",
    "python": "the interpreter inside that environment",
    "pip": "the pip inside that environment",
    "repo": "the absolute path of the benchmark checkout",
    "workdir": "a scratch directory the build may use and discard",
}

PLAN_SYSTEM = (
    "You are building the environment a benchmark repository needs in order to run. You are "
    "given what the repository declares about its dependencies and how it is laid out, and "
    "you return a plan: which Python version to build the environment with, and the sequence "
    "of shell commands that installs what the repository needs.\n\n"
    "Commands are run one at a time, in a shell, and each is recorded only if it succeeds, "
    "so order them "
    "the way a person would: create the environment first, then install, then anything the "
    "repository needs done to itself. Use the placeholders given in `placeholders`, which are "
    "substituted before running. Do not write a command whose success you cannot tell from "
    "its output, and do not include a step you know will fail.\n\n"
    "The repository's own requirements are the starting point, not the whole answer: a file "
    "lists what its authors used, which is not always everything a component needs, and a "
    "library imported by the code that runs is a dependency whether or not it is listed. "
    "Where you add something the repository does not mention, say so in the reasoning.\n\n"
    "Where the repository pins a version, use it even if a newer one exists -- the repository "
    "was written against what it pins. The exception is a pin that cannot work on "
    "`machine`: a build for a compute capability this GPU does not report installs and then "
    "cannot execute, which is worse than failing, because the build looks successful. Where "
    "that is so, say which pin you are departing from and why, and what the departure risks.\n\n"
    "Return one JSON object: {\"python\": \"<version, e.g. 3.9>\", \"commands\": [\"<command>\", "
    "...], \"probe\": \"<one command using {python} that shows the environment can do the "
    "thing this benchmark exists to do>\", \"reasoning\": \"<what you based this on, and "
    "anything you added or could not determine>\"}."
)

RESUME_SYSTEM = (
    "You are continuing to build an environment. Commands are run one at a time and only the "
    "ones that succeed are kept, so you are shown the commands that have already survived and "
    "the failure of the one that did not.\n\n"
    "Change as little as possible. If the failing command is wrong, return a corrected "
    "version of it. If it failed because something it needs is not installed yet, return the "
    "missing command followed by the one that failed. Use the placeholders in `placeholders`.\n\n"
    "Do not re-create the environment. Recreating the prefix removes everything installed "
    "into it, and the commands already recorded are not re-run, so the result is an empty "
    "environment and a recipe that claims otherwise.\n\n"
    "Return one JSON object: {\"commands\": [\"<the commands to run next, in order>\"], "
    "\"reasoning\": \"<what the failure told you>\", \"unbuildable\": \"<if this cannot be "
    "fixed by installing something, say why; otherwise omit>\"}."
)


RECONSIDER_SYSTEM = (
    "You are building an environment and the approach is not working: a whole round of "
    "commands produced nothing that survived. Fixing the failing command one more time has "
    "not helped, so reconsider the constraint instead.\n\n"
    "The usual cause is a requirement that cannot be satisfied as stated on this machine -- "
    "a version that is no longer published, a build for hardware this machine does not have, "
    "a pin that predates something it now has to coexist with. When that is so, the useful "
    "move is to name the constraint that is failing and substitute it, saying what you are "
    "giving up and what would have to be checked afterwards. A package whose pinned build "
    "targets hardware the machine does not have will install and then fail at first use; "
    "substituting a build that targets it is the fix, and the risk is that the rest of the "
    "stack was written against the old one.\n\n"
    "You are given everything tried so far, with what each attempt reported. Do not repeat "
    "an approach that has already failed.\n\n"
    "Return one JSON object: {\"constraint\": \"<the requirement that cannot hold, as it was "
    "stated>\", \"substitute\": \"<what to use instead>\", \"why\": \"<what makes the "
    "original impossible here>\", \"cost\": \"<what this gives up, and what would have to "
    "be checked>\", \"commands\": [\"<the commands to run now>\"], \"unbuildable\": \"<if "
    "no substitution can work, why not; otherwise omit>\"}"
)

DIAGNOSE_SYSTEM = (
    "An environment build has stopped. Say what is in the way, in terms someone can act on, "
    "and do not overstate what is known.\n\n"
    "A dependency that is missing is not the same finding as a requirement that cannot be "
    "satisfied on this machine, and neither is the same as the machine itself being unable to "
    "host the workload -- a driver, a rendering backend, a compute capability the installed "
    "builds do not target. They have different remedies and only the first is a matter of "
    "installing something.\n\n"
    "The record is what survived; the transcript is everything tried. A command in the record "
    "has run successfully and is not evidence about the blocker unless it is the one that "
    "failed.\n\n"
    "Return one JSON object: {\"blocker\": \"<dependency | requirement-not-satisfiable | "
    "platform | unresolved>\", \"what\": \"<the specific thing in the way>\", \"evidence\": "
    "\"<the commands and output that show it>\", \"tried\": [\"<approaches attempted>\"], "
    "\"would_unblock\": \"<what a person would do, and what it would cost>\", "
    "\"confidence\": \"<high | medium | low>\"}"
)


def platform_facts(*, timeout: int = 60) -> dict[str, Any]:
    """What this machine is, so a plan can be checked against it rather than assumed.

    A build planned without this is planned against an imagined machine. The failure that
    costs most is invisible until first use: a wheel built for a compute capability the card
    does not have installs perfectly and cannot execute, and no amount of dependency
    resolution reveals it. Knowing the capability beforehand is what lets a plan avoid it.
    """
    facts: dict[str, Any] = {"os": os.uname().sysname, "release": os.uname().release}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
             "--format=csv,noheader"],
            text=True, capture_output=True, timeout=timeout, env=os.environ)
        facts["gpus"] = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        facts["gpus"] = []
    try:
        out = subprocess.run(["nvcc", "--version"], text=True, capture_output=True,
                             timeout=timeout, env=os.environ)
        facts["cuda_toolkit"] = next((line.strip() for line in out.stdout.splitlines()
                                      if "release" in line), "not found")
    except (OSError, subprocess.TimeoutExpired):
        facts["cuda_toolkit"] = "not found"
    return facts


def manifests_of(repo: Path, *, limit: int = 60_000) -> dict[str, str]:
    """What the repository says about its own dependencies, read verbatim."""
    found: dict[str, str] = {}
    for name in MANIFESTS:
        path = Path(repo) / name
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > limit:
                continue
            found[name] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return found


def _object(content: str) -> dict[str, Any]:
    text = content.strip()
    for opener in ("```json", "```"):
        if text.startswith(opener):
            text = text[len(opener):]
    text = text.removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the response")
    return json.loads(text[start:end + 1])


def plan_problems(value: dict[str, Any]) -> list[str]:
    """Shape faults in a plan, as messages. Empty means runnable, not correct."""
    if not isinstance(value, dict):
        return ["plan must be one JSON object"]
    problems: list[str] = []
    if not str(value.get("python", "")).strip():
        problems.append("python is required: which version to build the environment with")
    commands = value.get("commands")
    if not isinstance(commands, list) or not commands:
        problems.append("commands must be a non-empty list")
    else:
        for index, command in enumerate(commands):
            if not isinstance(command, str) or not command.strip():
                problems.append(f"commands[{index}] must be a non-empty string")
    if not str(value.get("probe", "")).strip():
        problems.append("probe is required: how to show the environment works")
    return problems


def substitute(command: str, values: dict[str, str]) -> str:
    """Fill the placeholders a command is allowed to use."""
    out = command
    for name, value in values.items():
        out = out.replace("{" + name + "}", value)
    return out


def unknown_placeholders(command: str) -> list[str]:
    import re
    return sorted(set(re.findall(r"\{([a-z_]+)\}", command)) - set(PLACEHOLDERS))


def run(command: str, *, env: dict[str, str], cwd: Path, timeout: int,
        output: Path) -> dict[str, Any]:
    """Run one command, keeping its output whether it worked or not."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {command}\n")
        log.flush()
        started = time.monotonic()
        try:
            # Through a shell, because provisioning is what shells are for. Running argv
            # directly made `CMAKE_ARGS=... pip install ...` -- the standard fix for a CMake
            # version policy rejecting an old package, and the one this build needed -- not
            # expressible, along with every other way a person sets something for a single
            # command. The command is a plan the system wrote from the repository it is
            # installing, run on the machine the user pointed it at.
            completed = subprocess.run(["bash", "-lc", command], text=True,
                                       capture_output=True, timeout=timeout, env=env,
                                       cwd=str(cwd))
        except subprocess.TimeoutExpired:
            log.write(f"[timed out after {timeout}s]\n")
            return {"command": command, "ok": False, "returncode": None,
                    "seconds": round(time.monotonic() - started, 1),
                    "failure_kind": "timeout",
                    "excerpt": f"the command did not finish within {timeout}s"}
        except OSError as exc:
            log.write(f"[{type(exc).__name__}: {exc}]\n")
            return {"command": command, "ok": False, "returncode": None, "seconds": 0.0,
                    "failure_kind": "platform",
                    "excerpt": f"{type(exc).__name__}: {exc}"}
        log.write(completed.stdout)
        log.write(completed.stderr)
    return {"command": command, "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "seconds": round(time.monotonic() - started, 1),
            "failure_kind": classify(completed.returncode, completed.stdout, completed.stderr),
            "excerpt": error_excerpt((completed.stderr or "") + "\n" + (completed.stdout or ""))}


def resets_environment(command: str, prefix: Path) -> bool:
    """Does this command throw the environment away and start again?

    Turning a prefix into an environment and turning it back into nothing look the same to
    a return code, and the difference decides whether everything recorded before it still
    exists. A build here recorded `pip install torch` -- a 1.6 GB download -- and then ran a
    second `conda create` on the same prefix, which removed it. The recipe went on claiming
    torch was installed, because the record only ever grew.

    A record that grows without ever being invalidated is not a record of what is present.
    This is the case snapshots exist for in the work this is modelled on; without them, the
    equivalent is knowing which commands make the record void.
    """
    lowered = command.lower()
    creating = ("conda create" in lowered or "conda env create" in lowered
                or "mamba create" in lowered or "micromamba create" in lowered)
    return creating and str(prefix) in command


def classify(returncode: int, stdout: str, stderr: str) -> str:
    """Whether the environment is wrong or the machine underneath it is.

    This is the distinction a build loop gets wrong by retrying forever. A missing package
    is a dependency problem and installing fixes it; a driver or a renderer that will not
    start is not, because no amount of installing fixes it. Reporting the second as the first
    produces a loop that is expensive and cannot terminate.
    """
    if returncode == 0:
        return "ok"
    text = (stdout + "\n" + stderr).lower()
    # Each signal has to be long enough not to appear inside a package name. A bare "egl"
    # matched `egl_probe`, so a dependency that failed to compile was reported as a platform
    # limit and the build stopped on a problem installing fixes -- which is exactly the
    # non-terminating loop this function exists to prevent, arrived at from the other side.
    signals = ("libegl", "libgl", "libcuda", "libnvidia", "nvidia driver", "cuda driver",
               "cannot open display", "no such device", "eglinitialize", "egl display",
               "vulkan", "nvml", "no kernel image", "not supported by the driver",
               "driver version", "could not load gpu")
    if returncode < 0 or any(signal in text for signal in signals):
        return "platform"
    if "modulenotfounderror" in text or "importerror" in text:
        return "dependency"
    return "unknown"


def error_excerpt(output: str, *, limit: int = 1500) -> str:
    """The part of the output that says what went wrong, not the tail.

    Learned one stage earlier: a program that prints a banner on import puts its cause above
    output that looks more recent, and feeding back the last few lines sends the model a
    header instead of an error.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return "(no output)"
    markers = ("error:", "traceback (most recent call last)", "cannot", "no such file",
               "not found", "unable to", "failed", "importerror", "modulenotfounderror")
    hits = [index for index, line in enumerate(lines)
            if any(marker in line.lower() for marker in markers)]
    if not hits:
        return "\n".join(lines[-8:])[:limit]
    # Both ends. A build log announces the failure at the top -- `error: subprocess-exited-
    # with-error` -- and states the cause hundreds of lines later inside the wrapped
    # compiler output. Reporting only the first gave the model a wrapper and no cause, and
    # it said so: the root cause "is not yet established".
    head = lines[max(0, hits[0] - 1):hits[0] + 12]
    # The cause of a build failure is at the end, and it does not have to announce itself
    # with a word this function thought to look for: `CMake Error at` and `make: *** Error 2`
    # both matched none of the markers, so a long log returned its opening and nothing else.
    # When there is a lot of output, the end is included on that reasoning alone.
    if len(lines) > hits[0] + 24:
        return ("\n".join(head) + "\n...\n" + "\n".join(lines[-14:]))[:limit]
    return "\n".join(head)[:limit]


def content_id(python: str, record: list[dict[str, Any]], repo: Path) -> str:
    """The identity of an environment: the commands that built it and the repository.

    Keyed on the commands rather than on a name, for the reason SWE-bench keys its
    environment images on a hash of their setup script: an environment left over from a
    changed recipe is not this environment, and reusing it would attribute to the new
    dependencies whatever the old ones produced.
    """
    return object_digest({"python": python,
                          "commands": [row["command"] for row in record if row.get("ok")],
                          "repo": str(repo)})


def values_for(prefix: Path, repo: Path, workdir: Path, conda: str) -> dict[str, str]:
    return {"conda": conda, "prefix": str(prefix), "python": str(prefix / "bin/python"),
            "pip": str(prefix / "bin/pip"), "repo": str(repo), "workdir": str(workdir)}


def conda_executable() -> str:
    for candidate in ("conda", "mamba", "micromamba"):
        from shutil import which
        found = which(candidate)
        if found:
            return found
    return "conda"


def build(repo: Path, *, client: Any, prefix: Path, output: Path, python: str | None = None,
          max_rounds: int = 14, step_timeout: int = 3600,
          manifests: dict[str, str] | None = None) -> dict[str, Any]:
    """Build until the probe passes, recording only what survived.

    The loop is: run the next command; if it fails, ask what to change given the failure and
    the commands that already worked; when the plan runs out, probe; if the probe fails, ask
    what is missing. Nothing that failed is ever in the record, so the recipe is a
    description of an environment that exists.
    """
    repo = Path(repo).expanduser().resolve()
    prefix = Path(prefix).expanduser().resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    workdir = output / "work"
    workdir.mkdir(exist_ok=True)
    manifests = manifests if manifests is not None else manifests_of(repo)

    environment = {**os.environ, "AUTOSIM_REPO": str(repo), "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                   "PIP_NO_INPUT": "1", "CONDA_ALWAYS_YES": "true", "PYTHONUNBUFFERED": "1"}
    # A build is long and is expected to be resumed. Reading the previous attempt back is
    # what makes "rolled back by rebuilding" cheap rather than a full restart, and it is the
    # same content-addressed idea as everywhere else: what survives is what succeeded.
    previous = read_json(output / "transcript.json") if (output / "transcript.json").is_file() else None
    transcript: list[dict[str, Any]] = list((previous or {}).get("rows") or [])
    # Replayed in order, because a command that recreated the environment invalidates what
    # came before it -- including in an earlier attempt, where the same mistake looks like
    # a longer recipe rather than a wrong one.
    record: list[dict[str, Any]] = []
    for row in transcript:
        if row.get("kind") == "reset":
            record = []
        elif row.get("ok") and row.get("kind") != "probe":
            record.append(row)
    survived = [row["command"] for row in record]

    if python is None:
        planned = plan(client, repo, manifests=manifests)
        python, pending, probe = planned["python"], list(planned["commands"]), planned["probe"]
        transcript.append({"stage": "plan", **{k: planned[k] for k in ("reasoning",)},
                           "attempts": planned.get("attempts")})
    else:
        pending, probe = [], ""
    atomic_json(output / "plan.json", {"python": python, "commands": pending, "probe": probe,
                                       "manifests": sorted(manifests), "created_at": now()})

    values = values_for(prefix, repo, workdir, conda_executable())
    index = 0
    for round_index in range(max_rounds):
        before = len(record)
        while index < len(pending):
            if substitute(pending[index], values) in survived:
                # Already done in an earlier attempt; re-running it would be slow and could
                # regress a step that works.
                index += 1
                continue
            command = substitute(pending[index], values)
            missing = unknown_placeholders(pending[index])
            if missing:
                result = {"command": command, "ok": False, "returncode": None, "seconds": 0.0,
                          "failure_kind": "dependency",
                          "excerpt": f"the command uses placeholders that do not exist: {missing}"}
            else:
                result = run(command, env=environment, cwd=repo,
                             timeout=step_timeout,
                             output=output / "build.log")
            transcript.append(result)
            if result["ok"]:
                if resets_environment(command, prefix):
                    # Everything before this is gone. Keeping it would make the recipe a
                    # description of an environment that no longer exists.
                    transcript.append({"kind": "reset", "command": command,
                                       "invalidated": len(record)})
                    record = []
                    survived = []
                record.append({**result, "index": index, "round": round_index + 1})
                index += 1
                continue
            if result["failure_kind"] == "platform":
                # Retrying cannot help and cannot terminate. The machine cannot host this,
                # which is a finding about the platform and not about the recipe.
                return _finish(output, repo, python, record, transcript, probe,
                               {"passed": False, "reason": "platform limit",
                                "detail": result["excerpt"]}, client=client)
            pending = pending[:index] + resume(client, repo, record, result,
                                               manifests=manifests, transcript=transcript,
                                               values=values)
            break
        else:
            outcome = probe_environment(probe, values=values, env=environment, cwd=repo,
                                        output=output / "probe.log")
            transcript.append({**outcome, "kind": "probe"})
            if outcome["ok"]:
                record.append({**outcome, "kind": "probe", "round": round_index + 1})
                return _finish(output, repo, python, record, transcript, probe,
                               {"passed": True, "seconds": outcome.get("seconds")},
                               client=client)
            if outcome["failure_kind"] == "platform":
                return _finish(output, repo, python, record, transcript, probe,
                               {"passed": False, "reason": "platform limit",
                                "detail": outcome["excerpt"]}, client=client)
            # A round that added nothing is a round that made no progress. Fixing one more
            # command has been tried and has not worked, so the thing to question is the
            # requirement rather than the command -- and that is a different question, which
            # nothing in the loop was asking.
            pending = resume(client, repo, record, {**outcome, "kind": "probe"},
                             manifests=manifests, transcript=transcript, values=values,
                             probing=True)
            index = 0
        # A round that added nothing made no progress, whether it stopped at the probe or at
        # a command. Escalating only on the probe left the case that actually occurred --
        # a build that never reached the probe because one command would not run --
        # retrying the same class of fix until the rounds ran out.
        if len(record) == before:
            revision = reconsider(client, repo, record=record, transcript=transcript,
                                  manifests=manifests)
            if revision is None:
                return _finish(output, repo, python, record, transcript, probe,
                               {"passed": False, "reason": "no approach left",
                                "rounds": round_index + 1}, client=client)
            pending = list(revision["commands"])
            index = 0
    return _finish(output, repo, python, record, transcript, probe,
                   {"passed": False, "reason": "rounds exhausted"}, client=client)


def plan(client: Any, repo: Path, *, manifests: dict[str, str], attempts: int = 3
         ) -> dict[str, Any]:
    payload = json.dumps({"placeholders": PLACEHOLDERS, "machine": platform_facts(),
                          "declared_dependencies": manifests}, ensure_ascii=False)
    log: list[dict[str, Any]] = []
    for repair in range(attempts):
        content, _ = client.chat_with_metadata(
            PLAN_SYSTEM, payload + ("" if repair == 0 else json.dumps(
                {"rejected": log[-1]["error"],
                 "instruction": "Return the corrected plan."}, ensure_ascii=False)),
            max_tokens=4000, timeout=300, thinking="disabled")
        try:
            value = _object(content)
            faults = plan_problems(value)
            if faults:
                raise ValueError("; ".join(faults))
            value["attempts"] = log
            return value
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            log.append({"attempt": repair + 1, "status": "rejected",
                        "error": redact(f"{type(exc).__name__}: {exc}")})
    raise ValueError(f"no runnable plan: {log[-1].get('error')}")


def resume(client: Any, repo: Path, record: list[dict[str, Any]], failure: dict[str, Any], *,
           manifests: dict[str, str], transcript: list[dict[str, Any]], values: dict[str, str],
           probing: bool = False, attempts: int = 3) -> list[str]:
    """What to run next, given what survived and what just failed."""
    payload = json.dumps({
        "placeholders": PLACEHOLDERS,
        "declared_dependencies": {k: v[:4000] for k, v in manifests.items()},
        "machine": platform_facts(),
        "survived_commands": [row["command"] for row in record if row.get("kind") != "probe"],
        "failure": {"command": failure.get("command"), "kind": failure.get("failure_kind"),
                    "excerpt": failure.get("excerpt")},
        "note": ("The probe -- the repository's own smallest real action -- failed. Say what "
                 "is missing." if probing else
                 "The command above failed. Correct it, or put what it needs in front of it."),
    }, ensure_ascii=False)
    log: list[dict[str, Any]] = []
    for repair in range(attempts):
        content, _ = client.chat_with_metadata(
            RESUME_SYSTEM, payload + ("" if repair == 0 else json.dumps(
                {"rejected": log[-1]["error"]}, ensure_ascii=False)),
            max_tokens=3000, timeout=300, thinking="disabled")
        try:
            value = _object(content)
            if str(value.get("unbuildable", "")).strip():
                transcript.append({"kind": "unbuildable", "reason": value["unbuildable"]})
                return []
            commands = [str(c) for c in (value.get("commands") or []) if str(c).strip()]
            if not commands:
                raise ValueError("commands must be a non-empty list")
            transcript.append({"kind": "resume", "commands": commands,
                               "reasoning": value.get("reasoning")})
            return commands
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            log.append({"attempt": repair + 1, "status": "rejected",
                        "error": redact(f"{type(exc).__name__}: {exc}")})
    return []


def brief(transcript: list[dict[str, Any]], *, limit: int = 30) -> list[dict[str, Any]]:
    """What was tried, compressed to what a reader needs to avoid repeating it."""
    rows = []
    for row in transcript[-limit:]:
        if row.get("kind") in {"reset", "resume", "plan"}:
            rows.append({"note": row.get("kind"), "commands": row.get("commands"),
                         "reasoning": str(row.get("reasoning"))[:300]})
            continue
        rows.append({"command": str(row.get("command"))[:400], "ok": bool(row.get("ok")),
                     "kind": row.get("failure_kind"), "said": str(row.get("excerpt"))[:500]})
    return rows


def reconsider(client: Any, repo: Path, *, record: list[dict[str, Any]],
               transcript: list[dict[str, Any]], manifests: dict[str, str],
               attempts: int = 2) -> dict[str, Any] | None:
    """Ask what constraint to give up, rather than how to run the same command again.

    The distinction is the one a build loop needs and does not have: a command that fails
    can be fixed, and a requirement that cannot hold on this machine cannot. Retrying the
    second is what turns a build into a loop that never terminates and never concludes --
    which is what happened here, where a pin built for hardware the machine does not have
    was retried with every variation of index URL.
    """
    payload = json.dumps({
        "declared_dependencies": {k: v[:3000] for k, v in manifests.items()},
        "placeholders": PLACEHOLDERS,
        "machine": platform_facts(),
        "survived": [row["command"] for row in record],
        "everything_tried": brief(transcript),
    }, ensure_ascii=False)
    for repair in range(attempts):
        content, _ = client.chat_with_metadata(
            RECONSIDER_SYSTEM, payload, max_tokens=3000, timeout=300, thinking="disabled")
        try:
            value = _object(content)
            if str(value.get("unbuildable", "")).strip():
                transcript.append({"kind": "unbuildable", "reason": value["unbuildable"]})
                return None
            commands = [str(c) for c in (value.get("commands") or []) if str(c).strip()]
            if not commands:
                raise ValueError("commands must be a non-empty list")
            transcript.append({"kind": "reconsidered", "constraint": value.get("constraint"),
                               "substitute": value.get("substitute"), "why": value.get("why"),
                               "cost": value.get("cost"), "commands": commands})
            return value
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            if repair == attempts - 1:
                transcript.append({"kind": "reconsider_failed", "error": str(exc)[:300]})
    return None


def diagnose(client: Any, repo: Path, *, record: list[dict[str, Any]],
             transcript: list[dict[str, Any]], probe: str,
             verdict: dict[str, Any]) -> dict[str, Any] | None:
    """Say what is in the way, so a build that stopped is a finding rather than a failure.

    'Rounds exhausted' is the shape a build takes when nobody was asked to conclude. What a
    reader needs is which of the three kinds of blocker this is, on what evidence, and what
    would unblock it -- and the three are worth keeping apart because only one of them is a
    matter of installing something.
    """
    payload = json.dumps({
        "repository": str(repo), "probe": probe, "verdict": verdict,
        "machine": platform_facts(),
        "survived": [row["command"] for row in record],
        "everything_tried": brief(transcript, limit=40),
    }, ensure_ascii=False)
    try:
        content, _ = client.chat_with_metadata(DIAGNOSE_SYSTEM, payload, max_tokens=2000,
                                              timeout=300, thinking="disabled")
        return _object(content)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def probe_environment(probe: str, *, values: dict[str, str], env: dict[str, str], cwd: Path,
                      output: Path, timeout: int = 1800) -> dict[str, Any]:
    return run(substitute(probe, values), env=env, cwd=cwd, timeout=timeout, output=output)


def _finish(output: Path, repo: Path, python: str, record: list[dict[str, Any]],
            transcript: list[dict[str, Any]], probe: str, verdict: dict[str, Any],
            *, client: Any = None) -> dict[str, Any]:
    """Write the result, and if the build did not succeed, write down what is in the way.

    A build that stops without a diagnosis is a build that failed silently, however much
    work it did: "rounds exhausted" tells a reader nothing they can act on, and the loop's
    own record of what it tried is the evidence for the answer.
    """
    result = {"schema_version": 1, "created_at": now(), "repo": str(repo), "python": python,
              "probe": probe, "record": record, "verdict": verdict,
              "content_id": content_id(python, record, repo),
              "survived": len([r for r in record if r.get("kind") != "probe"]),
              "attempted": len(transcript)}
    if not verdict.get("passed") and client is not None:
        result["diagnosis"] = diagnose(client, repo, record=record, transcript=transcript,
                                       probe=probe, verdict=verdict)
        if result["diagnosis"]:
            transcript.append({"kind": "diagnosis", **result["diagnosis"]})
    atomic_json(output / "environment.json", result)
    atomic_json(output / "transcript.json", {"rows": transcript})
    return result


def provision(repo: Path, *, client: Any, output: Path) -> dict[str, Any]:
    """Plan and build, writing down every step whether it worked or not."""
    repo = Path(repo).expanduser().resolve()
    output = Path(output)
    return build(repo, client=client, prefix=output / "env", output=output)
