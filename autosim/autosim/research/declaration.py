"""A benchmark described as data, and the checks that decide how much of it is true.

The adapter protocol settled *what* a benchmark must tell the system. It left open *who
writes it down*, and the answer was a person. This module closes that: a declaration is a
JSON document, a model can draft one from a repository survey, and everything a declaration
claims is either checked against the filesystem or marked unverified.

The split matters, because a model drafting facts about a benchmark it has never run will
be confidently wrong about some of them. So a declaration is never trusted:

* A **claim** is what the model wrote -- a path, an entry point, a dimension.
* A **check** is what the system did about it: the path exists, the file parses, the
  command ran.
* A **capability** is only ever as verified as its weakest check, and the vocabulary of
  capabilities is the system's own need rather than any benchmark's feature list.

That last point is why `CAPABILITIES` lives here and not in an adapter. "Can this benchmark
produce new successful trajectories" is a question the system asks of every benchmark; that
LIBERO answers it by handing you a teleoperation script is LIBERO's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .adapter_protocol import Axis, OptimizationSpace, check_space
from .common import digest, read_json
from .contracts import CapabilityRecord
from .registry import TaskSpec


#: What the system needs from any benchmark, in its own words. A declaration must answer
#: every one of these, including answering "no". The prompt is built from this mapping, so
#: adding a capability the system learns to use is one entry here and no prompt edit.
CAPABILITIES: dict[str, str] = {
    "native_evaluation": "run this benchmark's own evaluator to score a policy on the "
                         "benchmark's own initial states and success predicate",
    "official_dataset": "a released set of demonstrations this benchmark authorises as "
                        "training data",
    "official_policy": "a released, already-trained policy to start from or compare against",
    "new_trajectory_generation": "produce new successful trajectories without a human in "
                                 "the loop; this is the one that decides whether the "
                                 "research loop can collect data at all",
    "targeted_generation": "bias the above toward a chosen region of the initial-state or "
                           "failure distribution",
    "training": "train or fine-tune a policy on demonstrations",
    "export": "package a trained policy so the benchmark's own evaluator loads it",
}

#: Statuses a declaration may give a capability, and what each obliges.
STATUSES = ("declared", "unsupported", "unknown")

#: Per-status required keys. `declared` is a promise, so it must name its evidence; the
#: other two are answers, so they must give a reason.
_STATUS_REQUIRES = {
    "declared": ("entrypoint", "evidence"),
    "unsupported": ("why",),
    "unknown": ("why",),
}

AXIS_KINDS = ("choice", "integer", "number", "structure")

#: Characters that make a declared path a pattern rather than a literal location.
_GLOB_CHARS = frozenset("*?[")


#: Every key a complete declaration carries, in the order a reader wants them.
REQUIRED_KEYS = ("benchmark", "evidence", "repo_markers", "tasks", "task_contract",
                 "assets", "capabilities", "optimization_space")


def problems_in(declaration: dict[str, Any],
                required: Sequence[str] = REQUIRED_KEYS) -> list[str]:
    """Structural faults in a draft, as messages. Empty means well-formed, not true.

    Only shape is checked here. Whether the paths exist is `verify`'s business, because
    the two failures need different answers: a malformed field is a retry, a missing file
    is a finding.

    ``required`` narrows which keys must be present, so a draft can be judged on the part
    it was asked for. A declaration is drafted in more than one pass, and a fault in the
    first pass reported against the second sends the model to fix something it was never
    asked about -- which it then changes, for no reason, twice.
    """
    if not isinstance(declaration, dict):
        return ["declaration must be one JSON object"]
    problems: list[str] = []
    for key in required:
        if key not in declaration:
            problems.append(f"missing required key: {key}")
    if problems:
        return problems
    absent = set(REQUIRED_KEYS) - set(declaration)

    if not str(declaration["benchmark"]).strip():
        problems.append("benchmark must name the project")
    if not isinstance(declaration["evidence"], str) or not declaration["evidence"].strip():
        problems.append("evidence must say what in the repository supports this reading")

    markers = declaration.get("repo_markers")
    if markers is not None and (not isinstance(markers, list) or not markers or not all(
            isinstance(m, str) and m.strip() for m in markers)):
        problems.append("repo_markers must be a non-empty list of paths that prove identity")

    if "tasks" not in declaration:
        return problems
    tasks = declaration["tasks"]
    if not isinstance(tasks, dict):
        problems.append("tasks must be an object")
    else:
        for key in ("pattern", "task_id_from", "kind"):
            if not str(tasks.get(key, "")).strip():
                problems.append(f"tasks.{key} is required")
        if tasks.get("kind") not in {"glob", "directory", "module_registry", "explicit"}:
            problems.append(f"tasks.kind must be one of glob, directory, module_registry, explicit")
        pattern = str(tasks.get("pattern", ""))
        if "<" in pattern or ">" in pattern:
            # A template reads like a pattern and is not one: it matches nothing, so the
            # task list comes back empty and the run stops at a stage whose cause is two
            # fields away. If the answer is "every suite", the glob is `*/*.bddl`.
            problems.append("tasks.pattern still contains a placeholder; give a glob that "
                            "matches the task files, or use kind=directory")

    contract = declaration.get("task_contract")
    if contract is None:
        pass
    elif not isinstance(contract, dict):
        problems.append("task_contract must be an object")
    else:
        for key in ("state_dim", "action_dim", "cameras", "max_episode_steps"):
            if contract.get(key) in (None, "", {}, []):
                problems.append(f"task_contract.{key} is required")
        if not isinstance(contract.get("cameras", {}), dict):
            problems.append("task_contract.cameras must map a camera name to its shape")
        for key in ("state_dim", "action_dim", "max_episode_steps"):
            value = contract.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value < 1):
                problems.append(f"task_contract.{key} must be a positive integer")

    caps = declaration.get("capabilities")
    if caps is None:
        pass
    elif not isinstance(caps, dict):
        problems.append("capabilities must be an object")
    else:
        # An unrecognised name is rejected rather than ignored. The system only knows how
        # to verify the capabilities it has, so a model that invents one -- or renames one
        # to fit its answer -- would otherwise reach verification with a name nothing
        # checks, which is how a claim silently becomes absent.
        unknown = sorted(set(caps) - set(CAPABILITIES))
        if unknown:
            problems.append(f"capabilities has no such entry: {unknown}; "
                            f"answer exactly {sorted(CAPABILITIES)}")
        for name in CAPABILITIES:
            if name not in caps:
                problems.append(f"capabilities.{name} must be answered, including with "
                                f"'unsupported'")
                continue
            row = caps[name]
            if not isinstance(row, dict):
                problems.append(f"capabilities.{name} must be an object")
                continue
            status = row.get("status")
            if status not in STATUSES:
                problems.append(f"capabilities.{name}.status must be one of {list(STATUSES)}")
                continue
            for key in _STATUS_REQUIRES[status]:
                if not str(row.get(key, "")).strip():
                    problems.append(f"capabilities.{name} claims {status} and must give {key}")
            entrypoint = row.get("entrypoint")
            if isinstance(entrypoint, str) and entrypoint.strip() and _is_prose(entrypoint):
                problems.append(f"capabilities.{name}.entrypoint must be a path, not a "
                                f"sentence; the explanation belongs in evidence")

    if "assets" in declaration:
        for name, asset in (declaration["assets"] or {}).items():
            if not isinstance(asset, dict):
                problems.append(f"assets.{name} must be an object")
                continue
            path = asset.get("path")
            if path in (None, ""):
                continue
            if not isinstance(path, str):
                problems.append(f"assets.{name}.path must be a string or omitted")
            elif not _is_named_path(path):
                problems.append(f"assets.{name}.path does not name a location; if there is "
                                f"none, omit path and answer in assets.{name}.why")
            elif _is_prose(path):
                # A remark typed into the path field fails every existence check while
                # looking like it ought to pass. Both facts are wanted; they belong in
                # different fields.
                problems.append(f"assets.{name}.path must be a path, not a sentence; the "
                                f"explanation belongs in assets.{name}.why")

    if "optimization_space" not in declaration:
        return problems
    problems.extend(_space_problems(declaration["optimization_space"]))
    if not problems:
        # Structural faults in the space are reported here rather than at first use, so a
        # draft that would fail later fails while the model can still be asked to fix it.
        try:
            check_space(space_from(declaration))
        except ValueError as exc:
            problems.extend(str(exc).split("; "))
    return problems


def _space_problems(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return ["optimization_space must be an object"]
    problems: list[str] = []
    for section in ("collection", "training"):
        if not isinstance(raw.get(section), list):
            problems.append(f"optimization_space.{section} must be a list of axes")
    extra = raw.get("extra", {})
    if not isinstance(extra, dict):
        problems.append("optimization_space.extra must map a section name to a list of axes")
    return problems


def _axes(raw: Sequence[dict[str, Any]], section: str, problems: list[str]) -> tuple[Axis, ...]:
    built = []
    for row in raw:
        if not isinstance(row, dict):
            problems.append(f"{section}: each axis must be an object")
            continue
        name, kind = row.get("name"), row.get("kind")
        if not str(name or "").strip():
            problems.append(f"{section}: an axis has no name")
            continue
        if kind not in AXIS_KINDS:
            problems.append(f"{section}.{name}: kind must be one of {list(AXIS_KINDS)}")
            continue
        kwargs: dict[str, Any] = {}
        if kind == "choice":
            values = row.get("values")
            if not isinstance(values, list) or not values:
                problems.append(f"{section}.{name}: a choice axis needs a non-empty values list")
                continue
            kwargs["values"] = tuple(values)
        elif kind == "structure":
            # The decision layer cannot check a shape it does not understand, and a
            # declaration cannot ship a predicate, so a declared structure is unconstrained
            # by construction and says so rather than pretending to validate.
            kwargs["validator"] = None
        else:
            low, high = row.get("low"), row.get("high")
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) \
                    or isinstance(low, bool) or isinstance(high, bool) or low >= high:
                problems.append(f"{section}.{name}: a {kind} axis needs low < high")
                continue
            kwargs.update(low=float(low), high=float(high))
        built.append(Axis(name=str(name), kind=kind,
                          description=str(row.get("description", "")),
                          default=row.get("default"),
                          group=str(row.get("group", "") or ""),
                          optional=bool(row.get("optional", False)), **kwargs))
    return tuple(built)


def space_from(declaration: dict[str, Any]) -> OptimizationSpace:
    """The declared optimization space, or a ValueError naming what is wrong with it."""
    raw = declaration.get("optimization_space")
    problems = _space_problems(raw)
    if problems:
        raise ValueError("; ".join(problems))
    # Axis-level faults are collected, not discarded. An axis the builder refuses is not
    # merely absent from the space -- it is a knob the controller was told it had and does
    # not, so the declaration says one thing and the run does another unless this raises.
    faults: list[str] = []
    extra = tuple((str(name), _axes(axes, f"extra.{name}", faults))
                  for name, axes in (raw.get("extra") or {}).items())
    space = OptimizationSpace(
        collection=_axes(raw["collection"], "optimization_space.collection", faults),
        training=_axes(raw["training"], "optimization_space.training", faults),
        extra=extra)
    if faults:
        raise ValueError("; ".join(faults))
    structural = check_space(space)
    if structural:
        raise ValueError("; ".join(structural))
    return space


@dataclass
class DeclarativeAdapter:
    """An adapter that reads its facts instead of containing them.

    It implements the same protocol the hand-written adapters do, so the decision layer
    cannot tell the difference -- which is the point. What it cannot do is execute: it has
    no trainer and no collector, so any claim it makes about *behaviour* is reported with
    the verification that actually backs it rather than being assumed.
    """

    declaration: dict[str, Any]
    repo: Path
    verification: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.repo = Path(self.repo).expanduser().resolve()
        self.benchmark = str(self.declaration["benchmark"])

    # -- the required protocol -------------------------------------------------------

    def tasks(self) -> list[str]:
        return list(self.verification.get("tasks") or [])

    def select_task(self, requested: str) -> str:
        available = self.tasks()
        if requested == "auto":
            if not available:
                raise ValueError("the declaration enumerates no tasks")
            return available[0]
        if requested not in available:
            raise ValueError(f"unknown task: {requested}")
        return requested

    def discover(self, task: str) -> dict[str, Any]:
        if task not in self.tasks():
            raise ValueError(f"unknown task: {task}")
        return {
            "benchmark": self.benchmark,
            "task": task,
            "repo": str(self.repo),
            "declaration_evidence": self.declaration["evidence"],
            "located": {name: row for name, row in
                        (self.verification.get("assets") or {}).items()},
            "declared": {name: {"status": row.get("status"), "entrypoint": row.get("entrypoint"),
                                "why": row.get("why")}
                         for name, row in self.declaration["capabilities"].items()},
            "verification_state": self.verification.get("state", "unverified"),
        }

    def task_contract(self, task: str) -> dict[str, Any]:
        return self.task_spec(task).as_dict()

    def task_spec(self, task: str) -> TaskSpec:
        if task not in self.tasks():
            raise ValueError(f"unknown task: {task}")
        contract = self.declaration["task_contract"]
        cameras = tuple(contract["cameras"])
        return TaskSpec(
            name=task,
            env_id=str(contract.get("env_id", task)),
            setting=str(contract.get("setting", "default")),
            max_episode_steps=int(contract["max_episode_steps"]),
            state_dim=int(contract["state_dim"]),
            action_dim=int(contract["action_dim"]),
            cameras=cameras,
            camera_shapes={name: list(shape) for name, shape in contract["cameras"].items()},
            control_parts=tuple(contract.get("control_parts", ())),
            recorded_fps=float(contract.get("recorded_fps", 0.0)),
            instruction=self._instruction(task),
            gym_config=str(contract.get("gym_config", "")),
            action_config=str(contract.get("action_config", "")),
            roles={"declared": True})

    def _instruction(self, task: str) -> str:
        instruction = self.declaration["task_contract"].get("instruction")
        if isinstance(instruction, str):
            return instruction
        if isinstance(instruction, dict):
            # A per-task sentence the scout could read from a catalogue but not restate.
            path = instruction.get("from")
            if path and Path(path).is_file():
                table = read_json(Path(path))
                if isinstance(table, dict):
                    return str(table.get(task, ""))
        return task.replace("_", " ")

    def capabilities(self, task: str) -> list[CapabilityRecord]:
        """One record per capability, never more verified than the checks behind it."""
        records = []
        for name, declared in self.declaration["capabilities"].items():
            check = (self.verification.get("capabilities") or {}).get(name, {})
            records.append(CapabilityRecord(
                capability_id=name, scope=task,
                status=check.get("status", "unknown"),
                evidence={"declared_status": declared.get("status"),
                          "entrypoint": declared.get("entrypoint"),
                          "checks": check.get("checks", []),
                          "declared_evidence": declared.get("evidence")},
                limitation=check.get("limitation") or declared.get("why"),
                probe_version=1))
        return records

    def optimization_space(self, task: str) -> OptimizationSpace:
        return space_from(self.declaration)

    def resolve_assets(self, task: str) -> dict[str, Any]:
        return dict(self.verification.get("assets") or {})

    def challenge_context(self, spec: TaskSpec) -> dict[str, Any]:
        return {"benchmark": self.benchmark,
                "declaration_evidence": self.declaration["evidence"],
                "instruction": spec.instruction}


def _is_named_path(value: str) -> bool:
    """Does this name a location, rather than answer a question in the path field?

    A path either has a directory separator or a filename extension. `unsupported`, `none`
    and `null` have neither, and each of them is a status that belongs in `why` -- but they
    pass every prose check, because a single word has no space in it.
    """
    text = value.strip()
    return "/" in text or Path(text).suffix != ""


def _is_prose(value: str) -> bool:
    """Does this look like an explanation someone typed into a path field?

    Two ways it happens: a space, or a length no filesystem path reaches. Both are cheap to
    catch, and left alone each one fails an existence check in a way that reads as "the
    file is missing" rather than "the wrong thing is in this field".
    """
    text = value.strip()
    return " " in text or len(text) > 300


def _check_path(repo: Path, relative: str) -> dict[str, Any]:
    """Does this path, or this pattern, resolve inside the checkout?

    A declaration may legitimately name a set of files -- one task definition per suite, one
    demonstration file per task -- and a pattern is the honest way to say so. Requiring one
    literal path would force the scout to name a single file and let it stand for all of
    them, which is a smaller claim than the truth and a weaker check than the count.
    """
    if _GLOB_CHARS & set(relative):
        root, _, name = relative.rpartition("/")
        matches = sorted((repo / root).glob(name)) if root else sorted(repo.glob(name))
        if not matches:
            matches = sorted(repo.glob(relative))
        return {"check": "path_exists", "subject": relative, "passed": bool(matches),
                "kind": "pattern", "matches": len(matches),
                "examples": [str(path.relative_to(repo)) for path in matches[:3]
                             if path.is_relative_to(repo)]}
    target = Path(relative)
    if not target.is_absolute():
        target = repo / target
    if target.is_dir():
        return {"check": "path_exists", "subject": relative, "passed": True, "kind": "directory"}
    if target.is_file():
        return {"check": "path_exists", "subject": relative, "passed": True, "kind": "file",
                "bytes": target.stat().st_size, "sha256": digest(target)}
    return {"check": "path_exists", "subject": relative, "passed": False}


def verify(declaration: dict[str, Any], repo: Path, *,
           survey_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check what can be checked without running the benchmark's own stack.

    The checks here are cheap and honest: does the path exist, does the file parse, does the
    enumerated task list agree with what was surveyed. They are deliberately *not* a smoke
    run -- executing an unknown benchmark's evaluator needs its environment, and pretending
    otherwise is how a report ends up claiming more than was tested.

    A capability is verified when every check behind it passed. It is `unsupported` when its
    own declaration says so, `failed` when a check contradicted it, and `unknown` when there
    was nothing cheap to check -- which is the truthful answer for anything about behaviour.
    """
    repo = Path(repo).expanduser().resolve()
    report: dict[str, Any] = {"benchmark": declaration["benchmark"], "repo": str(repo),
                              "checks": [], "capabilities": {}, "assets": {}}

    for marker in declaration["repo_markers"]:
        report["checks"].append(_check_path(repo, marker))
    # One marker resolving is enough to confirm identity. Markers are corroboration the
    # model offers, not a checklist it must complete: requiring all of them lets a single
    # typo in a fifth marker invalidate a correct identification, and every capability that
    # depends on identity then fails for a reason that has nothing to do with the benchmark.
    markers = [row for row in report["checks"] if row.get("check") == "path_exists"]
    resolved = [row for row in markers if row["passed"]]
    report["identity"] = "confirmed" if resolved else "not_found"
    report["identity_failures"] = [row["subject"] for row in markers if not row["passed"]]
    report["identity_markers_resolved"] = [row["subject"] for row in resolved]

    # The task list is a path claim like any other when it is stated as one, so the pattern
    # is counted rather than trusted to have matched.
    pattern = str(declaration["tasks"].get("pattern", ""))
    if declaration["tasks"]["kind"] in {"glob", "directory"} and pattern:
        report["checks"].append(_check_path(repo, pattern))
    tasks = _enumerate_tasks(declaration, repo, report)
    report["tasks"] = tasks
    report["task_count"] = len(tasks)
    report["task_pattern"] = declaration["tasks"].get("pattern")
    kind = declaration["tasks"]["kind"]
    if kind == "module_registry":
        # The names came from a module the system never imported, so there is nothing here
        # that confirms them. Reporting that as a pass would launder an assertion into a
        # check, which is the one thing this report exists to prevent.
        report["checks"].append({"check": "tasks_enumerated", "passed": None,
                                 "count": len(tasks), "examples": tasks[:5],
                                 "note": "names were listed from a module the system did not "
                                         "import; a glob over the task files would be "
                                         "countable"})
    elif not tasks:
        report["checks"].append({"check": "tasks_enumerated", "passed": False,
                                 "pattern": declaration["tasks"].get("pattern")})
    else:
        report["checks"].append({"check": "tasks_enumerated", "passed": True,
                                 "count": len(tasks), "examples": tasks[:5]})

    if survey_report is not None:
        surveyed = {row["path"] for row in survey_report.get("datasets", [])}
        surveyed |= {row["path"] for row in survey_report.get("model_artifacts", [])}
        report["survey_agreement"] = _agree(declaration, repo, surveyed)
        observed = observed_contract(survey_report)
        report["observed_contract"] = observed
        # Computed by the survey, which is the only place that opened the files. Recomputing
        # it here would be a second implementation of the same question, and the two would
        # eventually disagree about which of them had the newer answer.
        provenance = survey_report.get("recorded_provenance") or {}
        report["recorded_provenance"] = provenance
        report["checks"].append({
            "check": "asset_provenance_resolves",
            "passed": bool(provenance.get("resolve_inside_repo")) or None,
            "resolved": (provenance.get("resolve_inside_repo") or [])[:5],
            "note": "paths the recorded data names about itself; ones that resolve inside "
                    "the checkout establish that the file belongs to this repository"})
        if observed:
            report["checks"].append(_contract_agreement(declaration["task_contract"], observed))

    for name, declared in declaration["capabilities"].items():
        report["capabilities"][name] = _verify_capability(name, declared, repo, report)
    for name, asset in (declaration.get("assets") or {}).items():
        report["assets"][name] = _verify_asset(asset, repo)
    if survey_report is not None:
        # A failed path is more useful with the file it was reaching for than with the
        # string it wrote: the difference between the two is the whole finding, and saying
        # only "it does not exist" invites the reader to conclude the file is absent when
        # it was found under a slightly different name.
        candidates = [item["path"] for item in
                      survey_report.get("datasets", []) + survey_report.get("model_artifacts", [])]
        for row in report["assets"].values():
            if row.get("status") == "failed":
                row["nearest_surveyed"] = _nearest(row.get("path") or "", candidates)

    report["state"] = ("identity_unconfirmed" if not resolved else
                       "verified" if all(row["status"] != "failed"
                                         for row in report["capabilities"].values())
                       else "partly_contradicted")
    return report


def _enumerate_tasks(declaration: dict[str, Any], repo: Path,
                     report: dict[str, Any]) -> list[str]:
    """The task list, read the way the declaration says this benchmark states it."""
    tasks = declaration["tasks"]
    kind, pattern = tasks["kind"], str(tasks["pattern"])
    if kind == "explicit":
        return [str(name) for name in tasks.get("names", [])]
    if kind == "module_registry":
        report["checks"].append({
            "check": "task_registry_import", "passed": False,
            "reason": "reading a module registry needs the benchmark's environment; "
                      "the declaration lists no names instead"})
        return [str(name) for name in tasks.get("names", [])]
    if kind == "directory":
        root = repo / pattern
        if not root.is_dir():
            return []
        return sorted(child.name for child in root.iterdir() if child.is_dir())
    found = sorted(path.stem for path in repo.glob(pattern) if path.is_file())
    return found


def _nearest(name: str, candidates: list[str], limit: int = 3) -> list[str]:
    """Surveyed files whose name shares the most with a path that failed to resolve.

    A path that misses by a directory is the usual way a correct answer is written wrongly,
    because the data was downloaded beside the checkout rather than inside it.
    """
    stem = Path(str(name)).name
    if not stem or set(stem) <= {"*", "_", "."}:
        return candidates[:limit]
    parts = {piece for piece in stem.replace("*", "").split("_") if len(piece) > 3}
    return sorted(candidates,
                  key=lambda row: -len(parts & set(Path(row).name.split("_"))))[:limit]


def _agree(declaration: dict[str, Any], repo: Path, surveyed: set[str]) -> dict[str, Any]:
    """Do the assets the declaration claims appear in the survey that was taken of the repo?

    Not a contradiction check -- a scout is allowed to find a file the survey's budget
    skipped -- but a disagreement is worth recording, because it is exactly how a
    hallucinated path shows up.
    """
    out: dict[str, Any] = {}
    for name, asset in (declaration.get("assets") or {}).items():
        for key in ("path",):
            value = asset.get(key)
            if isinstance(value, str) and value:
                target = Path(value)
                out[name] = {"declared": value, "seen_in_survey": str(target) in surveyed,
                             "exists": (target if target.is_absolute() else repo / target).exists()}
                break
    return out


def observed_contract(survey_report: dict[str, Any]) -> dict[str, Any]:
    """The task contract as the *data* states it, where the survey could read it.

    A model that never opened a demonstration file writes a plausible action dimension and
    moves on. This reads the real one out of the recorded trajectories, so the two can be
    compared -- the disagreement is the useful part, and it is invisible to any check that
    only asks whether a field is present and positive.
    """
    for row in survey_report.get("dataset_structure", []):
        groups = row.get("groups") or {}
        for group in groups.values():
            first = (group.get("first") or {})
            if not isinstance(first, dict):
                continue
            actions = first.get("actions")
            observation = (first.get("obs") or {}).get("group")
            if not isinstance(actions, dict) or not isinstance(observation, dict):
                continue
            cameras = {name: shape for name, value in observation.items()
                       if name.endswith("_rgb") and (shape := value.get("shape"))
                       and len(shape) == 4}
            return {"source": row["path"],
                    "action_dim": actions["shape"][-1],
                    "cameras": {name: shape[1:] for name, shape in cameras.items()},
                    "observed_keys": sorted(observation),
                    "episodes_in_file": group.get("count")}
    return {}


def _contract_agreement(declared: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Does the declared contract match what the trajectories actually contain?"""
    disagreements = []
    if declared.get("action_dim") != observed["action_dim"]:
        disagreements.append(f"action_dim: declared {declared.get('action_dim')}, "
                             f"recorded {observed['action_dim']}")
    seen = {name: list(shape) for name, shape in observed["cameras"].items()}
    if seen and not set(declared.get("cameras", {})) & set(seen):
        disagreements.append(f"cameras: declared {sorted(declared.get('cameras', {}))}, "
                             f"recorded {sorted(seen)}")
    return {"check": "contract_matches_recorded_data", "passed": not disagreements,
            "source": observed["source"], "disagreements": disagreements,
            "observed_cameras": seen, "observed_keys": observed["observed_keys"]}


def _verify_asset(asset: dict[str, Any], repo: Path,
                  prior: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Exists is not the same as belongs here, and the record says which was checked.

    A checkpoint belonging to a neighbouring project passes every cheap check there is:
    the path is real, the file opens, the size is plausible. Nothing here can decide
    ownership, so it records where the asset sits relative to the checkout and leaves the
    claim legible rather than silently promoting it.
    """
    path = asset.get("path") or asset.get("checkpoint") or asset.get("dataset")
    if not path or not _is_named_path(str(path)):
        return {"status": "unknown", "why": asset.get("why", "the declaration names no path")}
    check = _check_path(repo, str(path))
    target = Path(str(path))
    if not target.is_absolute():
        target = repo / target
    try:
        inside = target.resolve().is_relative_to(repo)
    except OSError:
        inside = False
    status = "verified" if check["passed"] else "failed"
    limitation = None
    if status == "verified" and not inside:
        # A pattern can point outside the checkout by construction: `../datasets/libero/*.hdf5`
        # is how a benchmark's downloaded data is usually found. Ownership is then a claim
        # this pass cannot make, so it says so rather than promoting the path to verified.
        status = "declared"
        limitation = ("the path exists but lies outside the checkout, so nothing here "
                      "establishes that this benchmark owns it")
    return {"status": status, "path": str(path), "check": check,
            "inside_repo": inside, "declared_format": asset.get("format"),
            "limitation": limitation, "why": asset.get("why")}


def _verify_capability(name: str, declared: dict[str, Any], repo: Path,
                       report: dict[str, Any]) -> dict[str, Any]:
    """A capability is as verified as the checks behind it, and no more.

    `unsupported` needs no check: a benchmark that says it cannot do this is making a claim
    about absence, and the honest response is to record it and note that absence is not
    something cheap checks can confirm. It is reported as `unsupported` with the reason
    attached, so a run that stops here stops with a sentence rather than a stack trace.
    """
    status = declared.get("status")
    checks: list[dict[str, Any]] = []
    if status == "unsupported":
        return {"status": "unsupported", "limitation": declared.get("why"),
                "declared_status": status, "checks": []}
    if status == "unknown":
        return {"status": "unknown", "limitation": declared.get("why"),
                "declared_status": status, "checks": []}

    entrypoint = declared.get("entrypoint")
    if isinstance(entrypoint, str) and entrypoint:
        check = _check_path(repo, entrypoint)
        checks.append(check)
        if not check["passed"]:
            return {"status": "failed", "checks": checks,
                    "limitation": f"the declared entry point does not exist: {entrypoint}"}
    if report.get("identity") != "confirmed":
        checks.append({"check": "repo_identity", "passed": False})
        return {"status": "failed", "checks": checks,
                "limitation": "the repository does not match the declared identity"}
    return {"status": "declared", "checks": checks,
            "limitation": "no check in this pass exercises behaviour; run the benchmark's "
                          "own stack to promote this beyond a promise"}
