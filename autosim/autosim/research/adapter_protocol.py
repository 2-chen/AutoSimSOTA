"""What a benchmark must tell the system, and what the system must never assume.

The system had two research loops, each with its own proposal schema, its own
prompt, and its own validation. Adding a benchmark meant writing another one of
each, and the two copies drifted. The parts that differed were almost entirely
*facts about the benchmark* -- RoboSynChallenge varies a collection profile and a
sampling mass, RoboTwin varies a setting and an epoch count -- but the decision
layer had those facts written into it rather than asked for.

This module makes the facts explicit. An adapter declares what can be varied; the
decision layer builds its prompt and its validation from that declaration and
never mentions a benchmark by name. Adding a benchmark is then writing an
adapter, with no change to the layer that decides.

Two rules hold this together:

* **The adapter declares, the decision layer consumes.** If the decision layer
  needs to know something, it is a field here, not a branch on a task name.
* **Absent means unsupported.** An adapter that does not implement an optional
  method is treated as not offering that capability, rather than the system
  guessing a default for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Sequence

from .contracts import CapabilityRecord


@dataclass(frozen=True)
class Axis:
    """One thing the controller may vary, and what it may set it to.

    `kind` decides which fields matter:

    ``choice``   a finite set in `values`
    ``integer``  a whole number in ``[low, high]``
    ``number``   a real number in ``[low, high]``

    `default` is what the prompt shows as the starting point. It is a suggestion:
    it is the value the adapter has reason to believe works, not the boundary of
    what may be proposed.
    """

    name: str
    kind: str
    description: str
    values: tuple[Any, ...] = ()
    low: float | None = None
    high: float | None = None
    default: Any = None
    #: Optional nesting inside the proposal, for benchmarks whose trainer takes its
    #: parameters as a group. An axis with no group sits directly under
    #: ``collection`` or ``training``; one with a group sits under that group. The
    #: nesting is the benchmark's shape, so it is declared here rather than assumed
    #: by the decision layer.
    group: str = ""
    #: For ``kind='structure'``: a predicate the adapter supplies, returning truthy
    #: when the value is one it can execute. Absent means the axis accepts anything.
    validator: Any = None
    #: An optional axis may be left out of a proposal entirely. It is still validated when
    #: present, and it is not shown in the skeleton -- omitting it means "use the default".
    optional: bool = False

    def accepts(self, value: Any) -> bool:
        try:
            if self.kind == "structure":
                # A shape the decision layer has no business understanding -- a list of
                # progress bins, a per-phase weight map. The adapter owns the check; the
                # generic layer only insists that one exists.
                return self.validator is None or bool(self.validator(value))
            if self.kind == "choice":
                return value in self.values
            if self.kind == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    return False
                return self.low <= value <= self.high
            if self.kind == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return False
                return self.low <= float(value) <= self.high
        except TypeError:
            return False
        return False

    def explain(self) -> str:
        """What this axis accepts, in a form the controller can act on."""
        if self.kind == "choice":
            return f"one of {list(self.values)}"
        if self.kind == "structure":
            return self.description
        return f"a {self.kind} in [{self.low}, {self.high}]"

    def describe(self) -> dict[str, Any]:
        if self.kind == "choice":
            return {"kind": self.kind, "values": list(self.values),
                    "default": self.default, "meaning": self.description}
        return {"kind": self.kind, "min": self.low, "max": self.high,
                "default": self.default, "meaning": self.description}


@dataclass(frozen=True)
class OptimizationSpace:
    """Everything the controller may vary for one task.

    Two dicts rather than one because the system treats them differently: the
    collection half decides what data exists, the training half decides what is
    done with it, and a proposal names both.
    """

    collection: tuple[Axis, ...] = ()
    training: tuple[Axis, ...] = ()
    #: Further top-level sections beyond collection and training. RoboSynChallenge uses
    #: one for the round's own evidence resolution, which is neither a collection knob
    #: nor a training knob. Declaring sections rather than fixing two of them is what
    #: keeps the next benchmark's third section from needing a change here.
    extra: tuple[tuple[str, tuple[Axis, ...]], ...] = ()

    def sections(self) -> dict[str, tuple[Axis, ...]]:
        out = {"collection": self.collection, "training": self.training}
        out.update(dict(self.extra))
        return out

    def axes(self, group: str) -> tuple[Axis, ...]:
        return self.sections().get(group, ())

    def axis(self, group: str, name: str) -> Axis | None:
        return next((a for a in self.axes(group) if a.name == name), None)

    def describe(self) -> dict[str, Any]:
        return {name: {a.name: a.describe() for a in axes}
                for name, axes in self.sections().items()}

    def skeleton(self) -> dict[str, Any]:
        """The starting point shown to the controller: declared defaults only."""
        return {name: _nest(axes) for name, axes in self.sections().items()}


def _nest(axes: Sequence[Axis]) -> dict[str, Any]:
    """Group axes into the nesting their declaration asks for."""
    out: dict[str, Any] = {}
    for axis in axes:
        if axis.optional and not axis.group:
            continue
        if not axis.group:
            out[axis.name] = axis.default
            continue
        out.setdefault(axis.group, {})[axis.name] = axis.default
    return out


def check_space(space: OptimizationSpace) -> list[str]:
    """Structural problems with a declared space, as messages (empty is good)."""
    problems = []
    for group, axes in space.sections().items():
        seen = set()
        for axis in axes:
            if axis.name in seen:
                problems.append(f"duplicate axis {group}.{axis.name}")
            seen.add(axis.name)
            if axis.kind not in {"choice", "integer", "number", "structure"}:
                problems.append(f"{group}.{axis.name}: unknown kind {axis.kind!r}")
            elif axis.kind == "choice" and not axis.values:
                problems.append(f"{group}.{axis.name}: choice axis declares no values")
            elif axis.kind in {"integer", "number"} and (
                    axis.low is None or axis.high is None or axis.low >= axis.high):
                problems.append(f"{group}.{axis.name}: needs low < high")
            if axis.default is not None and not axis.accepts(axis.default):
                problems.append(f"{group}.{axis.name}: default {axis.default!r} is not accepted")
    return problems


#: Methods every adapter must provide. The decision layer is written against
#: exactly this list; anything else an adapter offers is its own business.
REQUIRED_METHODS = (
    "tasks",              # () -> list[str]
    "select_task",        # (requested: str) -> str
    "discover",           # (task: str) -> dict
    "task_contract",      # (task: str) -> dict   what the controller is told about the task
    "capabilities",       # (task: str) -> list[CapabilityRecord]
    "optimization_space", # (task: str) -> OptimizationSpace
)

#: Methods an adapter may omit. Omitting one is a statement that the benchmark
#: does not offer that capability, not a gap for the system to paper over.
OPTIONAL_METHODS = (
    "resolve_assets",     # (task: str) -> dict   benchmarks that ship pinned assets
    "challenge_context",  # (task: str) -> dict   benchmark-authored framing for the controller
)


def check_adapter(adapter: Any, *, known_tasks: Sequence[str] | None = None) -> list[str]:
    """Report what an adapter is missing or has got wrong. Empty means usable.

    Called before a run starts rather than at the first failure, so an incomplete
    adapter is a message instead of a crash halfway through a collection.
    """
    problems = [f"missing required method: {name}" for name in REQUIRED_METHODS
                if not callable(getattr(adapter, name, None))]
    if problems:
        return problems
    try:
        tasks = adapter.tasks()
    except Exception as exc:
        return [f"tasks() raised {type(exc).__name__}: {exc}"]
    if not tasks:
        return ["adapter enumerates no tasks"]
    if known_tasks is not None:
        unknown = sorted(set(tasks) - set(known_tasks))
        if unknown:
            problems.append(f"adapter returns tasks the system does not know: {unknown}")
    sample = tasks[0]
    try:
        space = adapter.optimization_space(sample)
    except Exception as exc:
        return problems + [f"optimization_space({sample!r}) raised {type(exc).__name__}: {exc}"]
    problems.extend(f"{sample}: {message}" for message in check_space(space))
    if not space.collection and not space.training:
        problems.append(f"{sample}: declares nothing the controller may vary")
    return problems
