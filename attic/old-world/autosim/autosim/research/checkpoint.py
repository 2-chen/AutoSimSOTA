"""What a checkpoint is, as opposed to where its weights happen to be.

The system identified a trained checkpoint by `sha256(model.safetensors)` -- forty-eight call
sites of that one expression, spread across the harness, the probes and the validators. It is
a good hash of the wrong thing.

**A checkpoint is not its weights.** A directory holding weights, the config that says what
shape they are, the optimizer state that says how far the training got, and the RNG state that
says what it would do next is *four* facts, and hashing one of them makes two checkpoints
identical when they differ in every part but the weights. Two runs stopped at different steps,
with different optimizers and different random streams, had the same identity -- and the
identity is what the ledger, the evaluation request and the round comparison all key on.

**And its layout is a convention, not a law.** `train/checkpoints/{steps:06d}/pretrained_model`
with `model.safetensors` inside is LeRobot's, and it was written into the runtime as the shape
a checkpoint has. A benchmark whose trainer writes `ckpt-{step}.pt` beside a `config.yaml` is
not wrong; it is a different trainer.

So this reads the directory and reports what is in it, by role. Roles come from names, because
names are what a trainer chose deliberately; the identity covers the ordered list of files that
were found, because that is what a checkpoint *is*. Nothing here decides whether the contents
are good -- `environment_contract.inspect_act_checkpoint` and
`experiment_validation.checkpoint_audit` do that, and they are separate questions asked
separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import object_digest


#: Role names, matched against a file's name. Ordered by how specific each is, so that
#: `optimizer_state.safetensors` is the optimizer rather than a second copy of the weights.
_ROLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("optimizer", re.compile(r"optim(izer)?[_-]?(state|param)", re.IGNORECASE)),
    ("rng", re.compile(r"(rng|random)[_-]?state", re.IGNORECASE)),
    ("step", re.compile(r"^(training_)?step.*\.json$", re.IGNORECASE)),
    ("config", re.compile(r"^(train_)?config\.(json|yaml|yml)$", re.IGNORECASE)),
    ("weights", re.compile(r"\.(safetensors|pt|pth|ckpt|bin)$", re.IGNORECASE)),
)

#: Names that are pointers or bookkeeping rather than content. `last` is a symlink to the most
#: recent step; hashing it would make two checkpoints equal whenever they point at the same
#: place, which is the opposite of what an identity is for.
_NOT_CONTENT = re.compile(r"^(last|latest|current)$", re.IGNORECASE)


def role_of(name: str) -> str:
    for role, pattern in _ROLES:
        if pattern.search(name):
            return role
    return "other"


@dataclass(frozen=True)
class Checkpoint:
    """A directory of files that together are one trained policy."""

    root: Path
    #: role -> the files that carry it, relative to the root, in a stable order.
    roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Files that exist but matched no role. Kept, because a file nobody named is still part
    #: of what the checkpoint is, and dropping it would make two different checkpoints equal.
    other: tuple[str, ...] = ()

    @classmethod
    def at(cls, root: Path) -> "Checkpoint | None":
        """Read a checkpoint directory, or return None if there is not one there.

        Recursive, because state lives in subdirectories as often as beside the weights, and
        bounded by being a directory read rather than a search: whatever the trainer wrote
        under this path is the checkpoint.
        """
        root = Path(root)
        if not root.is_dir():
            return None
        roles: dict[str, list[str]] = {}
        other: list[str] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            name = path.name
            if _NOT_CONTENT.match(name):
                continue
            relative = str(path.relative_to(root))
            if name.startswith("."):
                continue
            role = role_of(name)
            (other if role == "other" else roles.setdefault(role, [])).append(relative)
        return cls(root=root, roles={k: tuple(v) for k, v in sorted(roles.items())},
                   other=tuple(other))

    # -- what it is ---------------------------------------------------------------------

    def files(self) -> tuple[str, ...]:
        """Every content file, in a stable order: roles first, then the unnamed ones."""
        ordered: list[str] = []
        for role in sorted(self.roles):
            ordered.extend(self.roles[role])
        ordered.extend(self.other)
        return tuple(ordered)

    def role(self, name: str) -> Path | None:
        """The path for a role, or None when the checkpoint does not carry it."""
        files = self.roles.get(name)
        return self.root / files[0] if files else None

    def has(self, *roles: str) -> bool:
        return all(self.roles.get(role) for role in roles)

    def missing(self, *roles: str) -> list[str]:
        """Which of these roles are absent. A finding, not a fault: not every trainer saves
        optimizer state, and a checkpoint that does not is not a broken one."""
        return [role for role in roles if not self.roles.get(role)]

    def identity(self, *, roles: tuple[str, ...] | None = None) -> str:
        """A digest of the files, by role and content.

        Paths *and* contents, because two checkpoints whose files were reordered or renamed
        are not the same checkpoint, and a digest over contents alone would say they were.
        Bounded to the roles asked for when a caller wants a cheaper answer, and defaulting
        to everything -- which is the honest default, since the whole point is that the
        weights are not the whole of it.
        """
        wanted = set(roles) if roles is not None else None
        parts: list[dict[str, Any]] = []
        for relative in self.files():
            role = role_of(Path(relative).name)
            if wanted is not None and role not in wanted:
                continue
            path = self.root / relative
            if not path.is_file():
                continue
            from .common import digest
            parts.append({"path": relative, "role": role, "sha256": digest(path)})
        return object_digest(parts)

    def as_dict(self) -> dict[str, Any]:
        return {"root": str(self.root), "roles": {k: list(v) for k, v in self.roles.items()},
                "other": list(self.other), "identity": self.identity()}


def identity_of(path: Path, *, roles: tuple[str, ...] | None = None) -> str | None:
    """The identity of a checkpoint at this path, or None when there is not one.

    The convenience the forty-eight call sites wanted: they wrote
    `digest(checkpoint / "model.safetensors")`, and what they meant was "which checkpoint is
    this".
    """
    found = Checkpoint.at(path)
    return found.identity(roles=roles) if found is not None else None
