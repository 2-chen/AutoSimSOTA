"""Changing the benchmark's own source, in the smallest form that can be checked and undone.

Most real improvements are here and not in a configuration value. AutoSOTA's report puts 51%
of its improvements in algorithmic change against 33% in tuning, and its search space is
explicitly "the entire unrestricted source code" rather than a declared set of knobs -- the
paper frames that as the thing that separates it from AutoML and from sandboxed systems.

This loop could not reach it at all. A proposal named settings from a declared space, so a
fault in the benchmark's own code -- a path computed one way and read another, a loader that
opens a name the data does not have, a guard whose condition is subtly wrong -- could be
diagnosed correctly and never fixed. RoboTwin produced three such diagnoses in one run and
stopped on each of them.

**A find-and-replace, and not a file rewrite.** Both can express the change; only one can be
checked before it is applied and undone exactly afterwards. `find` has to occur exactly once,
which is what makes the patch a statement about *this* text rather than about a text that
happened to match; the result has to parse, when the file is Python; and the text it replaced
is kept, so reverting is exact rather than a second guess at what was there.

What bounds it is not here. **The red lines bound it**, before the patch is applied: a file
the benchmark declares as its evaluator cannot be patched at all, and a change that reads as a
change to measurement is refused with the line named. This module is the mechanism; the
permission is `redlines`' and the record of what was changed is the snapshot's.
"""

from __future__ import annotations

import ast
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: How much of a file may be replaced. A patch is a correction, not a rewrite: a `find` longer
#: than this is a rewrite expressed as a patch, and the reason for the size limit is that the
#: review it gets is reading a diff.
MAX_FIND = 4000
MAX_PATCHES = 8


@dataclass
class Patch:
    """One replacement in one file."""

    file: str
    find: str
    replace: str

    @classmethod
    def from_change(cls, change: Any) -> "Patch | None":
        if not isinstance(change, dict):
            return None
        file = str(change.get("file") or "").strip()
        find = str(change.get("find") or "")
        replace = str(change.get("replace") or "")
        if not file or not find:
            return None
        return cls(file=file, find=find, replace=replace)

    def as_dict(self) -> dict[str, Any]:
        return {"file": self.file, "find": self.find, "replace": self.replace}


def patches_from_change(change: Any) -> list[Patch]:
    """One legacy replacement or a bounded multi-file transaction."""
    if isinstance(change, dict) and "patches" in change:
        raw = change.get("patches")
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PATCHES:
            return []
        patches = [Patch.from_change(row) for row in raw]
        return [one for one in patches if one is not None] if all(patches) else []
    one = Patch.from_change(change)
    return [one] if one else []


def check_many(patches: list[Patch], *, repo: Path) -> list[str]:
    if not patches or len(patches) > MAX_PATCHES:
        return [f"a code change needs 1 to {MAX_PATCHES} concrete patches"]
    files = [one.file for one in patches]
    if len(files) != len(set(files)):
        return ["multiple replacements in one file need one unambiguous combined patch"]
    return [problem for one in patches for problem in check(one, repo=repo)]


def apply_many(patches: list[Patch], *, repo: Path) -> dict[str, str]:
    """Apply all checked files, rolling back every earlier file if any write fails."""
    problems = check_many(patches, repo=repo)
    if problems:
        raise ValueError("; ".join(problems))
    before: dict[str, str] = {}
    try:
        for one in patches:
            before[one.file] = apply(one, repo=repo)
    except Exception:                                                    # noqa: BLE001
        for one in reversed(patches):
            if one.file in before:
                revert(one, before[one.file], repo=repo)
        raise
    return before


def check(patch: Patch, *, repo: Path) -> list[str]:
    """Why this cannot be applied, as messages. Empty means applicable — not *correct*.

    Every fault here is one that would otherwise be discovered by running, which is the more
    expensive way to discover all of them. `find` occurring twice is the interesting one: the
    change would be applied to one of two identical places and the run would not know which,
    and the fix is a longer `find` rather than a guess.
    """
    problems: list[str] = []
    root = Path(repo).resolve()
    relative = Path(patch.file)
    if (relative.is_absolute() or not relative.parts or
            any(part in {"", ".", ".."} for part in relative.parts) or
            relative.as_posix() != patch.file):
        return [f"{patch.file} escapes this checkout"]
    target = root / relative
    if not target.is_relative_to(root):
        return [f"{patch.file} escapes this checkout"]
    parent = root
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            return [f"{patch.file} escapes this checkout through a symlink"]
        if not parent.is_dir():
            return [f"{patch.file} is not a file in this checkout"]
    if target.is_symlink():
        return [f"{patch.file} escapes this checkout through a symlink"]
    if not target.is_file():
        return [f"{patch.file} is not a file in this checkout"]
    if len(patch.find) > MAX_FIND:
        problems.append(f"the text to replace is {len(patch.find)} characters, which is a "
                        f"rewrite rather than a patch; a patch is something a reader can "
                        f"review as a diff")
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{patch.file} could not be read as text: {type(exc).__name__}"]
    found = text.count(patch.find)
    if found == 0:
        problems.append(f"the text to replace does not occur in {patch.file}. It has to be a "
                        f"line the file actually contains, exactly as the file spells it")
    elif found > 1:
        problems.append(f"the text to replace occurs {found} times in {patch.file}, so which "
                        f"one is meant is undecidable. Make it longer until it is unique")
    if patch.find == patch.replace:
        problems.append("the patch replaces the text with itself")
    if not problems and target.suffix == ".py":
        after = text.replace(patch.find, patch.replace, 1)
        try:
            ast.parse(after)
        except SyntaxError as exc:
            problems.append(f"the file would no longer parse: {type(exc).__name__}: {exc}")
    return problems


def apply(patch: Patch, *, repo: Path) -> str:
    """Make the change and return what was there, so it can be put back exactly.

    Raises `ValueError` when `check` found something — a caller that applies without checking
    has decided the faults do not matter, and it should have to say so by catching rather than
    by not looking.
    """
    problems = check(patch, repo=repo)
    if problems:
        raise ValueError("; ".join(problems))
    target = Path(repo).resolve() / patch.file
    before = target.read_text(encoding="utf-8")
    after = before.replace(patch.find, patch.replace, 1)
    _atomic_replace(target, after.encode("utf-8"), stat.S_IMODE(target.stat().st_mode),
                    expected=before.encode("utf-8"), relative=patch.file)
    return before


def revert(patch: Patch, before: str, *, repo: Path) -> None:
    """Put the file back exactly.

    From the text that was there and not by replacing in the other direction: a `replace` that
    occurs elsewhere in the file would be undone in the wrong place, and a patch that was
    applied and then edited by something else would be undone over the top of that.
    """
    root = Path(repo).resolve()
    relative = Path(patch.file)
    if (relative.is_absolute() or not relative.parts or
            any(part in {"", ".", ".."} for part in relative.parts) or
            relative.as_posix() != patch.file):
        raise ValueError(f"unsafe code patch path: {patch.file}")
    target = root / relative
    parent = root
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"unsafe code patch parent: {patch.file}")
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"unsafe code patch target: {patch.file}")
    before_bytes = before.encode("utf-8")
    if not patch.find or before.count(patch.find) != 1:
        raise ValueError(f"saved source does not uniquely match patch: {patch.file}")
    after_bytes = before.replace(patch.find, patch.replace, 1).encode("utf-8")
    current = target.read_bytes()
    if current == before_bytes:
        return
    if current != after_bytes:
        raise ValueError(f"source diverged after the candidate change: {patch.file}")
    _atomic_replace(target, before_bytes, stat.S_IMODE(target.stat().st_mode),
                    expected=after_bytes, relative=patch.file)


def _atomic_replace(target: Path, contents: bytes, mode: int, *, expected: bytes,
                    relative: str) -> None:
    """Replace one already-validated source file without exposing a partial write."""
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.patch")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        # Do not overwrite a concurrent edit made after validation/read.
        if target.is_symlink() or not target.is_file() or target.read_bytes() != expected:
            raise RuntimeError(f"source changed while applying patch: {relative}")
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def diff(patch: Patch) -> str:
    """The change as a reader sees it, for the record."""
    removed = "\n".join(f"- {line}" for line in patch.find.splitlines())
    added = "\n".join(f"+ {line}" for line in patch.replace.splitlines())
    return f"--- {patch.file}\n{removed}\n{added}"
