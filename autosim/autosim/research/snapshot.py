"""What the run changed, kept so it can be put back, and the best state kept so it can be kept.

Once a run can change the benchmark's code and its environment, two things become necessary
and neither existed.

**Undo.** A change that does not work has to leave nothing behind, or the next round runs on
top of it and the round after that on top of both. AutoSOTA commits before every modification
and restores with `git checkout` when a crash or an exhausted debug budget intervenes -- its
seven-step iteration has the snapshot as step two, before the change and not after it.

**A best.** AutoSOTA advances a `_best` tag whenever an iteration beats the recorded optimum
and restores it at the end of the run, so what is exported is the best state reached and not
the last one. Without it a run's result is whatever it happened to finish on, and a loop whose
later rounds go worse reports the worse number as its finding.

**Content-addressed, not a git repository.** The obvious implementation is the one AutoSOTA
uses, and it is available here -- the checkouts are git repositories. Writing commits into a
benchmark's own history to record what an experiment did is a larger side effect than the
experiment, and it is the user's repository. What this keeps instead is the text of the files
it touched, keyed by digest, in the run's own directory: the same idea as everywhere else in
this system, where what survives is what succeeded and it is addressed by what it is.

The set of files is the run's own -- what it changed -- and not the checkout. A snapshot of
the whole repository would be a second copy of the benchmark, and the parts of it that matter
are the parts something wrote to.
"""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_json, digest, now

#: The scale that is the run's answer rather than a measure of how close it is to one.
_THE_ANSWER = "success_rate"


@dataclass
class Snapshot:
    """One moment: the files that were not as the checkout had them, by digest."""

    name: str
    at: str
    files: dict[str, str] = field(default_factory=dict)   # path -> sha256 of its content
    score: float | None = None
    why: str = ""
    #: What `score` is measured in. Two scales reach `_best` and they are not comparable: the
    #: benchmark's success rate, which is the answer, and the objective's completion, which is
    #: how far a run that cannot yet produce the answer has got. A completion of 0.6 is not
    #: better than a success rate of 0.4 -- they are different questions -- and a `_best` that
    #: compared them would restore a state on the strength of the wrong one.
    scale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "at": self.at, "files": self.files, "score": self.score,
                "why": self.why, "scale": self.scale}


class Snapshots:
    """The states this run passed through, and the best of them.

    `blobs/` holds each file's content once, keyed by its digest; a snapshot is a mapping from
    path to digest. Two snapshots that share a file share the bytes, which matters because a
    loop that tries ten variations of one file should not hold ten copies of it to be able to
    go back.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.path = self.root / "snapshots.json"
        self.rows: list[Snapshot] = []
        self._best: str = ""
        if self.path.is_file():
            try:
                document = json.loads(self.path.read_text(encoding="utf-8"))
                self.rows = [Snapshot(**row) for row in document.get("snapshots") or []]
                self._best = str(document.get("best") or "")
            except (OSError, ValueError, TypeError):
                self.rows, self._best = [], ""

    # -- reading ------------------------------------------------------------------------

    def get(self, name: str) -> Snapshot | None:
        return next((one for one in self.rows if one.name == name), None)

    def best(self) -> Snapshot | None:
        return self.get(self._best)

    def history(self) -> list[dict[str, Any]]:
        return [one.as_dict() for one in self.rows]

    # -- writing ------------------------------------------------------------------------

    def capture(self, paths: Iterable[str], *, repo: Path, name: str, score: float | None = None,
                why: str = "") -> Snapshot:
        """Record what these files hold right now.

        Named by the caller rather than numbered, because a reader of the record has to be able
        to tell which state is which: `before-round-3` says something a sequence number does
        not. A file that is not there is recorded as absent rather than skipped -- a snapshot
        that silently omits a file cannot restore the state it claims to describe.
        """
        files: dict[str, str] = {}
        root = Path(repo).resolve()
        for relative in paths:
            target = (root / relative).resolve()
            if Path(relative).is_absolute() or not target.is_relative_to(root):
                raise ValueError(f"snapshot path escapes repository: {relative}")
            if not target.is_file():
                files[relative] = ""
                continue
            try:
                text = target.read_bytes()
            except OSError:
                files[relative] = ""
                continue
            key = digest(target)
            files[relative] = key
            self.blobs.mkdir(parents=True, exist_ok=True)
            blob = self.blobs / key
            if not blob.is_file() or digest(blob) != key:
                temporary = blob.with_name(blob.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    temporary.write_bytes(text)
                    if digest(temporary) != key:
                        raise RuntimeError(f"source changed while saving snapshot: {relative}")
                    os.replace(temporary, blob)
                finally:
                    temporary.unlink(missing_ok=True)
        snapshot = Snapshot(name=name, at=now(), files=files, score=score, why=why)
        self.rows = [one for one in self.rows if one.name != name] + [snapshot]
        self._write()
        return snapshot

    def restore(self, name: str, *, repo: Path) -> list[str]:
        """Put every file in that snapshot back, and say which ones moved.

        The ones that moved, not the ones that were in it: a restore that reports everything it
        touched cannot be told from one that changed nothing, and the run's next decision
        depends on knowing whether anything was actually undone.
        """
        snapshot = self.get(name)
        if snapshot is None:
            return []
        moved: list[str] = []
        root = Path(repo).resolve()
        checked: list[tuple[str, Path, bytes | None]] = []
        for relative, key in snapshot.files.items():
            target = (root / relative).resolve()
            if Path(relative).is_absolute() or not target.is_relative_to(root):
                raise ValueError(f"snapshot restore path escapes repository: {relative}")
            if not key:
                checked.append((relative, target, None))
                continue
            blob = self.blobs / key
            if not blob.is_file():
                raise FileNotFoundError(f"snapshot blob missing for {relative}: {key}")
            if digest(blob) != key:
                raise ValueError(f"snapshot blob corrupt for {relative}: {key}")
            checked.append((relative, target, blob.read_bytes()))
        for relative, target, after in checked:
            if after is None:
                if target.is_file():
                    target.unlink()
                    moved.append(relative)
                continue
            if target.is_file():
                try:
                    if target.read_bytes() == after:
                        continue
                except OSError:
                    pass
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(after)
            moved.append(relative)
        return moved

    def restore_patched(self, name: str, patches: Iterable[Any], *, repo: Path) -> list[str]:
        """Idempotently undo one recorded patch transaction, refusing later edits.

        Each source file must still be exactly either the state saved by ``name`` or the
        complete result of its corresponding find/replace. This accepts a crash between
        files (and a repeated recovery) while refusing to overwrite an edit that is not part
        of this transaction. All paths and snapshot blobs are checked before the first write;
        individual restores use same-directory atomic replacement.
        """
        from .codepatch import Patch

        snapshot = self.get(name)
        if snapshot is None:
            raise FileNotFoundError(f"pre-change snapshot is missing: {name}")
        if self.root.is_symlink() or self.blobs.is_symlink():
            raise ValueError("guarded rollback snapshot storage contains a symlink")
        rows = list(patches)
        if not rows or any(not isinstance(one, Patch) for one in rows):
            raise ValueError("guarded rollback needs concrete source patches")
        paths = [one.file for one in rows]
        if len(paths) != len(set(paths)) or set(paths) != set(snapshot.files):
            raise ValueError("patch paths do not exactly match the pre-change snapshot")

        root = Path(repo).resolve()
        checked: list[tuple[str, Path, bytes, bytes, int]] = []
        for patch in rows:
            relative = Path(patch.file)
            if (relative.is_absolute() or not relative.parts or
                    any(part in {"", ".", ".."} for part in relative.parts) or
                    relative.as_posix() != patch.file):
                raise ValueError(f"unsafe guarded rollback path: {patch.file}")
            target = root / relative
            if not target.is_relative_to(root):
                raise ValueError(f"guarded rollback path escapes repository: {patch.file}")
            parent = root
            for part in relative.parts[:-1]:
                parent = parent / part
                if parent.is_symlink() or not parent.is_dir():
                    raise ValueError(f"guarded rollback parent is unsafe: {patch.file}")
            if target.is_symlink():
                raise ValueError(f"guarded rollback refuses a symlink: {patch.file}")

            key = snapshot.files.get(patch.file)
            if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError(f"pre-change snapshot has no file bytes: {patch.file}")
            blob = self.blobs / key
            if blob.is_symlink() or not blob.is_file():
                raise FileNotFoundError(f"snapshot blob missing for {patch.file}: {key}")
            if digest(blob) != key:
                raise ValueError(f"snapshot blob corrupt for {patch.file}: {key}")
            before = blob.read_bytes()
            try:
                before_text = before.decode("utf-8")
                find = patch.find.encode("utf-8")
                after_text = before_text.replace(patch.find, patch.replace, 1)
            except (AttributeError, UnicodeDecodeError) as exc:
                raise ValueError(f"patch or snapshot is not valid UTF-8: {patch.file}") from exc
            if not patch.find or before_text.count(patch.find) != 1:
                raise ValueError(f"patch does not uniquely match its snapshot: {patch.file}")
            after = after_text.encode("utf-8")
            if not target.exists() or not target.is_file():
                raise ValueError(f"guarded rollback target is missing or not a file: {patch.file}")
            info = target.stat()
            current = target.read_bytes()
            if current not in (before, after):
                raise ValueError(f"source diverged after the candidate change: {patch.file}")
            checked.append((patch.file, target, before, current, stat.S_IMODE(info.st_mode)))

        moved: list[str] = []
        for relative, target, before, expected_current, mode in checked:
            # Narrow the race window: never replace a file that changed after preflight.
            if target.is_symlink() or not target.is_file() or target.read_bytes() != expected_current:
                raise ValueError(f"source changed during guarded rollback: {relative}")
            if expected_current == before:
                continue
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.rollback")
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(before)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, mode)
                os.replace(temporary, target)
                directory = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)
            moved.append(relative)
        return moved

    def advance(self, name: str, *, score: float | None, why: str = "",
                scale: str = "") -> bool:
        """Point `_best` at this state, when it is better than what is there.

        Better, and not merely newer: a run whose later rounds go worse has to be able to report
        the earlier, better state as its result, and a tag that advances on every capture makes
        the last attempt the answer. Where nothing was measured, nothing advances -- a state
        that produced no number is not better than one that produced a number.

        Two scales reach here and are never compared. A state carrying the benchmark's success
        rate is the run's answer and outranks one carrying only the objective's completion; a
        later state on the completion scale does not displace an earlier measured one, and vice
        versa. Comparing them would let a run restore a state because it had got further toward
        measuring, over one that had measured.
        """
        snapshot = self.get(name)
        if snapshot is None or score is None:
            return False
        current = self.best()
        if current is not None and current.score is not None:
            if current.scale == scale and current.score >= score:
                return False
            if current.scale != scale and (
                    current.scale == _THE_ANSWER or current.scale.startswith("metric:")):
                return False
        snapshot.score = score
        snapshot.scale = scale
        snapshot.why = why or snapshot.why
        self._best = name
        self._write()
        return True

    def _write(self) -> None:
        atomic_json(self.path, {"schema_version": 1, "at": now(), "best": self._best,
                                "snapshots": [one.as_dict() for one in self.rows]})
