"""A library of methods the research controller may draw on.

Skills live as `SKILL.md` files on disk rather than as entries in this module, so
adding a method is adding a file -- no code change, no release. Drop one in
`autosim/skills/`, or point `AUTOSIM_SKILLS_DIR` at your own directory and it is
picked up too.

Nothing here is enforced. A skill is evidence and guidance offered to the
controller, which may contradict any of it and should say why when it does. That
is the whole point: a rule narrows what the controller can try, whereas a method
with its evidence attached widens what it can reason about.

Scope decides what a skill is shown for:

    scope: general                 any benchmark, any task
    scope: benchmark:RoboSynChallenge
    scope: task:water_pouring

A benchmark-scoped skill is only surfaced for that benchmark, so a measurement
made on one simulator never masquerades as a universal truth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

#: Where skills are looked for, in order. Later directories do not shadow earlier
#: ones -- every discovered skill is offered.
SEARCH_PATH_ENV = "AUTOSIM_SKILLS_DIR"


def _search_dirs() -> list[Path]:
    dirs = [Path(__file__).resolve().parents[1] / "skills"]
    extra = os.environ.get(SEARCH_PATH_ENV)
    if extra:
        dirs.extend(Path(part).expanduser() for part in extra.split(os.pathsep) if part)
    return [d for d in dirs if d.is_dir()]


def _skill_files() -> list[Path]:
    files: list[Path] = []
    for directory in _search_dirs():
        files.extend(sorted(directory.glob("*/SKILL.md")))   # one directory per skill
        files.extend(sorted(directory.glob("*.md")))         # or a flat file
    # A name appearing twice means the same skill was found in two places; keep the first,
    # which follows the documented search order rather than the filesystem's. The name of a
    # `<name>/SKILL.md` entry is its directory -- keying off the stem would collapse every
    # such skill onto the literal "SKILL" and keep only one of them.
    seen, unique = set(), []
    for path in files:
        key = path.parent.name if path.name == "SKILL.md" else path.stem
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _parse(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        raise ValueError("missing YAML frontmatter")
    _, _, remainder = text.partition("---")
    front, _, body = remainder.partition("\n---")
    meta = yaml.safe_load(front) or {}
    if not isinstance(meta, dict) or not meta.get("name"):
        raise ValueError("frontmatter needs a name")
    return {
        "name": str(meta["name"]),
        "description": str(meta.get("description") or "").strip(),
        "scope": str(meta.get("scope") or "general"),
        "confidence": str(meta.get("confidence") or "unspecified"),
        "evidence": str(meta.get("evidence") or "").strip(),
        "method": body.strip(),
        "source": str(path),
    }


def discover() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Every readable skill, plus a list of files that could not be read.

    A malformed skill is reported, never fatal: a run should not end because
    somebody's new method file has a typo in it.
    """
    skills: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []
    for path in _skill_files():
        try:
            skills.append(_parse(path))
        except Exception as exc:
            problems.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return skills, problems


def _applies(skill: dict[str, Any], benchmark: str | None, task: str | None) -> bool:
    scope = skill["scope"].strip()
    if scope in ("", "general"):
        return True
    kind, _, value = scope.partition(":")
    if kind == "benchmark":
        return benchmark is not None and value == benchmark
    if kind == "task":
        return task is not None and value == task
    # An unrecognised scope is shown rather than silently dropped: a typo should be
    # visible in the prompt, not hide a method from the controller.
    return True


def skills_reference(benchmark: str | None = None, task: str | None = None) -> dict[str, Any]:
    """The reference block handed to the controller each round."""
    skills, problems = discover()
    return {
        "status": "reference_only_not_enforced",
        "how_to_use": (
            "Methods distilled from experience with simulator benchmarks, offered so you do "
            "not have to rediscover them. None is a constraint: nothing validates against "
            "them and you may contradict any of them. Each carries how it was measured and "
            "how confident that measurement is -- weigh them accordingly, and treat a "
            "benchmark-scoped entry as a finding about that benchmark only. If your task or "
            "your evidence says otherwise, do that instead and say why in the hypothesis."
        ),
        "library": {
            "search_paths": [str(d) for d in _search_dirs()],
            "add_a_method_by": "dropping a SKILL.md into a skill directory",
            "unreadable": problems,
        },
        "skills": [s for s in skills if _applies(s, benchmark, task)],
    }
