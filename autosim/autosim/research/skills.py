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

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import yaml

from . import failure_memory

#: Where skills are looked for, in order. Later directories do not shadow earlier
#: ones -- every discovered skill is offered.
SEARCH_PATH_ENV = "AUTOSIM_SKILLS_DIR"

#: The entry that is about the library rather than about benchmarks: what the fields mean, how
#: to weigh a claim against its evidence, and which entry to reach for in which situation. It
#: goes first, because an index sorted into the middle of what it indexes is an index nobody
#: finds -- and because what it says about reading the rest is worth reading before the rest.
GUIDE = "using-the-method-library"


def _search_dirs() -> list[Path]:
    dirs = [Path(__file__).resolve().parents[1] / "skills"]
    extra = os.environ.get(SEARCH_PATH_ENV)
    if extra:
        dirs.extend(Path(part).expanduser() for part in extra.split(os.pathsep) if part)
    # What the system has learned by running, beside what a person wrote. AutoSOTA calls this
    # the third level of its failure memory -- "knowledge notes distilling repeated successes
    # into explicit skills" -- and it is the only level that changes what the system can do at
    # a repository it has never seen: a remedy that held in a second repository has stopped
    # being an anecdote, and it should reach the next controller without a person carrying it.
    #
    # A separate directory, and its entries say in their own `confidence` that nobody wrote
    # them and what they rest on. Mixed into the curated library they would be believed at its
    # weight, which is the failure this system keeps finding in other forms.
    dirs.append(Path(__file__).resolve().parents[3] / "autoresearch_runs"
                / failure_memory.DISTILLED_DIR)
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


_LIST_FIELDS = ("capabilities", "keywords", "inputs", "outputs", "preconditions",
                "verification", "permissions", "resources", "invalid_when")
_MANIFEST_REQUIRED = ("id", "version", *_LIST_FIELDS)
_MAX_FRONTMATTER_BYTES = 128_000


def _read_frontmatter(path: Path) -> dict[str, Any]:
    """Read only the bounded YAML header; retrieval must not load every skill body."""
    lines: list[str] = []
    size = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        if handle.readline().strip() != "---":
            raise ValueError("missing YAML frontmatter")
        for line in handle:
            if line.strip() == "---":
                value = yaml.safe_load("".join(lines)) or {}
                if not isinstance(value, dict) or not str(value.get("name") or "").strip():
                    raise ValueError("frontmatter needs a name")
                return value
            size += len(line.encode("utf-8", errors="replace"))
            if size > _MAX_FRONTMATTER_BYTES:
                raise ValueError("YAML frontmatter exceeds the metadata limit")
            lines.append(line)
    raise ValueError("unterminated YAML frontmatter")


def _metadata_problem(row: dict[str, Any]) -> str:
    missing = [key for key in _MANIFEST_REQUIRED if key not in row]
    if missing:
        return "missing metadata fields: " + ", ".join(missing)
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        return "id must be a non-empty string"
    if not isinstance(row.get("version"), (str, int, float)):
        return "version must be a string or number"
    if not str(row["version"]).strip():
        return "version must be non-empty"
    for field in _LIST_FIELDS:
        values = row.get(field)
        if (not isinstance(values, list) or
                any(not isinstance(value, str) or not value.strip() for value in values)):
            return f"{field} must be a list of non-empty strings"
    if not row["capabilities"]:
        return "capabilities must contain at least one tag"
    return ""


def _scope_problem(value: Any) -> str:
    """Validate the retrieval boundary before a method can reach a controller.

    Unknown scope strings used to be treated as globally applicable. A typo in
    ``benchmark:<name>`` could therefore leak a repository-specific method into every
    unfamiliar benchmark -- exactly the cross-repository contamination scoped skills exist
    to prevent. Invalid scopes remain visible as library diagnostics, not as advice.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return ""
    if not isinstance(value, str):
        return "scope must be 'general', 'benchmark:<name>', or 'task:<name>'"
    scope = value.strip()
    if scope == "general":
        return ""
    kind, separator, target = scope.partition(":")
    if (separator and kind in ("benchmark", "task") and target.strip() and
            target == target.strip()):
        return ""
    return "scope must be 'general', 'benchmark:<name>', or 'task:<name>'"


def _load_manifest() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    root = Path(__file__).resolve().parents[1] / "skills"
    path = root / "manifest.yaml"
    problems: list[dict[str, str]] = []
    if path.is_symlink() or not path.is_file():
        return {}, [{"path": str(path), "error": "skill manifest is missing or unsafe"}]
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, [{"path": str(path), "error": f"{type(exc).__name__}: {exc}"}]
    if (not isinstance(document, dict) or document.get("schema_version") != 1 or
            not isinstance(document.get("skills"), list)):
        return {}, [{"path": str(path), "error": "invalid manifest schema"}]
    rows: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for index, raw in enumerate(document["skills"]):
        if not isinstance(raw, dict):
            problems.append({"path": f"{path}#skills[{index}]", "error": "entry is not a map"})
            continue
        file_value = str(raw.get("file") or "")
        relative = Path(file_value)
        if (not file_value or relative.is_absolute() or ".." in relative.parts or
                relative.suffix.lower() != ".md"):
            problems.append({"path": f"{path}#skills[{index}]", "error": "unsafe skill file path"})
            continue
        problem = _metadata_problem(raw)
        if problem:
            problems.append({"path": f"{path}#{raw.get('id', index)}", "error": problem})
            continue
        skill_path = root / relative
        if skill_path.is_symlink() or not skill_path.is_file():
            problems.append({"path": str(skill_path), "error": "manifest target is missing or unsafe"})
            continue
        try:
            resolved = skill_path.resolve()
            resolved.relative_to(root.resolve())
            front = _read_frontmatter(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            problems.append({"path": str(skill_path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        expected_name = relative.parent.name if relative.name == "SKILL.md" else relative.stem
        if str(front["name"]) != expected_name:
            problems.append({"path": str(skill_path), "error": "frontmatter name does not match manifest path"})
            continue
        scope_problem = _scope_problem(front.get("scope", "general"))
        if scope_problem:
            problems.append({"path": str(skill_path), "error": scope_problem})
            continue
        identity = str(raw["id"])
        if identity in seen_ids:
            problems.append({"path": str(skill_path), "error": f"duplicate skill id: {identity}"})
            continue
        seen_ids.add(identity)
        rows[file_value] = {**raw, "name": str(front["name"]),
                            "description": str(front.get("description") or "").strip(),
                            "scope": str(front.get("scope") or "general"),
                            "confidence": str(front.get("confidence") or "unspecified"),
                            "evidence": str(front.get("evidence") or "").strip(),
                            "source": str(resolved), "maturity": "curated"}
    return rows, problems


def _external_metadata(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Custom/distilled skills must carry their own versioned retrieval contract."""
    try:
        front = _read_frontmatter(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    problem = _metadata_problem(front)
    if problem:
        return None, problem
    scope_problem = _scope_problem(front.get("scope", "general"))
    if scope_problem:
        return None, scope_problem
    return {**front, "name": str(front["name"]),
            "description": str(front.get("description") or "").strip(),
            "scope": str(front.get("scope") or "general"),
            "confidence": str(front.get("confidence") or "unspecified"),
            "evidence": str(front.get("evidence") or "").strip(),
            "source": str(path.resolve()), "maturity": "external_unverified"}, ""


def _metadata_index() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = (Path(__file__).resolve().parents[1] / "skills").resolve()
    curated, problems = _load_manifest()
    entries: list[dict[str, Any]] = []
    files = _skill_files()
    found_curated: set[str] = set()
    seen_ids: set[str] = set()
    for path in files:
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else None
        except (OSError, RuntimeError, ValueError):
            relative = None
        if relative is not None:
            row = curated.get(relative)
            if row is None:
                problems.append({"path": str(path), "error": "built-in skill is not in manifest"})
                continue
            found_curated.add(relative)
        else:
            row, problem = _external_metadata(path)
            if row is None:
                problems.append({"path": str(path), "error": problem})
                continue
        identity = str(row.get("id") or "").strip()
        if identity in seen_ids:
            problems.append({"path": str(path), "error": f"duplicate skill id: {identity}"})
            continue
        seen_ids.add(identity)
        entries.append(dict(row))
    for relative in sorted(set(curated) - found_curated):
        problems.append({"path": relative, "error": "manifest entry was not discovered"})
    # Keep first-seen order, with the method-library guide first as its stable index.
    entries = ([row for row in entries if row["name"] == GUIDE]
               + [row for row in entries if row["name"] != GUIDE])
    return entries, problems


def _parse(path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    first, _, remainder = text.partition("\n")
    if first.strip() != "---":
        raise ValueError("missing YAML frontmatter")
    front, delimiter, body = remainder.partition("\n---")
    meta = yaml.safe_load(front) or {}
    if not isinstance(meta, dict) or not meta.get("name") or not delimiter:
        raise ValueError("frontmatter needs a name and closing delimiter")
    row = dict(metadata or {})
    if row and str(row.get("name")) != str(meta["name"]):
        raise ValueError("skill body identity changed after indexing")
    contract = {field: row.get(field, meta.get(field)) for field in
                ("id", "version", *_LIST_FIELDS)}
    return {**row, **contract,
            "name": str(meta["name"]),
            "description": str(meta.get("description") or row.get("description") or "").strip(),
            "scope": str(meta.get("scope") or row.get("scope") or "general"),
            "confidence": str(meta.get("confidence") or row.get("confidence") or "unspecified"),
            "evidence": str(meta.get("evidence") or row.get("evidence") or "").strip(),
            "method": body.strip(),
            "source": str(path.resolve()),
            "body_sha256": hashlib.sha256(body.strip().encode("utf-8")).hexdigest()}


def discover() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Every schema-valid readable skill, plus files that need curation.

    A malformed skill is reported, never fatal: a run should not end because
    somebody's new method file has a typo in it.
    """
    index, problems = _metadata_index()
    by_source = {row["source"]: row for row in index}
    skills: list[dict[str, Any]] = []
    for path in _skill_files():
        try:
            row = by_source.get(str(path.resolve()))
            if row is None:
                continue
            skills.append(_parse(path, row))
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
    # Discovery validates and reports invalid scopes before retrieval. Keep the final
    # applicability check fail-closed as a defence against callers passing an ad-hoc row.
    return False


def _tokens(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        text = " ".join(str(one) for one in value)
    else:
        text = str(value or "")
    return {word for word in re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
            if len(word) > 1}


def _relevance(query: str, skill: dict[str, Any]) -> tuple[int, list[str]]:
    query_words = _tokens(query)
    if not query_words:
        return 0, []
    caps = _tokens(skill.get("capabilities"))
    keywords = _tokens(skill.get("keywords"))
    descriptive = _tokens([skill.get("name"), skill.get("description"),
                           skill.get("preconditions"), skill.get("invalid_when")])
    cap_hits = sorted(query_words & caps)
    keyword_hits = sorted(query_words & keywords)
    text_hits = sorted(query_words & descriptive)
    score = 4 * len(cap_hits) + 2 * len(keyword_hits) + len(text_hits)
    return score, cap_hits + keyword_hits + text_hits


def skills_catalog(benchmark: str | None = None, task: str | None = None, *,
                   query: str = "", max_recommended: int = 3) -> dict[str, Any]:
    """A short, body-free directory from which the controller chooses what to read.

    Lexical relevance only marks recommendations; it never removes an applicable method
    or authorizes reading one. The main agent's explicit ID selection is a separate step.
    """
    if max_recommended < 0 or max_recommended > 12:
        raise ValueError("max_recommended must be between 0 and 12")
    index, problems = _metadata_index()
    applicable = [row for row in index if _applies(row, benchmark, task)]
    scores = {row["id"]: _relevance(query, row)[0] for row in applicable}
    ranked = sorted((row for row in applicable if row["name"] != GUIDE and
                     scores[row["id"]] > 0),
                    key=lambda row: -scores[row["id"]])
    recommended = {row["id"] for row in ranked[:max_recommended]}
    return {
        "contract": "This is a directory, not skill content or an applicability verdict. "
                    "Choose zero to three IDs based on the current uncertainty and give a "
                    "reason for each. Keyword recommendations are hints only; any listed "
                    "ID may be chosen. Read bodies before relying on a method. Skills grant "
                    "no permissions or resources.",
        "entries": [{"id": row["id"], "version": str(row["version"]),
                     "name": row["name"], "description": str(row["description"])[:200],
                     "scope": row["scope"], "confidence": str(row["confidence"])[:120],
                     "recommended_by_keyword": row["id"] in recommended}
                    for row in applicable],
        "unreadable_count": len(problems),
    }


def read_selected_skills(ids: list[str], *, benchmark: str | None = None,
                         task: str | None = None) -> dict[str, Any]:
    """Read only explicitly selected, applicable skill bodies, with stable identities."""
    if (len(ids) > 3 or any(not isinstance(identity, str) for identity in ids) or
            len(ids) != len(set(ids))):
        raise ValueError("select at most three distinct skills")
    index, problems = _metadata_index()
    allowed = {row["id"]: row for row in index if _applies(row, benchmark, task)}
    if any(not isinstance(identity, str) or identity not in allowed for identity in ids):
        raise ValueError("skill ID is not in the applicable catalog")
    bodies: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for identity in ids:
        row = allowed[identity]
        try:
            body = _parse(Path(row["source"]), row)
        except Exception as exc:
            problems.append({"path": row["source"],
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        bodies.append(body)
        selection.append({"id": identity, "version": str(row["version"]),
                          "scope": row["scope"], "body_sha256": body["body_sha256"],
                          "applicability": "candidate_requires_current_run_verification"})
    return {"status": "reference_only_not_enforced", "selection": selection,
            "skills": bodies, "index": [], "unreadable": problems}


def skills_reference(benchmark: str | None = None, task: str | None = None, *,
                     query: str = "", max_selected: int = 3,
                     include_index: bool = True) -> dict[str, Any]:
    """Return a compact metadata index plus only the few bodies matching this question."""
    if max_selected < 0 or max_selected > 12:
        raise ValueError("max_selected must be between 0 and 12")
    index, problems = _metadata_index()
    applicable = [skill for skill in index if _applies(skill, benchmark, task)]
    ranked = []
    for order, skill in enumerate(applicable):
        score, matches = _relevance(query, skill)
        if skill["name"] != GUIDE and score > 0:
            ranked.append((score, order, matches, skill))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen = [row[3] for row in ranked[:max_selected]]
    guide = next((skill for skill in applicable if skill["name"] == GUIDE), None)
    selected_rows = ([guide] if guide else []) + chosen
    # Deduplicate the guide if the query itself matched it.
    selected_rows = list({row["id"]: row for row in selected_rows}.values())
    full_by_source: dict[str, dict[str, Any]] = {}
    selection = []
    ranked_by_id = {row[3]["id"]: (row[0], row[2]) for row in ranked}
    for row in selected_rows:
        path = Path(row["source"])
        try:
            full_by_source[row["source"]] = _parse(path, row)
        except Exception as exc:
            problems.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        score, matches = ranked_by_id.get(row["id"], (0, []))
        selection.append({"id": row["id"], "version": str(row["version"]),
                          "scope": row["scope"], "score": score,
                          "reason": ("library orientation included first" if
                                     row["name"] == GUIDE else
                                     "matched metadata: " + ", ".join(matches)),
                          "applicability": "candidate_requires_current_run_verification",
                          "body_sha256": full_by_source[row["source"]]["body_sha256"]})
    index_view = [{key: skill.get(key) for key in (
        "id", "version", "name", "description", "scope", "confidence", "capabilities",
        "keywords", "inputs", "outputs", "preconditions", "verification", "permissions",
        "resources", "invalid_when", "maturity")}
        for skill in applicable] if include_index else []
    return {
        "status": "reference_only_not_enforced",
        "contract": ("The metadata index, when included, is advisory. Selection is lexical "
                     "relevance, not an "
                     "applicability verdict. For each selected method, compare preconditions "
                     "and invalid_when against this run and record whether it was used or "
                     "declined with evidence; recheck its recommendation against current source "
                     "and receipts. Bodies are suggestions, not rules; skills grant no permissions "
                     "or resources. No name match or prior use proves applicability."),
        "query": query,
        "selection": selection,
        "library": {
            "search_paths": [str(d) for d in _search_dirs()],
            "add_a_method_by": "add a versioned manifest entry or complete metadata to an external SKILL.md",
            "unreadable": problems,
        },
        "index": index_view,
        "skills": [full_by_source[row["source"]] for row in selected_rows
                   if row["source"] in full_by_source],
    }
