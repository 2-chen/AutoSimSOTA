"""The method library indexes first and loads only bodies useful to this question."""

import hashlib
from pathlib import Path
import re

import pytest
import yaml

from autosim.research import skills as skill_library
from autosim.research.skills import (GUIDE, discover, read_selected_skills,
                                     skills_catalog, skills_reference)


NEW_SIMULATION_METHODS = (
    "choosing-simulation-improvements",
    "aligning-policy-inputs-and-actions",
    "tuning-action-chunks-and-feedback",
    "validating-policy-training",
    "improving-observations-and-memory",
    "adapting-pretrained-policies",
    "learning-from-policy-failures",
)


@pytest.mark.parametrize("name", NEW_SIMULATION_METHODS)
def test_simulation_methods_are_selectable_without_keyword_recommendation(name):
    identity = "autosimsota." + name
    for benchmark in ("RoboTwin", "LIBERO", "UnseenArmRepository"):
        catalog = skills_catalog(benchmark=benchmark, query="zzzxxy qqqqv")
        row = next(item for item in catalog["entries"] if item["id"] == identity)
        assert not row["recommended_by_keyword"]
        assert "method" not in row
        result = read_selected_skills([identity], benchmark=benchmark)
        selected = result["skills"][0]
        assert selected["id"] == identity
        assert selected["scope"] == "general"
        assert selected["body_sha256"] == hashlib.sha256(
            selected["method"].encode()).hexdigest()


def test_curated_manifest_covers_files_and_local_references(monkeypatch):
    root = Path(skill_library.__file__).resolve().parents[1] / "skills"
    monkeypatch.setattr(skill_library, "_search_dirs", lambda: [root])
    methods, problems = discover()
    assert not problems
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    registered = {row["file"] for row in manifest["skills"]}
    files = {path.relative_to(root).as_posix() for path in root.glob("*/SKILL.md")}
    assert registered == files
    assert len({row["id"] for row in methods}) == len(files)
    # Reference documents are reachable but must not become independent catalog entries.
    for path in root.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#", 1)[0]).is_file(), (path, target)
    assert not any("references/" in row["source"] for row in methods)


def test_catalog_does_not_read_method_bodies_or_literature(monkeypatch):
    original_read = Path.read_text

    def guarded_read(path, *args, **kwargs):
        assert path.name != "SKILL.md"
        assert path.name not in ("literature-review.md", "library-audit.md")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    catalog = skills_catalog(benchmark="UnseenArmRepository", query="采集 动作 预训练")
    assert catalog["entries"]


def test_main_agent_catalog_is_body_free_and_read_is_explicit():
    catalog = skills_catalog(query="runtime environment build", max_recommended=2)
    assert len(catalog["entries"]) >= 20
    assert sum(bool(row["recommended_by_keyword"]) for row in catalog["entries"]) <= 2
    assert all("method" not in row and "source" not in row for row in catalog["entries"])

    chosen = "autosimsota.building-an-environment-that-runs"
    result = read_selected_skills([chosen])
    assert [row["id"] for row in result["skills"]] == [chosen]
    assert result["selection"][0]["body_sha256"] == result["skills"][0]["body_sha256"]
    assert result["index"] == []
    # Ranking is a hint, not an allowlist: the Agent may choose another catalog entry.
    unranked = next(row["id"] for row in catalog["entries"]
                    if not row["recommended_by_keyword"] and row["id"] != chosen)
    assert read_selected_skills([unranked])["selection"][0]["id"] == unranked


def test_explicit_skill_read_cannot_bypass_catalog_scope_or_count():
    with pytest.raises(ValueError, match="applicable catalog"):
        read_selected_skills(["autosimsota.robosynchallenge-measurements"],
                             benchmark="LIBERO")
    with pytest.raises(ValueError, match="distinct"):
        read_selected_skills(["autosimsota.using-the-method-library"] * 2)
    with pytest.raises(ValueError, match="applicable catalog"):
        read_selected_skills(["unknown.skill"])


def test_index_is_compact_and_unqueried_calls_only_load_the_guide():
    reference = skills_reference()
    assert reference["status"] == "reference_only_not_enforced"
    assert reference["skills"][0]["name"] == GUIDE
    assert len(reference["index"]) >= 20
    assert len(reference["skills"]) == 1
    assert all("method" not in row and "body_sha256" not in row
               for row in reference["index"])


def test_query_loads_a_bounded_relevant_set_and_records_why():
    reference = skills_reference(
        query="find a native train evaluate collect entrypoint command from repository source",
        max_selected=2)
    names = [skill["name"] for skill in reference["skills"]]
    assert names[0] == GUIDE
    assert len(names) <= 3  # the guide plus no more than two task-relevant bodies
    assert "finding-how-a-benchmark-runs" in names
    selected = {row["id"]: row for row in reference["selection"]}
    assert set(selected) == {row["id"] for row in reference["skills"]}
    for skill in reference["skills"]:
        assert skill["body_sha256"] == hashlib.sha256(skill["method"].encode()).hexdigest()
        assert selected[skill["id"]]["version"] == skill["version"]
        assert selected[skill["id"]]["body_sha256"] == skill["body_sha256"]
        assert selected[skill["id"]]["applicability"] == \
            "candidate_requires_current_run_verification"
    assert selected["autosimsota.finding-how-a-benchmark-runs"]["score"] > 0
    assert "matched metadata" in selected["autosimsota.finding-how-a-benchmark-runs"]["reason"]


def test_guide_explains_scope_confidence_evidence_and_not_policy():
    library = {skill["name"]: skill for skill in discover()[0]}
    guide = library[GUIDE]["method"]
    assert guide.count("```") == 0, "an index is prose, not code"
    general = {name: skill for name, skill in library.items()
               if name != GUIDE and not skill["scope"].startswith("benchmark:")}
    missing = [name for name in general if name not in guide]
    assert not missing, f"the guide does not mention {missing}"
    for field in ("description", "confidence", "evidence", "method"):
        assert field in guide, field
    assert "contract" in guide.lower()


def test_scope_limits_benchmark_specific_skill_bodies_not_the_metadata_index():
    all_methods = skills_reference(query="click_bell sampling mass measured values")
    assert "robosynchallenge-measurements" not in {
        row["name"] for row in all_methods["skills"]}
    assert any(row["name"] == "robosynchallenge-measurements"
               for row in discover()[0])

    scoped = skills_reference(benchmark="RoboSynChallenge",
                              query="click_bell sampling mass measured values")
    assert "robosynchallenge-measurements" in {
        row["name"] for row in scoped["skills"]}
    assert all(not row["scope"].startswith("benchmark:") or
               row["scope"] == "benchmark:RoboSynChallenge"
               for row in scoped["index"])


def test_unmatched_query_does_not_invent_relevance_or_hide_metadata():
    reference = skills_reference(query="zzzxxy qqqqv 6a92f")
    assert [row["name"] for row in reference["skills"]] == [GUIDE]
    assert len(reference["index"]) >= 20


def test_malformed_external_skill_fails_closed_but_does_not_break_discovery(tmp_path, monkeypatch):
    (tmp_path / "custom.md").write_text(
        "---\nname: incomplete\ndescription: missing the retrieval schema\n---\nBody\n",
        encoding="utf-8")
    monkeypatch.setenv("AUTOSIM_SKILLS_DIR", str(tmp_path))
    reference = skills_reference(query="custom external method")
    assert any("missing metadata fields" in row["error"]
               for row in reference["library"]["unreadable"])
    assert "incomplete" not in {row["name"] for row in reference["index"]}


def test_misspelled_benchmark_scope_is_diagnostic_not_global_advice(tmp_path, monkeypatch):
    (tmp_path / "scoped.md").write_text(
        "---\n"
        "name: scoped-runtime-method\n"
        "description: A benchmark-specific runtime method.\n"
        "id: external.scoped-runtime-method\n"
        "version: '1.0'\n"
        "scope: benchamrk:RoboTwin\n"
        "capabilities: [runtime.validation]\n"
        "keywords: [runtime, dependency, repair]\n"
        "inputs: [A dependency failure.]\n"
        "outputs: [A bounded recommendation.]\n"
        "preconditions: [A runtime failure is evidenced.]\n"
        "verification: [Repeat the bounded import check.]\n"
        "permissions: [Advisory only.]\n"
        "resources: [No direct compute.]\n"
        "invalid_when: [The issue is unrelated to runtime.]\n"
        "---\n\nRepository-specific method body.\n",
        encoding="utf-8")
    monkeypatch.setenv("AUTOSIM_SKILLS_DIR", str(tmp_path))

    reference = skills_reference(benchmark="ManiSkill", task="PushCube-v1",
                                 query="runtime dependency repair")

    assert "scoped-runtime-method" not in {row["name"] for row in reference["index"]}
    assert "scoped-runtime-method" not in {row["name"] for row in reference["skills"]}
    assert any(row["path"].endswith("scoped.md") and "scope must be" in row["error"]
               for row in reference["library"]["unreadable"])


def test_external_skills_need_unique_ids_and_their_selected_body_is_versioned(tmp_path,
                                                                              monkeypatch):
    (tmp_path / "custom.md").write_text(
        "---\n"
        "name: custom-runtime-repair\n"
        "description: A custom runtime repair technique.\n"
        "id: autosimsota.custom-runtime-repair\n"
        "version: '2.1.0'\n"
        "scope: general\n"
        "capabilities: [runtime.validation]\n"
        "keywords: [runtime, dependency, repair]\n"
        "inputs: [A dependency failure and current environment.]\n"
        "outputs: [A bounded repair recommendation.]\n"
        "preconditions: [A concrete runtime failure exists.]\n"
        "verification: [Rerun the same bounded import check.]\n"
        "permissions: [Advisory only; no permissions.]\n"
        "resources: [No direct compute.]\n"
        "invalid_when: [Failure cause is unrelated to runtime.]\n"
        "---\n\nCheck the package metadata first.\n",
        encoding="utf-8")
    monkeypatch.setenv("AUTOSIM_SKILLS_DIR", str(tmp_path))
    reference = skills_reference(query="runtime dependency repair custom-runtime technique",
                                 max_selected=3)
    selected = {row["id"]: row for row in reference["skills"]}
    assert selected["autosimsota.custom-runtime-repair"]["version"] == "2.1.0"
    assert selected["autosimsota.custom-runtime-repair"]["body_sha256"]

    # Reusing a curated identity cannot silently replace that method.
    path = tmp_path / "duplicate.md"
    path.write_text((tmp_path / "custom.md").read_text(encoding="utf-8").replace(
        "autosimsota.custom-runtime-repair", "autosimsota.where-successes-come-from")
        .replace("custom-runtime-repair", "duplicate-runtime-repair"), encoding="utf-8")
    duplicate = skills_reference(query="runtime dependency repair")
    assert any("duplicate skill id" in row["error"]
               for row in duplicate["library"]["unreadable"])
