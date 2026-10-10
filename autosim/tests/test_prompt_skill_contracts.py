"""Prompt composition, packaging and metadata boundaries, without model calls."""

import ast
from pathlib import Path
import re

import pytest
import yaml

from autosim.research import prompt_library, skills
from autosim.research.agent_roles import ROLE_PROFILES, role_profile
from autosim.research.agent_runtime import _role_cli_args
from autosim.research import runtime_freshness


def test_every_registered_template_is_present_and_no_unregistered_templates():
    assert {path.stem for path in prompt_library.ROOT.glob("*.md")} == (
        prompt_library.PROMPT_NAMES | prompt_library.LEGACY_NAMES)
    for name in prompt_library.PROMPT_NAMES:
        value = prompt_library.load_prompt(name)
        assert value.strip()
        if name in prompt_library.ROLE_PROMPTS:
            assert value.count(prompt_library.load_prompt("core")) == 1
        else:
            assert value == prompt_library._template(name)


@pytest.mark.parametrize("name", ["../skills/manifest", "/etc/passwd", "unknown", "scheduler.md",
                                 *sorted(prompt_library.LEGACY_NAMES)])
def test_model_cannot_choose_prompt_paths(name):
    with pytest.raises(ValueError, match="unknown prompt"):
        prompt_library.load_prompt(name)


def test_scheduler_formatting_preserves_runtime_operations_and_literal_json():
    from autosim.research.prepare import STEP_SYSTEM
    result = STEP_SYSTEM.format(operations="CURRENT_OPERATIONS_SENTINEL")
    assert result.count("CURRENT_OPERATIONS_SENTINEL") == 1
    assert '{"state_revision"' in result


def test_scheduler_projects_unavailable_procedures_not_safety_or_response_rules():
    from autosim.research.prepare import OPERATIONS
    names={'read_the_checkout','stop'}
    compact=prompt_library.scheduler_request_prompt(operations='OPERATIONS_SENTINEL',permitted=names,
                                                   operation_names=OPERATIONS)
    full=prompt_library.load_prompt('scheduler').format(operations='OPERATIONS_SENTINEL')
    assert len(compact)<len(full)
    assert compact.count('OPERATIONS_SENTINEL')==1
    assert '{"state_revision"' in compact and 'Method selection and review' in compact
    assert 'not official baseline reproduction' in compact and 'held-out' in compact
    complete=prompt_library.scheduler_request_prompt(operations='OPERATIONS_SENTINEL',permitted=OPERATIONS,
                                                    operation_names=OPERATIONS)
    assert complete==full


def test_data_agent_prompt_routing_capabilities_and_scheduler_followthrough():
    from autosim.research.main_agent import task_system_for_role, TASK_SYSTEM, READ_ONLY_ROLES
    assert "data" in READ_ONLY_ROLES
    system = task_system_for_role("data")
    assert system.count(prompt_library.load_prompt("core")) == 1
    assert "Dedicated Data Agent assignment" in system
    assert "recommended_next_actions" in system
    assert "preserve raw/source datasets" in system.lower()
    assert "held-out evaluation" in system
    assert "conversion" in system and "actual trainer" in system
    assert "no arbitrary shell" in system
    assert task_system_for_role("resource").startswith(TASK_SYSTEM)
    assert "local budget ownership" in task_system_for_role("resource")
    with pytest.raises(ValueError):
        task_system_for_role("../../unknown")
    profile = role_profile("data")
    assert profile.title == "Data Agent" and profile.can_run_agent
    assert profile.builtin_tools == ("Read", "Glob", "Grep")
    assert not profile.can_edit_checkout and not profile.can_execute_diagnostics
    assert "mcp__autosim_exec__inspect_native_environment" in profile.mcp_tools
    assert "mcp__autosim_exec__run_command" not in profile.mcp_tools
    scheduler = prompt_library.load_prompt("scheduler").format(operations="fixture")
    assert "main_agent.data_agent_handoff" in scheduler
    assert "Do not invoke it every heartbeat" in scheduler
    assert "old missing-interpreter/metric rationale" in scheduler


def test_native_observability_guidance_preserves_live_jobs_and_work_semantics():
    scheduler = prompt_library.load_prompt("scheduler").format(operations="fixture")
    monitor = prompt_library.load_prompt("monitor")
    assert "Quiet stdout alone is not evidence of a stuck trainer" in scheduler
    assert "native CLI's logging/progress initialization" in scheduler
    assert "effective native config" in scheduler
    assert "hidden work floors" in scheduler
    assert "interrupt a healthy" in scheduler
    assert "formal baseline and frozen evaluation protocol" in scheduler
    assert "printed configuration are not completed-work evidence" in monitor
    assert "uncertain, not automatically stalled" in monitor
    assert "Do not recommend interrupting a healthy active job" in monitor


@pytest.mark.parametrize("kind", ["empty", "oversize", "symlink"])
def test_prompt_files_reject_unsafe_or_unusable_content(tmp_path, monkeypatch, kind):
    root = tmp_path / "prompts"
    root.mkdir()
    target = root / "core.md"
    if kind == "symlink":
        outside = tmp_path / "outside.md"
        outside.write_text("untrusted replacement")
        target.symlink_to(outside)
    else:
        target.write_text("  " if kind == "empty" else "x" * 128_001)
    monkeypatch.setattr(prompt_library, "ROOT", root)
    prompt_library._template.cache_clear()
    try:
        with pytest.raises(ValueError):
            prompt_library.load_prompt("core")
    finally:
        prompt_library._template.cache_clear()


def test_prompt_cache_is_stable_until_controlled_refresh(tmp_path, monkeypatch):
    root = tmp_path / "prompts"
    root.mkdir()
    target = root / "core.md"
    target.write_text("original shared contract")
    monkeypatch.setattr(prompt_library, "ROOT", root)
    prompt_library._template.cache_clear()
    try:
        initial = prompt_library.load_prompt("core")
        target.write_text("edited shared contract")
        assert prompt_library.load_prompt("core") == initial
        prompt_library._template.cache_clear()
        assert prompt_library.load_prompt("core") != initial
    finally:
        prompt_library._template.cache_clear()


def test_static_global_system_literals_are_not_maintained_in_python():
    root = Path(prompt_library.__file__).parent
    # All top-level static global SYSTEM constants belong to the library.
    # Per-call prompts with dynamic contracts remain local to their operation.
    for path in root.glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str) and node.targets
                    and isinstance(node.targets[0], ast.Name)
                    and "SYSTEM" in node.targets[0].id):
                pytest.fail(f"inline global system template: {path.name}:{node.lineno}")


def test_prompt_changes_version_the_controller_without_claiming_repair(monkeypatch):
    runtime_freshness.controller_identity.cache_clear()
    initial = runtime_freshness.controller_identity()
    try:
        monkeypatch.setattr(runtime_freshness, "prompt_identity", lambda: "changed-prompts")
        runtime_freshness.controller_identity.cache_clear()
        assert runtime_freshness.controller_identity() != initial
    finally:
        runtime_freshness.controller_identity.cache_clear()


def test_packaging_declares_all_prompt_and_skill_reference_files():
    root = Path(prompt_library.__file__).resolve().parents[2]
    config = (root / "pyproject.toml").read_text()
    section = config.split("[tool.setuptools.package-data]", 1)[1].split("\n[", 1)[0]
    patterns = ast.literal_eval(re.search(r"autosim\s*=\s*(\[[^\]]*\])", section)[1])
    package = root / "autosim"
    included = {path for pattern in patterns for path in package.glob(pattern)}
    required = set(prompt_library.ROOT.glob("*.md"))
    required |= set((package / "skills").rglob("*.md"))
    required.add(package / "skills" / "manifest.yaml")
    assert required <= included


def test_recorder_has_no_unavailable_tools_or_competing_handoff_schema():
    recorder = role_profile("recorder")
    assert not recorder.builtin_tools and not recorder.mcp_tools
    assert "inspect_native_environment" not in recorder.instruction
    assert "search_public_sources" not in recorder.instruction
    assert "venv" not in recorder.instruction
    assert "concise handoff" not in recorder.instruction
    for name in ROLE_PROFILES:
        profile = role_profile(name, read_only=True)
        assert "Edit" not in profile.builtin_tools and "Write" not in profile.builtin_tools
        assert "mcp__autosim_exec__run_command" not in profile.mcp_tools
        if profile.can_inspect_native:
            assert "mcp__autosim_exec__inspect_native_environment" in profile.mcp_tools


@pytest.mark.parametrize("role", ["scheduler", "init", "fix", "objective", "recorder"])
def test_decision_only_prompt_matches_its_actual_empty_tool_allowlist(tmp_path, role):
    args = _role_cli_args(role_profile(role), workspace=tmp_path, output=tmp_path,
                          python="python", model="fixture", max_budget_usd=1,
                          decision_only=True)
    assert args[args.index("--tools") + 1] == ""
    assert "--allowedTools" not in args
    instruction = args[args.index("--append-system-prompt") + 1]
    assert "no tools" in instruction
    assert "inspect_native_environment" not in instruction
    assert "search_public_sources" not in instruction


def _curated(tmp_path, monkeypatch, *, manifest_scope="general", front_scope=None):
    package = tmp_path / "package"
    root = package / "skills"
    path = root / "example" / "SKILL.md"
    path.parent.mkdir(parents=True)
    front = {"name": "example", "description": "Example method."}
    if front_scope is not None:
        front["scope"] = front_scope
    path.write_text("---\n" + yaml.safe_dump(front) + "---\n\nMethod body.")
    row = {"id": "test.example", "version": "1.0.0", "file": "example/SKILL.md",
           "scope": manifest_scope, "confidence": "source-backed guidance",
           "evidence": "Verified source audit; not a measured improvement."}
    row.update({field: ["bounded guidance"] for field in skills._LIST_FIELDS})
    (root / "manifest.yaml").write_text(yaml.safe_dump({"schema_version": 1, "skills": [row]}))
    monkeypatch.setattr(skills, "__file__", str(package / "research" / "skills.py"))
    return path, row


def test_manifest_evidence_and_scope_survive_minimal_frontmatter(tmp_path, monkeypatch):
    path, expected = _curated(tmp_path, monkeypatch, manifest_scope="benchmark:RoboTwin")
    rows, problems = skills._load_manifest()
    assert not problems
    row = rows["example/SKILL.md"]
    for field in ("scope", "confidence", "evidence"):
        assert row[field] == expected[field]
    assert not skills._applies(row, "LIBERO", None)
    assert skills._parse(path, row)["scope"] == "benchmark:RoboTwin"


@pytest.mark.parametrize("scope", ["benchamrk:RoboTwin", {"benchmark": "RoboTwin"}])
def test_invalid_manifest_scope_is_not_global_advice(tmp_path, monkeypatch, scope):
    _curated(tmp_path, monkeypatch, manifest_scope=scope)
    rows, problems = skills._load_manifest()
    assert not rows
    assert any("scope must be" in row["error"] for row in problems)


def test_conflicting_header_cannot_broaden_manifest_scope(tmp_path, monkeypatch):
    _curated(tmp_path, monkeypatch, manifest_scope="benchmark:RoboTwin", front_scope="general")
    rows, problems = skills._load_manifest()
    assert not rows
    assert any("conflicting scope" in row["error"] for row in problems)


def test_scope_cannot_change_between_catalog_and_body_read(tmp_path, monkeypatch):
    path, _ = _curated(tmp_path, monkeypatch)
    rows, _ = skills._load_manifest()
    path.write_text(path.read_text().replace("name: example", "name: example\nscope: benchmark:RoboTwin"))
    with pytest.raises(ValueError, match="scope changed"):
        skills._parse(path, rows["example/SKILL.md"])


def test_curated_metadata_has_one_canonical_location(monkeypatch):
    root = Path(skills.__file__).resolve().parents[1] / "skills"
    monkeypatch.setattr(skills, "_search_dirs", lambda: [root])
    methods, problems = skills.discover()
    assert not problems
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    rows = {row["id"]: row for row in manifest["skills"]}
    assert len(methods) == len(rows)
    for method in methods:
        front = skills._read_frontmatter(Path(method["source"]))
        assert {"name", "description"} <= front.keys()
        assert not {"scope", "confidence", "evidence"} & front.keys()
        for field in ("scope", "confidence", "evidence"):
            assert method[field] == rows[method["id"]][field]
def test_scheduler_template_formats_its_only_runtime_placeholder():
    from autosim.research.prompt_library import load_prompt
    prompt=load_prompt('scheduler').format(operations='available operations fixture')
    assert 'available operations fixture' in prompt and 'reuse_policy' in prompt
