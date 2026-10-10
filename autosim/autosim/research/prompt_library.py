"""Versioned, package-local role prompts; runtime facts stay in the request.

Names are an allowlist, not model-controlled file paths. Cached templates form a
stable prefix; a controller restart is required after editing prompt files.
"""

from functools import lru_cache
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "prompts"
LEGACY_NAMES = frozenset({"onboard_task_prompt", "ideas_task_prompt", "optimize_task_prompt"})
CORE_ROLES = frozenset({
    "core", "scheduler", "monitor", "fix", "specialist", "ideator",
    "ideator-leap", "idea-repair", "recorder", "skill-selection", "runtime-tools",
})
OPERATION_PROMPTS = frozenset({
    "claims-summary", "contract-codegen", "contract-paths", "parameter-decision",
    "execution-derive", "execution-triage", "execution-argv", "execution-diagnose",
    "execution-revise", "objective-decompose", "objective-answer",
    "environment-plan", "environment-resume", "environment-reconsider",
    "environment-diagnose", "repository-scout", "repository-triage", "strategy",
    "supervisor", "candidate-execution-review", "recorder-fact-review", "data", "resource",
})
PROMPT_NAMES = CORE_ROLES | OPERATION_PROMPTS
ROLE_PROMPTS = CORE_ROLES - {"core", "runtime-tools"}


@lru_cache(maxsize=len(PROMPT_NAMES))
def _template(name: str) -> str:
    if name not in PROMPT_NAMES:
        raise ValueError(f"unknown prompt template: {name}")
    path = ROOT / f"{name}.md"
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"unsafe prompt template: {name}")
    if path.stat().st_size > 128_000:
        raise ValueError(f"prompt template exceeds size limit: {name}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty prompt template: {name}")
    return value


def load_prompt(name: str) -> str:
    """Compose core once, then the role's own task and exact response contract."""
    content = _template(name)
    return (_template("core") + "\n\n" + content
            if name in ROLE_PROMPTS else content)


def scheduler_request_prompt(*, operations: str, permitted, operation_names) -> str:
    """Keep strategic/safety/response rules; omit clauses for unavailable operations.

    No summarizing model or heuristic selects the next action. This only projects
    the existing executor capability boundary; complete templates stay versioned.
    The stable strategic prefix is unchanged for provider caching.
    """
    template=load_prompt('scheduler')
    start='Choose by what is missing and by what failed. A few things are worth knowing:\n'
    end='`method_library` contains candidate methods retrieved from metadata, not executable policy.'
    if start not in template or end not in template:
        return template.format(operations=operations)
    prefix,rest=template.split(start,1)
    procedures,suffix=rest.split(end,1)
    parts=re.split(r'(?m)(?=^\* )',procedures)
    known=set(operation_names)
    selected=[]
    for clause in parts:
        names=set(re.findall(r'`([a-z][a-z0-9_]*)`',clause)) & known
        if not names or names & set(permitted):
            selected.append(clause)
    result=prefix+start+''.join(selected)+end+suffix
    return result.format(operations=operations)


def prompt_identity() -> str:
    """Fingerprint the loaded templates, including core and tool guidance."""
    payload = "\n".join(name + ":" + hashlib.sha256(
        _template(name).encode("utf-8")).hexdigest() for name in sorted(PROMPT_NAMES))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
