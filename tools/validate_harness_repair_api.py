#!/usr/bin/env python3
"""Exercise real API coding on the archived production barrier defect.

The original AST predicate is projected onto the production membership helper's
JSON interface.  No new defect is invented.  Expected answers remain in this
host process; candidate code runs in the probed kernel sandbox.  The output is
an activation request, never an implicit edit of the current research process.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "autosim"))

from autosim.research.common import atomic_json, digest, immutable_json, object_digest, read_json
from autosim.research.compute_agent import client_from_file
from autosim.research.incidents import IncidentStore, build_incident
from autosim.research.repair_agent import RepairAgent
from autosim.research.repair_tools import RepairTools, verify_activation_request, membership_validation_spec
from autosim.research.repair_validation import ValidationSpec, isolation_capability


TARGET = "autosim/autosim/research/probe_barrier_contract.py"


def membership_cases() -> tuple[dict, ...]:
    return membership_validation_spec(TARGET).cases


def historical_predicate(path: Path) -> tuple[str, dict]:
    """Extract the exact old expression; the only transformation maps inputs."""
    source = path.read_text()
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        text = ast.get_source_segment(source, node.test) or ""
        if 'barrier.glob("*.json")' in text and "expected" in text:
            matches.append(node)
    if not matches or len({ast.dump(item.test, include_attributes=False) for item in matches}) != 1:
        raise ValueError("historical barrier predicates are absent or semantically different")
    node = matches[0]
    original = ast.get_source_segment(source, node.test)

    class ProjectInputs(ast.NodeTransformer):
        def visit_Call(self, value):
            if (isinstance(value.func, ast.Name) and value.func.id == "list" and len(value.args) == 1 and
                    isinstance(value.args[0], ast.Call) and isinstance(value.args[0].func, ast.Attribute) and
                    isinstance(value.args[0].func.value, ast.Name) and value.args[0].func.value.id == "barrier" and
                    value.args[0].func.attr == "glob"):
                return ast.Name(id="ready", ctx=ast.Load())
            return self.generic_visit(value)

        def visit_Name(self, value):
            if value.id == "expected":
                return ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=[ast.Name(id="expected", ctx=ast.Load())], keywords=[])
            return value

    projected = ProjectInputs().visit(copy.deepcopy(node.test))
    ast.fix_missing_locations(projected)
    expression = ast.unparse(projected)
    candidate_base = ('"""Archived production barrier predicate, projected onto membership inputs.\n'
                      'The host retains the original source/hash and verifies this projection.\n"""\n\n'
                      'def ready_members_match(expected: list[dict], ready: list[dict], generation: str) -> bool:\n'
                      f'    return not ({expression})\n')
    return candidate_base, {"historical_path": str(path.absolute()), "historical_sha256": digest(path),
                            "historical_lines": sorted(item.lineno for item in matches), "original_predicate": original,
                            "projected_predicate": expression,
                            "projection": "list(barrier.glob('*.json')) -> ready; expected count -> len(expected); release is not wait-condition",
                            "claim_scope": "Real archived membership bug; not an explanation or repair of URDF engine SIGSEGV."}


def prepare(root: Path, historical: Path, *, deadline_epoch: float,
            preparation_only: bool = False) -> tuple[dict, RepairTools]:
    root = root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    projected, provenance = historical_predicate(historical)
    frozen = root / "historical_execution"
    target = frozen / TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text() != projected:
        raise ValueError("historical repair source changed")
    target.write_text(projected)
    immutable_json(root / "historical_projection.json", provenance | {"projected_module_sha256": digest(target)})
    contract = membership_validation_spec(TARGET)
    incident = build_incident(run_id="historical_barrier_coding", stage="native_probe_barrier",
        purpose="smoke", worker_roots={}, source_revision=digest(target),
        exception="The archived ready-file count accepts stale or wrong worker identities; verify the real historical predicate.",
        generation="historical-control", budget={"deadline_epoch": deadline_epoch})
    incident["failure_kind"] = "barrier_membership_validation"
    # This contract is part of the legitimate tool task; expected test outputs
    # remain private. The model must implement the behavior, not guess the tests.
    incident["supervisor_error"] = (
        "Archived file-count barrier admitted stale/mismatched workers. Repair the registered production helper. "
        "Contract: expected and ready are nonempty lists of dicts, identical complete memberships with unique worker and uuid. "
        "worker, uuid, generation, claim and attempt_id must be nonempty strings matching the expected row. "
        "Every expected generation equals the generation argument (a nonempty str); ready state is exactly 'ready' "
        "and alive is exactly True. Malformed/missing fields, stale generation, duplicate workers/UUIDs, "
        "missing or extra ready members must return False. Host coordinator independently measures alive/PID identity.")
    # Review preparation must never create an execution incident or tool ledger:
    # its timer may expire while specific outbound authorization is pending.
    session_root = root / "preparation" if preparation_only else root
    incident = IncidentStore(session_root / "incidents").snapshot(incident)
    tools = RepairTools(session_root / "tools", source_root=frozen, allowed_code_paths=[TARGET], incident=incident,
                        validators={contract.name: contract}, deadline_epoch=deadline_epoch)
    return incident, tools


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-evaluation", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--max-api-calls", type=int, default=12)
    parser.add_argument("--max-api-tokens", type=int, default=160000)
    args = parser.parse_args()
    if not 185 <= args.max_seconds <= 1200:
        parser.error("max-seconds must be between 185 and 1200")
    args.root.mkdir(parents=True, exist_ok=True)
    # Preparing reviewable material sends no requests and must not start the
    # execution session's deadline while its specific egress approval is pending.
    budget_path = args.root / "preparation/budget.json" if args.prepare_only else args.root / "budget.json"
    if budget_path.exists():
        deadline = min(read_json(budget_path)["deadline_epoch"], time.time() + args.max_seconds)
    else:
        deadline = time.time() + args.max_seconds
    atomic_json(budget_path, {"deadline_epoch": deadline})
    incident, broker = prepare(args.root, args.historical_evaluation, deadline_epoch=deadline,
                               preparation_only=args.prepare_only)
    capability = isolation_capability()
    atomic_json(args.root / "isolation.json", capability)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "l2_available": capability["l2_available"],
                          "root": str(args.root.absolute()), "incident_id": incident["incident_id"]}))
        return 0 if capability["l2_available"] else 3
    if not capability["l2_available"]:
        print(json.dumps({"status": "unavailable", "reason": "L2 kernel isolation unavailable"}))
        return 3
    if args.env_file is None:
        parser.error("--env-file is required for the real API coding validation")
    result = RepairAgent(args.root / "agent", client_from_file(args.env_file), tools=broker,
                         max_calls=args.max_api_calls, max_tokens=args.max_api_tokens).run(incident)
    if result["status"] == "activation_requested":
        verify_activation_request(result["result"])
        atomic_json(args.root / "activation_request.json", result["result"])
    print(json.dumps({"status": result["status"], "api_used": result.get("api_used", False),
                      "steps": len(result["steps"]), "root": str(args.root.absolute())}))
    return 0 if result["status"] == "activation_requested" else 2


if __name__ == "__main__":
    raise SystemExit(main())
