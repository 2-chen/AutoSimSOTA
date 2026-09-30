"""Host-owned repair operations, immutable candidates and activation requests.

There is deliberately no shell tool.  Files, inputs and validators are registered
by the supervisor; the API can neither widen that scope nor edit its own judge.
"""
from __future__ import annotations

import ast
import copy
import difflib
import json
import math
import re
import random
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

from .common import atomic_json, digest, exclusive, object_digest, read_json
from .incidents import evidence_view, permitted_relative_path, redact_text
from .repair_validation import (IsolationUnavailable, ValidationSpec, check_pure_component,
                                isolation_capability, run_contract, validate_candidate)


def membership_validation_spec(target: str = "autosim/autosim/research/probe_barrier_contract.py") -> ValidationSpec:
    """Trusted membership tests live outside the candidate's mounted source."""
    cases = []
    rng = random.Random(15915)

    def add(expected, ready, generation, answer):
        cases.append({"args": copy.deepcopy([expected, ready, generation]), "expected": answer})

    fields = ("worker", "uuid", "generation", "claim", "attempt_id")
    for count in (1, 2, 4, 8):
        generation = f"generation-{rng.randrange(10**9)}"
        expected = [{"worker": f"worker-{i}", "uuid": f"GPU-{rng.randrange(10**12)}",
                     "generation": generation, "claim": f"claim-{rng.randrange(10**12)}",
                     "attempt_id": f"attempt-{rng.randrange(10**12)}"} for i in range(count)]
        ready = [row | {"state": "ready", "alive": True} for row in expected]
        add(expected, ready, generation, True)
        shuffled = copy.deepcopy(ready)
        rng.shuffle(shuffled)
        add(expected, shuffled, generation, True)
        add(expected, ready[:-1], generation, False)
        add(expected, ready + [ready[0]], generation, False)
        for field in fields:
            for value in (None, True, 1, "", "different-identity"):
                bad = copy.deepcopy(ready)
                bad[0][field] = value
                add(expected, bad, generation, False)
            bad = copy.deepcopy(ready)
            del bad[0][field]
            add(expected, bad, generation, False)
        for field, values in (("alive", (False, 1, None, "true")), ("state", ("running", "failed", None, True))):
            for value in values:
                bad = copy.deepcopy(ready)
                bad[0][field] = value
                add(expected, bad, generation, False)
        for field in fields:
            for value in (None, True, 1, ""):
                bad = copy.deepcopy(expected)
                bad[0][field] = value
                add(bad, ready, generation, False)
        if count > 1:
            for field in ("worker", "uuid"):
                bad_expected = copy.deepcopy(expected)
                bad_expected[1][field] = bad_expected[0][field]
                add(bad_expected, [row | {"state": "ready", "alive": True} for row in bad_expected], generation, False)
            add(expected, [ready[0]] * count, generation, False)
        for wrong in (None, True, 1, "", "old-generation"):
            add(expected, ready, wrong, False)
        for bad_expected, bad_ready in ((expected, None), (None, ready), ({}, ready), (expected, {}), (expected, [None] * count)):
            add(bad_expected, bad_ready, generation, False)
    add([], [], "empty", False)
    return ValidationSpec("native_membership", target, "ready_members_match", tuple(cases), native_required=True, pure_component=True)


def production_repair_tools(incident: dict, source_root: Path, output: Path, deadline_epoch: float,
                            ledger=None, *, max_candidates: int = 2) -> "RepairTools":
    """Only the registered pure lifecycle helper is currently editable by L2.

The coordinator still owns PID checks, peer cleanup and process launch. Native
engine defects have no matching reproducer here and cannot pass activation.
"""
    root = Path(source_root).resolve()
    choices = ("autosim/autosim/research/probe_barrier_contract.py", "autosim/research/probe_barrier_contract.py",
               "research/probe_barrier_contract.py")
    relative = next((name for name in choices if (root / name).is_file()), None)
    if relative is None:
        raise ValueError("source root has no registered production repair component")
    spec = membership_validation_spec(relative)
    return RepairTools(output, source_root=root, allowed_code_paths=[relative], incident=incident,
                       validators={spec.name: spec}, deadline_epoch=min(deadline_epoch, time.time() + 1200),
                       ledger=ledger, max_candidates=max_candidates)


class RepairTools:
    def __init__(self, root: Path, *, source_root: Path, allowed_code_paths: Sequence[str],
                 incident: dict, validators: Mapping[str, ValidationSpec], deadline_epoch: float,
                 max_candidates: int = 2, max_actions: int = 32, ledger=None):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.source_root = Path(source_root).resolve()
        self.incident = incident
        self.validators = dict(validators)
        self.ledger = ledger
        if not math.isfinite(deadline_epoch):
            raise ValueError("repair deadline must be a finite absolute timestamp")
        if not 1 <= max_candidates <= 4 or not 1 <= max_actions <= 64:
            raise ValueError("repair operation budget outside limits")
        allowed = tuple(sorted(set(permitted_relative_path(name) for name in allowed_code_paths)))
        if not allowed or len(allowed) > 16:
            raise ValueError("repair requires 1..16 explicit source files")
        self.allowed = allowed
        self.base_hashes = {}
        for relative in self.allowed:
            path = self.source_root / relative
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.source_root):
                raise PermissionError("repair source is not an ordinary registered file")
            if path.suffix != ".py" or path.stat().st_size > 131072:
                raise PermissionError("repair source must be a bounded Python component")
            self.base_hashes[relative] = digest(path)
        if not self.validators or any(spec.target not in allowed for spec in self.validators.values()):
            raise ValueError("all repair validators must target registered source components")
        identity = {"incident_id": incident["incident_id"], "source_root": str(self.source_root),
                    "base_hashes": self.base_hashes,
                    "validators": {key: spec.identity() for key, spec in self.validators.items()}}
        with exclusive(self.root / "state.lock"):
            state_path = self.root / "session.json"
            if state_path.exists():
                state = read_json(state_path)
                if state["identity"] != identity:
                    raise ValueError("repair session identity or trusted validators changed")
                state["deadline_epoch"] = min(state["deadline_epoch"], deadline_epoch)
                state["max_candidates"] = min(state["max_candidates"], max_candidates)
                state["max_actions"] = min(state["max_actions"], max_actions)
            else:
                state = {"identity": identity, "deadline_epoch": deadline_epoch,
                         "max_candidates": max_candidates, "max_actions": max_actions,
                         "actions": [], "candidates": []}
            atomic_json(state_path, state)
            reference = self.root / "reference"
            for relative in allowed:
                destination = reference / relative
                if destination.exists() and digest(destination) != self.base_hashes[relative]:
                    raise ValueError("immutable repair reference changed")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(self.source_root / relative, destination)
            self.reference_root = reference

    @property
    def deadline_epoch(self) -> float:
        return float(read_json(self.root / "session.json")["deadline_epoch"])

    def description(self) -> dict:
        return {"tools": {
            "read_evidence": {"arguments": {}, "description": "Read purpose-scoped measured incident evidence."},
            "read_code": {"arguments": {"path": "registered source path", "start_line": "optional integer", "line_count": "optional integer <= 240"}},
            "search_code": {"arguments": {"query": "literal text, <= 200 characters"}},
            "run_reproducer": {"arguments": {"validator": "registered validator name"}},
            "apply_patch_candidate": {"arguments": {"hypothesis": "one measurable hypothesis", "edits": [{"path": "registered source path", "old": "unique exact source fragment", "new": "replacement"}]}},
            "run_validation": {"arguments": {"candidate_id": "candidate identity"}},
            "request_activation": {"arguments": {"candidate_id": "validated candidate identity"}},
            "request_rollback": {"arguments": {"execution_revision": "previous revision", "reason": "evidence-based reason"}},
        }, "allowed_code_paths": list(self.allowed),
            "validators": {key: {"function": value.function, "target": value.target, "cases": len(value.cases),
                                   "native_required": value.native_required}
                           for key, value in self.validators.items()},
            "deadline_epoch": self.deadline_epoch,
            "scientific_boundary": "No judge, score, sealed bank, weights, physics or budget edits."}

    def _ensure_source(self) -> None:
        for relative, expected in self.base_hashes.items():
            path = self.source_root / relative
            if path.is_symlink() or not path.is_file() or digest(path) != expected:
                raise RuntimeError("registered source changed during repair")
            if digest(self.reference_root / relative) != expected:
                raise RuntimeError("immutable repair reference changed")

    def _candidate(self, candidate_id: str) -> tuple[Path, dict]:
        if not isinstance(candidate_id, str) or not re.fullmatch(r"[a-f0-9]{64}", candidate_id):
            raise ValueError("invalid repair candidate identity")
        directory = self.root / "candidates" / candidate_id
        manifest = read_json(directory / "manifest.json")
        if manifest["candidate_id"] != candidate_id or manifest["base_hashes"] != self.base_hashes:
            raise ValueError("candidate is based on a different source revision")
        for relative, expected in manifest["candidate_hashes"].items():
            path = directory / "source" / relative
            if path.is_symlink() or digest(path) != expected:
                raise ValueError("candidate changed after registration")
            for spec in self.validators.values():
                if spec.target == relative and spec.pure_component:
                    check_pure_component(path.read_text(), spec.function)
        return directory, manifest

    def execute(self, name: str, arguments: dict) -> dict:
        if name not in self.description()["tools"] or not isinstance(arguments, dict):
            raise ValueError("unregistered repair operation")
        action_id = object_digest({"incident": self.incident["incident_id"], "base": self.base_hashes,
                                   "tool": name, "arguments": arguments})
        path = self.root / "actions" / f"{action_id}.json"
        with exclusive(self.root / "state.lock"):
            if path.exists():
                record = read_json(path)
                if record["status"] in {"verified", "failed"}:
                    self._settle(record)
                    return record["result"]
                raise RuntimeError("unresolved repair action; reconcile before another execution")
            self._ensure_source()
            state = read_json(self.root / "session.json")
            remaining = state["deadline_epoch"] - time.time()
            if remaining <= 0 or len(state["actions"]) >= state["max_actions"]:
                raise RuntimeError("repair action budget exhausted")
            estimate = (2 * sum(spec.timeout_seconds + 3 for spec in self.validators.values()) + 8
                        if name in {"run_reproducer", "run_validation"} else 2)
            if remaining < estimate:
                raise RuntimeError("insufficient wall budget for repair validation")
            state["actions"].append(action_id)
            atomic_json(self.root / "session.json", state)
            record = {"action_id": action_id, "action": name, "status": "intent",
                      "arguments_digest": object_digest(arguments), "started_epoch": time.time()}
            atomic_json(path, record)
            reservation = f"repair:{action_id}"
            if self.ledger is not None:
                self.ledger.reserve(reservation, devices=0, estimate_seconds=estimate, category="retry")
            record["status"] = "executing"
            atomic_json(path, record)
            try:
                result = self._dispatch(name, arguments, state)
                result = {"action_id": action_id, "action": name, **result}
                record.update(status="verified", result=result)
            except Exception as exc:
                result = {"action_id": action_id, "action": name, "status": "failed",
                          "reason": redact_text(str(exc), limit=500), "error_type": type(exc).__name__}
                record.update(status="failed", result=result)
            finally:
                record["elapsed_seconds"] = time.time() - record["started_epoch"]
                # Publication and settlement share an identity and can be
                # reconciled without rerunning the candidate or double charging.
                atomic_json(path, record)
                self._settle(record)
            return result

    def _settle(self, record: dict) -> None:
        if self.ledger is not None:
            from .accounting import Charge
            reservation = f"repair:{record['action_id']}"
            self.ledger.settle(Charge(job=reservation, category="retry", status=record["status"],
                wall_seconds=record["elapsed_seconds"], detail={"action_id": record["action_id"]}),
                reservation_key=reservation, settlement_id=reservation)

    @staticmethod
    def _fields(arguments: dict, required: set[str], optional: set[str] | None = None) -> None:
        if not required <= set(arguments) or set(arguments) - required - (optional or set()):
            raise ValueError("repair tool arguments do not match its schema")

    def _dispatch(self, name: str, arguments: dict, state: dict) -> dict:
        if name == "read_evidence":
            self._fields(arguments, set())
            return {"status": "completed", "evidence": evidence_view(self.incident)}
        if name == "read_code":
            self._fields(arguments, {"path"}, {"start_line", "line_count"})
            path = permitted_relative_path(arguments["path"])
            if path not in self.allowed:
                raise PermissionError("source file is not in this incident's repair scope")
            start, count = arguments.get("start_line", 1), arguments.get("line_count", 240)
            if type(start) is not int or type(count) is not int or start < 1 or not 1 <= count <= 240:
                raise ValueError("invalid source range")
            lines = (self.reference_root / path).read_text().splitlines()
            return {"status": "completed", "path": path, "sha256": self.base_hashes[path],
                    "start_line": start, "source": redact_text("\n".join(lines[start - 1:start - 1 + count]), limit=24000)}
        if name == "search_code":
            self._fields(arguments, {"query"})
            query = arguments["query"]
            if not isinstance(query, str) or not 1 <= len(query) <= 200:
                raise ValueError("invalid literal search query")
            matches = [{"path": name, "line": i, "text": redact_text(line, limit=400)}
                       for name in self.allowed
                       for i, line in enumerate((self.reference_root / name).read_text().splitlines(), 1)
                       if query in line]
            return {"status": "completed", "matches": matches[:40]}
        if name == "run_reproducer":
            self._fields(arguments, {"validator"})
            spec = self.validators[arguments["validator"]]
            capability = isolation_capability()
            if not capability["l2_available"]:
                raise IsolationUnavailable("L2 unavailable: kernel isolation probe failed")
            receipt = run_contract(self.reference_root, spec)
            # Do not hand expected answers to candidate code or the API.
            return {"status": "completed", "baseline_reproduced": not receipt["passed"],
                    "receipt": receipt}
        if name == "apply_patch_candidate":
            self._fields(arguments, {"hypothesis", "edits"})
            hypothesis, edits = arguments["hypothesis"], arguments["edits"]
            if not isinstance(hypothesis, str) or not 1 <= len(hypothesis) <= 2000:
                raise ValueError("a bounded repair hypothesis is required")
            if not isinstance(edits, list) or not 1 <= len(edits) <= 16:
                raise ValueError("repair requires 1..16 explicit edits")
            if len(state["candidates"]) >= state["max_candidates"]:
                raise RuntimeError("repair candidate budget exhausted")
            if not isolation_capability()["l2_available"]:
                raise IsolationUnavailable("L2 unavailable: refusing an executable candidate")
            updated = {name: (self.reference_root / name).read_text() for name in self.allowed}
            for edit in edits:
                self._fields(edit, {"path", "old", "new"})
                relative = permitted_relative_path(edit["path"])
                if relative not in updated:
                    raise PermissionError("candidate attempted to modify an unregistered file")
                old, new = edit["old"], edit["new"]
                if not isinstance(old, str) or not isinstance(new, str) or not old or len(new) > 65536:
                    raise ValueError("invalid bounded patch fragment")
                if updated[relative].count(old) != 1:
                    raise ValueError("patch source fragment is not unique in the immutable base")
                updated[relative] = updated[relative].replace(old, new, 1)
            for content in updated.values():
                if len(content.encode()) > 131072:
                    raise ValueError("candidate source exceeds size limit")
                ast.parse(content)
            for spec in self.validators.values():
                if spec.pure_component:
                    check_pure_component(updated[spec.target], spec.function)
            candidate_id = object_digest({"base": self.base_hashes, "sources": updated})
            if all(updated[name] == (self.reference_root / name).read_text() for name in updated):
                raise ValueError("patch makes no source change")
            destination = self.root / "candidates" / candidate_id
            diffs = []
            hashes = {}
            for relative, content in updated.items():
                target = destination / "source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                hashes[relative] = digest(target)
                diffs.extend(difflib.unified_diff((self.reference_root / relative).read_text().splitlines(True),
                    content.splitlines(True), fromfile=f"a/{relative}", tofile=f"b/{relative}"))
            (destination / "patch.diff").write_text("".join(diffs))
            manifest = {"schema_version": 1, "candidate_id": candidate_id, "incident_id": self.incident["incident_id"],
                        "base_hashes": self.base_hashes, "candidate_hashes": hashes,
                        "hypothesis": redact_text(hypothesis), "patch_sha256": digest(destination / "patch.diff")}
            atomic_json(destination / "manifest.json", manifest)
            state["candidates"].append(candidate_id)
            atomic_json(self.root / "session.json", state)
            return {"status": "candidate_created", "candidate_id": candidate_id, "manifest": manifest}
        if name == "run_validation":
            self._fields(arguments, {"candidate_id"})
            destination, manifest = self._candidate(arguments["candidate_id"])
            verdict = validate_candidate(self.reference_root, destination / "source", tuple(self.validators.values()))
            self._candidate(arguments["candidate_id"])
            atomic_json(destination / "validation.json", verdict)
            return {"status": "validated" if verdict["passed"] else "rejected", "candidate_id": manifest["candidate_id"],
                    "validation": verdict, "validation_sha256": digest(destination / "validation.json")}
        if name == "request_activation":
            self._fields(arguments, {"candidate_id"})
            destination, manifest = self._candidate(arguments["candidate_id"])
            verdict = read_json(destination / "validation.json")
            if not verdict.get("passed") or not verdict.get("baseline_reproduced"):
                raise ValueError("activation requires trusted baseline-failure/candidate-pass evidence")
            self._ensure_source()
            record = {"schema_version": 1, "status": "activation_requested", "candidate_id": manifest["candidate_id"],
                      "incident_id": self.incident["incident_id"], "candidate_root": str(destination / "source"),
                      "manifest_path": str(destination / "manifest.json"),
                      "manifest_sha256": digest(destination / "manifest.json"),
                      "validation_path": str(destination / "validation.json"),
                      "validation_sha256": digest(destination / "validation.json"),
                      "base_hashes": self.base_hashes, "candidate_hashes": manifest["candidate_hashes"],
                      "requires_native_acceptance": verdict["requires_native_acceptance"],
                      "pure_components": {spec.target: spec.function for spec in self.validators.values() if spec.pure_component},
                      "activation_preconditions": ["old_workers_reaped", "stage_boundary", "same_base_revision",
                                                   "new_supervisor_process", "native_readmission"],
                      "deadline_epoch": self.deadline_epoch}
            atomic_json(destination / "activation_request.json", record)
            return record
        if name == "request_rollback":
            self._fields(arguments, {"execution_revision", "reason"})
            revision, reason = arguments["execution_revision"], arguments["reason"]
            if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{64}", revision):
                raise ValueError("rollback revision must be an existing source digest")
            if not isinstance(reason, str) or not 1 <= len(reason) <= 2000:
                raise ValueError("rollback requires a bounded evidence reason")
            return {"status": "rollback_requested", "execution_revision": revision,
                    "reason": redact_text(reason), "activation_preconditions": ["host_registered_revision", "old_workers_reaped", "new_supervisor_process"]}
        raise ValueError("unknown repair operation")


def verify_activation_request(record: dict) -> dict:
    """Recheck filesystem identities in the outer host before a new process."""
    if record.get("status") != "activation_requested" or record.get("deadline_epoch", 0) <= time.time():
        raise ValueError("activation request is not valid within its original budget")
    manifest_path, validation_path = Path(record["manifest_path"]), Path(record["validation_path"])
    if digest(manifest_path) != record["manifest_sha256"] or digest(validation_path) != record["validation_sha256"]:
        raise ValueError("repair receipts changed after validation")
    manifest, verdict = read_json(manifest_path), read_json(validation_path)
    if not verdict.get("passed") or manifest["candidate_id"] != record["candidate_id"]:
        raise ValueError("invalid repair validation receipt")
    root = Path(record["candidate_root"]).resolve()
    for relative, expected in record["candidate_hashes"].items():
        permitted_relative_path(relative)
        target = root / relative
        if target.is_symlink() or not target.resolve().is_relative_to(root) or digest(target) != expected:
            raise ValueError("candidate no longer matches activation request")
        if relative in record.get("pure_components", {}):
            check_pure_component(target.read_text(), record["pure_components"][relative])
    return record
