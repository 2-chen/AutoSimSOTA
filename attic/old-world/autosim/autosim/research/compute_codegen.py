"""API-generated numeric audit patches, checked against the unchanged reference."""
from __future__ import annotations
import ast
import difflib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

from .common import atomic_json, digest, now, object_digest, read_json
from .patch_validation import checked_function


def extract_function(path: Path, name: str) -> str:
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"


def measure(source: str, cases: list) -> dict:
    import time
    runner = Path(__file__).with_name("patch_validation.py")
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-I", str(runner)],
                            input=json.dumps({"source": source, "cases": cases}), text=True,
                            capture_output=True, timeout=20, env={"PATH": os.defpath})
    if result.returncode:
        raise ValueError(f"bounded audit kernel rejected (exit {result.returncode})")
    measured = json.loads(result.stdout)
    measured["subprocess_seconds"] = time.perf_counter()-started
    return measured


def validate(source: str, reference: str) -> dict:
    checked_function(source)
    rng = random.Random(70114)
    cases = [{"lengths": values, "chunk_size": chunk}
             for values in ([], [0], [1], [49, 50, 51], [2000, 4, 60])
             for chunk in (1, 2, 50, 128)]
    cases += [{"lengths": [rng.randrange(0, 1000) for _ in range(rng.randrange(1, 40))],
               "chunk_size": rng.randrange(1, 500)} for _ in range(100)]
    expected, actual = measure(reference, cases), measure(source, cases)
    if expected["outputs"] != actual["outputs"]:
        differences = [{"input": case, "expected": ref, "actual": candidate}
                       for case, ref, candidate in zip(cases, expected["outputs"], actual["outputs"])
                       if ref != candidate]
        return {"passed": False, "reason": "output_parity", "cases": len(cases),
                "counterexamples": differences[:4]}
    workload = [{"lengths": [300 + i % 200 for i in range(5000)], "chunk_size": 50}]
    reference_times, candidate_times = [], []
    for iteration in range(3):
        order = [(reference, reference_times, False), (source, candidate_times, True)]
        if iteration % 2: order.reverse()
        for code, times, isolated in order:
            measured = measure(code, workload)
            times.append(measured["subprocess_seconds"] if isolated else measured["elapsed_seconds"])
    ratio = statistics.median(candidate_times) / max(statistics.median(reference_times), 1e-12)
    return {"passed": ratio <= .9, "reason": "measured_gain" if ratio <= .9 else "no_net_kernel_gain",
            "parity_cases": len(cases), "reference_seconds": reference_times,
            "candidate_seconds": candidate_times, "time_ratio": ratio,
            "candidate_execution": "bounded subprocess, including launch and JSON overhead",
            "scope": "audit kernel only; whole-run payback requires remaining-work estimate"}


def optimize(agent, source_path: Path, root: Path, *, max_revisions=2) -> dict:
    root = Path(root)
    reference = extract_function(source_path, "_padding_audit")
    context = {"function": "robosyn_data._padding_audit", "source_sha256": digest(source_path), "source": reference,
               "contract": "Nonnegative integer episode lengths; positive integer chunk_size; identical output dict."}
    for revision in range(max_revisions + 1):
        response = agent.request("compute_codegen", context, output_tokens=4000, thinking="disabled", system=(
            "Optimize the supplied existing audit function without changing any returned value. "
            "Use algebra to eliminate per-frame work. Return JSON with source containing exactly one "
            "Python function named _padding_audit(lengths, chunk_size=50), and explanation. "
            "No imports, attributes, decorators, file operations, nested functions or recursion. "
            "Only numeric builtins min/max/sum/int/float/len/range/enumerate and local containers. "
            "Code must handle empty inputs and zero episode lengths. Each frame is an anchor; "
            "these are sliding windows, not disjoint chunks. For each length L, sum min(C,k) "
            "over k=1..L using a triangular sum and a constant tail. Count k<C separately. "
            "Keep integer arithmetic until the final division. Explain the derivation briefly."))
        try:
            payload = json.loads(response["content"], strict=False)
        except (ValueError, TypeError):
            context = {**context, "previous_verdict": {"passed": False, "reason": "invalid_json"}}
            receipt = {"status": "rejected", "request_id": response["request_id"],
                       "verdict": context["previous_verdict"], "time": now()}
            atomic_json(root / "invalid_responses" / (response["request_id"] + ".json"), receipt)
            continue
        source = payload["source"]
        patch_id = object_digest({"base": context["source_sha256"], "source": source})
        destination = root / patch_id
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "candidate.py").write_text(source)
        (destination / "patch.diff").write_text("".join(difflib.unified_diff(
            reference.splitlines(True), source.splitlines(True), fromfile="reference", tofile="candidate")))
        try:
            verdict = validate(source, reference)
        except (ValueError, SyntaxError, subprocess.TimeoutExpired, KeyError) as exc:
            verdict = {"passed": False, "reason": type(exc).__name__}
        receipt = {"patch_id": patch_id, "base_sha256": context["source_sha256"],
                   "candidate_sha256": digest(destination / "candidate.py"),
                   "source_path": str(source_path), "request_id": response["request_id"],
                   "verdict": verdict, "time": now(), "status": "validated" if verdict["passed"] else "rejected"}
        atomic_json(destination / "receipt.json", receipt)
        if verdict["passed"]:
            return receipt
        context = {**context, "previous_candidate": source, "previous_verdict": verdict}
    return receipt


def activate(root: Path, receipt: dict) -> Path:
    if receipt["status"] != "validated" or not receipt["verdict"]["passed"]:
        raise ValueError("unvalidated patch cannot be activated")
    candidate = Path(root) / receipt["patch_id"] / "candidate.py"
    if digest(candidate) != receipt["candidate_sha256"]:
        raise ValueError("candidate changed after validation")
    checked_function(candidate.read_text())
    registry = Path(root) / "active.json"
    atomic_json(registry, {**receipt, "candidate": str(candidate.absolute()), "activated_at": now()})
    return registry


def payback_decision(receipt: dict, *, remaining_calls: int, measured_reference_seconds: float,
                     preparation_seconds: float, safety_margin: float = 1.25) -> dict:
    if min(remaining_calls, measured_reference_seconds, preparation_seconds) < 0 or safety_margin < 1:
        raise ValueError("invalid payback inputs")
    valid = receipt.get("status") == "validated" and receipt.get("verdict", {}).get("passed")
    saving = remaining_calls * measured_reference_seconds * (1-receipt.get("verdict", {}).get("time_ratio", 1))
    return {"activate": bool(valid and saving > preparation_seconds*safety_margin),
            "remaining_calls": remaining_calls, "expected_saved_seconds": saving,
            "preparation_seconds": preparation_seconds, "safety_margin": safety_margin,
            "reason": "positive_payback" if valid and saving > preparation_seconds*safety_margin else "insufficient_remaining_work"}


def deactivate(root: Path, *, reason: str) -> None:
    registry = Path(root) / "active.json"
    if registry.exists():
        record = read_json(registry)
        atomic_json(Path(root) / "rollbacks" / (object_digest(record) + ".json"),
                    {"previous":record,"reason":reason,"time":now()})
        registry.unlink()


class AuditOptimizer:
    """Tune the imminent audit only when measured work can repay bounded API work."""

    def __init__(self, root: Path, agent, *, minimum_projected_seconds=150):
        self.root, self.agent = Path(root), agent
        self.minimum_projected_seconds = minimum_projected_seconds

    def prepare(self, dataset: Path, *, remaining_seconds: float) -> Path | None:
        import time
        from autosim import robosyn_data
        from .common import exclusive
        self.root.mkdir(parents=True, exist_ok=True)
        with exclusive(self.root / "optimization.lock"):
            started = time.monotonic()
            record = {"status":"skipped", "reason":"insufficient_wall_budget", "time":now()}
            registry = None
            try:
                # At most three calls with 180 s timeouts, plus bounded validators.
                if remaining_seconds < 660:
                    return None
                rows = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line]
                lengths = [int(row["length"]) for row in rows]
                if not lengths or any(length < 0 for length in lengths):
                    record["reason"] = "no_valid_length_workload"
                    return None
                reference = extract_function(Path(robosyn_data.__file__), "_padding_audit")
                sample = lengths[:1000]
                measurement = measure(reference, [{"lengths":sample,"chunk_size":50}])
                # A conservative lower estimate avoids speculative savings across future rounds.
                projected = measurement["elapsed_seconds"] * sum(lengths) / max(1,sum(sample)) * .5
                record.update(projected_kernel_seconds=projected, measured_sample=measurement["elapsed_seconds"],
                              episode_count=len(lengths), remaining_calls=1)
                if projected < self.minimum_projected_seconds:
                    record["reason"] = "imminent_audit_too_small_to_repay_codegen"
                    return None
                # The API sees exactly the permitted function, never the source module.
                source = self.root / "reference_function.py"
                source.write_text(reference)
                receipt = optimize(self.agent, source, self.root / "patches", max_revisions=2)
                decision = payback_decision(receipt, remaining_calls=1,
                    measured_reference_seconds=projected, preparation_seconds=time.monotonic()-started)
                record.update(status=receipt["status"], receipt=receipt, payback=decision, reason=decision["reason"])
                if decision["activate"]:
                    registry = activate(self.root / "patches", receipt)
                return registry
            except Exception as exc:
                record.update(status="fallback", reason=type(exc).__name__)
                return None
            finally:
                record.update(elapsed_seconds=time.monotonic()-started, registry=str(registry) if registry else None)
                atomic_json(self.root / "decisions" / (object_digest(record)+".json"), record)
