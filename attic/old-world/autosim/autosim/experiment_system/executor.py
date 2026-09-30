"""Durable jobs with inherited locks, receipts, bounded costs and fail-closed evals.

The supervisor survives a controller exit and keeps the job/GPU leases. Recovery
commits verified receipts; it never interprets policy failure as a retry request.
This is a cooperative local runtime, NOT an adversarial security sandbox.
"""
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from autosim.research.common import atomic_json, digest, immutable_json, now, object_digest, read_json


@dataclass(frozen=True)
class Job:
    name: str
    stage: str
    command: list[str]
    cwd: str
    outputs: dict[str, str]  # relative path -> nonempty/json/result
    sources: dict[str, str]
    dependencies: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 1800
    max_attempts: int = 1
    budget_seconds: int = 1800
    gpu: str | None = None

    def validate(self):
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", self.name):
            raise ValueError("invalid job name")
        if self.stage not in {"probe", "collect", "admit", "train", "evaluate", "analyze"}:
            raise ValueError("invalid stage")
        if not self.command or not Path(self.cwd).is_dir() or not self.sources:
            raise ValueError("command, cwd, pinned sources required")
        if self.timeout_seconds < 1 or not 1 <= self.max_attempts <= 999 or self.budget_seconds < 1:
            raise ValueError("positive finite budget required")
        if self.gpu is not None and not re.fullmatch(r"[0-9]+", self.gpu):
            raise ValueError("GPU must be a local numeric ID")
        for path, validator in self.outputs.items():
            p = Path(path)
            if p.is_absolute() or ".." in p.parts or validator not in {"nonempty", "json", "result"}:
                raise ValueError("invalid artifact specification")
        if not self.outputs:
            raise ValueError("at least one verified output required")
        # No implicit credential storage in immutable job specifications.
        if any(re.search(r"token|secret|password|api_key", k, re.I) for k in self.environment):
            raise ValueError("credentials must not enter job manifests")
        return self


@contextmanager
def lease(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream
    finally:
        # Do not LOCK_UN: an inherited supervisor may still own this open-file
        # description when a controller is interrupted.
        stream.close()


def verify_sources(sources):
    for name, expected in sources.items():
        p = Path(name)
        if not p.is_file() or digest(p) != expected:
            raise ValueError(f"pinned input changed: {p}")


def verify_outputs(directory: Path, specs: dict) -> dict:
    result = {}
    for name, validator in specs.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing/escaped/empty artifact: {name}")
        if validator in {"json", "result"}:
            data = read_json(path)
            if validator == "result" and (not isinstance(data, dict) or data.get("status") != "completed"):
                raise ValueError(f"result did not certify completion: {name}")
            if validator == "result":
                verify_sources(data.get("verified_files", {}))
        result[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return result


def failure_action(stage: str, receipt: dict | None) -> str:
    if stage == "evaluate":
        # No retry inference from exit code, score, or partial metrics.
        return "quarantine_evaluation"
    if receipt is None:
        return "requires_audit_unknown_execution"
    if receipt.get("failure_kind") in {"infrastructure", "timeout"}:
        return "retry_from_new_attempt"
    return "requires_adapter_or_data_fix"


class Executor:
    def __init__(self, root: Path):
        self.root = root.absolute()
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, job: Job):
        job.validate()
        immutable_json(self.root / job.name / "job.json", asdict(job))

    def committed(self, name):
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
            raise ValueError("invalid dependency")
        path = self.root / name / "commit.json"
        if not path.exists():
            return None
        record = read_json(path)
        spec = read_json(self.root / name / "job.json")
        if record["job_signature"] != object_digest(spec):
            raise ValueError("commit input signature mismatch")
        verify_sources(spec["sources"])
        receipt = self.root / name / record["attempt"] / "receipt.json"
        if digest(receipt) != record["receipt_sha256"]:
            raise ValueError("committed receipt was modified")
        actual = verify_outputs(self.root / name / record["attempt"], spec["outputs"])
        if actual != record["artifacts"]:
            raise ValueError("committed artifact was modified")
        for dep, expected in record.get("dependency_commits", {}).items():
            if digest(self.root / dep / "commit.json") != expected:
                raise ValueError("upstream commit changed")
            self.committed(dep)
        return record

    def _commit(self, job, directory, attempt, receipt):
        verify_sources(job.sources)
        artifacts = verify_outputs(attempt, job.outputs)
        record = {"status": "committed", "job_signature": object_digest(asdict(job)),
                  "attempt": attempt.name, "artifacts": artifacts, "committed_at": now(),
                  "receipt_sha256": digest(attempt / "receipt.json"),
                  "elapsed_seconds": receipt["elapsed_seconds"]}
        record["dependency_commits"] = {dep: digest(self.root / dep / "commit.json") for dep in job.dependencies}
        immutable_json(directory / "commit.json", record)
        return record

    def run(self, job: Job) -> dict:
        self.register(job)
        directory = self.root / job.name
        try:
            with ExitStack() as stack:
                job_lock = stack.enter_context(lease(directory / "job.lock"))
                committed = self.committed(job.name)
                if committed:
                    return committed
                for dep in job.dependencies:
                    if not self.committed(dep):
                        return {"status": "waiting_dependency", "dependency": dep}
                verify_sources(job.sources)
                attempts = sorted(directory.glob("attempt_[0-9][0-9][0-9]"))
                charged = 0.0
                for attempt in attempts:
                    receipt_path = attempt / "receipt.json"
                    receipt = read_json(receipt_path) if receipt_path.exists() else None
                    charged += receipt["elapsed_seconds"] if receipt else job.timeout_seconds
                    if receipt and receipt.get("returncode") == 0:
                        try:
                            return self._commit(job, directory, attempt, receipt)
                        except ValueError as exc:
                            return {"status": "requires_artifact_audit", "error": str(exc)}
                    action = failure_action(job.stage, receipt)
                    if action != "retry_from_new_attempt":
                        return {"status": action, "attempt": attempt.name, "charged_seconds": charged}
                if len(attempts) >= job.max_attempts or charged >= job.budget_seconds:
                    return {"status": "budget_exhausted", "charged_seconds": charged}
                if shutil.disk_usage(self.root).free < 1024**3:
                    return {"status": "waiting_disk_capacity"}
                locks = [job_lock.fileno()]
                if job.gpu is not None:
                    gpu_lock = stack.enter_context(lease(Path(f"/tmp/autosim-robosyn-gpu-{job.gpu}.lock")))
                    locks.append(gpu_lock.fileno())
                attempt = directory / f"attempt_{len(attempts) + 1:03d}"
                attempt.mkdir()
                command = [arg.replace("{attempt}", str(attempt)) for arg in job.command]
                request = {"command": command, "cwd": job.cwd, "environment": job.environment,
                           "timeout": min(job.timeout_seconds, job.budget_seconds - charged),
                           "stage": job.stage, "outputs": job.outputs, "sources": job.sources}
                atomic_json(attempt / "request.json", request)
                # Write request before spawning: an ambiguous crash remains auditable.
                with (attempt / "supervisor.log").open("a") as log:
                    process = subprocess.Popen([sys.executable, "-m", "autosim.experiment_system.worker",
                                                str(attempt)], pass_fds=tuple(locks), start_new_session=True,
                                               stdout=log, stderr=subprocess.STDOUT)
                    process.wait()
                receipt = read_json(attempt / "receipt.json") if (attempt / "receipt.json").exists() else None
                if receipt and receipt.get("returncode") == 0:
                    try:
                        return self._commit(job, directory, attempt, receipt)
                    except ValueError as exc:
                        return {"status": "requires_artifact_audit", "error": str(exc)}
                return {"status": failure_action(job.stage, receipt), "attempt": attempt.name}
        except BlockingIOError:
            return {"status": "waiting_lease"}

    def summary(self):
        rows = []
        for job_file in sorted(self.root.glob("*/job.json")):
            directory = job_file.parent
            receipts = [read_json(p) for p in directory.glob("attempt_*/receipt.json")]
            rows.append({"job": directory.name, "committed": (directory / "commit.json").exists(),
                         "attempts": len(list(directory.glob("attempt_*"))),
                         "elapsed_seconds": sum(r["elapsed_seconds"] for r in receipts),
                         "failed_attempts": sum(r["returncode"] != 0 for r in receipts)})
        return {"jobs": rows, "policy_performance_claim": False}
