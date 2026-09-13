"""Repository-in, optimized-repository-out simulator AutoResearch runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from autosim.llm_client import LLMClient
from autosim.robosyn_data import evaluation_seed_bank
from autosim.robosyn_mvp import gpu_lock

from .analysis import analyze
from .accounting import BudgetLedger, Charge
from .common import atomic_json, digest, immutable_json, now, object_digest, read_json, redact
from .decision_controllers import control_proposal
from .data_version import fingerprint_dataset
from .device_cli import (DevicePlanRefused, DeviceReceiptMissing, devices_protocol_block,
                         plan_summary,
                         protocol_mismatch, resolve_plan, write_plan)
from .devices import NoCompatibleDevice
from .job_runner import PhaseJob, run_phase as run_phase_jobs
from .ledger import SeedLedger, compare
from .registry import TASK_IDS, load_task
from .robosyn_adapter import PROFILE_FAMILIES, RoboSynAdapter
from .runtime import (Runtime, retryable_collection_startup, retryable_evaluation_startup,
                      startup_receipt)
from .task_diagnostics import task_failure_analysis


SCHEMA_VERSION = 1
BENCHMARK = "RoboSynChallenge"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
COLLECTION_PROFILES = set(PROFILE_FAMILIES)
TRAINING_SPACE = {
    "optimizer_lr": {5e-6, 1e-5, 2e-5},
    "n_action_steps": {10, 25, 50},
    "action_loss_profile": {"legacy_mask_mean", "valid_mean"},
    "image_augmentation_profile": {"none", "photometric_mild", "camera_geometry_mild"},
}


def load_deepseek_environment(repo: Path | None = None, *, project_root: Path | None = None) -> dict[str, Any]:
    """Load a narrowly scoped local dotenv without ever returning values.

    A dotenv file is not an operating-system environment.  For the simple
    repository-only CLI reads only the AutoSimSOTA project-level `.env` (or an
    explicitly named `AUTOSIM_ENV_FILE`), while rejecting credential files
    readable by group/other users.  It never searches parent projects.
    """
    root = (project_root or PROJECT_ROOT).absolute()
    explicit = os.environ.get("AUTOSIM_ENV_FILE")
    candidates = [Path(explicit).absolute()] if explicit else [root / ".env"]
    source = next((path.absolute() for path in candidates if path.is_file()), None)
    loaded = []
    if source is not None:
        mode = source.stat().st_mode & 0o777
        if mode & 0o077:
            raise PermissionError(f"credential file must not be group/world accessible: {source}")
        for line in source.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.removeprefix("export ").split("=", 1)
            key, value = key.strip(), value.strip()
            if key not in {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"}:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key not in os.environ and value:
                os.environ[key] = value
                loaded.append(key)
    os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")
    return {"source": str(source) if source else "process_environment_only",
            "loaded_variable_names": sorted(loaded),
            "model": os.environ["DEEPSEEK_MODEL"],
            "base_url": os.environ["DEEPSEEK_BASE_URL"],
            "api_key_available": bool(os.environ.get("DEEPSEEK_API_KEY"))}


@dataclass(frozen=True)
class MilestoneConfig:
    repo_input: str
    output_root: Path
    run_id: str
    task: str = "auto"
    controller: str = "api"
    gpu: str = "0"
    rounds: int = 2
    attempts_per_round: int = 100
    screen_steps: int = 20_000
    development_episodes: int = 40
    selection_episodes: int = 100
    final_episodes: int = 200
    train_seed: int = 1_000
    min_original_fraction: float = 0.25
    hours: float = 24.0
    full_budget: bool = False
    allow_api_egress: bool = False
    dry_run: bool = False
    probe_only: bool = False
    # Multi-device knobs.  All of them default to the absent value: a plain `--gpu i`
    # invocation never builds a device plan, so its frozen protocol stays byte-identical
    # to the runs that predate this feature.
    gpus: str | None = None
    device_mode: str = "auto"
    max_parallel_jobs: int | None = None
    gpu_hours: float | None = None
    accept_device_plan: bool = False
    multi_gpu_probe: bool = False
    probe_if_needed: bool = False
    probe_smoke_timeout: int = 900

    def validate(self) -> "MilestoneConfig":
        if self.task != "auto" and self.task not in TASK_IDS:
            raise ValueError(f"unknown RoboSyn task: {self.task}")
        if self.controller not in {"api", "fixed", "random", "heuristic"}:
            raise ValueError(f"unknown research controller: {self.controller}")
        if self.device_mode not in {"auto", "pinned_index", "identity", "node_isolated"}:
            raise ValueError(f"unknown device addressing mode: {self.device_mode}")
        if self.max_parallel_jobs is not None and self.max_parallel_jobs < 1:
            raise ValueError("--max-parallel-jobs must be at least 1")
        if self.gpu_hours is not None and self.gpu_hours <= 0:
            raise ValueError("--gpu-hours must be positive")
        if not 1 <= self.rounds <= 8:
            raise ValueError("research rounds must be in [1,8]")
        if self.attempts_per_round < 2 or self.screen_steps < 1:
            raise ValueError("collection attempts and training steps must be positive")
        if not 0 <= self.min_original_fraction <= 1:
            raise ValueError("minimum original-distribution fraction must be in [0,1]")
        if min(self.development_episodes, self.selection_episodes, self.final_episodes) < 1:
            raise ValueError("evaluation banks must be non-empty")
        if not 0 < self.hours <= 24:
            raise ValueError("run budget must be in (0,24] hours")
        return self


def protocol_budget(config: MilestoneConfig) -> dict[str, Any]:
    """The frozen budget block, minus the knobs that are invocation-level.

    Egress authorization is a permission, not an experiment variable, so an explicitly
    approved resume keeps the same frozen protocol.  The device knobs are invocation-level
    in the same way: the resolved plan (``protocol["devices"]``) records what the run
    actually executes under -- mode, device set, parallel limit, GPU-hour limit -- so
    dropping the raw flags keeps every legacy ``--gpu i`` budget byte-identical to the runs
    that predate the multi-device feature.
    """
    budget = asdict(config) | {"output_root": str(config.output_root)}
    budget.pop("allow_api_egress", None)
    for key in ("gpus", "device_mode", "max_parallel_jobs", "gpu_hours",
                "accept_device_plan", "multi_gpu_probe", "probe_if_needed",
                "probe_smoke_timeout"):
        budget.pop(key, None)
    return budget


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def resolve_repository(value: str, run_root: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve a local repository or clone a URL into this run's source area."""
    candidate = Path(value).expanduser()
    if candidate.exists():
        repo = candidate.absolute()
        origin = "local_path"
    elif value.startswith(("https://", "http://", "git@")):
        repo = run_root / "source_repo"
        if not repo.exists():
            process_dir = run_root / "jobs" / "clone"
            process_dir.mkdir(parents=True, exist_ok=True)
            log = process_dir / "stdout.log"
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(["git", "clone", "--depth", "1", value, str(repo)],
                                        stdout=stream, stderr=subprocess.STDOUT)
            atomic_json(process_dir / "process.json", {
                "command": ["git", "clone", "--depth", "1", value, str(repo)],
                "returncode": result.returncode, "status": "completed" if result.returncode == 0 else "failed",
            })
            if result.returncode:
                raise RuntimeError(f"repository clone failed: {log}")
        origin = "url_clone"
    else:
        raise FileNotFoundError(f"repository input does not exist: {value}")
    if not repo.is_dir():
        raise ValueError("repository input must resolve to a directory")
    status = _git(repo, "status", "--porcelain")
    identity = {
        "input": value, "resolved_path": str(repo), "origin_kind": origin,
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "git_remote": _git(repo, "remote", "get-url", "origin"),
        "worktree_dirty": bool(status and status != "unknown"),
        "worktree_status_sha256": object_digest(status.splitlines()) if status != "unknown" else None,
    }
    return repo, identity


def discover_robosyn(repo: Path, task: str = "click_bell") -> dict[str, Any]:
    """Backward-compatible public discovery wrapper around the backend adapter."""
    return RoboSynAdapter(repo).discover(task)


def resolve_assets(repo: Path, task: str = "click_bell") -> dict[str, Any]:
    """Backward-compatible public asset resolver around the backend adapter."""
    return RoboSynAdapter(repo).resolve_assets(task)


def _json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("proposal must be one JSON object")
    return parsed


def validate_proposal(raw: dict[str, Any], *, round_index: int, evidence_id: str,
                      parent_checkpoint_sha256: str, parent_data_version: str,
                      allowed_profiles: set[str] | None = None,
                      allowed_collection_modes: set[str] | None = None,
                      attempts_per_round: int = 100,
                      training_steps: int = 20_000,
                      min_original_fraction: float = 0.5) -> dict[str, Any]:
    required = {"proposal_id", "parent_checkpoint_sha256", "parent_data_version",
                "development_evidence_id", "hypothesis", "primary_intervention",
                "collection", "training", "expected_validation"}
    missing, extra = sorted(required - set(raw)), sorted(set(raw) - required)
    # `decision` is an optional schema field. `round` is not part of the schema at all:
    # build_prompt() sends the request as {"round": n, "required_schema": {...}, ...}, and
    # the model sometimes echoes that wrapper key back into its answer. It carries no
    # information the validator does not already bind (round_index is validated against
    # proposal_id and the evidence ids), so it must not fail an otherwise valid proposal --
    # it is dropped below to keep the persisted proposal canonical.
    extra = [field for field in extra if field not in {"decision", "round"}]
    if missing or extra:
        raise ValueError(f"proposal schema mismatch; missing={missing}; extra={extra}")
    raw.pop("round", None)
    if raw["development_evidence_id"] != evidence_id:
        raise ValueError("proposal cites the wrong development evidence")
    if raw["parent_checkpoint_sha256"] != parent_checkpoint_sha256:
        raise ValueError("proposal cites the wrong parent checkpoint")
    if raw["parent_data_version"] != parent_data_version:
        raise ValueError("proposal cites the wrong data version")
    if raw["primary_intervention"] not in {"targeted_data", "policy_correction", "data_processing"}:
        raise ValueError("unregistered primary intervention")
    decision = raw.get("decision", "experiment")
    if decision not in {"experiment", "stop"}:
        raise ValueError("decision must be experiment or stop")
    collection = raw["collection"]
    if set(collection) != {"enabled", "mode", "profile", "targeted_attempts", "original_attempts", "target_episodes"}:
        raise ValueError("collection schema mismatch")
    modes = allowed_collection_modes or {"expert", "policy_correction"}
    if collection["mode"] not in modes:
        raise ValueError("unsupported collection mode")
    if (raw["primary_intervention"] == "policy_correction") != (
            collection["mode"] == "policy_correction"):
        raise ValueError("primary intervention and collection mode are inconsistent")
    profiles = COLLECTION_PROFILES if allowed_profiles is None else allowed_profiles
    if collection["profile"] not in profiles:
        raise ValueError("unsupported targeted collection profile")
    targeted, original = int(collection["targeted_attempts"]), int(collection["original_attempts"])
    if targeted + original > attempts_per_round or min(targeted, original) < 0:
        raise ValueError("collection must fit the attempt budget")
    minimum_original = int(attempts_per_round * min_original_fraction + 0.999999)
    if decision != "stop" and original < minimum_original:
        raise ValueError("collection does not preserve the protocol's minimum original-distribution coverage")
    minimum_probe = min(20, max(1, attempts_per_round // 2))
    if decision != "stop" and round_index == 1 and (
            not collection["enabled"] or targeted < minimum_probe):
        raise ValueError(f"round one must execute a targeted collection probe of at least {minimum_probe} attempts")
    if collection["enabled"] and not 1 <= int(collection["target_episodes"]) <= max(1, targeted):
        raise ValueError("invalid targeted episode request")
    training = raw["training"]
    if set(training) != {"steps", "params", "targeted_sampling_mass", "phase_weights", "horizon_floor"}:
        raise ValueError("training schema mismatch")
    if int(training["steps"]) != training_steps:
        raise ValueError("candidate training steps differ from the frozen run protocol")
    params = training["params"]
    if set(params) - set(TRAINING_SPACE):
        raise ValueError("unregistered training parameter")
    for name, value in params.items():
        if value not in TRAINING_SPACE[name]:
            raise ValueError(f"training parameter outside allowed space: {name}")
    if not 0.05 <= float(training["targeted_sampling_mass"]) <= 0.5:
        raise ValueError("targeted sampling mass must be in [0.05,0.5]")
    phase_names = set(training["phase_weights"])
    if phase_names not in ({"early", "middle", "late"},
                           {"early", "approach", "contact_recovery"}):
        raise ValueError("phase weights must cover the three registered progress bins")
    if any(float(v) <= 0 for v in training["phase_weights"].values()):
        raise ValueError("phase weights must be positive")
    if not 0 < float(training["horizon_floor"]) <= 1:
        raise ValueError("horizon floor must be in (0,1]")
    if not str(raw["hypothesis"]).strip() or not str(raw["expected_validation"]).strip():
        raise ValueError("hypothesis and validation expectation are required")
    if decision == "stop":
        if round_index == 1:
            raise ValueError("round one cannot stop before the required bounded probe")
        if collection["enabled"] or targeted or original or int(collection["target_episodes"]):
            raise ValueError("stop decision must request zero collection work")
    return raw


def build_prompt(round_index: int, context: dict[str, Any]) -> tuple[str, str]:
    profiles = context.get("allowed_collection_profiles", sorted(COLLECTION_PROFILES))
    modes = context.get("allowed_collection_modes", ["expert"])
    attempts = int(context.get("attempts_per_round", 100))
    steps = int(context.get("training_steps", 20_000))
    original_fraction = float(context.get("min_original_fraction", 0.25))
    original = max(1, int(attempts * original_fraction + 0.999999))
    targeted = attempts - original
    system = (
        "You are the research decision module for a simulator manipulation task. Return exactly one JSON object "
        "matching the supplied schema. You may only select registered collection and ACT training controls. "
        "Never change observations, the success judge, evaluation seeds, horizon, evaluator, or execute code. "
        "Treat diagnostics as evidence with stated limitations, not proven causality. Round 1 must request at "
        "least one targeted attempt and at least as many original-distribution attempts. For any enabled "
        f"collection, targeted_attempts + original_attempts must be <={attempts}, with at least {original} "
        "original-distribution attempts. From round 2 onward, choose decision=stop when current evidence is "
        "already saturated or no legal experiment is justified; a stop object must set collection.enabled=false "
        "and all three collection counts to zero. Do not infer task semantics absent from the supplied contract."
    )
    schema = {
        "decision": "experiment|stop",
        "proposal_id": f"round_{round_index}_<short_name>",
        "parent_checkpoint_sha256": context["parent_checkpoint_sha256"],
        "parent_data_version": context["parent_data_version"],
        "development_evidence_id": context["development_evidence_id"],
        "hypothesis": "string", "primary_intervention": "targeted_data|policy_correction|data_processing",
        "collection": {"enabled": True, "mode": "|".join(modes),
                       "profile": "|".join(profiles),
                       "targeted_attempts": targeted, "original_attempts": original,
                       "target_episodes": targeted},
        "training": {"steps": steps, "params": {"action_loss_profile": "valid_mean"},
                     "targeted_sampling_mass": 0.5,
                     "phase_weights": {"early": 0.75, "middle": 1.0, "late": 3.0},
                     "horizon_floor": 0.25},
        "expected_validation": "string",
    }
    user = json.dumps({"round": round_index, "required_schema": schema,
                       "allowed_training_values": {k: sorted(v, key=str) for k, v in TRAINING_SPACE.items()},
                       "context": context}, ensure_ascii=False, sort_keys=True)
    return system, user


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return dict(metrics["summary"])


def clickbell_failure_analysis(evaluation: Path, output: Path) -> dict[str, Any]:
    """Combine generic motion evidence with ClickBell scene/contact slices.

    The press-depth signal is sampled every ten steps and is therefore kept as
    an exploratory diagnostic, never promoted to the official success judge.
    """
    result = analyze(evaluation)
    try:
        from .failure_slice_analysis import _load_repeat

        metrics = read_json(evaluation / "evaluation_metrics.json")
        artifact = Path(metrics.get("artifact_directory", evaluation))
        _, episodes, _ = _load_repeat(artifact)
        labels = ("success", "near_threshold_press", "contact_insufficient_press",
                  "no_button_contact", "unclassified")
        result["clickbell_diagnostics"] = {
            "contact_outcomes_from_ten_step_sampled_qpos": {
                label: sum(row["contact_class"] == label for row in episodes) for label in labels
            },
            "camera_high": {
                "episodes": sum(bool(row["camera_high"]) for row in episodes),
                "failures": sum(bool(row["camera_high"]) and not row["success"] for row in episodes),
            },
            "clutter_near": {
                "episodes": sum(bool(row["clutter_near"]) for row in episodes),
                "failures": sum(bool(row["clutter_near"]) and not row["success"] for row in episodes),
            },
            "measurement_status": "exploratory_ten_step_sampling_not_official_judge",
        }
    except (KeyError, ValueError, FileNotFoundError) as exc:
        result["clickbell_diagnostics"] = {
            "measurement_status": "unavailable", "reason": redact(f"{type(exc).__name__}: {exc}")}
    atomic_json(output, result)
    return result


class RepositoryAutoResearch:
    def __init__(self, config: MilestoneConfig):
        self.config = config.validate()
        self.run_root = (config.output_root / BENCHMARK / config.run_id).absolute()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_root / "run_state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "schema_version": SCHEMA_VERSION, "kind": "robosyn_repository_autoresearch",
            "run_id": config.run_id, "status": "initializing", "created_at": now(), "stage": "initialize",
            "rounds": [], "api_calls": [], "final_confirmation_opened": False,
        }
        self.repo: Path | None = None
        self.adapter: RoboSynAdapter | None = None
        self.task = config.task
        self.runtime: Runtime | None = None
        self.spec = None
        self.assets: dict[str, Any] = {}
        self.allowed_profiles: set[str] = set()
        self.plan: dict[str, Any] | None = None   # stays None for every legacy invocation
        self.ledger = None                        # only a multi-device run keeps accounts
        # Parallel jobs in one phase share this object's state file and capability index;
        # both are read-modify-write, so every mutation goes through one lock.
        self.lock = threading.RLock()

    def _master_seed(self, role: str) -> int:
        """Derive independent, reproducible banks for each research replicate.

        Seed 1000 retains the preregistered v1 constants so completed runs stay
        comparable.  Other research seeds use a domain-separated digest rather
        than merely changing optimization randomness while silently reusing the
        same collection/evaluation episodes.
        """
        legacy = {
            "development": 310_911_001,
            "selection_validation": 310_911_101,
            "final_confirmation": 310_911_201,
            "export_smoke_1": 310_911_301,
        }
        for round_index in range(1, self.config.rounds + 1):
            legacy[f"collection_round_{round_index}_original"] = (
                310_912_001 + (round_index - 1) * 100)
            legacy[f"collection_round_{round_index}_targeted"] = (
                310_912_051 + (round_index - 1) * 100)
        if self.config.train_seed == 1_000 and role in legacy:
            return legacy[role]
        material = f"autosim-repository-banks-v2:{self.config.train_seed}:{role}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")

    def _require_api_egress_authorization(self) -> None:
        if (self.config.controller == "api" and self.task != "click_bell"
                and not self.config.allow_api_egress):
            raise PermissionError(
                "non-ClickBell DeepSeek use requires explicit --allow-api-egress; "
                "structured task contracts and failure summaries leave the local machine")

    def _completed_result(self) -> dict[str, Any] | None:
        """A completed run is a read-only query, independent of cwd and credentials."""
        if self.state.get("status") not in {"completed", "completed_without_target_improvement"}:
            return None
        requested = self.config.repo_input
        if not (requested.startswith("http://") or requested.startswith("https://")):
            expected = Path(requested).expanduser().absolute()
            recorded = self.state.get("repo")
            if recorded and Path(recorded).absolute() != expected:
                raise ValueError(
                    f"run-id belongs to a different repository: recorded={recorded}, requested={expected}")
        if self.config.task != "auto" and self.state.get("task") != self.config.task:
            raise ValueError(
                f"run-id belongs to task {self.state.get('task')}, requested={self.config.task}")
        protocol_path = self.run_root / "protocol.json"
        if self.state.get("kind") in {
                "robosyn_repository_autoresearch", "robosyn_repository_api_autoresearch"
        } and protocol_path.is_file():
            budget = read_json(protocol_path).get("budget", {})
            requested_fields = {
                "controller": self.config.controller,
                "rounds": self.config.rounds,
                "attempts_per_round": self.config.attempts_per_round,
                "screen_steps": self.config.screen_steps,
                "development_episodes": self.config.development_episodes,
                "selection_episodes": self.config.selection_episodes,
                "final_episodes": self.config.final_episodes,
                "train_seed": self.config.train_seed,
                "min_original_fraction": self.config.min_original_fraction,
                "full_budget": self.config.full_budget,
            }
            changed = {key: {"recorded": budget.get(key), "requested": value}
                       for key, value in requested_fields.items()
                       if key in budget and budget[key] != value}
            if changed:
                raise ValueError(f"completed run protocol differs from request: {changed}")
            recorded_devices = budget.get("devices")
            requested_plan = self._requested_plan_identity()
            if recorded_devices and not requested_plan:
                raise ValueError(
                    "completed run was executed under a multi-device plan; re-running it "
                    "without --gpus/--device-mode would change its protocol "
                    f"(recorded plan: {recorded_devices.get('plan_digest')})")
            if requested_plan and not recorded_devices:
                raise ValueError(
                    "completed run was executed single-device; --gpus would change its protocol")
            if recorded_devices:
                changed_plan = {key: {"recorded": recorded_devices.get(key), "requested": value}
                                for key, value in requested_plan.items()
                                if recorded_devices.get(key) != value}
                if changed_plan:
                    raise ValueError(f"completed run device plan differs from request: {changed_plan}")
        return self.state

    def save(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates, updated_at=now())
            atomic_json(self.state_path, self.state)

    def update_capability(self, capability_id: str, status: str, evidence: dict[str, Any],
                          limitation: str | None = None) -> None:
        path = self.run_root / "capabilities.json"
        with self.lock:
            document = read_json(path)
            matches = [row for row in document["records"] if row["capability_id"] == capability_id]
            if len(matches) != 1:
                raise RuntimeError(f"capability record is not unique: {capability_id}")
            matches[0].update(status=status, evidence=evidence, limitation=limitation,
                              verified_at=now() if status == "verified" else None)
            atomic_json(path, document)

    def start_or_resume_budget(self) -> None:
        """Use one wall-clock budget across retries and process restarts."""
        started_text = self.state.get("budget_started_at") or self.state["created_at"]
        started = datetime.fromisoformat(started_text)
        current = datetime.now().astimezone()
        elapsed_seconds = max(0.0, (current - started).total_seconds())
        remaining_seconds = self.config.hours * 3600 - elapsed_seconds
        self.save(budget_started_at=started_text,
                  budget_elapsed_seconds_at_resume=elapsed_seconds,
                  budget_remaining_seconds_at_resume=max(0.0, remaining_seconds))
        if remaining_seconds <= 0:
            raise TimeoutError(
                f"AutoResearch wall-clock budget of {self.config.hours:g} hours is exhausted")
        assert self.runtime is not None
        self.runtime.deadline = time.monotonic() + remaining_seconds

    def device_ledger(self) -> Any:
        """The GPU-hour account.  A legacy run keeps none, so its artifacts are unchanged."""
        if self.plan is None or self.plan["legacy_equivalence"]:
            return None
        if self.ledger is None:
            self.ledger = BudgetLedger(self.run_root / "accounting.json",
                                       wall_limit_seconds=self.config.hours * 3600.0,
                                       gpu_hours_limit=self.plan["gpu_hours_limit"],
                                       devices=self.plan["usable"])
        return self.ledger

    def phase_estimate(self, episodes: int) -> float:
        """Seconds a job of this size is expected to cost: construction plus per-episode marginal."""
        model = getattr(self.runtime, "cost_model", None) or {}
        return (float(model.get("construction_seconds", 360.0))
                + max(0, int(episodes)) * float(model.get("marginal_seconds", 40.0)))

    def run_phase(self, phase: str, jobs: Sequence[PhaseJob]) -> dict[str, Any]:
        """Run one phase's independent jobs: in order on one device, in parallel on several.

        The order *between* phases is the research protocol and is untouched; this only
        stops jobs that do not depend on each other from queueing.  With no multi-device
        plan -- every legacy invocation, and every single-device run -- ``run_phase`` is a
        plain loop over the same calls.
        """
        assert self.runtime is not None
        ledger = self.device_ledger()
        results = run_phase_jobs(phase=phase, jobs=jobs, runtime=self.runtime, plan=self.plan,
                                 output=self.run_root / "schedule",
                                 max_parallel_jobs=self.config.max_parallel_jobs,
                                 ledger=ledger, run_id=self.config.run_id)
        if ledger is not None:
            with self.lock:
                ledger.write()
        return results

    def _resolve_device_plan(self, workspace: Path) -> dict[str, Any] | None:
        """Build the device plan, or refuse with the reason.  Legacy builds none.

        A plan exists only when ``--gpus`` was passed; the other multi-device knobs are
        rejected without it (see ``main``), so "no plan" is exactly "the legacy path".
        """
        if self.config.gpus is None:
            return None
        try:
            plan = self._plan_once(workspace)
        except DeviceReceiptMissing as exc:
            # No receipt for *this* container.  A receipt is keyed by the node it was taken
            # on and this cluster's container hostname is job-scoped, so a receipt from
            # another job can never authorize this one -- measuring here is the only way a
            # multi-device run starts without a hand-carried receipt.  Only a missing
            # receipt is treated this way: "device busy" or "incompatible" is a fact about
            # the node that measuring again would not change.
            if not self.config.probe_if_needed:
                raise RuntimeError(f"multi-device plan refused: {exc}") from exc
            outcome = self._run_device_probe(workspace)
            if not outcome["published"]:
                raise RuntimeError(
                    f"multi-device plan refused: {exc}; the in-run device probe did not "
                    f"publish a usable receipt ({outcome['detail']})"
                ) from exc
            try:
                plan = self._plan_once(workspace)
            except (DevicePlanRefused, NoCompatibleDevice) as again:
                raise RuntimeError(
                    f"multi-device plan refused: {again}; measured in this container by "
                    f"{outcome['artifact']}"
                ) from again
        except (DevicePlanRefused, NoCompatibleDevice) as exc:
            raise RuntimeError(f"multi-device plan refused: {exc}") from exc
        write_plan(self.run_root, plan)
        if self.runtime is not None:
            self.runtime.plan = plan
        self.save(stage="device_plan_resolved", device_plan=plan_summary(plan),
                  device_plan_digest=plan["plan_digest"],
                  device_plan_legacy_equivalence=plan["legacy_equivalence"],
                  devices=[{"index": gpu["index"], "uuid": gpu["uuid"],
                            "class_name": gpu.get("class_name"), "renderer": gpu.get("renderer")}
                           for gpu in plan["usable"]])
        return plan

    def _plan_once(self, workspace: Path) -> dict:
        """One resolution attempt against whatever receipt this container already has."""
        return resolve_plan(platform_root=workspace, requested=self.config.gpus,
                            mode=self.config.device_mode,
                            max_parallel_jobs=self.config.max_parallel_jobs,
                            gpu_hours=self.config.gpu_hours, hours=self.config.hours,
                            accept_partial=self.config.accept_device_plan,
                            image=os.environ.get("AUTOSIM_IMAGE"),
                            worker_spec=os.environ.get("SCO_WORKER_SPEC"))

    def _run_device_probe(self, workspace: Path, *, runner: Callable[..., int] | None = None,
                          ) -> dict[str, Any]:
        """Measure this container's devices, the way the full run will address them.

        Runs the ladder as its own process (`device_probe.main`), for the same reason the
        launcher does: it builds and tears down real simulations, and a crash in one arm
        must not take the research run's process with it.  It costs about an hour of wall
        clock and one episode per device -- charged to this run's GPU-hour account, because
        the cards were held for it whether or not it produced anything else.

        The probe honours the run's own ``--device-mode``, so a run that asks for a mode
        this node cannot do fails here, in the measurement, rather than later mid-phase.
        """
        output = self.run_root / "device_probe"
        output.mkdir(parents=True, exist_ok=True)
        mode = (self.config.device_mode if self.config.device_mode in {"pinned_index", "identity"}
                else "pinned_index")
        command = [sys.executable, "-m", "autosim.research.device_probe",
                   "--workspace", str(workspace), "--output", str(output),
                   "--task", self.task, "--gpus", self.config.gpus,
                   "--mode", mode, "--smoke-timeout", str(int(self.config.probe_smoke_timeout))]
        self.save(stage="device_probe_running", device_probe={
            "command": " ".join(command), "output": str(output),
            "note": "no device receipt for this container; measuring this node before the plan",
        })
        started = time.time()
        returncode = (runner or subprocess.call)(command)
        artifact = output / "device_probe.json"
        receipt = read_json(artifact) if artifact.is_file() else {}
        self.probe_charge = {
            "job": "device_probe", "category": "probe", "wall_seconds": time.time() - started,
            "status": "completed" if returncode == 0 else f"failed(rc={returncode})",
            "devices": [str(uuid) for uuid in (receipt.get("verified_devices")
                                               or [row.get("uuid") for row in
                                                   receipt.get("allocation") or []])],
            "detail": {"mode": receipt.get("mode"), "passed": bool(receipt.get("passed")),
                       "verdict": receipt.get("verdict"), "returncode": returncode,
                       "artifact": str(artifact)},
        }
        published = bool(receipt.get("passed"))
        self.save(stage="device_probe_finished", device_probe={
            "returncode": returncode, "passed": published,
            "mode": receipt.get("mode"), "verified_devices": receipt.get("verified_devices"),
            "verdict": receipt.get("verdict"), "artifact": str(artifact),
        })
        return {"published": published, "artifact": str(artifact),
                "detail": f"rc={returncode}, passed={published}, mode={receipt.get('mode')}"}

    def _charge_device_probe(self) -> None:
        """Put the in-run probe on the same account as everything else it paid for."""
        charge = getattr(self, "probe_charge", None)
        ledger = self.device_ledger()
        if not charge or ledger is None:
            return
        ledger.charge(Charge(job=charge["job"], category=charge["category"],
                             status=charge["status"], devices=tuple(charge["devices"]),
                             wall_seconds=float(charge["wall_seconds"]),
                             detail=charge["detail"]))

    def _requested_plan_identity(self) -> dict[str, Any]:
        """What this invocation asks for, in the plan-identity vocabulary.

        Only knobs the operator actually passed are compared: an omitted knob means "let the
        node decide", which must not become a false conflict against whatever clamped value
        the recorded plan happens to carry.
        """
        if self.config.gpus is None:
            return {}
        identity: dict[str, Any] = {"requested": self.config.gpus}
        if self.config.device_mode != "auto":
            identity["mode"] = self.config.device_mode
        if self.config.max_parallel_jobs is not None:
            identity["max_parallel_jobs"] = self.config.max_parallel_jobs
        if self.config.gpu_hours is not None:
            identity["gpu_hours_limit"] = self.config.gpu_hours
        return identity

    def _check_recorded_plan(self, protocol: dict[str, Any]) -> None:
        """A resume must not silently change which devices the experiment ran on.

        ``immutable_json`` would catch the difference one step later, but a bare frozen-file
        error does not say *which* device knob changed, so the plan identity is compared
        here and reported field by field.
        """
        recorded_path = self.run_root / "protocol.json"
        if not recorded_path.is_file():
            return
        recorded = read_json(recorded_path)
        recorded_devices = (recorded.get("budget") or {}).get("devices")
        resolved = (protocol.get("budget") or {}).get("devices")
        if recorded_devices and not resolved:
            raise ValueError(
                "this run was executed under a multi-device plan; resuming it without "
                "--gpus would change its protocol. Pass the same --gpus/--device-mode/"
                f"--max-parallel-jobs (recorded plan: {recorded_devices.get('plan_digest')})")
        if not recorded_devices and resolved:
            raise ValueError(
                f"this run was executed single-device; --gpus would change its protocol "
                f"(requested plan: {resolved.get('plan_digest')})")
        if recorded_devices and resolved:
            changed = protocol_mismatch(recorded_devices, resolved)
            if changed:
                raise ValueError(f"resume device plan differs from the recorded one: {changed}")

    def initialize(self) -> None:
        self.repo, source = resolve_repository(self.config.repo_input, self.run_root)
        self.adapter = RoboSynAdapter(self.repo)
        self.task = self.adapter.select_task(self.config.task)
        discovery = self.adapter.discover(self.task)
        try:
            self.assets = self.adapter.resolve_assets(self.task)
        except FileNotFoundError as exc:
            if not self.config.probe_only:
                raise
            self.assets = {"task": self.task, "status": "incomplete",
                           "error": redact(f"{type(exc).__name__}: {exc}"),
                           "inventory": self.adapter.asset_inventory(self.task)}
        atomic_json(self.run_root / "source_identity.json", source)
        if source["worktree_dirty"]:
            patch_text = _git(self.repo, "diff", "--binary", "HEAD")
            (self.run_root / "source_worktree.patch").write_text(patch_text + "\n", encoding="utf-8")
            atomic_json(self.run_root / "source_untracked_files.json", {
                "paths": [line[3:] for line in _git(self.repo, "status", "--porcelain").splitlines()
                          if line.startswith("?? ")],
                "note": "file names only; large/untracked assets are inventoried separately",
            })
        atomic_json(self.run_root / "benchmark_discovery.json", discovery)
        atomic_json(self.run_root / "asset_manifest.json", self.assets)
        self.spec = load_task(self.repo, self.task)
        self.allowed_profiles = self.adapter.collection_profiles(self.spec)
        if not self.allowed_profiles:
            raise RuntimeError(f"task {self.task} exposes no legal targeted collection profile")
        atomic_json(self.run_root / "capabilities.json", {
            "schema_version": 1,
            "benchmark": BENCHMARK,
            "task": self.task,
            "records": [row.as_dict() for row in self.adapter.capabilities(self.spec)],
        })
        workspace = Path(__file__).resolve().parents[3]
        eval_candidate = self.repo.with_name(self.repo.name + "_eval_clean")
        eval_repo = eval_candidate if eval_candidate.is_dir() else self.repo
        self.runtime = Runtime(workspace, self.run_root / "runtime", self.config.gpu,
                               repo_path=self.repo, eval_repo_path=eval_repo)
        self.runtime.output.mkdir(parents=True, exist_ok=True)
        project_root = self.runtime.platform_root.absolute()
        owned_paths = [Path(__file__).resolve(), self.repo, eval_repo.absolute(),
                       self.runtime.python.absolute()]
        escaped = [str(path) for path in owned_paths
                   if path != project_root and project_root not in path.parents]
        if escaped:
            raise RuntimeError(f"standalone project boundary violation: {escaped}")
        atomic_json(self.run_root / "project_boundary.json", {
            "status": "passed" if (self.run_root == project_root or project_root in self.run_root.parents)
                      else "explicit_external_output",
            "project_root": str(project_root),
            "owned_runtime_paths": [str(path) for path in owned_paths],
            "artifact_output": str(self.run_root),
            "cross_project_research_reads": [],
            "external_system_dependencies": ["GPU driver", "system shared libraries"],
            "note": "credential file is loaded from the project root; historical priors are literal protocol context, not runtime reads",
        })
        self.plan = self._resolve_device_plan(workspace)
        # Open the GPU-hour account with the run's wall clock, not with the first job: the
        # ledger is what bounds the second, third and tenth job, so it has to have been
        # counting since the budget did.
        self.device_ledger()
        self._charge_device_probe()
        budget = protocol_budget(self.config)
        protocol = {
            "schema_version": SCHEMA_VERSION, "benchmark": BENCHMARK, "task": self.task,
            "source": source, "discovery_sha256": object_digest(discovery), "assets": self.assets,
            "budget": budget,
            "budget_mode": "80k_full" if self.config.full_budget else "20k_compact",
            "comparison_models": ["official_act", "official_data_continuation", "api_selected_candidate"],
            "success_gate_pp": 3.0, "final_feedback_forbidden": True,
        }
        # Only a plan that is *not* legacy equivalent changes the run's identity.  A
        # `--gpus 0` plan addresses exactly the card and the way `--gpu 0` does, so it stays
        # out of the frozen protocol.
        if self.plan is not None and not self.plan["legacy_equivalence"]:
            protocol["devices"] = devices_protocol_block(self.plan)
        self._check_recorded_plan(protocol)
        immutable_json(self.run_root / "run_request.json", {
            "schema_version": 1, "run_id": self.config.run_id,
            "repository": str(self.repo), "source_commit": source["git_commit"],
            "task": self.task, "task_signature": self.spec.signature,
            "protocol_sha256": object_digest(protocol),
        })
        immutable_json(self.run_root / "protocol.json", protocol)
        if not self.config.probe_only:
            self._reserve_banks()
        self.save(status="running", stage="initialized", repo=str(self.repo), benchmark=BENCHMARK, task=self.task)

    def _reserve_banks(self) -> None:
        ledger = SeedLedger(self.run_root / "seed_ledger.sqlite")
        # Retire every locally recorded training/evaluation seed before
        # allocating this run. This is stronger than merely choosing a large,
        # visually different master seed.
        assert self.runtime is not None
        from .controller import import_known_seeds

        import_known_seeds(self.runtime, ledger)
        masters = {name: self._master_seed(name) for name in
                   ("development", "selection_validation", "final_confirmation")}
        for round_index in range(1, self.config.rounds + 1):
            for suffix in ("original", "targeted"):
                name = f"collection_round_{round_index}_{suffix}"
                masters[name] = self._master_seed(name)
        counts = {"development": self.config.development_episodes,
                  "selection_validation": self.config.selection_episodes,
                  "final_confirmation": self.config.final_episodes}
        for round_index in range(1, self.config.rounds + 1):
            counts[f"collection_round_{round_index}_original"] = self.config.attempts_per_round
            counts[f"collection_round_{round_index}_targeted"] = self.config.attempts_per_round
        banks = {}
        for name, master in masters.items():
            purpose = "collection" if name.startswith("collection") else name
            seeds = ledger.reserve(self.task, name, purpose, master, counts[name])
            banks[name] = {"master_seed": master, "count": counts[name],
                           "seed_sha256": object_digest(seeds),
                           "sealed_from_research_context": name == "final_confirmation"}
        ledger.close()
        immutable_json(self.run_root / "seed_banks.json", banks)

    def _existing_evaluation(self, path: Path, purpose: str, count: int) -> dict[str, Any] | None:
        metrics = path / "evaluation_metrics.json"
        if not metrics.is_file():
            return None
        data = read_json(metrics)
        if data.get("purpose") != purpose or len(data.get("episodes", [])) != count:
            raise RuntimeError(f"incompatible cached evaluation: {metrics}")
        return data

    def evaluate(self, checkpoint: Path, label: str, purpose: str, master: int, count: int,
                 *, bound: Runtime | None = None) -> dict[str, Any]:
        """One evaluation, on ``bound``'s device when a phase placed it on one.

        ``bound`` is a per-job view; the unbound path is what every legacy invocation and
        every single-device run takes, and it is unchanged.
        """
        assert self.runtime is not None and self.spec is not None
        runtime = bound or self.runtime
        path = self.run_root / "evaluations" / f"{label}_{purpose}"
        cached = self._existing_evaluation(path, purpose, count)
        if cached is not None:
            result = cached
        else:
            result = runtime.evaluate(self.spec, checkpoint, path, episodes=count,
                                      master_seed=master, purpose=purpose)
        self.update_capability("native_evaluation", "verified", {
            "evaluation": str(path), "execution_mode": result.get("execution_mode"),
            "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
            "episodes": count,
        })
        return result

    def propose(self, round_index: int, context: dict[str, Any]) -> dict[str, Any]:
        path = self.run_root / "rounds" / f"round_{round_index}" / "proposal.json"
        if path.is_file():
            return read_json(path)["proposal"]
        if self.config.controller != "api":
            proposal = control_proposal(
                self.config.controller, round_index, context, seed=self.config.train_seed)
            proposal = validate_proposal(
                proposal, round_index=round_index,
                evidence_id=context["development_evidence_id"],
                parent_checkpoint_sha256=context["parent_checkpoint_sha256"],
                parent_data_version=context["parent_data_version"],
                allowed_profiles=self.allowed_profiles,
                allowed_collection_modes=(
                    {"expert", "policy_correction"} if self.spec.correction_supported else {"expert"}),
                attempts_per_round=self.config.attempts_per_round,
                training_steps=self.config.screen_steps,
                min_original_fraction=self.config.min_original_fraction)
            atomic_json(path, {
                "schema_version": SCHEMA_VERSION, "proposal": proposal,
                "provider": {"kind": self.config.controller, "api_used": False},
                "validation": "passed", "created_at": now(),
            })
            return proposal
        client = LLMClient()
        if not client.available:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in the autosim process environment")
        system, user = build_prompt(round_index, context)
        request_record = {
            "round": round_index, "request_sha256": object_digest({"system": system, "user": user}),
            "requested_model": client.model, "base_url": client.base_url,
            "development_evidence_id": context["development_evidence_id"], "started_at": now(),
        }
        call_dir = path.parent / "api"
        existing_requests = list(call_dir.glob("request*.json")) if call_dir.exists() else []
        proposal_attempt = len(existing_requests) + 1
        request_path = call_dir / ("request.json" if proposal_attempt == 1 else f"request_attempt_{proposal_attempt}.json")
        atomic_json(request_path, {**request_record, "proposal_attempt": proposal_attempt})
        errors = []
        for repair in range(3):
            current_system = system if repair == 0 else system + (
                " Return the COMPLETE object from required_schema, with every required field exactly once and no extra fields.")
            current_user = user if repair == 0 else user + (
                "\nYour previous object was rejected. Previous validation error: " + errors[-1] +
                "\nCopy required_schema as the output shape, replace placeholders with valid values, and omit wrapper fields.")
            try:
                content, metadata = client.chat_with_metadata(
                    current_system, current_user, max_tokens=4096, timeout=120,
                    thinking="disabled")
                proposal = validate_proposal(_json_object(content), round_index=round_index,
                    evidence_id=context["development_evidence_id"],
                    parent_checkpoint_sha256=context["parent_checkpoint_sha256"],
                    parent_data_version=context["parent_data_version"],
                    allowed_profiles=self.allowed_profiles,
                    allowed_collection_modes=(
                        {"expert", "policy_correction"} if self.spec.correction_supported else {"expert"}),
                    attempts_per_round=self.config.attempts_per_round,
                    training_steps=self.config.screen_steps,
                    min_original_fraction=self.config.min_original_fraction)
                record = {"schema_version": SCHEMA_VERSION, "proposal": proposal,
                          "provider": metadata, "repair_attempt": repair,
                          "response_sha256": object_digest(content), "validation": "passed", "created_at": now()}
                atomic_json(path, record)
                self.state["api_calls"].append({"round": round_index, "proposal_id": proposal["proposal_id"],
                                                "provider": metadata, "response_sha256": record["response_sha256"]})
                self.save()
                return proposal
            except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError) and not any(
                    marker in str(exc) for marker in ("empty content", "no choices")
                ):
                    raise
                errors.append(redact(f"{type(exc).__name__}: {exc}"))
                shape: dict[str, Any] = {"parsed_type": "unavailable"}
                try:
                    parsed = _json_object(locals().get("content", ""))
                    required = {"proposal_id", "parent_checkpoint_sha256", "parent_data_version",
                                "development_evidence_id", "hypothesis", "primary_intervention",
                                "collection", "training", "expected_validation"}
                    shape = {"parsed_type": "object", "field_names": sorted(parsed),
                             "missing_fields": sorted(required - set(parsed)),
                             "extra_fields": sorted(set(parsed) - required)}
                except (ValueError, json.JSONDecodeError):
                    pass
                atomic_json(call_dir / f"attempt_{proposal_attempt}_invalid_response_{repair + 1}.json",
                            {"response_sha256": object_digest(locals().get("content", "")),
                             "response_length": len(locals().get("content", "")),
                             "error": errors[-1], "shape": shape})
        raise RuntimeError(f"DeepSeek proposal failed schema validation after repairs: {errors[-1]}")

    def api_preflight(self, client: LLMClient) -> dict[str, Any]:
        path = self.run_root / "api_preflight.json"
        if path.is_file():
            result = read_json(path)
            if result.get("status") != "passed":
                raise RuntimeError("previous DeepSeek API preflight did not pass")
            return result
        content, metadata = client.chat_with_metadata(
            "Return only a JSON object with exactly one field named status whose value is ready.",
            '{"requested_response":{"status":"ready"}}', max_tokens=64, timeout=45,
            thinking="disabled")
        parsed = _json_object(content)
        if parsed != {"status": "ready"}:
            raise RuntimeError("DeepSeek API preflight JSON capability check failed")
        result = {"status": "passed", "provider": metadata,
                  "response_sha256": object_digest(content), "completed_at": now(),
                  "not_a_research_proposal": True}
        atomic_json(path, result)
        return result

    def _prepare_official_data(self) -> dict[str, Any]:
        """Full-audit exact bytes once, then reuse only after a metadata-safe fingerprint hit."""
        assert self.runtime is not None and self.spec is not None
        dataset = Path(self.assets["official_dataset"])
        destination = self.run_root / "official_data_audit"
        published = destination / "data_audit.json"
        if published.is_file():
            return read_json(published)
        cache_root = PROJECT_ROOT / "autoresearch_cache" / "official_data_audits"
        version = fingerprint_dataset(dataset, cache_root / "data_versions")
        auditor = PROJECT_ROOT / "autosim" / "autosim" / "robosyn_data.py"
        identity = {
            "content_id": version["content_id"],
            "task_signature": self.spec.signature,
            "auditor_sha256": digest(auditor),
            "full_video_decode": True,
        }
        shared = cache_root / f"audit_{object_digest(identity)}.json"
        if shared.is_file():
            result = read_json(shared)
            if (not result.get("passed")
                    or result.get("shared_cache_identity") != identity
                    or result.get("dataset", {}).get("video_decode_scope") != "all"):
                raise RuntimeError(f"invalid shared official-data audit cache: {shared}")
            result = dict(result)
            result["content_version"] = version
            result["audit_cache_reuse"] = {
                "status": "reused_after_all_relative_paths_size_and_mtime_ns_matched",
                "shared_artifact": str(shared),
            }
            atomic_json(published, result)
            return result
        result = self.runtime.prepare_data(
            self.spec, dataset, destination, full_video_decode=True)
        result["content_version"] = version
        result["shared_cache_identity"] = identity
        result["audit_cache_reuse"] = {"status": "new_full_audit"}
        atomic_json(published, result)
        immutable_json(shared, result)
        return result

    def _collect_one(self, round_index: int, label: str, *, attempts: int, target: int,
                     profile: str, mode: str, checkpoint: Path,
                     bound: Runtime | None = None) -> dict[str, Any] | None:
        if attempts == 0:
            return None
        assert self.runtime is not None and self.spec is not None
        runtime = bound or self.runtime
        base = self.run_root / "rounds" / f"round_{round_index}" / "collection" / label
        for candidate in [base, base / "startup_attempt_2", base / "startup_attempt_3"]:
            result_path = candidate / "bounded_collection_result.json"
            if result_path.is_file():
                return read_json(result_path)
        master = self._master_seed(f"collection_round_{round_index}_{label}")
        for attempt_index in range(1, 4):
            destination = base if attempt_index == 1 else base / f"startup_attempt_{attempt_index}"
            process_path = destination / "process/process.json"
            if process_path.is_file():
                process = read_json(process_path)
                if retryable_collection_startup(destination) and attempt_index < 3:
                    atomic_json(base / f"startup_attempt_{attempt_index}_retry_decision.json", {
                        "reason": "native_crash_before_first_collection_reset",
                        "returncode": process.get("returncode"), "accepted_episodes": 0,
                        "attempt_budget_consumed": 0, "same_collection_seed_master": True,
                        "next_attempt": attempt_index + 1,
                    })
                    continue
                raise RuntimeError(f"unclassified prior collection attempt: {process_path}")
            try:
                result = runtime.collect_bounded(
                    self.spec, destination, attempt_budget=attempts, target_episodes=min(target, attempts),
                    master_seed=master, profile=profile, collection_mode=mode,
                    correction_checkpoint=checkpoint if mode == "policy_correction" else None,
                    correction_prefix_min=40, correction_prefix_max=160, correction_replan_steps=10,
                    correction_safe_return_steps=20, timeout=7200)
                capability_id = ("policy_prefix_expert_takeover" if mode == "policy_correction"
                                 else "original_distribution_collection" if profile == "full_random"
                                 else f"targeted_collection.{profile}")
                scene_audit = read_json(destination / "scene_evidence_audit.json")
                factor_status = scene_audit.get("requested_factor_readback", {}).get("status", "unknown")
                capability_status = "verified" if factor_status == "verified" else "unknown"
                result["requested_factor_readback"] = factor_status
                result["requested_factor_readback_evidence"] = str(
                    destination / "scene_evidence_audit.json")
                atomic_json(destination / "bounded_collection_result.json", result)
                self.update_capability(capability_id, capability_status, {
                    "result": str(destination / "bounded_collection_result.json"),
                    "profile": profile, "mode": mode,
                    "attempts_consumed": result["attempts_consumed"],
                    "accepted_episodes": result["accepted_episodes"],
                    "capability_state": result["capability_state"],
                    "requested_factor_readback": factor_status,
                }, limitation=(scene_audit.get("requested_factor_readback", {}).get("reason")
                               if factor_status != "verified" else
                               "zero successful expert episodes in bounded probe"
                               if result["accepted_episodes"] == 0 else None))
                return result
            except RuntimeError:
                if not process_path.is_file():
                    raise
                process = read_json(process_path)
                crash_before_reset = (process.get("status") == "failed"
                                      and process.get("returncode") in {-11, -6}
                                      and not (destination / "collection.json").exists()
                                      and not (destination / "scene_resets.jsonl").exists())
                if not crash_before_reset or attempt_index == 3:
                    raise
                atomic_json(base / f"startup_attempt_{attempt_index}_retry_decision.json", {
                    "reason": "native_crash_before_first_collection_reset",
                    "returncode": process.get("returncode"), "accepted_episodes": 0,
                    "attempt_budget_consumed": 0, "same_collection_seed_master": True,
                    "next_attempt": attempt_index + 1,
                })
        raise RuntimeError("collection startup retries exhausted")

    def _make_mixture(self, round_index: int, proposal: dict[str, Any], roots: list[tuple[str, Path]]) -> Path:
        path = self.run_root / "rounds" / f"round_{round_index}" / "mixture.json"
        entries = []
        official = Path(self.assets["official_dataset"])
        all_roots = [("full_random", official)] + roots
        for profile, root in all_roots:
            info = read_json(root / "meta/info.json")
            entries.append({"root": str(root), "profile": profile,
                            "source_kind": "official_full_random" if root == official else "research_requested_collection",
                            "episode_count": int(info["total_episodes"]), "frame_count": int(info["total_frames"]),
                            "info_sha256": digest(root / "meta/info.json")})
        targeted_mass = float(proposal["training"]["targeted_sampling_mass"]) if roots else 0.0
        targeted_profiles = sorted({profile for profile, _ in roots if profile != "full_random"})
        profile_masses = {"full_random": 1.0 - targeted_mass}
        if targeted_profiles:
            for profile in targeted_profiles:
                profile_masses[profile] = targeted_mass / len(targeted_profiles)
        else:
            profile_masses["full_random"] = 1.0
        weights = proposal["training"]["phase_weights"]
        middle_name = "middle" if "middle" in weights else "approach"
        late_name = "late" if "late" in weights else "contact_recovery"
        sampling = {"strategy": "stratified_phase", "profile_masses": profile_masses,
                    "phase_bins": [
                        {"name": "early", "start": 0.0, "end": 0.45, "weight": float(weights["early"])},
                        {"name": "middle", "start": 0.45, "end": 0.75,
                         "weight": float(weights[middle_name])},
                        {"name": "late", "start": 0.75, "end": 1.01,
                         "weight": float(weights[late_name])},
                    ], "horizon_weighting": {"mode": "linear_floor", "chunk_size": 50,
                                              "floor": float(proposal["training"]["horizon_floor"])}}
        payload = {"schema_version": 4, "kind": "robosyn_research_mixture",
                   "datasets": entries, "sampling": sampling,
                   "total_episodes": sum(x["episode_count"] for x in entries),
                   "total_frames": sum(x["frame_count"] for x in entries),
                   "proposal_id": proposal["proposal_id"],
                   "decision_controller": self.config.controller}
        immutable_json(path, payload)
        return path

    def run_round(self, round_index: int, context: dict[str, Any], cumulative: list[tuple[str, Path]]) -> tuple[dict, list[tuple[str, Path]]]:
        assert self.runtime is not None and self.spec is not None
        round_dir = self.run_root / "rounds" / f"round_{round_index}"
        result_path = round_dir / "round_result.json"
        if result_path.is_file():
            result = read_json(result_path)
            return result, [(x["profile"], Path(x["root"])) for x in result["cumulative_data"]]
        proposal = self.propose(round_index, context)
        if proposal.get("decision", "experiment") == "stop":
            result = {"round": round_index, "proposal_id": proposal["proposal_id"],
                      "proposal": proposal, "status": "stopped_by_controller",
                      "reason": proposal["hypothesis"], "completed_at": now()}
            atomic_json(result_path, result)
            self.state["rounds"].append({
                "round": round_index, "proposal_id": proposal["proposal_id"],
                "status": "stopped_by_controller"})
            self.save(stage=f"round_{round_index}_stopped")
            return result, cumulative
        collection = proposal["collection"]
        enabled = bool(collection["enabled"])
        acting_checkpoint = Path(context.get(
            "current_policy_checkpoint", self.assets["official_checkpoint"]))
        targeted_attempts = int(collection["targeted_attempts"]) if enabled else 0
        original_attempts = int(collection["original_attempts"]) if enabled else 0
        # The two collections of a round share no seed bank, no output directory and no
        # checkpoint, and neither result can change the other's -- which is what makes them
        # the phase a second device actually buys: on one device they run in this order.
        collected = self.run_phase(f"round_{round_index}_collection", [
            PhaseJob("collection_targeted", "collection",
                     lambda bound: self._collect_one(
                         round_index, "targeted", attempts=targeted_attempts,
                         target=int(collection["target_episodes"]), profile=collection["profile"],
                         mode=collection["mode"], checkpoint=acting_checkpoint, bound=bound),
                     estimate_seconds=self.phase_estimate(targeted_attempts)),
            PhaseJob("collection_original", "collection",
                     lambda bound: self._collect_one(
                         round_index, "original", attempts=original_attempts,
                         target=original_attempts, profile="full_random", mode="expert",
                         checkpoint=Path(self.assets["official_checkpoint"]), bound=bound),
                     estimate_seconds=self.phase_estimate(original_attempts)),
        ])
        target = collected["collection_targeted"]
        original = collected["collection_original"]
        admitted = []
        excluded = []
        for profile, result in ((collection["profile"], target), ("full_random", original)):
            targeting_verified = (
                profile == "full_random"
                or result is not None
                and result.get("requested_factor_readback") == "verified"
            )
            if result and not targeting_verified:
                excluded.append({
                    "profile": profile,
                    "reason": "requested targeting was not verified by realized-scene readback",
                    "requested_factor_readback": result.get("requested_factor_readback", "unknown"),
                    "accepted_episodes": result.get("accepted_episodes", 0),
                    "dataset_root": result.get("dataset_root"),
                })
                continue
            if result and int(result.get("accepted_episodes", 0)) > 0 and result.get("dataset_root"):
                # A spread collection leaves one dataset per attempt block; every one of
                # them is admitted, and each carries its own shard's yield rather than the
                # merged total, so the mixture's parts stay per-dataset facts.
                per_root = {row.get("dataset_root"): row for row in result.get("shards") or []}
                for root in map(Path, result.get("dataset_roots") or [result["dataset_root"]]):
                    shard = per_root.get(str(root)) or {}
                    self.runtime.prepare_data(self.spec, root,
                                              round_dir / "data_audit" / f"{profile}_{len(admitted)}")
                    cumulative.append((profile, root))
                    admitted.append({"profile": profile, "root": str(root),
                                     "accepted_episodes": shard.get("accepted_episodes",
                                                                    result["accepted_episodes"]),
                                     "attempts_consumed": shard.get("attempts",
                                                                    result["attempts_consumed"])})
        atomic_json(round_dir / "data_admission.json", {
            "admitted": admitted, "excluded": excluded,
            "rule": "targeted sources require verified realized-scene readback; full_random is admitted by its original-distribution contract",
        })
        if round_index == 1 and not any(x["profile"] != "full_random" for x in admitted):
            raise RuntimeError("required targeted/correction probe produced no admitted training data")
        mixture = self._make_mixture(round_index, proposal, cumulative)
        checkpoint = self.runtime.train(self.spec, Path(self.assets["official_dataset"]),
            round_dir / "candidate", steps=self.config.screen_steps,
            params=proposal["training"]["params"], mixture=mixture,
            seed=self.config.train_seed, pretrained=Path(self.assets["official_checkpoint"]))
        self.update_capability("act_training", "verified", {
            "checkpoint": str(checkpoint), "steps": self.config.screen_steps,
            "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
        })
        exposure = read_json(round_dir / "candidate" / "training_exposure_audit.json")
        if not any(int(x.get("yielded_samples", 0)) > 0 and x.get("source_kind") == "research_requested_collection"
                   for x in exposure.get("parts", [])):
            raise RuntimeError("admitted API-requested data received no actual training batch exposure")
        dev = self.evaluate(checkpoint, f"round_{round_index}_candidate", "development",
                            self._master_seed("development"), self.config.development_episodes)
        evaluation_dir = self.run_root / "evaluations" / f"round_{round_index}_candidate_development"
        analysis_path = round_dir / "candidate_failure_analysis.json"
        if analysis_path.is_file():
            candidate_analysis = read_json(analysis_path)
        elif self.task == "click_bell":
            candidate_analysis = clickbell_failure_analysis(evaluation_dir, analysis_path)
        else:
            candidate_analysis = task_failure_analysis(self.task, evaluation_dir, analysis_path)
        result = {"round": round_index, "proposal_id": proposal["proposal_id"],
                  "proposal": proposal, "admitted_data": admitted,
                  "cumulative_data": [{"profile": p, "root": str(r)} for p, r in cumulative],
                  "mixture": str(mixture), "checkpoint": str(checkpoint),
                  "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
                  "development_evaluation": str(evaluation_dir),
                  "candidate_failure_analysis": str(analysis_path),
                  "candidate_evidence_id": object_digest(candidate_analysis),
                  "development_summary": _summary(dev), "training_exposure_audit": str(round_dir / "candidate" / "training_exposure_audit.json"),
                  "status": "completed", "completed_at": now()}
        atomic_json(result_path, result)
        self.state["rounds"].append({k: result[k] for k in ("round", "proposal_id", "checkpoint", "development_summary", "status")})
        self.save(stage=f"round_{round_index}_completed")
        return result, cumulative

    def _feedback_context(self, baseline_analysis: dict[str, Any], prior: dict[str, Any] | None) -> dict[str, Any]:
        compact_baseline = {
            "summary": baseline_analysis.get("summary"),
            "categories": baseline_analysis.get("categories"),
            "clickbell_diagnostics": baseline_analysis.get("clickbell_diagnostics"),
            "limitations": baseline_analysis.get("limitations"),
        }
        exposure = read_json(Path(prior["training_exposure_audit"])) if prior is not None else None
        current_analysis = (read_json(Path(prior["candidate_failure_analysis"]))
                            if prior is not None else baseline_analysis)
        evidence = baseline_analysis if prior is None else {
            "baseline_failure_analysis_summary": compact_baseline,
            "previous_round": prior["round"],
            "previous_proposal": prior["proposal"],
            "previous_admitted_data": [{k: row[k] for k in ("profile", "accepted_episodes", "attempts_consumed")}
                                       for row in prior["admitted_data"]],
            "previous_training_exposure": [{k: row.get(k) for k in
                ("profile", "source_kind", "expected_sampling_mass", "realized_sampling_mass", "yielded_samples")}
                for row in exposure.get("parts", [])],
            "previous_development_summary": prior["development_summary"],
            "current_policy_checkpoint_sha256": prior["checkpoint_sha256"],
            "current_policy_failure_analysis": current_analysis,
            "current_policy_evidence_id": prior["candidate_evidence_id"],
        }
        evidence_id = object_digest(evidence)
        suffix = "baseline.json" if prior is None else f"round_{prior['round']}_feedback.json"
        atomic_json(self.run_root / "evidence" / suffix,
                    {"evidence_id": evidence_id, "evidence": evidence})
        assert self.adapter is not None
        historical = ({
            "official_plus_1500_targeted_local_data": "49% to 74% on a historical same-seed 500-episode local evaluation",
            "residual": "47 of 48 historical candidate failures were heuristically labeled insufficient press depth",
            "warning": "historical combination changed data and processing together and is not this run's result",
        } if self.task == "click_bell" else {
            "status": "no_task_specific_prior_injected",
        })
        return {"task_contract": self.spec.as_dict(),
                "benchmark_challenges": self.adapter.challenge_context(self.spec),
                "capabilities": [row.as_dict() for row in self.adapter.capabilities(self.spec)],
                "allowed_collection_profiles": sorted(self.allowed_profiles),
                "allowed_collection_modes": (["expert", "policy_correction"]
                                             if self.spec.correction_supported else ["expert"]),
                "attempts_per_round": self.config.attempts_per_round,
                "training_steps": self.config.screen_steps,
                "min_original_fraction": self.config.min_original_fraction,
                "historical_noncausal_prior": historical,
                "development_evidence_id": evidence_id, "development_evidence": evidence,
                "parent_checkpoint_sha256": self.assets["checkpoint_weight_sha256"],
                "parent_data_version": self.assets["dataset_info_sha256"],
                "current_policy_checkpoint": (prior["checkpoint"] if prior is not None
                                              else self.assets["official_checkpoint"]),
                "must_use_current_policy_feedback": prior is not None}

    def _select(self, baseline: dict[str, Any], rounds: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Score the baseline and every candidate on one frozen bank.

        These evaluations share a seed bank and a checkpoint per row, and none of them can
        change another's number, so they are one phase: on several devices they run at the
        same time, and on one they run in the order they always did, official first.
        """
        seed = self._master_seed("selection_validation")
        episodes = self.config.selection_episodes
        estimate = self.phase_estimate(episodes)
        jobs = [PhaseJob("round_1_selection_official_act", "evaluation",
                         lambda bound, checkpoint=Path(self.assets["official_checkpoint"]): self.evaluate(
                             checkpoint, "official_act", "selection_validation", seed, episodes,
                             bound=bound), estimate_seconds=estimate)]
        for row in rounds:
            label, checkpoint = f"round_{row['round']}_candidate", Path(row["checkpoint"])
            jobs.append(PhaseJob(f"round_{row['round']}_selection_candidate", "evaluation",
                                 lambda bound, checkpoint=checkpoint, label=label: self.evaluate(
                                     checkpoint, label, "selection_validation", seed, episodes,
                                     bound=bound), estimate_seconds=estimate))
        measured = self.run_phase("selection_validation", jobs)
        official = measured["round_1_selection_official_act"]
        candidates = []
        for row in rounds:
            metrics = measured[f"round_{row['round']}_selection_candidate"]
            candidates.append((row, metrics, compare(metrics, official)))
        def key(item):
            summary = item[1]["summary"]
            return (summary["success_rate"], -summary["average_action_steps"],
                    -summary.get("average_inference_time_per_episode_seconds", float("inf")))
        selected_row, selected_metrics, comparison = max(candidates, key=key)
        selection = {"official_summary": _summary(official), "candidates": [
            {"round": row["round"], "summary": _summary(metrics), "vs_official": comp}
            for row, metrics, comp in candidates], "selected_round": selected_row["round"],
            "selected_checkpoint": selected_row["checkpoint"], "selected_summary": _summary(selected_metrics),
            "selected_vs_official": comparison,
            "qualifies_for_final": comparison["success_delta"] >= 0.03,
            "rule": "success_rate, then average_action_steps, then inference time; retain official on complete tie"}
        atomic_json(self.run_root / "selection.json", selection)
        return selected_row, selection

    def _official_continuation(self, steps: int) -> Path:
        assert self.runtime is not None and self.spec is not None
        output = self.run_root / "controls" / "official_data_continuation"
        prior = output / "train/checkpoints/020000/pretrained_model/model.safetensors"
        return self.runtime.train(self.spec, Path(self.assets["official_dataset"]),
            output, steps=steps, params={}, seed=self.config.train_seed,
            resume=steps > 20_000 and prior.is_file(),
            pretrained=None if steps > 20_000 and prior.is_file() else Path(self.assets["official_checkpoint"]))

    def _export(self, deployment: Path, selected: Path | None, final: dict[str, Any] | None) -> Path:
        assert self.repo is not None and self.runtime is not None
        destination = self.run_root / "optimized_repo"
        # Evaluation during research uses the clean compatibility checkout, so
        # the exported evaluator must use those exact bytes as well. Overlay
        # only the audited collection/training extensions from the working
        # repository; exporting the whole dirty worktree previously produced a
        # deployment repository different from the one used for model ranking.
        export_base = self.runtime.eval_repo if self.runtime.eval_repo.is_dir() else self.repo
        overlay_paths = [
            Path("scripts/run_env.py"),
            Path("policy/act/scripts/train.py"),
            Path("robosynchallenge/managers/datasets.py"),
        ]
        export_composition = {
            "base_repository": str(export_base),
            "base_native_evaluator_sha256": digest(export_base / "scripts/eval_policy.py"),
            "base_policy_loader_sha256": digest(export_base / "policy/act/deploy_policy.py"),
            "overlays": {str(path): digest(self.repo / path) for path in overlay_paths},
        }
        if destination.exists():
            prior_manifest = destination / "AUTORESEARCH_MANIFEST.json"
            prior_composition = (read_json(prior_manifest).get("export_composition")
                                 if prior_manifest.is_file() else None)
            if prior_composition != export_composition:
                raise RuntimeError(
                    "existing optimized_repo has stale or unknown export composition; "
                    "preserve it and rebuild under a new run identity"
                )
        if not destination.exists():
            # `.venv` is machine state, not artifact content: policy/act ships an 11 GB
            # uv venv (torch cu128 wheels) that would be copied here and again into the
            # short /tmp staging tree below, and the exported README tells the deployer
            # to install the dependencies anyway. Validation runs the staged evaluator
            # with this platform's interpreter, so it never reads that venv.
            shutil.copytree(export_base, destination, ignore=shutil.ignore_patterns(
                ".git", ".venv", "lerobot_dataset", "eval_result", "evaluation_results", "__pycache__", "*.pyc", "checkpoints"))
            for relative in overlay_paths:
                source = self.repo / relative
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        checkpoint_relative = Path("checkpoints") / f"ACT_sim_{self.task}"
        checkpoint_dest = destination / checkpoint_relative
        if checkpoint_dest.exists():
            shutil.rmtree(checkpoint_dest)
        shutil.copytree(deployment, checkpoint_dest)
        manifest = {"schema_version": SCHEMA_VERSION, "benchmark": BENCHMARK, "task": self.task,
                    "deployment_policy": "candidate" if selected and deployment == selected else "official_act",
                    "checkpoint_sha256": digest(checkpoint_dest / "model.safetensors"),
                    "source_identity": read_json(self.run_root / "source_identity.json"),
                    "export_composition": export_composition,
                    "runtime_path_requirement": {
                        "kind": "short_resolved_repository_path",
                        "example": "/tmp/robosyn",
                        "reason": "DexSim native DFString material identifiers abort for deeply nested paths",
                    },
                    "run_artifacts": str(self.run_root), "final_result": final,
                    "training_assets": read_json(self.run_root / "asset_manifest.json")}
        atomic_json(destination / "AUTORESEARCH_MANIFEST.json", manifest)
        readme = (
            f"# AutoResearch {self.task} output\n\n"
            "This is a derived RoboSynChallenge repository. The deployed ACT checkpoint is at "
            f"`{checkpoint_relative}`.\n\n"
            "Evaluate from the repository root with the native `scripts/eval_policy.py` entry after installing "
            "the official environment. Reproduction inputs and hashes are in `AUTORESEARCH_MANIFEST.json`; "
            "large training datasets remain external assets and are not silently duplicated.\n\n"
            "DexSim requires a short *resolved* repository path for material creation. Copy or extract this "
            "repository to a path such as `/tmp/robosyn` before native evaluation; a symbolic link to a deeply "
            "nested directory is insufficient because asset paths are resolved.\n"
        )
        (destination / "README_AUTORESEARCH.md").write_text(readme, encoding="utf-8")
        if digest(checkpoint_dest / "model.safetensors") != digest(deployment / "model.safetensors"):
            raise RuntimeError("exported checkpoint hash differs from selected deployment")
        # A file-existence check is insufficient: load and run the exported
        # repository itself through one native smoke episode. This result is
        # operational validation only and never enters model selection.
        ledger = SeedLedger(self.run_root / "seed_ledger.sqlite")
        smoke_banks = []
        # One exported-repository probe is sufficient per research run. The
        # engine fault is already preserved across independent runs/tasks;
        # repeating identical pre-reset hangs in every matched control only
        # consumes budget without adding policy evidence.
        for index in range(1):
            master_seed = self._master_seed(f"export_smoke_{index + 1}")
            seeds = ledger.reserve(
                self.task, f"export_smoke_{index + 1}", "smoke", master_seed, 1)
            smoke_banks.append({"master_seed": master_seed,
                                "seed_sha256": object_digest(seeds)})
        ledger.close()
        immutable_json(self.run_root / "export_smoke_seeds.json", {
            "banks": smoke_banks, "count": len(smoke_banks), "purpose": "operational_smoke",
            "selection_role": "none; operational deployment check only"})
        # DexSim stores material identifiers in a fixed-size native DFString.
        # A valid repository can abort during CreateMaterial when its resolved
        # path is deeply nested.  Validate the exact exported bytes after a
        # physical (not symlinked) copy to a short temporary directory.
        staging_root = Path(tempfile.mkdtemp(prefix="asr_", dir="/tmp"))
        staged_repo = staging_root / "r"
        shutil.copytree(destination, staged_repo, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", "eval_result", "evaluation_results"))
        staged_checkpoint = staged_repo / checkpoint_relative
        if digest(staged_checkpoint / "model.safetensors") != digest(
                checkpoint_dest / "model.safetensors"):
            raise RuntimeError("short-path staged checkpoint differs from exported checkpoint")
        export_runtime = Runtime(self.runtime.workspace, self.run_root / "export_validation",
                                 self.config.gpu, repo_path=staged_repo,
                                 eval_repo_path=staged_repo, python_path=self.runtime.python)
        export_spec = load_task(staged_repo, self.task)
        smoke = None
        startup_failures = []
        native_limitation = None
        last_error = None
        for index, bank in enumerate(smoke_banks, 1):
            smoke_dir = self.run_root / f"export_validation/native_episode_seed_{index}"
            cached = smoke_dir / "evaluation_metrics.json"
            if cached.is_file():
                smoke = read_json(cached)
                break
            try:
                smoke = export_runtime.evaluate(export_spec, staged_checkpoint, smoke_dir,
                                                episodes=1, master_seed=bank["master_seed"],
                                                purpose="smoke", startup_attempts=1,
                                                # 900 s is a real budget, not a token bound. The
                                                # environment takes ~360 s to construct before the
                                                # first reset in this cluster (measured on the run's
                                                # own development evaluation: process start
                                                # 10:21:44.96 -> first reset 10:27:44.68), while
                                                # every real evaluation is allowed >=1800 s. At the
                                                # previous 120 s bound the probe was killed mid-
                                                # construction on every attempt -- ``TimeoutExpired``
                                                # with ``startup.json: environment_constructing`` and
                                                # no ``initializations.jsonl`` -- so it could only
                                                # ever report a startup fault that had not occurred.
                                                smoke_timeout=900)
                break
            except RuntimeError as exc:
                last_error = exc
                # A different smoke bank is legal only when this attempt is
                # positively certified as failing before the first reset. An
                # episode-level or post-action failure is never skipped.
                if not retryable_evaluation_startup(smoke_dir):
                    raise
                startup_failures.append({
                    "master_seed": bank["master_seed"],
                    "artifact_directory": str(smoke_dir),
                    "status": "failed_before_first_reset",
                    "error_type": type(exc).__name__,
                    "startup_receipt": startup_receipt(smoke_dir),
                })
        if smoke is None:
            # All bounded operational probes failed before reset. Loading
            # still validates packaging, but must not set runtime_verified.
            assert last_error is not None
            check_code = (
                "from policy.act.deploy_policy import get_model; "
                f"m=get_model({{'checkpoint_path':'{checkpoint_relative}',"
                "'pytorch_device':'cpu','act_step':10}); "
                "print(type(m).__name__,sum(p.numel() for p in m.parameters()))"
            )
            load_result = export_runtime.run(
                [str(export_runtime.python), "-c", check_code],
                self.run_root / "export_validation/checkpoint_load", 300)
            if deployment == Path(self.assets["official_checkpoint"]):
                real_simulation_evidence = (
                    self.run_root / "evaluations/official_act_selection_validation")
            elif final is not None:
                real_simulation_evidence = (
                    self.run_root /
                    f"evaluations/{self.config.controller}_selected_candidate_final_confirmation")
            else:
                selected_round = read_json(self.run_root / "selection.json")["selected_round"]
                real_simulation_evidence = self.run_root / (
                    f"evaluations/round_{selected_round}_candidate_selection_validation")
            native_limitation = {
                "status": "blocked_before_first_reset",
                # Quote the probes rather than name a cause: each attempt carries
                # its ``startup_receipt`` (termination, phase, whether any
                # initialization was recorded). Saying "renderer fault" here
                # instead would be a diagnosis these artifacts do not support.
                "reason": ("every operational probe ended before its first reset; see "
                           "attempts[].startup_receipt for the recorded phase and termination"),
                "attempts": startup_failures,
                "completed_evaluation_episodes": 0,
                "same_checkpoint_real_simulation_evidence": True,
                "export_smoke_seed_bank_reused": False,
                "error_type": type(last_error).__name__,
                "checkpoint_load_status": load_result["status"],
                "real_simulation_evidence": str(real_simulation_evidence),
            }
            atomic_json(self.run_root / "export_native_smoke_limitation.json", native_limitation)
            smoke = {"execution_mode": "checkpoint_load_fallback", "summary": None}
        export_status = (("passed_after_bounded_native_startup_failures"
                          if startup_failures else "passed") if native_limitation is None
                         else "passed_with_native_startup_limitation")
        atomic_json(self.run_root / "export_validation.json", {
            "status": export_status,
            "export_runtime_verified": native_limitation is None,
            "optimized_repo": str(destination),
            "checkpoint_sha256": digest(checkpoint_dest / "model.safetensors"),
            "native_episode_execution_mode": smoke.get("execution_mode"),
            "native_episode_summary": smoke.get("summary"),
            "native_startup_failures": startup_failures,
            "native_startup_limitation": native_limitation,
            "runtime_staging": {
                "mode": "physical_short_path_copy",
                "resolved_path_length": len(str(staged_repo.resolve())),
                "reason": "DexSim DFString material identifiers fail under deeply nested repository paths",
                "staged_checkpoint_sha256": digest(staged_checkpoint / "model.safetensors"),
            },
            "score_claim": False,
        })
        shutil.rmtree(staging_root)
        return destination

    def execute(self) -> dict[str, Any]:
        try:
            completed = self._completed_result()
            if completed is not None:
                return completed
            self.initialize()
            assert self.runtime is not None and self.spec is not None
            if self.config.dry_run or self.config.probe_only:
                self.save(status="prepared", stage="dry_run_complete",
                          note=("capability/asset probe completed; no API, simulator, training, or score claim"
                                if self.config.probe_only else
                                "discovery/assets/protocol validated; no API, simulator, training, or score claim"))
                return self.state
            if self.config.controller == "api":
                self._require_api_egress_authorization()
                credential_config = load_deepseek_environment(self.repo)
                atomic_json(self.run_root / "llm_configuration.json", credential_config)
                client = LLMClient()
                if not client.available:
                    raise RuntimeError("DEEPSEEK_API_KEY is not set; refusing to substitute a local rule proposer")
                preflight = self.api_preflight(client)
                self.save(stage="api_preflight_passed", api_preflight={
                    "status": preflight["status"], "provider": preflight["provider"]})
            else:
                self.save(stage="control_controller_ready", decision_controller=self.config.controller)
            self.start_or_resume_budget()
            # The legacy index lock guards *this run* against a concurrent single-device
            # run; it is the mechanism the UUID leases replace.  Taking both would be
            # self-defeating -- the leases treat a held legacy lock as occupancy, so a
            # multi-device run holding index 0 would block its own first lease.
            holding_legacy = self.plan is None or self.plan["legacy_equivalence"]
            with gpu_lock(self.config.gpu) if holding_legacy else nullcontext():
                official_audit_path = self.run_root / "official_data_audit"
                official_audit_file = official_audit_path / "data_audit.json"
                if official_audit_file.is_file():
                    official_audit = read_json(official_audit_file)
                else:
                    official_audit = self._prepare_official_data()
                self.update_capability("official_dataset_quality", "verified", {
                    "audit": str(official_audit_file),
                    "info_sha256": official_audit.get("info_sha256"),
                    "video_decode_scope": official_audit.get("dataset", {}).get("video_decode_scope"),
                })
                official_dev = self.evaluate(Path(self.assets["official_checkpoint"]), "official_act", "development",
                                             self._master_seed("development"), self.config.development_episodes)
                analysis_path = self.run_root / "evidence" / "official_development_failure_analysis.json"
                if analysis_path.is_file():
                    baseline_analysis = read_json(analysis_path)
                elif self.task == "click_bell":
                    baseline_analysis = clickbell_failure_analysis(
                        self.run_root / "evaluations/official_act_development", analysis_path)
                else:
                    baseline_analysis = task_failure_analysis(
                        self.task, self.run_root / "evaluations/official_act_development", analysis_path)
                cumulative: list[tuple[str, Path]] = []
                rounds = []
                prior = None
                for round_index in range(1, self.config.rounds + 1):
                    prior, cumulative = self.run_round(
                        round_index, self._feedback_context(baseline_analysis, prior), cumulative)
                    if prior["status"] == "stopped_by_controller":
                        break
                    rounds.append(prior)
                if not rounds:
                    raise RuntimeError("controller stopped without producing any candidate")
                selected_row, selection = self._select(official_dev, rounds)
                compact_steps = self.config.screen_steps
                official_cont: Path | None = None
                candidate = Path(selected_row["checkpoint"])
                # The full 80K path is explicit and decided before final confirmation.
                if self.config.full_budget:
                    candidate = self.runtime.train(self.spec, Path(self.assets["official_dataset"]),
                        self.run_root / "rounds" / f"round_{selected_row['round']}" / "candidate",
                        steps=80_000, params=selected_row["proposal"]["training"]["params"],
                        mixture=Path(selected_row["mixture"]), seed=self.config.train_seed, resume=True)
                    official_cont = self._official_continuation(80_000)
                    recheck_seed = self._master_seed("selection_validation")
                    recheck_episodes = self.config.selection_episodes
                    recheck_estimate = self.phase_estimate(recheck_episodes)
                    rechecks = self.run_phase("full_budget_recheck", [
                        PhaseJob("official_act_80k_recheck", "evaluation",
                                 lambda bound, checkpoint=Path(self.assets["official_checkpoint"]): self.evaluate(
                                     checkpoint, "official_act_80k_recheck", "selection_validation",
                                     recheck_seed, recheck_episodes, bound=bound),
                                 estimate_seconds=recheck_estimate),
                        PhaseJob("selected_candidate_80k_recheck", "evaluation",
                                 lambda bound, checkpoint=candidate: self.evaluate(
                                     checkpoint, f"{self.config.controller}_selected_candidate_80k",
                                     "selection_validation", recheck_seed, recheck_episodes,
                                     bound=bound), estimate_seconds=recheck_estimate),
                        PhaseJob("official_data_continuation_80k_recheck", "evaluation",
                                 lambda bound, checkpoint=official_cont: self.evaluate(
                                     checkpoint, "official_data_continuation_80k",
                                     "selection_validation", recheck_seed, recheck_episodes,
                                     bound=bound), estimate_seconds=recheck_estimate)])
                    official_recheck = rechecks["official_act_80k_recheck"]
                    candidate_recheck = rechecks["selected_candidate_80k_recheck"]
                    control_recheck = rechecks["official_data_continuation_80k_recheck"]
                    selection["full_budget_recheck"] = {
                        "official_summary": _summary(official_recheck),
                        "candidate_summary": _summary(candidate_recheck),
                        "official_continuation_summary": _summary(control_recheck),
                        "candidate_vs_official": compare(candidate_recheck, official_recheck),
                        "candidate_vs_official_continuation": compare(candidate_recheck, control_recheck),
                    }
                    selection["qualifies_for_final"] = (
                        selection["full_budget_recheck"]["candidate_vs_official"]["success_delta"] >= 0.03)
                    atomic_json(self.run_root / "selection.json", selection)
                final = None
                deployment = Path(self.assets["official_checkpoint"])
                if selection["qualifies_for_final"]:
                    if official_cont is None:
                        official_cont = self._official_continuation(compact_steps)
                    self.save(stage="final_confirmation", final_confirmation_opened=True)
                    final_seed = self._master_seed("final_confirmation")
                    final_episodes = self.config.final_episodes
                    final_estimate = self.phase_estimate(final_episodes)
                    metrics = self.run_phase("final_confirmation", [
                        PhaseJob(key, "evaluation",
                                 lambda bound, checkpoint=checkpoint, label=label: self.evaluate(
                                     checkpoint, label, "final_confirmation", final_seed,
                                     final_episodes, bound=bound),
                                 estimate_seconds=final_estimate)
                        for key, label, checkpoint in (
                            ("official_act", "official_act", Path(self.assets["official_checkpoint"])),
                            ("official_data_continuation", "official_data_continuation", official_cont),
                            ("selected_candidate", f"{self.config.controller}_selected_candidate", candidate),
                        )])
                    final = {"summaries": {k: _summary(v) for k, v in metrics.items()},
                             "decision_controller": self.config.controller,
                             "candidate_vs_official": compare(metrics["selected_candidate"], metrics["official_act"]),
                             "candidate_vs_official_continuation": compare(metrics["selected_candidate"], metrics["official_data_continuation"])}
                    final["engineering_improvement_gate_passed"] = final["candidate_vs_official"]["success_delta"] >= 0.03
                    atomic_json(self.run_root / "final_confirmation_report.json", final)
                    if final["engineering_improvement_gate_passed"]:
                        deployment = candidate
                else:
                    atomic_json(self.run_root / "final_confirmation_not_opened.json", {
                        "reason": "selected candidate did not improve at least 3pp on selection validation",
                        "selection": selection})
                exported = self._export(deployment, candidate, final)
                export_validation = read_json(self.run_root / "export_validation.json")
                completed = bool(final and final["engineering_improvement_gate_passed"])
                result_contract = {
                    "execution_complete": True,
                    "result_valid": True,
                    "performance_improved": completed,
                    "hypothesis_supported": None,
                    "export_runtime_verified": bool(export_validation.get("export_runtime_verified")),
                    "reasons": {
                        "performance": "final confirmation improvement gate" if completed else
                                       "candidate did not pass final improvement gate",
                        "hypothesis": "performance comparison does not isolate a causal mechanism",
                        "export": export_validation["status"],
                    },
                }
                self.save(status="completed" if completed else "completed_without_target_improvement",
                          stage="complete", selected_checkpoint=str(candidate), deployment_checkpoint=str(deployment),
                          official_continuation_checkpoint=(str(official_cont) if official_cont else None),
                          optimized_repo=str(exported),
                          api_closed_loop_completed=self.config.controller == "api",
                          decision_controller=self.config.controller,
                          performance_target_achieved=completed,
                          execution_complete=True, result_valid=True,
                          hypothesis_supported=None,
                          export_runtime_verified=result_contract["export_runtime_verified"],
                          run_result=result_contract,
                          error=None)
                return self.state
        except KeyboardInterrupt:
            self.save(status="interrupted", stage=self.state.get("stage", "unknown"),
                      result_valid=False,
                      error="KeyboardInterrupt: run stopped by operator; partial artifacts retained")
            raise
        except Exception as exc:
            self.save(status="blocked", stage=self.state.get("stage", "unknown"),
                      error=redact(f"{type(exc).__name__}: {exc}"))
            raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Official benchmark repository URL or local path")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "autoresearch_runs")
    parser.add_argument("--run-id", default=datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--task", default="auto",
                        help="Task to optimize; auto uses the backend's deterministic default")
    parser.add_argument("--controller", choices=["api", "fixed", "random", "heuristic"],
                        default="api", help="Research decision policy; controls share the same executor and budget")
    parser.add_argument("--gpu", default=None,
                        help="Single-device form: the GPU index this run uses (default 0). "
                             "Cannot be combined with the --gpus family, which would make "
                             "the intended device set ambiguous")
    parser.add_argument("--gpus", default=None,
                        help="Multi-device form: 'auto' or a comma-separated device list such "
                             "as 0,1. Requires a passing device-probe receipt for this node "
                             "(python -m autosim.research.device_probe)")
    parser.add_argument("--device-mode", choices=["auto", "pinned_index", "identity", "node_isolated"],
                        default="auto",
                        help="Addressing scheme; 'auto' adopts the mode the device probe verified")
    parser.add_argument("--max-parallel-jobs", type=int, default=None,
                        help="Upper bound on heavy jobs running at the same time")
    parser.add_argument("--gpu-hours", type=float, default=None,
                        help="Cumulative GPU-hour budget; defaults to --hours x usable devices")
    parser.add_argument("--accept-device-plan", action="store_true",
                        help="Proceed on the usable subset when some requested devices are busy "
                             "or incompatible, and record the accepted reasons")
    parser.add_argument("--multi-gpu-probe", action="store_true",
                        help="Run the device-alignment ladder on this node, publish its receipt, "
                             "and exit without starting a research run")
    parser.add_argument("--probe-if-needed", action="store_true",
                        help="For a multi-device run: if no device receipt matches this container, "
                             "run the ladder here (same budget) before building the plan")
    parser.add_argument("--probe-smoke-timeout", type=int, default=900,
                        help="Per-arm wall-clock bound for --multi-gpu-probe and --probe-if-needed; "
                             "a real episode needs minutes because DexSim construction alone "
                             "costs ~6 minutes")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--attempts-per-round", type=int, default=100)
    parser.add_argument("--training-steps", type=int, default=20_000)
    parser.add_argument("--development-episodes", type=int, default=40)
    parser.add_argument("--selection-episodes", type=int, default=100)
    parser.add_argument("--final-episodes", type=int, default=200)
    parser.add_argument("--train-seed", type=int, default=1000)
    parser.add_argument("--min-original-fraction", type=float, default=0.25)
    parser.add_argument("--full-budget", action="store_true", help="Precommit to 80K candidate/control continuation")
    parser.add_argument("--allow-api-egress", action="store_true",
                        help="Explicitly authorize sending this non-ClickBell task's structured contract and failure summaries to the configured API")
    parser.add_argument("--robotwin-training-epochs", type=int, choices=[2000, 6000], default=6000,
                        help="RoboTwin ACT epoch budget; ignored by RoboSyn")
    parser.add_argument("--robotwin-evaluation-episodes", type=int, default=20,
                        help="Episodes per clean/randomized RoboTwin development bank")
    parser.add_argument("--dry-run", action="store_true", help="Validate discovery/assets/protocol without API or GPU work")
    parser.add_argument("--probe-only", action="store_true",
                        help="Inventory a task even when training assets are missing; performs no API or GPU work")
    return parser


# Compatibility name for historical callers and frozen ClickBell run IDs.
ClickBellAutoResearch = RepositoryAutoResearch


def _reject_ambiguous_devices(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """`--gpu` and the `--gpus` family describe the same thing; refuse to guess.

    Two device models in one invocation would let a multi-device plan be built and then
    executed with the single-device device, which is exactly the silent mislabelling the
    plan is supposed to prevent.
    """
    multi = []
    if args.gpus is not None:
        multi.append("--gpus")
    if args.max_parallel_jobs is not None:
        multi.append("--max-parallel-jobs")
    if args.gpu_hours is not None:
        multi.append("--gpu-hours")
    if args.device_mode != "auto":
        multi.append("--device-mode")
    if args.accept_device_plan:
        multi.append("--accept-device-plan")
    if args.multi_gpu_probe:
        multi.append("--multi-gpu-probe")
    if args.probe_if_needed:
        multi.append("--probe-if-needed")
    if args.gpu is not None and multi:
        parser.error(f"--gpu uses the single-device form and cannot be combined with "
                     f"{', '.join(multi)}; pass --gpus instead of --gpu")
    if args.multi_gpu_probe:
        # The ladder measures the node; it schedules no jobs, so job-level budgets and
        # acknowledgements have nothing to apply to.
        unusable = [name for name, value in (("--max-parallel-jobs", args.max_parallel_jobs),
                                            ("--gpu-hours", args.gpu_hours),
                                            ("--accept-device-plan", args.accept_device_plan or None),
                                            ("--probe-if-needed", args.probe_if_needed or None))
                    if value is not None]
        if unusable:
            parser.error(f"{', '.join(unusable)} do not apply to --multi-gpu-probe, "
                         f"which only measures this node and publishes its receipt")
        return
    dependent = [name for name in multi if name not in {"--gpus", "--multi-gpu-probe"}]
    if dependent and args.gpus is None:
        parser.error(f"{', '.join(dependent)} only applies to a multi-device plan; "
                     f"add --gpus auto (or a device list)")


def _probe_argv(args: argparse.Namespace) -> list[str]:
    """The device ladder runs as its own process, exactly as the launcher runs it."""
    return ["-m", "autosim.research.device_probe",
            "--workspace", str(PROJECT_ROOT),
            "--output", str(Path(args.output_root) / BENCHMARK / f"multigpu_probe_{args.run_id}"),
            "--task", args.task,
            "--gpus", args.gpus or "auto",
            "--mode", args.device_mode if args.device_mode in {"pinned_index", "identity"} else "pinned_index",
            "--smoke-timeout", str(int(args.probe_smoke_timeout))]


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    _reject_ambiguous_devices(parser, args)
    if args.multi_gpu_probe:
        # No research run, no plan: this invocation only measures the node and publishes the
        # receipt a later --gpus run consumes.
        return subprocess.call([sys.executable, *_probe_argv(args)])
    gpu = args.gpu if args.gpu is not None else "0"
    local_repo = Path(args.repo).expanduser()
    if (local_repo / "env_cfg/task_config/demo_randomized.yml").is_file() and (
            local_repo / "collect_data.sh").is_file():
        from .robotwin_adapter import RoboTwinAdapter

        adapter = RoboTwinAdapter(local_repo)
        task = adapter.select_task(args.task)
        run_root = (args.output_root / adapter.benchmark / args.run_id).absolute()
        run_root.mkdir(parents=True, exist_ok=True)
        discovery = adapter.discover(task)
        capabilities = {"schema_version": 1, "benchmark": adapter.benchmark,
                        "task": task,
                        "records": [row.as_dict() for row in adapter.capabilities(task)]}
        atomic_json(run_root / "benchmark_discovery.json", discovery)
        atomic_json(run_root / "capabilities.json", capabilities)
        missing = [row for row in capabilities["records"]
                   if row["status"] in {"unsupported", "unknown"}]
        probe_mode = bool(args.probe_only or args.dry_run)
        executor_error = ("RoboTwin execution capabilities are incomplete: " +
                          ", ".join(row["capability_id"] for row in missing)) if missing else (
                          "RoboTwin native capabilities are verified, but the unified candidate "
                          "research executor is not implemented yet")
        state = {
            "schema_version": 1, "kind": "robotwin_repository_probe",
            "run_id": args.run_id, "repo": str(local_repo.absolute()),
            "benchmark": adapter.benchmark, "task": task,
            "status": "prepared" if probe_mode else "blocked",
            "stage": "static_capability_probe", "created_at": now(),
            "capability_probe_complete": True,
            "execution_complete": probe_mode, "result_valid": probe_mode,
            "performance_improved": False, "export_runtime_verified": False,
            "missing_execution_capabilities": [row["capability_id"] for row in missing],
            "error": None if probe_mode else executor_error,
        }
        if probe_mode:
            atomic_json(run_root / "run_state.json", state)
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0
        from .robotwin_autoresearch import RoboTwinAutoResearch
        runner = RoboTwinAutoResearch(
            project_root=PROJECT_ROOT, repo=local_repo, output_root=args.output_root,
            run_id=args.run_id, task=task, controller=args.controller, gpu=gpu,
            hours=args.hours, train_seed=args.train_seed,
            training_epochs=args.robotwin_training_epochs,
            evaluation_episodes=args.robotwin_evaluation_episodes,
            allow_api_egress=args.allow_api_egress)
        try:
            result = runner.execute()
        except Exception as exc:
            print(f"AutoResearch blocked: {redact(str(exc))}")
            print(f"State: {runner.run_root / 'run_state.json'}")
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 3
    config = MilestoneConfig(repo_input=args.repo, output_root=args.output_root,
                             run_id=args.run_id, task=args.task, controller=args.controller,
                             gpu=gpu, hours=args.hours,
                             rounds=args.rounds, attempts_per_round=args.attempts_per_round,
                             screen_steps=args.training_steps,
                             development_episodes=args.development_episodes,
                             selection_episodes=args.selection_episodes,
                             final_episodes=args.final_episodes,
                             train_seed=args.train_seed,
                             min_original_fraction=args.min_original_fraction,
                             full_budget=args.full_budget,
                             allow_api_egress=args.allow_api_egress,
                             dry_run=args.dry_run,
                             probe_only=args.probe_only,
                             gpus=args.gpus, device_mode=args.device_mode,
                             max_parallel_jobs=args.max_parallel_jobs,
                             gpu_hours=args.gpu_hours,
                             accept_device_plan=args.accept_device_plan,
                             multi_gpu_probe=args.multi_gpu_probe,
                             probe_if_needed=args.probe_if_needed,
                             probe_smoke_timeout=args.probe_smoke_timeout)
    runner = RepositoryAutoResearch(config)
    try:
        result = runner.execute()
    except Exception as exc:
        print(f"AutoResearch blocked: {redact(str(exc))}")
        print(f"State: {runner.state_path}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"completed", "prepared"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
