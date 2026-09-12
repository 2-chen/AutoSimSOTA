"""Minimal, auditable AutoResearch loop for RoboSynChallenge.

The MVP deliberately keeps the mutable surface small: it searches ACT training
recipes, never edits the evaluator, serializes access to one GPU, evaluates all
candidates with the same seeds, and records every command/result in a run
manifest.  Source-code mutation and LLM proposal generation belong in a later
layer once this execution loop is trustworthy.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


@dataclass(frozen=True)
class Recipe:
    name: str
    params: Dict[str, Any]


@dataclass
class MVPConfig:
    name: str
    repo: Path
    python: Path
    gpu_id: str
    task: str
    setting: str
    baseline_checkpoint: Path
    dataset_root: Optional[Path]
    dataset_mixture_manifest: Optional[Path]
    output_root: Path
    embodichain_data_root: Optional[Path]
    extra_library_paths: List[Path] = field(default_factory=list)
    train: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, List[Any]] = field(default_factory=dict)
    frozen_files: List[str] = field(default_factory=list)
    dry_run_baseline_score: float = 0.37

    @classmethod
    def load(cls, path: str | Path) -> "MVPConfig":
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text()) or {}

        def resolve(
            value: Optional[str], *, base: Path = config_path.parent
        ) -> Optional[Path]:
            if value in (None, ""):
                return None
            candidate = Path(value).expanduser()
            # Path.resolve() follows the venv's python symlink and silently loses
            # its site-packages.  Normalize without dereferencing symlinks.
            candidate = candidate if candidate.is_absolute() else base / candidate
            return Path(os.path.abspath(candidate))

        repo = resolve(raw.get("repo"))
        python_bin = resolve(raw.get("python"))
        baseline = resolve(raw.get("baseline_checkpoint"))
        output_root = resolve(raw.get("output_root", "output/robosyn_mvp"))
        if not all((repo, python_bin, baseline, output_root)):
            raise ValueError(
                "repo, python, baseline_checkpoint, and output_root are required"
            )

        task = str(raw.get("task", "click_bell"))
        setting = str(raw.get("setting", "random"))
        frozen = list(
            raw.get("frozen_files")
            or [
                "scripts/eval_policy.py",
                "policy/act/deploy_policy.yml",
                "policy/act/deploy_policy.py",
                "policy/inference_timing.py",
                f"robosynchallenge/tasks/{task}/{task}.py",
                f"configs/{task}/{setting}/gym_config.json",
                f"configs/{task}/action_config.json",
            ]
        )
        return cls(
            name=str(raw.get("name", "robosyn-act-mvp")),
            repo=repo,
            python=python_bin,
            gpu_id=str(raw.get("gpu_id", "0")),
            task=task,
            setting=setting,
            baseline_checkpoint=baseline,
            dataset_root=resolve(raw.get("dataset_root")),
            dataset_mixture_manifest=resolve(raw.get("dataset_mixture_manifest")),
            output_root=output_root,
            embodichain_data_root=resolve(raw.get("embodichain_data_root")),
            extra_library_paths=[
                resolve(item) for item in raw.get("extra_library_paths", [])
            ],
            train=dict(raw.get("train") or {}),
            evaluation=dict(raw.get("evaluation") or {}),
            diagnostics=dict(raw.get("diagnostics") or {}),
            search_space={
                key: list(values)
                for key, values in (raw.get("search_space") or {}).items()
            },
            frozen_files=frozen,
            dry_run_baseline_score=float(raw.get("dry_run_baseline_score", 0.37)),
        )

    @property
    def train_defaults(self) -> Dict[str, Any]:
        defaults = {
            "steps": 20_000,
            "batch_size": 64,
            "num_workers": 4,
            "seed": 1000,
            "chunk_size": 50,
            "n_action_steps": 50,
            "use_amp": False,
            "optimizer_lr": 1e-5,
            "optimizer_weight_decay": 1e-4,
            "kl_weight": 10.0,
            "dropout": 0.1,
            "image_augmentation_profile": "none",
            "action_loss_profile": "legacy_mask_mean",
            "save_freq": 5_000,
            "timeout_seconds": 14_400,
        }
        defaults.update(self.train)
        return defaults

    @property
    def eval_defaults(self) -> Dict[str, Any]:
        defaults = {
            "rungs": [3, 20, 100],
            "training_steps": [5_000, 20_000, 80_000],
            "seed": 0,
            "protocol": "frozen_final",
            "promote_top_k": 1,
            "max_trials": 3,
            "acceptance_min_episodes": 100,
            "acceptance_min_success_delta": 0.03,
            "acceptance_max_p_value": 0.05,
            "timeout_seconds": 3_600,
            "renderer": "hybrid",
            "video": False,
            # DexSim/Vulkan can occasionally abort while creating a fresh
            # process even though the checkpoint and simulator inputs are
            # valid.  Retry startup crashes only; policy/runtime exceptions
            # remain hard failures.
            "startup_retry_exit_codes": [-11, -6],
            "startup_retries": 2,
            "startup_retry_delay_seconds": 5.0,
        }
        defaults.update(self.evaluation)
        defaults["rungs"] = [int(value) for value in defaults["rungs"]]
        defaults["training_steps"] = [
            int(value) for value in defaults["training_steps"]
        ]
        if len(defaults["training_steps"]) != len(defaults["rungs"]):
            raise ValueError("evaluation.training_steps must align with evaluation.rungs")
        raw_seeds = defaults.get("seeds")
        defaults["seeds"] = (
            [int(value) for value in raw_seeds]
            if raw_seeds is not None
            else [int(defaults["seed"])] * len(defaults["rungs"])
        )
        if len(defaults["seeds"]) != len(defaults["rungs"]):
            raise ValueError("evaluation.seeds must align with evaluation.rungs")
        return defaults

    @property
    def diagnostic_defaults(self) -> Dict[str, Any]:
        defaults = {
            "episodes": 5,
            "profiles": ["appearance", "camera", "robot_pose", "clutter"],
            "feedback_action_steps": [8, 10, 16, 50],
        }
        defaults.update(self.diagnostics)
        defaults["episodes"] = int(defaults["episodes"])
        defaults["profiles"] = [str(value) for value in defaults["profiles"]]
        defaults["feedback_action_steps"] = [
            int(value) for value in defaults["feedback_action_steps"]
        ]
        return defaults


def metric_order(metrics: Dict[str, Any]) -> tuple[float, float, float]:
    """Order candidates without allowing speed to compensate for failures."""
    latency = metrics.get("average_inference_time_per_episode_seconds")
    return (
        float(metrics.get("success_rate", 0.0)),
        -float(metrics.get("average_action_steps", math.inf)),
        -float(latency if latency is not None else math.inf),
    )


def paired_success_comparison(
    candidate: Dict[str, Any], baseline: Dict[str, Any]
) -> Dict[str, Any]:
    """Exact one-sided sign test over same-seed binary episode outcomes."""

    def outcomes(metrics: Dict[str, Any]) -> Dict[int, bool]:
        return {
            int(episode["episode_seed"]): bool(episode["success"])
            for episode in metrics.get("episodes", [])
        }

    candidate_outcomes = outcomes(candidate)
    baseline_outcomes = outcomes(baseline)
    seeds = sorted(set(candidate_outcomes) & set(baseline_outcomes))
    wins = sum(candidate_outcomes[seed] and not baseline_outcomes[seed] for seed in seeds)
    losses = sum(baseline_outcomes[seed] and not candidate_outcomes[seed] for seed in seeds)
    discordant = wins + losses
    if discordant == 0 or wins <= losses:
        p_value = 1.0
    else:
        p_value = sum(
            math.comb(discordant, count)
            for count in range(wins, discordant + 1)
        ) / (2**discordant)
    return {
        "paired_episode_count": len(seeds),
        "candidate_only_successes": wins,
        "baseline_only_successes": losses,
        "same_outcome_count": len(seeds) - discordant,
        "success_rate_delta": (
            float(candidate.get("success_rate", 0.0))
            - float(baseline.get("success_rate", 0.0))
        ),
        "one_sided_sign_test_p_value": p_value,
    }


DIAGNOSTIC_PROPOSAL_FAMILIES = {
    "clutter": ["clutter_occlusion_mild"],
    "appearance": ["photometric_mild", "photometric_strong"],
    "robot_pose": ["proprioceptive_noise_not_implemented"],
    "camera": ["multiview_geometry_not_implemented"],
}


def route_diagnostic_proposals(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Route low-performing diagnostic factors to bounded proposal families."""
    ordered = sorted(
        results,
        key=lambda name: (
            float(results[name].get("success_rate", 0.0)),
            -float(results[name].get("average_action_steps", math.inf)),
        ),
    )
    proposals = []
    for profile in ordered:
        for family in DIAGNOSTIC_PROPOSAL_FAMILIES.get(profile, []):
            proposals.append(
                {
                    "diagnostic_profile": profile,
                    "proposal_family": family,
                    "implemented": not family.endswith("_not_implemented"),
                }
            )
    return {
        "factor_priority": ordered,
        "proposals": proposals,
        "implemented_next": [
            item["proposal_family"] for item in proposals if item["implemented"]
        ],
    }


def route_feedback_sweep(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    best_key = max(results, key=lambda key: metric_order(results[key]))
    return {
        "selected_n_action_steps": int(best_key),
        "ordered_n_action_steps": [
            int(key)
            for key in sorted(
                results, key=lambda key: metric_order(results[key]), reverse=True
            )
        ],
        "retrain_required": False,
    }


def acceptance_decision(
    candidate: Optional[Dict[str, Any]],
    baseline: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the final statistical gate; small rungs can promote, never accept."""
    if not candidate:
        return {"accepted": False, "status": "no_candidate"}
    comparison = paired_success_comparison(candidate, baseline)
    episode_count = int(candidate.get("episode_count", 0))
    minimum_episodes = int(evaluation["acceptance_min_episodes"])
    minimum_delta = float(evaluation["acceptance_min_success_delta"])
    maximum_p_value = float(evaluation["acceptance_max_p_value"])
    reasons = []
    if candidate.get("evaluation_config", {}).get("diagnostic_profile"):
        reasons.append("diagnostic slices are not ranking eligible")
    if evaluation.get("protocol") == "development":
        reasons.append("development seed banks can promote candidates but cannot accept them")
    if episode_count < minimum_episodes:
        reasons.append(f"requires at least {minimum_episodes} episodes")
    if comparison["paired_episode_count"] != episode_count:
        reasons.append("candidate and baseline do not have identical paired seed coverage")
    if comparison["success_rate_delta"] < minimum_delta:
        reasons.append(f"success-rate delta is below {minimum_delta:.3f}")
    if comparison["one_sided_sign_test_p_value"] > maximum_p_value:
        reasons.append(f"paired p-value is above {maximum_p_value:.3f}")
    return {
        "accepted": not reasons,
        "status": "accepted" if not reasons else "inconclusive_or_rejected",
        "reasons": reasons,
        "thresholds": {
            "minimum_episodes": minimum_episodes,
            "minimum_success_rate_delta": minimum_delta,
            "maximum_p_value": maximum_p_value,
        },
        "comparison": comparison,
    }


class FrozenEvaluatorGuard:
    """Hash files that an optimization candidate must never change."""

    def __init__(self, repo: Path, relative_paths: Iterable[str]):
        self.repo = repo
        self.paths = [repo / path for path in relative_paths]
        self.before = self.snapshot()

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def snapshot(self) -> Dict[str, str]:
        return {
            str(path.relative_to(self.repo)): self._digest(path)
            for path in self.paths
            if path.exists()
        }

    def assert_unchanged(self) -> None:
        after = self.snapshot()
        if after != self.before:
            changed = sorted(set(self.before) | set(after))
            changed = [
                path for path in changed if self.before.get(path) != after.get(path)
            ]
            raise RuntimeError(
                f"frozen evaluator files changed during the run: {changed}"
            )


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, **payload: Any) -> None:
        record = {
            "time": datetime.now().astimezone().isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


@contextmanager
def gpu_lock(gpu_id: str):
    """Prevent two local AutoResearch jobs from accidentally sharing one GPU."""
    lock_path = Path(f"/tmp/autosim-robosyn-gpu-{gpu_id}.lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"GPU {gpu_id} is already leased by another AutoResearch run"
            ) from exc
        lock_file.write(f"pid={os.getpid()} started={datetime.now().isoformat()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def generate_recipes(config: MVPConfig) -> List[Recipe]:
    """Deterministic Cartesian search capped by max_trials."""
    defaults = config.train_defaults
    space = config.search_space or {
        "chunk_size": [32, 50, 64],
        "n_action_steps": [32, 50],
        "use_amp": [False, True],
    }
    keys = sorted(space)
    recipes: List[Recipe] = []
    for values in itertools.product(*(space[key] for key in keys)):
        params = dict(defaults)
        params.update(dict(zip(keys, values)))
        if all(params.get(key) == defaults.get(key) for key in keys):
            continue
        digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[
            :8
        ]
        label = "_".join(f"{key}-{params[key]}" for key in keys)
        recipes.append(Recipe(name=f"{label}_{digest}", params=params))
    if not recipes and config.dataset_mixture_manifest is not None:
        recipes.append(Recipe(name="targeted_data_only", params=dict(defaults)))
    return recipes[: int(config.eval_defaults["max_trials"])]


class RoboSynMVPRunner:
    def __init__(
        self,
        config: MVPConfig,
        *,
        dry_run: bool = False,
        run_dir: Optional[Path] = None,
    ):
        self.config = config
        self.dry_run = dry_run
        if run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = config.output_root / f"{config.name}_{stamp}_{os.getpid()}"
            self.run_dir.mkdir(parents=True, exist_ok=False)
        else:
            self.run_dir = Path(run_dir).expanduser().resolve()
            if not self.run_dir.is_dir():
                raise FileNotFoundError(f"resume run directory does not exist: {self.run_dir}")
        self.events = EventLog(self.run_dir / "events.jsonl")
        self.guard = FrozenEvaluatorGuard(config.repo, config.frozen_files)
        self.records: List[Dict[str, Any]] = []

    def preflight(self, *, require_dataset: bool) -> List[str]:
        return preflight_config(self.config, require_dataset=require_dataset)

    def _environment(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.gpu_id
        env["PYTHONWARNINGS"] = "ignore::UserWarning"
        env.setdefault("MPLCONFIGDIR", "/tmp/robosyn_autoresearch_matplotlib")
        env.setdefault("XDG_CACHE_HOME", "/tmp/robosyn_autoresearch_cache")
        if self.config.embodichain_data_root:
            env["EMBODICHAIN_DATA_ROOT"] = str(self.config.embodichain_data_root)
        workspace = self.config.repo.parent
        python_paths = [
            str(self.config.repo),
            str(self.config.repo / "policy"),
            str(workspace / "EmbodiChain"),
        ]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        library_paths = [str(path) for path in self.config.extra_library_paths if path]
        if env.get("LD_LIBRARY_PATH"):
            library_paths.append(env["LD_LIBRARY_PATH"])
        if library_paths:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
        return env

    def _run_command(self, command: List[str], log_path: Path, timeout: int) -> None:
        self.events.append("command_started", command=command, log=str(log_path))
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=self.config.repo,
                    env=self._environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                self.events.append("command_timeout", command=command, timeout=timeout)
                raise RuntimeError(
                    f"command timed out after {timeout}s; see {log_path}"
                ) from exc
        elapsed = time.monotonic() - started
        self.events.append(
            "command_finished",
            command=command,
            returncode=result.returncode,
            elapsed_seconds=elapsed,
            log=str(log_path),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"command failed with exit code {result.returncode}; see {log_path}"
            )

    def _eval_command(
        self,
        checkpoint: Path,
        episodes: int,
        model_name: str,
        diagnostic_profile: Optional[str] = None,
        act_n_action_steps_override: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        evaluation = self.config.eval_defaults
        command = [
            str(self.config.python),
            "scripts/eval_policy.py",
            "--config",
            "policy/act/deploy_policy.yml",
            "--overrides",
            "--task_name",
            self.config.task,
            "--setting",
            self.config.setting,
            "--checkpoint_path",
            str(checkpoint),
            "--model_name",
            model_name,
            "--max_episodes",
            str(episodes),
            "--seed",
            str(evaluation["seed"] if seed is None else seed),
            "--eval_video_log",
            str(bool(evaluation["video"])),
            "--eval_reset_sync_steps",
            "0",
            "--headless",
            "True",
            "--renderer",
            str(evaluation["renderer"]),
            "--pytorch_device",
            "cuda",
        ]
        if diagnostic_profile:
            command.extend(["--eval_diagnostic_profile", diagnostic_profile])
        if act_n_action_steps_override is not None:
            command.extend(
                ["--act_n_action_steps_override", str(act_n_action_steps_override)]
            )
        return command

    def _new_metric_file(self, before: set[Path]) -> Path:
        root = (
            self.config.repo
            / "eval_result"
            / self.config.task
            / "act"
            / self.config.setting
        )
        after = set(root.rglob("evaluation_metrics.json")) if root.exists() else set()
        created = sorted(after - before, key=lambda path: path.stat().st_mtime_ns)
        if not created:
            raise RuntimeError(f"evaluation produced no new metrics under {root}")
        return created[-1]

    def evaluate(
        self,
        checkpoint: Path,
        episodes: int,
        name: str,
        diagnostic_profile: Optional[str] = None,
        act_n_action_steps_override: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.dry_run:
            if name == "baseline":
                score = self.config.dry_run_baseline_score
            else:
                recipe = next(
                    item for item in generate_recipes(self.config) if item.name == name
                )
                chunk = float(recipe.params.get("chunk_size", 50))
                actions = float(recipe.params.get("n_action_steps", 50))
                amp_bonus = 0.005 if recipe.params.get("use_amp") else 0.0
                score = max(
                    0.0,
                    min(
                        1.0,
                        0.40
                        - abs(chunk - 50) * 0.001
                        - abs(actions - 50) * 0.0005
                        + amp_bonus,
                    ),
                )
            return {
                "score": round(score, 6),
                "success_count": round(score * episodes),
                "episode_count": episodes,
                "dry_run": True,
            }

        metric_root = (
            self.config.repo
            / "eval_result"
            / self.config.task
            / "act"
            / self.config.setting
        )
        before = (
            set(metric_root.rglob("evaluation_metrics.json"))
            if metric_root.exists()
            else set()
        )
        profile_suffix = f"_{diagnostic_profile}" if diagnostic_profile else ""
        if act_n_action_steps_override is not None:
            profile_suffix += f"_actions{act_n_action_steps_override}"
        evaluation = self.config.eval_defaults
        evaluation_seed = evaluation["seed"] if seed is None else seed
        log_path = self.run_dir / (
            f"eval_{name}{profile_suffix}_{episodes}ep_seed_{evaluation_seed}.log"
        )
        retries = int(evaluation["startup_retries"])
        retry_codes = {int(value) for value in evaluation["startup_retry_exit_codes"]}
        command = self._eval_command(
            checkpoint,
            episodes,
            name,
            diagnostic_profile=diagnostic_profile,
            act_n_action_steps_override=act_n_action_steps_override,
            seed=seed,
        )
        for attempt in range(retries + 1):
            attempt_log = (
                log_path
                if attempt == 0
                else log_path.with_name(f"{log_path.stem}_attempt{attempt + 1}.log")
            )
            try:
                self._run_command(
                    command,
                    attempt_log,
                    int(evaluation["timeout_seconds"]),
                )
                log_path = attempt_log
                break
            except RuntimeError as exc:
                retryable = any(
                    f"exit code {code}" in str(exc) for code in retry_codes
                )
                if not retryable or attempt >= retries:
                    raise
                delay = float(evaluation["startup_retry_delay_seconds"])
                self.events.append(
                    "evaluation_startup_retry",
                    name=name,
                    episodes=episodes,
                    seed=seed,
                    failed_attempt=attempt + 1,
                    next_attempt=attempt + 2,
                    delay_seconds=delay,
                    error=str(exc),
                    log=str(attempt_log),
                )
                time.sleep(delay)
        metric_path = self._new_metric_file(before)
        payload = json.loads(metric_path.read_text())
        summary = dict(payload["summary"])
        summary["episodes"] = list(payload.get("episodes") or [])
        summary["evaluation_config"] = dict(payload.get("config") or {})
        summary["diagnostic"] = payload.get("diagnostic")
        summary["score"] = float(summary["success_rate"])
        summary["metrics_path"] = str(metric_path)
        summary["log_path"] = str(log_path)
        return summary

    def run_diagnostics_only(self) -> Dict[str, Any]:
        """Evaluate non-ranking challenge slices without touching acceptance."""
        problems = self.preflight(require_dataset=False)
        if problems:
            raise RuntimeError("preflight failed:\n- " + "\n- ".join(problems))
        diagnostic = self.config.diagnostic_defaults
        results = {}
        for profile in diagnostic["profiles"]:
            metrics = self.evaluate(
                self.config.baseline_checkpoint,
                diagnostic["episodes"],
                f"diagnostic_{profile}",
                diagnostic_profile=profile,
            )
            results[profile] = metrics
            self.guard.assert_unchanged()
        summary = {
            "schema_version": 1,
            "name": self.config.name,
            "created_at": datetime.now().astimezone().isoformat(),
            "dry_run": self.dry_run,
            "status": "diagnostics_only",
            "ranking_eligible": False,
            "checkpoint": str(self.config.baseline_checkpoint),
            "episodes_per_profile": diagnostic["episodes"],
            "frozen_hashes": self.guard.before,
            "diagnostics": results,
            "routing_decision": route_diagnostic_proposals(results),
            "accepted": False,
        }
        final = self._write_summary(summary)
        self.events.append("diagnostics_finished", summary=str(final))
        return summary

    def run_feedback_sweep_only(self) -> Dict[str, Any]:
        """Pairwise screen ACT replanning frequency with frozen model weights."""
        problems = self.preflight(require_dataset=False)
        if problems:
            raise RuntimeError("preflight failed:\n- " + "\n- ".join(problems))
        diagnostic = self.config.diagnostic_defaults
        results = {}
        for action_steps in diagnostic["feedback_action_steps"]:
            metrics = self.evaluate(
                self.config.baseline_checkpoint,
                diagnostic["episodes"],
                f"feedback_actions_{action_steps}",
                act_n_action_steps_override=action_steps,
            )
            results[str(action_steps)] = metrics
            self.guard.assert_unchanged()
        reference_key = str(max(diagnostic["feedback_action_steps"]))
        reference = results[reference_key]
        comparisons = {
            key: paired_success_comparison(metrics, reference)
            for key, metrics in results.items()
            if key != reference_key
        }
        summary = {
            "schema_version": 1,
            "name": self.config.name,
            "created_at": datetime.now().astimezone().isoformat(),
            "dry_run": self.dry_run,
            "status": "feedback_sweep_only",
            "ranking_eligible": False,
            "checkpoint": str(self.config.baseline_checkpoint),
            "episodes_per_candidate": diagnostic["episodes"],
            "reference_n_action_steps": int(reference_key),
            "frozen_hashes": self.guard.before,
            "feedback_sweep": results,
            "paired_comparisons": comparisons,
            "routing_decision": route_feedback_sweep(results),
            "accepted": False,
        }
        final = self._write_summary(summary)
        self.events.append("feedback_sweep_finished", summary=str(final))
        return summary

    def train(
        self, recipe: Recipe, *, steps: Optional[int] = None, resume: bool = False
    ) -> Path:
        candidate_dir = self.run_dir / "candidates" / recipe.name
        train_dir = candidate_dir / "train"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        target_steps = int(steps if steps is not None else recipe.params["steps"])
        command = [
            str(self.config.python),
            "policy/act/scripts/train.py",
            "--dataset-root",
            str(self.config.dataset_root),
            "--output-dir",
            str(train_dir),
            "--device",
            "cuda",
            "--steps",
            str(target_steps),
            "--batch-size",
            str(int(recipe.params["batch_size"])),
            "--num-workers",
            str(int(recipe.params["num_workers"])),
            "--seed",
            str(int(recipe.params["seed"])),
            "--chunk-size",
            str(int(recipe.params["chunk_size"])),
            "--n-action-steps",
            str(int(recipe.params["n_action_steps"])),
            "--optimizer-lr",
            str(float(recipe.params["optimizer_lr"])),
            "--optimizer-weight-decay",
            str(float(recipe.params["optimizer_weight_decay"])),
            "--kl-weight",
            str(float(recipe.params["kl_weight"])),
            "--dropout",
            str(float(recipe.params["dropout"])),
            "--image-augmentation-profile",
            str(recipe.params["image_augmentation_profile"]),
            "--action-loss-profile",
            str(recipe.params["action_loss_profile"]),
            "--save-freq",
            str(min(target_steps, int(recipe.params.get("save_freq", target_steps)))),
            "--log-freq",
            "100",
            "--data-pipeline-audit",
            str(candidate_dir / f"data_pipeline_{target_steps:06d}.json"),
        ]
        if self.config.dataset_mixture_manifest is not None:
            command.extend(
                [
                    "--dataset-mixture-manifest",
                    str(self.config.dataset_mixture_manifest),
                ]
            )
        if recipe.params.get("use_amp"):
            command.append("--use-amp")
        if resume:
            command.append("--resume")
        (candidate_dir / "recipe.json").write_text(
            json.dumps(asdict(recipe), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.dry_run:
            (candidate_dir / f"planned_train_{target_steps:06d}.json").write_text(
                json.dumps(command, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return train_dir / "checkpoints" / f"{target_steps:06d}" / "pretrained_model"

        self._run_command(
            command,
            candidate_dir / f"train_{target_steps:06d}.log",
            int(recipe.params["timeout_seconds"]),
        )
        expected = train_dir / "checkpoints" / f"{target_steps:06d}" / "pretrained_model"
        if (expected / "model.safetensors").is_file():
            return expected
        checkpoint_dirs = [
            path.parent
            for path in train_dir.rglob("model.safetensors")
            if (path.parent / "config.json").exists()
        ]
        if not checkpoint_dirs:
            raise RuntimeError(
                f"training produced no LeRobot checkpoint under {train_dir}"
            )
        return max(checkpoint_dirs, key=lambda path: path.stat().st_mtime_ns)

    def _record(self, **record: Any) -> Dict[str, Any]:
        self.records.append(record)
        self.events.append("result", **record)
        return record

    def _write_summary(self, summary: Dict[str, Any]) -> Path:
        temp = self.run_dir / "summary.json.tmp"
        final = self.run_dir / "summary.json"
        temp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        os.replace(temp, final)
        return final

    def run_baseline_only(self) -> Dict[str, Any]:
        problems = self.preflight(require_dataset=False)
        if problems:
            raise RuntimeError("preflight failed:\n- " + "\n- ".join(problems))
        episodes = self.config.eval_defaults["rungs"][0]
        self.events.append("baseline_started", episodes=episodes)
        metrics = self.evaluate(self.config.baseline_checkpoint, episodes, "baseline")
        self.guard.assert_unchanged()
        summary = {
            "schema_version": 1,
            "name": self.config.name,
            "created_at": datetime.now().astimezone().isoformat(),
            "dry_run": False,
            "status": "baseline_only",
            "hardware": {"gpu_id": self.config.gpu_id, "host": platform.node()},
            "frozen_hashes": self.guard.before,
            "baseline_by_episodes": {episodes: metrics},
            "best": None,
            "accepted": False,
            "records": [],
        }
        final = self._write_summary(summary)
        self.events.append(
            "baseline_finished", score=metrics["score"], summary=str(final)
        )
        return summary

    def run_reference_only(self) -> Dict[str, Any]:
        """Train and evaluate the fixed default recipe before any search."""
        problems = self.preflight(require_dataset=True)
        if problems:
            raise RuntimeError("preflight failed:\n- " + "\n- ".join(problems))
        episodes = self.config.eval_defaults["rungs"][0]
        baseline = self.evaluate(
            self.config.baseline_checkpoint, episodes, "baseline"
        )
        recipe = Recipe(name="reference", params=self.config.train_defaults)
        try:
            checkpoint = self.train(recipe)
            metrics = self.evaluate(checkpoint, episodes, recipe.name)
            best = self._record(
                name=recipe.name,
                status="completed",
                rung=0,
                episodes=episodes,
                score=metrics["score"],
                metrics=metrics,
                params=recipe.params,
                checkpoint=str(checkpoint),
            )
        except Exception as exc:
            best = self._record(
                name=recipe.name,
                status="failed",
                rung=0,
                episodes=episodes,
                score=0.0,
                metrics={},
                params=recipe.params,
                checkpoint=None,
                error=str(exc),
            )
        self.guard.assert_unchanged()
        decision = acceptance_decision(
            best.get("metrics") if best["status"] == "completed" else None,
            baseline,
            self.config.eval_defaults,
        )
        accepted = bool(decision["accepted"])
        summary = {
            "schema_version": 1,
            "name": self.config.name,
            "created_at": datetime.now().astimezone().isoformat(),
            "dry_run": False,
            "status": "reference_only",
            "hardware": {"gpu_id": self.config.gpu_id, "host": platform.node()},
            "frozen_hashes": self.guard.before,
            "baseline_by_episodes": {episodes: baseline},
            "best": best,
            "accepted": accepted,
            "acceptance_decision": decision,
            "records": self.records,
        }
        final = self._write_summary(summary)
        self.events.append("reference_finished", accepted=accepted, summary=str(final))
        if best["status"] != "completed":
            raise RuntimeError(best["error"])
        return summary

    def run(self) -> Dict[str, Any]:
        problems = self.preflight(require_dataset=not self.dry_run)
        if problems:
            raise RuntimeError("preflight failed:\n- " + "\n- ".join(problems))

        recipes = generate_recipes(self.config)
        rungs = self.config.eval_defaults["rungs"]
        if not recipes:
            raise RuntimeError(
                "search_space generated no recipe different from train defaults"
            )
        self.events.append(
            "run_started", config=asdict(self.config), dry_run=self.dry_run
        )

        baseline_by_rung: Dict[str, Dict[str, Any]] = {}
        recipe_by_name = {recipe.name: recipe for recipe in recipes}
        active_names = list(recipe_by_name)
        trained: Dict[str, Path] = {}
        training_steps = self.config.eval_defaults["training_steps"]
        final_rung = -1
        final_baseline: Dict[str, Any] = {}
        for rung_index, episodes in enumerate(rungs):
            if not active_names:
                break
            rung_seed = self.config.eval_defaults["seeds"][rung_index]
            final_baseline = self.evaluate(
                self.config.baseline_checkpoint,
                episodes,
                "baseline",
                seed=rung_seed,
            )
            baseline_by_rung[
                f"rung_{rung_index}_{episodes}ep_seed_{rung_seed}"
            ] = final_baseline
            rung_records = []
            for name in active_names:
                recipe = recipe_by_name[name]
                try:
                    if (
                        rung_index == 0
                        or training_steps[rung_index] > training_steps[rung_index - 1]
                    ):
                        trained[name] = self.train(
                            recipe,
                            steps=training_steps[rung_index],
                            resume=rung_index > 0,
                        )
                    elif training_steps[rung_index] < training_steps[rung_index - 1]:
                        raise ValueError("evaluation.training_steps must be nondecreasing")
                    metrics = self.evaluate(
                        trained[name],
                        episodes,
                        name,
                        seed=rung_seed,
                    )
                    rung_records.append(
                        self._record(
                            name=name,
                            status="completed",
                            rung=rung_index,
                            episodes=episodes,
                            training_steps=training_steps[rung_index],
                            score=metrics["score"],
                            metrics=metrics,
                            params=recipe.params,
                            checkpoint=str(trained[name]),
                        )
                    )
                except Exception as exc:
                    self._record(
                        name=name,
                        status="failed",
                        rung=rung_index,
                        episodes=episodes,
                        training_steps=training_steps[rung_index],
                        score=0.0,
                        metrics={},
                        params=recipe.params,
                        checkpoint=str(trained[name]),
                        error=str(exc),
                    )
                self.guard.assert_unchanged()
            rung_records.sort(
                key=lambda record: metric_order(record["metrics"]), reverse=True
            )
            active_names = [
                record["name"]
                for record in rung_records[: int(self.config.eval_defaults["promote_top_k"])]
            ]
            final_rung = rung_index

        finalists = [
            record
            for record in self.records
            if record.get("status") == "completed" and record.get("rung") == final_rung
        ]
        best = (
            max(finalists, key=lambda record: metric_order(record["metrics"]))
            if finalists
            else None
        )
        final_episodes = rungs[final_rung] if final_rung >= 0 else rungs[0]
        baseline = final_baseline
        decision = acceptance_decision(
            best.get("metrics") if best else None,
            baseline,
            self.config.eval_defaults,
        )
        accepted = bool(decision["accepted"])
        summary = {
            "schema_version": 1,
            "name": self.config.name,
            "created_at": datetime.now().astimezone().isoformat(),
            "dry_run": self.dry_run,
            "status": "completed",
            "evaluation_protocol": self.config.eval_defaults["protocol"],
            "ranking_eligible": self.config.eval_defaults["protocol"] == "frozen_final",
            "hardware": {
                "gpu_id": self.config.gpu_id,
                "host": platform.node(),
                "orchestrator_python": sys.version,
                "experiment_python": str(self.config.python),
            },
            "frozen_hashes": self.guard.before,
            "baseline_by_episodes": baseline_by_rung,
            "best": best,
            "accepted": accepted,
            "acceptance_decision": decision,
            "records": self.records,
        }
        final = self._write_summary(summary)
        self.events.append("run_finished", accepted=accepted, summary=str(final))
        return summary


def preflight_config(config: MVPConfig, *, require_dataset: bool) -> List[str]:
    """Validate an experiment without creating run artifacts."""
    problems: List[str] = []
    checks = {
        "repository": config.repo,
        "python": config.python,
        "baseline checkpoint": config.baseline_checkpoint,
        "ACT evaluator config": config.repo / "policy/act/deploy_policy.yml",
        "evaluator": config.repo / "scripts/eval_policy.py",
    }
    for label, path in checks.items():
        if not path.exists():
            problems.append(f"missing {label}: {path}")
    for relative in config.frozen_files:
        if not (config.repo / relative).exists():
            problems.append(f"missing frozen file: {relative}")
    checkpoint_files = [
        config.baseline_checkpoint / "config.json",
        config.baseline_checkpoint / "model.safetensors",
    ]
    for path in checkpoint_files:
        if not path.exists():
            problems.append(f"incomplete baseline checkpoint: {path}")
    if config.dataset_root is None:
        if require_dataset:
            problems.append("dataset_root is required for real training")
    elif not config.dataset_root.exists():
        problems.append(f"dataset_root does not exist: {config.dataset_root}")
    else:
        problems.extend(_validate_lerobot_dataset(config.dataset_root))
    if config.dataset_mixture_manifest is not None:
        if not config.dataset_mixture_manifest.is_file():
            problems.append(
                f"dataset mixture manifest does not exist: {config.dataset_mixture_manifest}"
            )
        else:
            try:
                mixture = json.loads(
                    config.dataset_mixture_manifest.read_text(encoding="utf-8")
                )
                roots = [Path(item["root"]) for item in mixture["datasets"]]
                if not roots:
                    problems.append("dataset mixture manifest is empty")
                for root in roots:
                    if not root.exists():
                        problems.append(f"mixture dataset does not exist: {root}")
                    else:
                        problems.extend(
                            f"mixture {root}: {problem}"
                            for problem in _validate_lerobot_dataset(root)
                        )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                problems.append(f"invalid dataset mixture manifest: {exc}")
    return problems


def _validate_lerobot_dataset(dataset_root: Path) -> List[str]:
    """Reject partial Hugging Face snapshots before an expensive GPU run."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return [f"dataset metadata is missing: {info_path}"]
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        total_episodes = int(info["total_episodes"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"invalid dataset metadata {info_path}: {exc}"]

    problems: List[str] = []
    parquet_count = sum(1 for _ in (dataset_root / "data").rglob("*.parquet"))
    if parquet_count != total_episodes:
        problems.append(
            f"incomplete dataset parquet files: {parquet_count}/{total_episodes}"
        )

    video_keys = [
        key
        for key, feature in dict(info.get("features") or {}).items()
        if feature.get("dtype") == "video"
    ]
    for video_key in video_keys:
        video_count = sum(
            1
            for _ in (dataset_root / "videos").glob(
                f"chunk-*/{video_key}/episode_*.mp4"
            )
        )
        if video_count != total_episodes:
            problems.append(
                f"incomplete dataset video {video_key}: "
                f"{video_count}/{total_episodes}"
            )
    return problems


def _print_preflight(config: MVPConfig, require_dataset: bool) -> int:
    problems = preflight_config(config, require_dataset=require_dataset)
    if problems:
        print("Preflight: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("Preflight: OK")
    print(f"- repo: {config.repo}")
    print(f"- baseline: {config.baseline_checkpoint}")
    print(f"- recipes: {len(generate_recipes(config))}")
    print(f"- rungs: {config.eval_defaults['rungs']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML experiment specification")
    parser.add_argument(
        "--episodes", type=int, help="override evaluation rungs with one episode count"
    )
    parser.add_argument(
        "--checkpoint", help="override baseline checkpoint for standalone evaluation"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight", action="store_true", help="validate paths without using the GPU"
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise the loop with deterministic fake metrics",
    )
    mode.add_argument(
        "--baseline-only",
        action="store_true",
        help="run the first real evaluation rung without training",
    )
    mode.add_argument(
        "--reference-only",
        action="store_true",
        help="train and evaluate the fixed train defaults before searching",
    )
    mode.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="run non-ranking appearance/camera/pose/clutter slices",
    )
    mode.add_argument(
        "--feedback-sweep-only",
        action="store_true",
        help="compare ACT n_action_steps with identical frozen weights",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="run real ACT training and simulation evaluation",
    )
    args = parser.parse_args(argv)

    config = MVPConfig.load(args.config)
    if args.episodes is not None:
        if args.episodes <= 0:
            parser.error("--episodes must be positive")
        config.evaluation["rungs"] = [args.episodes]
        config.evaluation["training_steps"] = [config.train_defaults["steps"]]
        if args.diagnostics_only or args.feedback_sweep_only:
            config.diagnostics["episodes"] = args.episodes
    if args.checkpoint:
        config.baseline_checkpoint = Path(os.path.abspath(Path(args.checkpoint).expanduser()))
    if args.preflight:
        return _print_preflight(config, require_dataset=False)

    runner = RoboSynMVPRunner(config, dry_run=args.dry_run)
    try:
        if (
            args.execute
            or args.baseline_only
            or args.reference_only
            or args.diagnostics_only
            or args.feedback_sweep_only
        ):
            with gpu_lock(config.gpu_id):
                if args.baseline_only:
                    summary = runner.run_baseline_only()
                elif args.reference_only:
                    summary = runner.run_reference_only()
                elif args.diagnostics_only:
                    summary = runner.run_diagnostics_only()
                elif args.feedback_sweep_only:
                    summary = runner.run_feedback_sweep_only()
                else:
                    summary = runner.run()
        else:
            summary = runner.run()
    except Exception as exc:
        runner.events.append("run_failed", error=str(exc))
        print(f"AutoResearch failed: {exc}", file=sys.stderr)
        return 1

    print(f"Run directory: {runner.run_dir}")
    if summary.get("diagnostics"):
        for profile, metrics in summary["diagnostics"].items():
            print(f"Diagnostic {profile}: {metrics['score']:.4f}")
        print("Accepted: False (diagnostic-only results are not ranking eligible)")
        return 0
    if summary.get("feedback_sweep"):
        for action_steps, metrics in summary["feedback_sweep"].items():
            print(f"Feedback n_action_steps={action_steps}: {metrics['score']:.4f}")
        print("Accepted: False (20-episode feedback sweep is screening only)")
        return 0
    best = summary.get("best")
    baseline_map = summary["baseline_by_episodes"]
    final_baseline = baseline_map[max(baseline_map)]
    print(f"Baseline: {final_baseline['score']:.4f}")
    if best:
        print(f"Best: {best['name']} {best['score']:.4f}")
    print(f"Accepted: {summary['accepted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
