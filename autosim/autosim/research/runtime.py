"""Single-GPU RoboSyn adapter; artifacts belong to an explicit experiment."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .common import assert_frozen, atomic_json, digest, immutable_json, read_json, run_command
from .registry import TaskSpec


def retryable_evaluation_startup(directory: Path) -> bool:
    """Only native failures positively recorded before the first evaluation reset."""
    process, startup = directory / "process/process.json", directory / "startup.json"
    if not process.is_file() or not startup.is_file():
        return False
    record = read_json(process)
    native_exit = (record.get("status") == "failed"
                   and record.get("returncode") in {-11, -6})
    native_startup_hang = (record.get("status") == "interrupted"
                           and record.get("error") == "TimeoutExpired"
                           and record.get("returncode") in {-15, -9})
    return ((native_exit or native_startup_hang)
            and read_json(startup).get("phase") in {"environment_constructing", "environment_ready"}
            and not (directory / "initializations.jsonl").exists()
            and not (directory / "evaluation_metrics.json").exists())


@dataclass
class Runtime:
    workspace: Path
    output: Path
    gpu: str = "0"
    deadline: float | None = None
    repo_path: Path | None = None
    eval_repo_path: Path | None = None
    python_path: Path | None = None

    @property
    def platform_root(self) -> Path:
        """Support both a standalone AutoSimSOTA root and the legacy parent layout."""
        return (self.workspace if (self.workspace / "RoboSynChallenge").is_dir()
                else self.workspace / "AutoSimSOTA")

    @property
    def repo(self) -> Path:
        return (self.repo_path or self.platform_root / "RoboSynChallenge").absolute()

    @property
    def eval_repo(self) -> Path:
        return (self.eval_repo_path or self.platform_root / "RoboSynChallenge_eval_clean").absolute()

    @property
    def python(self) -> Path:
        # Do not resolve the venv executable symlink.
        return self.python_path or self.platform_root / ".venv/bin/python"

    def environment(self, repo: Path | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=self.gpu, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
                   AUTOSIM_EXPERT_SETTLE_STEPS="75",
                   EMBODICHAIN_DATA_ROOT=str(self.platform_root / "embodichain_data"),
                   XDG_CACHE_HOME="/tmp/robosyn_autoresearch_cache",
                   MPLCONFIGDIR="/tmp/robosyn_autoresearch_matplotlib", OMP_NUM_THREADS="4")
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in [
            self.platform_root / "autosim", repo or self.repo,
            self.platform_root / "EmbodiChain"])
        libraries = [self.platform_root / ".venv/lib/python3.10/site-packages/nvidia/cudnn/lib",
                     Path("/home/wbc/miniconda3/envs/robotwin5090/lib")]
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(p) for p in libraries] +
                                                  ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        return env

    def run(self, command: list[str], output: Path, timeout: int, *, evaluation=False) -> dict:
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining < 1:
                raise RuntimeError("task process-time budget exhausted")
            timeout = min(timeout, max(1, int(remaining)))
        if shutil.disk_usage(self.output).free < 12 * 1024**3:
            raise RuntimeError("less than 12 GiB free: refusing further data/checkpoint writes")
        repo = self.eval_repo if evaluation else self.repo
        return run_command(command, cwd=repo, env=self.environment(repo), output=output, timeout=timeout)

    def collect(self, spec: TaskSpec, destination: Path, *, episodes: int,
                master_seed: int, profile: str = "full_random", timeout=1800) -> Path:
        result = self.collect_bounded(
            spec, destination, attempt_budget=episodes * 12,
            target_episodes=episodes, master_seed=master_seed,
            profile=profile, timeout=timeout)
        if result["capability_state"] != "target_reached":
            raise RuntimeError(f"collector did not reach target: {destination / 'collection.json'}")
        if result["dataset_root"] is None:
            raise RuntimeError(f"collector returned no dataset: {destination / 'collection.json'}")
        return Path(result["dataset_root"])

    def collect_bounded(self, spec: TaskSpec, destination: Path, *,
                        attempt_budget: int, master_seed: int,
                        profile: str = "full_random",
                        target_episodes: int | None = None,
                        collection_mode: str = "expert",
                        correction_checkpoint: Path | None = None,
                        correction_prefix_min: int = 20,
                        correction_prefix_max: int = 120,
                        correction_replan_steps: int = 10,
                        correction_safe_return_steps: int = 20,
                        timeout=1800) -> dict:
        """Collect under an attempt budget and preserve legitimate partial yield.

        This is the research-facing contract.  Zero yield and partial yield are
        evidence, not process failures; callers decide whether they are useful.
        The legacy ``collect`` method remains strict for workflows that require a
        fixed number of accepted episodes.
        """
        if attempt_budget < 1:
            raise ValueError("attempt_budget must be positive")
        if target_episodes is None:
            target_episodes = attempt_budget
        if not 1 <= target_episodes <= attempt_budget:
            raise ValueError("target_episodes must be in [1, attempt_budget]")
        if collection_mode not in {"expert", "policy_correction"}:
            raise ValueError("unknown collection_mode")
        if collection_mode == "policy_correction":
            if not spec.correction_supported:
                raise ValueError(f"{spec.name} has no validated correction adapter")
            if correction_checkpoint is None:
                raise ValueError("policy_correction requires correction_checkpoint")
            if not 1 <= correction_prefix_min <= correction_prefix_max:
                raise ValueError("invalid correction prefix range")
        manifest = destination / "collection.json"
        module = "autosim.research.collection_worker"
        command = [str(self.python), "-m", module, "--gym_config", spec.gym_config,
                   "--action_config", spec.action_config, "--num_envs", "1", "--headless",
                   "--max_episodes", str(target_episodes), "--collection_seed", str(master_seed),
                   "--collection_profile", profile, "--collection_mode", collection_mode,
                   "--collection_manifest", str(manifest), "--dataset_save_path", str(destination / "data"),
                   "--collection_max_attempts", str(attempt_budget), "--collection_quiet"]
        if collection_mode == "policy_correction":
            command += ["--correction_checkpoint", str(correction_checkpoint),
                        "--correction_prefix_min", str(correction_prefix_min),
                        "--correction_prefix_max", str(correction_prefix_max),
                        "--correction_replan_steps", str(correction_replan_steps),
                        "--correction_safe_return_steps", str(correction_safe_return_steps)]
        contract_path = destination / "expert_contract.json"
        if not contract_path.exists():
            atomic_json(contract_path, {"task": spec.name, "adapter": spec.expert_adapter,
                        "collection_mode": collection_mode,
                        "parent_checkpoint": (str(correction_checkpoint) if correction_checkpoint else None),
                        "parent_model_sha256": (digest(correction_checkpoint / "model.safetensors")
                                                if correction_checkpoint else None),
                        "training_only": True, "success_function_modified": False,
                        "expert_success_cadence": "one_original_judge_query_after_each_actual_action_like_policy_adapter",
                        "terminal_hold_max_steps": 75,
                        "terminal_hold": "real recorded actions until original judge success, bounded below task horizon",
                        "retrospective_capture": (destination / "process/process.json").exists(),
                        "collector_source_sha256": digest(self.repo / "scripts/run_env.py"),
                        "adapter_source_sha256": digest(Path(__file__).with_name("collection_worker.py"))})
        self.run(command, destination / "process", timeout)
        data = read_json(manifest)
        budget_exhausted = (
            data.get("status") == "failed"
            and data.get("error") in {
                f"RuntimeError: Collection exceeded {attempt_budget} expert attempts.",
                f"RuntimeError: Correction collection exceeded {attempt_budget} attempts.",
            }
            and len(data.get("attempts", [])) == attempt_budget
        )
        if data["status"] != "completed" and not budget_exhausted:
            raise RuntimeError(f"collector process did not complete: {manifest}")
        if data["task"] != spec.name or len(data.get("dataset_paths", [])) > 1:
            raise RuntimeError(f"unexpected collection contract: {manifest}")
        from .diagnostic_audit import audit_collection_scene_evidence
        scene_audit = audit_collection_scene_evidence(
            manifest, destination / "scene_resets.jsonl")
        atomic_json(destination / "scene_evidence_audit.json", scene_audit)
        if not scene_audit["passed"]:
            raise RuntimeError(
                f"collection scene evidence audit failed: "
                f"{destination / 'scene_evidence_audit.json'}")
        accepted = len(data.get("successful_episode_seeds", []))
        capability_state = (
            "target_reached" if accepted >= target_episodes
            else "partial_yield" if accepted else "no_observed_yield"
        )
        result = {
            "schema_version": 1,
            "kind": "bounded_collection_result",
            "task": spec.name,
            "profile": profile,
            "collection_mode": collection_mode,
            "attempt_budget": attempt_budget,
            "target_episodes": target_episodes,
            "attempts_consumed": len(data.get("attempts", [])),
            "accepted_episodes": accepted,
            "capability_state": capability_state,
            "termination": "attempt_budget_exhausted" if budget_exhausted else "target_reached",
            "dataset_root": (data.get("dataset_paths") or [None])[0],
            "manifest": str(manifest.resolve()),
            "scene_evidence_audit": str(
                (destination / "scene_evidence_audit.json").resolve()),
            "zero_yield_does_not_prove_incapability": accepted == 0,
        }
        atomic_json(destination / "bounded_collection_result.json", result)
        return result

    def prepare_data(self, spec: TaskSpec, root: Path, output: Path, *,
                     full_video_decode: bool = False) -> dict:
        from autosim.robosyn_data import validate_dataset

        info = read_json(root / "meta/info.json")
        if info["codebase_version"] == "v3.0":
            # The official converter renames the source, keeping a v3 backup.
            # Refuse any existing intermediate directory rather than overwriting it.
            for suffix in ("_v3.0", "_v2.1"):
                if root.with_name(root.name + suffix).exists():
                    raise FileExistsError(f"conversion intermediate exists: {root}{suffix}")
            self.run([str(self.python), "scripts/convert_lerobot3.0_to_2.1.py",
                      "--repo-id", root.name, "--root", str(root.parent)], output / "conversion", 1800)
        aliases = {c: (f"observation.images.{c}", f"{c}.color") for c in spec.cameras}
        # A recorded frame is not necessarily a simulator control step. Do not
        # enforce the evaluation step limit on training recording lengths.
        result = validate_dataset(root, state_dim=spec.state_dim, action_dim=spec.action_dim,
                                  max_episode_frames=None, camera_aliases=aliases,
                                  full_video_decode=full_video_decode)
        result["task_signature"] = spec.signature
        result["recorded_fps"] = read_json(root / "meta/info.json").get("fps")
        result["info_sha256"] = digest(root / "meta/info.json")
        atomic_json(output / "data_audit.json", result)
        if not result["passed"]:
            raise RuntimeError(f"data audit failed: {result['errors']}")
        from .data_version import fingerprint_dataset
        result["content_version"] = fingerprint_dataset(root, self.output / "data_versions")
        atomic_json(output / "data_audit.json", result)
        return result

    def train(self, spec: TaskSpec, root: Path, output: Path, *, steps: int,
              params: dict | None = None, mixture: Path | None = None,
              seed: int = 1000, resume: bool = False,
              pretrained: Path | None = None) -> Path:
        params = dict(params or {})
        allowed = {"batch_size", "chunk_size", "n_action_steps", "optimizer_lr", "kl_weight",
                   "dropout", "action_loss_profile", "image_augmentation_profile", "num_workers"}
        if set(params) - allowed:
            raise ValueError(f"unapproved training parameters: {set(params) - allowed}")
        defaults = dict(batch_size=32, chunk_size=50, n_action_steps=50, optimizer_lr=1e-5,
                        kl_weight=10.0, dropout=0.1, action_loss_profile="legacy_mask_mean",
                        image_augmentation_profile="none", num_workers=4)
        defaults.update(params)
        if not 1 <= int(defaults["n_action_steps"]) <= int(defaults["chunk_size"]):
            raise ValueError("action execution horizon must be in [1, chunk_size]")
        if resume and pretrained is not None:
            raise ValueError("resume and pretrained initialization are mutually exclusive")
        train_dir = output / "train"
        from .data_version import training_versions
        versions = training_versions(root, mixture, self.output / "data_versions")
        immutable_json(output / "observation_contract.json", spec.as_dict())
        command = [str(self.python), "policy/act/scripts/train.py", "--dataset-root", str(root),
                   "--observation-contract", str(output / "observation_contract.json"),
                   "--output-dir", str(train_dir), "--device", "cuda", "--steps", str(steps),
                   "--seed", str(seed), "--save-freq", str(min(20000, steps)), "--log-freq", "100"]
        for key, value in defaults.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        if mixture:
            command += ["--dataset-mixture-manifest", str(mixture),
                        "--data-pipeline-audit", str(output / "training_data_audit.json"),
                        "--sampling-exposure-audit", str(output / "training_exposure_audit.json")]
        if pretrained is not None:
            command += ["--pretrained-policy", str(pretrained)]
        if resume:
            command += ["--resume"]
        recipe = {"task_signature": spec.signature,
                    "steps": steps, "save_freq": min(20000, steps),
                    "seed": seed, "params": defaults, "dataset": str(root),
                    "training_data_content_ids": {row["root"]: row["content_id"] for row in versions},
                    "dataset_info_sha256": digest(root / "meta/info.json"),
                    "mixture_sha256": digest(mixture) if mixture else None,
                    "pretrained_checkpoint": str(pretrained) if pretrained else None,
                    "pretrained_model_sha256": (
                        digest(pretrained / "model.safetensors") if pretrained else None),
                    "pretrained_config_sha256": (
                        digest(pretrained / "config.json") if pretrained else None),
                    "trainer_sha256": digest(self.repo / "policy/act/scripts/train.py"),
                    "training_exposure_audit": (
                        str(output / "training_exposure_audit.json") if mixture else None
                    ),
                    "mixture": str(mixture) if mixture else None, "resume": resume}
        recipe_path = output / f"recipe_{steps}.json"
        if recipe_path.exists() and read_json(recipe_path) != recipe:
            raise RuntimeError("training inputs changed: use a new experiment directory")
        atomic_json(recipe_path, recipe)
        self.run(command, output / f"process_train_{steps}", max(1800, int(steps * 1.5)))
        checkpoint = train_dir / "checkpoints" / f"{steps:06d}" / "pretrained_model"
        if not (checkpoint / "model.safetensors").is_file():
            raise RuntimeError(f"training returned without checkpoint: {checkpoint}")
        return checkpoint

    def evaluate(self, spec: TaskSpec, checkpoint: Path, output: Path, *, episodes: int,
                 master_seed: int, purpose: str = "development", policy: str = "act",
                 startup_attempts: int = 3, smoke_timeout: int = 300) -> dict:
        if purpose not in {"smoke", "development", "selection_validation",
                           "final_confirmation", "confirmation", "final"}:
            raise ValueError(f"invalid evaluation purpose: {purpose}")
        if not 1 <= startup_attempts <= 3:
            raise ValueError("startup_attempts must be in [1, 3]")
        if smoke_timeout < 30:
            raise ValueError("smoke_timeout must be at least 30 seconds")
        immutable_json(output / "evaluation_request.json", {
            "task_signature": spec.signature, "checkpoint": str(checkpoint),
            "weight_sha256": digest(checkpoint / "model.safetensors"),
            "model_config_sha256": digest(checkpoint / "config.json"),
            "episodes": episodes, "master_seed": master_seed, "purpose": purpose, "policy": policy})
        first_attempt = 1
        published = output / "evaluation_metrics.json"
        if published.is_file():
            previous = read_json(published)
            if previous.get("execution_mode") == "real_simulation":
                first_attempt = int(previous.get("startup_attempt_count", 1))
                if first_attempt not in set(range(1, startup_attempts + 1)):
                    raise ValueError("invalid startup attempt provenance")
        else:
            prior_process = output / "process/process.json"
            if prior_process.is_file():
                prior = read_json(prior_process)
                # A launcher path failure before Popen produced a historical
                # ``running`` record without a pid. Preserve it and continue in
                # a separate attempt directory; no simulator reset occurred.
                never_started = (prior.get("status") in {"running", "failed_to_start"}
                                 and "pid" not in prior
                                 and not (output / "initializations.jsonl").exists()
                                 and not (output / "startup.json").exists())
                retryable_startup = retryable_evaluation_startup(output)
                if not never_started and not retryable_startup:
                    raise RuntimeError(f"unclassified prior evaluation attempt: {prior_process}")
                first_attempt = 2
                atomic_json(output / "startup_attempt_1_recovery.json", {
                    "reason": ("launcher_failed_before_process_creation" if never_started
                               else "native_startup_failure_before_first_reset"),
                    "prior_process_status": prior.get("status"),
                    "completed_evaluation_episodes": 0,
                    "same_checkpoint_and_seed_bank": True,
                    "next_attempt": 2,
                })
        for attempt in range(first_attempt, startup_attempts + 1):
            destination = output if attempt == 1 else output / f"startup_attempt_{attempt}"
            command = [str(self.python), "-m", "autosim.research.evaluation", "--task", spec.name,
                       "--checkpoint", str(checkpoint), "--output", str(destination), "--episodes", str(episodes),
                       "--seed", str(master_seed), "--purpose", purpose, "--policy", policy]
            try:
                timeout = smoke_timeout if purpose == "smoke" else max(1800, episodes * 60)
                self.run(command, destination / "process", timeout, evaluation=True)
            except RuntimeError:
                if attempt == startup_attempts or not retryable_evaluation_startup(destination):
                    raise
                assert_frozen(read_json(destination / "protocol.json")["frozen_files"])
                atomic_json(destination / "startup_retry_decision.json", {
                    "reason": "native_crash_before_first_evaluation_reset", "next_attempt": attempt + 1,
                    "completed_evaluation_episodes": 0, "same_checkpoint_and_seed_bank": True})
                continue
            assert_frozen(read_json(destination / "protocol.json")["frozen_files"])
            data = read_json(destination / "evaluation_metrics.json")
            data["artifact_directory"] = str(destination)
            data["startup_attempt_count"] = attempt
            break
        if data.get("execution_mode") != "real_simulation" or data.get("purpose") != purpose:
            raise RuntimeError("evaluation worker did not certify a real result under the requested purpose")
        rows = data["episodes"]
        if len(rows) != episodes or len({r["episode_seed"] for r in rows}) != episodes:
            raise RuntimeError("evaluation has missing/duplicate episode seeds")
        if data["config"]["task"] != spec.name or data["config"]["timeout_action_steps"] != spec.max_episode_steps:
            raise RuntimeError("evaluation task/horizon differs from contract")
        atomic_json(output / "evaluation_metrics.json", data)
        return data
