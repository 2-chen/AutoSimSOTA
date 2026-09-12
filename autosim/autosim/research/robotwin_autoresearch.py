"""Resumable repository-in/repository-out AutoResearch loop for RoboTwin ACT."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from autosim.robosyn_mvp import gpu_lock

from .common import atomic_json, digest, immutable_json, now, read_json, run_command
from .robotwin_adapter import RoboTwinAdapter
from .robotwin_data_audit import audit_dataset
from .robotwin_decision import decide
from .robotwin_evidence import paired
from .robotwin_jobs import evaluate, runtime_environment, train
from .robotwin_mix_dataset import build_mix


class RoboTwinAutoResearch:
    """One bounded baseline -> evidence -> acquisition -> candidate loop."""

    def __init__(self, *, project_root: Path, repo: Path, output_root: Path, run_id: str,
                 task: str, controller: str, gpu: str, hours: float, train_seed: int,
                 training_epochs: int, evaluation_episodes: int,
                 allow_api_egress: bool):
        self.project_root, self.repo = project_root.absolute(), repo.absolute()
        if Path(run_id).name != run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError("run_id must be one safe path component")
        self.adapter = RoboTwinAdapter(self.repo)
        self.task = self.adapter.select_task(task)
        self.run_root = (output_root / "RoboTwin" / run_id).absolute()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.controller, self.gpu, self.train_seed = controller, str(gpu), int(train_seed)
        self.training_epochs, self.evaluation_episodes = int(training_epochs), int(evaluation_episodes)
        self.allow_api_egress = bool(allow_api_egress)
        self.deadline = time.monotonic() + float(hours) * 3600
        if controller not in {"api", "fixed", "random", "heuristic"}:
            raise ValueError("unknown RoboTwin controller")
        if training_epochs not in {2000, 6000} or evaluation_episodes < 1:
            raise ValueError("RoboTwin training_epochs must be 2000/6000 and evaluation episodes positive")

    def remaining(self) -> int:
        return max(0, int(self.deadline - time.monotonic()))

    def state(self, status: str, stage: str, **extra: Any) -> dict[str, Any]:
        value = {"schema_version": 1, "kind": "robotwin_act_autoresearch", "run_id": self.run_root.name,
                 "repo": str(self.repo), "benchmark": "RoboTwin", "task": self.task,
                 "controller": self.controller, "status": status, "stage": stage,
                 "updated_at": now(), "execution_complete": status == "completed",
                 "result_valid": bool(extra.pop("result_valid", False)),
                 "performance_improved": bool(extra.pop("performance_improved", False)),
                 "export_runtime_verified": bool(extra.pop("export_runtime_verified", False)), **extra}
        atomic_json(self.run_root / "run_state.json", value)
        return value

    def require_budget(self, seconds: int, stage: str) -> None:
        if self.remaining() < seconds:
            raise RuntimeError(f"insufficient remaining wall-clock budget for {stage}: "
                               f"need {seconds}s, have {self.remaining()}s")

    def initialize(self) -> dict[str, Any]:
        discovery = self.adapter.discover(self.task)
        capabilities = {"schema_version": 1, "benchmark": "RoboTwin", "task": self.task,
                        "records": [row.as_dict() for row in self.adapter.capabilities(self.task)]}
        atomic_json(self.run_root / "benchmark_discovery.json", discovery)
        atomic_json(self.run_root / "capabilities.json", capabilities)
        missing = [row["capability_id"] for row in capabilities["records"] if row["status"] != "verified"]
        if missing:
            raise RuntimeError(f"RoboTwin capabilities are not verified: {missing}")
        protocol = {"schema_version": 1, "task_signature": discovery["task_signature"],
                    "controller": self.controller, "train_seed": self.train_seed,
                    "training_epochs": self.training_epochs,
                    "evaluation_episodes": self.evaluation_episodes,
                    "evaluation_settings": ["demo_clean", "demo_randomized"],
                    "development_evaluation_seed": 0, "confirmation_evaluation_seed": 1,
                    "new_episode_budget": 10,
                    "judge_modified": False, "final_seed_feedback_allowed": False}
        immutable_json(self.run_root / "frozen_protocol.json", protocol)
        self.state("running", "baseline")
        return discovery

    def _baseline(self) -> Path:
        output = self.run_root / "checkpoint"
        receipt = self.run_root / "training_receipt.json"
        if receipt.is_file():
            checkpoint = Path(read_json(receipt)["checkpoint"])
            if checkpoint.is_file() and digest(checkpoint) == read_json(receipt)["checkpoint_sha256"]:
                return checkpoint.parent
            raise RuntimeError("baseline training receipt/checkpoint mismatch")
        self.require_budget(2 * 3600, "official ACT baseline training")
        dataset_key = f"demo_clean-{self.task}-aloha_agilex-joint"
        train(self.project_root, self.repo, dataset_key, output, gpu=self.gpu,
              epochs=self.training_epochs, seed=self.train_seed, lr=1e-5,
              chunk_size=50, kl_weight=10.0, timeout=min(self.remaining(), 3 * 3600))
        return output

    def _evaluate_pair(self, checkpoint: Path, label: str, purpose: str,
                       *, eval_seed: int = 0,
                       seed_banks: dict[str, list[int]] | None = None) -> dict[str, dict]:
        values = {}
        for setting in ("demo_clean", "demo_randomized"):
            target = self.run_root / "evaluations" / f"{label}_{setting}"
            evaluate(self.project_root, self.repo, checkpoint, self.task, setting, target,
                     episodes=self.evaluation_episodes, eval_seed=eval_seed, gpu=self.gpu,
                     max_seed_attempts=max(100, self.evaluation_episodes * 20),
                     timeout=min(self.remaining(), 3 * 3600), purpose=purpose,
                     seed_bank=None if seed_banks is None else seed_banks[setting])
            values[setting] = read_json(target / "evidence.json")
        return values

    def _proposal(self, baseline: dict[str, dict], discovery: dict[str, Any]) -> dict[str, Any]:
        target = self.run_root / "proposal.json"
        if target.is_file():
            return read_json(target)["proposal"]
        proposal, provenance = decide(self.controller, baseline, discovery["task_contract"],
                                      seed=self.train_seed, project_root=self.project_root,
                                      allow_api_egress=self.allow_api_egress)
        atomic_json(target, {"schema_version": 1, "proposal": proposal, "provenance": provenance})
        return proposal

    def _materialize_collection_config(self, proposal: dict[str, Any]) -> tuple[str, Path]:
        setting = proposal["collection"]["setting"]
        base = self.repo / "env_cfg/task_config" / f"{setting}.yml"
        config = yaml.safe_load(base.read_text(encoding="utf-8"))
        identity = hashlib.sha256((self.run_root.name + proposal["proposal_id"]).encode()).hexdigest()[:10]
        name = "autosim_" + re.sub(r"[^a-zA-Z0-9_]+", "_", identity)
        path = self.repo / "env_cfg/task_config" / f"{name}.yml"
        config.update(episode_num=10, max_seed_attempts=100,
                      seed_start=100_000_000 + self.train_seed * 10_000,
                      save_path="./data", use_seed=False, collect_data=True)
        serialized = yaml.safe_dump(config, sort_keys=False)
        if path.is_file() and path.read_text(encoding="utf-8") != serialized:
            raise ValueError("generated collection config identity collision")
        path.write_text(serialized, encoding="utf-8")
        immutable_json(self.run_root / "collection_protocol.json", {
            "base_setting": setting, "base_config_sha256": digest(base), "generated_config": str(path),
            "generated_config_sha256": digest(path), "episode_target": 10, "attempt_limit": 100,
            "seed_start": config["seed_start"], "domain_randomization": config["domain_randomization"]})
        return name, path

    def _acquire_and_mix(self, proposal: dict[str, Any]) -> str:
        config_name, _ = self._materialize_collection_config(proposal)
        raw = self.repo / "data" / config_name / self.task / "aloha_agilex"
        audit_path = self.run_root / "collection_audit.json"
        if not audit_path.is_file():
            self.require_budget(3600, "RoboTwin expert acquisition")
            env = runtime_environment(self.project_root, self.gpu)
            run_command(["bash", "collect_data.sh", self.task, config_name, self.gpu], cwd=self.repo,
                        env=env, output=self.run_root / "collection_process",
                        timeout=min(self.remaining(), 3 * 3600))
            result = audit_dataset(raw, decode_all_images=True)
            if result["episode_count"] != 10:
                raise RuntimeError("bounded RoboTwin collection did not yield exactly 10 audited episodes")
            atomic_json(audit_path, result)
        act = self.repo / "XPolicyLab/policy/ACT"
        extra = act / "processed_data" / config_name / self.task / "aloha_agilex-joint"
        if not (extra / "episode_9.hdf5").is_file():
            run_command(["bash", "process_data.sh", config_name, self.task, "aloha_agilex", "joint"],
                        cwd=act, env=runtime_environment(self.project_root, self.gpu),
                        output=self.run_root / "process_new_data", timeout=min(self.remaining(), 3600))
        official = act / "processed_data/demo_clean" / self.task / "aloha_agilex-joint"
        mixed = act / "processed_data/autosim_mix" / self.run_root.name / "aloha_agilex-joint"
        key = f"autosim_mix-{self.run_root.name}-aloha_agilex-joint"
        build_mix(official, extra, mixed, act / "TASK_CONFIGS.json", key)
        return key

    def _candidate(self, dataset_key: str, proposal: dict[str, Any]) -> Path:
        output = self.run_root / "candidate" / "checkpoint"
        receipt = output.parent / "training_receipt.json"
        if receipt.is_file():
            return output
        params = proposal["training"]
        self.require_budget(2 * 3600, "candidate ACT training")
        train(self.project_root, self.repo, dataset_key, output, gpu=self.gpu,
              epochs=int(params["epochs"]), seed=self.train_seed, lr=float(params["lr"]),
              chunk_size=int(params["chunk_size"]), kl_weight=float(params["kl_weight"]),
              timeout=min(self.remaining(), 3 * 3600))
        return output

    def _export(self, checkpoint: Path) -> tuple[Path, bool]:
        destination = self.run_root / "optimized_repo"
        if not destination.is_dir():
            def ignore(directory: str, names: list[str]) -> set[str]:
                relative = Path(directory).absolute().relative_to(self.repo)
                blocked = {".git", "__pycache__"}
                if relative == Path("."):
                    blocked |= {"data", "eval_result"}
                if relative == Path("XPolicyLab/policy/ACT"):
                    blocked |= {"processed_data", "checkpoints"}
                return {name for name in names if name in blocked or name.endswith(".pyc")}
            shutil.copytree(self.repo, destination, ignore=ignore, copy_function=os.link)
        deployed = destination / "XPolicyLab/policy/ACT/checkpoints/autoresearch_deployment"
        if not deployed.exists():
            shutil.copytree(checkpoint, deployed, copy_function=os.link)
        atomic_json(destination / "AUTORESEARCH_MANIFEST.json", {
            "schema_version": 1, "source_repo": str(self.repo), "task": self.task,
            "checkpoint": str(deployed.relative_to(destination)),
            "checkpoint_sha256": digest(deployed / "policy_last.ckpt"),
            "run_artifacts": str(self.run_root), "assets_hard_linked_from_project": True,
            "export_portability": "standalone inside this filesystem; materialize hard links before transfer",
            "source_task_signature": read_json(self.run_root / "benchmark_discovery.json")["task_signature"],
            "compatibility_files": {str(path.relative_to(self.repo)): digest(path) for path in (
                self.repo / "scripts/collect_data.py",
                self.repo / "scripts/eval_policy_xpolicylab.py",
                self.repo / "XPolicyLab/policy/ACT/setup_eval_policy_server.sh",
                self.repo / "XPolicyLab/utils/run_sim_env_client.sh",
            )}})
        smoke = self.run_root / "export_validation"
        try:
            evaluate(self.project_root, destination, deployed, self.task, "demo_clean", smoke,
                     episodes=1, eval_seed=97, gpu=self.gpu, max_seed_attempts=20,
                     timeout=min(self.remaining(), 1200), purpose="deployment_smoke")
            return destination, True
        except Exception as exc:
            atomic_json(smoke / "limitation.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            return destination, False

    def execute(self) -> dict[str, Any]:
        try:
            existing = self.run_root / "run_state.json"
            if existing.is_file() and read_json(existing).get("status") == "completed":
                return read_json(existing)
            discovery = self.initialize()
            with gpu_lock(self.gpu):
                baseline_checkpoint = self._baseline()
                self.state("running", "baseline_evaluation")
                baseline = self._evaluate_pair(baseline_checkpoint, "baseline", "development")
                proposal = self._proposal(baseline, discovery)
                if proposal["decision"] == "stop":
                    exported, verified = self._export(baseline_checkpoint)
                    return self.state("completed", "complete", result_valid=True,
                                      performance_improved=False, export_runtime_verified=verified,
                                      selected_checkpoint=str(baseline_checkpoint),
                                      optimized_repo=str(exported),
                                      hypothesis_supported=False,
                                      stop_reason="controller_selected_stop")
                self.state("running", "acquisition")
                dataset_key = self._acquire_and_mix(proposal)
                self.state("running", "candidate_training")
                candidate_checkpoint = self._candidate(dataset_key, proposal)
                self.state("running", "candidate_evaluation")
                development_banks = {
                    setting: [int(row["episode_seed"])
                              for row in baseline[setting]["metrics"]["episodes"]]
                    for setting in ("demo_clean", "demo_randomized")
                }
                candidate = self._evaluate_pair(candidate_checkpoint, "candidate", "development",
                                                seed_banks=development_banks)
                comparisons = {setting: paired(candidate[setting], baseline[setting])
                               for setting in ("demo_clean", "demo_randomized")}
                qualifies = all(row["success_delta"] > 0 for row in comparisons.values())
                confirmation = None
                if qualifies:
                    self.state("running", "confirmation_evaluation")
                    baseline_final = self._evaluate_pair(
                        baseline_checkpoint, "baseline_confirmation", "confirmation", eval_seed=1)
                    confirmation_banks = {
                        setting: [int(row["episode_seed"])
                                  for row in baseline_final[setting]["metrics"]["episodes"]]
                        for setting in ("demo_clean", "demo_randomized")
                    }
                    candidate_final = self._evaluate_pair(
                        candidate_checkpoint, "candidate_confirmation", "confirmation", eval_seed=1,
                        seed_banks=confirmation_banks)
                    confirmation = {setting: paired(candidate_final[setting], baseline_final[setting])
                                    for setting in ("demo_clean", "demo_randomized")}
                improved = bool(confirmation) and all(
                    row["success_delta"] > 0 for row in confirmation.values())
                atomic_json(self.run_root / "comparison.json", {
                    "development": comparisons, "qualified_for_confirmation": qualifies,
                    "confirmation": confirmation,
                    "selection_rule": "positive paired delta on both clean and randomized banks; repeat on untouched seed-1 confirmation"})
                deployment = candidate_checkpoint if improved else baseline_checkpoint
                exported, verified = self._export(deployment)
            return self.state("completed", "complete", result_valid=True,
                              performance_improved=improved, export_runtime_verified=verified,
                              selected_checkpoint=str(deployment), optimized_repo=str(exported),
                              comparisons={"development": comparisons, "confirmation": confirmation},
                              hypothesis_supported=improved)
        except Exception as exc:
            self.state("blocked", "failed", error=f"{type(exc).__name__}: {exc}")
            raise
