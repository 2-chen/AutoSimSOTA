"""ACT adapter for RoboTwin.

The adapter exposes RoboTwin's ACT training script through the generic
TaskAdapter interface so AutoSim can optimize model hyperparameters and, when
requested by the closed-loop optimizer, source-level CODE/ALGO changes.
"""

import hashlib
import importlib.util
import json
import math
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from autosim.adapters.base import Candidate, TaskAdapter, ParamDef, EvalResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBOTWIN = os.environ.get("ROBOTWIN_HOME", str(PROJECT_ROOT / "RoboTwin"))


class ACTAdapter(TaskAdapter):
    """ACT (Action Chunking Transformer) VLA 模型适配器"""

    def __init__(self, repo_path: str = ROBOTWIN,
                 task_name: str = "beat_block_hammer",
                 data_name: Optional[str] = None,
                 task_config: str = "demo_randomized",
                 expert_data_num: int = 15,
                 epochs: int = 100,
                 seed: int = 42,
                 batch_size: int = 4,
                 timeout: int = 1800,
                 gpu_id: Optional[str] = None,
                 output_root: Optional[str] = None,
                 dry_run: bool = False,
                 eval_metric: str = "success_rate",
                 eval_episodes: int = 10,
                 eval_seed: int = 0,
                 instruction_type: str = "unseen",
                 temporal_agg: bool = True,
                 device: str = "cuda:0",
                 python_executable: Optional[str] = None):
        super().__init__(repo_path)
        self.repo = Path(repo_path).resolve()
        self.act_dir = self.repo / "policy" / "ACT"
        self.task_name = task_name
        self.task_config = task_config
        self.expert_data_num = int(expert_data_num)
        self.data_name = data_name or f"sim-{task_name}-{task_config}-{self.expert_data_num}"
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.timeout = int(timeout)
        self.gpu_id = gpu_id
        self.dry_run = bool(dry_run or os.environ.get("AUTOSIM_DRY_RUN") == "1")
        self.eval_metric = eval_metric
        self.eval_episodes = int(eval_episodes)
        self.eval_seed = int(eval_seed)
        self.instruction_type = instruction_type
        self.temporal_agg = bool(temporal_agg)
        self.device = device
        self.python_executable = python_executable or sys.executable
        self.output_root = Path(output_root) if output_root else (
            self.act_dir / "act_ckpt" / f"act-{task_name}" / "autosim"
        )

    def get_param_space(self) -> Dict[str, ParamDef]:
        return {
            "kl_weight": ParamDef("kl_weight", 10, (2, 30),
                                  "CVAE KL divergence weight", "int"),
            "lr": ParamDef("lr", 1e-5, (5e-6, 5e-4),
                           "AdamW learning rate", "float"),
            "chunk_size": ParamDef("chunk_size", 50, (20, 80),
                                   "Action chunk size (horizon)", "int"),
        }

    def _load_sim_task_configs(self) -> Dict:
        config_path = self.act_dir / "SIM_TASK_CONFIGS.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing ACT config: {config_path}")
        return json.loads(config_path.read_text())

    def _dataset_dir(self) -> Path:
        configs = self._load_sim_task_configs()
        if self.data_name not in configs:
            known = ", ".join(sorted(configs)[:8])
            raise KeyError(
                f"Dataset {self.data_name!r} is not registered in SIM_TASK_CONFIGS.json. "
                f"Known examples: {known}"
        )
        return self.act_dir / configs[self.data_name]["dataset_dir"]

    def _missing_python_deps(self) -> List[str]:
        deps = ["torch", "einops", "h5py", "tqdm", "matplotlib"]
        if Path(self.python_executable).resolve() == Path(sys.executable).resolve():
            return [pkg for pkg in deps if importlib.util.find_spec(pkg) is None]
        code = "import " + ", ".join(deps)
        try:
            result = subprocess.run(
                [self.python_executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            return deps
        if result.returncode == 0:
            return []
        missing = []
        text = result.stderr + result.stdout
        for dep in deps:
            if dep in text:
                missing.append(dep)
        return missing or deps

    def _validate_real_run(self) -> Optional[str]:
        if not self.act_dir.exists():
            return f"ACT directory not found: {self.act_dir}"
        missing = self._missing_python_deps()
        if missing:
            return (
                f"Python executable {self.python_executable} is missing ACT dependencies: "
                f"{', '.join(missing)}. Activate RoboTwin/policy/ACT/conda_env.yaml "
                "or install the ACT requirements before running without --dry-run."
            )
        try:
            dataset_dir = self._dataset_dir()
        except Exception as exc:
            return str(exc)
        if not dataset_dir.exists():
            return (
                f"Processed ACT dataset not found: {dataset_dir}. "
                f"Run RoboTwin/policy/ACT/process_data.sh for {self.task_name} "
                f"{self.task_config} {self.expert_data_num} first, or pass --dry-run."
            )
        first_episode = dataset_dir / "episode_0.hdf5"
        if not first_episode.exists():
            return f"Dataset exists but episode_0.hdf5 is missing: {dataset_dir}"
        return None

    def _coerce_params(self, params: Optional[Dict]) -> Dict:
        params = params or {}
        return {
            "kl_weight": int(float(params.get("kl_weight", 10))),
            "lr": float(params.get("lr", 1e-5)),
            "chunk_size": int(float(params.get("chunk_size", 50))),
        }

    def _safe_name(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "candidate"

    def _ckpt_dir(self, params: Dict, run_name: Optional[str] = None) -> Path:
        if run_name:
            return self.output_root / self._safe_name(run_name)
        key = json.dumps(params, sort_keys=True)
        digest = hashlib.sha1(key.encode()).hexdigest()[:8]
        lr = f"{params['lr']:.1e}".replace("+", "")
        name = f"kl{params['kl_weight']}_lr{lr}_chunk{params['chunk_size']}_{digest}"
        return self.output_root / name

    def _build_train_command(self, params: Dict, ckpt_dir: Path) -> list:
        return [
            self.python_executable, str(self.act_dir / "imitate_episodes.py"),
            "--task_name", self.data_name,
            "--ckpt_dir", str(ckpt_dir),
            "--policy_class", "ACT",
            "--kl_weight", str(params["kl_weight"]),
            "--chunk_size", str(params["chunk_size"]),
            "--hidden_dim", "512",
            "--batch_size", str(self.batch_size),
            "--dim_feedforward", "3200",
            "--num_epochs", str(self.epochs),
            "--lr", str(params["lr"]),
            "--save_freq", str(self.epochs),
            "--state_dim", "14",
            "--seed", str(self.seed),
        ]

    def _dry_run_score(self, params: Dict) -> float:
        """Deterministic proxy surface for validating the optimization loop."""
        kl = params["kl_weight"]
        chunk = params["chunk_size"]
        log_lr = math.log10(params["lr"])
        loss = (
            0.38
            + 0.0018 * (kl - 8) ** 2
            + 0.00035 * (chunk - 48) ** 2
            + 0.18 * (log_lr + 4.35) ** 2
        )
        if self.eval_metric == "success_rate":
            return round(max(0.0, min(1.0, 1.0 - loss)), 6)
        return round(loss, 6)

    def _failure_score(self) -> float:
        return 0.0 if self.eval_metric == "success_rate" else 999.0

    def _build_eval_command(self, ckpt_dir: Path) -> list:
        cmd = [
            self.python_executable, "-m", "autosim.adapters.act_success_eval",
            "--repo", str(self.repo),
            "--task-name", self.task_name,
            "--task-config", self.task_config,
            "--ckpt-setting", ckpt_dir.name,
            "--ckpt-dir", str(ckpt_dir),
            "--seed", str(self.eval_seed),
            "--test-num", str(self.eval_episodes),
            "--instruction-type", self.instruction_type,
            "--device", self.device,
        ]
        if self.temporal_agg:
            cmd.append("--temporal-agg")
        return cmd

    def _write_candidate_manifest(
        self,
        ckpt_dir: Path,
        params: Dict,
        status: str,
        train_command: Optional[list] = None,
        eval_command: Optional[list] = None,
        candidate: Optional[Candidate] = None,
        extra: Optional[Dict] = None,
    ) -> Path:
        manifest = {
            "status": status,
            "adapter": self.__class__.__name__,
            "repo": str(self.repo),
            "task_name": self.task_name,
            "task_config": self.task_config,
            "data_name": self.data_name,
            "eval_metric": self.eval_metric,
            "eval_episodes": self.eval_episodes,
            "eval_seed": self.eval_seed,
            "instruction_type": self.instruction_type,
            "temporal_agg": self.temporal_agg,
            "device": self.device,
            "python_executable": self.python_executable,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "params": params,
            "ckpt_dir": str(ckpt_dir),
            "train_command": train_command,
            "eval_command": eval_command,
            "candidate": {
                "name": candidate.name,
                "kind": candidate.kind,
                "payload": candidate.payload,
                "description": candidate.description,
            } if candidate else None,
        }
        if extra:
            manifest.update(extra)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "autosim_candidate_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        return path

    def _write_json_record(self, path: Path, payload: Dict) -> Optional[Path]:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            return path
        except OSError:
            fallback = self.output_root / "_eval_records" / path.parent.name / path.name
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            return fallback

    def _eval_env(self) -> Dict:
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        env.setdefault("MPLCONFIGDIR", "/tmp/autosim_matplotlib")
        env.setdefault("XDG_CACHE_HOME", "/tmp/autosim_cache")
        py_path = Path(self.python_executable)
        if py_path.name == "python" and py_path.parent.name == "bin":
            conda_lib = py_path.parent.parent / "lib"
            if conda_lib.exists():
                current_ld = env.get("LD_LIBRARY_PATH", "")
                lib_str = str(conda_lib)
                if lib_str not in current_ld.split(os.pathsep):
                    env["LD_LIBRARY_PATH"] = lib_str + (os.pathsep + current_ld if current_ld else "")
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        return env

    def _run_success_eval(
        self,
        ckpt_dir: Path,
        env: Dict,
        record_dir: Optional[Path] = None,
    ) -> EvalResult:
        cmd = self._build_eval_command(ckpt_dir)
        py_path = env.get("PYTHONPATH", "")
        autosim_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = autosim_root + (os.pathsep + py_path if py_path else "")
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
                cwd=self.repo, env=env
            )
        except subprocess.TimeoutExpired:
            return EvalResult(
                score=0.0, success=False,
                info=f"Task success evaluation timed out after {self.timeout}s"
            )

        match = re.search(r"AUTOSIM_EVAL_RESULT=(\{.*\})", r.stdout)
        payload = {}
        if match:
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                payload = {}
        success_rate = float(payload.get("success_rate", 0.0))
        record = {
            "command": cmd,
            "returncode": r.returncode,
            "success_rate": success_rate,
            "payload": payload,
            "stdout_tail": r.stdout[-4000:],
            "stderr_tail": r.stderr[-4000:],
        }
        record_path = self._write_json_record(
            (record_dir or ckpt_dir) / "autosim_success_eval.json",
            record,
        )
        if payload:
            return EvalResult(
                score=success_rate,
                success=(r.returncode == 0),
                metrics={
                    "success_rate": success_rate,
                    "success_count": payload.get("success_count"),
                    "eval_episodes": payload.get("test_num"),
                    "ckpt_dir": str(ckpt_dir),
                    "returncode": r.returncode,
                    "eval_record": str(record_path),
                },
                info="task success evaluation complete",
            )
        return EvalResult(
            score=0.0,
            success=False,
            metrics={"returncode": r.returncode, "ckpt_dir": str(ckpt_dir), "eval_record": str(record_path)},
            info=f"Could not parse success evaluation. stderr: {r.stderr[-500:]}"
        )

    def candidate_from_checkpoint(
        self,
        name: str,
        ckpt_dir: Optional[str] = None,
        description: str = "",
    ) -> Candidate:
        """Create a system-level candidate from an existing ACT checkpoint dir."""
        path = Path(ckpt_dir) if ckpt_dir else (
            self.act_dir / "act_ckpt" / f"act-{self.task_name}" / name
        )
        return Candidate(
            name=name,
            kind="checkpoint",
            payload={"ckpt_dir": str(path), "ckpt_setting": name},
            description=description,
        )

    def discover_checkpoint_candidates(
        self,
        limit: Optional[int] = None,
        include_prefix: Optional[str] = None,
        exclude_names: Optional[set] = None,
    ) -> List[Candidate]:
        """Discover ACT checkpoint directories that can be evaluated.

        This is a system-level candidate source: training/generation may happen
        elsewhere, while AutoSim repeatedly scans and evaluates what appears.
        """
        root = self.act_dir / "act_ckpt" / f"act-{self.task_name}"
        if not root.exists():
            return []
        exclude_names = exclude_names or set()
        candidates = []
        dirs = [p for p in root.iterdir() if p.is_dir() and (p / "policy_last.ckpt").exists()]
        dirs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        for path in dirs:
            if path.name in exclude_names:
                continue
            if include_prefix and not path.name.startswith(include_prefix):
                continue
            candidates.append(
                self.candidate_from_checkpoint(
                    path.name,
                    ckpt_dir=str(path),
                    description="discovered ACT checkpoint candidate",
                )
            )
            if limit is not None and len(candidates) >= limit:
                break
        return candidates

    def evaluate_candidate(self, candidate: Candidate) -> EvalResult:
        """Evaluate a system-level ACT candidate.

        checkpoint candidates are evaluated directly. train_recipe/params
        candidates go through evaluate(params), which trains in real mode and
        uses the deterministic proxy in dry-run mode.
        """
        if candidate.kind in ("patch_train_recipe", "algo_recipe", "code_recipe"):
            return self._evaluate_patch_candidate(candidate)

        if candidate.kind in ("params", "train_recipe"):
            params = candidate.payload.get("params", candidate.payload)
            result = self.evaluate(params, run_name=candidate.name, candidate=candidate)
            result.metrics.setdefault("candidate", candidate.name)
            result.metrics.setdefault("candidate_kind", candidate.kind)
            result.metrics.setdefault("recipe_type", candidate.payload.get("recipe_type"))
            return result

        if candidate.kind != "checkpoint":
            raise TypeError(f"ACTAdapter does not support candidate kind {candidate.kind!r}")

        ckpt_dir_value = candidate.payload.get("ckpt_dir")
        ckpt_setting = candidate.payload.get("ckpt_setting", candidate.name)
        ckpt_dir = Path(ckpt_dir_value) if ckpt_dir_value else (
            self.act_dir / "act_ckpt" / f"act-{self.task_name}" / ckpt_setting
        )
        if self.dry_run:
            score = float(candidate.payload.get("score", self._dry_run_score(self._coerce_params({}))))
            return EvalResult(
                score=score,
                success=True,
                metrics={
                    "mode": "dry_run",
                    "candidate": candidate.name,
                    self.eval_metric: score,
                    "ckpt_dir": str(ckpt_dir),
                },
                info="dry-run checkpoint candidate evaluation",
            )
        if not ckpt_dir.exists():
            return EvalResult(
                score=0.0 if self.eval_metric == "success_rate" else 999.0,
                success=False,
                metrics={"candidate": candidate.name, "ckpt_dir": str(ckpt_dir)},
                info=f"checkpoint directory not found: {ckpt_dir}",
            )
        if not (ckpt_dir / "policy_last.ckpt").exists():
            return EvalResult(
                score=0.0 if self.eval_metric == "success_rate" else 999.0,
                success=False,
                metrics={"candidate": candidate.name, "ckpt_dir": str(ckpt_dir)},
                info=f"policy_last.ckpt not found in {ckpt_dir}",
            )
        record_dir = self.output_root / "_eval_records" / self._safe_name(candidate.name)
        result = self._run_success_eval(ckpt_dir, self._eval_env(), record_dir=record_dir)
        result.metrics.setdefault("candidate", candidate.name)
        result.metrics.setdefault("ckpt_dir", str(ckpt_dir))
        return result

    def _apply_candidate_patches(self, candidate: Candidate) -> tuple[bool, str, List[Path]]:
        changed = []
        for patch in candidate.payload.get("patches", []):
            file_path = patch.get("file", "")
            old_code = patch.get("old_code", "")
            new_code = patch.get("new_code", "")
            full_path = self._resolve_change_path(file_path)
            if full_path is None:
                return False, f"patch target not found: {file_path}", changed
            applied = self.apply_change(file_path, old_code, new_code)
            if not applied:
                return False, f"old_code not found in {file_path}", changed
            changed.append(full_path)
        return True, "", changed

    def _compile_patch_files(self, files: List[Path]) -> Optional[str]:
        for path in files:
            if path.suffix != ".py":
                continue
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                return f"patched file failed to compile: {path}: {exc.msg}"
        return None

    def _evaluate_patch_candidate(self, candidate: Candidate) -> EvalResult:
        """Train/evaluate a temporary ALGO/CODE patch candidate, then rollback."""
        params = candidate.payload.get("params", {})
        coerced = self._coerce_params(params)
        ckpt_dir = self._ckpt_dir(coerced, run_name=candidate.name)
        ok, error, changed_files = self._apply_candidate_patches(candidate)
        if not ok:
            self.revert_all()
            self._write_candidate_manifest(
                ckpt_dir=ckpt_dir,
                params=coerced,
                status="patch_apply_failed",
                train_command=self._build_train_command(coerced, ckpt_dir),
                eval_command=self._build_eval_command(ckpt_dir),
                candidate=candidate,
                extra={"error": error, "patches": candidate.payload.get("patches", [])},
            )
            return EvalResult(
                score=self._failure_score(),
                success=False,
                metrics={
                    "candidate": candidate.name,
                    "candidate_kind": candidate.kind,
                    "ckpt_dir": str(ckpt_dir),
                    "manifest": str(ckpt_dir / "autosim_candidate_manifest.json"),
                    "patch_error": error,
                },
                info=error,
            )

        compile_error = self._compile_patch_files(changed_files)
        if compile_error:
            self.revert_all()
            self._write_candidate_manifest(
                ckpt_dir=ckpt_dir,
                params=coerced,
                status="patch_compile_failed",
                train_command=self._build_train_command(coerced, ckpt_dir),
                eval_command=self._build_eval_command(ckpt_dir),
                candidate=candidate,
                extra={"error": compile_error, "patches": candidate.payload.get("patches", [])},
            )
            return EvalResult(
                score=self._failure_score(),
                success=False,
                metrics={
                    "candidate": candidate.name,
                    "candidate_kind": candidate.kind,
                    "ckpt_dir": str(ckpt_dir),
                    "manifest": str(ckpt_dir / "autosim_candidate_manifest.json"),
                    "patch_error": compile_error,
                },
                info=compile_error,
            )

        try:
            result = self.evaluate(coerced, run_name=candidate.name, candidate=candidate)
            result.metrics.setdefault("candidate", candidate.name)
            result.metrics.setdefault("candidate_kind", candidate.kind)
            result.metrics.setdefault("recipe_type", candidate.payload.get("recipe_type"))
            result.metrics.setdefault("patches_applied", len(changed_files))
            result.metrics.setdefault("patched_files", [str(path) for path in changed_files])
            return result
        finally:
            self.revert_all()

    def evaluate(
        self,
        params: Dict = None,
        run_name: Optional[str] = None,
        candidate: Optional[Candidate] = None,
    ) -> EvalResult:
        """Train ACT with given params, return val_loss."""
        params = self._coerce_params(params)
        ckpt_dir = self._ckpt_dir(params, run_name=run_name)

        if self.dry_run:
            score = self._dry_run_score(params)
            self._write_candidate_manifest(
                ckpt_dir=ckpt_dir,
                params=params,
                status="dry_run_complete",
                train_command=self._build_train_command(params, ckpt_dir),
                eval_command=self._build_eval_command(ckpt_dir),
                candidate=candidate,
                extra={"score": score},
            )
            return EvalResult(
                score=score,
                success=True,
                metrics={
                    "mode": "dry_run",
                    self.eval_metric: score,
                    "epochs_trained": self.epochs,
                    "dataset": self.data_name,
                    "params": params,
                    "ckpt_dir": str(ckpt_dir),
                    "manifest": str(ckpt_dir / "autosim_candidate_manifest.json"),
                },
                info=f"deterministic ACT proxy {self.eval_metric}",
            )

        validation_error = self._validate_real_run()
        if validation_error:
            self._write_candidate_manifest(
                ckpt_dir=ckpt_dir,
                params=params,
                status="blocked_validation",
                train_command=self._build_train_command(params, ckpt_dir),
                eval_command=self._build_eval_command(ckpt_dir),
                candidate=candidate,
                extra={"error": validation_error},
            )
            return EvalResult(
                score=self._failure_score(),
                success=False,
                metrics={"ckpt_dir": str(ckpt_dir), "manifest": str(ckpt_dir / "autosim_candidate_manifest.json")},
                info=validation_error,
            )

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_train_command(params, ckpt_dir)
        self._write_candidate_manifest(
            ckpt_dir=ckpt_dir,
            params=params,
            status="training_started",
            train_command=cmd,
            eval_command=self._build_eval_command(ckpt_dir),
            candidate=candidate,
        )

        env = self._eval_env()

        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
                cwd=self.act_dir, env=env
            )
            v_losses = [float(m.group(1)) for m in
                       re.finditer(r'Val loss:\s+([\d.]+)', r.stdout)]
            t_losses = [float(m.group(1)) for m in
                       re.finditer(r'Train loss:\s*([\d.]+)', r.stdout)]
            best_match = re.search(r'Best ckpt, val loss\s+([\d.]+)', r.stdout)
            best_loss = float(best_match.group(1)) if best_match else (
                min(v_losses) if v_losses else 999.0
            )

            record = {
                "params": params,
                "dataset": self.data_name,
                "command": cmd,
                "returncode": r.returncode,
                "min_val_loss": best_loss,
                "epochs_trained": len(t_losses),
                "stdout_tail": r.stdout[-4000:],
                "stderr_tail": r.stderr[-4000:],
            }
            (ckpt_dir / "autosim_eval.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False)
            )
            self._write_candidate_manifest(
                ckpt_dir=ckpt_dir,
                params=params,
                status="training_complete" if v_losses else "training_failed",
                train_command=cmd,
                eval_command=self._build_eval_command(ckpt_dir),
                candidate=candidate,
                extra={
                    "returncode": r.returncode,
                    "min_val_loss": best_loss,
                    "epochs_trained": len(t_losses),
                    "autosim_eval": str(ckpt_dir / "autosim_eval.json"),
                },
            )

            if v_losses:
                if self.eval_metric == "success_rate":
                    eval_result = self._run_success_eval(ckpt_dir, env)
                    self._write_candidate_manifest(
                        ckpt_dir=ckpt_dir,
                        params=params,
                        status="eval_complete" if eval_result.success else "eval_failed",
                        train_command=cmd,
                        eval_command=self._build_eval_command(ckpt_dir),
                        candidate=candidate,
                        extra={
                            "returncode": r.returncode,
                            "min_val_loss": best_loss,
                            "epochs_trained": len(t_losses),
                            "score": eval_result.score,
                            "eval_metrics": eval_result.metrics,
                            "autosim_eval": str(ckpt_dir / "autosim_eval.json"),
                            "autosim_success_eval": str(ckpt_dir / "autosim_success_eval.json"),
                        },
                    )
                    eval_result.metrics.setdefault("min_val_loss", best_loss)
                    eval_result.metrics.setdefault("epochs_trained", len(t_losses))
                    eval_result.metrics.setdefault("manifest", str(ckpt_dir / "autosim_candidate_manifest.json"))
                    return eval_result
                return EvalResult(
                    score=best_loss,
                    success=True,
                    metrics={
                        "min_val_loss": best_loss,
                        "final_train_loss": t_losses[-1] if t_losses else None,
                        "epochs_trained": len(t_losses),
                        "ckpt_dir": str(ckpt_dir),
                        "returncode": r.returncode,
                        "manifest": str(ckpt_dir / "autosim_candidate_manifest.json"),
                    },
                )
            return EvalResult(
                score=self._failure_score(), success=False,
                metrics={
                    "returncode": r.returncode,
                    "ckpt_dir": str(ckpt_dir),
                    "manifest": str(ckpt_dir / "autosim_candidate_manifest.json"),
                },
                info=f"Training produced no Val loss. stderr: {r.stderr[-500:]}"
            )
        except subprocess.TimeoutExpired:
            self._write_candidate_manifest(
                ckpt_dir=ckpt_dir,
                params=params,
                status="training_timeout",
                train_command=cmd,
                eval_command=self._build_eval_command(ckpt_dir),
                candidate=candidate,
                extra={"error": f"Training timed out after {self.timeout}s"},
            )
            return EvalResult(
                score=self._failure_score(), success=False,
                metrics={"ckpt_dir": str(ckpt_dir), "manifest": str(ckpt_dir / "autosim_candidate_manifest.json")},
                info=f"Training timed out after {self.timeout}s: {' '.join(cmd)}"
            )
        except Exception as e:
            self._write_candidate_manifest(
                ckpt_dir=ckpt_dir,
                params=params,
                status="training_exception",
                train_command=cmd,
                eval_command=self._build_eval_command(ckpt_dir),
                candidate=candidate,
                extra={"error": str(e)},
            )
            return EvalResult(
                score=self._failure_score(),
                success=False,
                metrics={"ckpt_dir": str(ckpt_dir), "manifest": str(ckpt_dir / "autosim_candidate_manifest.json")},
                info=str(e),
            )

    def get_source_files(self) -> Dict[str, str]:
        files = {}
        for fname in [
            "policy/ACT/act_policy.py",
            "policy/ACT/detr/models/detr_vae.py",
            "policy/ACT/detr/models/transformer.py",
            "policy/ACT/detr/models/backbone.py",
            "policy/ACT/imitate_episodes.py",
        ]:
            path = os.path.join(self.repo_path, fname)
            if os.path.exists(path):
                files[fname] = Path(path).read_text()
        return files

    def get_primary_metric_name(self) -> str:
        return self.eval_metric

    def get_metric_direction(self) -> str:
        return "higher" if self.eval_metric == "success_rate" else "lower"
