"""Bounded, receipt-producing RoboTwin ACT training and native evaluation jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .common import atomic_json, digest, now, read_json, run_command
from .robotwin_evidence import analyze


def runtime_environment(project_root: Path, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    python = project_root / ".venv_robotwin/bin/python"
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), XPOLICYLAB_PYTHON=str(python),
               PYTHONUNBUFFERED="1", MPLCONFIGDIR="/tmp/autosim_robotwin_matplotlib",
               AUTOSIM_QUIET_STEPS="1", ROBOTWIN_SUPPRESS_EVAL_CONFIG="1")
    path = [str(python.parent), "/home/wbc/miniconda3/condabin",
            "/home/wbc/miniconda3/envs/robotwin5090/bin", "/usr/bin", "/bin"]
    env["PATH"] = os.pathsep.join(path)
    library = "/home/wbc/miniconda3/envs/robotwin5090/lib"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [library] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    return env


def train_command(project_root: Path, repo: Path, dataset_key: str, output: Path,
                  *, epochs: int, seed: int, lr: float, chunk_size: int,
                  kl_weight: float) -> list[str]:
    python = project_root / ".venv_robotwin/bin/python"
    script = repo / "XPolicyLab/policy/ACT/imitate_episodes.py"
    return [str(python), str(script), "--bench_name", "AutoResearch", "--task_name", dataset_key,
            "--ckpt_setting", dataset_key, "--ckpt_dir", str(output), "--policy_class", "ACT",
            "--kl_weight", str(int(kl_weight)), "--chunk_size", str(int(chunk_size)),
            "--hidden_dim", "512", "--batch_size", "16", "--dim_feedforward", "3200",
            "--num_epochs", str(int(epochs)), "--lr", str(float(lr)), "--save_freq", str(int(epochs)),
            "--seed", str(int(seed))]


def train(project_root: Path, repo: Path, dataset_key: str, output: Path, *, gpu: str,
          epochs: int, seed: int, lr: float, chunk_size: int, kl_weight: float,
          timeout: int) -> dict[str, Any]:
    act = repo / "XPolicyLab/policy/ACT"
    registry = read_json(act / "TASK_CONFIGS.json")
    if dataset_key not in registry:
        raise KeyError(f"ACT dataset is not registered: {dataset_key}")
    dataset = act / registry[dataset_key]["dataset_dir"]
    files = sorted(dataset.glob("episode_*.hdf5"))
    if len(files) != int(registry[dataset_key]["num_episodes"]):
        raise ValueError("ACT registry episode count differs from processed dataset")
    command = train_command(project_root, repo, dataset_key, output, epochs=epochs, seed=seed,
                            lr=lr, chunk_size=chunk_size, kl_weight=kl_weight)
    env = runtime_environment(project_root, gpu)
    env["ACT_ACTION_DIM"] = "14"
    process = run_command(command, cwd=act, env=env, output=output.parent / "train_process",
                          timeout=timeout)
    checkpoint = output / "policy_last.ckpt"
    if not checkpoint.is_file() or not (output / "dataset_stats.pkl").is_file():
        raise RuntimeError("ACT training exited without checkpoint and deployment statistics")
    receipt = {"schema_version": 1, "kind": "robotwin_act_training", "created_at": now(),
               "status": "completed", "dataset_key": dataset_key, "dataset_episode_count": len(files),
               "parameters": {"epochs": epochs, "seed": seed, "lr": lr,
                              "chunk_size": chunk_size, "kl_weight": kl_weight},
               "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
               "dataset_stats_sha256": digest(output / "dataset_stats.pkl"), "process": process}
    atomic_json(output.parent / "training_receipt.json", receipt)
    return receipt


def adopt_completed_training(repo: Path, dataset_key: str, output: Path, *, epochs: int,
                             seed: int, lr: float, chunk_size: int,
                             kl_weight: float) -> dict[str, Any]:
    """Register an already completed foreground job without inventing process telemetry."""
    act = repo / "XPolicyLab/policy/ACT"
    registry = read_json(act / "TASK_CONFIGS.json")
    if dataset_key not in registry:
        raise KeyError(f"ACT dataset is not registered: {dataset_key}")
    dataset = act / registry[dataset_key]["dataset_dir"]
    files = sorted(dataset.glob("episode_*.hdf5"))
    checkpoint = output / "policy_last.ckpt"
    epoch_checkpoint = output / f"policy_epoch_{epochs}_seed_{seed}.ckpt"
    stats = output / "dataset_stats.pkl"
    missing = [str(path) for path in (checkpoint, epoch_checkpoint, stats) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot adopt incomplete ACT training output: {missing}")
    if len(files) != int(registry[dataset_key]["num_episodes"]):
        raise ValueError("ACT registry episode count differs from processed dataset")
    if digest(checkpoint) != digest(epoch_checkpoint):
        raise ValueError("final and named epoch checkpoints differ")
    receipt = {"schema_version": 1, "kind": "robotwin_act_training", "created_at": now(),
               "status": "completed", "execution_origin": "foreground_job_adopted_after_completion",
               "process_telemetry_available": False, "dataset_key": dataset_key,
               "dataset_episode_count": len(files),
               "parameters": {"epochs": epochs, "seed": seed, "lr": lr,
                              "chunk_size": chunk_size, "kl_weight": kl_weight},
               "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
               "epoch_checkpoint": str(epoch_checkpoint),
               "dataset_stats_sha256": digest(stats)}
    atomic_json(output.parent / "training_receipt.json", receipt)
    return receipt


def eval_command(repo: Path, checkpoint_dir: Path, task: str, eval_seed: int,
                 gpu: str) -> list[str]:
    return ["bash", "eval.sh", "AutoResearch", task, str(checkpoint_dir.absolute()),
            "aloha_agilex", "joint", str(eval_seed), str(gpu), str(gpu),
            "autosim_robotwin", "autosim_robotwin"]


def evaluate(project_root: Path, repo: Path, checkpoint_dir: Path, task: str, setting: str,
             output: Path, *, episodes: int, eval_seed: int, gpu: str,
             max_seed_attempts: int, timeout: int, purpose: str,
             seed_bank: list[int] | None = None) -> dict[str, Any]:
    receipt_path = output / "evaluation_receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        expected_bank_hash = None
        if seed_bank is not None:
            expected_bank_hash = digest(output / "seed_bank.json")
        if receipt.get("seed_bank_sha256") != expected_bank_hash:
            raise RuntimeError("evaluation receipt seed bank differs from request")
        return receipt
    output.mkdir(parents=True, exist_ok=True)
    args_file = output / "eval_args.txt"
    instruction = "unseen" if setting == "demo_randomized" else "seen"
    args_file.write_text(f"--task_config\n{setting}\n--test_num\n{episodes}\n"
                         f"--max_seed_attempts\n{max_seed_attempts}\n"
                         f"--instruction_type\n{instruction}\n", encoding="utf-8")
    seed_bank_hash = None
    if seed_bank is not None:
        if len(seed_bank) != episodes or len(set(seed_bank)) != len(seed_bank):
            raise ValueError("explicit evaluation seed bank must be unique and match episode count")
        bank_path = output / "seed_bank.json"
        atomic_json(bank_path, {"schema_version": 1, "seeds": [int(seed) for seed in seed_bank]})
        seed_bank_hash = digest(bank_path)
        with args_file.open("a", encoding="utf-8") as stream:
            stream.write(f"--expert_check\nfalse\n--seed_bank_file\n{bank_path}\n")
    result_root = repo / "eval_result" / task / "ACT" / setting / checkpoint_dir.name
    before = set(result_root.glob("*/_result.txt")) if result_root.is_dir() else set()
    command = eval_command(repo, checkpoint_dir, task, eval_seed, gpu)
    env = runtime_environment(project_root, gpu)
    env["ROBOTWIN_EVAL_ARGS_FILE"] = str(args_file)
    process = run_command(command, cwd=repo / "XPolicyLab/policy/ACT", env=env,
                          output=output / "process", timeout=timeout)
    after = set(result_root.glob("*/_result.txt"))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        raise RuntimeError(f"native evaluation produced {len(created)} new result files")
    native_dir = created[0].parent
    trace = native_dir / "episode_results.jsonl"
    if not trace.is_file():
        raise RuntimeError("native evaluation did not emit per-episode audit trace")
    limits = yaml.safe_load((repo / "env_cfg/task_config/_eval_step_limit.yml").read_text())
    evidence = analyze(trace, checkpoint_dir / "policy_last.ckpt", task=task, setting=setting,
                       max_actions=int(limits[task]), purpose=purpose, evaluation_seed=eval_seed)
    atomic_json(output / "evidence.json", evidence)
    receipt = {"schema_version": 1, "kind": "robotwin_native_evaluation", "created_at": now(),
               "status": "completed", "task": task, "setting": setting,
               "checkpoint_sha256": digest(checkpoint_dir / "policy_last.ckpt"),
               "native_result": str(created[0]), "native_result_sha256": digest(created[0]),
               "trace": str(trace), "trace_sha256": digest(trace),
               "seed_bank_sha256": seed_bank_hash,
               "summary": evidence["metrics"]["summary"], "evidence_id": evidence["evidence_id"],
               "process": process}
    atomic_json(receipt_path, receipt)
    return receipt
