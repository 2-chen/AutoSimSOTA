from pathlib import Path

from autosim.research.robotwin_jobs import (adopt_completed_training, eval_command,
                                            runtime_environment, train_command)


def test_commands_use_isolated_runtime_and_explicit_budget(tmp_path):
    project, repo = tmp_path / "project", tmp_path / "repo"
    train = train_command(project, repo, "dataset-key", tmp_path / "checkpoint",
                          epochs=6000, seed=1000, lr=1e-5, chunk_size=50, kl_weight=10)
    assert train[0] == str(project / ".venv_robotwin/bin/python")
    assert train[train.index("--num_epochs") + 1] == "6000"
    assert train[train.index("--ckpt_setting") + 1] == "dataset-key"
    evaluate = eval_command(repo, tmp_path / "checkpoint", "beat_block_hammer", 3, "0")
    assert evaluate[:2] == ["bash", "eval.sh"]
    assert evaluate[3] == "beat_block_hammer"


def test_runtime_environment_does_not_use_external_project(tmp_path):
    env = runtime_environment(tmp_path, "0")
    assert env["XPOLICYLAB_PYTHON"] == str(tmp_path / ".venv_robotwin/bin/python")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "/home/wbc/\u4e0b\u8f7d/autoresearch/RoboTwin" not in env.get("PYTHONPATH", "")


def test_adopt_training_requires_matching_named_and_last_checkpoint(tmp_path):
    repo = tmp_path / "repo"
    act = repo / "XPolicyLab/policy/ACT"
    dataset = act / "processed"
    dataset.mkdir(parents=True)
    (dataset / "episode_0.hdf5").write_bytes(b"data")
    (act / "TASK_CONFIGS.json").write_text(
        '{"key":{"dataset_dir":"processed","num_episodes":1}}')
    output = tmp_path / "run/checkpoint"
    output.mkdir(parents=True)
    (output / "policy_last.ckpt").write_bytes(b"model")
    (output / "policy_epoch_2_seed_1.ckpt").write_bytes(b"model")
    (output / "dataset_stats.pkl").write_bytes(b"stats")
    receipt = adopt_completed_training(repo, "key", output, epochs=2, seed=1,
                                       lr=1e-5, chunk_size=50, kl_weight=10)
    assert receipt["status"] == "completed"
    assert receipt["execution_origin"] == "foreground_job_adopted_after_completion"
