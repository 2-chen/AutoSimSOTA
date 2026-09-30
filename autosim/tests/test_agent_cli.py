import subprocess
from pathlib import Path

import pytest

from autosim.research import agent_cli


def test_agent_cli_creates_tracked_only_copy_and_calls_runtime(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.py").write_text("print('fixture')\n", encoding="utf-8")
    (source / ".env").write_text("API_KEY=never-copy", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "program.py"], check=True)
    output = tmp_path / "runs" / "one"
    observed = {}

    def runtime(**kwargs):
        observed.update(kwargs)
        return {"status": "completed", "total_cost_usd": 0.01}

    monkeypatch.setattr(agent_cli, "run_coding_agent", runtime)
    code = agent_cli.main([str(source), str(output), "--prompt", "inspect fixture",
                           "--role", "supervisor", "--max-total-budget-usd", "2"])

    assert code == 0
    assert (output / "checkout" / "program.py").is_file()
    assert not (output / "checkout" / ".env").exists()
    assert observed["workspace"] == output / "checkout"
    assert observed["resume"] is False
    assert observed["role"] == "supervisor"
    assert observed["max_total_budget_usd"] == 2
    assert "completed" in capsys.readouterr().out


def test_agent_cli_requires_explicit_resume_for_an_existing_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(SystemExit) as error:
        agent_cli.main([str(source), str(output), "--prompt", "task"])

    assert error.value.code == 2


def test_agent_cli_cannot_create_output_inside_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(SystemExit):
        agent_cli.main([str(source), str(source / "run"), "--prompt", "task"])
    assert not list(source.iterdir())
