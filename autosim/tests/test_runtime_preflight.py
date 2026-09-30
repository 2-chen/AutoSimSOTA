import json
from types import SimpleNamespace
from autosim.research import runtime_preflight


def test_socket_denial_is_actionable_without_a_provider_request(monkeypatch):
    def denied(*a, **k):
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(runtime_preflight.socket, "socket", denied)
    monkeypatch.setattr(runtime_preflight.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(runtime_preflight.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    result = runtime_preflight.check_runtime()
    assert result["status"] == "infrastructure_blocked"
    assert result["checks"][0]["errno"] == 1


def test_cli_preflight_blocks_before_copy_and_agent(tmp_path, monkeypatch):
    from autosim.research.derive_and_run import main
    monkeypatch.setattr(runtime_preflight, "check_runtime", lambda: {
        "status": "infrastructure_blocked", "checks": [
            {"name": "loopback_gateway", "ok": False, "errno": 1}]})
    monkeypatch.setattr("autosim.llm_client.load_credential_file", lambda **_: {})
    output = tmp_path / "run"
    assert main([str(tmp_path / "repo"), str(output), "--framework", "autosota_sim_v1"]) == 2
    assert "启动受阻" in (output / "RUN.md").read_text()
    assert not (output / "checkout").exists()
    assert json.loads((output / "task_gpu_budget.json").read_text())["charged_seconds"] == 0
