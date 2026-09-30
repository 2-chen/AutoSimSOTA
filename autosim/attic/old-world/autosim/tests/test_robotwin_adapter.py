from pathlib import Path
import yaml

from autosim.research.robotwin_adapter import RoboTwinAdapter
from autosim.research.repository_autoresearch import main


PROJECT = Path(__file__).resolve().parents[1]
ROBOTWIN = PROJECT.parent / "RoboTwin"


def test_robotwin_static_contract_and_capabilities():
    adapter = RoboTwinAdapter(ROBOTWIN)
    assert len(adapter.tasks()) == 50
    result = adapter.discover("beat_block_hammer")
    assert result["benchmark"] == "RoboTwin"
    assert result["task_contract"]["max_episode_steps"] == 400
    assert result["task_contract"]["action_semantics"].startswith("14D dual-arm")
    assert result["task_contract"]["control_frequency_hz"] == 15
    assert result["evidence"]["task_data"]["episode_count"] == 50
    capabilities = {row.capability_id: row for row in adapter.capabilities("beat_block_hammer")}
    assert capabilities["repository_recognition"].status == "verified"
    assert capabilities["official_expert"].status == "verified"
    assert capabilities["action_unit_and_timing"].status == "verified"
    assert capabilities["native_evaluation"].status == "verified"
    assert capabilities["embodiment_assets"].status == "verified"
    assert capabilities["training_data"].status == "verified"
    assert capabilities["runtime_environment"].status == "verified"
    assert capabilities["act_training"].status == "verified"


def test_robotwin_auto_task_is_deterministic():
    assert RoboTwinAdapter(ROBOTWIN).select_task("auto") == "beat_block_hammer"


def test_robotwin_probe_only_is_a_completed_valid_probe(tmp_path):
    code = main([str(ROBOTWIN), "--probe-only", "--run-id", "probe",
                 "--output-root", str(tmp_path)])
    assert code == 0
    import json
    state = json.loads((tmp_path / "RoboTwin/probe/run_state.json").read_text())
    assert state["status"] == "prepared"
    assert state["capability_probe_complete"] is True
    assert state["execution_complete"] is True
    assert state["result_valid"] is True
    assert state["missing_execution_capabilities"] == []


def test_bounded_randomized_collection_preserves_official_factor_settings():
    root = ROBOTWIN / "env_cfg/task_config"
    official = yaml.safe_load((root / "demo_randomized.yml").read_text())
    bounded = yaml.safe_load((root / "autosim_randomized_10.yml").read_text())
    assert bounded["domain_randomization"] == official["domain_randomization"]
    assert bounded["camera"] == official["camera"]
    assert bounded["embodiment"] == official["embodiment"]
    assert bounded["episode_num"] == 10
    assert bounded["max_seed_attempts"] == 100
