import json

from autosim.research.robotwin_mix_dataset import build_mix


def test_build_mix_is_hard_linked_and_registered(tmp_path):
    act = tmp_path / "ACT"
    official, extra = tmp_path / "official", tmp_path / "extra"
    official.mkdir(parents=True)
    extra.mkdir()
    (official / "episode_0.hdf5").write_bytes(b"official-0")
    (official / "episode_1.hdf5").write_bytes(b"official-1")
    (extra / "episode_0.hdf5").write_bytes(b"extra-0")
    registry = act / "TASK_CONFIGS.json"
    act.mkdir()
    registry.write_text("{}")
    output = act / "processed_data/mix"
    result = build_mix(official, extra, output, registry, "mix-task")
    assert result["status"] == "passed"
    assert result["official_episode_count"] == 2
    assert result["new_episode_count"] == 1
    assert result["all_hard_linked"] is True
    assert (output / "episode_2.hdf5").samefile(extra / "episode_0.hdf5")
    assert json.loads(registry.read_text())["mix-task"]["num_episodes"] == 3


def test_build_mix_refuses_existing_different_file(tmp_path):
    act = tmp_path / "ACT"
    official, extra, output = tmp_path / "official", tmp_path / "extra", act / "mix"
    official.mkdir(parents=True)
    extra.mkdir()
    output.mkdir(parents=True)
    (official / "episode_0.hdf5").write_bytes(b"official")
    (extra / "episode_0.hdf5").write_bytes(b"extra")
    (output / "episode_0.hdf5").write_bytes(b"different")
    registry = act / "TASK_CONFIGS.json"
    registry.write_text("{}")
    import pytest
    with pytest.raises(FileExistsError):
        build_mix(official, extra, output, registry, "mix-task")
