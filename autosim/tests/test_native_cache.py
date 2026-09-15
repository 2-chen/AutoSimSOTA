import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from autosim.research.native_cache import configure_native_cache


def test_unconfigured_cache_never_imports_engine(monkeypatch):
    monkeypatch.delenv("AUTOSIM_NATIVE_CACHE", raising=False)
    with patch.dict(sys.modules, {"embodichain.lab.sim": None}):
        assert configure_native_cache() is None


@pytest.mark.parametrize("instances", [0, 1])
def test_cache_is_set_before_construction_only(monkeypatch, instances):
    engine = types.SimpleNamespace(SimulationManager=types.SimpleNamespace(
        get_instance_num=lambda: instances))
    module = types.ModuleType("embodichain.lab.sim")
    module.sim_manager = engine
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        monkeypatch.setenv("AUTOSIM_NATIVE_CACHE", str(root / "gpu_cache"))
        with patch.dict(sys.modules, {"embodichain.lab.sim": module}):
            if instances:
                with pytest.raises(RuntimeError, match="before simulation"):
                    configure_native_cache(root / "receipt")
                assert not (root / "gpu_cache").exists()
            else:
                receipt = configure_native_cache(root / "receipt")
                assert engine.SIM_CACHE_DIR == root / "gpu_cache"
                assert all(Path(p).is_dir() for p in receipt["cache_paths"].values())
                assert (root / "receipt/native_cache.json").is_file()
