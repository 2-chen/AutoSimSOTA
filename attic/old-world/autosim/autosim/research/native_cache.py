"""Configure process-local engine cache paths before creating a simulation."""
from __future__ import annotations
import os
from pathlib import Path


def configure_native_cache(output: Path | None = None):
    value = os.environ.get("AUTOSIM_NATIVE_CACHE")
    if not value:
        return None
    from embodichain.lab.sim import sim_manager
    from .common import atomic_json
    if sim_manager.SimulationManager.get_instance_num():
        raise RuntimeError("native cache must be configured before simulation construction")
    root = Path(value)
    paths = {"SIM_CACHE_DIR":root,"MATERIAL_CACHE_DIR":root / "mat_cache",
             "CONVEX_DECOMP_DIR":root / "convex","REACHABLE_XPOS_DIR":root / "reachable"}
    for name,path in paths.items():
        path.mkdir(parents=True,exist_ok=True)
        setattr(sim_manager,name,path)
    receipt = {"cache_paths":{k:str(v) for k,v in paths.items()},
               "scope":"exclusive GPU lease; shared read-only assets remain separate"}
    if output:
        atomic_json(Path(output) / "native_cache.json",receipt)
    return receipt
