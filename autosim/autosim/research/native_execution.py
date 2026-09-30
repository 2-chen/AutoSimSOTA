"""One resource boundary for native preparation and diagnostic operations."""
from contextlib import contextmanager
from pathlib import Path
import time
import uuid
import math


@contextmanager
def resource_window(output: Path, repo: Path, timeout: float, decision=None):
    """CPU by default; accelerator access requires fresh occupancy, lease and budget."""
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("native resource window must be finite and positive")
    if decision is None or not decision.on_gpu:
        if decision is not None and decision.device == "unavailable":
            raise ValueError("selected native resource is unavailable")
        yield float(timeout), False, {}
        return
    from .compute_decision import recheck_gpu
    from .gpu_lease import GPUResourceLease
    from .task_budget import TaskGPUBudget
    device = (decision.evidence.get("selected_device") or {}).get("uuid")
    if not device:
        raise ValueError("native GPU operation requires a physical device identity")
    recheck_gpu(decision, settle_seconds=0)
    identity = uuid.uuid4().hex
    with GPUResourceLease(device, metadata={"stage": "native_probe", "lease_id": identity}):
        budget = TaskGPUBudget(output)
        if not budget.path.exists():
            budget.initialize(repo)
        window = min(float(timeout), budget.snapshot()["available_gpu_seconds"])
        budget.reserve(identity, device, requested_seconds=window)
        started = time.monotonic()
        try:
            yield window, True, {**decision.environment, "AUTOSIM_GPU_LEASE_ID": identity}
        finally:
            budget.finish(identity, time.monotonic() - started)
