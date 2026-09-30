import json

import pytest

from autosim.research.gpu_lease import GPULeaseBusy, GPUResourceLease


def test_gpu_lease_serializes_runs_by_physical_uuid(tmp_path):
    first = GPUResourceLease("GPU-abcd", metadata={"run_id": "first"},
                             directory=tmp_path / "leases")
    second = GPUResourceLease("GPU-abcd", metadata={"run_id": "second"},
                              directory=tmp_path / "leases")

    first.acquire()
    try:
        record = json.loads(first.path.read_text(encoding="utf-8"))
        assert record["run_id"] == "first"
        assert record["device_uuid"] == "GPU-abcd"
        with pytest.raises(GPULeaseBusy, match="active AutoSim lease"):
            second.acquire()
    finally:
        first.release()

    assert first.path.read_text(encoding="utf-8") == ""
    second.acquire()
    second.release()


def test_gpu_lease_fails_closed_for_a_non_private_directory(tmp_path):
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    lease = GPUResourceLease("GPU-abcd", directory=directory)

    with pytest.raises(PermissionError, match="private"):
        lease.acquire()
