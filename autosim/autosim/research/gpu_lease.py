"""A cooperative per-device lease for serialized AutoSim GPU stages."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping


class GPULeaseBusy(RuntimeError):
    """Another AutoSim process currently owns this device lease."""


class GPUResourceLease:
    """Hold a process-scoped exclusive lease keyed by the physical device UUID.

    This coordinates AutoSim processes that honor the lease. External applications are
    handled separately by the fresh NVML occupancy check in ``compute_decision``; a lock
    file alone is never evidence that an unmanaged process has released a GPU.
    """

    def __init__(self, device_uuid: str, *, metadata: Mapping[str, Any] | None = None,
                 directory: Path | None = None):
        self.device_uuid = str(device_uuid or "").strip()
        if not self.device_uuid:
            raise ValueError("a GPU lease requires a physical device UUID")
        self.directory = Path(directory or
                              f"/tmp/autosimsota-gpu-leases-{os.getuid()}")
        self.metadata = dict(metadata or {})
        key = hashlib.sha256(self.device_uuid.encode("utf-8")).hexdigest()
        self.path = self.directory / f"{key}.lock"
        self._fd: int | None = None

    def acquire(self) -> "GPUResourceLease":
        if self._fd is not None:
            return self
        try:
            import fcntl
        except ImportError as exc:
            # CPU-only use of AutoSim must remain importable on non-POSIX systems. GPU
            # execution fails closed there until a native lock backend is provided.
            raise OSError("exclusive GPU leases require POSIX flock on this platform") from exc
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise OSError("GPU lease directory is not a real directory")
        directory_info = self.directory.stat(follow_symlinks=False)
        if (directory_info.st_uid != os.getuid() or
                directory_info.st_mode & 0o077):
            raise PermissionError("GPU lease directory must be owned by this user and private")
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) |
            getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise PermissionError("GPU lease file must be private and regular")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.lseek(descriptor, 0, os.SEEK_SET)
                held = os.read(descriptor, 4096).decode("utf-8", errors="replace").strip()
                detail = f"; holder={held[:500]}" if held else ""
                raise GPULeaseBusy(
                    f"GPU {self.device_uuid} has an active AutoSim lease{detail}") from exc
            record = {**self.metadata, "pid": os.getpid(),
                      "device_uuid": self.device_uuid}
            encoded = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, encoded[:4096])
            os.fsync(descriptor)
            self._fd = descriptor
            return self
        except BaseException:
            os.close(descriptor)
            raise

    def release(self) -> None:
        descriptor, self._fd = self._fd, None
        if descriptor is None:
            return
        try:
            os.ftruncate(descriptor, 0)
            try:
                import fcntl
            except ImportError:
                pass  # acquire() cannot succeed without this backend
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "GPUResourceLease":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()
