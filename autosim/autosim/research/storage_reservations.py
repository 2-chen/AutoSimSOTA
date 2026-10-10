"""Cooperative, conservative future-byte reservations on the same filesystem.

Reservations do not preallocate disk and cannot account for unrelated host writes.
Other live reservations are subtracted from current free space; their full size
is retained until exit (conservative even after a producer has written bytes).
"""
import fcntl
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from .common import atomic_json, now, read_json
from .storage_guard import preflight
from .process_executor import _proc_identity, inspect_process_identity


def _directory():
    return Path('/tmp')/f'autosim-storage-reservations-{os.getuid()}'


@contextmanager
def reserve(output, *, stage, required_bytes=None, path=None):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    target=Path(path or output).resolve()
    while not target.exists(): target=target.parent
    device=target.stat().st_dev
    directory=_directory()
    if directory.is_symlink(): raise ValueError('unsafe reservation store')
    directory.mkdir(mode=0o700,exist_ok=True)
    if directory.stat().st_uid!=os.getuid() or directory.stat().st_mode & 0o077:
        raise ValueError('reservation store must be private to its owner')
    lock=directory/'lock'
    if lock.is_symlink(): raise ValueError('unsafe reservation lock')
    path=directory/f'{uuid.uuid4().hex}.json'
    with lock.open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        held=0
        for item in directory.glob('*.json'):
            if item.is_symlink(): raise ValueError('unsafe reservation record')
            row=read_json(item)
            if row.get('device')!=device: continue
            status=inspect_process_identity(row['identity']).get('status')
            if status=='not_running':
                item.unlink()  # expired lease metadata only, never a payload
            else: held+=row['bytes']
        probe=preflight(target,stage=stage,required_bytes=required_bytes)
        requested=max(probe['required_bytes'],(required_bytes or 0)+probe['safety_floor_bytes'])
        if probe['free_bytes']-held<requested:
            raise ValueError(f"storage admission refused: {probe['free_bytes']} free, "
                             f"{held} held, {requested} requested")
        identity=_proc_identity(os.getpid())
        if not identity: raise ValueError('storage reservation identity unavailable')
        identity['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        atomic_json(path,{'at':now(),'device':device,'output':str(output),
                         'bytes':requested,'identity':identity})
    try:
        yield {**probe,'other_reserved_bytes':held,'reservation_bytes':requested,
               'scope':'cooperative conservative future-byte reservation; not disk preallocation'}
    finally:
        with lock.open('a+b') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            path.unlink(missing_ok=True)
