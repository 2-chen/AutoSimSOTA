"""Durable, explicitly selected invocation revisions; never verified commands."""
from pathlib import Path
import re

from .common import atomic_json, now, object_digest, read_json, redact


def save(output: Path, *, stage: str, base: dict, row: dict, reason: str,
         evidence_ref: str = '') -> dict:
    if not reason.strip():
        raise ValueError('an invocation draft needs an Agent rationale')
    payload = {'schema_version': 1, 'status': 'unverified', 'stage': stage,
        'base_sha256': object_digest(base), 'base_context': base,
        'row': row, 'reason': redact(reason)[:2000],
        'native_evidence_ref': evidence_ref, 'created_at': now(),
        'authority': 'Agent revision; requires explicit selection and new native verification'}
    identity = object_digest(payload)
    root = Path(output).resolve()
    directory = root / 'invocation_drafts'
    if directory.is_symlink():
        raise ValueError('unsafe invocation draft directory')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{identity}.json'
    if path.is_symlink():
        raise ValueError('unsafe invocation draft path')
    atomic_json(path, {'id': identity, **payload})
    return {'id': identity, 'ref': path.relative_to(root).as_posix(), **payload}


def load(output: Path, ref: str, *, stage: str, base: dict | None = None,
         rebind_inputs: list[str] | None = None) -> dict:
    if (rebind_inputs is not None and (not isinstance(rebind_inputs, list) or
            any(key != 'checkpoint' for key in rebind_inputs) or
            len(rebind_inputs) != len(set(rebind_inputs)))):
        raise ValueError('draft rebind_inputs currently permits checkpoint only, once')
    if rebind_inputs and base is None:
        raise ValueError('draft rebind needs the current source/input/interpreter context')
    root = Path(output).resolve()
    if isinstance(ref, str) and re.fullmatch(r'[0-9a-f]{64}', ref):
        # The catalog publishes both id and ref. Either identifies the same
        # confined immutable draft; choosing an id is not implicit adoption.
        ref = f'invocation_drafts/{ref}.json'
    relative = Path(ref)
    if (relative.is_absolute() or len(relative.parts) != 2 or
            relative.parts[0] != 'invocation_drafts' or relative.suffix != '.json'):
        raise ValueError('select the offered 64-hex draft id or invocation_drafts/<id>.json ref')
    path = root / relative
    if (path.is_symlink() or path.parent.is_symlink() or not path.is_file() or
            not path.resolve().is_relative_to(root) or path.stat().st_size > 256 * 1024):
        raise ValueError('unsafe or missing invocation draft')
    record = read_json(path)
    payload = {key: value for key, value in record.items() if key != 'id'}
    if (record.get('id') != relative.stem or object_digest(payload) != relative.stem or
            record.get('schema_version') != 1 or record.get('status') != 'unverified' or
            record.get('stage') != stage or not isinstance(record.get('row'), dict)):
        raise ValueError('invocation draft identity changed')
    if base is not None and record.get('base_sha256') != object_digest(base):
        previous = record.get('base_context')
        if (not rebind_inputs or not isinstance(previous, dict) or
                object_digest(previous) != record.get('base_sha256') or
                {k:v for k,v in previous.items() if k not in rebind_inputs} !=
                {k:v for k,v in base.items() if k not in rebind_inputs}):
            raise ValueError('invocation draft belongs to changed source/input/interpreter context')
        # This transfers an unverified blueprint, never accepted execution/weights.
        # Main still chooses the new producer; caller binding and native postconditions
        # are enforced by make_runnable under the unchanged isolation/protocol.
    return {**record, 'ref': relative.as_posix(),
            'rebound_inputs': list(rebind_inputs or [])}


def catalog(output: Path, limit: int = 6) -> list[dict]:
    directory = Path(output) / 'invocation_drafts'
    if directory.is_symlink():
        return []
    rows = []
    for path in sorted(directory.glob('*.json'), key=lambda p: p.stat().st_mtime,
                       reverse=True)[:limit]:
        try:
            if path.is_symlink() or path.stat().st_size > 256 * 1024:
                continue
            candidate = read_json(path)
            record = load(output, path.relative_to(output).as_posix(),
                          stage=candidate.get('stage'))
            rows.append({key: record.get(key) for key in ('id', 'ref', 'stage', 'status',
                'base_sha256', 'row', 'reason', 'native_evidence_ref', 'authority')} |
                {'rebindable_inputs': ['checkpoint'] if isinstance(record.get('base_context'), dict) else []})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return rows
