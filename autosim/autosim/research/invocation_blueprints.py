"""Immutable historical invocations: explicit starting points, never approvals.

A source edit invalidates live adapters, not the knowledge of how they ran.
Only a Main-selected blueprint can supply an initial row; a new derivation and
native verification remain mandatory. Checkpoint/work settings stay caller-owned.
"""
from pathlib import Path
import json
import re

from .common import atomic_json, digest, object_digest, read_json
from .experiment_transaction import safe


def _read(output, relative, *, limit=8 * 1024 * 1024):
    path = safe(output, relative)
    if path.stat().st_size > limit:
        raise ValueError('oversized invocation blueprint record')
    return read_json(path)


def _bindings(output):
    doc = _read(output, 'workspace_snapshot.json')
    return object_digest(doc.get('resource_bindings', doc.get('resources', [])))


def _proof(output, interface):
    from .evidence_store import read_attempt_evidence
    ref = interface.get('native_evidence_ref')
    if not isinstance(ref, str) or not re.fullmatch(r'evidence/[0-9a-f]{32}\.json', ref):
        raise ValueError('unsafe blueprint native evidence reference')
    proof = read_attempt_evidence(output, Path(ref).stem, limit=1)
    startup = interface.get('verification_scope') == 'training_startup_only'
    if proof.get('status') != ('timed_out' if startup else 'completed'):
        raise ValueError('blueprint historical proof changed')
    if not startup and (type(proof.get('returncode')) is not int or proof['returncode'] != 0):
        raise ValueError('blueprint historical verification was unsuccessful')


def remember(output: Path, repo: Path, stage: str, entry: dict, *, interpreter: str,
             task: str) -> dict:
    if (not isinstance(stage, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', stage)
            or not isinstance(entry, dict) or not isinstance(entry.get('row'), dict)
            or not isinstance(entry.get('source'), str) or not entry['source'].strip()):
        raise ValueError('blueprint needs a complete historical stage adapter')
    from .patch_validation import checked_function
    from .execution_derive import argv_function_name
    try:
        checked_function(entry['source'], argv_function_name(stage))
    except (ValueError, SyntaxError) as exc:
        raise ValueError('invalid historical blueprint source') from exc
    interface = None
    directory = safe(output, 'execution_interfaces/' + stage)
    for path in sorted(directory.glob('*.json'))[-24:]:
        candidate = _read(output, path.relative_to(output).as_posix())
        body = {k: v for k, v in candidate.items() if k != 'id'}
        if (candidate.get('id') == path.stem == object_digest(body)
                and candidate.get('stage') == stage
                and candidate.get('stage_sha256') == object_digest(entry)):
            _proof(output, candidate)
            log_path = safe(output, candidate['derivation_ref'])
            if digest(log_path) != candidate.get('derivation_sha256'):
                continue
            arguments = _read(output, candidate['derivation_ref']).get('arguments') or {}
            if arguments.get('task') != task:
                continue
            interface = {**candidate, 'ref': path.relative_to(output).as_posix()}
            break
    if interface is None:
        raise ValueError('no published interface binds this historical stage record')
    # Legacy archives may be imported only while the runtime/bindings still match.
    # Changed source is precisely why this is an unverified blueprint, not reuse.
    if (interface.get('environment_sha256') != digest(safe(output, 'environment.json'))
            or interface.get('bindings_sha256') != _bindings(output)):
        raise ValueError('historical blueprint environment or bindings changed')
    body = {'schema_version': 1, 'status': 'unverified', 'stage': stage,
            'repository': str(repo.resolve()), 'interpreter': str(interpreter),
            'task': task, 'entry': entry, 'interface_ref': interface['ref'],
            'interface_id': interface['id'],
            'environment_sha256': interface['environment_sha256'],
            'bindings_sha256': interface['bindings_sha256'],
            'native_evidence_ref': interface['native_evidence_ref'],
            'authority': 'historical blueprint only; Main selection and NEW native verification required'}
    if len(json.dumps(body).encode()) > 256 * 1024:
        raise ValueError('oversized invocation blueprint')
    identity = object_digest(body)
    ref = 'invocation_blueprints/' + identity + '.json'
    path = safe(output, ref)
    document = {**body, 'id': identity}
    if path.exists() and _read(output, ref) != document:
        raise ValueError('immutable invocation blueprint changed')
    if not path.exists():
        atomic_json(path, document)
    return {**document, 'ref': ref}


def load(output: Path, ref: str, *, repo: Path, stage: str, interpreter: str,
         task: str, current_row: dict) -> dict:
    if isinstance(ref, str) and re.fullmatch(r'[0-9a-f]{64}', ref):
        ref = 'invocation_blueprints/' + ref + '.json'
    if not isinstance(ref, str) or not re.fullmatch(r'invocation_blueprints/[0-9a-f]{64}\.json', ref):
        raise ValueError('select an offered invocation blueprint ID or ref')
    record = _read(output, ref, limit=256 * 1024)
    body = {k: v for k, v in record.items() if k != 'id'}
    if (record.get('id') != Path(ref).stem or object_digest(body) != record.get('id')
            or record.get('schema_version') != 1 or record.get('status') != 'unverified'):
        raise ValueError('invocation blueprint identity changed')
    if (record.get('stage') != stage or record.get('repository') != str(repo.resolve())
            or record.get('interpreter') != str(interpreter) or record.get('task') != task):
        raise ValueError('invocation blueprint belongs to another stage/repository/task/interpreter')
    interface = _read(output, record['interface_ref'])
    if (interface.get('id') != record['interface_id']
            or object_digest({k: v for k, v in interface.items() if k != 'id'}) != record['interface_id']
            or interface.get('stage') != stage
            or interface.get('stage_sha256') != object_digest(record['entry'])
            or interface.get('native_evidence_ref') != record.get('native_evidence_ref')):
        raise ValueError('invocation blueprint provenance changed')
    _proof(output, interface)
    if (record['environment_sha256'] != digest(safe(output, 'environment.json'))
            or record['bindings_sha256'] != _bindings(output)):
        raise ValueError('invocation blueprint environment or bindings changed; derive fresh')
    row = record['entry']['row']
    if row.get('entrypoint') != current_row.get('entrypoint'):
        raise ValueError('invocation blueprint entrypoint differs; derive fresh')
    return {**record, 'ref': ref}


def import_archives(output: Path, repo: Path, *, interpreter: str) -> list[str]:
    """Migrate bounded old snapshots, without modifying/approving their contents."""
    imported = []
    paths = sorted(safe(output, 'superseded_stages').glob('*.json'))[-6:]
    current = safe(output, 'derived_stages.json')
    if current.is_file():
        paths.append(current)
    for path in paths:
        try:
            archive = _read(output, path.relative_to(output).as_posix())
            kept = archive if path == current else archive.get('kept') or {}
            for stage, entry in kept.items():
                # The old task must come from the same historical derivation, not
                # from today's declaration. Changed/overwritten logs cannot fill it.
                directory = safe(output, 'execution_interfaces/' + stage)
                for raw in sorted(directory.glob('*.json'))[-24:]:
                    interface = _read(output, raw.relative_to(output).as_posix())
                    if (interface.get('id') != raw.stem
                            or object_digest({k: v for k, v in interface.items() if k != 'id'}) != raw.stem
                            or interface.get('stage') != stage
                            or interface.get('stage_sha256') != object_digest(entry)):
                        continue
                    log_path = safe(output, interface['derivation_ref'])
                    if digest(log_path) != interface.get('derivation_sha256'):
                        continue
                    arguments = _read(output, interface['derivation_ref']).get('arguments') or {}
                    task = arguments.get('task')
                    if not isinstance(task, str):
                        continue
                    imported.append(remember(output, repo, stage, entry,
                                             interpreter=interpreter, task=task)['ref'])
                    break
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue  # No guessed task/runtime, no historical approval transfer.
    return imported


def catalog(output: Path) -> list[dict]:
    rows = []
    try:
        directory = safe(output, 'invocation_blueprints')
    except (OSError, ValueError):
        return []
    files = []
    for path in directory.glob('*.json'):
        try:
            if not path.is_symlink():
                files.append((path.stat().st_mtime, path))
        except OSError:
            continue
    for _, path in sorted(files, reverse=True)[:12]:
        try:
            doc = _read(output, path.relative_to(output).as_posix(), limit=256 * 1024)
            if (doc.get('id') != path.stem
                    or object_digest({k: v for k, v in doc.items() if k != 'id'}) != path.stem):
                continue
            rows.append({k: doc.get(k) for k in
                         ('id', 'stage', 'task', 'status', 'authority', 'native_evidence_ref')} |
                        {'ref': path.relative_to(output).as_posix(),
                         'entrypoint': doc['entry']['row'].get('entrypoint'),
                         'environment_keys': sorted((doc['entry']['row'].get('environment') or {}).keys()),
                         'runtime_fields': sorted(doc['entry']['row']),
                         'proposed_operation': {'do': 'derive_a_command',
                             'arguments': {'stage': doc['stage'], 'blueprint_ref': path.stem}},
                         'selection': 'Main chooses; not automatic reuse or native readiness'})
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return rows
