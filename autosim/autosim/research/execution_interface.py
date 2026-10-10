"""Versioned accepted stage adapters, reusable only with exact bound evidence.

No repository names, guessed paths or automatic cache adoption. Init/LLM still
creates the adapter. Reuse avoids derivation calls, not native runtime checks.
"""
from pathlib import Path
import re

from .common import atomic_json, digest, object_digest, read_json
from .experiment_transaction import safe
from .native_jobs import source_identity


def _read(output, relative):
    path = safe(output, relative)
    if path.stat().st_size > 8*1024**2:
        raise ValueError('oversized interface evidence')
    return read_json(path)


def _bindings(output):
    # Explicit mount manifest, not a claim that every dataset byte was read.
    doc = _read(output, 'workspace_snapshot.json')
    return object_digest(doc.get('resource_bindings', doc.get('resources', [])))


def _inputs(output, repo, stage):
    return {'source_identity':source_identity(output, repo, stage_scope=[stage]),
            'environment_sha256':digest(safe(output, 'environment.json')),
            'bindings_sha256':_bindings(output),
            'stage_sha256':object_digest(_read(output, 'derived_stages.json')[stage])}


def publish(output: Path, repo: Path, stage: str) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', stage):
        raise ValueError('unsafe interface stage')
    log_ref = 'derivation_attempts/'+stage+'.json'
    log = _read(output, log_ref)
    accepted = next((r for r in reversed(log.get('attempts') or []) if r.get('status')=='accepted'), None)
    if not accepted or not accepted.get('native_evidence_ref'):
        raise ValueError('interface lacks an accepted native verification receipt')
    from .evidence_store import read_attempt_evidence
    evidence_ref = accepted['native_evidence_ref']
    if not re.fullmatch(r'evidence/[0-9a-f]{32}\.json', evidence_ref):
        raise ValueError('unsafe interface verification ID')
    identity = Path(evidence_ref).stem
    proof = read_attempt_evidence(output, identity, limit=1)
    startup = accepted.get('verification_scope') == 'training_startup_only'
    if proof['status'] != ('timed_out' if startup else 'completed'):
        raise ValueError('interface native verification status differs')
    if not startup and (type(proof.get('returncode')) is not int or proof['returncode']!=0):
        raise ValueError('interface native verification did not exit successfully')
    body = {'schema_version':1,'stage':stage, **_inputs(output, repo, stage),
            'derivation_ref':log_ref, 'derivation_sha256':digest(safe(output, log_ref)),
            'native_evidence_ref':evidence_ref,
            'verification_scope':accepted.get('verification_scope','completed_stage'),
            'authority':'accepted adapter under these bindings; not baseline, dataset consumption or convergence'}
    key = object_digest(body)
    path = safe(output, 'execution_interfaces/'+stage+'/'+key+'.json')
    if path.exists() and _read(output, path.relative_to(output).as_posix()) != {**body,'id':key}:
        raise ValueError('immutable execution interface changed')
    if not path.exists():
        atomic_json(path, {**body,'id':key})
    return {**body,'id':key,'ref':path.relative_to(output).as_posix()}


def reuse(output: Path, repo: Path, stage: str) -> dict:
    if not isinstance(stage, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',stage):
        raise ValueError('unsafe interface stage')
    current = _inputs(output, repo, stage)
    for raw in sorted((output/'execution_interfaces'/stage).glob('*.json')):
        relative = raw.relative_to(output).as_posix()
        row = _read(output, relative)
        if row.get('id') != raw.stem or row.get('id') != object_digest({k:v for k,v in row.items() if k!='id'}):
            raise ValueError('execution interface identity changed')
        if any(row.get(k) != v for k,v in current.items()):
            continue
        # Recheck sealed log bytes, not just presence of an old cache filename.
        if digest(safe(output, row['derivation_ref'])) != row['derivation_sha256']:
            continue
        from .evidence_store import read_attempt_evidence
        proof = read_attempt_evidence(output, Path(row['native_evidence_ref']).stem, limit=1)
        if proof.get('status') != ('timed_out' if row['verification_scope']=='training_startup_only' else 'completed'):
            raise ValueError('execution interface proof changed')
        return {**row,'ref':relative}
    raise ValueError('no exact verified interface is reusable; inspect changed source/bindings or derive explicitly')


def catalog(output: Path) -> list[dict]:
    rows = []
    for raw in sorted((output/'execution_interfaces').glob('*/*.json'))[-12:]:
        try:
            row = _read(output, raw.relative_to(output).as_posix())
            if (row.get('id') != raw.stem or row.get('stage') != raw.parent.name or
                    row.get('id') != object_digest({k:v for k,v in row.items() if k!='id'})):
                continue
            rows.append({'stage':row['stage'],'id':row['id'],
                'ref':raw.relative_to(output).as_posix(),'verification_scope':row['verification_scope'],
                'status':'historically_verified; current reuse requires exact recheck',
                'proposed_operation':{'do':'derive_a_command',
                    'arguments':{'stage':row['stage'],'reuse_verified':True}}})
        except (OSError,ValueError,TypeError,KeyError):
            continue
    return rows
