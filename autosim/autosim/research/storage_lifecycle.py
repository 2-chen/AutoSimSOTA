"""Run-owned payload registry and reviewed, operator-authorized garbage collection.

The model supplies semantic interpretation; the kernel owns IDs, byte inventories,
references, paths, leases and deletion transactions. No age/filename-only GC.
"""
import argparse
import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from .common import atomic_json, digest, now, object_digest, read_json


@contextmanager
def locked(output):
    root = Path(output).resolve()/ 'storage'
    if root.is_symlink():
        raise ValueError('unsafe storage directory')
    root.mkdir(parents=True, exist_ok=True)
    with safe(output,'storage/lock').open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield root


def safe(output, ref):
    root = Path(output).resolve()
    relative = Path(str(ref))
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('payload references must be run-relative')
    path = root/relative
    for ancestor in [path, *path.parents]:
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError('payload path contains a symlink')
    if not path.resolve().is_relative_to(root) or path == root:
        raise ValueError('payload escaped run')
    return path


def configure(output, *, auto_gc=False):
    """Operator-only API/CLI. Never exposed as an Agent operation."""
    if not isinstance(auto_gc, bool):
        raise ValueError('automatic GC authorization must be a boolean')
    with locked(output) as root:
        row = {'schema_version':1, 'auto_gc':bool(auto_gc),
               'authorized_scope':'reviewed unscored obsolete run-native payloads only'}
        atomic_json(safe(output,'storage/policy.json'), row)
    return row


def policy(output):
    path = Path(output)/'storage/policy.json'
    if not path.exists():
        return {'auto_gc':False}
    return read_json(safe(output,'storage/policy.json'))


def idle(output):
    from .native_jobs import active_jobs
    from .process_executor import inspect_process_identity
    for path in Path(output).glob('storage/stage_leases/*.json'):
        row = read_json(safe(output,path.relative_to(output)))
        if inspect_process_identity(row['identity']).get('status') != 'not_running':
            raise ValueError('live/unknown native stage lease prevents retention changes')
    if active_jobs(Path(output)):
        raise ValueError('active or unresolved native jobs prevent retention changes')
    for path in Path(output).glob('processes/*.json'):
        row = read_json(safe(output,path.relative_to(output)))
        identity = row.get('process_identity') or row.get('identity')
        if identity and inspect_process_identity(identity).get('status') != 'not_running':
            raise ValueError('live/unknown process prevents payload mutation')
    for path in Path(output).glob('research/*/controller_session.json'):
        row = read_json(safe(output,path.relative_to(output)))
        if row.get('status') == 'running' or row.get('pending_action'):
            raise ValueError('running/unknown research action prevents payload mutation')


@contextmanager
def stage_lease(output):
    """Admission and GC share one lock, including synchronous native stages.

    Dead leases do not grant permission to ignore other receipts/jobs. Unknown
    identity remains blocked; there is no automatic signalling of a PID.
    """
    from .process_executor import _proc_identity
    identity = _proc_identity(os.getpid())
    if not identity:
        raise ValueError('native stage owner identity unavailable')
    identity['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    ref = f'storage/stage_leases/{uuid.uuid4().hex}.json'
    with locked(output):
        if any(read_json(safe(output,p.relative_to(output))).get('status')=='committing'
               for p in Path(output).glob('storage/plans/*.json')):
            raise ValueError('incomplete retention transaction requires reconciliation')
        atomic_json(safe(output,ref),{'at':now(),'identity':identity})
    try:
        yield
    finally:
        with locked(output):
            safe(output,ref).unlink(missing_ok=True)


def inventory(output, refs):
    if not isinstance(refs,list) or not refs or len(refs)>256:
        raise ValueError('offer 1..256 exact file references')
    rows = []
    for ref in sorted(set(refs)):
        path = safe(output,ref)
        if not path.is_file():
            raise ValueError('payload must be an existing regular file')
        stat = path.stat()
        rows.append({'ref':str(ref),'bytes':stat.st_size, 'sha256':digest(path),
                     'device':stat.st_dev,'inode':stat.st_ino,'mtime_ns':stat.st_mtime_ns})
    return rows


def native_owner(output, evidence_id):
    from .evidence_store import read_attempt_evidence
    native = read_attempt_evidence(Path(output),evidence_id,limit=3000)
    if native.get('status') not in {'completed','failed'}:
        raise ValueError('native evidence has unresolved outcome')
    receipt = read_json(safe(output,native['receipt_ref']))
    if receipt.get('stage',receipt.get('node_id')) not in {'train','evaluate','record_demo'}:
        raise ValueError('data producers are not eligible for automatic payload GC')
    if receipt.get('evidence_id',receipt.get('attempt_id',receipt.get('id')))!=evidence_id:
        raise ValueError('native receipt identity disagrees with sealed evidence')
    raw = Path(str(receipt.get('output_directory') or receipt.get('verification_output') or ''))
    if not raw.is_absolute() or not raw.resolve().is_relative_to(Path(output).resolve()):
        raise ValueError('sealed native receipt has no run-local producer root')
    if not any(raw.resolve().is_relative_to(Path(output).resolve()/name)
               for name in ('research','verification_outputs')):
        raise ValueError('producer root is not a native research/verification output')
    native['receipt_facts']={key:receipt.get(key) for key in ('stage','node_id','argv',
        'status','returncode','termination_reason','training_progress','settings_digest','storage_pressure')}
    return raw.resolve(), native


def protected(output):
    """Automatic pins from authoritative measurements/handoffs/data/baseline records."""
    paths = set()
    patterns = ['research/*/measurements/*.json','research/*/best.json',
                'research/*/confirmation.json','verification_checkpoint_handoffs/*.json',
                'data_versions/**/*.json','baseline_references/**/*.json',
                'research/*/media/*.json','report/demo_requests.json']
    def visit(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key in {'path','source','selected_path','checkpoint_ref','declared_path','root','data_root','ref'} and isinstance(child,str):
                    path = Path(child)
                    if not path.is_absolute(): path = Path(output)/path
                    resolved = path.resolve()
                    if resolved.is_relative_to(Path(output).resolve()): paths.add(resolved)
                visit(child)
        elif isinstance(value,list):
            for child in value: visit(child)
    for pattern in patterns:
        for path in Path(output).glob(pattern):
            visit(read_json(safe(output,path.relative_to(output))))
    from .checkpoint_handoff import catalog as handoffs
    handoff=handoffs(Path(output))
    if handoff.get('status')=='available':
        paths.add((Path(output)/handoff['output_ref']).resolve())
    return paths


def validate_registration(output, *, evidence_id, refs, role, reason):
    if not isinstance(refs,list) or not refs or len(refs)>256 or not all(isinstance(ref,str) for ref in refs):
        raise ValueError('offer 1..256 exact file references')
    if role not in {'intermediate','terminal','resume','verification'} or not str(reason).strip():
        raise ValueError('payload role and applicability rationale required')
    owner, native = native_owner(output,evidence_id)
    if role=='verification' and not owner.is_relative_to(Path(output).resolve()/'verification_outputs'):
        raise ValueError('verification retention role requires a verification producer')
    for ref in refs:
        path = safe(output,ref)
        if not path.resolve().is_relative_to(owner):
            raise ValueError('payload is not owned by the sealed native attempt')
        if set(path.relative_to(Path(output)).parts) & {'artifact_store','export','snapshots','measurements',
                'attempts','training_progress_audits','policy_consumption_audits'}:
            raise ValueError('framework archives and evidence are not native GC payloads')
        if path.suffix.lower() in {'.json','.jsonl','.md','.yaml','.yml','.log','.csv','.py','.txt','.toml','.sh'}:
            raise ValueError('metadata, evidence and source cannot be garbage-collected')
    return owner,native


def register(output, *, evidence_id, refs, role, reason, source_evidence):
    validate_registration(output,evidence_id=evidence_id,refs=refs,role=role,reason=reason)
    with locked(output) as root:
        idle(output)
        rows = inventory(output,refs)
        identity = object_digest({'evidence_id':evidence_id,'files':rows})
        record = {'id':identity,'at':now(),'evidence_id':evidence_id,'files':rows,
                  'bytes':sum(r['bytes'] for r in rows),'role':role,'reason':reason,
                  'source_evidence':source_evidence,'status':'retained'}
        path = root/'artifacts'/f'{identity}.json'
        if path.exists():
            old=read_json(safe(output,path.relative_to(output)))
            if old['role']!=role or old['source_evidence']!=source_evidence:
                raise ValueError('registered payload cannot be silently reclassified')
            return old
        atomic_json(safe(output,path.relative_to(output)),record)
    return record


def catalog(output):
    rows = []
    pins = protected(output)
    # A measured file may also have a hard-link alias under another producer.
    # Never reclaim that alias based solely on different path spelling.
    pin_inodes = set()
    for pin in pins:
        if pin.is_file():
            stat=pin.stat()
            pin_inodes.add((stat.st_dev,stat.st_ino))
    for path in Path(output).glob('storage/artifacts/*.json'):
        row = read_json(safe(output,path.relative_to(output)))
        if object_digest({'evidence_id':row['evidence_id'],'files':row['files']})!=row['id']:
            raise ValueError('registered payload identity changed')
        linked=any(safe(output,f['ref']).exists() and safe(output,f['ref']).stat().st_nlink>1
                   for f in row['files'])
        pinned = linked or row['role'] in {'terminal','resume'} or any(
            (file['device'],file['inode']) in pin_inodes or any(safe(output,file['ref']).resolve() == p or safe(output,file['ref']).resolve().is_relative_to(p)
                for p in pins) for file in row['files'])
        rows.append({key:row[key] for key in ('id','role','bytes','status','reason')})
        rows[-1].update(protected=pinned, file_refs=[r['ref'] for r in row['files']])
    return rows


def plan(output, *, artifact_ids, reason, review):
    """Independent Supervisor verdict is mandatory; operator policy gates commit."""
    if (not isinstance(artifact_ids,list) or not artifact_ids or len(artifact_ids)>64 or
            not all(isinstance(identity,str) and len(identity)==64 and
                    all(c in '0123456789abcdef' for c in identity) for identity in artifact_ids)):
        raise ValueError('select 1..64 registered IDs')
    if not reason.strip() or review.get('verdict') != 'verified' or review.get('unneeded_for_recovery') is not True:
        raise ValueError('source-backed independent retention review required')
    if sorted(review.get('artifact_ids') or []) != sorted(set(artifact_ids)):
        raise ValueError('retention review must bind exact artifact IDs')
    with locked(output) as root:
        idle(output)
        offered = {r['id']:r for r in catalog(output)}
        rows = []
        for identity in sorted(set(artifact_ids)):
            row = offered.get(identity)
            if not row or row['protected'] or row['status'] != 'retained':
                raise ValueError('artifact is unknown, protected, or no longer retained')
            rows.append(read_json(safe(output,f'storage/artifacts/{identity}.json')))
        # Native intermediate snapshots need a separately registered, pinned
        # terminal/recovery group from that producer. Verification-only obsolete
        # outputs may be reclaimed after the independent review and handoff pins.
        for row in rows:
            if row['role']=='intermediate' and not any(
                    entry['role'] in {'terminal','resume'} and entry['status']=='retained'
                    and read_json(safe(output,f"storage/artifacts/{entry['id']}.json"))['evidence_id']==row['evidence_id']
                    for entry in offered.values()):
                raise ValueError('register and protect producer terminal/recovery payload before intermediate GC')
        identity = object_digest({'artifact_ids':sorted(set(artifact_ids)),
                                  'inventories':rows,'reason':reason,'review':review})
        result = {'id':identity,'at':now(),'artifact_ids':sorted(set(artifact_ids)),
                  'reason':reason,'review':review,'inventories':rows,'status':'planned',
                  'bytes':sum(r['bytes'] for r in rows)}
        path=safe(output,f'storage/plans/{identity}.json')
        if path.exists(): return {k:read_json(path)[k] for k in ('id','artifact_ids','status','bytes')}
        atomic_json(path,result)
    return {key:result[key] for key in ('id','artifact_ids','status','bytes')}


def commit(output, *, plan_id, repo=None):
    if not isinstance(plan_id,str) or len(plan_id)!=64 or any(c not in '0123456789abcdef' for c in plan_id):
        raise ValueError('unsafe retention plan ID')
    with locked(output) as root:
        if policy(output).get('auto_gc') is not True:
            return {'status':'authorization_required','reason':'operator must enable run-scoped auto GC'}
        idle(output)
        path = safe(output,f'storage/plans/{plan_id}.json')
        record = read_json(path)
        if object_digest({k:record[k] for k in ('artifact_ids','inventories','reason','review')})!=plan_id:
            raise ValueError('retention plan identity changed')
        if record['status'] != 'planned':
            return {'status':record['status'],'reason':'transaction already started; do not blindly replay'}
        from .storage_controls import source_path
        for source in record['review'].get('sources') or []:
            code=source_path(output,repo or Path(output)/'checkout',source['path'])
            if digest(code)!=source['sha256']:
                raise ValueError('retention source changed; obtain a new independent review')
        current = {r['id']:r for r in catalog(output)}
        # Revalidate every retained terminal/resume pin before deleting anything.
        for entry in current.values():
            if entry['role'] in {'terminal','resume'} and entry['status']=='retained':
                pin=read_json(safe(output,f"storage/artifacts/{entry['id']}.json"))
                if inventory(output,[r['ref'] for r in pin['files']])!=pin['files']:
                    raise ValueError('protected terminal/recovery payload changed')
        for row in record['inventories']:
            offered = current.get(row['id'])
            if not offered or offered['protected'] or offered['status']!='retained':
                raise ValueError('references changed since retention planning')
            if inventory(output,[r['ref'] for r in row['files']]) != row['files']:
                raise ValueError('payload bytes or identity changed since retention planning')
        record.update(status='committing',removed_refs=[])
        atomic_json(path,record)
        for row in record['inventories']:
            for file in row['files']:
                safe(output,file['ref']).unlink()
                record['removed_refs'].append(file['ref'])
                atomic_json(path,record)
            updated={**row,'status':'pruned','pruned_at':now(),'plan_id':plan_id}
            atomic_json(safe(output,f"storage/artifacts/{row['id']}.json"),updated)
        record.update(status='completed',completed_at=now())
        atomic_json(path,record)
        return {'status':'completed','removed_bytes':record['bytes'],
                'manifest_ref':path.relative_to(Path(output)).as_posix(),
                'recoverability':'deleted payloads are not recoverable; evidence and protected artifacts retained'}


def reconcile(output, *, plan_id):
    """Account for an interrupted transaction, never replay its deletions."""
    if not isinstance(plan_id,str) or len(plan_id)!=64 or any(c not in '0123456789abcdef' for c in plan_id):
        raise ValueError('unsafe retention plan ID')
    with locked(output):
        idle(output)
        path=safe(output,f'storage/plans/{plan_id}.json')
        row=read_json(path)
        if object_digest({k:row[k] for k in ('artifact_ids','inventories','reason','review')})!=plan_id:
            raise ValueError('retention plan identity changed')
        if row['status']!='committing':
            return {'status':row['status'],'reason':'no interrupted deletion to reconcile'}
        removed=[]
        remaining=[]
        for group in row['inventories']:
            for file in group['files']:
                target=safe(output,file['ref'])
                if not target.exists(): removed.append(file['ref'])
                else:
                    if inventory(output,[file['ref']])!=[file]:
                        raise ValueError('remaining transaction payload changed; human review required')
                    remaining.append(file['ref'])
            status='pruned' if all(f['ref'] in removed for f in group['files']) else 'partially_pruned' if any(f['ref'] in removed for f in group['files']) else 'retained'
            atomic_json(safe(output,f"storage/artifacts/{group['id']}.json"),{**group,'status':status,'plan_id':plan_id})
        row.update(status='reconciled_partial' if remaining else 'completed',removed_refs=removed,
                   remaining_refs=remaining,reconciled_at=now())
        atomic_json(path,row)
        return {'status':row['status'],'removed_refs':removed,'remaining_refs':remaining,
                'reason':'accounted only; no deletion replayed. Register remaining payloads for a new review.'}


def pruned_refs(output):
    refs = set()
    for path in Path(output).glob('storage/plans/*.json'):
        row = read_json(safe(output,path.relative_to(output)))
        refs.update(row.get('removed_refs') or [])
    return refs


def main():
    parser = argparse.ArgumentParser(description='Operator storage authorization, not an Agent tool')
    parser.add_argument('output',type=Path)
    parser.add_argument('--auto-gc',choices=['enable','disable'],required=True)
    args=parser.parse_args()
    print(json.dumps(configure(args.output,auto_gc=args.auto_gc=='enable')))


if __name__ == '__main__': main()
