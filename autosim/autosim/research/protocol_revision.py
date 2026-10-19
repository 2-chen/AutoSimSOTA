"""Reviewed, prospective repair of a pre-search freeze; never rewrite old protocols.

A failed guard is not an observed rollout. Unknown/running/completed operations
are nevertheless barriers, not permission to retroactively change evaluation.
"""
from pathlib import Path
import difflib
import hashlib
import re

from .common import atomic_json, digest, immutable_json, now, object_digest, read_json

CHECKS = ('presearch_only', 'instrumentation_or_setup_only', 'preserves_task_distribution',
          'preserves_metric_and_success', 'preserves_reserved_conditions', 'sources_supported')


def safe(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    if not path.is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)):
        raise ValueError('unsafe protocol revision path')
    return path


def record(root, path):
    path = safe(root, path)
    if not path.is_file() or path.stat().st_size > 2*1024**2:
        raise ValueError('protocol record missing or oversized')
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError('protocol record must be an object')
    return value


def effective(root):
    """Verify the whole immutable chain, not merely a mutable active pointer."""
    root = Path(root)
    original = safe(root, root/'protocol_frozen.json')
    pointer = safe(root, root/'active_protocol_revision.json')
    if not pointer.exists():
        return original, None
    identity = record(root,pointer).get('id')
    seen, chain = set(), []
    while identity:
        if not isinstance(identity,str) or not re.fullmatch(r'[0-9a-f]{64}',identity) or identity in seen or len(seen)>=8:
            raise ValueError('invalid protocol revision chain')
        seen.add(identity)
        directory=root/'protocol_revisions'/identity
        row=record(root,directory/'revision.json')
        if row.get('id')!=identity or object_digest({k:v for k,v in row.items() if k!='id'})!=identity:
            raise ValueError('protocol revision identity changed')
        if any((row.get('review') or {}).get(k) is not True for k in CHECKS):
            raise ValueError('protocol revision lacks independent approval')
        planned={k:v for k,v in row.items() if k not in
                 {'id','review','reason','at','frozen_sha256','formal_paper_eligible','plan_id'}}
        if (object_digest(planned)!=row.get('plan_id') or
                row['review'].get('plan_id')!=row.get('plan_id')):
            raise ValueError('review does not identify the exact revision plan')
        frozen=record(root,directory/'protocol_frozen.json')
        if object_digest(frozen)!=row.get('frozen_sha256') or frozen.get('hashes')!=row.get('hashes'):
            raise ValueError('revised frozen source identity changed')
        chain.append((directory,row,frozen))
        identity=row.get('parent_revision_id')
    parent_path=original
    for directory,row,frozen in reversed(chain):
        if digest(parent_path)!=row.get('parent_file_sha256'):
            raise ValueError('protocol revision parent bytes changed')
        previous=record(root,parent_path).get('hashes')
        if not isinstance(previous,dict) or set(previous)!=set(frozen.get('hashes') or {}):
            raise ValueError('revision cannot drop or add protected files')
        parent_path=directory/'protocol_frozen.json'
    return chain[0][0]/'protocol_frozen.json',chain[0][1]


def prestart_measurement(root, row):
    if row.get('ok') is not False or row.get('confirmation'):
        return False
    if row.get('native_operation_status')=='not_started' and row.get('ran') is False:
        identity=row.get('evidence_id')
        if not identity:return False
        from .evidence_store import read_attempt_evidence
        proof=read_attempt_evidence(Path(root).parents[1],identity)
        return '"native_operation_status": "not_started"' in proof['text']
    evaluation=row.get('evaluate') or {}
    attempt=evaluation.get('attempt_id')
    if not isinstance(attempt,str) or not re.fullmatch(r'[0-9a-f]{32}',attempt):return False
    receipt=record(root,Path(root)/'attempts'/attempt/'receipt.json')
    # Legacy guard receipts lacked ran/command_started. Their typed guard terminal
    # is produced before staging/launch; any evidence of launch still vetoes it.
    return (receipt.get('attempt_id')==attempt and receipt.get('status')=='blocked'
        and receipt.get('termination_reason')=='protocol_changed' and receipt.get('returncode') is None
        and receipt.get('ran') is not True and receipt.get('command_started') is not True
        and not receipt.get('pid') and not receipt.get('process_identity'))


def boundary(root):
    root=Path(root)
    for name in ('confirmation_attempts','measurements','attempts','reference_protocols'):
        safe(root,root/name)
    if any(safe(root,root/name).exists() for name in ('controller_session.json','comparison_protocol.json')):
        raise ValueError('local research session/protocol already initialized')
    if any((root/'confirmation_attempts').glob('*.json')) or (root/'confirmation.json').exists():
        raise ValueError('held-out boundary already exposed')
    facts={}
    for path in sorted((root/'measurements').glob('*.json')):
        if not prestart_measurement(root,record(root,path)):
            raise ValueError('measurement ran or its outcome is unknown; preserve study and use a new run')
        facts[path.relative_to(root).as_posix()]=digest(path)
    # Include detached attempts even when they have not produced measurements.
    for path in sorted((root/'attempts').glob('*/receipt.json')):
        row=record(root,path)
        if row.get('node_id')!='evaluate':continue
        if not (row.get('status')=='blocked' and row.get('termination_reason')=='protocol_changed'
                and row.get('returncode') is None and row.get('ran') is not True
                and row.get('command_started') is not True and not row.get('pid') and not row.get('process_identity')):
            raise ValueError('native evaluation started or is unresolved')
        facts[path.relative_to(root).as_posix()]=digest(path)
    return facts


def plan(output, repo, run_id, current_hashes):
    from .native_jobs import active_jobs
    output,repo=Path(output),Path(repo).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',run_id):raise ValueError('invalid protocol namespace')
    if active_jobs(output):raise ValueError('native jobs still own workspace')
    root=output/'research'/run_id
    facts=boundary(root)
    parent,previous=effective(root)
    old=record(root,parent).get('hashes')
    if not isinstance(old,dict) or not old or set(old)!=set(current_hashes):
        raise ValueError('revision requires the same nonempty protected source set')
    changed=[key for key in old if old[key]!=current_hashes[key]]
    if not 1<=len(changed)<=4:raise ValueError('revision needs 1..4 changed protected files')
    snapshot=record(output,output/'workspace_snapshot.json')
    if Path(snapshot.get('destination','')).resolve()!=repo:
        raise ValueError('isolated workspace identity differs')
    source=Path(snapshot['source']).resolve()
    changes=[]
    for key in sorted(changed):
        current=safe(repo,Path(key))
        if current.stat().st_size>256*1024 or digest(current)!=current_hashes[key]:
            raise ValueError('changed source is oversized or unstable')
        rel=current.relative_to(repo)
        original=safe(source,source/rel)
        # The original source must still have the exact old frozen bytes.
        # Missing historical bytes require a new study, not reconstruction by LLM.
        if not original.is_file() or original.stat().st_size>256*1024 or digest(original)!=old[key]:
            raise ValueError('authentic old protected source bytes unavailable')
        before,after=original.read_text(),current.read_text()
        diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                                        fromfile='original/'+rel.as_posix(),tofile='current/'+rel.as_posix()))
        if len(diff.encode())>48000:raise ValueError('source change exceeds review window')
        changes.append({'ref':rel.as_posix(),'before_sha256':old[key],
            'after_sha256':current_hashes[key],'diff':diff})
    refs={p.relative_to(root).as_posix():digest(p) for p in sorted((root/'reference_protocols').glob('*.json'))
          if safe(root,p).is_file()}
    payload={'schema_version':1,'run_id':run_id,'parent_revision_id':previous['id'] if previous else None,
        'parent_file_sha256':digest(parent),'hashes':current_hashes,'changes':changes,
        'prestart_evidence':facts,'historical_reference_protocols':refs,
        'authority':'prospective development protocol only; old failures/costs remain, no score promotion'}
    payload['plan_id']=object_digest(payload)
    return payload


def activate(output,repo,run_id,planned,review,reason):
    """Only executor writes a new immutable version after independent review."""
    if not isinstance(reason,str) or not reason.strip() or len(reason)>1000:raise ValueError('revision rationale required')
    if any(review.get(k) is not True for k in CHECKS) or review.get('plan_id')!=planned['plan_id']:
        raise ValueError('independent review did not approve this exact protocol revision')
    # Repeat the full native/boundary/source checks immediately before committing.
    hashes={key:digest(Path(key)) for key in planned['hashes']}
    if plan(output,repo,run_id,hashes)!=planned:raise ValueError('protocol revision inputs changed during review')
    root=Path(output)/'research'/run_id
    frozen={'schema_version':1,'hashes':hashes,'created_at':now(),
            'freeze_reason':'reviewed prospective pre-search setup repair'}
    row={**planned,'review':review,'reason':reason,'at':now(),'frozen_sha256':object_digest(frozen),
         'formal_paper_eligible':False}
    row['id']=object_digest(row)
    directory=safe(root,root/'protocol_revisions'/row['id'])
    immutable_json(directory/'protocol_frozen.json',frozen)
    immutable_json(directory/'revision.json',row)
    atomic_json(safe(root,root/'active_protocol_revision.json'),{'id':row['id']})
    effective(root)
    return row


def reference_path(root,reference_id):
    _,revision=effective(root)
    name=reference_id if revision is None else ('v'+revision['id'][:16]+'_'+
        hashlib.sha256(reference_id.encode()).hexdigest()[:16])
    return Path(root)/'reference_protocols'/(name+'.json')


def view(output,repo,run_id,current_hashes):
    try:
        planned=plan(output,repo,run_id,current_hashes)
        return {'status':'review_available','plan_id':planned['plan_id'],
            'changes':[{k:c[k] for k in ('ref','before_sha256','after_sha256')} for c in planned['changes']],
            'next_operation':'revise_presearch_protocol','authority':planned['authority']}
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        return {'status':'not_available','why':str(exc)[:500]}
