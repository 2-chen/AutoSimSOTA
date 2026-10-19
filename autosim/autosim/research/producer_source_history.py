"""Read-only candidate source history addressed by a sealed training producer.

Historical bytes are not an installed patch, current approval or native capability.
"""
import hashlib
import json
from pathlib import Path
import re
import uuid

from .common import atomic_text, digest, immutable_json, now, object_digest, read_json, sanitize_model_payload
from .experiment_transaction import safe

SUFFIXES = {'.py', '.pyi', '.sh', '.yaml', '.yml', '.toml', '.json', '.md', '.rst', '.cfg', '.ini'}


def _json(output, ref, maximum=2*1024**2):
    path = safe(output, ref)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('historical source metadata is missing or oversized')
    raw = path.read_bytes()
    if len(raw) > maximum:
        raise ValueError('historical source metadata exceeds its bound')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('historical source metadata must be an object')
    return value, hashlib.sha256(raw).hexdigest()


def _blob(output, base, sha):
    if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('invalid historical source blob identity')
    path = safe(output, base+'/snapshots/blobs/'+sha)
    if not path.is_file() or path.stat().st_size > 2*1024**2:
        raise ValueError('historical source blob missing or oversized')
    raw = path.read_bytes()
    if len(raw) > 2*1024**2 or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('historical source blob changed')
    return raw


def load_view(output, repo, request):
    """Validate and read archives only. This function never captures/edits anything."""
    from .codepatch import patches_from_change
    from .evidence_store import read_attempt_evidence
    from .workspace_resources import bindings_for
    if (not isinstance(request, dict) or set(request)-{'ref','producer_evidence_id','view'}
            or not {'ref','producer_evidence_id'} <= set(request)):
        raise ValueError('producer source needs ref, producer_evidence_id and optional source|patch view')
    producer, ref, view = request['producer_evidence_id'], request['ref'], request.get('view','source')
    if (not isinstance(producer, str) or not re.fullmatch('[0-9a-f]{32}', producer)
            or not isinstance(ref, str) or not 1 <= len(ref) <= 1024 or view not in ('source','patch')):
        raise ValueError('invalid producer source identity/reference/view')
    match = re.fullmatch(r'(.+):([1-9][0-9]*)(?:-([1-9][0-9]*))?', ref)
    name = match[1] if match else ref
    relative = Path(name)
    if (relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name
            or relative.as_posix() != name or relative.suffix.lower() not in SUFFIXES):
        raise ValueError('historical source requires a confined public source/configuration path')
    output, repo = Path(output).resolve(), Path(repo).resolve()
    manifest, _ = _json(output, 'workspace_snapshot.json', maximum=8*1024**2)
    if (Path(str(manifest.get('destination') or '')).resolve() != repo or name not in {
            row[1] for row in manifest.get('source_entries', [])
            if isinstance(row,list) and len(row)>1 and row[0]=='file'}):
        raise ValueError('historical source must belong to the isolated public source manifest')
    if any((repo/name).is_relative_to(repo/row['target']) for row in bindings_for(repo)):
        raise ValueError('historical source cannot read a dataset/checkpoint/resource binding')
    proof = read_attempt_evidence(output, producer, limit=1)
    receipt, receipt_sha = _json(output, proof['receipt_ref'])
    run_id = receipt.get('run_id')
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',run_id):
        raise ValueError('historical source producer run identity is invalid')
    base = 'research/'+run_id
    if (proof['receipt_ref'] != base+'/attempts/'+producer+'/receipt.json'
            or receipt.get('attempt_id') != producer or receipt.get('stage') != 'train'
            or receipt.get('status') not in ('completed','failed')
            or proof.get('status') != receipt['status']
            or proof.get('returncode') != receipt.get('returncode')):
        raise ValueError('historical source needs a sealed terminal training producer from this run')
    session, _ = _json(output, base+'/controller_session.json')
    if Path(str(session.get('repository') or '')).resolve() != repo:
        raise ValueError('historical source producer belongs to a different checkout')
    paths = list(safe(output,base+'/measurements').glob('round_*.json'))
    if len(paths)>256:
        raise ValueError('historical producer measurement lookup exceeds bound')
    rounds = set()
    for path in paths:
        row, _ = _json(output,path.relative_to(output).as_posix(),maximum=8*1024**2)
        label = row.get('label')
        matched = re.fullmatch(r'round_([1-9][0-9]*)(?:_handoff_correction)?',str(label or ''))
        if (matched and path.stem == label and not row.get('confirmation')
                and (row.get('train') or {}).get('attempt_id') == producer):
            rounds.add(int(matched[1]))
    if len(rounds)!=1:
        raise ValueError('producer has no unique archived formal candidate round')
    number = rounds.pop()
    change_ref = base+'/code_changes/round-'+str(number)+'.json'
    change, change_sha = _json(output,change_ref,maximum=256*1024)
    if change.get('status')!='applied' or change.get('round')!=number:
        raise ValueError('historical source transaction is not an applied candidate record')
    patches = [p for p in patches_from_change({'patches':change.get('patches')}) if p.file==name]
    index, _ = _json(output,base+'/snapshots/snapshots.json')
    rows = index.get('snapshots')
    if not isinstance(rows,list):
        raise ValueError('historical source snapshot index is invalid')
    before = [r for r in rows if isinstance(r,dict) and r.get('name')==change.get('snapshot')]
    after = [r for r in rows if isinstance(r,dict) and r.get('name')=='round-'+str(number)]
    if len(before)!=1 or len(after)!=1 or not patches:
        raise ValueError('requested source has no unique archived patch/preimage/postimage')
    old_sha, new_sha = before[0]['files'].get(name), after[0]['files'].get(name)
    old, new = _blob(output,base,old_sha), _blob(output,base,new_sha)
    # Match the text semantics used by codepatch/read_text, including universal newlines.
    rebuilt = old.decode('utf-8').replace('\r\n','\n').replace('\r','\n')
    for patch in patches:
        if rebuilt.count(patch.find)!=1:
            raise ValueError('historical patch preimage is ambiguous')
        rebuilt = rebuilt.replace(patch.find,patch.replace,1)
    if hashlib.sha256(rebuilt.encode()).hexdigest()!=new_sha:
        raise ValueError('historical patches do not reproduce the archived source postimage')
    if view=='patch':
        if match:
            raise ValueError('patch view selects a whole changed file, not a line range')
        text = json.dumps([{'file':p.file,'find':p.find,'replace':p.replace} for p in patches],ensure_ascii=False,indent=2)
        line_range = None
    else:
        lines = new.decode('utf-8').splitlines(keepends=True)
        start, end = (int(match[2]),int(match[3] or match[2])) if match else (1,min(80,len(lines)))
        if not 1<=start<=end<=len(lines) or end-start>=160:
            raise ValueError('historical source needs an existing range of at most 160 lines')
        text = ''.join(lines[start-1:end])
        line_range = {'first':start,'last':end,'total':len(lines),'next_line':end+1 if end<len(lines) else None}
    if not text or len(text.encode())>12000:
        raise ValueError('historical source/patch excerpt exceeds 12000 bytes; request a narrower source range')
    return {'producer_evidence_id':producer,'producer_status':receipt['status'],
        'producer_receipt_ref':proof['receipt_ref'],'producer_receipt_sha256':receipt_sha,
        'run_id':run_id,'round':number,'source_path':name,'view':view,
        'code_change_ref':change_ref,'code_change_sha256':change_sha,
        'snapshot_name':after[0]['name'],'snapshot_entry_sha256':object_digest(after[0]),
        'before_sha256':old_sha,'postimage_sha256':new_sha,
        'text':text,'line_range':line_range,'installed_status':'not_asserted',
        'execution_source_binding':'candidate archive only; this read does not certify the complete executed import graph',
        'authority':'authenticated historical candidate source/patch, not current source, installed repair, '
                    'new execution approval, data consumption, scientific validity or score'}


def capture(output, repo, request):
    """Publish a bounded historical read for ordinary evidence-ID paging; never restore it."""
    from .evidence_store import capture_attempt_evidence
    output, repo = Path(output).resolve(), Path(repo).resolve()
    view = load_view(output,repo,request)
    directory = output/'producer_source_citations'
    if directory.is_symlink():
        raise ValueError('unsafe producer-source evidence store')
    identity = uuid.uuid4().hex
    snapshot = directory/(identity+'.source')
    shown = sanitize_model_payload({'excerpt':view['text']},local_roots=(repo,output))['excerpt']
    # Paged evidence must be as private as the first tool response. Raw bytes
    # remain in the original owned archive, never in this model-facing copy.
    atomic_text(snapshot,shown)
    record = {'schema_version':1,'id':identity,'at':now(),'repo':str(repo),
        'request':dict(request), **{k:v for k,v in view.items() if k!='text'},
        'original_excerpt_sha256':hashlib.sha256(view['text'].encode()).hexdigest(),
        'display_redacted':shown != view['text'],
        'snapshot_ref':snapshot.relative_to(output).as_posix(),'snapshot_sha256':digest(snapshot)}
    record['sha256']=object_digest(record)
    immutable_json(directory/(identity+'.json'),record)
    proof = capture_attempt_evidence(output,attempt_id=identity,log=snapshot,
        receipt_ref=f'producer_source_citations/{identity}.json',status='historical_source_captured',
        returncode=None,termination_reason='read_only_producer_source')
    return {**proof, 'status':'historical_source_captured',
        'historical_source_ref':'producer-source:'+identity,
        **{k:v for k,v in view.items() if k!='text'}, 'excerpt':shown[:12000],
        'excerpt_complete':len(shown)<=12000,'display_redacted':record['display_redacted'],
        'guidance':'Read this evidence_id for the bounded historical source/patch display. '
            'Redacted displays are not raw executable code: never copy privacy placeholders into a patch. '
            'Cite it only as historical '
            'evidence; it is NOT a checkout-source reference for current protocol/capability approval. '
            'Inspect current source separately and explicitly compose any new audited proposal. '
            'This tool does not apply, install, replay, mutate history or start an experiment.'}
