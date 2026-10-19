"""Trusted citations of installed source, separate from model-written diagnostic output.

The Agent selects a relative file under a published read-only native dependency root.
The executor reads/seals actual bytes; citations recheck both context and backing bytes.
Printing a guessed source snippet in a diagnostic cannot grant citation authority.
"""
import re
import uuid
from pathlib import Path

from .common import atomic_text, digest, immutable_json, now, object_digest, read_json
from .native_context import load_context


def roots(context):
    return [Path(context['interpreter']).parent.parent,
            *(Path(p) for p in context.get('dependency_roots', []))]


def catalog(output, repo):
    """A bounded discovery route for the already-authorized dependency roots."""
    context = load_context(Path(output), Path(repo))
    rows = []
    for index, root in enumerate(roots(context)):
        entries = []
        for p in sorted(root.iterdir(), key=lambda p: p.name):
            name = p.name
            if (p.is_symlink() or name.startswith('.') or
                    any(s in name.lower() for s in ('credential', 'secret', 'id_rsa', 'id_ed25519'))):
                continue
            if p.is_dir() or p.suffix.lower() in {'.py', '.pyi', '.sh', '.yaml', '.yml', '.toml'}:
                entries.append(name + ('/' if p.is_dir() else ''))
            if len(entries) >= 65:
                break
        rows.append({'root_index': index, 'entries': entries[:64], 'truncated': len(entries) > 64})
    if load_context(Path(output), Path(repo))['identity'] != context['identity']:
        raise ValueError('native source context changed during discovery')
    return {'status': 'source_catalog', 'context_identity': context['identity'], 'roots': rows,
            'authority': 'authorized-root names only; not source bytes or protocol approval',
            'guidance': 'Select source_request={root_index,path} to capture exact public source bytes. '
                        'Builtin Read cannot read external dependencies; no edit-mode escalation required.'}


def locate(context, request):
    if not isinstance(request,dict) or set(request) != {'root_index','path'}:
        raise ValueError('source_request requires root_index and dependency-root-relative path')
    index, name = request['root_index'], request['path']
    available = roots(context)
    if type(index) is not int or not 0 <= index < len(available) or not isinstance(name,str):
        raise ValueError('source_request root index is not in the selected native dependency roots')
    relative = Path(name)
    if (not name or len(name)>512 or relative.is_absolute() or '..' in relative.parts
            or '\\' in name or relative.suffix.lower() not in {'.py','.pyi','.sh','.yaml','.yml','.toml'}
            or any(p.startswith('.') or p.lower() in {'credentials','secrets'} for p in relative.parts)
            or any(t in name.lower() for t in ('credential','secret','id_rsa','id_ed25519'))):
        raise ValueError('native source path is not an admissible public code/config file')
    root = available[index].resolve(strict=True)
    path = root/relative
    if any(p.is_symlink() for p in [path, *path.parents] if p != root and p.is_relative_to(root)):
        raise ValueError('native source path contains a symbolic link')
    if not path.resolve(strict=True).is_relative_to(root) or not path.is_file() or path.stat().st_size>2*1024**2:
        raise ValueError('native source is absent, escaped or exceeds 2 MiB')
    return path


def capture(output, repo, request):
    context = load_context(Path(output), Path(repo))
    directory = Path(output)/'native_source_citations'
    if directory.is_symlink() or Path(output).is_symlink():
        raise ValueError('unsafe native source citation store')
    path = locate(context,request)
    before = digest(path)
    text = path.read_text(encoding='utf-8')
    identity = uuid.uuid4().hex
    snapshot = directory/(identity+'.source')
    if before != digest(path) or load_context(Path(output),Path(repo))['identity'] != context['identity']:
        raise ValueError('native source/context changed while reading')
    atomic_text(snapshot,text)
    if digest(snapshot) != before:
        raise ValueError('native source bytes did not survive exact snapshot')
    row = {'schema_version':1,'id':identity,'at':now(),'request':request,
        'context_identity':context['identity'],'source_sha256':before,
        'snapshot_ref':snapshot.relative_to(output).as_posix(),
        'authority':'executor-read native dependency source; not a score or protocol approval'}
    row['sha256'] = object_digest(row)
    immutable_json(directory/(identity+'.json'),row)
    from .evidence_store import capture_attempt_evidence
    proof = capture_attempt_evidence(Path(output),attempt_id=identity,log=snapshot,
        receipt_ref=f'native_source_citations/{identity}.json',status='native_source_captured',
        returncode=None,termination_reason='read_only_source_capture')
    return {**proof, 'ref':'native-source:'+identity,'sha256':before,'source_request':request,
        'excerpt':text[:4000], 'bytes':snapshot.stat().st_size,
        'snapshot_ref':row['snapshot_ref'],
        'native_source_roots':[{'root_index':i,'path':str(p)} for i,p in enumerate(roots(context))],
        'guidance':'Cite ref and an exact source quote. Use read_evidence(evidence_id,offset,limit) for full paged source; '
            'context/backing bytes must still match during reservation review and later use.'}


def resolve(output, repo, ref, quote):
    if not isinstance(ref,str) or not re.fullmatch(r'native-source:[0-9a-f]{32}',ref):
        raise ValueError('invalid native source citation handle')
    from .experiment_transaction import safe
    metadata = safe(output,'native_source_citations/'+ref.split(':')[1]+'.json')
    if not metadata.is_file():
        raise ValueError('native source citation is missing; model diagnostic text is not source proof')
    if metadata.stat().st_size>16000:
        raise ValueError('native source citation record exceeds bounds')
    row = read_json(metadata)
    if (row.get('schema_version') != 1 or row.get('id') != ref.split(':')[1]
            or row.get('sha256') != object_digest({k:v for k,v in row.items() if k!='sha256'})):
        raise ValueError('native source citation identity changed')
    context = load_context(Path(output),Path(repo))
    if context['identity'] != row['context_identity']:
        raise ValueError('native source environment changed; capture a fresh citation')
    snapshot = safe(output,row['snapshot_ref'])
    if snapshot != safe(output,'native_source_citations/'+row['id']+'.source'):
        raise ValueError('native source snapshot is not bound to its citation ID')
    source = locate(context,row['request'])
    if digest(source) != row['source_sha256'] or digest(snapshot) != row['source_sha256']:
        raise ValueError('native source or snapshot changed; capture and review current evidence')
    if not isinstance(quote,str) or not quote or len(quote)>4000 or quote not in snapshot.read_text(encoding='utf-8'):
        raise ValueError('native source citation quote is not exact')
    return {'ref':ref,'quote':quote,'sha256':row['source_sha256']}
