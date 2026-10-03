"""Source-backed AgentSupervisor interpretation of fresh native progress metadata.

No benchmark names, counter filenames or accepted key-name recipes. A zero exit and
new weights alone never prove learning; the reviewer must trace a completed-work
counter to its native producer. The kernel checks the selected bytes and citations.
"""
import ast
import json
import math
from pathlib import Path

from .common import atomic_json, digest, now, object_digest, read_json, sanitize_model_text


def numeric_fields(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_fields(child, (*path, str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield {'field':list(path), 'value':value}


def candidates(output, since):
    result = []
    for path in output.rglob('*.json'):
        if len(result) >= 24:
            break
        if (path.is_symlink() or not path.resolve().is_relative_to(output.resolve())
                or not since <= path.stat().st_mtime or path.stat().st_size > 32768):
            continue
        try:
            document = json.loads(path.read_text())
            fields = list(numeric_fields(document))[:80]
        except (OSError, ValueError, RecursionError):
            continue
        if fields:
            result.append({'path':path.relative_to(output).as_posix(),
                           'sha256':digest(path), 'fields':fields})
    return result


def sources(repo, root, argv):
    """Trace static Python imports only within the checkout/registered dependency roots."""
    if (root/'native_context.json').is_file():
        from .native_context import load_context
        context = load_context(root, repo)
    else:
        context = {}
    roots = [repo, *reversed([Path(p) for p in context.get('dependency_roots', [])])]
    pending = [Path(p) for p in argv if str(p).endswith('.py') and Path(p).is_absolute()]
    seen, result = set(), []
    while pending and len(result) < 24:
        path = pending.pop(0)
        if path in seen:
            continue
        seen.add(path)
        if (path.suffix != '.py' or not path.is_file() or path.is_symlink()
                or not any(path.resolve().is_relative_to(base.resolve()) for base in roots)
                or path.stat().st_size > 65536):
            continue
        content = path.read_text(encoding='utf-8')
        entry = {'id':'source-'+str(len(result)), 'path':path,
                 'sha256':digest(path), 'content':content,
                 'excerpt':content if len(content) <= 16000 else content[:10000]+'\n…\n'+content[-6000:]}
        result.append(entry)
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
        for module in dict.fromkeys(modules):
            relative = Path(*module.split('.'))
            located = False
            for base in roots:
                for folder in (base, *list((base/'lib').glob('python*/site-packages'))[:8]):
                    candidate = (folder/relative).with_suffix('.py')
                    if candidate.is_file():
                        pending.append(candidate)
                        located = True
                        break
                if located:
                    break
    return result


def validate(answer, *, output, docs, code):
    if answer.get('verdict') != 'verified' or not isinstance(answer.get('counter'), dict):
        raise ValueError('Supervisor did not verify a completed-work counter')
    counter = answer['counter']
    doc = next((row for row in docs if row['path'] == counter.get('path')), None)
    field = counter.get('field')
    if doc is None or not isinstance(field, list) or not field or not all(isinstance(x,str) for x in field):
        raise ValueError('counter must select an offered native document/field')
    selected = next((row for row in doc['fields'] if row['field'] == field), None)
    if (selected is None or isinstance(counter.get('value'), bool)
            or selected['value'] != counter.get('value')
            or not selected['value'] > 0 or not float(selected['value']).is_integer()):
        raise ValueError('counter is not an exact positive completed-work integer')
    path = output/doc['path']
    if (path.is_symlink() or not path.resolve().is_relative_to(output.resolve())
            or digest(path) != doc['sha256']):
        raise ValueError('native counter changed during review')
    citations = answer.get('source_evidence')
    if not isinstance(citations, list) or not citations:
        raise ValueError('completed-work semantics need source citations')
    verified = []
    for citation in citations:
        source = next((row for row in code if row['id'] == citation.get('source_id')), None)
        quote = citation.get('quote')
        # Exact native update statements may be short (e.g. "step += 1"). Length
        # is not evidence of semantics; the independent review traces the chain,
        # while the kernel verifies offered source identity and unchanged bytes.
        if source is None:
            raise ValueError('source citation did not select an offered source ID')
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError('source citation quote is empty')
        if quote not in source['content']:
            raise ValueError('source citation quote is not an exact native substring')
        if digest(source['path']) != source['sha256']:
            raise ValueError('source citation bytes changed during review')
        verified.append({'source_id':source['id'], 'sha256':source['sha256'], 'quote':quote})
    return {'status':'observed', 'authority':'source-backed independent Supervisor review',
            'evidence':{'native_counter':counter, 'document_sha256':doc['sha256'],
                        'source_evidence':verified}, 'why':str(answer.get('why') or '')[:1500]}


def audit(client, *, repo, root, output, argv, since, native_evidence_ref):
    docs, code = candidates(output, since), sources(repo, root, argv)
    if not docs or not code:
        return {'status':'unknown', 'reason':'no fresh numeric metadata/source trace'}
    identity = object_digest({'audit_contract':'completed_work_role_scope_v2',
                              'documents':docs, 'source_hashes':[r['sha256'] for r in code],
                              'native_evidence_ref':native_evidence_ref})
    path = root/'training_progress_audits'/f'{identity}.json'
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('unsafe progress audit path')
    if path.exists():
        old = read_json(path)
        # Revalidate the selected bytes; a saved verdict is not proof of unchanged files.
        try:
            return {**validate(old['answer'], output=output, docs=docs, code=code),
                    'audit_ref':path.relative_to(root).as_posix()}
        except (ValueError, KeyError, OSError):
            return {'status':'unknown', 'reason':'saved audit no longer verifies'}
    payload = {'native_evidence_ref':native_evidence_ref, 'numeric_documents':docs,
               'sources':[{'id':r['id'], 'sha256':r['sha256'], 'excerpt':r['excerpt']} for r in code]}
    system = ('Independently audit actual TRAINING PROGRESS, not benchmark score. The native '
        'trainer exited zero and produced a fresh artifact, but stdout has no recognizable '
        'work counter. Select an exact positive completed optimizer/update/epoch counter '
        'from offered native metadata only when source shows it advances after real training '
        'work and is persisted by that native producer. Requested steps, configuration, '
        'learning-rate schedules, initialization-only weights and data-loader rows do NOT '
        'prove learning. Do not fabricate output, alter code or run commands. Trace producer '
        'and save flow using provided source excerpts; quote exact substrings. If evidence '
        'is insufficient, say unverified. Return ONE JSON object: '
        '{"verdict":"verified|unverified","counter":{"path":"offered relative path",'
        '"field":["exact","keys"],"value":positive_integer},'
        '"source_evidence":[{"source_id":"offered id","quote":"exact source"}],"why":"reason"}.')
    record = {'id':identity, 'at':now(), 'native_evidence_ref':native_evidence_ref,
              'documents':docs, 'source_hashes':[r['sha256'] for r in code]}
    try:
        from .agent_client import role_scope
        with role_scope(client, 'supervisor'):
            content, _ = client.chat_with_metadata(system, sanitize_model_text(json.dumps(payload)),
                max_tokens=1600, timeout=180, thinking='disabled', read_only=True)
        from .execution_derive import _objects
        answers = [obj for obj in _objects(content) if 'verdict' in obj]
        if len(answers) != 1:
            raise ValueError('Supervisor must return one unambiguous verdict')
        record['answer'] = answers[0]
        result = validate(answers[0], output=output, docs=docs, code=code)
    except Exception as exc:
        record.update(status='unverified', error=f'{type(exc).__name__}: {exc}'[:1200])
        result = {'status':'unknown', 'reason':record['error']}
    else:
        record['status'] = 'verified'
    atomic_json(path, record)
    return {**result, 'audit_ref':path.relative_to(root).as_posix()}
