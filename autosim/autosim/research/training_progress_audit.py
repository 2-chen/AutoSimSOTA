"""Source-backed AgentSupervisor interpretation of fresh native progress metadata.

No benchmark names, counter filenames or accepted key-name recipes. A zero exit and
new weights alone never prove learning; the reviewer must trace a completed-work
counter to its native producer. The kernel checks the selected bytes and citations.
"""
import ast
import json
import math
import time
from pathlib import Path

from .common import atomic_json, digest, now, object_digest, read_json, sanitize_model_text


def revalidate_detached(receipt, *, path, root, repo, client=None,
                        allow_model=False, failure_details=None):
    """Review a detached producer without modifying its worker-owned sealed receipt.

    Read projections only replay existing exact-byte/source proof. An explicit job
    inspection/adoption may buy a new review; listing jobs never calls the model.
    """
    from .evidence_store import read_attempt_evidence
    root, repo, path = Path(root).resolve(), Path(repo).resolve(), Path(path)
    sidecar = path.with_name('training_progress_revalidation.json')
    try:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            return None
        original_hash = digest(path)
        if read_json(path) != receipt:
            return None
        output = Path(str(receipt.get('output_directory') or '')).resolve(strict=True)
        if not output.is_relative_to(root/'research'):
            return None
        reference = str(receipt.get('evidence_ref') or '')
        evidence_id = str(receipt.get('evidence_id') or Path(reference).stem)
        native = read_attempt_evidence(root, evidence_id, limit=1)
        if (native.get('status') != 'completed' or native.get('returncode') != 0 or
                native.get('receipt_ref') != path.relative_to(root).as_posix()):
            return None
        since = float(receipt.get('started_epoch') or 0)
        if sidecar.exists():
            if sidecar.is_symlink():
                return None
            saved = read_json(sidecar)
            if saved.get('native_receipt_sha256') != original_hash:
                return None
            audit_path = root / str(saved.get('audit_ref') or '')
            if (audit_path.is_symlink() or not audit_path.resolve().is_relative_to(
                    (root/'training_progress_audits').resolve()) or
                    digest(audit_path) != saved.get('audit_sha256')):
                return None
            document = read_json(audit_path)
            if document.get('native_evidence_ref') != reference:
                return None
            code = sources(repo, root, receipt['argv'])
            if document.get('source_hashes') != [row['sha256'] for row in code]:
                return None
            progress = validate(document['answer'], output=output,
                docs=candidates(output, since), code=code)
            progress['audit_ref'] = saved['audit_ref']
        elif allow_model and getattr(client, 'supports_native_progress_audit', False):
            progress = audit(client, repo=repo, root=root, output=output,
                argv=receipt['argv'], since=since, native_evidence_ref=reference)
            if progress.get('status') != 'observed' or digest(path) != original_hash:
                if failure_details is not None:
                    failure_details.update(progress)
                return None
            audit_path = root / str(progress.get('audit_ref') or '')
            if (audit_path.is_symlink() or not audit_path.resolve().is_relative_to(
                    (root/'training_progress_audits').resolve())):
                return None
            atomic_json(sidecar, {'schema_version':1, 'at':now(),
                'native_receipt_sha256':original_hash, 'native_evidence_ref':reference,
                'audit_ref':progress['audit_ref'], 'audit_sha256':digest(audit_path)})
        else:
            return None
        return {**receipt, 'training_progress':progress,
                'training_progress_revalidation_ref':sidecar.relative_to(root).as_posix()}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        if failure_details is not None:
            failure_details.update(status='unknown', reason=f'{type(exc).__name__}: {exc}')
        return None


def numeric_fields(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_fields(child, (*path, str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield {'field':list(path), 'value':value}


def candidates(output, since):
    """Complete, deterministic numeric inventory, or an explicit bounded failure.

    Never silently return a prefix: directory enumeration order is not evidence of
    which checkpoint finished training. Bounds protect the reviewer, not select it.
    """
    result = []
    scanned = 0
    for path in output.rglob('*.json'):
        scanned += 1
        if scanned > 8192:
            raise ValueError('numeric inventory incomplete: metadata scan budget exceeded')
        if (path.is_symlink() or not path.resolve().is_relative_to(output.resolve())
                or not since <= path.stat().st_mtime or path.stat().st_size > 32768):
            continue
        try:
            document = json.loads(path.read_text())
            fields = list(numeric_fields(document))
        except (OSError, ValueError, RecursionError):
            continue
        if len(fields) > 256:
            raise ValueError('numeric inventory incomplete: document field budget exceeded')
        if fields:
            result.append({'path':path.relative_to(output).as_posix(),
                           'sha256':digest(path), 'fields':fields})
            if len(result) > 2048:
                raise ValueError('numeric inventory incomplete: document budget exceeded')
    return sorted(result, key=lambda row: row['path'])


def sources(repo, root, argv, *, cwd=None):
    """Trace static Python imports only within the checkout/registered dependency roots."""
    if (root/'native_context.json').is_file():
        from .native_context import load_context
        context = load_context(root, repo)
    else:
        context = {}
    roots = [repo, *reversed([Path(p) for p in context.get('dependency_roots', [])])]
    working = Path(cwd) if cwd is not None else repo
    pending = [Path(p) if Path(p).is_absolute() else working / p
               for p in argv if str(p).endswith('.py')]
    seen, result = set(), []
    while pending and len(result) < 24:
        # Native orchestration/update semantics come before transitive utility
        # dependencies; large SDK imports must not exhaust the bounded trace.
        pending.sort(key=lambda candidate: not candidate.resolve().is_relative_to(repo.resolve()))
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
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    relative = Path(*node.module.split('.')) if node.module else Path()
                    for target in [base / relative, *(base / relative / n.name for n in node.names if n.name != '*')]:
                        for candidate in (target.with_suffix('.py'), target / '__init__.py'):
                            if candidate.is_file():
                                pending.append(candidate)
                elif node.module:
                    modules.append(node.module)
                    modules.extend(node.module + '.' + n.name for n in node.names if n.name != '*')
        for module in dict.fromkeys(modules):
            relative = Path(*module.split('.'))
            located = False
            for base in roots:
                for folder in (base, *list((base/'lib').glob('python*/site-packages'))[:8]):
                    for candidate in ((folder/relative).with_suffix('.py'), folder/relative/'__init__.py'):
                        if candidate.is_file():
                            pending.append(candidate)
                            located = True
                            break
                    if located:
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
    # A positive intermediate counter is not a terminal selection proof. The
    # reviewer must interpret competing counters, rather than seeing a prefix or
    # having the kernel assume that every numerically larger value is better.
    competing = [row['path'] for row in docs if row['path'] != doc['path']
                 and any(item['field'] == field and item['value'] != selected['value']
                         for item in row['fields'])]
    scope = answer.get('counter_selection') or {}
    if competing:
        reviewed = scope.get('reviewed_documents')
        required = {doc['path'], *competing}
        offered = {row['path'] for row in docs}
        if (scope.get('kind') not in {'terminal_completed', 'native_best'}
                or not isinstance(scope.get('why'), str) or not scope['why'].strip()
                or not isinstance(reviewed, list)
                or not all(isinstance(item, str) for item in reviewed)
                or len(reviewed) != len(set(reviewed))
                or not required <= set(reviewed) <= offered):
            raise ValueError('competing progress counters require explicit complete selection review; '
                             'include all competing counters and only offered documents')
        if scope.get('kind') == 'terminal_completed' and any(
                item['field'] == field and item['value'] > selected['value']
                for row in docs for item in row['fields']):
            raise ValueError('terminal counter claim conflicts with a later completed-work counter; '
                             'review producer families before claiming terminal completion')
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
                        'source_evidence':verified, 'counter_selection':scope},
            'why':str(answer.get('why') or '')[:1500]}


def audit(client, *, repo, root, output, argv, since, native_evidence_ref):
    try:
        docs, code = candidates(output, since), sources(repo, root, argv)
    except ValueError as exc:
        return {'status':'unknown', 'reason':str(exc)}
    if not docs or not code:
        return {'status':'unknown', 'reason':'no fresh numeric metadata/source trace'}
    identity = object_digest({'audit_contract':'completed_work_bounded_review_v4',
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
        except (ValueError, KeyError, OSError) as exc:
            return {'status':'unknown',
                    'reason':f'saved audit no longer verifies: {type(exc).__name__}: {exc}'[:1200],
                    'audit_ref':path.relative_to(root).as_posix()}
    payload = {'native_evidence_ref':native_evidence_ref, 'numeric_documents':docs,
               'native_invocation':sanitize_model_text(json.dumps(argv), local_roots=(repo,root)),
               'sources':[{'id':r['id'], 'sha256':r['sha256'], 'excerpt':r['excerpt']} for r in code]}
    system = ('Independently audit actual TRAINING PROGRESS, not benchmark score. The native '
        'trainer exited zero and produced a fresh artifact, but stdout has no recognizable '
        'work counter. Select an exact positive completed optimizer/update/epoch counter '
        'from offered native metadata only when source shows it advances after real training '
        'work and is persisted by that native producer. Requested steps, configuration, '
        'learning-rate schedules, initialization-only weights and data-loader rows do NOT '
        'prove learning. Do not fabricate output, alter code or run commands. Trace producer '
        'and save flow using provided source excerpts; quote exact substrings. Review ALL '
        'offered counter documents. Prefer the actual completed terminal policy for this '
        'training invocation, not any intermediate positive counter. A native best-policy '
        'selection instead requires a source-backed rule independent of held-out scores. '
        'Never infer terminal or best from filenames, enumeration order or a maximum alone. '
        'Trace the invocation to requested work in producer source; explain whether the '
        'selected policy satisfies that request, uses legitimate native early stopping, '
        'or represents a native best-policy rule. Do not equate epochs and updates. '
        'If other documents have the same field with differing values, counter_selection '
        'must explain the terminal/native-best rule and list the selected document '
        'plus ALL those competing documents, even ones ultimately excluded as another producer. '
        'List only the selected document and same-field competitors: do not enumerate every '
        'configuration document. Keep each explanation under 100 words and quote only the '
        'few exact update/save statements required for the proof. No Markdown or commentary. '
        'Other offered documents reviewed for exclusions may also be listed; do not invent '
        'paths or duplicate entries. '
        'If evidence '
        'is insufficient, say unverified. Return ONE JSON object: '
        '{"verdict":"verified|unverified","counter":{"path":"offered relative path",'
        '"field":["exact","keys"],"value":positive_integer},'
        '"counter_selection":{"kind":"terminal_completed|native_best",'
        '"reviewed_documents":["offered paths"],"why":"source-backed choice and exclusions"},'
        '"source_evidence":[{"source_id":"offered id","quote":"exact source"}],"why":"reason"}.')
    record = {'id':identity, 'at':now(), 'native_evidence_ref':native_evidence_ref,
              'documents':docs, 'source_hashes':[r['sha256'] for r in code]}
    record['responses'] = []
    deadline = time.monotonic() + 180
    try:
        from .agent_client import role_scope
        from .execution_derive import _objects
        request = payload
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                raise ValueError('progress review original deadline exhausted')
            with role_scope(client, 'supervisor'):
                content, provider = client.chat_with_metadata(system,
                    sanitize_model_text(json.dumps(request)), max_tokens=2400,
                    timeout=remaining, thinking='disabled', read_only=True,
                    decision_only=True, auto_skills=False, include_research_context=False,
                    resume_session=False)
            response = {'response_sha256':object_digest(content),
                        'text':sanitize_model_text(str(content))[:65536],
                        'truncated':len(str(content))>65536,
                        'provider_turn':{k:provider[k] for k in
                            ('turn_id','decision_attempt_id','agent_session_key')
                            if isinstance(provider,dict) and k in provider}}
            record['responses'].append(response)
            # Preserve the failed reply before a bounded correction or process restart.
            atomic_json(path, {**record, 'status':'reviewing'})
            try:
                answers = [obj for obj in _objects(content) if 'verdict' in obj]
                if len(answers) != 1:
                    raise ValueError('Supervisor must return one unambiguous verdict')
                record['answer'] = answers[0]
                if answers[0].get('verdict') == 'unverified':
                    raise RuntimeError('Supervisor explicitly found insufficient evidence: ' +
                                       str(answers[0].get('why') or '')[:700])
                result = validate(answers[0], output=output, docs=docs, code=code)
                break
            except ValueError as exc:
                response['validation_error'] = str(exc)[:1200]
                if attempt:
                    raise
                request = {**payload, 'format_feedback':str(exc)[:1200],
                           'previous_response':response['text'][:12000],
                           'correction_scope':'One bounded schema/citation correction; reconsider '
                               'the supplied evidence. Never invent counters, sources or success.'}
    except Exception as exc:
        record.update(status='unverified', error=f'{type(exc).__name__}: {exc}'[:1200])
        result = {'status':'unknown', 'reason':record['error']}
    else:
        record['status'] = 'verified'
    atomic_json(path, record)
    return {**result, 'audit_ref':path.relative_to(root).as_posix()}
