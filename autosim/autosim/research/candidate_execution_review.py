"""Source-bound LLM admission; not proof of metric gain or native consumption."""
import json
import hashlib
import re
import time
from pathlib import Path

from .common import atomic_json, digest, now, object_digest, read_json, sanitize_model_payload
from .prompt_library import load_prompt

FORMAT_FEEDBACK_VERSION = 5


def requested_native_ids(ideas, *, limit=12) -> list[str]:
    """Bounded IDs explicitly cited by Agent-authored proposals, not keyword relevance."""
    ids = []
    for idea in reversed(list(ideas)):
        evidence = getattr(idea, 'evidence', None)
        for text in evidence if isinstance(evidence, list) else []:
            if not isinstance(text, str):
                continue
            for identity in re.findall(r'(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])', text):
                if identity not in ids and len(ids) < limit:
                    ids.append(identity)
    return ids


def select_native_observations(pool, requested_ids=()):
    """The same per-proposal selection for catalog lookup and real admission."""
    groups = {}
    for excerpt in pool:
        groups.setdefault(excerpt['evidence_id'], []).append(excerpt)
    selected = list(dict.fromkeys(identity for identity in requested_ids if identity in groups))[:3]
    recent = [identity for identity, group in groups.items()
              if group[0].get('_selection_recent', True)]
    selected += [identity for identity in recent[-3:] if identity not in selected]
    return [{key:value for key,value in excerpt.items() if key != '_selection_recent'}
            for identity in selected for excerpt in groups[identity]]


def native_observations(root: Path, repo: Path, *, requested_ids=(), return_pool=False) -> list[dict]:
    """Offer sealed CPU diagnostics for the current environment, never metric authority."""
    from .native_context import load_context
    from .evidence_store import read_attempt_evidence
    from .diagnostic_source_identity import source_identity
    try:
        context=load_context(root,repo)
        current_source=source_identity(repo)
        if current_source is None:
            return []
        directory=root/'native_diagnostics'
        if directory.is_symlink() or not directory.is_dir():
            return []
        paths=[]
        for index,path in enumerate(directory.glob('*.json')):
            if index>=512:
                return []
            if not path.is_symlink():
                paths.append((path.stat().st_mtime_ns,path))
    except (OSError,ValueError,TypeError,KeyError,RuntimeError):
        return []
    requested = list(dict.fromkeys(identity for identity in list(requested_ids)[:(128 if return_pool else 12)]
        if isinstance(identity, str) and re.fullmatch(r'[0-9a-f]{32}', identity)))
    # Retain Agent-selected older evidence, but validate it exactly as recent evidence.
    # Never take a path or a quote from the proposal as trusted evidence.
    requested_paths = {directory/f'{identity}.json' for identity in requested}
    candidates = sorted(paths)[-8:]
    recent_paths = {row[1] for row in candidates}
    candidates += [row for row in sorted(paths) if row[1] in requested_paths and row not in candidates]
    candidates.sort()
    rows=[]
    for _,path in candidates:
        try:
            if not re.fullmatch(r'[0-9a-f]{32}',path.stem) or path.stat().st_size>65536:
                continue
            receipt=read_json(path)
            if (receipt.get('context_identity')!=context['identity'] or
                    receipt.get('repository_source_identity') != current_source or
                    receipt.get('repository_source_stable') is not True or
                    (receipt.get('diagnostic_policy_binding') is not None and
                     receipt.get('diagnostic_policy_binding_stable') is not True) or
                    receipt.get('capability')!='cpu_diagnostic_not_simulation_readiness' or
                    receipt.get('gpu_access') is not False or receipt.get('returncode')!=0 or
                    receipt.get('status')!='completed' or receipt.get('evidence_id')!=path.stem):
                continue
            if receipt.get('diagnostic_policy_binding') is not None:
                from .diagnostic_artifacts import binding_unchanged
                if not binding_unchanged(root, receipt['diagnostic_policy_binding']):
                    continue
            proof=read_attempt_evidence(root,path.stem,limit=8000)
            if (proof['receipt_ref']!=path.relative_to(root).as_posix() or
                    proof['status']!='completed' or proof['returncode']!=0):
                continue
            excerpt={'id':f'runtime_diagnostic_{path.stem}','text':proof['text'],
                         '_selection_recent':path in recent_paths,
                         'context_identity':context['identity'],'evidence_id':path.stem,
                         'offset':proof['offset'],'next_offset':proof['next_offset'],
                         'complete':proof['next_offset']==proof['log_bytes'],
                         'authority':'sealed read-only CPU diagnostic; inspect relevance, not a score or proof of all claims'}
            group=[excerpt]
            if proof['next_offset'] < proof['log_bytes']:
                # Native probes often inspect the consumer *after* repository
                # source. Offer a separate bounded end page, not an invented
                # contiguous quote or a prefix that silently hides that evidence.
                tail=read_attempt_evidence(root,path.stem,
                    offset=max(proof['next_offset'],proof['log_bytes']-8000),limit=8000)
                group.append({**excerpt,'id':f'runtime_diagnostic_{path.stem}_tail',
                              'text':tail['text'],'offset':tail['offset'],
                              'next_offset':tail['next_offset'],'complete':False})
            rows.append(group)
        except (OSError,ValueError,TypeError,KeyError,AttributeError):
            continue
    pool = [excerpt for group in rows for excerpt in group]
    return pool if return_pool else select_native_observations(pool, requested)


def material(repo: Path, stages: dict, idea) -> dict:
    repo = Path(repo).resolve()
    sources = [{'id':f'workflow_{name}', 'text':text}
               for name,text in sorted(stages.items()) if isinstance(text,str) and text.strip()]
    change = idea.change or {}
    patches = change.get('patches') or ([change] if idea.granularity == 'code' else [])
    offered_candidate_paths = set()
    for index, patch in enumerate(patches):
        path = repo / str(patch.get('file') or '')
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repo)
                or path.stat().st_size > 4*1024*1024):
            raise ValueError('candidate source unavailable for execution review; inspect before GPU work')
        source_path = path.resolve().relative_to(repo).as_posix()
        if source_path in offered_candidate_paths:
            continue  # Multiple hunks in one file must not create competing source IDs.
        offered_candidate_paths.add(source_path)
        raw=path.read_bytes()
        text=raw.decode('utf-8')
        complete=len(text.encode())<=65536
        if not complete:
            block=str(patch.get('find') or '')
            offset=text.find(block) if block else -1
            if offset<0:
                raise ValueError('large candidate file needs an exact patch context')
            text=(text[:6000]+'\n# bounded excerpt near the proposed patch\n'+
                  text[max(0,offset-4000):offset+len(block)+4000]+
                  '\n# bounded final excerpt\n'+text[-4000:])
        sources.append({'id':f'candidate_source_{index}',
                        'path':source_path, 'sha256':hashlib.sha256(raw).hexdigest(),
                        'complete':complete, 'text':text})
    # A code intervention is source, not a second JSON string serialization layer.
    # Offer literal before/after blocks so citation quoting does not reconstruct
    # JSON escapes; the structured change and file identities remain in the packet.
    change_text = ('\n\n'.join(f"file: {p.get('file', '')}\nfind:\n{p.get('find', '')}"
                              f"\nreplace:\n{p.get('replace', '')}" for p in patches)
                   if patches else json.dumps(change, ensure_ascii=False))
    sources.append({'id':'candidate_change', 'text':change_text})
    # Source-reference inventory is a reading aid, not an execution/reachability rule.
    # Include wrappers and their local script references so the reviewer need not infer
    # a policy from a filename or a generic dynamic import alone.
    native_refs=[]
    offered={row.get('path'):row['id'] for row in sources if row.get('path')}
    pending=[(row,0) for row in sources if row['id'].startswith('workflow_')]
    seen=set()
    for parent,depth in pending:
        if depth>=3:
            continue
        for ref in re.findall(r'[A-Za-z0-9_./-]+\.(?:py|sh)\b',parent['text']):
            relative=Path(ref.lstrip('/'))
            if '..' in relative.parts:
                continue
            path=repo/relative
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repo):
                continue
            if path.stat().st_size>4*1024*1024:
                continue
            key=path.relative_to(repo).as_posix()
            if key not in offered:
                if len(seen)>=16:
                    continue
                raw=path.read_bytes()
                text=raw.decode('utf-8')
                complete=len(raw)<=65536
                if not complete:
                    text=text[:8000]+'\n# incomplete native source excerpt\n'+text[-8000:]
                row={'id':f'native_source_{len(seen)}','path':key,'text':text,
                     'complete':complete,'sha256':hashlib.sha256(raw).hexdigest()}
                sources.append(row)
                offered[key]=row['id']
                seen.add(key)
                pending.append((row,depth+1))
            native_refs.append({'from':parent['id'],'source_id':offered[key],
                                'path':key,'authority':'source reference hint; not proof of execution'})
    # Dynamic imports commonly enter through a package initializer rather than
    # the patched module filename. Supply bounded package glue, not an assumed
    # import edge or a repository-specific adapter map.
    for candidate in list(sources):
        if not candidate['id'].startswith('candidate_source_'):
            continue
        module = repo / candidate['path']
        if module.suffix != '.py':
            continue
        for parent in list(module.parents)[:4]:
            if not parent.is_relative_to(repo):
                break
            path = parent / '__init__.py'
            if (path.is_symlink() or not path.is_file() or
                    any(p.is_symlink() for p in path.parents if p.is_relative_to(repo))
                    or path.stat().st_size > 65536):
                continue
            key = path.relative_to(repo).as_posix()
            if key not in offered:
                if len(seen) >= 16:
                    break
                raw = path.read_bytes()
                sources.append({'id': f'native_source_{len(seen)}', 'path': key,
                    'text': raw.decode('utf-8'), 'complete': True,
                    'sha256': hashlib.sha256(raw).hexdigest()})
                offered[key] = sources[-1]['id']
                seen.add(key)
            native_refs.append({'from': candidate['id'], 'source_id': offered[key],
                'path': key, 'authority': 'package initialization hint; inspect exports and routing'})
    packet = {'candidate':{key:getattr(idea,key) for key in
                          ('label','granularity','mechanism','why','risk')},
              'sources':sources, 'native_references':native_refs,
              'boundary':'selected stage builders, before this candidate; not a score'}
    # Host paths stay redacted. These are validated PUBLIC checkout-relative
    # filenames, useful for reasoning about routing, not arbitrary path grants.
    for row in sources + native_refs:
        if isinstance(row.get('path'), str):
            row['repository_file'] = row['path']
    if len(json.dumps(packet,ensure_ascii=False).encode()) > 192*1024:
        raise ValueError('execution review packet incomplete: inspect a smaller intervention before GPU work')
    return packet


def validate(answer, packet):
    if (not isinstance(answer,dict) or answer.get('verdict') not in
            {'applicable','inapplicable','uncertain'} or
            not isinstance(answer.get('reason'),str) or not answer['reason'].strip()
            or len(answer['reason']) > 1500):
        raise ValueError('execution review needs one bounded verdict and reason')
    citations = answer.get('source_evidence')
    if not isinstance(citations,list) or len(citations)>12:
        raise ValueError('execution review requires bounded exact source citations')
    ids=set()
    for index,citation in enumerate(citations):
        if not isinstance(citation,dict):
            raise ValueError('execution review citation must be an object')
        source=next((row for row in packet['sources'] if row['id']==citation.get('source_id')),None)
        quote=citation.get('quote')
        if source is None or not isinstance(quote,str) or not 4<=len(quote)<=800 or quote not in source['text']:
            raise ValueError(f"source_evidence[{index}] ({citation.get('source_id')!r}): "
                             'quote must be 4..800 characters and an exact substring of this offered source')
        ids.add(source['id'])
    if answer['verdict']!='uncertain' and ('candidate_change' not in ids or
                                          not any(s.startswith('workflow_') for s in ids)):
        raise ValueError('execution relevance needs both workflow and intervention citations')
    if (answer['verdict']!='uncertain' and (packet.get('execution_plan') or {}).get('stages')==['evaluate']
            and 'workflow_evaluate' not in ids):
        raise ValueError('evaluation-only policy trial requires the actual evaluate workflow citation')
    if (answer['verdict'] != 'uncertain' and
            (packet.get('execution_plan') or {}).get('will_train') is True and
            'workflow_train' not in ids):
        raise ValueError('new-training trial requires the actual train workflow citation')
    native_ids={row['source_id'] for row in packet.get('native_references') or []}
    if answer['verdict']!='uncertain' and native_ids and not native_ids.intersection(ids):
        raise ValueError('execution review must cite actual offered native source, not infer a routed policy from a filename')


def cached(root: Path, packet: dict) -> dict | None:
    identity = object_digest(packet)
    directory = root/'candidate_execution_reviews'
    path = directory/f'{identity}.json'
    if directory.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 256*1024:
        return None
    try:
        row = read_json(path)
        signature = row.pop('record_sha256')
        if (object_digest(row) != signature or row.get('material_sha256') != identity
                or row.get('schema_version') != 1):
            return None
        if row.get('request_ref'):
            request_path=root/row['request_ref']
            if (Path(row['request_ref']).is_absolute() or '..' in Path(row['request_ref']).parts
                    or request_path.is_symlink() or request_path.parent.is_symlink()
                    or digest(request_path)!=row.get('request_sha256')):
                return None
        if row.get('status')=='unavailable':
            if not row.get('error'):
                return None
        else:
            validate(row['answer'], packet)
        return {**row,'record_sha256':signature}
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return None


def citation_feedback(answer, packet):
    """Recommend exact byte matches; never rewrite or approve a review."""
    suggestions, errors = [], []
    citations = answer.get('source_evidence') if isinstance(answer, dict) else None
    if isinstance(citations, list):
        for index, citation in enumerate(citations[:12]):
            if not isinstance(citation, dict) or not isinstance(citation.get('quote'), str):
                errors.append(f'source_evidence[{index}] needs a source_id and string quote')
                continue
            quote = citation['quote']
            supplied = next((s for s in packet['sources'] if s['id'] == citation.get('source_id')), None)
            if supplied and 4 <= len(quote) <= 800 and quote in supplied['text']:
                continue
            errors.append(f'source_evidence[{index}] is not an exact quote of its supplied source')
            # Escaping help is a suggestion only. The second response must itself
            # pass the original exact-substring and source-ID validator.
            if len(quote) > 4000:
                continue
            decoded = quote.replace('\\n', '\n').replace('\\"', '"')
            alternatives = [quote, quote.strip(), decoded, decoded.strip()]
            encoded = quote
            for _ in range(2):
                encoded = json.dumps(encoded, ensure_ascii=False)[1:-1]
                alternatives.append(encoded)
            for alternative in dict.fromkeys(alternatives):
                if not 4 <= len(alternative) <= 800:
                    continue
                for source in packet['sources']:
                    if alternative in source['text']:
                        candidate = {'source_id': source['id'], 'quote': alternative}
                        if candidate not in suggestions and len(suggestions) < 8:
                            suggestions.append(candidate)
    if isinstance(answer, dict) and answer.get('verdict') in {'applicable', 'inapplicable'}:
        ids = {c.get('source_id') for c in citations or [] if isinstance(c, dict)}
        evaluate_only = (packet.get('execution_plan') or {}).get('stages') == ['evaluate']
        if (evaluate_only and 'workflow_evaluate' not in ids or
                not evaluate_only and not any(str(key).startswith('workflow_') for key in ids)):
            errors.append('missing required workflow_evaluate quote' if evaluate_only else
                          'missing a required workflow_* quote')
        if 'candidate_change' not in ids:
            errors.append('missing required candidate_change quote')
        # Offer compact literal anchors for ALL mandatory categories, not merely
        # corrections to quotes the reviewer happened to include. No verdict is
        # supplied and the next answer must still independently pass validate().
        required = ['candidate_change']
        workflows = [s['id'] for s in packet['sources'] if s['id'].startswith('workflow_')]
        if evaluate_only:
            required.append('workflow_evaluate')
        elif (packet.get('execution_plan') or {}).get('will_train') is True:
            required.append('workflow_train')
            if 'workflow_train' not in ids:
                errors.append('missing required workflow_train quote: training is scheduled')
        elif workflows:
            required.append(workflows[0])
        native_ids = [r['source_id'] for r in packet.get('native_references') or []]
        if native_ids and not set(native_ids).intersection(ids):
            errors.append('missing an offered repository native routing source quote; '
                          'runtime_diagnostic_* is supplemental, not one of these source IDs')
            preferred = next((key for key in native_ids if key.startswith('candidate_source_')), native_ids[0])
            required.append(preferred)
        for key in required:
            source = next((s for s in packet['sources'] if s['id'] == key), None)
            if source:
                line = next((line.strip() for line in source['text'].splitlines()
                             if 4 <= len(line.strip()) <= 300), None)
                suggestion = {'source_id': key, 'quote': line}
                if line and suggestion not in suggestions and len(suggestions) < 8:
                    suggestions.append(suggestion)
    return {'citation_errors': errors[:16], 'exact_quote_suggestions': suggestions,
            'citation_transport_hint': 'A supplied diagnostic can contain nested JSON-encoded '
                'source strings. Quote the offered literal bytes (including backslashes), not '
                'the decoded code. Suggestions are exact offered substrings; copy their quote '
                'value with correct JSON encoding, not invented leading indentation. Correct '
                'every listed invalid citation, or omit an optional citation while retaining '
                'all required sources. No source or verdict is repaired for you.',
            'required_native_source_ids': list(dict.fromkeys(
                r['source_id'] for r in packet.get('native_references') or [])),
            'authority': 'verbatim source matches only; reviewer chooses and revalidates; no verdict inferred'}


def packet_for(root, repo, stages, idea, *, reuse_policy=None, execution_plan=None,
               policy_metadata=None, native_observation_snapshot=None):
    packet=material(repo,stages,idea)
    # A read-only catalog may project many plans in one synchronous state read.
    # Share its freshly checked evidence only within that read; execution callers
    # omit this argument and independently revalidate current source/artifacts.
    requested = requested_native_ids([idea])
    if (isinstance(native_observation_snapshot, dict) and
            set(requested) <= set(native_observation_snapshot['requested_ids'])):
        observations = select_native_observations(native_observation_snapshot['pool'], requested)
    elif native_observation_snapshot is not None and not requested:
        observations = select_native_observations(native_observation_snapshot)
    else:
        observations = (native_observations(Path(root),Path(repo),requested_ids=requested)
                        if requested else native_observations(Path(root),Path(repo)))
    packet['sources'].extend(observations)
    if reuse_policy is not None and policy_metadata:
        packet['sources'].extend(policy_metadata)
    packet['review_contract_version']=7
    # Negative results remain immutable, but a changed bounded correction
    # contract must not silently reuse an unavailable result from the old one.
    packet['format_feedback_version'] = FORMAT_FEEDBACK_VERSION
    packet['review_prompt_sha256']=object_digest(load_prompt('candidate-execution-review'))
    packet['required_native_source_ids'] = list(dict.fromkeys(
        r['source_id'] for r in packet.get('native_references') or []))
    if reuse_policy is not None:
        packet['execution_plan']={'mode':'evaluation_only_local_frozen_policy','stages':['evaluate'],
                                  'reuse_policy':dict(reuse_policy),
                                  'boundary':'no training, new data or policy-byte changes; evaluate changed deployment source'}
    elif execution_plan is not None:
        packet['execution_plan']=dict(execution_plan)
    else:
        scheduled=[name for name in ('train','evaluate') if stages.get(name)]
        packet['execution_plan']={'mode':'standard_training_and_evaluation',
            'stages':scheduled,'will_train':'train' in scheduled,
            'boundary':'a scheduled train writes new weights; unchanged recipe is not proof of identical existing weights'}
    if len(json.dumps(packet,ensure_ascii=False).encode())>192*1024:
        raise ValueError('bounded review including native diagnostics is too large')
    return sanitize_model_payload(packet,local_roots=(repo,root))


class ReviewUnavailable(ValueError):
    def __init__(self, message, record_ref):
        self.record_ref=record_ref
        self.evidence_id='execution-review:'+Path(record_ref).stem
        super().__init__(message)


def read_review_evidence(root: Path, evidence_id: str, *, offset=0, limit=8000) -> dict:
    """Expose signed review evidence by namespace ID, not arbitrary host paths."""
    from .common import sanitize_model_text
    if not isinstance(evidence_id,str) or not re.fullmatch(r'execution-review:[0-9a-f]{64}',evidence_id):
        raise ValueError('unsafe execution review evidence id')
    if isinstance(offset,bool) or not isinstance(offset,int) or offset<0:
        raise ValueError('offset must be a nonnegative byte position')
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=12000:
        raise ValueError('limit must be 1..12000 bytes')
    root=Path(root).resolve(strict=True)
    identity=evidence_id.split(':',1)[1]
    directory=root/'candidate_execution_reviews'
    path=directory/f'{identity}.json'
    request_path=directory/'requests'/f'{identity}.json'
    for candidate in (directory,directory/'requests',path,request_path):
        if candidate.is_symlink() or not candidate.exists():
            raise ValueError('execution review evidence missing or unsafe')
    if path.stat().st_size>256*1024 or request_path.stat().st_size>256*1024:
        raise ValueError('execution review evidence exceeds bounded store')
    request=read_json(request_path)
    if not isinstance(request,dict) or object_digest(request.get('payload'))!=identity:
        raise ValueError('execution review request identity changed')
    record=read_json(path)
    if not isinstance(record,dict):
        raise ValueError('execution review evidence is not a record')
    signature=record.pop('record_sha256',None)
    if (signature!=object_digest(record) or record.get('material_sha256')!=identity
            or record.get('schema_version')!=1
            or record.get('status') not in {'reviewing','reviewed','unavailable'}
            or record.get('request_ref')!=request_path.relative_to(root).as_posix()
            or record.get('request_sha256')!=digest(request_path)):
        raise ValueError('execution review evidence signature or request binding changed')
    if record['status']=='reviewed':
        validate(record.get('answer'),request['payload'])
    record['record_sha256']=signature
    # Sanitize the whole document before paging, so a boundary cannot expose a
    # partial private path or credential. Original immutable files remain intact.
    data=sanitize_model_text(json.dumps({'record':record,'request':request},
        ensure_ascii=False),local_roots=(root,)).encode('utf-8')
    return {'evidence_id':evidence_id,'receipt_ref':path.relative_to(root).as_posix(),
            'status':record['status'],'authority':record.get('authority'),
            'log_bytes':len(data),'offset':offset,'next_offset':min(len(data),offset+limit),
            'text':data[offset:offset+limit].decode('utf-8','replace')}


def review(root: Path, repo: Path, stages: dict, idea, client, *, reuse_policy=None, execution_plan=None, policy_metadata=None) -> dict:
    packet=packet_for(root,repo,stages,idea,reuse_policy=reuse_policy,execution_plan=execution_plan,
                      policy_metadata=policy_metadata)
    previous=cached(root,packet)
    if previous:
        if previous.get('status')=='unavailable':
            raise ReviewUnavailable(previous['error'],previous['record_ref'])
        return previous
    path=root/'candidate_execution_reviews'/f'{object_digest(packet)}.json'
    if path.exists() or path.is_symlink():
        raise ValueError('invalid existing execution review preserved; reconcile before reuse')
    directory=path.parent
    if directory.is_symlink():
        raise ValueError('execution review destination is unsafe')
    identity=object_digest(packet)
    request_path=directory/'requests'/f'{identity}.json'
    if request_path.parent.is_symlink() or request_path.is_symlink() or request_path.exists():
        raise ValueError('existing execution review request preserved; reconcile before reuse')
    system=load_prompt('candidate-execution-review')
    atomic_json(request_path,{'system':system,'payload':packet})
    row={'schema_version':1,'material_sha256':identity,'status':'reviewing','at':now(),
         'record_ref':path.relative_to(root).as_posix(),'evidence_id':'execution-review:'+identity,
         'request_ref':request_path.relative_to(root).as_posix(),'request_sha256':digest(request_path),
         'responses':[],'authority':'source plausibility review, not native consumption or score'}
    def seal():
        projected=dict(row)
        projected['record_sha256']=object_digest(row)
        atomic_json(path,projected)
    seal()
    from .agent_client import role_scope
    from .execution_derive import _object
    request=packet
    deadline=time.monotonic()+90
    for attempt in range(2):
        remaining=deadline-time.monotonic()
        if remaining<=0:
            row.update(status='unavailable',error='execution review original window exhausted')
            break
        try:
            with role_scope(client,'supervisor'):
                content, metadata=client.chat_with_metadata(system,
                    json.dumps(request,ensure_ascii=False),max_tokens=1800,timeout=remaining,
                    thinking='disabled',decision_only=True,read_only=True,auto_skills=False,
                    resume_session=False,include_research_context=False)
        except (OSError,ValueError,TypeError,RuntimeError) as exc:
            row.update(status='unavailable',error=f'{type(exc).__name__}: {exc}'[:1200])
            break
        if not isinstance(content,str):
            row.update(status='unavailable',error='execution review response is not text')
            break
        response={'text':content[:65536],'text_sha256':hashlib.sha256(content.encode()).hexdigest(),
                  'truncated':len(content)>65536,
                  'provider_turn':{key:metadata[key] for key in ('turn_id','decision_attempt_id','agent_session_key')
                                   if isinstance(metadata,dict) and key in metadata}}
        row['responses'].append(response)
        answer = None
        try:
            if response['truncated']:
                raise ValueError('execution review response exceeds bounded delivery limit')
            answer=_object(content)
            validate(answer,packet)
            row.pop('error',None)
            row.update(status='reviewed',answer=answer,provider_turn=response['provider_turn'])
            break
        except ValueError as exc:
            response['validation_error']=str(exc)[:1200]
            row.update(status='unavailable',error=str(exc)[:1200])
            seal()
            if attempt:
                break
            request={**packet,'format_feedback':str(exc)[:700],
                     **citation_feedback(answer, packet),
                     'previous_answer': answer if isinstance(answer, dict) else None,
                     'required_citation_ids':'Use 1..4 SHORT exact quotes from the supplied source strings; never reconstruct spacing or JSON escaping. Applicable/inapplicable require candidate_change, workflow_* and offered native routing source. Uncertain may use an empty source_evidence list. Preserve substantive uncertainty, not just repair labels.'}
    seal()
    if row['status']=='unavailable':
        raise ReviewUnavailable(row['error'],row['record_ref'])
    return read_json(path)
