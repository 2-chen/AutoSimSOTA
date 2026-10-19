"""Source-bound LLM admission; not proof of metric gain or native consumption."""
import json
import hashlib
import re
import time
from copy import deepcopy
from pathlib import Path

from .common import atomic_json, digest, now, object_digest, read_json, sanitize_model_payload
from .prompt_library import load_prompt

FORMAT_FEEDBACK_VERSION = 10
MAX_REVIEW_BYTES = 384 * 1024


def decision_evidence_refs(action) -> list[str]:
    """Explicit current Main citations, not its rationale or incidental tool calls."""
    decision = action.get('research_decision') if isinstance(action,dict) else None
    refs = decision.get('input_evidence') if isinstance(decision,dict) else None
    return [ref[:300] for ref in refs if isinstance(ref,str)][:20] if isinstance(refs,list) else []


def decision_context(action) -> dict | None:
    """Forward the current Main target, not an inference from a historical idea."""
    decision = action.get('research_decision') if isinstance(action, dict) else None
    if not isinstance(decision, dict):
        return None
    context = {key: deepcopy(decision[key]) for key in (
        'question', 'hypothesis', 'expected_outputs', 'postconditions', 'stop_condition')
        if key in decision and decision[key]}
    if not context:
        return None
    context['operation'] = action.get('step')
    context['authority'] = 'current Main-declared experimental target; not evidence it is satisfied'
    if len(json.dumps(context, ensure_ascii=False).encode()) > 32768:
        raise ValueError('current research decision exceeds bounded execution review; narrow its contract')
    return context


def requested_native_ids(ideas, *, limit=12, additional_evidence_refs=()) -> list[str]:
    """Bounded explicit Agent citations; current decision before older proposal."""
    ids = []
    texts = list(additional_evidence_refs) if isinstance(additional_evidence_refs,(list,tuple)) else []
    for idea in reversed(list(ideas)):
        evidence = getattr(idea, 'evidence', None)
        texts.extend(evidence if isinstance(evidence,list) else [])
    for text in texts:
        if not isinstance(text,str):
            continue
        for identity in re.findall(r'(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])',text):
            if identity not in ids and len(ids)<limit:
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


def framework_observations(root: Path, requested_ids=()) -> list[dict]:
    """Forward Agent-cited contract pages, not its interpretation of those pages.

    Recheck both the sealed log and the current source slice. A path mentioned
    in a proposal is never a file read grant; refused/stale/cross-run pages are
    not evidence. These excerpts establish executor semantics, NOT native work.
    """
    from . import harness_repair
    from .common import sanitize_model_text
    from .evidence_store import read_attempt_evidence
    source = Path(harness_repair.__file__).resolve().parents[1]
    root = Path(root)
    directory = root/'framework_source_reads'
    if directory.is_symlink() or not directory.is_dir():
        return []
    rows = []
    for identity in dict.fromkeys(list(requested_ids)[:12]):
        if not isinstance(identity, str) or not re.fullmatch(r'[0-9a-f]{32}', identity):
            continue
        path = directory/(identity+'.json')
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
                continue
            record = read_json(path)
            relative = record.get('source_ref')
            if relative not in harness_repair.READABLE:
                continue
            native = source/relative
            if (native.is_symlink() or not native.resolve().is_relative_to(source)
                    or native.stat().st_size > harness_repair.MAX_SOURCE_BYTES):
                continue
            fingerprint = digest(native)
            start, end = record.get('offset'), record.get('next_offset')
            if (record.get('source_sha256') != fingerprint or type(start) is not int
                    or type(end) is not int or not 0 <= start <= end <= start+240):
                continue
            lines = native.read_text().splitlines()
            if end > len(lines):
                continue
            expected = sanitize_model_text('\n'.join(f'{i+1}: {lines[i]}' for i in range(start,end)))
            if (record.get('excerpt') != expected or
                    record.get('log_sha256') != hashlib.sha256(expected.encode()).hexdigest()):
                continue
            proof = read_attempt_evidence(root,identity,offset=0,limit=8000)
            if (record.get('status') != 'completed' or record.get('returncode') != 0
                    or proof['receipt_ref'] != path.relative_to(root).as_posix()
                    or proof['termination_reason'] != 'framework_source_readonly'
                    or proof['status'] != 'completed' or proof['returncode'] != 0
                    or proof['log_bytes'] != len(expected.encode())
                    or proof['text'] != expected.encode()[:8000].decode('utf-8','replace')
                    or digest(native) != fingerprint):
                continue
            rows.append({'id':f'framework_contract_{identity}', 'text':proof['text'],
                'evidence_id':identity, 'source_ref':relative, 'source_sha256':fingerprint,
                'complete':proof['next_offset']==proof['log_bytes'],
                'authority':'current sealed executor contract excerpt; not native consumption or a score'})
            # requested_native_ids already bounds explicit Agent selections to
            # twelve. A hidden three-page cutoff dropped the fourth cited
            # producer/payload contract and made a correct proposal look
            # unsupported. Keep every authenticated requested page; this is
            # source transport, never an approval or native witness.
        except (OSError,ValueError,TypeError,KeyError,AttributeError):
            continue
    return rows


def source_citation_observations(root: Path, repo: Path, requested_ids=()) -> list[dict]:
    """Deliver explicitly selected captured source, not model-written diagnostics.

    Installed-source and checkout-source reads have different stores from CPU
    diagnostics. Dropping these handles makes a reviewer repeatedly demand code
    Main already read. Resolve the original handles against CURRENT backing bytes
    and environment; do not infer authority from a path in an idea or scan stores.
    """
    from .experiment_transaction import safe
    from .evidence_store import read_attempt_evidence
    from .checkout_source_citations import resolve as resolve_checkout
    from .native_source_citations import resolve as resolve_native
    root, repo = Path(root), Path(repo)
    rows, delivered = [], 0
    stores = (
        ('checkout_source_citations', 'checkout-source', 'checkout_source_captured', resolve_checkout),
        ('native_source_citations', 'native-source', 'native_source_captured', resolve_native))
    for identity in dict.fromkeys(list(requested_ids)[:12]):
        if not isinstance(identity, str) or not re.fullmatch(r'[0-9a-f]{32}', identity):
            continue
        for directory, prefix, status, resolve_source in stores:
            try:
                path = safe(root, f'{directory}/{identity}.json')
                snapshot = safe(root, f'{directory}/{identity}.source')
                if (not path.is_file() or path.stat().st_size > 32000
                        or not snapshot.is_file() or snapshot.stat().st_size > 65536
                        or delivered + snapshot.stat().st_size > 128*1024):
                    continue
                record = read_json(path)
                if record.get('snapshot_ref') != snapshot.relative_to(root).as_posix():
                    continue
                text = snapshot.read_text(encoding='utf-8')
                if not text:
                    continue
                ref = f'{prefix}:{identity}'
                # Checkout captures are already bounded complete source pages;
                # native handles allow an exact <=4000-character subquote while
                # verifying the entire stored file and current environment.
                resolved = resolve_source(root, repo, ref,
                    text if prefix == 'checkout-source' else text[:4000])
                proof = read_attempt_evidence(root, identity, offset=0, limit=1)
                seal = read_json(safe(root, f'evidence/{identity}.json'))
                if (proof['receipt_ref'] != path.relative_to(root).as_posix()
                        or proof['status'] != status or proof['returncode'] is not None
                        or proof['termination_reason'] != 'read_only_source_capture'
                        or proof['log_bytes'] != snapshot.stat().st_size
                        or seal.get('log_ref') != snapshot.relative_to(root).as_posix()
                        or seal.get('log_sha256') != digest(snapshot)
                        or digest(snapshot) != (record.get('snapshot_sha256') or resolved['sha256'])):
                    continue
                rows.append({'id':f'captured_source_{identity}', 'text':text,
                    'evidence_id':identity,'source_ref':ref,'source_sha256':resolved['sha256'],
                    'complete':True,
                    'authority':'Agent-selected sealed source, revalidated against current backing bytes; '
                        'not an execution edge, actual consumption, approval or a score'})
                delivered += len(text.encode('utf-8'))
            except (OSError,ValueError,TypeError,KeyError,AttributeError,RuntimeError):
                continue
    return rows


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


def _hunk_preview(text: str, spans: list[tuple[int, int]], limit: int) -> tuple[str, bool]:
    """Prioritize every changed region; disjoint fragments are explicitly marked."""
    merged = []
    for start, end in sorted((max(0, a-500), min(len(text), b+500)) for a,b in spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    sections = [(f'# [prospective changed-hunk excerpt {i+1}]\n', text[a:b])
                for i,(a,b) in enumerate(merged)]
    complete_hunks = '\n'.join(head+body for head,body in sections)
    if len(complete_hunks.encode()) <= limit:
        return complete_hunks, True
    header_bytes = sum(len((head+'\n').encode()) for head,_ in sections)
    if not sections or header_bytes >= limit:
        return '# [postimage excerpts omitted at byte ceiling]'[:limit], False
    room = (limit-header_bytes)//len(sections)
    marker = b'\n# [middle of changed-hunk excerpt omitted]\n'
    parts = []
    for head,body in sections:
        raw = body.encode()
        if len(raw) > room:
            if room <= len(marker):
                body = raw[:room].decode('utf-8','ignore')
            else:
                edge = (room-len(marker))//2
                body = (raw[:edge].decode('utf-8','ignore')+marker.decode()+
                        raw[-edge:].decode('utf-8','ignore') if edge else marker.decode())
        parts.append(head+body)
    return '\n'.join(parts), False


def _prospective_source(base: dict, text: str, patches: list[dict], *, limit: int | None) -> dict:
    """Read-only exact postimage, not inherited history or executable approval."""
    # codepatch reads UTF-8 text with universal-newline translation before writing.
    text=text.replace('\r\n','\n').replace('\r','\n')
    after, spans = text, []
    for patch in patches:
        find, replace = patch.get('find'), patch.get('replace')
        if (not isinstance(find,str) or not find or not isinstance(replace,str)
                or after.count(find) != 1):
            return {'id':base['id'].replace('candidate_source_','candidate_effective_source_'),
                'repository_file':base['path'], 'complete':False,
                'source_phase':'prospective_postimage_unavailable',
                'text':'Exact patch composition is unavailable against the current checkout. '
                    'Inspect current source and correct the proposal; no historical patch was inherited.'}
        start = after.index(find)
        end = start+len(find)
        delta = len(replace)-len(find)
        # Later patches may precede or overlap earlier ones. Keep coordinates
        # in the final postimage rather than reusing a stale first/last offset.
        spans = [(a if a<start else a+delta if a>=end else start,
                  b if b<=start else b+delta if b>end else start+len(replace))
                 for a,b in spans]
        spans.append((start,start+len(replace)))
        after = after.replace(find,replace,1)
    encoded = after.encode('utf-8')
    complete = limit is None or len(encoded) <= limit
    if complete:
        preview = after
        hunks_complete = True
    else:
        preview, hunks_complete = _hunk_preview(after, spans, limit)
    return {'id':base['id'].replace('candidate_source_','candidate_effective_source_'),
        'repository_file':base['path'], 'source_phase':'prospective_postimage_not_executed',
        'base_sha256':base['sha256'], 'sha256':hashlib.sha256(encoded).hexdigest(),
        'complete':complete, 'changed_hunks_complete':hunks_complete, 'text':preview,
        'authority':'exact current preimage plus ONLY this proposal; not inherited candidate code, execution or a score'}


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
                        'complete':complete, 'text':text,
                        'source_phase':'current_checkout_preimage'})
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
    bases = [row for row in sources if row['id'].startswith('candidate_source_')]
    if bases:
        packet['candidate_base_contract'] = {
            'base':'current checkout bytes, not the last attempted candidate',
            'prior_candidate_effects_assumed':False,
            'rule':'An unsuccessful candidate may have been rolled back. Compose every '
                'still-required repair explicitly against this preimage; descriptions and '
                'historical diagnostic success do not install prior patches. Prospective '
                'sources are deterministic text views, not native readiness or approval.'}
        # Keep the ORIGINAL packet ceiling. Do not evict native/contract evidence
        # to fit new presentation. Escaping and source metadata need headroom.
        packet['prospective_source_delivery_version'] = 2
        prepared = []
        full = []
        for base in bases:
            path=repo/base['path'];raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=base['sha256']:
                raise ValueError('candidate preimage changed during read-only postimage inspection')
            related=[p for p in patches if (repo/str(p.get('file') or '')).resolve()==path.resolve()]
            prepared.append((base, raw.decode('utf-8'), related))
            full.append(_prospective_source(base,raw.decode('utf-8'),related,limit=None))
        full_packet = {**packet, 'sources':sources+full, 'prospective_source_delivery':'complete'}
        full_fits = (all(row.get('complete') for row in full) and
                     len(json.dumps(full_packet,ensure_ascii=False).encode())<=192*1024)
        room = 192*1024-len(json.dumps(packet,ensure_ascii=False).encode())
        limit = min(12000,max(0,(room-1024*len(bases))//(3*len(bases))))
        packet['prospective_source_delivery'] = ('bounded' if limit>=1024 else
                                                'omitted_at_original_packet_ceiling')
        if full_fits:
            sources.extend(full)
            packet['prospective_source_delivery'] = 'complete'
        elif limit>=1024:
            for base,text,related in prepared:
                sources.append(_prospective_source(base,text,related,limit=limit))
    # Host paths stay redacted. These are validated PUBLIC checkout-relative
    # filenames, useful for reasoning about routing, not arbitrary path grants.
    for row in sources + native_refs:
        if isinstance(row.get('path'), str):
            row['repository_file'] = row['path']
    if len(json.dumps(packet,ensure_ascii=False).encode()) > 192*1024:
        raise ValueError('execution review packet incomplete: inspect a smaller intervention before GPU work')
    return packet


def normalize_citation_boundaries(answer, packet):
    """Only harmless outer ASCII whitespace; strict validation follows.

    Never edit provider text, repair code tokens/indentation inside a quote,
    remove line numbers or infer a missing citation. Every canonical quote
    must already be an exact offered substring; retain raw responses separately.
    """
    if not isinstance(answer,dict) or not isinstance(answer.get('source_evidence'),list):
        return answer, []
    normalized=deepcopy(answer);changes=[]
    sources={row['id']:row['text'] for row in packet['sources']}
    for index,citation in enumerate(normalized['source_evidence'][:12]):
        if not isinstance(citation,dict):
            continue
        quote=citation.get('quote');source_id=citation.get('source_id')
        source=sources.get(source_id) if isinstance(source_id,str) else None
        if not isinstance(quote,str) or not 4<=len(quote)<=800 or not isinstance(source,str):
            continue
        exact=quote.strip(' \t\r\n')
        if quote not in source and quote!=exact and 4<=len(exact)<=800 and exact in source:
            citation['quote']=exact
            changes.append({'citation_index':index,'source_id':citation['source_id'],
                'kind':'outer_ascii_whitespace_only',
                'original_quote_sha256':hashlib.sha256(quote.encode()).hexdigest(),
                'exact_quote_sha256':hashlib.sha256(exact.encode()).hexdigest()})
    return normalized,changes


def expand_citation_choices(answer, packet, offered_choices=()):
    """Resolve only reviewer-selected, previously offered literal citations.

    Choice IDs bind the complete original packet and exact source/quote pair.
    This is transport, not fuzzy quote repair or a scientific decision. The
    normalized answer still passes validate(); raw provider text is retained.
    """
    if not isinstance(answer, dict) or not isinstance(answer.get('source_evidence'), list):
        return answer, []
    normalized = deepcopy(answer)
    packet_sha = object_digest(packet)
    resolved = []
    for index, citation in enumerate(normalized['source_evidence'][:12]):
        if not isinstance(citation, dict) or 'citation_choice' not in citation:
            continue
        identity = citation['citation_choice']
        if set(citation) != {'citation_choice'} or not isinstance(identity, str):
            raise ValueError('citation_choice must be a sole string field')
        choice = next((row for row in offered_choices
                       if isinstance(row, dict) and row.get('citation_choice') == identity), None)
        if choice is None:
            raise ValueError('citation_choice was not offered in this format retry')
        pair = {key: choice.get(key) for key in ('source_id', 'quote')}
        expected = 'cq_' + object_digest({'packet_sha256': packet_sha, **pair})
        source = next((row for row in packet['sources']
                       if row['id'] == pair['source_id']), None)
        quote = pair['quote']
        if (identity != expected or source is None or not isinstance(quote, str)
                or not 4 <= len(quote) <= 800 or quote not in source['text']):
            raise ValueError('citation_choice no longer binds this exact packet/source')
        normalized['source_evidence'][index] = pair
        resolved.append({'citation_index': index, 'citation_choice': identity,
                         'source_id': pair['source_id'], 'packet_sha256': packet_sha,
                         'exact_quote_sha256': hashlib.sha256(quote.encode()).hexdigest()})
    return normalized, resolved


def bind_offered_source_ids(answer, packet):
    """Bind an explicit source-ID citation; never manufacture a quotation.

    Like consumer review, the reviewer may cite an offered source as a whole.
    Record exact delivered-text AND material identities, separately from literal
    quotes. Old contracts and legacy quotes retain their original validation.
    """
    if not isinstance(answer, dict) or not isinstance(answer.get('source_evidence'), list):
        return answer, []
    normalized = deepcopy(answer)
    bindings = []
    for index, citation in enumerate(normalized['source_evidence'][:12]):
        if not isinstance(citation, dict) or set(citation) != {'source_id'}:
            continue
        if packet.get('review_contract_version', 0) < 12:
            raise ValueError('source-ID references require review contract 12')
        identity = citation['source_id']
        matches = [s for s in packet['sources'] if s['id'] == identity]
        if not isinstance(identity, str) or len(matches) != 1:
            raise ValueError('source-ID reference must uniquely name an offered source')
        bound = {'source_id': identity, 'citation_kind': 'offered_source',
                 'offered_text_sha256': hashlib.sha256(matches[0]['text'].encode()).hexdigest(),
                 'material_sha256': object_digest(packet)}
        normalized['source_evidence'][index] = bound
        bindings.append({'citation_index': index, **bound})
    return normalized, bindings


def _bound_source_matches(citation, packet):
    if (packet.get('review_contract_version', 0) < 12 or not isinstance(citation, dict)
            or set(citation) != {'source_id', 'citation_kind', 'offered_text_sha256', 'material_sha256'}
            or citation['citation_kind'] != 'offered_source'
            or not isinstance(citation['source_id'], str)):
        return False
    matches = [s for s in packet['sources'] if s['id'] == citation['source_id']]
    return (len(matches) == 1 and citation['material_sha256'] == object_digest(packet)
            and citation['offered_text_sha256'] == hashlib.sha256(matches[0]['text'].encode()).hexdigest())


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
        if 'citation_kind' in citation:
            if not _bound_source_matches(citation, packet):
                raise ValueError(f'source_evidence[{index}]: source-ID reference lost its exact material/text binding')
            ids.add(citation['source_id'])
            continue
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
            if _bound_source_matches(citation, packet):
                continue
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
            # Numbered source pages cannot support a reconstructed contiguous
            # multiline block with the line prefixes removed. Recommend literal
            # component lines, still requiring the reviewer to choose and quote
            # the offered bytes; do not normalize a citation into acceptance.
            alternatives.extend(line.strip() for line in decoded.splitlines()[:12])
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
        ids = {c['source_id'] for c in citations or [] if isinstance(c, dict)
               and isinstance(c.get('source_id'), str)}
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
        if native_ids:
            if not set(native_ids).intersection(ids):
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
    packet_sha = object_digest(packet)
    choices = [{'citation_choice': 'cq_' + object_digest({'packet_sha256': packet_sha, **pair}),
                **pair} for pair in suggestions]
    return {'citation_errors': errors[:16], 'exact_quote_suggestions': suggestions,
            'citation_choices': choices,
            'source_id_transport_hint': 'Prefer source_evidence entries containing ONLY '
                '{"source_id":"exact offered id"}. The executor binds the entire delivered '
                'source text and original material hashes, not an invented quote. Required '
                'workflow/candidate/native categories and the independent verdict do not change.',
            'citation_transport_hint': 'A supplied diagnostic can contain nested JSON-encoded '
                'source strings. Quote the offered literal bytes (including backslashes), not '
                'the decoded code. Suggestions are exact offered substrings; copy their quote '
                'value with correct JSON encoding, not invented leading indentation. Correct '
                'every listed invalid citation, or omit an optional citation while retaining '
                'all required sources. Prefer selecting an offered citation_choices ID with '
                '{"citation_choice":"cq_..."} instead of retyping its quote. The executor '
                'expands only your explicit selection to its exact offered source/quote and '
                'revalidates all original rules. No source or verdict is repaired for you.',
            'required_native_source_ids': list(dict.fromkeys(
                r['source_id'] for r in packet.get('native_references') or [])),
            'authority': 'verbatim source matches only; reviewer chooses and revalidates; no verdict inferred'}


def packet_for(root, repo, stages, idea, *, reuse_policy=None, execution_plan=None,
               policy_metadata=None, native_observation_snapshot=None,
               selected_training_settings=None, additional_evidence_refs=(), research_root=None,
               research_decision_context=None):
    packet=material(repo,stages,idea)
    # A read-only catalog may project many plans in one synchronous state read.
    # Share its freshly checked evidence only within that read; execution callers
    # omit this argument and independently revalidate current source/artifacts.
    requested = requested_native_ids([idea],additional_evidence_refs=additional_evidence_refs)
    if (isinstance(native_observation_snapshot, dict) and
            set(requested) <= set(native_observation_snapshot['requested_ids'])):
        observations = select_native_observations(native_observation_snapshot['pool'], requested)
    elif native_observation_snapshot is not None and not requested:
        observations = select_native_observations(native_observation_snapshot)
    else:
        observations = (native_observations(Path(root),Path(repo),requested_ids=requested)
                        if requested else native_observations(Path(root),Path(repo)))
    packet['sources'].extend(observations)
    packet['sources'].extend(framework_observations(root,requested))
    packet['sources'].extend(source_citation_observations(root,repo,requested))
    if reuse_policy is not None and policy_metadata:
        packet['sources'].extend(policy_metadata)
    if research_root is not None:
        from .development_evidence import reader_failure_diagnostics
        directory=Path(research_root)
        if (directory.is_symlink() or directory.resolve().parent !=
                Path(root).resolve()/'research'):
            raise ValueError('historical reader evidence must belong to this run output')
        from .data_campaign import execution_protocols
        protocols = execution_protocols(root, directory.name)
        if protocols:
            packet['frozen_data_studies'] = protocols
            packet['sources'].insert(0, {'id': 'frozen_data_study_requirements',
                'text': json.dumps(protocols, ensure_ascii=False, indent=2),
                'authority': 'full registered comparison requirements; not native execution '
                             'evidence or approval of existing arm observations'})
        for index,failure in enumerate(reader_failure_diagnostics(directory)):
            packet['sources'].append({'id':f'historical_reader_failure_{index}',
                'text':json.dumps(failure,ensure_ascii=False,indent=2),
                'authority':'authenticated historical negative diagnostic; '
                    'not a verdict about this candidate, current source approval or consumption'})
    if research_decision_context is not None:
        if (not isinstance(research_decision_context, dict) or
                len(json.dumps(research_decision_context, ensure_ascii=False).encode()) > 32768):
            raise ValueError('invalid or oversized current research decision context')
        packet['current_research_decision'] = deepcopy(research_decision_context)
        packet['sources'].insert(0, {'id': 'current_research_decision',
            'text': json.dumps(research_decision_context, ensure_ascii=False, indent=2),
            'authority': 'Main-declared target/postconditions, not verified execution or native evidence'})
    packet['review_contract_version']=13
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
    if selected_training_settings is not None:
        packet['selected_training_settings'] = dict(selected_training_settings)
        packet['selected_training_settings_authority'] = (
            'explicit requested call inputs before candidate transformations/registry resolution; '
            'inspect candidate_change for overrides; not proof of training or data consumption')
        from .data_versions import PREFIX
        inputs = {'requested_training_settings':dict(selected_training_settings),
            'requested_data_handles':[{'setting':key,'data_version_id':value[len(PREFIX):]}
                for key,value in selected_training_settings.items()
                if isinstance(value,str) and value.startswith(PREFIX)],
            'authority':packet['selected_training_settings_authority']}
        from .candidate_settings import TRAINING_SEED
        if TRAINING_SEED in selected_training_settings:
            inputs['seed_scopes'] = {
                'requested_train_seed':selected_training_settings[TRAINING_SEED],
                'evaluation_seed':selected_training_settings.get('seed'),
                'authority':'executor phase-scoped inputs, not proof of native RNG consumption; '
                    'only the train builder seed slot receives the training override'}
        # A tool-less reviewer can cite caller facts instead of claiming that
        # keys are absent because they are metadata outside the source list.
        packet['sources'].insert(0,{'id':'selected_training_inputs',
            'text':json.dumps(inputs,ensure_ascii=False,indent=2),
            'authority':'controller-provided requested inputs; not native execution evidence'})
        # Keep the small execution context ahead of large repository excerpts.
        # This changes delivery order, not source authority or cache identity.
        packet = {key:packet[key] for key in (
            'candidate','execution_plan','selected_training_settings',
            'selected_training_settings_authority')} | {
                key:value for key,value in packet.items() if key not in {
                    'candidate','execution_plan','selected_training_settings',
                    'selected_training_settings_authority'}}
    size = len(json.dumps(packet,ensure_ascii=False).encode())
    if size > MAX_REVIEW_BYTES:
        raise ValueError('bounded review including native diagnostics is too large: '
            f'{size} bytes exceeds {MAX_REVIEW_BYTES}; select narrower authenticated source '
            'pages/diagnostics, not an unchanged retry or a repository-infeasibility claim')
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
    # The request encloses the allowed packet plus its short system prompt.
    # Read bounds must not be smaller than the writer's packet bound.
    if path.stat().st_size>256*1024 or request_path.stat().st_size>MAX_REVIEW_BYTES+65536:
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


def review(root: Path, repo: Path, stages: dict, idea, client, *, reuse_policy=None, execution_plan=None,
           policy_metadata=None, selected_training_settings=None, additional_evidence_refs=(),
           research_root=None, research_decision_context=None) -> dict:
    packet=packet_for(root,repo,stages,idea,reuse_policy=reuse_policy,execution_plan=execution_plan,
                      policy_metadata=policy_metadata, selected_training_settings=selected_training_settings,
                      additional_evidence_refs=additional_evidence_refs, research_root=research_root,
                      research_decision_context=research_decision_context)
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
    offered_choices=[]
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
            answer,source_bindings=bind_offered_source_ids(answer,packet)
            answer,choice_resolutions=expand_citation_choices(answer,packet,offered_choices)
            answer,boundary_changes=normalize_citation_boundaries(answer,packet)
            validate(answer,packet)
            if source_bindings:
                response['source_id_bindings']=source_bindings
            if choice_resolutions:
                response['citation_choice_resolutions']=choice_resolutions
            if boundary_changes:
                response['citation_boundary_normalizations']=boundary_changes
            row.pop('error',None)
            row.update(status='reviewed',answer=answer,provider_turn=response['provider_turn'])
            break
        except ValueError as exc:
            response['validation_error']=str(exc)[:1200]
            row.update(status='unavailable',error=str(exc)[:1200])
            seal()
            if attempt:
                break
            feedback=citation_feedback(answer, packet)
            offered_choices=feedback['citation_choices']
            request={**packet,'format_feedback':str(exc)[:700],
                     **feedback,
                     'previous_answer': answer if isinstance(answer, dict) else None,
                     'required_citation_ids':'Prefer source_id-only references for all required categories, up to 12, instead of copying escaped code. Legacy quotes must remain exact offered substrings. Applicable/inapplicable require candidate_change, workflow_* and offered native routing source. Uncertain may use an empty source_evidence list. Preserve substantive uncertainty, not just repair labels.'}
    seal()
    if row['status']=='unavailable':
        raise ReviewUnavailable(row['error'],row['record_ref'])
    return read_json(path)
