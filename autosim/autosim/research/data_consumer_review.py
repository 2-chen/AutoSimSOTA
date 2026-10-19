"""Re-review changed consumers without rewriting immutable data registrations.

Only an executor-owned, applied candidate transaction may request a new review.
Cached independent approvals still require the same registry, current cited source
bytes, environment and selection key. They are not quality/consumption certificates.
"""
from __future__ import annotations

import re
import json
import ast
from pathlib import Path

from .common import digest, immutable_json, object_digest, read_json, sanitize_model_payload
from .experiment_transaction import safe

CHECKS = ('allowed_training_data', 'loader_connection_supported', 'sources_supported')
RESPONSE_PROTOCOL = 3
MAX_REASON_CHARS = 8192
MAX_CURRENT_CONTEXT_BYTES = 64 * 1024
CONTEXT_CODE_SUFFIXES = {'.py', '.pyi', '.sh', '.bash', '.c', '.cc', '.cpp', '.h', '.hpp', '.rs', '.js', '.ts'}


def _executor_contract_specs():
    """Bound current public first-party I/O code, never repository/resource reads.

    A native argv builder does not itself export executor-owned metadata. Show
    that separate, fixed interface without disclosing the selected host path.
    These exact current source excerpts are admission context, not runtime proof.
    """
    root=Path(__file__).resolve().parent
    specs=[];total=0
    for filename,names in [('data_versions.py',{'resolve','training_metadata','native_selection_payload'}),
                           ('declarative_backend.py',{'staging_environment'}),
                           ('derived_research.py',set())]:
        path=root/filename;text=path.read_text();lines=text.splitlines(keepends=True)
        tree=ast.parse(text);nodes=[]
        if names:
            nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
                   and n.name in names]
            if {n.name for n in nodes}!=names:
                raise ValueError('public executor I/O contract changed; explicit source review required')
        else:
            for node in ast.walk(tree):
                if isinstance(node,ast.Assign):
                    if any((isinstance(t,ast.Name) and t.id=='selected_training_data') or
                           (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name)
                            and t.value.id=='surroundings' and isinstance(t.slice,ast.Constant)
                            and t.slice.value=='AUTOSIM_TRAIN_SELECTION_JSON') for t in node.targets):
                        nodes.append(node)
                elif (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                      and node.func.id=='isolated_argv' and any(
                          k.arg=='native_environment' and isinstance(k.value,ast.Name)
                          and k.value.id=='surroundings' for k in node.keywords)):
                    nodes.append(node)
        for node in sorted(nodes,key=lambda n:n.lineno):
            start,end=node.lineno-1,node.end_lineno
            # Include the immediate call site after creation of typed metadata.
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and
                    t.id=='selected_training_data' for t in node.targets):
                end=min(len(lines),end+5)
            quote=''.join(lines[start:end]);total+=len(quote.encode())
            if total>16384:
                raise ValueError('public executor I/O contract exceeds the bounded review context')
            specs.append({'ref':f'executor-contract:research/{filename}:{start+1}-{end}',
                          'quote':quote,'sha256':digest(path),
                          'authority':'current public executor I/O source; not native reads, batches or score'})
    return specs


def _format_failure(record):
    review = record.get('review') or {}
    return (record.get('status') == 'unavailable' or
            (record.get('response_protocol') is None and
             bool(review.get('response_sha256')) and
             str(review.get('why', '')).startswith('completed review had an invalid response/citation:')))


def _authenticate(record, material):
    if (record.get('material') != material or record.get('sha256') != object_digest(
            {k:v for k,v in record.items() if k != 'sha256'})):
        raise ValueError('cached consumer review is corrupt; cannot reuse its decision')


def _record_refs(base):
    # Keep genuine approvals/refusals from earlier protocols reachable. A newer
    # parser is not permission to erase or repurchase a scientific verdict.
    return [base+'.json', *[base+f'.format-v{version}.json'
                            for version in range(2, RESPONSE_PROTOCOL+1)]]


def _response(answer, material, shown):
    from .execution_derive import _object
    from .source_citations import resolve
    review = _object(answer)
    if any(type(review.get(key)) is not bool for key in CHECKS):
        raise ValueError('all three consumer checks require JSON booleans')
    why = review.get('why')
    if not isinstance(why, str) or not why.strip() or len(why) > MAX_REASON_CHARS:
        raise ValueError(f'why requires nonempty prose of at most {MAX_REASON_CHARS} characters')
    if ('manifest_observations_supported' in review and
            type(review['manifest_observations_supported']) is not bool):
        raise ValueError('manifest_observations_supported requires a JSON boolean')
    review['citations'] = resolve(material['source_specs'], shown, review.get('citations'))
    return review


def _original_source(output, repo, citation):
    """Authenticate historical checkout evidence, without pretending it is current."""
    ref = citation['ref']
    if not ref.startswith('checkout-source:'):
        return ref
    if not re.fullmatch(r'checkout-source:[0-9a-f]{32}', ref):
        raise ValueError('invalid original checkout source handle')
    identity = ref.split(':')[1]
    row = read_json(safe(output, f'checkout_source_citations/{identity}.json'))
    if (row.get('schema_version') != 1 or row.get('id') != identity
            or row.get('repo') != str(repo.resolve())
            or row.get('sha256') != object_digest({k:v for k,v in row.items() if k != 'sha256'})
            or row.get('source_sha256') != citation.get('sha256')
            or row.get('snapshot_ref') != f'checkout_source_citations/{identity}.source'):
        raise ValueError('original source evidence identity changed')
    snapshot = safe(output, row['snapshot_ref'])
    if (digest(snapshot) != row['snapshot_sha256'] or
            citation.get('quote') not in (snapshot.read_text(), row['display_quote'])):
        raise ValueError('original source snapshot changed; cannot reapprove corrupted history')
    return row['source_ref']


def _material(output, repo, row, setting):
    from .baseline_reference import checkout_source, source_evidence
    from .experiment_bundle import artifact_identity
    from .baseline_reference import checkpoint_location
    from .data_lineage import verify
    version = row['id']
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', version):
        raise ValueError('invalid data version identity')
    path = safe(output, f'data_versions/{version}.json')
    if read_json(path) != row or row.get('status') != 'reviewed_not_consumed' or not all(
            row.get('review', {}).get(key) is True for key in CHECKS):
        raise ValueError('original data registration/review is not intact')
    source, _ = checkpoint_location(output, repo, row.get('native_ref') or row['dataset_ref'])
    if str(source) != row['source'] or artifact_identity(source) != (row['identity_field'], row['identity']):
        raise ValueError('data changed; consumer review cannot approve different data')
    verify(output, repo, row.get('asset') or {})
    refs = row.get('source_refs')
    if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
        raise ValueError('consumer refresh requires original source evidence')
    specs = []
    for citation in refs:
        if citation['ref'].startswith('native-source:'):
            # Changed external SDKs need explicit Data/Fix investigation, not guessed
            # root indexes or a silent recapture from another environment.
            current = source_evidence(repo, [citation], output=output)[0]
            if current != citation:
                raise ValueError('native SDK changed; explicit source review required')
            specs.append(current)
        else:
            relative = _original_source(output, repo, citation)
            _, text, canonical, sha = checkout_source(repo, relative)
            if not text or len(text) > 4000:
                raise ValueError('current source page needs an explicit bounded recapture')
            specs.append({'ref': canonical, 'quote': text, 'sha256': sha})
    environment = safe(output, 'environment.json')
    stages = safe(output, 'derived_stages.json')
    train_interface = read_json(stages).get('train') if stages.is_file() else None
    return {'schema_version': 1, 'version_id': version, 'record_sha256': digest(path),
            'setting': setting, 'source_specs': specs,
            'train_interface_sha256': object_digest(train_interface),
            'environment_sha256': digest(environment) if environment.is_file() else None}


def _expanded_material(output, repo, row, setting, *, base=None, include_executor_contract=True):
    """Current context for ALREADY cited checkout files, not a new path grant.

    Historical line ranges can omit the constructor/loader after a candidate
    inserts code. Bind bounded whole-file context separately; never rewrite the
    registration or invalidate a still-current legacy approval just for formatting.
    External SDK citations retain their original explicit source checks.
    """
    from .baseline_reference import checkout_source
    material = base if base is not None else _material(output, repo, row, setting)
    extra, omitted, seen, total = [], [], set(), 0
    # Prefer exact code paths named by the verified TRAIN caller over historical
    # collection/config references. This is a source edge, not a benchmark-name
    # heuristic or a policy/data choice. Otherwise an unrelated large collection
    # module can consume the bound before the actual loader is reached.
    stage_path = safe(output,'derived_stages.json')
    train = read_json(stage_path).get('train') if stage_path.is_file() else None
    if isinstance(train,dict):
        train = train.get('source')
    direct = set()
    if isinstance(train,str):
        try:
            direct = {node.value.lstrip('/') for node in ast.walk(ast.parse(train))
                if isinstance(node,ast.Constant) and isinstance(node.value,str)
                and Path(node.value).suffix.lower() in CONTEXT_CODE_SUFFIXES}
        except SyntaxError:
            pass
    citations = sorted(material['source_specs'],key=lambda item:
        re.sub(r':[1-9][0-9]*(?:-[1-9][0-9]*)?$', '', item['ref']) not in direct)
    for citation in citations:
        reference = citation['ref']
        if reference.startswith('native-source:'):
            continue
        relative = re.sub(r':[1-9][0-9]*(?:-[1-9][0-9]*)?$', '', reference)
        if relative in seen or Path(relative).suffix.lower() not in CONTEXT_CODE_SUFFIXES:
            continue
        seen.add(relative)
        _, text, canonical, sha = checkout_source(repo, relative)
        if any(item['quote'] == text for item in material['source_specs']
               if item['ref'].split(':')[0] == relative):
            continue  # The original page already contains this complete file.
        size = len(text.encode('utf-8'))
        if size + total > MAX_CURRENT_CONTEXT_BYTES:
            omitted.append({'ref':canonical,'sha256':sha,'bytes':size,
                            'reason':'complete source exceeds remaining 64 KiB context bound'})
            continue
        extra.append({'ref':canonical,'quote':text,'sha256':sha})
        total += size
    if not extra and not omitted:
        return material
    return {**material, 'source_specs':[*material['source_specs'], *extra,
                *(_executor_contract_specs() if include_executor_contract else [])],
            'current_context_version':3 if include_executor_contract else 2, 'context_omissions':omitted,
            'context_authority':'bounded complete CURRENT checkout files already explicitly cited '
                'by the registration; omissions are not evidence; no execution or SDK access implied'}


def accepted(output, repo, row, setting):
    """Return a still-current approval; absence never restores a stale permission."""
    try:
        base_material = _material(output, repo, row, setting)
        materials=[base_material]
        for contracts in (False,True):
            expanded=_expanded_material(output,repo,row,setting,base=base_material,
                                        include_executor_contract=contracts)
            if expanded not in materials:
                materials.append(expanded)
        for material in materials:
            identity = object_digest(material)
            base = f'data_consumer_reviews/{row["id"]}/{identity}'
            for ref in _record_refs(base):
                path = safe(output, ref)
                if not path.exists():
                    continue
                record = read_json(path)
                _authenticate(record, material)
                if _format_failure(record) or not all(
                        (record.get('review') or {}).get(key) is True for key in CHECKS):
                    continue
                if ((row.get('asset') or {}).get('profile_witness', {}).get('inspector') and
                        record['review'].get('manifest_observations_supported') is not True):
                    continue
                return {'record_ref': ref, 'record_sha256': digest(path),
                        'authority': 'independent current consumer review; not actual training consumption'}
        return None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def accepted_for_completed_training(output, repo, row, setting, *, context):
    """Reuse only the sealed ORIGINAL approval of an already completed producer.

    Fresh training still needs accepted() against the current executor. Historical
    training is not replayed; its native source/data/environment must still match,
    while a later first-party writer implementation is not what produced its log.
    This returns the original review identity, not a new permission or consumption
    certificate. Only restore_selection supplies its authenticated selection body.
    """
    from .evidence_store import read_attempt_evidence
    from .data_versions import native_selection_payload
    measurement=context['measurement']; run_id=context['run_id']
    bindings=context['bindings']
    binding=next(b for b in bindings if b['setting']==setting and b['id']==row['id'])
    held=binding.get('consumer_review') or {}
    if (set(held)!={'record_ref','record_sha256','authority'} or
            not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',run_id)):
        raise ValueError('historical consumer approval is missing or malformed')
    prefix=f'data_consumer_reviews/{row["id"]}/'
    ref=held['record_ref']
    if not isinstance(ref,str) or not ref.startswith(prefix):
        raise ValueError('historical consumer approval leaves its registered version')
    path=safe(output,ref)
    if not path.is_file() or path.stat().st_size>2*1024**2 or digest(path)!=held['record_sha256']:
        raise ValueError('historical consumer approval changed')
    record=read_json(path);material=record.get('material')
    if not isinstance(material,dict):
        raise ValueError('historical consumer approval lacks bound source material')
    _authenticate(record,material)
    if (_format_failure(record) or not all(record.get('review',{}).get(k) is True for k in CHECKS)
            or ((row.get('asset') or {}).get('profile_witness',{}).get('inspector') and
                record['review'].get('manifest_observations_supported') is not True)):
        raise ValueError('historical consumer approval was not accepted')
    label=measurement.get('label','')
    if not re.fullmatch(r'round_[1-9][0-9]*',label):
        raise ValueError('historical consumer recovery requires its exact candidate label')
    transaction_ref=f'research/{run_id}/code_changes/round-{label[6:]}.json'
    if record.get('candidate_transaction_ref')!=transaction_ref:
        raise ValueError('historical consumer approval belongs to another candidate')
    transaction=read_json(safe(output,transaction_ref))
    if transaction.get('status')!='applied' or transaction.get('round')!=int(label[6:]):
        raise ValueError('historical candidate transaction is not intact')
    train=measurement.get('train') or {}; attempt=train.get('attempt_id','')
    if not re.fullmatch(r'[0-9a-f]{32}',attempt):
        raise ValueError('historical consumer needs its exact completed training producer')
    receipt_ref=f'research/{run_id}/attempts/{attempt}/receipt.json'
    receipt=read_json(safe(output,receipt_ref));proof=read_attempt_evidence(output,attempt,limit=1)
    if (receipt.get('attempt_id')!=attempt or receipt.get('run_id')!=run_id or
            receipt.get('stage')!='train' or receipt.get('status')!='completed' or
            receipt.get('returncode')!=0 or receipt.get('termination_reason')!='normal_exit' or
            proof.get('status')!='completed' or proof.get('returncode')!=0 or
            proof.get('receipt_ref')!=receipt_ref or
            receipt.get('settings_digest')!=object_digest(measurement['settings']) or
            Path(receipt.get('working_directory','')).resolve()!=Path(repo).resolve()):
        raise ValueError('historical consumer training receipt is not consistent')
    labels=[{'id':b['id'],'setting':b['setting'],'path':measurement['settings'][b['setting']]}
            for b in bindings]
    expected=json.loads(native_selection_payload({'selected_training_data':labels}))
    environment=receipt.get('environment') or {}
    if (environment!=train.get('environment') or
            json.loads(environment.get('AUTOSIM_TRAIN_SELECTION_JSON','null'))!=expected):
        raise ValueError('historical training did not seal this exact selection metadata')
    # Keep full hashes of native consumers, dependency citations, dataset/registry,
    # train interface and environment. Only old executor-context source excerpts
    # belong to the historical writer rather than today's writer implementation.
    historical=dict(material)
    historical['source_specs']=[s for s in material['source_specs']
        if not re.fullmatch(r'executor-contract:research/(?:data_versions|declarative_backend|derived_research)\.py:[1-9][0-9]*-[1-9][0-9]*',str(s.get('ref','')))]
    if material.get('current_context_version')==3:
        historical['current_context_version']=2
    base=_material(output,repo,row,setting)
    current=[base,_expanded_material(output,repo,row,setting,base=base,include_executor_contract=False)]
    if historical not in current:
        raise ValueError('historical consumer native source/data/environment changed')
    return dict(held)


def _review_evidence(output, ref, record, material, *, approved):
    """Make a completed, authenticated review reachable by the Agent's ID reader."""
    from .evidence_store import capture_attempt_evidence
    _authenticate(record, material)
    identity_fields = {'kind': 'data_consumer_review', 'material': material}
    if record.get('response_protocol') is not None:
        identity_fields['response_ref'] = ref
    identity = object_digest(identity_fields)[:32]
    return capture_attempt_evidence(output, attempt_id=identity, log=safe(output, ref),
        receipt_ref=ref,
        status=('consumer_review_approved' if approved else
                'consumer_review_unavailable' if _format_failure(record) else 'consumer_review_refused'),
        returncode=None, termination_reason='independent_data_consumer_review')


def resolve_candidate(research, settings, *, round_index):
    """One bounded independent review of an already-selected code intervention.

    No automatic source edits, data regeneration, rollback or retry of native work.
    Normal resolve stays closed unless an exact approval has been saved.
    """
    from .data_versions import resolve, PREFIX
    from .baseline_reference import SourceCitationError, source_evidence
    try:
        return resolve(research.output, research.repo, settings)
    except (SourceCitationError, ValueError) as original:
        # The transaction, not an Agent claim, establishes the pre-native boundary.
        from .agent_client import role_scope
        from .source_citations import project, INSTRUCTION
        output, repo = research.output, research.repo
        transaction_ref = f'research/{research.run_id}/code_changes/round-{round_index}.json'
        failure_evidence = None
        failure_review_summary = None
        pre_native_verified = False
        review_call_active = False
        try:
            path = safe(output, transaction_ref)
            transaction = read_json(path)
            transaction_sha = digest(path)
            session_path = safe(output, f'research/{research.run_id}/controller_session.json')
            session = read_json(session_path)
            session_sha = digest(session_path)
            pending = session.get('pending_action') or {}
            from .native_jobs import active_jobs
            if (transaction.get('status') != 'applied' or transaction.get('round') != round_index
                    or session.get('status') != 'running'
                    or session.get('action') != f'round_{round_index}'
                    or pending.get('round') != round_index
                    or pending.get('phase') != 'preparing_idea'
                    or transaction.get('idea_label') != (pending.get('idea') or {}).get('label')
                    or safe(output, f'research/{research.run_id}/measurements/round_{round_index}.json').exists()
                    or active_jobs(output)
                    or not getattr(research.client, 'supports_candidate_execution_review', False)):
                raise ValueError('current applied candidate and independent reviewer required')
            pre_native_verified = True
            for setting, handle in settings.items():
                if not isinstance(handle, str) or not handle.startswith(PREFIX):
                    continue
                version = handle[len(PREFIX):]
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', version):
                    raise ValueError('invalid data version handle')
                row = read_json(safe(output, f'data_versions/{version}.json'))
                try:
                    if source_evidence(repo, row['source_refs'], output=output) == row['source_refs']:
                        continue  # Another error is not a reason to buy a new review.
                except SourceCitationError:
                    pass
                base_material = _material(output, repo, row, setting)
                legacy_materials=[base_material,_expanded_material(output,repo,row,setting,
                    base=base_material,include_executor_contract=False)]
                for legacy_material in legacy_materials:
                    base_name = f'data_consumer_reviews/{version}/{object_digest(legacy_material)}'
                    for legacy_ref in _record_refs(base_name):
                        legacy_path = safe(output,legacy_ref)
                        if legacy_path.exists():
                            _authenticate(read_json(legacy_path),legacy_material)
                material = _expanded_material(output, repo, row, setting,base=base_material)
                identity = object_digest(material)
                ref = f'data_consumer_reviews/{version}/{identity}.json'
                destination = safe(output, ref)
                revalidated_from = None
                reusable_review = None
                reusable_responses = []
                if destination.exists():
                    previous = read_json(destination)
                    _authenticate(previous, material)
                    # A legacy format-v2 approval/refusal must not disappear just
                    # because the base record was a pre-protocol format failure.
                    for older_ref in _record_refs(f'data_consumer_reviews/{version}/{identity}')[1:-1]:
                        older_path = safe(output,older_ref)
                        if older_path.exists():
                            older = read_json(older_path)
                            _authenticate(older,material)
                            previous, ref = older, older_ref
                    if (_format_failure(previous) and
                            (previous.get('response_protocol') or 0) < RESPONSE_PROTOCOL):
                        old_ref = ref
                        ref = f'data_consumer_reviews/{version}/{identity}.format-v{RESPONSE_PROTOCOL}.json'
                        destination = safe(output, ref)
                        # Revalidate sealed paid output before buying another call.
                        # Strict booleans/citations/current material still apply;
                        # never do this for a valid negative judgment. All original
                        # bytes and stable evidence IDs remain untouched.
                        # The correction supersedes its first answer. Never pick
                        # an earlier favorable verdict if the last answer cannot
                        # validate under the new protocol.
                        for index, response in list(enumerate(previous.get('responses') or []))[-1:]:
                            if (not isinstance(response,dict) or response.get('truncated') is not False
                                    or not isinstance(response.get('text'),str)
                                    or len(response['text']) > 32768):
                                continue
                            try:
                                specs = material['source_specs']
                                _, old_shown = project({},specs,local_roots=(repo,output))
                                reusable_review = _response(response['text'],material,old_shown)
                            except (ValueError,TypeError,KeyError):
                                continue
                            reusable_responses = previous['responses']
                            revalidated_from = {'record_ref':old_ref,
                                'record_sha256':digest(safe(output,old_ref)), 'response_index':index,
                                'authority':'same authenticated material and sealed paid response; parser upgrade only'}
                            break
                    else:
                        destination = safe(output,ref)
                if destination.exists():
                    if accepted(output, repo, row, setting):
                        continue
                    cached = read_json(destination)
                    failure_evidence = _review_evidence(output, ref, cached, material, approved=False)
                    failure_review_summary = {'record_ref':ref,
                        'record_sha256':digest(destination),
                        'failed_checks':[key for key in CHECKS if (cached.get('review') or {}).get(key) is not True],
                        'review':{key:(cached.get('review') or {}).get(key) for key in
                            (*CHECKS,'manifest_observations_supported','why')}}
                    state = 'unavailable' if _format_failure(cached) else 'refused'
                    raise ValueError(f'this exact consumer review was {state}; obtain new relevant evidence. '
                        + str((cached.get('review') or {}).get('why', 'no review reason'))[:800])
                payload = {'data': {key:row.get(key) for key in
                    ('id', 'identity_field', 'identity', 'bytes', 'files', 'asset')},
                    'selection_key': setting, 'current_source_refs': material['source_specs'],
                    'original_registration_review': row.get('review'),
                    'original_review_authority':'existing immutable approval of data scope; '
                        'not proof of current loader routing, quality, batches or performance',
                    'source_context_omissions':material.get('context_omissions',[]),
                    'requested_training_settings': settings,
                    'registered_input_selection':{'version_id':version,'input_setting':setting,
                        'selection_kind':'registered data-version handle',
                        'record_sha256':material['record_sha256'],
                        'preflight_checks':'immutable registration, allowed original scope, '
                            'current source asset bytes and lineage independently checked before review',
                        'route_authority':'data_versions.resolve replaces this exact input setting '
                            'with its authorized native mount; trainer_builder source must still '
                            'connect the same setting to its actual reader. No private path implied.',
                        'runtime_authority':'not yet consumed; actual reader identity, counts, '
                            'training updates and policy rollout remain mandatory post-execution checks'},
                    'native_train_builder': getattr(research, 'sources', {}).get('train'),
                    'consumer_contract': 'Version ID and selection key are labels only. '
                        'Expected data hashes/counters must never be echoed as execution evidence. '
                        'Actual byte reads, sample_read and completed training_batch updates '
                        'remain separately verified; a training log is not an evaluator input. '
                        'This is source admission BEFORE native execution: missing current batch '
                        'counters are not source-routing disproof. Global episodes/seed describe '
                        'the fixed rollout protocol, not the number of dataset episodes; an explicit '
                        '_autosim_training_seed overrides randomness only for training. Unknown '
                        'asset quality/provenance remains unknown; do not infer a manifest inspector '
                        'requirement from missing metadata or certify it from original permission. '
                        'The registered_input_selection preserves version/slot association across '
                        'private path redaction. Public executor-contract excerpts explain '
                        'resolution and the environment writer outside the native argv builder. '
                        'Missing private paths, future runtime hashes or future batch counts '
                        'are not by themselves evidence against PRE-execution source admission; '
                        'unsupported actual source edges still require abstention.',
                    'candidate_transaction': transaction,
                    'authority': 'same registered data; consumer source changed, no execution claimed'}
                packet, shown = project(payload, material['source_specs'], local_roots=(repo, output))
                system = ('Independently re-review the CURRENT consumer of an unchanged registered training dataset. '
                        'Read only. Original data scope is not a proof of current loader routing. Trace selected '
                        'input key through current loader source and candidate patch, including version label '
                        'and identity computed from actual opened data. Require allowed_training_data, '
                        'loader_connection_supported, sources_supported booleans; also '
                        'manifest_observations_supported when applicable, why and citations. '
                        'Abstain if evidence is incomplete. Do not certify actual batches, quality, score or '
                        'apply changes. Preserve the original registration and all unknowns. '
                        'Return exactly one JSON object, no Markdown or commentary, with keys '
                        'allowed_training_data, loader_connection_supported, sources_supported '
                        f'(JSON booleans), why (nonempty, at most {MAX_REASON_CHARS} characters; prefer concise prose), citations '
                        '(nonempty array of {"source_id":"review-source-N"}). Include the JSON '
                        'boolean manifest_observations_supported when a manifest inspector applies. '
                        'Use false for unsupported claims; false is an actual review verdict, '
                        'not a format correction.' + INSTRUCTION)
                responses = reusable_responses.copy()
                review = reusable_review
                for index in range(0 if reusable_review is not None else 2):
                    review_call_active = True
                    with role_scope(research.client, 'supervisor'):
                        answer, _ = research.client.chat_with_metadata(system,
                            json.dumps(packet, ensure_ascii=False), max_tokens=2400,
                            thinking='disabled', timeout=90, resume_session=False, auto_skills=False,
                            decision_only=True, read_only=True, include_research_context=False)
                    review_call_active = False
                    response = {'attempt': index, 'response_sha256': object_digest(answer),
                        'text': sanitize_model_payload(str(answer)[:32768], local_roots=(repo, output)),
                        'truncated': len(str(answer)) > 32768}
                    responses.append(response)
                    # Seal each paid answer before parsing or asking once for a
                    # format correction; even a second-call transport failure must
                    # not erase the first answer's diagnostic evidence.
                    response_ref = (f'data_consumer_reviews/{version}/responses/'
                        f'{identity}-{index}-{response["response_sha256"]}.json')
                    immutable_json(safe(output, response_ref), response)
                    try:
                        review = _response(answer, material, shown)
                        break
                    except (ValueError, TypeError, KeyError) as exc:
                        review = {'why': 'completed review had an invalid response/citation: '+str(exc)[:600],
                                  'response_sha256': object_digest(answer)}
                        packet = {**packet, 'format_feedback': str(exc)[:600],
                            'previous_response': response['text'],
                            'correction_scope': 'Correct JSON format/types and supplied citation IDs only; '
                                'independently preserve the verdict warranted by the same source evidence.'}
                if (_expanded_material(output, repo, row, setting) != material
                        or digest(path) != transaction_sha or digest(session_path) != session_sha):
                    raise ValueError('consumer, registration or transaction changed during review')
                record = {'schema_version': 1, 'material': material, 'review': review,
                    'response_protocol': RESPONSE_PROTOCOL, 'responses': responses,
                    'status': 'unavailable' if 'citations' not in review else 'reviewed',
                    'candidate_transaction_ref': transaction_ref,
                    'authority': 'current source/selection review only; not quality or training consumption'}
                if revalidated_from is not None:
                    record['revalidated_response_from'] = revalidated_from
                record['sha256'] = object_digest(record)
                immutable_json(destination, record)
                approved = accepted(output, repo, row, setting) is not None
                evidence = _review_evidence(output, ref, record, material, approved=approved)
                if not approved:
                    failure_evidence = evidence
                    failure_review_summary = {'record_ref':ref,
                        'record_sha256':digest(destination),
                        'failed_checks':[key for key in CHECKS if review.get(key) is not True],
                        'review':{key:review.get(key) for key in
                            (*CHECKS,'manifest_observations_supported','why')}}
                    raise ValueError('independent current consumer review did not pass. '
                        + str(review.get('why', 'no review reason'))[:800])
            return resolve(output, repo, settings)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
            if review_call_active:
                # Preserve provider/transport/budget identity for the normal runtime
                # recovery path. An unavailable review is not a dataset/source fault.
                raise
            # Keep the existing source-preparation recovery path and stable reason.
            evidence_hint = (f'; read_evidence evidence_id={failure_evidence["evidence_id"]}'
                             if failure_evidence else '')
            failure = SourceCitationError('consumer_review_required', index=0, ref=None,
                detail=f'current consumer revalidation refused{evidence_hint}: {exc}; preserve data and applied patch')
            if failure_evidence:
                failure.details.update({key:failure_evidence[key] for key in
                    ('evidence_id', 'evidence_ref', 'log_ref', 'log_sha256')})
            if pre_native_verified:
                # Keep the authentic source judgment separate from the failed
                # admission operation. Its full why/failed checks must reach Fix
                # before large source packets, not disappear behind an 800-char
                # exception prefix or an unrelated older artifact failure.
                from .operation_failure import seal
                sealed = seal(output,operation='data_consumer_revalidation',exc=exc,
                    native_operation_status='not_started',context={
                        'failure_category':'data_consumer_review_required','repair_owner':'fix',
                        'consumer_review_summary':failure_review_summary,
                        'consumer_review_evidence_id':(failure_evidence or {}).get('evidence_id'),
                        'candidate_transaction_ref':transaction_ref,
                        'candidate_transaction_sha256':transaction_sha,
                        'controller_session_sha256':session_sha,
                        'pre_native_boundary':'authenticated applied candidate, preparing_idea phase, no native jobs',
                        'authority':'read-only repair handoff; not a review override, rollback or native retry'})
                failure.planning_failure = {**sealed,
                    'failure_category':'data_consumer_review_required','repair_owner':'fix',
                    'retry_original_arguments':False,'next_operation':'research_task',
                    'guidance':'Read the sealed current consumer refusal and inspect the actual opened-data '
                        'connection. Preserve all datasets, reviews, scores and pending source ownership; '
                        'Main chooses reconciliation and a new audited action. No automatic edit or replay.'}
            raise failure from original
