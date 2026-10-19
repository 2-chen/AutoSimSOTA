"""Audited source-only workflow investments, separate from scored policy trials.

No jobs, metric adoption, environment installation, or budget changes happen here.
Only Main can request an exact cleared patch; a separate source review must bind
the non-scoring stage and preservation of the learning/evaluation contracts.
"""
from pathlib import Path
import hashlib
import json
import re
import uuid

from .common import atomic_json, digest, immutable_json, now, object_digest, read_json
from .codepatch import apply_many, check_many, patches_from_change, revert
from .experiment_transaction import safe
from .ideas import CLEARED, IdeaLibrary

STAGES = frozenset({'collect', 'prepare_data', 'convert', 'preprocess'})
CHECKS = ('target_stage_only', 'evaluation_preserved', 'training_preserved',
          'existing_data_preserved', 'protocol_preserved')


def _save(path, row):
    body = {k: v for k, v in row.items() if k != 'record_sha256'}
    atomic_json(path, {**body, 'record_sha256': object_digest(body)})


def _read(path):
    if path.is_symlink() or path.stat().st_size > 2*1024**2:
        raise ValueError('unsafe or oversized workflow patch record')
    row = read_json(path)
    if not isinstance(row,dict):
        raise ValueError('workflow patch record must be an object')
    body = {k: v for k, v in row.items() if k != 'record_sha256'}
    if row.get('record_sha256') != object_digest(body):
        raise ValueError('workflow patch record identity changed')
    if (row.get('schema_version')!=1 or not isinstance(row.get('id'),str)
            or not re.fullmatch('[0-9a-f]{32}',row['id'])
            or not isinstance(row.get('status'),str)
            or row.get('status') not in {'reviewing','refused','prepared','applied','rolled_back'}):
        raise ValueError('workflow patch record schema changed')
    return row


def _record_path(output, identity):
    if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{32}', identity):
        raise ValueError('workflow patch needs its exact transaction ID')
    return safe(Path(output), 'workflow_patches/'+identity+'/state.json')


def view(output, run_id, *, limit=8):
    output = Path(output)
    if output.is_symlink():
        raise ValueError('workflow output cannot be a symlink')
    if not output.exists():
        return []
    directory = safe(output, 'workflow_patches')
    paths = list(directory.glob('*/state.json')) if directory.exists() else []
    if len(paths) > 128:
        raise ValueError('workflow patch history exceeds bounded catalog')
    rows = []
    for path in paths:
        path = _record_path(output, path.parent.name)
        if path.stat().st_size > 2*1024**2:
            raise ValueError('workflow patch record exceeds bound')
        row = _read(path)
        if row.get('run_id') != run_id or row.get('id') != path.parent.name:
            raise ValueError('workflow patch belongs to a different research session')
        rows.append({k: row.get(k) for k in ('id', 'idea_label', 'target_stage', 'status',
                    'created_at', 'verification_required',
                    'source_transition_started', 'evidence_id')})
    rows = sorted(rows, key=lambda r: str(r.get('created_at')))
    return rows if limit is None else rows[-limit:]


def generation(output, run_id):
    """Transition identity for command re-verification, not a source certificate."""
    rows = [r for r in view(output, run_id, limit=None) if r.get('source_transition_started')]
    return object_digest([(r['id'], r['status']) for r in rows]) if rows else ''


def _idle(research):
    from .agent_tasks import records
    from .native_jobs import active_jobs
    root = Path(research.run_root)
    session = read_json(safe(research.output, str((root/'controller_session.json').relative_to(research.output))))
    if not isinstance(session,dict):
        raise ValueError('workflow source session must be an object')
    if (session.get('run_id') != research.run_id or session.get('status') != 'paused'
            or session.get('action') or session.get('pending_action')
            or Path(str(session.get('repository') or '')).resolve() != research.repo.resolve()
            or any((root/'confirmation_attempts').glob('*.json'))):
        raise ValueError('workflow source changes require idle paused development before held-out exposure')
    if active_jobs(research.output) or any(r.get('status') in {'running', 'submitted'}
                                         for r in records(research.output)):
        raise ValueError('native/specialist jobs still own the checkout')
    gpu = Path(research.output)/'task_gpu_budget.json'
    if gpu.exists() and any(r.get('status') == 'active' for r in read_json(gpu)['leases'].values()):
        raise ValueError('an active GPU lease owns the checkout')
    return session


def _owned(research, patches, *, check_preimage=True):
    from .workspace_resources import bindings_for
    output, repo = Path(research.output), Path(research.repo).resolve()
    if Path(research.repo).is_symlink() or not repo.is_relative_to(output.resolve()) or repo==output.resolve():
        raise ValueError('workflow source must be a distinct run-owned isolated checkout')
    manifest = safe(output, 'workspace_snapshot.json')
    if manifest.stat().st_size > 8*1024**2:
        raise ValueError('source manifest exceeds bound')
    document = read_json(manifest)
    if not isinstance(document,dict):
        raise ValueError('workflow source manifest must be an object')
    owned = {row[1] for row in document.get('source_entries', [])
             if isinstance(row, list) and len(row) > 1 and row[0] == 'file'}
    if Path(str(document.get('destination') or '')).resolve() != repo:
        raise ValueError('workflow patch requires this isolated owned source checkout')
    for patch in patches:
        safe(repo, patch.file)
        if patch.file not in owned or any((repo/patch.file).is_relative_to(repo/r['target'])
                                         for r in bindings_for(repo)):
            raise ValueError('workflow patches cannot modify resource bindings or unowned source')
    problems = check_many(patches, repo=repo) if check_preimage else []
    if problems:
        raise ValueError('; '.join(problems))


def _binding(research, label, stage, *, ignore_patch_id=None):
    session = _idle(research)
    library = IdeaLibrary(Path(research.run_root)/'ideas.json')
    idea = library.get(label)
    if idea is None or idea.status != CLEARED or idea.granularity != 'code':
        raise ValueError('choose one exact currently cleared code idea')
    if stage not in STAGES or not research.sources.get(stage):
        raise ValueError('workflow patch needs a declared non-scoring collect/data-preparation stage')
    from .redlines import audit
    verdict = audit(idea, research.red_lines())
    if not verdict.cleared:
        raise ValueError('workflow patch violates current red lines: '+verdict.because)
    patches = patches_from_change(idea.change)
    _owned(research, patches)
    for row in view(research.output, research.run_id, limit=None):
        if row['id'] == ignore_patch_id:
            continue
        if row['status'] == 'applied':
            raise ValueError('restore the active workflow source transaction before applying another')
        if row['status'] not in {'applied', 'rolled_back', 'refused'}:
            raise ValueError('reconcile the unfinished workflow patch before further mutation')
    identities = {p.file: digest(research.repo/p.file) for p in patches}
    bound = {'idea_sha256': object_digest(idea.as_dict()), 'session_sha256': object_digest(session),
             'baseline_sha256': digest(Path(research.run_root)/'measurements/baseline.json'),
             'protocol_sha256': digest(Path(research.run_root)/'comparison_protocol.json'),
             'stage_sources_sha256': object_digest(research.sources), 'files': identities}
    return library, idea, patches, bound


def apply(research, *, idea_label, target_stage, reason, reviewer, decision_context=None,
          before_apply=None):
    if not isinstance(reason, str) or not 1 <= len(reason) <= 1500:
        raise ValueError('a bounded Main-authored workflow investment reason is required')
    library, idea, patches, binding = _binding(research, idea_label, target_stage)
    from .candidate_execution_review import packet_for, validate
    packet = packet_for(research.output, research.repo, research.sources, idea,
        research_root=research.run_root, research_decision_context=decision_context,
        execution_plan={'mode': 'source_only_workflow_investment', 'stages': [target_stage],
                        'will_train': False, 'will_evaluate': False, 'will_collect': False})
    packet['workflow_patch_contract'] = {'target_stage': target_stage,
        'required_checks': list(CHECKS), 'binding': binding,
        'authority': 'source application only; later native commands need fresh verification, '
                     'source-bound admission and separate jobs; not data quality or a score'}
    identity = uuid.uuid4().hex
    directory = safe(research.output, 'workflow_patches/'+identity)
    request_ref = str((directory/'request.json').relative_to(research.output))
    immutable_json(directory/'request.json', packet)
    request_sha = digest(directory/'request.json')
    review_ref = str((directory/'review.json').relative_to(research.output))
    row = {'schema_version': 1, 'id': identity, 'run_id': research.run_id,
           'evidence_id':'workflow-patch:'+identity,
           'idea_label': idea_label, 'target_stage': target_stage, 'created_at': now(),
           'request_ref': request_ref, 'review_ref': review_ref, 'binding': binding,
           'request_sha256': request_sha,
           'reason': reason, 'status': 'reviewing', 'authority': 'not a scored policy experiment',
           'verification_required': True}
    _save(directory/'state.json', row)
    try:
        answer = reviewer(packet)
        validate(answer, packet)
        cited = {r.get('source_id') for r in answer.get('source_evidence', [])}
        postimages = [s for s in packet['sources'] if s['id'].startswith('candidate_effective_source_')]
        if answer.get('verdict') == 'applicable' and ('workflow_'+target_stage not in cited or
                not postimages or not {s['id'] for s in postimages} <= cited or
                not all(s.get('changed_hunks_complete',s.get('complete',False)) for s in postimages) or
                not all(answer.get(k) is True for k in CHECKS)):
            raise ValueError('workflow permission requires exact target/postimage citations and all preservation checks')
    except Exception as exc:
        row.update(status='refused', review_error=type(exc).__name__+': '+str(exc)[:800])
        _save(directory/'state.json', row)
        raise
    immutable_json(directory/'review.json', {'packet_sha256': object_digest(packet), 'answer': answer})
    row['review_sha256'] = digest(directory/'review.json')
    if answer['verdict'] != 'applicable':
        row['status'] = 'refused'
        _save(directory/'state.json', row)
        return row
    if (digest(directory/'request.json') != request_sha or
            _binding(research, idea_label, target_stage, ignore_patch_id=identity)[3] != binding):
        row.update(status='refused', review_error='source or contract changed during review')
        _save(directory/'state.json', row)
        raise ValueError('workflow source/protocol/idea/session changed during review')
    originals = {p.file: (research.repo/p.file).read_text() for p in patches}
    if sum(len(t.encode()) for t in originals.values()) > 1024**2:
        raise ValueError('workflow rollback originals exceed bound')
    immutable_json(directory/'rollback.json', {'before': originals, 'patches': idea.change,
        'after_sha256': {p.file: hashlib.sha256(
            originals[p.file].replace(p.find, p.replace, 1).encode()).hexdigest() for p in patches}})
    row.update(rollback_sha256=digest(directory/'rollback.json'),
               request_sha256=digest(directory/'request.json'), review_sha256=digest(directory/'review.json'))
    row.update(status='prepared', source_transition_started=True)
    _save(directory/'state.json', row)
    if before_apply is not None:
        before_apply(row)
    try:
        apply_many(patches, repo=research.repo)
    except Exception:
        if all(digest(research.repo/p.file) == binding['files'][p.file] for p in patches):
            row.update(status='rolled_back', rollback_reason='apply failed; exact originals restored')
            _save(directory/'state.json', row)
        raise
    row.update(status='applied', after_sha256={p.file: digest(research.repo/p.file) for p in patches},
               applied_at=now())
    _save(directory/'state.json', row)
    try:
        library.note(idea_label, worked=False, outcome='workflow source applied; no score or native job; '
                     'fresh stage verification required', event_id='workflow:'+identity)
    except (OSError, ValueError) as exc:
        row['library_note_error'] = type(exc).__name__
        _save(directory/'state.json', row)
    return row


def restore(research, *, patch_id, reason, before_restore=None):
    _idle(research)
    if not isinstance(reason, str) or not 1 <= len(reason) <= 1500:
        raise ValueError('bounded workflow rollback reason required')
    path = _record_path(research.output, patch_id)
    row = _read(path)
    if row.get('run_id') != research.run_id or row.get('id') != patch_id:
        raise ValueError('workflow rollback belongs to another study')
    if row.get('status') == 'rolled_back':
        return row
    if row.get('status') == 'reviewing':
        # Code application cannot start before a prepared record and rollback archive.
        for name, sha in row['binding']['files'].items():
            if digest(safe(research.repo, name)) != sha:
                raise ValueError('review-only source changed; operator reconciliation required')
        row.update(status='rolled_back', rollback_reason=reason, rolled_back_at=now())
        _save(path, row)
        return row
    if row.get('status') not in {'prepared', 'applied'}:
        raise ValueError('workflow transaction has no restorable source change')
    rollback = safe(research.output, str((path.parent/'rollback.json').relative_to(research.output)))
    if rollback.stat().st_size > 2*1024**2 or digest(rollback) != row.get('rollback_sha256'):
        raise ValueError('workflow rollback archive identity changed')
    original = read_json(rollback)
    patches = patches_from_change(original['patches'])
    if not patches:
        raise ValueError('workflow rollback has no bounded original patches')
    _owned(research, patches, check_preimage=False)
    # Check ALL files before touching any; never overwrite unrelated later changes.
    for patch in patches:
        before = original['before'][patch.file]
        if hashlib.sha256(before.encode()).hexdigest() != row['binding']['files'].get(patch.file):
            raise ValueError('workflow rollback original bytes lost their binding')
        if (research.repo/patch.file).read_text() not in (before, before.replace(patch.find, patch.replace, 1)):
            raise ValueError('source diverged; do not overwrite later work during rollback')
    row.update(status='prepared',transition_phase='restoring')
    _save(path,row)
    if before_restore is not None:
        before_restore(row)
    for patch in reversed(patches):
        revert(patch, original['before'][patch.file], repo=research.repo)
    row.update(status='rolled_back', rollback_reason=reason, rolled_back_at=now(),
               verification_required=True)
    _save(path, row)
    return row


def read_evidence(output, evidence_id, *, offset=0, limit=8000):
    """Paged public request/review, never rollback originals or arbitrary paths."""
    from .common import sanitize_model_payload
    if not isinstance(evidence_id,str) or not re.fullmatch('workflow-patch:[0-9a-f]{32}',evidence_id):
        raise ValueError('unsafe workflow patch evidence ID')
    if isinstance(offset,bool) or not isinstance(offset,int) or offset<0 or isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=12000:
        raise ValueError('bounded nonnegative byte offset and 1..12000 limit required')
    path=_record_path(output,evidence_id.split(':')[1]);row=_read(path)
    if row['id']!=path.parent.name:
        raise ValueError('workflow evidence record belongs to another transaction')
    request=safe(output,str((path.parent/'request.json').relative_to(output)))
    if request.stat().st_size>2*1024**2 or digest(request)!=row.get('request_sha256'):
        raise ValueError('workflow public request identity changed')
    document={'transaction':{k:v for k,v in row.items() if k not in {'binding'}},
              'request':read_json(request),'authority':'read-only source transaction; not execution, data quality or score'}
    review=safe(output,str((path.parent/'review.json').relative_to(output)))
    if review.exists():
        if review.stat().st_size>65536 or digest(review)!=row.get('review_sha256'):
            raise ValueError('workflow review identity changed')
        document['review']=read_json(review)
    # Put the decision before the much larger source packet: callers can inspect
    # the reason without reloading every offered source into their context.
    ordered={k:document[k] for k in ('authority','transaction','review','request') if k in document}
    display=sanitize_model_payload(ordered,local_roots=(Path(output),))
    data=json.dumps(display,ensure_ascii=False).encode();end=min(len(data),offset+limit)
    return {'evidence_id':evidence_id,'record_sha256':row['record_sha256'],
            'display_sha256':hashlib.sha256(data).hexdigest(),'offset':offset,'total_bytes':len(data),
            'next_offset':end if end<len(data) else None,'text':data[offset:end].decode(errors='replace')}
