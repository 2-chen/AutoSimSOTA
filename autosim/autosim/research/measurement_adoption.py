"""Explicit adoption of linked development measurements, without rerunning experiments.

The agent chooses a label and the exact expected incumbent. The kernel verifies
archived bytes/protocol/receipts; only best metadata and its report projection move.
"""
import math
import re
from pathlib import Path

from .common import atomic_json, digest, exclusive, now, object_digest, read_json
from .snapshot import Snapshot, Snapshots


def _safe(root, relative):
    root = Path(root).resolve(strict=True)
    path = root/relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
            or not path.resolve().is_relative_to(root)):
        raise ValueError('unsafe adoption evidence path')
    return path


def offered(root):
    """Cheap recommendations only. Full byte checks occur on explicit adoption."""
    root = Path(root)
    rows = []
    try:
        for path in sorted((root/'measurement_adoptions').glob('*.json')):
            receipt = read_json(_safe(root, path.relative_to(root)))
            request = {k:receipt.get(k) for k in ('label','expected_best','measurement_sha256')}
            if (receipt.get('status') == 'prepared' and receipt.get('id') == object_digest(request)
                    and path.name == receipt['id']+'.json'):
                rows.append({**request, 'kind':'reconcile_prepared_adoption',
                             'why':'finish only exact before/after metadata projections; '
                                   'do not rerun training or evaluation'})
        if rows:
            return rows
        if ((root/'confirmation.json').exists()
                or any((root/'confirmation_attempts').glob('*.json'))):
            return []
        best = Snapshots(root/'snapshots').best()
        protocol = read_json(_safe(root, 'comparison_protocol.json'))
        for path in sorted((root/'measurements').glob('*.json')):
            if path.is_symlink():
                continue
            row = read_json(path)
            label = row.get('label', '')
            score = row.get('metric_utility')
            scale = ('success_rate' if (row.get('metric') or {}).get('name') == 'success_rate'
                     else 'metric:'+str((row.get('metric') or {}).get('name') or ''))
            if (not re.fullmatch(r'round_[1-9][0-9]*_handoff_correction', label)
                    or path.name != label+'.json' or row.get('ok') is not True
                    or not row.get('handoff_correction') or row.get('confirmation')
                    or row.get('metric') != protocol.get('metric')
                    or (row.get('guardrails') or {}).get('status') in {'unknown', 'violated'}
                    or type(score) not in (int, float) or not math.isfinite(score)
                    or not row.get('source_state')
                    or best is None or best.scale != scale or best.score is None
                    or score <= best.score):
                continue
            rows.append({'label':label, 'metric_value':row.get('metric_value'),
                         'utility':score, 'expected_best':best.name,
                         'measurement_sha256':digest(path),
                         'why':'linked development measurement improves the current incumbent; '
                               'explicit adoption rechecks sealed receipts and archived bytes'})
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return []
    return rows


def _finish(root, receipt_path, receipt):
    """Recover only exact before/after projections after an interrupted commit."""
    for relative, expected in receipt['preserved'].items():
        if digest(_safe(root, relative)) != expected:
            raise ValueError('adoption preserved evidence changed')
    for relative, expected in receipt['preserved_output'].items():
        if digest(_safe(root.parent.parent, relative)) != expected:
            raise ValueError('adoption sealed evidence changed')
    from .receipt_verifier import verify_measurement
    if verify_measurement(root, receipt['label']).get('status') != 'consistent':
        raise ValueError('adoption evidence changed before commit')
    for key in set(receipt['best']['files'].values())- {''}:
        if not re.fullmatch(r'[a-f0-9]{64}',key) or digest(_safe(root,'snapshots/blobs/'+key)) != key:
            raise ValueError('adoption source blob changed before commit')
    for relative, versions in receipt['projections'].items():
        path = _safe(root, relative)
        current = object_digest(read_json(path))
        if current not in {versions['before_sha256'], object_digest(versions['after'])}:
            raise ValueError('adoption projection changed outside its transaction')
    for relative, versions in receipt['projections'].items():
        path = _safe(root, relative)
        if object_digest(read_json(path)) == versions['before_sha256']:
            atomic_json(path, versions['after'])
    atomic_json(receipt_path, {**receipt, 'status':'applied'})
    return {'ok':True, 'adoption_id':receipt['id'], 'best':receipt['best'],
            'baseline_and_history_unchanged':True,
            'receipt_ref':str(receipt_path.relative_to(root.parent.parent))}


def adopt(research, *, label, expected_best, measurement_sha256):
    root = Path(research.run_root)
    _safe(research.output, root.relative_to(research.output))
    request = {'label':label, 'expected_best':expected_best,
               'measurement_sha256':measurement_sha256}
    identity = object_digest(request)
    with exclusive(_safe(root, '.measurement-adoption.lock')):
        receipt_path = _safe(root, 'measurement_adoptions/'+identity+'.json')
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt.get('id') != identity or any(receipt.get(k) != v for k,v in request.items()):
                raise ValueError('adoption receipt identity mismatch')
            if receipt.get('status') == 'applied':
                return {'ok':True, 'adoption_id':identity, 'best':receipt['best'],
                        'already_applied':True}
            if receipt.get('status') == 'prepared':
                result = _finish(root, receipt_path, receipt)
                research.snapshots = Snapshots(root/'snapshots')
                return result
            raise ValueError('unknown adoption transaction')
        for path in (root/'measurement_adoptions').glob('*.json'):
            if read_json(_safe(root, path.relative_to(root))).get('status') == 'prepared':
                raise ValueError('reconcile the previously prepared adoption first')
        match = next((row for row in offered(root) if all(row.get(k)==v for k,v in request.items())), None)
        if match is None:
            raise ValueError('no improved linked measurement matches the expected incumbent and bytes')
        from .native_jobs import active_jobs
        from .agent_tasks import records
        if active_jobs(research.output) or any(r.get('status') in
                {'submitted','running','outcome_unknown'} for r in records(research.output)):
            raise ValueError('unresolved native jobs prevent adoption')
        session_path = _safe(root, 'controller_session.json')
        session = read_json(session_path)
        if (session.get('status') not in {'paused','completed'} or session.get('action')
                or session.get('pending_action')):
            raise ValueError('adoption requires an idle research session')
        if (getattr(research, '_held', {}) or (root/'confirmation.json').exists()
                or any((root/'confirmation_attempts').glob('*.json'))
                or (root/'supervisor_verdict.json').exists()):
            raise ValueError('adoption cannot cross an active candidate or held-out boundary')
        measurement_path = _safe(root, 'measurements/'+label+'.json')
        row = read_json(measurement_path)
        from .receipt_verifier import verify_measurement
        proof = verify_measurement(root, label)
        if proof.get('status') != 'consistent':
            raise ValueError('measurement receipt revalidation failed: '+str(proof.get('issues'))[:800])
        from .evidence_store import read_attempt_evidence
        receipt_hashes = {}
        output_hashes = {'workspace_snapshot.json':digest(_safe(research.output,'workspace_snapshot.json'))}
        for stage in ('train','evaluate'):
            attempt = (row.get(stage) or {}).get('attempt_id', '')
            if not re.fullmatch(r'[a-f0-9]{32}', attempt):
                raise ValueError('missing native attempt identity')
            evidence = read_attempt_evidence(research.output, attempt, limit=1)
            receipt = _safe(root, 'attempts/'+attempt+'/receipt.json')
            native = read_json(receipt)
            if (evidence.get('receipt_ref') != str(receipt.relative_to(research.output))
                    or evidence.get('status') != 'completed' or evidence.get('returncode') != 0
                    or native.get('attempt_id') != attempt or native.get('status') != 'completed'
                    or native.get('returncode') != 0 or native.get('termination_reason') != 'normal_exit'):
                raise ValueError('native receipt/sealed evidence identity mismatch')
            receipt_hashes[str(receipt.relative_to(root))] = digest(receipt)
            evidence_path = _safe(research.output, 'evidence/'+attempt+'.json')
            log_ref = read_json(evidence_path)['log_ref']
            output_hashes['evidence/'+attempt+'.json'] = digest(evidence_path)
            output_hashes[log_ref] = digest(_safe(research.output, log_ref))
        state = row['source_state']
        workspace = read_json(_safe(research.output, 'workspace_snapshot.json'))
        from .workspace_snapshot import _state_identity, _validated_base
        _validated_base(workspace)
        expected = _state_identity(workspace, state['source_snapshot'], state['overlay_files'],
                                   state.get('overlay_modes') or {}, state.get('overlay_directories') or {})
        source = research.snapshots.get(state['source_snapshot'])
        if (state.get('identity_sha256') != expected
                or state.get('source_tree_fingerprint') != workspace.get('source_tree_fingerprint')
                or state.get('copy_mode') != workspace.get('copy_mode', 'full_worktree')
                or state.get('untracked_assets_omitted') != bool(workspace.get('untracked_assets_omitted'))
                or source is None or source.files != state['overlay_files']):
            raise ValueError('archived source identity mismatch')
        for name, key in source.files.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('unsafe archived source path')
            if key and (not re.fullmatch(r'[a-f0-9]{64}', key)
                        or digest(_safe(root, 'snapshots/blobs/'+key)) != key):
                raise ValueError('archived source blob changed')
        snapshots_path = _safe(root, 'snapshots/snapshots.json')
        document = read_json(snapshots_path)
        best = research.snapshots.best()
        actual_best = next((s for s in document.get('snapshots', []) if s.get('name')==expected_best),None)
        if (document.get('best') != expected_best or best is None or best.name != expected_best
                or actual_best != best.as_dict()):
            raise ValueError('incumbent changed during adoption')
        name = label.replace('_','-',1)
        if research.snapshots.get(name) is not None:
            raise ValueError('adoption snapshot name already exists')
        adopted = Snapshot(name=name, at=now(), files=dict(source.files),
                           score=row['metric_utility'], scale=best.scale,
                           why='explicit receipt-verified linked measurement adoption').as_dict()
        report_path = _safe(root, 'research_report.json')
        report = read_json(report_path)
        if report.get('rounds') != session.get('history'):
            raise ValueError('report/session history mismatch')
        projections = {
            'snapshots/snapshots.json':{'before_sha256':object_digest(document),
                'after':{**document,'at':now(),'best':name,
                         'snapshots':[*document['snapshots'],adopted]}},
            'research_report.json':{'before_sha256':object_digest(report),
                'after':{**report,'best':adopted,'best_export':{},'measurement_adoption_id':identity}}}
        preserved = {relative:digest(_safe(root, relative)) for relative in (
            'measurements/baseline.json','measurements/'+label+'.json',
            'comparison_protocol.json','controller_session.json')}
        receipt = {'id':identity,'status':'prepared','at':now(),**request,'best':adopted,
                   'proof':proof,'source_identity_sha256':expected,
                   'preserved':{**preserved,**receipt_hashes},'preserved_output':output_hashes,
                   'projections':projections}
        atomic_json(receipt_path, receipt)
        result = _finish(root, receipt_path, receipt)
        research.snapshots = Snapshots(root/'snapshots')
        research._append_local_event({'at':now(),'event':'measurement_adoption',
                                     'label':label,'adoption_id':identity,'metric_value':row['metric_value']})
        return result
