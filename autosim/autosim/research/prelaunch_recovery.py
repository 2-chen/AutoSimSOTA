"""Narrow recovery of sealed pre-native failures left in running checkpoints.

Never replay a possibly launched job or undo a code patch. Normal new rejections
return through measurement finalization; this explicit reconciliation repairs old
parameter/citation checkpoints using a verified original action receipt. Applied
patches remain interrupted until Main explicitly resolves the source transaction.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .action_receipt import read_action_receipt
from .common import atomic_json, now, object_digest, read_json


def _stopped_children(output, run_id, receipt):
    """Authenticate every owned helper; disappearance alone proves no completion."""
    import re
    from .common import digest
    from .experiment_transaction import safe
    from .process_executor import inspect_process_identity
    if receipt.get('process_identity'):
        raise ValueError('parent owns a process; normal process reconciliation required')
    children = []
    for identity in receipt.get('child_attempt_ids') or []:
        if not isinstance(identity, str) or not re.fullmatch(r'[0-9a-f]{32}', identity):
            raise ValueError('invalid owned child identity')
        ref = 'processes/' + identity + '.json'
        path = safe(output, ref)
        child = read_json(path)
        process = child.get('process_identity')
        if (child.get('run_id') != run_id or child.get('attempt_id') != identity
                or (child.get('action') or {}).get('decision_id') != receipt['decision_id']
                or not isinstance(process, dict)
                or inspect_process_identity(process).get('status') != 'not_running'):
            raise ValueError('owned helper is live, unknown or not receipt-bound')
        children.append({'ref': ref, 'sha256': digest(path),
                         'process_status': 'not_running', 'native_completion': 'not_inferred'})
    return children


def failed_preparation(output: Path, repo: Path, run_id: str, steps: list[dict]) -> dict:
    """A sealed failed parent with a write-ahead patch, but no native launch in it.

    Later independent jobs do not belong to this failed action. They must be terminal;
    their data, weights and scores are never discarded or automatically replayed.
    This only exposes reconciliation; source rollback remains Main's separate choice.
    The exception class is not an execution boundary: source lookup, patch commit,
    snapshot and independent-review failures use the same authenticated ownership
    checks. No unknown native attempt is made retryable by its error message.
    """
    from .experiment_transaction import safe
    from .native_jobs import active_jobs
    try:
        root = safe(output, 'research/' + run_id)
        path = safe(output, 'research/' + run_id + '/controller_session.json')
        session = read_json(path)
        pending = session.get('pending_action') or {}
        index = pending.get('round')
        idea = pending.get('idea') or {}
        if (session.get('status') != 'running' or session.get('run_id') != run_id
                or Path(session.get('repository', '')).resolve() != repo.resolve()
                or type(index) is not int or index < 1 or session.get('next_round') != index
                or session.get('action') != f'round_{index}'
                or pending.get('phase') != 'preparing_idea'
                or pending.get('source_transaction_protocol') != 1
                or idea.get('granularity') != 'code' or active_jobs(output)):
            return {}
        measurement = safe(output, f'research/{run_id}/measurements/round_{index}.json')
        if measurement.exists():
            return {}
        transaction_path = safe(output, f'research/{run_id}/code_changes/round-{index}.json')
        transaction = read_json(transaction_path)
        if (transaction.get('round') != index or transaction.get('idea_label') != idea.get('label')
                or transaction.get('status') not in {'applying', 'applied'}):
            return {}
        candidate = next((s for s in reversed(steps) if s.get('step') == 'run_the_loop'
                          and (s.get('arguments') or {}).get('idea_label') == idea.get('label')), None)
        if not candidate or candidate.get('outcome') != 'raised':
            return {}
        receipt = read_action_receipt(output, candidate['receipt_ref'], run_id=run_id, repository=repo)
        if (receipt.get('receipt_sha256') != candidate.get('receipt_sha256')
                or receipt.get('status') != 'raised' or receipt.get('action') != 'run_the_loop'
                or (receipt.get('arguments') or {}).get('idea_label') != idea.get('label')):
            return {}
        begin, end = map(datetime.fromisoformat, (receipt['started_at'], receipt['finished_at']))
        cutoff = datetime.fromisoformat(pending['updated_at'])
        if not begin <= cutoff <= end:
            return {}
        if (root / 'attempts').is_symlink():
            return {}
        for raw in (root / 'attempts').glob('*/receipt.json'):
            native = read_json(safe(output, raw.relative_to(output).as_posix()))
            if begin <= datetime.fromisoformat(native['started_at']) <= end:
                return {}
        # CPU/source helpers may have durable process records even though no native
        # stage was launched. Verify every owned child has stopped; stale "running"
        # presentation alone is not proof of liveness or completion. Never kill it here.
        children = _stopped_children(output, run_id, receipt)
        return {'status': 'failed_candidate_checkpoint_reconcilable',
            'round': index, 'idea_label': idea['label'], 'phase': pending['phase'],
            'receipt_ref': candidate['receipt_ref'], 'receipt_sha256': receipt['receipt_sha256'],
            'session_sha256': object_digest(session), 'transaction_sha256': object_digest(transaction),
            'stopped_children': children,
            'reason': receipt['reason'], 'next_action': 'reconcile_interrupted_action',
            'guidance': 'mark the failed candidate interrupted; preserve patch and later independent work. '
                        'Main may separately choose audited discard/rollback; never infer a score or retry.'}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def inspect(output: Path, repo: Path, run_id: str, steps: list[dict]) -> dict:
    failed = failed_preparation(output, repo, run_id, steps)
    if failed:
        return failed
    root = output / 'research' / run_id
    try:
        path = root / 'controller_session.json'
        if any(p.is_symlink() for p in (path, root, root.parent, root / 'attempts', root / 'measurements')):
            return {}
        session = read_json(path)
        pending = session.get('pending_action') or {}
        index = pending.get('round')
        if (session.get('status') != 'running' or session.get('run_id') != run_id
                or Path(session.get('repository', '')).resolve() != repo.resolve()
                or not isinstance(index, int) or isinstance(index, bool) or index < 1
                or session.get('next_round') != index or session.get('action') != f'round_{index}'
                or pending.get('phase') not in {'measurement_starting', 'preparing_idea'}
                or pending.get('changed') or pending.get('collected')
                or (pending.get('phase') == 'measurement_starting'
                    and not isinstance(pending.get('settings'), dict))
                or not isinstance(session.get('history'), list)
                or len(session['history']) != index
                or (session['history'][0] or {}).get('label') != 'baseline'
                or (root / 'measurements' / f'round_{index}.json').exists()
                or (root / 'measurements' / f'round_{index}.json').is_symlink()):
            return {}
        preparing = pending.get('phase') == 'preparing_idea'
        if preparing:
            # Version 1 commits this manifest BEFORE the first candidate source write.
            # Do not infer a safe pre-patch boundary for older/unknown protocols.
            transaction = root / 'code_changes' / f'round-{index}.json'
            if (pending.get('source_transaction_protocol') != 1
                    or (root / 'code_changes').is_symlink()
                    or transaction.exists() or transaction.is_symlink()):
                return {}
        from .native_jobs import active_jobs
        if active_jobs(output):
            return {}
        cutoff = datetime.fromisoformat(pending['updated_at'])
        for receipt_path in (root / 'attempts').glob('*/receipt.json'):
            if receipt_path.is_symlink():
                return {}
            attempt = read_json(receipt_path)
            # Any attempt since the pending checkpoint requires the normal process/
            # receipt reconciliation, even if it claims no launch or a failed exit.
            if datetime.fromisoformat(attempt['started_at']) >= cutoff:
                return {}
        idea = (pending.get('idea') or {}).get('label')
        candidate = next((s for s in reversed(steps) if s.get('step') == 'run_the_loop'
                          and (s.get('arguments') or {}).get('idea_label') == idea), None)
        if not candidate or candidate.get('outcome') != 'raised':
            return {}
        receipt = read_action_receipt(output, candidate['receipt_ref'],
                                      run_id=run_id, repository=repo)
        reason = receipt.get('reason') or ''
        if (receipt.get('receipt_sha256') != candidate.get('receipt_sha256')
                or receipt.get('status') != 'raised' or receipt.get('action') != 'run_the_loop'
                or (receipt.get('arguments') or {}).get('idea_label') != idea
                or (not preparing and (not reason.startswith('ValueError: stage_argv_')
                    or ' produced an ambiguous command:' not in reason))
                or datetime.fromisoformat(receipt['started_at']) > cutoff
                or datetime.fromisoformat(receipt['finished_at']) < cutoff):
            return {}
        children = _stopped_children(output, run_id, receipt)
        return {'status': 'prelaunch_rejection_reconcilable', 'idea_label': idea,
                'round': index, 'receipt_ref': candidate['receipt_ref'],
                'receipt_sha256': receipt['receipt_sha256'],
                'session_sha256': object_digest(session), 'reason': reason,
                'stopped_children': children,
                'phase': pending['phase'],
                'next_action': 'reconcile_interrupted_action'}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def reconcile(research, steps: list[dict]) -> dict:
    evidence = inspect(research.output, research.repo, research.run_id, steps)
    if not evidence:
        return {'outcome': 'blocked', 'because': 'no verified, unlaunched parameter-only refusal to reconcile'}
    path = research.run_root / 'controller_session.json'
    session = read_json(path)
    pending = session['pending_action']
    if evidence['status'] == 'failed_candidate_checkpoint_reconcilable':
        # No rollback and no attempt/adoption here. The existing explicit discard
        # transaction verifies source pre/postimages before it may change files.
        index = evidence['round']
        boundary = {'schema_version': 1, 'status': 'unresolved',
            'controller_action': f'round_{index}', 'round': index,
            'pending_action_identity': 'matched', 'phase': pending['phase'],
            'idea': pending['idea'], 'candidate_measurement': {'status': 'missing'},
            'attempt_evidence': {'attempt_id': None, 'receipt_ref': None,
                'receipt_status': 'not_recorded', 'process_status': 'not_recorded'},
            'failed_parent': evidence, 'automatic_replay': False}
        if object_digest(read_json(path)) != evidence['session_sha256']:
            raise ValueError('session changed during candidate reconciliation')
        interruption = {'controller_action': f'round_{index}', 'pending_action': pending,
                        'boundary': boundary, 'failed_parent': evidence, 'outcome_known': False}
        session.update(status='interrupted', interruption=interruption,
            reconciliation={'status': 'reconciled', 'failed_parent': evidence, 'native_replayed': False},
            updated_at=now())
        atomic_json(path, session)
        return {'outcome': 'reconciled', 'because': 'failed source-preparation checkpoint is now interrupted; '
                'Main must explicitly resolve the candidate, no native replay or source change',
                'resume_scope': 'preparation',
                'prelaunch_reconciliation': evidence, 'native_replayed': False,
                'evidence_refs': [evidence['receipt_ref']]}
    if session.get('input_identity'):
        if research._research_session_identity(rounds=session['rounds'], settings=session['base_settings']) != session['input_identity']:
            raise ValueError('research inputs changed; prelaunch reconciliation cannot mix protocols')
    preparing = evidence['phase'] == 'preparing_idea'
    stage = (None if preparing else
             evidence['reason'].split('stage_argv_', 1)[1].split(' ', 1)[0])
    # A stale citation needs new evidence, not an automatic patch/retry. The argv
    # branch still rechecks its exact construction; neither branch launches work.
    if not preparing:
        inputs = research._inputs(stage, settings=pending['settings'])
        research.backend.argv(stage, inputs)
    from .operation_failure import seal
    sealed = seal(research.output, operation='candidate_source_preflight' if preparing else f'stage_argv_{stage}',
                  exc=ValueError(evidence['reason']), context=evidence,
                  native_operation_status='not_started')
    if object_digest(read_json(path)) != evidence['session_sha256']:
        raise ValueError('session changed during prelaunch reconciliation')
    reconciliation = {**evidence, **sealed, 'status': 'prelaunch_rejection_reconciled',
                      'pending_action': pending, 'at': now(),
                      'native_revalidation': (('not_run; citation refresh required'
                        if evidence['reason'].startswith('SourceCitationError:') else
                        'not_run; preparation evidence/repair required') if preparing else
                                              'not_run; corrected argv construction only')}
    session.update(status='paused', action='', pending_action=None,
                   prelaunch_reconciliation=reconciliation, updated_at=now())
    atomic_json(path, session)
    # Baseline, protocol, history, next_round and all budgets remain byte-for-byte
    # logically unchanged. Main must select an audited candidate for real replay.
    return {'outcome': 'reconciled', 'because': 'verified prelaunch refusal reconciled; no native job replayed',
            'resume_scope': 'preparation',
            'native_replayed': False,
            **sealed, 'prelaunch_reconciliation': reconciliation}
