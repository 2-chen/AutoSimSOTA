"""Narrow recovery of legacy argv refusals left in an unknown-running checkpoint.

Never replay a possibly launched job or undo a code patch. Normal new rejections
return through measurement finalization; this explicit reconciliation repairs old
parameter-only checkpoints using a verified original action receipt.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .action_receipt import read_action_receipt
from .common import atomic_json, now, object_digest, read_json


def inspect(output: Path, repo: Path, run_id: str, steps: list[dict]) -> dict:
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
                or pending.get('phase') != 'measurement_starting' or pending.get('changed')
                or pending.get('collected') or not isinstance(pending.get('settings'), dict)
                or not isinstance(session.get('history'), list)
                or len(session['history']) != index
                or (session['history'][0] or {}).get('label') != 'baseline'
                or (root / 'measurements' / f'round_{index}.json').exists()
                or (root / 'measurements' / f'round_{index}.json').is_symlink()):
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
                or not reason.startswith('ValueError: stage_argv_')
                or ' produced an ambiguous command:' not in reason
                or datetime.fromisoformat(receipt['started_at']) > cutoff
                or datetime.fromisoformat(receipt['finished_at']) < cutoff):
            return {}
        return {'status': 'prelaunch_rejection_reconcilable', 'idea_label': idea,
                'round': index, 'receipt_ref': candidate['receipt_ref'],
                'receipt_sha256': receipt['receipt_sha256'],
                'session_sha256': object_digest(session), 'reason': reason,
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
    if session.get('input_identity'):
        if research._research_session_identity(rounds=session['rounds'], settings=session['base_settings']) != session['input_identity']:
            raise ValueError('research inputs changed; prelaunch reconciliation cannot mix protocols')
    stage = evidence['reason'].split('stage_argv_', 1)[1].split(' ', 1)[0]
    # Revalidate the exact failed construction, not the train/eval process. A still
    # broken builder leaves the checkpoint untouched and emits a citable failure.
    inputs = research._inputs(stage, settings=pending['settings'])
    research.backend.argv(stage, inputs)
    from .operation_failure import seal
    sealed = seal(research.output, operation=f'stage_argv_{stage}',
                  exc=ValueError(evidence['reason']), context=evidence,
                  native_operation_status='not_started')
    if object_digest(read_json(path)) != evidence['session_sha256']:
        raise ValueError('session changed during prelaunch reconciliation')
    reconciliation = {**evidence, **sealed, 'status': 'prelaunch_rejection_reconciled',
                      'pending_action': pending, 'at': now(),
                      'native_revalidation': 'not_run; corrected argv construction only'}
    session.update(status='paused', action='', pending_action=None,
                   prelaunch_reconciliation=reconciliation, updated_at=now())
    atomic_json(path, session)
    # Baseline, protocol, history, next_round and all budgets remain byte-for-byte
    # logically unchanged. Main must select an audited candidate for real replay.
    return {'outcome': 'reconciled', 'because': 'verified prelaunch refusal reconciled; no native job replayed',
            **sealed, 'prelaunch_reconciliation': reconciliation}
