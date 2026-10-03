"""Explicit allocation of additional development rounds, never a new baseline."""
from pathlib import Path

from .common import atomic_json, digest, exclusive, now, object_digest, read_json
from .research_state import ResearchStateError


def resume_rounds(root: Path, session: dict, requested: int) -> int:
    """Accept an earlier CLI allocation only through applied extension receipts."""
    total=session.get('rounds')
    current=session
    for _ in range(64):
        if current.get('rounds')==requested:
            return total
        identity=current.get('round_extension_id','')
        if (not isinstance(identity,str) or len(identity)!=64
                or any(c not in '0123456789abcdef' for c in identity)):
            break
        path=root/'round_extensions'/(identity+'.json')
        if (path.is_symlink() or path.parent.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size>2*1024**2):
            break
        receipt=read_json(path)
        if not isinstance(receipt,dict):
            break
        request={k:receipt.get(k) for k in ('additional_rounds','expected_rounds','reason')}
        added,expected=request['additional_rounds'],request['expected_rounds']
        before=receipt.get('before_session') or {}
        preserved=receipt.get('preserved') or {}
        if not isinstance(before,dict) or not isinstance(preserved,dict):
            break
        if any(p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in
                (root/'measurements/baseline.json',root/'comparison_protocol.json')):
            break
        if (receipt.get('status')!='applied' or receipt.get('id')!=identity
                or object_digest(request)!=identity
                or type(added) is not int or not 1<=added<=1000
                or type(expected) is not int or expected<0
                or expected+added!=current.get('rounds')
                or receipt.get('new_rounds')!=current.get('rounds')
                or before.get('rounds')!=expected
                or before.get('run_id')!=session.get('run_id')
                or before.get('repository')!=session.get('repository')
                or before.get('base_settings')!=session.get('base_settings')
                or preserved.get('base_settings')!=object_digest(session.get('base_settings'))
                or preserved.get('baseline_sha256')!=digest(root/'measurements/baseline.json')
                or preserved.get('comparison_protocol_sha256')!=digest(root/'comparison_protocol.json')):
            break
        current=before
    raise ValueError('paused research session has different frozen rounds/settings; '
                     'no intact applied extension chain connects the requested rounds')


def extend(research, *, additional_rounds: int, expected_rounds: int, reason: str) -> dict:
    if (isinstance(additional_rounds, bool) or not isinstance(additional_rounds, int)
            or not 1 <= additional_rounds <= 1000 or isinstance(expected_rounds, bool)
            or not isinstance(expected_rounds, int) or expected_rounds < 0):
        raise ValueError('additional_rounds must be integer 1..1000; expected_rounds must be nonnegative integer')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('reason must be a nonempty string')
    if len(reason) > 1000:
        raise ValueError(f'reason has {len(reason)} characters; shorten to at most 1000 characters')
    root = research.run_root
    request = {'additional_rounds': additional_rounds, 'expected_rounds': expected_rounds,
               'reason': reason}
    identity = object_digest(request)
    receipt_path = root/'round_extensions'/f'{identity}.json'
    if receipt_path.parent.is_symlink() or receipt_path.is_symlink():
        raise ResearchStateError('unsafe round-extension ledger')
    from .native_jobs import active_jobs
    from .agent_tasks import records
    with exclusive(root/'round_extensions.lock'):
        if receipt_path.is_file():
            previous = read_json(receipt_path)
            if previous.get('status') == 'applied':
                return previous
            raise ResearchStateError('unfinished extension requires reconciliation; do not allocate twice')
        if research.budget is None or research.budget.remaining() <= 0:
            raise ResearchStateError('no remaining hard run budget')
        if active_jobs(research.output) or any(r.get('status') in
                {'submitted', 'running', 'outcome_unknown'} for r in records(research.output)):
            raise ResearchStateError('active or unresolved jobs own the research workspace')
        if any((root/'confirmation_attempts').glob('*.json')):
            raise ResearchStateError('development cannot reopen after held-out exposure')
        session_path, report_path = root/'controller_session.json', root/'research_report.json'
        if session_path.is_symlink() or report_path.is_symlink():
            raise ResearchStateError('linked research session/report')
        session, report = read_json(session_path), read_json(report_path)
        if (session.get('schema_version') != 1 or session.get('run_id') != research.run_id
                or Path(session.get('repository', '')).resolve() != research.repo
                or session.get('status') not in {'paused', 'completed'}
                or session.get('pending_action') or session.get('action')
                or report.get('run_id') != research.run_id
                or report.get('run_status') != session.get('status')
                or session.get('rounds') != expected_rounds):
            raise ResearchStateError('extension requires the exact idle, matching research session')
        settings = session.get('base_settings')
        if not isinstance(settings, dict):
            raise ResearchStateError('missing immutable baseline settings')
        hashes = research._research_session_input_hashes(rounds=expected_rounds, settings=settings)
        if (session.get('input_identity') != research._research_session_identity(
                rounds=expected_rounds, settings=settings)
                or session.get('input_component_sha256') != hashes):
            raise ResearchStateError('research inputs changed; an extension cannot repair the protocol')
        baseline = root/'measurements/baseline.json'
        baseline_row = read_json(baseline)
        if baseline_row.get('ok') is not True:
            raise ResearchStateError('extension requires a valid original baseline')
        from .receipt_verifier import verify_measurement
        proof = verify_measurement(root, 'baseline')
        if proof.get('status') != 'consistent':
            raise ResearchStateError('original baseline receipt is not consistent')
        history = session.get('history')
        next_round = session.get('next_round')
        if (not isinstance(history, list) or not history or report.get('rounds') != history
                or not isinstance(next_round, int)
                or isinstance(next_round, bool) or not 1 <= next_round <= expected_rounds+1):
            raise ResearchStateError('invalid continuation history or next-round cursor')
        preserved = {'baseline_sha256': digest(baseline), 'history': object_digest(history),
                     'base_settings': object_digest(settings),
                     'best': object_digest(research.snapshots.best().as_dict()
                         if research.snapshots.best() else None),
                     'comparison_protocol_sha256': digest(root/'comparison_protocol.json')}
        total = expected_rounds + additional_rounds
        updated = {**session, 'rounds': total, 'status': 'paused', 'action': '',
                   'stopped_because': '', 'updated_at': now(), 'round_extension_id': identity,
                   'input_identity': research._research_session_identity(rounds=total, settings=settings),
                   'input_component_sha256': research._research_session_input_hashes(rounds=total, settings=settings)}
        next_report = {**report, 'run_status': 'paused', 'planned_rounds': total,
                       'next_round': next_round, 'stopped_because': '', 'round_extension_id': identity}
        receipt = {'id': identity, 'status': 'prepared', 'at': now(), **request,
                   'receipt_ref': str(receipt_path.relative_to(research.output)),
                   'new_rounds': total, 'preserved': preserved,
                   'before_session': session, 'before_report': report,
                   'after_session_sha256': object_digest(updated),
                   'after_report_sha256': object_digest(next_report)}
        atomic_json(receipt_path, receipt)
        # An interruption between writes fails closed on session/report mismatch;
        # prepared evidence preserves both originals for explicit reconciliation.
        atomic_json(report_path, next_report)
        atomic_json(session_path, updated)
        receipt['status'] = 'applied'
        atomic_json(receipt_path, receipt)
        return {**receipt, 'receipt_ref': str(receipt_path.relative_to(research.output))}
