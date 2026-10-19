"""Explicit CPU-only recovery of a completed candidate's missing result locator.

The original failure stays intact. Main chooses the locator; an independent
reviewer and the existing receipt verifier must approve the same native result.
This module does not run native work or promote a baseline/incumbent.
"""
from pathlib import Path
import re

from .common import read_json
from .experiment_transaction import safe


def offered(root):
    root=Path(root)
    if (root/'confirmation.json').exists() or any((root/'confirmation_attempts').glob('*.json')):
        return []
    rows=[]
    for index,path in enumerate(sorted((root/'measurements').glob('round_*.json'))):
        if index>=512:
            break
        try:
            if path.is_symlink() or path.stat().st_size>2*1024**2:
                continue
            old=read_json(path);label=old.get('label')
            if (not isinstance(label,str) or not re.fullmatch(r'round_[1-9][0-9]*',label)
                    or path.name!=label+'.json' or old.get('ok') is not False
                    or old.get('where')!='evaluate' or old.get('baseline_origin')!='locally_trained'
                    or old.get('confirmation') or old.get('metric_value') is not None
                    or old.get('result_artifact')
                    or (old.get('metric_artifact_evidence') or {}).get('status')!='missing'):
                continue
            attempt=(old.get('evaluate') or {}).get('attempt_id')
            if not isinstance(attempt,str) or not re.fullmatch(r'[0-9a-f]{32}',attempt):
                continue
            receipt=read_json(safe(root,f'attempts/{attempt}/receipt.json'))
            if (receipt.get('attempt_id')!=attempt or receipt.get('node_id')!='evaluate'
                    or receipt.get('status')!='completed' or receipt.get('returncode')!=0
                    or receipt.get('termination_reason')!='normal_exit'):
                continue
            rows.append({'label':label,'kind':'completed_evaluation_missing_locator',
                'evaluation_attempt_id':attempt,'native_replayed':False,
                'required_argument':'artifact_revision',
                'why':'Native evaluation completed; select an independently source-reviewed '
                    'attempt-owned result locator for CPU recovery. Do not repeat evaluation.'})
        except (OSError,ValueError,TypeError,KeyError):
            continue
    return rows


def record(research, *, label, artifact_revision):
    from .baseline_score_recovery import _idle
    from .reference_score_recovery import recover
    from .receipt_verifier import verify_measurement
    from .handoff_correction import candidate_sources
    _idle(research)
    session=read_json(safe(research.run_root,'controller_session.json'))
    if (session.get('status') not in {'paused','completed','finalizing'} or session.get('action')
            or session.get('pending_action') or getattr(research,'_held',{})):
        raise ValueError('CPU score recovery needs an idle development session')
    if not any(row['label']==label for row in offered(research.run_root)):
        raise ValueError('select an offered completed development evaluation with a missing locator')
    original=f'research/{research.run_id}/measurements/{label}.json'
    for index,path in enumerate(sorted((research.run_root/'measurements').glob('score_recovered_*.json'))):
        if index>=512:
            raise ValueError('bounded score recovery catalog exceeded')
        if path.is_symlink() or path.stat().st_size>2*1024**2:
            raise ValueError('unsafe existing score recovery')
        saved=read_json(path)
        if (saved.get('scoring_recovery') or {}).get('original_measurement_ref')==original:
            if verify_measurement(research.run_root,saved.get('label','')).get('status')!='consistent':
                raise ValueError('existing CPU recovery no longer verifies; preserve it for Fix')
            return saved
    # Restore only the original audited candidate while source/locator checks run,
    # then return to the exact prior checkout. This is not a new intervention.
    with candidate_sources(research,label):
        result,_=recover(research,original_ref=original,
            locator_revision=artifact_revision,client=research.client)
    return result
