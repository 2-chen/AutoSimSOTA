"""Small factual handoff projection. Advice is not a second workflow controller."""
import re
from pathlib import Path

from .common import read_json


def _read(root, path):
    if (path.is_symlink() or not path.resolve().is_relative_to(root)
            or path.stat().st_size>2*1024**2):
        raise ValueError('unsafe milestone source')
    row=read_json(path)
    if not isinstance(row,dict):
        raise ValueError('milestone source must be an object')
    return row


def view(output, run_id, facts):
    root=Path(output).resolve()
    path=root/'research'/run_id/'measurements/baseline.json'
    basis={'status':'not_measured','official_reproduction':False}
    try:
        measured=_read(root,path)
        reference = facts.get('baseline_reference') or {}
        reproduced = reference.get('status') == 'reproduced'
        basis={'status':'local_measurement_recorded' if measured.get('ok') else 'measurement_failed',
            'metric_value':measured.get('metric_value'),
            'samples':(measured.get('metric_reading') or {}).get('samples'),
            'measurement_ref':path.relative_to(root).as_posix(),
            'official_reproduction':reproduced and reference.get('measurement_ref') == path.relative_to(root).as_posix(),
            'released_reference_reproduced':reproduced,
            'data_consumption':(measured.get('data_consumption') or {}).get('status','not_recorded'),
            'instruction':'A real local score is not proof of official fidelity. Compare policy family, '
                'released revision/recipe, actual updates and batches, loaded data population, '
                'normalization/action mapping and evaluator/reset settings using source evidence. '
                'Diagnose a weak baseline before claiming SOTA; never replace its historical score.'}
        attempt=(measured.get('train') or {}).get('attempt_id') or ''
        if re.fullmatch(r'[0-9a-f]{32}',attempt):
            receipt_path=path.parents[1]/'attempts'/attempt/'receipt.json'
            receipt=_read(root,receipt_path)
            basis.update(training_receipt_ref=receipt_path.relative_to(root).as_posix(),
                actual_training_argv=receipt.get('argv',[])[:96])
    except (OSError,ValueError,TypeError,KeyError):
        pass
    catalog=facts.get('verified_train_artifacts') or {}
    events=[]
    for producer in catalog.get('producers') or []:
        if producer.get('origin')=='native_job':
            events.append({'kind':'completed_training_available','job_id':producer.get('job_id'),
                'next_question':'Was this producer formally adopted/scored? Inspect receipts first; do not retrain.',
                'evidence_ref':producer.get('native_evidence_ref')})
    for producer in catalog.get('pending_producers') or []:
        events.append({'kind':'producer_audit_pending',**producer})
    last=(facts.get('recent_actions') or [])[-1:] or []
    transient=[{k:row.get(k) for k in ('step','failure_category','retryable','retry_when','repair_owner')}
               for row in last if row.get('retryable') is True]
    return {'schema_version':1,'authority':'advisory receipt projection; Main chooses the route',
        'baseline_basis':basis,'producer_handoffs':events[:24],
        'retryable_blocks':transient,
        'data_next_question':'Ask Data at meaningful milestones: failed development rollout → '
            'targeted acquisition plan → native collection → quality/profile → immutable version → '
            'actual training batch consumption → candidate score. Collection/conversion success '
            'alone is not a gain. Reuse an applicable handoff; do not summon Data every heartbeat.'}
