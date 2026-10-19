"""Bound the observer to this action's sealed failure, not the newest model."""
import json
import re


def current_failure(output, action):
    from .evidence_store import read_attempt_evidence
    ident = action.get('evidence_id')
    if not isinstance(ident, str) or not re.fullmatch(r'[0-9a-f]{32}', ident):
        return {'status': 'unavailable', 'why': 'current action has no sealed failure ID; do not infer from recent producers'}
    try:
        proof = read_attempt_evidence(output, ident, limit=12000)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {'status':'unavailable', 'evidence_id':ident,
                'why':'current failure evidence could not be verified: '+str(exc)[:300]}
    subject = {}
    # Only this sealed operation envelope can name an associated producer.
    # A successful producer is not relabeled as the failed operation.
    if proof.get('status') == 'operation_contract_failed':
        try:
            row = json.loads(proof['text'])
            context = row.get('context') or {}
            if isinstance(context, dict):
                for key in ('training_attempt_id','evaluation_attempt_id','measurement_label'):
                    value = context.get(key)
                    if isinstance(value, str) and len(value) <= 128:
                        subject[key] = value
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
    elif proof.get('status') == 'failed':
        subject['failed_native_attempt_id'] = ident
    return {'status':'verified_sealed_failure', 'evidence_id':ident,
        'failure_status':proof['status'], 'returncode':proof.get('returncode'),
        'receipt_ref':proof.get('receipt_ref'), 'log_ref':proof.get('log_ref'),
        'excerpt':proof['text'][:8000], 'subject':subject,
        'authority':'this action failure and explicitly associated subjects only; '
            'recent/latest producers and previous observer claims are not the failed subject'}
