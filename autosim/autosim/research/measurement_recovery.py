"""Receipt-bound development recovery; an Agent chooses the repair, not a filename rule.

Completed training is retained after a failed consumer. The offered packet is not a
score or permission to retry held-out evaluation. Execution must revalidate it and
reuse the same producer under the original development protocol.
"""
import re
from pathlib import Path

from .common import digest, object_digest, read_json
from .experiment_transaction import safe
from .evidence_store import read_attempt_evidence


def offered(output: Path, run_id: str, *, _producers: list[dict] | None = None) -> list[dict]:
    from .checkpoint_handoff import formal_producers
    output = Path(output).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('unsafe recovery run identity')
    if not output.is_dir():
        return []
    root = safe(output, 'research/'+run_id)
    if (root/'confirmation.json').exists() or any((root/'confirmation_attempts').glob('*.json')):
        return []
    # One state projection may share its freshly audited inventory. Explicit
    # select/recovery callers omit this private argument and revalidate afresh.
    inventory = formal_producers(output) if _producers is None else _producers
    producers = {row['training_attempt_id']:row for row in inventory
                 if row.get('origin') == 'formal_measurement' and row.get('run_id') == run_id}
    rows = []
    for path in sorted((root/'measurements').glob('*.json'))[-100:]:
        try:
            path = safe(output, path.relative_to(output).as_posix())
            measurement = read_json(path)
            label = str(measurement.get('label') or '')
            if (not re.fullmatch(r'baseline|round_[1-9][0-9]*', label) or path.stem != label or
                    measurement.get('ok') is not False or measurement.get('confirmation') or
                    measurement.get('zero_work_reported') or
                    measurement.get('where') not in {'evaluate','policy_artifact'}):
                continue
            train = measurement.get('train') or measurement.get('training_outcome') or {}
            attempt = str(train.get('attempt_id') or '')
            producer = producers.get(attempt)
            if not producer:
                continue
            train_path = safe(output, f'research/{run_id}/attempts/{attempt}/receipt.json')
            receipt = read_json(train_path)
            # A known detached producer may use train-only settings. Actual replay
            # checks the native argv too; the catalog does not infer that equivalence.
            evaluation_id = str((measurement.get('evaluate') or {}).get('attempt_id') or '')
            evidence_id = attempt
            evaluation_ref = None
            boundary = measurement.get('policy_handoff_failure') or {}
            if measurement.get('where') == 'policy_artifact' and boundary:
                evidence_id = str(boundary.get('evidence_id') or '')
                envelope = read_attempt_evidence(output, evidence_id, limit=1)
                detail = read_json(safe(output,envelope['receipt_ref']))
                if (envelope.get('status') != 'operation_contract_failed' or
                        detail.get('operation') != 'policy_checkpoint_boundary' or
                        (detail.get('context') or {}).get('training_attempt_id') != attempt):
                    continue
            if measurement.get('where') == 'evaluate':
                if not re.fullmatch(r'[0-9a-f]{32}', evaluation_id):
                    continue
                evaluation_ref = f'research/{run_id}/attempts/{evaluation_id}/receipt.json'
                native_path = safe(output, evaluation_ref)
                native = read_json(native_path)
                sealed = read_attempt_evidence(output, evaluation_id, limit=1)
                if (native.get('attempt_id') != evaluation_id or native.get('run_id') != run_id or
                        native.get('stage') != 'evaluate' or native.get('status') != 'failed' or
                        type(native.get('returncode')) is not int or native['returncode'] == 0 or
                        native.get('termination_reason') != 'nonzero_exit' or native.get('ran') is not True or
                        native.get('settings_digest') != object_digest(measurement.get('settings') or {}) or
                        sealed.get('status') != 'failed' or sealed.get('returncode') != native['returncode'] or
                        sealed.get('receipt_ref') != evaluation_ref):
                    continue
                evidence_id = evaluation_id
            recovery_state = (root/'attempts'/attempt/f'baseline_handoff_recovery_{evidence_id}.json'
                              if label == 'baseline' else root/'attempts'/attempt/'handoff_correction.json')
            if recovery_state.exists():
                # A still-linked attempt has already been submitted. A new native
                # failure gets a new ID; an unknown original outcome is not replayed.
                continue
            rows.append({'label':label, 'failure_category':'development_consumer_failed',
                'repair_owner':'fix', 'evidence_id':evidence_id,
                'measurement_ref':path.relative_to(output).as_posix(),
                'measurement_sha256':digest(path),
                'training_attempt_id':attempt, 'training_receipt_sha256':digest(train_path),
                'training_evidence_id':attempt, 'training_output_ref':producer['output_ref'],
                'completed_work':producer['progress'], 'evaluation_attempt_id':evaluation_id or None,
                'evaluation_receipt_ref':evaluation_ref,
                'evaluation_receipt_sha256':digest(safe(output,evaluation_ref)) if evaluation_ref else None,
                'settings':measurement.get('settings') or {},
                'required_checkpoint_producer':producer['native_evidence_ref'],
                'next_action':'recover_unscored_baseline' if label == 'baseline' else 'correct_policy_handoff',
                'checkpoint_ref_required':True,
                'instruction':'Fix reads this sealed failure and native save/load source, then proposes '
                    'an exact checkpoint_ref inside THIS completed formal producer. Main executes the '
                    'offered recovery under the same protocol. Do not substitute smoke-test weights, '
                    'retrain, repeat completed diagnostic jobs, or change evaluation semantics.'})
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return rows


def select_for_recovery(output: Path, run_id: str, label: str, checkpoint_ref: str) -> tuple[dict, dict]:
    from .checkpoint_handoff import catalog, select
    packet = next((row for row in offered(output, run_id) if row['label'] == label), None)
    if packet is None:
        raise ValueError('no sealed failed development consumer with the same completed producer is offered')
    choice = select(output, checkpoint_ref, catalog(output))
    if choice['producer_evidence_ref'] != packet['required_checkpoint_producer']:
        raise ValueError('recovery must use the exact formal training producer, not a probe or different candidate')
    receipt = read_json(safe(output, f'research/{run_id}/attempts/{packet["training_attempt_id"]}/receipt.json'))
    artifact = receipt.get('artifact') or {}
    if artifact.get('matched') == 1:
        candidates = artifact.get('candidate_paths') or artifact.get('examples') or []
        if len(candidates) != 1:
            raise ValueError('unique producer payload is not explicitly recorded')
        payload = Path(candidates[0])
        if not payload.is_absolute():
            payload = Path(receipt['working_directory'] if artifact.get('found_beside_the_command')
                           else receipt['output_directory'])/payload
        payload = payload.resolve(strict=True)
        selected = Path(choice['selected_path']).resolve(strict=True)
        if selected != payload and not (selected.is_dir() and payload.is_relative_to(selected)):
            raise ValueError('recovery boundary must retain the unique payload actually produced by this training')
        measurement = read_json(safe(output,packet['measurement_ref']))
        frozen = measurement.get('policy_artifact') or {}
        if (frozen.get('content_sha256') and Path(frozen.get('source') or '').resolve() == payload and
                digest(payload) != frozen['content_sha256']):
            raise ValueError('formal producer payload changed after its original archival')
    return packet, choice


def verified_consumer_kind(output: Path, stage: str = 'evaluate') -> dict:
    """A successful invocation supplies a boundary example, not a universal loader rule."""
    try:
        path = safe(Path(output).resolve(), 'derivation_attempts/'+stage+'.json')
        document = read_json(path)
        accepted = next(row for row in reversed(document.get('attempts') or [])
                        if row.get('status') == 'accepted')
        identity = Path(accepted['native_evidence_ref']).stem
        proof = read_attempt_evidence(output, identity, limit=1)
        if proof.get('status') != 'completed' or proof.get('returncode') != 0:
            return {}
        checkpoint = Path((document.get('arguments') or {}).get('checkpoint') or '')
        checkpoint = checkpoint.resolve(strict=True)
        if not checkpoint.is_relative_to(Path(output).resolve()):
            return {}
        return {'kind':'directory' if checkpoint.is_dir() else 'file',
                'native_evidence_ref':accepted['native_evidence_ref'],
                'authority':'successful consumer example; different shapes require Agent source review'}
    except (OSError, ValueError, TypeError, KeyError, StopIteration):
        return {}
