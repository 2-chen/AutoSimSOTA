"""Additive re-scoring of a completed native evaluation; never launch native work.

The original process receipt and failed measurement remain unchanged. Only a
missing structured-result postcondition can be repaired, under the same frozen
protocol, policy and source identity. The independent verifier must agree.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
import uuid

from .common import atomic_json, digest, immutable_json, now, object_digest, read_json
from .experiment_transaction import safe
from .experiment_bundle import freeze_artifact
from .policy_consumption import (verify_policy_consumption, verify_rollout_evidence,
                                 verify_metric_lineage, verified_episode_count)


class LocatorReviewRefused(ValueError):
    def __init__(self,message,ref,review):
        super().__init__(message)
        self.review_ref,self.review=ref,review


def _review_locator(research, *, client, revision, old, original_path, receipt,
                    receipt_path, protocol_path, log):
    """A source-reviewed locator supplement, never a new metric or stage adapter."""
    import json
    from .artifact_revision import validate
    from .baseline_reference import source_evidence
    from .agent_client import role_scope
    from .execution_derive import _object
    from .metric_contract import resolve_metric_artifact
    from .source_citations import project, resolve, INSTRUCTION
    revision = validate('evaluate', revision)
    if client is None or research.metric_spec.artifact_root != 'output':
        raise ValueError('reviewed recovery locator requires a reviewer and an attempt-owned output root')
    output_root = Path(str(receipt.get('output_directory') or ''))
    if (not output_root.is_absolute() or output_root.is_symlink() or
            not output_root.resolve().is_relative_to(research.run_root.resolve()) or
            output_root.resolve() == research.run_root.resolve()):
        raise ValueError('reference result output root is not confined to the native attempt')
    citations = source_evidence(research.repo, revision['source_refs'], output=research.output)
    kwargs = dict(roots={'output':output_root},started_at=float(receipt.get('started_epoch') or 0),
                  allowed_roots=[research.run_root])
    located = resolve_metric_artifact(replace(research.metric_spec,artifact_pattern=revision['pattern']),**kwargs)
    if located.get('status') != 'matched':
        raise ValueError('proposed reference locator has no unique fresh native result: '+str(located.get('status')))
    # The sealed terminal log also bounds legacy receipts without finished_epoch.
    end = receipt.get('finished_epoch')
    if not isinstance(end,(int,float)) or isinstance(end,bool):
        end = log.stat().st_mtime
    if float(located.get('mtime') or 0) > float(end) + .001:
        raise ValueError('proposed result was written after the sealed native operation')
    subject = dict(attempt_id=receipt['attempt_id'], native_receipt_sha256=digest(receipt_path),
        original_measurement_sha256=digest(original_path), reference_protocol_sha256=digest(protocol_path),
        metric=old['metric'], root='output', pattern=revision['pattern'],
        result_sha256=located['sha256'], result_path=located['path'])
    packet,shown=project({'subject':subject,'revision':{**revision,'source_refs':citations},
        'candidate_output_relative_path':Path(located['path']).relative_to(output_root.resolve()).as_posix(),
        'actual_loaded_policy_basename':Path(str(old.get('evaluated_policy_path') or old['policy_artifact']['path'])).name,
        'actual_native_argv':receipt.get('argv') or [],
        'authority':'completed native result locator only; no result values, replay, protocol or stage mutation'},
        citations,local_roots=(research.repo,research.output))
    verdict={};error=''
    try:
        with role_scope(client,'supervisor'):
            answer,_=client.chat_with_metadata(
                'Review ONLY the output locator of this completed native evaluation operation. '
                'The original metric, policy, source, settings and native receipt remain frozen. '
                'Trace exact native writer source and actual caller inputs to the proposed unique '
                'fresh result under this attempt output root. Check checkpoint/model-derived directory '
                'names; do not guess a basename or widen to other runs, input data or policy configs. '
                'Do not select by result value. No edits or native replay. Return JSON booleans '
                'source_output_connection_supported, original_attempt_scope_supported, '
                'native_semantics_preserved, frozen_metric_mapping_preserved, plus why and citations.'+INSTRUCTION,
                json.dumps(packet,ensure_ascii=False),max_tokens=1800,read_only=True,include_research_context=False)
        verdict=_object(answer)
        verdict['citations']=resolve(citations,shown,verdict.get('citations'))
    except (ValueError,TypeError,KeyError) as exc:
        error=str(exc)
    current=resolve_metric_artifact(replace(research.metric_spec,artifact_pattern=revision['pattern']),**kwargs)
    if (source_evidence(research.repo,citations,output=research.output)!=citations or
            digest(receipt_path)!=subject['native_receipt_sha256'] or
            digest(original_path)!=subject['original_measurement_sha256'] or
            digest(protocol_path)!=subject['reference_protocol_sha256'] or
            current.get('status')!='matched' or current.get('sha256')!=subject['result_sha256'] or
            current.get('path')!=subject['result_path']):
        raise ValueError('reference locator subject changed during independent review')
    approved=not error and all(verdict.get(k) is True for k in (
        'source_output_connection_supported','original_attempt_scope_supported',
        'native_semantics_preserved','frozen_metric_mapping_preserved'))
    row=dict(schema_version=1,created_at=now(),subject=subject,
        revision={**revision,'source_refs':citations},review=verdict,review_error=error,
        status='approved_locator_only' if approved else 'rejected',
        authority='only an additive locator certificate; unchanged metric and new independent scoring checks required')
    identity=object_digest(row);ref='reference_locator_reviews/'+identity+'.json'
    immutable_json(safe(research.run_root,ref),row)
    if not approved:
        raise LocatorReviewRefused('reference locator review rejected; '+ref+': '+
            (error or str(verdict.get('why') or 'failed checks'))[:700],ref,verdict)
    # Preserve the declared metric role separately from the reviewed physical locator.
    evidence={**current,'pattern':research.metric_spec.artifact_pattern,
        'locator_pattern':revision['pattern'],'source':'reviewed_completed_reference_locator'}
    return Path(current['path']),evidence,{'ref':ref,'sha256':identity}


def _local_training_proof(root: Path, old: dict) -> dict:
    """Re-read the completed local producer and exact original load bundle."""
    from .completed_evaluation_recovery import _producer_capsule
    from .experiment_bundle import artifact_identity
    train=old.get('train') or {};identity=train.get('attempt_id')
    if not re.fullmatch(r'[0-9a-f]{32}',str(identity or '')):
        raise ValueError('local score recovery requires its completed formal producer')
    path=safe(root,f'attempts/{identity}/receipt.json');receipt=read_json(path)
    if (receipt.get('run_id')!=root.name or receipt.get('attempt_id')!=identity
            or receipt.get('node_id')!='train' or receipt.get('stage')!='train'
            or receipt.get('status')!='completed' or receipt.get('returncode')!=0
            or receipt.get('termination_reason')!='normal_exit'
            or receipt.get('settings_digest')!=object_digest(old.get('settings'))
            or (receipt.get('comparison_protocol_sha256') and
                receipt['comparison_protocol_sha256']!=digest(safe(root,'comparison_protocol.json')))):
        raise ValueError('completed local training identity/settings/protocol changed')
    policy=old.get('policy_artifact') or {}
    source=Path(str(policy.get('source') or ''))
    output=Path(str(receipt.get('output_directory') or ''))
    if (not source.is_absolute() or not output.is_absolute() or
            not source.resolve(strict=True).is_relative_to(output.resolve(strict=True)) or
            not output.resolve().is_relative_to(root.resolve()) or
            any(p.is_symlink() for p in (source,*source.parents) if p.is_relative_to(root))):
        raise ValueError('local evaluated policy is not within its completed producer')
    field,value=artifact_identity(source)
    frozen=Path(str(policy.get('path') or ''))
    if policy.get(field)!=value or artifact_identity(frozen)!=(field,value):
        raise ValueError('local producer and actual evaluated frozen policy bytes differ')
    proof=_producer_capsule(root.parent.parent,root,receipt)
    from .evidence_store import read_attempt_evidence
    sealed=read_attempt_evidence(root.parent.parent,identity)
    if (sealed.get('returncode')!=0 or sealed.get('status')!='completed' or
            sealed.get('receipt_ref')!=path.relative_to(root.parent.parent).as_posix()):
        raise ValueError('completed local training sealed evidence changed')
    return {**proof,'training_attempt_id':identity,'policy_identity_field':field,
            'policy_identity':value,'source_relative_path':source.relative_to(output).as_posix()}


def scoring_receipt(root: Path, measured: dict, receipt: dict) -> dict:
    """Validate an additive supplement, without mutating the native receipt."""
    claim = measured.get('scoring_recovery')
    if claim is None:
        return receipt
    if not isinstance(claim, dict):
        raise ValueError('invalid scoring recovery claim')
    ref = str(claim.get('attachment_ref') or '')
    if not re.fullmatch(r'metric_recoveries/[0-9a-f]{64}\.json', ref):
        raise ValueError('unsafe scoring recovery supplement')
    attachment = read_json(safe(root, ref))
    if object_digest(attachment) != Path(ref).stem or object_digest(attachment) != claim.get('attachment_sha256'):
        raise ValueError('scoring recovery supplement changed')
    attempt = (measured.get('evaluate') or {}).get('attempt_id')
    native_path = safe(root, f'attempts/{attempt}/receipt.json') if re.fullmatch(r'[0-9a-f]{32}', str(attempt)) else None
    old_ref = str(attachment.get('original_measurement_ref') or '')
    if not re.fullmatch(r'measurements/[A-Za-z0-9_-]{1,80}\.json', old_ref):
        raise ValueError('unsafe original measurement')
    old_path = safe(root, old_ref)
    old = read_json(old_path)
    if (attachment.get('schema_version') != 1 or native_path is None
            or attachment.get('native_receipt_sha256') != digest(native_path)
            or attachment.get('original_measurement_sha256') != digest(old_path)
            or attachment.get('attempt_id') != attempt
            or read_json(native_path) != receipt
            or old.get('ok') is not False or old.get('where') != 'evaluate'
            or (old.get('metric_artifact_evidence') or {}).get('status') != 'missing'
            or old.get('result_artifact') or old.get('metric_value') is not None
            or old.get('baseline_origin') not in {'released_reference','locally_trained'}
            or old.get('confirmation') or measured.get('confirmation')
            or any(measured.get(k) != old.get(k) for k in (
                'metric', 'settings', 'policy_artifact', 'source_state', 'reference_protocol_ref',
                'baseline_origin', 'evaluate', 'train', 'data_consumption'))
            or receipt.get('status') != 'completed' or receipt.get('returncode') != 0
            or receipt.get('termination_reason') != 'normal_exit'):
        raise ValueError('scoring recovery is not bound to the original completed operation')
    if old.get('baseline_origin')=='locally_trained':
        if (old.get('reference_protocol_ref') or
                attachment.get('local_training_proof')!=_local_training_proof(root,old)):
            raise ValueError('local scoring supplement lost its completed training proof')
    conditions = deepcopy(receipt.get('postconditions') or [])
    if any(not isinstance(row, dict) or
           (row.get('passed') is False and row.get('id') != 'metric_result_artifact')
           for row in conditions):
        raise ValueError('recovery cannot waive another native postcondition')
    evidence = attachment.get('metric_result_artifact') or {}
    if evidence.get('status') != 'matched':
        raise ValueError('scoring supplement has no unique fresh result')
    locator = attachment.get('locator_review')
    if evidence.get('source') == 'reviewed_completed_reference_locator' or locator is not None:
        if (not isinstance(locator,dict) or not re.fullmatch(
                r'reference_locator_reviews/[0-9a-f]{64}\.json',str(locator.get('ref') or ''))):
            raise ValueError('missing reviewed reference locator certificate')
        review=read_json(safe(root,locator['ref']));subject=review.get('subject') or {}
        if (object_digest(review)!=locator.get('sha256') or
                object_digest(review)!=Path(locator['ref']).stem or
                review.get('status')!='approved_locator_only' or review.get('review_error') or
                not all(review.get('review',{}).get(k) is True for k in (
                    'source_output_connection_supported','original_attempt_scope_supported',
                    'native_semantics_preserved','frozen_metric_mapping_preserved')) or
                not review.get('review',{}).get('citations') or
                subject.get('attempt_id')!=attempt or
                subject.get('native_receipt_sha256')!=digest(native_path) or
                subject.get('original_measurement_sha256')!=digest(old_path) or
                subject.get('reference_protocol_sha256')!=digest(safe(root,
                    old.get('reference_protocol_ref') or 'comparison_protocol.json')) or
                subject.get('metric')!=old['metric'] or subject.get('root')!=evidence.get('root') or
                subject.get('pattern')!=evidence.get('locator_pattern') or
                subject.get('result_path')!=evidence.get('path') or
                subject.get('result_sha256')!=evidence.get('sha256')):
            raise ValueError('reference locator certificate does not bind this completed operation')
    for row in conditions:
        if row.get('id') == 'metric_result_artifact':
            row['passed'] = True
    return {**receipt, 'metric_result_artifact': evidence, 'postconditions': conditions}


def recover(research, *, reference_id: str = '', original_ref: str,
            locator_revision=None, client=None) -> tuple[dict, str]:
    """Prepare a new independently verified reading; no launch or adoption."""
    if reference_id and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', reference_id):
        raise ValueError('invalid reference identity')
    if not re.fullmatch(r'research/[A-Za-z0-9_-]{1,80}/measurements/[A-Za-z0-9_-]{1,80}\.json', original_ref):
        raise ValueError('unsafe original reference measurement')
    original_path = safe(research.output, original_ref)
    if original_path.parent.parent != research.run_root:
        raise ValueError('reference measurement belongs to another research run')
    old = read_json(original_path)
    if (not isinstance(old, dict) or old.get('ok') is not False
            or old.get('where') != 'evaluate' or old.get('confirmation')
            or old.get('baseline_origin') != ('released_reference' if reference_id else 'locally_trained')
            or (old.get('metric_artifact_evidence') or {}).get('status') != 'missing'
            or old.get('result_artifact') or old.get('metric_value') is not None):
        raise ValueError('only a missing result from a failed reference can be recovered')
    attempt = (old.get('evaluate') or {}).get('attempt_id')
    if not re.fullmatch(r'[0-9a-f]{32}', str(attempt)):
        raise ValueError('reference has no completed evaluation identity')
    receipt_path = safe(research.run_root, f'attempts/{attempt}/receipt.json')
    receipt = read_json(receipt_path)
    if (receipt.get('node_id') != 'evaluate' or receipt.get('attempt_id') != attempt
            or receipt.get('status') != 'completed' or receipt.get('returncode') != 0
            or receipt.get('termination_reason') != 'normal_exit'
            or any(not isinstance(row, dict) or (row.get('passed') is False and
                row.get('id') != 'metric_result_artifact') for row in receipt.get('postconditions') or [])):
        raise ValueError('native operation is not completed or has another failed postcondition')
    local_proof=None
    if not reference_id:
        if any((research.run_root/'confirmation_attempts').glob('*.json')) or (research.run_root/'confirmation.json').exists():
            raise ValueError('local score recovery forbidden after held-out exposure')
        local_proof=_local_training_proof(research.run_root,old)
    protocol_ref = f'reference_protocols/{reference_id}.json' if reference_id else 'comparison_protocol.json'
    protocol_path = safe(research.run_root, protocol_ref)
    if (old.get('reference_protocol_ref') != (protocol_ref if reference_id else None)
            or receipt.get('reference_protocol_ref') != (protocol_ref if reference_id else None)
            or digest(protocol_path) != receipt.get('comparison_protocol_sha256')
            or research.metric_spec.as_dict() != old.get('metric')
            or object_digest(old.get('settings')) != receipt.get('settings_digest')):
        raise ValueError('frozen reference protocol or metric changed')
    from .workspace_snapshot import verify_state
    manifest = read_json(safe(research.output, 'workspace_snapshot.json'))
    if not verify_state(research.repo, manifest, old.get('source_state') or {}, research.snapshots):
        raise ValueError('target source changed since reference evaluation')
    from .evidence_store import read_attempt_evidence
    sealed = read_attempt_evidence(research.output, attempt)
    sealed_record = read_json(safe(research.output, f'evidence/{attempt}.json'))
    log = Path(str(receipt.get('log') or ''))
    if (sealed.get('returncode') != 0 or not log.is_file()
            or not log.resolve().is_relative_to(research.run_root.resolve())
            or sealed.get('receipt_ref') != receipt_path.relative_to(research.output).as_posix()
            or safe(research.output, str(sealed_record.get('log_ref') or '')).resolve() != log.resolve()):
        raise ValueError('completed native evidence is unavailable')
    prior = getattr(research, '_reference_protocol_id', '')
    if prior:
        raise ValueError('another reference evaluation owns the controller')
    research._reference_protocol_id = reference_id
    try:
        artifact, evidence = research._metric_artifact_for_record(receipt)
    finally:
        research._reference_protocol_id = prior
    locator_review=None
    if locator_revision is not None:
        if artifact is not None and evidence.get('status')=='matched':
            raise ValueError('frozen producer already locates the result; omit artifact_revision')
        artifact,evidence,locator_review=_review_locator(research,client=client,revision=locator_revision,
            old=old,original_path=original_path,receipt=receipt,receipt_path=receipt_path,
            protocol_path=protocol_path,log=log)
    if artifact is None or evidence.get('status') != 'matched':
        raise ValueError('no unique fresh result under the frozen producer rule')
    if not verify_state(research.repo,manifest,old.get('source_state') or {},research.snapshots):
        raise ValueError('target source changed during completed score recovery')
    policy = verify_policy_consumption(log, Path(old['policy_artifact']['path']))
    rollout = verify_rollout_evidence(log, policy_sha256=str(policy.get('sha256') or ''))
    label = ('reference_recovered_' if reference_id else 'score_recovered_') + uuid.uuid4().hex[:16]
    frozen = freeze_artifact(artifact, research.run_root/'experiments'/label/
                            ('result'+artifact.suffix), max_bytes=64*1024**2)
    frozen['content_sha256'] = digest(Path(frozen['path']))
    if frozen['content_sha256'] != evidence.get('sha256'):
        raise ValueError('native result changed while being archived')
    reading = research.metric_spec.read(said='', artifact=Path(frozen['path']))
    lineage = verify_metric_lineage(log, policy_sha256=str(policy.get('sha256') or ''),
        episode_ids=rollout.get('episode_ids') or [], value=reading.get('value'))
    expected_samples = (read_json(protocol_path).get('settings') or {}).get('episodes')
    native_count=verified_episode_count({'policy_consumption':policy,'rollout_evidence':rollout,
        'metric_lineage':lineage,'metric_reading':reading,'metric':old['metric'],
        'metric_value':reading.get('value')})
    if (not isinstance(expected_samples, int) or isinstance(expected_samples, bool)
            or expected_samples < 1 or native_count != expected_samples
            or rollout.get('episodes_completed') != expected_samples
            or reading.get('value') is None
            or any(row.get('status') != 'verified' for row in (policy, rollout, lineage))):
        raise ValueError('native load, rollout and result lineage did not reverify')
    supplement = {'schema_version': 1, 'attempt_id': attempt,
        **({'local_training_proof':local_proof} if local_proof else {}),
        'original_measurement_ref': original_path.relative_to(research.run_root).as_posix(),
        'original_measurement_sha256': digest(original_path),
        'native_receipt_sha256': digest(receipt_path),
        **({'locator_review':locator_review} if locator_review else {}),
        'metric_result_artifact': {**{k:evidence.get(k) for k in (
            'status','path','root','pattern','sha256','size_bytes','mtime',
            'source','locator_pattern')},
            'frozen_copy_ref': Path(frozen['path']).relative_to(research.run_root).as_posix(),
            'frozen_copy_sha256': frozen['content_sha256']}}
    identity = object_digest(supplement)
    attachment_ref = f'metric_recoveries/{identity}.json'
    atomic_json(safe(research.run_root, attachment_ref), supplement)
    result = {**deepcopy(old), 'label': label, 'ok': True, 'where': '', 'why': '',
        'result_artifact': frozen, 'metric_artifact_evidence': evidence,
        'metric_reading': reading, 'metric_value': reading['value'],
        'metric_utility': research.metric_spec.utility(reading['value']),
        'policy_consumption': policy, 'rollout_evidence': rollout, 'metric_lineage': lineage,
        'scoring_recovery': {'attachment_ref': attachment_ref, 'attachment_sha256': identity,
            'native_replayed': False, 'original_measurement_ref': original_ref}}
    if research.metric_spec.name == 'success_rate':
        result.update(success_rate=reading['value'], success_reading=reading)
    scoring_receipt(research.run_root, result, receipt)
    measured_path = safe(research.run_root, f'measurements/{label}.json')
    atomic_json(measured_path, result)
    from .receipt_verifier import verify_measurement
    audit = verify_measurement(research.run_root, label)
    if audit.get('status') != 'consistent':
        result.update(ok=False, where='scoring_recovery',
                      why='independent receipt verification refused recovery',
                      metric_value=None, metric_utility=None, success_rate=None,
                      recovery_verification=audit)
        atomic_json(measured_path, result)
        raise ValueError('recovered reading failed independent receipt checks: ' + '; '.join(audit.get('issues') or []))
    return result, measured_path.relative_to(research.output).as_posix()
