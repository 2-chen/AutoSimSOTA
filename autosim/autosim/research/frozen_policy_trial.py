"""Agent-selected evaluation-only candidates with immutable local training identity."""
import json
import re
from pathlib import Path

from .common import digest, object_digest, read_json


def metadata_sources(root, request):
    """Bounded JSON from the requested archive; never read model tensors.

    This is inspectable metadata, not full policy byte verification. The native
    executor still uses load() to verify the entire immutable policy before work.
    """
    try:
        root = Path(root).resolve(strict=True)
        if (not isinstance(request, dict) or set(request) != {'label', 'measurement_sha256'}
                or not isinstance(request['label'], str)
                or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', request['label'])):
            return []
        measured = _safe(root, 'measurements/' + request['label'] + '.json')
        if measured.stat().st_size > 2 * 1024 * 1024 or digest(measured) != request['measurement_sha256']:
            return []
        artifact = (read_json(measured).get('policy_artifact') or {})
        raw = Path(artifact.get('path') or '')
        if not raw.is_absolute() or not raw.is_relative_to(root):
            return []
        archive = _safe(root, raw.relative_to(root))
        if not archive.is_dir():
            return []  # File-only weights have no identity-bound sibling metadata.
        rows, total = [], 0
        paths = []
        for index, path in enumerate(archive.rglob('*.json')):
            if index >= 128:
                return []  # No unbounded inventory or arbitrary partial selection.
            paths.append(path)
        for path in sorted(paths):
            if len(rows) >= 8:
                break
            if not re.fullmatch(r'(?:.*[._-])?(?:config|configuration|metadata)\.json', path.name, re.IGNORECASE):
                continue  # Do not forward arbitrary JSON arrays/data/weight files.
            path = _safe(root, path.relative_to(root))
            size = path.stat().st_size
            if size > 16384 or total + size > 32768 or not path.is_file():
                continue
            text = path.read_text(encoding='utf-8')
            json.loads(text)  # Data only, never import/deserialise policy code.
            rows.append({'id': f'frozen_policy_metadata_{len(rows)}',
                'path': path.relative_to(archive).as_posix(), 'text': text,
                'sha256': digest(path), 'policy_sha256': artifact.get('sha256'),
                'complete': True,
                'authority': 'requested run-owned archive JSON; metadata bytes only, not loader consumption or a score'})
            total += size
        return rows  # Existing external-model boundaries sanitize the full packet.
    except (OSError, ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        return []


def offered(root):
    """Small incumbent offer, not byte verification or automatic execution."""
    try:
        state_path=_safe(root,'snapshots/snapshots.json')
        if not state_path.is_file() or state_path.stat().st_size>2*1024*1024:
            return []
        state=read_json(state_path)
        best=state.get('best')
        if not isinstance(best,str) or not best:
            return []
        label=best.replace('-','_',1)
        path=Path(root)/'measurements'/f'{label}.json'
        row=read_json(_safe(root,path.relative_to(root)))
        if not row.get('ok') or row.get('confirmation') or not row.get('policy_artifact'):
            return []
        return [{'label':label,'measurement_sha256':digest(path),
                 'policy_metadata_sources':metadata_sources(root, {'label':label,'measurement_sha256':digest(path)}),
                 'policy_sha256':row['policy_artifact'].get('sha256'),
                 'training_attempt_id':(row.get('train') or {}).get('attempt_id'),
                 'authority':'unverified offer; executor rechecks native receipts and bytes',
                 'guidance':'For a code-only deployment candidate explicitly choose '
                     'run_the_loop(reuse_policy={label,measurement_sha256}) without '
                     'training_settings/job_id. Reuses exact selected weights, not new '
                     'data/training; runs a fresh frozen-protocol evaluation.'}]
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        return []


def _safe(root, relative):
    root=Path(root).resolve(strict=True)
    relative=Path(relative)
    path=root/relative
    if (relative.is_absolute() or '..' in relative.parts or
            any(p.is_symlink() for p in (path,*path.parents) if p.is_relative_to(root)) or
            not path.resolve().is_relative_to(root)):
        raise ValueError('unsafe frozen policy evidence path')
    return path


def load(research, request):
    """Reverify a scored local policy, never infer a checkpoint from its filename."""
    from .receipt_verifier import verify_measurement
    from .evidence_store import read_attempt_evidence
    from .experiment_bundle import artifact_identity
    if (not isinstance(request,dict) or set(request)!={'label','measurement_sha256'} or
            not isinstance(request['label'],str) or
            not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',request['label']) or
            not isinstance(request['measurement_sha256'],str) or
            not re.fullmatch(r'[0-9a-f]{64}',request['measurement_sha256'])):
        raise ValueError('reuse_policy requires an exact label and measurement_sha256')
    root=research.run_root.resolve(strict=True)
    if research.execution_graph:
        raise ValueError('frozen policy graph input needs an explicit verified graph edge')
    if ((root/'confirmation.json').exists() or
            any((root/'confirmation_attempts').glob('*.json'))):
        raise ValueError('development cannot resume after held-out exposure')
    path=_safe(root,'measurements/'+request['label']+'.json')
    if not path.is_file() or path.stat().st_size>2*1024*1024 or digest(path)!=request['measurement_sha256']:
        raise ValueError('frozen policy parent measurement changed or unavailable')
    parent=read_json(path)
    if not isinstance(parent,dict) or not isinstance(parent.get('guardrails') or {},dict):
        raise ValueError('frozen policy parent metadata is malformed')
    if (parent.get('label')!=request['label'] or parent.get('ok') is not True or
            parent.get('confirmation') or parent.get('baseline_origin')!='locally_trained' or
            (parent.get('guardrails') or {}).get('status') in {'unknown','violated'} or
            not isinstance(parent.get('settings'),dict)):
        raise ValueError('frozen policy requires a valid locally trained development measurement')
    if verify_measurement(root,request['label']).get('status')!='consistent':
        raise ValueError('frozen policy parent evidence is inconsistent')
    train=parent.get('train') or {}
    if not isinstance(train,dict) or not isinstance(train.get('artifact'),dict):
        raise ValueError('frozen policy training metadata is malformed')
    attempt=train.get('attempt_id')
    if not isinstance(attempt,str) or not re.fullmatch(r'[0-9a-f]{32}',attempt):
        raise ValueError('frozen policy has no exact local training attempt')
    proof=read_attempt_evidence(research.output,attempt,limit=1)
    receipt_path=_safe(root,'attempts/'+attempt+'/receipt.json')
    sidecar=receipt_path.with_name('training_progress_revalidation.json')
    if sidecar.exists() or sidecar.is_symlink():
        sidecar=_safe(root,sidecar.relative_to(root))
        reviewed=read_json(sidecar)
        if not isinstance(reviewed,dict) or reviewed.get('native_receipt_sha256')!=digest(receipt_path):
            raise ValueError('frozen policy audited training receipt identity changed')
    if (proof.get('receipt_ref')!=receipt_path.relative_to(research.output).as_posix() or
            proof.get('status')!='completed' or proof.get('returncode')!=0 or
            proof.get('termination_reason')!='normal_exit'):
        raise ValueError('frozen policy native training receipt is not sealed normal completion')
    receipt=research._verified_training_receipt(attempt,parent['settings'])
    if receipt is None:
        raise ValueError('frozen policy training settings/progress/protocol identity did not verify')
    artifact=train.get('artifact') or {}
    native_artifact=receipt.get('artifact') or {}
    for key in ('checked','matched','candidate_paths','examples'):
        if artifact.get(key)!=native_artifact.get(key):
            raise ValueError('frozen policy selection differs from native artifact inventory')
    trained={**receipt,'artifact':dict(artifact),
             'artifact_selection':dict(train.get('artifact_selection') or {})}
    from .checkpoint_handoff import bound_handoff
    handoff = bound_handoff(research.output, parent, producer_evidence_ref=receipt['evidence_ref'])
    if handoff is not None:
        trained['explicit_checkpoint_handoff'] = handoff
    produced=research._recorded_artifact(trained)
    if produced is None:
        raise ValueError('frozen policy original artifact selection is unverifiable')
    policy=dict(parent.get('policy_artifact') or {})
    raw=Path(str(policy.get('path') or ''))
    if not raw.is_absolute() or not raw.is_relative_to(root):
        raise ValueError('frozen policy archive is not run-owned')
    archived=_safe(root,raw.relative_to(root))
    field,identity=artifact_identity(archived)
    source=Path(str(policy.get('source') or ''))
    if (policy.get(field)!=identity or not source.is_absolute() or
            source.resolve(strict=True)!=produced.resolve(strict=True) or
            artifact_identity(produced)!=(field,identity)):
        raise ValueError('frozen policy bytes or selected training source identity changed')
    manifest_identity=(object_digest([(produced.name,identity)])
                       if field=='content_sha256' else identity)
    if policy.get('sha256')!=manifest_identity:
        raise ValueError('frozen policy archive manifest identity changed')
    if research._comparison_protocol_violation(parent['settings'],target='evaluate',freeze=False):
        raise ValueError('frozen policy parent does not match the current frozen protocol')
    return {'request':dict(request),'settings':dict(parent['settings']),
            'trained':trained,'policy_artifact':policy,
            'identity':{'mode':'evaluation_only_local_frozen_policy',**request,
                        'training_attempt_id':attempt,'policy_sha256':manifest_identity,
                        'training_receipt_sha256':digest(receipt_path),
                        'policy_byte_identity':{'field':field,'digest':identity},
                        'no_new_training_or_data_claim':True}}
