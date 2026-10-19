"""Link an existing terminal evaluation to its exact completed training producer.

No native work, model judgment, score selection or historical receipt rewrite occurs.
Training recipe and evaluation-call settings remain distinct and independently bound.
"""
from copy import deepcopy
from pathlib import Path
import re
from types import SimpleNamespace

from .common import digest, immutable_json, object_digest, read_json
from .experiment_transaction import safe


def _document(root, ref, bound=2*1024**2):
    path=safe(root,ref)
    if not path.is_file() or path.stat().st_size>bound:
        raise ValueError('completed evaluation document missing or oversized')
    row=read_json(path)
    if not isinstance(row,dict):
        raise ValueError('completed evaluation document must be an object')
    return row,path


def _normalized(argv):
    """Resolve existing executor aliases, not arbitrary substitutions or flags."""
    result=[]
    for value in argv:
        prefix,path='',value
        if '=' in value:
            lhs,rhs=value.split('=',1)
            if rhs.startswith('/'):
                prefix,path=lhs+'=',rhs
        result.append(prefix+str(Path(path).resolve()) if path.startswith('/') else value)
    return result


def _source_capsule(output,repo,request):
    from .native_jobs import _checkout_source
    if request.get('source_stage_scope')!=['evaluate']:
        raise ValueError('completed evaluation linkage currently requires a scoped evaluator source identity')
    kept,_=_document(output,'derived_stages.json')
    capsule={**_checkout_source(output,repo),'derived_stages':{
        'scope_version':1,'stages':{'evaluate':kept['evaluate']}}}
    if object_digest(capsule)!=request.get('source_identity'):
        raise ValueError('current source cannot be captured as the original evaluation identity')
    return capsule


def _producer_capsule(output,root,receipt):
    progress=receipt.get('training_progress') or {}
    audit_ref=progress.get('audit_ref')
    if not isinstance(audit_ref,str) or not re.fullmatch(r'training_progress_audits/[0-9a-f]{64}\.json',audit_ref):
        raise ValueError('completed training lacks its original independent progress audit')
    _,audit=_document(output,audit_ref)
    counter=(progress.get('evidence') or {}).get('native_counter') or {}
    directory=Path(receipt['output_directory']).resolve(strict=True)
    if not directory.is_relative_to(root):raise ValueError('training output escaped research')
    path=safe(directory,counter['path'])
    document=read_json(path);actual=document
    for field in counter['field']:actual=actual[field]
    if (progress.get('status')!='observed' or type(actual) not in (int,float) or
            actual<=0 or actual!=actual or abs(actual)==float('inf') or actual!=counter['value'] or
            digest(path)!=(progress.get('evidence') or {}).get('document_sha256')):
        raise ValueError('completed training native counter or audited document changed')
    return {'receipt_sha256':digest(root/f"attempts/{receipt['attempt_id']}/receipt.json"),
        'audit_ref':audit_ref,'audit_sha256':digest(audit),'counter':counter,
        'counter_document_sha256':digest(path),
        'authority':'independent source/progress validation at linkage; immutable native counter retained'}


def _argv_equivalence(output, repo, protocol, parent_settings, request, native, policy,source_capsule,
                      interpreter=None):
    """Replay the frozen pure builder for BOTH phase settings and actual argv."""
    from .declarative_backend import DeclarativeBackend
    from .derived_research import DerivedResearch
    source=protocol.get('evaluator_argv_source')
    entry=source_capsule['derived_stages']['stages']['evaluate']
    current=entry.get('source') if isinstance(entry,dict) else entry
    if not isinstance(source,str) or current!=source:
        raise ValueError('frozen evaluator builder changed')
    backend=DeclarativeBackend(repo=repo,answer={'stages':{'evaluate':protocol['evaluator_stage']}},
        sources={'evaluate':source},parameters={'evaluate':protocol['evaluator_parameters']})
    if interpreter is None:
        environment,_=_document(output,'environment.json')
        interpreter=environment.get('interpreter') or environment.get('python')
    if not isinstance(interpreter,str) or not Path(interpreter).is_absolute():
        raise ValueError('recorded native interpreter unavailable')
    directory=Path(native['output_directory'])
    if not directory.resolve().is_relative_to(output):
        raise ValueError('native output is outside its run')
    stub=SimpleNamespace(repo=repo,interpreter=Path(interpreter),backend=backend,
                         stage_directory=lambda stage:directory)
    stub._task=lambda:DerivedResearch._task(stub)
    overrides={'output':str(directory),'checkpoint':str(policy),'experiment_dir':str(policy),
        'previous_artifact':str(policy),'device':native.get('device') or '',
        'device_index':int((native.get('compute_environment') or {}).get('AUTOSIM_DEFAULT_CUDA_ORDINAL') or 0)}
    commands=[backend.argv('evaluate',DerivedResearch._inputs(stub,'evaluate',settings=settings,
        **overrides)) for settings in (parent_settings,request['settings'])]
    observed=native.get('argv')
    if (not isinstance(observed,list) or not observed or
            any(not isinstance(v,str) for v in observed) or
            _normalized(commands[0])!=_normalized(commands[1]) or
            _normalized(commands[1])!=_normalized(observed)):
        raise ValueError('training-recipe and actual evaluation inputs do not reproduce the same frozen argv')
    return interpreter


def _context(output, repo, run_id, label, job_id, *, producers=None,
             source_capsule=None,producer_capsule=None,interpreter=None,historical=False):
    from .native_jobs import source_matches, checkpoint_overrides
    from .checkpoint_handoff import formal_producers
    from .evidence_store import read_attempt_evidence
    from .metric_contract import MetricSpec
    from .policy_consumption import (verify_policy_consumption,verify_rollout_evidence,
                                     verify_metric_lineage)
    output=Path(output).resolve(strict=True);repo=Path(repo).resolve(strict=True)
    if (not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',run_id) or
            not re.fullmatch(r'round_[1-9][0-9]*',label) or
            not re.fullmatch(r'[0-9a-f]{32}',job_id)):
        raise ValueError('invalid bounded completed-evaluation identity')
    root=safe(output,f'research/{run_id}')
    if not historical and ((root/'confirmation.json').exists() or
            any((root/'confirmation_attempts').glob('*.json'))):
        raise ValueError('completed evaluation cannot be linked after held-out exposure')
    parent,parent_path=_document(root,f'measurements/{label}.json')
    train=parent.get('train') or {};train_id=train.get('attempt_id')
    if (parent.get('label')!=label or parent.get('ok') is not False or
            parent.get('where') not in {'evaluate','policy_artifact'} or
            parent.get('confirmation') or parent.get('zero_work_reported') or
            not re.fullmatch(r'[0-9a-f]{32}',str(train_id or ''))):
        raise ValueError('original incomplete local training measurement required')
    training_receipt,_=_document(root,f'attempts/{train_id}/receipt.json')
    if (training_receipt.get('settings_digest')!=object_digest(parent.get('settings')) or
            training_receipt.get('status')!='completed' or training_receipt.get('returncode')!=0 or
            training_receipt.get('termination_reason')!='normal_exit'):
        raise ValueError('original training recipe is not bound to its native receipt')
    if producer_capsule is None:
        inventory=formal_producers(output) if producers is None else producers
        producer=next((p for p in inventory if p.get('origin')=='formal_measurement' and
            p.get('run_id')==run_id and p.get('training_attempt_id')==train_id),None)
        if producer is None:
            raise ValueError('completed training source/progress/receipt is not verified')
        producer_capsule=_producer_capsule(output,root,training_receipt)
    elif _producer_capsule(output,root,training_receipt)!=producer_capsule:
        raise ValueError('historical completed-training proof changed')
    train_proof=read_attempt_evidence(output,train_id,limit=1)
    if (train_proof.get('status')!='completed' or train_proof.get('returncode')!=0 or
            train_proof.get('receipt_ref')!=f'research/{run_id}/attempts/{train_id}/receipt.json'):
        raise ValueError('original sealed training identity changed')
    request,request_path=_document(output,f'native_jobs/{job_id}/request.json')
    result,result_path=_document(output,f'native_jobs/{job_id}/result.json')
    selection=request.get('checkpoint_selection') or {}
    if (request.get('job_id')!=job_id or result.get('job_id')!=job_id or
            request.get('stage')!='evaluate' or request.get('run_id')!=run_id or
            request.get('repo')!=str(repo) or result.get('status')!='completed' or
            not request.get('source_identity') or
            result.get('source_identity_after')!=request['source_identity'] or
            selection.get('producer_evidence_ref')!=f'evidence/{train_id}.json'):
        raise ValueError('job source/operation or exact training producer disagrees')
    if source_capsule is None:
        if not source_matches(output,repo,request):raise ValueError('evaluation source changed before linkage')
        source_capsule=_source_capsule(output,repo,request)
    elif (object_digest(source_capsule)!=request['source_identity'] or
            (source_capsule.get('derived_stages') or {}).get('scope_version')!=1 or
            set(source_capsule['derived_stages'].get('stages') or {})!={'evaluate'}):
        raise ValueError('historical evaluation source capsule changed')
    checkpoint_overrides(output,'evaluate',selection)
    policy=Path(selection['selected_path']).resolve(strict=True)
    native=result.get('stage_result') or {};attempt=native.get('attempt_id')
    if not re.fullmatch(r'[0-9a-f]{32}',str(attempt or '')):
        raise ValueError('completed evaluation has no native identity')
    receipt,receipt_path=_document(root,f'attempts/{attempt}/receipt.json')
    proof=read_attempt_evidence(output,attempt,limit=1)
    if (receipt.get('attempt_id')!=attempt or receipt.get('controller_decision_id')!=job_id or
            receipt.get('run_id')!=run_id or receipt.get('stage')!='evaluate' or
            receipt.get('node_id')!='evaluate' or receipt.get('status')!='completed' or
            receipt.get('returncode')!=0 or receipt.get('termination_reason')!='normal_exit' or
            native.get('status')!='completed' or native.get('returncode')!=0 or
            native.get('termination_reason')!='normal_exit' or
            native.get('argv')!=receipt.get('argv') or native.get('evidence_id')!=attempt or
            native.get('output_directory')!=receipt.get('output_directory') or
            proof.get('status')!='completed' or proof.get('returncode')!=0 or
            proof.get('receipt_ref')!=receipt_path.relative_to(output).as_posix() or
            receipt.get('settings_digest')!=object_digest(request.get('settings')) or
            any(not isinstance(p,dict) or p.get('passed') is False
                for p in receipt.get('postconditions') or [])):
        raise ValueError('evaluation must have exact sealed normal-exit ownership and postconditions')
    protocol,protocol_path=_document(root,'comparison_protocol.json')
    if receipt.get('comparison_protocol_sha256')!=digest(protocol_path):
        raise ValueError('frozen evaluation protocol changed')
    spec=MetricSpec(**protocol['metric'])
    interpreter=_argv_equivalence(output,repo,protocol,parent['settings'],request,receipt,policy,
                                  source_capsule,interpreter=interpreter)
    evidence=receipt.get('metric_result_artifact') or {}
    if (native.get('metric_result_artifact')!=evidence or evidence.get('status')!='matched' or
            evidence.get('matched')!=1 or evidence.get('pattern')!=spec.artifact_pattern or
            evidence.get('root')!=spec.artifact_root or spec.source not in {'json','csv'}):
        raise ValueError('one original frozen-contract structured metric required')
    artifact=Path(evidence.get('path') or '').resolve(strict=True)
    if (not artifact.is_relative_to(root) or artifact.stat().st_size>64*1024**2 or
            digest(artifact)!=evidence.get('sha256')):
        raise ValueError('original structured result bytes changed')
    raw,_=_document(output,f'evidence/{attempt}.json')
    log=safe(output,raw['log_ref'])
    if Path(receipt.get('log') or '').resolve()!=log.resolve():
        raise ValueError('native log reference disagrees')
    reading=spec.read(said='',artifact=artifact)
    policy_proof=verify_policy_consumption(log,policy)
    expected=(protocol.get('settings') or {}).get('episodes')
    if type(expected) is not int or expected<1:
        raise ValueError('explicit frozen episode population required for completed evaluation recovery')
    rollout=verify_rollout_evidence(log,policy_sha256=policy_proof.get('sha256') or '',min_episodes=expected)
    lineage=verify_metric_lineage(log,policy_sha256=policy_proof.get('sha256') or '',
        episode_ids=rollout.get('episode_ids') or [],value=reading.get('value'))
    if (reading.get('value') is None or reading.get('samples')!=expected or
            rollout.get('episodes_completed')!=expected or any(p.get('status')!='verified'
                for p in (policy_proof,rollout,lineage))):
        raise ValueError('actual policy loading, complete rollout or metric lineage did not verify')
    return locals()


def registered_training_blocker(output, run_id, measurement):
    """Do not advertise a score recovery whose ORIGINAL data binding fails.

    This is a read-only prerequisite check, never a retrospective registration.
    Unbound native training remains unknown and is not falsely rejected as having
    a broken registered binding. Execution independently repeats its own checks.
    """
    if (measurement.get('data_selection') is None and
            not (measurement.get('data_consumption') or {}).get('versions')):
        return None
    try:
        verified_training_data(output, run_id, measurement)
        return None
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {'label': measurement.get('label'),
                'training_attempt_id': (measurement.get('train') or {}).get('attempt_id'),
                'measurement_ref': f'research/{run_id}/measurements/{measurement.get("label")}.json',
                'status': 'blocked_registered_training_data', 'why': str(exc)[:700],
                'next_action': 'research_task',
                'authority': 'read-only prerequisite failure, not a score or repository boundary; '
                    'inspect original sealed selection/reader evidence; do not repeat unchanged '
                    'handoff or invent past consumption; Main chooses a justified new action'}


def offered(output,repo,run_id,*,producers=None,_blocked=None):
    """Read-only possible links; Main explicitly chooses one, never a best score."""
    output=Path(output);root=output/'research'/run_id
    if not root.is_dir():return []
    parents=[]
    for path in sorted((root/'measurements').glob('round_*.json'))[-100:]:
        if re.fullmatch(r'round_[1-9][0-9]*',path.stem):
            try:
                row=read_json(path)
                parents.append((path.stem,(row.get('train') or {}).get('attempt_id')))
            except (OSError,ValueError,TypeError):pass
    jobs=sorted((output/'native_jobs').glob('*/request.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:12]
    result=[]
    for path in jobs:
        try:
            request=read_json(path);producer=Path((request.get('checkpoint_selection') or {}).get('producer_evidence_ref') or '').stem
            for label,train_id in parents:
                if producer!=train_id:continue
                target=f'{label}_eval_{path.parent.name[:12]}'
                if (root/'measurements'/f'{target}.json').exists():continue
                c=_context(output,repo,run_id,label,path.parent.name,producers=producers)
                blocker=registered_training_blocker(output,run_id,c['parent'])
                if blocker:
                    if _blocked is not None:
                        _blocked.append({**blocker,'job_id':path.parent.name,
                            'evaluation_attempt_id':c['attempt'],
                            'native_metric_value_not_linked':c['reading']['value']})
                    continue
                result.append({'label':label,'job_id':path.parent.name,'training_attempt_id':train_id,
                    'evaluation_attempt_id':c['attempt'],'metric_value':c['reading']['value'],
                    'next_action':'correct_policy_handoff','arguments':{'label':label,'job_id':path.parent.name},
                    'authority':'existing native work only; execution revalidates all identities; no adoption'})
        except (OSError,ValueError,TypeError,KeyError,AttributeError):continue
    return result


def _training_data(c):
    """Recheck a PREEXISTING training selection, never bind an unbound old train."""
    from .data_versions import verify_consumption
    parent=c['parent'];output=c['output'];proof=parent.get('data_selection')
    if proof is None:
        if (parent.get('data_consumption') or {}).get('versions'):
            raise ValueError('unbound training cannot acquire a retrospective data identity')
        return [],verify_consumption(output,[],parent['train'],training_performed=True)
    expected=f"research/{c['run_id']}/data_selections/{proof.get('id')}.json"
    body,path=_document(output,expected)
    if (proof.get('record_ref')!=expected or digest(path)!=proof.get('record_sha256') or
            object_digest(body)!=proof.get('id') or body.get('label')!=c['label'] or
            body.get('settings_sha256')!=object_digest(parent['settings']) or
            proof.get('training_attempt_id')!=c['train_id'] or not body.get('bindings')):
        raise ValueError('original pre-training data selection changed')
    bindings=[]
    for binding in body['bindings']:
        registered,path=_document(output,binding['record_ref'])
        if (digest(path)!=binding.get('record_sha256') or
                registered.get('status')!='reviewed_not_consumed' or
                any(registered.get(k)!=binding.get(k) for k in ('id','source','identity_field','identity')) or
                not all(registered.get('review',{}).get(k) is True for k in
                    ('allowed_training_data','loader_connection_supported','sources_supported'))):
            raise ValueError('original registered data/review identity changed')
        if ((registered.get('asset') or {}).get('profile_witness',{}).get('inspector') and
                registered.get('review',{}).get('manifest_observations_supported') is not True):
            raise ValueError('original native manifest observation review is not accepted')
        review=binding.get('consumer_review')
        if review:
            _,path=_document(output,review['record_ref'])
            if digest(path)!=review.get('record_sha256'):
                raise ValueError('original training-time consumer approval changed')
        bindings.append({**registered,**binding})
    usage=verify_consumption(output,bindings,parent['train'],training_performed=True)
    if usage.get('status')!='verified':
        raise ValueError('original completed training consumption did not reverify: '+str(usage.get('why')))
    return bindings,usage


def verified_training_data(output,run_id,measurement):
    """Reverify a completed producer's ORIGINAL data binding, not a future run.

    A caller reading a historical score must not replace its native setting key
    with a presentation alias or demand approval for today's different trainer.
    """
    output=Path(output).resolve(strict=True)
    train=measurement.get('train') or {};identity=train.get('attempt_id')
    if not re.fullmatch(r'[0-9a-f]{32}',str(identity or '')):
        raise ValueError('historical data binding requires its exact training producer')
    receipt,_=_document(output,f'research/{run_id}/attempts/{identity}/receipt.json')
    if (receipt.get('attempt_id')!=identity or receipt.get('run_id')!=run_id or
            receipt.get('stage')!='train' or receipt.get('status')!='completed' or
            receipt.get('returncode')!=0 or receipt.get('termination_reason')!='normal_exit' or
            receipt.get('settings_digest')!=object_digest(measurement.get('settings'))):
        raise ValueError('historical training settings or receipt identity changed')
    return _training_data({'output':output,'run_id':run_id,'label':measurement['label'],
                          'train_id':identity,'parent':measurement})


def verify_link(root, measured, receipt):
    """Verify an additive link; return only explicit metric-copy supplements.

    Raw settings_digest, argv, process identity, log and receipt are NOT changed.
    The caller separately checks the actual evaluation phase settings and path.
    """
    root=Path(root).resolve();output=root.parent.parent
    claim=measured.get('completed_evaluation_link')
    if not isinstance(claim,dict) or claim.get('native_replayed') is not False:
        raise ValueError('invalid completed evaluation link')
    ref=claim.get('attachment_ref')
    if not isinstance(ref,str) or not re.fullmatch(r'completed_evaluation_links/[0-9a-f]{64}\.json',ref):
        raise ValueError('unsafe completed evaluation attachment')
    link,path=_document(root,ref)
    if object_digest(link)!=path.stem or path.stem!=claim.get('attachment_sha256'):
        raise ValueError('completed evaluation attachment changed')
    c=_context(output,Path(link['repo']),root.name,link['original_label'],link['job_id'],
        source_capsule=link['source_capsule'],producer_capsule=link['producer_capsule'],
        interpreter=link['caller_interpreter'],historical=True)
    expected_refs={'original_measurement':c['parent_path'],'job_request':c['request_path'],
        'job_result':c['result_path'],'evaluation_receipt':c['receipt_path'],
        'training_receipt':root/'attempts'/c['train_id']/'receipt.json',
        'protocol':c['protocol_path']}
    if (link.get('schema_version')!=1 or link.get('label')!=measured.get('label') or
            link.get('preserved_sha256')!={k:digest(v) for k,v in expected_refs.items()} or
            receipt!=c['receipt'] or measured.get('settings')!=c['parent']['settings'] or
            measured.get('evaluation_settings')!=c['request']['settings'] or
            (measured.get('evaluate') or {}).get('attempt_id')!=c['attempt'] or
            (measured.get('train') or {}).get('attempt_id')!=c['train_id'] or
            (measured.get('train') or {}).get('reused') is not True or
            measured.get('consumed_policy_path')!=str(c['policy']) or
            measured.get('metric')!=c['spec'].as_dict() or measured.get('confirmation')):
        raise ValueError('completed evaluation linkage or original identities changed')
    if object_digest(measured.get('source_state'))!=link.get('evaluation_source_state_sha256'):
        raise ValueError('archived evaluation source state changed')
    from .experiment_bundle import artifact_identity
    frozen=measured.get('policy_artifact') or {}
    policy_path=Path(frozen.get('path') or '').resolve(strict=True)
    if not policy_path.is_relative_to(root):raise ValueError('frozen policy escaped research')
    field,identity=artifact_identity(policy_path)
    if artifact_identity(c['policy'])!=(field,identity) or frozen.get(field)!=identity:
        raise ValueError('frozen copy differs from the policy actually loaded')
    bindings,usage=_training_data(c)
    if measured.get('data_consumption')!=usage:
        raise ValueError('training-data claim differs from preserved native evidence')
    if bindings:
        proof=measured.get('data_selection') or {}
        body,path=_document(output,str(proof.get('record_ref') or ''))
        old,_=_document(output,c['parent']['data_selection']['record_ref'])
        if (proof.get('training_attempt_id')!=c['train_id'] or
                proof.get('record_sha256')!=digest(path) or object_digest(body)!=proof.get('id') or
                body!={**old,'label':measured['label']}):
            raise ValueError('linked data selection must preserve the original pre-training binding')
    elif measured.get('data_selection'):
        raise ValueError('unbound parent cannot gain a selected-data claim')
    frozen_result=measured.get('result_artifact') or {}
    result_path=Path(frozen_result.get('path') or '').resolve(strict=True)
    evidence=link.get('metric_result_artifact') or {}
    if (not result_path.is_relative_to(root) or
            digest(result_path)!=c['evidence']['sha256'] or
            frozen_result.get('content_sha256')!=c['evidence']['sha256'] or
            evidence!={**c['evidence'],'frozen_copy_ref':result_path.relative_to(root).as_posix(),
                'frozen_copy_sha256':c['evidence']['sha256']}):
        raise ValueError('frozen metric copy or native source attachment changed')
    return {**receipt,'metric_result_artifact':evidence}


def record(research, *, label, job_id):
    """Explicit Main action; reuse complete native work and keep every old result."""
    from .native_jobs import active_jobs
    from .experiment_bundle import freeze_artifact
    from .common import atomic_json,now
    from .data_versions import seal_selection
    root=research.run_root
    if active_jobs(research.output) or getattr(research,'_held',{}):
        raise ValueError('active native work or source transaction prevents completed evaluation linkage')
    session,_=_document(root,'controller_session.json')
    if (session.get('status') not in {'paused','completed','finalizing'} or session.get('action') or
            session.get('pending_action')):
        raise ValueError('completed evaluation linkage requires an idle research session')
    target=f'{label}_eval_{job_id[:12]}'
    measured_path=safe(root,f'measurements/{target}.json')
    if measured_path.exists():
        from .receipt_verifier import verify_measurement
        old=read_json(measured_path)
        if ((old.get('completed_evaluation_link') or {}).get('job_id')!=job_id or
                verify_measurement(root,target).get('status')!='consistent'):
            raise ValueError('existing linked evaluation is not the same verified result')
        return old
    c=_context(research.output,research.repo,research.run_id,label,job_id)
    bindings,usage=_training_data(c)
    policy=freeze_artifact(c['policy'],root/'experiments'/target/'policy',
        max_bytes=2*1024**3,store=root/'artifact_store')
    frozen=freeze_artifact(c['artifact'],root/'experiments'/target/('result'+c['artifact'].suffix),
        max_bytes=64*1024**2)
    frozen['content_sha256']=digest(Path(frozen['path']))
    evidence={**c['evidence'],'frozen_copy_ref':Path(frozen['path']).relative_to(root).as_posix(),
              'frozen_copy_sha256':frozen['content_sha256']}
    source=research._capture_workspace_source_state(target)
    train=deepcopy(c['parent']['train']);train['reused']=True
    native=c['native']
    result={'label':target,'ok':True,'where':'','why':'','baseline_origin':'locally_trained',
        'settings':deepcopy(c['parent']['settings']),'evaluation_settings':deepcopy(c['request']['settings']),
        'training_source_label':label,'train':train,
        'evaluate':{k:native.get(k) for k in ('attempt_id','returncode','status','ran',
            'termination_reason','requested_seed','evidence_id','evidence_ref','device','gpu_device_uuid')},
        'policy_artifact':policy,'consumed_policy_path':str(c['policy']),
        'policy_consumption':c['policy_proof'],'rollout_evidence':c['rollout'],
        'metric_lineage':c['lineage'],'metric':c['spec'].as_dict(),
        'metric_value':c['reading']['value'],'metric_utility':c['spec'].utility(c['reading']['value']),
        'metric_reading':c['reading'],'metric_artifact_evidence':c['evidence'],
        'result_artifact':frozen,'source_state':source,'data_consumption':usage,
        'restorable_policy':True,'evaluated_policy_path':str(c['policy']),
        'scored':'completed evaluation of this exact locally trained producer; no replay'}
    if bindings:
        result['data_selection']={**seal_selection(research.output,research.run_id,target,
            result['settings'],bindings),'training_attempt_id':c['train_id']}
    refs={'original_measurement':c['parent_path'],'job_request':c['request_path'],
        'job_result':c['result_path'],'evaluation_receipt':c['receipt_path'],
        'training_receipt':root/'attempts'/c['train_id']/'receipt.json','protocol':c['protocol_path']}
    link={'schema_version':1,'at':now(),'label':target,'original_label':label,'job_id':job_id,
          'repo':str(research.repo),'preserved_sha256':{k:digest(v) for k,v in refs.items()},
          'source_capsule':c['source_capsule'],'producer_capsule':c['producer_capsule'],
          'caller_interpreter':c['interpreter'],
          'evaluation_source_state_sha256':object_digest(source),
          'metric_result_artifact':evidence,'authority':'additive reuse; no native execution or adoption'}
    identity=object_digest(link);ref=f'completed_evaluation_links/{identity}.json'
    immutable_json(safe(root,ref),link)
    result['completed_evaluation_link']={'attachment_ref':ref,'attachment_sha256':identity,
        'job_id':job_id,'native_replayed':False,'original_measurement_ref':
        c['parent_path'].relative_to(research.output).as_posix()}
    if c['spec'].name=='success_rate':result['success_rate']=c['reading']['value']
    immutable_json(measured_path,result)
    from .receipt_verifier import verify_measurement
    audit=verify_measurement(root,target)
    if audit.get('status')!='consistent':
        atomic_json(measured_path,{**result,'ok':False,'where':'completed_evaluation_link',
            'metric_value':None,'metric_utility':None,'why':'independent linked-receipt verification failed',
            'link_verification':audit})
        raise ValueError('linked evaluation failed independent verification: '+str(audit.get('issues')))
    research._append_local_event({'at':now(),'event':'completed_evaluation_link',
        'original_label':label,'label':target,'job_id':job_id,'metric_value':result['metric_value'],
        'native_replayed':False,'best_unchanged':True})
    from . import selection
    selection.note(root,protocol_sha256=digest(c['protocol_path']),label=target,
        started=True,metric_value=result['metric_value'],metric_utility=result['metric_utility'],
        baseline=False,settings=c['protocol']['settings'],
        metric_kind='binary' if c['spec'].name=='success_rate' else 'continuous',
        actual_episodes=c['reading'].get('episodes_completed'),successes=c['reading'].get('successes'))
    return result
