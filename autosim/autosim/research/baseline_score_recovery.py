"""Explicit Main-owned CPU recovery of an already completed baseline evaluation.

Keep the failed reading and process receipt; independently re-read the same result.
An additive journal makes baseline/history promotion forward-resumable, not a replay.
"""
from copy import deepcopy
from pathlib import Path

from .common import atomic_json, digest, immutable_json, read_json
from .experiment_transaction import safe


def _idle(research):
    from .native_jobs import active_jobs
    from .agent_tasks import records
    root=research.run_root
    budget_path=research.output/'task_gpu_budget.json'
    active_gpu=(any(x.get('status')=='active' for x in
        (read_json(budget_path).get('leases') or {}).values()) if budget_path.is_file() else False)
    if (active_jobs(research.output) or any(r.get('status') in
            {'submitted','running','outcome_unknown'} for r in records(research.output))
            or active_gpu or any((root/'confirmation_attempts').glob('*.json')) or (root/'confirmation.json').exists()):
        raise ValueError('baseline CPU recovery requires idle development before held-out exposure')


def _context(research):
    root=research.run_root
    session=read_json(safe(root,'controller_session.json'))
    report=read_json(safe(root,'research_report.json'))
    if (session.get('run_id')!=research.run_id or report.get('run_id')!=research.run_id
            or session.get('status')!='paused' or report.get('run_status')!='paused'
            or session.get('pending_action') or len(session.get('history') or [])!=1
            or len(report.get('rounds') or [])!=1
            or session['history'][0].get('label')!='baseline'
            or report['rounds'][0].get('label')!='baseline'
            or session['history'][0].get('measured') is True
            or report['rounds'][0].get('measured') is True):
        raise ValueError('CPU baseline recovery requires unchanged paused unscored round zero')
    return session,report


def available(root):
    """Cheap advisory; execution repeats every receipt/protocol/identity check."""
    try:
        root=Path(root);pending=root/'baseline_score_recovery.json'
        if pending.is_file() and not pending.is_symlink() and read_json(pending).get('status')=='committing':
            return {'available':True,'kind':'resume_cpu_baseline_score_commit',
                    'why':'finish the journaled CPU baseline adoption; no new model review or native job'}
        m=read_json(safe(root,'measurements/baseline.json'));ev=m.get('evaluate') or {}
        r=read_json(safe(root,f"attempts/{ev['attempt_id']}/receipt.json"))
        if (m.get('ok') is False and m.get('where')=='evaluate'
                and m.get('baseline_origin')=='locally_trained' and not m.get('confirmation')
                and (m.get('metric_artifact_evidence') or {}).get('status')=='missing'
                and r.get('node_id')=='evaluate' and r.get('status')=='completed'
                and r.get('returncode')==0 and r.get('termination_reason')=='normal_exit'):
            return {'available':True,'kind':'completed_baseline_result_locator',
                'evaluation_attempt_id':ev['attempt_id'],'checkpoint_ref_required':False,
                'why':'native evaluation completed; Main may propose artifact_revision '
                    '{pattern,reason,source_refs:[{ref,quote}]} for independent same-attempt '
                    'locator review and zero-GPU scoring. Do not replay this evaluator or retrain.'}
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        pass
    return {'available':False}


def _row(original, result, backup_ref, attempt_id, run_id):
    row={**original,'measured':True,'status':'measured',
         'metric_value':result.get('metric_value'),'metric_utility':result.get('metric_utility'),
         'success_rate':result.get('success_rate'),
         'recovered_from':{'measurement_ref':f'research/{run_id}/'+backup_ref,
             'training_attempt_id':attempt_id,'training_reused':True,
             'original_failure':{k:original[k] for k in ('why_not','failure') if k in original}}}
    row.pop('failure',None);row.pop('why_not',None)
    return row


def _finish(research, state):
    from .receipt_verifier import verify_measurement
    _idle(research);root=research.run_root
    result_path=safe(root,state['recovered_ref']);result=read_json(result_path)
    if digest(result_path)!=state['recovered_sha256'] or verify_measurement(root,result['label']).get('status')!='consistent':
        raise ValueError('journaled baseline score no longer independently verifies')
    original_path=safe(root,state['original_ref']);original=read_json(original_path)
    context_path=safe(root,state['context_ref']);context=read_json(context_path)
    if digest(original_path)!=state['original_sha256'] or digest(context_path)!=state['context_sha256']:
        raise ValueError('baseline recovery original/context changed')
    claim=result.get('scoring_recovery') or {}
    if (original.get('label')!='baseline' or result.get('baseline_origin')!='locally_trained'
            or claim.get('original_measurement_ref')!=original_path.relative_to(research.output).as_posix()
            or (result.get('train') or {}).get('attempt_id')!=state['training_attempt_id']):
        raise ValueError('journal does not own this original baseline and completed producer')
    session,report=deepcopy(context['session']),deepcopy(context['report'])
    current_session=read_json(safe(root,'controller_session.json'))
    current_report=read_json(safe(root,'research_report.json'))
    expected_row=_row(session['history'][0],result,state['original_ref'],
                      state['training_attempt_id'],research.run_id)
    for current,before,rows_key,mutable in (
        (current_session,session,'history',{'history','baseline_rate','current_measurement_label','updated_at'}),
        (current_report,report,'rounds',{'rounds','best','objective'})):
        if ({k:v for k,v in current.items() if k not in mutable}!=
                {k:v for k,v in before.items() if k not in mutable} or
                current.get(rows_key) not in (before[rows_key],[expected_row])):
            raise ValueError('baseline history or frozen controller context changed during recovery')
    adopted={**result,'label':'baseline'}
    baseline=safe(root,'measurements/baseline.json')
    if digest(baseline)!=state['baseline_original_sha256'] and read_json(baseline)!=adopted:
        raise ValueError('another baseline replaced the original during score recovery')
    # The original failed record and the verified separate reading remain immutable.
    atomic_json(baseline,adopted)
    if verify_measurement(root,'baseline').get('status')!='consistent':
        raise ValueError('adopted baseline score failed independent verification')
    committed=research._commit_recovered_baseline(adopted,
        attempt_id=state['training_attempt_id'],backup_path=original_path,session=session,report=report)
    atomic_json(root/'baseline_score_recovery.json',{**state,'status':'completed',
        'native_replayed':False,'training_reused':True})
    return committed


def record(research, *, revision=None, client=None):
    from .reference_score_recovery import recover
    _idle(research);root=research.run_root;state_path=root/'baseline_score_recovery.json'
    if state_path.is_symlink():raise ValueError('unsafe baseline recovery journal')
    if state_path.exists():
        state=read_json(state_path)
        if state.get('status') in {'committing','completed'}:
            return _finish(research,state)
        raise ValueError('unknown baseline score recovery journal; preserve evidence')
    if revision is None:raise ValueError('Main must propose a bounded source-backed artifact_revision')
    session,report=_context(research)
    original=safe(root,'measurements/baseline.json');old=read_json(original)
    original_sha=digest(original)
    backup_ref='measurements/baseline_score_source_'+original_sha[:16]+'.json'
    backup=safe(root,backup_ref);immutable_json(backup,old)
    context_ref='baseline_score_contexts/'+original_sha+'.json'
    context=safe(root,context_ref);immutable_json(context,{'session':session,'report':report})
    result,recovered_ref=recover(research,original_ref=backup.relative_to(research.output).as_posix(),
        locator_revision=revision,client=client)
    if (digest(original)!=original_sha or _context(research)!=(session,report)):
        raise ValueError('baseline/controller changed during independent score recovery')
    state={'schema_version':1,'status':'committing','original_ref':backup_ref,
        'baseline_original_sha256':original_sha,
        'original_sha256':digest(backup),'context_ref':context_ref,'context_sha256':digest(context),
        'recovered_ref':Path(recovered_ref).relative_to('research/'+research.run_id).as_posix(),
        'recovered_sha256':digest(research.output/recovered_ref),
        'training_attempt_id':old['train']['attempt_id']}
    atomic_json(state_path,state)
    return _finish(research,state)
