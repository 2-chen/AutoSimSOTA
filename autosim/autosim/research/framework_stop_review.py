"""Independent bounded review of a Main claim that the framework cannot proceed."""
import json
import re

from .common import atomic_json, object_digest, read_json, sanitize_model_payload


def review(controller, choice):
    arguments=choice.get('arguments') or {}
    if (choice.get('do')!='stop' or arguments.get('stop_scope') not in {'framework','pause'} or
            choice.get('decision_failure') or not getattr(controller.client,'supports_main_agent',False)):
        return choice
    # Operator pauses do not enter this model-choice function. A model cannot turn
    # an unsupported boundary into an unchecked pause by changing its scope label.
    if controller.budget and controller.budget.remaining()<=0:
        return choice
    facts=controller.state()
    action=controller.last_action or {}
    identity=action.get('evidence_id') or ''
    evidence=None
    if re.fullmatch(r'[0-9a-f]{32}',str(identity)):
        try:
            from .evidence_store import read_attempt_evidence
            evidence=read_attempt_evidence(controller.output,identity,limit=4000)
        except (OSError,ValueError,TypeError):
            pass
    inputs={'reason':choice.get('why'),'last_action':action,
            'available':facts.get('available',[]),'failure_evidence':evidence,
            'native_jobs':facts.get('native_jobs',[]),
            'controller_runtime':facts.get('controller_runtime'),
            'to_measure':facts.get('to_measure')}
    fingerprint=object_digest(inputs)
    directory=controller.output/'framework_stop_reviews'
    path=directory/(fingerprint+'.json')
    approved=False
    why='No current sealed framework failure proves that available recovery paths are exhausted.'
    cached=False
    try:
        if directory.is_symlink() or path.is_symlink():
            raise ValueError('unsafe framework stop review')
        if path.is_file():
            cached=True
            answer=read_json(path)['answer']
        else:
            from .agent_client import role_scope
            from .execution_derive import _object
            packet=sanitize_model_payload(inputs,local_roots=(controller.repo,controller.output))
            with role_scope(controller.client,'monitor'):
                text,_=controller.client.chat_with_metadata(
                    'Independently review a proposed AGENT pause/framework stop, read only. No edits or native '
                    'execution. Temporary GPU/job ownership, malformed arguments, missing source '
                    'quotes and unfinished producer/data handoffs are not proof of permanent '
                    'infrastructure failure. Inspect the current sealed failure and available '
                    'operations and current admission requirements. Historical attempt totals do not '
                    'prove a current fault. Approve a bounded pause when current evidence establishes '
                    'no safe in-scope recovery, not merely an unsupported permanent claim. '
                    'Insufficient evidence means abstain with a concrete next safe investigation. Return JSON: approved boolean, '
                    'why, evidence_id and an exact quote from failure_evidence.text. '
                    'Do not invent IDs or assume a previous failure is current.',
                    json.dumps(packet,ensure_ascii=False),max_tokens=1000,
                    timeout=max(1,min(float(getattr(controller.client,'timeout',180)),
                                      controller.budget.remaining() if controller.budget else 180)),
                    read_only=True,include_research_context=False,auto_skills=False)
            answer=_object(text)
            # Save a bounded exact decision; never let output prose become executable code.
            answer={k:answer.get(k) for k in ('approved','why','evidence_id','quote')}
            atomic_json(path,{'input_digest':fingerprint,'answer':answer})
        runtime = facts.get('controller_runtime') or {}
        current = not runtime or any(
            row.get('evidence_id') == identity and row.get('validity') == 'observed_in_current_runtime'
            for row in runtime.get('previous_action_failures') or [])
        approved=(answer.get('approved') is True and evidence is not None and current and
                  answer.get('evidence_id')==identity and
                  isinstance(answer.get('quote'),str) and bool(answer['quote'].strip()) and
                  answer['quote'] in evidence['text'] and action.get('retryable') is not True)
        why=str(answer.get('why') or why)[:600]
    except Exception as exc:
        from .agent_client import run_model_budget_exhausted
        if run_model_budget_exhausted(exc):
            return {**choice,'why':'model budget exhausted during framework stop review',
                    'resource_boundary_review_budget_exhausted':True}
        why=f'Framework stop review unavailable: {type(exc).__name__}: {exc}'[:600]
    if approved:
        return {**choice,'framework_stop_review_ref':path.relative_to(controller.output).as_posix()}
    correction_scope=object_digest({'available':inputs['available'],
        'evidence_id':identity,'native_jobs':inputs['native_jobs'],
        'controller_runtime':inputs['controller_runtime'],'to_measure':inputs['to_measure']})
    repeated=getattr(controller,'_framework_stop_correction_scope',None)==correction_scope or cached
    controller._framework_stop_correction_scope=correction_scope
    # Existing decision recovery gives Scheduler a correction turn. If unavailable it
    # pauses with an unresolved claim, never reports an unproven permanent boundary.
    return {**choice,'arguments':{**arguments,'stop_scope':'pause'},'why':why,
        **({} if repeated else {'decision_failure':{
            'kind':'framework_stop_unproven','attempts':1,'reason':why}}),
        'framework_stop_review_ref':path.relative_to(controller.output).as_posix() if path.exists() else None}
