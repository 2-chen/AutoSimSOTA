"""Bind paid native training to one audited candidate, never replace a baseline."""
from pathlib import Path

from .common import digest, object_digest, read_json
from .ideas import CLEARED


def allocation(research, idea_label: str, training_settings: dict | None = None) -> dict:
    root = research.run_root
    session = read_json(root/'controller_session.json')
    if (session.get('status') != 'paused' or session.get('action') or session.get('pending_action')
            or session.get('next_round', 0) > session.get('rounds', 0)
            or any((root/'confirmation_attempts').glob('*.json'))):
        raise ValueError('candidate job requires an idle allocated development round before held-out exposure')
    if session.get('input_identity') != research._research_session_identity(
            rounds=session['rounds'],settings=session['base_settings']):
        raise ValueError('candidate job inputs differ from the existing research session')
    idea = research.library.get(idea_label)
    if idea is None or idea.status != CLEARED:
        raise ValueError('select one exact currently cleared idea')
    if idea.granularity not in {'param', 'algo'}:
        raise ValueError('detached code candidates need a prepared patch transaction; use the synchronous audited path')
    if research.execution_graph:
        raise ValueError('graph candidates require graph-node artifact binding; use the existing graph experiment path')
    own = idea.change or {}
    requested = own.get('collection') or {}
    if requested and requested.get('enabled', True):
        raise ValueError('collect and validate the data version before detached training; do not silently skip collection')
    from .candidate_settings import proposed
    settings = proposed(research,session['base_settings'],idea)
    violation = research._comparison_protocol_violation(settings, target='evaluate')
    if violation:
        raise ValueError('candidate changes the frozen evaluation protocol')
    if training_settings is not None:
        from .candidate_settings import apply
        settings=apply(research,settings,training_settings)
    from .data_versions import resolve
    settings,data_versions=resolve(research.output,research.repo,settings)
    from .training_work import missing_formal_work
    problem=missing_formal_work(research.sources.get('train',''),
        research._inputs('train',settings=settings))
    if problem:
        raise ValueError(problem)
    return {'schema_version':1, 'run_id':research.run_id, 'idea_label':idea_label,
            'idea_sha256':object_digest(idea.as_dict()), 'round':session['next_round'],
            'baseline_sha256':digest(root/'measurements/baseline.json'),
            'protocol_sha256':digest(root/'comparison_protocol.json'),
            'history_sha256':object_digest(session['history']), 'settings':settings,
            'training_settings':training_settings,
            'data_versions':data_versions}


def adopt(research, job_id: str, idea_label: str) -> dict:
    from .native_jobs import status, source_identity, _path
    job = status(research.output, job_id)
    request = read_json(_path(research.output, job_id)/'request.json')
    expected = allocation(research, idea_label,
        (request.get('candidate_binding') or {}).get('training_settings'))
    if (job.get('status') != 'completed' or request.get('candidate_binding') != expected
            or request.get('repo') != str(research.repo) or request.get('run_id') != research.run_id
            or request.get('stage') != 'train' or request.get('settings') != expected['settings']
            or not request.get('source_identity')
            or request['source_identity'] != source_identity(research.output, research.repo)):
        raise ValueError('completed job does not belong to this exact candidate/source/protocol/round')
    result = (job.get('result') or {}).get('stage_result') or {}
    attempt = result.get('attempt_id') or ''
    if not research._verified_training_receipt(attempt, expected['settings'], audit_progress=True):
        raise ValueError('candidate training receipt or observed updates are unverified')
    return {**expected, 'job_id':job_id, 'training_attempt_id':attempt}
