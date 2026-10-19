"""Durable, evidence-bound baseline triage; Agent reasoning is not a new score.

No benchmark recipe or minimum number of updates is prescribed. A Main Agent may
continue a deliberately budgeted local study after inspecting its assumptions, but
cannot silently treat missing consumption/reproduction evidence as verified.
"""
import re
from pathlib import Path

from .common import atomic_json, digest, object_digest, read_json, now


def _read(root, path):
    if (path.is_symlink() or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in path.parents if p.is_relative_to(root))
            or path.stat().st_size > 2*1024**2):
        raise ValueError('unsafe baseline diagnosis input')
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError('baseline diagnosis input must be an object')
    return value


def view(output, run_id):
    root = Path(output).resolve()
    # The run id is an executor-owned namespace, not an Agent-provided file path.
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('invalid baseline diagnosis namespace')
    from .baseline_reference import view as reference_view
    reference = reference_view(root)
    path = root/'research'/run_id/'measurements/baseline.json'
    subject = {'run_id':run_id, 'problems':[], 'evidence_ids':[], 'evidence_refs':[]}
    try:
        measured = _read(root, path)
        subject.update(measurement_ref=path.relative_to(root).as_posix(),
                       measurement_sha256=digest(path))
        subject['evidence_refs'].append(subject['measurement_ref'])
        consumption = measured.get('data_consumption') or {}
        trained = measured.get('train') or {}
        training_ran = bool(trained.get('ran') or (
            trained.get('ran') is not False and trained.get('attempt_id')
            and trained.get('returncode') == 0))
        if training_ran and consumption.get('status') != 'verified':
            subject['problems'].append('training_population_or_batch_consumption_unverified')
        metric = measured.get('metric') or {}
        reading = measured.get('metric_reading') or {}
        floor = metric.get('minimum')
        if metric.get('name') == 'success_rate' and floor is None:
            floor = 0  # Defined canonical metric semantics, not a benchmark name.
        values = reading.get('episode_values') or []
        at_floor = (isinstance(values,list) and bool(values)
                    and len(values) == reading.get('samples')
                    and all(isinstance(v,(int,float,bool)) and v == floor for v in values))
        if (measured.get('ok') is True and floor is not None
                and metric.get('direction','maximize') == 'maximize'
                and measured.get('metric_value') == floor
                and reading.get('samples',0) > 0
                and (at_floor or metric.get('name') == 'success_rate')):
            subject['problems'].append('all_development_episodes_at_metric_floor')
        for stage in ('train','evaluate'):
            row = measured.get(stage) or {}
            identity = row.get('evidence_id') or row.get('attempt_id')
            if re.fullmatch(r'[0-9a-f]{32}', str(identity or '')):
                subject['evidence_ids'].append(identity)
    except FileNotFoundError:
        pass
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        subject['problems'].append('baseline_evidence_unreadable')
    if reference.get('status') in {'performance_mismatch','evaluation_failed'}:
        subject['problems'].append('released_reference_'+reference['status'])
        subject['reference'] = {k:reference.get(k) for k in
            ('id','status','record_ref','measurement_ref','expected_metric','measured_metric')}
        try:
            ref_path = root/reference['measurement_ref']
            measurement = _read(root, ref_path)
            subject['reference']['measurement_sha256'] = digest(ref_path)
            subject['evidence_refs'].extend([reference['record_ref'],reference['measurement_ref']])
            identity = ((measurement.get('evaluate') or {}).get('evidence_id') or
                        measurement.get('evidence_id'))
            if re.fullmatch(r'[0-9a-f]{32}', str(identity or '')):
                subject['evidence_ids'].append(identity)
        except (OSError,ValueError,TypeError,KeyError):
            subject['problems'].append('reference_evidence_unreadable')
    if subject['problems'] and not subject['evidence_ids']:
        subject['problems'].append('sealed_failure_evidence_missing')
    subject['evidence_ids'] = list(dict.fromkeys(subject['evidence_ids']))
    subject['evidence_refs'].extend(f'evidence/{identity}.json' for identity in subject['evidence_ids'])
    identity = object_digest(subject)
    result = {'status':'required' if subject['problems'] else 'not_required',
        'diagnosis_recordable':bool(subject['evidence_ids']),
        'subject_sha256':identity, 'subject':subject,
        'authority':'triage obligation, not an assertion that a valid zero score is an executor failure',
        'next_operation':'research_task', 'repair_owner':'main_agent',
        'suggested_task':{'role':'objective','mode':'inspect',
            'task':'Inspect the supplied native evidence IDs and frozen policy/recipe. Distinguish '
                   'undertraining from wrong loader/population, normalization/action/control mapping, '
                   'and evaluation/reset mismatch. Ask Data to trace actual batches if needed. '
                   'Compare the same policy family with a pinned release; absence of resources is '
                   'a boundary, not permission to label short verification work an official baseline.',
            'expected_result':'Evidence-backed four-part diagnosis (training/data/policy/evaluation), '
                              'explicit unknowns, and a concrete next operation; preserve old scores.',
            'evidence_refs':subject['evidence_refs']}}
    if subject['problems'] and not subject['evidence_ids']:
        result['evidence_gap'] = ('Historical failure has no sealed native/operation evidence ID. '
            'Do not fabricate an ID or repeat record_baseline_diagnosis. Inspect original ownership '
            'and receipts; if proven idle, explicitly request bounded reference revalidation '
            'under the current harness to seal the current cause. Frozen protocol and old measurements stay unchanged.')
    path = root/'research'/run_id/'baseline_diagnoses'/(identity+'.json')
    try:
        record = _read(root, path)
        if record.get('subject') != subject or record.get('subject_sha256') != identity:
            raise ValueError('diagnosis subject changed')
        unsigned = {k:v for k,v in record.items() if k != 'record_sha256'}
        if record.get('record_sha256') != object_digest(unsigned):
            raise ValueError('diagnosis record changed')
        result.update(diagnosis_ref=path.relative_to(root).as_posix(),
                      decision=record['diagnosis']['decision'],
                      next_action=record['diagnosis']['next_action'])
        if result['status'] == 'required' and record['diagnosis']['decision'] == 'proceed_local':
            result['status'] = 'triaged_local_only'
    except FileNotFoundError:
        pass
    except (OSError,ValueError,TypeError,KeyError):
        result['status'] = 'required'
        result['record_problem'] = 'diagnosis record requires revalidation'
    return result


def record(output, repo, run_id, diagnosis):
    from .operation_contracts import BASELINE_DIAGNOSIS, validate
    from .baseline_reference import source_evidence
    from .evidence_store import read_attempt_evidence
    validate(BASELINE_DIAGNOSIS, diagnosis)
    current = view(output, run_id)
    if current['status'] == 'not_required' or diagnosis['subject_sha256'] != current['subject_sha256']:
        raise ValueError('choose the current diagnosis subject; historical measurements are not rewritten')
    expected = set(current['subject']['evidence_ids'])
    if not expected or not expected <= set(diagnosis['evidence_ids']):
        raise ValueError('diagnosis must inspect all offered native baseline/reference evidence IDs')
    for identity in diagnosis['evidence_ids']:
        if identity not in expected:
            raise ValueError('unrelated evidence cannot resolve baseline triage')
        read_attempt_evidence(Path(output), identity)
    sources = source_evidence(Path(repo), diagnosis['source_refs'], output=Path(output))
    held = {**diagnosis, 'source_refs':sources}
    row = {'schema_version':1,'at':now(), 'subject':current['subject'],
           'subject_sha256':current['subject_sha256'],'diagnosis':held,
           'authority':'source-cited Agent triage, not verified batch consumption or official reproduction'}
    row['record_sha256'] = object_digest(row)
    path = Path(output)/'research'/run_id/'baseline_diagnoses'/(current['subject_sha256']+'.json')
    # Investigations can be extended; scores/receipts are never modified. Keep each
    # revision so changing a conclusion is visible rather than erasing the report.
    from .common import immutable_json
    if any(p.is_symlink() for p in (path, path.parent, path.parent.parent)):
        raise ValueError('unsafe baseline diagnosis destination')
    immutable_json(path.parent/'history'/(row['record_sha256']+'.json'), row)
    atomic_json(path, row)
    return view(output, run_id)
