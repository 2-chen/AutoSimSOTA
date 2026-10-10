"""Read-only experiment lifecycle projection, not another decision owner."""
from pathlib import Path

from .common import read_json, object_digest
from .experiment_transaction import safe, view as transaction_view, recovery_edge


def _read(output: Path, path: Path) -> dict:
    if (path.is_symlink() or not path.resolve().is_relative_to(output)
            or path.stat().st_size>2*1024**2):
        raise ValueError('unsafe experiment projection source')
    row=read_json(path)
    if not isinstance(row,dict):
        raise ValueError('experiment source must be an object')
    return row


def view(output: Path, run_id: str) -> dict:
    import re
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', run_id) or run_id in {'.','..'}:
        raise ValueError('unsafe experiment run id')
    if not Path(output).exists():
        return {'schema_version':1, 'experiments':[], 'candidates':[], 'transactions':[],
                'authority':'receipt projection; no run evidence yet'}
    output = Path(output).resolve(strict=True)
    root = safe(output, 'research/'+run_id)
    candidates = []
    history = {}
    try:
        session = _read(output, safe(output, 'research/'+run_id+'/controller_session.json'))
        if session.get('run_id') == run_id:
            history = {r.get('label'):r for r in session.get('history',[]) if isinstance(r,dict)}
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        pass
    measurements = {}
    errors = []
    # Displayed candidates are a bounded recommendation tree, not a new search
    # owner. Protocol/receipt verification remains independent of Agent prose.
    paths = sorted((root/'measurements').glob('*.json'))
    for raw in paths[-100:]:
        try:
            path = safe(output, raw.relative_to(output).as_posix())
            row = _read(output, path)
            if row.get('label') != path.stem:
                raise ValueError('measurement label differs from filename')
            if row.get('confirmation') or path.stem == 'confirmation' or path.stem.startswith('confirmation_'):
                continue  # No held-out labels, attempts or results in Main/Recorder development context.
            measurements[path.stem] = row
            consistent = False
            if row.get('ok') is True:
                from .receipt_verifier import verify_measurement
                consistent = verify_measurement(root, path.stem).get('status') == 'consistent'
            candidates.append({'label': path.stem,
                'scope': 'confirmation' if row.get('confirmation') else
                         'reference' if row.get('baseline_origin') == 'released_reference' else 'development',
                'status': 'scored' if consistent else 'unverified' if row.get('ok') else 'unscored',
                'metric_value': row.get('metric_value') if consistent else None,
                'metric_name': (row.get('metric') or {}).get('name'),
                'training_attempt_id': (row.get('train') or {}).get('attempt_id'),
                'evaluation_attempt_id': (row.get('evaluate') or {}).get('attempt_id'),
                'data_consumption': (row.get('data_consumption') or {}).get('status', 'unknown'),
                'settings_sha256':object_digest(row.get('settings') or {}),
                'data_versions':(row.get('data_consumption') or {}).get('versions', []),
                'policy_identity':(row.get('policy_artifact') or {}).get('content_sha256') or
                                  (row.get('policy_artifact') or {}).get('sha256'),
                'compared_incumbent':(history.get(path.stem) or {}).get('comparison_reference'),
                'idea':(history.get(path.stem) or {}).get('idea'),
                'varied':(history.get(path.stem) or {}).get('varied') or {},
                'source_identity': (row.get('source_state') or {}).get('identity_sha256'),
                'measurement_ref': path.relative_to(output).as_posix(),
                'recovery': recovery_edge(row)})
        except (OSError,ValueError,TypeError,KeyError,AttributeError):
            errors.append(raw.relative_to(output).as_posix())
    jobs = []
    for path in sorted((output/'native_jobs').glob('*/request.json'))[-100:]:
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(output):
                continue
            path = safe(output, path.relative_to(output).as_posix())
            request = _read(output,path)
            binding = request.get('candidate_binding')
            if request.get('run_id') != run_id or request.get('stage') != 'train':
                continue
            if path.parent.name != request.get('job_id'):
                raise ValueError('job id differs from its path')
            result_path = path.with_name('result.json')
            if result_path.is_symlink():
                raise ValueError('unsafe job result')
            result = _read(output,safe(output,result_path.relative_to(output).as_posix())) if result_path.is_file() else {}
            if result and result.get('job_id') != request.get('job_id'):
                raise ValueError('job result identity differs')
            attempt = ((result.get('stage_result') or {}).get('attempt_id') or
                       ((result.get('result') or {}).get('stage_result') or {}).get('attempt_id'))
            linked_labels = [name for name, value in measurements.items()
                if attempt and (value.get('train') or {}).get('attempt_id') == attempt]
            label = (f'round_{binding["round"]}' if binding else
                     linked_labels[0] if len(linked_labels) == 1 else 'baseline')
            measured_path = root/'measurements'/(label+'.json')
            measured = measurements.get(label, {})
            linked = bool(attempt and (measured.get('train') or {}).get('attempt_id') == attempt)
            verified = linked and any(c['label']==label and c['status']=='scored' for c in candidates)
            jobs.append({'experiment_id':f'{run_id}/{label}', 'job_id':request['job_id'],
                'idea_label':binding['idea_label'] if binding else 'baseline / unallocated training', 'stage':'train',
                'status':'scored' if linked and verified else
                         'unscored' if linked else result.get('status','pending'),
                'measurement_ref':str(measured_path.relative_to(output)) if linked else None,
                'metric_value':measured.get('metric_value') if verified else None,
                'adoption': 'known' if linked else 'requires_explicit_adoption',
                'native_status':result.get('status','pending'),
                'training_attempt_id':attempt,
                'next_operation': None if linked and verified else
                    'inspect_native_job' if result.get('status') in {'completed','failed','cancelled'} else 'wait_for_jobs',
                'guidance': 'retain exact paid training; inspect then explicitly adopt under the frozen protocol',
                'request_ref':str(path.relative_to(output))})
        except (OSError,ValueError,TypeError,KeyError,AttributeError):
            continue
    transactions = transaction_view(output, run_id)
    return {'schema_version':1,'authority':'receipt projection; Scheduler alone decides actions',
            'experiments':jobs, 'candidates':candidates[-30:],
            'transactions':[r for r in transactions['transactions'] if r.get('scope')!='confirmation'],
            'index_errors':errors[:10]+transactions.get('index_errors', []),
            'metrics': {'valid_scored_development': sum(c['status']=='scored' and c['scope']=='development' for c in candidates),
                'completed_training_pending_score':sum(j['native_status']=='completed' and j['status']!='scored' for j in jobs),
                'completed_training_jobs':sum(j['native_status']=='completed' for j in jobs),
                'completed_training_with_verified_score':sum(j['native_status']=='completed' and j['status']=='scored' for j in jobs),
                'counts_scope':'bounded recent evidence, not whole-run accounting',
                'measurement_scan_truncated':len(paths)>100},
            'reference_scope':'published reference is separate from candidate gain'}
