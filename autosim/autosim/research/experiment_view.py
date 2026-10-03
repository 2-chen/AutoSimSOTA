"""Read-only experiment lifecycle projection, not another decision owner."""
from pathlib import Path

from .common import read_json


def _read(output: Path, path: Path) -> dict:
    if (path.is_symlink() or not path.resolve().is_relative_to(output)
            or path.stat().st_size>2*1024**2):
        raise ValueError('unsafe experiment projection source')
    row=read_json(path)
    if not isinstance(row,dict):
        raise ValueError('experiment source must be an object')
    return row


def view(output: Path, run_id: str) -> dict:
    output = Path(output).resolve()
    root = output/'research'/run_id
    jobs = []
    for path in sorted((output/'native_jobs').glob('*/request.json'))[-100:]:
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(output):
                continue
            request = _read(output,path)
            binding = request.get('candidate_binding')
            if not binding or request.get('run_id') != run_id:
                continue
            result_path = path.with_name('result.json')
            result = _read(output,result_path) if result_path.is_file() else {}
            label = f'round_{binding["round"]}'
            measured_path = root/'measurements'/(label+'.json')
            measured = _read(output,measured_path) if measured_path.is_file() else {}
            attempt = ((result.get('stage_result') or {}).get('attempt_id') or
                       ((result.get('result') or {}).get('stage_result') or {}).get('attempt_id'))
            linked = bool(attempt and (measured.get('train') or {}).get('attempt_id') == attempt)
            verified=False
            if linked and measured.get('ok'):
                from .receipt_verifier import verify_measurement
                verified=verify_measurement(root,label).get('status') == 'consistent'
            jobs.append({'experiment_id':f'{run_id}/{label}', 'job_id':request['job_id'],
                'idea_label':binding['idea_label'], 'stage':'train',
                'status':'scored' if linked and verified else
                         'unscored' if linked else result.get('status','pending'),
                'measurement_ref':str(measured_path.relative_to(output)) if linked else None,
                'metric_value':measured.get('metric_value') if verified else None,
                'adoption': 'known' if linked else 'requires_explicit_adoption',
                'request_ref':str(path.relative_to(output))})
        except (OSError,ValueError,TypeError,KeyError,AttributeError):
            continue
    return {'schema_version':1,'authority':'receipt projection; Scheduler alone decides actions',
            'experiments':jobs,'reference_scope':'published reference is separate from candidate gain'}
