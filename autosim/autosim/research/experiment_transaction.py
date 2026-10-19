"""One evidence index for an experiment, not a second execution authority.

The journal indexes existing receipts; it never launches/replays work, certifies
a metric, changes an incumbent or resets a budget. Interrupted side effects remain
owned by the existing process/session recovery machinery.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import re
import uuid

from .common import atomic_json, digest, now, object_digest, read_json


def safe(root: Path, relative: str) -> Path:
    root = Path(root).resolve(strict=True)
    name = Path(relative)
    path = root/name
    if (name.is_absolute() or '..' in name.parts or
            any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
            or not path.resolve().is_relative_to(root)):
        raise ValueError('unsafe experiment evidence index')
    return path


def _save(research, journal):
    path = safe(research.run_root, 'experiment_transactions/'+journal['id']+'.json')
    body = {**deepcopy(journal), 'updated_at': now()}
    atomic_json(path, {**body, 'sha256': object_digest(body)})


def _note_failure(research, exc):
    # Presentation/index failure must not erase a native measurement or trigger
    # its replay. The existing event log is the fallback, not a fabricated score.
    try:
        research._note('experiment_index_unwritten', [{'why': type(exc).__name__}])
    except (OSError, ValueError, TypeError):
        pass  # Never discard an already sealed native result for optional indexing.


def stage(research, name: str, *, result=None):
    journal = getattr(research, '_experiment_journal', None)
    if journal is None:
        return
    journal['phase'] = name if result is None else name+'_returned'
    if result is not None:
        journal['stages'].append({
            'stage': name, **{k: result.get(k) for k in (
                'attempt_id', 'status', 'returncode', 'termination_reason',
                'evidence_id', 'evidence_ref', 'ran', 'reused')},
            'recorded_at': now()})
        journal['stages'] = journal['stages'][-64:]
    try:
        _save(research, journal)
    except (OSError, ValueError, TypeError) as exc:
        _note_failure(research, exc)


@contextmanager
def track(research, *, label: str, settings: dict, confirmation=False,
          reference=False, reused_attempt='', data_selection=None):
    if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', label):
        raise ValueError('unsafe experiment label')
    previous = getattr(research, '_experiment_journal', None)
    journal = {'schema_version': 1, 'id': uuid.uuid4().hex, 'run_id': research.run_id,
               'label': label, 'settings_sha256': object_digest(settings),
               'scope': 'confirmation' if confirmation else 'reference' if reference else 'development',
               'phase': 'planned', 'status': 'running', 'stages': [],
               'reused_training_attempt_id': reused_attempt, 'started_at': now(),
               'authority': 'evidence index only; native receipts/session own execution'}
    if data_selection is not None:
        journal['data_selection'] = deepcopy(data_selection)
    research._experiment_journal = journal
    try:
        _save(research, journal)
    except (OSError, ValueError, TypeError) as exc:
        _note_failure(research, exc)
    try:
        yield journal
    except BaseException:
        journal.update(status='interrupted', phase='reconcile_existing_ownership')
        try:
            _save(research, journal)
        except (OSError, ValueError, TypeError) as exc:
            _note_failure(research, exc)
        raise
    finally:
        research._experiment_journal = previous


def finish(research, journal, result):
    # A returned refusal need not have written a measurement; do not invent one.
    journal.update(status='returned', phase='measurement_returned',
                   result_ok=result.get('ok') is True,
                   failure_where=result.get('where'),
                   finished_at=now())
    try:
        path = safe(research.run_root, 'measurements/'+journal['label']+'.json')
        if path.is_file() and path.stat().st_size <= 8*1024**2:
            stored = read_json(path)
            if stored == result:
                journal.update(measurement_ref=path.relative_to(research.output).as_posix(),
                               measurement_sha256=digest(path))
        _save(research, journal)
    except (OSError, ValueError, TypeError) as exc:
        _note_failure(research, exc)


def recovery_edge(measurement: dict) -> dict:
    """Explain the broken edge, never grant permission to replay a native job."""
    if measurement.get('ok') is True:
        return {'edge': 'measurement_to_commit', 'owner': 'scheduler',
                'guidance': 'verify existing measurement and finish adoption/report; do not rerun'}
    where = measurement.get('where') or 'unknown'
    if where in {'policy_artifact', 'policy_archive', 'source_identity'}:
        return {'edge': 'training_to_policy', 'owner': 'fix',
                'guidance': 'inspect the exact completed attempt/bytes; preserve paid training before retry'}
    if where == 'training_data':
        return {'edge': 'data_to_training', 'owner': 'data',
                'guidance': 'trace actual loader consumption; presence/argv is not a batch witness'}
    if where in {'evaluate', 'metric_artifact', 'metric_lineage'}:
        return {'edge': 'policy_to_score', 'owner': 'fix',
                'guidance': 'inspect sealed evaluation evidence, retain policy and frozen protocol; reconcile before replay'}
    return {'edge': str(where), 'owner': 'scheduler',
            'guidance': 'resolve exact evidence and process ownership before deciding a bounded repair'}


def view(output: Path, run_id: str, *, limit=20) -> dict:
    output = Path(output).resolve(strict=True)
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', run_id) or run_id in {'.','..'}:
        raise ValueError('unsafe transaction run id')
    root = safe(output, 'research/'+run_id)
    if not root.is_dir():
        return {'schema_version': 1, 'transactions': []}
    rows = []
    problems = []
    for raw in (root/'experiment_transactions').glob('*.json'):
        try:
            if not re.fullmatch(r'[0-9a-f]{32}\.json', raw.name):
                raise ValueError('unsafe transaction id')
            path = safe(root, raw.relative_to(root).as_posix())
            if path.stat().st_size > 256*1024:
                raise ValueError('oversized transaction')
            row = read_json(path)
            if (not isinstance(row, dict) or row.get('id')+'.json' != path.name
                    or row.get('run_id') != run_id or row.get('schema_version') != 1
                    or row.get('sha256') != object_digest({k:v for k,v in row.items() if k!='sha256'})):
                raise ValueError('transaction index changed')
            if row.get('scope') == 'confirmation':
                continue  # Development readers cannot see held-out IDs/paths through an index.
            measurement = {}
            ref = row.get('measurement_ref')
            if ref:
                measured = safe(output, ref)
                if (measured != root/'measurements'/(row['label']+'.json') or
                        measured.stat().st_size > 8*1024**2 or digest(measured) != row.get('measurement_sha256')):
                    raise ValueError('transaction measurement changed')
                measurement = read_json(measured)
            summary = {k: row.get(k) for k in ('id','label','scope','status','phase',
                'started_at','updated_at','measurement_ref','reused_training_attempt_id','stages')}
            summary['recovery'] = recovery_edge(measurement) if measurement else {
                'edge': row['phase'], 'owner': 'scheduler',
                'guidance': 'inspect existing session/jobs/attempt receipts; index alone cannot prove completion or safe replay'}
            rows.append(summary)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            problems.append({'ref': raw.relative_to(output).as_posix(), 'error': type(exc).__name__})
    rows.sort(key=lambda row: (row.get('updated_at') or '', row['id']))
    return {'schema_version': 1, 'transactions': rows[-limit:],
            'index_errors': problems[:10], 'authority': 'evidence index, not score or replay authorization'}
