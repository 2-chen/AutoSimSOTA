"""Observed orchestration cost, never a reason to falsify scientific checks."""
from pathlib import Path
from .common import read_json


def view(output: Path) -> dict:
    path=output/'scheduling_metrics.json'
    if not path.is_file() or path.is_symlink() or path.stat().st_size>4*1024**2:
        return {'status':'not_recorded'}
    try:
        doc=read_json(path)
        aggregates=doc.get('aggregates') or {}
        selected={key:aggregates[key] for key in (
            'model_turn','formal_measurement','controller_action','recorder','resource_wait')
            if key in aggregates}
        recent=doc.get('events') or []
        failures=sum(row.get('kind')=='controller_action' and row.get('status') in
            {'not attempted','raised','no runnable command'} for row in recent)
        return {'status':'observed','totals':selected,'recent_event_count':len(recent),
                'recent_rejected_or_failed_actions':failures,'source_ref':'scheduling_metrics.json',
                'limitation':'durations may overlap; not wall/GPU/paid cost, '
                    'and rejected actions are not all native failures or duplicate work',
                'guidance':'reuse paid evidence; avoid repeated unchanged investigations. '
                    'Screening is advisory; low-fidelity scores cannot replace formal evaluation.'}
    except (OSError,ValueError,TypeError,AttributeError,KeyError):
        return {'status':'unreadable'}
