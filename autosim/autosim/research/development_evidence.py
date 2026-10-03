"""Bounded aggregate DEVELOPMENT evidence, not episode rows or held-out scores."""
import math
from collections import Counter
from pathlib import Path

from .common import digest, read_json


def summaries(root: Path) -> list[dict]:
    root = Path(root).resolve()
    directory = root/'measurements'
    if directory.is_symlink():
        return []
    paths = [directory/'baseline.json']
    recent = []
    for path in directory.glob('round_*.json'):
        try:
            if not path.is_symlink() and path.is_file():
                recent.append((path.stat().st_mtime, path))
        except OSError:
            continue
    paths += [path for _, path in sorted(recent)[-5:]]
    result = []
    for path in paths:
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 8*1024*1024:
                continue
            row = read_json(path)
            if not isinstance(row, dict):
                continue
            value = row.get('metric_value')
            if (row.get('ok') is not True or not isinstance(value, (int, float))
                    or isinstance(value, bool) or not math.isfinite(value)):
                continue
            item = {'label':row.get('label'), 'metric_value':value,
                    'samples':(row.get('metric_reading') or {}).get('samples'),
                    'measurement_ref':str(path.relative_to(root)),
                    'authority':'formal development measurement, not command verification',
                    'identity':{key:(row.get(key) or {}).get('status') for key in
                        ['policy_consumption','rollout_evidence','metric_lineage']},
                    'training_settings':{k:v for k,v in (row.get('settings')or{}).items()
                        if isinstance(v,(bool,int,float))}}
            artifact = row.get('result_artifact') or {}
            raw = Path(str(artifact.get('path') or '__missing__'))
            raw = raw if raw.is_absolute() else root/raw
            # Directory/manifest identity is not the file's byte hash.
            expected = artifact.get('content_sha256') or artifact.get('sha256')
            if (expected and raw.is_file() and not raw.is_symlink()
                    and not (root/'experiments').is_symlink()
                    and raw.resolve().is_relative_to(root/'experiments')
                    and raw.stat().st_size <= 8*1024*1024 and digest(raw)==expected):
                document = read_json(raw)
                rows = document.get('episodes') if isinstance(document,dict) else None
                if isinstance(rows,list) and rows and len(rows)<=100000:
                    counts = Counter(str(x['failure_stage'])[:120] for x in rows
                        if isinstance(x,dict) and isinstance(x.get('failure_stage'),str))
                    if counts:
                        item['failure_stage_counts'] = dict(counts.most_common(16))
                    fields = sorted({k for x in rows if isinstance(x,dict) for k,v in x.items()
                        if isinstance(v,(int,float)) and not isinstance(v,bool)
                        and not k.endswith(('_id','_index','_seed'))})[:12]
                    numeric = {}
                    for key in fields:
                        values = [x[key] for x in rows if isinstance(x,dict)
                            and isinstance(x.get(key),(int,float)) and not isinstance(x[key],bool)
                            and math.isfinite(x[key])]
                        if values:
                            numeric[key] = {'count':len(values),'min':min(values),
                                'max':max(values),'mean':sum(values)/len(values)}
                    item['diagnostic_numeric_aggregates'] = numeric
                    item['diagnostic_artifact_verified'] = True
            result.append(item)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return result
