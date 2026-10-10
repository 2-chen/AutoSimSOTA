"""Bounded aggregate DEVELOPMENT evidence, not episode rows or held-out scores."""
import math
from collections import Counter
from pathlib import Path

from .common import digest, read_json, object_digest
from .native_runtime_summary import stage_runtime


def summaries(root: Path) -> list[dict]:
    root = Path(root).resolve()
    directory = root/'measurements'
    if directory.is_symlink():
        return []
    from .handoff_correction import eligible
    handoffs = {row['label']: row for row in eligible(root)}
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
            consumption = row.get('data_consumption')
            status = consumption.get('status') if isinstance(consumption, dict) else None
            item['data_consumption_status'] = status if status in (
                'verified', 'not_required', 'not_applicable', 'unverified', 'failed') else 'unknown'
            if status == 'verified':
                versions = consumption.get('versions') or []
                item['data_versions'] = [v for v in versions[:16] if isinstance(v,str)] if isinstance(versions,list) else []
                exposure = []
                reads = consumption.get('exposure') or []
                for read in (reads[:16] if isinstance(reads,list) else []):
                    if not isinstance(read,dict):continue
                    counters = {k:read[k] for k in ('samples_read','batches','unique_samples')
                        if isinstance(read.get(k),int) and not isinstance(read[k],bool) and read[k]>=0}
                    sources = read.get('source_samples') or {}
                    sources = {k:v for k,v in (list(sources.items())[:16] if isinstance(sources,dict) else [])
                               if isinstance(v,int) and not isinstance(v,bool) and v>=0}
                    exposure.append({'data_version_id':read.get('data_version_id'),
                                     'status':read.get('status'),**counters,'source_samples':sources})
                item['data_exposure'] = exposure
                item['data_exposure_digest'] = object_digest(exposure)
                item['data_exposure_meaning'] = 'actual sample reads including repetition; not additional episodes or adequate learning'
            item['training_settings_meaning'] = 'requested inputs, not proof of work in the evaluated policy'
            timings = {stage: timing for stage in ('train', 'evaluate')
                       if (timing := stage_runtime(root, row, stage)) is not None}
            if timings:
                item['native_stage_runtime'] = timings
            if item['label'] in handoffs:
                item['training_handoff'] = handoffs[item['label']]
                item['interpretation_warning'] = (
                    'The scored artifact has an incomplete completed-work selection. '
                    'Do not interpret its score as the terminal training policy score; '
                    'inspect or correct the handoff before claiming adequate training failed.')
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
