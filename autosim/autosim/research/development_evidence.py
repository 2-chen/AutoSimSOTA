"""Bounded aggregate DEVELOPMENT evidence, not episode rows or held-out scores."""
import math
import json
import re
from collections import Counter
from pathlib import Path

from .common import digest, read_json, object_digest
from .native_runtime_summary import stage_runtime


def training_selection_transport(receipt: dict) -> dict:
    """Bounded receipt observation; never data-consumption or registry authority.

    Paths are intentionally not projected. Missing metadata is unknown, whereas
    the executor's explicit [] means no registered selection was transported;
    it does NOT mean the native trainer read no data.
    """
    from .common import sanitize_model_text
    result = {'status':'unknown',
        'authority':'recorded train receipt environment only; not opened bytes, '
            'registry approval, training consumption or data quality',
        'guidance':'An omitted training input does not inherit the last candidate call. '
            'No registered selection can still train on native default data; '
            'do not attribute it to a registered acquisition.'}
    environment = receipt.get('environment')
    if not isinstance(environment,dict) or 'AUTOSIM_TRAIN_SELECTION_JSON' not in environment:
        return result
    raw = environment['AUTOSIM_TRAIN_SELECTION_JSON']
    if not isinstance(raw,str) or len(raw.encode('utf-8'))>16384:
        return {**result,'status':'invalid_transport_metadata'}
    try:
        rows = json.loads(raw)
    except (TypeError,ValueError):
        return {**result,'status':'invalid_transport_metadata'}
    if not isinstance(rows,list) or len(rows)>64:
        return {**result,'status':'invalid_transport_metadata'}
    bindings = []
    for row in rows:
        if not isinstance(row,dict):
            return {**result,'status':'invalid_transport_metadata'}
        identity, setting, path = row.get('id'),row.get('setting'),row.get('path')
        if (not isinstance(identity,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',identity)
                or not isinstance(setting,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',setting)
                or not isinstance(path,str) or not path or len(path)>4096
                or sanitize_model_text(identity)!=identity or sanitize_model_text(setting)!=setting):
            return {**result,'status':'invalid_transport_metadata'}
        bindings.append({'data_version_id':identity,'setting':setting})
    return {**result,'status':'transported' if bindings else 'no_registered_selection',
            'binding_count':len(bindings),'bindings':bindings}


def reader_issue_summary(log: Path, bindings: list[dict], expected_sha256: str) -> dict:
    """Bounded negative-only diagnostics; never salvage or certify witness rows."""
    import hashlib
    from .data_versions import MARKER
    unavailable = {'status': 'unavailable', 'authority': 'No log-schema diagnostics or consumption inferred.'}
    if log.is_symlink() or not log.is_file() or log.stat().st_size > 16*1024*1024:
        return unavailable
    raw = log.read_bytes()
    if len(raw) > 16*1024*1024 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        return unavailable
    selected = {}
    for binding in bindings:
        identity = binding.get('id')
        if isinstance(identity, str):
            selected.setdefault(identity, []).append(binding)
    issues = Counter()
    stages = Counter()
    markers = parsed = 0
    for line in raw.decode('utf-8', 'replace').splitlines():
        if MARKER not in line:
            continue
        markers += 1
        try:
            row = json.loads(line.split(MARKER, 1)[1])
        except (ValueError, TypeError):
            issues['malformed_json_lines'] += 1
            continue
        if not isinstance(row, dict):
            issues['non_object_records'] += 1
            continue
        parsed += 1
        stage = row.get('consumption_stage')
        if not isinstance(stage, str) or stage not in {'sample_read', 'training_batch'}:
            stage = 'unknown'
            issues['missing_or_unknown_consumption_stage'] += 1
        stages[stage] += 1
        reader = row.get('reader_id')
        if not isinstance(reader, str) or not reader or len(reader) > 256:
            issues['missing_or_unbounded_reader_id'] += 1
        identity = row.get('data_version_id')
        matches = selected.get(identity, []) if isinstance(identity, str) else []
        if not matches:
            issues['unselected_or_missing_version_id'] += 1
        else:
            matching_slot = [binding for binding in matches if binding.get('setting') == row.get('setting')]
            if not matching_slot:
                issues['selected_input_slot_mismatch'] += 1
            elif not any(binding.get('identity_field') == row.get('identity_field') and
                         binding.get('identity') == row.get('identity') for binding in matching_slot):
                issues['selected_bytes_identity_mismatch'] += 1
        count = row.get('samples_read')
        if type(count) is not int or count < 0:
            issues['missing_or_invalid_sample_counter'] += 1
    return {'status': 'bounded_schema_diagnostics', 'scope': 'complete sealed log within 16MiB bound',
        'marker_lines': markers, 'parsed_object_records': parsed,
        'events_by_stage': dict(stages), 'issue_counts': dict(issues),
        'authority': 'Partial schema diagnostics only, not the strict validator, actual consumption, '
            'unique samples, training effect or replay permission. Counts omit unparseable payloads; '
            'zero listed issues does not establish validity. Omissions do not retroactively '
            'invalidate a legacy schema. Do not split/repair old log rows.'}


def reader_failure_diagnostics(root: Path) -> list[dict]:
    """Expose sealed counter failures even when an earlier handoff masked them.

    This is a negative-only historical observation, not restore_selection(), a
    current source approval, a consumption certificate or a replay permission.
    Do not write it into original measurements or repair their sealed logs.
    """
    from .experiment_transaction import safe
    from .evidence_store import read_attempt_evidence
    from .data_versions import loader_witnesses, training_metadata
    root = Path(root).resolve()
    if root.parent.name != 'research' or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', root.name):
        return []
    output = root.parent.parent
    try:
        directory = safe(output, f'research/{root.name}/measurements')
        paths = sorted((p for p in directory.glob('round_*.json')
                        if not p.is_symlink() and p.is_file()),
                       key=lambda p:(p.stat().st_mtime_ns,p.name))[-32:]
    except (OSError, ValueError):
        return []
    result = []
    # Select AFTER authentication/filtering. New successful or unbound rows
    # must not evict recent failures. This bounded window is not all history.
    for path in reversed(paths):
        try:
            if path.stat().st_size > 8*1024*1024:
                continue
            row = read_json(path)
            if (not isinstance(row, dict) or row.get('ok') is not False
                    or row.get('label') != path.stem or row.get('where') not in
                    {'policy_artifact', 'training_data'}):
                continue
            proof = row['data_selection']; train = row['train']
            attempt = train['attempt_id']; identity = proof['id']
            if (not re.fullmatch(r'[a-f0-9]{32}', attempt)
                    or not re.fullmatch(r'[a-f0-9]{64}', identity)
                    or proof['training_attempt_id'] != attempt):
                continue
            selection_ref = f'research/{root.name}/data_selections/{identity}.json'
            selection_path = safe(output, selection_ref)
            if (proof['record_ref'] != selection_ref or selection_path.stat().st_size > 256*1024
                    or digest(selection_path) != proof['record_sha256']):
                continue
            body = read_json(selection_path)
            if (object_digest(body) != identity or body['label'] != row['label']
                    or body['settings_sha256'] != object_digest(row['settings'])):
                continue
            bindings = body['bindings']
            if not isinstance(bindings, list) or not 1 <= len(bindings) <= 64:
                continue
            fresh = []
            for binding in bindings:
                if (not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', binding['id'])
                        or binding['record_ref'] != f"data_versions/{binding['id']}.json"):
                    raise ValueError('invalid data card reference')
                card_path = safe(output, binding['record_ref'])
                if card_path.stat().st_size > 8*1024*1024 or digest(card_path) != binding['record_sha256']:
                    raise ValueError('historical card unavailable')
                card = read_json(card_path)
                if any(card.get(key) != binding[key] for key in ('id', 'source', 'identity_field', 'identity')):
                    raise ValueError('selected card identity disagrees')
                fresh.append({**card, **binding})
            verified = read_attempt_evidence(output, attempt, limit=1)
            receipt_ref = f'research/{root.name}/attempts/{attempt}/receipt.json'
            receipt = read_json(safe(output, receipt_ref))
            if (verified['receipt_ref'] != receipt_ref or verified['status'] != 'completed'
                    or verified['returncode'] != 0 or verified['log_bytes'] > 16*1024*1024
                    or receipt.get('run_id') != root.name or receipt.get('stage') != 'train'
                    or receipt.get('attempt_id') != attempt or receipt.get('status') != 'completed'
                    or receipt.get('returncode') != 0 or receipt.get('settings_digest') != object_digest(row['settings'])
                    or receipt.get('environment') != train.get('environment')
                    or json.loads(receipt['environment']['AUTOSIM_TRAIN_SELECTION_JSON']) !=
                    training_metadata(bindings, row['settings'])):
                continue
            evidence = read_json(safe(output, f'evidence/{attempt}.json'))
            log = safe(output, evidence['log_ref'])
            try:
                loader_witnesses(log, fresh)
            except (ValueError, TypeError, KeyError) as exc:
                result.append({'label':row['label'], 'measurement_ref':str(path.relative_to(output)),
                    'training_attempt_id':attempt, 'evidence_id':attempt,
                    'selection_ref':selection_ref, 'log_sha256':evidence['log_sha256'],
                    'status':'sealed_reader_events_invalid', 'why':str(exc)[:700],
                    'schema_diagnostics':reader_issue_summary(log, fresh, evidence['log_sha256']),
                    'simultaneous_measurement_failure':row['where'],
                    'repair_owner':'data',
                    'authority':'negative historical log diagnostic only; not current source approval, score or consumption certification',
                    'guidance':'A bundle-only correction cannot repair these immutable reader events. '
                        'Data/Fix may inspect the original consumer and Main chooses the next audited action; '
                        'do not edit old logs, infer successful consumption, or replay an unknown operation.'})
                if len(result)>=3:
                    break
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return list(reversed(result))


def summaries(root: Path, *, _handoffs: list[dict] | None = None) -> list[dict]:
    root = Path(root).resolve()
    directory = root/'measurements'
    if directory.is_symlink():
        return []
    from .handoff_correction import eligible
    handoffs = {row['label']: row for row in (eligible(root) if _handoffs is None else _handoffs)}
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
                    stage = read.get('consumption_stage')
                    readers = read.get('reader_streams')
                    exposure.append({'data_version_id':read.get('data_version_id'),
                        'setting':read.get('setting'), 'status':read.get('status'),
                        'consumption_stage':stage if stage in ('sample_read','training_batch') else 'unknown',
                        **({'reader_streams':readers} if type(readers) is int and readers>0 else {}),
                        **counters,'source_samples':sources})
                item['data_exposure'] = exposure
                item['data_exposure_digest'] = object_digest(exposure)
                item['data_exposure_meaning'] = (
                    'consumption_stage distinguishes prefetched sample_read from training_batch; '
                    'counters include repetition, not unique samples, additional episodes or adequate learning; '
                    'reader_streams counts whole-run identities, not concurrent workers')
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
