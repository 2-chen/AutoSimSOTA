"""Explicit data versions and actual-loader witnesses, independent of repository names."""
import json
import re
from pathlib import Path

from .common import atomic_json, digest, read_json, now, object_digest
from .experiment_bundle import artifact_identity
from .baseline_reference import resolve_checkpoint, checkpoint_location

MARKER = 'AUTOSIM_TRAIN_DATA_CONSUMPTION '
PREFIX = 'data-version:'
_BINDING_FIELDS = ('id', 'setting', 'record_ref', 'record_sha256',
                   'identity_field', 'identity', 'source')


def _binding_record(binding):
    result = {key: binding[key] for key in _BINDING_FIELDS}
    if binding.get('consumer_review'):
        result['consumer_review'] = binding['consumer_review']
    return result


def training_metadata(bindings: list, settings: dict) -> list[dict]:
    """Transport selected labels to a native builder, never expected witness hashes.

    resolve() already verifies selection and bytes. This channel only preserves
    the registered ID and canonical slot after a handle becomes a native path.
    Native readers must still compute identity and count actual training work.
    """
    if not isinstance(bindings, list) or len(bindings) > 64:
        raise ValueError('training metadata needs bounded resolved bindings')
    result, seen = [], set()
    for binding in bindings:
        ident, slot = binding['id'], binding['setting']
        path = settings.get(slot)
        if (not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', ident)
                or not isinstance(slot, str) or not 1 <= len(slot) <= 128 or slot in seen
                or not isinstance(path, str) or not Path(path).is_absolute()):
            raise ValueError('training metadata disagrees with resolved data slots')
        seen.add(slot)
        result.append({'id': ident, 'setting': slot, 'path': path})
    return result


def native_selection_payload(inputs: dict) -> str:
    """Explicit labels for native bridges, independent of a frozen argv builder.

    No expected identity/counter is available through this channel. Consumers
    remain responsible for independent byte identification and observed work.
    An empty selection is serialized too, overriding stale ambient metadata.
    """
    selected = inputs.get('selected_training_data', [])
    if (not isinstance(selected, list) or len(selected) > 64 or any(
            not isinstance(row, dict) or set(row) != {'id', 'setting', 'path'}
            or not isinstance(row['setting'], str) or not isinstance(row['path'], str)
            or len(row['path']) > 4096 for row in selected)):
        raise ValueError('native training selection needs bounded id/setting/path labels only')
    labels = training_metadata(selected, {row['setting']:row['path'] for row in selected})
    payload = json.dumps(labels, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if len(payload.encode()) > 65536:
        raise ValueError('native training selection exceeds the environment payload bound')
    return payload


def seal_selection(output: Path, run_id: str, label: str, settings: dict,
                   bindings: list) -> dict:
    """Seal the selected versions before execution, not their later consumption.

    Keep a small projection of the reviewed records, not their full manifests.
    The native receipt still owns training and the loader log owns batch counts.
    """
    if (not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', run_id)
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', label)
            or not 1 <= len(bindings) <= 64):
        raise ValueError('invalid bounded data selection identity')
    body = {'schema_version': 1, 'label': label,
            'settings_sha256': object_digest(settings),
            'bindings': [_binding_record(b) for b in bindings]}
    identity = object_digest(body)
    ref = f'research/{run_id}/data_selections/{identity}.json'
    from .experiment_transaction import safe
    path = safe(output, ref)
    if path.exists():
        if read_json(path) != body:
            raise ValueError('immutable data selection changed')
    else:
        atomic_json(path, body)
    return {'id': identity, 'record_ref': ref, 'record_sha256': digest(path)}


def restore_selection(output: Path, repo: Path, run_id: str, measurement: dict) -> list:
    """Revalidate the ORIGINAL selection for a same-producer recovery.

    Never infer a data version from a path or retrospectively certify an old
    unbound measurement. A present proof must survive every check before replay.
    """
    proof = measurement.get('data_selection')
    if proof is None:
        if (measurement.get('data_consumption') or {}).get('versions'):
            raise ValueError('legacy data measurement lacks an explicit selected-version proof')
        return []  # Legacy unbound experiments remain unbound/unknown.
    if (not isinstance(proof, dict) or
            set(proof) != {'id', 'record_ref', 'record_sha256', 'training_attempt_id'} or
            not re.fullmatch(r'[a-f0-9]{64}', str(proof.get('id') or '')) or
            proof.get('training_attempt_id') != (measurement.get('train') or {}).get('attempt_id') or
            not re.fullmatch(r'[a-f0-9]{32}', str(proof.get('training_attempt_id') or ''))):
        raise ValueError('data selection must retain its exact original training producer')
    from .experiment_transaction import safe
    expected = f'research/{run_id}/data_selections/{proof["id"]}.json'
    if proof['record_ref'] != expected:
        raise ValueError('data selection proof leaves the original research')
    path = safe(output, expected)
    if (not path.is_file() or path.stat().st_size > 256 * 1024
            or digest(path) != proof['record_sha256']):
        raise ValueError('sealed data selection is missing or changed')
    body = read_json(path)
    if (not isinstance(body, dict) or
            set(body) != {'schema_version', 'label', 'settings_sha256', 'bindings'} or
            type(body['schema_version']) is not int or body['schema_version'] != 1 or
            object_digest(body) != proof['id'] or
            body['label'] != measurement.get('label') or
            body['settings_sha256'] != object_digest(measurement.get('settings'))):
        raise ValueError('selected data proof disagrees with the original measurement')
    held = body['bindings']
    if (not isinstance(held, list) or not 1 <= len(held) <= 64 or
            any(not isinstance(b, dict) or set(b) not in
                (set(_BINDING_FIELDS), set(_BINDING_FIELDS) | {'consumer_review'}) or
                not isinstance(b['setting'], str) or not 1 <= len(b['setting']) <= 128 or
                not isinstance(b['id'], str) or
                not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', b['id']) for b in held) or
            len({b['setting'] for b in held}) != len(held)):
        raise ValueError('invalid or duplicate selected data bindings')
    resolved, fresh = resolve(output, repo,
        {b['setting']: PREFIX + b['id'] for b in held},
        _completed_training={'measurement':measurement,'bindings':held,'run_id':run_id})
    if ([_binding_record(b) for b in fresh] != held or
            any(measurement['settings'].get(k) != v for k, v in resolved.items())):
        raise ValueError('original reviewed data selection or native input changed')
    return fresh


def loader_witnesses(log: Path, bindings: list) -> dict:
    """Reduce monotonic native reader counters without retaining a huge training log.

    One anonymous final event remains valid. Repeated cumulative events need increasing
    counts; different reader processes require explicit distinct reader_id values. Counts
    are summed across reader streams, never across successive progress events.
    """
    wanted = {(row['id'], row['setting']): row for row in bindings}
    version_ids = {row['id'] for row in bindings}
    streams, events = {}, 0
    with log.open(errors='replace') as handle:
        for line in handle:
            if not line.startswith(MARKER):
                continue
            if len(line.encode()) > 16000 or events >= 1000000:
                raise ValueError('loader witness stream exceeds bounded event size/population')
            row = json.loads(line[len(MARKER):])
            if not isinstance(row, dict):
                raise ValueError('data witness must be an object')
            ident = row.get('data_version_id')
            events += 1
            if ident not in version_ids:
                continue
            slot = (ident, row.get('setting'))
            if slot not in wanted:
                raise ValueError('loader witness setting disagrees with the selected version binding')
            binding = wanted[slot]
            if (
                    (row.get('identity_field'), row.get('identity')) !=
                    (binding['identity_field'], binding['identity'])):
                raise ValueError('loader witness version/setting/bytes disagree with the selected binding')
            count = row.get('samples_read')
            if type(count) is not int or not 0 < count <= 10**15:
                raise ValueError('actual positive integer sample reads are required')
            reader = row.get('reader_id', 'anonymous')
            if not isinstance(reader, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', reader):
                raise ValueError('reader_id must be a stable scoped reader identity')
            stage = row.get('consumption_stage', 'sample_read')
            if stage not in {'sample_read', 'training_batch'}:
                raise ValueError('consumption_stage must distinguish sample reads from actual training batches')
            key = (ident, binding['setting'], stage, reader)
            previous = streams.get(key)
            if previous is not None and count <= previous['samples_read']:
                raise ValueError('non-monotonic or duplicate loader counts; use distinct reader_id for each reader lifetime')
            from .data_lineage import exposure
            exposure(row, binding)  # Validate every event, not just the convenient last one.
            streams[key] = row
    result = {}
    for (ident, setting), binding in wanted.items():
        available = [(stage, row) for (i, s, stage, _), row in streams.items()
                     if i == ident and s == setting]
        stage = 'training_batch' if any(s == 'training_batch' for s, _ in available) else 'sample_read'
        readers = [row for s, row in available if s == stage]
        if not readers:
            raise ValueError('actual loader emitted no valid data-consumption witness for ' + ident)
        if len(readers) == 1:
            result[(ident, setting)] = {**readers[0], 'consumption_stage': stage,
                                       'reader_streams': 1, 'witness_events': events}
            continue
        if any(row.get('reader_id') is None for row in readers):
            raise ValueError('anonymous and identified reader streams cannot be mixed')
        combined = {'data_version_id': ident, 'setting': binding['setting'],
            'identity_field': binding['identity_field'], 'identity': binding['identity'],
            'samples_read': sum(row['samples_read'] for row in readers),
            'reader_streams': len(readers), 'witness_events': events, 'consumption_stage': stage}
        # Source-read counts add; distinct samples do NOT add across workers without a
        # global set/identity proof. Leave unique exposure unknown rather than inflate it.
        if all(isinstance(row.get('exposure'), dict) for row in readers):
            combined['exposure'] = {}
            if all('batches' in row['exposure'] for row in readers):
                combined['exposure']['batches'] = sum(row['exposure']['batches'] for row in readers)
            if all('source_samples' in row['exposure'] for row in readers):
                sources = {}
                for row in readers:
                    for source, count in row['exposure']['source_samples'].items():
                        sources[source] = sources.get(source, 0) + count
                combined['exposure']['source_samples'] = sources
        result[(ident, setting)] = combined
    return result


def describe(output: Path, repo: Path, spec: dict) -> dict:
    """Inspect bounded bytes/profile before Supervisor review; does not register data."""
    identity = spec.get('id')
    if not isinstance(identity,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',identity):
        raise ValueError('data version needs a path-safe id')
    path = output/'data_versions'/(identity+'.json')
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError('data version ids are immutable')
    source,native = checkpoint_location(output,repo,spec['dataset_ref'])
    files, size = [], 0
    for item in ([source] if source.is_file() else source.rglob('*')):
        if len(files)>=10000 or item.is_symlink():
            raise ValueError('data manifest exceeds bounds or contains symbolic links')
        size += item.stat().st_size if item.is_file() else 0
        if size>64*1024**3:
            raise ValueError('data version exceeds bounded 64 GiB manifest scan')
        files.append(item)
    field,value = artifact_identity(source)
    from .data_lineage import build, verify
    asset = build(output, repo, spec.get('asset'), source, native, field, value)
    verify(output, repo, asset)
    row = {'schema_version':2, 'at':now(), 'id':identity,
           'reason':spec.get('reason'), 'asset':asset,
           'dataset_ref':spec['dataset_ref'],'native_ref':str(native.relative_to(output.resolve())),
           'source':str(source),
           'identity_field':field,'identity':value,'bytes':size,'files':sum(p.is_file() for p in files),
           'source_refs':spec['source_refs'],'status':'reviewed_not_consumed',
           'authority':'registered data bytes; not proof of training consumption or quality'}
    return row


def register(output: Path, repo: Path, spec: dict, review: dict, *, described: dict | None = None) -> dict:
    if not all(review.get(k) is True for k in
        ('allowed_training_data','loader_connection_supported','sources_supported')):
        raise ValueError('independent data scope/loader/source review did not pass')
    row = dict(described) if described is not None else describe(output, repo, spec)
    if ((row.get('asset') or {}).get('profile_witness', {}).get('inspector') and
            review.get('manifest_observations_supported') is not True):
        raise ValueError('native source manifest requires independent observation/reader-source review')
    # described is internal executor output, never an Agent-facing argument. Revalidate
    # bytes across the model review window; an atomic registry lock only protects IDs.
    source, native = checkpoint_location(output, repo, spec['dataset_ref'])
    if (row['id'] != spec['id'] or row['dataset_ref'] != spec['dataset_ref']
            or str(source) != row['source'] or str(native.relative_to(output.resolve())) != row['native_ref']
            or artifact_identity(source) != (row['identity_field'], row['identity'])):
        raise ValueError('data identity changed during independent review')
    from .data_lineage import verify
    verify(output, repo, row['asset'])
    row.update(review=review, source_refs=spec['source_refs'])
    path = output/'data_versions'/(row['id']+'.json')
    if path.parent.is_symlink():
        raise ValueError('unsafe data registry')
    from .scheduling import locked
    with locked(path.parent/'registry.lock'):
        if path.exists() or path.is_symlink() or path.parent.is_symlink():
            raise ValueError('data version ids are immutable')
        atomic_json(path,row)
    return row


def resolve(output: Path, repo: Path, settings: dict, *, _completed_training=None) -> tuple[dict,list]:
    result,bindings = dict(settings),[]
    for key,value in settings.items():
        if not isinstance(value,str) or not value.startswith(PREFIX):
            continue
        identity = value[len(PREFIX):]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',identity):
            raise ValueError('invalid data-version handle')
        path = output/'data_versions'/(identity+'.json')
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('unsafe data registry')
        row = read_json(path)
        if row.get('status')!='reviewed_not_consumed' or not all(row.get('review',{}).get(k) is True
                for k in ('allowed_training_data','loader_connection_supported','sources_supported')):
            raise ValueError('data version lacks accepted independent review')
        if ((row.get('asset') or {}).get('profile_witness', {}).get('inspector') and
                row.get('review', {}).get('manifest_observations_supported') is not True):
            raise ValueError('native manifest observation review is no longer accepted')
        consumer_review = None
        if row.get('source_refs'):
            from .baseline_reference import source_evidence
            try:
                if source_evidence(repo,row['source_refs'],output=output)!=row['source_refs']:
                    raise ValueError('reviewed data loader sources changed')
            except (OSError, ValueError, TypeError, KeyError):
                from .data_consumer_review import accepted, accepted_for_completed_training
                consumer_review = accepted(output, repo, row, key)
                if consumer_review is None and _completed_training is not None:
                    try:
                        consumer_review=accepted_for_completed_training(output,repo,row,key,
                            context=_completed_training)
                    except (OSError,ValueError,TypeError,KeyError,StopIteration):
                        consumer_review=None
                if consumer_review is None:
                    raise  # No current consumer approval: do not reuse stale review.
        source,native = checkpoint_location(output,repo,row.get('native_ref') or row['dataset_ref'])
        if (row.get('id')!=identity or str(source)!=row['source']
                or artifact_identity(source)!=(row['identity_field'],row['identity'])):
            raise ValueError('registered data bytes changed')
        from .data_lineage import verify
        verify(output, repo, row.get('asset') or {})
        # Native processes see the authorized resource mount, not its host backing source.
        result[key] = str(native)
        bindings.append({**row,'setting':key,'record_ref':str(path.relative_to(output)),
                         'record_sha256':digest(path),
                         **({'consumer_review':consumer_review} if consumer_review else {})})
    return result,bindings


def verify_consumption(output: Path, bindings: list, trained: dict, *, training_performed: bool | None = None) -> dict:
    if not bindings:
        # Absence of a version handle says nothing about the native loader. Existing
        # demonstrations can be misrouted just as easily as newly collected data.
        # Online RL/source controllers need no demonstration witness, but that is a
        # source-backed Agent diagnosis, not something inferred from an empty list.
        if training_performed is None:
            training_performed = bool(trained.get('ran') or (
                trained.get('ran') is not False and trained.get('attempt_id')
                and trained.get('returncode') == 0))
        return ({'status':'unknown','meaning':'training ran without a reviewed data binding; '
                 'inspect the native loader, actual population and batch consumption',
                 'repair_owner':'data','next_operation':'research_task'}
                if training_performed else
                {'status':'not_applicable','meaning':'evaluation-only; no training data consumed in this measurement'})
    try:
        from .evidence_store import read_attempt_evidence
        evidence_id = trained.get('evidence_id') or trained.get('attempt_id') or ''
        verified = read_attempt_evidence(output,evidence_id)  # Validate sealed log ownership and bytes.
        if verified.get('status')!='completed' or verified.get('returncode')!=0:
            raise ValueError('data consumption requires a completed successful training attempt')
        evidence = read_json(output/'evidence'/(evidence_id+'.json'))
        log = output/evidence['log_ref']
        rows = loader_witnesses(log, bindings)
        exposures = []
        for binding in bindings:
            path=output/binding['record_ref']
            if digest(path)!=binding['record_sha256']:
                raise ValueError('data review changed')
            field,value=artifact_identity(Path(binding['source']))
            if (field,value)!=(binding['identity_field'],binding['identity']):
                raise ValueError('data bytes changed during training')
            from .data_lineage import verify, exposure
            verify(output, None, binding.get('asset') or {})
            matched = rows[(binding['id'], binding['setting'])]
            exposures.append({'data_version_id':binding['id'], 'setting':binding['setting'],
                              **exposure(matched, binding),
                              'consumption_stage': matched['consumption_stage'],
                              'reader_streams': matched['reader_streams'],
                              'witness_events': matched['witness_events']})
        return {'status':'verified','evidence_id':evidence_id,
                'versions':[b['id'] for b in bindings],
                'exposure':exposures,
                'authority':'sealed loader witness under independently reviewed source; not a quality/SOTA claim'}
    except (OSError,ValueError,TypeError,KeyError) as exc:
        return {'status':'unverified','why':str(exc)[:700], 'repair_owner':'data',
                'next_operation':'research_task',
                'missing_edge':'inspect the selected version/setting/bytes and actual reader counters; '
                               'Fix source routing, then revalidate the same training consumer'}


def catalog(output: Path, *, offset: int = 0, limit: int = 64) -> list:
    if (isinstance(offset,bool) or not isinstance(offset,int) or offset<0
            or isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=64):
        raise ValueError('data catalog needs nonnegative offset and limit 1..64')
    rows=[]
    if (output/'data_versions').is_symlink():
        return rows
    for path in sorted((output/'data_versions').glob('*.json'))[offset:offset+limit]:
        try:
            if path.is_symlink():continue
            row=read_json(path)
            if (not isinstance(row.get('id'),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',row['id'])
                    or row['id']!=path.stem):
                continue
            asset = row.get('asset') or {}
            if not isinstance(asset,dict):continue
            profile = asset.get('profile') or {}
            profile = {k:profile[k] for k in ('scope','format','counts') if k in profile}
            full_profile = asset.get('profile') or {}
            for key in ('issues','coverage','funnel'):
                values = full_profile.get(key) or {}
                profile[key] = dict(list(values.items())[:6])
                if len(values)>6:profile[key+'_categories_omitted'] = len(values)-6
            if full_profile.get('manifest'):
                profile['manifest'] = full_profile['manifest']
            if full_profile.get('selection'):
                selected = full_profile['selection']
                profile['selection'] = {k:selected[k] for k in ('counts','rejected','unknown','manifest_sha256') if k in selected}
            rows.append({**{k:row[k] for k in ('id','dataset_ref','status','files','bytes')},
                         'kind':asset.get('kind','unknown'),
                         'parents':[p['id'] for p in asset.get('parents',[])],
                         'profile':profile,
                         'producer_ref':(asset.get('producer') or {}).get('evidence_ref'),
                         'producer_status':(asset.get('producer') or {}).get('producer_status'),
                         'partial_output':(asset.get('producer') or {}).get('partial_output',False),
                         'producer_scope':(asset.get('producer') or {}).get('scope','this_run'),
                         'recipe':{'operation':asset['recipe']['operation']} if asset.get('recipe') else {},
                         'record_sha256':digest(path)})
        except (OSError,ValueError,TypeError,KeyError,AttributeError):pass
    return rows
