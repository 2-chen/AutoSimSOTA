"""Explicit data versions and actual-loader witnesses, independent of repository names."""
import json
import re
from pathlib import Path

from .common import atomic_json, digest, read_json, now
from .experiment_bundle import artifact_identity
from .baseline_reference import resolve_checkpoint, checkpoint_location

MARKER = 'AUTOSIM_TRAIN_DATA_CONSUMPTION '
PREFIX = 'data-version:'


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


def resolve(output: Path, repo: Path, settings: dict) -> tuple[dict,list]:
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
        if row.get('source_refs'):
            from .baseline_reference import source_evidence
            if source_evidence(repo,row['source_refs'])!=row['source_refs']:
                raise ValueError('reviewed data loader sources changed')
        source,native = checkpoint_location(output,repo,row.get('native_ref') or row['dataset_ref'])
        if (row.get('id')!=identity or str(source)!=row['source']
                or artifact_identity(source)!=(row['identity_field'],row['identity'])):
            raise ValueError('registered data bytes changed')
        from .data_lineage import verify
        verify(output, repo, row.get('asset') or {})
        # Native processes see the authorized resource mount, not its host backing source.
        result[key] = str(native)
        bindings.append({**row,'setting':key,'record_ref':str(path.relative_to(output)),
                         'record_sha256':digest(path)})
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
        rows = []
        for line in log.read_text(errors='replace').splitlines():
            if line.startswith(MARKER):
                row=json.loads(line[len(MARKER):])
                if not isinstance(row,dict):
                    raise ValueError('data witness must be an object')
                rows.append(row)
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
            matched=[row for row in rows if row.get('data_version_id')==binding['id']
                and row.get('identity_field')==field and row.get('identity')==value
                and isinstance(row.get('samples_read'),int) and not isinstance(row['samples_read'],bool)
                and row['samples_read']>0 and row.get('setting')==binding['setting']]
            if len(matched)!=1:
                raise ValueError('actual loader did not emit exactly one valid data-consumption witness')
            exposures.append({'data_version_id':binding['id'], 'setting':binding['setting'],
                              **exposure(matched[0], binding)})
        return {'status':'verified','evidence_id':evidence_id,
                'versions':[b['id'] for b in bindings],
                'exposure':exposures,
                'authority':'sealed loader witness under independently reviewed source; not a quality/SOTA claim'}
    except (OSError,ValueError,TypeError,KeyError) as exc:
        return {'status':'unverified','why':str(exc)[:700]}


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
            for key in ('issues','coverage'):
                values = full_profile.get(key) or {}
                profile[key] = dict(list(values.items())[:6])
                if len(values)>6:profile[key+'_categories_omitted'] = len(values)-6
            rows.append({**{k:row[k] for k in ('id','dataset_ref','status','files','bytes')},
                         'kind':asset.get('kind','unknown'),
                         'parents':[p['id'] for p in asset.get('parents',[])],
                         'profile':profile,
                         'producer_ref':(asset.get('producer') or {}).get('evidence_ref'),
                         'recipe':{'operation':asset['recipe']['operation']} if asset.get('recipe') else {},
                         'record_sha256':digest(path)})
        except (OSError,ValueError,TypeError,KeyError,AttributeError):pass
    return rows
