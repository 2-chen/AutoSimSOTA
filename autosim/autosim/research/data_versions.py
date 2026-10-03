"""Explicit data versions and actual-loader witnesses, independent of repository names."""
import json
import re
from pathlib import Path

from .common import atomic_json, digest, read_json
from .experiment_bundle import artifact_identity
from .baseline_reference import resolve_checkpoint, checkpoint_location

MARKER = 'AUTOSIM_TRAIN_DATA_CONSUMPTION '
PREFIX = 'data-version:'


def register(output: Path, repo: Path, spec: dict, review: dict) -> dict:
    identity = spec.get('id')
    if not isinstance(identity,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',identity):
        raise ValueError('data version needs a path-safe id')
    path = output/'data_versions'/(identity+'.json')
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError('data version ids are immutable')
    if not all(review.get(k) is True for k in
        ('allowed_training_data','loader_connection_supported','sources_supported')):
        raise ValueError('independent data scope/loader/source review did not pass')
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
    row = {'id':identity,'dataset_ref':spec['dataset_ref'],'native_ref':str(native.relative_to(output.resolve())),
           'source':str(source),
           'identity_field':field,'identity':value,'bytes':size,'files':sum(p.is_file() for p in files),
           'review':review,'source_refs':spec['source_refs'],'status':'reviewed_not_consumed',
           'authority':'registered data bytes; not proof of training consumption or quality'}
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
        # Native processes see the authorized resource mount, not its host backing source.
        result[key] = str(native)
        bindings.append({**row,'setting':key,'record_ref':str(path.relative_to(output)),
                         'record_sha256':digest(path)})
    return result,bindings


def verify_consumption(output: Path, bindings: list, trained: dict) -> dict:
    if not bindings:
        return {'status':'not_required','meaning':'no explicit new data version was selected'}
    try:
        from .evidence_store import read_attempt_evidence
        evidence_id = trained.get('evidence_id') or trained.get('attempt_id') or ''
        read_attempt_evidence(output,evidence_id)  # Validate sealed log ownership and bytes.
        evidence = read_json(output/'evidence'/(evidence_id+'.json'))
        log = output/evidence['log_ref']
        rows = []
        for line in log.read_text(errors='replace').splitlines():
            if line.startswith(MARKER):
                row=json.loads(line[len(MARKER):])
                if not isinstance(row,dict):
                    raise ValueError('data witness must be an object')
                rows.append(row)
        for binding in bindings:
            path=output/binding['record_ref']
            if digest(path)!=binding['record_sha256']:
                raise ValueError('data review changed')
            field,value=artifact_identity(Path(binding['source']))
            if (field,value)!=(binding['identity_field'],binding['identity']):
                raise ValueError('data bytes changed during training')
            matched=[row for row in rows if row.get('data_version_id')==binding['id']
                and row.get('identity_field')==field and row.get('identity')==value
                and isinstance(row.get('samples_read'),int) and not isinstance(row['samples_read'],bool)
                and row['samples_read']>0 and row.get('setting')==binding['setting']]
            if len(matched)!=1:
                raise ValueError('actual loader did not emit exactly one valid data-consumption witness')
        return {'status':'verified','evidence_id':evidence_id,
                'versions':[b['id'] for b in bindings],
                'authority':'sealed loader witness under independently reviewed source; not a quality/SOTA claim'}
    except (OSError,ValueError,TypeError,KeyError) as exc:
        return {'status':'unverified','why':str(exc)[:700]}


def catalog(output: Path) -> list:
    rows=[]
    if (output/'data_versions').is_symlink():
        return rows
    for path in sorted((output/'data_versions').glob('*.json'))[:64]:
        try:
            if path.is_symlink():continue
            row=read_json(path)
            rows.append({k:row[k] for k in ('id','dataset_ref','status','files','bytes')})
        except (OSError,ValueError,TypeError,KeyError):pass
    return rows
