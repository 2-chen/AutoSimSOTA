"""Offer verified producer artifacts; the Agent chooses native save/load boundaries.

No filename, policy family or size-ranking recipe chooses a checkpoint here. Byte and
ownership checks do not claim loader compatibility; native loading must still succeed.
"""
from pathlib import Path
import hashlib

from .common import atomic_json, digest, now, object_digest, read_json
from .evidence_store import read_attempt_evidence


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            hasher.update(block)
    return hasher.hexdigest()


def catalog(root: Path) -> dict:
    unavailable = {'status':'unavailable', 'entries':[]}
    try:
        document = root/'derivation_attempts/train.json'
        if document.is_symlink() or document.stat().st_size > 2*1024**2:
            return unavailable
        accepted = next(row for row in reversed(read_json(document).get('attempts') or [])
                        if row.get('status') == 'accepted')
        progress = accepted.get('training_progress') or {}
        if progress.get('status') != 'observed':
            return unavailable
        ref = accepted.get('native_evidence_ref') or ''
        identity = Path(ref).stem
        native = read_attempt_evidence(root, identity, limit=1)
        if native['returncode'] != 0 or native['status'] != 'completed':
            return unavailable
        output = Path(accepted['verification_output']).resolve(strict=True)
        if not output.is_dir() or not output.is_relative_to(root.resolve()):
            return unavailable
        artifact = accepted.get('verified_artifact') or {}
        declared = Path(artifact['declared_path']).resolve(strict=True)
        if not declared.is_file() or not declared.is_relative_to(output):
            return unavailable
        for item in artifact.get('structured_candidates') or []:
            path = Path(item['path'])
            if not path.resolve().is_relative_to(output) or digest(path) != item['sha256']:
                return unavailable
        counter = progress.get('evidence') or {}
        if counter.get('document_sha256'):
            path = output/counter['native_counter']['path']
            if not path.resolve().is_relative_to(output) or digest(path) != counter['document_sha256']:
                return unavailable
        entries = []
        for path in sorted(output.rglob('*')):
            if len(entries) >= 96:
                break
            if path.is_symlink() or not path.resolve().is_relative_to(output):
                continue
            entries.append({'ref':path.relative_to(root.resolve()).as_posix(),
                'kind':'directory' if path.is_dir() else 'file',
                'bytes':path.stat().st_size if path.is_file() else None})
        return {'status':'available', 'output_ref':output.relative_to(root.resolve()).as_posix(),
                'declared_artifact_ref':declared.relative_to(root.resolve()).as_posix(),
                'native_evidence_ref':ref, 'progress':progress, 'entries':entries,
                'instruction':'Choose the exact policy file OR bundle directory from producer '
                'and loader source, then pass its run-relative checkpoint_ref to '
                'derive_a_command(stage=score_target). Do not choose an optimizer state, '
                'training counter or largest file just because it exists. No default is '
                'selected. This is command verification, not a scored baseline.'}
    except (OSError, ValueError, TypeError, KeyError, StopIteration, AttributeError):
        return unavailable


def select(root: Path, ref: str, offered: dict) -> dict:
    if offered.get('status') != 'available' or not isinstance(ref, str) or not ref:
        raise ValueError('checkpoint_ref requires a currently verified producer catalog')
    relative = Path(ref)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('checkpoint_ref must be a confined run-relative path')
    path = (root/relative).resolve(strict=True)
    output = (root/offered['output_ref']).resolve(strict=True)
    if not path.is_relative_to(output) or not path.exists():
        raise ValueError('checkpoint_ref is not from the accepted native producer output')
    files = [path] if path.is_file() else sorted(path.rglob('*'))
    manifest = []
    for candidate in files:
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(output):
            raise ValueError('selected bundle contains an unowned or symbolic-link entry')
        if not candidate.is_file():
            continue
        if len(manifest) >= 256:
            raise ValueError('bundle manifest too large; select a source-backed smaller native bundle')
        manifest.append({'ref':candidate.relative_to(root.resolve()).as_posix(),
                         'sha256':file_digest(candidate),'bytes':candidate.stat().st_size})
    if not manifest:
        raise ValueError('selected policy file/bundle has no bytes')
    identity = object_digest({'producer':offered['native_evidence_ref'], 'files':manifest})
    record = {'id':identity, 'at':now(), 'selected_path':str(path),
              'checkpoint_ref':ref, 'producer_evidence_ref':offered['native_evidence_ref'],
              'files':manifest, 'load_compatibility':'unverified_requires_native_load',
              'authority':'Agent choice, frozen producer bytes; not performance evidence'}
    proof_ref = f'verification_checkpoint_handoffs/{identity}.json'
    atomic_json(root/proof_ref, record)
    return {**record, 'evidence_ref':proof_ref}


def validate(root: Path, selected: dict) -> None:
    """Repeat the frozen input identity check immediately before each native attempt."""
    base = Path(selected['selected_path']).resolve(strict=True)
    if not base.is_relative_to(root.resolve()):
        raise ValueError('selected checkpoint escaped the run')
    for row in selected['files']:
        path = root/row['ref']
        resolved = path.resolve(strict=True)
        if (path.is_symlink() or not (resolved == base or resolved.is_relative_to(base)) or
                path.stat().st_size != row['bytes'] or file_digest(path) != row['sha256']):
            raise ValueError('selected producer bytes changed before native loading: '+row['ref'])
