"""Offer verified producer artifacts; the Agent chooses native save/load boundaries.

No filename, policy family or size-ranking recipe chooses a checkpoint here. Byte and
ownership checks do not claim loader compatibility; native loading must still succeed.
"""
from pathlib import Path
import hashlib
import re

from .common import atomic_json, digest, now, object_digest, read_json
from .evidence_store import read_attempt_evidence


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            hasher.update(block)
    return hasher.hexdigest()


def _verification_catalog(root: Path) -> dict:
    root = Path(root).resolve()
    def unavailable(why):
        return {'status':'unavailable', 'entries':[], 'because':why,
                'evidence_refs':['derivation_attempts/train.json'],
                'instruction':'Inspect the accepted training attempt, its sealed native evidence '
                    'and progress/artifact audit. Storage GC registration cannot authorize '
                    'a producer. Correct changed or missing evidence before choosing a policy.'}
    try:
        document = root/'derivation_attempts/train.json'
        if document.is_symlink() or document.stat().st_size > 2*1024**2:
            return unavailable('training attempt document is unsafe or too large')
        accepted = next(row for row in reversed(read_json(document).get('attempts') or [])
                        if row.get('status') == 'accepted')
        progress = accepted.get('training_progress') or {}
        if progress.get('status') != 'observed':
            return unavailable('accepted training has no independently observed native progress')
        ref = accepted.get('native_evidence_ref') or ''
        identity = Path(ref).stem
        native = read_attempt_evidence(root, identity, limit=1)
        if native['returncode'] != 0 or native['status'] != 'completed':
            return unavailable('accepted producer has no completed zero-exit native evidence')
        output = Path(accepted['verification_output']).resolve(strict=True)
        if not output.is_dir() or not output.is_relative_to(root.resolve()):
            return unavailable('accepted producer output is not a run-owned directory')
        artifact = accepted.get('verified_artifact') or {}
        declared = None
        if artifact.get('declared_path'):
            declared = Path(artifact['declared_path']).resolve(strict=True)
            if not declared.is_file() or not declared.is_relative_to(output):
                return unavailable('unique declared artifact is outside producer output or absent')
        elif not (artifact.get('matched', 0) > 0 and
                  artifact.get('structured_candidates') and artifact.get('pattern')):
            # Native checkpoint globs legitimately match several snapshots/bundle
            # members. Ambiguity is for the Agent to resolve from save/load source,
            # not a reason to erase a completed, independently verified producer.
            return unavailable('producer lacks a declared artifact or verified multi-match artifact evidence')
        for item in artifact.get('structured_candidates') or []:
            path = Path(item['path'])
            if path.is_symlink() or not path.resolve().is_relative_to(output) or digest(path) != item['sha256']:
                return unavailable('verified structured producer artifact changed or escaped output')
        counter = progress.get('evidence') or {}
        if counter.get('document_sha256'):
            path = output/counter['native_counter']['path']
            if not path.resolve().is_relative_to(output) or digest(path) != counter['document_sha256']:
                return unavailable('independently reviewed native training counter changed or escaped output')
        entries = []
        entry_count = 0
        for path in sorted(output.rglob('*')):
            if path.is_symlink() or not path.resolve().is_relative_to(output):
                continue
            entry_count += 1
            if len(entries) < 96:
                entries.append({'ref':path.relative_to(root.resolve()).as_posix(),
                    'kind':'directory' if path.is_dir() else 'file',
                    'bytes':path.stat().st_size if path.is_file() else None})
        return {'status':'available', 'output_ref':output.relative_to(root.resolve()).as_posix(),
                'declared_artifact_ref':declared.relative_to(root.resolve()).as_posix() if declared else None,
                'native_evidence_ref':ref, 'progress':progress, 'entries':entries,
                'entry_count':entry_count, 'entries_truncated':entry_count > len(entries),
                'instruction':'Choose the exact policy file OR bundle directory from producer '
                'and loader source, then pass its run-relative checkpoint_ref to '
                'derive_a_command(stage=score_target). Do not choose an optimizer state, '
                'training counter or largest file just because it exists. No default is '
                'selected. For the complete paged file inventory use inspect_storage with '
                'the producer evidence_id (stem of native_evidence_ref); the first 96 entries '
                'are only a preview, not the allowed set. This is command verification, '
                'not a scored baseline.'}
    except (OSError, ValueError, TypeError, KeyError, StopIteration, AttributeError) as exc:
        return unavailable('producer verification could not complete: '+type(exc).__name__+': '+str(exc)[:300])


def catalog(root: Path) -> dict:
    """One handoff surface for derivation probes and independently audited long jobs."""
    from .training_progress_audit import revalidate_detached
    root = Path(root).resolve()
    verification = _verification_catalog(root)
    producers, pending = [], []
    if verification['status'] == 'available':
        producers.append({**verification, 'origin':'command_verification',
                          'producer_id':Path(verification['native_evidence_ref']).stem})
    jobs_root = root/'native_jobs'
    paths = sorted(jobs_root.glob('*/request.json')) if not jobs_root.is_symlink() else []
    for path in paths[:128]:
        try:
            if (path.is_symlink() or path.parent.is_symlink() or
                    not re.fullmatch(r'[0-9a-f]{32}',path.parent.name) or path.stat().st_size>65536):
                continue
            request = read_json(path)
            if request.get('stage') != 'train' or request.get('job_id') != path.parent.name:
                continue
            result_path = path.with_name('result.json')
            if not result_path.exists():
                continue
            if result_path.is_symlink() or result_path.stat().st_size>2*1024**2:
                continue
            result = read_json(result_path)
            if result.get('job_id') != request['job_id'] or result.get('status') != 'completed':
                continue
            if not request.get('source_identity') or request['source_identity'] != result.get('source_identity_after'):
                raise ValueError('worker source identity was not stable')
            native_result = result.get('stage_result') or {}
            attempt = native_result.get('attempt_id') or ''
            run_id = request.get('run_id') or ''
            if not re.fullmatch(r'[0-9a-f]{32}',attempt) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',run_id):
                raise ValueError('invalid producer attempt/run identity')
            receipt_path = root/'research'/run_id/'attempts'/attempt/'receipt.json'
            if receipt_path.is_symlink() or receipt_path.stat().st_size>2*1024**2:
                raise ValueError('unsafe native receipt')
            receipt = read_json(receipt_path)
            if (receipt.get('stage') != 'train' or receipt.get('attempt_id') != attempt or
                    receipt.get('controller_decision_id') != request['job_id'] or
                    native_result.get('evidence_ref') != receipt.get('evidence_ref')):
                raise ValueError('job is not bound to its native training receipt')
            repo = Path(request['repo']).resolve(strict=True)
            if not repo.is_relative_to(root):
                raise ValueError('producer checkout is not run-owned')
            reviewed = revalidate_detached(receipt, path=receipt_path, root=root, repo=repo)
            if not reviewed:
                raise ValueError('inspect_native_job must independently audit/revalidate training progress')
            output = Path(receipt['output_directory']).resolve(strict=True)
            if not output.is_dir() or not output.is_relative_to(root/'research'/run_id):
                raise ValueError('producer output escaped its run namespace')
            artifact = receipt.get('artifact') or {}
            if not (artifact.get('checked') and artifact.get('freshness_checked') and artifact.get('matched',0)>0):
                raise ValueError('native output lacks fresh artifact verification')
            entries, entry_count = [], 0
            for item in sorted(output.rglob('*')):
                if item.is_symlink() or not item.resolve().is_relative_to(output):
                    continue
                entry_count += 1
                if len(entries)<96:
                    entries.append({'ref':item.relative_to(root).as_posix(),
                        'kind':'directory' if item.is_dir() else 'file',
                        'bytes':item.stat().st_size if item.is_file() else None})
            from .native_jobs import source_matches
            try:
                source_current = source_matches(root,repo,request)
            except (OSError,ValueError,TypeError,KeyError):
                source_current = False
            producers.append({'status':'available','origin':'native_job',
                'formal_adoption_source_status':'current' if source_current else 'historical_source_requires_resolution',
                'formal_adoption_ready':False,
                'formal_adoption_requirements':'explicit settings/argv, current source identity, frozen protocol and loaded-policy rollout; availability only proves historical production',
                'producer_id':request['job_id'],'job_id':request['job_id'],
                'output_ref':output.relative_to(root).as_posix(),
                'native_evidence_ref':receipt['evidence_ref'],
                'progress':reviewed['training_progress'],
                'training_progress_revalidation_ref':reviewed['training_progress_revalidation_ref'],
                'entries':entries,'entry_count':entry_count,'entries_truncated':entry_count>len(entries)})
        except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
            pending.append({'job_id':path.parent.name,'because':str(exc)[:300],
                            'next_operation':'inspect_native_job'})
    if not producers:
        return {**verification,'producers':[],'pending_producers':pending,
                'jobs_scan_truncated':len(paths)>128}
    return {**(verification if verification['status']=='available' else producers[0]),
        'status':'available','producers':producers,'pending_producers':pending,
        'jobs_scan_truncated':len(paths)>128,
        'instruction':'Agent chooses an explicit checkpoint_ref within ONE listed producer, '
            'using native save/load source. No latest/largest or best-policy default. '
            'Long-job production is not yet a scored baseline/candidate; retain its receipt '
            'and use explicit adoption for formal scoring. Pending producers require '
            'inspect_native_job, not retraining or storage GC authorization.'}


def select(root: Path, ref: str, offered: dict) -> dict:
    if offered.get('status') != 'available' or not isinstance(ref, str) or not ref:
        raise ValueError('checkpoint_ref requires a currently verified producer catalog')
    relative = Path(ref)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('checkpoint_ref must be a confined run-relative path')
    path = (root/relative).resolve(strict=True)
    producers = offered.get('producers') or [offered]
    matches = [row for row in producers
               if path.is_relative_to((root/row['output_ref']).resolve(strict=True))]
    if len(matches) != 1:
        raise ValueError('checkpoint_ref must identify exactly one verified native producer output')
    offered = matches[0]
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
