"""Additive correction of scored handoffs, without retraining or rewriting history."""
import re
from contextlib import contextmanager
from pathlib import Path

from .common import atomic_json, digest, now, object_digest, read_json
from .training_progress_audit import candidates


def _sources_available(root, label):
    """Do not advertise a restoration the executor already knows is unsafe."""
    from types import SimpleNamespace
    from .snapshot import Snapshots
    session_path = root/'controller_session.json'
    if not session_path.exists():
        return True
    session = read_json(session_path)
    research = SimpleNamespace(run_root=root, repo=Path(session['repository']),
                               snapshots=Snapshots(root/'snapshots'))
    _candidate_source_plan(research, label)
    return True


def eligible(root):
    root = Path(root)
    offered = []
    for path in sorted((root/'measurements').glob('*.json')):
        if path.is_symlink():
            continue
        try:
            measurement = read_json(path)
            unscored = (measurement.get('ok') is False and
                        measurement.get('status') == 'unscored' and
                        measurement.get('where') in {'train','policy_artifact'} and
                        not measurement.get('evaluate') and not measurement.get('policy_artifact') and
                        not measurement.get('zero_work_reported'))
            if (not measurement.get('ok') and not unscored) or measurement.get('handoff_correction'):
                continue
            attempt = str((measurement.get('train') or {}).get('attempt_id') or '')
            if not re.fullmatch(r'[a-f0-9]{32}', attempt):
                continue
            folder = root/'attempts'/attempt
            receipt_path, selection_path = folder/'receipt.json', folder/'artifact_selection.json'
            if receipt_path.is_symlink() or selection_path.is_symlink():
                continue
            receipt, selected = read_json(receipt_path), read_json(selection_path)
            if unscored:
                if (receipt.get('attempt_id') != attempt or receipt.get('status') != 'completed'
                        or receipt.get('returncode') != 0
                        or receipt.get('termination_reason') != 'normal_exit'
                        or selected.get('attempt_id') != attempt
                        or selected.get('status') not in {'abstained','unavailable'}):
                    continue
                label = str(measurement.get('label') or '')
                if not re.fullmatch(r'round_[1-9][0-9]*', label) or path.name != label+'.json':
                    continue
                corrected_label = label+'_handoff_correction'
                completed = root/'measurements'/f'{corrected_label}.json'
                if completed.is_symlink() or (completed.exists() and read_json(completed).get('ok')):
                    continue
                _sources_available(root, label)
                offered.append({'label':label, 'training_attempt_id':attempt,
                    'kind':'completed_unscored_training', 'corrected_label':corrected_label,
                    'why':'native training completed; explicitly re-audit progress and resolve '
                          'the existing policy before any retraining. Eligibility is advisory; '
                          'execution repeats receipt/settings/protocol/source checks.'})
                continue
            progress = receipt.get('training_progress') or {}
            counter = (progress.get('evidence') or {}).get('native_counter') or {}
            if ((progress.get('evidence') or {}).get('counter_selection') or {}).get('kind') == 'native_best':
                continue
            if (receipt.get('status') != 'completed' or receipt.get('returncode') != 0
                    or selected.get('status') != 'selected'
                    or selected.get('attempt_id') != attempt
                    or selected.get('answer', {}).get('selection_basis') != 'verified_completed_work_container'
                    or not counter.get('field')):
                continue
            output = Path(receipt['output_directory']).resolve(strict=True)
            if not output.is_relative_to(root.resolve()):
                continue
            old_source = Path((measurement.get('policy_artifact') or {}).get('source') or '').resolve()
            selected_path = Path(selected['selected_path'])
            if not selected_path.is_absolute():
                selected_path = output/selected_path
            if old_source != selected_path.resolve():
                continue
            docs = candidates(output, float(receipt['started_epoch']))
            later = [item['value'] for doc in docs for item in doc['fields']
                     if item['field'] == counter['field'] and item['value'] > counter['value']]
            if not later:
                continue
            label = str(measurement['label'])
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,55}', label) or path.name != label+'.json':
                continue
            corrected_label = label+'_handoff_correction'
            completed = root/'measurements'/f'{corrected_label}.json'
            if completed.is_symlink():
                continue
            if completed.exists() and read_json(completed).get('ok'):
                continue
            _sources_available(root, label)
            offered.append({'label':label, 'training_attempt_id':attempt,
                'selected_completed_work':counter['value'], 'other_counter_max':max(later),
                'why':'selected positive-progress proof omitted larger counters; independently re-audit native terminal/best semantics',
                'corrected_label':corrected_label})
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return offered


def _candidate_source_plan(research, label):
    """Restore only the exact archived intervention, and undo it after recovery.

    Reject unrelated edits and non-patch snapshot drift before any write. The held
    patch guard also keeps native staging from silently erasing the intervention.
    """
    from .codepatch import patches_from_change
    match = re.fullmatch(r'round_([1-9][0-9]*)', label)
    transaction = research.run_root/'code_changes'/f'round-{match.group(1)}.json' if match else None
    if transaction is None or not transaction.exists():
        return []
    if transaction.is_symlink():
        raise ValueError('unsafe candidate source transaction')
    row = read_json(transaction)
    patches = patches_from_change({'patches':row.get('patches')})
    before = research.snapshots.get(str(row.get('snapshot') or ''))
    after = research.snapshots.get('round-'+match.group(1))
    if not patches or before is None or after is None or row.get('status') != 'applied':
        raise ValueError('candidate source snapshots/transaction unavailable')
    by_name = {p.file:p for p in patches}
    for name, key in after.files.items():
        target = research.repo/name
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or target.is_symlink()
                or any(parent.is_symlink() for parent in target.parents
                       if parent.is_relative_to(research.repo))):
            raise ValueError('unsafe candidate source path')
        if name in by_name:
            old_key = before.files.get(name, '')
            blob = research.snapshots.blobs/old_key
            if not re.fullmatch(r'[a-f0-9]{64}',old_key) or blob.is_symlink() or digest(blob)!=old_key:
                raise ValueError('candidate preimage blob is not verified')
            original = blob.read_text()
            patch = by_name[name]
            if original.count(patch.find)!=1:
                raise ValueError('candidate patch preimage ambiguous')
            import hashlib
            if hashlib.sha256(original.replace(patch.find,patch.replace,1).encode()).hexdigest()!=key:
                raise ValueError('candidate postimage differs from archived snapshot')
            if digest(target)!=old_key:
                raise ValueError('candidate source changed beyond its recorded rollback')
        elif digest(target)!=key:
            raise ValueError('inherited candidate source differs from its recorded snapshot')
    if not set(by_name)<=set(after.files):
        raise ValueError('candidate patch omitted from archived snapshot')
    return patches


@contextmanager
def candidate_sources(research, label):
    from .codepatch import apply_many, revert
    patches = _candidate_source_plan(research, label)
    if not patches:
        yield
        return
    originals = apply_many(patches,repo=research.repo)
    prior = dict(research._held)
    research._held.update({p.file:(p,originals[p.file]) for p in patches})
    try:
        yield
    finally:
        for patch in reversed(patches):
            revert(patch,originals[patch.file],repo=research.repo)
        research._held = prior


def correct(research, *, label):
    if getattr(research, 'execution_graph', None):
        return {'ok':False, 'why':'graph handoff correction requires an explicit verified '
                'reuse edge; no graph stages started'}
    row = next((row for row in eligible(research.run_root) if row['label'] == label), None)
    if row is None:
        return {'ok':False, 'why':'no receipt-bound incomplete handoff is offered for that label'}
    original_path = research.run_root/'measurements'/f'{label}.json'
    original = read_json(original_path)
    original_hash = digest(original_path)
    attempt = row['training_attempt_id']
    folder = research.run_root/'attempts'/attempt
    selected = folder/'artifact_selection.json'
    reference = selected.relative_to(research.output).as_posix()
    state_path = folder/'handoff_correction.json'
    if state_path.is_symlink():
        return {'ok':False,'why':'unsafe correction state path'}
    if state_path.exists():
        saved = read_json(state_path)
        if saved.get('status') == 'revalidating':
            return {'ok':False, 'why':'previous correction outcome is unknown; reconcile '
                    'its selection/evaluation receipts before starting another operation'}
    state = {'at':now(), 'original_measurement_ref':original_path.relative_to(research.output).as_posix(),
             'original_measurement_sha256':original_hash, 'original_selection_sha256':digest(selected),
             'training_receipt_sha256':digest(folder/'receipt.json'),
             'training_attempt_id':attempt, 'training_reused':True,
             'corrected_label':row['corrected_label'], 'status':'revalidating'}
    atomic_json(state_path, state)
    try:
        with candidate_sources(research,label):
            result = research.measure(settings=original['settings'], label=row['corrected_label'],
                _reuse_training_attempt_id=attempt,
                **({'_audit_training_progress':True} if row.get('kind')=='completed_unscored_training'
                   else {'_selection_correction_ref':reference}))
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
        atomic_json(state_path,{**state,'status':'source_or_receipt_refused',
                              'error':f'{type(exc).__name__}: {exc}'[:1200]})
        return {'ok':False,'why':f'completed training recovery refused: {exc}',
                'handoff_correction':{**state,'status':'unscored'}}
    if digest(original_path) != original_hash:
        raise ValueError('original measurement changed during corrective evaluation')
    result['handoff_correction'] = {**state, 'status':'measured' if result.get('ok') else 'unscored',
                                    'baseline_and_incumbent_unchanged':True}
    if result.get('ok'):
        atomic_json(research.run_root/'measurements'/f"{row['corrected_label']}.json", result)
    atomic_json(state_path, {**result['handoff_correction'], 'result_digest':object_digest(result)})
    research._append_local_event({'at':now(), 'event':'policy_handoff_correction',
        'label':label, 'corrected_label':row['corrected_label'], 'training_reused':True,
        'ok':bool(result.get('ok')), 'metric_value':result.get('metric_value')})
    return result
