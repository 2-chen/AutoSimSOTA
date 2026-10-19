"""Read-only installation facts for recently measured candidate patches.

An archived transaction's `applied` records a past event, not the live checkout.
This view grants no patch/replay authority and never changes an old measurement.
"""
import re
from pathlib import Path

from .common import digest, read_json
from .experiment_transaction import safe
from .codepatch import patches_from_change


def view(output: Path, repo: Path, run_id: str) -> list[dict]:
    output, repo = Path(output).resolve(), Path(repo).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', run_id) or not repo.is_relative_to(output):
        return []
    base = f'research/{run_id}'
    try:
        directory = safe(output, base+'/measurements')
        index = safe(output, base+'/snapshots/snapshots.json')
        if not index.is_file() or index.stat().st_size > 2*1024*1024:
            return []
        snapshots = read_json(index).get('snapshots') or []
        paths = sorted((p for p in directory.glob('round_*.json') if not p.is_symlink()),
            key=lambda p:(p.stat().st_mtime_ns,p.name))[-32:]
    except (OSError, ValueError, TypeError, AttributeError):
        return []
    result, seen = [], set()
    for path in reversed(paths):
        try:
            if path.stat().st_size > 2*1024*1024:
                continue
            measured = read_json(path)
            match = re.fullmatch(r'round_([1-9][0-9]*)(?:_handoff_correction)?', path.stem)
            if (not match or measured.get('label') != path.stem or measured.get('ok') is not True
                    or measured.get('confirmation') or match[1] in seen):
                continue
            change_ref = base+'/code_changes/round-'+match[1]+'.json'
            change_path = safe(output, change_ref)
            if change_path.stat().st_size > 256*1024:
                continue
            change = read_json(change_path)
            patches = patches_from_change({'patches':change.get('patches')})
            if not patches or change.get('status') != 'applied' or change.get('round') != int(match[1]):
                continue
            old = [s for s in snapshots if s.get('name') == change.get('snapshot')]
            new = [s for s in snapshots if s.get('name') == 'round-'+match[1]]
            if len(old) != 1 or len(new) != 1 or len({p.file for p in patches}) != len(patches):
                continue
            files = []
            for patch in patches:
                if Path(patch.file).is_absolute() or '..' in Path(patch.file).parts or '\\' in patch.file:
                    raise ValueError('unsafe recorded source path')
                before, after = old[0]['files'][patch.file], new[0]['files'][patch.file]
                blobs = []
                for key in (before, after):
                    if not isinstance(key,str) or not re.fullmatch(r'[0-9a-f]{64}',key):
                        raise ValueError('source blob identity unavailable')
                    blob = safe(output, base+'/snapshots/blobs/'+key)
                    if blob.stat().st_size > 65536 or digest(blob) != key:
                        raise ValueError('source blob changed or exceeds observation bound')
                    blobs.append(blob.read_text())
                if (blobs[0].count(patch.find) != 1 or
                        blobs[0].replace(patch.find,patch.replace,1) != blobs[1]):
                    raise ValueError('recorded patch does not reproduce source postimage')
                target = safe(output, repo.relative_to(output).as_posix()+'/'+patch.file)
                if target.is_file() and target.stat().st_size > 65536:
                    raise ValueError('current source exceeds observation bound')
                current = digest(target) if target.is_file() else None
                files.append({'file':patch.file,'before_sha256':before,'postimage_sha256':after,
                    'current_sha256':current,
                    'status':('matches_postimage' if current == after else
                        'matches_preimage_not_postimage' if current == before else
                        'missing' if current is None else 'differs_from_both')})
            seen.add(match[1])
            result.append({'measurement_ref':path.relative_to(output).as_posix(),
                'measurement_sha256':digest(path),'code_change_ref':change_ref,
                'code_change_sha256':digest(change_path),'recorded_transaction_status':change['status'],
                'files':files,'all_changed_files_match_postimage':all(
                    f['status']=='matches_postimage' for f in files),
                'authority':'bounded source/snapshot observation only; recorded success is not '
                    'a new independent audit, full-source equivalence or installed-repair certificate',
                'rule':'An archived applied transaction and a successful measurement do NOT install '
                    'their patch permanently. Inspect current hashes. If a needed earlier repair '
                    'is absent, Main must explicitly compose it with a NEW audited candidate from '
                    'the current preimage; a parameter-only proposal does not inherit reverted code. '
                    'Do not auto-apply patches, rewrite history, reset the baseline or weaken evidence.'})
            if len(result) == 3:
                break
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue  # Missing/corrupt history is not an installed-repair claim.
    return result
