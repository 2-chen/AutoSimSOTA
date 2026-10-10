"""Keep audited source interventions alive across repeatable native staging.

Staging is allowed to recreate glue, but not silently turn a code candidate back
into its preimage. Only an exact audited preimage can be repaired automatically;
unrelated edits fail closed. Historical metrics remain untouched.
"""
import hashlib
import re
from pathlib import Path

from .codepatch import apply
from .common import read_json


def preserve(repo: Path, held: dict) -> dict:
    rows = []
    for name, (patch, before) in held.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or
                patch.file != name or not relative.parts):
            raise ValueError(f'audited candidate source path is unsafe: {name}')
        expected = before.replace(patch.find, patch.replace, 1)
        path = Path(repo) / name
        parent = Path(repo)
        for part in relative.parts[:-1]:
            parent = parent / part
            if parent.is_symlink():
                raise ValueError(f'audited candidate source parent is unsafe: {name}')
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'audited candidate source is missing or unsafe: {name}')
        actual = path.read_text(encoding='utf-8')
        restored = False
        if actual != expected:
            if actual != before:
                raise ValueError(f'staging changed candidate source beyond its audited preimage: {name}')
            apply(patch, repo=repo)  # Validates relative paths, parents and unique patch again.
            restored = True
        if path.read_text(encoding='utf-8') != expected:
            raise ValueError(f'candidate source changed during staging repair: {name}')
        rows.append({'file': name, 'expected_sha256': hashlib.sha256(expected.encode()).hexdigest(),
                     'reapplied_after_staging': restored})
    return {'status': 'verified' if rows else 'not_required', 'files': rows}


def historical_mismatches(root: Path) -> list[dict]:
    """Compare recorded code-candidate snapshots to the source actually measured.

This is an additive diagnosis, not an edit to old scores or a claim about tensor
contents. Missing legacy source manifests are not fabricated into positive proof.
"""
    snapshot_path = root / 'snapshots/snapshots.json'
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        return []
    try:
        snapshots = {row['name']: row.get('files') or {} for row in
                     read_json(snapshot_path).get('snapshots') or [] if isinstance(row, dict)}
        problems = []
        for transaction in sorted((root / 'code_changes').glob('round-*.json'))[-128:]:
            if transaction.is_symlink():
                continue
            match = re.fullmatch(r'round-([1-9][0-9]*)\.json', transaction.name)
            if not match:
                continue
            number = int(match[1])
            measurement_path = root / 'measurements' / f'round_{number}.json'
            if measurement_path.is_symlink() or not measurement_path.is_file():
                continue
            transaction_row = read_json(transaction)
            expected = snapshots.get(f'round-{number}', {})
            measurement = read_json(measurement_path)
            actual = (measurement.get('source_state') or {}).get('overlay_files') or {}
            for patch in transaction_row.get('patches') or []:
                name = patch.get('file')
                wanted, measured = expected.get(name), actual.get(name)
                if (isinstance(wanted, str) and isinstance(measured, str)
                        and re.fullmatch('[a-f0-9]{64}', wanted)
                        and re.fullmatch('[a-f0-9]{64}', measured) and wanted != measured):
                    problems.append({'label': f'round_{number}', 'idea': transaction_row.get('idea_label'),
                        'file': name, 'expected_sha256': wanted, 'measured_sha256': measured,
                        'status': 'candidate_source_mismatch',
                        'because': 'native measurement did not retain the audited source intervention; '
                                   'do not infer its hypothesis failed or count it as a valid code candidate'})
        return problems
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return [{'status': 'unverifiable', 'because': 'candidate source history could not be audited'}]
