"""Bounded operator diagnostics, advisory only: never scores or repair authority."""
import re
from pathlib import Path

from .common import read_json, sanitize_model_text
from .evidence_store import read_attempt_evidence


def view(output: Path, run_id: str) -> list[dict]:
    output = Path(output).resolve()
    directory = output / 'monitor_findings'
    if directory.is_symlink() or not directory.is_dir():
        return []
    paths = []
    for index, path in enumerate(directory.glob('*.json')):
        if index >= 512:
            return []  # An incomplete inventory must not pretend to be the latest.
        try:
            if not path.is_symlink():
                paths.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    rows = []
    for _, path in sorted(paths)[-8:]:
        try:
            if path.is_symlink() or path.stat().st_size > 16384:
                continue
            row = read_json(path)
            if (row.get('schema_version') != 1 or row.get('run_id') != run_id
                    or row.get('origin') != 'operator_diagnostic'
                    or not re.fullmatch(r'[0-9a-f]{32}', str(row.get('native_attempt_id') or ''))):
                continue
            receipt = output / 'research' / run_id / 'attempts' / row['native_attempt_id'] / 'receipt.json'
            if (receipt.is_symlink() or not receipt.is_file()
                    or not receipt.resolve().is_relative_to(output)
                    or receipt.resolve() != receipt
                    or receipt.stat().st_size > 8 * 1024 * 1024):
                continue
            native = read_json(receipt)
            if native.get('attempt_id') != row['native_attempt_id'] or native.get('run_id') != run_id:
                continue
            proof = read_attempt_evidence(output, row['evidence_id'], limit=3000)
            if (proof['status'] != 'operator_diagnostic'
                    or proof['receipt_ref'] != path.relative_to(output).as_posix()
                    or proof.get('artifact') != {'native_attempt_id': row['native_attempt_id']}):
                continue
            rows.append({'origin': 'operator_diagnostic',
                         'authority': 'advisory reproduction, not native outcome, score or permission; independently inspect before repair',
                         'native_attempt_id': row['native_attempt_id'],
                         'evidence_id': row['evidence_id'],
                         'summary': sanitize_model_text(str(row.get('summary') or ''))[:700],
                         'excerpt': proof['text'],
                         'evidence_ref': f"evidence/{row['evidence_id']}.json"})
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return rows[-3:]
