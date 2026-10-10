"""Immutable operator study intent, separate from native benchmark settings."""
from pathlib import Path

from .common import atomic_json, object_digest, read_json


def persist(output: Path, source: Path | None):
    output = Path(output)
    target = output/'research_brief.json'
    if target.is_symlink():
        raise ValueError('unsafe research brief record')
    saved = read_json(target) if target.exists() else None
    if source is None:
        if saved is not None:
            view(output)  # Validate on resume, never silently discard old intent.
        return
    source = Path(source)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 24000:
        raise ValueError('research brief must be a regular UTF-8 text file <=24KB')
    text = source.read_text(encoding='utf-8')
    if not text.strip() or len(text) > 6000:
        raise ValueError('research brief must contain 1..6000 characters')
    if saved is not None:
        current = view(output)
        if current['text'] != text:
            raise ValueError('research brief is frozen; use a new study run for changed intent')
        return
    if (output/'budget.json').exists() or (output/'run_state.json').exists():
        raise ValueError('cannot retrofit a research brief into an initialized run')
    atomic_json(target, {'schema_version': 1, 'text': text, 'sha256': object_digest(text),
                        'authority': 'operator study intent; not benchmark settings, evidence or new permissions'})


def view(output: Path):
    path = Path(output)/'research_brief.json'
    if not path.exists() and not path.is_symlink():
        return {'status': 'not_supplied'}
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 30000:
        raise ValueError('unsafe research brief record')
    row = read_json(path)
    if (not isinstance(row, dict) or not isinstance(row.get('text'), str)
            or not 1 <= len(row['text']) <= 6000 or object_digest(row['text']) != row.get('sha256')):
        raise ValueError('research brief identity changed')
    return {**row, 'status': 'frozen', 'record_ref': 'research_brief.json'}
