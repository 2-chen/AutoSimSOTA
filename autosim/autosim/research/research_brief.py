"""Immutable operator study intent, separate from native benchmark settings."""
from pathlib import Path

from .common import atomic_json, object_digest, read_json


def persist(output: Path, source: Path | None, *, require_data_comparison: bool | None = None):
    output = Path(output)
    target = output/'research_brief.json'
    if target.is_symlink():
        raise ValueError('unsafe research brief record')
    saved = read_json(target) if target.exists() else None
    if require_data_comparison is not None and type(require_data_comparison) is not bool:
        raise ValueError('data comparison requirement must be explicit Boolean intent')
    if saved is not None:
        current = view(output)
        required = (current.get('requirements') or {}).get('data_comparison', False)
        if require_data_comparison is not None and required != require_data_comparison:
            raise ValueError('research brief requirements are frozen; use a new study run')
    if source is None:
        if saved is not None:
            view(output)  # Validate on resume, never silently discard old intent.
        elif require_data_comparison:
            raise ValueError('a required data comparison needs a new --research-brief')
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
    if (output/'run_state.json').exists():
        raise ValueError('cannot retrofit a research brief into an initialized run')
    if (output/'budget.json').exists():
        from .run_successor import assert_pristine_bootstrap
        assert_pristine_bootstrap(output)
    row = {'schema_version': 1, 'text': text, 'sha256': object_digest(text),
           'authority': 'operator study intent; not benchmark settings, evidence or new permissions'}
    if require_data_comparison is not None:
        requirements = {'data_comparison':require_data_comparison}
        row.update(requirements=requirements, requirements_sha256=object_digest(requirements))
    atomic_json(target, row)


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
    if 'requirements' in row or 'requirements_sha256' in row:
        requirements = row.get('requirements')
        if (not isinstance(requirements, dict) or set(requirements) != {'data_comparison'} or
                type(requirements['data_comparison']) is not bool or
                row.get('requirements_sha256') != object_digest(requirements)):
            raise ValueError('research brief requirements identity changed')
    return {**row, 'status': 'frozen', 'record_ref': 'research_brief.json'}
