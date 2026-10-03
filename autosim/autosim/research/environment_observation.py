"""Sealed, cached metadata from the selected interpreter; never an install decision."""
from pathlib import Path
import json

from .common import atomic_json, digest, object_digest, read_json


CODE = """import json, importlib.metadata as m, importlib.util as u
rows=[]
for d in sorted(m.distributions(), key=lambda x: x.metadata.get('Name','').lower()):
    name=d.metadata.get('Name','')
    hints=(d.read_text('top_level.txt') or '').splitlines()
    if not hints:
        hints=sorted({str(f).split('/')[0].removesuffix('.py') for f in (d.files or [])
                      if str(f).split('/')[0].removesuffix('.py').isidentifier()})
    rows.append({'distribution':name,'version':d.version,
                 'import_names':[h for h in hints if h.isidentifier()][:16]})
print(json.dumps({'packages':rows[:512], 'complete':len(rows)<=512,
    'authority':'metadata_only_not_import_or_consumer_readiness'}))
"""


def observe(output: Path, repo: Path) -> dict:
    """Inspect once per metadata/context identity; reuse a hash-verified native receipt."""
    from .native_context import load_context
    from .evidence_store import read_attempt_evidence
    try:
        context = load_context(output, repo)
        python = Path(context['interpreter'])
        if not python.is_file():
            return {'status': 'unavailable', 'reason': 'selected interpreter does not exist'}
        prefix = python.parent.parent
        stamps = []
        for base in sorted((prefix / 'lib').glob('python*/site-packages')):
            for path in sorted(base.glob('*.dist-info/*')):
                if path.name not in {'METADATA', 'top_level.txt', 'RECORD', 'direct_url.json'} or not path.is_file():
                    continue
                stat = path.stat()
                stamps.append((str(path.relative_to(prefix)), stat.st_size, stat.st_mtime_ns,
                               stat.st_ctime_ns))
        config = prefix / 'pyvenv.cfg'
        identity = object_digest({'context':context['identity'], 'metadata':stamps,
                                  'interpreter':python.stat().st_mtime_ns,
                                  'venv_config':digest(config) if config.is_file() else None,
                                  'implementation':digest(Path(__file__))})
        cache = output / 'environment_observation.json'
        previous = read_json(cache) if cache.is_file() and not cache.is_symlink() else {}
        if previous.get('identity') == identity and previous.get('evidence_id'):
            read_attempt_evidence(output, previous['evidence_id'], limit=1)
            return previous
        from .agent_runtime import _McpExecutor
        from .budget import RunBudget
        budget = RunBudget.existing(output)
        timeout = min(30, budget.remaining()) if budget else 30
        if timeout <= 0:
            return {'status':'unavailable', 'reason':'hard wall budget exhausted; no diagnostic launched',
                    'authority':'diagnostic_unavailable_not_package_absence'}
        executor = _McpExecutor(workspace=repo, output=output, allow_commands=False, allow_native=True)
        receipt = executor.native_probe({'code':CODE,
            'purpose':'Inspect actual installed distributions and import-name metadata; no imports, install or score.',
            'timeout_seconds':timeout})
        result = {'identity':identity, 'status':'unavailable',
                  'evidence_id':receipt['evidence_id'], 'evidence_ref':receipt['evidence_ref'],
                  'authority':'metadata_only_not_environment_readiness'}
        if receipt.get('returncode') == 0:
            value = json.loads(receipt['stdout'])
            result.update(status='observed', packages=value['packages'], complete=value['complete'])
        else:
            result['reason'] = str(receipt.get('stderr') or '')[:1000]
        atomic_json(cache, result)
        return result
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {'status':'unavailable', 'reason':f'{type(exc).__name__}: {exc}'[:1000],
                'authority':'diagnostic_unavailable_not_package_absence'}


def prompt_view(observation: dict, context: str = '') -> dict:
    """Small recommendation view; the sealed receipt retains all collected metadata."""
    import re
    tokens = set(re.findall(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', context.lower()))
    normalize = lambda name: name.lower().replace('_', '-')
    tokens = {normalize(token) for token in tokens}
    rows = observation.get('packages') or []
    relevant = [row for row in rows if any(normalize(name) in tokens for name in
        [row.get('distribution', ''), *(row.get('import_names') or [])])]
    selected = relevant[:48]
    return {**{k:v for k,v in observation.items() if k != 'packages'},
            'packages':selected, 'observed_package_count':len(rows),
            'projection_complete':len(selected) == len(rows),
            'selection':'lexical recommendation only; read sealed evidence for other packages; '
                        'omitted packages are not evidence of absence'}


def cached_view(output: Path, context: str = '') -> dict:
    """Expose cached leads to Scheduler without starting any native operation."""
    from .evidence_store import read_attempt_evidence
    path = output / 'environment_observation.json'
    if not path.is_file() or path.is_symlink():
        return {'status':'not_observed', 'authority':'not_environment_readiness'}
    try:
        value = read_json(path)
        read_attempt_evidence(output, value['evidence_id'], limit=1)
        return {**prompt_view(value, context), 'freshness':'cached_metadata; revalidate before relying on readiness'}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {'status':'unavailable', 'reason':str(exc)[:500],
                'authority':'diagnostic_unavailable_not_package_absence'}
