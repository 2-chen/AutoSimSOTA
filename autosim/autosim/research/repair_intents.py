"""Avoid repeating a paid recovery turn for the same evidence and executable inputs."""
from pathlib import Path
from .common import atomic_json, digest, object_digest, read_json


def begin(output, *, failure, commands, probes, manifests, observation, proposal=None,
          repo=None, resource_identity=None):
    meaningful = None if proposal is None else {k:v for k,v in proposal.items()
        if k != 'reasoning'}
    source = None
    if repo is not None:
        from .source_tracking import materialized_files
        try:
            source = {str(path): digest(Path(repo) / path) for path in materialized_files(Path(repo))}
        except (OSError, ValueError) as exc:
            # Large/unreadable repositories must not crash the repair path. A bounded
            # scan failure is not evidence of absence; explicit revised proposals remain
            # available when automatic source-change discovery cannot certify a change.
            source = {'status':'scan_incomplete', 'reason':str(exc)[:500]}
    identity = object_digest({'version':2, 'failure_id':failure.get('evidence_id'),
        'operation':failure.get('template') or failure.get('probe') or failure.get('command'),
        'commands':commands, 'probes':probes, 'manifests':manifests,
        'environment':observation.get('identity') or observation.get('reason'),
        'source':source, 'resources':resource_identity,
        'proposal':meaningful, 'framework':{name:digest(Path(__file__).with_name(name))
            for name in ('provision.py', 'agent_runtime.py', 'process_executor.py', 'repair_intents.py')}})
    root = Path(output) / 'repair_intents'
    path = root / (identity + '.json')
    if root.is_symlink() or path.is_symlink():
        raise ValueError('unsafe repair intention path')
    previous = read_json(path) if path.exists() else {}
    if previous:
        return identity, False
    atomic_json(path, {'identity':identity, 'status':'started',
        'failure_evidence_id':failure.get('evidence_id'),
        'authority':'duplicate_recovery_guard_not_resource_boundary'})
    return identity, True


def finish(output, identity, status):
    path = Path(output) / 'repair_intents' / (identity + '.json')
    value = read_json(path)
    atomic_json(path, {**value, 'status':status})
