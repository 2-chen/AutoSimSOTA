"""Source-reviewed pre-search reservation; never weakens held-out enforcement."""
import json
from pathlib import Path

from .baseline_reference import source_evidence
from .common import atomic_json, immutable_json, now, object_digest, read_json


def search_started(output, run_id):
    root = Path(output)/'research'/run_id
    return ((root/'controller_session.json').exists() or (root/'comparison_protocol.json').exists()
            or any((root/'measurements').glob('*.json'))
            or any((root/'confirmation_attempts').glob('*.json')))


def validate(confirmation, reason, protocol_keys):
    if (not isinstance(confirmation, dict) or not 1 <= len(confirmation) <= 8
            or any(not isinstance(k, str) or not k.strip() or len(k) > 100 for k in confirmation)):
        raise ValueError('confirmation must map 1..8 native evaluation keys to reserved values')
    if len(json.dumps(confirmation, allow_nan=False).encode()) > 4000:
        raise ValueError('confirmation settings exceed bounds')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError('a concise source-backed reservation reason is required')
    if (not isinstance(protocol_keys, list) or len(protocol_keys) > 8
            or any(not isinstance(k, str) or k not in confirmation for k in protocol_keys)
            or len(set(protocol_keys)) != len(protocol_keys)):
        raise ValueError('extra protocol_keys must be distinct keys in this reservation')


def record(output, repo, run_id, proposal, review):
    if search_started(output, run_id):
        raise ValueError('reservation must precede the first search/session/measurement')
    validate(proposal['confirmation'], proposal['reason'], proposal['protocol_keys'])
    citations = source_evidence(repo, proposal['source_refs'], output=output)
    pointer = Path(output)/'confirmation_reservation.json'
    archive = Path(output)/'confirmation_reservations'
    if pointer.is_symlink() or archive.is_symlink():
        raise ValueError('unsafe confirmation reservation registry')
    if any(review.get(k) is not True for k in (
            'native_evaluation_input_supported', 'disjoint_conditions_supported',
            'preserves_policy_and_metric', 'source_connection_supported')):
        raise ValueError('independent review did not establish a genuine evaluation reservation')
    cited = review.get('citations')
    if (not isinstance(cited, list) or not cited or any(not isinstance(c, dict) or not any(
            c.get('ref') == s['ref'] and c.get('quote') == s['quote'] for s in citations) for c in cited)):
        raise ValueError('reservation review needs exact supplied source citations')
    row = {**proposal, 'source_refs': citations, 'review': review, 'run_id': run_id,
           'schema_version': 1, 'at': now(),
           'authority': 'reviewed pre-search intent; actual disjoint rollout still needs verification'}
    row['sha256'] = object_digest(row)
    immutable_json(archive/(row['sha256']+'.json'), row)
    atomic_json(pointer, row)
    return row


def apply(output, repo, run_id, declaration):
    path = Path(output)/'confirmation_reservation.json'
    if not path.exists() and not path.is_symlink():
        return declaration
    if path.is_symlink() or path.stat().st_size > 128*1024:
        raise ValueError('unsafe confirmation reservation')
    row = read_json(path)
    unsigned = dict(row)
    sha = unsigned.pop('sha256', None)
    if row.get('run_id') != run_id or object_digest(unsigned) != sha:
        raise ValueError('confirmation reservation identity changed')
    if source_evidence(repo, row['source_refs'], output=output) != row['source_refs']:
        raise ValueError('reviewed confirmation sources changed; re-review before search')
    goal = dict(declaration.get('research_goal') or {})
    if goal.get('confirmation') and goal['confirmation'] != row['confirmation']:
        raise ValueError('declaration disagrees with the reviewed reservation')
    goal['confirmation'] = row['confirmation']
    goal['protocol_keys'] = list(dict.fromkeys([*(goal.get('protocol_keys') or []), *row['protocol_keys']]))
    return {**declaration, 'research_goal': goal}
