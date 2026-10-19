"""Source-reviewed pre-search reservation; never weakens held-out enforcement."""
import json
from pathlib import Path

from .baseline_reference import source_evidence
from .common import atomic_json, immutable_json, now, object_digest, read_json


EVALUATION_PROTOCOL_KEYS = frozenset({'task', 'tasks', 'suite', 'benchmark', 'seed', 'seeds',
    'episodes', 'n_episodes', 'horizon', 'max_episode_steps', 'initial_states', 'eval_seed'})


def unrecognized_protocol_keys(confirmation, protocol_keys):
    exact = EVALUATION_PROTOCOL_KEYS | set(protocol_keys)
    return sorted(key for key in confirmation
                  if key not in exact and not key.startswith(('eval.', 'evaluation.')))


def with_reviewed_intent(declaration, confirmation, protocol_keys, *, replaces_confirmation=None):
    """Compile one explicit reviewed replacement, never an implicit mismatch bypass."""
    goal = dict(declaration.get('research_goal') or {})
    previous = goal.get('confirmation')
    if previous and previous != confirmation and previous != replaces_confirmation:
        raise ValueError('declaration disagrees with the reviewed reservation')
    goal['confirmation'] = json.loads(json.dumps(confirmation, allow_nan=False))
    goal['protocol_keys'] = list(dict.fromkeys([*(goal.get('protocol_keys') or []), *protocol_keys]))
    return {**declaration, 'research_goal':goal}


def search_started(output, run_id):
    root = Path(output)/'research'/run_id
    from .protocol_revision import boundary
    try:
        boundary(root)
        return False
    except (OSError,ValueError,TypeError,KeyError,RuntimeError):
        # Detached evaluator attempts count even before a measurement is written.
        return True


def validate(confirmation, reason, protocol_keys, *, require_protocol_keys=False):
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
    outside = unrecognized_protocol_keys(confirmation, protocol_keys)
    if require_protocol_keys and outside:
        raise ValueError('native reservation keys require explicit protocol_keys before review: ' +
                         ', '.join(outside))


def view(output, repo, run_id):
    """Expose the sealed pre-search intent, not a held-out outcome or new approval.

    A stale dependency citation must not hide the parameters that Main/Objective
    need to re-review. Returning the original intent never makes stale sources
    current, changes the declaration, or reopens search after finalization.
    """
    from .experiment_transaction import safe
    path = Path(output) / 'confirmation_reservation.json'
    if not path.exists() and not path.is_symlink():
        return {'status': 'not_registered'}
    try:
        path = safe(output, 'confirmation_reservation.json')
        if path.stat().st_size > 128 * 1024:
            raise ValueError('confirmation reservation exceeds bounds')
        row = read_json(path)
        unsigned = dict(row)
        sha = unsigned.pop('sha256', None)
        if row.get('run_id') != run_id or object_digest(unsigned) != sha:
            raise ValueError('confirmation reservation identity changed')
        archive_ref = f'confirmation_reservations/{sha}.json'
        archive = safe(output, archive_ref)
        if archive.stat().st_size > 128 * 1024 or read_json(archive) != row:
            raise ValueError('original confirmation reservation archive changed')
        validate(row['confirmation'], row['reason'], row['protocol_keys'])
        refs = row['source_refs']
        if (not isinstance(refs, list) or not 1 <= len(refs) <= 12 or any(
                not isinstance(s, dict) or not isinstance(s.get('ref'), str) or
                len(s['ref']) > 1024 or not isinstance(s.get('quote'), str) or
                not 1 <= len(s['quote']) <= 4000 for s in refs)):
            raise ValueError('original confirmation source inventory exceeds bounds')
        errors = []
        for index, citation in enumerate(refs):
            try:
                source_evidence(repo, [citation], output=output)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append({'index': index, 'ref': citation['ref'],
                               'because': str(exc)[:500]})
        return {'status': 'sources_require_refresh' if errors else 'registered',
                'confirmation': row['confirmation'], 'protocol_keys': row['protocol_keys'],
                'reason': row['reason'], 'source_refs': refs, 'registered_at': row['at'],
                'sha256': sha, 'archive_ref': archive_ref, 'source_errors': errors,
                'presearch_revision_allowed': not search_started(output, run_id),
                'authority': 'immutable original intent, not current source approval or held-out results',
                'next': 'Preserve the original conditions; capture fresh source citations and request '
                        'independent pre-search review if needed. Never infer that this view grants execution.'}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {'status': 'invalid', 'because': str(exc)[:500],
                'authority': 'no recoverable authenticated intent; no execution approval'}


def record(output, repo, run_id, proposal, review):
    if search_started(output, run_id):
        raise ValueError('reservation must precede the first search/session/measurement')
    validate(proposal['confirmation'], proposal['reason'], proposal['protocol_keys'],
             require_protocol_keys=True)
    replaced = proposal.get('replaces_confirmation')
    if replaced is not None and not isinstance(replaced, dict):
        raise ValueError('replaced reservation must be an explicit prior declaration object')
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
    try:
        restored = source_evidence(repo, row['source_refs'], output=output)
    except ValueError as original_error:
        try:
            restored = _frozen_intent_sources(output, repo, run_id, row)
        except (OSError, ValueError, TypeError, KeyError):
            raise original_error
    if restored != row['source_refs']:
        raise ValueError('reviewed confirmation sources changed; re-review before search')
    return with_reviewed_intent(declaration, row['confirmation'], row['protocol_keys'],
                               replaces_confirmation=row.get('replaces_confirmation'))


def _frozen_intent_sources(output, repo, run_id, row):
    """Legal optimization must not invalidate the historical reservation intent.

    No new split, source approval or score: restore only the exact archived intent,
    with unchanged quoted bytes and still-frozen evaluator. Evaluation admission
    independently verifies current frozen baseline/candidate and disjoint rollout.
    """
    from .experiment_transaction import safe
    from .protocol_revision import effective
    from .common import digest
    root = Path(output) / 'research' / run_id
    comparison = read_json(safe(output, f'research/{run_id}/comparison_protocol.json'))
    archive = safe(output, f"confirmation_reservations/{row['sha256']}.json")
    if (read_json(archive) != row or comparison.get('confirmation') != row['confirmation']):
        raise ValueError('frozen comparison and original reservation archive must agree')
    protected_path, _ = effective(root)
    protected = read_json(protected_path).get('hashes')
    if not isinstance(protected, dict) or not protected:
        raise ValueError('protected evaluator identity unavailable')
    for name, sha in protected.items():
        path = Path(name)
        if not path.is_absolute() or not path.resolve().is_relative_to(Path(repo).resolve()):
            raise ValueError('protected evaluator escaped its checkout')
        relative = path.relative_to(Path(repo).resolve()).as_posix()
        if digest(safe(repo, relative)) != sha:
            raise ValueError('protected evaluator changed; cannot restore frozen intent')
    restored = []
    for citation in row['source_refs']:
        try:
            restored.extend(source_evidence(repo, [citation], output=output))
        except ValueError:
            from .checkout_source_citations import resolve_frozen_intent
            restored.append(resolve_frozen_intent(output, repo, citation.get('ref'),
                                                  citation.get('quote'), citation.get('sha256')))
    return restored
