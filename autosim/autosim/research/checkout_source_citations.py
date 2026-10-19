"""Run-owned source handles survive privacy projection without exposing host paths."""
import re
import uuid
from pathlib import Path

from .common import atomic_text, digest, immutable_json, now, object_digest, read_json, sanitize_model_payload


def capture(output, repo, request):
    if isinstance(request, dict) and 'producer_evidence_id' in request:
        from .producer_source_history import capture as capture_history
        return capture_history(output, repo, request)
    from .baseline_reference import checkout_source
    from .evidence_store import capture_attempt_evidence
    from .workspace_resources import bindings_for
    if not isinstance(request, dict) or set(request) != {'ref'}:
        raise ValueError('checkout_source_request requires only ref=file[:line[-line]]')
    output, repo = Path(output).resolve(), Path(repo).resolve()
    path, text, canonical, sha = checkout_source(repo, request['ref'])
    if path.suffix.lower() not in {'.py', '.pyi', '.sh', '.yaml', '.yml', '.toml', '.json', '.md', '.rst', '.cfg', '.ini'}:
        raise ValueError('capture public source/configuration, not datasets or binary artifacts')
    if any(path.is_relative_to(repo / row['target']) for row in bindings_for(repo)):
        raise ValueError('checkout source capture cannot read a bound dataset/checkpoint/resource')
    manifest = output / 'workspace_snapshot.json'
    if manifest.is_file():
        if manifest.is_symlink():
            raise ValueError('unsafe checkout source manifest')
        entries = read_json(manifest).get('source_entries')
        if entries is not None and path.relative_to(repo).as_posix() not in {
                row[1] for row in entries if isinstance(row, list) and len(row) > 1 and row[0] == 'file'}:
            raise ValueError('source capture requires a file in the isolated source manifest, not a generated data artifact')
    if not text or len(text) > 4000:
        raise ValueError('capture a nonempty source line range of at most 4000 characters')
    directory = output / 'checkout_source_citations'
    if directory.is_symlink():
        raise ValueError('unsafe checkout source store')
    identity = uuid.uuid4().hex
    snapshot = directory / (identity + '.source')
    atomic_text(snapshot, text)
    if digest(path) != sha:
        raise ValueError('checkout source changed during capture')
    displayed = sanitize_model_payload({'quote': text}, local_roots=(repo, output))['quote']
    row = {'schema_version': 1, 'id': identity, 'at': now(), 'repo': str(repo),
           'source_ref': canonical, 'source_sha256': sha, 'display_quote': displayed,
           'snapshot_ref': snapshot.relative_to(output).as_posix(), 'snapshot_sha256': digest(snapshot)}
    row['sha256'] = object_digest(row)
    immutable_json(directory / (identity + '.json'), row)
    proof = capture_attempt_evidence(output, attempt_id=identity, log=snapshot,
        receipt_ref=f'checkout_source_citations/{identity}.json', status='checkout_source_captured',
        returncode=None, termination_reason='read_only_source_capture')
    return {**proof, 'excerpt': displayed, 'ref': 'checkout-source:' + identity, 'quote': displayed,
            'guidance': 'Copy this exact ref/quote pair into source_refs. The executor retains original '
                        'bytes and revalidates them locally; never reconstruct a redacted path.',
            'authority': 'source bytes only; not scientific approval or simulation readiness'}


def resolve(output, repo, ref, quote):
    from .baseline_reference import checkout_source
    from .experiment_transaction import safe
    if not isinstance(ref, str) or not re.fullmatch(r'checkout-source:[0-9a-f]{32}', ref):
        raise ValueError('invalid checkout source handle')
    identity = ref.split(':')[1]
    metadata = safe(output, 'checkout_source_citations/' + identity + '.json')
    if not metadata.is_file() or metadata.stat().st_size > 32000:
        raise ValueError('checkout source handle is missing or exceeds bounds')
    row = read_json(metadata)
    if (row.get('schema_version') != 1 or row.get('id') != identity or
            row.get('repo') != str(Path(repo).resolve()) or
            row.get('sha256') != object_digest({k: v for k, v in row.items() if k != 'sha256'})):
        raise ValueError('checkout source handle identity changed')
    _, text, _, sha = checkout_source(repo, row['source_ref'])
    snapshot = safe(output, row['snapshot_ref'])
    if (snapshot != safe(output, 'checkout_source_citations/' + identity + '.source') or
            sha != row['source_sha256'] or digest(snapshot) != row['snapshot_sha256'] or
            snapshot.read_text(encoding='utf-8') != text):
        raise ValueError('checkout source/snapshot changed; capture and review current evidence')
    if not isinstance(quote, str) or not quote or quote not in (text, row['display_quote']):
        raise ValueError('copy the exact quote returned with this checkout source ID; subquotes require a new bounded capture')
    return {'ref': ref, 'quote': text, 'sha256': sha}


def resolve_frozen_intent(output, repo, ref, quote, original_sha):
    """Restore reviewed pre-search intent, not approve mutable candidate source.

    Only the reservation restorer calls this after verifying the original archive,
    frozen comparison protocol and every protected file. It requires authentic
    isolated-base provenance and identical cited bytes; regular resolve remains
    strict for new proposals, including changes outside the quoted line range.
    """
    from .baseline_reference import checkout_source
    from .experiment_transaction import safe
    from .workspace_snapshot import _validated_base
    if not isinstance(ref, str) or not re.fullmatch(r'checkout-source:[0-9a-f]{32}', ref):
        raise ValueError('invalid frozen source handle')
    identity = ref.split(':')[1]
    metadata = safe(output, 'checkout_source_citations/' + identity + '.json')
    if not metadata.is_file() or metadata.stat().st_size > 32000:
        raise ValueError('missing or oversized frozen source handle')
    row = read_json(metadata)
    if (row.get('schema_version') != 1 or row.get('id') != identity or
            row.get('repo') != str(Path(repo).resolve()) or
            row.get('source_sha256') != original_sha or
            row.get('sha256') != object_digest({k: v for k, v in row.items() if k != 'sha256'})):
        raise ValueError('frozen source handle identity changed')
    path, text, _, _ = checkout_source(repo, row['source_ref'])
    manifest = read_json(safe(output, 'workspace_snapshot.json'))
    if Path(str(manifest.get('destination') or '')).resolve() != Path(repo).resolve():
        raise ValueError('frozen source belongs to a different checkout')
    base = _validated_base(manifest).get(path.relative_to(Path(repo).resolve()).as_posix())
    if not base or base[0] != 'file' or base[3] != original_sha:
        raise ValueError('authentic isolated-base source provenance is unavailable')
    snapshot = safe(output, 'checkout_source_citations/' + identity + '.source')
    if (row.get('snapshot_ref') != snapshot.relative_to(output).as_posix() or
            snapshot.stat().st_size > 16000 or digest(snapshot) != row['snapshot_sha256'] or
            snapshot.read_text(encoding='utf-8') != text or quote not in (text, row['display_quote'])):
        raise ValueError('cited reservation bytes changed; no frozen-intent recovery')
    return {'ref': ref, 'quote': text, 'sha256': original_sha}
