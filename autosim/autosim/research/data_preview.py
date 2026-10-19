"""Bounded previews of reviewed native data, using the existing Recorder request queue."""
from pathlib import Path
import shutil
import time

from .common import digest, read_json, now
from .data_lineage import _record
from .data_quality import validate_manifest
from .data_health import PROFILE_GENERATORS
from .data_versions import resolve
from .frozen_policy_trial import _safe
from .media_manifest import MEDIA_SUFFIXES, _preview


def capture(output, repo, *, version_id, card_sha256, timeout_seconds=60):
    output = Path(output).resolve(strict=True)
    started = time.monotonic()
    card = _safe(output, f'data_versions/{version_id}.json')
    if not card.is_file() or digest(card) != card_sha256:
        raise ValueError('data preview version changed since the Recorder request')
    _, bindings = resolve(output, Path(repo), {'preview': 'data-version:' + version_id})
    binding = bindings[0]
    source = Path(binding['source'])
    witness = (binding.get('asset') or {}).get('profile_witness') or {}
    _, receipt = _record(output, witness.get('receipt_ref'), max_bytes=2 * 1024**2)
    if (receipt.get('data_profile_generator') not in PROFILE_GENERATORS
            or receipt.get('gpu_access') is not False or receipt.get('repository_source_stable') is not True
            or receipt.get('data_profile_status') or
            ('diagnostic_data_binding' in receipt and receipt.get('diagnostic_data_binding_stable') is not True)):
        raise ValueError('data preview requires a sealed native manifest, not a model video path')
    manifest = validate_manifest(receipt.get('data_profile', {}).get('episode_manifest'), root=source)
    rows = []
    for episode in manifest['episodes']:
        if not episode['committed'] or episode['format_valid'] is not True:
            continue
        candidates = sorted((name, sha) for name, sha in episode['files'].items()
                            if Path(name).suffix.lower() in MEDIA_SUFFIXES)
        for name, sha in candidates:
            path = _safe(source, name)
            if path.stat().st_size > 32 * 1024**2:
                continue
            if time.monotonic() - started >= timeout_seconds:
                break
            target = _safe(output, f'report/data_previews/{version_id}/{sha[:24]}{path.suffix.lower()}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != sha:
                    raise ValueError('existing data preview identity changed')
            else:
                # Copy only a small preview, never an entire raw pool or a mutable link.
                temporary = _safe(output, str(target.relative_to(output)) + '.partial')
                if temporary.exists():
                    raise ValueError('interrupted preview copy needs reconciliation')
                shutil.copyfile(path, temporary)
                if digest(temporary) != sha:
                    raise ValueError('native video changed during preview copy')
                temporary.replace(target)
            relative = target.relative_to(output).as_posix()
            row = {'path': relative, 'sha256': sha, 'bytes': target.stat().st_size,
                   'data_version_id': version_id, 'episode_id': episode['id'],
                   'source_relative_path': name, 'manifest_sha256': binding['asset']['profile']['manifest']['sha256']}
            if timeout_seconds - (time.monotonic() - started) >= 20:
                row['preview'] = _preview(output, relative, sha)
            rows.append(row)
            break  # one camera per episode; sample is not a representative outcome set
        if len(rows) >= 3 or time.monotonic() - started >= timeout_seconds:
            break
    resolve(output, Path(repo), {'preview': 'data-version:' + version_id})
    if digest(card) != card_sha256:
        raise ValueError('data preview registration changed during capture')
    return {'kind': 'dataset_preview', 'status': 'captured' if rows else 'unavailable',
            'why': 'bounded native episode video sample' if rows else 'no bounded committed native videos',
            'at': now(), 'data_version_id': version_id, 'card_sha256': card_sha256,
            'seconds': round(time.monotonic()-started, 3), 'gpu_access': False,
            'preview_bytes': sum(row['bytes'] for row in rows), 'media': rows,
            'selection_rule': 'first bounded committed episodes, one native camera each; not representative sampling',
            'authority': 'reviewed dataset examples; no new policy rollout, score or expert-quality certification'}


def view(output):
    """Short verified preview projection; do not rehash raw datasets on heartbeat."""
    output = Path(output).resolve(strict=True)
    result = []
    try:
        store = _safe(output, 'report/demo_requests.json')
        if not store.is_file() or store.stat().st_size > 2 * 1024**2:
            return result
        requests = read_json(store).get('requests') or []
        for request in requests[-64:]:
            event = request.get('result') or {}
            if (request.get('kind') != 'dataset_preview' or request.get('status') != 'captured'
                    or event.get('kind') != 'dataset_preview' or event.get('gpu_access') is not False):
                continue
            card = _safe(output, f'data_versions/{request["data_version_id"]}.json')
            if digest(card) != event.get('card_sha256'):
                continue
            for row in event.get('media', [])[:3]:
                path = _safe(output, row['path'])
                if path.stat().st_size > 32 * 1024**2 or digest(path) != row.get('sha256'):
                    continue
                shown = {**row, 'purpose': request['purpose'], 'selection_rule': event['selection_rule']}
                preview = row.get('preview') or {}
                if preview.get('status') == 'decoded_frame':
                    image = _safe(output, preview['path'])
                    if image.stat().st_size <= 2 * 1024**2 and digest(image) == preview['sha256']:
                        shown['preview_ref'] = preview['path']
                result.append(shown)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return result
    return result
