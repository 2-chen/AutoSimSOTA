"""Keep source-review caveats visible after a successful data registration.

Training permission, structural validity and task-success/label quality are
different claims. This view copies the retained review; it does not interpret
free prose, certify quality, choose an experiment or change a data card.
"""
from pathlib import Path
import re

from .common import digest, read_json


def latest(output, catalog):
    root = Path(output) / 'data_versions'
    if root.is_symlink():
        return None
    candidates = []
    for item in catalog:
        ident = item.get('id') if isinstance(item,dict) else None
        if not isinstance(ident,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',ident):
            continue
        path = root / (ident + '.json')
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 256*1024:
                continue
            if digest(path) != item.get('record_sha256'):
                continue
            row = read_json(path)
            if (row.get('id') != ident or row.get('status') != 'reviewed_not_consumed'
                    or digest(path) != item.get('record_sha256')):
                continue
            candidates.append((path.stat().st_mtime_ns,ident,row))
        except (OSError,ValueError,TypeError):
            continue
    if not candidates:
        return None
    _, ident, row = max(candidates,key=lambda entry:(entry[0],entry[1]))
    profile = (row.get('asset') or {}).get('profile') or {}
    return {'data_version_id':ident, 'data_card_ref':f'data_versions/{ident}.json',
        'review':row.get('review') or {},
        'manifest_unverified':profile.get('manifest') or {},
        'scope':'source-backed permission to train and observed format, not a quality/success certificate',
        'instruction':'Read the full review, including caveats, before claiming a campaign quality '
            'floor. Recorder-committed episodes or a successful process do not by themselves '
            'prove native task success, expert labels, useful diversity or learning benefit. '
            'This view does not authorize an experiment or change the frozen comparison.'}
