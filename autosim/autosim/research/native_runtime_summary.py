"""Small, receipt-bound native timings for Agent planning; never budget charges."""

import math
import re
from pathlib import Path

from .common import read_json
from .evidence_store import read_attempt_evidence


def stage_runtime(root: Path, measurement: dict, stage: str) -> dict | None:
    """Expose only a completed train/evaluate attempt linked by the measurement.

    A native executor's `seconds` is observed process elapsed time, not charged
    GPU-seconds, proof of convergence, or a prediction for another configuration.
    Missing, malformed, unsealed or oversized evidence leaves the timing unknown.
    """
    if stage not in ('train', 'evaluate') or not isinstance(measurement, dict):
        return None
    try:
        root = Path(root).resolve(strict=True)
        if root.parent.name != 'research':
            return None
        output = root.parent.parent
        row = measurement.get(stage)
        if not isinstance(row, dict) or type(row.get('returncode')) is not int or row['returncode'] != 0:
            return None
        attempt = row.get('attempt_id')
        if not isinstance(attempt, str) or not re.fullmatch(r'[0-9a-f]{32}', attempt):
            return None
        if row.get('evidence_id') != attempt:
            return None
        receipt = root/'attempts'/attempt/'receipt.json'
        # Resolving inside the run is not enough: reject aliases to other attempts.
        expected = receipt.relative_to(output)
        if (receipt.is_symlink() or not receipt.is_file()
                or receipt.resolve(strict=True).relative_to(output) != expected
                or receipt.stat().st_size > 8 * 1024 * 1024):
            return None
        native = read_json(receipt)
        if (not isinstance(native, dict) or native.get('run_id') != root.name
                or native.get('attempt_id') != attempt or native.get('node_id') != stage
                or native.get('status') != 'completed'
                or type(native.get('returncode')) is not int or native['returncode'] != 0
                or native.get('termination_reason') != 'normal_exit'):
            return None
        seconds = native.get('seconds')
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or seconds <= 0):
            return None
        document = output/'evidence'/f'{attempt}.json'
        if (document.is_symlink() or not document.is_file()
                or document.resolve(strict=True) != document
                or document.stat().st_size > 256 * 1024):
            return None
        sealed = read_json(document)
        log_bytes = sealed.get('log_bytes') if isinstance(sealed, dict) else None
        # Keep this optional planning view cheap; agents can inspect larger logs by ID.
        if type(log_bytes) is not int or not 0 <= log_bytes <= 2 * 1024 * 1024:
            return None
        relative = Path(str(sealed.get('log_ref') or ''))
        if relative.is_absolute() or '..' in relative.parts:
            return None
        log = output/relative
        if (log.is_symlink() or not log.is_file() or log.resolve(strict=True) != log
                or log.stat().st_size != log_bytes):
            return None
        proof = read_attempt_evidence(output, attempt, limit=1)
        if (proof.get('receipt_ref') != expected.as_posix()
                or proof.get('status') != 'completed'
                or type(proof.get('returncode')) is not int or proof['returncode'] != 0
                or proof.get('termination_reason') != 'normal_exit'):
            return None
        return {'attempt_id': attempt, 'seconds': seconds,
                'receipt_ref': expected.as_posix(),
                'evidence_ref': f'evidence/{attempt}.json',
                'reused': row.get('reused') is True,
                'meaning': 'reported native process elapsed seconds; not GPU charges, '
                           'convergence, or a forecast for changed settings'}
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError):
        return None
