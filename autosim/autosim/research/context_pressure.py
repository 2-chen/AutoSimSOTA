"""Bounded, cache-aware session rotation using actual provider usage, not characters."""
from __future__ import annotations


def rotation_decision(receipt: dict) -> dict:
    """Keep a bounded warm prefix without overriding local runtime headroom.

    This is an engineering heuristic, not a hit-rate promise. It never extends model,
    GPU or wall budgets. Protocol/capability/transport invalidation is handled first
    by the caller and cannot be overridden by a favorable cache statistic.
    """
    usage = receipt.get('usage') or {}
    names = ('input_hit_tokens', 'input_miss_tokens', 'output_tokens')
    values = [usage.get(name, 0) for name in names]
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
        return {'rotate': True, 'reason': 'invalid provider context usage'}
    hit, miss, output = values
    size = hit + miss
    ratio = hit / size if size else None
    warm = (receipt.get('status') == 'priced' and ratio is not None and ratio >= .90
            and miss <= 12_000 and output <= 8_192)
    # Provider capacity/cache pricing does not describe the CLI's compaction
    # behavior. A highly cached ~139k-input compaction request actually failed
    # to return a usable summary. Never grant warm histories a 240k exemption;
    # reserve room for the next controller request, tools and generated output.
    # These are conservative engineering thresholds, not provider context claims.
    limit = 120_000 if warm else 96_000
    return {'rotate': size >= limit, 'input_tokens': size, 'input_hit_ratio': ratio,
            'soft_limit_tokens': 96_000, 'effective_limit_tokens': limit,
            'local_input_ceiling_tokens':120_000,
            'reason': ('bounded warm-prefix reuse' if warm else
                       'rehydrate cold, costly or oversized context from current evidence')}
