"""Read provider receipts in a UTC-aware [start, end) window without repricing.

No model/native calls, no ledger updates, no raw prompt storage, no held-out reads.
Identical turn/index receipts are counted once; conflicting copies are quarantined.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('timezone-aware ISO timestamp required')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('timezone-aware ISO timestamp required')
    return result.astimezone(timezone.utc)


def _read(path):
    if (any(p.is_symlink() for p in (path, *path.parents)) or
            not path.is_file() or path.stat().st_size > 8 * 1024**2):
        raise ValueError('unsafe, absent or oversized receipt')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('object receipt required')
    return value


def _money(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _new_group():
    return dict(requests=0, requests_with_usage=0, requests_without_usage=0,
        input_hit_tokens=0, input_miss_tokens=0, output_tokens=0,
        known_cost_usd=0.0, requests_without_cost=0, unknown_reserved_usd=0.0,
        requests_below_half_hit=0, cost_of_below_half_hit_requests_usd=0.0,
        requests_at_output_limit=0, cost_of_output_limit_requests_usd=0.0,
        profiles_observed=0, _turns=set(), _systems=set(), _tools=set(), _first_messages=set())


def _finish(group):
    hit, miss = group['input_hit_tokens'], group['input_miss_tokens']
    group['cache_hit_ratio'] = hit / (hit + miss) if hit + miss else None
    group['turns'] = len(group.pop('_turns'))
    group['distinct_system_prefixes'] = len(group.pop('_systems'))
    group['distinct_tool_schemas'] = len(group.pop('_tools'))
    group['distinct_first_messages'] = len(group.pop('_first_messages'))
    return group


def report(output, start, end):
    start, end = timestamp(start), timestamp(end)
    if start >= end:
        raise ValueError('start must precede end')
    root = Path(output).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir():
        raise ValueError('existing non-symlink run root required')
    workers = root / 'agent_workers'
    roots = [root]
    skipped = []
    if not workers.is_symlink():
        roots.extend(p for p in sorted(workers.glob('*')) if p.is_dir() and not p.is_symlink())
    elif workers.exists():
        skipped.append('agent_workers:symlink')
    requests, conflicts, duplicates, undated = {}, set(), 0, 0
    for base in roots:
        roles = {}
        for path in (base / 'agent/processes').glob('*.json'):
            try:
                record = _read(path)
                role = record.get('role')
                roles[record['attempt_id']] = (role if isinstance(role, str) and
                    re.fullmatch('[a-z][a-z0-9_-]{0,31}', role) else 'unknown')
            except (OSError, ValueError, TypeError, KeyError):
                continue
        for path in sorted((base / 'agent/pricing').glob('*.json')):
            ref = path.relative_to(root).as_posix()
            try:
                record = _read(path)
                turn = record.get('turn_id')
                if not isinstance(turn, str) or not re.fullmatch('[0-9a-f]{32}', turn):
                    raise ValueError('invalid turn identity')
                receipts = (record.get('gateway') or {}).get('receipts', [])
                if not isinstance(receipts, list):
                    raise ValueError('invalid receipt list')
                for index, receipt in enumerate(receipts):
                    if not isinstance(receipt, dict):
                        skipped.append(f'{ref}#{index}:invalid_receipt')
                        continue
                    key = (turn, index)
                    normalized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
                    previous = requests.get(key)
                    if key in conflicts:
                        continue
                    if previous:
                        if previous['encoded'] == normalized:
                            duplicates += 1
                            if previous['role'] != roles.get(turn, 'unknown'):
                                previous['role'] = 'unknown_conflicting_role'
                        else:
                            conflicts.add(key)
                            requests.pop(key)
                        continue
                    requests[key] = {'receipt': receipt, 'encoded': normalized,
                        'role': roles.get(turn, 'unknown'), 'source_ref': ref}
            except (OSError, ValueError, TypeError, AttributeError):
                skipped.append(ref + ':unreadable_or_invalid')
    totals, roles, details = _new_group(), {}, []
    for (turn, index), stored in requests.items():
        receipt, role = stored['receipt'], stored['role']
        try:
            at = timestamp(receipt.get('at'))
        except (ValueError, TypeError):
            undated += 1
            continue  # Cannot allocate an undated cost into this window.
        if not start <= at < end:
            continue
        usage = receipt.get('usage')
        valid_usage = (isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
            for k in ('input_hit_tokens', 'input_miss_tokens', 'output_tokens')))
        priced = receipt.get('status') == 'priced' and _money(receipt.get('cost_usd'))
        cost = receipt['cost_usd'] if priced else None
        hit_ratio, at_limit = None, False
        profile = receipt.get('request_profile')
        if valid_usage:
            size = usage['input_hit_tokens'] + usage['input_miss_tokens']
            hit_ratio = usage['input_hit_tokens'] / size if size else None
            limit = profile.get('max_output_tokens') if isinstance(profile, dict) else None
            at_limit = type(limit) is int and limit > 0 and usage['output_tokens'] >= limit
        for group in (totals, roles.setdefault(role, _new_group())):
            group['requests'] += 1
            group['_turns'].add(turn)
            group['known_cost_usd'] += cost if cost is not None else 0
            group['requests_without_cost'] += not priced
            if receipt.get('status') == 'unknown' and _money(receipt.get('reserved_usd')):
                group['unknown_reserved_usd'] += receipt['reserved_usd']
            group['requests_with_usage' if valid_usage else 'requests_without_usage'] += 1
            if valid_usage:
                for key in ('input_hit_tokens', 'input_miss_tokens', 'output_tokens'):
                    group[key] += usage[key]
            if hit_ratio is not None and hit_ratio < .5:
                group['requests_below_half_hit'] += 1
                group['cost_of_below_half_hit_requests_usd'] += cost or 0
            if at_limit:
                group['requests_at_output_limit'] += 1
                group['cost_of_output_limit_requests_usd'] += cost or 0
            if isinstance(profile, dict):
                group['profiles_observed'] += 1
                for field, bucket in [('system_sha256', '_systems'), ('tools_sha256', '_tools'),
                                      ('first_message_sha256', '_first_messages')]:
                    value = profile.get(field)
                    if isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value):
                        group[bucket].add(value)
        details.append({'turn_id': turn, 'request_index': index, 'role': role,
            'at': receipt['at'], 'source_ref': stored['source_ref'], 'known_cost_usd': cost,
            'cache_hit_ratio': hit_ratio, 'output_at_limit': at_limit,
            'output_tokens': usage['output_tokens'] if valid_usage else None})
    return {'schema_version': 1, 'observed_at': datetime.now(timezone.utc).isoformat(),
        'window': {'start_inclusive': start.isoformat(),
        'end_exclusive': end.isoformat()}, 'basis': 'stored_provider_receipts_no_repricing',
        'scope': 'this run and direct agent_workers only; not full inherited ledger or provider invoice',
        'worker_roots_included': len(roots) - 1, 'totals': _finish(totals),
        'roles': {k: _finish(v) for k, v in sorted(roles.items())},
        'duplicate_copies_excluded': duplicates,
        'conflicting_requests_excluded': [{'turn_id': t, 'request_index': i} for t, i in sorted(conflicts)],
        'undated_receipts_unallocated': undated, 'skipped_records': skipped,
        'highest_known_cost_requests': sorted(details, key=lambda r: r['known_cost_usd'] or 0,
                                              reverse=True)[:10],
        'cost_per_valid_experiment_usd': None,
        'limitations': ['Missing usage, quarantined and undated records are not zero cost.',
            'Unknown holds are not settled expenditure; totals do not change ledger authority.',
            'A low cache ratio or output-limit observation alone does not identify compaction or waste.',
            'Late receipt files can change coverage of an old window; this is not an atomic settlement snapshot.',
            'No verified experiment denominator or causal before/after comparison in this billing window.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    args = parser.parse_args(argv)
    try:
        value = report(args.output, args.start, args.end)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
