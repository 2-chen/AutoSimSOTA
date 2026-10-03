"""Explicit operator-only reduction; never reset historical usage or wall start time."""
from pathlib import Path
import argparse
import json
import math

from .common import atomic_json, now, object_digest, read_json
from .agent_budget import AgentCostLedger
from .task_budget import TaskGPUBudget


def reduce(output: Path, *, factor: float, reason: str, model_scope: str = 'remaining',
           preserve_stopped_holds: bool = False) -> dict:
    return _amend(output, factor=factor, reason=reason, model_scope=model_scope,
                  preserve_stopped_holds=preserve_stopped_holds)


def extend(output: Path, *, model_usd: float, wall_seconds: float, reason: str,
           preserve_stopped_holds: bool = False) -> dict:
    """Operator-authorized addition; no usage, GPU cap or start-time reset."""
    if (not math.isfinite(model_usd) or model_usd < 0 or
            not math.isfinite(wall_seconds) or wall_seconds < 0 or
            model_usd + wall_seconds <= 0):
        raise ValueError('extension requires finite nonnegative additions, at least one positive')
    return _amend(output, factor=1.0, reason=reason, model_scope='extension',
                  preserve_stopped_holds=preserve_stopped_holds,
                  model_usd=model_usd, wall_seconds=wall_seconds)


def _amend(output: Path, *, factor: float, reason: str, model_scope: str,
           preserve_stopped_holds: bool, model_usd: float = 0, wall_seconds: float = 0) -> dict:
    extension = model_scope == 'extension'
    if not math.isfinite(factor) or not (factor == 1 if extension else 0 < factor < 1) or not reason.strip() or model_scope not in {'total', 'remaining', 'extension'}:
        raise ValueError('explicit reduction requires 0<factor<1, reason and total/remaining scope')
    output = Path(output).resolve(strict=True)
    request = {'factor':factor, 'reason':reason, 'model_scope':model_scope}
    if extension:
        request.update(model_add_usd=model_usd, wall_add_seconds=wall_seconds)
    request_id = object_digest(request)
    path = output/'budget_amendments'/f'{request_id}.json'
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError('unsafe amendment directory')
    if path.is_file():
        previous = read_json(path)
        if previous.get('status') == 'applied':
            # Compatibility with an already applied operator amendment: derive the
            # historical prefix length only while its full entry digest still matches.
            if 'model_entry_count' not in previous['preserved']:
                raw = read_json(output/'agent/cost_ledger.json')
                if object_digest(raw['entries']) == previous['preserved']['model_entries_digest']:
                    previous['preserved']['model_entry_count'] = len(raw['entries'])
                    atomic_json(path, previous)
            return previous
        raise ValueError('unfinished budget amendment requires operator reconciliation; do not halve twice')
    from .process_executor import inspect_process_identity
    for process_path in [*output.glob('processes/*.json'), *output.glob('agent/processes/*.json')]:
        if process_path.is_symlink():
            raise ValueError('unsafe process receipt')
        row = read_json(process_path)
        identity = row.get('process_identity') or row.get('identity')
        if identity and inspect_process_identity(identity).get('status') != 'not_running':
            raise ValueError('stop/reconcile recorded processes before budget amendment')
    paths = [output/'budget.json', output/'task_gpu_budget.json', output/'agent/cost_ledger.json',
             output/'agent/runtime_spec.json']
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError('budget amendment requires existing safe task, wall, ledger and runtime records')
    raw = read_json(paths[2])
    ledger = AgentCostLedger(output, run_id=raw['run_id'], limit_usd=raw['limit_usd'],
                             cost_basis=raw.get('cost_basis', 'cli_reported'))
    gpu = TaskGPUBudget(output)
    with ledger._locked(), gpu.locked() as gpu_record:
        raw = ledger._read()
        spent, held = ledger._totals(raw)
        if (not preserve_stopped_holds and any(row['status'] == 'reserved' for row in raw['entries'])) or any(
                row['status'] == 'active' for row in gpu_record['leases'].values()):
            raise ValueError('live/unreconciled reservations forbid budget amendment')
        wall, spec = read_json(paths[0]), read_json(paths[3])
        if spec['total_budget_usd'] != raw['limit_usd']:
            raise ValueError('runtime and model ledger limits disagree')
        available = max(0.0, raw['limit_usd'] - spent - held)
        limit = (raw['limit_usd'] + model_usd if extension else
                 raw['limit_usd']*factor if model_scope == 'total' else spent + held + available*factor)
        amendment = {'id':request_id, 'status':'prepared', 'factor':factor,
            'reason':reason, 'at':now(), 'model_scope':model_scope,
            'previous_amendment_id':raw.get('budget_amendment_id'),
            'stopped_legacy_holds_preserved':preserve_stopped_holds,
            'before':{'wall_seconds':wall['wall_seconds'], 'gpu_seconds':gpu_record['cap_seconds'],
                'model_usd':raw['limit_usd'], 'turn_usd':spec['turn_budget_usd']},
            'after':{'wall_seconds':wall['wall_seconds'] + wall_seconds if extension else wall['wall_seconds']*factor, 'gpu_seconds':gpu_record['cap_seconds']*factor,
                'model_usd':limit, 'turn_usd':spec['turn_budget_usd']*factor},
            'preserved':{'wall_started_epoch':wall['started_epoch'],
                'charged_gpu_seconds':gpu_record['charged_seconds'],
                'model_entries_digest':object_digest(raw['entries']), 'model_entry_count':len(raw['entries']),
                'spent_usd':spent, 'reserved_usd':held},
            'model_available_before':available, 'model_available_after':max(0.0, limit-spent-held)}
        if extension:
            amendment.update(model_add_usd=model_usd, wall_add_seconds=wall_seconds)
        atomic_json(path, amendment)
        # Each write is atomic. A crash between them is fail-closed: frozen runtime/ledger
        # validation refuses inconsistent limits; prepared evidence retains all old limits.
        wall.update(wall_seconds=amendment['after']['wall_seconds'], budget_amendment_id=amendment['id'])
        gpu_record.update(cap_seconds=amendment['after']['gpu_seconds'], budget_amendment_id=amendment['id'])
        raw.update(limit_usd=limit, budget_amendment_id=amendment['id'])
        spec.update(total_budget_usd=limit, turn_budget_usd=amendment['after']['turn_usd'], budget_amendment_id=amendment['id'])
        for target, value in zip(paths, (wall, gpu_record, raw, spec)):
            atomic_json(target, value)
        amendment['status'] = 'applied'
        atomic_json(path, amendment)
    from .budget import RunBudget
    RunBudget.existing(output).record()
    return amendment


def _verified_amendment(output: Path, identity: str, ledger: dict, wall: dict) -> dict:
    """Validate an audit record and its unchanged historical ledger prefix."""
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('invalid amendment identity')
    path = output/'budget_amendments'/f'{identity}.json'
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('unsafe amendment record')
    amendment = read_json(path)
    if amendment.get('status') != 'applied' or amendment.get('id') != identity:
        raise ValueError('unapplied amendment')
    keys = ('factor','reason','model_scope')
    extension = amendment.get('model_scope') == 'extension'
    if extension:
        keys += ('model_add_usd', 'wall_add_seconds')
    if object_digest({key:amendment[key] for key in keys}) != identity:
        raise ValueError('changed authorization')
    before, after, preserved = amendment['before'], amendment['after'], amendment['preserved']
    factor = amendment['factor']
    if not (factor == 1 if extension else 0 < factor < 1):
        raise ValueError('invalid amendment factor')
    if extension:
        model_add, wall_add = amendment['model_add_usd'], amendment['wall_add_seconds']
        if not (math.isfinite(model_add) and model_add >= 0 and
                math.isfinite(wall_add) and wall_add >= 0 and model_add + wall_add > 0):
            raise ValueError('invalid extension')
        expected = {**before, 'model_usd':before['model_usd'] + model_add,
                    'wall_seconds':before['wall_seconds'] + wall_add}
    else:
        if amendment['model_scope'] not in {'total', 'remaining'}:
            raise ValueError('invalid reduction scope')
        used = preserved['spent_usd'] + preserved['reserved_usd']
        limit = (before['model_usd'] * factor if amendment['model_scope'] == 'total' else
                 used + max(0.0, before['model_usd'] - used) * factor)
        expected = {key:value * factor for key,value in before.items()}
        expected['model_usd'] = limit
    count = preserved['model_entry_count']
    if (after != expected or preserved['wall_started_epoch'] != wall['started_epoch']
            or not isinstance(count, int) or not 0 <= count <= len(ledger['entries'])
            or object_digest(ledger['entries'][:count]) != preserved['model_entries_digest']):
        raise ValueError('amendment does not preserve authorized history')
    return amendment


def authorizes_transition(output: Path, *, old_limit: float, new_limit: float, run_id: str) -> bool:
    """Adopt a verified authorization chain, including model then wall-only additions."""
    try:
        if not (math.isfinite(old_limit) and math.isfinite(new_limit)):
            return False
        ledger = read_json(output/'agent/cost_ledger.json')
        identity = ledger.get('budget_amendment_id')
        records = [output/'agent/cost_ledger.json', output/'agent/runtime_spec.json',
                   output/'task_gpu_budget.json', output/'budget.json']
        if any(record.is_symlink() for record in records):
            return False
        ledger, spec, gpu, wall = map(read_json, records)
        amendment = _verified_amendment(output, identity, ledger, wall)
        after = amendment['after']
        if (any(row.get('budget_amendment_id') != identity for row in (ledger,spec,gpu,wall))
                or not (ledger['run_id'] == run_id == spec['run_id'])
                or after['model_usd'] != new_limit or new_limit != ledger['limit_usd']
                or new_limit != spec['total_budget_usd']
                or after['turn_usd'] != spec['turn_budget_usd']
                or after['gpu_seconds'] != gpu['cap_seconds']
                or after['wall_seconds'] != wall['wall_seconds']):
            return False
        seen = set()
        while amendment['before']['model_usd'] != old_limit:
            if amendment['id'] in seen:
                return False
            seen.add(amendment['id'])
            previous = amendment.get('previous_amendment_id')
            if 'previous_amendment_id' in amendment:
                predecessor = _verified_amendment(output, previous, ledger, wall)
            else:
                # Legacy records have no link: accept only a unique, verified earlier
                # transaction whose entire resulting budget matches this input budget.
                matches = []
                for path in (output/'budget_amendments').glob('*.json'):
                    if path.stem in seen:
                        continue
                    try:
                        row = _verified_amendment(output, path.stem, ledger, wall)
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
                    if row['after'] == amendment['before'] and row['at'] < amendment['at']:
                        matches.append(row)
                if len(matches) != 1:
                    return False
                predecessor = matches[0]
            if (predecessor['after'] != amendment['before']
                    or predecessor['at'] >= amendment['at']
                    or predecessor['preserved']['model_entry_count'] > amendment['preserved']['model_entry_count']):
                return False
            amendment = predecessor
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--factor', type=float, required=True)
    parser.add_argument('--reason', required=True)
    parser.add_argument('--model-scope', choices=['total', 'remaining'], default='remaining')
    parser.add_argument('--preserve-stopped-holds', action='store_true',
        help='operator confirms controller stopped; preserve stranded reserved entries unchanged (no release)')
    args = parser.parse_args()
    print(json.dumps(reduce(args.output, factor=args.factor, reason=args.reason,
                           model_scope=args.model_scope, preserve_stopped_holds=args.preserve_stopped_holds), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
