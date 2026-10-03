"""Operator recovery of the pre-launch amendment-path bug; no usage reset or run restart."""
import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

from autosim.research.common import atomic_json, digest, now, object_digest, read_json
from autosim.research.agent_budget import AgentCostLedger
from autosim.research.budget import RunBudget
from autosim.research.budget_amendment import authorizes_transition


def recover(output, misplaced_ref, count, original_digest, original_model_limit):
    output = Path(output).resolve(strict=True)
    misplaced = output / misplaced_ref
    if (misplaced.is_symlink() or misplaced.parent != output/'agent/processes'
            or not misplaced.resolve().is_relative_to(output)):
        raise ValueError('invalid exact affected process receipt')
    bad = read_json(misplaced)
    if bad.get('status') != 'applied' or bad.get('model_scope') != 'remaining':
        raise ValueError('affected receipt is not the known misplaced applied amendment')
    if read_json(output/'run_state.json').get('status') == 'running':
        raise ValueError('controller must remain stopped')
    ledger_path = output/'agent/cost_ledger.json'
    ledger = read_json(ledger_path)
    old_entries = ledger['entries'][:count]
    if len(old_entries) != count or object_digest(old_entries) != original_digest:
        raise ValueError('original ledger prefix cannot be verified; no budget change')
    spent, held = AgentCostLedger._totals({'entries':old_entries})
    first_after = bad['before']
    expected = spent+held+(original_model_limit-spent-held)*bad['factor']
    if not math.isclose(first_after['model_usd'], expected, abs_tol=1e-9):
        raise ValueError('first authorized reduction does not match original preserved usage')
    wall, gpu, spec = (read_json(output/path) for path in
        ('budget.json','task_gpu_budget.json','agent/runtime_spec.json'))
    if (wall['wall_seconds'] != bad['after']['wall_seconds'] or gpu['cap_seconds'] != bad['after']['gpu_seconds']
            or ledger['limit_usd'] != bad['after']['model_usd'] or spec['total_budget_usd'] != ledger['limit_usd']):
        raise ValueError('current limits no longer match the unintended second reduction')
    manifest = read_json(output/'run_events.json')
    base_path = output/manifest['base_ref']
    if digest(base_path) != manifest['base_sha256']:
        raise ValueError('original event history hash changed; do not reconstruct receipt')
    selected = subprocess.run(['jq','-c','--arg','ref',misplaced_ref,
        '.rows[]|select(.event=="agent_process_started" and .details.process_ref==$ref)|{at,details,event_hash}',
        str(base_path)], check=True, text=True, capture_output=True)
    starts = [json.loads(line) for line in selected.stdout.splitlines()]
    terminals = []
    with (output/'agent/events.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('process_ref') == misplaced_ref and row.get('type') == 'turn_finished':
                terminals.append(row)
    if len(starts) != 1 or len(terminals) != 1:
        raise ValueError('original process identity/terminal evidence not unique; no reconstruction')
    start, terminal = starts[0], terminals[0]
    identity = start['details']['process_identity']
    from autosim.research.process_executor import inspect_process_identity
    if inspect_process_identity(identity).get('status') != 'not_running':
        raise ValueError('original recorded process is not proven stopped')
    incident = output/'diagnostics/budget_amendment_path_incident.json'
    atomic_json(incident, {'at':now(), 'misplaced_ref':misplaced_ref, 'misplaced_payload':bad,
        'ledger_digest_at_recovery':digest(ledger_path), 'original_ledger_prefix_digest':original_digest,
        'process_start_event':start, 'process_terminal_event':terminal,
        'policy':'restore first authorized limits; preserve every ledger entry; reconstruct metadata explicitly, not byte-identical'})
    first = {**bad, 'before':{'wall_seconds':first_after['wall_seconds']/bad['factor'],
        'gpu_seconds':first_after['gpu_seconds']/bad['factor'], 'model_usd':original_model_limit,
        'turn_usd':first_after['turn_usd']/bad['factor']}, 'after':first_after,
        'preserved':{**bad['preserved'], 'model_entry_count':count, 'model_entries_digest':original_digest,
            'spent_usd':spent, 'reserved_usd':held}, 'recovered_at':now(),
        'recovery_ref':incident.relative_to(output).as_posix(),
        'model_available_before':original_model_limit-spent-held,
        'model_available_after':first_after['model_usd']-spent-held}
    amendment = output/'budget_amendments'/f"{first['id']}.json"
    atomic_json(amendment, {**first,'status':'prepared'})
    wall.update(wall_seconds=first_after['wall_seconds'], budget_amendment_id=first['id'])
    gpu.update(cap_seconds=first_after['gpu_seconds'], budget_amendment_id=first['id'])
    ledger.update(limit_usd=first_after['model_usd'], budget_amendment_id=first['id'])
    spec.update(total_budget_usd=first_after['model_usd'], turn_budget_usd=first_after['turn_usd'],
                budget_amendment_id=first['id'])
    for path, value in ((output/'budget.json',wall),(output/'task_gpu_budget.json',gpu),
                        (ledger_path,ledger),(output/'agent/runtime_spec.json',spec)):
        atomic_json(path,value)
    atomic_json(misplaced, {'schema_version':1, 'run_id':ledger['run_id'],
        'attempt_id':Path(misplaced_ref).stem, 'status':terminal['status'],
        'returncode':terminal['returncode'], 'timed_out':terminal.get('timed_out'),
        'process_identity':identity, 'role':terminal['role'], 'session_id':terminal['session_id'],
        'started_at':datetime.fromisoformat(start['at']).timestamp(),
        'finished_at':datetime.fromisoformat(terminal['at'].replace('Z','+00:00')).timestamp(),
        'reconstructed_metadata':True, 'not_byte_identical':True,
        'recovery_evidence_ref':incident.relative_to(output).as_posix()})
    atomic_json(amendment,{**first,'status':'applied'})
    RunBudget.existing(output).record()
    if not authorizes_transition(output,old_limit=original_model_limit,new_limit=first_after['model_usd'],run_id=ledger['run_id']):
        raise ValueError('recovered budget adoption contract did not verify')
    return {'status':'recovered', 'limits':first_after, 'usage_entries_preserved':len(ledger['entries']),
            'receipt_reconstructed_from_verified_history':misplaced_ref}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--misplaced-ref',required=True)
    parser.add_argument('--original-entry-count',type=int,required=True)
    parser.add_argument('--original-entries-digest',required=True)
    parser.add_argument('--original-model-limit',type=float,required=True)
    args=parser.parse_args()
    print(json.dumps(recover(args.output,args.misplaced_ref,args.original_entry_count,
        args.original_entries_digest,args.original_model_limit),ensure_ascii=False,indent=2))
