"""Current controller/budget facts versus historical observations, without erasing evidence."""
from functools import lru_cache
from pathlib import Path

from .common import digest, object_digest, read_json
from .prompt_library import prompt_identity


@lru_cache(maxsize=1)
def controller_identity():
    root = Path(__file__).parent
    # Automatically cover new admission validators, never mutable run artifacts.
    sources = {path.name: digest(path) for path in sorted(root.glob('*.py'))}
    return object_digest({'sources': sources, 'prompts': prompt_identity()})


def current_model_budget(output, run_id):
    path = Path(output)/'agent/cost_ledger.json'
    if not path.is_file():
        return {'status':'unavailable', 'reason':'no persistent model ledger'}
    try:
        from .agent_budget import AgentCostLedger
        if path.is_symlink():
            raise ValueError('unsafe model ledger')
        row = read_json(path)
        ledger = AgentCostLedger(output,run_id=run_id,limit_usd=row['limit_usd'],
                                 cost_basis=row.get('cost_basis','cli_reported'))
        return {'status':'verified_current', **ledger.snapshot(),
                'authority':'current ledger; historical role snapshots are not admission decisions'}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        return {'status':'unavailable','reason':type(exc).__name__,
                'authority':'unverified; retain holds and do not infer available allowance'}


def budget_observation(observation, current):
    result = dict(observation or {})
    original = result.get('run_budget') or {}
    if (result.get('status') == 'budget_exhausted' and
            current.get('status') == 'verified_current' and
            isinstance(original,dict) and isinstance(original.get('remaining_usd'),(int,float)) and
            current['remaining_usd'] > original['remaining_usd'] + 1e-9):
        result.update(status='historical_budget_exhausted', original_status='budget_exhausted',
            validity='historical_not_current_admission',
            invalidation_reason='current safe allowance exceeds the recorded blocked allowance',
            budget_snapshot_scope='historical; use current_model_budget, not this run_budget')
    return result


def failure_context(output, row):
    """Version a failed controller action without deleting or declaring it repaired."""
    recorded = None
    ref = row.get('receipt_ref')
    if isinstance(ref, str):
        path = Path(output)/ref
        try:
            if (not path.is_symlink() and path.resolve().is_relative_to(Path(output).resolve())
                    and path.is_file() and path.stat().st_size <= 2*1024**2):
                from .action_receipt import verify_action_receipt
                receipt = verify_action_receipt(read_json(path))
                recorded = (receipt.get('resource_limits') or {}).get('enforced', {}).get('controller_runtime_id')
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    if recorded is None and row.get('evidence_id'):
        try:
            import json
            from .evidence_store import read_attempt_evidence
            proof = read_attempt_evidence(output, row['evidence_id'], limit=12000)
            recorded = (json.loads(proof['text']).get('context') or {}).get('controller_runtime_id')
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return {'receipt_ref':ref, 'outcome':row.get('outcome'),
            'evidence_id':row.get('evidence_id'), 'recorded_at':row.get('finished_at') or row.get('at'),
            'failure_category':row.get('failure_category'), 'repair_owner':row.get('repair_owner'),
            'recorded_controller_runtime_id':recorded,
            'validity':'observed_in_current_runtime' if recorded == controller_identity()
                       else 'unverified_in_current_runtime',
            'repair_status':'not_proven; bounded native revalidation required'}


def controller_context(output, steps):
    identity = controller_identity()
    result = {'controller_runtime_id':identity,
        'guidance':'Historical failure is not proof the current controller still fails. '
                   'When runtime identity differs or is unrecorded, consider ONE bounded '
                   'source-backed revalidation using the trusted executor; preserve old evidence, '
                   'do not mark it repaired, do not replay unknown live operations.'}
    failures, seen = [], set()
    for row in reversed(steps):
        arguments = row.get('arguments') or {}
        key = (row.get('step'), arguments.get('stage') if isinstance(arguments, dict) else None)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        if row.get('outcome') in {'raised','failed','rejected','no runnable command','checkpoint failure','not attempted','blocked'}:
            failures.append({'step':key[0], 'stage':key[1], **failure_context(output,row)})
    result['previous_action_failures'] = failures[:8]
    for row in reversed(steps):
        if row.get('step') != 'build_the_environment':
            continue
        if row.get('outcome') not in {'raised','failed','checkpoint failure','repair proposal rejected'}:
            break
        ref = row.get('receipt_ref')
        recorded = None
        if isinstance(ref,str):
            path = Path(output)/ref
            try:
                if (not path.is_symlink() and path.resolve().is_relative_to(Path(output).resolve()) and
                        path.is_file() and path.stat().st_size <= 2*1024**2):
                    from .action_receipt import verify_action_receipt
                    receipt = verify_action_receipt(read_json(path))
                    recorded = (receipt.get('resource_limits') or {}).get('enforced',{}).get('controller_runtime_id')
            except (OSError,ValueError,TypeError,AttributeError):
                pass
        result['previous_setup_failure'] = {'receipt_ref':ref,'outcome':row.get('outcome'),
            'recorded_controller_runtime_id':recorded,
            'validity': 'observed_in_current_runtime' if recorded == identity else
                        'unverified_in_current_runtime',
            'repair_status':'not_proven; bounded native revalidation required'}
        break
    return result
