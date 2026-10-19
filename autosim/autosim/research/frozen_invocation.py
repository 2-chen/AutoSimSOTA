"""Keep diagnostic command verification from replacing a frozen scoring adapter."""
from pathlib import Path

from .common import (assert_frozen, atomic_json, digest, immutable_json, now,
                     object_digest, read_json)
from .experiment_transaction import safe


def frozen_entry(output, run_id, stage):
    path=safe(output,f'research/{run_id}/comparison_protocol.json')
    if not path.exists():
        return None
    row=read_json(path)
    if not isinstance(row,dict):
        raise ValueError('frozen comparison protocol must be an object')
    if row.get('target')!=stage:
        return None
    if (not isinstance(row.get('evaluator_argv_source'),str)
            or not isinstance(row.get('evaluator_parameters'),dict)
            or not isinstance(row.get('evaluator_stage'),dict)):
        raise ValueError('frozen evaluator invocation is malformed; no replacement authorized')
    return {'source':row['evaluator_argv_source'],
            'parameters':row['evaluator_parameters'],'row':row['evaluator_stage']}


def differences(output, run_id, stage, candidate):
    expected=frozen_entry(output,run_id,stage)
    if expected is None:
        return []
    execution=read_json(safe(output,'execution.json'))
    effective={**((execution.get('stages')or{}).get(stage)or{}),
               **(candidate.get('row')or{}),'available':True}
    actual={**candidate,'row':effective}
    return [key for key in ('source','parameters','row') if actual.get(key)!=expected[key]]


def retain_diagnostic(output, run_id, stage, candidate, attempts):
    """Preserve successful verification, without installing it or reporting a score."""
    protocol=safe(output,f'research/{run_id}/comparison_protocol.json')
    row={'schema_version':1,'run_id':run_id,'stage':stage,
         'frozen_protocol_sha256':digest(protocol),'candidate':candidate,
         'attempts':attempts,'status':'verified_diagnostic_not_installed',
         'authority':'command diagnostic only; not a frozen adapter, policy adoption or score'}
    identity=object_digest(row)
    ref=f'diagnostic_invocations/{identity}.json'
    immutable_json(safe(output,ref),row)
    return ref


def restore_legacy(output, repo, run_id, stage, *, expected_current_sha256, reason):
    """Stopped-run maintenance only: restore the exact independently verified freeze.

    This does not edit protocols, benchmark code, measurements, costs or policies.
    Unrelated entries and the displaced diagnostic remain preserved. No permission
    to change the evaluator or to replay native work is inferred from restoration.
    """
    output=Path(output).resolve(strict=True);repo=Path(repo).resolve(strict=True)
    if not isinstance(reason,str) or not reason.strip():
        raise ValueError('explicit maintenance reason required')
    kept_path=safe(output,'derived_stages.json')
    before_sha=digest(kept_path)
    if before_sha!=expected_current_sha256:
        raise ValueError('active invocation record changed; inspect before restoring')
    from .native_jobs import active_jobs
    if active_jobs(output):
        raise ValueError('native jobs prevent evaluator metadata restoration')
    # Never replace a live in-memory adapter while a recorded model/native process
    # still owns this run. The operator first stops the outer research service.
    from .process_executor import inspect_process_identity
    for directory in ('processes','agent/processes'):
        for path in (output/directory).glob('*.json'):
            row=read_json(safe(output,path.relative_to(output).as_posix()))
            identity=row.get('process_identity') or row.get('identity')
            if identity and inspect_process_identity(identity).get('status')!='not_running':
                raise ValueError('live or unverifiable process prevents metadata restoration')
    root=safe(output,f'research/{run_id}')
    protocol=safe(output,f'research/{run_id}/comparison_protocol.json')
    protocol_sha=digest(protocol)
    from .protocol_revision import effective
    protected,_=effective(root)
    hashes=read_json(protected).get('hashes')
    if not isinstance(hashes,dict) or not hashes or any(
            not Path(name).is_absolute() or not Path(name).resolve().is_relative_to(repo)
            for name in hashes):
        raise ValueError('original protected source identities required')
    assert_frozen(hashes)
    from .receipt_verifier import verify_measurement
    proof=verify_measurement(root,'baseline')
    if proof.get('status')!='consistent':
        raise ValueError('original baseline/protocol receipts must remain independently consistent')
    if digest(protocol)!=protocol_sha:
        raise ValueError('protocol changed during original baseline verification')
    expected=frozen_entry(output,run_id,stage)
    kept=read_json(kept_path)
    if expected is None or stage not in kept:
        raise ValueError('no frozen active evaluator entry to restore')
    if not differences(output,run_id,stage,kept[stage]):
        return {'status':'already_matches','stage':stage}
    if differences(output,run_id,stage,expected):
        raise ValueError('survey metadata also differs; no implicit multi-record restoration')
    record={'schema_version':1,'at':now(),'run_id':run_id,'stage':stage,
        'reason':reason,'before_record_sha256':before_sha,
        'protocol_sha256':protocol_sha,'before':kept[stage],'restored':expected,
        'baseline_proof':proof,'status':'restoration_prepared',
        'authority':'metadata repair only; no execution or scientific approval'}
    identity=object_digest(record)
    ref=f'frozen_invocation_recoveries/{identity}.json'
    immutable_json(safe(output,ref),record)
    if digest(kept_path)!=before_sha or digest(protocol)!=protocol_sha:
        raise ValueError('invocation/protocol changed during restoration; no write')
    after={**kept,stage:expected}
    atomic_json(kept_path,after)
    if read_json(kept_path)!=after or digest(protocol)!=protocol_sha:
        raise ValueError('records changed after restoration; inspect prepared audit before continuing')
    applied_ref=f'frozen_invocation_recoveries/{identity}.applied.json'
    applied={'schema_version':1,'at':now(),'status':'exact_frozen_adapter_restored',
             'prepared_ref':ref,'after_record_sha256':digest(kept_path),
             'after_object_sha256':object_digest(after),'authority':record['authority']}
    immutable_json(safe(output,applied_ref),applied)
    return {'status':applied['status'],'record_ref':ref,'applied_ref':applied_ref,'stage':stage}
