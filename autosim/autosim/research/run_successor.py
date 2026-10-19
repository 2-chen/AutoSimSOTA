"""Operator-authorized study restart without discarding financial/GPU history.

This is not a scientific promotion or resource reset. The old writer must be
stopped; it is retired before a successor may use the inherited ceilings.
"""
import json
import subprocess
import uuid
from pathlib import Path

from .common import atomic_json, digest, immutable_json, now, object_digest, read_json


def assert_not_retired(output):
    path = Path(output) / 'successor_run.json'
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise ValueError('unsafe successor retirement record')
        row = read_json(path)
        raise ValueError('this study was retired into ' + str(row.get('successor')) +
                         '; resume that output, not the retired budget writer')


def service_state(unit):
    if (not isinstance(unit, str) or not unit.startswith('autosim-') or
            any(not (c.isalnum() or c in '-_.') for c in unit)):
        raise ValueError('explicit AutoSim parent unit required')
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'MainPID', '-p', 'LoadState'],
        capture_output=True, text=True, timeout=15, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def _accounting_proof_names(parent):
    """Carry grants and prefix-rewrite proofs, never model/research sessions."""
    if (parent / 'agent').is_symlink():
        raise ValueError('unsafe accounting agent directory')
    names, total = [], 0
    for relative in ('budget_amendments', 'agent/ledger_history'):
        directory = parent / relative
        if directory.is_symlink():
            raise ValueError('unsafe accounting proof directory')
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError('accounting proof directory is not a directory')
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                raise ValueError('unsafe accounting proof link')
            if path.is_dir():
                continue
            if not path.is_file() or path.suffix != '.json':
                raise ValueError('unexpected accounting proof payload')
            total += path.stat().st_size
            names.append(path.relative_to(parent).as_posix())
            if len(names) > 4096 or total > 32 * 1024**2:
                raise ValueError('accounting proof inventory exceeds bounds')
    spec = parent / 'agent/runtime_spec.json'
    if spec.exists() or spec.is_symlink():
        if spec.is_symlink() or not spec.is_file() or spec.stat().st_size > 64 * 1024:
            raise ValueError('unsafe inherited runtime accounting configuration')
        names.append('agent/runtime_spec.json')
    return names


def _rebound_runtime_spec(value, successor):
    # Runtime configuration is not a provider conversation or environment approval.
    return {**value, 'workspace_identity': str(successor / 'checkout')}


def prepare(parent, successor, *, parent_unit, reason):
    """Copy only accounting, not measurements, declarations, approvals or sessions."""
    parent, successor = Path(parent), Path(successor)
    if parent.is_symlink() or successor.is_symlink():
        raise ValueError('study roots must not be symlinks')
    parent = parent.resolve(strict=True)
    successor = successor.absolute()
    if successor.parent.resolve() != parent.parent or successor.exists():
        raise ValueError('successor must be a new sibling study output')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
        raise ValueError('explicit authorization/restart reason required')
    assert_not_retired(parent)
    observed = service_state(parent_unit)
    if observed.get('MainPID') != '0' or observed.get('ActiveState') not in {'inactive', 'failed'}:
        raise ValueError('parent controller must be proven stopped before budget inheritance')
    finished = read_json(parent / 'native_jobs/controller_finished.json')
    if finished.get('state') not in {'running', 'finished'}:
        raise ValueError('parent controller ownership record is unreadable')
    from .native_jobs import active_jobs
    if active_jobs(parent):
        raise ValueError('native jobs still own the parent workspace')
    from .task_budget import TaskGPUBudget
    gpu_snapshot = TaskGPUBudget(parent).snapshot()
    if gpu_snapshot['active_leases']:
        raise ValueError('parent still has active GPU reservations')
    names = ('budget.json', 'budget_scope.json', 'task_gpu_budget.json', 'agent/cost_ledger.json',
             *_accounting_proof_names(parent))
    originals = {}
    hashes = {}
    for name in names:
        path = parent / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024**2:
            raise ValueError('unsafe/missing parent accounting: ' + name)
        originals[name] = read_json(path)
        hashes[name] = digest(path)
    if originals['budget_scope.json'].get('scope') != 'task':
        raise ValueError('repository-wide budget needs its own shared-ledger reconciliation')
    from .agent_budget import AgentCostLedger
    model = originals['agent/cost_ledger.json']
    ledger = AgentCostLedger(parent, run_id=model['run_id'], limit_usd=model['limit_usd'],
                             cost_basis=model.get('cost_basis', 'cli_reported'))
    model_snapshot = ledger.snapshot()
    if gpu_snapshot['cap_gpu_seconds'] > 86400:
        from .budget_amendment import authorizes_gpu_cap
        if not authorizes_gpu_cap(parent, gpu_snapshot['cap_gpu_seconds']):
            raise ValueError('parent above-default GPU cap lacks verified authorization')
    if any(digest(parent / name) != sha for name, sha in hashes.items()):
        raise ValueError('parent accounting changed during inheritance')
    row = {'schema_version': 1, 'created_at': now(), 'parent': str(parent),
           'successor': str(successor), 'parent_unit': parent_unit, 'reason': reason,
           'parent_accounting_sha256': hashes, 'model_at_transfer': model_snapshot,
           'gpu_at_transfer': gpu_snapshot, 'parent_service_at_transfer': observed,
           'parent_controller_record_at_transfer': finished,
           'termination_kind': ('normal_exit' if finished.get('state') == 'finished'
                                else 'operator_stopped_verified_idle_service'),
           'wall_origin_preserved': True, 'held_costs_preserved': True,
           'scientific_results_inherited': False, 'formal_paper_eligible': False,
           'cost_rollup_rule': 'deduplicate inherited model reservation IDs and GPU lease IDs; '
                               'never add both cumulative parent and successor totals'}
    row['id'] = object_digest(row)
    successor.mkdir()
    # Fail closed if preparation is interrupted: the old writer is retired before
    # the new accounting exists. An incomplete successor must not be launched.
    atomic_json(successor / 'successor_initialization.json', {'status': 'pending', 'id': row['id']})
    immutable_json(parent / 'successor_run.json', row)
    for name, value in originals.items():
        copied = json.loads(json.dumps(value))
        if name == 'task_gpu_budget.json':
            copied.update(output=str(successor), task_id=uuid.uuid4().hex,
                          parent_task_id=value['task_id'])
        elif name == 'agent/runtime_spec.json':
            copied = _rebound_runtime_spec(copied, successor)
        atomic_json(successor / name, copied)
    immutable_json(successor / 'budget_inheritance.json', row)
    atomic_json(successor / 'successor_initialization.json', {'status': 'prepared', 'id': row['id']})
    return row


def validate_inheritance(output):
    """Check retirement/provenance on every successor launch, including resume."""
    path = Path(output) / 'budget_inheritance.json'
    initialization = Path(output) / 'successor_initialization.json'
    if initialization.exists() or initialization.is_symlink():
        if (initialization.is_symlink() or read_json(initialization).get('status') != 'prepared'
                or not path.is_file()):
            raise ValueError('successor accounting initialization is incomplete')
    if not path.exists():
        return
    if path.is_symlink():
        raise ValueError('unsafe budget inheritance')
    row = read_json(path)
    unsigned = {k: v for k, v in row.items() if k != 'id'}
    if (row.get('id') != object_digest(unsigned) or
            row.get('successor') != str(Path(output).resolve())):
        raise ValueError('budget inheritance identity changed')
    parent = Path(row['parent'])
    if read_json(parent / 'successor_run.json') != row:
        raise ValueError('parent is not retired to this successor')
    for name, sha in row['parent_accounting_sha256'].items():
        if digest(parent / name) != sha:
            raise ValueError('retired parent accounting changed; reconcile before resume')


def preparation_reference(output, stage):
    """Read one authenticated predecessor recipe as advice, not inherited approval.

    A successor changes paths and may change environments/resources. Keep the
    old source-backed runtime connections readable, but never install the old
    adapter, supply its checkpoint/settings, or promote its native receipt.
    """
    output = Path(output)
    if not (output / 'budget_inheritance.json').exists():
        return {'status': 'not_available'}
    try:
        if stage not in {'train', 'evaluate', 'collect', 'prepare_data', 'record_demo'}:
            raise ValueError('select one declared native stage')
        validate_inheritance(output)
        from .experiment_transaction import safe
        from .invocation_blueprints import _proof
        parent = Path(read_json(output / 'budget_inheritance.json')['parent'])
        path = safe(parent, 'derived_stages.json')
        if path.stat().st_size > 256 * 1024:
            raise ValueError('predecessor adapter archive exceeds bounds')
        entry = read_json(path).get(stage)
        if (not isinstance(entry, dict) or not isinstance(entry.get('row'), dict) or
                not isinstance(entry.get('source'), str) or
                len(json.dumps(entry).encode()) > 16 * 1024):
            raise ValueError('no bounded predecessor stage recipe')
        from .execution_derive import argv_function_name
        from .patch_validation import checked_function
        checked_function(entry['source'], argv_function_name(stage))
        identity = object_digest(entry)
        directory = safe(parent, 'execution_interfaces/' + stage)
        for interface_path in sorted(directory.glob('*.json'))[-24:]:
            interface_path = safe(parent, interface_path.relative_to(parent).as_posix())
            if interface_path.stat().st_size > 64 * 1024:
                continue
            interface = read_json(interface_path)
            if (interface.get('id') != interface_path.stem or
                    object_digest({k: v for k, v in interface.items() if k != 'id'}) != interface_path.stem or
                    interface.get('stage') != stage or interface.get('stage_sha256') != identity):
                continue
            _proof(parent, interface)
            derivation_path = safe(parent, interface['derivation_ref'])
            if derivation_path.stat().st_size > 8 * 1024**2 or digest(derivation_path) != interface['derivation_sha256']:
                continue
            return {'status': 'historical_reference_requires_new_verification',
                    'stage': stage, 'entry': entry, 'entry_sha256': identity,
                    'interface_id': interface['id'], 'native_evidence_ref': interface['native_evidence_ref'],
                    'parent_inheritance_id': read_json(output / 'budget_inheritance.json')['id'],
                    'authority': 'historical runtime connection advice only; no current command, '
                                 'checkpoint, score, protocol or environment approval',
                    'instruction': 'Agent may inspect and adapt this recipe to CURRENT source and caller inputs. '
                        'Do not copy old task/data/checkpoint/work/settings or mount old paths by default. '
                        'Preserve valid interpreter/config/renderer connections when repairing another edge. '
                        'Any chosen revision needs a NEW native verification and the current scientific/budget guards.'}
        raise ValueError('no matching authenticated predecessor native interface')
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        return {'status': 'not_available', 'because': str(exc)[:300],
                'authority': 'missing historical reference is not a native failure'}


def assert_pristine_bootstrap(output):
    """Accounting inheritance is not research initialization.

    Only explicitly supported bootstrap artifacts may precede the first intent.
    Unknown or execution artifacts fail closed; inherited charges must be intact.
    """
    output = Path(output)
    if not (output / 'budget_inheritance.json').is_file():
        raise ValueError('first study intent requires verified budget inheritance')
    validate_inheritance(output)
    allowed = {'budget.json', 'budget_scope.json', 'task_gpu_budget.json',
               'task_gpu_budget.lock', 'budget_inheritance.json',
               'successor_initialization.json', 'workspace_snapshot.json',
               'checkout', 'env', '.borrowed_dependencies', 'agent',
               'bootstrap_source.json', 'bootstrap_dependency_delta.json',
               'OPERATOR_INTERVENTIONS.md', 'budget_amendments'}
    if any(p.name not in allowed for p in output.iterdir()):
        raise ValueError('successor already has execution or unknown artifacts')
    agent = output / 'agent'
    if agent.is_symlink() or any(p.name not in {'cost_ledger.json', 'cost_ledger.lock',
                                             'runtime_spec.json', 'ledger_history'}
                               or p.is_symlink() for p in agent.iterdir()):
        raise ValueError('successor agent activity already initialized')
    row = read_json(output / 'budget_inheritance.json')
    expected_proofs = set(row['parent_accounting_sha256']) - {
        'budget.json', 'budget_scope.json', 'task_gpu_budget.json', 'agent/cost_ledger.json'}
    if set(_accounting_proof_names(output)) != expected_proofs:
        raise ValueError('successor accounting proof inventory changed before first study intent')
    for name in expected_proofs:
        path = output / name
        if name == 'agent/runtime_spec.json':
            expected_spec = _rebound_runtime_spec(read_json(Path(row['parent']) / name), output)
            if read_json(path) != expected_spec:
                raise ValueError('successor runtime accounting configuration changed before first study intent')
        elif digest(path) != row['parent_accounting_sha256'][name]:
            raise ValueError('successor accounting proof changed before first study intent')
    for name in ('budget.json', 'budget_scope.json', 'agent/cost_ledger.json'):
        path = output / name
        if path.is_symlink() or digest(path) != row['parent_accounting_sha256'][name]:
            raise ValueError('successor accounting changed before first study intent')
    parent_gpu = read_json(Path(row['parent']) / 'task_gpu_budget.json')
    child_gpu = read_json(output / 'task_gpu_budget.json')
    expected = {**parent_gpu, 'output': str(output.resolve()),
                'task_id': child_gpu.get('task_id'), 'parent_task_id': parent_gpu['task_id']}
    if child_gpu != expected or (output / 'task_gpu_budget.json').is_symlink():
        raise ValueError('successor GPU activity changed before first study intent')


def bootstrap_source(parent, successor):
    """Reuse audited tracked-source deltas, never parent data or state directories.

    A fresh checkout is made by the normal resource-aware snapshotter first. Its
    original source hashes must match the parent's authentic original inventory.
    These are explicit engineering interventions, not scientific approvals.
    """
    import os
    from .common import atomic_text
    parent, successor = Path(parent).resolve(), Path(successor).resolve()
    validate_inheritance(successor)
    old = read_json(parent / 'workspace_snapshot.json')
    fresh = read_json(successor / 'workspace_snapshot.json')
    if old.get('source') != fresh.get('source'):
        raise ValueError('source bootstrap requires the same original repository')
    checkout = successor / 'checkout'
    originals = {row[1]: row for row in fresh['source_entries'] if row[0] == 'file'}
    edits = []
    for row in old['source_entries']:
        if row[0] != 'file':
            continue
        relative = Path(row[1])
        source = parent / 'checkout' / relative
        target = checkout / relative
        if (relative.is_absolute() or '..' in relative.parts or source.is_symlink()
                or not source.is_file() or not source.resolve().is_relative_to(parent / 'checkout')):
            raise ValueError('unsafe/missing parent tracked source')
        changed = digest(source)
        if changed == row[3]:
            continue
        if (source.stat().st_size > 1024**2 or originals.get(row[1]) != row
                or target.is_symlink() or digest(target) != row[3]
                or not target.resolve().is_relative_to(checkout)):
            raise ValueError('bootstrap source drift or oversized edit: ' + row[1])
        # UTF-8 source only. Binary artifacts are explicit read-only resources.
        text = source.read_text(encoding='utf-8')
        edits.append((source, target, row, changed, text))
    record = {'schema_version': 1, 'parent': str(parent), 'created_at': now(),
              'files': [{'path': row[1], 'original_sha256': row[3], 'reused_sha256': sha}
                        for _, _, row, sha, _ in edits],
              'authority': 'operator source bootstrap; must reverify in the new run',
              'formal_paper_eligible': False}
    for source, target, row, sha, text in edits:
        if digest(source) != sha:
            raise ValueError('parent source changed during bootstrap')
        atomic_text(target, text)
        os.chmod(target, source.stat().st_mode & 0o777)
        if digest(target) != sha:
            raise ValueError('source bootstrap bytes differ')
    immutable_json(successor / 'bootstrap_source.json', record)
    return record


def bootstrap_dependency_delta(parent_prefix, successor_prefix, *, output):
    """Copy an installed, ABI-matched local delta, not an entire environment.

    Distribution RECORDs delimit payloads. Startup hooks, scripts outside the
    site directory and unowned files are never inherited or executed.
    """
    import importlib.metadata as metadata
    import re
    import shutil
    parent_prefix, successor_prefix, output = (Path(p).resolve() for p in
        (parent_prefix, successor_prefix, output))
    validate_inheritance(output)
    if not successor_prefix.is_relative_to(output) or parent_prefix.is_relative_to(output):
        raise ValueError('dependency delta needs external base and run-owned target')
    versions = []
    for prefix in (parent_prefix, successor_prefix):
        text = (prefix / 'pyvenv.cfg').read_text()
        match = re.search(r'(?m)^version(?:_info)?\s*=\s*(\d+\.\d+)', text)
        if not match:
            raise ValueError('dependency delta requires known Python ABI')
        versions.append(match[1])
    if versions[0] != versions[1]:
        raise ValueError('dependency delta Python ABI differs')
    relative_site = Path('lib') / ('python' + versions[0]) / 'site-packages'
    source_site, target_site = parent_prefix / relative_site, successor_prefix / relative_site
    planned, distributions = {}, []
    skipped_scripts = []
    total = 0
    for dist in metadata.distributions(path=[str(source_site)]):
        distributions.append({'name': dist.metadata['Name'], 'version': dist.version})
        if dist.files is None:
            raise ValueError('installed delta has no RECORD')
        for item in dist.files:
            relative = Path(str(item))
            if '..' in relative.parts and (source_site / relative).resolve().is_relative_to(parent_prefix):
                skipped_scripts.append(str(relative))
                continue  # Prefix-local CLI/docs are not import payloads.
            if (relative.is_absolute() or '..' in relative.parts or relative.suffix == '.pth'
                    or relative.name in {'sitecustomize.py', 'usercustomize.py'}):
                raise ValueError('installed delta includes startup hooks or external payload')
            if '__pycache__' in relative.parts:
                continue
            source, target = source_site / relative, target_site / relative
            if (source.is_symlink() or not source.is_file() or
                    not source.resolve().is_relative_to(source_site) or
                    target.is_symlink() or not target.resolve().is_relative_to(target_site)):
                raise ValueError('unsafe installed delta payload')
            sha = digest(source)
            if target.exists() and digest(target) != sha:
                raise ValueError('installed delta would overwrite current target')
            if relative.as_posix() not in planned:
                total += source.stat().st_size
                planned[relative.as_posix()] = (source, target, sha)
            if total > 512 * 1024**2 or len(planned) > 20000:
                raise ValueError('installed delta exceeds bounded payload scope')
    for source, target, sha in planned.values():
        if digest(source) != sha:
            raise ValueError('installed dependency changed during transfer')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if digest(target) != sha:
            raise ValueError('installed dependency transfer hash mismatch')
    record = {'schema_version': 1, 'parent_prefix': str(parent_prefix),
              'successor_prefix': str(successor_prefix), 'python_abi': versions[0],
              'distributions': distributions, 'bytes': total,
              'skipped_non_import_payloads': skipped_scripts,
              'files': {name: data[2] for name, data in planned.items()},
              'authority': 'operator reuse of installed local delta; consumer probes required'}
    immutable_json(output / 'bootstrap_dependency_delta.json', record)
    return record


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('successor', type=Path)
    parser.add_argument('--parent-unit', required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args(argv)
    row = prepare(args.parent, args.successor, parent_unit=args.parent_unit, reason=args.reason)
    print(json.dumps({'id': row['id'], 'successor': row['successor'],
                      'scientific_results_inherited': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
