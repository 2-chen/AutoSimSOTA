import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from autosim.research import environment_pool as pool, provision as pv
from autosim.research.common import atomic_json, object_digest
from autosim.research.budget import RunBudget
from autosim.research.task_budget import TaskGPUBudget
from autosim.research.agent_budget import AgentCostLedger
from autosim.research.budget_amendment import reduce, extend, authorizes_transition


def base(path, *, editable=False, venv=False):
    (path/'bin').mkdir(parents=True)
    (path/'bin/python').write_text('fixture interpreter')
    if venv:
        (path/'pyvenv.cfg').write_text('version = 3.10.19')
    else:
        atomic_json(path/'conda-meta/python.json', {'name':'python', 'version':'3.10.19'})
    for name in ('torch', 'task-package'):
        metadata = path/f'lib/python3.10/site-packages/{name}-1.dist-info'
        metadata.mkdir(parents=True)
        (metadata/'METADATA').write_text(f'Name: {name}\nVersion: 1\n')
        if name == 'task-package' and editable:
            atomic_json(metadata/'direct_url.json', {'url':'file:///old/source', 'dir_info':{'editable':True}})
    return path


def test_editable_base_has_portable_pins_and_explicit_rebinding(tmp_path):
    row = pool.describe(base(tmp_path/'old', editable=True, venv=True))
    assert not row['cloneable'] and row['reconstructable']
    assert row['rebind_packages'] == ['task-package']
    assert row['portable_packages'] == {'torch':'1'}
    assert '/old/source' not in json.dumps(pool.short_catalog([row]))
    row['matching_package_count'] = 10
    other = pool.describe(base(tmp_path/'plain'))
    other['matching_package_count'] = 0
    rows = [{**other, 'id':f'generic-{i}'} for i in range(16)]
    assert pool.short_catalog([*rows, row])[0]['id'] == row['id']


def test_verified_venv_can_publish_template_without_copying_environment(tmp_path):
    output = tmp_path/'run'
    pool.configure(output, tmp_path/'store')
    prefix = base(output/'env', editable=True, venv=True)
    atomic_json(output/'environment.json', {'verdict':{'passed':True}, 'probes':['import actual_consumer'],
        'verified_entrypoints':{'train':'hash'}})
    result = pool.publish_snapshot(output, interpreter=prefix/'bin/python', manifests={}, machine={})
    assert result['status'] == 'not_cacheable' and result['portable_template']['status'] == 'published'
    rows = pool.catalog(output, {}, {})
    saved = next(row for row in rows if row.get('origin') == 'verified_portable_template')
    assert saved['reconstructable'] and not saved['cloneable'] and saved['prefix'] is None
    assert saved['rebind_packages'] == ['task-package']
    manifest = tmp_path/'store/templates'/saved['id']/'manifest.json'
    value = json.loads(manifest.read_text())
    value['payload']['environment']['portable_packages']['torch'] = 'fake'
    atomic_json(manifest, value)
    assert not any(row.get('origin') == 'verified_portable_template' for row in pool.catalog(output, {}, {}))


def test_existing_interpreter_can_switch_base_without_overwriting_and_resume_new_prefix(tmp_path, monkeypatch):
    output = tmp_path/'run'
    repo = output/'checkout'
    repo.mkdir(parents=True)
    old = base(output/'env')
    source = base(tmp_path/'source')
    old_bytes = (old/'bin/python').read_bytes()
    pool.configure(output, tmp_path/'store')
    row = pool.describe(source)
    monkeypatch.setattr(pool, 'catalog', lambda *a: [row])
    def plan(*a, **kw):
        assert kw['python'] is None
        assert kw['assets']['required_environment_selection']['base_environment_id'] == row['id']
        return {'python':'3.10', 'commands':['echo first', 'echo second'],
            'probes':['{python} -c "import task"'], 'assets':[], 'base_environment_id':row['id'],
            'environment_selection_reason':'different Python/ABI required'}
    monkeypatch.setattr(pv, 'plan', plan)
    clones = []
    def clone(source, **kw):
        clones.append(kw['destination'])
        shutil.copytree(source, kw['destination'])
        return {'ok':True, 'command':'clone', 'returncode':0}
    monkeypatch.setattr(pool, 'clone', clone)
    calls = []
    monkeypatch.setattr(pv, 'run', lambda command, **kw: calls.append(command) or {
        'command':command, 'ok':True, 'returncode':0, 'seconds':.01})
    monkeypatch.setattr(pv, 'probe_environment', lambda probe, **kw: {
        'command':pv.substitute(probe, kw['values']), 'ok':True, 'returncode':0, 'seconds':.01})
    arguments = dict(client=object(), prefix=old, output=output, manifests={}, assets={}, max_operations=1)
    first = pv.build(repo, python=str(old/'bin/python'), base_environment_id=row['id'],
        environment_selection_reason='resolve ABI incompatibility', **arguments)
    assert first['status'] == 'yielded'
    new_prefix = clones[0]
    assert new_prefix != old and new_prefix.is_relative_to(output/'environments')
    assert (old/'bin/python').read_bytes() == old_bytes
    cursor = json.loads((output/'provision_cursor.json').read_text())
    assert cursor['prefix'] == str(new_prefix)
    second = pv.build(repo, python=first['interpreter'], **arguments)
    third = pv.build(repo, python=second['interpreter'], **arguments)
    assert third['verdict']['passed']
    assert third['interpreter'] == str(new_prefix/'bin/python')
    assert len([path for path in clones if path.is_relative_to(output)]) == 1
    assert calls == ['echo first', 'echo second']
    assert (old/'bin/python').read_bytes() == old_bytes


def test_reconstructable_base_plan_uses_new_creation_and_no_old_editable_paths(tmp_path):
    row = pool.short_catalog([pool.describe(base(tmp_path/'old', editable=True, venv=True))])[0]
    class Client:
        def chat_with_metadata(self, system, payload, **kw):
            assert 'NEVER copying old venv/site hooks' in system
            return json.dumps({'python':'3.10', 'base_environment_id':row['id'],
                'base_environment_mode':'reconstruct', 'environment_selection_reason':'rebind editable package',
                'commands':['{conda} create --yes --prefix {prefix} python=3.10 pip',
                            '{pip} install -e {repo}'], 'probes':['{python} -c "import task"'], 'assets':[]}), {}
    result = pv.plan(Client(), tmp_path, manifests={}, assets={'environment_candidates':[row]})
    assert result['base_environment_mode'] == 'reconstruct'
    assert '/old/source' not in json.dumps(result)


def budgets(output):
    output.mkdir()
    RunBudget(output, wall_seconds=172800)
    TaskGPUBudget(output).initialize(output/'checkout')
    ledger = AgentCostLedger(output, run_id='run', limit_usd=60)
    reserved = ledger.reserve(requested_usd=4, session_id='session', role='scheduler')
    ledger.settle(reserved['reservation_id'], actual_usd=None, launched=True)
    atomic_json(output/'agent/runtime_spec.json', {'run_id':'run', 'total_budget_usd':60,
        'turn_budget_usd':4, 'timeout_seconds':900})
    return ledger


def test_reduce_remaining_budget_preserves_entries_gpu_usage_and_wall_start(tmp_path):
    output = tmp_path/'run'
    ledger = budgets(output)
    entries = json.loads(ledger.path.read_text())['entries']
    start = json.loads((output/'budget.json').read_text())['started_epoch']
    # Regression: scanning a historical process must not shadow the amendment path.
    historical = output/'agent/processes/old.json'
    atomic_json(historical, {'status':'completed', 'original_metadata':'must remain untouched'})
    historical_bytes = historical.read_bytes()
    result = reduce(output, factor=.5, reason='explicit user authorization')
    assert result['after'] == {'wall_seconds':86400, 'gpu_seconds':43200, 'model_usd':32, 'turn_usd':2}
    assert json.loads(ledger.path.read_text())['entries'] == entries
    assert json.loads((output/'budget.json').read_text())['started_epoch'] == start
    assert result['preserved']['model_entries_digest'] == object_digest(entries)
    assert reduce(output, factor=.5, reason='explicit user authorization') == result
    assert historical.read_bytes() == historical_bytes
    assert (output/'budget_amendments'/f"{result['id']}.json").is_file()
    assert authorizes_transition(output, old_limit=60, new_limit=32, run_id='run')
    assert not authorizes_transition(output, old_limit=60, new_limit=33, run_id='run')
    new_ledger = AgentCostLedger(output, run_id='run', limit_usd=32)
    new_ledger.reserve(requested_usd=2, session_id='new', role='scheduler')
    assert authorizes_transition(output, old_limit=60, new_limit=32, run_id='run')
    raw = json.loads(ledger.path.read_text())
    raw['entries'][0]['reserved_usd'] = 0
    atomic_json(ledger.path, raw)
    assert not authorizes_transition(output, old_limit=60, new_limit=32, run_id='run')
    # Resumed initialization uses the existing lower cap, not default 24 hours.
    TaskGPUBudget(output).initialize(output/'checkout')
    with pytest.raises(ValueError, match='cannot change'):
        TaskGPUBudget(output).initialize(output/'checkout', cap_seconds=86400)


@pytest.mark.parametrize('model_addition,wall_addition', [(1.49,0), (1.49,21600), (0,21600)])
def test_explicit_extension_keeps_usage_start_gpu_and_is_idempotent(tmp_path, model_addition, wall_addition):
    output = tmp_path/'run'
    ledger = budgets(output)
    ledger.reserve(requested_usd=4, session_id='old', role='scheduler')
    entries = json.loads(ledger.path.read_text())['entries']
    wall = json.loads((output/'budget.json').read_text())
    gpu = json.loads((output/'task_gpu_budget.json').read_text())
    kwargs = dict(model_usd=model_addition, wall_seconds=wall_addition, reason='explicit operator addition', preserve_stopped_holds=True)
    result = extend(output, **kwargs)
    assert result['after']['model_usd'] == 60 + model_addition
    assert result['after']['wall_seconds'] == wall['wall_seconds'] + wall_addition
    assert result['after']['gpu_seconds'] == gpu['cap_seconds']
    assert result['after']['turn_usd'] == result['before']['turn_usd']
    assert json.loads(ledger.path.read_text())['entries'] == entries
    assert json.loads((output/'budget.json').read_text())['started_epoch'] == wall['started_epoch']
    assert extend(output, **kwargs) == result
    assert authorizes_transition(output, old_limit=60, new_limit=60 + model_addition, run_id='run')
    audit = output/'budget_amendments'/f"{result['id']}.json"
    result['wall_add_seconds'] = 99999
    atomic_json(audit, result)
    assert not authorizes_transition(output, old_limit=60, new_limit=60 + model_addition, run_id='run')
    with pytest.raises(ValueError):
        extend(output, model_usd=float('nan'), wall_seconds=21600, reason='invalid')


def test_budget_amendment_refuses_live_model_reservation(tmp_path):
    output = tmp_path/'run'
    ledger = budgets(output)
    ledger.reserve(requested_usd=4, session_id='live', role='scheduler')
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match='reservations'):
        reduce(output, factor=.5, reason='user')
    assert ledger.path.read_bytes() == before
    entries = json.loads(before)['entries']
    result = reduce(output, factor=.5, reason='operator confirms stopped controller', preserve_stopped_holds=True)
    assert result['stopped_legacy_holds_preserved']
    assert json.loads(ledger.path.read_text())['entries'] == entries


@pytest.mark.parametrize('legacy', [False, True])
def test_model_then_wall_extension_can_resume_old_session(tmp_path, legacy):
    output = tmp_path/'run'
    ledger = budgets(output)
    entries = json.loads(ledger.path.read_text())['entries']
    model = extend(output, model_usd=10, wall_seconds=0, reason='user adds ten dollars')
    wall = extend(output, model_usd=0, wall_seconds=7200, reason='user adds two hours')
    assert wall['previous_amendment_id'] == model['id']
    if legacy:
        wall.pop('previous_amendment_id')
        atomic_json(output/'budget_amendments'/f"{wall['id']}.json", wall)
    assert authorizes_transition(output, old_limit=60, new_limit=70, run_id='run')
    assert not authorizes_transition(output, old_limit=59, new_limit=70, run_id='run')
    assert not authorizes_transition(output, old_limit=60, new_limit=71, run_id='run')
    assert not authorizes_transition(output, old_limit=60, new_limit=70, run_id='other-run')
    assert json.loads(ledger.path.read_text())['entries'] == entries
    # Applied audit history is necessary; a missing/tampered predecessor must not
    # make the latest wall-only authorization authorize a model increase itself.
    model['model_add_usd'] = 11
    atomic_json(output/'budget_amendments'/f"{model['id']}.json", model)
    assert not authorizes_transition(output, old_limit=60, new_limit=70, run_id='run')


def test_multiple_amendments_require_exact_full_budget_chain(tmp_path):
    output = tmp_path/'run'
    budgets(output)
    first = reduce(output, factor=.5, reason='halve remaining budget')
    extend(output, model_usd=10, wall_seconds=0, reason='add ten')
    last = extend(output, model_usd=0, wall_seconds=7200, reason='add wall')
    assert authorizes_transition(output, old_limit=60, new_limit=42, run_id='run')
    assert authorizes_transition(output, old_limit=32, new_limit=42, run_id='run')
    last['previous_amendment_id'] = first['id']
    atomic_json(output/'budget_amendments'/f"{last['id']}.json", last)
    assert not authorizes_transition(output, old_limit=60, new_limit=42, run_id='run')


def test_budget_amendment_refuses_matching_live_process(tmp_path, monkeypatch):
    from autosim.research import process_executor
    output = tmp_path/'run'
    budgets(output)
    atomic_json(output/'processes/live.json', {'process_identity':{'pid':123}})
    monkeypatch.setattr(process_executor, 'inspect_process_identity', lambda *a: {'status':'matching_running'})
    with pytest.raises(ValueError, match='processes'):
        reduce(output, factor=.5, reason='user')


def test_failed_clone_retains_old_prefix_and_does_not_fallback_to_recreating_it(tmp_path, monkeypatch):
    output = tmp_path/'run'
    repo = output/'checkout'
    repo.mkdir(parents=True)
    old = base(output/'env')
    row = pool.describe(base(tmp_path/'source'))
    pool.configure(output, tmp_path/'store')
    monkeypatch.setattr(pool, 'catalog', lambda *a: [row])
    monkeypatch.setattr(pv, 'plan', lambda *a, **k: {'python':'3.10',
        'commands':['create {prefix}'], 'probes':['{python} -c "import task"'],
        'base_environment_id':row['id'], 'environment_selection_reason':'ABI'})
    monkeypatch.setattr(pool, 'clone', lambda *a, **k: {'ok':False, 'command':'clone',
        'returncode':1, 'evidence_id':'a'*32})
    before = (old/'bin/python').read_bytes()
    first = pv.build(repo, client=object(), output=output, prefix=old,
        python=str(old/'bin/python'), manifests={}, assets={},
        base_environment_id=row['id'], environment_selection_reason='ABI')
    assert first['verdict']['reason'] == 'base_clone_failed'
    monkeypatch.setattr(pv, 'plan', lambda *a, **k: pytest.fail('no scratch fallback after clone failure'))
    second = pv.build(repo, client=object(), output=output, prefix=old,
        python=str(old/'bin/python'), manifests={}, assets={})
    assert second['verdict']['passed'] is False
    assert (old/'bin/python').read_bytes() == before
