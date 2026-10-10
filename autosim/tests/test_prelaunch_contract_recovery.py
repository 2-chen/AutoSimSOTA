"""Real argv rejection, sealed evidence, idle checkpoint and legacy reconciliation."""
import json
import sys

import pytest

from autosim.research.common import atomic_json, read_json
from autosim.research.declarative_backend import DeclarativeBackend
from autosim.research.execution_derive import coalesce_dynamic_overrides, conflicting_option_problem
from autosim.research.evidence_store import read_attempt_evidence
from autosim.research.ideas import Idea


@pytest.mark.parametrize('flag', ['--set', '--override', '--setting'])
@pytest.mark.parametrize('inline', [False, True])
def test_append_assignments_preserve_cli_groups_and_per_key_precedence(tmp_path, flag, inline):
    entry = tmp_path / 'entry.py'
    entry.write_text(f"import argparse\np=argparse.ArgumentParser()\np.add_argument('{flag}', action='append')\n")
    def group(v):
        return [flag + '=' + v] if inline else [flag, v]
    argv = [sys.executable, str(entry), *group('loader.workers=0'),
            *group('optimizer.lr=0.001'), *group('loader.workers=4')]
    assert 'conflicting' in conflicting_option_problem(argv)
    resolved = coalesce_dynamic_overrides(argv, {'loader.workers': 4})
    assert resolved == [sys.executable, str(entry), *group('optimizer.lr=0.001'), *group('loader.workers=4')]
    assert not conflicting_option_problem(resolved)


def test_append_permission_requires_native_source_and_scalar_options_remain_strict(tmp_path):
    entry = tmp_path / 'entry.py'
    entry.write_text('import argparse\np=argparse.ArgumentParser()\np.add_argument("--name")\n')
    argv = [sys.executable, str(entry), '--name', 'a=1', '--name', 'b=2']
    assert conflicting_option_problem(argv)
    assert conflicting_option_problem([sys.executable, str(entry), '--steps', '100', '--steps', '200'])
    assert not conflicting_option_problem(['tool', '--', '--steps', '100', '--steps', '200'])


def test_relative_entry_point_uses_native_working_directory(tmp_path):
    (tmp_path / 'train.py').write_text('import argparse\np=argparse.ArgumentParser()\np.add_argument("--knob", action="append")\n')
    backend = DeclarativeBackend(repo=tmp_path,
        answer={'stages': {'train': {'available': True}}},
        sources={'train': 'def stage_argv_train(i):\n    return [i["python"], "train.py", "--knob", "a=1", "--knob", "b=2"]'})
    assert backend.argv('train', {'python': sys.executable})[-4:] == ['--knob', 'a=1', '--knob', 'b=2']


def test_assignment_like_scalar_values_are_not_rewritten_as_config(tmp_path):
    entry = tmp_path / 'entry.py'
    entry.write_text('import argparse\np=argparse.ArgumentParser()\np.add_argument("--output")\n')
    for group in (['--output', 'loader.workers=0'], ['--output=loader.workers=0']):
        argv = [sys.executable, str(entry), *group]
        assert coalesce_dynamic_overrides(argv, {'loader.workers': 4}) == argv


def test_stage_argv_refusal_returns_sealed_unlaunched_receipt(tmp_path):
    from tests.test_derived_research import research
    real = research(tmp_path)
    source = 'def stage_argv_train(i):\n    return [i["python"], "train.py", "--size=1", "--size=2"]\n'
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                     sources={**real.sources, 'train': source})
    result = real.run_stage('train', settings={'train.n_epochs': 2})
    assert result['ran'] is False and result['command_started'] is False
    assert result['termination_reason'] == 'operation_contract_failed'
    evidence = read_attempt_evidence(real.output, result['evidence_id'])
    assert 'ambiguous command' in evidence['text'] and evidence['returncode'] is None
    assert not result.get('process_identity')
    assert not list(real.stage_directory('train').glob('native_outputs/*/models/*'))


def test_candidate_prelaunch_failure_finalizes_and_returns_idle_with_fix_evidence(tmp_path):
    from tests.test_derived_research import research
    real = research(tmp_path)
    real.library.add(Idea(label='more', granularity='param', change={'train.n_epochs': 2},
                          risk='low', crosses='none', why='probe worker configuration'))
    train = real.sources['train'].replace('    return argv',
        '    if i["settings"].get("train.n_epochs") == 2:\n'
        '        argv = argv + ["--bad=1", "--bad=2"]\n    return argv')
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                     sources={**real.sources, 'train': train},
                                     parameters={'train': {'train.n_epochs': {'value': '1'}}})
    real.sources = real.backend.sources
    baseline = real.run(rounds=2, yield_after_action=True)
    baseline_bytes = (real.run_root / 'measurements/baseline.json').read_bytes()
    real.library.audit(real.red_lines())
    result = real.run(rounds=2, yield_after_action=True, selected_idea_label='more',
                      main_controller_owns_selection=True)
    session = read_json(real.run_root / 'controller_session.json')
    assert result['run_status'] == session['status'] == 'paused'
    assert session['pending_action'] is None
    failed = read_json(real.run_root / 'measurements/round_1.json')
    assert failed['ok'] is False and failed['train']['ran'] is False
    assert len(failed['train']['evidence_id']) == 32
    assert (real.run_root / 'measurements/baseline.json').read_bytes() == baseline_bytes
    assert result['rounds'][-1]['status'] == 'nothing was measured'


def legacy_session(tmp_path):
    from tests.test_derived_research import research
    from autosim.research.action_receipt import ActionReceipt, write_action_receipt
    real = research(tmp_path)
    session = {'schema_version': 1, 'run_id': real.run_id, 'repository': str(real.repo),
               'status': 'running', 'action': 'round_1', 'next_round': 1,
               'history': [{'label': 'baseline', 'metric_value': .98}], 'rounds': 2,
               'pending_action': {'round': 1, 'phase': 'measurement_starting',
                                 'idea': {'label': 'workers'}, 'changed': {},
                                 'settings': {'train.n_epochs': 2},
                                 'updated_at': '2026-10-05T09:39:22+00:00'}}
    atomic_json(real.run_root / 'controller_session.json', session)
    ref, receipt = write_action_receipt(real.output, ActionReceipt(
        action_id='a' * 16, decision_id='a' * 16, run_id=real.run_id,
        repository=str(real.repo), action='run_the_loop', state_revision=1,
        started_at='2026-10-05T09:39:14+00:00', finished_at='2026-10-05T09:39:24+00:00',
        status='raised', reason='ValueError: stage_argv_train produced an ambiguous command: bad',
        arguments={'idea_label': 'workers'}))
    steps = [{'step': 'run_the_loop', 'outcome': 'raised',
              'arguments': {'idea_label': 'workers'}, 'receipt_ref': ref,
              'receipt_sha256': receipt['receipt_sha256']}]
    return real, session, steps


@pytest.mark.parametrize('fault', ['', 'changed', 'native', 'measurement', 'receipt', 'wrong_run', 'other_error'])
def test_legacy_reconciliation_is_explicit_narrow_and_preserves_history(tmp_path, fault):
    from autosim.research.prelaunch_recovery import inspect, reconcile
    real, session, steps = legacy_session(tmp_path)
    path = real.run_root / 'controller_session.json'
    if fault == 'changed':
        session['pending_action']['changed'] = {'file': 'policy.py'}
    if fault == 'wrong_run':
        session['run_id'] = 'other'
    if fault in {'changed', 'wrong_run'}:
        atomic_json(path, session)
    if fault == 'native':
        atomic_json(real.run_root / 'attempts' / ('b' * 32) / 'receipt.json',
                    {'started_at': '2026-10-05T09:39:23+00:00', 'ran': False})
    if fault == 'measurement':
        atomic_json(real.run_root / 'measurements/round_1.json', {'ok': False})
    if fault in {'receipt', 'other_error'}:
        p = real.output / steps[0]['receipt_ref']
        row = read_json(p); row['reason'] = 'some arbitrary unknown error'
        if fault == 'other_error':
            from autosim.research.common import object_digest
            row['receipt_sha256'] = object_digest({k: v for k, v in row.items() if k != 'receipt_sha256'})
            steps[0]['receipt_sha256'] = row['receipt_sha256']
        atomic_json(p, row)
    before = path.read_bytes()
    status = inspect(real.output, real.repo, real.run_id, steps)
    assert bool(status) == (not fault)
    if fault:
        assert reconcile(real, steps)['outcome'] == 'blocked'
        assert path.read_bytes() == before
        return
    result = reconcile(real, steps)
    after = read_json(path)
    assert result['outcome'] == 'reconciled'
    assert after['status'] == 'paused' and after['pending_action'] is None
    for key in ['history', 'rounds', 'next_round']:
        assert after[key] == session[key]
    assert after['prelaunch_reconciliation']['pending_action'] == session['pending_action']
    assert not list((real.run_root / 'attempts').glob('*/receipt.json'))
    assert not inspect(real.output, real.repo, real.run_id, steps)


def test_generic_operation_exception_is_citable_without_claiming_no_native_work(tmp_path, monkeypatch):
    from tests.test_prepare import preparation
    real = preparation(tmp_path)
    def handler(**kwargs):
        raise ValueError('one bounded controller error')
    monkeypatch.setattr(real, '_step_read_the_checkout', handler)
    result = real._do_operation('read_the_checkout')
    assert result['outcome'] == 'raised'
    assert result['native_operation_status'] == 'not_inferred'
    assert 'one bounded controller error' in read_attempt_evidence(real.output, result['evidence_id'])['text']


def test_evidence_identity_cannot_overwrite_or_escape(tmp_path):
    from autosim.research.operation_failure import seal
    identity = 'c' * 32
    sealed = seal(tmp_path, operation='construction', exc=ValueError('original'), attempt_id=identity)
    with pytest.raises(ValueError, match='already exists'):
        seal(tmp_path, operation='construction', exc=ValueError('replacement'), attempt_id=identity)
    assert 'original' in read_attempt_evidence(tmp_path, sealed['evidence_id'])['text']
    with pytest.raises(ValueError, match='unsafe'):
        seal(tmp_path, operation='construction', exc=ValueError('bad'), attempt_id='../escape')


def test_main_exposes_legacy_reconciliation_before_new_experiments(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    real, session, steps = legacy_session(tmp_path)
    class Client:
        model = 'fixture'
        supports_main_agent = True
    parent = Preparation(repo=real.repo, output=real.output, client=Client(), scouting=tmp_path/'scouting')
    parent.steps = steps
    monkeypatch.setattr(parent, '_research_controller', lambda: (real, {}))
    ops = parent._available_operations({'status': 'running'}, {})
    assert 'reconcile_interrupted_action' in ops and 'run_the_loop' not in ops
    result = parent._step_reconcile_interrupted_action()
    assert result['outcome'] == 'reconciled'
    assert read_json(real.run_root / 'controller_session.json')['status'] == 'paused'
