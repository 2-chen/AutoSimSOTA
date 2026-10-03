"""Audit regressions: assert repaired behavior, using only temporary fixtures."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

from autosim.research import agent_tasks, native_jobs, provision, run_record
from autosim.research.common import atomic_json, digest, object_digest
from autosim.research.budget import RunBudget
from autosim.research.main_agent import validate_report
from tests.test_main_agent import make
from autosim.research.prepare import _latest_per_step


def fixture(tmp_path):
    repo = tmp_path / 'checkout'; repo.mkdir()
    (repo / 'train.py').write_text('print(1)\n')
    entries = [['file', 'train.py', 9, digest(repo / 'train.py'), 0o644]]
    atomic_json(tmp_path / 'workspace_snapshot.json', {
        'schema_version': 2, 'destination': str(repo.resolve()),
        'source_entries': entries, 'source_tree_fingerprint': object_digest(entries)})
    atomic_json(tmp_path / 'derived_stages.json', {})
    return repo


def test_new_runtime_source_is_not_part_of_native_job_identity(tmp_path):
    repo = fixture(tmp_path)
    dependency = repo / 'new_dependency.py'; dependency.write_text('API = 1\n')
    before = native_jobs.source_identity(tmp_path, repo)
    dependency.write_text('API = 2\n')
    assert before != native_jobs.source_identity(tmp_path, repo)


def test_mixed_scope_silently_omits_materialized_source(tmp_path):
    repo = fixture(tmp_path)
    (repo / 'new_dependency.py').write_text('API = 1\n')
    _, reader = agent_tasks.snapshot(tmp_path, repo, 'a'*32, source_paths=['.'])
    assert (reader / 'train.py').exists()
    assert (reader / 'new_dependency.py').exists()


def test_cancel_dead_native_worker_does_not_free_exclusive_owner(tmp_path):
    identity = 'a'*32; root = tmp_path / 'native_jobs' / identity
    atomic_json(root / 'request.json', {'job_id': identity, 'stage': 'train'})
    atomic_json(root / 'control.json', {'state': 'running', 'cancelled': False})
    with patch.object(native_jobs, 'inspect_process_identity', return_value={'status':'not_running'}):
        atomic_json(root / 'worker_identity.json', {})
        assert native_jobs.status(tmp_path, identity)['status'] == 'outcome_unknown'
        cancelled = native_jobs.cancel(tmp_path, identity, reason='dead worker')
    assert cancelled['status'] == 'cancelled'
    assert identity not in native_jobs.active_jobs(tmp_path)
    assert (root / 'result.json').exists()


def test_popen_failure_leaves_unrecoverable_active_job(tmp_path):
    repo = fixture(tmp_path)
    RunBudget(tmp_path, wall_seconds=60)
    atomic_json(tmp_path / 'derived_stages.json', {'train':{'source':'verified'}})
    with patch.object(native_jobs.subprocess, 'Popen', side_effect=OSError('launch failed')):
        try:
            native_jobs.submit(tmp_path, repo=repo, stage='train', settings={'device':'cpu'},
                               requested_seconds=1, reason='audit')
        except OSError:
            pass
        else:
            raise AssertionError('expected launch failure')
    assert native_jobs.active_jobs(tmp_path) == []
    results = list((tmp_path / 'native_jobs').glob('*/result.json'))
    assert len(results) == 1
    assert json.loads(results[0].read_text())['launched'] is False


def test_live_agent_elapsed_drops_to_zero_when_fallback_label_differs(tmp_path):
    atomic_json(tmp_path / 'run_state.json', {'status':'running', 'current_action':{
        'step':'coding_agent_turn', 'started_at':'2026-10-01T00:00:00+00:00'}})
    target = tmp_path / 'RUN.md'
    target.write_text('# Audit\n')
    run_record.refresh_live_status(tmp_path, fallback_status='running',
                                  fallback_current='waiting for model/tool event')
    text = target.read_text()
    assert '（约 0 秒）' not in text


def test_supervision_fingerprint_changes_on_environment_telemetry_only(tmp_path):
    prep = make(tmp_path)
    facts = {'environment':{'latest_installation_attempt':{'evidence_id':'same','seconds':1}}}
    with patch.object(prep, 'state', return_value=facts):
        before = prep._supervision_fingerprint()
        facts['environment']['latest_installation_attempt']['seconds'] = 2
        assert before == prep._supervision_fingerprint()


def test_valid_field_lengths_can_still_exceed_report_byte_contract():
    report = {'summary':'摘要', **{key:['汉'*1000]*4 for key in (
        'findings','uncertainties','evidence_refs','recommended_next_actions')}}
    assert all(len(v) <= 20 for k,v in report.items() if isinstance(v,list))
    try:
        validate_report(report)
    except ValueError as exc:
        assert '16000 bytes' in str(exc)
    else:
        raise AssertionError('expected whole-report rejection')


def test_recent_actions_are_group_order_not_actual_recent_order():
    rows = [{'step':'build_the_environment','outcome':'checkpoint failure'}]
    rows += [{'step':f'other_{i}','outcome':'done'} for i in range(6)]
    rows += [{'step':'build_the_environment','outcome':'checkpoint failure'}]
    recent = _latest_per_step(rows)[-6:]
    assert recent[-1]['step'] == 'build_the_environment'


def test_large_legitimate_source_is_rejected_as_unsafe(tmp_path):
    (tmp_path / 'large.py').write_text('# public source\n' * 5000)
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            payload = json.loads(user)
            assert payload['source_ranges']['large.py']['start_line'] == 3000
            assert len(payload['sources']['large.py']) < 65536
            return json.dumps({'approved':True, 'reason':'correct source',
                'citations':[{'file':'large.py', 'quote':'# public source'}]}), {}
    result = provision.review_probe_replacement(Client(), tmp_path, {}, ['fixed'],
        {'same_capability':'same native consumer', 'source_refs':['large.py:3000-3002']})
    assert result['approved'] is True


def test_approved_unbuildable_is_downgraded_to_checkpoint_on_queue_refresh(tmp_path):
    repo = tmp_path / 'checkout'; repo.mkdir()
    identity = 'c'*32
    calls = []
    def execute(command, **kwargs):
        log = tmp_path / 'failure.log'; log.write_text('failed')
        from autosim.research.evidence_store import capture_attempt_evidence
        capture_attempt_evidence(tmp_path, attempt_id=identity, log=log,
            receipt_ref='receipt.json', status='failed', returncode=1,
            termination_reason='failed')
        return {'command':command, 'ok':False, 'returncode':1, 'seconds':.01,
                'failure_kind':'failed', 'evidence_id':identity}
    def repair(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return [], None
        kwargs['transcript'].append({'kind':'unbuildable','reason':'reviewed boundary',
                                    'boundary_review':{'approved':True}})
        return [], None
    seed = {'python':sys.executable,'templates':['false'],'probes':['{python} -c "import json"']}
    with patch.object(provision, 'run', execute), patch.object(provision, 'resume', repair):
        provision.build(repo, client=object(), prefix=tmp_path/'env', output=tmp_path,
            python=sys.executable, manifests={}, assets={}, max_operations=1, max_rounds=1, seed=seed)
        result = provision.build(repo, client=object(), prefix=tmp_path/'env', output=tmp_path,
            python=sys.executable, manifests={}, assets={}, max_operations=1, max_rounds=1, seed=seed)
    assert result['verdict']['reason'] == 'agent_declared_unbuildable'
