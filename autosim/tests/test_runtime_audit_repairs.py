"""Cross-layer audit regressions; no model calls or real benchmark mutations."""
import json
from pathlib import Path
import pytest

from autosim.research import native_jobs, provision, recovery_contract
from autosim.research.common import atomic_json, read_json
from autosim.research.main_agent import receive_report
from autosim.research.process_executor import ProcessAttempt
from autosim.research.runtime_recovery import seal_protocol_stream
from tests.test_main_agent import make, Client, REPORT


def test_report_recovery_preserves_original_and_only_requests_format(tmp_path):
    original = json.dumps({**REPORT, 'findings':['汉'*1000]*8,
                           'uncertainties':['未知'*700]*8}, ensure_ascii=False)
    class CompressionClient:
        def chat_with_metadata(self, system, user, **kwargs):
            payload = json.loads(user)
            assert 'already completed investigation' in system
            assert kwargs['read_only'] is True
            assert kwargs['auto_skills'] is False
            assert payload['original_report_complete'] is True
            assert '16000' in payload['validation_error']
            return json.dumps(REPORT), {}
    result = receive_report(original, client=CompressionClient(), output=tmp_path,
        workspace=tmp_path/'checkout', timeout=20)
    assert result['summary'] == REPORT['summary']
    assert result['delivery_recovery_evidence_ref']
    raw = next((tmp_path/'report_handoffs').glob('*.log')).read_text()
    assert '汉'*1000 in raw


def test_valid_report_never_adds_a_model_turn(tmp_path):
    class NoCalls:
        def chat_with_metadata(self, *args, **kwargs):
            raise AssertionError('valid handoff should not trigger compression')
    assert receive_report(json.dumps(REPORT), client=NoCalls(), output=tmp_path,
        workspace=tmp_path/'checkout', timeout=20)['summary'] == REPORT['summary']


def test_inspect_task_preserves_verified_commands_and_requests_readonly(tmp_path):
    client = Client(); client.timeout = 900
    prep = make(tmp_path, client)
    prep.stages = {'train':'verified trainer'}
    atomic_json(prep.output/'derived_stages.json', {'train':{'source':'verified'}})
    result = prep.do('research_task', role='init', task='Inspect actual loader',
        expected_result='No edits; evidence only', mode='inspect', timeout_seconds=700)
    assert result['outcome'] == 'reported'
    assert prep.stages == {'train':'verified trainer'}
    assert read_json(prep.output/'derived_stages.json')['train']['source'] == 'verified'
    assert not prep.main_agent.get('checkout_needs_resurvey')


def test_segment_budget_excludes_history_and_derive_substeps(tmp_path):
    prep = make(tmp_path)
    prep.steps = [{'step':'build_the_environment'}]*100
    prep._cycle_step_start = len(prep.steps)
    prep._step_budget = 8
    assert prep._remaining_actions() == 8
    prep.steps += [{'step':'derive:train'}, {'step':'derive:train'}, {'step':'derive_a_command'}]
    assert prep._remaining_actions() == 7


@pytest.mark.parametrize('observation', ['matching_running','unverifiable','identity_mismatch'])
def test_dead_job_reconciliation_never_releases_unproven_worker(tmp_path, monkeypatch, observation):
    identity = 'a'*32; root = tmp_path/'native_jobs'/identity
    atomic_json(root/'request.json', {'job_id':identity,'run_id':'derived','stage':'train'})
    atomic_json(root/'control.json', {'state':'running'})
    atomic_json(root/'worker_identity.json', {'recorded':'identity'})
    monkeypatch.setattr(native_jobs, 'inspect_process_identity', lambda _: {'status':observation})
    with pytest.raises(ValueError, match='not proven absent'):
        native_jobs.reconcile(tmp_path, identity, reason='audit')
    assert identity in native_jobs.active_jobs(tmp_path)


def test_dead_worker_with_live_native_child_remains_owned(tmp_path, monkeypatch):
    identity = 'b'*32; root = tmp_path/'native_jobs'/identity
    atomic_json(root/'request.json', {'job_id':identity,'run_id':'derived','stage':'train'})
    atomic_json(root/'control.json', {'state':'running'})
    atomic_json(root/'worker_identity.json', {'kind':'worker'})
    atomic_json(tmp_path/'research/derived/attempts/child/receipt.json', {
        'controller_decision_id':identity,'status':'running','process_identity':{'kind':'child'}})
    monkeypatch.setattr(native_jobs, 'inspect_process_identity', lambda row: {
        'status':'not_running' if row.get('kind') == 'worker' else 'matching_running'})
    with pytest.raises(ValueError, match='native child'):
        native_jobs.reconcile(tmp_path, identity, reason='audit')
    assert identity in native_jobs.active_jobs(tmp_path)


def test_unreadable_job_is_visible_and_corrupt_result_retains_ownership(tmp_path):
    prep = make(tmp_path)
    identity = 'c'*32; root = prep.output/'native_jobs'/identity
    atomic_json(root/'request.json', {'job_id':identity,'stage':'train'})
    atomic_json(root/'control.json', {'state':'running'})
    (root/'result.json').write_text('{broken')
    assert identity in native_jobs.active_jobs(prep.output)
    view = prep._native_job_view()
    assert view[0]['job_id'] == identity
    assert view[0]['status'] == 'unreadable'
    assert view[0]['failure']


def test_scheduler_submitted_repair_uses_same_independent_review(tmp_path, monkeypatch):
    class NoFixTurn:
        output = tmp_path
        def chat_with_metadata(self, *args, **kwargs):
            raise AssertionError('submitted proposal must not be rewritten by a new Fix turn')
    seen = []
    monkeypatch.setattr(provision, 'platform_facts', lambda: {})
    monkeypatch.setattr(provision, 'review_install_replacement',
        lambda *args, **kwargs: seen.append(args[2]['evidence_id']) or {'approved':True})
    transcript = []
    commands, probes = provision.resume(NoFixTurn(), tmp_path, [], {'evidence_id':'d'*32},
        manifests={}, transcript=transcript, values={}, submitted_proposal={
            'failure_evidence_id':'d'*32,'commands':['echo corrected'],
            'repair_mode':'replace_operation','install_replacement_evidence':{
                'same_capability':'source-backed correction','source_refs':['setup.py'],
                'failure_evidence_id':'d'*32}},
        current_probes=['original consumer'])
    assert commands == ['echo corrected'] and probes is None
    assert seen == ['d'*32]
    assert transcript[-1]['replacement_review']['approved'] is True


def test_stale_proposal_is_rejected_without_new_model_or_native_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(provision, 'platform_facts', lambda: {})
    transcript = []
    commands, probes = provision.resume(object(), tmp_path, [], {'evidence_id':'d'*32},
        manifests={}, transcript=transcript, values={}, submitted_proposal={
            'failure_evidence_id':'e'*32,'commands':['echo stale']})
    assert commands == [] and probes is None
    assert transcript[-1]['native_operation_launched'] is False


def test_alias_repair_rejects_stale_resource_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(provision, 'platform_facts', lambda: {})
    transcript = []
    commands, _ = provision.resume(object(), tmp_path, [], {'evidence_id':'d'*32},
        manifests={}, transcript=transcript, values={}, submitted_proposal={
            'failure_evidence_id':'d'*32,'commands':['install {run_path_1}'],
            'execution_aliases_digest':'obsolete'})
    assert commands == []
    assert 'alias catalog changed' in transcript[-1]['reason']


def test_rejected_proposal_is_not_relabelled_as_new_install_failure(tmp_path, monkeypatch):
    prep = make(tmp_path)
    prep.declaration = {'assets':{}}
    monkeypatch.setattr(prep, '_surveyed_runnable_stages', lambda: {'train':{'entrypoint':'train.py'}})
    monkeypatch.setattr(prep, '_select_execution_path', lambda **_: {'stages':['train'],'asset_keys':[],'why':'fixture'})
    monkeypatch.setattr(provision, 'build', lambda *args, **kwargs: {
        'status':'yielded','new_native_operation':False,'latest_attempt':{'ok':False},
        'repair_rejection':{'reason':'review rejected source citation'},
        'latest_failure':{'evidence_id':'d'*32}})
    result = prep._step_build_the_environment()
    assert result['outcome'] == 'repair proposal rejected'
    assert result['native_operation_launched'] is False
    assert result['parent_evidence_id'] == 'd'*32
    recovery = recovery_contract.view([{'step':'build_the_environment',**result}], {})
    assert recovery['unresolved_failure']['next_action'] == 'build_the_environment'


def test_protocol_raw_stream_is_sealed_and_credentials_removed(tmp_path):
    attempt = ProcessAttempt(True, 0, '{"type":"system","estimated_',
                             'authorization: api-secret-value\n')
    receipt = seal_protocol_stream(tmp_path, tmp_path/'checkout', turn_id='f'*32,
        attempt=attempt, secrets=('api-secret-value',))
    log = tmp_path/receipt['log_ref']
    text = log.read_text()
    assert 'estimated_' in text
    assert 'api-secret-value' not in text
    record = next((tmp_path/'agent/stream_failures').glob('*.json'))
    assert read_json(record)['stdout_newline_terminated'] is False
