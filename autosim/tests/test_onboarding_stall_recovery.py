import json
import threading
from types import SimpleNamespace

import pytest

from autosim.research import provision, recorder, installation_recovery
from autosim.research.common import atomic_json, object_digest
from autosim.research.prepare import Preparation
from autosim.research.agent_budget import AgentCostLedger
from autosim.research.execution_progress import execution_progress


def test_current_prefix_and_queue_exclude_legacy_failure(tmp_path):
    cursor = {'prefix':str(tmp_path/'new'), 'planned':{'commands':['install']},
              'pending':['install'], 'failed_probe':{}}
    old = {'template':'install', 'command':str(tmp_path/'old/bin/pip')+' install',
           'ok':False, 'evidence_id':'old'}
    atomic_json(tmp_path/'provision_cursor.json', cursor)
    atomic_json(tmp_path/'transcript.json', {'rows':[old]})
    assert provision.latest_failure(tmp_path, {'latest_failure':old}) == {}
    current = {**old, 'environment_prefix':cursor['prefix'], 'evidence_id':'new'}
    atomic_json(tmp_path/'transcript.json', {'rows':[old,current]})
    assert provision.latest_failure(tmp_path, {'latest_failure':old}) == current
    success = {**current, 'ok':True}
    atomic_json(tmp_path/'transcript.json', {'rows':[old,current,success]})
    assert provision.latest_failure(tmp_path, {'latest_failure':old}) == {}


def test_repair_ahead_of_original_does_not_hide_current_failure():
    cursor = {'prefix':'/run/env','pending':['repair','original']}
    failure = {'template':'original','environment_prefix':'/run/env',
               'ok':False,'evidence_id':'failed'}
    assert provision.current_queue_failure(cursor, [failure]) == failure
    assert provision.current_queue_failure(cursor, [failure], head_only=True) == {}
    assert provision.current_queue_failure({**cursor,'context_identity':'new'},
        [{**failure,'plan_identity':'old'}]) == {}


@pytest.mark.parametrize('scope', ['pause','framework'])
def test_pause_is_not_a_repository_impossibility_claim(tmp_path, scope):
    state = SimpleNamespace(output=tmp_path, interpreter=None, budget=None)
    choice = {'do':'stop','why':'contract unresolved','arguments':{'stop_scope':scope}}
    assert Preparation._review_environment_stop(state, choice) == choice


def test_stop_review_uses_exposed_snapshot_not_a_new_scan(tmp_path, monkeypatch):
    resources = {'local_wheels':[], 'environment_candidates':[], 'scan_complete':True}
    resources['digest'] = object_digest(resources)
    atomic_json(tmp_path/'installation_recovery_inventory.json', resources)
    atomic_json(tmp_path/'environment.json', {'latest_failure':{'evidence_id':'failed'}})
    seen = []
    def review(*args, **kwargs):
        seen.append(kwargs['resources'])
        return {'approved':True}
    monkeypatch.setattr(installation_recovery, 'review_boundary', review)
    state = SimpleNamespace(output=tmp_path, interpreter=None, budget=None, client=object())
    state._installation_recovery_inventory = lambda: pytest.fail('moving inventory was rescanned')
    choice = {'do':'stop','why':'resource unavailable','arguments':{}}
    assert Preparation._review_environment_stop(state, choice) == choice
    assert seen == [resources]


def test_resource_assessment_names_exact_invalid_fields():
    with pytest.raises(ValueError, match='local_artifacts must be') as exc:
        installation_recovery.review_boundary(None, resources={'digest':'same'}, reason='x',
            assessment={'inventory_digest':'same','local_artifacts':{},
                        'environment_reuse':'evidence','alternative_sources':'evidence'}, failure={})
    assert 'inventory_digest does not match' not in str(exc.value)


def test_inventory_snapshot_remains_bound_after_latest_view_changes(tmp_path, monkeypatch):
    snapshot = installation_recovery.inventory(tmp_path)
    frozen = tmp_path/'installation_inventory_snapshots'/f"{snapshot['digest']}.json"
    assert frozen.is_file()
    atomic_json(tmp_path/'installation_recovery_inventory.json', {'digest':'different'})
    atomic_json(tmp_path/'environment.json', {'latest_failure':{'evidence_id':'failed'}})
    seen = []
    monkeypatch.setattr(installation_recovery, 'review_boundary', lambda *a, **kw:
        seen.append(kw['resources']['digest']) or {'approved':True})
    state = SimpleNamespace(output=tmp_path, interpreter=None, budget=None, client=object())
    choice = {'do':'stop','why':'resource boundary','arguments':{
        'resource_assessment':{'inventory_digest':snapshot['digest']}}}
    assert Preparation._review_environment_stop(state, choice) == choice
    assert seen == [snapshot['digest']]
    value = json.loads(frozen.read_text())
    value['scan_complete'] = not value['scan_complete']
    atomic_json(frozen, value)
    assert Preparation._review_environment_stop(state, choice)['decision_failure']['kind'] == 'resource_boundary_unproven'
    assert len(seen) == 1


def test_environment_switch_retires_old_recovery_obligation(tmp_path):
    state = SimpleNamespace(steps=[
        {'step':'build_the_environment','outcome':'checkpoint failure','failure_domain':'native_install'},
        {'step':'build_the_environment','outcome':'checkpoint','arguments':{'base_environment_id':'new'}},
        {'step':'choose','outcome':'decision_rejected'}])
    assert Preparation._environment_epoch_steps(state) == state.steps[1:]


def test_run_report_exposes_current_queue_not_old_plan(tmp_path):
    snapshot = {'status':'paused','measurements':[], 'revision':'snapshot','event_revision':'event',
        'actions':[],'plan':{},'budget':{},'verified_media':[], 'run_id':'run',
        'environment_queue':{'prefix':'env/new','python':'3.10','passed':False,
                             'pending':['install current dependency'],'next_probe':0,'probe_count':2}}
    text = recorder.render(tmp_path, snapshot, {})
    assert '当前计划 Python：3.10' in text
    assert 'install current dependency' in text
    assert '不是历史环境的失败命令' in text


@pytest.mark.parametrize('category', ['resource_boundary_unproven','invalid_stop_scope'])
def test_resource_format_retries_are_bounded_without_monitor(tmp_path, category):
    state = SimpleNamespace(output=tmp_path, state_persistence_error='',
        client=SimpleNamespace(as_role=lambda *a: None))
    statuses = []
    state._persist_run_state = lambda **kw: statuses.append(kw['status'])
    failure = {'failure_category':category,'evidence_id':'old','because':'missing string'}
    assert Preparation._handoff_scheduler_decision_failure(state, failure=failure)
    assert Preparation._handoff_scheduler_decision_failure(state, failure=failure)
    assert not Preparation._handoff_scheduler_decision_failure(state, failure=failure)
    assert statuses == ['running','running','paused']
    assert state.last_action['recovery']['status'] == 'blocked'


@pytest.mark.parametrize('scope,status', [('pause','paused'),('framework','infrastructure_blocked')])
def test_stop_scope_survives_whole_controller_cycle(tmp_path, scope, status):
    class Client:
        model = 'fixture-no-provider'
        def chat_with_metadata(self, *args, **kwargs):
            return json.dumps({'do':'stop','why':'unresolved contract, not impossible repository',
                               'arguments':{'stop_scope':scope}}), {}
    controller = Preparation(repo=tmp_path, output=tmp_path/'out',
                             client=Client(), scouting=tmp_path/'scouting')
    report = controller.run(max_steps=1, max_relaunch=3)
    assert report['status'] == status
    assert report['steps'][-1]['outcome'] == status
    assert report['supervision']['scheduler_segments'] == 1


def test_interleaved_rejections_are_stagnation_not_progress():
    steps = []
    for _ in range(3):
        steps.extend([{'step':'choose','outcome':'decision_rejected'},
                      {'step':'monitor','outcome':'reported'}])
    assert execution_progress(steps)['needs_strategy_change']


def test_settlement_wait_preserves_unknown_and_observes_real_settlement(tmp_path):
    ledger = AgentCostLedger(tmp_path, run_id='run', limit_usd=1)
    old = ledger.reserve(session_id='old', role='scheduler', requested_usd=.8)
    ledger.settle(old['reservation_id'], actual_usd=None, launched=True)
    before = ledger.path.read_bytes()
    assert ledger.wait_for_settlement(timeout=0, minimum_usd=.25)['status'] == 'blocked'
    assert ledger.path.read_bytes() == before
    # This tests execution settlement, not Recorder's separate recovery floor.
    live = ledger.reserve(session_id='live', role='fix', requested_usd=.2)
    def finish():
        ledger.settle(live['reservation_id'], actual_usd=.01, launched=True)
    worker = threading.Timer(.01, finish)
    worker.start()
    result = ledger.wait_for_settlement(timeout=1, minimum_usd=.15)
    worker.join()
    assert result['status'] == 'available'
    assert json.loads(ledger.path.read_text())['entries'][0]['status'] == 'unknown'


def test_recorder_does_not_purchase_narration_for_rejected_decision(tmp_path):
    client = SimpleNamespace(supports_recorder=True)
    snapshot = {'status':'running','source_consistency':'stable',
                'actions':[{'step':'choose','outcome':'decision_rejected'}]}
    assert recorder.narrate(tmp_path, snapshot, client=client) == {}
    assert not (tmp_path/'report/recorder_trigger.json').exists()


def test_settlement_retries_remain_bounded_and_keep_hold_classification(tmp_path):
    ledger = AgentCostLedger(tmp_path, run_id='run', limit_usd=1)
    reservation = ledger.reserve(session_id='old', role='scheduler', requested_usd=.1)
    ledger.settle(reservation['reservation_id'], actual_usd=None, launched=True)
    statuses = []
    state = SimpleNamespace(output=tmp_path, budget=None)
    state._persist_run_state = lambda **kw: statuses.append(kw['status'])
    assert Preparation._await_model_settlement(state)
    assert Preparation._await_model_settlement(state)
    assert not Preparation._await_model_settlement(state)
    assert state._budget_settlement_pending
    assert statuses == ['budget_waiting','budget_waiting']
