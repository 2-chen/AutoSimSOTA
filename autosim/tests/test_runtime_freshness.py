"""Replay stale budget and framework evidence after an operator resumes the same run."""
import copy

import pytest

from autosim.research.agent_budget import AgentCostLedger
from autosim.research.common import atomic_json, read_json
from autosim.research.action_receipt import ActionReceipt, write_action_receipt
from autosim.research.runtime_freshness import (
    budget_observation, current_model_budget, controller_context, controller_identity)
from tests.test_main_agent import make


def old_observation():
    return {'status':'budget_exhausted','failure_category':'run_model_budget',
        'recorded_at':'2026-10-02T04:23:30Z',
        'run_budget':{'limit_usd':3,'remaining_usd':0,'entry_count':2}}


def test_historical_budget_snapshot_is_retained_but_not_current_admission():
    original = old_observation()
    before = copy.deepcopy(original)
    result = budget_observation(original,{'status':'verified_current','remaining_usd':11})
    assert original==before
    assert result['status']=='historical_budget_exhausted'
    assert result['run_budget']==before['run_budget']
    assert result['original_status']=='budget_exhausted'


@pytest.mark.parametrize('current',[
    {'status':'unavailable'}, {'status':'verified_current','remaining_usd':0},
])
def test_unverified_or_exhausted_current_budget_does_not_invalidate_old_block(current):
    assert budget_observation(old_observation(),current)['status']=='budget_exhausted'


def test_preparation_refreshes_monitor_and_fix_before_context_and_report(tmp_path,monkeypatch):
    real = make(tmp_path)
    ledger = AgentCostLedger(real.output,run_id=real.run_id,limit_usd=20)
    reservation = ledger.reserve(session_id='historical',role='init',requested_usd=2)
    ledger.settle(reservation['reservation_id'],actual_usd=None,launched=True)
    before = ledger.path.read_bytes()
    real.monitor_observation = old_observation()
    real.fix_observation = old_observation()
    view = real.state()
    assert view['current_model_budget']['remaining_usd']==18
    assert view['monitor_observation']['status']=='historical_budget_exhausted'
    assert view['fix_observation']['status']=='historical_budget_exhausted'
    assert ledger.path.read_bytes()==before
    monkeypatch.setattr(real,'choose',lambda:{'do':'stop','why':'current independent framework fault',
        'arguments':{'stop_scope':'framework'}})
    result = real._run_cycle(max_steps=1)
    assert result['status']=='infrastructure_blocked'
    assert result['steps'][-1]['stop_scope']=='framework'


@pytest.mark.parametrize('outcome', ['raised', 'repair proposal rejected'])
def test_legacy_failure_is_not_claimed_repaired_or_current_runtime_proof(tmp_path, outcome):
    ref='action_receipts/'+'a'*16+'.json'
    def receipt(identity, runtime=None):
        return write_action_receipt(tmp_path,ActionReceipt(action_id=identity,run_id='r',
            repository=str(tmp_path),action='build_the_environment',decision_id=identity,
            state_revision=1,started_at='before',finished_at='after',status='raised',reason='failed',
            resource_limits={'enforced':{'controller_runtime_id':runtime}}))[0]
    ref=receipt('a'*16)
    steps=[{'step':'build_the_environment','outcome':outcome,'receipt_ref':ref}]
    view=controller_context(tmp_path,steps)
    assert view['previous_setup_failure']['validity']=='unverified_in_current_runtime'
    assert view['previous_setup_failure']['repair_status'].startswith('not_proven')
    steps[0]['receipt_ref']=receipt('b'*16,controller_identity())
    assert controller_context(tmp_path,steps)['previous_setup_failure']['validity']=='observed_in_current_runtime'


def test_newer_success_does_not_reactivate_historical_setup_failure(tmp_path):
    rows=[{'step':'build_the_environment','outcome':'raised'},
          {'step':'build_the_environment','outcome':'done'}]
    assert 'previous_setup_failure' not in controller_context(tmp_path,rows)


def test_foreign_ledger_is_not_exposed_as_current_allowance(tmp_path):
    ledger=AgentCostLedger(tmp_path,run_id='other',limit_usd=20)
    ledger.reserve(session_id='s',role='init',requested_usd=2)
    before=ledger.path.read_bytes()
    assert current_model_budget(tmp_path,'mine')['status']=='unavailable'
    assert ledger.path.read_bytes()==before


def test_derivation_failure_is_versioned_and_newer_success_supersedes_it(tmp_path):
    receipt = ActionReceipt(action_id='c'*16, run_id='r', repository=str(tmp_path),
        action='derive_a_command',decision_id='c'*16,state_revision=1,
        started_at='before',finished_at='after',status='no runnable command',reason='failed',
        resource_limits={'enforced':{'controller_runtime_id':'old-runtime'}})
    ref, _ = write_action_receipt(tmp_path,receipt)
    rows = [{'step':'derive_a_command','arguments':{'stage':'train'},
             'outcome':'no runnable command','receipt_ref':ref}]
    failure = controller_context(tmp_path,rows)['previous_action_failures'][0]
    assert failure['stage'] == 'train'
    assert failure['validity'] == 'unverified_in_current_runtime'
    rows.append({'step':'derive_a_command','arguments':{'stage':'train'},'outcome':'done'})
    assert controller_context(tmp_path,rows)['previous_action_failures'] == []


def test_exhausted_harness_proposal_is_not_a_scheduler_option(tmp_path):
    real = make(tmp_path)
    real.client.fork_readonly = lambda **kw: None
    for index in range(3):
        atomic_json(real.output/f'harness_repairs/{index}/proposal.json',
                    {'id':str(index),'status':'candidate','created_at':str(index)})
    view = real.state()
    assert view['harness_repair']['proposal_count'] == 3
    assert view['harness_repair']['can_propose'] is False
    assert 'propose_harness_repair' not in view['available']
