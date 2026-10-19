"""Replay stale budget and framework evidence after an operator resumes the same run."""
import copy

import pytest

from autosim.research.agent_budget import AgentCostLedger
from autosim.research.common import atomic_json, read_json
from autosim.research.action_receipt import ActionReceipt, write_action_receipt
from autosim.research.runtime_freshness import (
    budget_observation, current_model_budget, controller_context, controller_identity)
from tests.test_main_agent import make


@pytest.mark.parametrize('filename', [
    'candidate_execution_review.py', 'monitor_findings.py', 'native_runtime_summary.py', 'telemetry.py',
    'agent_client.py', 'deepseek_gateway.py',
    'measurement_adoption.py', 'frozen_policy_trial.py', 'receipt_verifier.py',
    'run_record.py', 'selection.py', 'experiment_bundle.py',
    'baseline_reference.py', 'confirmation_reservation.py', 'framework_stop_review.py',
    'operation_contracts.py', 'native_source_citations.py',
])
def test_new_admission_and_evidence_modules_change_controller_identity(monkeypatch, filename):
    from autosim.research import runtime_freshness

    original_digest = runtime_freshness.digest
    controller_identity.cache_clear()
    try:
        before = controller_identity()
        monkeypatch.setattr(runtime_freshness, 'digest',
            lambda path: 'changed-module-content' if path.name == filename else original_digest(path))
        controller_identity.cache_clear()
        assert controller_identity() != before
    finally:
        # Other tests must not inherit the identity of a simulated source edit.
        controller_identity.cache_clear()


def old_observation():
    return {'status':'budget_exhausted','failure_category':'run_model_budget',
        'recorded_at':'2026-10-02T04:23:30Z',
        'run_budget':{'limit_usd':3,'remaining_usd':0,'entry_count':2}}


def test_current_admission_rules_are_source_bound_and_stale_routes_not_permanent(tmp_path):
    from autosim.research.common import digest
    from autosim.research import runtime_freshness
    old={'step':'submit_native_job','outcome':'rejected','because':'protocol changed'}
    original=copy.deepcopy(old)
    row=controller_context(tmp_path,[old])
    assert row['historical_admission_revalidation']['operations']==['submit_native_job']
    assert row['previous_action_failures'][0]['validity']=='unverified_in_current_runtime'
    assert old==original
    rules=row['current_admission_contracts']
    assert 'ONLY explicitly requested' in rules['candidate_settings']['rule']
    for contract in rules.values():
        path=runtime_freshness.__file__
        from pathlib import Path
        assert contract['source_sha256']==digest(Path(path).parent/Path(contract['source_ref']).name)
    assert 'verified repair' in row['historical_admission_revalidation']['authority']
    assert not (tmp_path/'native_jobs').exists()


def test_current_contract_distinguishes_available_transport_from_stale_builder(tmp_path):
    source = 'def stage_argv_train(i): return [i["python"], i["dataset"]]'
    facts = controller_context(tmp_path, [], train_source=source)
    rule = facts['current_admission_contracts']['training_selection_transport']['rule']
    assert 'selected_training_data' in rule and 'NOT expected hashes or counters' in rule
    assert 'AUTOSIM_TRAIN_SELECTION_JSON' in rule and 'new campaign ID' in rule
    assert 'AUDITED CODE CANDIDATES' in rule and 'not research_task(edit)' in rule
    assert facts['current_admission_contracts']['candidate_settings']['current_train_builder']['text'] == source
    assert not list(tmp_path.iterdir())


def test_main_uncertainty_contract_is_explicit_not_an_automatic_review_override(tmp_path):
    rule=controller_context(tmp_path,[])['current_admission_contracts'][
        'candidate_uncertainty_validation']['rule']
    assert 'execution_uncertainty' in rule and 'stage_window_seconds' in rule
    assert 'NOT approval' in rule and 'unavailable/malformed' in rule
    assert 'Existing actual-data' in rule and 'unchanged hard totals' in rule
    assert not list(tmp_path.iterdir())


def test_consumer_format_recovery_is_not_research_approval_or_an_unbounded_retry(tmp_path):
    rule=controller_context(tmp_path,[])['current_admission_contracts'][
        'consumer_review_response']['rule']
    assert 'consumer_review_unavailable' in rule
    assert 'One format-only' in rule and 'valid negative verdict' in rule
    assert 'Legacy cached format failures alone' in rule and 'format-v2' in rule
    assert 'format-v3' in rule and '8192' in rule
    assert 'sealed older paid responses' in rule and 'strict booleans/citations' in rule
    assert 'no training' in rule and 'remain cached unavailable' in rule
    assert 'ALREADY' in rule and '64 KiB' in rule and 'raw JSON datasets' in rule
    assert 'without deleting the old' in rule and 'legacy approvals stay valid' in rule
    assert not list(tmp_path.iterdir())


def test_current_runtime_refusal_is_not_offered_as_stale_retry(tmp_path):
    ref=write_action_receipt(tmp_path,ActionReceipt(action_id='b'*16,run_id='r',
        repository=str(tmp_path),action='submit_native_job',decision_id='b'*16,
        state_revision=1,started_at='before',finished_at='after',status='rejected',reason='refused',
        resource_limits={'enforced':{'controller_runtime_id':controller_identity()}}))[0]
    row=controller_context(tmp_path,[{'step':'submit_native_job','outcome':'rejected','receipt_ref':ref}])
    assert row['historical_admission_revalidation']['operations']==[]
    assert row['previous_action_failures'][0]['validity']=='observed_in_current_runtime'


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
    result = real._run_cycle(max_steps=3)
    # A mere Main assertion with no current sealed failure cannot authorize a
    # permanent framework boundary. One correction then a safe unresolved pause.
    assert result['status']=='paused'
    assert result['steps'][-1]['stop_scope']=='pause'
    assert list((real.output/'framework_stop_reviews').glob('*.json'))


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
                    {'id':str(index),'status':'candidate','created_at':str(index),
                     'runtime_identity':controller_identity()})
    view = real.state()
    assert view['harness_repair']['proposal_count'] == 3
    assert view['harness_repair']['can_propose'] is False
    assert 'propose_harness_repair' not in view['available']


def test_historical_harness_proposal_cap_does_not_lock_new_controller(tmp_path):
    real=make(tmp_path);real.client.fork_readonly=lambda **kw:None
    for index in range(3):
        atomic_json(real.output/f'harness_repairs/{index}/proposal.json',
            {'id':str(index),'status':'candidate','created_at':str(index),'runtime_identity':'old'})
    view=real.state()
    assert view['harness_repair']['proposal_count']==3
    assert view['harness_repair']['current_version_proposal_count']==0
    assert 'propose_harness_repair' in view['available']
