"""AutoSOTA-aligned recovery: cheap consumers, reusable grounding, safe budget lifecycle."""
import json
from pathlib import Path

import pytest

from autosim.research import provision, recorder, budget_receipts
from autosim.research.agent_budget import AgentCostLedger, AgentCostBudgetError
from autosim.research.common import atomic_json, read_json
from autosim.research.deepseek_pricing import PRICE_CARD_ID
from tests.test_environment_overlay import fixture, make, provision_fixture


class Planner:
    def __init__(self, output, answers):
        self.output, self.answers, self.payloads = output, iter(answers), []

    def chat_with_metadata(self, system, payload, **kwargs):
        self.payloads.append(payload)
        return json.dumps(next(self.answers)), {}


def basic_plan(**extra):
    return {'python':'3.10','commands':['{pip} install useful'],
            'probes':['{python} -c "import useful"'], **extra}


@pytest.mark.parametrize('bad', [
    {'commands':[{'command':'{pip} install useful'}]},
    {'probes':[{'command':'{python} -c "import useful"'}]},
    {'source_binding_ids':[{'id':'not-a-string'}]},
    {'resource_requests':[{'command':'{pip} install useful','resource':{},'why':'test'}]},
    {'assets':{'dataset':'wrong shape'}},
    {'base_environment_mode':{'mode':'overlay'}},
])
def test_malformed_plan_is_corrected_not_an_unhashable_framework_crash(tmp_path,bad):
    client = Planner(tmp_path,[basic_plan(**bad),basic_plan()])
    answer = provision.plan(client,tmp_path,manifests={},assets={},attempts=2)
    assert answer['commands'] == ['{pip} install useful']
    assert 'ValueError' in client.payloads[1]
    assert 'Correct this proposal' in client.payloads[1]
    assert answer['attempts'][0]['status'] == 'rejected'


def test_exhausted_shape_correction_has_stable_planner_failure_evidence(tmp_path):
    client = Planner(tmp_path,[basic_plan(commands=[{}])])
    with pytest.raises(ValueError) as held:
        provision.plan(client,tmp_path,manifests={},assets={},attempts=1)
    failure = held.value.planning_failure
    assert failure['failure_domain'] == 'framework_plan'
    assert failure['repair_owner'] == 'environment_planner'
    assert failure['native_operation_status'] == 'not_started'
    assert (tmp_path/failure['evidence_ref']).is_file()
    assert 'commands[0]' in failure['attempts'][0]['error']


def test_existing_environment_can_probe_without_install_or_recreation(fixture,monkeypatch):
    base,site,output,repo,version = fixture
    result = make(fixture)
    _, planned = provision_fixture(fixture,monkeypatch)
    planned.update(python='existing',base_environment_mode='existing',commands=[],
                   probes=['{python} -c "import dependency,task; assert dependency.VALUE==73"'])
    planned.pop('base_environment_id')
    answer = provision.build(repo,client=None,prefix=output/'env',output=output,
        python=result['interpreter'],assets={},manifests={},max_rounds=1)
    assert answer['verdict']['passed']
    assert answer['interpreter'] == result['interpreter']
    assert [r for r in answer['record'] if r.get('kind')=='probe']


def test_existing_plan_validates_the_supplied_run_owned_interpreter(fixture):
    result = make(fixture)
    base,site,output,repo,version = fixture
    plan = basic_plan(python='existing',base_environment_mode='existing',commands=[],
                      probes=['{python} -c "import dependency"'])
    client = Planner(output,[plan])
    assert provision.plan(client,repo,manifests={},assets={},python=result['interpreter'])['commands']==[]
    client = Planner(output,[plan])
    with pytest.raises(ValueError):
        provision.plan(client,repo,manifests={},assets={},python=str(base/'bin/python'),attempts=1)


def test_source_handoff_is_reused_until_source_changes(tmp_path):
    from tests.test_main_agent import make as preparation
    obj = preparation(tmp_path)
    (obj.repo/'train.py').write_text('torch.save(policy.state_dict(),checkpoint)\n')
    (obj.repo/'eval.py').write_text('policy.load_state_dict(torch.load(checkpoint))\n')
    rows = {'train':{'entrypoint':'train.py'},'evaluate':{'entrypoint':'eval.py'}}
    obj.client = Planner(obj.output,[{'compatible':True,'why':'matching state_dict save/load'}]*2)
    assert obj._review_policy_handoff(rows,['train','evaluate'],'evaluate','native policy')['compatible']
    assert obj._review_policy_handoff(rows,['train','evaluate'],'evaluate','native policy')['reused']
    assert len(obj.client.payloads)==1
    (obj.repo/'train.py').write_text('torch.save(policy.state_dict(),checkpoint)\n# changed source\n')
    assert obj._review_policy_handoff(rows,['train','evaluate'],'evaluate','native policy')['compatible']
    assert len(obj.client.payloads)==2


def priced_hold(tmp_path,monkeypatch,*,worker=False,unknown=False):
    ledger = AgentCostLedger(tmp_path,run_id='r',limit_usd=4,cost_basis='deepseek_official_estimate_v1')
    reservation = ledger.reserve(session_id='s',role='init',requested_usd=3)
    if unknown:
        ledger.settle(reservation['reservation_id'],actual_usd=None,launched=True)
    owner = tmp_path/'agent_tasks'/('b'*32) if worker else tmp_path
    turn = 'a'*32
    atomic_json(owner/'agent/sessions'/('c'*32+'.json'),{
        'run_id':'r','session_id':'s','role':'init','budget_output':str(tmp_path),
        'budget_reservation_id':reservation['reservation_id'],'process_attempt_id':turn,
        'cost_basis':'deepseek_official_estimate_v1','status':'interrupted'})
    atomic_json(owner/'agent/processes'/f'{turn}.json',{'run_id':'r','attempt_id':turn,
        'status':'interrupted','process_identity':{'pid':999999}})
    price = owner/'agent/pricing'/f'{turn}.json'
    atomic_json(price,{'schema_version':1,'turn_id':turn,'price_card':PRICE_CARD_ID,
        'cost_basis':'deepseek_official_estimate_v1','official_estimate_usd':.07,
        'gateway':{'unknown':False,'requests':2,'cost_usd':.07,'bound_valid':True}})
    monkeypatch.setattr(budget_receipts,'inspect_process_identity',lambda *a:{'status':'not_running'})
    return ledger,price,reservation


@pytest.mark.parametrize('snapshot,expected',[
    ({'unknown':False,'requests':0,'cost_usd':0,'bound_valid':True,
      'accounting_ceiling_usd':0,'held_usd':0},0),
    ({'unknown':False,'requests':0,'cost_usd':0},None),
    ({'unknown':True,'requests':0,'cost_usd':0,'bound_valid':True,
      'accounting_ceiling_usd':0,'held_usd':0},None),
    ({'unknown':False,'requests':1,'cost_usd':.04,'bound_valid':True},.04),
    ({'unknown':False,'requests':1,'cost_usd':.04,'bound_valid':False},None),
])
def test_zero_usage_requires_trusted_zero_forwarding_receipt(snapshot,expected):
    assert budget_receipts.known_gateway_cost(snapshot)==expected


def test_zero_request_receipt_releases_hold_without_guessing_from_process_exit(tmp_path,monkeypatch):
    ledger,price,_ = priced_hold(tmp_path,monkeypatch)
    receipt = read_json(price)
    receipt.update(official_estimate_usd=None,gateway={'unknown':False,'requests':0,
        'cost_usd':0,'held_usd':0,'accounting_ceiling_usd':0,'bound_valid':True})
    atomic_json(price,receipt)
    assert ledger.reconcile_receipts(apply=True)['results'][0]['actual_usd']==0
    assert ledger.snapshot()['reserved_usd']==0


def test_interruption_seals_provider_receipt_and_settles_only_observed_usage(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from autosim.research import agent_runtime
    output = tmp_path/'run'
    workspace = output/'checkout'
    workspace.mkdir(parents=True)
    atomic_json(output/'agent_fixture.json',{'schema_version':1,'workspace':str(workspace)})
    monkeypatch.setattr('autosim.llm_client.load_credential_file',lambda **_: {'source':'fixture'})
    monkeypatch.setenv('DEEPSEEK_API_KEY','fixture-secret')
    monkeypatch.setenv('DEEPSEEK_MODEL','deepseek-flash')
    snapshot = {'unknown':False,'requests':1,'cost_usd':.03,'bound_valid':True}
    @contextmanager
    def gate(**kwargs):
        yield SimpleNamespace(base_url='http://fixture',local_token='fixture',snapshot=lambda:snapshot)
    def interrupted(*args,**kwargs):
        raise KeyboardInterrupt('fixture interrupt')
    monkeypatch.setattr(agent_runtime,'deepseek_turn_gateway',gate)
    monkeypatch.setattr(agent_runtime,'run_process_stream',interrupted)
    with pytest.raises(KeyboardInterrupt):
        agent_runtime.run_coding_agent(workspace=workspace,output=output,prompt='fixture',
            run_id='r',max_budget_usd=.1,max_total_budget_usd=1,auto_skills=False,cli='/usr/bin/claude')
    ledger = AgentCostLedger(output,run_id='r',limit_usd=1,cost_basis='deepseek_official_estimate_v1')
    assert ledger.snapshot()['spent_usd']==pytest.approx(.03)
    assert ledger.snapshot()['reserved_usd']==0
    receipts = list((output/'agent/pricing').glob('*.json'))
    assert len(receipts)==1 and read_json(receipts[0])['interrupted']


@pytest.mark.parametrize('worker,unknown',[(False,False),(True,False),(False,True)])
def test_receipt_reconciliation_is_bounded_idempotent_and_not_cli_repricing(tmp_path,monkeypatch,worker,unknown):
    ledger,price,reservation = priced_hold(tmp_path,monkeypatch,worker=worker,unknown=unknown)
    before = ledger.path.read_bytes()
    assert ledger.reconcile_receipts()['results'][0]['status']=='settled_from_receipt'
    assert ledger.path.read_bytes()==before
    ledger.reconcile_receipts(apply=True)
    assert ledger.snapshot()['spent_usd']==pytest.approx(.07)
    assert ledger.snapshot()['reserved_usd']==0
    after = ledger.path.read_bytes()
    ledger.reconcile_receipts(apply=True)
    assert ledger.path.read_bytes()==after
    row = read_json(ledger.path)['entries'][0]
    assert row['reconciliation_evidence']==str(price.relative_to(tmp_path))


def test_missing_usage_or_live_process_never_clears_hold(tmp_path,monkeypatch):
    ledger,price,_ = priced_hold(tmp_path,monkeypatch)
    monkeypatch.setattr(budget_receipts,'inspect_process_identity',lambda *a:{'status':'matching_running'})
    assert ledger.reconcile_receipts(apply=True)['results'][0]['status']=='retained'
    assert ledger.snapshot()['reserved_usd']==3
    monkeypatch.setattr(budget_receipts,'inspect_process_identity',lambda *a:{'status':'not_running'})
    price.unlink()
    assert ledger.reconcile_receipts(apply=True)['results'][0]['status']=='retained'
    assert ledger.snapshot()['reserved_usd']==3


@pytest.mark.parametrize('change',[{'turn_id':'d'*32},{'price_card':'wrong'}, {'official_estimate_usd':0.0},
                                 {'cost_basis':'cli_reported'}])
def test_mismatched_receipt_cannot_free_reservation(tmp_path,monkeypatch,change):
    ledger,price,_ = priced_hold(tmp_path,monkeypatch)
    atomic_json(price,{**read_json(price),**change})
    assert ledger.reconcile_receipts(apply=True)['results'][0]['status']=='retained'
    assert ledger.snapshot()['reserved_usd']==3


def test_verified_request_bound_reduces_unknown_hold_without_claiming_actual_cost(tmp_path,monkeypatch):
    ledger,price,_ = priced_hold(tmp_path,monkeypatch)
    row = read_json(price)
    row.update(official_estimate_usd=None,gateway={'unknown':True,'accounting_ceiling_usd':.4,'bound_valid':True})
    atomic_json(price,row)
    ledger.reconcile_receipts(apply=True)
    usage = ledger.snapshot()
    assert usage['spent_usd']==0
    assert usage['unknown_reserved_usd']==pytest.approx(.4)
    assert usage['unknown_entries']==1


def test_historical_orphan_hold_needs_evidence_even_if_no_process_exists(tmp_path):
    ledger = AgentCostLedger(tmp_path,run_id='r',limit_usd=4)
    ledger.reserve(session_id='old',role='fix',requested_usd=3)
    assert ledger.reconcile_receipts(apply=True)['results'][0]['status']=='retained'
    assert ledger.snapshot()['reserved_usd']==3


def test_recorder_cannot_reserve_or_extend_execution_recovery_margin(tmp_path):
    atomic_json(tmp_path/'scheduler_policy.json',{'recovery_model_reserve_usd':.6})
    ledger = AgentCostLedger(tmp_path,run_id='r',limit_usd=1)
    narration = ledger.reserve(session_id='n',role='recorder',requested_usd=1,allow_partial=True)
    assert narration['allowed_usd']==pytest.approx(.4)
    with pytest.raises(AgentCostBudgetError):
        ledger.extend(narration['reservation_id'],.5)
    assert ledger.reserve(session_id='f',role='fix',requested_usd=.6)['allowed_usd']==pytest.approx(.6)


def test_low_budget_recorder_remains_fact_only_and_reads_current_ledger(tmp_path):
    from autosim.research import run_record
    ledger = AgentCostLedger(tmp_path,run_id='r',limit_usd=10)
    ledger.reserve(session_id='setup',role='init',requested_usd=9.8)
    class Client:
        supports_recorder = True
        def chat_with_metadata(self,*a,**kw): pytest.fail('report must not buy model calls')
        def fork_readonly(self,**kw): pytest.fail('report must not create paid worker')
    view = run_record.build_report_view(tmp_path,'r',status='running')
    text = recorder.refresh(tmp_path,view,context={'actions':[],'plan':{}},client=Client())
    assert '事实模式' in text
    assert '$0.20000' in text
    assert read_json(tmp_path/'report/recorder_deferred.json')['mode']=='fact_only'
    assert read_json(ledger.path)['entries'][0]['status']=='reserved'


def test_budget_report_after_settlement_does_not_keep_old_zero_balance(tmp_path):
    from autosim.research import run_record
    ledger = AgentCostLedger(tmp_path,run_id='r',limit_usd=10)
    turn = ledger.reserve(session_id='s',role='init',requested_usd=10)
    ledger.settle(turn['reservation_id'],actual_usd=.2,launched=True)
    view = run_record.build_report_view(tmp_path,'r',status='budget_waiting')
    snapshot = recorder.make_snapshot(tmp_path,view,{'actions':[],'plan':{}})
    text = recorder.render(tmp_path,snapshot,{})
    assert '$9.80000' in text
    assert '已有可用余额' in text
    assert '不代表已停止的实验自动恢复' in text
