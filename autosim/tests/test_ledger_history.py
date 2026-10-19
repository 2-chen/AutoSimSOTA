"""Authorized budget changes survive ONLY evidenced accounting transitions."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from autosim.research import budget_receipts, ledger_history
from autosim.research.agent_budget import AgentCostLedger
from autosim.research.budget import RunBudget
from autosim.research.budget_amendment import extend, authorizes_transition
from autosim.research.common import atomic_json, object_digest, read_json, now
from autosim.research.task_budget import TaskGPUBudget
from tests.test_environment_and_budget_alignment import priced_hold


def grant(tmp_path, monkeypatch, *, unknown=True, real_runtime=False):
    ledger, price, reservation = priced_hold(tmp_path, monkeypatch, unknown=unknown)
    monkeypatch.setattr('autosim.research.process_executor.inspect_process_identity',
                        lambda _: {'status': 'not_running'})
    workspace = tmp_path/'checkout';workspace.mkdir()
    RunBudget(tmp_path, wall_seconds=3600)
    TaskGPUBudget(tmp_path).initialize(workspace)
    if real_runtime:
        from autosim.research.agent_client import RoleAwareAgentClient
        RoleAwareAgentClient(workspace=workspace,output=tmp_path,run_id='r',
            turn_budget_usd=1,total_budget_usd=4,timeout=60)
    else:
        atomic_json(tmp_path/'agent/runtime_spec.json', {'run_id':'r',
            'total_budget_usd':4, 'turn_budget_usd':1, 'timeout_seconds':60})
    amendment = extend(tmp_path, model_usd=1, wall_seconds=0,
        reason='fixture explicit grant', preserve_stopped_holds=True)
    ledger = AgentCostLedger(tmp_path, run_id='r', limit_usd=5,
                              cost_basis='deepseek_official_estimate_v1')
    return ledger,price,reservation,amendment


def allowed(path):
    return authorizes_transition(path, old_limit=4, new_limit=5, run_id='r')


@pytest.mark.parametrize('unknown', [False, True])
@pytest.mark.parametrize('bounded', [False, True])
def test_grant_receipt_reconciliation_and_new_turn_preserve_original_history(tmp_path,monkeypatch,unknown,bounded):
    ledger,price,reservation,amendment=grant(tmp_path,monkeypatch,unknown=unknown)
    original=read_json(ledger.path)['entries']
    grant_bytes=(tmp_path/f'budget_amendments/{amendment["id"]}.json').read_bytes()
    if bounded:
        row=read_json(price);row.update(official_estimate_usd=None,
            gateway={'unknown':True,'accounting_ceiling_usd':.4,'bound_valid':True})
        atomic_json(price,row)
    ledger.reconcile_receipts(apply=True)
    assert read_json(ledger.path)['entries']!=original
    assert allowed(tmp_path)
    assert (tmp_path/f'budget_amendments/{amendment["id"]}.json').read_bytes()==grant_bytes
    before=ledger.path.read_bytes()
    assert allowed(tmp_path) and ledger.path.read_bytes()==before
    new=ledger.reserve(session_id='new',role='scheduler',requested_usd=1)
    ledger.settle(new['reservation_id'],actual_usd=.03,launched=True)
    assert allowed(tmp_path)
    if bounded:
        assert ledger.snapshot()['unknown_reserved_usd']==.4
        assert ledger.snapshot()['spent_usd']==.03
    else:
        assert ledger.snapshot()['spent_usd']==.1


@pytest.mark.parametrize('operation',['extend_settle','cancel','unknown_then_reconcile'])
def test_old_live_reservation_can_finish_after_grant_with_proved_accounting(tmp_path,monkeypatch,operation):
    ledger,_,reservation,_=grant(tmp_path,monkeypatch,unknown=False)
    identity=reservation['reservation_id']
    if operation=='extend_settle':
        ledger.extend(identity,3.5)
        assert allowed(tmp_path)
        ledger.settle(identity,actual_usd=.2,launched=True)
    elif operation=='cancel':
        ledger.settle(identity,actual_usd=None,launched=False)
    else:
        ledger.settle(identity,actual_usd=None,launched=True)
        assert allowed(tmp_path)
        ledger.reconcile_unknown(identity,actual_usd=.2,evidence='fixture explicit provider receipt')
    assert allowed(tmp_path)


@pytest.mark.parametrize('fault',['amount','role','deleted','reordered','missing_proof','proof_hash',
                                'proof_run','proof_before','forged_settled_history'])
def test_unproved_edits_and_corrupt_history_never_authorize_budget(tmp_path,monkeypatch,fault):
    ledger,price,reservation,_=grant(tmp_path,monkeypatch)
    ledger.reconcile_receipts(apply=True)
    assert allowed(tmp_path)
    raw=read_json(ledger.path)
    if fault in {'amount','role','deleted','reordered'}:
        if fault=='amount':raw['entries'][0]['actual_usd']=0
        elif fault=='role':raw['entries'][0]['role']='different-owner'
        elif fault=='deleted':raw['entries']=[]
        else:
            ledger.reserve(session_id='different',role='fix',requested_usd=1)
            raw=read_json(ledger.path);raw['entries'].reverse()
        atomic_json(ledger.path,raw)
    else:
        path=next((tmp_path/'agent/ledger_history').glob('*/*.json'))
        record=read_json(path)
        if fault=='missing_proof':path.unlink()
        else:
            if fault=='proof_hash':record['operation']='settle'
            elif fault=='proof_run':record['run_id']='other'
            elif fault=='proof_before':record['before_sha256']='a'*64
            else:record['changes'][0]['before']['status']='settled'
            if fault!='proof_hash':
                record['sha256']=object_digest({k:v for k,v in record.items() if k!='sha256'})
            atomic_json(path,record)
    assert not allowed(tmp_path)


def test_proof_write_failure_keeps_original_ledger_and_full_reservation(tmp_path,monkeypatch):
    ledger,price,_,_=grant(tmp_path,monkeypatch)
    before=ledger.path.read_bytes()
    monkeypatch.setattr(ledger_history,'immutable_json',lambda *a: (_ for _ in ()).throw(OSError('fixture disk full')))
    with pytest.raises(OSError):ledger.reconcile_receipts(apply=True)
    assert ledger.path.read_bytes()==before
    assert ledger.snapshot()['reserved_usd']==3
    assert allowed(tmp_path)


def test_crash_after_proof_before_ledger_commit_is_safe_and_retryable(tmp_path,monkeypatch):
    ledger,price,_,_=grant(tmp_path,monkeypatch)
    original=ledger.path.read_bytes()
    real_write=ledger_history.atomic_json
    monkeypatch.setattr(ledger_history,'atomic_json',
                        lambda *a: (_ for _ in ()).throw(OSError('fixture interrupted commit')))
    with pytest.raises(OSError):ledger.reconcile_receipts(apply=True)
    assert ledger.path.read_bytes()==original
    assert list((tmp_path/'agent/ledger_history').glob('*/*.json'))
    assert allowed(tmp_path)  # An unused write-ahead proof does not change accounts.
    monkeypatch.setattr(ledger_history,'atomic_json',real_write)
    ledger.reconcile_receipts(apply=True)
    assert allowed(tmp_path) and ledger.snapshot()['spent_usd']==.07


def legacy_change(tmp_path,monkeypatch):
    ledger,price,reservation,amendment=grant(tmp_path,monkeypatch)
    row=read_json(price);row.update(official_estimate_usd=None,
        gateway={'unknown':True,'accounting_ceiling_usd':.4,'bound_valid':True})
    atomic_json(price,row)
    # Emulate the old writer: update only the existing row, no new proof history.
    raw=read_json(ledger.path);entry=raw['entries'][0]
    entry.update(status='unknown',actual_usd=None,reserved_usd=.4,
        reconciled_at=now(),reconciliation_evidence=str(price.relative_to(tmp_path)),
        reconciliation_receipt_sha256=object_digest(row))
    atomic_json(ledger.path,raw)
    return ledger,price,reservation,amendment


def test_explicit_legacy_repair_proves_exact_old_prefix_without_changing_accounts(tmp_path,monkeypatch):
    ledger,price,reservation,amendment=legacy_change(tmp_path,monkeypatch)
    assert not allowed(tmp_path)
    accounts=[ledger.path,tmp_path/f'budget_amendments/{amendment["id"]}.json',tmp_path/'budget.json']
    original=[p.read_bytes() for p in accounts]
    repaired=ledger_history.recover_legacy_reconciliation(ledger,amendment['id'])
    assert repaired['status']=='exact_legacy_prefix_proved'
    assert [p.read_bytes() for p in accounts]==original
    assert ledger.snapshot()['unknown_reserved_usd']==.4
    assert allowed(tmp_path)
    assert ledger_history.recover_legacy_reconciliation(ledger,amendment['id'])['status']=='already_verified'
    # More grants/settlements still validate the complete original authorization chain.
    extend(tmp_path,model_usd=1,wall_seconds=0,reason='fixture second grant')
    ledger=AgentCostLedger(tmp_path,run_id='r',limit_usd=6,cost_basis=ledger.cost_basis)
    row=read_json(price);row.update(official_estimate_usd=.07,
        gateway={'unknown':False,'requests':2,'cost_usd':.07,'bound_valid':True})
    atomic_json(price,row)
    ledger.reconcile_receipts(apply=True)
    assert authorizes_transition(tmp_path,old_limit=4,new_limit=6,run_id='r')


def test_legacy_recovery_does_not_revert_older_already_reconciled_rows(tmp_path,monkeypatch):
    from autosim.research.deepseek_pricing import PRICE_CARD_ID
    ledger,price,first=priced_hold(tmp_path,monkeypatch,unknown=True)
    monkeypatch.setattr('autosim.research.process_executor.inspect_process_identity',
                        lambda _: {'status':'not_running'})
    row=read_json(price);row.update(official_estimate_usd=None,
        gateway={'unknown':True,'accounting_ceiling_usd':.4,'bound_valid':True})
    atomic_json(price,row)
    ledger.reconcile_receipts(apply=True)
    old=read_json(ledger.path)['entries'][0]
    second=ledger.reserve(session_id='second',role='init',requested_usd=1)
    ledger.settle(second['reservation_id'],actual_usd=None,launched=True)
    turn='d'*32
    atomic_json(tmp_path/'agent/sessions'/('e'*32+'.json'),{
        'run_id':'r','session_id':'second','role':'init','budget_output':str(tmp_path),
        'budget_reservation_id':second['reservation_id'],'process_attempt_id':turn,
        'cost_basis':ledger.cost_basis,'status':'interrupted'})
    atomic_json(tmp_path/'agent/processes'/f'{turn}.json',{'run_id':'r','attempt_id':turn,
        'status':'interrupted','process_identity':{'pid':999999}})
    price2=tmp_path/'agent/pricing'/f'{turn}.json'
    price_row={'schema_version':1,'turn_id':turn,'price_card':PRICE_CARD_ID,
        'cost_basis':ledger.cost_basis,'official_estimate_usd':None,
        'gateway':{'unknown':True,'accounting_ceiling_usd':.2,'bound_valid':True}}
    atomic_json(price2,price_row)
    RunBudget(tmp_path,wall_seconds=3600)
    TaskGPUBudget(tmp_path).initialize(tmp_path/'checkout')
    atomic_json(tmp_path/'agent/runtime_spec.json',{'run_id':'r','total_budget_usd':4,'turn_budget_usd':1})
    amendment=extend(tmp_path,model_usd=1,wall_seconds=0,reason='fixture grant after old reconciliation')
    ledger=AgentCostLedger(tmp_path,run_id='r',limit_usd=5,cost_basis=ledger.cost_basis)
    document=read_json(ledger.path)
    document['entries'][1].update(reserved_usd=.2,reconciled_at=now(),
        reconciliation_evidence=str(price2.relative_to(tmp_path)),
        reconciliation_receipt_sha256=object_digest(price_row))
    atomic_json(ledger.path,document)
    result=ledger_history.recover_legacy_reconciliation(ledger,amendment['id'])
    assert result['reconstructed_rows']==1 and allowed(tmp_path)
    assert read_json(ledger.path)['entries'][0]==old
    assert ledger.snapshot()['unknown_reserved_usd']==pytest.approx(.6)


@pytest.mark.parametrize('fault',['unrelated_edit','missing_receipt','wrong_receipt','unknown_without_bound'])
def test_legacy_repair_refuses_any_unproved_or_nonmatching_history(tmp_path,monkeypatch,fault):
    ledger,price,_,amendment=legacy_change(tmp_path,monkeypatch)
    if fault=='unrelated_edit':
        raw=read_json(ledger.path);raw['entries'][0]['role']='guessed';atomic_json(ledger.path,raw)
    elif fault=='missing_receipt':price.unlink()
    else:
        row=read_json(price)
        if fault=='wrong_receipt':row['turn_id']='d'*32
        else:row['gateway']['bound_valid']=False
        atomic_json(price,row)
    before=ledger.path.read_bytes()
    with pytest.raises(ValueError):ledger_history.recover_legacy_reconciliation(ledger,amendment['id'])
    assert ledger.path.read_bytes()==before
    assert not allowed(tmp_path)


def test_main_agent_model_turn_starts_after_grant_and_receipt_reconciliation(tmp_path,monkeypatch):
    """Full role client -> runtime -> authorization -> gateway -> process -> settlement."""
    import json
    from autosim.research import agent_runtime
    from autosim.research.agent_client import RoleAwareAgentClient
    from autosim.research.process_executor import ProcessAttempt
    monkeypatch.setenv('DEEPSEEK_API_KEY','fixture-secret')
    monkeypatch.setenv('DEEPSEEK_MODEL','deepseek-flash')
    monkeypatch.setattr('autosim.llm_client.load_credential_file',lambda **_: {'source':'fixture'})
    monkeypatch.setattr(agent_runtime.shutil,'which',lambda _: '/usr/bin/claude')
    ledger,price,reservation,amendment=grant(tmp_path,monkeypatch,real_runtime=True)
    ledger.reconcile_receipts(apply=True)
    workspace=tmp_path/'checkout'
    atomic_json(tmp_path/'agent_fixture.json',{'schema_version':1,'workspace':str(workspace)})
    # Preserve the actual old session cap; do not edit it to bypass admission.
    atomic_json(tmp_path/'agent/session.json',{'schema_version':1,'run_id':'r',
        'role':'scheduler','status':'interrupted','max_total_budget_usd':4})
    client=RoleAwareAgentClient(workspace=workspace,output=tmp_path,run_id='r',
        turn_budget_usd=1,total_budget_usd=5,timeout=60)
    snapshot={'unknown':False,'requests':1,'cost_usd':.03,'bound_valid':True}
    @contextmanager
    def gateway(**kwargs):
        kwargs['checkpoint'](snapshot)
        yield SimpleNamespace(base_url='http://fixture',local_token='fixture',snapshot=lambda:snapshot)
    started=[]
    def process(command,**kwargs):
        started.append(command)
        kwargs['on_stdout_line'](json.dumps({'type':'result','total_cost_usd':9,
            'duration_ms':1,'usage':{'input_tokens':10},'result':'fixture completed'}))
        return ProcessAttempt(launched=True,returncode=0,stdout='',stderr='',containment_mode='fixture')
    monkeypatch.setattr(agent_runtime,'deepseek_turn_gateway',gateway)
    monkeypatch.setattr(agent_runtime,'run_process_stream',process)
    answer,metadata=client.chat_with_metadata('fixture instructions','{}',auto_skills=False)
    assert started and answer=='fixture completed'
    assert ledger.snapshot()['spent_usd']==pytest.approx(.1)
    assert read_json(tmp_path/'agent/session.json')['max_total_budget_usd']==5
    assert allowed(tmp_path)
