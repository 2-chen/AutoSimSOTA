"""Exporter contracts use fixtures, not robot results."""
import pytest

from autosim.research import data_campaign, paper_evidence
from autosim.research.common import atomic_json, read_json
from tests.test_data_campaign import arm_fixture, score


@pytest.fixture(autouse=True)
def campaign_receipt_fixture(monkeypatch):
    # Exporter fixtures do not implement native receipts; the exporter verification
    # is patched independently in each test. Production registration now requires
    # its own receipt check, tested with genuine CPU measurements elsewhere.
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
                        lambda *a, **kw: {'status':'consistent'})


def test_empty_export_is_pending_not_zero(tmp_path):
    report=paper_evidence.build(tmp_path,tmp_path)
    assert not report['rows'] and not report['formal_paper_eligible']
    assert report['costs']['autonomy_status']=='not_audited'
    assert '待执行不是零分' in paper_evidence.markdown(report)
    assert report==paper_evidence.build(tmp_path,tmp_path)


def test_actual_data_and_receipts_required(tmp_path,monkeypatch):
    repo,root,raw,_,kwargs=arm_fixture(tmp_path)
    data_campaign.record_arm(root,repo,'derived',**kwargs)
    monkeypatch.setattr(paper_evidence,'verify_measurement',lambda *a: {'status':'inconsistent'})
    report=paper_evidence.build(root,repo)
    assert not report['rows'] and report['issues']
    monkeypatch.setattr(paper_evidence,'verify_measurement',lambda *a: {'status':'consistent'})
    report=paper_evidence.build(root,repo)
    assert len(report['rows'])==1 and report['rows'][0]['formal_paper_eligible'] is False
    assert report['seed_summaries'][0]['sample_std'] is None
    raw.write_text('[9]')
    report=paper_evidence.build(root,repo)
    assert not report['rows'] and report['issues']


def measurement(keys=None,column=None,values=None):
    return {'metric':{'initial_state_hash_column':column},
            'metric_reading':{'episode_keys':keys,'episode_values':values or [0,1]}}


def test_episode_indexes_and_shared_seed_are_not_pairing():
    row=measurement(['seed0\x1f0','seed0\x1f1'])
    assert paper_evidence._paired_values(row,row) is None
    row['metric']['initial_state_hash_column']='state_hash'
    assert paper_evidence._paired_values(row,row) is None


def test_initial_state_hash_pairing_aligns_values():
    keys=['seed0\x1f'+'a'*64,'seed0\x1f'+'b'*64]
    left=measurement(keys,'state_hash')
    right=measurement(keys[::-1],'state_hash',[1,0])
    assert paper_evidence._paired_values(left,right)==([0,1],[0,1])
    right['metric_reading']['episode_keys']=keys[:1]*2
    assert paper_evidence._paired_values(left,right) is None


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),[],['fake']])
def test_bad_episode_values_are_not_statistics(value):
    assert paper_evidence._values({'metric_reading':{'episode_values':[value]}}) is None


def test_unknown_usage_and_active_gpu_are_not_complete_cost(tmp_path):
    atomic_json(tmp_path/'agent/cost_ledger.json',{'entries':[
        {'status':'settled','actual_usd':.4},{'status':'unknown'}, {'status':'reserved'}]})
    atomic_json(tmp_path/'task_gpu_budget.json',{'charged_seconds':3600,
        'leases':{'one':{'status':'active'}}})
    atomic_json(tmp_path/'budget.json',{'elapsed_wall_seconds':7200})
    (tmp_path/'OPERATOR_INTERVENTIONS.md').write_text('framework refresh')
    costs=paper_evidence.build(tmp_path,tmp_path)['costs']
    assert costs['model_settled_usd']==.4 and costs['model_cost_complete'] is False
    assert costs['gpu_lease_hours']==1 and costs['gpu_cost_complete'] is False
    assert costs['elapsed_wall_hours']==2
    assert costs['autonomy_status']=='development_operator_interventions'


def test_missing_settled_amount_is_not_zero_cost(tmp_path):
    atomic_json(tmp_path/'agent/cost_ledger.json',{'entries':[{'status':'settled'}]})
    report=paper_evidence.build(tmp_path,tmp_path)
    assert report['costs']['model_settled_usd'] is None
    assert report['costs']['model_cost_complete'] is False


def test_export_cli_immutable_snapshot(tmp_path):
    from tools.export_data_research import main
    run=tmp_path/'run';run.mkdir()
    destination=tmp_path/'export'
    assert main([str(run),str(run),str(destination)])==0
    assert main([str(run),str(run),str(destination)])==0
    snapshots=list(destination.iterdir())
    assert len(snapshots)==1
    assert read_json(snapshots[0]/'analysis.json')['formal_paper_eligible'] is False


@pytest.mark.parametrize('mismatch',[False,True])
def test_arm_comparison_keeps_episode_and_training_uncertainty_separate(tmp_path,monkeypatch,mismatch):
    repo,root,_,_,kwargs=arm_fixture(tmp_path)
    for label,values in [('round_1',[0,1]),('round_2',[0,0])]:
        row=score(root,label,sum(values)/len(values),['new'])
        row['metric']={'name':'success','direction':'maximize','unit':'fraction','aggregation':'mean'}
        if mismatch and label=='round_2':row['metric']['unit']='percent'
        row['metric_reading']={'episode_values':values,'episode_keys':['seed0:0','seed0:1'],'samples':2}
        atomic_json(root/f'research/derived/measurements/{label}.json',row)
    data_campaign.record_arm(root,repo,'derived',**kwargs)
    data_campaign.record_arm(root,repo,'derived',**{**kwargs,'arm':'random','measurement_label':'round_2'})
    # Fixture receipts are replaced explicitly; production exporter performs native verification.
    monkeypatch.setattr(paper_evidence,'verify_measurement',lambda *a: {'status':'consistent'})
    report=paper_evidence.build(root,repo)
    assert len(report['rows'])==2
    assert all(row['training_seed_count']==1 and row['sample_std'] is None for row in report['seed_summaries'])
    if mismatch:
        assert not report['comparisons'] and report['issues']
    else:
        result=report['comparisons'][0]
        assert result['paired_by_initial_state_hash'] is False
        assert result['random_to_targeted']['difference']==.5
        assert report['formal_paper_eligible'] is False
