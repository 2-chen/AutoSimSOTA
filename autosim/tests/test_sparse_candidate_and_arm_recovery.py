"""Regressions for real sparse-candidate and boundary-review admission failures."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autosim.research.adapter_protocol import Axis, OptimizationSpace
from autosim.research.candidate_settings import proposed, merge_training
from autosim.research.common import read_json
from autosim.research.ideas import Idea, CLEARED, execution_compatibility
from tests.test_derived_research import research
from tests.test_prepare import preparation


@pytest.mark.parametrize('change', [{'steps':80000}, {'training':{'steps':80000}}])
def test_sparse_candidate_does_not_apply_validation_defaults(tmp_path, change):
    real=research(tmp_path)
    real.space=OptimizationSpace(training=(
        Axis('steps','integer','updates',low=1,high=100000,default=10000),
        Axis('seed','integer','native default is not the measured seed',low=0,high=99999999,default=1000),
        Axis('batch_size','integer','batch',low=1,high=128,default=64)))
    base={'task':'toy','episodes':50,'seed':20261013,'steps':1000,'batch_size':8}
    original=deepcopy(base)
    assert not real._comparison_protocol_violation(base,target='evaluate',freeze=True)
    idea=Idea(label='longer',granularity='param',change=change)
    selected=proposed(real,base,idea)
    assert selected=={**base,'steps':80000}
    assert not real._comparison_protocol_violation(selected,target='evaluate')
    assert base==original
    # A direct evaluation-seed mutation remains forbidden. Training seeds need
    # their own verified native input, not an excuse to mutate evaluation.
    idea.change={'steps':80000,'seed':1000}
    with pytest.raises(ValueError,match='trainer seed input'):
        proposed(real,base,idea)
    assert 'seed' in real._comparison_protocol_violation({**base,'seed':1000},target='evaluate')


def test_grouped_delta_preserves_other_members_and_code_changes_no_axes(tmp_path):
    real=research(tmp_path)
    real.space=OptimizationSpace(training=(
        Axis('lr','number','learning rate',low=0,high=1,default=.1,group='optimizer'),
        Axis('decay','number','decay',low=0,high=1,default=.2,group='optimizer')))
    idea=Idea(label='lr',granularity='param',change={'training':{'optimizer':{'lr':.01}}})
    base={'optimizer.lr':.05,'optimizer.decay':.9,'seed':12}
    proposal={'training':{'optimizer':{'lr':.01,'decay':.2}}}
    assert merge_training(real,base,idea,proposal)=={**base,'optimizer.lr':.01}
    idea.granularity='code'
    assert merge_training(real,base,idea,proposal)==base


def test_source_backed_steps_offering_agrees_with_executor_input():
    idea=Idea(label='steps',granularity='param',change={'steps':80000})
    assert not execution_compatibility(idea,stage_parameters={'--batch-size':{}},train_source='')[0]
    assert execution_compatibility(idea,stage_parameters={'--batch-size':{}},
        train_source='def train(i): return [str(i["steps"])]')[0]
    idea.change={'invented':1}
    assert not execution_compatibility(idea,stage_parameters={'--batch-size':{}},
        train_source='def train(i): return [str(i["steps"])]')[0]


@pytest.mark.parametrize('route',['synchronous','detached'])
def test_real_candidate_routes_preserve_the_measured_seed(tmp_path,route):
    from autosim.research.candidate_jobs import allocation
    real=research(tmp_path)
    real.space=OptimizationSpace(training=(
        Axis('steps','integer','updates',low=1,high=100000,default=10000),
        Axis('seed','integer','seed',low=0,high=99999999,default=1000)))
    base={'task':'toy','episodes':5,'seed':20261013,'steps':1000}
    real.run(rounds=2,settings=base,yield_after_action=True)
    idea=Idea(label='longer',granularity='param',change={'steps':2000},
        mechanism='more learning',why='same evaluator',status=CLEARED)
    real.library.add(idea)
    if route=='detached':
        assert allocation(real,idea.label)['settings']=={**base,'steps':2000}
    else:
        report=real.run(rounds=2,settings=base,yield_after_action=True,
            selected_idea_label=idea.label,main_controller_owns_selection=True)
        assert report['rounds'][-1]['status']=='measured'
        assert report['rounds'][-1]['varied']=={'steps':2000}
        measured=read_json(real.run_root/'measurements'/f'{report["rounds"][-1]["label"]}.json')
        assert measured['settings']=={**base,'steps':2000}
    assert read_json(real.run_root/'comparison_protocol.json')['settings']['seed']==base['seed']


def test_inspecting_an_attempt_id_does_not_claim_a_missing_job_is_corrupt(tmp_path,monkeypatch):
    from autosim.research.evidence_store import capture_attempt_evidence
    obj=preparation(tmp_path);obj.output.mkdir(exist_ok=True)
    ident='a'*32;log=obj.output/'train.log';log.write_text('loss: 1.2\n')
    capture_attempt_evidence(obj.output,attempt_id=ident,log=log,receipt_ref='original.json',
        status='completed',returncode=0,termination_reason='normal_exit')
    result=obj._step_inspect_native_job(job_id=ident,log_query='loss:')
    assert result['id_kind']=='attempt_evidence' and result['evidence_id']==ident
    assert 'job_id' not in result and 'read_evidence' in result['next_action']


def boundary_preparation(tmp_path,monkeypatch,answers):
    obj=preparation(tmp_path,answers)
    obj.output.mkdir(exist_ok=True)
    (obj.repo/'expert.py').write_text('EXPERT_MODE = "reset_only"\n')
    monkeypatch.setattr(obj,'_available_operations',lambda:[])
    monkeypatch.setattr(obj,'_task_gpu_budget_view',lambda:{})
    monkeypatch.setattr('autosim.research.runtime_freshness.current_model_budget',lambda *a:{})
    boundary={'status':'unsupported','reason':'test-only review transport, not a capability claim',
        'evidence_refs':[], 'source_refs':[{'ref':'expert.py','quote':'EXPERT_MODE = "reset_only"'}]}
    return obj,boundary


def test_conflicting_boundary_arguments_fail_before_review_with_stable_evidence(tmp_path,monkeypatch):
    obj,boundary=boundary_preparation(tmp_path,monkeypatch,[])
    result=obj._step_record_data_campaign_arm(campaign_id='campaign',arm='agent',seed=1,
        boundary=boundary,measurement_label='baseline')
    assert result['failure_category']=='data_arm_argument_conflict'
    assert result['repair_owner']=='main_agent' and result['native_operation_status']=='not_started'
    assert (obj.output/result['evidence_ref']).is_file()
    assert obj.client.seen==[]


def test_source_citation_refusal_is_not_an_unsupported_data_boundary(tmp_path,monkeypatch):
    obj,boundary=boundary_preparation(tmp_path,monkeypatch,[])
    boundary['source_refs']=['expert.py']
    result=obj._step_record_data_campaign_arm(campaign_id='campaign',arm='agent',seed=1,boundary=boundary)
    assert result['failure_category']=='source_citation_invalid'
    assert result['repair_owner']=='main_agent' and result['retryable']
    assert result['evidence_id'] and obj.client.seen==[]
    assert not (obj.output/'data_campaigns').exists()


@pytest.mark.parametrize('fault',['none','invalid_repair','rejected_science'])
def test_boundary_citation_format_repair_is_bounded_and_never_changes_verdict(tmp_path,monkeypatch,fault):
    initial={'boundary_evidence_supported':True,'no_safe_next_data_operation':fault!='rejected_science',
        'why':'independent test assessment',
        'citations':[{'source_id':'review-source-1','quote':'EXPERT_MODE'}]}
    repair={'citations':[{'source_id':'invented' if fault=='invalid_repair' else 'review-source-1'}],
        'why':'must not replace scientific decision','no_safe_next_data_operation':True}
    obj,boundary=boundary_preparation(tmp_path,monkeypatch,[initial,repair])
    captured=[]
    def record(*a,**kw):
        captured.append(kw)
        return {'metric_value':None}
    monkeypatch.setattr('autosim.research.data_campaign.record_boundary',record)
    result=obj._step_record_data_campaign_arm(campaign_id='campaign',arm='agent',seed=1,boundary=boundary)
    if fault=='none':
        assert result['outcome']=='recorded' and len(obj.client.seen)==2
        review=captured[0]['review']
        assert review['why']==initial['why']
        assert review['citations'][0]['quote']=='EXPERT_MODE = "reset_only"'
        assert review['citation_format_repair']['initial_review']==initial
        packet=json.loads(obj.client.seen[1])
        assert packet['review_sources'][0]['source_id']=='review-source-1'
    else:
        assert not captured and result['evidence_id']
        if fault=='rejected_science':
            assert result['failure_category']=='data_boundary_not_supported'
            assert result['repair_owner']=='data'
            assert result['independent_review']['no_safe_next_data_operation'] is False
            assert result['independent_review']['failed_checks']==['no_safe_next_data_operation']
        else:
            assert result['failure_category']=='review_citation_invalid'
            assert result['repair_owner']=='supervisor'
        assert len(obj.client.seen)==(1 if fault=='rejected_science' else 2)


def test_provider_failure_is_not_relabelled_as_data_infeasibility(tmp_path,monkeypatch):
    obj,boundary=boundary_preparation(tmp_path,monkeypatch,[])
    from autosim.research.agent_budget import AgentCostBudgetError
    def budget_error(*a,**kw):raise AgentCostBudgetError('test ceiling')
    obj.client.chat_with_metadata=budget_error
    with pytest.raises(AgentCostBudgetError):
        obj._step_record_data_campaign_arm(campaign_id='campaign',arm='agent',seed=1,boundary=boundary)
    assert not (obj.output/'data_campaigns').exists()


@pytest.mark.parametrize('kind',['prompt','transport'])
def test_provider_value_or_io_failure_keeps_its_runtime_identity(tmp_path,monkeypatch,kind):
    obj,boundary=boundary_preparation(tmp_path,monkeypatch,[])
    from autosim.research.agent_runtime import AgentPromptTooLarge
    fault=AgentPromptTooLarge('runner prompt ceiling') if kind=='prompt' else OSError('provider connection lost')
    def fail(*a,**kw):raise fault
    obj.client.chat_with_metadata=fail
    with pytest.raises(type(fault)) as caught:
        obj._step_record_data_campaign_arm(campaign_id='campaign',arm='agent',seed=1,boundary=boundary)
    assert caught.value is fault
    assert not (obj.output/'data_campaigns').exists()


def test_native_review_denial_is_not_hidden_by_mixed_citation_namespaces(tmp_path,monkeypatch):
    initial={'boundary_evidence_supported':False,'no_safe_next_data_operation':False,
        'why':'A completed producer opened the data; repair its current consumer proof.',
        'citations':[{'source_id':'evidence-id-not-a-source-id'},{'source_id':'review-source-1'}]}
    obj,boundary=boundary_preparation(tmp_path,monkeypatch,[initial])
    result=obj._step_record_data_campaign_arm(campaign_id='campaign',arm='agent',seed=1,boundary=boundary)
    assert result['failure_category']=='data_boundary_not_supported'
    assert result['repair_owner']=='data' and 'repair its current consumer' in result['because']
    assert len(obj.client.seen)==1
    assert 'not an operator-only gate' in result['retry_when']
    assert not (obj.output/'data_campaigns').exists()


def test_independent_denial_reaches_next_main_context_and_survives_restart(tmp_path,monkeypatch):
    obj=preparation(tmp_path,[{'do':'read_the_checkout','why':'fixture decision'}])
    judgment={'boundary_evidence_supported':False,'no_safe_next_data_operation':False,
        'failed_checks':['no_safe_next_data_operation'],'why':'Repair the current consumer proof.',
        'authority':'unapproved_boundary_not_a_completed_experiment'}
    monkeypatch.setattr(obj,'do',lambda *a,**kw:{'outcome':'not attempted',
        'because':'data boundary not established','failure_category':'data_boundary_not_supported',
        'repair_owner':'data','retryable':True,'independent_review':judgment})
    monkeypatch.setattr(obj,'_refresh_document',lambda **kw:None)
    result=obj._run_cycle(max_steps=1)
    assert result['steps'][-1]['independent_review']==judgment
    assert obj.last_action['independent_review']==judgment
    assert obj._decision_facts()['recent_actions'][-1]['independent_review']==judgment
    restored=preparation(tmp_path)
    assert restored.last_action['independent_review']==judgment


def test_fix_can_read_candidate_job_admission_without_patching_scientific_guards(tmp_path):
    from autosim.research.harness_repair import read_source,ALLOWED
    row=read_source(tmp_path,{'path':'research/candidate_jobs.py','limit':65})
    assert 'frozen evaluation protocol' in row['excerpt']
    assert not row['candidate_edits_allowed'] and not row['live_activation_allowed']
    assert 'research/candidate_jobs.py' not in ALLOWED
    for ref in ['research/decision.py','research/training_work.py']:
        item=read_source(tmp_path,{'path':ref,'limit':65})
        assert not item['candidate_edits_allowed'] and not item['live_activation_allowed']
