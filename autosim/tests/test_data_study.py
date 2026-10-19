"""Study control tests use synthetic contracts, never robotics performance data."""
from copy import deepcopy
from pathlib import Path

import pytest

from autosim.research.common import atomic_json, read_json
from tools import data_study as study


def manifest():
    return {'schema_version':1, 'study_id':'unit_study', 'phase':'draft',
        'track':'equal_accepted_data', 'tasks':[{'id':'task', 'repository':'fixture',
            'native_task':'task', 'family':'fixture', 'role':'development'}],
        'methods':['fixed_continuation','random_quality','fdas'],
        'training_seeds':[0,1,2], 'agent_replicas':[0],
        'protocol':{k:None for k in study.PROTOCOL_FIELDS}, 'bindings':{}}


def bind(document, root, method='fixed_continuation', seed=0, replica=0, label='baseline'):
    document['bindings'][f'task--{method}--s{seed}--a{replica}']={
        'output':str(root), 'repository':str(root/'checkout'), 'run_id':'derived',
        'measurement_label':label, 'campaign_id':'campaign' if method in study.ACQUISITION else None}


def measured(root, value=0, label='baseline', **extra):
    atomic_json(root/f'research/derived/measurements/{label}.json', {'label':label,
        'ok':True, 'metric_value':value,
        'metric':{'name':'success','unit':'fraction','direction':'maximize','aggregation':'mean'}, **extra})


def test_pending_is_not_zero_and_all_denominators_preserved():
    report=study.inspect(manifest())
    assert len(report['rows'])==9
    assert all(r['status']=='pending' and r['metric_value'] is None for r in report['rows'])
    assert all(r['mean'] is None and r['planned']==3 for r in report['summaries'])
    assert not report['formal_paper_eligible']
    assert '待安排' in study.markdown(report)


@pytest.mark.parametrize('field,value', [('schema_version',True),('training_seeds',[0,0]),
    ('training_seeds',[True]),('agent_replicas',[]),('methods',[{}]),
    ('track','both'),('phase','complete')])
def test_invalid_matrix_is_rejected(field,value):
    row=manifest();row[field]=value
    with pytest.raises(ValueError):study.validate(row)


def test_duplicate_task_ids_rejected():
    row=manifest();row['tasks']*=2
    with pytest.raises(ValueError,match='duplicate task'):study.validate(row)


def test_missing_protocol_is_not_freezable(tmp_path):
    row=manifest();row['phase']='formal'
    with pytest.raises(ValueError,match='unresolved'):study.freeze(row,tmp_path/'freeze')


def test_acquisition_only_formal_design_does_not_require_qt_or_fdas(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'harness_identity', lambda *a: 'a' * 64)
    row = finalized(tmp_path)
    row['methods'] = ['fixed_official', 'fixed_continuation', 'random_quality',
                      'rule_targeted', 'agent_targeted']
    row['bindings'] = {}
    for _, cell in study.matrix(row).items():
        bind(row, tmp_path / cell['id'], cell['method'], cell['training_seed'])
    frozen = study.freeze(row, tmp_path / 'acquisition-freeze')
    report = study.inspect(row, frozen=frozen)
    assert len(report['rows']) == 15 and not report['formal_paper_eligible']
    assert not report['design']['selection_increment_planned']
    assert {r['id'] for r in report['design']['planned_contrasts']} == {
        'targeting_vs_random', 'reasoning_vs_rule'}
    assert '不检验Qt' in study.markdown(report)
    assert all(r['status'] == 'pending' for r in report['rows'])


@pytest.mark.parametrize('methods', [
    ['random_quality', 'agent_targeted'],
    ['rule_targeted', 'agent_targeted'],
    ['random_quality', 'rule_targeted'],
])
def test_formal_acquisition_cannot_omit_its_direct_controls(tmp_path, methods):
    row = finalized(tmp_path)
    row['methods'], row['bindings'] = methods, {}
    with pytest.raises(ValueError, match='formal'):
        study.validate(row)


def test_design_questions_do_not_promote_selection_or_resource_authority():
    row = manifest()
    view = study.design_view(row)
    assert any(r['id'] == 'legacy_joint_vs_random' for r in view['planned_contrasts'])
    assert not view['selection_increment_planned']
    row['methods'].append('targeted_quality')
    assert study.design_view(row)['selection_increment_planned']
    assert view['not_a_budget_grant_or_task_capability_check']
    assert view['equal_total_compute_not_implied']


def test_legacy_report_still_renders_without_inventing_design_metadata():
    report = study.inspect(manifest())
    report.pop('design')
    original = deepcopy(report)
    assert '旧快照未记录' in study.markdown(report)
    assert report == original and not report['formal_paper_eligible']


def finalized(tmp_path):
    row=manifest();row['phase']='formal'
    row['protocol']={k:(2 if k=='accepted_new_episodes' else 'fixture protocol') for k in study.PROTOCOL_FIELDS}
    for cell, plan in study.matrix(row).items():
        bind(row,tmp_path/cell,plan['method'],plan['training_seed'])
    return row


def test_freeze_is_immutable_and_does_not_promote_scores(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'harness_identity',lambda *a:'a'*64)
    row=finalized(tmp_path)
    frozen=study.freeze(row,tmp_path/'frozen')
    assert frozen==study.freeze(row,tmp_path/'frozen')
    assert study.inspect(row,frozen=frozen)['current_harness_matches_freeze'] is True
    assert study.inspect(row,frozen=frozen)['formal_paper_eligible'] is False
    changed=deepcopy(row);changed['protocol']['training_work']='different'
    with pytest.raises(ValueError,match='immutable'):study.freeze(changed,tmp_path/'frozen')
    with pytest.raises(ValueError,match='identity'):study.inspect(changed,frozen=frozen)
    monkeypatch.setattr(study,'harness_identity',lambda *a:'b'*64)
    report=study.inspect(row,frozen=frozen)
    assert report['current_harness_matches_freeze'] is False
    assert not report['formal_paper_eligible']


def test_cannot_relabel_initialized_run_as_preregistered(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'harness_identity',lambda *a:'a'*64)
    row=finalized(tmp_path)
    first=Path(next(iter(row['bindings'].values()))['output']);first.mkdir()
    with pytest.raises(ValueError,match='uninitialized'):study.freeze(row,tmp_path/'freeze')


def test_preallocated_but_unstarted_formal_runs_are_pending_not_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'harness_identity', lambda *a: 'a' * 64)
    row = finalized(tmp_path)
    snapshot = study.freeze(row, tmp_path / 'freeze')
    report = study.inspect(row, frozen=snapshot)
    assert all(r['status'] == 'pending' and r['metric_value'] is None for r in report['rows'])
    assert all('no run started' in r['pending_reason'] for r in report['rows'])
    assert not report['formal_paper_eligible']


def test_formal_preallocation_cannot_follow_a_symlink_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'harness_identity', lambda *a: 'a' * 64)
    row = finalized(tmp_path)
    target = tmp_path / 'elsewhere'
    target.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(target, target_is_directory=True)
    first = next(iter(row['bindings'].values()))
    first['output'] = str(alias / 'not-created')
    with pytest.raises(ValueError, match='symlinks'):
        study.freeze(row, tmp_path / 'freeze')
    assert not (target / 'not-created').exists()


def test_one_measurement_cannot_stand_for_different_seeds(tmp_path):
    row=manifest();bind(row,tmp_path,seed=0);bind(row,tmp_path,seed=1)
    with pytest.raises(ValueError,match='multiple cells'):study.inspect(row)


def test_zero_valid_native_measurement_is_preserved(tmp_path,monkeypatch):
    row=manifest();bind(row,tmp_path)
    measured(tmp_path)
    monkeypatch.setattr(study,'verify_measurement',lambda *a:{'status':'consistent'})
    report=study.inspect(row)
    assert report['rows'][0]['metric_value']==0
    assert report['rows'][0]['status']=='verified_development'
    summary=report['summaries'][0]
    assert summary['verified']==1 and summary['planned']==3 and not summary['complete']
    assert summary['mean']==0 and summary['sample_std'] is None


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),None])
def test_invalid_numeric_score_is_not_a_result(tmp_path,monkeypatch,value):
    row=manifest();bind(row,tmp_path);measured(tmp_path,0)
    # The real writer already forbids NaN/Inf. Exercise the reader against a
    # corrupt/foreign producer independently, without weakening that writer.
    read=study.regular_json
    def corrupt(path):
        result=read(path)
        result['metric_value']=value
        return result
    monkeypatch.setattr(study,'regular_json',corrupt)
    monkeypatch.setattr(study,'verify_measurement',lambda *a:{'status':'consistent'})
    assert study.inspect(row)['rows'][0]['status']=='evidence_unavailable'


def test_receipt_and_heldout_cannot_be_bypassed(tmp_path,monkeypatch):
    row=manifest();bind(row,tmp_path);measured(tmp_path)
    monkeypatch.setattr(study,'verify_measurement',lambda *a:{'status':'inconsistent'})
    assert study.inspect(row)['rows'][0]['metric_value'] is None
    measured(tmp_path,confirmation=True)
    monkeypatch.setattr(study,'verify_measurement',lambda *a:{'status':'consistent'})
    assert study.inspect(row)['rows'][0]['status']=='evidence_unavailable'


def test_acquisition_requires_exact_campaign_arm_and_consumption(tmp_path,monkeypatch):
    row=manifest();bind(row,tmp_path,method='fdas',label='round_1');measured(tmp_path,.25,'round_1')
    monkeypatch.setattr(study,'verify_measurement',lambda *a:{'status':'consistent'})
    observation={'campaign':'campaign','arm':'targeted','seed':0,'measurement_label':'round_1'}
    monkeypatch.setattr(study,'data_evidence',lambda *a,**kw:{'rows':[observation]})
    report=study.inspect(row)
    assert next(r for r in report['rows'] if r['method']=='fdas' and r['training_seed']==0)['metric_value']==.25
    observation['arm']='random'
    report=study.inspect(row)
    assert next(r for r in report['rows'] if r['method']=='fdas' and r['training_seed']==0)['metric_value'] is None


def test_agent_replicas_are_not_pooled_as_training_seeds(tmp_path,monkeypatch):
    row=manifest();row['methods']=['fixed_continuation'];row['agent_replicas']=[0,1]
    for replica,value in [(0,0),(1,1)]:
        root=tmp_path/str(replica);bind(row,root,replica=replica);measured(root,value)
    monkeypatch.setattr(study,'verify_measurement',lambda *a:{'status':'consistent'})
    report=study.inspect(row)
    assert [r['mean'] for r in report['summaries']]==[0,1]
    assert all(r['sample_std'] is None and r['planned']==3 for r in report['summaries'])


def test_different_metric_units_are_not_averaged(tmp_path,monkeypatch):
    row=manifest();row['methods']=['fixed_continuation']
    bind(row,tmp_path,seed=0,label='baseline');bind(row,tmp_path,seed=1,label='round_1')
    measured(tmp_path,.5);measured(tmp_path,50,'round_1',metric={'unit':'percent'})
    monkeypatch.setattr(study,'verify_measurement',lambda *a:{'status':'consistent'})
    summary=study.inspect(row)['summaries'][0]
    assert not summary['metric_semantics_match'] and summary['mean'] is None


def test_symlink_records_and_destinations_are_refused(tmp_path):
    target=tmp_path/'target';target.mkdir()
    alias=tmp_path/'alias';alias.symlink_to(target,target_is_directory=True)
    atomic_json(target/'manifest.json',manifest())
    with pytest.raises(ValueError,match='symlinks'):study.regular_json(alias/'manifest.json')
    with pytest.raises(ValueError,match='symlinks'):study.safe_destination(alias/'out')


def test_cli_snapshot_reproducible_and_readonly(tmp_path):
    source=tmp_path/'manifest.json';atomic_json(source,manifest())
    old=source.read_bytes()
    assert study.main(['inspect',str(source),str(tmp_path/'reports')])==0
    assert study.main(['inspect',str(source),str(tmp_path/'reports')])==0
    assert source.read_bytes()==old
    reports=list((tmp_path/'reports').glob('*/analysis.json'))
    assert len(reports)==1 and not read_json(reports[0])['formal_paper_eligible']


def test_public_draft_has_four_tasks_and_sixty_pending_cells():
    path=Path(__file__).resolve().parents[2]/'docs/paper/EXPERIMENT_MATRIX.json'
    report=study.inspect(study.regular_json(path))
    assert len(report['rows'])==60
    assert all(row['status']=='pending' for row in report['rows'])
    assert {r['method'] for r in report['rows']} == {
        'fixed_official', 'fixed_continuation', 'random_quality', 'rule_targeted', 'agent_targeted'}
    assert not report['design']['selection_increment_planned']


def test_qt_draft_is_separate_unbound_and_not_a_completed_result():
    path = Path(__file__).resolve().parents[2] / 'docs/paper/EXPERIMENT_MATRIX_QT.json'
    document = study.regular_json(path)
    report = study.inspect(document)
    assert document['phase'] == 'draft' and document['bindings'] == {}
    assert len(report['rows']) == 72 and report['design']['selection_increment_planned']
    assert report['design']['unbound_cells'] == 72
    assert not report['formal_paper_eligible']


def test_costs_deduplicate_whole_output_not_research_run_id(tmp_path):
    row=manifest()
    bind(row,tmp_path,seed=0,label='baseline')
    bind(row,tmp_path,seed=1,label='round_1')
    row['bindings']['task--fixed_continuation--s1--a0']['run_id']='another_scope'
    atomic_json(tmp_path/'agent/cost_ledger.json',{'entries':[{'status':'settled','actual_usd':.2}]})
    report=study.inspect(row)
    assert len(report['costs'])==1 and report['costs'][0]['model_settled_usd']==.2


def test_changed_protocol_has_distinct_report_snapshot():
    row=manifest()
    first=study.inspect(row)
    row['protocol']['exclusion_rules']='different exclusions'
    second=study.inspect(row)
    assert first['manifest_sha256'] != second['manifest_sha256']
    assert first['snapshot_id'] != second['snapshot_id']
