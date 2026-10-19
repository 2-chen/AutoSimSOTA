"""Actual CPU producer/consumer work, then additive linkage without new execution."""
import json
from pathlib import Path
import sys

import pytest

from autosim.research import completed_evaluation_recovery as recovery
from autosim.research.common import atomic_json,digest,read_json
from autosim.research.metric_contract import MetricSpec
from autosim.research.receipt_verifier import verify_measurement


def fixture(tmp_path,monkeypatch,*,success=False,selected_data=False,wrong_consumption=False):
    from autosim.research.workspace_snapshot import create
    from autosim.research.declarative_backend import DeclarativeBackend
    from autosim.research.derived_research import DerivedResearch
    from autosim.research.compute_decision import ComputeDecision
    from autosim.research.adapter_protocol import OptimizationSpace
    from autosim.research.native_jobs import source_identity
    from autosim.research.checkpoint_handoff import catalog,select
    from tests.test_measurement_recovery import Review
    source=tmp_path/'source';source.mkdir()
    (source/'produce.py').write_text('''import json,sys
from pathlib import Path
root=Path(sys.argv[1]);bundle=root/'bundle';bundle.mkdir(parents=True)
completed=0
for n in range(17):
    completed += 1
(bundle/'learned.bin').write_bytes(b'toy policy, not a real benchmark')
(bundle/'config.json').write_text('{"architecture":"toy"}')
(root/'progress.json').write_text(json.dumps({'completed':completed}))
''')
    (source/'consume.py').write_text('''import json,sys,hashlib
from pathlib import Path
bundle=Path(sys.argv[1]);assert bundle.is_dir(),'complete bundle required'
files=sorted(p for p in bundle.rglob('*') if p.is_file())
rows=[(str(p.relative_to(bundle)),hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
sha=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
print('AUTOSIM_POLICY_LOADED '+json.dumps({'path':str(bundle),'sha256':sha}))
episodes=[{'episode_id':str(i),'success':SUCCESS} for i in range(2)]
for row in episodes:
    print('AUTOSIM_ROLLOUT_COMPLETED '+json.dumps({'episode_id':row['episode_id'],'policy_sha256':sha}))
value=sum(r['success'] for r in episodes)/2
print('AUTOSIM_METRIC_REPORTED '+json.dumps({'episode_ids':[r['episode_id'] for r in episodes],'policy_sha256':sha,'value':value}))
output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
(output/'result.json').write_text(json.dumps({'episodes':episodes}))
'''.replace('SUCCESS',repr(success)))
    output=tmp_path/'run';repo=output/'checkout';create(source,repo)
    if selected_data:
        script=repo/'produce.py'
        script.write_text(script.read_text()+'''
import hashlib
data=Path(sys.argv[2]);samples=json.loads(data.read_text())
print('AUTOSIM_TRAIN_DATA_CONSUMPTION '+json.dumps({
    'data_version_id':'extra','setting':'dataset_root','identity_field':'content_sha256',
    'identity':hashlib.sha256(data.read_bytes()).hexdigest(),'samples_read':len(samples),
    'reader_id':'main','consumption_stage':'sample_read'}))
''')
        if wrong_consumption:
            script.write_text(script.read_text().replace(
                "'data_version_id':'extra'", "'data_version_id':'not_the_selected_version'"))
    stages={'train':'def stage_argv_train(i):\n    return [i["python"],i["repo"]+"/produce.py",i["output"]]',
        'evaluate':'def stage_argv_evaluate(i):\n    return [i["python"],i["repo"]+"/consume.py",i["checkpoint"],i["output"]]'}
    if selected_data:
        stages['train']='def stage_argv_train(i):\n    return [i["python"],i["repo"]+"/produce.py",i["output"],i["settings"]["dataset_root"]]'
    answer={'stages':{'train':{'available':True,'entrypoint':'produce.py','artifact':'bundle/learned.bin'},
        'evaluate':{'available':True,'entrypoint':'consume.py','artifact':'result.json'}}}
    real=DerivedResearch(repo=repo,output=output,run_id='derived',
        backend=DeclarativeBackend(repo=repo,answer=answer,sources=stages,parameters={}),
        stages=stages,space=OptimizationSpace(),client=Review(),interpreter=Path(sys.executable),
        require_training_progress=True,require_policy_consumption=True,
        compute=ComputeDecision(device='cpu',device_index=0,environment={},why='CPU contract fixture'))
    real.metric_spec=MetricSpec(name='success_rate',direction='maximize',minimum=0,maximum=1,
        source='json',artifact_root='output',artifact_pattern='result.json',json_key='episodes',
        json_value_key='success',episode_id_column='episode_id',aggregation='mean',min_samples=2)
    real.controller_decision_id='a'*16
    settings={'episodes':2,'seed':11,'steps':17}
    if selected_data:
        from autosim.research.data_versions import register
        atomic_json(output/'training.json',[1,2,3])
        register(output,repo,{'id':'extra','dataset_ref':'training.json',
            'source_refs':[],'reason':'CPU selection fixture, not robot data'},
            {'allowed_training_data':True,'loader_connection_supported':True,'sources_supported':True})
        settings['dataset_root']='data-version:extra'
    failed=real.measure(settings=settings,label='round_4')
    assert not failed['ok'] and failed['train']['returncode']==0
    atomic_json(real.run_root/'controller_session.json',{'status':'paused','action':'',
        'pending_action':None,'run_id':'derived'})
    atomic_json(output/'derived_stages.json',{s:{'source':code,'row':answer['stages'][s],
        'parameters':{}} for s,code in stages.items()})
    atomic_json(output/'environment.json',{'interpreter':sys.executable})
    producer=next(p for p in catalog(output)['producers']
                  if p['training_attempt_id']==failed['train']['attempt_id'])
    selection=select(output,producer['output_ref']+'/bundle',catalog(output))
    job='b'*32;native_settings={'episodes':2,'seed':11}
    request={'job_id':job,'run_id':'derived','repo':str(repo),'stage':'evaluate',
        'settings':native_settings,'source_stage_scope':['evaluate'],
        'source_identity':source_identity(output,repo,stage_scope=['evaluate']),
        'checkpoint_selection':selection}
    atomic_json(output/f'native_jobs/{job}/request.json',request)
    real.controller_decision_id=job
    native=real.run_stage('evaluate',settings=native_settings,checkpoint=selection['selected_path'])
    assert native['status']=='completed' and native['returncode']==0,native
    atomic_json(output/f'native_jobs/{job}/result.json',{'job_id':job,'status':'completed',
        'source_identity_after':request['source_identity'],'stage_result':native})
    monkeypatch.setattr(real,'run_stage',lambda *a,**kw:pytest.fail('must not repeat native work'))
    monkeypatch.setattr(real.client,'chat_with_metadata',lambda *a,**kw:pytest.fail('no paid review permitted'))
    return real,job,failed,native


@pytest.mark.parametrize('success',[False,True])
def test_completed_evaluation_reuses_exact_native_work_and_preserves_failures(tmp_path,monkeypatch,success):
    real,job,failed,native=fixture(tmp_path,monkeypatch,success=success)
    parent=real.run_root/'measurements/round_4.json'
    receipt=real.run_root/f"attempts/{native['attempt_id']}/receipt.json"
    original={p:p.read_bytes() for p in [parent,receipt,real.output/f'native_jobs/{job}/result.json']}
    rows=recovery.offered(real.output,real.repo,'derived')
    assert rows[0]['arguments']=={'label':'round_4','job_id':job}
    result=recovery.record(real,label='round_4',job_id=job)
    assert result['ok'] and result['metric_value']==float(success)
    assert result['train']['attempt_id']==failed['train']['attempt_id'] and result['train']['reused']
    assert result['evaluate']['attempt_id']==native['attempt_id']
    assert result['settings']['steps']==17 and result['evaluation_settings']=={'episodes':2,'seed':11}
    assert verify_measurement(real.run_root,result['label'])['status']=='consistent'
    assert all(p.read_bytes()==v for p,v in original.items())
    assert recovery.record(real,label='round_4',job_id=job)==result
    assert recovery.offered(real.output,real.repo,'derived')==[]


@pytest.mark.parametrize('fault',['source','recipe','protocol','request_seed','result_bytes',
    'producer','receipt','held_out','log','unknown_job','active_transaction'])
def test_changed_or_unknown_context_never_creates_a_linked_score(tmp_path,monkeypatch,fault):
    real,job,failed,native=fixture(tmp_path,monkeypatch)
    if fault=='source':(real.repo/'consume.py').write_text('# changed evaluator')
    elif fault=='recipe':
        path=real.run_root/'measurements/round_4.json';row=read_json(path)
        row['settings']['steps']=999;atomic_json(path,row)
    elif fault=='protocol':
        path=real.run_root/'comparison_protocol.json';row=read_json(path)
        row['settings']['seed']=22;atomic_json(path,row)
    elif fault=='request_seed':
        path=real.output/f'native_jobs/{job}/request.json';row=read_json(path)
        row['settings']['seed']=22;atomic_json(path,row)
    elif fault=='result_bytes':Path(native['metric_result_artifact']['path']).write_text('{}')
    elif fault=='producer':
        path=real.output/f'native_jobs/{job}/request.json';row=read_json(path)
        row['checkpoint_selection']['producer_evidence_ref']='evidence/'+('f'*32)+'.json'
        atomic_json(path,row)
    elif fault=='receipt':
        path=real.run_root/f"attempts/{native['attempt_id']}/receipt.json";row=read_json(path)
        row['status']='running';atomic_json(path,row)
    elif fault=='held_out':atomic_json(real.run_root/'confirmation_attempts/exposed.json',{})
    elif fault=='log':(real.run_root/f"evaluate/attempts/{native['attempt_id']}/output.log").write_text('changed')
    elif fault=='active_transaction':real._held={'native-change':True}
    else:job='f'*32
    with pytest.raises((ValueError,OSError,KeyError)):
        recovery.record(real,label='round_4',job_id=job)
    assert not list((real.run_root/'measurements').glob('round_4_eval_*.json'))


def test_link_verifier_rejects_changed_alias_or_supplement(tmp_path,monkeypatch):
    real,job,_,_=fixture(tmp_path,monkeypatch)
    result=recovery.record(real,label='round_4',job_id=job)
    path=real.run_root/'measurements'/f"{result['label']}.json"
    result['consumed_policy_path']=result['policy_artifact']['path']
    atomic_json(path,result)
    assert verify_measurement(real.run_root,result['label'])['status']!='consistent'


def test_linked_historical_score_uses_archives_not_future_checkout_or_environment(tmp_path,monkeypatch):
    real,job,_,_=fixture(tmp_path,monkeypatch)
    result=recovery.record(real,label='round_4',job_id=job)
    (real.repo/'consume.py').write_text('# unrelated subsequent candidate')
    (real.repo/'produce.py').write_text('# later trainer revision')
    atomic_json(real.output/'environment.json',{'interpreter':'/future/unavailable/python'})
    audit=verify_measurement(real.run_root,result['label'])
    assert audit['status']=='consistent',audit


def test_historical_training_counter_is_still_rechecked_after_linkage(tmp_path,monkeypatch):
    real,job,failed,_=fixture(tmp_path,monkeypatch)
    result=recovery.record(real,label='round_4',job_id=job)
    receipt=read_json(real.run_root/f"attempts/{failed['train']['attempt_id']}/receipt.json")
    path=Path(receipt['output_directory'])/'progress.json'
    atomic_json(path,{'completed':999})
    assert verify_measurement(real.run_root,result['label'])['status']!='consistent'


def test_later_confirmation_does_not_erase_preexisting_development_evidence(tmp_path,monkeypatch):
    real,job,_,_=fixture(tmp_path,monkeypatch)
    result=recovery.record(real,label='round_4',job_id=job)
    atomic_json(real.run_root/'confirmation_attempts/later.json',{'status':'failed'})
    assert verify_measurement(real.run_root,result['label'])['status']=='consistent'
    assert recovery.offered(real.output,real.repo,'derived')==[]


def test_completed_job_keeps_preexisting_data_selection_and_rechecks_actual_bytes(tmp_path,monkeypatch):
    real,job,failed,_=fixture(tmp_path,monkeypatch,selected_data=True)
    old=failed['data_selection']
    old_bytes=(real.output/old['record_ref']).read_bytes()
    result=recovery.record(real,label='round_4',job_id=job)
    assert result['data_consumption']['status']=='verified'
    assert result['data_consumption']['exposure'][0]['samples_read']==3
    assert result['data_consumption']['exposure'][0]['consumption_stage']=='sample_read'
    assert result['data_selection']['id']!=old['id']
    assert result['data_selection']['training_attempt_id']==old['training_attempt_id']
    assert (real.output/old['record_ref']).read_bytes()==old_bytes
    assert verify_measurement(real.run_root,result['label'])['status']=='consistent'
    from autosim.research.data_campaign import _measurement
    linked,_=_measurement(real.output,'derived',result['label'])
    assert linked==result
    from autosim.research.data_campaign import _arm_data_bindings
    monkeypatch.setattr('autosim.research.data_versions.resolve',
        lambda *a,**kw:pytest.fail('historical score must not get a fake current loader slot'))
    (real.repo/'produce.py').write_text('# a future candidate, not the past producer')
    bindings,training,acquired=_arm_data_bindings(real.output,real.repo,'derived',result,
        fixed=False,acquired_version_id='extra',training_version_id='extra')
    assert training['setting']=='dataset_root' and training['id']=='extra'
    assert acquired['id']=='extra' and len(bindings)==2
    with pytest.raises(ValueError,match='original native training binding'):
        _arm_data_bindings(real.output,real.repo,'derived',result,fixed=True,
            acquired_version_id='',training_version_id='some_other_version')
    atomic_json(real.output/'training.json',[9])
    assert verify_measurement(real.run_root,result['label'])['status']!='consistent'
    with pytest.raises(ValueError):
        _arm_data_bindings(real.output,real.repo,'derived',result,fixed=True,
            acquired_version_id='',training_version_id='extra')


@pytest.mark.parametrize('fault', ['native_reader_label', 'selection_proof', 'data_bytes'])
def test_recovery_catalog_does_not_offer_a_link_the_execution_data_gate_will_refuse(
        tmp_path, monkeypatch, fault):
    real, job, parent, native = fixture(tmp_path, monkeypatch, selected_data=True,
                                       wrong_consumption=fault == 'native_reader_label')
    if fault == 'selection_proof':
        selection = real.output/parent['data_selection']['record_ref']
        document = read_json(selection)
        document['label'] = 'invented_old_round'
        atomic_json(selection, document)
    elif fault == 'data_bytes':
        atomic_json(real.output/'training.json', [999])
    original_path = real.run_root/'measurements/round_4.json'
    old = original_path.read_bytes()
    blocked = []
    assert recovery.offered(real.output, real.repo, 'derived', _blocked=blocked) == []
    if fault == 'native_reader_label':
        # Current producers already stop at training_data, before the older
        # recovery context accepts them. Retain that stronger existing guard.
        assert parent['where'] == 'training_data' and blocked == []
        blocker = recovery.registered_training_blocker(real.output, 'derived', parent)
    else:
        assert len(blocked) == 1
        blocker = blocked[0]
        assert blocker['job_id'] == job and blocker['native_metric_value_not_linked'] == 0.0
    assert blocker['status'] == 'blocked_registered_training_data'
    assert blocker['label'] == 'round_4'
    assert blocker['training_attempt_id'] == parent['train']['attempt_id']
    assert blocker['next_action'] == 'research_task' and 'arguments' not in blocker
    with pytest.raises(ValueError):
        recovery.record(real, label='round_4', job_id=job)
    assert original_path.read_bytes() == old
    assert not (real.run_root/'measurements'/f'round_4_eval_{job[:12]}.json').exists()


@pytest.mark.parametrize('selected_data', [False, True])
def test_recovery_catalog_preserves_unknown_native_training_and_verified_registered_data(
        tmp_path, monkeypatch, selected_data):
    real, job, parent, _ = fixture(tmp_path, monkeypatch, selected_data=selected_data)
    blocked = []
    assert recovery.registered_training_blocker(real.output, 'derived', parent) is None
    offered = recovery.offered(real.output, real.repo, 'derived', _blocked=blocked)
    assert len(offered) == 1 and offered[0]['job_id'] == job and not blocked
    assert offered[0]['metric_value'] == 0.0  # A valid negative result is still offered.


@pytest.mark.parametrize('success',[False,True])
def test_linked_result_can_be_explicitly_adopted_only_if_better_without_replay(tmp_path,monkeypatch,success):
    from autosim.research import measurement_adoption as adoption
    real,job,_,_=fixture(tmp_path,monkeypatch,success=success)
    # Toy incumbent metadata exercises the transaction; it is not a robot baseline.
    real.snapshots.capture([],repo=real.repo,name='toy-incumbent')
    real.snapshots.advance('toy-incumbent',score=0.0,scale='success_rate')
    atomic_json(real.run_root/'measurements/baseline.json',{'label':'baseline','ok':False})
    session=read_json(real.run_root/'controller_session.json')
    atomic_json(real.run_root/'controller_session.json',{**session,'history':[]})
    atomic_json(real.run_root/'research_report.json',{'rounds':[],
        'best':real.snapshots.best().as_dict(),'best_export':{}})
    result=recovery.record(real,label='round_4',job_id=job)
    offers=adoption.offered(real.run_root)
    assert bool(offers) is success
    assert real.snapshots.best().name=='toy-incumbent'  # No automatic adoption.
    if success:
        original=(real.run_root/'measurements/round_4.json').read_bytes()
        request={k:offers[0][k] for k in ('label','expected_best','measurement_sha256')}
        adopted=adoption.adopt(real,**request)
        assert adopted['ok'] and real.snapshots.best().score==1.0
        assert (real.run_root/'measurements/round_4.json').read_bytes()==original
        assert verify_measurement(real.run_root,result['label'])['status']=='consistent'
        assert adoption.adopt(real,**request)['already_applied']


def test_prepare_exposes_explicit_completed_job_without_reselecting_policy(monkeypatch):
    from types import SimpleNamespace
    from autosim.research.prepare import Preparation
    calls=[]
    controller=SimpleNamespace()
    prep=SimpleNamespace(run_id='derived',_research_controller=lambda:(controller,None))
    def record(research,**kwargs):
        calls.append((research,kwargs))
        return {'label':'round_4_eval_bbbbbbbbbbbb','metric_value':0.0,
            'train':{'attempt_id':'a'*32},'evaluate':{'attempt_id':'c'*32},
            'completed_evaluation_link':{'attachment_ref':'completed_evaluation_links/proof.json'}}
    monkeypatch.setattr(recovery,'record',record)
    result=Preparation._step_correct_policy_handoff(prep,label='round_4',job_id='b'*32)
    assert calls==[(controller,{'label':'round_4','job_id':'b'*32})]
    assert result['outcome']=='measurement_recorded' and result['metric_value']==0.0
    assert result['training_reused'] is True
    with pytest.raises(ValueError,match='omit checkpoint_ref'):
        Preparation._step_correct_policy_handoff(prep,label='round_4',job_id='b'*32,
                                                checkpoint_ref='unrelated-policy')
    assert len(calls)==1
