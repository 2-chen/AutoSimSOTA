"""Real CPU producer -> sample reads -> trained bytes -> evaluator, not a robot score."""
import json
from pathlib import Path

from autosim.research import data_versions, recorder, run_record
from autosim.research.adapter_protocol import Axis, OptimizationSpace
from autosim.research.common import digest
from autosim.research.declarative_backend import checked_function
from autosim.research.ideas import Idea, CLEARED
from tests.test_derived_research import research


def test_collection_data_version_training_and_formal_comparison(tmp_path):
    real=research(tmp_path,stages=('collect','train','evaluate'))
    original=tmp_path/'original.json';original.write_text('[1]')
    trainer='''import pathlib,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
data=pathlib.Path(sys.argv[2]);raw=data.read_bytes();samples=json.loads(raw)
if len(samples)>1:
 print('AUTOSIM_TRAIN_DATA_CONSUMPTION '+json.dumps(dict(data_version_id='targeted',setting='dataset_root',identity_field='content_sha256',identity=hashlib.sha256(raw).hexdigest(),samples_read=len(samples)*2,exposure=dict(batches=2,unique_samples=len(samples),source_samples=dict(targeted=len(samples)*2)))))
(p/'models').mkdir();(p/'models/model.pth').write_text(str(len(samples)))
print('global_step=2')
'''
    evaluator="import pathlib,sys; n=int(pathlib.Path(sys.argv[1]).read_text()); print('succ: 0.80' if n>=3 else 'succ: 0.20')"
    collector='''import pathlib,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
data=p/'data.json';data.write_text('[1,2,3]')
raw=data.read_bytes();episodes=json.loads(raw)
print('AUTOSIM_DATA_PROFILE '+json.dumps(dict(identity_field='content_sha256',identity=hashlib.sha256(raw).hexdigest(),scope='full',format='JSON scalar episodes (CPU fixture)',training_only=True,counts=dict(episodes=len(episodes),samples=len(episodes)),issues=dict(invalid_values=sum(not isinstance(e,int) for e in episodes)))))
'''
    sources={
        'train':f'def stage_argv_train(i):\n    return [i["python"],"-c",{trainer!r},i["output"],i["settings"].get("dataset_root",{str(original)!r})]\n',
        'evaluate':f'def stage_argv_evaluate(i):\n    return [i["python"],"-c",{evaluator!r},i["checkpoint"]]\n',
        'collect':f'def stage_argv_collect(i):\n    return [i["python"],"-c",{collector!r},i["output"]]\n'}
    for stage,source in sources.items():
        real.sources[stage]=source
        real.backend.sources[stage]=source
        real.backend._functions[stage]=checked_function(source,'stage_argv_'+stage)
    real.backend.stages['collect']['artifact']='data.json'
    real.backend.stages['evaluate']['artifact']=''
    real.space=OptimizationSpace(training=(Axis('dataset_root','choice','training data version',
        values=('data-version:targeted',),optional=True),))
    real.require_training_progress=True
    baseline=real.run(rounds=1,yield_after_action=True)
    before=digest(real.run_root/'measurements/baseline.json')
    collected=real.collect({})
    assert collected['ran']
    data_versions.register(real.output,real.repo,
        dict(id='targeted',dataset_ref=str(Path(collected['data']).relative_to(real.output)),
             source_refs=[],reason='synthetic train-only coverage extension',
             asset=dict(kind='synthetic',producer_evidence_id=collected['evidence_id'])),
        dict(allowed_training_data=True,loader_connection_supported=True,sources_supported=True))
    real.library.add(Idea(label='targeted data',granularity='param',risk='low',
        mechanism='additional development samples',change={'dataset_root':'data-version:targeted'},
        status=CLEARED,why='verify pipeline'))
    report=real.run(rounds=1,yield_after_action=True,selected_idea_label='targeted data',
                    main_controller_owns_selection=True)
    assert report['rounds'][0]['metric_value']==0.2
    assert report['rounds'][-1]['metric_value']==0.8
    measured=json.loads((real.run_root/'measurements/round_1.json').read_text())
    assert measured['data_consumption']['status']=='verified'
    assert measured['data_consumption']['exposure'][0]['source_samples']=={'targeted':6}
    card=json.loads((real.output/'data_versions/targeted.json').read_text())
    assert card['asset']['producer']['evidence_id']==collected['evidence_id']
    assert card['asset']['profile']['counts']['episodes']==3
    view=run_record.build_report_view(real.output,real.run_id)
    snap=recorder.make_snapshot(real.output,view,{'actions':[],'plan':{}})
    document=recorder.render(real.output,snap,{})
    assert '原生采集' not in document  # synthetic test data, not a robotics collection claim
    assert '合成数据' in document and 'targeted：6' in document
    assert '3 条 episode' in document and '实读 6 样本次' in document
    assert digest(real.run_root/'measurements/baseline.json')==before
    assert report['performance_claims']['global_sota_verified'] is False


def test_native_reversible_filter_to_real_batch_reads_and_score(tmp_path):
    """A second real process transforms data; its metadata alone cannot earn a score."""
    real=research(tmp_path,stages=('collect','train','evaluate'))
    real.output.mkdir(exist_ok=True)
    original=real.output/'original.json';original.write_text('[1]');raw_before=digest(original)
    train='''import pathlib,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
data=pathlib.Path(sys.argv[2]);raw=data.read_bytes();samples=json.loads(raw)
if data.name=='selected.json':
 batches=[samples,samples];reads=sum(len(batch) for batch in batches)
 print('AUTOSIM_TRAIN_DATA_CONSUMPTION '+json.dumps(dict(data_version_id='selected',setting='dataset_root',identity_field='content_sha256',identity=hashlib.sha256(raw).hexdigest(),samples_read=reads,exposure=dict(batches=len(batches),unique_samples=len(set(samples)),source_samples=dict(collected=reads)))))
(p/'models').mkdir();(p/'models/model.pth').write_text(str(len(samples)))
print('global_step=2')
'''
    collect='''import pathlib,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
data=p/'collected.json';data.write_text('[1,2,3]');raw=data.read_bytes();rows=json.loads(raw)
print('AUTOSIM_DATA_PROFILE '+json.dumps(dict(identity_field='content_sha256',identity=hashlib.sha256(raw).hexdigest(),scope='full',format='JSON scalar episodes (CPU fixture)',training_only=True,counts=dict(episodes=len(rows),samples=len(rows)))))
'''
    evaluate="import pathlib,sys; print('succ: 0.80' if int(pathlib.Path(sys.argv[1]).read_text())>=2 else 'succ: 0.20')"
    for name,body,args in [('train',train,f'i["settings"].get("dataset_root",{str(original)!r})'),
                           ('collect',collect,''),('evaluate',evaluate,'i["checkpoint"]')]:
        first='i["checkpoint"]' if name=='evaluate' else 'i["output"]'
        extra=', '+args if args and name!='evaluate' else ''
        source=f'def stage_argv_{name}(i):\n    return [i["python"],"-c",{body!r},{first}{extra}]\n'
        real.sources[name]=source;real.backend.sources[name]=source
        real.backend._functions[name]=checked_function(source,'stage_argv_'+name)
    real.backend.stages['collect']['artifact']='collected.json'
    real.backend.stages['evaluate']['artifact']=''
    real.space=OptimizationSpace(training=(Axis('dataset_root','choice','data view',values=('data-version:selected',),optional=True),))
    real.require_training_progress=True
    parent_record=real.output/'data_versions/collected.json'
    process='''import pathlib,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
record=pathlib.Path(sys.argv[2])
if not record.exists():
 print('Original baseline uses original inputs; no derived view requested.');sys.exit(0)
parent=json.loads(record.read_text())
raw=pathlib.Path(parent['source']).read_bytes();rows=json.loads(raw)
assert parent['identity']==hashlib.sha256(raw).hexdigest()
selected=[episode for episode in rows if episode>=2]
data=p/'selected.json';data.write_text(json.dumps(selected));loaded=data.read_bytes()
parents=[dict(id=parent['id'],identity_field='content_sha256',identity=hashlib.sha256(raw).hexdigest(),record_sha256=hashlib.sha256(record.read_bytes()).hexdigest())]
print('AUTOSIM_DATA_PROFILE '+json.dumps(dict(identity_field='content_sha256',identity=hashlib.sha256(loaded).hexdigest(),scope='full',format='JSON scalar episodes (CPU fixture)',training_only=True,counts=dict(episodes=len(selected),samples=len(selected)),parents=parents,recipe=dict(operation='filter complete episodes',parameters=dict(minimum_value=2),preserves_episode_groups=True))))
'''
    source=f'def stage_argv_prepare_data(i):\n    return [i["python"],"-c",{process!r},i["output"],{str(parent_record)!r}]\n'
    real.sources['prepare_data']=source;real.backend.sources['prepare_data']=source
    real.backend._functions['prepare_data']=checked_function(source,'stage_argv_prepare_data')
    real.backend.stages['prepare_data']=dict(available=True,artifact='selected.json')
    real.run(rounds=1,yield_after_action=True)
    baseline_hash=digest(real.run_root/'measurements/baseline.json')
    result=real.collect({});collected=Path(result['data']);collected_before=digest(collected)
    review=dict(allowed_training_data=True,loader_connection_supported=True,sources_supported=True)
    data_versions.register(real.output,real.repo,dict(id='collected',dataset_ref=str(collected.relative_to(real.output)),
        source_refs=[],reason='CPU fixture native production',asset=dict(kind='synthetic',producer_evidence_id=result['evidence_id'])),review)
    processed=real.run_stage('prepare_data',settings={})
    assert processed['status']=='completed'
    real._prepared=True  # this fixture has explicitly completed the once-per-run native stage
    selected=Path(processed['output_directory'])/'selected.json'
    data_versions.register(real.output,real.repo,dict(id='selected',dataset_ref=str(selected.relative_to(real.output)),
        source_refs=[],reason='reversible whole-episode selection',asset=dict(kind='processed',parents=['collected'],
        producer_evidence_id=processed['evidence_id'])),review)
    real.library.add(Idea(label='filtered data',granularity='param',risk='low',mechanism='whole-episode data selection',
        change=dict(dataset_root='data-version:selected'),status=CLEARED,why='verify actual native pipeline'))
    report=real.run(rounds=1,yield_after_action=True,selected_idea_label='filtered data',main_controller_owns_selection=True)
    measured=json.loads((real.run_root/'measurements/round_1.json').read_text())
    assert measured['ok'] and measured['metric_value']==0.8
    assert measured['data_consumption']['exposure'][0]['source_samples']=={'collected':4}
    assert digest(original)==raw_before and digest(collected)==collected_before
    assert digest(real.run_root/'measurements/baseline.json')==baseline_hash
    view=run_record.build_report_view(real.output,real.run_id)
    document=recorder.render(real.output,recorder.make_snapshot(real.output,view,{'actions':[],'plan':{}}),{})
    assert '可逆处理结果' in document and 'collected：4' in document
    assert '-data-episodes.svg' in document and '不能相加为新增数据量' in document
    assert report['performance_claims']['global_sota_verified'] is False
