"""Real CPU producer -> sample reads -> trained bytes -> evaluator, not a robot score."""
import json
from pathlib import Path

from autosim.research import data_versions
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
 print('AUTOSIM_TRAIN_DATA_CONSUMPTION '+json.dumps(dict(data_version_id='targeted',setting='dataset_root',identity_field='content_sha256',identity=hashlib.sha256(raw).hexdigest(),samples_read=len(samples))))
(p/'models').mkdir();(p/'models/model.pth').write_text(str(len(samples)))
print('global_step=2')
'''
    evaluator="import pathlib,sys; n=int(pathlib.Path(sys.argv[1]).read_text()); print('succ: 0.80' if n>=3 else 'succ: 0.20')"
    collector="import pathlib,sys; p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True);(p/'data.json').write_text('[1,2,3]')"
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
             source_refs=[],reason='synthetic train-only coverage extension'),
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
    assert digest(real.run_root/'measurements/baseline.json')==before
    assert report['performance_claims']['global_sota_verified'] is False
