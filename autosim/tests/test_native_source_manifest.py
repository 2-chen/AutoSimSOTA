"""Other native formats keep their reader: no benchmark switch or universal converter."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autosim.research import data_health, data_lineage, data_versions
from autosim.research.common import atomic_json, digest
from autosim.research.evidence_store import capture_attempt_evidence
from tests.test_data_versions import registered
from tests.test_data_lineage import producer, REVIEW

INSPECTOR = '''import hashlib, json, math, os
from pathlib import Path
root=Path(os.environ['AUTOSIM_DATA_ROOT'])
pairs=[]; rows=[]
for path in sorted(root.glob('*.json')):
    native=json.loads(path.read_text())
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    pairs.append((path.name,sha))
    rows.append({'id':path.stem,'committed':True,'files':{path.name:sha},'content_sha256':sha,
        'format_valid':isinstance(native['action'],list) and all(math.isfinite(x) for x in native['action']),
        'source_family':native.get('source_family'),'source_instance':native.get('source_instance'),
        'split':native.get('split'),'supervision':native.get('supervision'),'conditions':{}})
identity=hashlib.sha256(json.dumps(pairs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
print(json.dumps({'scope':'full','format':'native JSON episode fixture','counts':{'episodes':len(rows),'samples':len(rows)},
    'issues':[],'video_decode_scope':'not_applicable','episode_manifest':{'schema_version':1,
        'identity_field':'sha256','identity':identity,'scope':'full','episodes':rows}}))
'''


def observation(tmp_path):
    repo,output,_,_=registered(tmp_path)
    root=output/'data';root.mkdir()
    for i in range(2):(root/f'{i}.json').write_text(json.dumps({'action':[i]}))
    (repo/'native_check.py').write_text(INSPECTOR)
    request={'format':'source_manifest','root':'data','inspector_ref':'native_check.py'}
    code=data_health.native_profile_code(request,workspace=repo)
    # Plain CPU source invocation, simulating the readonly executor result; no robot result.
    env={**os.environ,'AUTOSIM_DIAGNOSTIC_DATA':str(root)}
    code=data_health.native_profile_code({**request,'root':'@producer','producer':{'evidence_id':'a'*32,'artifact_index':0}},workspace=repo)
    attempt=subprocess.run([sys.executable,'-c',code],cwd=repo,env=env,text=True,capture_output=True,check=True)
    ident='b'*32;ref=f'native_diagnostics/{ident}.json';log=output/f'native_diagnostics/{ident}.log'
    atomic_json(output/ref,dict(data_profile_generator='autosim_native_source_manifest_v1',
        data_profile_inspector=data_health.inspector_source(repo,'native_check.py'),data_profile=json.loads(attempt.stdout),
        gpu_access=False,repository_source_stable=True,data_profile_request=request))
    log.write_text(attempt.stdout)
    capture_attempt_evidence(output,attempt_id=ident,log=log,receipt_ref=ref,status='completed',returncode=0,
                            termination_reason='cpu_native_diagnostic')
    parent,_=producer(output,root,status='timed_out')
    spec=dict(id='native',dataset_ref='data',source_refs=[],reason='partial native-format fixture',
              asset={'kind':'collected','producer_evidence_id':parent,'manifest_evidence_id':ident})
    return repo,output,root,spec


def test_generic_native_manifest_requires_reader_source_review_and_preserves_partial(tmp_path):
    repo,output,root,spec=observation(tmp_path)
    with pytest.raises(ValueError,match='independent observation'):
        data_versions.register(output,repo,spec,REVIEW)
    before=digest(output/'producer'/('a'*32)/'receipt.json')
    card=data_versions.register(output,repo,spec,{**REVIEW,'manifest_observations_supported':True})
    assert card['asset']['profile']['format']=='native JSON episode fixture'
    assert card['asset']['profile']['counts']['samples']==2
    assert card['asset']['profile']['manifest']['unknown_supervision']==2
    assert card['asset']['producer']['partial_output'] is True
    assert digest(output/'producer'/('a'*32)/'receipt.json')==before
    _,bound=data_versions.resolve(output,repo,{'data':'data-version:native'})
    assert bound[0]['id']=='native'
    (repo/'native_check.py').write_text(INSPECTOR+'# changed reader\n')
    with pytest.raises(ValueError):data_versions.resolve(output,repo,{'data':'data-version:native'})


@pytest.mark.parametrize('ref',['../outside.py','/tmp/outside.py','native_check.sh'])
def test_generic_inspection_does_not_grant_external_executables_or_paths(tmp_path,ref):
    with pytest.raises((OSError,ValueError)):
        data_health.native_profile_code({'format':'source_manifest','root':'data','inspector_ref':ref},workspace=tmp_path)


def test_real_native_source_manifest_uses_owned_readonly_artifact_not_host_root(tmp_path):
    from autosim.research.agent_runtime import _McpExecutor
    from autosim.research.native_context import publish_context
    repo,output,_,spec=observation(tmp_path)
    repo=output/'checkout';repo.mkdir();(repo/'native_check.py').write_text(INSPECTOR)
    atomic_json(output/'agent_fixture.json',{'schema_version':1,'workspace':str(repo)})
    atomic_json(output/'run_state.json',{'run_id':'derived'})
    subprocess.run(['/usr/bin/python3','-m','venv','--copies','--without-pip',str(output/'env')],check=True)
    publish_context(output,repo,output/'env/bin/python',{})
    executor=_McpExecutor(output=output,workspace=repo,allow_commands=False,allow_native=True)
    result=executor.native_probe({'purpose':'actual CPU native reader fixture','data_profile_request':
        {'format':'source_manifest','root':'@producer','producer':{'evidence_id':'a'*32,'artifact_index':0},
         'inspector_ref':'native_check.py'}})
    assert result['status']=='completed' and result['diagnostic_data_binding_stable'] is True
    assert result['data_profile']['episode_manifest_summary']['episodes']==2
    assert 'code' not in result['data_profile_inspector']
    spec['asset']['manifest_evidence_id']=result['evidence_id']
    card=data_versions.register(output,repo,spec,{**REVIEW,'manifest_observations_supported':True})
    assert card['asset']['profile']['counts']['samples']==2
