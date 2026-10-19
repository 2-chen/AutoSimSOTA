"""Labels reach an unchanged command, without weakening data-byte verification."""
import json
import os
import subprocess
import sys

import pytest

from autosim.research.declarative_backend import staging_environment
from autosim.research.data_versions import native_selection_payload


def test_registered_labels_reach_unchanged_native_command(tmp_path):
    path = tmp_path / 'data with spaces.json'; path.write_text('[1,2,3]')
    labels = [{'id':'selected_v1','setting':'dataset','path':str(path)}]
    inputs = {'repo':str(tmp_path),'output':str(tmp_path/'output'),'python':sys.executable,
              'dataset':str(path),'selected_training_data':labels}
    # An old argv builder does not need editing, so its frozen source is unchanged.
    namespace = {}
    source = 'def train(i): return [i["python"], i["repo"] + "/reader.py", i["dataset"]]'
    exec(source,namespace)
    before = namespace['train'](inputs)
    reader = tmp_path / 'reader.py'
    reader.write_text('import json,os,sys,hashlib\n'
        'labels=json.loads(os.environ["AUTOSIM_TRAIN_SELECTION_JSON"])\n'
        'data=open(sys.argv[1],"rb").read()\n'
        'assert labels[0]["path"]==sys.argv[1]\n'
        'print(json.dumps({"labels":labels,"actual_sha256":hashlib.sha256(data).hexdigest()}))\n')
    env = staging_environment(os.environ.copy(),inputs)
    result = json.loads(subprocess.run(before,env=env,check=True,capture_output=True,text=True).stdout)
    assert namespace['train'](inputs) == before
    assert result['labels'] == labels and result['actual_sha256'] not in env['AUTOSIM_TRAIN_SELECTION_JSON']
    assert set(result['labels'][0]) == {'id','setting','path'}


def test_absent_selection_clears_ambient_labels_without_mutating_caller():
    original = {'AUTOSIM_TRAIN_SELECTION_JSON':'{"forged":"binding"}'}
    result = staging_environment(original,{})
    assert json.loads(result['AUTOSIM_TRAIN_SELECTION_JSON']) == []
    assert original['AUTOSIM_TRAIN_SELECTION_JSON'] == '{"forged":"binding"}'


def test_actual_formal_native_process_gets_labels_not_only_staging(tmp_path, monkeypatch):
    from tests.test_derived_research import research
    real = research(tmp_path)
    data = tmp_path/'data.json'; data.write_text('[1,2,3]')
    labels = [{'id':'selected','setting':'dataset','path':str(data)}]
    original_builder = real.backend._functions['train']
    def command(i):
        argv = original_builder(i)
        argv[2] += '; import os; print("NATIVE_SELECTION="+os.environ["AUTOSIM_TRAIN_SELECTION_JSON"])'
        return argv
    real.backend._functions['train'] = command
    monkeypatch.setenv('AUTOSIM_TRAIN_SELECTION_JSON','ambient_must_not_win')
    result = real.run_stage('train',settings={},selected_training_data=labels)
    assert result['returncode'] == 0
    payload = next(line.split('=',1)[1] for line in result['said'].splitlines()
                   if line.startswith('NATIVE_SELECTION='))
    assert json.loads(payload) == labels
    result = real.run_stage('train',settings={})
    assert result['returncode'] == 0 and 'NATIVE_SELECTION=[]' in result['said']


@pytest.mark.parametrize('labels', [None, {}, [{'id':'../escape','setting':'dataset','path':'/data'}],
    [{'id':'good','setting':'dataset','path':'relative'}],
    [{'id':'good','setting':'dataset','path':'/data','identity':'expected_hash'}],
    [{'id':'good','setting':'dataset','path':'/data','samples_read':10}],
    [{'id':'good','setting':'dataset','path':'/data'}]*2])
def test_native_label_channel_rejects_malformed_or_expected_witness_payload(labels):
    with pytest.raises(ValueError):
        native_selection_payload({'selected_training_data':labels})


def test_native_label_channel_is_bounded_and_empty_without_binding():
    assert json.loads(native_selection_payload({})) == []
    with pytest.raises(ValueError):
        native_selection_payload({'selected_training_data':[
            {'id':str(i),'setting':'slot'+str(i),'path':'/'+('a'*4095)} for i in range(20)]})
