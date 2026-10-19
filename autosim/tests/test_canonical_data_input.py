import json
import subprocess
from types import SimpleNamespace

import pytest

from autosim.research import candidate_settings, data_versions, harness_repair
from autosim.research.adapter_protocol import Axis, OptimizationSpace
from tests.test_derived_research import research


def test_registered_canonical_slot_reaches_a_differently_named_native_flag(tmp_path):
    real = research(tmp_path)
    script = real.repo/'trainer.py'
    script.write_text('import argparse,json\np=argparse.ArgumentParser()\n'
        'p.add_argument("--training-root",required=True)\na=p.parse_args()\n'
        'print(json.dumps(json.load(open(a.training_root))))\n')
    source = ('def stage_argv_train(i):\n'
        '    return [i["python"], i["repo"] + "/trainer.py", "--training-root", i["dataset"]]\n')
    real.sources['train'] = source
    data=real.output/'examples.json';data.write_text('[1,2,3]')
    data_versions.register(real.output, real.repo, {'id':'extra','dataset_ref':'examples.json',
        'source_refs':[],'reason':'CPU input-routing fixture'},
        {'allowed_training_data':True,'loader_connection_supported':True,'sources_supported':True})
    selected = candidate_settings.apply(real, {}, {'dataset':'data-version:extra'})
    resolved, bindings = data_versions.resolve(real.output,real.repo,selected)
    inputs=real._inputs('train',settings=resolved)
    namespace={};exec(source,namespace)
    argv=namespace['stage_argv_train'](inputs)
    assert argv[-2:]==['--training-root',str(data)]
    assert 'dataset' not in inputs['settings']  # no duplicate unrecognized native flag
    result=subprocess.run(argv,check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)==[1,2,3]
    assert bindings[0]['setting']=='dataset' and bindings[0]['id']=='extra'
    assert 'consumer_review' not in bindings[0]


@pytest.mark.parametrize('key,value,source', [
    ('dataset','/unreviewed/path','return i["dataset"]'),
    ('dataset','data-version:../escape','return i["dataset"]'),
    ('dataset','data-version:extra','return ["native_default"]'),
    ('dataset_root','data-version:extra','return i["dataset"]'),
    ('seed',1,'return i["dataset"]')])
def test_canonical_admission_is_not_an_alias_or_arbitrary_parameter_grant(key,value,source):
    real=SimpleNamespace(space=OptimizationSpace(),sources={'train':'def train(i):\n    '+source})
    with pytest.raises(ValueError):candidate_settings.apply(real,{}, {key:value})


def test_explicit_axis_and_frozen_protocol_restrictions_still_apply():
    real=SimpleNamespace(space=OptimizationSpace(training=(Axis('dataset','choice','restricted',
        values=('data-version:allowed',)),)),sources={'train':'def train(i):\n    return i["dataset"]'},
        _comparison_protocol_violation=lambda *a,**k:'protected data split cannot change')
    with pytest.raises(ValueError,match='declared/source-backed'):
        candidate_settings.apply(real,{}, {'dataset':'data-version:other'})
    with pytest.raises(ValueError,match='frozen evaluation protocol'):
        candidate_settings.apply(real,{}, {'dataset':'data-version:allowed'})


def test_fix_can_read_input_merge_contract_but_not_patch_or_hot_load_it(tmp_path):
    result=harness_repair.read_source(tmp_path,{'path':'research/candidate_settings.py','limit':70})
    assert 'canonical_dataset_input' in result['excerpt']
    assert result['candidate_edits_allowed'] is False
    assert result['live_activation_allowed'] is False
