import json

import pytest

from autosim.research.development_evidence import training_selection_transport


def test_empty_selection_is_not_unknown_or_no_training_data():
    result = training_selection_transport({'environment':{'AUTOSIM_TRAIN_SELECTION_JSON':'[]'}})
    assert result['status']=='no_registered_selection'
    assert result['bindings']==[] and result['binding_count']==0
    assert 'native default data' in result['guidance']
    assert 'not opened bytes' in result['authority']


def test_registered_selection_projection_is_path_free_not_a_consumption_certificate():
    receipt={'environment':{'AUTOSIM_TRAIN_SELECTION_JSON':json.dumps([
        {'id':'acquired_v1','setting':'dataset','path':'/private/simulation/data'}]),
        'PRIVATE_TOKEN':'never-ship-this-value'}}
    before=json.dumps(receipt,sort_keys=True)
    result=training_selection_transport(receipt)
    assert result['status']=='transported'
    assert result['bindings']==[{'data_version_id':'acquired_v1','setting':'dataset'}]
    assert 'private' not in json.dumps(result) and 'never-ship' not in json.dumps(result)
    assert json.dumps(receipt,sort_keys=True)==before


@pytest.mark.parametrize('receipt',[{}, {'environment':None}, {'environment':{}}])
def test_absent_selection_remains_unknown(receipt):
    assert training_selection_transport(receipt)['status']=='unknown'


@pytest.mark.parametrize('raw',[None,'null','{}','[null]','not-json',' '*16385,
    json.dumps([{'id':'ok','setting':'dataset','path':'/private'}]*65),
    json.dumps([{'id':'ok','setting':'../private','path':'/private'}]),
    json.dumps([{'id':'ok','setting':'dataset'}]),
    json.dumps([{'id':'sk-123456789012345678901234567890','setting':'dataset','path':'/private'}])])
def test_malformed_or_sensitive_metadata_never_becomes_an_empty_selection(raw):
    result=training_selection_transport({'environment':{'AUTOSIM_TRAIN_SELECTION_JSON':raw}})
    assert result['status']=='invalid_transport_metadata'
    assert 'bindings' not in result and '123456789' not in json.dumps(result)


def test_observation_and_prompt_keep_parameter_continuity_explicit():
    from autosim.research.prompt_library import load_prompt
    prompt=' '.join(load_prompt('scheduler').split())
    assert 'do not inherit the previous experiment' in prompt
    assert 'no_registered_selection' in prompt
    assert 'not opened-byte or batch verification' in prompt
    assert 'slot does not require inventing a dataset optimization axis' in prompt
    assert 'Registration alone and a previous round' in prompt
