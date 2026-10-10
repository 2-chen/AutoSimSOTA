from collections import namedtuple

import pytest

from autosim.research import storage_guard, scheduling


@pytest.mark.parametrize('free,projected,expected', [(100,200,False),(300,200,True),(300,None,True)])
def test_storage_projection_and_floor_are_checked(tmp_path, monkeypatch, free, projected, expected):
    monkeypatch.setenv('AUTOSIM_MIN_FREE_STORAGE_BYTES','128')
    usage = namedtuple('usage','total used free')(1000,1000-free,free)
    monkeypatch.setattr(storage_guard.shutil,'disk_usage',lambda path: usage)
    result = storage_guard.preflight(tmp_path/'absent/output', stage='train', required_bytes=projected)
    assert result['sufficient'] is expected
    assert result['required_bytes'] == max(128,projected or 0)


@pytest.mark.parametrize('value', [True,0,-1,'2048',1.5])
def test_projection_rejects_invalid_counts(tmp_path, value):
    with pytest.raises(ValueError):
        storage_guard.preflight(tmp_path,stage='train',required_bytes=value)
    with pytest.raises(ValueError):
        scheduling.resource_profile({'storage_required_bytes':value})


def test_agent_native_job_projection_survives_resource_validation():
    assert scheduling.resource_profile({'storage_required_bytes':123456})['storage_required_bytes'] == 123456


def test_low_storage_blocks_before_native_launch(tmp_path, monkeypatch):
    from tests.test_derived_research import research
    real = research(tmp_path)
    monkeypatch.setenv('AUTOSIM_MIN_FREE_STORAGE_BYTES','1')
    usage = namedtuple('usage','total used free')(1000,1000,0)
    monkeypatch.setattr(storage_guard.shutil,'disk_usage',lambda path: usage)
    monkeypatch.setattr(real,'_run_stage',lambda *a,**k: pytest.fail('native command must not launch'))
    result = real.run_stage('train',settings={})
    assert result['termination_reason'] == 'storage_insufficient'
    assert result['ran'] is False
    assert result['resource_block']['evidence']['free_bytes'] == 0
