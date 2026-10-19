"""A short view must not erase reader-vs-batch authority."""
import pytest
from autosim.research.common import atomic_json
from autosim.research.development_evidence import summaries


def measure(root, stage='sample_read', streams=24):
    exposure = {'data_version_id': 'registered', 'setting': 'dataset',
        'status': 'basic_read_only', 'samples_read': 8000}
    if stage is not None:
        exposure['consumption_stage'] = stage
    if streams is not None:
        exposure['reader_streams'] = streams
    atomic_json(root/'measurements/baseline.json', {'label': 'baseline', 'ok': True,
        'metric_value': 0.06, 'data_consumption': {'status': 'verified',
        'versions': ['registered'], 'exposure': [exposure]}})
    return summaries(root, _handoffs=[])[0]


@pytest.mark.parametrize('stage,streams', [('training_batch', 1), ('sample_read', 24)])
def test_short_view_preserves_actual_stage_not_generic_status(tmp_path, stage, streams):
    row = measure(tmp_path, stage, streams)
    exposure = row['data_exposure'][0]
    assert exposure['consumption_stage'] == stage
    assert exposure['setting'] == 'dataset'
    assert exposure['reader_streams'] == streams
    assert exposure['samples_read'] == 8000
    assert 'training_batch' in row['data_exposure_meaning']
    assert 'unique' in row['data_exposure_meaning']


def test_stage_change_refreshes_milestone_digest(tmp_path):
    a = measure(tmp_path, 'sample_read', 1)
    b = measure(tmp_path, 'training_batch', 1)
    assert a['data_exposure_digest'] != b['data_exposure_digest']


@pytest.mark.parametrize('stage,streams', [(None, None), ('unproven', True)])
def test_missing_or_invalid_stage_never_becomes_batch(tmp_path, stage, streams):
    row = measure(tmp_path, stage, streams)
    assert row['data_exposure'][0]['consumption_stage'] == 'unknown'
    assert 'reader_streams' not in row['data_exposure'][0]
