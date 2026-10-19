import hashlib
import json

import pytest

from autosim.research.data_versions import MARKER
from autosim.research.development_evidence import reader_issue_summary


def row(**change):
    return {**dict(data_version_id='registered',setting='dataset',identity_field='sha256',
        identity='a'*64,reader_id='worker-1',consumption_stage='training_batch',samples_read=8),**change}


def binding():
    return dict(id='registered',setting='dataset',identity_field='sha256',identity='a'*64)


def inspect(tmp_path,lines):
    path=tmp_path/'sealed.log';path.write_text('\n'.join(lines)+'\n')
    raw=path.read_bytes();value=reader_issue_summary(path,[binding()],hashlib.sha256(raw).hexdigest())
    assert path.read_bytes()==raw
    return value


def test_valid_batch_is_observed_not_promoted_to_consumption(tmp_path):
    value=inspect(tmp_path,[MARKER+json.dumps(row())])
    assert value['events_by_stage']=={'training_batch':1}
    assert value['issue_counts']=={}
    assert value['status']=='bounded_schema_diagnostics'
    assert 'not the strict validator' in value['authority']
    assert 'zero listed issues does not establish validity' in value['authority']


def test_bad_framing_does_not_hide_other_parseable_schema_gaps(tmp_path):
    old=row(data_version_id='private/data/legacy-root');old.pop('reader_id');old.pop('consumption_stage')
    value=inspect(tmp_path,[MARKER+json.dumps(old),MARKER+json.dumps(row())+json.dumps(row()),
        MARKER+json.dumps(row())])
    assert value['marker_lines']==3 and value['parsed_object_records']==2
    assert value['events_by_stage']=={'unknown':1,'training_batch':1}
    assert value['issue_counts']=={'missing_or_unknown_consumption_stage':1,
        'missing_or_unbounded_reader_id':1,'unselected_or_missing_version_id':1,'malformed_json_lines':1}
    assert 'private/data' not in json.dumps(value)
    assert 'Do not split/repair old log rows' in value['authority']


@pytest.mark.parametrize('change,issue',[
    ({'setting':'wrong'},'selected_input_slot_mismatch'),
    ({'identity':'wrong'},'selected_bytes_identity_mismatch'),
    ({'identity_field':'wrong'},'selected_bytes_identity_mismatch'),
    ({'reader_id':''},'missing_or_unbounded_reader_id'),
    ({'consumption_stage':{}},'missing_or_unknown_consumption_stage'),
    ({'data_version_id':[]},'unselected_or_missing_version_id'),
    ({'samples_read':True},'missing_or_invalid_sample_counter'),
    ({'samples_read':-1},'missing_or_invalid_sample_counter')])
def test_independent_omissions_are_bounded_observations(tmp_path,change,issue):
    assert inspect(tmp_path,[MARKER+json.dumps(row(**change))])['issue_counts'][issue]==1


def test_non_object_and_noise_are_not_fake_witnesses(tmp_path):
    result=inspect(tmp_path,['ordinary log',MARKER+'[]',MARKER+'not JSON'])
    assert result['parsed_object_records']==0
    assert result['issue_counts']=={'non_object_records':1,'malformed_json_lines':1}


def test_changed_or_symlink_log_not_summarized(tmp_path):
    path=tmp_path/'log';path.write_text(MARKER+json.dumps(row()))
    assert reader_issue_summary(path,[binding()],'0'*64)['status']=='unavailable'
    link=tmp_path/'link';link.symlink_to(path)
    assert reader_issue_summary(link,[binding()],hashlib.sha256(path.read_bytes()).hexdigest())['status']=='unavailable'
