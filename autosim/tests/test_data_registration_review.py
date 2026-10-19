import pytest

from autosim.research.common import atomic_json,digest
from autosim.research.data_registration_review import latest


def card(root,ident='fresh',**changes):
    path=root/'data_versions'/f'{ident}.json'
    value={'id':ident,'status':'reviewed_not_consumed',
        'review':{'allowed_training_data':True,'loader_connection_supported':True,
                  'sources_supported':True,'why':'Format is valid, but expert-path save does not check task success.'},
        'asset':{'profile':{'manifest':{'unknown_supervision':20,'unknown_source_families':20}}}}
    value.update(changes)
    atomic_json(path,value)
    return path,{'id':ident,'record_sha256':digest(path)}


def test_successful_registration_keeps_negative_quality_review_and_unknowns(tmp_path):
    path,entry=card(tmp_path)
    before=digest(path)
    result=latest(tmp_path,[entry])
    assert result['review']['allowed_training_data'] is True
    assert 'does not check task success' in result['review']['why']
    assert result['manifest_unverified']['unknown_supervision'] == 20
    assert 'not a quality/success certificate' in result['scope']
    assert digest(path) == before


@pytest.mark.parametrize('fault',['changed','symlink','unknown_id','missing'])
def test_unverified_or_escaping_card_is_not_presented_as_a_review(tmp_path,fault):
    path,entry=card(tmp_path)
    if fault=='changed':path.write_text('{}')
    elif fault=='symlink':
        moved=tmp_path/'other.json';path.rename(moved);path.symlink_to(moved)
    elif fault=='unknown_id':entry['id']='../other'
    else:path.unlink()
    assert latest(tmp_path,[entry]) is None


def test_main_receives_full_current_registration_review(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research import data_versions
    real=preparation(tmp_path)
    _,entry=card(real.output)
    monkeypatch.setattr(data_versions,'catalog',lambda _: [entry])
    result=real._decision_facts()['latest_data_registration_review']
    assert result['data_version_id']=='fresh'
    assert 'does not check task success' in result['review']['why']
