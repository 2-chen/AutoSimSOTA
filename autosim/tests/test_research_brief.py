import pytest

from autosim.research.common import atomic_json, read_json
from autosim.research.research_brief import persist, view
from tests.test_prepare import preparation


def test_study_intent_frozen_separately_from_native_settings(tmp_path):
    output = tmp_path/'out'
    source = tmp_path/'brief.md'
    source.write_text('失败驱动的数据获取与筛选。')
    persist(output, source)
    assert view(output)['status'] == 'frozen'
    persist(output, None)
    persist(output, source)
    real = preparation(tmp_path)
    assert real._research_brief_view()['text'] == source.read_text()
    assert 'research_brief' not in real.base_settings
    source.write_text('改变研究目标')
    with pytest.raises(ValueError, match='frozen'):
        persist(output, source)


@pytest.mark.parametrize('kind', ['empty', 'too_long', 'symlink', 'retrofit', 'tamper'])
def test_brief_rejects_unsafe_or_changed_intent(tmp_path, kind):
    output = tmp_path/'out'
    source = tmp_path/'brief.md'
    source.write_text('合法研究目标')
    if kind == 'empty': source.write_text(' ')
    elif kind == 'too_long': source.write_text('x'*6001)
    elif kind == 'symlink':
        link = tmp_path/'link.md'; link.symlink_to(source); source = link
    elif kind == 'retrofit': atomic_json(output/'budget.json', {})
    else:
        persist(output, source)
        row = read_json(output/'research_brief.json'); row['text'] = '篡改'
        atomic_json(output/'research_brief.json', row)
    with pytest.raises(ValueError):
        persist(output, source)
