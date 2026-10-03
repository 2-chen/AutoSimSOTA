import json

import pytest

from autosim.research import invocation_drafts as drafts


def test_draft_survives_restart_but_requires_explicit_context_and_selection(tmp_path):
    base = {'stage': 'evaluate', 'entrypoint_sha256': 'abc', 'checkpoint': 'frozen'}
    row = {'environment': {'WRAPPER_PYTHON': '/run/env/bin/python', 'DISPLAY_MODE': 'chosen'}}
    saved = drafts.save(tmp_path, stage='evaluate', base=base, row=row,
                        reason='Agent read native loader failure', evidence_ref='evidence/native.json')
    assert not (tmp_path / 'derived_stages.json').exists()
    assert drafts.catalog(tmp_path)[0]['status'] == 'unverified'
    assert drafts.load(tmp_path, saved['ref'], stage='evaluate', base=base)['row'] == row
    assert drafts.load(tmp_path, saved['id'], stage='evaluate', base=base)['ref'] == saved['ref']
    with pytest.raises(ValueError, match='changed'):
        drafts.load(tmp_path, saved['ref'], stage='evaluate', base={**base, 'checkpoint': 'different'})
    with pytest.raises(ValueError, match='changed'):
        drafts.load(tmp_path, saved['id'], stage='evaluate', base={**base, 'checkpoint':'different'})
    with pytest.raises(ValueError):
        drafts.load(tmp_path, saved['ref'], stage='train', base=base)


def test_changed_draft_bytes_and_escaping_refs_are_rejected(tmp_path):
    saved = drafts.save(tmp_path, stage='train', base={}, row={}, reason='native evidence')
    path = tmp_path / saved['ref']
    record = json.loads(path.read_text())
    record['row'] = {'environment': {'UNREVIEWED': 'change'}}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='identity'):
        drafts.load(tmp_path, saved['ref'], stage='train', base={})
    with pytest.raises(ValueError, match='identity'):
        drafts.load(tmp_path, saved['id'], stage='train', base={})
    assert drafts.catalog(tmp_path) == []
    for ref in ('../foreign.json', '/tmp/foreign.json', 'invocation_drafts/../foreign.json'):
        with pytest.raises(ValueError):
            drafts.load(tmp_path, ref, stage='train')


def test_symlink_and_missing_rationale_fail_closed(tmp_path):
    with pytest.raises(ValueError, match='rationale'):
        drafts.save(tmp_path, stage='train', base={}, row={}, reason=' ')
    (tmp_path / 'invocation_drafts').symlink_to(tmp_path / 'other')
    assert drafts.catalog(tmp_path) == []
    with pytest.raises(ValueError, match='unsafe'):
        drafts.save(tmp_path, stage='train', base={}, row={}, reason='native evidence')
