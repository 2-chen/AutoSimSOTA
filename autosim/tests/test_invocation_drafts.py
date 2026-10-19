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


def test_explicit_checkpoint_rebind_preserves_draft_not_execution_authority(tmp_path):
    base = {'stage':'evaluate', 'entrypoint_sha256':'code', 'interpreter':'runtime',
            'task':'frozen', 'checkpoint':{'path':'old', 'evidence_ref':'owned-old'}}
    row = {'environment':{'WRAPPER_PYTHON':'{verified_interpreter}', 'DISPLAY_MODE':'chosen'}}
    saved = drafts.save(tmp_path, stage='evaluate', base=base, row=row, reason='native caller handoff')
    current = {**base, 'checkpoint':{'path':'new', 'evidence_ref':'owned-new'}}
    with pytest.raises(ValueError, match='changed'):
        drafts.load(tmp_path, saved['ref'], stage='evaluate', base=current)
    loaded = drafts.load(tmp_path, saved['ref'], stage='evaluate', base=current, rebind_inputs=['checkpoint'])
    assert loaded['row'] == row and loaded['status'] == 'unverified'
    assert loaded['rebound_inputs'] == ['checkpoint']
    assert drafts.catalog(tmp_path)[0]['rebindable_inputs'] == ['checkpoint']
    assert not (tmp_path/'derived_stages.json').exists()


@pytest.mark.parametrize('key', ['entrypoint_sha256', 'interpreter', 'task', 'row'])
def test_checkpoint_rebind_never_waives_other_context(tmp_path, key):
    base = {'checkpoint':'old', 'entrypoint_sha256':'code', 'interpreter':'runtime',
            'task':'frozen', 'row':{'entrypoint':'eval.py'}}
    saved = drafts.save(tmp_path, stage='evaluate', base=base, row={}, reason='native failure')
    with pytest.raises(drafts.DraftContextChanged, match='changed') as error:
        drafts.load(tmp_path, saved['ref'], stage='evaluate',
                    base={**base, 'checkpoint':'new', key:'changed'}, rebind_inputs=['checkpoint'])
    assert error.value.changed_fields==sorted(['checkpoint',key])
    assert not error.value.checkpoint_only


@pytest.mark.parametrize('keys', [['interpreter'], ['checkpoint','checkpoint'], 'checkpoint', [{}]])
def test_rebind_cannot_expand_requested_context(tmp_path, keys):
    saved = drafts.save(tmp_path, stage='evaluate', base={}, row={}, reason='failure')
    with pytest.raises(ValueError, match='rebind'):
        drafts.load(tmp_path, saved['ref'], stage='evaluate', base={}, rebind_inputs=keys)


def test_legacy_opaque_draft_cannot_be_rebound(tmp_path):
    from autosim.research.common import atomic_json, object_digest
    payload = {'schema_version':1, 'status':'unverified', 'stage':'evaluate',
               'base_sha256':object_digest({'checkpoint':'old'}), 'row':{}}
    identity = object_digest(payload)
    ref = 'invocation_drafts/'+identity+'.json'
    atomic_json(tmp_path/ref, {'id':identity, **payload})
    assert drafts.catalog(tmp_path)[0]['rebindable_inputs'] == []
    with pytest.raises(ValueError, match='changed'):
        drafts.load(tmp_path, ref, stage='evaluate', base={'checkpoint':'new'}, rebind_inputs=['checkpoint'])


def test_symlink_and_missing_rationale_fail_closed(tmp_path):
    with pytest.raises(ValueError, match='rationale'):
        drafts.save(tmp_path, stage='train', base={}, row={}, reason=' ')
    (tmp_path / 'invocation_drafts').symlink_to(tmp_path / 'other')
    assert drafts.catalog(tmp_path) == []
    with pytest.raises(ValueError, match='unsafe'):
        drafts.save(tmp_path, stage='train', base={}, row={}, reason='native evidence')
