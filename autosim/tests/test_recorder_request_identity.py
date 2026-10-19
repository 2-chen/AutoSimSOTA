"""Request handles avoid error-prone digest copying without weakening fact review."""
import json

import pytest

from autosim.research import recorder
from autosim.research.common import read_json
from tests.test_recorder import snapshot, Writer


def answer_for(held, revision):
    return {'event_revision': revision, 'sections': {
        key: {'text': '本次证据尚不足以判断结果。', 'evidence_ids': ['run']}
        for key in recorder.FIELDS}}


def test_short_handle_is_bound_to_both_material_event_and_exact_snapshot(tmp_path):
    held = snapshot(tmp_path)
    handle = recorder._event_handle(held)
    assert handle.startswith('report-') and len(handle) == 31
    assert recorder._validate(answer_for(held, handle), held)['sections']
    for key in ('event_revision', 'revision'):
        changed = {**held, key: 'different-' + held[key]}
        with pytest.raises(ValueError, match='event revision'):
            recorder._validate(answer_for(held, handle), changed)
    # Compatibility never accepts an incorrect, truncated or one-character typo hash.
    assert recorder._validate(answer_for(held, held['event_revision']), held)
    with pytest.raises(ValueError, match='event revision'):
        recorder._validate(answer_for(held, held['event_revision'] + 'a'), held)
    incomplete = {key: value for key, value in held.items() if key != 'revision'}
    assert recorder._validate(answer_for(held, held['event_revision']), incomplete)
    with pytest.raises(ValueError, match='event revision'):
        recorder._validate(answer_for(held, handle), incomplete)


def test_narration_receives_only_short_outer_handle_but_saves_executor_identity(tmp_path):
    held = snapshot(tmp_path); writer = Writer()
    recorded = recorder.narrate(tmp_path, held, client=writer)
    assert recorded['status'] == 'written'
    assert recorded['event_revision'] == held['event_revision']
    assert recorded['revision'] == held['revision']
    packets = [row[0] for row in writer.calls]
    writing = next(row for row in packets if row.get('phase') == 'write_report')
    assert writing['event_revision'] == recorder._event_handle(held)
    assert recorded['fact_review_contract_version'] == recorder.FACT_REVIEW_CONTRACT_VERSION
    assert any(row.get('proposed_report') for row in packets)  # Independent review is retained.
    stored = read_json(tmp_path / 'report/narratives' / (held['event_revision'] + '.json'))
    assert stored == recorded


def test_stale_handle_is_not_relabelled_as_current_or_saved_as_prose(tmp_path):
    held = snapshot(tmp_path); stale = {**held, 'revision': 'a different snapshot'}
    writer = Writer(); original = writer.chat_with_metadata
    def stale_reply(system, user, **kwargs):
        content, metadata = original(system, user, **kwargs)
        payload = json.loads(user)
        if payload.get('phase') in {'write_report', 'format_repair'}:
            reply = json.loads(content)
            reply['event_revision'] = recorder._event_handle(stale)
            return json.dumps(reply, ensure_ascii=False), metadata
        return content, metadata
    writer.chat_with_metadata = stale_reply
    assert recorder.narrate(tmp_path, held, client=writer) == {}
    assert not (tmp_path / 'report/narrative.json').exists()
    assert read_json(tmp_path / 'report/recorder_trigger.json')['status'] == 'failed'
