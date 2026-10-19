"""Current read/proposal facts cannot be filtered out of prose verification."""
from copy import deepcopy
import json

import pytest

from autosim.research import recorder, harness_repair
from autosim.research.common import atomic_json
from tests.test_recorder import snapshot, Writer


@pytest.mark.parametrize('current_count', [0, 3])
def test_historical_proposal_counts_do_not_describe_current_read_permission(
        tmp_path, monkeypatch, current_count):
    monkeypatch.setattr('autosim.research.runtime_freshness.controller_identity', lambda: 'current')
    for i in range(3):
        atomic_json(tmp_path / f'harness_repairs/old-{i}/proposal.json', {
            'id': f'old-{i}', 'runtime_identity': 'old', 'status': 'refused'})
    for i in range(current_count):
        atomic_json(tmp_path / f'harness_repairs/new-{i}/proposal.json', {
            'id': f'new-{i}', 'runtime_identity': 'current', 'status': 'proposed'})
    row = harness_repair.status(tmp_path)
    assert row['proposal_count'] == 3 + current_count
    assert row['can_propose'] is (current_count < 3)
    assert row['proposal_admission']['remaining'] == 3 - current_count
    assert row['source_access']['independent_of_proposal_quota'] is True
    assert row['source_access']['requires_native_process'] is False
    assert row['automatic_python_activation'] is False
    if not current_count:
        assert 'remain available' in row['admission_guidance']
    else:
        assert 'quota is exhausted' in row['admission_guidance']
    assert 'Edit/activation denial is not read denial' in row['admission_guidance']


def test_fact_reviewer_always_receives_current_access_even_if_writer_omits_it(tmp_path):
    held = snapshot(tmp_path)
    held['harness_repair'] = harness_repair.status(tmp_path)
    writer = Writer()
    result = recorder.narrate(tmp_path, held, client=writer)
    assert result['status'] == 'written'
    packet = next(p for p, _ in writer.calls if p.get('phase') == 'review_report')
    assert packet['evidence']['harness_repair']['can_propose'] is True
    assert packet['evidence']['harness_repair']['source_access']['independent_of_proposal_quota']
    assert 'harness_repair' in packet['allowed_evidence_ids']
    # Giving the reviewer facts does not invent citations in the writer's prose.
    assert all(row['evidence_ids'] == ['run'] for row in result['sections'].values())


def test_read_scope_change_invalidates_paid_prose_once_not_each_heartbeat(tmp_path):
    held = snapshot(tmp_path); writer = Writer()
    held['harness_repair'] = {'can_propose': False, 'source_catalog': []}
    recorder.narrate(tmp_path, held, client=writer)
    changed = deepcopy(held); changed['event_revision'] = 'new-source-permission'
    changed['harness_repair'] = harness_repair.status(tmp_path)
    recorder.narrate(tmp_path, changed, client=writer)
    assert len(writer.calls) == 5  # Same writing skills, new prose and independent review.
    changed['event_revision'] = 'heartbeat-only'
    recorder.narrate(tmp_path, changed, client=writer)
    assert len(writer.calls) == 5


def test_plan_only_data_prose_still_has_actual_data_facts_in_review(tmp_path):
    held = snapshot(tmp_path)
    held['plan'] = {'accepted_episode_budget': 20}
    held['data_versions'] = [{'id': 'registered', 'count_status': 'unknown'}]
    writer = Writer(); original = writer.chat_with_metadata
    def plan_only(system, user, **kwargs):
        packet = json.loads(user); content, metadata = original(system, user, **kwargs)
        if packet.get('phase') in {'write_report', 'format_repair'}:
            answer = json.loads(content)
            answer['sections']['purpose'] = {'text': '先按计划核对数据。', 'evidence_ids': ['plan']}
            return json.dumps(answer, ensure_ascii=False), metadata
        return content, metadata
    writer.chat_with_metadata = plan_only
    result = recorder.narrate(tmp_path, held, client=writer)
    assert result['status'] == 'written'
    request = next(p for p, _ in writer.calls if p.get('phase') == 'review_report')
    assert request['evidence']['data_versions'][0]['count_status'] == 'unknown'
    assert 'data_campaigns' in request['evidence']


def test_stale_closed_route_is_corrected_by_independent_review_not_auto_approved(tmp_path):
    held = snapshot(tmp_path); held['harness_repair'] = harness_repair.status(tmp_path)
    class Reviewed(Writer):
        def chat_with_metadata(self, system, user, **kwargs):
            packet = json.loads(user)
            content, metadata = super().chat_with_metadata(system, user, **kwargs)
            if packet.get('phase') == 'review_report':
                current = packet['evidence']['harness_repair']
                assert current['can_propose'] and current['source_access']['independent_of_proposal_quota']
                text = packet['proposed_report']['sections']['blocker']['text']
                if '只能等待操作者' in text:
                    return json.dumps({'verdict': 'unsupported', 'issues': [{
                        'section': 'blocker', 'reason': '当前读取目录和提案额度不能被历史拒绝覆盖。',
                        'evidence_ids': ['harness_repair']}]}), metadata
                return content, metadata
            if packet.get('phase') in {'write_report', 'format_repair'}:
                answer = json.loads(content)
                answer['sections']['blocker'] = {
                    'text': ('消费连接尚未验证，旧拒绝是否仍然存在需要复验。' if
                             packet['phase'] == 'format_repair' else
                             '当前公开框架源码不可读，只能等待操作者。'),
                    'evidence_ids': ['harness_repair']}
                return json.dumps(answer, ensure_ascii=False), metadata
            return content, metadata
    result = recorder.narrate(tmp_path, held, client=Reviewed())
    assert result['status'] == 'written' and result['format_repair_attempted']
    assert [r['answer']['verdict'] for r in result['fact_reviews']] == ['unsupported', 'supported']
    assert '只能等待操作者' not in result['sections']['blocker']['text']
