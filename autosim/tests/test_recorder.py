"""The reporting agent chooses presentation, never measurements or execution."""
import json
import hashlib
from contextlib import contextmanager
import xml.etree.ElementTree as ET

import pytest

from autosim.research import recorder, run_record, report_page
from autosim.research.common import atomic_json
from autosim.research.prepare import Preparation
from autosim.research.agent_roles import role_profile
from autosim.research.agent_runtime import _role_cli_args


def test_readable_closure_panel_separates_paid_training_and_verified_score(tmp_path):
    held = snapshot(tmp_path)
    held['experiment_lifecycle'] = {'metrics':{'valid_scored_development':1,
        'completed_training_pending_score':2}, 'candidates':[
        {'label':'baseline','status':'scored','metric_value':.4,'metric_name':'reward',
         'measurement_ref':'research/derived/measurements/baseline.json',
         'data_consumption':'unknown','recovery':{'edge':'measurement_to_commit'}},
        {'label':'round_1','status':'unscored','data_consumption':'verified',
         'recovery':{'edge':'policy_to_score'}}]}
    text = recorder.render(tmp_path,held,{})
    assert '训练完成、尚待计分：2' in text
    assert 'reward：0.4' in text and '尚缺实际消费证据' in text
    assert '保留权重和冻结协议' in text
    assert 'SOTA' not in text


def test_report_includes_partial_build_facts_and_separate_model_budget(tmp_path):
    atomic_json(tmp_path / "provision_progress.json", {"status": "building", "attempts": [
        {"ok": True, "seconds": 12, "evidence_ref": "evidence/" + "a"*32 + ".json"},
        {"ok": False, "seconds": 3, "evidence_ref": "evidence/" + "b"*32 + ".json"}]})
    atomic_json(tmp_path / "agent" / "cost_ledger.json", {
        "limit_usd": 30, "entries": [{"status": "settled", "actual_usd": .24}]})
    view = run_record.build_report_view(tmp_path, "derived", status="budget_exhausted")
    snapshot = recorder.make_snapshot(tmp_path, view, {"actions": [], "plan": {}})
    assert recorder._evidence(snapshot)["provisioning"]["attempts"][0]["ok"]
    result = recorder.render(tmp_path, snapshot, {})
    assert "环境准备的实际操作" in result
    assert "$0.24000" in result and "$29.76000" in result


def test_native_verification_is_displayed_without_claiming_scored_experiment(tmp_path):
    from autosim.research.evidence_store import capture_attempt_evidence
    identity = 'd'*32
    log = tmp_path/f'derivation_native/{identity}/native.log'
    log.parent.mkdir(parents=True)
    log.write_text('\x1b[31mValueError: unsupported native device\n\x1b[0m\n')
    original_bytes = log.read_bytes()
    sealed = capture_attempt_evidence(tmp_path, attempt_id=identity, log=log,
        receipt_ref=f'derivation_native/{identity}/receipt.json', status='failed',
        returncode=1,termination_reason='failed')
    atomic_json(log.parent/'receipt.json', {**sealed,'id':identity,'stage':'train',
        'status':'failed','seconds':4.2})
    view = run_record.build_report_view(tmp_path,'derived',status='running')
    snapshot = recorder.make_snapshot(tmp_path,view,{'actions':[],'plan':{}})
    document = recorder.render(tmp_path,snapshot,{})
    assert '最近的原生试跑（不计分）' in document
    assert 'unsupported native device' in document
    assert '\x1b' not in document
    assert log.read_bytes() == original_bytes
    assert f'evidence/{identity}.json' in document
    assert '已记录 0 项开发实验' in document
    assert snapshot['measurements'] == []


def test_live_native_panel_advances_during_action_without_model_or_narrative_rewrite(tmp_path, monkeypatch):
    from autosim.research.stage_verification import recent_attempts
    path = tmp_path / 'RUN.md'
    path.write_text('# Run\n' + recorder.START + '\n人类叙述保持\n' + recorder.END + '\n技术详情保持')
    identity = 'c' * 32
    directory = tmp_path / 'derivation_native' / identity
    directory.mkdir(parents=True)
    atomic_json(directory / 'receipt.json', {'id': identity, 'stage': 'evaluate', 'status': 'running'})
    monkeypatch.setattr(recorder, 'refresh', lambda *args, **kwargs: pytest.fail('heartbeat cannot narrate'))
    run_record.refresh_live_status(tmp_path, fallback_status='running')
    content = path.read_text()
    assert 'cccccccccc' in content and '运行记录待核验' in content
    assert '人类叙述保持' in content and '技术详情保持' in content
    assert content.count(recorder.NATIVE_START) == 1
    assert recent_attempts(tmp_path)[0]['status'] == 'running_unverified'
    assert not (tmp_path / 'report/recorder_trigger.json').exists()
    assert 'SOTA' not in content


def test_report_distinguishes_environment_availability_from_readiness(tmp_path):
    atomic_json(tmp_path / "environment_pool.json", {"enabled": True})
    atomic_json(tmp_path / "environment_selection.json", {
        "id": "opaque-base", "reason": "Python 与现有依赖匹配，尚需探针",
        "result": {"ok": True}})
    atomic_json(tmp_path / "package_cache_view.json", {"links": ["file:///fixture.whl"]})
    atomic_json(tmp_path / "environment_cache_publication.json", {
        "snapshot": {"status": "capacity_skipped"}})
    view = run_record.build_report_view(tmp_path, "derived", status="running")
    state = recorder.make_snapshot(tmp_path, view, {"actions": [], "plan": {}})
    evidence = recorder._evidence(state)["environment_reuse"]
    assert evidence["clone_ok"] and evidence["available_wheels"] == 1
    document = recorder.render(tmp_path, state, {})
    assert "环境复用" in document and "不是实测命中数" in document
    assert "Python 与现有依赖匹配" in document
    assert "仍需原生探针" in document and "空间不足，未发布" in document


def measurement(root, label, value, protocol="a" * 64, *, direction="maximize", confirmation=False):
    attempt = hashlib.sha256(label.encode()).hexdigest()[:32]
    base = root / "research" / "derived"
    atomic_json(base / "attempts" / attempt / "receipt.json", {
        "attempt_id": attempt, "stage": "evaluate", "seconds": 12,
        "comparison_protocol_sha256": protocol})
    path = base / "measurements" / (label + ".json")
    atomic_json(path, {"label": label, "ok": value is not None, "metric_value": value,
                      "metric": {"name": "reward", "unit": "points", "direction": direction},
                      "metric_reading": {"episodes_completed": 20},
                      "evaluate": {"attempt_id": attempt}, "varied": {"chunk": 8},
                      "settings": {"task": "demo"}, "confirmation": confirmation})
    return path


def snapshot(root):
    return recorder.make_snapshot(root, run_record.build_report_view(root, "derived", status="running"),
                                  {"actions": [{"step": "train", "outcome": "done"}], "plan": {}})


def test_report_current_cursor_and_latest_action_override_stale_working_plan(tmp_path):
    session = {'run_id':'derived','status':'paused','rounds':22,'next_round':19,'history':[]}
    atomic_json(tmp_path/'research/derived/controller_session.json', session)
    stale = {'objective':'old setup gate', 'remaining_rounds':1, 'candidate_pool':'empty'}
    actions = [{'step':'reconcile_interrupted_action','outcome':'interrupted; outcome unknown'},
               {'step':'generate_research_ideas','outcome':'ideas available','because':'3 selectable ideas'}]
    view = run_record.build_report_view(tmp_path, 'derived', status='running', current_action='choose')
    held = recorder.make_snapshot(tmp_path, view, {'actions':actions,'plan':stale})
    evidence = recorder._evidence(held)
    cursor = evidence['research_progress']
    assert cursor['next_round'] == 19 and cursor['planned_rounds'] == 22
    assert cursor['remaining_candidate_rounds'] == 4
    assert evidence['last_completed_action'] == actions[-1]
    assert evidence['plan']['document'] == stale and 'NOT current' in evidence['plan']['authority']
    session.update(rounds=24)
    atomic_json(tmp_path/'research/derived/controller_session.json', session)
    later = recorder.make_snapshot(tmp_path,
        run_record.build_report_view(tmp_path,'derived',status='running'), {'actions':actions,'plan':stale})
    assert later['event_revision'] != held['event_revision']


def test_foreign_controller_cursor_is_not_narration_authority(tmp_path):
    atomic_json(tmp_path/'research/derived/controller_session.json',
        {'run_id':'foreign','rounds':99,'next_round':1})
    view = run_record.build_report_view(tmp_path,'derived',status='running')
    assert view['research_progress']['status'] == 'unavailable'


def test_heartbeat_marks_recovery_prose_historical_without_rewriting_it(tmp_path):
    narrative = {'written_at':'2026-10-07T11:29:00Z',
                 'observed_run_scope':{'status':'paused','current_action':'finished'}}
    atomic_json(tmp_path/'report/narrative.json', narrative)
    text = '外层暂停，原实验结果保留。'
    content = recorder._narrative_scope(narrative,'paused','finished',tmp_path) + '\n' + text
    assert '状态或动作已改变' not in content
    updated = recorder.refresh_narrative_scope(content,tmp_path,status='running',current_action='choose')
    assert '状态或动作已改变' in updated and '不是当前状态' in updated
    assert text in updated and (tmp_path/'report/narrative.json').is_file()
    assert recorder.read(tmp_path,'report/narrative.json') == narrative
    assert recorder.refresh_narrative_scope('old unmarked report',tmp_path,status='running',current_action='choose') == 'old unmarked report'


def test_reference_is_not_candidate_gain_and_baseline_meaning_is_chinese(tmp_path):
    measurement(tmp_path,'baseline',0)
    measurement(tmp_path,'reference_published',0.51)
    held = snapshot(tmp_path)
    row = next(r for r in held['measurements'] if r['label']=='reference_published')
    assert not row['comparable'] and row['delta'] is None
    document = recorder.render(tmp_path,held,{})
    assert '本地 baseline 的有效测量不等于官方性能复现' in document
    assert '已发布基线：尚未核验' in document


def test_reused_native_training_is_visible_but_cost_chart_counts_one_attempt(tmp_path):
    first=measurement(tmp_path,'baseline',0)
    linked=measurement(tmp_path,'round_1_handoff_correction',0.2)
    attempt='f'*32
    base=tmp_path/'research/derived'
    atomic_json(base/'attempts'/attempt/'receipt.json',{'attempt_id':attempt,'stage':'train','seconds':120})
    for path,reused in [(first,False),(linked,True)]:
        row=json.loads(path.read_text());row['train']={'attempt_id':attempt,'reused':reused};atomic_json(path,row)
    held=snapshot(tmp_path)
    points=recorder.cost_points(held['measurements'])
    assert [v for name,v in points if name.endswith('/train')]==[120]
    assert len([name for name,v in points if name.endswith('/evaluate')])==2
    document=recorder.render(tmp_path,held,{})
    assert 'train 已复用（原尝试 120 秒）' in document
    assert '不是又训练一次' in document


def test_cost_chart_does_not_guess_identity_from_equal_durations():
    rows=[{'label':'one','stage_seconds':{'train':10}},
          {'label':'two','stage_seconds':{'train':10}}]
    assert len(recorder.cost_points(rows))==2


def _active_session(root):
    session = {'run_id': 'derived', 'status': 'running', 'action': 'round_12',
               'pending_action': {'round': 12, 'phase': 'measurement_starting',
                   'idea': {'label': 'shorter-action-chunk', 'mechanism': '计划缩短动作开环区间',
                            'why': '依据已有开发侧失败分析，仍需新实验核验'},
                   'changed': {'files': ['policy/deploy.py', '../private', '/private']},
                   'settings': {'steps': 16384, 'private_path': '/private', 'bad': 'not_numeric'}}}
    atomic_json(root / 'research/derived/controller_session.json', session)
    atomic_json(root / 'run_state.json', {'run_id': 'derived', 'status': 'running'})
    return session


def test_active_candidate_reports_requested_work_not_completion(tmp_path):
    session = _active_session(tmp_path)
    current = recorder.active_candidate(tmp_path, 'derived')
    assert current['files'] == ['policy/deploy.py']
    assert current['requested_settings'] == {'steps': 16384}
    assert current['proposed_hypothesis']['mechanism'] == '计划缩短动作开环区间'
    assert 'not verified' in current['proposed_hypothesis']['authority']
    view = run_record.build_report_view(tmp_path, 'derived', status='running')
    snapshot = recorder.make_snapshot(tmp_path, view, {'actions': [], 'plan': {}})
    assert recorder._evidence(snapshot)['active_candidate'] == current
    document = recorder.render(tmp_path, snapshot, {})
    assert '第 12 轮' in document and 'steps=16384' in document
    assert '设置是请求量，不是已完成步数' in document
    assert '未请求冻结权重复用' in document
    assert '尚无本轮分数' not in document  # A measurement may precede cursor commit.
    session['pending_action']['idea']['mechanism'] = 'x'*2000
    session['pending_action']['idea']['why'] = 'y'*2000
    atomic_json(tmp_path / 'research/derived/controller_session.json', session)
    bounded = recorder.active_candidate(tmp_path, 'derived')['proposed_hypothesis']
    assert len(bounded['mechanism']) == 1000 and len(bounded['reason']) == 700
    assert bounded['mechanism_truncated'] and bounded['reason_truncated']
    session['pending_action']['frozen_policy_request'] = {'label': 'parent'}
    atomic_json(tmp_path / 'research/derived/controller_session.json', session)
    panel = recorder.active_candidate_panel(recorder.active_candidate(tmp_path, 'derived'))
    assert '实际加载仍待核验' in panel
    assert '本轮仅评测，不安排新训练' in panel and '继承的原训练设置' in panel
    assert '不是本轮新增训练量' in panel and '请求的数值设置' not in panel


@pytest.mark.parametrize('change', [
    {'status': 'completed'}, {'run_id': 'other'}, {'action': 'round_13'},
    {'pending_action': None}, {'pending_action': {'round': True}},
    {'pending_action': {'round': 12, 'idea': [], 'changed': {}}},
    {'pending_action': {'round': 12, 'idea': {'label': 'x'}, 'changed': []}},
    {'pending_action': {'round': 12, 'idea': {'label': 'x'}, 'changed': {'files': 'policy.py'}}},
])
def test_active_candidate_refuses_stale_or_malformed_state(tmp_path, change):
    session = _active_session(tmp_path)
    session.update(change)
    atomic_json(tmp_path / 'research/derived/controller_session.json', session)
    assert recorder.active_candidate(tmp_path, 'derived') == {}


def test_active_candidate_refuses_symlink_and_invalid_run_id(tmp_path):
    _active_session(tmp_path)
    assert recorder.active_candidate(tmp_path, '../private') == {}
    path = tmp_path / 'research/derived/controller_session.json'
    path.rename(path.with_name('real.json'))
    path.symlink_to(path.with_name('real.json'))
    assert recorder.active_candidate(tmp_path, 'derived') == {}


def test_live_candidate_panel_advances_without_narration_and_clears_on_stop(tmp_path, monkeypatch):
    _active_session(tmp_path)
    path = tmp_path / 'RUN.md'
    path.write_text('# Run\n' + recorder.START + '\n原叙述保留\n' + recorder.END + '\n历史保留')
    monkeypatch.setattr(recorder, 'refresh', lambda *a, **kw: pytest.fail('no LLM on heartbeat'))
    run_record.refresh_live_status(tmp_path, fallback_status='running')
    document = path.read_text()
    assert '第 12 轮' in document and '原叙述保留' in document and '历史保留' in document
    atomic_json(tmp_path / 'run_state.json', {'run_id': 'derived', 'status': 'stopped'})
    atomic_json(tmp_path / 'report/health.json', {})
    run_record.refresh_live_status(tmp_path, fallback_status='stopped')
    document = path.read_text()
    assert '第 12 轮' not in document and '原叙述保留' in document
    assert document.count(recorder.ACTIVE_START) == 1
    atomic_json(tmp_path / 'run_state.json', {'run_id': 'derived', 'status': 'running'})
    cleared = recorder.refresh_active_candidate_panel(document, tmp_path, status='stopped')
    assert '第 12 轮' not in cleared


@pytest.mark.parametrize('outer_destination', [False, True])
def test_native_research_root_heartbeat_resolves_owner_candidate(tmp_path, outer_destination):
    _active_session(tmp_path)
    native = tmp_path / 'research/derived'
    destination = (tmp_path if outer_destination else native) / 'RUN.md'
    destination.write_text('# Run\n' + recorder.START + '\n旧叙述\n' + recorder.END)
    atomic_json(native / 'running.json', {'stage': 'train', 'started_at': 1})
    run_record.refresh_live_status(native, destination=destination)
    content = destination.read_text()
    assert '第 12 轮' in content and 'steps=16384' in content and '旧叙述' in content
    expected = 'research/derived/controller_session.json' if outer_destination else 'controller_session.json'
    assert f'[候选提交记录]({expected})' in content


def test_linked_measurement_shows_adopted_best_and_receipt_bound_original_idea(tmp_path):
    from autosim.research.common import digest
    original=measurement(tmp_path,'round_1',0.1)
    linked=measurement(tmp_path,'round_1_handoff_correction',0.3)
    row=json.loads(linked.read_text());row.pop('varied',None)
    row['handoff_correction']={'original_measurement_ref':str(original.relative_to(tmp_path)),
                              'original_measurement_sha256':digest(original)}
    atomic_json(linked,row)
    atomic_json(tmp_path/'research/derived/controller_session.json',{'run_id':'derived',
        'history':[{'label':'round_1','idea':'source-backed-training',
                    'mechanism':'原训练干预，恢复时没有重训','undone':True}]})
    atomic_json(tmp_path/'research/derived/research_report.json',
                {'best':{'name':'round-1_handoff_correction','score':0.3}})
    current=snapshot(tmp_path)
    candidate=next(r for r in current['measurements'] if r['label']=='round_1_handoff_correction')
    assert candidate['idea']=='source-backed-training'
    assert candidate['changes']=='原训练干预，恢复时没有重训'
    assert candidate['disposition']=='当前最佳；保留'
    assert candidate['is_incumbent'] is True
    assert '当前已采用最佳' in recorder.render(tmp_path,current,{})
    assert '已回退' not in candidate['disposition']
    row['handoff_correction']['original_measurement_sha256']='0'*64;atomic_json(linked,row)
    candidate=next(r for r in snapshot(tmp_path)['measurements'] if r['label']=='round_1_handoff_correction')
    assert not candidate['idea'] and not candidate['changes']


def test_projection_joins_real_history_and_excludes_invalid_candidate(tmp_path, monkeypatch):
    from autosim.research import candidate_source_guard
    measurement(tmp_path, 'baseline', .98)
    path = measurement(tmp_path, 'round_1', .74)
    row = json.loads(path.read_text())
    row.pop('varied')
    row['metric_reading'] = {'value': .74}
    row['metric_lineage'] = {'status': 'verified', 'episodes_completed': 50}
    atomic_json(path, row)
    atomic_json(tmp_path / 'research/derived/controller_session.json',
        {'run_id': 'derived', 'history': [{'label': 'round_1', 'idea': 'normalization',
         'mechanism': '观测归一化', 'undone': True, 'verdict': '未超过基线'}]})
    current = snapshot(tmp_path)
    candidate = next(r for r in current['measurements'] if r['label'] == 'round_1')
    assert candidate['changes'] == '观测归一化'
    assert candidate['samples'] == 50 and '已回退' in candidate['disposition']
    monkeypatch.setattr(candidate_source_guard, 'historical_mismatches',
                        lambda root: [{'label': 'round_1', 'status': 'candidate_source_mismatch'}])
    current = snapshot(tmp_path)
    invalid = next(r for r in current['measurements'] if r['label'] == 'round_1')
    assert not invalid['comparable'] and invalid['delta'] is None
    assert invalid['metric_value'] == .74  # Keep original measurement; never falsify it.
    document = recorder.render(tmp_path, current, {})
    assert '候选修改未实际生效' in document and 'normalization' in document


def test_framework_blocker_is_not_mislabeled_environment_failure(tmp_path):
    current = recorder.make_snapshot(tmp_path,
        run_record.build_report_view(tmp_path, 'derived', status='infrastructure_blocked'),
        {'actions': [{'step': 'run_the_loop', 'outcome': 'raised',
          'because': 'ValueError: stage_argv_train produced an ambiguous command: duplicate',
          'receipt_ref': 'action_receipts/' + 'd'*16 + '.json'}]})
    document = recorder.render(tmp_path, current, {})
    assert '运行基础设施／框架阻塞' in document
    assert '命令组装／参数合同拒绝' in document
    assert '运行环境阻塞，需修复启动条件' not in document


@pytest.mark.parametrize('field,value,expected', [
    ('text', '中' * 701, 'text must contain Chinese'),
    ('text', 'only English', 'text must contain Chinese'),
    ('evidence_ids', ['run'] * 7, '1..6 existing IDs'),
    ('evidence_ids', ['invented'], 'unknown or invalid evidence ID'),
])
def test_recorder_rejection_identifies_exact_section_without_exposing_content(
        tmp_path, field, value, expected):
    current = snapshot(tmp_path)
    answer = {'event_revision': current['event_revision'],
        'sections': {key: {'text': '当前训练尚需核验。', 'evidence_ids': ['run']}
                     for key in recorder.FIELDS}}
    answer['sections']['finding'][field] = value
    with pytest.raises(ValueError, match='Recorder section finding:') as error:
        recorder._validate(answer, current)
    assert expected in str(error.value)
    assert 'invented' not in str(error.value)


def test_recorder_prompt_explains_actual_length_and_reference_limits():
    assert '1–700' in recorder.SYSTEM and '1–6' in recorder.SYSTEM
    assert '100–250' in recorder.SYSTEM
    assert '内层研究光标 paused' in recorder.SYSTEM


def test_recorder_gets_operator_provenance_and_new_finding_is_semantic_event(tmp_path, monkeypatch):
    from autosim.research import monitor_findings
    rows = [{'evidence_id': 'a'*32, 'origin': 'operator_diagnostic',
             'authority': 'advisory, not a score or autonomous native failure',
             'summary': '操作者取消了一条已复现接口错误的训练。'}]
    monkeypatch.setattr(monitor_findings, 'view', lambda *args: rows)
    first = snapshot(tmp_path)
    assert recorder._evidence(first)['monitor_findings'] == rows
    rows = [{**rows[0], 'evidence_id': 'b'*32, 'summary': '新增只读反例，尚未核验真实 batch。'}]
    second = snapshot(tmp_path)
    assert first['event_revision'] != second['event_revision']
    assert not second['measurements']


def test_performance_claim_check_does_not_treat_label_digits_as_metric_values(tmp_path):
    measurement(tmp_path, 'baseline', 0.0)
    measurement(tmp_path, 'round_6', 0.02)
    held = snapshot(tmp_path)
    rows = {row['label']: row for row in held['measurements']}
    refs = [rows[label]['id'] for label in ('baseline', 'round_6')]
    answer = {'event_revision': held['event_revision'], 'sections': {
        key: {'text': '当前训练尚需核验。', 'evidence_ids': ['run']}
        for key in recorder.FIELDS}}
    answer['sections']['finding'] = {
        'text': '本地基线成功率为 0.0，round_6 为 0.02，仍需独立确认。',
        'evidence_ids': refs, 'metric_claims': [
            {'measurement_id': rows[label]['id'], 'field': 'metric_value',
             'value': value, 'unit': 'points'}
            for label, value in [('baseline', 0.0), ('round_6', 0.02)]]}
    assert recorder._validate(answer, held)['sections']['finding']['text'].endswith('独立确认。')
    answer['sections']['finding']['metric_claims'].pop()
    with pytest.raises(ValueError, match='typed metric claims'):
        recorder._validate(answer, held)

@pytest.mark.parametrize('suffix', ['（20集）', '（20 episodes）', '（20个评测回合）'])
def test_performance_claim_check_separates_evidenced_sample_count_from_metric(tmp_path, suffix):
    measurement(tmp_path, 'baseline', 0.23)
    held = snapshot(tmp_path)
    row = held['measurements'][0]
    answer = {'event_revision': held['event_revision'], 'sections': {
        key: {'text': '当前训练尚需核验。', 'evidence_ids': ['run']}
        for key in recorder.FIELDS}}
    answer['sections']['finding'] = {
        'text': '本地成功率为0.23' + suffix + '，尚需确认。',
        'evidence_ids': [row['id']], 'metric_claims': [
            {'measurement_id': row['id'], 'field': 'metric_value', 'value': 0.23, 'unit': 'points'}]}
    recorder._validate(answer, held)
    original_text = answer['sections']['finding']['text']
    answer['sections']['finding']['text'] = '本地成功率为0.23（999集），尚需确认。'
    with pytest.raises(ValueError, match='numbers absent'):
        recorder._validate(answer, held)
    answer['sections']['finding']['text'] = original_text
    answer['sections']['finding']['metric_claims'] = []
    with pytest.raises(ValueError, match='typed metric claims'):
        recorder._validate(answer, held)


def test_report_exposes_frozen_policy_lineage_and_literal_rollback_without_new_training(tmp_path):
    train_id = 'a' * 32
    policy_id = 'b' * 64
    parent = measurement(tmp_path, 'baseline', 0.23)
    candidate = measurement(tmp_path, 'round_14', 0.19)
    atomic_json(tmp_path / 'research/derived/attempts' / train_id / 'receipt.json', {
        'attempt_id': train_id, 'seconds': 120})
    for path in (parent, candidate):
        row = json.loads(path.read_text())
        row['train'] = {'attempt_id': train_id, 'reused': path == candidate}
        row['policy_artifact'] = {'sha256': policy_id}
        if path == candidate:
            row['policy_reuse'] = {'mode': 'evaluation_only_local_frozen_policy',
                'label': 'baseline', 'training_attempt_id': train_id,
                'policy_sha256': policy_id, 'no_new_training_or_data_claim': True,
                'private_path': '/do/not/include', 'unexpected': 'not report authority'}
        atomic_json(path, row)
    atomic_json(tmp_path / 'research/derived/controller_session.json', {
        'run_id': 'derived', 'history': [{'label': 'baseline', 'undone': False},
            {'label': 'round_14', 'undone': True}]})
    rows = {row['label']: row for row in snapshot(tmp_path)['measurements']}
    assert rows['baseline']['new_stage_attempt_ids']['train'] == train_id
    frozen = rows['round_14']
    assert frozen['reused_stages'] == ['train']
    assert set(frozen['new_stage_attempt_ids']) == {'evaluate'}
    assert frozen['policy_sha256'] == rows['baseline']['policy_sha256'] == policy_id
    assert frozen['policy_identity_group'] == rows['baseline']['policy_identity_group']
    from autosim.research.common import sanitize_model_payload
    external = sanitize_model_payload(rows)
    assert policy_id not in json.dumps(external)
    assert external['round_14']['policy_identity_group'] == external['baseline']['policy_identity_group']
    assert frozen['policy_reuse']['label'] == 'baseline'
    assert 'private_path' not in frozen['policy_reuse'] and 'unexpected' not in frozen['policy_reuse']
    assert frozen['candidate_rollback_applied'] is True
    assert rows['baseline']['candidate_rollback_applied'] is False


def test_format_feedback_lists_all_sections_but_never_changes_proposal(tmp_path):
    held = snapshot(tmp_path)
    answer = {'event_revision': held['event_revision'], 'sections': {
        key: {'text': '当前训练尚需核验。', 'evidence_ids': ['run']}
        for key in recorder.FIELDS}}
    answer['sections']['finding']['text'] = '中' * 701
    answer['sections']['blocker']['evidence_ids'] = ['invented']
    before = json.dumps(answer, ensure_ascii=False)
    errors = recorder._section_validation_errors(answer, held)
    assert len(errors) == 2
    assert any('section finding:' in error for error in errors)
    assert any('section blocker:' in error for error in errors)
    assert json.dumps(answer, ensure_ascii=False) == before
    assert not (tmp_path/'report/narrative.json').exists()


def test_unsupported_number_feedback_identifies_the_missing_values_without_rounding_evidence(tmp_path):
    held = snapshot(tmp_path)
    held['budget'] = {'elapsed_seconds': 12.345}
    answer = {'event_revision': held['event_revision'], 'sections': {
        key: {'text': '当前训练尚需核验。', 'evidence_ids': ['run']}
        for key in recorder.FIELDS}}
    answer['sections']['current'] = {'text': '预算已用12.3秒。', 'evidence_ids': ['budget']}
    with pytest.raises(ValueError, match=r"numbers absent.*12\.3"):
        recorder._validate(answer, held)
    assert held['budget']['elapsed_seconds'] == 12.345
    answer['sections']['current']['text'] = '预算已用12.345秒。'
    recorder._validate(answer, held)


@pytest.mark.parametrize('text,allowed', [
    ('本地回报为 -3.0，仍需确认。', True),
    ('本地回报为 3.0，仍需确认。', False),
    ('本地回报为 -3.0。另一项指标为 2.0。', False),
])
def test_metric_claim_validation_checks_sign_and_each_performance_mention(tmp_path, text, allowed):
    measurement(tmp_path, 'baseline', -3.0)
    measurement(tmp_path, 'round_6', 2.0)
    held = snapshot(tmp_path)
    rows = {row['label']: row for row in held['measurements']}
    answer = {'event_revision': held['event_revision'], 'sections': {
        key: {'text': '当前结果尚需核验。', 'evidence_ids': ['run']}
        for key in recorder.FIELDS}}
    answer['sections']['finding'] = {'text': text,
        'evidence_ids': [row['id'] for row in rows.values()],
        'metric_claims': [{'measurement_id': rows['baseline']['id'],
            'field': 'metric_value', 'value': -3.0, 'unit': 'points'}]}
    if allowed:
        recorder._validate(answer, held)
    else:
        with pytest.raises(ValueError, match='typed metric claims'):
            recorder._validate(answer, held)


class Writer:
    supports_recorder = True
    supports_main_agent = True

    def __init__(self, mode="normal"):
        self.calls, self.roles, self.mode = [], [], mode

    @contextmanager
    def as_role(self, role):
        self.roles.append(role)
        yield

    def chat_with_metadata(self, system, user, **kwargs):
        packet = json.loads(user)
        self.calls.append((packet, kwargs))
        if self.mode == "service_failure":
            raise RuntimeError("fixture service unavailable")
        if packet.get('phase') == 'review_report':
            return json.dumps({'verdict': 'supported', 'issues': []}), {'role': 'supervisor'}
        if "skill_catalog" in packet:
            return json.dumps({"skill_reads": [{"id": "autosimsota.writing-research-updates", "why": "解释当前结果"}]}), {}
        if self.mode == 'malformed_json' and 'format_correction' not in packet or self.mode == 'always_malformed_json':
            return '{"sections":{"current":{"text":"正文里有"未转义引号""}}}', {"role": "recorder"}
        if self.mode == 'oversized_malformed_json':
            return 'x' * 16001, {"role": "recorder"}
        answer = {"event_revision": packet["event_revision"],
                  "sections": {key: {"text": "当前证据支持继续核对训练结果，而非宣称完成研究。",
                                     "evidence_ids": ["run"]} for key in recorder.FIELDS},
                  "charts": [{"kind": "comparison", "why": "比较同协议的开发指标。"},
                             {"kind": "cost", "why": "看清阶段耗时。"}], "demo_requests": []}
        ids = [key for key in packet["evidence"] if key.startswith("m-")]
        if ids:
            answer["demo_requests"] = [{"measurement_id": ids[0], "purpose": "观察接触阶段的行为。"}]
        if self.mode == "fabricated":
            answer["sections"]["finding"]["text"] = "成功率提高了99%。"
        if self.mode == "unknown_ref" or (
                self.mode == 'repairable_ref' and 'format_correction' not in packet):
            answer["sections"]["finding"]["evidence_ids"] = ["invented"]
        if self.mode == "command":
            answer["charts"][0]["command"] = "run arbitrary code"
        return json.dumps(answer, ensure_ascii=False), {"role": "recorder"}


@pytest.mark.parametrize('mode', ['correctable', 'repeated', 'bad_reference', 'bad_protocol', 'transport'])
def test_independent_fact_review_corrects_once_or_preserves_previous(tmp_path, mode):
    class ReviewedWriter(Writer):
        def __init__(self):
            super().__init__()
            self.reviews = 0

        def chat_with_metadata(self, system, user, **options):
            packet = json.loads(user)
            if packet.get('phase') != 'review_report':
                return super().chat_with_metadata(system, user, **options)
            self.calls.append((packet, options)); self.reviews += 1
            if mode == 'transport':
                raise RuntimeError('review unavailable')
            issue = {'section': 'finding', 'reason': '没有证据支持复用权重的说法。',
                     'evidence_ids': ['run' if mode != 'bad_reference' else 'invented']}
            answer = {'verdict': 'unsupported', 'issues': [issue]}
            if mode == 'correctable' and self.reviews == 2:
                answer = {'verdict': 'supported', 'issues': []}
            if mode == 'bad_protocol':
                answer['issues'] = []
            return json.dumps(answer), {'role': 'supervisor'}

    measurement(tmp_path, 'baseline', 1)
    held = snapshot(tmp_path)
    writer = ReviewedWriter()
    result = recorder.narrate(tmp_path, held, client=writer, timeout=40)
    trigger = recorder.read(tmp_path, 'report/recorder_trigger.json')
    if mode == 'correctable':
        assert result['status'] == 'written' and len(writer.calls) == 5
        assert result['fact_review_contract_version'] == recorder.FACT_REVIEW_CONTRACT_VERSION
        assert [row['answer']['verdict'] for row in result['fact_reviews']] == ['unsupported', 'supported']
        repair = next(packet for packet, _ in writer.calls if packet['phase'] == 'format_repair')
        assert repair['format_correction']['factual_issues'][0]['section'] == 'finding'
    else:
        assert not result and trigger['status'] == 'failed'
        assert not recorder.pending_requests(tmp_path)
        assert not (tmp_path/'report/narrative.json').exists()
        assert len(writer.calls) == (5 if mode == 'repeated' else 3)
    assert all(0 < options['timeout'] <= 20 for _, options in writer.calls)
    assert all(options['read_only'] and options['resume_session'] is False for _, options in writer.calls)
    recorder.narrate(tmp_path, held, client=writer, timeout=40)
    assert len(writer.calls) == (5 if mode in {'correctable', 'repeated'} else 3)


def test_recorder_corrects_format_once_in_same_time_budget(tmp_path):
    held = snapshot(tmp_path)
    writer = Writer('repairable_ref')
    result = recorder.narrate(tmp_path, held, client=writer, timeout=40)
    assert result['status'] == 'written' and result['format_repair_attempted']
    assert len(writer.calls) == 4 and len(result['turns']) == 4
    assert [packet['phase'] for packet,_ in writer.calls]==['skill_selection','write_report','format_repair','review_report']
    assert [options['max_tokens'] for _,options in writer.calls]==[1000,5000,5000,1500]
    assert all(options['thinking']=='disabled' and options['resume_session'] is False for _,options in writer.calls)
    packet, options = writer.calls[-2]
    assert 'Recorder section finding' in packet['format_correction']['error']
    assert options['read_only'] and not options['auto_skills']
    assert 0 < options['timeout'] <= 20
    assert not (tmp_path/'research/derived/measurements').exists()
    recorder.narrate(tmp_path, held, client=writer, timeout=40)
    assert len(writer.calls) == 4  # Cached success does not repeat model calls.


def test_recorder_format_repair_is_bounded_and_never_accepts_unknown_evidence(tmp_path):
    writer = Writer('unknown_ref')
    assert not recorder.narrate(tmp_path, snapshot(tmp_path), client=writer)
    assert len(writer.calls) == 3
    assert recorder.read(tmp_path, 'report/recorder_trigger.json')['status'] == 'failed'
    assert not (tmp_path/'report/narrative.json').exists()


def test_recorder_malformed_json_gets_one_bounded_correction(tmp_path):
    writer = Writer('malformed_json')
    result = recorder.narrate(tmp_path, snapshot(tmp_path), client=writer, timeout=40)
    assert result['status'] == 'written' and result['format_repair_attempted']
    assert [packet['phase'] for packet, _ in writer.calls] == [
        'skill_selection', 'write_report', 'format_repair', 'review_report']
    correction, options = writer.calls[-2]
    assert isinstance(correction['format_correction']['previous_answer'], str)
    assert '合法JSON' in correction['format_correction']['instruction']
    assert 0 < options['timeout'] <= 20 and options['max_tokens'] == 5000


@pytest.mark.parametrize('mode,call_count', [('always_malformed_json', 3), ('oversized_malformed_json', 2)])
def test_recorder_does_not_loop_or_publish_unparseable_content(tmp_path, mode, call_count):
    writer = Writer(mode)
    assert not recorder.narrate(tmp_path, snapshot(tmp_path), client=writer)
    assert len(writer.calls) == call_count
    assert recorder.read(tmp_path, 'report/recorder_trigger.json')['status'] == 'failed'
    assert not (tmp_path/'report/narrative.json').exists()


def test_recorder_event_to_charts_request_and_cached_narration(tmp_path):
    measurement(tmp_path, "baseline", -3)
    measurement(tmp_path, "candidate", -1)
    held = snapshot(tmp_path)
    client = Writer()
    narrative = recorder.narrate(tmp_path, held, client=client)
    assert narrative["status"] == "written"
    assert narrative["skill_selection"][0]["body_sha256"]
    assert client.roles == ["recorder", "supervisor"]
    assert all(options["read_only"] and not options["auto_skills"] for _, options in client.calls)
    assert len(recorder.pending_requests(tmp_path)) == 1
    doc = recorder.render(tmp_path, held, narrative)
    assert "最新发现" in doc and "相对基线差值" in doc and "同协议" in doc
    assert held["measurements"][1]["delta"] == 2
    assert held["measurements"][0]["samples"] == 20
    charts = list((tmp_path / "report" / "charts").glob("*.svg"))
    assert len(charts) == 2
    for path in charts:
        ET.fromstring(path.read_text())
    recorder.narrate(tmp_path, held, client=client)
    assert len(client.calls) == 3
    # New heartbeat/budget/current-action bytes do not create a new research event.
    view = run_record.build_report_view(tmp_path, "derived", status="running", current_action="poll")
    view["budget"] = {"elapsed_wall_seconds": 999}
    later = recorder.make_snapshot(tmp_path, view, {"actions": held["actions"], "plan": {}})
    assert later["event_revision"] == held["event_revision"]
    recorder.narrate(tmp_path, later, client=client)
    assert len(client.calls) == 3
    assert "已有更新" in recorder.render(tmp_path, later, narrative)


@pytest.mark.parametrize("mode", ["service_failure", "fabricated", "unknown_ref", "command"])
def test_bad_narration_falls_back_and_is_not_retried_on_heartbeat(tmp_path, mode):
    held = snapshot(tmp_path)
    client = Writer(mode)
    assert not recorder.narrate(tmp_path, held, client=client)
    count = len(client.calls)
    recorder.narrate(tmp_path, held, client=client)
    assert len(client.calls) == count
    doc = recorder.render(tmp_path, held, {})
    assert "写作服务本轮未完成" in doc and "没有分数不等于成功率为零" in doc
    assert not recorder.pending_requests(tmp_path)


def test_comparison_rejects_mismatched_protocol_and_hidden_confirmation(tmp_path):
    measurement(tmp_path, "baseline", 1)
    measurement(tmp_path, "candidate", 5, protocol="b" * 64)
    measurement(tmp_path, "hidden", 99, confirmation=True)
    measurement(tmp_path, "failed", None)
    held = snapshot(tmp_path)
    assert all(row["label"] != "hidden" for row in held["measurements"])
    assert not next(row for row in held["measurements"] if row["label"] == "candidate")["comparable"]
    assert "hidden" not in json.dumps(recorder._evidence(held))
    assert "不计算" in recorder.render(tmp_path, held, {})
    assert not list((tmp_path / "report/charts").glob("*-comparison.svg"))


def test_failed_writer_does_not_retry_each_routine_event_but_new_evidence_bypasses(tmp_path):
    import copy
    held = snapshot(tmp_path)
    writer = Writer('service_failure')
    recorder.narrate(tmp_path, held, client=writer)
    assert len(writer.calls) == 1
    assert writer.calls[0][1]['timeout'] == 90
    routine = copy.deepcopy(held)
    routine['event_revision'] = 'routine-after-failure'
    routine['actions'] = [{'step':'inspect', 'outcome':'completed'}]
    recorder.narrate(tmp_path, routine, client=writer)
    assert len(writer.calls) == 1
    assert (tmp_path/'report/recorder_deferred.json').is_file()
    routine['runtime_failure'] = {'category':'new_native_failure'}
    routine['event_revision'] = 'new-evidence-after-failure'
    recorder.narrate(tmp_path, routine, client=writer)
    assert len(writer.calls) == 2


def test_explicit_recorder_window_stays_respected(tmp_path):
    writer = Writer()
    assert recorder.narrate(tmp_path, snapshot(tmp_path), client=writer, timeout=40)['status'] == 'written'
    assert all(options['timeout'] == 20 for _, options in writer.calls)


@pytest.mark.parametrize('selection_seconds',[31,41])
def test_recorder_selection_and_prose_share_original_window(tmp_path,monkeypatch,selection_seconds):
    import time
    clock=[0]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    class DelayedSelection(Writer):
        def chat_with_metadata(self,system,user,**options):
            result=super().chat_with_metadata(system,user,**options)
            if len(self.calls)==1:
                clock[0]=selection_seconds
            return result
    writer=DelayedSelection()
    result=recorder.narrate(tmp_path,snapshot(tmp_path),client=writer,timeout=40)
    if selection_seconds==31:
        assert result['status']=='written' and writer.calls[1][1]['timeout']==9
    else:
        assert len(writer.calls)==1 and not result
        assert json.loads((tmp_path/'report/recorder_trigger.json').read_text())['status']=='failed'


def test_new_research_event_stales_narration_and_generates_a_new_turn(tmp_path):
    measurement(tmp_path, "baseline", 1)
    first = snapshot(tmp_path)
    writer = Writer()
    narrative = recorder.narrate(tmp_path, first, client=writer)
    measurement(tmp_path, "candidate", 2)
    second = snapshot(tmp_path)
    assert "上一份叙述已过期" in recorder.render(tmp_path, second, narrative)
    recorder.narrate(tmp_path, second, client=writer)
    assert len(writer.calls) == 6


def test_running_action_is_separate_from_old_stop_and_plan(tmp_path):
    held = snapshot(tmp_path)
    held['current_action'] = 'train'
    held['actions'] = [{'step': 'budget', 'outcome': 'exhausted',
                        'because': 'old wall deadline expired'}]
    held['plan'] = {'objective': 'reach native score',
                    'open_questions': ['old missing interpreter question'],
                    'next_actions': ['old environment preparation'],
                    'data_strategy': {'route': 'old deferred collection route'}}
    document = recorder.render(tmp_path, held, {})
    assert '正在执行：**train**' in document
    assert '最近已返回的操作（历史）' in document
    assert '不是当前动作的结果或停止原因' in document
    assert '上次计划中的问题（未复核当前状态）' in document
    assert '上次计划中的后续动作（不是当前指令）' in document
    assert '不证明当前仍采用该路线' in document
    assert 'old wall deadline expired' in document
    assert '<summary>历史操作说明摘录（按需展开）</summary>' in document
    assert '具体说明：old wall deadline expired' not in document
    assert 'old deferred collection route' in document
    held['status'] = 'budget_exhausted'
    held['current_action'] = None
    stopped = recorder.render(tmp_path, held, {})
    assert '最近一次操作' in stopped
    assert '正在执行' not in stopped
    assert '最近已返回的操作（历史）' not in stopped
    assert '具体说明：old wall deadline expired' in stopped
    assert '<summary>历史操作说明摘录（按需展开）</summary>' not in stopped


@pytest.mark.parametrize('status', ['running', 'paused'])
def test_stop_reason_is_historical_after_resume(tmp_path, status):
    held = snapshot(tmp_path)
    held.update(status=status, current_action='derive_a_command' if status == 'running' else None)
    held['actions'] = [{'step': 'stop', 'outcome': 'infrastructure_blocked',
                        'because': 'old failure before resume'}]
    document = recorder.render(tmp_path, held, {})
    assert 'old failure before resume' in document
    assert ('### 为什么停止' in document) == (status != 'running')
    assert ('<summary>上次停止说明（历史，当前已续跑）</summary>' in document) == (status == 'running')


@pytest.mark.parametrize('outcome, label', [
    ('not attempted', '未执行（原因见操作记录）'),
    ('yielded', '本段已返回主控，不代表实验成功'),
])
def test_folded_history_localizes_outcome_without_rendering_untrusted_html(tmp_path, outcome, label):
    held = snapshot(tmp_path)
    held['current_action'] = 'train'
    held['actions'] = [{'step': 'run_the_loop', 'outcome': outcome,
                       'because': '</details><script>old rejected review</script>'}]
    document = recorder.render(tmp_path, held, {})
    assert f'状态：**{label}**' in document
    assert '<summary>历史操作说明摘录（按需展开）</summary>' in document
    assert '&lt;script&gt;old rejected review&lt;/script&gt;' in document
    assert '<script>' not in document
    assert document.count('</details>') == 1


def test_monitor_fixture_is_shown_as_operator_evidence_not_measurement(tmp_path):
    held = snapshot(tmp_path)
    held['monitor_findings'] = [{'native_attempt_id':'a'*32,
                                'evidence_id':'b'*32,
                                'evidence_ref':f"evidence/{'b'*32}.json",
                                'summary':'操作者取消了有缺陷的训练，建议独立核验序列边界'}]
    document = recorder.render(tmp_path, held, {})
    assert '外部监视诊断（不是原生实验结果）' in document
    assert '不是系统自主发现' in document
    assert '操作者取消了有缺陷的训练' in document
    assert '已记录 0 项开发实验' in document
    assert '查看反例' in document


def test_recorder_coalesces_routine_prose_but_new_failure_bypasses(tmp_path):
    import copy
    held = snapshot(tmp_path)
    writer = Writer()
    original = recorder.narrate(tmp_path, held, client=writer)
    assert "evidence" not in writer.calls[0][0]
    assert "task_outline" in writer.calls[0][0]
    assert "evidence" in writer.calls[1][0]
    assert writer.calls[1][0]['allowed_evidence_ids'] == list(writer.calls[1][0]['evidence'])
    routine = copy.deepcopy(held)
    routine["event_revision"] = "routine-change"
    routine["actions"] = [{"step": "inspect", "outcome": "completed"}]
    assert recorder.narrate(tmp_path, routine, client=writer) == original
    assert len(writer.calls) == 3
    routine["runtime_failure"] = {"category": "new_failure"}
    routine["event_revision"] = "new-failure"
    recorder.narrate(tmp_path, routine, client=writer)
    assert len(writer.calls) == 6


def test_repeated_argument_refusals_do_not_buy_new_prose_for_receipt_ids(tmp_path):
    import copy
    held=snapshot(tmp_path)
    held['actions']=[{'step':'register_data_version','outcome':'rejected',
        'failure_category':'operation_arguments_invalid','because':'version.source_refs missing',
        'action_id':'a'*16,'finished_at':'old'}]
    writer=Writer()
    original=recorder.narrate(tmp_path,held,client=writer)
    assert len(writer.calls)==3
    repeated=copy.deepcopy(held)
    repeated['event_revision']='new-receipt'
    repeated['actions'][0].update(action_id='b'*16,finished_at='new',
        because='version.reason too long')
    assert recorder.narrate(tmp_path,repeated,client=writer)==original
    assert len(writer.calls)==3
    repeated['runtime_failure']={'category':'native_crash','evidence_id':'c'*32}
    repeated['event_revision']='native-crash'
    recorder.narrate(tmp_path,repeated,client=writer)
    assert len(writer.calls)==6


def test_writer_sees_recent_environment_facts_not_whole_install_inventory(tmp_path):
    held = snapshot(tmp_path)
    attempts = [{"ok": i > 5, "attempt_id": str(i)} for i in range(30)]
    held["provisioning"] = {"status": "completed", "attempts": attempts}
    held["installation_recovery_inventory"] = {"local_wheels": list(range(100)),
                                                "scan_complete": True}
    evidence = recorder._evidence(held)
    assert evidence["provisioning"]["attempt_count"] == 30
    assert evidence["provisioning"]["failed_attempt_count"] == 6
    assert evidence["provisioning"]["attempts"] == attempts[-8:]
    assert evidence["installation_recovery_inventory"]["local_wheel_count"] == 100
    assert held["provisioning"]["attempts"] == attempts


def test_recorder_has_no_tools_or_mcp_and_cannot_execute(tmp_path):
    profile = role_profile("recorder")
    args = _role_cli_args(profile, workspace=tmp_path, output=tmp_path,
                          python="python", model="fixture", max_budget_usd=.1)
    assert not profile.builtin_tools and not profile.mcp_tools
    assert args[args.index("--tools") + 1] == ""
    assert json.loads(args[args.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert "--allowedTools" not in args


def test_scheduler_capture_is_explicit_frozen_and_not_replayed(tmp_path, monkeypatch):
    output = tmp_path / "out"
    real = Preparation(repo=tmp_path, output=output, client=Writer(), scouting=tmp_path / "scout")
    path = measurement(output, "baseline", 1)
    held = snapshot(output)
    recorder._enqueue(output, held, [{"measurement_id": held["measurements"][0]["id"], "purpose": "观察失败行为"}])
    request = recorder.pending_requests(output)[0]
    calls = []

    class Research:
        def capture_demo(self, **kwargs):
            calls.append(kwargs)
            assert not recorder.pending_requests(output)  # capturing persisted before side effects
            return {"status": "captured", "attempt_id": "3" * 32}

    monkeypatch.setattr(real, "state", lambda: {"research_progress": {"status": "paused"}})
    monkeypatch.setattr(real, "_research_controller", lambda: (Research(), {}))
    result = real.do("review_report_demo", request_id=request["id"], decision="capture", reason="预留预算内展示")
    assert result["outcome"] == "captured" and len(calls) == 1
    assert calls[0]["settings"] == {"task": "demo"}
    assert real._step_review_report_demo(request_id=request["id"], decision="capture", reason="重复")["outcome"] == "rejected"
    recorder._enqueue(output, held, [{"measurement_id": held["measurements"][0]["id"], "purpose": "换一种措辞"}])
    assert not recorder.pending_requests(output)


def test_scheduler_decline_never_constructs_executor(tmp_path, monkeypatch):
    output = tmp_path / "out"
    real = Preparation(repo=tmp_path, output=output, client=Writer(), scouting=tmp_path / "scout")
    measurement(output, "baseline", 1)
    held = snapshot(output)
    recorder._enqueue(output, held, [{"measurement_id": held["measurements"][0]["id"], "purpose": "观察"}])
    request = recorder.pending_requests(output)[0]
    monkeypatch.setattr(real, "_research_controller", lambda: pytest.fail("must not launch"))
    assert real._step_review_report_demo(request_id=request["id"], decision="decline", reason="保留复评预算")["outcome"] == "declined"


def test_preparation_publishes_narration_and_survives_telemetry_refresh(tmp_path):
    output = tmp_path / "out"
    writer = Writer()
    real = Preparation(repo=tmp_path, output=output, client=writer, scouting=tmp_path / "scout")
    real.steps = [{"step": "build_the_environment", "outcome": "done", "because": "fixture"}]
    real._refresh_document(status="running", current="choose")
    path = output / "RUN.md"
    original = path.read_text()
    assert all(options['timeout'] == (45 if packet.get('phase') == 'review_report' else recorder.DEFAULT_TIMEOUT / 2)
               for packet, options in writer.calls)
    assert "最新发现" in original and original.index("当前进展") < original.index("技术详情")
    assert "最新发现" in (output / "RUN.html").read_text()
    trigger = recorder.read(output, "report/recorder_trigger.json")
    run_record.refresh_live_status(output, fallback_status="running", fallback_current="train")
    assert recorder.read(output, "report/recorder_trigger.json") == trigger
    assert "最新发现" in path.read_text()
    run_record.refresh_research_projection(output, "derived")
    assert path.read_text().count(recorder.START) == 1


def test_live_html_removes_active_content_and_external_images(tmp_path):
    path = tmp_path / "RUN.html"
    report_page.write_live_page(path,
        '# 中文报告\n\n<script>alert(1)</script>\n\n'
        '<img src="https://external.invalid/pixel" onerror="alert(2)">'
        '<a href="javascript:alert(3)">恶意链接</a>\n\n'
        '![本地曲线](report/charts/real.svg)')
    rendered = path.read_text()
    assert "<script" not in rendered and "onerror" not in rendered
    assert "https://external" not in rendered and "javascript:" not in rendered
    assert 'src="report/charts/real.svg"' in rendered


@pytest.mark.parametrize("block", ["changed", "heldout", "running", "exhausted"])
def test_capture_gates_reject_unsafe_or_unbudgeted_requests(tmp_path, monkeypatch, block):
    output = tmp_path / "out"
    real = Preparation(repo=tmp_path, output=output, client=Writer(), scouting=tmp_path / "scout")
    path = measurement(output, "baseline", 1)
    held = snapshot(output)
    recorder._enqueue(output, held, [{"measurement_id": held["measurements"][0]["id"], "purpose": "观察"}])
    request = recorder.pending_requests(output)[0]
    progress = {"status": "paused"}
    if block == "running":
        progress["status"] = "running"
    elif block == "exhausted":
        class Exhausted:
            def remaining(self):
                return 0
        real.budget = Exhausted()
    else:
        row = json.loads(path.read_text())
        row["confirmation" if block == "heldout" else "metric_value"] = True if block == "heldout" else 20
        atomic_json(path, row)
    monkeypatch.setattr(real, "state", lambda: {"research_progress": progress})
    monkeypatch.setattr(real, "_research_controller", lambda: pytest.fail("must not launch"))
    assert real._step_review_report_demo(request_id=request["id"], decision="capture", reason="观察")["outcome"] in {"blocked", "rejected"}


def test_runtime_report_includes_actual_measurement_table_and_charts(tmp_path):
    output = tmp_path / "out"
    client = Writer()
    real = Preparation(repo=tmp_path, output=output, client=client, scouting=tmp_path / "scout")
    measurement(output, "baseline", .2)
    measurement(output, "candidate", .4)
    real.steps = [{"step": "run_the_loop", "outcome": "done", "because": "synthetic fixture, not real benchmark"}]
    real._refresh_document(status="running", current="choose")
    text = (output / "RUN.md").read_text()
    assert "baseline" in text and "candidate" in text and "report/charts/" in text
    assert recorder.pending_requests(output)
    assert real.state()["report_demo_requests"]
    assert "review_report_demo" in real.state()["available"]
    assert "report/charts/" in (output / "RUN.html").read_text()


def test_generic_negative_metric_is_not_lost_or_mixed_with_other_units():
    source = {"measurements": [{"record": {"label": "a", "ok": True,
               "metric_value": -2, "metric": {"name": "energy", "direction": "minimize"}}}]}
    assert report_page._points(source) == [("a", -2)]
    ET.fromstring(report_page.curve_svg(report_page._points(source)))
    source["measurements"].append({"record": {"label": "b", "ok": True, "metric_value": 8,
                                              "metric": {"name": "reward"}}})
    assert report_page._points(source) == []


def test_training_plot_is_frozen_to_its_snapshot_and_changed_source_is_withheld(tmp_path):
    held = snapshot(tmp_path)
    path = tmp_path / "report/telemetry/train.svg"
    path.parent.mkdir(parents=True)
    content = '<svg xmlns="http://www.w3.org/2000/svg"><text>loss</text></svg>'
    path.write_text(content)
    held["telemetry_attempts"] = [{"chart_ref": "report/telemetry/train.svg"}]
    expected = hashlib.sha256(content.encode()).hexdigest()
    held["source_revision_vector"]["report/telemetry/train.svg"] = expected
    document = recorder.render(tmp_path, held, {})
    assert f"report/charts/telemetry-{expected}.svg" in document
    path.write_text('<svg/>')
    assert "快照暂不一致" in recorder.render(tmp_path, held, {})
    assert (tmp_path / f"report/charts/telemetry-{expected}.svg").read_text() == content
