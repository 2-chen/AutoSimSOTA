import json

from tools import external_baseline_trial as pilot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_benchmark_submission_is_found_without_restarting_native_work(tmp_path):
    write(tmp_path / 'benchmark/pilot_results.json', {'terminal_boundary': True, 'baseline': None})
    result = pilot.submitted_result(tmp_path)
    assert result['status'] == 'ready'
    assert result['paths'] == ['benchmark/pilot_results.json']
    assert result['value']['baseline'] is None


def test_conflicting_submissions_are_not_silently_picked(tmp_path):
    write(tmp_path / 'pilot_results.json', {'baseline': 0})
    write(tmp_path / 'benchmark/pilot_results.json', {'baseline': .5})
    assert pilot.submitted_result(tmp_path)['status'] == 'conflict'
    assert json.loads((tmp_path / 'pilot_results.json').read_text()) == {'baseline': 0}


def test_equivalent_submissions_are_accepted_once(tmp_path):
    for relative in ['pilot_results.json', 'benchmark/pilot_results.json']:
        write(tmp_path / relative, {'completed': True})
    result = pilot.submitted_result(tmp_path)
    assert result['status'] == 'ready' and len(result['paths']) == 2


def test_bad_submission_is_not_a_request_to_run_more_experiments(tmp_path):
    (tmp_path / 'pilot_results.json').write_text('{broken')
    assert pilot.submitted_result(tmp_path)['status'] == 'invalid'


def test_symlinked_report_or_parent_is_not_followed(tmp_path):
    write(tmp_path / 'elsewhere/pilot_results.json', {'completed': True})
    (tmp_path / 'benchmark').symlink_to(tmp_path / 'elsewhere', target_is_directory=True)
    assert pilot.submitted_result(tmp_path)['status'] == 'invalid'


def test_metric_discovery_includes_native_formats_not_invalid_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'STUDY', tmp_path)
    trial = tmp_path / 'trial'
    for relative in ['benchmark/outputs/baseline/summary.json',
                     'benchmark/research_out/eval/evaluation_metrics.json',
                     'outputs/_invalid_attempt/summary.json',
                     'benchmark/research_out/.hidden/metrics.json']:
        write(trial / relative, {})
    write(tmp_path / 'resource/metrics.json', {})
    (trial / 'outputs/external').symlink_to(tmp_path / 'resource', target_is_directory=True)
    assert pilot.native_metrics(trial) == [
        'trial/benchmark/outputs/baseline/summary.json',
        'trial/benchmark/research_out/eval/evaluation_metrics.json',
    ]


def test_render_links_report_in_benchmark_without_changing_old_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'STUDY', tmp_path)
    write(tmp_path / 'manifest.json', {'trials': [
        {'name': 'trial', 'system': 'example', 'benchmark': 'native', 'allow_images': True}]})
    write(tmp_path / 'trial/state.json', {'status': 'needs_followup'})
    path = tmp_path / 'trial/benchmark/RESEARCH.md'
    path.parent.mkdir(parents=True)
    path.write_text('原始研究记录')
    pilot.render()
    rendered = (tmp_path / 'RUN.md').read_text()
    assert 'trial/benchmark/RESEARCH.md' in rendered
    assert 'needs_followup' in rendered
    assert '未经独立核验不代表有效分数' in rendered
    assert path.read_text() == '原始研究记录'
