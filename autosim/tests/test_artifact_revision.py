import json
import sys
from pathlib import Path

import pytest

from autosim.research import artifact_revision as revision
from autosim.research.common import read_json
from tests.test_prepare import preparation


def request(pattern='generated/new_episodes'):
    return {'pattern': pattern, 'reason': 'native collector writes this complete dataset root',
            'source_refs': [{'ref': 'collect.py', 'quote': 'OUTPUT = "generated/new_episodes"'}]}


class Reviewer:
    def __init__(self, fault=None):
        self.fault = fault
        self.calls = []

    def chat_with_metadata(self, system, user, **kwargs):
        self.calls.append((system, user, kwargs))
        verdict = {'artifact_scope_supported': True, 'source_output_connection_supported': True,
                   'new_native_verification_required': True, 'why': 'Exact native source supports the root',
                   'citations': [{'source_id': 'review-source-1'}]}
        if self.fault:
            verdict[self.fault] = False
        return json.dumps(verdict), {}


def setup(tmp_path):
    repo = tmp_path / 'checkout'
    repo.mkdir()
    (repo / 'collect.py').write_text('OUTPUT = "generated/new_episodes"\n')
    return repo


@pytest.mark.parametrize('pattern', ['', '.', '././', '..', '../data', 'data/../other',
                                    '/tmp/data', '*', '**/data', 'data/**bad', 'data\\new', '\x00'])
def test_unsafe_scope_never_reaches_review(tmp_path, pattern):
    repo = setup(tmp_path)
    client = Reviewer()
    with pytest.raises(ValueError):
        revision.review(tmp_path, repo, client, 'collect', {}, request(pattern))
    assert client.calls == []
    assert not (tmp_path / 'artifact_revisions').exists()


@pytest.mark.parametrize('stage', ['', 'arbitrary'])
def test_unknown_stage_contract_cannot_be_changed(tmp_path, stage):
    with pytest.raises(ValueError, match='native output stage'):
        revision.validate(stage, request())


@pytest.mark.parametrize('stage', ['train', 'evaluate'])
def test_prebaseline_policy_score_locator_requires_preserved_native_semantics(tmp_path, stage):
    repo = setup(tmp_path)
    with pytest.raises(revision.ArtifactRevisionRefused):
        revision.review(tmp_path, repo, Reviewer(), stage, {'artifact':'old'}, request())
    class ExactReviewer(Reviewer):
        def chat_with_metadata(self, *args, **kwargs):
            content, meta = super().chat_with_metadata(*args, **kwargs)
            verdict = json.loads(content)
            verdict['native_semantics_preserved'] = True
            return json.dumps(verdict), meta
    before = {'artifact':'old', 'entrypoint':'collect.py', 'metric':'native_success'}
    selected, ref = revision.review(tmp_path, repo, ExactReviewer(), stage, before, request())
    assert {k:v for k,v in selected.items() if k not in {'artifact','artifact_revision_ref'}} == {
        k:v for k,v in before.items() if k!='artifact'}
    assert before['artifact'] == 'old'
    record = read_json(tmp_path / ref)
    assert record['scope'] == 'prebaseline_output_locator_only'
    assert record['status'] == 'reviewed_unverified'


@pytest.mark.parametrize('marker', ['controller_session.json', 'comparison_protocol.json',
                                  'research_report.json', 'measurements/baseline.json',
                                  'measurements/unscored_trial.json'])
@pytest.mark.parametrize('stage', ['train', 'evaluate'])
def test_formal_history_blocks_policy_score_locator_revision_before_model(tmp_path, marker, stage):
    from autosim.research.common import atomic_json
    repo = setup(tmp_path)
    atomic_json(tmp_path/'research/derived'/marker, {'untouched':True})
    client = Reviewer()
    with pytest.raises(ValueError, match='pre-baseline only'):
        revision.review(tmp_path, repo, client, stage, {'artifact':'old'}, request())
    assert not client.calls


def test_baseline_created_during_review_cannot_authorize_locator(tmp_path):
    from autosim.research.common import atomic_json
    repo = setup(tmp_path)
    class RacingReviewer(Reviewer):
        def chat_with_metadata(self, *args, **kwargs):
            result = super().chat_with_metadata(*args, **kwargs)
            atomic_json(tmp_path/'research/derived/measurements/baseline.json', {'untouched':True})
            return result
    with pytest.raises(ValueError, match='pre-baseline only'):
        revision.review(tmp_path, repo, RacingReviewer(), 'evaluate', {'artifact':'old'}, request())
    assert not (tmp_path/'artifact_revisions').exists()


def test_prepare_refuses_postbaseline_locator_without_running_verifier(tmp_path):
    from autosim.research.common import atomic_json
    real = preparation(tmp_path)
    atomic_json(real.output/'research'/real.run_id/'controller_session.json', {'status':'paused'})
    result = real._step_derive_a_command(stage='evaluate',artifact_revision=request())
    assert result['failure_category'] == 'artifact_revision_invalid'
    assert 'pre-baseline only' in result['because']


@pytest.mark.parametrize('stage', ['train', 'evaluate'])
@pytest.mark.parametrize('baseline_race', [False, True])
def test_prepare_uses_reviewed_locator_with_fresh_native_verification(tmp_path, monkeypatch, stage, baseline_race):
    from autosim.research import prepare as module
    from autosim.research.common import atomic_json
    real = preparation(tmp_path)
    (tmp_path/'collect.py').write_text('OUTPUT = "generated/new_episodes"\n')
    class ExactReviewer(Reviewer):
        def chat_with_metadata(self, *args, **kwargs):
            text, meta = super().chat_with_metadata(*args, **kwargs)
            verdict = json.loads(text)
            verdict['native_semantics_preserved'] = True
            return json.dumps(verdict), meta
    real.client, real.interpreter = ExactReviewer(), Path(sys.executable)
    real.execution = {'stages':{stage:{'available':True,'entrypoint':'collect.py','artifact':'old'}}}
    real.declaration = {'optimization_space':{'training':[
        {'name':'steps','kind':'integer','low':1,'high':2,'default':1,'description':'updates'}]}}
    monkeypatch.setattr(real, '_confirmation_admission_problem', lambda: None)
    monkeypatch.setattr(module.execution_derive, 'checkpoint_for_verification',lambda *a, **k:{'path':''})
    seen=[]
    def native(*args, **kwargs):
        seen.append(args[2].copy())
        assert kwargs['unique_verification_outputs'] is True
        if baseline_race:
            atomic_json(real.output/'research'/real.run_id/'controller_session.json',{'status':'paused'})
        return 'def stage_argv_'+stage+'(i): return [i["python"], "collect.py"]', {}, [], args[2]
    monkeypatch.setattr(module.execution_derive,'make_runnable',native)
    result=real._step_derive_a_command(stage=stage,artifact_revision=request())
    assert len(seen)==1 and seen[0]['artifact']=='generated/new_episodes'
    if baseline_race:
        assert result['failure_category']=='artifact_revision_context_changed'
        assert stage not in real.stages
        assert not (real.output/'derived_stages.json').exists()
    else:
        assert result['outcome']=='done'
        assert read_json(real.output/'derived_stages.json')[stage]['row']['artifact']=='generated/new_episodes'
        assert real.execution['stages'][stage]['artifact']=='old'


@pytest.mark.parametrize('pattern', ['generated/new_episodes', 'episodes/export-*', 'raw/episode.h5',
                                    'generated/new_episodes/meta/info.json'])
def test_agent_explicitly_selects_generic_scope_without_layout_inference(tmp_path, pattern):
    repo = setup(tmp_path)
    before = {'artifact': 'old/marker.json', 'invocation': 'collect.py'}
    selected, ref = revision.review(tmp_path, repo, Reviewer(), 'collect', before, request(pattern))
    assert selected['artifact'] == pattern
    assert selected['artifact_revision_ref'] == ref
    assert before == {'artifact': 'old/marker.json', 'invocation': 'collect.py'}
    record = read_json(tmp_path / ref)
    assert record['status'] == 'reviewed_unverified'
    assert record['revision']['source_refs'][0]['sha256']
    assert not (tmp_path / 'derived_stages.json').exists()
    assert not (tmp_path / 'execution.json').exists()


@pytest.mark.parametrize('check', ['artifact_scope_supported', 'source_output_connection_supported',
                                 'new_native_verification_required'])
def test_independent_refusal_cannot_be_overridden(tmp_path, check):
    repo = setup(tmp_path)
    client = Reviewer(check)
    with pytest.raises(ValueError, match='not approved'):
        revision.review(tmp_path, repo, client, 'collect', {}, request())
    assert client.calls[0][2]['read_only'] is True
    record = read_json(next((tmp_path / 'artifact_revisions').glob('*.json')))
    assert record['status'] == 'review_rejected'
    assert record['review'][check] is False


def test_fake_source_is_rejected_before_model_and_source_race_after_review(tmp_path):
    repo = setup(tmp_path)
    client = Reviewer()
    spec = request()
    spec['source_refs'][0]['quote'] = 'not in source'
    with pytest.raises(ValueError):
        revision.review(tmp_path, repo, client, 'collect', {}, spec)
    assert not client.calls
    class Changing(Reviewer):
        def chat_with_metadata(self, *args, **kwargs):
            result = super().chat_with_metadata(*args, **kwargs)
            (repo / 'collect.py').write_text('OUTPUT = "elsewhere"\n')
            return result
    with pytest.raises(ValueError):
        revision.review(tmp_path, repo, Changing(), 'collect', {}, request())
    assert not (tmp_path / 'artifact_revisions').exists()


def test_reviewed_row_reaches_new_native_verification_and_formal_backend(tmp_path, monkeypatch):
    from autosim.research import prepare as module
    real = preparation(tmp_path)
    (tmp_path / 'collect.py').write_text('OUTPUT = "generated/new_episodes"\n')
    real.client = Reviewer()
    real.interpreter = Path(sys.executable)
    real.declaration = {'optimization_space': {'collection': [], 'training': [
        {'name':'steps', 'kind':'integer', 'low':1, 'high':2, 'default':1,
         'description':'native updates'}]}}
    real.execution = {'stages': {'collect': {'available': True, 'entrypoint': 'collect.py',
                                            'artifact': 'old/marker.json'}}}
    monkeypatch.setattr(module.execution_derive, 'checkpoint_for_verification', lambda *a, **k: {'path': ''})
    seen = []
    source = 'def stage_argv_collect(i): return [i["python"], "collect.py"]'
    def native(*args, **kwargs):
        seen.append(args[2].copy())
        assert args[2]['artifact'] == 'generated/new_episodes'
        assert kwargs['unique_verification_outputs'] is True
        return source, {}, [], args[2]
    monkeypatch.setattr(module.execution_derive, 'make_runnable', native)
    result = real._step_derive_a_command(stage='collect', artifact_revision=request())
    assert result['outcome'] == 'done'
    assert result['formal_producer_status'] == 'interface_verified_not_formal_acquisition'
    assert result['next_operation'] == 'submit_native_job'
    assert len(seen) == 1
    kept = read_json(real.output / 'derived_stages.json')['collect']['row']
    assert kept['artifact'] == 'generated/new_episodes'
    assert real._research_controller()[0].backend.artifact_pattern('collect') == kept['artifact']
    assert kept['artifact_revision_ref'] in result['evidence_refs']
    assert real.execution['stages']['collect']['artifact'] == 'old/marker.json'


def test_main_sees_formal_acquisition_distinct_from_successful_probe(tmp_path):
    real = preparation(tmp_path)
    real.stages['collect'] = 'def stage_argv_collect(i): return []'
    real.verified['collect'] = {'artifact':'generated/new_episodes'}
    contract = real._decision_facts()['executor_input_contracts']['data_acquisition']
    assert contract['collect_interface_installed'] is True
    assert contract['installed_artifact_pattern'] == 'generated/new_episodes'
    assert contract['formal_producer_operation'] == 'submit_native_job'
    assert 'empty artifact there does not invalidate' in contract['guidance']


def test_native_failure_does_not_install_reviewed_scope(tmp_path, monkeypatch):
    from autosim.research import prepare as module
    real = preparation(tmp_path)
    (tmp_path / 'collect.py').write_text('OUTPUT = "generated/new_episodes"\n')
    real.client = Reviewer()
    real.interpreter = Path(sys.executable)
    real.execution = {'stages': {'collect': {'available': True, 'entrypoint': 'collect.py', 'artifact': 'old'}}}
    monkeypatch.setattr(module.execution_derive, 'checkpoint_for_verification', lambda *a, **k: {'path': ''})
    monkeypatch.setattr(module.execution_derive, 'make_runnable', lambda *a, **k: (None, {}, [], a[2]))
    result = real._step_derive_a_command(stage='collect', artifact_revision=request())
    assert result['outcome'] == 'no runnable command'
    assert 'collect' not in real.stages
    assert not (real.output / 'derived_stages.json').exists()
    assert len(list((real.output / 'artifact_revisions').glob('*.json'))) == 1


@pytest.mark.parametrize('option', ['reuse_verified', 'draft_ref', 'blueprint_ref'])
def test_cached_verification_cannot_grant_revised_scope(tmp_path, option):
    real = preparation(tmp_path)
    result = real._step_derive_a_command(stage='collect', artifact_revision=request(),
                                        **{option: True if option == 'reuse_verified' else 'old'})
    assert result['failure_category'] == 'artifact_revision_invalid'


def test_nested_contract_publishes_exact_request_shape(tmp_path):
    from autosim.research.operation_contracts import schema
    spec = schema(preparation(tmp_path)._step_derive_a_command)['properties']['artifact_revision']
    assert spec['required'] == ['pattern', 'reason', 'source_refs']
    assert spec['additionalProperties'] is False


def test_native_revision_cannot_install_a_different_scope(tmp_path, monkeypatch):
    from autosim.research import prepare as module
    real = preparation(tmp_path)
    (tmp_path / 'collect.py').write_text('OUTPUT = "generated/new_episodes"\n')
    real.client, real.interpreter = Reviewer(), Path(sys.executable)
    real.execution = {'stages': {'collect': {'available': True, 'entrypoint':'collect.py', 'artifact':'old'}}}
    monkeypatch.setattr(module.execution_derive, 'checkpoint_for_verification', lambda *a, **k: {'path': ''})
    monkeypatch.setattr(module.execution_derive, 'make_runnable',
                        lambda *a, **k: ('builder', {}, [], {**a[2], 'artifact':'generated'}))
    result = real._step_derive_a_command(stage='collect', artifact_revision=request())
    assert result['failure_category'] == 'artifact_revision_context_changed'
    assert 'collect' not in real.stages
    assert not (real.output / 'derived_stages.json').exists()


def test_symlink_review_store_is_not_followed(tmp_path):
    repo = setup(tmp_path)
    target = tmp_path / 'elsewhere'
    target.mkdir()
    (tmp_path / 'artifact_revisions').symlink_to(target, target_is_directory=True)
    client = Reviewer()
    with pytest.raises(ValueError, match='unsafe'):
        revision.review(tmp_path, repo, client, 'collect', {}, request())
    assert not client.calls and list(target.iterdir()) == []


def test_real_cpu_collector_reports_only_explicit_fresh_scope_and_old_marker_stays_file(tmp_path):
    import subprocess
    import time
    from autosim.research.declarative_backend import DeclarativeBackend
    repo = setup(tmp_path)
    script = repo / 'collect.py'
    script.write_text(script.read_text() + '\nfrom pathlib import Path\n'
        'root = Path(OUTPUT)\nroot.mkdir(parents=True)\n'
        '(root / "payload.bin").write_bytes(b"real generated bytes")\n'
        '(root / "manifest.json").write_text("{}")\n')
    old_row = {'artifact': 'generated/new_episodes/manifest.json'}
    selected, _ = revision.review(tmp_path, repo, Reviewer(), 'collect', old_row, request())
    started = time.time()
    subprocess.run([sys.executable, str(script)], cwd=repo, check=True, capture_output=True)
    def check(row):
        return DeclarativeBackend(repo=repo, answer={'stages': {'collect': row}}, sources={}).check_artifact(
            'collect', repo, since=started)
    actual = check(selected)
    assert actual['matched'] == 1 and actual['freshness_checked'] is True
    assert actual['candidate_paths'] == ['generated/new_episodes']
    assert check(old_row)['candidate_paths'] == ['generated/new_episodes/manifest.json']
    # The actual native bytes, not the review, decide whether a scope was produced.
    assert check({'artifact': 'generated/not_produced'})['matched'] == 0
