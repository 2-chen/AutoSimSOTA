"""Actual failures are not readiness; unchanged failures do not buy another repair turn."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autosim.research import provision as pv
from autosim.research import environment_observation as observation
from autosim.research import repair_intents
from autosim.research.common import atomic_json
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.installation_recovery import inventory, review_boundary
from autosim.research.prepare import Preparation


@pytest.mark.parametrize('passed', [False, True])
def test_interpreter_alone_is_not_a_passing_environment(tmp_path, monkeypatch, passed):
    prep = Preparation(repo=tmp_path / 'repo', output=tmp_path / 'run',
                       client=object(), scouting=tmp_path / 'scouting')
    prep.repo.mkdir(exist_ok=True)
    prep.declaration = {'assets':{}}
    monkeypatch.setattr(prep, '_surveyed_runnable_stages', lambda: {'train':{'entrypoint':'train.py'}})
    monkeypatch.setattr(prep, '_select_execution_path', lambda **_: {'stages':['train'], 'asset_keys':[], 'why':'fixture'})
    monkeypatch.setattr(pv, 'env_python', lambda *_: Path(sys.executable))
    monkeypatch.setattr(pv, 'build', lambda *a, **k: {'verdict':{'passed':passed, 'reason':'probe failed'},
        'latest_failure':{'kind':'probe', 'evidence_id':'a'*32, 'probe':'import task'}})
    result = prep._step_build_the_environment()
    assert result['outcome'] == ('done' if passed else 'failed')
    if not passed:
        assert result['failure_domain'] == 'native_probe'
        assert result['evidence_id'] == 'a'*32


@pytest.mark.parametrize('cooperative', [None, 1])
def test_failed_probe_is_not_replayed_after_rejected_repair(tmp_path, monkeypatch, cooperative):
    repo = tmp_path / 'repo'
    repo.mkdir()
    calls = []
    def failed(probe, **kw):
        calls.append(probe)
        log = tmp_path / 'failed.log'
        log.write_text('ModuleNotFoundError: wrong import name')
        capture_attempt_evidence(tmp_path, attempt_id='a'*32, log=log,
            receipt_ref='fixture.json', status='failed', returncode=1, termination_reason='failed')
        return {'command':probe, 'ok':False, 'returncode':1, 'evidence_id':'a'*32}
    repairs = []
    def reject(*a, **k):
        repairs.append(a[3])
        return [], None
    monkeypatch.setattr(pv, 'probe_environment', failed)
    monkeypatch.setattr(pv, 'resume', reject)
    seed = {'python':sys.executable, 'templates':[], 'probes':['{python} -c "import wrong"']}
    monkeypatch.setattr(pv, 'plan', lambda *a, **k: {'python':sys.executable,
        'commands':[], 'probes':seed['probes'], 'assets':{}})
    for _ in range(3):
        result = pv.build(repo, client=object(), prefix=tmp_path / 'env', output=tmp_path,
            python=sys.executable, manifests={}, assets={}, max_operations=cooperative,
            max_rounds=1, seed=seed)
        assert result['status'] == 'yielded' and result['new_native_operation'] is False
        # Exercise a pre-upgrade cursor lacking the new failed_probe field.
        cursor = json.loads((tmp_path / 'provision_cursor.json').read_text())
        cursor.pop('failed_probe', None)
        atomic_json(tmp_path / 'provision_cursor.json', cursor)
    assert len(calls) == 1
    assert all(row['probe'] == seed['probes'][0] for row in repairs)


def test_repair_intent_requires_changed_input_but_allows_real_source_change(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    source = repo / 'task.py'
    source.write_text('VALUE=1')
    kwargs = dict(failure={'evidence_id':'a'*32, 'probe':'import task'}, commands=[],
                  probes=['import task'], manifests={}, observation={'identity':'env'}, repo=repo)
    first, allowed = repair_intents.begin(tmp_path, **kwargs, proposal={'commands':['repair'], 'reasoning':'one'})
    assert allowed
    second, allowed = repair_intents.begin(tmp_path, **kwargs, proposal={'commands':['repair'], 'reasoning':'new prose'})
    assert first == second and not allowed
    source.write_text('VALUE=2')
    third, allowed = repair_intents.begin(tmp_path, **kwargs, proposal={'commands':['repair']})
    assert allowed and third != first
    kwargs['observation'] = {'identity':'new installed metadata'}
    assert repair_intents.begin(tmp_path, **kwargs, proposal={'commands':['repair']})[1]


def test_resume_unchanged_failure_does_not_call_model_again(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    class Client:
        output = tmp_path
        calls = 0
        def chat_with_metadata(self, *a, **k):
            self.calls += 1
            return json.dumps({'commands':['false']}), {}
    client = Client()
    monkeypatch.setattr(observation, 'observe', lambda *a: {'identity':'env'})
    monkeypatch.setattr(pv, 'platform_facts', lambda: {})
    transcript = []
    for _ in range(3):
        assert pv.resume(client, repo, [], {'command':'false', 'evidence_id':'a'*32},
            manifests={}, transcript=transcript, values={}, attempts=1,
            current_probes=['import task'], pending_commands=['false']) == ([], None)
    assert client.calls == 1
    assert 'Unchanged recovery intention' in transcript[-1]['reason']


def test_empty_local_inventory_does_not_prove_unbuildable(tmp_path):
    resources = inventory(tmp_path)
    with pytest.raises(ValueError, match='assessment'):
        review_boundary(object(), resources=resources, reason='nothing local', assessment=None, failure={})


def test_metadata_prompt_is_small_and_omission_not_absence():
    value = {'identity':'env', 'complete':True, 'evidence_id':'a'*32,
        'packages':[{'distribution':'native-engine', 'version':'1', 'import_names':['native']},
                    *[{'distribution':f'other-{i}', 'version':'1', 'import_names':[]} for i in range(200)]]}
    view = observation.prompt_view(value, 'import native; dependency native-engine')
    assert view['packages'] == value['packages'][:1]
    assert view['observed_package_count'] == 201 and not view['projection_complete']
    assert 'not evidence of absence' in view['selection']


def test_metadata_cache_requires_verified_receipt_and_refreshes_on_import_metadata(tmp_path, monkeypatch):
    from autosim.research import native_context, agent_runtime
    prefix = tmp_path / 'env'
    python = prefix / 'bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('fixture')
    metadata = prefix / 'lib/python3.11/site-packages/native_engine-1.dist-info'
    metadata.mkdir(parents=True)
    (metadata / 'METADATA').write_text('Name: native-engine\nVersion: 1')
    monkeypatch.setattr(native_context, 'load_context', lambda *a: {'identity':'ctx', 'interpreter':str(python)})
    calls = []
    class Executor:
        def __init__(self, **kw):
            assert kw['allow_commands'] is False
        def native_probe(self, args):
            calls.append(args)
            evidence_id = str(len(calls))*32
            log = tmp_path / f'metadata_{len(calls)}.log'
            stdout = json.dumps({'packages':[{'distribution':'native-engine', 'version':'1',
                'import_names':['native']}], 'complete':True})
            log.write_text(stdout)
            capture_attempt_evidence(tmp_path, attempt_id=evidence_id, log=log,
                receipt_ref='fixture.json', status='completed', returncode=0, termination_reason='completed')
            return {'evidence_id':evidence_id, 'evidence_ref':str(log.name), 'returncode':0, 'stdout':stdout}
    monkeypatch.setattr(agent_runtime, '_McpExecutor', Executor)
    first = observation.observe(tmp_path, tmp_path / 'repo')
    assert first['status'] == 'observed'
    assert observation.observe(tmp_path, tmp_path / 'repo') == first and len(calls) == 1
    (metadata / 'top_level.txt').write_text('native')
    assert observation.observe(tmp_path, tmp_path / 'repo')['identity'] != first['identity']
    assert len(calls) == 2
    (tmp_path / 'metadata_2.log').write_text('tampered')
    assert observation.observe(tmp_path, tmp_path / 'repo')['status'] == 'unavailable'
    assert len(calls) == 2


def test_actual_selected_environment_metadata_is_sealed_without_importing_package(tmp_path):
    from autosim.research.native_context import publish_context
    from autosim.research.common import run_local_environment
    output = tmp_path / 'run'
    repo = output / 'checkout'
    repo.mkdir(parents=True)
    atomic_json(output / 'workspace_snapshot.json', {'destination':str(repo)})
    subprocess.run(['/usr/bin/python3', '-m', 'venv', '--without-pip', str(output / 'env')], check=True)
    publish_context(output, repo, output / 'env/bin/python', run_local_environment(output, {}))
    result = observation.observe(output, repo)
    assert result['status'] == 'observed', result
    assert result['complete'] and result['packages'] == []
    assert result['evidence_id']
    assert observation.observe(output, repo) == result
