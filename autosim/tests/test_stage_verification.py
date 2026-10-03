import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from autosim.research import stage_verification as sv, common
from autosim.research.common import atomic_json
from autosim.research.workspace_resources import validate_bindings


def bound_repo(tmp_path):
    original = tmp_path/'original'
    original.mkdir()
    source = original/'dataset'
    (source/'task'/'meta').mkdir(parents=True)
    (source/'task'/'meta'/'info.json').write_text('private data content')
    output = tmp_path/'run'
    repo = output/'checkout'
    (repo/'attached').mkdir(parents=True)
    bindings = validate_bindings(original, repo, [{'target':'attached','source':str(source)}])
    atomic_json(output/'workspace_snapshot.json', {'source':str(original),
        'destination':str(repo), 'resource_bindings':bindings})
    return repo, output


def test_resource_context_supplies_real_names_without_reading_data(tmp_path):
    repo, output = bound_repo(tmp_path)
    context = sv.resource_context(repo)
    row = context['resources'][0]
    assert row['target'] == 'attached'
    assert row['entries'][0]['name'] == 'task'
    assert row['children'][0]['entries'][0]['name'] == 'meta'
    assert 'private data content' not in json.dumps(context)
    assert 'source' not in row
    assert 'dataset' not in context  # Agent still chooses the consumer path.
    assert (output/row['evidence_ref']).is_file()


def test_missing_input_check_uses_bound_source_not_empty_placeholder(tmp_path):
    repo, output = bound_repo(tmp_path)
    assert sv.missing_checkout_paths([str(repo/'attached/task/meta/info.json')],
                                    repo=repo, output=output/'v') == []
    assert sv.missing_checkout_paths([str(repo/'old_default'),
        'data="'+str(repo/'attached/missing')+'"', str(output/'v/future_model')],
        repo=repo, output=output/'v') == ['attached/missing', 'old_default']


def test_future_output_inside_checkout_is_not_treated_as_input(tmp_path):
    repo, _ = bound_repo(tmp_path)
    assert sv.missing_checkout_paths([str(repo/'results/new_model')],
                                    repo=repo, output=repo/'results') == []


def test_running_native_tail_is_bounded_unsealed_and_never_a_verdict(tmp_path):
    identity = 'e' * 32
    directory = tmp_path/'derivation_native'/identity
    directory.mkdir(parents=True)
    atomic_json(directory/'receipt.json', {'id':identity, 'stage':'evaluate',
        'status':'running', 'started_at':common.now()})
    (directory/'native.log').write_text('old output\n' + 'x'*6000 + '\nEpisode 7/100\n')
    row = sv.recent_attempts(tmp_path)[0]
    assert row['status'] == 'running_unverified' and row['evidence_verified'] is False
    assert len(row['live_telemetry']['tail'].encode()) <= 4096
    assert 'Episode 7/100' in row['live_telemetry']['tail']
    assert 'old output' not in row['live_telemetry']['tail']
    assert 'returncode' not in row and 'evidence_ref' not in row
    assert row['elapsed_wall_seconds'] >= 0
    from autosim.research.recorder import native_table
    panel = native_table([row])
    assert '未封存' in panel and '未结算' in panel
    assert '命令退出成功' not in panel


def test_running_log_symlink_cannot_expose_external_text(tmp_path):
    identity = 'f' * 32
    directory = tmp_path/'derivation_native'/identity
    directory.mkdir(parents=True)
    atomic_json(directory/'receipt.json', {'id':identity, 'stage':'evaluate','status':'running'})
    external = tmp_path/'private.txt'
    external.write_text('never expose this')
    (directory/'native.log').symlink_to(external)
    row = sv.recent_attempts(tmp_path)[0]
    assert 'live_telemetry' not in row
    assert 'never expose this' not in json.dumps(row)


@pytest.mark.parametrize('timeout', [False, True])
def test_native_output_is_streamed_and_sealed_on_success_or_timeout(tmp_path, monkeypatch, timeout):
    repo = tmp_path/'checkout'
    repo.mkdir()
    monkeypatch.setattr(common, 'isolated_argv', lambda argv, **kw: argv)
    script = 'import time; print("native progress reached", flush=True); '
    script += 'time.sleep(10)' if timeout else 'print("optimizer_step=1")'
    entered = []
    @contextmanager
    def window(seconds):
        entered.append(seconds)
        try:
            yield seconds, False, {}
        finally:
            entered.append('closed')
    kwargs = dict(repo=repo, output=tmp_path, cwd=repo, env={},
                  timeout=.4 if timeout else 5, stdin=None, window=window)
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            sv.run_native([sys.executable,'-c',script], **kwargs)
        assert 'native progress reached' in error.value.output
    else:
        result = sv.run_native([sys.executable,'-c',script], **kwargs)
        assert result.returncode == 0
    assert entered[-1] == 'closed'
    receipt = json.loads(next((tmp_path/'derivation_native').glob('*/receipt.json')).read_text())
    assert receipt['status'] == ('timed_out' if timeout else 'completed')
    assert 'native progress reached' in (tmp_path/receipt['log_ref']).read_text()
    assert (tmp_path/receipt['evidence_ref']).is_file()
    rows = sv.recent_attempts(tmp_path)
    assert rows[0]['evidence_verified'] is True
    assert rows[0]['evidence_id'] == receipt['id']
    assert 'not baseline' in rows[0]['authority']
    (tmp_path/receipt['log_ref']).write_text('altered')
    assert sv.recent_attempts(tmp_path) == []


def test_argv_generator_receives_explicit_resources_and_actual_input_slots(tmp_path):
    from autosim.research.execution_derive import generate_argv
    class Client:
        def chat_with_metadata(self, system, payload, **kw):
            data = json.loads(payload)
            assert data['native_resource_context']['resources'][0]['target'] == 'attached'
            assert data['verification_inputs']['dataset'] == ''
            return json.dumps({'source':'def stage_argv_train(i): return [i["python"], i["repo"]+"/train.py", i["repo"]+"/attached/task"]'}), {}
    source, _, _ = generate_argv(Client(), 'train', entrypoint='train.py', invocation='python train.py <data>',
        repository_files=['train.py'], inputs_for_verify={'dataset':''},
        verification_context={'resources':[{'target':'attached'}]})
    assert source and '/attached/task' in source


@pytest.mark.parametrize('cause', ['capture', 'orphan'])
def test_native_zero_exit_failure_seals_specific_internal_cause(tmp_path, monkeypatch, cause):
    import autosim.research.process_executor as executor
    repo = tmp_path/'checkout'
    repo.mkdir()
    monkeypatch.setattr(common, 'isolated_argv', lambda argv, **kw: argv)
    failure = ValueError('capture callback failed') if cause == 'capture' else None
    monkeypatch.setattr(executor, 'run_process_stream', lambda *a, **kw:
        executor.ProcessAttempt(True, 0, '100 episodes completed', 'diagnostic',
            error=failure, orphaned_children=cause == 'orphan',
            containment_mode='pid_namespace',
            cleanup_diagnostics={'status':'terminated_orphans' if cause == 'orphan' else 'clean'}))
    kwargs = dict(repo=repo, output=tmp_path, cwd=repo, env={}, timeout=5, stdin=None)
    if failure:
        with pytest.raises(ValueError) as error:
            sv.run_native(['true'], **kwargs)
        assert (tmp_path/error.value.evidence_ref).is_file()
    else:
        result = sv.run_native(['true'], **kwargs)
        assert result.returncode == 125 and result.native_returncode == 0
        assert result.native_cleanup['orphaned_children']
    receipt = json.loads(next((tmp_path/'derivation_native').glob('*/receipt.json')).read_text())
    assert receipt['returncode'] == 0 and receipt['status'] == 'failed'
    assert receipt['orphaned_children'] == (cause == 'orphan')
    assert receipt['effective_returncode'] == (125 if cause == 'orphan' else 0)
    assert receipt['stdout_tail'] == '100 episodes completed'
    assert receipt['capture_error'] == ({'type':'ValueError','message':'capture callback failed'}
                                        if failure else None)
    row = sv.recent_attempts(tmp_path)[0]
    assert row['evidence_verified']
    assert '[AutoSim execution audit; not a metric]' in row['excerpt']
    assert ('capture callback failed' if failure else 'terminated_orphans') in row['excerpt']
    assert 'not an artifact/identity validation verdict' in row['receipt_scope']


def test_preflight_rejection_does_not_anchor_unseen_command_shape():
    from autosim.research.execution_derive import generate_argv
    proposals = iter([
        {'source':'def stage_argv_train(i): return ["python", "wrong.py"]', 'shape_was_accepted':True},
        {'source':'def stage_argv_train(i): return ["python", "correct.py"]'}])
    class Client:
        def chat_with_metadata(self, *a, **kw):
            return json.dumps(next(proposals)), {}
    seen = []
    def verify(argv):
        seen.append(argv)
        return {'ok':argv[-1] == 'correct.py', 'launch_verified':False,
                'failure_kind':'resource_path_missing', 'error':'missing local path'}
    source, _, log = generate_argv(Client(), 'train', entrypoint='correct.py',
        invocation='python correct.py', repository_files=['correct.py'], verify=verify, attempts=2)
    assert source and 'correct.py' in source
    assert seen == [['python','wrong.py'], ['python','correct.py']]
    assert log[0]['failure_kind'] == 'resource_path_missing'


def test_failed_train_preserves_native_error_instead_of_unbound_progress(tmp_path, monkeypatch):
    from autosim.research import execution_derive as ed
    seen = []
    class Client:
        def chat_with_metadata(self, system, payload, **kw):
            seen.append(payload)
            return json.dumps({'source':'def stage_argv_train(i): return [i["python"], "-c", "fail", "--steps", str(i["steps"])]'}), {}
    monkeypatch.setattr(ed, 'isolated_argv', lambda argv, **kw: argv)
    monkeypatch.setattr(ed, 'bounded_run', lambda *a, **kw:
        subprocess.CompletedProcess(a[0], 1, '', 'ValueError: native device rejected'))
    source, _, log, _ = ed.make_runnable(Client(), 'train',
        {'entrypoint':'train.py','invocation':'python train.py','parameters':[]},
        repo=tmp_path, repository_files=[], inputs_for_verify={'python':sys.executable,'steps':1024},
        attempts=2, rounds=1, require_step_control=True)
    assert source is None
    assert 'native device rejected' in json.dumps(log)
    assert 'UnboundLocalError' not in json.dumps(log)
    assert any('native device rejected' in payload for payload in seen[1:])


def test_separate_native_streams_and_window_survive_generator_diagnosis_revision(tmp_path, monkeypatch):
    from autosim.research import execution_derive as ed
    seen = []
    stderr = '\x1b[31m[Engine fatal] SDL_Init failed: No available displays\x1b[0m'
    stdout = 'verbose configuration banner\n' * 1000
    class Client:
        def chat_with_metadata(self, system, payload, **kw):
            seen.append((system, payload, kw['timeout']))
            if 'command_built' in payload:
                if 'WHAT IS WRONG' in payload:
                    return json.dumps({'not_an_invocation_problem':'requires a renderer fix'}), {}
                return json.dumps({'finding':'display missing','about':'the command'}), {}
            return json.dumps({'source':'def stage_argv_evaluate(i): return [i["python"], "-c", "fail"]'}), {}
    monkeypatch.setattr(ed, 'isolated_argv', lambda argv, **kw: argv)
    monkeypatch.setattr(ed, 'bounded_run', lambda *a, **kw:
                        subprocess.CompletedProcess(a[0], -6, stdout, stderr))
    _, _, log, _ = ed.make_runnable(Client(), 'evaluate',
        {'entrypoint':'eval.py','invocation':'python eval.py'}, repo=tmp_path,
        repository_files=[], inputs_for_verify={'python':sys.executable},
        attempts=2, rounds=1, agent_timeout_seconds=777)
    assert len(seen) >= 4  # first draft, retry, diagnosis, revision
    assert all(timeout == 777 for _, _, timeout in seen)
    assert all('SDL_Init failed' in payload for _, payload, _ in seen[1:])
    rejection = next(row for row in log if row.get('status') == 'rejected')
    assert rejection['native_diagnostics']['returncode'] == -6
    assert rejection['native_diagnostics']['stderr_tail'] == stderr
    assert len(rejection['native_diagnostics']['stdout_tail']) <= 2000


def test_legacy_model_window_and_preflight_feedback_remain_supported():
    from autosim.research import execution_derive as ed
    windows = []
    class Client:
        def chat_with_metadata(self, *a, **kw):
            windows.append(kw['timeout'])
            return json.dumps({'source':'def stage_argv_train(i): return ["true"]'}), {}
    ed.generate_argv(Client(), 'train', entrypoint='train.py', invocation='true', repository_files=[])
    assert windows == [180]
    assert ed.native_failure_feedback({'error':'ValueError: no input'}) == 'ValueError: no input'


def test_revision_receives_and_resolves_selected_interpreter_without_host_path(tmp_path):
    from autosim.research import execution_derive as ed
    selected = str(tmp_path/'run'/'environments'/'selected'/'bin'/'python')
    repo = tmp_path/'run'/'checkout'
    seen = []
    class Client:
        def chat_with_metadata(self, system, payload, **kw):
            seen.append(payload)
            return json.dumps({'working_directory':'{repo}',
                'environment':{'WRAPPER_PYTHON':'{verified_interpreter}'}}), {}
    change = ed.revise_invocation(Client(), 'evaluate', {'interpreter':selected}, repo=repo,
        argv=['bash','eval.sh'],failure='ModuleNotFoundError: dependency')
    assert selected not in seen[0]
    assert json.loads(seen[0])['selected_stage_interpreter']['execution_ref'] == '{verified_interpreter}'
    assert change['environment']['WRAPPER_PYTHON'] == selected


@pytest.mark.parametrize('row,value', [({},'{verified_interpreter}'),
    ({'interpreter':'/usr/bin/python3'},'{repo}/{verified_interpreter}')])
def test_interpreter_reference_rejects_unbacked_or_misrooted_handle(tmp_path,row,value):
    from autosim.research import execution_derive as ed
    class Client:
        def chat_with_metadata(self,*a,**kw):
            return json.dumps({'working_directory':'{repo}','environment':{'PYTHON':value}}), {}
    assert ed.revise_invocation(Client(),'evaluate',row,repo=tmp_path,argv=[],failure='failure',attempts=1) is None


def test_first_draft_receives_bounded_same_stage_native_history(tmp_path,monkeypatch):
    from autosim.research import execution_derive as ed
    monkeypatch.setattr(sv,'recent_attempts',lambda *a:[
        {'id':str(n),'stage':'evaluate' if n else 'train','status':'failed','excerpt':'display missing'}
        for n in range(6)])
    seen = []
    class Client:
        def chat_with_metadata(self,system,payload,**kw):
            seen.append(json.loads(payload)['native_resource_context'])
            return json.dumps({'source':'def stage_argv_evaluate(i): return ["true"]'}), {}
    monkeypatch.setattr(ed,'isolated_argv',lambda argv,**kw:argv)
    monkeypatch.setattr(ed,'bounded_run',lambda *a,**kw:subprocess.CompletedProcess(a[0],0,'',''))
    ed.make_runnable(Client(),'evaluate',{'entrypoint':'eval.py','invocation':'true'},
        repo=tmp_path/'checkout', repository_files=[],
        inputs_for_verify={'output':str(tmp_path/'v'),'python':sys.executable},rounds=1,attempts=1)
    assert len(seen[0]['previous_native_verifications']) == 3
    assert all(row['stage']=='evaluate' for row in seen[0]['previous_native_verifications'])
    assert 'historical' in seen[0]['history_instruction']


@pytest.mark.parametrize('reason',['native wrapper chooses the wrong interpreter',''])
def test_failed_parsed_source_replacement_is_never_silently_discarded(reason):
    from autosim.research import execution_derive as ed
    sources = iter([
        {'source':'def stage_argv_evaluate(i): return ["bash", "wrapper.sh"]',
         'shape_was_accepted':True},
        {'source':'def stage_argv_evaluate(i): return [i["python"], "evaluate.py"]',
         'shape_was_accepted':True,'reasoning':reason}])
    class Client:
        def chat_with_metadata(self,*a,**kw): return json.dumps(next(sources)), {}
    seen = []
    def verify(argv):
        seen.append(argv)
        return {'ok':argv[0]!='bash','error':'ModuleNotFoundError: wrapper interpreter',
            'native_evidence_ref':'evidence/'+'b'*32+'.json'}
    source,_,log = ed.generate_argv(Client(),'evaluate',entrypoint='wrapper.sh',
        invocation='bash wrapper.sh',repository_files=[],inputs_for_verify={'python':'python'},
        verify=verify,attempts=2)
    if reason:
        assert seen == [['bash','wrapper.sh'],['python','evaluate.py']]
        assert source and 'evaluate.py' in source
        assert log[-1]['source_replacement']['prior_native_evidence_ref'].startswith('evidence/')
    else:
        assert source is None and seen == [['bash','wrapper.sh']]
        assert 'explicit reasoning rationale' in log[-1]['error']


def test_malformed_retry_does_not_inherit_previous_draft_or_native_output():
    from autosim.research import execution_derive as ed
    replies = iter([json.dumps({'source':'def stage_argv_evaluate(i): return ["false"]'}), 'not json'])
    class Client:
        def chat_with_metadata(self,*a,**kw):return next(replies),{}
    _,_,log = ed.generate_argv(Client(),'evaluate',entrypoint='eval.py',invocation='false',
        repository_files=[],verify=lambda _: {'ok':False,'error':'first failure'},attempts=2)
    assert log[-1]['source'] == 'not json'
    assert log[-1]['argv'] == [] and log[-1]['said'] == ''


def test_verified_artifacts_belong_to_current_unique_output(tmp_path, monkeypatch):
    from autosim.research import execution_derive as ed
    repo = tmp_path/'checkout'
    repo.mkdir()
    class Client:
        def chat_with_metadata(self, *a, **kw):
            return json.dumps({'source':'def stage_argv_train(i): return [i["python"], i["output"], "--steps", str(i["steps"])]'}), {}
    monkeypatch.setattr(ed, 'isolated_argv', lambda argv, **kw: argv)
    def run(argv, **kw):
        root = Path(argv[1]); root.mkdir()
        (root/'model.bin').write_bytes(b'weights')
        (root/'metric.json').write_text('{"score":0.25}')
        for path in root.iterdir():
            os.utime(path, (time.time()+1, time.time()+1))
        return subprocess.CompletedProcess(argv, 0, 'optimizer_step=2', '')
    monkeypatch.setattr(ed, 'bounded_run', run)
    source, _, log, _ = ed.make_runnable(Client(), 'train',
        {'entrypoint':'train.py','invocation':'python train.py','artifact':'model.bin'},
        repo=repo, repository_files=[],
        inputs_for_verify={'python':sys.executable,'steps':1024,'output':str(tmp_path/'v')},
        attempts=1, rounds=1, unique_verification_outputs=True)
    assert source, json.dumps(log)
    accepted = next(row for row in log if row.get('status') == 'accepted')
    root = Path(accepted['verification_output'])
    evidence = accepted['verified_artifact']
    assert root.parent == tmp_path/'verification_outputs'
    assert evidence['output_directory'] == str(root)
    assert evidence['declared_path'] == str(root/'model.bin')
    assert evidence['structured_candidates'][0]['relative_path'] == 'metric.json'


def test_changed_producer_guard_rejects_before_native_launch(tmp_path,monkeypatch):
    from autosim.research import execution_derive as ed
    class Client:
        def chat_with_metadata(self,*a,**kw):
            return json.dumps({'source':'def stage_argv_evaluate(i): return [i["python"], "-c", "pass"]'}), {}
    def guard():
        raise ValueError('producer bytes changed')
    def launch(*a,**kw):
        pytest.fail('native command launched after changed producer identity')
    monkeypatch.setattr(ed,'bounded_run',launch)
    source, _, log, _ = ed.make_runnable(Client(),'evaluate',
        {'entrypoint':'eval.py','invocation':'python eval.py'},repo=tmp_path,
        repository_files=[],inputs_for_verify={'python':sys.executable},
        verification_inputs_guard=guard,attempts=1,rounds=1)
    assert source is None
    assert log[0]['failure_kind'] == 'producer_identity_changed'
    assert 'producer bytes changed' in json.dumps(log)
