import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from autosim.research.budget import RunBudget
from autosim.research.native_jobs import adjust, cancel, source_identity, status, submit
from autosim.research.common import digest, object_digest


def scoped_repo(tmp_path):
    from autosim.research.common import atomic_json
    repo = tmp_path/'checkout'; repo.mkdir()
    source = repo/'train.py'; source.write_text('print("train")\n')
    output = tmp_path/'run'; output.mkdir()
    entries = [['file','train.py',source.stat().st_size,digest(source),0o644]]
    atomic_json(output/'workspace_snapshot.json',{'schema_version':2,'destination':str(repo),
        'source_entries':entries,'source_tree_fingerprint':object_digest(entries)})
    atomic_json(output/'derived_stages.json',{'train':{'source':'train-builder','parameters':{}}})
    return output, repo


def test_refused_job_window_reports_exact_wall_constraint_without_start_or_clipping(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from autosim.research import native_jobs as jobs
    output,repo=scoped_repo(tmp_path)
    monkeypatch.setattr(jobs.RunBudget,'existing',lambda _:SimpleNamespace(remaining=lambda:11800.25))
    before=sorted(p.relative_to(output).as_posix() for p in output.rglob('*'))
    with pytest.raises(ValueError) as failure:
        jobs.submit(output,repo=repo,stage='train',settings={'steps':80000},
                    requested_seconds=12000,reason='bounded native work')
    reason=str(failure.value)
    assert 'requested_seconds=12000.000' in reason
    assert 'remaining_wall_seconds=11800.250' in reason
    assert 'GPU availability is separate' in reason
    assert 'silently shortened' in reason
    assert sorted(p.relative_to(output).as_posix() for p in output.rglob('*'))==before


@pytest.mark.parametrize('candidate', [False, True])
def test_detached_train_preserves_registered_labels_and_rechecks_bytes(tmp_path, candidate):
    from autosim.research import data_versions
    from autosim.research.native_jobs import training_overrides
    from autosim.research.common import atomic_json
    output, repo = scoped_repo(tmp_path)
    data = output/'data.json'; data.write_text('[1,2,3]')
    data_versions.register(output,repo, {'id':'chosen', 'dataset_ref':'data.json',
        'source_refs':[], 'reason':'CPU detached identity fixture'},
        {'allowed_training_data':True,'loader_connection_supported':True,'sources_supported':True})
    settings, bindings = data_versions.resolve(output,repo,{'dataset':'data-version:chosen'})
    request = {'stage':'train','settings':settings}
    if candidate:
        request['candidate_binding'] = {'data_versions':bindings}
    else:
        request['data_bindings'] = bindings
    assert training_overrides(output,repo,request) == {'selected_training_data':[
        {'id':'chosen','setting':'dataset','path':str(data)}]}
    assert training_overrides(output,repo,{**request,'stage':'evaluate'}) == {}
    assert training_overrides(output,repo,{'stage':'train','settings':{}}) == {}
    data.write_text('[4]')
    with pytest.raises(ValueError, match='bytes changed'):
        training_overrides(output,repo,request)


@pytest.mark.parametrize('settings', [{}, {'episodes': 0}, {'episodes': True},
    {'episodes': -1}, {'episodes': '100'}, {'eval.n_eval': 100}])
def test_collection_submission_rejects_zero_work_before_worker(tmp_path, monkeypatch, settings):
    from autosim.research.common import atomic_json
    output, repo = scoped_repo(tmp_path)
    RunBudget(output, wall_seconds=1000)
    monkeypatch.setattr('autosim.research.scheduling.policy', lambda _: {'enabled': True})
    atomic_json(output/'derived_stages.json', {'collect': {
        'source': 'def stage_argv_collect(i): return [str(i["episodes"])]'}})
    with pytest.raises(ValueError, match='explicit positive episode target'):
        submit(output, repo=repo, stage='collect', settings=settings,
               requested_seconds=100, reason='registered acquisition')
    assert not (output/'native_jobs').exists()


def registered_policy(output):
    from autosim.research.common import atomic_json
    artifact = output/'verification_outputs'/'producer'/'policy.bin'
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'actual immutable toy policy bytes')
    files = [{'ref':artifact.relative_to(output).as_posix(),'sha256':digest(artifact),
              'bytes':artifact.stat().st_size}]
    producer = 'evidence/'+'a'*32+'.json'
    identity = object_digest({'producer':producer,'files':files})
    row = {'id':identity,'selected_path':str(artifact),'files':files,
           'producer_evidence_ref':producer,'checkpoint_ref':files[0]['ref']}
    ref = 'verification_checkpoint_handoffs/'+identity+'.json'
    atomic_json(output/ref,row)
    return {**row,'evidence_ref':ref}


@pytest.mark.parametrize('fault',[None,'settings','policy','source','native_log','failed'])
def test_exact_completed_evaluation_is_reusable_but_changed_work_is_not(tmp_path,fault):
    from autosim.research.native_jobs import completed_evaluation
    from autosim.research.common import atomic_json
    from autosim.research.evidence_store import capture_attempt_evidence
    output,repo=scoped_repo(tmp_path)
    policy=registered_policy(output)
    job='a'*32;attempt='b'*32;stamp='c'*64;settings={'episodes':20,'seed':7}
    atomic_json(output/'native_jobs'/job/'request.json',dict(job_id=job,run_id='derived',repo=str(repo),
        stage='evaluate',settings=settings,source_identity=stamp,checkpoint_selection=policy))
    atomic_json(output/'native_jobs'/job/'result.json',dict(job_id=job,status='completed',
        source_identity_after=stamp,stage_result=dict(attempt_id=attempt,returncode=0)))
    log=output/'native_eval.log';log.write_text('finished development evaluation')
    capture_attempt_evidence(output,attempt_id=attempt,log=log,
        receipt_ref=f'research/derived/attempts/{attempt}/receipt.json',status='completed',
        returncode=0,termination_reason='normal_exit')
    if fault=='settings':settings={**settings,'seed':8}
    if fault=='policy':policy={**policy,'id':'f'*64}
    if fault=='source':stamp='d'*64
    if fault=='native_log':log.write_text('changed log')
    if fault=='failed':atomic_json(output/'native_jobs'/job/'result.json',dict(job_id=job,status='failed'))
    result=completed_evaluation(output,repo=repo,run_id='derived',settings=settings,
                                checkpoint_selection=policy,source_stamp=stamp)
    assert result==(job if fault is None else None)


@pytest.mark.parametrize('fault',[None,'changed_bytes','changed_proof','wrong_stage','forged_id'])
def test_detached_policy_selection_revalidated_not_guessed(tmp_path,fault):
    from autosim.research.native_jobs import checkpoint_overrides
    from autosim.research.common import atomic_json
    output,repo = scoped_repo(tmp_path)
    chosen = registered_policy(output)
    if fault == 'changed_bytes':Path(chosen['selected_path']).write_bytes(b'other policy')
    if fault == 'changed_proof':atomic_json(output/chosen['evidence_ref'],{})
    if fault == 'forged_id':chosen['id']='0'*64
    if fault:
        with pytest.raises(ValueError):checkpoint_overrides(output,'train' if fault=='wrong_stage' else 'evaluate',chosen)
    else:
        assert checkpoint_overrides(output,'evaluate',chosen)=={'checkpoint':chosen['selected_path']}
        assert checkpoint_overrides(output,'evaluate',None)=={}


@pytest.mark.parametrize('preparation_budget_initialized', [True, False])
@pytest.mark.parametrize('persisted_budget_available', [True, False])
def test_worker_routes_explicit_policy_and_eval_settings_into_native_call(
        tmp_path,monkeypatch,preparation_budget_initialized,persisted_budget_available):
    from types import SimpleNamespace
    from autosim.research import native_jobs as nj
    from autosim.research.common import atomic_json,read_json
    output,repo = scoped_repo(tmp_path)
    if persisted_budget_available:
        RunBudget(output, wall_seconds=1000)
    atomic_json(output/'derived_stages.json',{'evaluate':{'source':'evaluator-builder'}})
    chosen = registered_policy(output)
    jid = 'd'*32; directory=output/'native_jobs'/jid;directory.mkdir(parents=True)
    settings={'task':'toy','episodes':7,'seed':19}
    atomic_json(directory/'request.json',{'job_id':jid,'repo':str(repo),'run_id':'derived',
        'stage':'evaluate','settings':settings,'requested_seconds':100,
        'source_stage_scope':['evaluate'],'source_identity':source_identity(output,repo,stage_scope=['evaluate']),
        'checkpoint_selection':chosen})
    seen=[]
    class Research:
        def run_stage(self,stage,**kwargs):
            seen.append((stage,kwargs))
            return {'status':'completed','stage':stage}
    research=Research()
    class Prep:
        def __init__(self,**kwargs):
            self.interpreter=sys.executable;self.stages={'evaluate':'builder'}
            self.budget=(SimpleNamespace(remaining=lambda:100)
                         if preparation_budget_initialized else None)
        def _research_controller(self):return research,None
    monkeypatch.setattr('autosim.research.prepare.Preparation',Prep)
    if not preparation_budget_initialized and not persisted_budget_available:
        assert nj._worker(output,jid)==1
        assert seen==[]
        result=read_json(directory/'result.json')
        assert 'existing hard run budget' in result['error']
        assert result['evidence_id']==jid
        assert (output/result['evidence_ref']).is_file()
        return
    assert nj._worker(output,jid)==0
    assert len(seen)==1 and seen[0][0]=='evaluate'
    assert seen[0][1]['settings']==settings
    assert seen[0][1]['checkpoint']==chosen['selected_path']
    if preparation_budget_initialized:
        assert seen[0][1]['timeout']==100
    else:
        # Detached Preparation construction does not run _run_cycle. Never freeze
        # the mutable 100-second request window as the process's hard timeout.
        assert 900 < seen[0][1]['timeout'] <= 1000
    assert read_json(directory/'request.json')['checkpoint_selection']==chosen


def test_detached_worker_soft_window_extension_reaches_real_process(tmp_path,monkeypatch):
    import os
    import threading
    from autosim.research import native_jobs as nj
    from autosim.research.common import atomic_json,read_json
    from autosim.research.process_executor import run_process
    output,repo=scoped_repo(tmp_path)
    RunBudget(output,wall_seconds=1000)
    jid='e'*32;directory=output/'native_jobs'/jid;directory.mkdir(parents=True)
    atomic_json(directory/'request.json',{'job_id':jid,'repo':str(repo),'run_id':'derived',
        'stage':'train','settings':{},'requested_seconds':.15})
    executions=[]
    class Research:
        def run_stage(self,stage,**kwargs):
            control=self.job_control_path
            atomic_json(control,{'deadline_epoch':time.time()+.2,'cancelled':False})
            def extend():
                time.sleep(.08)
                atomic_json(control,{'deadline_epoch':time.time()+1,'cancelled':False})
            thread=threading.Thread(target=extend);thread.start()
            try:
                attempt=run_process([sys.executable,'-c',
                    "import time; time.sleep(.35); print('completed-after-initial-window')"],
                    cwd=repo,env=os.environ.copy(),timeout=kwargs['timeout'],
                    live_control=lambda:read_json(control))
            finally:
                thread.join()
            executions.append(attempt)
            return {'stage':stage,'status':'completed' if attempt.returncode==0
                    and not attempt.timed_out else 'failed'}
    research=Research()
    class Prep:
        def __init__(self,**kwargs):
            self.interpreter=sys.executable;self.stages={'train':'builder'};self.budget=None
        def _research_controller(self):return research,None
    monkeypatch.setattr('autosim.research.prepare.Preparation',Prep)
    assert nj._worker(output,jid)==0
    assert len(executions)==1 and not executions[0].timed_out
    assert executions[0].stdout.strip()=='completed-after-initial-window'


def test_job_scope_does_not_invalidate_paid_training_for_new_evaluator(tmp_path):
    from autosim.research.native_jobs import request_source_identity, source_matches
    from autosim.research.common import atomic_json, read_json
    output,repo = scoped_repo(tmp_path)
    legacy = {'stage':'train','source_identity':source_identity(output,repo)}
    request = {'stage':'train','source_stage_scope':['train'],
        'source_identity':source_identity(output,repo,stage_scope=['train'])}
    atomic_json(output/'derived_stages.json',{
        **read_json(output/'derived_stages.json'), 'evaluate':{'source':'new evaluator'}})
    assert source_matches(output,repo,request)
    assert not source_matches(output,repo,legacy)
    assert request_source_identity(output,repo,request)==request['source_identity']
    (repo/'train.py').write_text('print("different trainer")\n')
    assert not source_matches(output,repo,request)


@pytest.mark.parametrize('scope',[None,[],['evaluate'],['train','train'],['../train']])
def test_invalid_scope_cannot_weaken_job_identity(tmp_path,scope):
    from autosim.research.native_jobs import request_source_identity
    output,repo = scoped_repo(tmp_path)
    with pytest.raises(ValueError):
        request_source_identity(output,repo,{'stage':'train','source_stage_scope':scope})


def test_training_builder_or_parameters_change_still_invalidates_job(tmp_path):
    from autosim.research.native_jobs import source_matches
    from autosim.research.common import atomic_json
    output,repo = scoped_repo(tmp_path)
    request = {'stage':'train','source_stage_scope':['train'],
        'source_identity':source_identity(output,repo,stage_scope=['train'])}
    atomic_json(output/'derived_stages.json',{'train':{'source':'train-builder','parameters':{'seed':1}}})
    assert not source_matches(output,repo,request)


def completed_legacy_job(tmp_path):
    from autosim.research.common import atomic_json
    from autosim.research.evidence_store import capture_attempt_evidence
    output,repo = scoped_repo(tmp_path)
    job='a'*32; attempt='b'*32
    request={'job_id':job,'run_id':'derived','repo':str(repo),'stage':'train',
             'source_identity':source_identity(output,repo)}
    atomic_json(output/'native_jobs'/job/'request.json',request)
    relative='research/derived/attempts/'+attempt+'/receipt.json'
    log=output/'native.log'; log.write_text('native training completed fixture\n')
    capture_attempt_evidence(output,attempt_id=attempt,log=log,receipt_ref=relative,
        status='completed',returncode=0,termination_reason='normal_exit')
    atomic_json(output/relative,{'attempt_id':attempt,'run_id':'derived',
        'controller_decision_id':job,'node_id':'train','status':'completed','returncode':0})
    atomic_json(output/'native_jobs'/job/'result.json',{'job_id':job,'status':'completed',
        'source_identity_after':request['source_identity'],
        'stage_result':{'attempt_id':attempt,'stage':'train'}})
    return output,repo,request


def test_explicit_legacy_bridge_preserves_paid_work_without_changing_raw_evidence(tmp_path):
    from autosim.research.native_jobs import bind_completed_source, source_matches
    from autosim.research.common import atomic_json,read_json
    output,repo,request = completed_legacy_job(tmp_path)
    path=output/'native_jobs'/request['job_id']/'request.json'
    before=path.read_bytes()
    assert bind_completed_source(output,repo,request)['status']=='bound'
    atomic_json(output/'derived_stages.json',{
        **read_json(output/'derived_stages.json'),'evaluate':{'source':'new score builder'}})
    assert source_matches(output,repo,request)
    assert path.read_bytes()==before
    (repo/'train.py').write_text('changed training code')
    assert not source_matches(output,repo,request)


def test_legacy_job_cannot_be_bridged_after_original_identity_already_changed(tmp_path):
    from autosim.research.native_jobs import bind_completed_source
    from autosim.research.common import atomic_json,read_json
    output,repo,request = completed_legacy_job(tmp_path)
    atomic_json(output/'derived_stages.json',{
        **read_json(output/'derived_stages.json'),'evaluate':{'source':'new score builder'}})
    with pytest.raises(ValueError,match='original global'):
        bind_completed_source(output,repo,request)
    assert not (output/'native_jobs'/request['job_id']/'source_binding.json').exists()


@pytest.mark.parametrize('change',['request','result','receipt','bridge'])
def test_additive_source_bridge_rejects_tampered_original_or_sidecar(tmp_path,change):
    from autosim.research.native_jobs import bind_completed_source,source_matches
    from autosim.research.common import atomic_json,read_json
    output,repo,request=completed_legacy_job(tmp_path)
    bind_completed_source(output,repo,request)
    atomic_json(output/'derived_stages.json',{
        **read_json(output/'derived_stages.json'),'collect':{'source':'new producer'}})
    relative = {'request':'native_jobs/'+request['job_id']+'/request.json',
        'result':'native_jobs/'+request['job_id']+'/result.json',
        'receipt':'research/derived/attempts/'+('b'*32)+'/receipt.json',
        'bridge':'native_jobs/'+request['job_id']+'/source_binding.json'}[change]
    path=output/relative; row=read_json(path); atomic_json(path,{**row,'tampered':True})
    assert not source_matches(output,repo,request)


def test_job_source_identity_tracks_existing_code_not_new_training_output(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    source = repo / "train.py"
    source.write_text("print('train')\n", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    (output / "derived_stages.json").write_text("{}", encoding="utf-8")
    entries = [["file", "train.py", source.stat().st_size, digest(source), 0o644]]
    (output / "workspace_snapshot.json").write_text(json.dumps({
        "schema_version": 2, "destination": str(repo),
        "source_entries": entries, "source_tree_fingerprint": object_digest(entries)}),
        encoding="utf-8")
    first = source_identity(output, repo)
    (repo / "checkpoint.pt").write_bytes(b"new training product")
    assert source_identity(output, repo) == first
    source.write_text("print('changed')\n", encoding="utf-8")
    assert source_identity(output, repo) != first


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="native isolation unavailable")
def test_detached_native_job_can_be_adjusted_and_cancelled(tmp_path):
    repo = tmp_path / "toy_sim"
    repo.mkdir()
    (repo / "README.md").write_text("synthetic trainer", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    RunBudget(output, wall_seconds=12)
    (output / "environment.json").write_text(json.dumps({
        "verdict": {"passed": True}, "interpreter": sys.executable}), encoding="utf-8")
    code = "import time; print('epoch=1', flush=True); time.sleep(8)"
    source = ("def stage_argv_train(i):\n"
              f"    return [i['python'], '-c', {code!r}]\n")
    (output / "derived_stages.json").write_text(json.dumps({
        "train": {"source": source, "parameters": {}}}), encoding="utf-8")
    (output / "execution.json").write_text(json.dumps({"stages": {
        "train": {"available": True, "entrypoint": "train.py",
                  "invocation": "python train.py", "artifact": ""}}}), encoding="utf-8")
    scout = output / "scouting" / "toy_sim_prepared"
    scout.mkdir(parents=True)
    declaration = {
        "benchmark": "toy_sim", "evidence": "synthetic test", "repo_markers": ["README.md"],
        "tasks": {"kind": "module_registry", "pattern": "", "names": ["ToyTask"]},
        "task_contract": {"primary_metric": {"name": "success_rate",
                     "direction": "maximize", "unit": "fraction", "source": "log"}},
        "assets": {}, "capabilities": {},
        "optimization_space": {"collection": [], "training": [{
            "name": "epochs", "kind": "integer", "description": "epochs",
            "low": 1, "high": 2, "default": 1}]}}
    (scout / "declaration.json").write_text(
        json.dumps({"declaration": declaration}), encoding="utf-8")
    job = submit(output, repo=repo, stage="train", settings={"device": "cpu"},
                 requested_seconds=4, reason="test native long job")
    job_id = job["job_id"]
    try:
        adjusted = adjust(output, job_id, seconds_from_now=5,
                          reason="first epoch has started")
        assert adjusted["deadline_epoch"] > job["deadline_epoch"]
        cancellation = cancel(output, job_id, reason="fixture has enough evidence")
        assert cancellation["cancel_requested"]
        for _ in range(100):
            current = status(output, job_id)
            if current["status"] in {"cancelled", "failed", "completed"}:
                break
            time.sleep(.1)
        assert current["status"] == "cancelled", current
        assert current["result"]["stage_result"]["termination_reason"] == "cancelled"
    finally:
        cancel(output, job_id, reason="test cleanup")
