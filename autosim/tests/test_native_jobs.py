import json
import shutil
import sys
import time

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
