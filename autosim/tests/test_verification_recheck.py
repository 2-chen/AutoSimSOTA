"""Real CPU evaluation receipt; artifact-only correction runs once, not twice."""
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from autosim.research.common import atomic_json
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.verification_recheck import ArtifactRecheck
from autosim.research import execution_derive as ed


def native_receipt(root, argv, output, cwd):
    ident=uuid.uuid4().hex
    log=root/f'derivation_native/{ident}/native.log'
    log.parent.mkdir(parents=True)
    done=subprocess.run(argv,cwd=cwd,capture_output=True,text=True)
    log.write_text(done.stdout+done.stderr)
    ref=f'derivation_native/{ident}/receipt.json'
    sealed=capture_attempt_evidence(root,attempt_id=ident,log=log,receipt_ref=ref,
        status='completed' if done.returncode==0 else 'failed',returncode=done.returncode,
        termination_reason='normal_exit')
    atomic_json(root/ref,{**sealed,'id':ident,'status':'completed' if done.returncode==0 else 'failed',
        'argv':argv,'cwd':str(cwd),'verification_output':str(output),'stage':'evaluate'})
    done.evidence_ref=f'evidence/{ident}.json'
    return done


@pytest.fixture
def saved(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    root=tmp_path/'run';root.mkdir()
    script=repo/'eval.py'
    script.write_text('import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True); '
                      '(p/"metrics.json").write_text("{}")')
    policy=root/'policy.bin';policy.write_bytes(b'policy')
    out=root/'verification_outputs/first'
    args=[sys.executable,str(script),str(out)]
    row={'entrypoint':'eval.py','artifact':'wrong/*.json'}
    inputs={'output':str(out),'checkpoint':str(policy),'episodes':1,'seed':1}
    cache=ArtifactRecheck('evaluate',repo,root)
    before=cache.identity(args,row,inputs,{},repo,None)
    started=time.time();done=native_receipt(root,args,out,repo)
    cache.capture(done,started,args,row,inputs,{},repo,None,before=before)
    assert cache.saved
    return cache,args,row,inputs,repo,root


def test_artifact_only_recheck_reuses_original_sealed_output(saved):
    cache,args,row,inputs,repo,root=saved
    next_out=root/'verification_outputs/second'
    changed_args=[*args[:-1],str(next_out)]
    changed={**inputs,'output':str(next_out)}
    result=cache.lookup(changed_args,{**row,'artifact':'metrics.json'},changed,{},repo,None)
    assert result and result['output']==inputs['output']
    assert not next_out.exists()


@pytest.mark.parametrize('change',['source','policy','seed','argv','environment','output','receipt','staging'])
def test_changed_execution_or_evidence_cannot_reuse(saved,change):
    cache,args,row,inputs,repo,root=saved
    row={**row,'artifact':'metrics.json'}
    environment={}
    if change=='source':(repo/'eval.py').write_text('print("changed")')
    elif change=='policy':(root/'policy.bin').write_bytes(b'other')
    elif change=='seed':inputs={**inputs,'seed':2}
    elif change=='argv':args=[*args,'--new']
    elif change=='environment':environment={'MODE':'other'}
    elif change=='output':(Path(inputs['output'])/'metrics.json').write_text('{"changed":true}')
    elif change=='receipt':cache.saved['receipt'].write_text('{}')
    else:row={**row,'staging':['touch new-input']}
    assert cache.lookup(args,row,inputs,environment,repo,None) is None


def test_training_cannot_reuse_native_evaluation(saved):
    cache,args,row,inputs,repo,root=saved
    train=ArtifactRecheck('train',repo,root)
    train.capture(cache.saved['done'],time.time(),args,row,inputs,{},repo,None,
        before=cache.identity(args,row,inputs,{},repo,None))
    assert train.saved is None


def test_unchanged_artifact_does_not_silently_reuse_a_repeat(saved):
    cache,args,row,inputs,repo,_=saved
    assert cache.lookup(args,row,inputs,{},repo,None) is None


def test_source_changed_during_execution_does_not_get_retrospective_identity(saved):
    cache,args,row,inputs,repo,root=saved
    other=ArtifactRecheck('evaluate',repo,root)
    before=other.identity(args,row,inputs,{},repo,None)
    (repo/'eval.py').write_text('print("after")')
    other.capture(cache.saved['done'],time.time(),args,row,inputs,{},repo,None,before=before)
    assert other.saved is None


def test_make_runnable_artifact_correction_has_one_real_native_execution(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    root=tmp_path/'run';root.mkdir()
    (repo/'eval.py').write_text('import pathlib,sys; p=pathlib.Path(sys.argv[1]); '
        'p.mkdir(parents=True); (p/"metrics.json").write_text("{}")')
    calls=[]
    def run(argv,**kw):
        calls.append(argv)
        return native_receipt(root,argv,Path(argv[-1]),repo)
    monkeypatch.setattr(ed,'isolated_argv',lambda argv,**kw:argv)
    monkeypatch.setattr(ed,'bounded_run',run)
    def generate(client,stage,**kw):
        i=kw['verification_inputs_factory']()
        result=kw['verify']([sys.executable,str(repo/'eval.py'),i['output']])
        item={'status':'accepted' if result['ok'] else 'rejected',
              'said':result.get('error',''),'argv':calls[0],**result}
        return ('def stage_argv_evaluate(i): return []' if result['ok'] else None),{},[item]
    monkeypatch.setattr(ed,'generate_argv',generate)
    monkeypatch.setattr(ed,'diagnose',lambda *a,**kw:{'about':'the invocation','finding':'artifact only'})
    monkeypatch.setattr(ed,'revise_invocation',lambda *a,**kw:{'artifact':'metrics.json','why':'native writer'})
    source,_,log,_=ed.make_runnable(None,'evaluate',{'entrypoint':'eval.py','artifact':'wrong/*.json'},
        repo=repo,repository_files=['eval.py'],inputs_for_verify={'repo':str(repo),'python':sys.executable,
        'output':str(root/'initial')},rounds=2,attempts=1,unique_verification_outputs=True)
    assert source is not None,log
    assert len(calls)==1
    assert log[-1]['native_diagnostics']['reused_native_execution'] is True
    assert log[-1]['verification_output']==calls[0][-1]
    assert log[-1]['verified_artifact']['matched']==1
