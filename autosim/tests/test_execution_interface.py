import pytest

from autosim.research.common import atomic_json, read_json
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.execution_interface import publish, reuse
from tests.test_native_jobs import scoped_repo


def interface(tmp_path):
    output,repo = scoped_repo(tmp_path)
    atomic_json(output/'environment.json',{'interpreter':'test-python','verdict':{'passed':True}})
    identity = 'a'*32
    log = output/'derivation_native'/identity/'native.log'; log.parent.mkdir(parents=True)
    log.write_text('real fixture update and checkpoint\n')
    proof = capture_attempt_evidence(output,attempt_id=identity,log=log,
        receipt_ref='derivation_native/'+identity+'/receipt.json',status='completed',
        returncode=0,termination_reason='normal_exit')
    atomic_json(output/'derivation_attempts/train.json',{'attempts':[
        {'status':'accepted','native_evidence_ref':proof['evidence_ref']}]})
    return output,repo,proof


def test_adapter_versions_reuse_exact_evidence_without_model_or_execution(tmp_path):
    output,repo,_ = interface(tmp_path)
    row = publish(output,repo,'train')
    assert reuse(output,repo,'train')['id']==row['id']
    assert publish(output,repo,'train')['id']==row['id']
    before = (output/row['ref']).read_bytes()
    stages = read_json(output/'derived_stages.json')
    atomic_json(output/'derived_stages.json',{**stages,'collect':{'source':'new producer'}})
    assert reuse(output,repo,'train')['id']==row['id']
    assert (output/row['ref']).read_bytes()==before


@pytest.mark.parametrize('changed',['code','environment','binding','builder','evidence'])
def test_adapter_reuse_rejects_changed_dependencies(tmp_path,changed):
    output,repo,proof = interface(tmp_path)
    publish(output,repo,'train')
    if changed=='code': (repo/'train.py').write_text('changed code')
    elif changed=='environment': atomic_json(output/'environment.json',{'interpreter':'different'})
    elif changed=='binding':
        manifest = read_json(output/'workspace_snapshot.json')
        atomic_json(output/'workspace_snapshot.json',{**manifest,'resource_bindings':[{'target':'new'}]})
    elif changed=='builder': atomic_json(output/'derived_stages.json',{'train':{'source':'different-builder'}})
    else:
        (output/proof['log_ref']).write_text('changed log')
    with pytest.raises(ValueError): reuse(output,repo,'train')


def test_reuse_is_explicit_and_does_not_skip_requested_reverification(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    output,repo,_ = interface(tmp_path)
    publish(output,repo,'train')
    real = preparation(tmp_path/'controller'); real.output=output; real.repo=repo
    monkeypatch.setattr(real.client,'chat_with_metadata',lambda *a,**kw:pytest.fail('cache cannot call model'))
    result = real._step_derive_a_command(stage='train',reuse_verified=True)
    assert result['interface_reused']
    assert real._step_derive_a_command(stage='train',reuse_verified=True,draft_ref='new')['outcome']=='rejected'


def test_unaccepted_native_evidence_cannot_publish_an_adapter(tmp_path):
    output,repo,_ = interface(tmp_path)
    atomic_json(output/'derivation_attempts/train.json',{'attempts':[{'status':'rejected'}]})
    with pytest.raises(ValueError,match='accepted native'):
        publish(output,repo,'train')
