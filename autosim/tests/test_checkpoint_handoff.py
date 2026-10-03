import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from autosim.research import checkpoint_handoff as ch
from autosim.research.common import atomic_json, digest
from autosim.research.evidence_store import capture_attempt_evidence


def producer(tmp_path):
    root = tmp_path/'run'
    output = root/'verification_outputs/attempt'
    bundle = output/'native_bundle'
    bundle.mkdir(parents=True)
    (bundle/'arbitrary_weights.bin').write_bytes(b'learned weights')
    config = bundle/'configuration.json'
    atomic_json(config, {'architecture':'native-format'})
    counter = output/'work.json'
    atomic_json(counter, {'updates':17})
    (output/'optimizer_state.bin').write_bytes(b'x'*1000)
    identity = 'e'*32
    log = root/f'derivation_native/{identity}/native.log'
    log.parent.mkdir(parents=True)
    log.write_text('native trainer finished\n')
    capture_attempt_evidence(root, attempt_id=identity, log=log,
        receipt_ref=f'derivation_native/{identity}/receipt.json',status='completed',
        returncode=0,termination_reason='completed')
    accepted = {'status':'accepted','native_evidence_ref':f'evidence/{identity}.json',
        'verification_output':str(output),
        'training_progress':{'status':'observed','evidence':{
            'native_counter':{'path':'work.json','field':['updates'],'value':17},
            'document_sha256':digest(counter)}},
        'verified_artifact':{'declared_path':str(config),
            'structured_candidates':[{'path':str(config),'sha256':digest(config)}]}}
    atomic_json(root/'derivation_attempts/train.json',{'attempts':[accepted]})
    return root, output, bundle, counter


def test_agent_selects_bundle_not_largest_optimizer_file(tmp_path):
    root, output, bundle, _ = producer(tmp_path)
    offered = ch.catalog(root)
    assert offered['status'] == 'available'
    assert 'selected_path' not in offered
    assert any(item['ref'].endswith('optimizer_state.bin') for item in offered['entries'])
    selected = ch.select(root,bundle.relative_to(root).as_posix(),offered)
    assert selected['selected_path'] == str(bundle)
    assert len(selected['files']) == 2
    assert selected['load_compatibility'] == 'unverified_requires_native_load'
    assert (root/selected['evidence_ref']).is_file()
    ch.validate(root, selected)
    (bundle/'arbitrary_weights.bin').write_bytes(b'changed weights')
    with pytest.raises(ValueError,match='changed'):
        ch.validate(root, selected)


@pytest.mark.parametrize('fault',['counter','config','native_log'])
def test_changed_verified_producer_evidence_is_not_offered(tmp_path,fault):
    root, _, bundle, counter = producer(tmp_path)
    path = counter if fault == 'counter' else bundle/'configuration.json' if fault == 'config' else next((root/'derivation_native').glob('*/native.log'))
    path.write_text('changed')
    assert ch.catalog(root)['status'] == 'unavailable'


@pytest.mark.parametrize('ref',['../elsewhere','/tmp/unowned','different_attempt/model'])
def test_checkpoint_choice_cannot_escape_accepted_output(tmp_path,ref):
    root, _, _, _ = producer(tmp_path)
    with pytest.raises((ValueError,FileNotFoundError)):
        ch.select(root,ref,ch.catalog(root))


def test_source_supported_bundle_alias_is_resolved_without_allowing_external_files(tmp_path):
    root, output, bundle, _ = producer(tmp_path)
    (output/'latest').symlink_to(bundle, target_is_directory=True)
    selected = ch.select(root,'verification_outputs/attempt/latest',ch.catalog(root))
    assert selected['selected_path'] == str(bundle)
    (bundle/'external').symlink_to(tmp_path/'foreign')
    with pytest.raises(ValueError,match='symbolic-link'):
        ch.select(root,bundle.relative_to(root).as_posix(),ch.catalog(root))


def test_controller_exposes_producer_and_forwards_agent_checkpoint_choice(tmp_path,monkeypatch):
    from tests.test_main_agent import make
    from autosim.research import prepare
    real = make(tmp_path)
    root, _, bundle, _ = producer(tmp_path)
    real.interpreter = Path(sys.executable)
    real.decision = SimpleNamespace(device='cuda',device_index=0,environment={},why='fixture',evidence={})
    real.execution = {'stages':{'train':{'available':True,'entrypoint':'train.py'},
        'evaluate':{'available':True,'entrypoint':'eval.py',
                    'invocation':'python eval.py --checkpoint <checkpoint>'}}}
    monkeypatch.setattr(prepare.execution_derive,'checkpoint_for_verification',
                        lambda *a,**kw:{'path':str(root/'old_optimizer.bin')})
    calls = []
    def runnable(client,stage,row,**kw):
        calls.append(kw['inputs_for_verify']['checkpoint'])
        kw['verification_inputs_guard']()
        return 'def stage_argv_evaluate(i): return ["true"]', {}, [], row
    monkeypatch.setattr(prepare.execution_derive,'make_runnable',runnable)
    monkeypatch.setattr(real,'_record_attempts',lambda *a:{})
    assert real._step_derive_a_command(stage='evaluate')['outcome'] == 'not attempted'
    assert not calls
    ref = bundle.relative_to(root).as_posix()
    result = real.do('derive_a_command',stage='evaluate',checkpoint_ref=ref)
    assert result['outcome'] == 'done', result
    assert calls == [str(bundle)]
    assert any('verification_checkpoint_handoffs/' in ref for ref in result['evidence_refs'])


def test_agent_can_allocate_model_window_without_changing_native_ceiling(tmp_path, monkeypatch):
    from tests.test_main_agent import make
    from autosim.research import prepare
    real = make(tmp_path)
    real.interpreter = Path(sys.executable)
    real.decision = SimpleNamespace(device='cuda',device_index=0,environment={},why='fixture',evidence={})
    real.execution = {'stages':{'train':{'available':True,'entrypoint':'train.py'}}}
    real.client.timeout = 900
    real.budget = SimpleNamespace(remaining=lambda: 600)
    monkeypatch.setattr(real,'_record_attempts',lambda *a:{})
    monkeypatch.setattr(prepare.execution_derive,'checkpoint_for_verification',lambda *a,**kw:{'path':''})
    calls = []
    def runnable(client,stage,row,**kw):
        calls.append((kw['agent_timeout_seconds'],kw['verification_timeout']))
        return 'def stage_argv_train(i): return ["true"]', {}, [], row
    monkeypatch.setattr(prepare.execution_derive,'make_runnable',runnable)
    result = real.do('derive_a_command',stage='train',timeout_seconds=120,agent_timeout_seconds=555)
    assert result['outcome'] == 'done', result
    assert calls == [(555,120)]
    real._step_derive_a_command(stage='train',agent_timeout_seconds=9999)
    assert calls[-1] == (600,600)
    real.budget = None
    real._step_derive_a_command(stage='train')
    assert calls[-1] == (900,900)
    for bad in (True, 0, -1, float('inf'),float('nan'),'900'):
        assert real._step_derive_a_command(stage='train',agent_timeout_seconds=bad)['outcome'] == 'rejected'
