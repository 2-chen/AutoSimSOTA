"""Prospective data identities survive CPU handoff failures; no robot score fixtures."""
import copy
import json
from pathlib import Path

import pytest

from autosim.research import data_campaign as dc, data_versions as dv, handoff_correction as hc
from autosim.research.common import atomic_json, digest, object_digest, read_json
from tests.test_data_versions import registered
from tests.test_measurement_recovery import failed_bundle


def selected(tmp_path):
    repo, output, data, _ = registered(tmp_path)
    settings, bindings = dv.resolve(output, repo, {'dataset_root':'data-version:extra'})
    proof = dv.seal_selection(output, 'derived', 'round_2', settings, bindings)
    measured = {'label':'round_2', 'settings':settings, 'train':{'attempt_id':'a'*32},
                'data_selection':{**proof, 'training_attempt_id':'a'*32}}
    return repo, output, data, bindings, measured


def test_selection_is_small_sealed_and_revalidated_not_a_consumption_certificate(tmp_path):
    repo, output, _, bindings, measured = selected(tmp_path)
    proof = measured['data_selection']
    body = read_json(output/proof['record_ref'])
    assert 'asset' not in body['bindings'][0]
    assert 'data_consumption' not in measured
    assert dv.restore_selection(output, repo, 'derived', measured) == bindings
    assert dv.restore_selection(output, repo, 'derived', {'data_consumption':{'status':'unknown'}}) == []
    with pytest.raises(ValueError, match='legacy'):
        dv.restore_selection(output, repo, 'derived', {'data_consumption':{'versions':['extra']}})


@pytest.mark.parametrize('fault', ['proof', 'producer', 'settings', 'bytes', 'review', 'symlink', 'other_run'])
def test_recovery_never_relabels_or_reselects_original_data(tmp_path, fault):
    repo, output, data, _, measured = selected(tmp_path)
    proof = output/measured['data_selection']['record_ref']
    if fault == 'proof': proof.write_text('{}')
    elif fault == 'producer': measured['train']['attempt_id'] = 'b'*32
    elif fault == 'settings': measured['settings']['dataset_root'] = 'different'
    elif fault == 'bytes': data.write_text('[4]')
    elif fault == 'review':
        path = output/'data_versions/extra.json'; row = read_json(path)
        row['review']['loader_connection_supported'] = False; atomic_json(path, row)
    elif fault == 'other_run': measured['data_selection']['record_ref'] = 'research/other/data_selections/'+proof.name
    else:
        destination = output/'external.json'; destination.write_bytes(proof.read_bytes())
        proof.unlink(); proof.symlink_to(destination)
    with pytest.raises((ValueError, OSError)):
        dv.restore_selection(output, repo, 'derived', measured)


def data_failure(tmp_path, label='round_2'):
    real, _ = failed_bundle(tmp_path)
    data = real.output/'training.json'; data.write_text('[1,2,3]')
    dv.register(real.output, real.repo, {
        'id':'extra', 'dataset_ref':'training.json', 'source_refs':[], 'reason':'CPU fixture'},
        {'allowed_training_data':True, 'loader_connection_supported':True, 'sources_supported':True})
    producer = real.repo/'produce.py'
    producer.write_text(producer.read_text()+'''
import hashlib
selected=Path(sys.argv[2]); reads=0
for iteration in range(completed):
    batch=json.loads(selected.read_text()); reads+=len(batch)
print('AUTOSIM_TRAIN_DATA_CONSUMPTION '+json.dumps({
    'data_version_id':'extra', 'setting':'dataset_root', 'identity_field':'content_sha256',
    'identity':hashlib.sha256(selected.read_bytes()).hexdigest(), 'samples_read':reads,
    'consumption_stage':'training_batch', 'reader_id':'main'}))
''')
    real.sources['train'] = ('def stage_argv_train(i):\n'
        '    return [i["python"], i["repo"]+"/produce.py", i["output"], i["settings"]["dataset_root"]]')
    from autosim.research.declarative_backend import DeclarativeBackend
    real.backend = DeclarativeBackend(repo=real.repo, answer={'stages':real.backend.stages},
                                     sources=real.sources, parameters={})
    session_path = real.run_root/'controller_session.json'
    session = read_json(session_path)
    atomic_json(session_path, {**session, 'schema_version':1, 'repository':str(real.repo)})
    failed = real.measure(settings={'dataset_root':'data-version:extra'}, label=label)
    return real, failed


def test_native_failed_bundle_recovery_retains_actual_training_batch_proof(tmp_path, monkeypatch):
    real, failed = data_failure(tmp_path)
    assert failed['ok'] is False and failed['where'] == 'evaluate', failed
    assert failed['data_consumption']['status'] == 'verified'
    proof = failed['data_selection']; assert proof['training_attempt_id'] == failed['train']['attempt_id']
    journals = [read_json(p) for p in (real.run_root/'experiment_transactions').glob('*.json')]
    assert next(j for j in journals if j['label']=='round_2')['data_selection']['id'] == proof['id']
    original_sha = digest(real.run_root/'measurements/round_2.json')
    native = real.run_stage; calls = []
    def evaluate_only(stage, **kw):
        assert stage == 'evaluate'
        calls.append(stage)
        return native(stage, **kw)
    monkeypatch.setattr(real, 'run_stage', evaluate_only)
    packet = next(p for p in hc.eligible(real.run_root) if p['label']=='round_2')
    result = hc.correct(real, label='round_2',
        checkpoint_ref=packet['training_output_ref']+'/arbitrary_native_bundle')
    assert result['ok'] and calls == ['evaluate'], result
    assert result['train']['attempt_id'] == failed['train']['attempt_id']
    assert result['data_consumption']['status'] == 'verified'
    assert result['data_consumption']['exposure'][0]['samples_read'] == 51
    assert result['data_consumption']['exposure'][0]['consumption_stage'] == 'training_batch'
    assert digest(real.run_root/'measurements/round_2.json') == original_sha


@pytest.mark.parametrize('phase',['before_measure','measurement_started'])
def test_handoff_source_refusal_is_sealed_and_reaches_fix_with_safe_native_status(tmp_path,monkeypatch,phase):
    from autosim.research.evidence_store import read_attempt_evidence
    from autosim.research.prepare import _sealed_fix_handoff
    from tests.test_prepare import preparation
    real,failed=data_failure(tmp_path)
    original=real.run_root/'measurements/round_2.json';before=digest(original)
    packet=next(p for p in hc.eligible(real.run_root) if p['label']=='round_2')
    def refuse(*a,**kw):
        raise ValueError('independent source/receipt revalidation refused')
    if phase=='before_measure':monkeypatch.setattr(dv,'restore_selection',refuse)
    else:monkeypatch.setattr(real,'measure',refuse)
    parent=preparation(tmp_path/'parent')
    parent.output=real.output;parent.run_id=real.run_id
    monkeypatch.setattr(parent,'_research_controller',lambda:(real,{}))
    result=parent._step_correct_policy_handoff(label='round_2',
        checkpoint_ref=packet['training_output_ref']+'/arbitrary_native_bundle')
    assert result['outcome']=='not recovered' and result['repair_owner']=='fix'
    assert _sealed_fix_handoff(result)
    assert result['training_attempt_id']==failed['train']['attempt_id']
    assert result['native_operation_status']==('not_started' if phase=='before_measure' else 'unknown')
    proof=read_attempt_evidence(real.output,result['evidence_id'])
    assert proof['status']=='operation_contract_failed'
    assert 'independent source/receipt revalidation refused' in proof['text']
    assert digest(original)==before


def test_baseline_recovery_preserves_original_data_selection_and_never_retrains(tmp_path, monkeypatch):
    from autosim.research import measurement_recovery as mr
    real, failed = data_failure(tmp_path, label='baseline')
    original = (real.run_root/'measurements/baseline.json').read_bytes()
    packet = next(p for p in mr.offered(real.output, real.run_id) if p['label']=='baseline')
    _, choice = mr.select_for_recovery(real.output, real.run_id, 'baseline',
        packet['training_output_ref']+'/arbitrary_native_bundle')
    native = real.run_stage; calls = []
    def evaluate_only(stage, **kw):
        assert stage == 'evaluate'
        calls.append(stage)
        return native(stage, **kw)
    monkeypatch.setattr(real, 'run_stage', evaluate_only)
    result = real.recover_unscored_baseline(checkpoint_handoff=choice)
    assert result['ok'] and calls == ['evaluate'], result
    assert result['data_consumption']['status'] == 'verified'
    assert result['data_selection'] == failed['data_selection']
    assert (real.run_root/result['recovery']['original_measurement_ref']).read_bytes() == original


def corrected_registry_fixture(tmp_path):
    from tests.test_data_campaign import arm_fixture
    repo, output, _, _, kwargs = arm_fixture(tmp_path)
    source = output/'research/derived/measurements/round_1.json'
    original = read_json(source)
    original.update(settings={'dataset_root':'fixture native path'}, train={'attempt_id':'c'*32})
    atomic_json(source, original)
    receipt = output/f'research/derived/attempts/{"c"*32}/receipt.json'
    atomic_json(receipt, {'attempt_id':'c'*32, 'returncode':0})
    correction = {'status':'measured', 'original_measurement_ref':source.relative_to(output).as_posix(),
        'original_measurement_sha256':digest(source), 'corrected_label':'round_1_handoff_correction',
        'training_reused':True, 'training_attempt_id':'c'*32, 'training_receipt_sha256':digest(receipt)}
    row = {**copy.deepcopy(original), 'label':'round_1_handoff_correction', 'handoff_correction':correction}
    atomic_json(source.with_name(row['label']+'.json'), row)
    atomic_json(receipt.with_name('handoff_correction.json'),
                {**correction, 'result_digest':object_digest(row)})
    return repo, output, kwargs, row


def test_registry_accepts_only_receipt_linked_same_producer_corrective_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
                        lambda *a, **kw:{'status':'consistent'})
    repo, output, kwargs, row = corrected_registry_fixture(tmp_path)
    result = dc.record_arm(output, repo, 'derived',
        **{**kwargs, 'measurement_label':row['label']})
    assert result['measurement_label'] == row['label']
    assert result['status'] == 'consumed_and_measured'


@pytest.mark.parametrize('fault', ['missing', 'original', 'producer', 'settings', 'state', 'receipt', 'selection'])
def test_corrective_name_alone_never_admits_a_data_arm(tmp_path, monkeypatch, fault):
    monkeypatch.setattr('autosim.research.receipt_verifier.verify_measurement',
                        lambda *a, **kw:{'status':'consistent'})
    repo, output, kwargs, row = corrected_registry_fixture(tmp_path)
    root = output/'research/derived'
    if fault == 'missing': row.pop('handoff_correction')
    elif fault == 'original': atomic_json(root/'measurements/round_1.json', {'changed':True})
    elif fault == 'producer': row['train']['attempt_id'] = 'd'*32
    elif fault == 'settings': row['settings'] = {'dataset_root':'different'}
    elif fault == 'state': atomic_json(root/f'attempts/{"c"*32}/handoff_correction.json', {'status':'revalidating'})
    elif fault == 'receipt': atomic_json(root/f'attempts/{"c"*32}/receipt.json', {'changed':True})
    else: row['data_selection'] = {'id':'e'*64}
    atomic_json(root/'measurements'/f'{row["label"]}.json', row)
    with pytest.raises((ValueError, KeyError)):
        dc.record_arm(output, repo, 'derived', **{**kwargs, 'measurement_label':row['label']})
