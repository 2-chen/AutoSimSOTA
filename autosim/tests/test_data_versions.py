import json
from pathlib import Path

import pytest

from autosim.research import data_versions
from autosim.research.common import atomic_json, digest
from autosim.research.experiment_bundle import artifact_identity
from autosim.research.evidence_store import capture_attempt_evidence
from tests.test_derived_research import research


def registered(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    output=tmp_path/'out';output.mkdir()
    data=output/'data.json';data.write_text('[1,2,3]')
    row=data_versions.register(output,repo,
        dict(id='extra',dataset_ref='data.json',source_refs=[],reason='synthetic development data'),
        dict(allowed_training_data=True,loader_connection_supported=True,sources_supported=True))
    return repo,output,data,row


def test_handles_resolve_explicit_data_and_reject_changed_bytes(tmp_path):
    repo,output,data,row=registered(tmp_path)
    settings,bindings=data_versions.resolve(output,repo,dict(dataset_root='data-version:extra'))
    assert settings['dataset_root']==str(data)
    assert bindings[0]['identity']==row['identity']
    data.write_text('[4]')
    with pytest.raises(ValueError):data_versions.resolve(output,repo,dict(dataset_root='data-version:extra'))


@pytest.mark.parametrize('samples',[0,True,3])
def test_consumption_requires_sealed_actual_loader_witness(tmp_path,samples):
    repo,output,data,row=registered(tmp_path)
    _,bindings=data_versions.resolve(output,repo,dict(dataset_root='data-version:extra'))
    attempt='d'*32;log=output/'native.log'
    log.write_text(data_versions.MARKER+json.dumps(dict(data_version_id='extra',setting='dataset_root',
        identity_field=row['identity_field'],identity=row['identity'],samples_read=samples)))
    capture_attempt_evidence(output,attempt_id=attempt,log=log,receipt_ref='receipt.json',
        status='completed',returncode=0,termination_reason='normal_exit')
    proof=data_versions.verify_consumption(output,bindings,dict(attempt_id=attempt))
    assert (proof['status']=='verified')==(samples==3 and not isinstance(samples,bool))
    log.write_text('changed')
    assert data_versions.verify_consumption(output,bindings,dict(attempt_id=attempt))['status']=='unverified'


def test_new_data_without_loader_witness_never_gets_formal_score(tmp_path):
    real=research(tmp_path)
    real.output.mkdir(exist_ok=True)
    data=real.output/'data.json';data.write_text('[1,2,3]')
    data_versions.register(real.output,real.repo,
        dict(id='extra',dataset_ref='data.json',source_refs=[],reason='synthetic'),
        dict(allowed_training_data=True,loader_connection_supported=True,sources_supported=True))
    result=real.measure(settings=dict(dataset_root='data-version:extra'),label='data_candidate')
    assert not result['ok'] and result['where']=='training_data'
    assert result['metric_value'] is None
    assert not (real.run_root/'evaluate').exists()
