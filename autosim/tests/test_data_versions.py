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


def test_selected_label_survives_path_resolution_without_expected_identity(tmp_path):
    repo, output, data, row = registered(tmp_path)
    settings, bindings = data_versions.resolve(output, repo, {'dataset': 'data-version:extra'})
    metadata = data_versions.training_metadata(bindings, settings)
    assert metadata == [{'id': 'extra', 'setting': 'dataset', 'path': str(data)}]
    assert row['identity'] not in json.dumps(metadata)
    assert 'identity_field' not in metadata[0] and 'samples_read' not in metadata[0]
    assert 'source' not in metadata[0]


def test_formal_train_caller_receives_verified_label_separate_from_native_flags(tmp_path, monkeypatch):
    real = research(tmp_path)
    path = real.output / 'extra.json'
    path.write_text('[1,2,3]')
    data_versions.register(real.output, real.repo,
        {'id': 'chosen', 'dataset_ref': 'extra.json', 'source_refs': [], 'reason': 'test caller binding'},
        {'allowed_training_data': True, 'loader_connection_supported': True, 'sources_supported': True})
    seen = []
    def capture(stage, **kwargs):
        assert stage == 'train'
        seen.append(kwargs)
        raise RuntimeError('stop after actual train caller')
    monkeypatch.setattr(real, 'run_stage', capture)
    with pytest.raises(RuntimeError, match='stop after actual train caller'):
        real.measure(settings={'dataset': 'data-version:chosen'}, label='baseline')
    assert seen[0]['settings']['dataset'] == str(path)
    assert 'extra' not in seen[0]
    assert seen[0]['selected_training_data'] == [
        {'id': 'chosen', 'setting': 'dataset', 'path': str(path)}]
    inputs = real._inputs('train', settings=seen[0]['settings'],
                          selected_training_data=seen[0]['selected_training_data'])
    assert inputs['selected_training_data'][0]['id'] == 'chosen'
    assert inputs['extra'] == {}
    assert data_versions.training_metadata([], {}) == []


@pytest.mark.parametrize('settings', [{'dataset': 'relative/path'}, {}])
def test_selection_metadata_does_not_invent_unresolved_paths(settings):
    with pytest.raises(ValueError, match='resolved data slots'):
        data_versions.training_metadata([{'id': 'selected', 'setting': 'dataset'}], settings)


def test_selection_metadata_cannot_become_an_unknown_native_cli_flag(tmp_path):
    real = research(tmp_path)
    # Existing builders often append EVERY settings/extra item as a native flag.
    # Selection labels must have a typed separate channel, not pollute either map.
    def old_builder(i):
        argv = [i['python'], '-c', 'pass', '--steps', str(i['steps'])]
        for name, value in {**i['settings'], **i['extra']}.items():
            argv += ['--' + name.replace('_', '-'), str(value)]
        return argv
    real.backend._functions['train'] = old_builder
    inputs = real._inputs('train', settings={'steps': 3}, extra={'native_option': 7},
        selected_training_data=[{'id': 'selected', 'setting': 'dataset', 'path': '/data'}])
    argv = real.backend.argv('train', inputs)
    assert '--native-option' in argv
    assert not any('selected_training_data' in word or 'selected-training-data' in word for word in argv)


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


def test_data_proof_failure_is_not_hidden_behind_a_simultaneous_bundle_failure(tmp_path,monkeypatch):
    real=research(tmp_path)
    data=real.output/'data.json'; data.write_text('[1,2,3]')
    data_versions.register(real.output,real.repo,
        dict(id='extra',dataset_ref='data.json',source_refs=[],reason='synthetic'),
        dict(allowed_training_data=True,loader_connection_supported=True,sources_supported=True))
    def no_boundary(trained,*a,**kw):
        return {**trained,'artifact_selection':{'status':'native_boundary_review_required',
            'why':'complete source-backed bundle not yet selected'}}
    monkeypatch.setattr(real,'_select_policy_artifact',no_boundary)
    monkeypatch.setattr(real,'_recorded_artifact',lambda *a:None)
    result=real.measure(settings=dict(dataset_root='data-version:extra'),label='both_failed')
    assert result['ok'] is False and result['where']=='policy_artifact'
    assert result['data_consumption']['status']=='unverified'
    assert result['train']['status']=='completed' and result['train']['returncode']==0
    assert result['artifact_selection']['status']=='native_boundary_review_required'
    assert result['policy_handoff_failure']['evidence_id']
    assert result['metric_value'] is None and not (real.run_root/'evaluate').exists()


def test_registration_and_resolution_keep_native_sdk_source_context(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research import native_source_citations as sources
    real=preparation(tmp_path)
    prefix=tmp_path/'environment';path=prefix/'lib/loader.py';path.parent.mkdir(parents=True)
    path.write_text('def load(root):\n    return read_training_samples(root)\n')
    monkeypatch.setattr(sources,'load_context',lambda *a:
                        {'identity':'native-loader-context','interpreter':str(prefix/'bin/python')})
    captured=sources.capture(real.output,real.repo,{'root_index':0,'path':'lib/loader.py'})
    quote='return read_training_samples(root)'
    real.client.choices=[dict(allowed_training_data=True,loader_connection_supported=True,sources_supported=True,
        citations=[dict(source_id='review-source-1',quote=quote)])]
    data=real.output/'training.json';data.write_text('[1,2,3]')
    result=real._step_register_data_version(version=dict(id='sdk_data',dataset_ref='training.json',
        reason='development data only',source_refs=[dict(ref=captured['ref'],quote=quote)]))
    assert result['outcome']=='registered',result
    settings,bindings=data_versions.resolve(real.output,real.repo,dict(dataset_root=result['handle']))
    assert settings['dataset_root']==str(data) and bindings[0]['source_refs'][0]['ref']==captured['ref']
    path.write_text('changed loader\n')
    with pytest.raises(ValueError,match='changed'):
        data_versions.resolve(real.output,real.repo,dict(dataset_root=result['handle']))


def test_registration_reviews_frozen_intent_without_requiring_prior_training(tmp_path):
    from tests.test_prepare import preparation
    from autosim.research import research_brief
    from autosim.research.common import object_digest
    real=preparation(tmp_path)
    real.output.mkdir(parents=True,exist_ok=True)
    text='Compare newly collected development demonstrations; never use held-out states.'
    atomic_json(real.output/'research_brief.json',
        {'text':text,'sha256':object_digest(text)})
    (real.output/'training.json').write_text('[1,2,3]')
    (tmp_path/'loader.py').write_text('return native_training_reader(root)\n')
    calls=[]
    def review(system,user,**kwargs):
        calls.append((system,json.loads(user),kwargs))
        return json.dumps(dict(allowed_training_data=True,loader_connection_supported=True,
            sources_supported=True,why='Source route is supported, runtime use still unverified.',
            citations=[dict(source_id='review-source-1')])),{}
    real.client.chat_with_metadata=review
    request=dict(id='before_training',dataset_ref='training.json',reason='prospective development',
        source_refs=[dict(ref='loader.py',quote='return native_training_reader(root)')])
    result=real._step_register_data_version(version=request)
    assert result['outcome']=='registered',result
    system,packet,options=calls[0]
    assert packet['frozen_study_intent']['text']==research_brief.view(real.output)['text']
    assert packet['frozen_study_intent']['status']=='frozen'
    # The model gets intent, not a private run identity/hash.
    assert packet['frozen_study_intent']['sha256']=='[LOCAL_RESOURCE_IDENTITY_OMITTED]'
    assert packet['review_phase']=='source_and_asset_registration_not_runtime_consumption'
    assert 'Do NOT require a prior training run' in system
    assert 'Explicit benchmark restrictions' in system
    assert options['read_only'] and options['include_research_context'] is False
    _,bindings=data_versions.resolve(real.output,tmp_path,{'dataset':result['handle']})
    assert data_versions.verify_consumption(real.output,bindings,{},training_performed=True)['status']=='unverified'
    assert not (real.output/'research/derived/train').exists()


@pytest.mark.parametrize('denied',[
    'allowed_training_data','loader_connection_supported','sources_supported'])
def test_source_only_registration_never_overrides_a_negative_review(tmp_path,denied):
    from tests.test_prepare import preparation
    real=preparation(tmp_path)
    real.output.mkdir(parents=True,exist_ok=True)
    (real.output/'training.json').write_text('[1,2,3]')
    (tmp_path/'loader.py').write_text('return native_training_reader(root)\n')
    answer=dict(allowed_training_data=True,loader_connection_supported=True,sources_supported=True,
        why='Independent reviewer found a genuine source restriction.',
        citations=[dict(source_id='review-source-1')])
    answer[denied]=False
    real.client.choices=[answer]
    result=real._step_register_data_version(version=dict(id='denied',dataset_ref='training.json',
        reason='development only',source_refs=[dict(ref='loader.py',quote='return native_training_reader(root)')]))
    assert result['outcome']=='not attempted'
    assert not (real.output/'data_versions/denied.json').exists()


def test_refused_data_review_is_sealed_for_data_agent_not_generic_checkout_fix(tmp_path):
    from tests.test_prepare import preparation
    from autosim.research.evidence_store import read_attempt_evidence
    real=preparation(tmp_path)
    code=tmp_path/'loader.py';code.write_text('return native_training_reader(root)\n')
    real.output.mkdir(parents=True,exist_ok=True)
    (real.output/'training.json').write_text('[1,2,3]')
    real.client.choices=[dict(allowed_training_data=True,loader_connection_supported=False,sources_supported=True,
        why='The source never verifies actual samples_read for this selected version.',
        citations=[dict(source_id='review-source-1',quote='return native_training_reader(root)')])]
    result=real._step_register_data_version(version=dict(id='rejected',dataset_ref='training.json',
        reason='development data',source_refs=[dict(ref='loader.py',quote='return native_training_reader(root)')]))
    assert result['outcome']=='not attempted' and result['repair_owner']=='data'
    assert result['native_operation_status']=='not_started'
    proof=read_attempt_evidence(real.output,result['evidence_id'])
    evidence=json.loads(proof['text'])
    assert evidence['context']['failed_checks']==['loader_connection_supported']
    assert 'samples_read' in evidence['context']['review']['why']
    assert not (real.output/'data_versions/rejected.json').exists()
