"""Changing a loader invalidates its review, not the unchanged data asset."""
from types import SimpleNamespace
import json

import pytest

from autosim.research import data_consumer_review as consumer, data_versions
from autosim.research.baseline_reference import SourceCitationError, source_evidence
from autosim.research.common import atomic_json, read_json, object_digest
from autosim.research.checkout_source_citations import capture


class Reviewer:
    supports_candidate_execution_review = True
    allow = True
    calls = 0
    malformed = False

    def chat_with_metadata(self, system, user, **kwargs):
        self.calls += 1
        assert kwargs['read_only'] and kwargs['decision_only']
        assert kwargs['include_research_context'] is False
        # This source-only JSON judgment is not an open-ended research session.
        # The gateway bounds visible output; do not exhaust it with provider
        # reasoning, trigger CLI automatic continuation, or reload unrelated skills.
        assert kwargs['max_tokens']==2400 and kwargs['thinking']=='disabled'
        assert kwargs['timeout']==90 and kwargs['resume_session'] is False
        assert kwargs['auto_skills'] is False
        assert 'Return exactly one JSON object' in system
        payload = json.loads(user)
        assert payload['selection_key'] == 'dataset'
        assert payload['review_sources']
        answer = {key:self.allow for key in consumer.CHECKS}
        answer.update(why='CPU fixture review; not a native training result',
                      citations=[{'source_id':'review-source-1'}])
        if self.malformed: answer['citations'] = [{'source_id':'guessed-invalid'}]
        return json.dumps(answer), {}


def setup(tmp_path):
    output = tmp_path/'run';repo = output/'checkout';repo.mkdir(parents=True)
    code = repo/'trainer.py';code.write_text('def load(root):\n    return reader(root)\n')
    data = output/'data.json';data.write_text('[1,2,3]')
    old = capture(output,repo,{'ref':'trainer.py:1-2'})
    refs = source_evidence(repo,[{'ref':old['ref'],'quote':old['quote']}],output=output)
    row = data_versions.register(output,repo,{'id':'extra','dataset_ref':'data.json',
        'reason':'fixture training data','source_refs':refs},
        {key:True for key in consumer.CHECKS})
    run_root = output/'research/derived'
    transaction = {'status':'applied','round':1,'idea_label':'repair-reader',
        'patches':[{'file':'trainer.py','find':'reader(root)','replace':'correct_reader(root)'}]}
    atomic_json(run_root/'code_changes/round-1.json',transaction)
    atomic_json(run_root/'controller_session.json',{'status':'running','action':'round_1',
        'pending_action':{'round':1,'phase':'preparing_idea','idea':{'label':'repair-reader'}}})
    code.write_text('def load(root):\n    return correct_reader(root)\n')
    client = Reviewer()
    real = SimpleNamespace(repo=repo,output=output,run_root=run_root,run_id='derived',client=client)
    return real,row,data,code


def test_current_independent_review_preserves_data_and_historical_registration(tmp_path):
    real,row,data,code=setup(tmp_path)
    registered=real.output/'data_versions/extra.json';before=registered.read_bytes()
    raw=data.read_bytes()
    with pytest.raises(SourceCitationError):data_versions.resolve(real.output,real.repo,{'dataset':'data-version:extra'})
    settings,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert settings=={'dataset':str(data)}
    assert bindings[0]['consumer_review']['record_ref'].startswith('data_consumer_reviews/extra/')
    assert real.client.calls==1
    assert registered.read_bytes()==before and data.read_bytes()==raw
    assert row['source_refs']==read_json(registered)['source_refs']
    assert data_versions.resolve(real.output,real.repo,{'dataset':'data-version:extra'})==(settings,bindings)
    assert real.client.calls==1
    # A review is not actual consumption and cannot fabricate a training witness.
    assert data_versions.verify_consumption(real.output,bindings,{},training_performed=True)['status']=='unverified'


@pytest.mark.parametrize('fault',[None,'native_source','data','failed_producer','missing_seal',
    'wrong_metadata','review_tamper','wrong_settings','transaction'])
def test_completed_training_recovery_keeps_original_review_across_executor_only_change(tmp_path,monkeypatch,fault):
    from autosim.research.common import digest
    from autosim.research.evidence_store import capture_attempt_evidence
    real,row,data,code=setup(tmp_path)
    # A bounded page plus a complete-file expansion yields executor context.
    code.write_text(code.read_text()+'# complete consumer source beyond the cited page\n')
    old=consumer._executor_contract_specs
    specs=old()
    monkeypatch.setattr(consumer,'_executor_contract_specs',lambda:specs)
    settings,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    proof=data_versions.seal_selection(real.output,'derived','round_1',settings,bindings)
    attempt='a'*32;ref=f'research/derived/attempts/{attempt}/receipt.json'
    labels=[{'id':'extra','setting':'dataset','path':settings['dataset']}]
    receipt={'attempt_id':attempt,'run_id':'derived','stage':'train','status':'completed',
        'returncode':0,'termination_reason':'normal_exit','settings_digest':object_digest(settings),
        'working_directory':str(real.repo),
        'environment':{'AUTOSIM_TRAIN_SELECTION_JSON':json.dumps(labels)}}
    atomic_json(real.output/ref,receipt)
    log=real.output/'completed.log';log.write_text('completed training; no consumption certificate')
    sealed=capture_attempt_evidence(real.output,attempt_id=attempt,log=log,receipt_ref=ref,
        status='completed',returncode=0,termination_reason='normal_exit')
    measurement={'label':'round_1','settings':settings,'train':receipt,
        'data_selection':{**proof,'training_attempt_id':attempt}}
    # Historical writer context changes, while native consumer/data stay exact.
    revised=[{**s,'sha256':'f'*64} for s in specs]
    monkeypatch.setattr(consumer,'_executor_contract_specs',lambda:revised)
    before=(real.output/proof['record_ref']).read_bytes()
    registry=(real.output/'data_versions/extra.json').read_bytes()
    if fault=='native_source':code.write_text('def load(root):\n    return unrelated_reader(root)\n')
    elif fault=='data':data.write_text('[9]')
    elif fault=='failed_producer':
        receipt['status']='failed';atomic_json(real.output/ref,receipt)
    elif fault=='missing_seal':(real.output/sealed['evidence_ref']).unlink()
    elif fault=='wrong_metadata':
        receipt['environment']={'AUTOSIM_TRAIN_SELECTION_JSON':'[]'};atomic_json(real.output/ref,receipt)
    elif fault=='review_tamper':
        p=real.output/bindings[0]['consumer_review']['record_ref'];p.write_text('{}')
    elif fault=='wrong_settings':
        receipt['settings_digest']='f'*64;atomic_json(real.output/ref,receipt)
    elif fault=='transaction':
        p=real.run_root/'code_changes/round-1.json';j=read_json(p);j['status']='applying';atomic_json(p,j)
    with pytest.raises((ValueError,OSError)):
        data_versions.resolve(real.output,real.repo,{'dataset':'data-version:extra'})
    if fault is None:
        assert data_versions.restore_selection(real.output,real.repo,'derived',measurement)==bindings
    else:
        with pytest.raises((ValueError,OSError)):
            data_versions.restore_selection(real.output,real.repo,'derived',measurement)
    assert (real.output/proof['record_ref']).read_bytes()==before
    assert (real.output/'data_versions/extra.json').read_bytes()==registry


def test_selection_seals_consumer_review_and_cannot_restore_different_scope(tmp_path):
    real,row,data,code=setup(tmp_path)
    settings,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    proof=data_versions.seal_selection(real.output,'derived','round_1',settings,bindings)
    attempt='a'*32
    measurement={'label':'round_1','settings':settings,'train':{'attempt_id':attempt},
        'data_selection':{**proof,'training_attempt_id':attempt}}
    assert data_versions.restore_selection(real.output,real.repo,'derived',measurement)==bindings
    with pytest.raises(SourceCitationError):
        data_versions.resolve(real.output,real.repo,{'other_key':'data-version:extra'})
    code.write_text('def load(root):\n    return a_different_reader(root)\n')
    with pytest.raises(SourceCitationError):
        data_versions.restore_selection(real.output,real.repo,'derived',measurement)


@pytest.mark.parametrize('fault',['data_changed','snapshot_changed','transaction_applying',
                                'not_owner','environment_changed_during_review'])
def test_no_automatic_approval_of_corruption_or_unknown_side_effects(tmp_path,fault):
    real,row,data,code=setup(tmp_path)
    if fault=='data_changed':data.write_text('[9]')
    if fault=='snapshot_changed':next((real.output/'checkout_source_citations').glob('*.source')).write_text('forged')
    if fault=='transaction_applying':
        path=real.run_root/'code_changes/round-1.json';tx=read_json(path);tx['status']='applying';atomic_json(path,tx)
    if fault=='not_owner':
        path=real.run_root/'controller_session.json';s=read_json(path);s['pending_action']['idea']['label']='other';atomic_json(path,s)
    if fault=='environment_changed_during_review':
        original=real.client.chat_with_metadata
        def changing(*args,**kwargs):
            result=original(*args,**kwargs);atomic_json(real.output/'environment.json',{'changed':True});return result
        real.client.chat_with_metadata=changing
    with pytest.raises(SourceCitationError,match='consumer revalidation refused'):
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert not list((real.output/'data_consumer_reviews').glob('*/*.json'))
    assert real.client.calls==(1 if fault=='environment_changed_during_review' else 0)


@pytest.mark.parametrize('malformed',[False,True])
def test_refused_or_invalid_review_is_cached_without_rebuying_same_call(tmp_path,malformed):
    from autosim.research.evidence_store import read_attempt_evidence
    real,row,data,code=setup(tmp_path);real.client.malformed=malformed;real.client.allow=malformed
    ids=[]
    for _ in range(2):
        with pytest.raises(SourceCitationError) as failure:
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        identity=failure.value.details['evidence_id'];ids.append(identity)
        assert identity in str(failure.value)[:600]
        evidence=read_attempt_evidence(real.output,identity)
        assert evidence['status']==('consumer_review_unavailable' if malformed else 'consumer_review_refused')
        reason='invalid response/citation' if malformed else 'CPU fixture review'
        assert reason in evidence['text']
    assert len(set(ids))==1
    assert real.client.calls==(2 if malformed else 1)
    assert consumer.accepted(real.output,real.repo,row,'dataset') is None
    assert len(list((real.output/'data_consumer_reviews').glob('*/*.json')))==1


def test_one_format_correction_preserves_first_answer_and_does_not_execute_training(tmp_path):
    real,row,data,code=setup(tmp_path)
    original=real.client.chat_with_metadata
    calls=[]
    def correcting(system,user,**kwargs):
        packet=json.loads(user);calls.append(packet)
        if len(calls)==1:
            return 'The connection appears plausible, but this is not a JSON review.',{}
        assert packet['format_feedback']=='no JSON object in the response'
        assert packet['previous_response'].startswith('The connection')
        return original(system,user,**kwargs)
    real.client.chat_with_metadata=correcting
    before=(real.output/'data_versions/extra.json').read_bytes()
    _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    record=read_json(real.output/bindings[0]['consumer_review']['record_ref'])
    assert len(calls)==2 and len(record['responses'])==2
    assert record['status']=='reviewed' and record['response_protocol']==consumer.RESPONSE_PROTOCOL
    assert record['responses'][0]['text'].startswith('The connection')
    assert len(list((real.output/'data_consumer_reviews/extra/responses').glob('*.json')))==2
    assert before==(real.output/'data_versions/extra.json').read_bytes()
    assert data_versions.verify_consumption(real.output,bindings,{},training_performed=True)['status']=='unverified'


@pytest.mark.parametrize('allow', [True, False])
def test_detailed_bounded_reason_does_not_invalidate_a_structured_verdict(tmp_path, allow):
    real,row,data,code=setup(tmp_path)
    calls=[]
    def detailed(*args,**kwargs):
        calls.append(1)
        return json.dumps({**{key:allow for key in consumer.CHECKS},
            'why':'Source-backed routing explanation. '*60,
            'citations':[{'source_id':'review-source-1'}]}),{}
    real.client.chat_with_metadata=detailed
    if allow:
        _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        record=read_json(real.output/bindings[0]['consumer_review']['record_ref'])
        assert len(record['review']['why'])>1500
    else:
        with pytest.raises(SourceCitationError) as failure:
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        from autosim.research.evidence_store import read_attempt_evidence
        proof=read_attempt_evidence(real.output,failure.value.details['evidence_id'])
        assert proof['status']=='consumer_review_refused'
    assert len(calls)==1


def test_legacy_format_failure_is_upgraded_without_rewriting_old_record_or_evidence(tmp_path):
    from autosim.research.evidence_store import read_attempt_evidence, capture_attempt_evidence
    real,row,data,code=setup(tmp_path)
    material=consumer._material(real.output,real.repo,row,'dataset')
    ref=f'data_consumer_reviews/extra/{object_digest(material)}.json'
    legacy={'schema_version':1,'material':material,
        'review':{**{key:False for key in consumer.CHECKS},
            'why':'completed review had an invalid response/citation: no JSON object in the response',
            'response_sha256':object_digest('legacy non-JSON answer')},
        'candidate_transaction_ref':'research/derived/code_changes/round-1.json'}
    legacy['sha256']=object_digest(legacy)
    atomic_json(real.output/ref,legacy)
    legacy_id=object_digest({'kind':'data_consumer_review','material':material})[:32]
    old_evidence=capture_attempt_evidence(real.output,attempt_id=legacy_id,
        log=real.output/ref,receipt_ref=ref,status='consumer_review_refused',
        returncode=None,termination_reason='independent_data_consumer_review')
    old_bytes=(real.output/ref).read_bytes()
    old_proof=read_attempt_evidence(real.output,old_evidence['evidence_id'])
    _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    approval=bindings[0]['consumer_review']
    assert approval['record_ref'].endswith(f'.format-v{consumer.RESPONSE_PROTOCOL}.json')
    assert real.client.calls==1 and (real.output/ref).read_bytes()==old_bytes
    assert read_attempt_evidence(real.output,old_evidence['evidence_id'])==old_proof
    assert consumer.accepted(real.output,real.repo,row,'dataset')==approval
    consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert real.client.calls==1


@pytest.mark.parametrize('bad',['not-json','string-boolean','empty-reason'])
def test_exhausted_format_correction_is_unavailable_not_a_negative_scientific_review(tmp_path,bad):
    from autosim.research.evidence_store import read_attempt_evidence
    real,row,data,code=setup(tmp_path)
    calls=[]
    def invalid(*args,**kwargs):
        calls.append(1)
        answer={**{key:True for key in consumer.CHECKS},'why':'review',
            'citations':[{'source_id':'review-source-1'}]}
        if bad=='not-json':return 'No JSON response.',{}
        if bad=='string-boolean':answer['allowed_training_data']='true'
        if bad=='empty-reason':answer['why']=''
        return json.dumps(answer),{}
    real.client.chat_with_metadata=invalid
    for _ in range(2):
        with pytest.raises(SourceCitationError) as failure:
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        evidence=read_attempt_evidence(real.output,failure.value.details['evidence_id'])
        assert evidence['status']=='consumer_review_unavailable'
        record=read_json(real.output/evidence['receipt_ref'])
        assert record['status']=='unavailable'
        assert all(key not in record['review'] for key in consumer.CHECKS)
    assert len(calls)==2 and consumer.accepted(real.output,real.repo,row,'dataset') is None


def test_second_format_call_transport_failure_retains_first_answer_without_caching_a_verdict(tmp_path):
    real,row,data,code=setup(tmp_path)
    calls=[];error=OSError('transport unavailable on correction')
    def interrupted(*args,**kwargs):
        calls.append(1)
        if len(calls)==1:return 'Non JSON first answer.',{}
        raise error
    real.client.chat_with_metadata=interrupted
    with pytest.raises(OSError) as failure:
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert failure.value is error and len(calls)==2
    assert len(list((real.output/'data_consumer_reviews/extra/responses').glob('*.json')))==1
    assert not list((real.output/'data_consumer_reviews').glob('*/*.json'))
    assert consumer.accepted(real.output,real.repo,row,'dataset') is None


@pytest.mark.parametrize('allow', [True, False])
def test_v2_length_failure_reuses_sealed_paid_response_without_buying_or_erasing(tmp_path,monkeypatch,allow):
    from autosim.research.evidence_store import read_attempt_evidence
    real,row,data,code=setup(tmp_path)
    calls=[]
    def detailed(*args,**kwargs):
        calls.append(1)
        return json.dumps({**{key:allow for key in consumer.CHECKS},
            'why':'A genuine source routing explanation. '*60,
            'citations':[{'source_id':'review-source-1'}]}),{}
    real.client.chat_with_metadata=detailed
    with monkeypatch.context() as old:
        old.setattr(consumer,'RESPONSE_PROTOCOL',2)
        old.setattr(consumer,'MAX_REASON_CHARS',1500)
        with pytest.raises(SourceCitationError) as failure:
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    prior=read_attempt_evidence(real.output,failure.value.details['evidence_id'])
    old_path=real.output/prior['receipt_ref'];old_bytes=old_path.read_bytes()
    assert prior['status']=='consumer_review_unavailable' and len(calls)==2
    if allow:
        _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        new_path=real.output/bindings[0]['consumer_review']['record_ref']
    else:
        with pytest.raises(SourceCitationError) as negative:
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        evidence=read_attempt_evidence(real.output,negative.value.details['evidence_id'])
        assert evidence['status']=='consumer_review_refused'
        new_path=real.output/evidence['receipt_ref']
    record=read_json(new_path)
    assert record['response_protocol']==3 and record['status']=='reviewed'
    assert record['revalidated_response_from']['record_ref']==prior['receipt_ref']
    assert record['revalidated_response_from']['response_index']==1
    assert record['responses']==read_json(old_path)['responses']
    assert len(calls)==2 and old_path.read_bytes()==old_bytes
    assert read_attempt_evidence(real.output,prior['evidence_id'])==prior


@pytest.mark.parametrize('allow', [True, False])
def test_v2_format_upgrade_keeps_real_v2_verdict_without_new_call(tmp_path,monkeypatch,allow):
    real,row,data,code=setup(tmp_path)
    material=consumer._material(real.output,real.repo,row,'dataset')
    base=f'data_consumer_reviews/extra/{object_digest(material)}'
    legacy={'material':material,'status':'unavailable','review':{'why':'format failure'}}
    legacy['sha256']=object_digest(legacy);atomic_json(real.output/(base+'.json'),legacy)
    real.client.allow=allow
    with monkeypatch.context() as old:
        old.setattr(consumer,'RESPONSE_PROTOCOL',2)
        if allow:
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        else:
            with pytest.raises(SourceCitationError):
                consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    old_path=real.output/(base+'.format-v2.json');old_bytes=old_path.read_bytes()
    calls=real.client.calls
    if allow:
        _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        assert bindings[0]['consumer_review']['record_ref']==base+'.format-v2.json'
    else:
        with pytest.raises(SourceCitationError):
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert real.client.calls==calls and old_path.read_bytes()==old_bytes
    assert not (real.output/(base+'.format-v3.json')).exists()


def test_parser_upgrade_does_not_cherry_pick_an_earlier_favorable_response(tmp_path,monkeypatch):
    real,row,data,code=setup(tmp_path)
    calls=[]
    def answer(*args,**kwargs):
        calls.append(1)
        result={**{key:len(calls)<3 for key in consumer.CHECKS},
            'why':('Earlier long source reasoning. '*65 if len(calls)<3 else 'Current refusal'),
            'citations':[{'source_id':'invalid' if len(calls)==2 else 'review-source-1'}]}
        return json.dumps(result),{}
    real.client.chat_with_metadata=answer
    with monkeypatch.context() as old:
        old.setattr(consumer,'RESPONSE_PROTOCOL',2)
        old.setattr(consumer,'MAX_REASON_CHARS',1500)
        with pytest.raises(SourceCitationError):
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    with pytest.raises(SourceCitationError,match='Current refusal'):
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert len(calls)==3
    assert consumer.accepted(real.output,real.repo,row,'dataset') is None


@pytest.mark.parametrize('bad', ['oversize','bad-citation','string-boolean'])
def test_long_reason_does_not_relax_verdict_or_citation_validation(tmp_path,bad):
    real,row,data,code=setup(tmp_path)
    def invalid(*args,**kwargs):
        answer={**{key:True for key in consumer.CHECKS},'why':'Detailed routing. '*120,
            'citations':[{'source_id':'review-source-1'}]}
        if bad=='oversize': answer['why']='x'*(consumer.MAX_REASON_CHARS+1)
        if bad=='bad-citation': answer['citations']=[{'source_id':'invented'}]
        if bad=='string-boolean': answer['sources_supported']='true'
        return json.dumps(answer),{}
    real.client.chat_with_metadata=invalid
    with pytest.raises(SourceCitationError):
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert consumer.accepted(real.output,real.repo,row,'dataset') is None


@pytest.mark.parametrize('allow',[True,False])
def test_valid_legacy_judgment_keeps_its_original_binding_without_purchasing_upgrade(tmp_path,allow):
    real,row,data,code=setup(tmp_path)
    material=consumer._material(real.output,real.repo,row,'dataset')
    ref=f'data_consumer_reviews/extra/{object_digest(material)}.json'
    legacy={'schema_version':1,'material':material,
        'review':{**{key:allow for key in consumer.CHECKS},'why':'legacy source judgment',
                  'citations':material['source_specs']},
        'candidate_transaction_ref':'research/derived/code_changes/round-1.json'}
    legacy['sha256']=object_digest(legacy);atomic_json(real.output/ref,legacy)
    before=(real.output/ref).read_bytes()
    if allow:
        _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        assert bindings[0]['consumer_review']['record_ref']==ref
    else:
        with pytest.raises(SourceCitationError,match='legacy source judgment'):
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert real.client.calls==0 and (real.output/ref).read_bytes()==before
    assert not list((real.output/'data_consumer_reviews/extra').glob('*.format-v2.json'))


def test_reviewer_sees_complete_current_consumer_not_just_historical_line_range(tmp_path):
    real,row,data,code=setup(tmp_path)
    code.write_text(code.read_text()+'\ndef initialize_loader(root):\n    return load(root)\n')
    original=real.client.chat_with_metadata
    def inspect(system,user,**kwargs):
        packet=json.loads(user)
        assert any('def initialize_loader(root)' in item['quote'] for item in packet['review_sources'])
        assert packet['original_registration_review']==row['review']
        assert 'Global episodes/seed' in packet['consumer_contract']
        assert packet['source_context_omissions']==[]
        binding=packet['registered_input_selection']
        assert binding['version_id']=='extra' and binding['input_setting']=='dataset'
        assert 'not yet consumed' in binding['runtime_authority']
        contracts=[s for s in packet['review_sources'] if s['ref'].startswith('executor-contract:')]
        assert any('def resolve(' in s['quote'] for s in contracts)
        assert any("surroundings['AUTOSIM_TRAIN_SELECTION_JSON']" in s['quote'] for s in contracts)
        assert any('native_environment=surroundings' in s['quote'] for s in contracts)
        assert str(data) not in user
        return original(system,user,**kwargs)
    real.client.chat_with_metadata=inspect
    registered=(real.output/'data_versions/extra.json').read_bytes()
    _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra','episodes':50},round_index=1)
    approval=bindings[0]['consumer_review']
    record=read_json(real.output/approval['record_ref'])
    assert record['material']['current_context_version']==3
    assert any(x['ref']=='trainer.py' for x in record['material']['source_specs'])
    assert (real.output/'data_versions/extra.json').read_bytes()==registered
    code.write_text(code.read_text().replace('return load(root)','return another_loader(root)'))
    assert consumer.accepted(real.output,real.repo,row,'dataset') is None


def test_current_executor_contract_change_invalidates_new_consumer_review(tmp_path,monkeypatch):
    real,row,data,code=setup(tmp_path)
    code.write_text(code.read_text()+'\ndef initialize_loader(root):\n    return load(root)\n')
    specs=consumer._executor_contract_specs()
    monkeypatch.setattr(consumer,'_executor_contract_specs',lambda:specs)
    _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert consumer.accepted(real.output,real.repo,row,'dataset')==bindings[0]['consumer_review']
    specs=[{**s,'quote':s['quote']+'\n# executor changed\n','sha256':'f'*64} for s in specs]
    assert consumer.accepted(real.output,real.repo,row,'dataset') is None


def test_unchanged_context_v2_approval_remains_usable_without_new_paid_review(tmp_path,monkeypatch):
    real,row,data,code=setup(tmp_path)
    code.write_text(code.read_text()+'\ndef initialize_loader(root):\n    return load(root)\n')
    original=consumer._expanded_material
    def old_context(*args,**kwargs):
        return original(*args,**{**kwargs,'include_executor_contract':False})
    with monkeypatch.context() as old:
        old.setattr(consumer,'_expanded_material',old_context)
        _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    ref=bindings[0]['consumer_review']['record_ref'];before=(real.output/ref).read_bytes()
    assert read_json(real.output/ref)['material']['current_context_version']==2
    calls=real.client.calls
    assert consumer.accepted(real.output,real.repo,row,'dataset')==bindings[0]['consumer_review']
    consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert real.client.calls==calls and (real.output/ref).read_bytes()==before


@pytest.mark.parametrize('legacy_allowed,corrupt',[(False,False),(True,False),(False,True)])
def test_expanded_source_can_revisit_incomplete_refusal_without_erasing_legacy_judgment(tmp_path,legacy_allowed,corrupt):
    from autosim.research.evidence_store import read_attempt_evidence, capture_attempt_evidence
    real,row,data,code=setup(tmp_path)
    code.write_text(code.read_text()+'\ndef initialize_loader(root):\n    return load(root)\n')
    base=consumer._material(real.output,real.repo,row,'dataset')
    ref=f'data_consumer_reviews/extra/{object_digest(base)}.json'
    legacy={'schema_version':1,'material':base,
        'review':{**{key:legacy_allowed for key in consumer.CHECKS},
                  'why':'historical source judgment','citations':base['source_specs']}}
    legacy['sha256']=object_digest(legacy)
    if corrupt:legacy['review']['why']='tampered after signing'
    atomic_json(real.output/ref,legacy)
    before=(real.output/ref).read_bytes()
    identity=object_digest({'kind':'data_consumer_review','material':base})[:32]
    capture_attempt_evidence(real.output,attempt_id=identity,log=real.output/ref,
        receipt_ref=ref,status='consumer_review_approved' if legacy_allowed else 'consumer_review_refused',
        returncode=None,termination_reason='independent_data_consumer_review')
    proof=read_attempt_evidence(real.output,identity)
    if corrupt:
        with pytest.raises(SourceCitationError,match='corrupt'):
            consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        assert real.client.calls==0
    else:
        _,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        selected=bindings[0]['consumer_review']['record_ref']
        assert (selected==ref)==legacy_allowed
        assert real.client.calls==(0 if legacy_allowed else 1)
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
        assert real.client.calls==(0 if legacy_allowed else 1)
    assert (real.output/ref).read_bytes()==before
    assert read_attempt_evidence(real.output,identity)==proof


def test_expanded_context_is_bounded_and_does_not_read_uncited_files_or_raw_json(tmp_path):
    real,row,data,code=setup(tmp_path)
    code.write_text(code.read_text()+'\n# '+('x'*consumer.MAX_CURRENT_CONTEXT_BYTES))
    material=consumer._expanded_material(real.output,real.repo,row,'dataset')
    assert material['context_omissions'][0]['ref']=='trainer.py'
    assert material['context_omissions'][0]['bytes']>consumer.MAX_CURRENT_CONTEXT_BYTES
    checkout_specs=[x for x in material['source_specs'] if not x['ref'].startswith('executor-contract:')]
    assert len(checkout_specs)==len(row['source_refs'])
    assert all(len(x['quote'])<4000 for x in checkout_specs)
    assert sum(len(x['quote'].encode()) for x in material['source_specs']
               if x['ref'].startswith('executor-contract:'))<=16384
    base={'source_specs':[{'ref':'raw_episode.json:1','quote':'old bounded page','sha256':'a'*64}]}
    assert consumer._expanded_material(real.output,real.repo,row,'dataset',base=base)==base


@pytest.mark.parametrize('stage_record',[False,True])
def test_context_bound_prioritizes_verified_train_source_edge_not_old_collection_page_order(tmp_path,stage_record):
    real,row,data,code=setup(tmp_path)
    code.write_text(code.read_text()+'\n# '+('t'*40000))
    collector=real.repo/'collector.py';collector.write_text('# '+('c'*40000))
    train='def stage_argv_train(i):\n    return [i["python"],i["repo"]+"/trainer.py"]\n'
    atomic_json(real.output/'derived_stages.json',{'train':{'source':train,'parameters':{}}
                                                if stage_record else train})
    base=consumer._material(real.output,real.repo,row,'dataset')
    from autosim.research.baseline_reference import checkout_source
    _,_,_,sha=checkout_source(real.repo,'collector.py')
    base={**base,'source_specs':[{'ref':'collector.py:1','quote':'# old collection excerpt','sha256':sha},
                               *base['source_specs']]}
    material=consumer._expanded_material(real.output,real.repo,row,'dataset',base=base)
    expanded=[x for x in material['source_specs'][len(base['source_specs']):]
              if not x['ref'].startswith('executor-contract:')]
    assert [x['ref'] for x in expanded]==['trainer.py']
    assert [x['ref'] for x in material['context_omissions']]==['collector.py']


def test_refused_review_evidence_detects_tampering_without_a_paid_retry(tmp_path):
    from autosim.research.evidence_store import read_attempt_evidence
    real,row,data,code=setup(tmp_path);real.client.allow=False
    with pytest.raises(SourceCitationError) as failure:
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    identity=failure.value.details['evidence_id']
    path=next((real.output/'data_consumer_reviews').glob('*/*.json'))
    record=read_json(path);record['review']['why']='tampered';atomic_json(path,record)
    with pytest.raises(ValueError,match='changed'):
        read_attempt_evidence(real.output,identity)
    with pytest.raises(SourceCitationError,match='corrupt'):
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert real.client.calls==1


def test_consumer_refusal_has_a_separate_compact_fix_handoff_and_preserves_review(tmp_path):
    from autosim.research.evidence_store import read_attempt_evidence
    from autosim.research.prepare import Preparation, _sealed_fix_handoff
    real,row,data,code=setup(tmp_path)
    class Refusing(Reviewer):
        allow=False
        def chat_with_metadata(self,*args,**kwargs):
            text,metadata=super().chat_with_metadata(*args,**kwargs)
            answer=json.loads(text)
            answer['why']='Legality and historical scope are unchanged. '*55+'ACTUAL_OPENED_ROOT_NOT_CONNECTED'
            return json.dumps(answer),metadata
    real.client=Refusing()
    card_before=(real.output/'data_versions/extra.json').read_bytes()
    data_before=data.read_bytes();code_before=code.read_bytes()
    with pytest.raises(SourceCitationError) as raised:
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    failure=raised.value;handoff=failure.planning_failure
    assert handoff['repair_owner']=='fix' and handoff['native_operation_status']=='not_started'
    assert handoff['retry_original_arguments'] is False
    assert handoff['evidence_id']!=failure.details['evidence_id']
    page=read_attempt_evidence(real.output,handoff['evidence_id'],limit=12000)
    assert page['status']=='operation_contract_failed'
    body=json.loads(page['text']);summary=body['context']['consumer_review_summary']
    assert 'loader_connection_supported' in summary['failed_checks']
    assert 'ACTUAL_OPENED_ROOT_NOT_CONNECTED' in summary['review']['why']
    assert 'citations' not in summary['review']
    assert body['context']['consumer_review_evidence_id']==failure.details['evidence_id']
    # Public log paths may be redacted; use the authenticated stable ID, never
    # a display placeholder as a local executable/read path.
    original_review=read_attempt_evidence(real.output,failure.details['evidence_id'],limit=1)
    review_path=real.output/original_review['receipt_ref'];before=review_path.read_bytes()
    assert read_json(review_path)['review']['loader_connection_supported'] is False
    # The ordinary controller exception adapter must retain the typed handoff,
    # even though the operation's result is a refusal rather than a process exit.
    preparation=Preparation(repo=real.repo,output=real.output,scouting=real.output/'scout',client=None)
    def fail():raise failure
    preparation._step_read_the_checkout=fail
    result=preparation.do('read_the_checkout')
    assert result['repair_owner']=='fix'
    assert result['failure_category']=='data_consumer_review_required'
    assert result['evidence_id']==handoff['evidence_id']
    assert result['native_operation_status']=='not_started'
    assert _sealed_fix_handoff(result)
    assert review_path.read_bytes()==before
    assert (real.output/'data_versions/extra.json').read_bytes()==card_before
    assert data.read_bytes()==data_before and code.read_bytes()==code_before
    assert real.client.calls==1 and not (real.run_root/'attempts').exists()


def test_unverified_pre_native_boundary_cannot_claim_a_fix_launch_boundary(tmp_path):
    real,row,data,code=setup(tmp_path)
    session=read_json(real.run_root/'controller_session.json')
    session['pending_action']['phase']='training'
    atomic_json(real.run_root/'controller_session.json',session)
    with pytest.raises(SourceCitationError) as failure:
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert not hasattr(failure.value,'planning_failure')
    assert real.client.calls==0


@pytest.mark.parametrize('kind',['run_model_budget','provider_payment_required','network','prompt_limit'])
def test_review_runtime_failures_keep_their_identity_and_do_not_cache_a_data_refusal(tmp_path,kind):
    from autosim.research.agent_client import AgentRuntimeClientError
    real,row,data,code=setup(tmp_path)
    if kind=='network':
        error=OSError('fixture transport unavailable')
    elif kind=='prompt_limit':
        error=ValueError('fixture prompt exceeds runtime limit')
    else:
        error=AgentRuntimeClientError('fixture model refusal',failure_category=kind,
            status='budget_exhausted' if kind=='run_model_budget' else 'failed',
            run_budget={'remaining_usd':0} if kind=='run_model_budget' else {})
    def unavailable(*args,**kwargs):
        raise error
    real.client.chat_with_metadata=unavailable
    before=(real.output/'data_versions/extra.json').read_bytes()
    with pytest.raises(type(error)) as failure:
        consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert failure.value is error
    assert not list((real.output/'data_consumer_reviews').glob('*/*.json'))
    assert (real.output/'data_versions/extra.json').read_bytes()==before


def test_cached_review_tampering_or_environment_drift_never_restores_permission(tmp_path):
    real,row,data,code=setup(tmp_path)
    consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    path=next((real.output/'data_consumer_reviews').glob('*/*.json'))
    record=read_json(path);record['review']['why']='tampered';atomic_json(path,record)
    assert consumer.accepted(real.output,real.repo,row,'dataset') is None
    with pytest.raises(SourceCitationError):data_versions.resolve(real.output,real.repo,{'dataset':'data-version:extra'})


def test_existing_current_review_does_not_buy_another_model_call(tmp_path):
    real,row,data,code=setup(tmp_path)
    code.write_text('def load(root):\n    return reader(root)\n')
    settings,bindings=consumer.resolve_candidate(real,{'dataset':'data-version:extra'},round_index=1)
    assert real.client.calls==0 and 'consumer_review' not in bindings[0]


def test_full_candidate_revalidates_changed_loader_before_native_training(tmp_path,monkeypatch):
    from tests.test_derived_research import research
    from autosim.research.ideas import Idea
    from autosim.research.declarative_backend import DeclarativeBackend
    real=research(tmp_path)
    monkeypatch.setattr(real,'_summarize',lambda:None)
    monkeypatch.setattr(real,'capture_demo',lambda **kw:{'captured':False})
    source=real.sources['train'].replace('    return argv',
        '    if i["dataset"]:\n        argv = argv + ["--data-root", i["dataset"]]\n    return argv')
    real.sources['train']=source
    real.backend=DeclarativeBackend(repo=real.repo,answer=real.backend.answer,
        sources=real.sources,parameters={})
    real.run(rounds=2,yield_after_action=True)
    code=real.repo/'loader.py';code.write_text('def load(root):\n    return reader(root)\n')
    data=real.output/'new.json';data.write_text('[1,2,3]')
    old=capture(real.output,real.repo,{'ref':'loader.py:1-2'})
    refs=source_evidence(real.repo,[{'ref':old['ref'],'quote':old['quote']}],output=real.output)
    data_versions.register(real.output,real.repo,{'id':'extra','dataset_ref':'new.json',
        'source_refs':refs,'reason':'CPU fixture'}, {key:True for key in consumer.CHECKS})
    real.library.add(Idea(label='reader-candidate',granularity='code',risk='low',crosses='none',
        change={'file':'loader.py','find':'reader(root)','replace':'correct_reader(root)'}))
    real.library.audit(real.red_lines())
    real.client=Reviewer()
    result=real.run(rounds=2,yield_after_action=True,selected_idea_label='reader-candidate',
        main_controller_owns_selection=True,candidate_training_settings={'dataset':'data-version:extra'})
    assert result['run_status']=='paused'
    assert real.client.calls==1
    measured=read_json(real.run_root/'measurements/round_1.json')
    # The toy producer ran, but emits NO actual batch witness. A current source
    # approval must not manufacture a data-consumption certificate or valid score.
    assert measured['train']['attempt_id']
    assert measured['ok'] is False and measured['metric_value'] is None
    assert measured['data_consumption']['status']=='unverified'
    selection=read_json(real.output/measured['data_selection']['record_ref'])
    assert selection['bindings'][0]['consumer_review']['record_ref']
