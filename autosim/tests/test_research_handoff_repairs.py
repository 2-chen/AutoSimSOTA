"""Regression for paid work, redaction-safe reviews and nonterminal refusals."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosim.research.common import atomic_json, digest, read_json
from autosim.research import checkpoint_handoff as handoff, source_citations
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.training_progress_audit import revalidate_detached


def long_producer(tmp_path):
    root=tmp_path/'run'; repo=root/'checkout'; repo.mkdir(parents=True)
    source=repo/'train.py';source.write_text('optimizer.step()\nstep += 1\nsave(step)\n')
    identity='a'*32; job_id='b'*32
    output=root/'research/derived/train/native_outputs'/identity
    bundle=output/'native-policy';bundle.mkdir(parents=True)
    (bundle/'weights.bin').write_bytes(b'native trained bytes')
    atomic_json(output/'work.json',{'updates':17})
    log=root/'research/derived/train/attempts'/identity/'output.log'
    log.parent.mkdir(parents=True);log.write_text('native training completed')
    path=root/'research/derived/attempts'/identity/'receipt.json'
    sealed=capture_attempt_evidence(root,attempt_id=identity,log=log,
        receipt_ref=path.relative_to(root).as_posix(),status='completed',returncode=0,
        termination_reason='completed')
    receipt={**sealed,'attempt_id':identity,'stage':'train','controller_decision_id':job_id,
        'argv':['python',str(source)],'started_epoch':0,'output_directory':str(output),
        'artifact':{'checked':True,'freshness_checked':True,'matched':1,'pattern':'native-policy/*'},
        'training_progress':{'status':'unknown'}}
    atomic_json(path,receipt)
    answer={'verdict':'verified','counter':{'path':'work.json','field':['updates'],'value':17},
        'source_evidence':[{'source_id':'source-0','quote':'step += 1'}]}
    audit_path=root/'training_progress_audits/audit.json'
    atomic_json(audit_path,{'native_evidence_ref':sealed['evidence_ref'],'answer':answer,
                           'source_hashes':[digest(source)]})
    atomic_json(path.with_name('training_progress_revalidation.json'),{
        'native_receipt_sha256':digest(path),'audit_ref':audit_path.relative_to(root).as_posix(),
        'audit_sha256':digest(audit_path)})
    job=root/'native_jobs'/job_id
    request={'job_id':job_id,'run_id':'derived','stage':'train','repo':str(repo),'source_identity':'c'*64}
    atomic_json(job/'request.json',request)
    atomic_json(job/'result.json',{'job_id':job_id,'status':'completed','stage_result':receipt,
        'source_identity_after':'c'*64})
    return root,repo,bundle,path,job


def test_long_training_is_available_without_derivation_or_comparison_protocol(tmp_path):
    root,repo,bundle,path,_=long_producer(tmp_path)
    original=path.read_bytes()
    assert not (root/'research/derived/comparison_protocol.json').exists()
    reviewed=revalidate_detached(read_json(path),path=path,root=root,repo=repo)
    assert reviewed['training_progress']['status']=='observed'
    catalog=handoff.catalog(root)
    assert catalog['status']=='available' and catalog['producers'][0]['origin']=='native_job'
    chosen=handoff.select(root,bundle.relative_to(root).as_posix(),catalog)
    handoff.validate(root,chosen)
    assert chosen['producer_evidence_ref']==read_json(path)['evidence_ref']
    assert path.read_bytes()==original
    assert not list((root/'research/derived').glob('measurements/*.json'))


@pytest.mark.parametrize('fault',['counter','source','source_same_quote','receipt','audit','native_log','worker_source','job_binding','missing_review'])
def test_long_producer_revalidates_every_proof_before_offer(tmp_path,fault):
    root,repo,bundle,path,job=long_producer(tmp_path)
    if fault=='counter': atomic_json(bundle.parent/'work.json',{'updates':99})
    elif fault=='source': (repo/'train.py').write_text('print("not training")')
    elif fault=='source_same_quote': (repo/'train.py').write_text('optimizer.zero_grad()\nstep += 1\nsave(step)\n')
    elif fault=='receipt':
        receipt=read_json(path);receipt['argv']=['echo'];atomic_json(path,receipt)
    elif fault=='audit': atomic_json(root/'training_progress_audits/audit.json',{})
    elif fault=='native_log': (root/read_json(path)['log_ref']).write_text('changed')
    elif fault=='worker_source':
        result=read_json(job/'result.json');result['source_identity_after']='d'*64;atomic_json(job/'result.json',result)
    elif fault=='job_binding':
        receipt=read_json(path);receipt['controller_decision_id']='e'*32;atomic_json(path,receipt)
    else: path.with_name('training_progress_revalidation.json').unlink()
    catalog=handoff.catalog(root)
    assert catalog['status']=='unavailable'
    assert catalog['pending_producers'][0]['job_id']==job.name


def test_long_job_inspection_does_not_create_research_protocol(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research import native_jobs
    real=preparation(tmp_path/'controller')
    root,repo,_,path,job=long_producer(tmp_path)
    real.output=root;real.repo=repo;real.run_id='derived'
    result=read_json(job/'result.json')
    monkeypatch.setattr(native_jobs,'status',lambda *a:{'status':'completed','result':result,
        'job_id':job.name,'job_ref':(job/'request.json').relative_to(root).as_posix()})
    monkeypatch.setattr(native_jobs,'source_identity',lambda *a:'c'*64)
    monkeypatch.setattr(real,'_research_controller',lambda:pytest.fail('inspection cannot create a scoring protocol'))
    answer=real._step_inspect_native_job(job_id=job.name)
    assert answer['training_progress_review']['status']=='verified'
    assert not (root/'research/derived/comparison_protocol.json').exists()


def test_one_catalog_keeps_agent_choice_between_two_origins(tmp_path):
    from tests.test_checkpoint_handoff import producer
    root,_,probe,_=producer(tmp_path)
    longroot,_,bundle,_,_=long_producer(tmp_path)
    # Composition must not choose the newest origin implicitly.
    offered=handoff.catalog(root)
    assert longroot==root and len(offered['producers'])==2
    selected=handoff.select(root,probe.relative_to(root).as_posix(),offered)
    assert selected['producer_evidence_ref']==offered['producers'][0]['native_evidence_ref']
    selected=handoff.select(root,bundle.relative_to(root).as_posix(),offered)
    assert selected['producer_evidence_ref']==offered['producers'][1]['native_evidence_ref']


def test_redacted_citation_roundtrip_uses_exact_opaque_source_id(tmp_path):
    original=[{'ref':'train.py','quote':f'dataset_root = "{tmp_path}/private-data"','sha256':'a'*64}]
    packet,shown=source_citations.project({},original,local_roots=(tmp_path,))
    assert str(tmp_path) not in json.dumps(packet)
    resolved=source_citations.resolve(original,shown,[{
        'source_id':shown[0]['source_id'],'quote':shown[0]['quote']}])
    assert resolved==original
    for bad in [{'source_id':'invented','quote':shown[0]['quote']},
                {'source_id':shown[0]['source_id'],'quote':original[0]['quote']},
                {'source_id':shown[0]['source_id'],'quote':shown[0]['quote']+' '}]:
        with pytest.raises(ValueError):source_citations.resolve(original,shown,[bad])


def test_temporary_job_ownership_can_retry_reservation_without_model_call(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research import native_jobs
    real=preparation(tmp_path)
    monkeypatch.setattr(native_jobs,'active_jobs',lambda *a:['a'*32])
    result=real._step_configure_confirmation_protocol(confirmation={'seed':1},source_refs=[],reason='test')
    assert result['failure_category']=='resource_busy' and result['retryable'] is True
    assert real.client.seen==[]
    atomic_json(real.output/'research'/real.run_id/'comparison_protocol.json',{})
    result=real._step_configure_confirmation_protocol(confirmation={'seed':1},source_refs=[],reason='test')
    assert result['failure_category']=='protocol_frozen' and result['retryable'] is False


def test_busy_then_release_preserves_original_redacted_reservation_sources(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    from autosim.research import native_jobs
    real=preparation(tmp_path)
    quote=f'return initial_bank[seed]  # local binding {tmp_path}/resources'
    (tmp_path/'evaluate.py').write_text(quote+'\n')
    calls=[];active=['a'*32]
    def reviewer(system,user,**options):
        packet=json.loads(user);calls.append(packet)
        shown=packet['review_sources'][0]
        assert str(tmp_path) not in user
        return json.dumps({'native_evaluation_input_supported':True,
            'disjoint_conditions_supported':True,'preserves_policy_and_metric':True,
            'source_connection_supported':True,'citations':[{
                'source_id':shown['source_id'],'quote':shown['quote']}]}),{}
    real.client.chat_with_metadata=reviewer
    monkeypatch.setattr(native_jobs,'active_jobs',lambda *a:list(active))
    arguments={'confirmation':{'seed':101},'source_refs':[{'ref':'evaluate.py','quote':quote}],
               'reason':'source-backed independent initial bank'}
    assert real._step_configure_confirmation_protocol(**arguments)['retryable'] is True
    assert not calls
    active.clear()
    answer=real._step_configure_confirmation_protocol(**arguments)
    assert answer['outcome']=='registered' and len(calls)==1
    saved=read_json(real.output/'confirmation_reservation.json')
    assert saved['review']['citations'][0]['quote']==quote
    assert saved['source_refs'][0]['quote']==quote


def test_data_registration_maps_model_visible_quote_back_without_path_guessing(tmp_path):
    from tests.test_prepare import preparation
    real=preparation(tmp_path)
    quote=f'dataset_root = "{tmp_path}/training-only"'
    (tmp_path/'train.py').write_text(quote+'\n')
    data=real.output/'training-data.json';data.parent.mkdir(exist_ok=True);data.write_text('[1,2,3]')
    def reviewer(system,user,**options):
        packet=json.loads(user);shown=packet['review_sources'][0]
        assert str(tmp_path) not in user
        return json.dumps({'allowed_training_data':True,'loader_connection_supported':True,
            'sources_supported':True,'citations':[{'source_id':shown['source_id'],
                                                  'quote':shown['quote']}]}),{}
    real.client.chat_with_metadata=reviewer
    answer=real._step_register_data_version(version={'id':'more','dataset_ref':data.name,
        'reason':'development-only toy fixture','source_refs':[{'ref':'train.py','quote':quote}]})
    assert answer['outcome']=='registered'
    saved=read_json(real.output/'data_versions/more.json')
    assert saved['review']['citations'][0]['quote']==quote
    assert saved['status']=='reviewed_not_consumed'


def test_framework_stop_requires_current_exact_failure_and_independent_review(tmp_path):
    from autosim.research.framework_stop_review import review
    calls=[]
    client=SimpleNamespace(supports_main_agent=True,as_role=lambda role:nullcontext(),timeout=5,
        chat_with_metadata=lambda *a,**kw:(calls.append(kw) or json.dumps({'approved':True,
            'why':'cannot proceed','evidence_id':'a'*32,'quote':'kernel fault'}),{}))
    controller=SimpleNamespace(output=tmp_path,repo=tmp_path,budget=None,client=client,
        last_action={'step':'train','failure_category':'resource_busy','retryable':True},
        state=lambda:{'available':['wait_for_jobs','inspect_native_job']})
    choice={'do':'stop','arguments':{'stop_scope':'framework'},'why':'blocked'}
    answer=review(controller,choice)
    assert answer['arguments']['stop_scope']=='pause'
    assert answer['decision_failure']['kind']=='framework_stop_unproven'
    assert calls[0]['read_only'] and not calls[0]['include_research_context']
    review(controller,choice)
    assert len(calls)==1
    log=tmp_path/'native.log';log.write_text('kernel fault')
    sealed=capture_attempt_evidence(tmp_path,attempt_id='a'*32,log=log,
        receipt_ref='failure.json',status='failed',returncode=1,termination_reason='failed')
    controller.last_action={'step':'train','failure_category':'framework_exception',**sealed}
    assert 'decision_failure' not in review(controller,choice)
    assert len(calls)==2


def test_milestones_expose_local_baseline_basis_without_rewriting_or_official_claim(tmp_path):
    from autosim.research.research_milestones import view
    root,_,_,receipt,_=long_producer(tmp_path)
    path=root/'research/derived/measurements/baseline.json'
    atomic_json(path,{'ok':True,'metric_value':0.0,'metric_reading':{'samples':100},
                     'train':{'attempt_id':read_json(receipt)['attempt_id']}})
    original=path.read_bytes()
    result=view(root,'derived',{'baseline_reference':{'status':'ready'}})
    basis=result['baseline_basis']
    assert basis['status']=='local_measurement_recorded' and basis['metric_value']==0
    assert not basis['official_reproduction']
    assert basis['data_consumption']=='not_recorded' and basis['actual_training_argv']
    assert path.read_bytes()==original


def test_new_measurement_retains_data_consumption_even_when_no_new_version(tmp_path):
    from tests.test_derived_research import research
    result=research(tmp_path).measure(settings={},label='baseline')
    assert result['ok'] and result['data_consumption']['status']=='unknown'


def test_reproduced_reference_does_not_relabel_an_unrelated_local_baseline(tmp_path):
    from autosim.research.research_milestones import view
    root=tmp_path/'run';path=root/'research/derived/measurements/baseline.json'
    atomic_json(path,{'ok':True,'metric_value':0,'data_consumption':{'status':'unknown'}})
    facts={'baseline_reference':{'status':'reproduced',
                               'measurement_ref':'research/derived/measurements/reference_published.json'}}
    basis=view(root,'derived',facts)['baseline_basis']
    assert basis['released_reference_reproduced'] and not basis['official_reproduction']
    assert basis['data_consumption']=='unknown'
    facts['baseline_reference']['measurement_ref']='research/derived/measurements/baseline.json'
    assert view(root,'derived',facts)['baseline_basis']['official_reproduction']


def test_initial_native_training_adoption_reaches_real_cpu_evaluation_without_retraining(tmp_path,monkeypatch):
    from tests.test_derived_research import ambiguous_policy_research
    from tests.test_prepare import preparation
    from autosim.research import native_jobs
    research=ambiguous_policy_research(tmp_path)
    job_id='b'*32
    research.controller_decision_id=job_id
    trained=research.run_stage('train',settings={})
    assert trained['returncode']==0
    assert not (research.run_root/'comparison_protocol.json').exists()
    atomic_json(research.output/'native_jobs'/job_id/'request.json',{
        'repo':str(research.repo),'run_id':research.run_id,'settings':{},'source_identity':'c'*64})
    monkeypatch.setattr(native_jobs,'status',lambda *a:{'status':'completed',
        'result':{'stage_result':trained}})
    monkeypatch.setattr(native_jobs,'source_identity',lambda *a:'c'*64)
    real=preparation(tmp_path)
    real.declaration={'benchmark':'synthetic'};real.interpreter=research.interpreter
    real.stages={'train':'verified','evaluate':'verified'}
    monkeypatch.setattr(real,'_research_loop_ready',lambda:True)
    monkeypatch.setattr(real,'_research_controller',lambda:(research,{}))
    native=research.run_stage
    def no_retraining(stage,**kwargs):
        assert stage!='train','paid completed training was run twice'
        return native(stage,**kwargs)
    monkeypatch.setattr(research,'run_stage',no_retraining)
    result=real._step_run_the_loop(job_id=job_id)
    measured=read_json(research.run_root/'measurements/baseline.json')
    assert measured['ok'] and measured['metric_value']==0.75
    assert measured['train']['attempt_id']==trained['attempt_id'] and measured['train']['reused']
    assert (research.run_root/'comparison_protocol.json').is_file()
