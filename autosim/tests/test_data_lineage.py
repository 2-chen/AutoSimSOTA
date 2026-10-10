"""Data evidence contracts, not simulation performance tests."""
import json

import pytest

from autosim.research import data_lineage, data_versions, recorder, run_record
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.experiment_bundle import artifact_identity
from tests.test_data_versions import registered
from tests.test_recorder import measurement, snapshot
from tests.test_prepare import preparation

REVIEW = dict(allowed_training_data=True,loader_connection_supported=True,sources_supported=True)


def producer(output, data, *, stage='collect', profile=None, status='completed', ident='a'*32):
    field, value = artifact_identity(data)
    profile = profile if profile is not None else dict(identity_field=field, identity=value,
        format='scalar JSON episodes (fixture)',scope='full',training_only=True,
        counts=dict(episodes=3,samples=3),issues=dict(invalid_actions=0),coverage=dict(task_count=1))
    artifact = {'matched':1,'examples':[data.name]}
    context = dict(stage=stage,output_directory=str(data.parent),working_directory=str(data.parent))
    receipt = 'producer/'+ident+'/receipt.json'
    atomic_json(output/receipt,dict(attempt_id=ident,status=status,artifact=artifact,**context))
    log = output/'producer'/ident/'native.log'
    log.write_text(data_lineage.PROFILE_MARKER+json.dumps(profile)+'\n')
    capture_attempt_evidence(output,attempt_id=ident,log=log,receipt_ref=receipt,
        status=status,returncode=0,termination_reason='normal_exit',artifact=artifact,native_context=context)
    return ident, profile


def spec(ref='data.json', **asset):
    return dict(id='new',dataset_ref=ref,source_refs=[],reason='training-only fixture',asset=asset)


def test_collected_asset_is_linked_to_successful_sealed_native_producer(tmp_path):
    repo,output,data,_ = registered(tmp_path)
    ident,_ = producer(output,data)
    card = data_versions.register(output,repo,spec(kind='collected',producer_evidence_id=ident),REVIEW)
    assert card['asset']['profile']['counts']['episodes'] == 3
    assert card['asset']['producer']['evidence_id'] == ident
    assert card['reason'] == 'training-only fixture'
    _,bindings = data_versions.resolve(output,repo,dict(dataset_root='data-version:new'))
    assert bindings[0]['asset']['kind'] == 'collected'
    assert data_versions.catalog(output)[1]['producer_ref'] == f'evidence/{ident}.json'
    assert data_versions.catalog(output,offset=1,limit=1)[0]['id'] == 'new'


@pytest.mark.parametrize('change', ['unsealed','failed','wrong_stage','wrong_identity','wrong_context',
                                   'duplicate_profile','heldout','negative','bool_count','outcomes',
                                   'unknown_field','wrong_artifact'])
def test_invalid_native_asset_claim_is_not_registered(tmp_path,change):
    repo,output,data,_ = registered(tmp_path)
    ident, profile = producer(output,data)
    if change == 'unsealed':
        ident = 'b'*32
    elif change in {'failed','wrong_stage','wrong_identity','heldout','negative','bool_count','outcomes','unknown_field'}:
        if change=='wrong_identity':profile['identity']='0'*64
        if change=='heldout':profile['training_only']=False
        if change=='negative':profile['counts']['samples']=-1
        if change=='bool_count':profile['counts']['samples']=True
        if change=='outcomes':profile['counts'].update(successful_episodes=3,failed_episodes=2)
        if change=='unknown_field':profile['episode_rows']=[dict(action=1)]
        ident,_=producer(output,data,ident='c'*32,profile=profile,
            stage='train' if change=='wrong_stage' else 'collect',status='failed' if change=='failed' else 'completed')
    elif change in {'wrong_context','wrong_artifact'}:
        path=output/'producer'/ident/'receipt.json';row=read_json(path)
        if change=='wrong_context':row['output_directory']=str(output/'unrelated')
        else:row['artifact']['examples']=['unrelated.json']
        atomic_json(path,row)
    else:
        log=output/'producer'/ident/'native.log';log.write_text(log.read_text()*2)
    with pytest.raises((ValueError,OSError)):
        data_versions.register(output,repo,spec(kind='collected',producer_evidence_id=ident),REVIEW)
    assert not (output/'data_versions/new.json').exists()


def processed(tmp_path):
    repo,output,raw,parent=registered(tmp_path)
    view=output/'selected.json';view.write_text('[1,3]')
    field,value=artifact_identity(view)
    profile=dict(identity_field=field,identity=value,scope='full',format='scalar JSON episodes (fixture)',
        training_only=True,counts=dict(episodes=2,samples=2),
        parents=[dict(id=parent['id'],identity_field=parent['identity_field'],identity=parent['identity'],
                      record_sha256=digest(output/'data_versions/extra.json'))],
        recipe=dict(operation='select complete episodes',parameters=dict(selected_ids=[0,2]),
                    preserves_episode_groups=True))
    ident,_=producer(output,view,stage='prepare_data',profile=profile)
    card=data_versions.register(output,repo,spec(ref='selected.json',kind='processed',parents=['extra'],
        producer_evidence_id=ident),REVIEW)
    return repo,output,raw,view,card


def test_processed_view_retains_raw_identity_recipe_and_parent(tmp_path):
    repo,output,raw,view,card=processed(tmp_path)
    assert raw.read_text()=='[1,2,3]'
    assert view.read_text()=='[1,3]'
    assert card['asset']['parents'][0]['id']=='extra'
    assert card['asset']['recipe']['parameters']['selected_ids']==[0,2]
    settings,_=data_versions.resolve(output,repo,dict(dataset_root='data-version:new'))
    assert settings['dataset_root']==str(view)
    raw.write_text('[4]')
    with pytest.raises(ValueError,match='parent data bytes changed'):
        data_versions.resolve(output,repo,dict(dataset_root='data-version:new'))


@pytest.mark.parametrize('asset', [dict(kind='processed'),dict(kind='collected'),
    dict(kind='processed',parents=['extra'],producer_evidence_id='a'*32),
    dict(kind='existing',parents=['extra']),dict(kind='existing',parents=['missing']),
    dict(kind='existing',parents=['extra','extra']),dict(kind='existing',license='')])
def test_asset_requires_nonoverlapping_valid_parents_and_evidence(tmp_path,asset):
    repo,output,data,_=registered(tmp_path)
    producer(output,data)
    with pytest.raises((ValueError,OSError)):
        data_versions.register(output,repo,spec(**asset),REVIEW)


def consumption(output, bindings, exposure, *, samples=6, status='completed'):
    binding=bindings[0]
    log=output/'train.log';ident='d'*32
    row=dict(data_version_id=binding['id'],setting=binding['setting'],
        identity_field=binding['identity_field'],identity=binding['identity'],samples_read=samples)
    if exposure is not None:row['exposure']=exposure
    log.write_text(data_versions.MARKER+json.dumps(row))
    capture_attempt_evidence(output,attempt_id=ident,log=log,receipt_ref='train.json',
        status=status,returncode=0,termination_reason='normal_exit')
    return data_versions.verify_consumption(output,bindings,dict(evidence_id=ident))


@pytest.mark.parametrize('exposure,valid', [
    (dict(batches=3,unique_samples=2,source_samples=dict(extra=6)),True),
    (None,True), (dict(batches=True),False), (dict(batches=0),False),
    (dict(unique_samples=9),False), (dict(unique_samples=3),False),
    (dict(source_samples=dict(unknown=6)),False),(dict(source_samples=dict(extra=2)),False),
    (dict(source_samples=dict(extra=-6)),False),(dict(configured_ratio=1),False)])
def test_exposure_uses_actual_read_counts_not_requested_mixture(tmp_path,exposure,valid):
    repo,output,_,_,_=processed(tmp_path)
    _,bindings=data_versions.resolve(output,repo,dict(dataset_root='data-version:new'))
    result=consumption(output,bindings,exposure)
    assert (result['status']=='verified') is valid
    if valid:
        assert result['exposure'][0]['samples_read']==6
        assert result['exposure'][0]['status']==('basic_read_only' if exposure is None else 'aggregate_reported')


def test_changed_parent_during_training_prevents_formal_data_adoption(tmp_path):
    repo,output,raw,_,_=processed(tmp_path)
    _,bindings=data_versions.resolve(output,repo,dict(dataset_root='data-version:new'))
    raw.write_text('[7]')
    assert consumption(output,bindings,None)['status']=='unverified'


def test_failed_trainer_cannot_certify_data_consumption(tmp_path):
    repo,output,_,_,_=processed(tmp_path)
    _,bindings=data_versions.resolve(output,repo,dict(dataset_root='data-version:new'))
    assert consumption(output,bindings,None,status='failed')['status']=='unverified'


@pytest.mark.parametrize('change',['recipe','parent','log','receipt','card'])
def test_processed_evidence_remains_bound_after_registration(tmp_path,change):
    repo,output,raw,view,card=processed(tmp_path)
    if change in {'log','receipt'}:
        witness=card['asset']['producer']
        path=output/witness['receipt_ref'] if change=='receipt' else output/'producer'/witness['evidence_id']/'native.log'
        path.write_text(path.read_text()+'\n ')
        with pytest.raises(ValueError):data_versions.resolve(output,repo,dict(dataset_root='data-version:new'))
    elif change=='card':
        path=output/'data_versions/extra.json';row=read_json(path);row['reason']='changed';atomic_json(path,row)
        with pytest.raises(ValueError):data_versions.resolve(output,repo,dict(dataset_root='data-version:new'))
    else:
        field,value=artifact_identity(view)
        profile=dict(identity_field=field,identity=value,scope='full',format='fixture',training_only=True,
                     parents=[{k:p[k] for k in ('id','identity_field','identity','record_sha256')}
                              for p in card['asset']['parents']],recipe=card['asset']['recipe'])
        if change=='recipe':profile['recipe']['preserves_episode_groups']=False
        else:profile['parents'][0]['identity']='0'*64
        ident,_=producer(output,view,stage='prepare_data',profile=profile,ident='b'*32)
        request=spec(ref='selected.json',kind='processed',parents=['extra'],producer_evidence_id=ident);request['id']='next'
        with pytest.raises(ValueError):data_versions.register(output,repo,request,REVIEW)


def test_descendant_rechecks_grandparent_bytes(tmp_path):
    repo,output,raw,view,card=processed(tmp_path)
    next_data=output/'next.json';next_data.write_text('[3]')
    field,value=artifact_identity(next_data)
    profile=dict(identity_field=field,identity=value,scope='full',format='fixture',training_only=True,
        parents=[dict(id='new',identity_field=card['identity_field'],identity=card['identity'],
                      record_sha256=digest(output/'data_versions/new.json'))],
        recipe=dict(operation='select complete episodes',parameters=dict(selected_ids=[1]),preserves_episode_groups=True))
    ident,_=producer(output,next_data,stage='prepare_data',profile=profile,ident='b'*32)
    request=spec(ref='next.json',kind='processed',parents=['new'],producer_evidence_id=ident);request['id']='next'
    data_versions.register(output,repo,request,REVIEW)
    raw.write_text('[9]')
    with pytest.raises(ValueError):data_versions.resolve(output,repo,dict(dataset_root='data-version:next'))


def test_supervisor_receives_real_card_before_registration(tmp_path):
    real=preparation(tmp_path)
    real.output.mkdir(parents=True,exist_ok=True)
    data=real.output/'data.json';data.write_text('[1,2,3]')
    (real.repo/'loader.py').write_text('samples = read_training_samples()\n')
    ident,_=producer(real.output,data)
    request=spec(kind='collected',producer_evidence_id=ident)
    request['source_refs']=[dict(ref='loader.py',quote='samples = read_training_samples()')]
    real.client.choices=[{**REVIEW,'citations':request['source_refs'],'why':'fixture review'}]
    result=real._step_register_data_version(version=request)
    assert result['outcome']=='registered'
    payload=json.loads(real.client.seen[-1])
    assert payload['native_data_card']['asset']['profile']['counts']['episodes']==3
    assert payload['native_data_card']['asset']['producer']['evidence_id']==ident


def test_chinese_data_panel_joins_lineage_exposure_and_measurement(tmp_path):
    repo,output,_,_,_=processed(tmp_path)
    path=measurement(output,'round_1',0.5)
    row=read_json(path)
    row['data_consumption']=dict(status='verified',versions=['new'],
        exposure=[dict(data_version_id='new',samples_read=6,batches=3,unique_samples=2,source_samples=dict(extra=6))])
    atomic_json(path,row)
    held=snapshot(output)
    assert held['measurements'][0]['data_summary']=='new 实读 6 样本次'
    doc=recorder.render(output,held,{})
    assert '数据资产与使用情况' in doc and '可逆处理结果' in doc and '2 条 episode' in doc
    assert '哪些数据真正进入训练' in doc and 'extra：6' in doc
    assert 'reward：0.5' in doc and '6 样本次' in doc
    # Only full-profile totals produce the data-size chart, never unknown/raw size.
    assert '-data-episodes.svg' not in doc


def test_full_counts_chart_does_not_promote_sample_profile_to_total(tmp_path):
    repo,output,data,_=registered(tmp_path)
    ident,profile=producer(output,data)
    data_versions.register(output,repo,spec(kind='collected',producer_evidence_id=ident),REVIEW)
    profile['scope']='sample'
    ident,_=producer(output,data,ident='b'*32,profile=profile)
    second=spec(kind='collected',producer_evidence_id=ident);second['id']='sampled'
    data_versions.register(output,repo,second,REVIEW)
    doc=recorder.render(output,snapshot(output),{})
    assert '抽样检查，非全量' in doc and '-data-episodes.svg' not in doc
    third=spec(kind='collected',producer_evidence_id='a'*32);third['id']='full_again'
    data_versions.register(output,repo,third,REVIEW)
    assert '-data-episodes.svg' in recorder.render(output,snapshot(output),{})
    assert list((output/'report/charts').glob('*-data-episodes.svg'))


def test_development_evidence_exposes_bounded_reads_not_raw_samples(tmp_path):
    from autosim.research.development_evidence import summaries
    path=measurement(tmp_path,'round_1',0.5)
    row=read_json(path)
    row['data_consumption']=dict(status='verified',versions=['new'],exposure=[dict(
        data_version_id='new',status='aggregate_reported',samples_read=6,batches=3,unique_samples=2,
        source_samples=dict(extra=6),episode_rows='DO_NOT_EXPORT')])
    atomic_json(path,row)
    evidence=summaries(tmp_path/'research/derived')
    assert evidence[0]['data_exposure'][0]['source_samples']=={'extra':6}
    assert len(evidence[0]['data_exposure_digest'])==64
    assert 'DO_NOT_EXPORT' not in json.dumps(evidence)
    before=evidence[0]['data_exposure_digest']
    row['data_consumption']['exposure'][0]['samples_read']=8
    row['data_consumption']['exposure'][0]['source_samples']['extra']=8
    atomic_json(path,row)
    assert summaries(tmp_path/'research/derived')[0]['data_exposure_digest']!=before


def test_catalog_is_metadata_only_and_registration_never_overwrites(tmp_path,monkeypatch):
    repo,output,_,_,_=processed(tmp_path)
    monkeypatch.setattr(data_versions,'artifact_identity',lambda *args:pytest.fail('catalog reads raw bytes'))
    assert len(data_versions.catalog(output))==2
    for kwargs in (dict(offset=-1),dict(limit=0),dict(limit=True)):
        with pytest.raises(ValueError):data_versions.catalog(output,**kwargs)
    before=digest(output/'data_versions/new.json')
    with pytest.raises(ValueError,match='immutable'):
        data_versions.register(output,repo,spec(ref='selected.json',kind='existing'),REVIEW)
    assert digest(output/'data_versions/new.json')==before
