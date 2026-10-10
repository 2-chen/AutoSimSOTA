"""Only disposable fixture payloads are removed; no real experiment is touched."""
import json
import os
from collections import namedtuple
from pathlib import Path

import pytest

from autosim.research import storage_lifecycle as sl, storage_controls as sc, storage_runtime as sr
from autosim.research.common import atomic_json, read_json
from autosim.research.evidence_store import capture_attempt_evidence


def producer(tmp_path,stage='train',verification=False):
    output=tmp_path/'run'
    identity='a'*32
    folder=output/('verification_outputs/probe' if verification else 'research/test/train/native_outputs/attempt')
    folder.mkdir(parents=True)
    log=folder/'native.log'
    log.write_text('completed updates 8192\n')
    receipt=output/'research/test/attempts'/identity/'receipt.json'
    atomic_json(receipt,{'attempt_id':identity,'stage':stage,'status':'completed','output_directory':str(folder)})
    capture_attempt_evidence(output,attempt_id=identity,log=log,
        receipt_ref=receipt.relative_to(output).as_posix(),status='completed',returncode=0,
        termination_reason='normal_exit')
    for name in ('middle.pt','final.pt','resume.pt'):
        (folder/name).write_bytes(name.encode())
    return output,folder,identity


def group(output,folder,identity,name,role):
    return sl.register(output,evidence_id=identity,refs=[(folder/name).relative_to(output).as_posix()],
                       role=role,reason='source-reviewed native group',source_evidence={'verified':True})


def approved_plan(output,ids):
    return sl.plan(output,artifact_ids=ids,reason='unscored intermediate superseded',
        review={'verdict':'verified','artifact_ids':ids,'unneeded_for_recovery':True})


def test_gc_operator_authorization_and_terminal_metadata_retention(tmp_path):
    output,folder,identity=producer(tmp_path)
    middle=group(output,folder,identity,'middle.pt','intermediate')
    final=group(output,folder,identity,'final.pt','terminal')
    plan=approved_plan(output,[middle['id']])
    assert sl.commit(output,plan_id=plan['id'])['status']=='authorization_required'
    assert (folder/'middle.pt').exists()
    sl.configure(output,auto_gc=True)
    result=sl.commit(output,plan_id=plan['id'])
    assert result['status']=='completed'
    assert result['removed_bytes']==len(b'middle.pt')
    assert not (folder/'middle.pt').exists()
    assert (folder/'final.pt').exists() and (folder/'native.log').exists()
    assert next(r for r in sl.catalog(output) if r['id']==final['id'])['protected']
    assert sl.commit(output,plan_id=plan['id'])['status']=='completed'
    assert sl.pruned_refs(output)=={(folder/'middle.pt').relative_to(output).as_posix()}


def test_no_intermediate_gc_without_producer_recovery(tmp_path):
    output,folder,identity=producer(tmp_path)
    middle=group(output,folder,identity,'middle.pt','intermediate')
    with pytest.raises(ValueError,match='terminal/recovery'):
        approved_plan(output,[middle['id']])


@pytest.mark.parametrize('mutation',['bytes','new_measurement','terminal_bytes','plan_tampering'])
def test_commit_rechecks_contents_and_references(tmp_path,mutation):
    output,folder,identity=producer(tmp_path)
    middle=group(output,folder,identity,'middle.pt','intermediate')
    group(output,folder,identity,'final.pt','terminal')
    plan=approved_plan(output,[middle['id']])
    sl.configure(output,auto_gc=True)
    if mutation=='bytes': (folder/'middle.pt').write_bytes(b'changed')
    elif mutation=='terminal_bytes': (folder/'final.pt').write_bytes(b'changed')
    elif mutation=='new_measurement':
        atomic_json(output/'research/test/measurements/candidate.json',{'policy_artifact':{'source':str(folder/'middle.pt')}})
    else:
        path=output/'storage/plans'/f"{plan['id']}.json"
        row=read_json(path); row['reason']='changed'
        atomic_json(path,row)
    with pytest.raises(ValueError): sl.commit(output,plan_id=plan['id'])
    assert (folder/'middle.pt').exists()


@pytest.mark.parametrize('kind',['train','collect','metadata','symlink','external','verification_role'])
def test_invalid_or_protected_artifact_registration(tmp_path,kind):
    output,folder,identity=producer(tmp_path,stage='collect' if kind=='collect' else 'train')
    ref=(folder/'middle.pt').relative_to(output).as_posix()
    role='intermediate'
    if kind=='metadata': ref=(folder/'native.log').relative_to(output).as_posix()
    elif kind=='symlink':
        (folder/'link.pt').symlink_to(folder/'middle.pt'); ref=(folder/'link.pt').relative_to(output).as_posix()
    elif kind=='external': ref='../outside.pt'
    elif kind=='verification_role': role='verification'
    elif kind=='train':
        atomic_json(output/'research/test/measurements/baseline.json',{'policy_artifact':{'source':str(folder/'middle.pt')}})
        row=group(output,folder,identity,'middle.pt','intermediate')
        with pytest.raises(ValueError,match='protected'): approved_plan(output,[row['id']])
        return
    with pytest.raises(ValueError):
        sl.register(output,evidence_id=identity,refs=[ref],role=role,reason='test',source_evidence=[])


def test_stage_lease_excludes_gc_and_controls(tmp_path):
    output,folder,identity=producer(tmp_path)
    with sl.stage_lease(output):
        with pytest.raises(ValueError,match='stage lease'): group(output,folder,identity,'middle.pt','intermediate')
    assert group(output,folder,identity,'middle.pt','intermediate')['status']=='retained'


def test_partial_deletion_reconciles_without_replay(tmp_path,monkeypatch):
    output,folder,identity=producer(tmp_path)
    row=sl.register(output,evidence_id=identity,refs=[(folder/name).relative_to(output).as_posix()
        for name in ('middle.pt','resume.pt')],role='intermediate',reason='test',source_evidence=[])
    group(output,folder,identity,'final.pt','terminal')
    plan=approved_plan(output,[row['id']])
    sl.configure(output,auto_gc=True)
    unlink=Path.unlink
    def interrupted(path,*args,**kwargs):
        if path.name=='resume.pt': raise OSError('simulated interrupted unlink')
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',interrupted)
    with pytest.raises(OSError): sl.commit(output,plan_id=plan['id'])
    assert not (folder/'middle.pt').exists() and (folder/'resume.pt').exists()
    assert sl.commit(output,plan_id=plan['id'])['status']=='committing'
    with pytest.raises(ValueError,match='reconciliation'):
        with sl.stage_lease(output): pytest.fail('must not admit a job')
    result=sl.reconcile(output,plan_id=plan['id'])
    assert result['status']=='reconciled_partial'
    assert (folder/'resume.pt').exists()
    assert sl.catalog(output)[0]['status'] in {'retained','partially_pruned'}


def test_hardlink_of_measured_policy_is_protected(tmp_path):
    output,folder,identity=producer(tmp_path)
    alias=output/'measured/policy.pt'; alias.parent.mkdir()
    os.link(folder/'middle.pt',alias)
    atomic_json(output/'research/test/measurements/baseline.json',{'policy_artifact':{'path':str(alias)}})
    row=group(output,folder,identity,'middle.pt','intermediate')
    assert next(r for r in sl.catalog(output) if r['id']==row['id'])['protected']


class Reviewer:
    def __init__(self,verdict='verified'): self.verdict=verdict; self.calls=0
    def chat_with_metadata(self,system,user,**kwargs):
        self.calls+=1
        body=json.loads(user)
        source=body['native_sources'][0]
        packet=body['operation']
        return json.dumps({'verdict':self.verdict,'why':'native source preserves final save',
            'source_evidence':[{'path':source['path'],'quote':source['quote']}],
            'preserves_final_and_selection':True,'preserves_learning_and_protocol':True,
            'role':packet.get('role'),'artifact_ids':packet.get('artifact_ids'),
            'unneeded_for_recovery':True}),{}


def controls_fixture(tmp_path):
    output=tmp_path/'out'; output.mkdir()
    repo=output/'checkout'; repo.mkdir()
    (repo/'train.py').write_text('save_interval = args.save_interval\nif done: save_final()\n')
    citations=[{'path':'train.py','quote':'if done: save_final()'}]
    return output,repo,citations


def test_source_reviewed_controls_override_literal_without_changing_learning(tmp_path):
    output,repo,citations=controls_fixture(tmp_path)
    client=Reviewer()
    row=sc.configure(output,repo,client,controls=[{'flag':'--save-interval','value':4096}],
                     reason='avoid full snapshots every 500 updates',source_evidence=citations,
                     checkpoint_bytes=1000,projected_bytes=2000)
    argv=[str(repo/'train.py'),'--save-interval','500','--steps','8192']
    result,reviewed=sc.apply(output,repo,stage='train',argv=argv)
    assert result==[str(repo/'train.py'),'--save-interval','4096','--steps','8192']
    assert argv[2]=='500' and reviewed['id']==row['id']
    assert sc.apply(output,repo,stage='evaluate',argv=['eval'])[0]==['eval']
    assert client.calls==1
    (repo/'train.py').write_text('different producer')
    with pytest.raises(ValueError,match='source changed'): sc.apply(output,repo,stage='train',argv=argv)


@pytest.mark.parametrize('controls',[ [{'flag':'../unsafe','value':1}], [{'flag':'--save','value':True}],
    [{'flag':'--save','value':0}], [{'flag':'--save','value':1},{'flag':'--save','value':2}] ])
def test_bad_controls_never_reach_model(tmp_path,controls):
    output,repo,citations=controls_fixture(tmp_path)
    client=Reviewer()
    with pytest.raises(ValueError): sc.configure(output,repo,client,controls=controls,reason='test',source_evidence=citations)
    assert client.calls==0


def test_unverified_controls_not_written(tmp_path):
    output,repo,citations=controls_fixture(tmp_path)
    with pytest.raises(ValueError,match='did not approve'):
        sc.configure(output,repo,Reviewer('unverified'),controls=[{'flag':'--save','value':10}],
                     reason='test',source_evidence=citations)
    assert not (output/'storage/controls/train.json').exists()


def test_runtime_observation_cancel_and_no_model_poll(tmp_path,monkeypatch):
    monkeypatch.setenv('AUTOSIM_MIN_FREE_STORAGE_BYTES','200000000')
    usage=namedtuple('usage','total used free')
    free=[400000000]
    from autosim.research import storage_guard
    monkeypatch.setattr(storage_guard.shutil,'disk_usage',lambda p:usage(500000000,500000000-free[0],free[0]))
    watch=sr.StorageWatch(tmp_path,stage='train',attempt_id='b'*32,path=tmp_path,deadline_epoch=1000,checkpoint_bytes=100000000)
    assert not watch()['cancelled'] and watch.status=='normal'
    first=watch.record_path.stat().st_mtime_ns
    assert not watch()['cancelled'] and watch.record_path.stat().st_mtime_ns==first
    free[0]=150000000
    assert not watch()['cancelled'] and watch.status=='pressure'
    free[0]=100000000
    assert watch()['cancellation_reason']=='storage_pressure'
    panel=sr.report_panel('# 研究记录\n',tmp_path)
    assert '空间危险' in panel and '未开启' in panel and '未保存更新' in panel
    assert sr.report_panel(panel,tmp_path).count('## 存储与中间产物')==1


def test_full_disk_persistence_failure_still_cancels(tmp_path,monkeypatch):
    watch=sr.StorageWatch(tmp_path,stage='train',attempt_id='c'*32,path=tmp_path,deadline_epoch=1000)
    monkeypatch.setattr(sr,'atomic_json',lambda *a: (_ for _ in ()).throw(OSError('ENOSPC')))
    assert watch()['cancelled']
    assert 'ENOSPC' in watch.observation['persistence_error']


def test_cooperative_reservations_across_two_runs(tmp_path,monkeypatch):
    from autosim.research import storage_reservations as reservations, storage_guard
    usage=namedtuple('usage','total used free')(1000,0,1000)
    monkeypatch.setattr(reservations,'_directory',lambda:tmp_path/'private-reservations')
    monkeypatch.setenv('AUTOSIM_MIN_FREE_STORAGE_BYTES','1')
    monkeypatch.setattr(storage_guard.shutil,'disk_usage',lambda p:usage)
    with reservations.reserve(tmp_path/'first',stage='train',required_bytes=700):
        with pytest.raises(ValueError,match='held'):
            with reservations.reserve(tmp_path/'second',stage='train',required_bytes=400): pass
    with reservations.reserve(tmp_path/'second',stage='train',required_bytes=400) as row:
        assert row['other_reserved_bytes']==0


def test_content_store_deduplicates_and_isolates_native_bytes(tmp_path):
    from autosim.research.experiment_bundle import freeze_artifact
    source=tmp_path/'native/policy.pt'; source.parent.mkdir(); source.write_bytes(b'weights')
    store=tmp_path/'store'
    first=freeze_artifact(source,tmp_path/'one/policy.pt',max_bytes=100,store=store)
    second=freeze_artifact(source,tmp_path/'two/policy.pt',max_bytes=100,store=store)
    assert first['path']==second['path']
    assert Path(first['path']).stat().st_ino!=source.stat().st_ino
    source.write_bytes(b'mutated')
    assert Path(first['path']).read_bytes()==b'weights'
    Path(first['path']).write_bytes(b'corrupt')
    source.write_bytes(b'weights')
    with pytest.raises(ValueError,match='changed'): freeze_artifact(source,tmp_path/'three/policy.pt',max_bytes=100,store=store)


def test_main_operations_require_review_and_do_not_enable_policy(tmp_path):
    from tests.test_prepare import preparation
    real=preparation(tmp_path)
    output,folder,identity=producer(tmp_path/'payload')
    real.output=output; real.repo=output/'checkout'; real.repo.mkdir()
    (real.repo/'train.py').write_text('if done: save_final()\n')
    real.client=Reviewer()
    refs=[(folder/'final.pt').relative_to(output).as_posix()]
    result=real._step_register_storage_artifact(evidence_id=identity,refs=refs,role='terminal',
        reason='completed native final',source_evidence=[{'path':'train.py','quote':'if done: save_final()'}])
    assert result['outcome']=='registered' and real.client.calls==1
    assert sl.policy(output)['auto_gc'] is False


def test_retired_artifacts_no_longer_offered_to_policy_selector(tmp_path):
    from tests.test_derived_research import research
    real=research(tmp_path)
    output=real.output
    ref='research/test/native/middle/policy.pt'
    atomic_json(output/'storage/plans'/('f'*64+'.json'),{'removed_refs':[ref]})
    kept,why=real._selection_candidates({'output_directory':str(output/'research/test/native'),
        'artifact':{'matched':2,'candidate_paths':['middle','final']}})
    assert kept==['final'] and not why


def test_streaming_native_disk_pressure_reaches_sealed_receipt(tmp_path,monkeypatch):
    import sys
    from autosim.research import stage_verification, common
    monkeypatch.setattr(common,'isolated_argv',lambda argv,**kw:argv)
    original=sr.StorageWatch.__call__
    def dangerous(watch,base=None):
        watch.critical=True
        watch.observation={'status':'critical','free_bytes':0}
        return {'deadline_epoch':watch.deadline_epoch,'cancelled':True}
    monkeypatch.setattr(sr.StorageWatch,'__call__',dangerous)
    result=stage_verification.run_native([sys.executable,'-c','import time; time.sleep(30)'],
        repo=tmp_path,output=tmp_path,cwd=tmp_path,env={},timeout=10,stdin=None,stage='train')
    assert result.returncode!=0
    row=read_json(next((tmp_path/'derivation_native').glob('*/receipt.json')))
    assert row['termination_reason']=='storage_pressure' and row['status']=='failed'
    assert (tmp_path/row['evidence_ref']).exists()
    assert not list((tmp_path/'storage/stage_leases').glob('*.json'))


def test_storage_projection_blocks_before_formal_native_launch(tmp_path,monkeypatch):
    from tests.test_derived_research import research
    from autosim.research import storage_guard
    real=research(tmp_path)
    usage=namedtuple('usage','total used free')(10000000000,0,3000000000)
    monkeypatch.setattr(storage_guard.shutil,'disk_usage',lambda p:usage)
    real.job_resources={'storage_required_bytes':4000000000}
    monkeypatch.setattr(real,'_execute_stage',lambda *a,**kw:pytest.fail('must not start'))
    row=real._run_stage('train',settings={})
    assert row['termination_reason']=='storage_admission_unavailable'
    assert row['ran'] is False


def test_incomplete_cas_manifest_is_reconciled_without_overwriting(tmp_path):
    from autosim.research.experiment_bundle import freeze_artifact
    source=tmp_path/'policy.pt'; source.write_bytes(b'weights')
    row=freeze_artifact(source,tmp_path/'dest',max_bytes=100,store=tmp_path/'store')
    manifest=Path(row['path']).parent/'manifest.json'
    manifest.unlink()
    before=Path(row['path']).stat().st_ino
    recovered=freeze_artifact(source,tmp_path/'another',max_bytes=100,store=tmp_path/'store')
    assert recovered['path']==row['path'] and Path(row['path']).stat().st_ino==before
    assert manifest.exists()


def test_retention_review_source_change_vetoes_gc(tmp_path):
    from autosim.research.common import digest
    output,folder,identity=producer(tmp_path)
    checkout=output/'checkout'; checkout.mkdir()
    source=checkout/'train.py'; source.write_text('save_final()')
    middle=group(output,folder,identity,'middle.pt','intermediate')
    group(output,folder,identity,'final.pt','terminal')
    row=sl.plan(output,artifact_ids=[middle['id']],reason='reviewed obsolete group',
                review={'verdict':'verified','artifact_ids':[middle['id']],
                        'unneeded_for_recovery':True,'sources':[{'path':'train.py','sha256':digest(source)}]})
    sl.configure(output,auto_gc=True)
    source.write_text('new producer')
    with pytest.raises(ValueError,match='source changed'): sl.commit(output,plan_id=row['id'])
    assert (folder/'middle.pt').exists()


def test_registered_dependency_citations_are_bound_to_validated_context(tmp_path,monkeypatch):
    from autosim.research import native_context
    output,repo,citations=controls_fixture(tmp_path)
    dependency=tmp_path/'bound'; (dependency/'lib').mkdir(parents=True)
    path=dependency/'lib/save.py'; path.write_text('save_final()')
    monkeypatch.setattr(native_context,'load_context',lambda output,repo:{'dependency_roots':[str(dependency)]})
    assert sc.source_path(output,repo,'dependency:0:lib/save.py')==path
    for ref in ('dependency:1:lib/save.py','dependency:0:../outside','dependency:0:/etc/passwd'):
        with pytest.raises(ValueError): sc.source_path(output,repo,ref)


@pytest.mark.parametrize('mutate', [False, True])
def test_storage_review_supplies_registered_import_sources_and_rechecks_bytes(tmp_path, monkeypatch, mutate):
    from autosim.research import native_context, training_progress_audit
    from autosim.research.common import digest
    output, repo, citations = controls_fixture(tmp_path)
    dependency = tmp_path/'installed'
    dependency.mkdir()
    native = dependency/'save.py'
    native.write_text('if completed: save_final()\n')
    outside = tmp_path/'outside.py'
    outside.write_text('not registered; must never be supplied')
    atomic_json(output/'native_context.json', {})
    monkeypatch.setattr(native_context, 'load_context',
        lambda *a: {'dependency_roots':[str(dependency)]})
    monkeypatch.setattr(training_progress_audit, 'sources', lambda *a: [
        {'path':p, 'sha256':digest(p), 'content':p.read_text()} for p in (native, outside)])

    class NativeReviewer(Reviewer):
        def chat_with_metadata(self, system, user, **kw):
            packet = json.loads(user)
            sources = packet['native_sources']
            assert all('not registered' not in r['source'] for r in sources)
            offered = next(r for r in sources if r['path']=='dependency:0:save.py')
            assert offered['source'] == 'if completed: save_final()\n'
            assert packet['source_discovery']['complete'] is False
            if mutate:
                native.write_text('changed native producer')
            return json.dumps({'verdict':'verified', 'why':'native terminal save survives',
                'source_evidence':[{'path':offered['path'], 'quote':'save_final()'}],
                'preserves_final_and_selection':True, 'preserves_learning_and_protocol':True}), {}

    if mutate:
        with pytest.raises(ValueError, match='source changed'):
            sc.configure(output, repo, NativeReviewer(), controls=[{'flag':'--save','value':10}],
                         reason='reduce payload writes', source_evidence=citations)
        assert not (output/'storage/controls/train.json').exists()
    else:
        result = sc.configure(output, repo, NativeReviewer(), controls=[{'flag':'--save','value':10}],
                              reason='reduce payload writes', source_evidence=citations)
        assert any(r['path']=='dependency:0:save.py' for r in result['review']['sources'])


def test_storage_skill_is_body_free_and_explicitly_readable():
    from autosim.research.skills import skills_catalog, read_selected_skills
    identity='autosimsota.managing-research-artifacts'
    row=next(r for r in skills_catalog(benchmark='UnseenArmRepository')['entries'] if r['id']==identity)
    assert 'method' not in row
    selected=read_selected_skills([identity],benchmark='UnseenArmRepository')
    assert selected['selection'][0]['body_sha256']


def test_producer_inventory_pages_do_not_hide_terminal_tail(tmp_path):
    output,folder,identity=producer(tmp_path)
    for index in range(30): (folder/f'{index:03d}.pt').write_bytes(b'x')
    first=sr.inspect_producer(output,evidence_id=identity,limit=24)
    second=sr.inspect_producer(output,evidence_id=identity,offset=first['next_offset'],limit=24)
    assert first['total_files']==34 and second['next_offset'] is None
    assert len(first['files'])+len(second['files'])==34
    assert any(row['ref'].endswith('final.pt') for row in second['files'])


def test_registration_review_gets_execution_and_progress_not_just_file_names(tmp_path):
    output,folder,identity=producer(tmp_path)
    _,native=sl.native_owner(output,identity)
    assert 'completed updates 8192' in native['text']
    assert native['receipt_facts']['status']=='completed'
    assert native['receipt_facts']['stage']=='train'


def test_reconstructed_legacy_candidates_also_exclude_retired_payloads(tmp_path):
    from tests.test_derived_research import research
    real=research(tmp_path)
    producer=real.run_root/'train/native_outputs/legacy'
    for name in ('early','final'):
        (producer/'models'/name).mkdir(parents=True)
        (producer/'models'/name/'config.json').write_text('{}')
    ref=(producer/'models/early/policy.pt').relative_to(real.output).as_posix()
    atomic_json(real.output/'storage/plans'/('e'*64+'.json'),{'removed_refs':[ref]})
    atomic_json(real.run_root/'attempts/legacy/receipt.json',{'started_at':'2000-01-01T00:00:00+00:00'})
    kept,why=real._selection_candidates({'attempt_id':'legacy','stage':'train',
        'output_directory':str(producer),'artifact':{'matched':2,'pattern':'models/*',
        'examples':['models/early','models/final']}})
    assert kept==['models/final'] and not why


def test_publication_automatically_appends_storage_panel(tmp_path):
    from autosim.research.run_record import publish_document
    atomic_json(tmp_path/'storage/runtime/one.json',{'status':'normal','at':'fixture',
        'filesystem':{'free_bytes':10*1024**3}})
    path=publish_document(tmp_path/'RUN.md','# 研究记录\n\n当前已暂停。\n')
    text=path.read_text()
    assert '## 存储与中间产物' in text and '10.00 GiB' in text
    publish_document(path,text)
    assert path.read_text().count('## 存储与中间产物')==1


def test_registry_snapshot_is_paged_without_sending_all_file_lists(tmp_path):
    output,folder,identity=producer(tmp_path)
    for index in range(3):
        name=f'group{index}.pt'; (folder/name).write_bytes(b'fixture')
        group(output,folder,identity,name,'intermediate')
    first=sr.snapshot(output,registry_limit=2)
    assert first['registry_total']==3 and first['registry_next_offset']==2
    assert len(first['registered_artifacts'])==2
    last=sr.snapshot(output,registry_offset=2,registry_limit=2)
    assert len(last['registered_artifacts'])==1 and last['registry_next_offset'] is None
