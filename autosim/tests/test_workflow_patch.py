import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosim.research import workflow_patch as w
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.ideas import Idea, IdeaLibrary, CLEARED
from autosim.research.redlines import RedLines


def fixture(tmp_path):
    output = tmp_path/'run'
    repo = output/'checkout'
    root = output/'research'/'toy'
    repo.mkdir(parents=True)
    (root/'measurements').mkdir(parents=True)
    (repo/'collector.py').write_text('def collect():\n    return 1\n')
    atomic_json(output/'workspace_snapshot.json', {'destination': str(repo),
        'source_entries': [['file', 'collector.py']]})
    atomic_json(root/'controller_session.json', {'run_id': 'toy', 'status': 'paused',
        'repository': str(repo), 'history': [], 'next_round': 1, 'rounds': 2})
    atomic_json(root/'measurements/baseline.json', {'metric': 0.12})
    atomic_json(root/'comparison_protocol.json', {'settings': {'episodes': 50}})
    library = IdeaLibrary(root/'ideas.json')
    library.add(Idea(label='collector_observer', granularity='code', status=CLEARED,
        change={'file': 'collector.py', 'find': 'return 1', 'replace': 'return 2'},
        mechanism='test-only source change', why='test', touches=['collector.py']))
    return SimpleNamespace(output=output, repo=repo, run_root=root, run_id='toy',
        library=library, sources={'collect': 'def stage_argv_collect():\n    return ["python", "collector.py"]\n',
        'train': 'def stage_argv_train():\n    return ["python", "train.py"]\n',
        'evaluate': 'def stage_argv_evaluate():\n    return ["python", "eval.py"]\n'},
        red_lines=lambda: RedLines(benchmark='toy'))


def review(packet, **overrides):
    wanted = {'workflow_collect', 'candidate_change', 'candidate_effective_source_0', 'candidate_source_0'}
    return {'verdict': 'applicable', 'reason': 'toy source-only review, not scientific evidence',
        'source_evidence': [{'source_id': r['id'], 'quote': r['text'][:40]}
                            for r in packet['sources'] if r['id'] in wanted or
                            r['id'].startswith('candidate_effective_source_')],
        **dict.fromkeys(w.CHECKS, True), **overrides}


def apply(real, reviewer=review, **kwargs):
    return w.apply(real, idea_label='collector_observer', target_stage='collect',
                   reason='source-only fixture', reviewer=reviewer, **kwargs)


def test_source_application_not_a_native_or_scored_trial(tmp_path):
    real = fixture(tmp_path)
    before = {p: digest(real.run_root/p) for p in ['controller_session.json',
              'measurements/baseline.json', 'comparison_protocol.json']}
    row = apply(real)
    assert row['status'] == 'applied' and row['verification_required']
    assert (real.repo/'collector.py').read_text().endswith('return 2\n')
    assert before == {p: digest(real.run_root/p) for p in before}
    assert not (real.output/'native_jobs').exists()
    assert not (real.output/'agent').exists()
    assert w.view(real.output, real.run_id)[0]['id'] == row['id']
    idea = IdeaLibrary(real.run_root/'ideas.json').get('collector_observer')
    assert idea.status != CLEARED and 'no score' in idea.outcome
    assert row['binding']['files']['collector.py'] == hashlib.sha256(b'def collect():\n    return 1\n').hexdigest()


def test_exact_restore_and_idempotence(tmp_path):
    real = fixture(tmp_path)
    row = apply(real)
    result = w.restore(real, patch_id=row['id'], reason='fixture rollback')
    assert result['status'] == 'rolled_back'
    assert (real.repo/'collector.py').read_text().endswith('return 1\n')
    assert w.restore(real, patch_id=row['id'], reason='again')['status'] == 'rolled_back'


def test_restore_never_overwrites_later_source_changes(tmp_path):
    real = fixture(tmp_path)
    row = apply(real)
    (real.repo/'collector.py').write_text('def collect():\n    return 3\n')
    with pytest.raises(ValueError, match='diverged'):
        w.restore(real, patch_id=row['id'], reason='must not overwrite')
    assert (real.repo/'collector.py').read_text().endswith('return 3\n')


@pytest.mark.parametrize('stage', ['train', 'evaluate', 'score', 'unknown'])
def test_no_scoring_or_unoffered_stage(tmp_path, stage):
    real = fixture(tmp_path)
    with pytest.raises(ValueError):
        w.apply(real, idea_label='collector_observer', target_stage=stage, reason='x',
                reviewer=lambda p: pytest.fail('must refuse before model'))


@pytest.mark.parametrize('check', w.CHECKS)
def test_every_preservation_check_is_mandatory(tmp_path, check):
    real = fixture(tmp_path)
    with pytest.raises(ValueError, match='preservation'):
        apply(real, lambda p: review(p, **{check: False}))
    assert (real.repo/'collector.py').read_text().endswith('return 1\n')


@pytest.mark.parametrize('verdict', ['uncertain', 'inapplicable'])
def test_negative_or_unknown_review_never_applies(tmp_path, verdict):
    real = fixture(tmp_path)
    row = apply(real, lambda p: review(p, verdict=verdict))
    assert row['status'] == 'refused'
    assert (real.repo/'collector.py').read_text().endswith('return 1\n')


def test_missing_exact_postimage_citation_rejected(tmp_path):
    real = fixture(tmp_path)
    def bad(p):
        answer = review(p)
        answer['source_evidence'] = [x for x in answer['source_evidence']
                                    if not x['source_id'].startswith('candidate_effective')]
        return answer
    with pytest.raises(ValueError, match='postimage'):
        apply(real, bad)


@pytest.mark.parametrize('change', ['source', 'protocol', 'session', 'idea', 'stages'])
def test_review_races_cannot_authorize_other_source_or_contract(tmp_path, change):
    real = fixture(tmp_path)
    def racing(p):
        answer = review(p)
        if change == 'source': (real.repo/'collector.py').write_text('def collect():\n    return 3\n')
        if change == 'protocol': atomic_json(real.run_root/'comparison_protocol.json', {'episodes': 1})
        if change == 'session':
            f=real.run_root/'controller_session.json'; d=read_json(f); d['rounds']=3; atomic_json(f,d)
        if change == 'idea':
            l=IdeaLibrary(real.run_root/'ideas.json'); l.get('collector_observer').why='changed'; l._write()
        if change == 'stages': real.sources['collect'] += '# changed\n'
        return answer
    with pytest.raises(ValueError): apply(real, racing)
    assert not list((real.output/'workflow_patches').glob('*/rollback.json'))


@pytest.mark.parametrize('state', ['running', 'completed', 'interrupted'])
def test_no_active_or_closed_research_source_change(tmp_path, state):
    real=fixture(tmp_path);f=real.run_root/'controller_session.json';d=read_json(f);d['status']=state;atomic_json(f,d)
    with pytest.raises(ValueError): apply(real)


def test_heldout_exposure_remains_closed(tmp_path):
    real=fixture(tmp_path);p=real.run_root/'confirmation_attempts';p.mkdir();atomic_json(p/'one.json', {})
    with pytest.raises(ValueError, match='held-out'): apply(real)


def test_unowned_or_resource_source_is_refused(tmp_path):
    real=fixture(tmp_path);atomic_json(real.output/'workspace_snapshot.json', {'destination':str(real.repo),'source_entries':[]})
    with pytest.raises(ValueError, match='unowned'): apply(real)


def test_no_source_change_through_symlink(tmp_path):
    real=fixture(tmp_path);target=tmp_path/'outside.py';target.write_text('def collect():\n    return 1\n')
    (real.repo/'collector.py').unlink();(real.repo/'collector.py').symlink_to(target)
    with pytest.raises(ValueError): apply(real)
    assert target.read_text().endswith('return 1\n')


def test_record_and_rollback_corruption_are_rejected(tmp_path):
    real=fixture(tmp_path);row=apply(real);f=real.output/'workflow_patches'/row['id']/'rollback.json';f.write_text('{}')
    with pytest.raises(ValueError, match='archive'):w.restore(real,patch_id=row['id'],reason='test')
    f=f.with_name('state.json');d=read_json(f);d['run_id']='foreign';atomic_json(f,d)
    with pytest.raises(ValueError, match='identity'):w.view(real.output,real.run_id)


def test_active_native_and_specialist_jobs_are_refused(tmp_path, monkeypatch):
    real=fixture(tmp_path)
    monkeypatch.setattr('autosim.research.native_jobs.active_jobs',lambda o:[{'status':'running'}])
    with pytest.raises(ValueError, match='jobs'):apply(real)
    monkeypatch.setattr('autosim.research.native_jobs.active_jobs',lambda o:[])
    monkeypatch.setattr('autosim.research.agent_tasks.records',lambda o:[{'status':'submitted'}])
    with pytest.raises(ValueError, match='jobs'):apply(real)


def test_gpu_lease_refuses_source_mutation(tmp_path):
    real=fixture(tmp_path);atomic_json(real.output/'task_gpu_budget.json',{'leases':{'x':{'status':'active'}}})
    with pytest.raises(ValueError, match='GPU lease'):apply(real)


def test_prepared_crash_restores_exact_originals_without_fake_score(tmp_path):
    real=fixture(tmp_path)
    def crash(row):raise RuntimeError('crash before apply')
    with pytest.raises(RuntimeError):
        w.apply(real,idea_label='collector_observer',target_stage='collect',reason='test',
                reviewer=review,before_apply=crash)
    row=w.view(real.output,real.run_id)[0]
    assert row['status']=='prepared'
    assert w.restore(real,patch_id=row['id'],reason='reconcile')['status']=='rolled_back'
    assert (real.repo/'collector.py').read_text().endswith('return 1\n')


def test_review_crash_record_can_be_safely_cancelled(tmp_path):
    real=fixture(tmp_path)
    def crash(packet):raise KeyboardInterrupt('simulate process interruption')
    with pytest.raises(KeyboardInterrupt):apply(real,crash)
    row=w.view(real.output,real.run_id)[0]
    assert row['status']=='reviewing'
    assert w.restore(real,patch_id=row['id'],reason='cancel review')['status']=='rolled_back'


def test_bridge_main_review_invalidates_prior_commands_and_gates_native_work(tmp_path,monkeypatch):
    from autosim.research.prepare import Preparation
    real=fixture(tmp_path)
    class Client:
        supports_main_agent=True
        role='scheduler'
        def chat_with_metadata(self,system,user,**kwargs):
            assert kwargs['read_only'] and not kwargs['include_research_context']
            return json.dumps(review(json.loads(user))),{}
    prep=Preparation(repo=real.repo,output=real.output,client=Client(),scouting=real.output/'scouting')
    prep.run_id=real.run_id
    prep.stages=dict(real.sources)
    prep.verified={'collect':{'old':'approval'}}
    atomic_json(real.output/'derived_stages.json',{'collect':{'source':real.sources['collect'],
        'parameters':{},'row':prep.verified['collect']}})
    monkeypatch.setattr(prep,'_research_controller',lambda:(real,{}))
    result=prep._step_apply_workflow_candidate(idea_label='collector_observer',target_stage='collect',reason='test')
    assert result['outcome']=='applied' and not prep.verified
    assert read_json(real.output/'derived_stages.json')['collect']['row']=={}
    blocked=prep._do_operation('submit_native_job',stage='collect',window_seconds=60)
    assert blocked['failure_category']=='workflow_stage_reverification'
    assert 'collect' in blocked['stages']
    assert prep.stages==real.sources  # Runtime builders preserved, not approval.


def test_stage_generation_changes_after_rollback(tmp_path):
    real=fixture(tmp_path)
    assert not w.generation(real.output,real.run_id)
    row=apply(real);applied=w.generation(real.output,real.run_id)
    w.restore(real,patch_id=row['id'],reason='test')
    assert w.generation(real.output,real.run_id)!=applied


def test_empty_cold_start_inventory_does_not_create_files(tmp_path):
    missing=tmp_path/'not-yet-created'
    assert w.view(missing,'toy')==[] and w.generation(missing,'toy')==''
    assert not missing.exists()


def test_refused_review_does_not_change_command_generation(tmp_path):
    real=fixture(tmp_path)
    apply(real,lambda p:review(p,verdict='uncertain'))
    assert w.generation(real.output,real.run_id)==''


def test_source_namespace_paging_and_tamper_guard(tmp_path):
    real=fixture(tmp_path);row=apply(real);identity='workflow-patch:'+row['id']
    first=w.read_evidence(real.output,identity,limit=120)
    second=w.read_evidence(real.output,identity,offset=first['next_offset'],limit=120)
    assert first['display_sha256']==second['display_sha256']
    assert 'rollback.json' not in w.read_evidence(real.output,identity,limit=12000)['text']
    f=real.output/row['request_ref'];f.write_text('{}')
    with pytest.raises(ValueError,match='request identity'):w.read_evidence(real.output,identity)


@pytest.mark.parametrize('identity,offset,limit',[
    ('../../other',0,1),('workflow-patch:'+'a'*32,True,1),
    ('workflow-patch:'+'a'*32,0,12001),('workflow-patch:'+'a'*32,-1,10)])
def test_source_namespace_denies_unbounded_or_non_id_reads(tmp_path,identity,offset,limit):
    real=fixture(tmp_path)
    with pytest.raises(ValueError):w.read_evidence(real.output,identity,offset=offset,limit=limit)


def test_native_training_blocked_while_workflow_source_is_active(tmp_path,monkeypatch):
    from autosim.research.prepare import Preparation
    real=fixture(tmp_path);apply(real)
    class Client:supports_main_agent=True
    prep=Preparation(repo=real.repo,output=real.output,client=Client(),scouting=real.output/'scouting')
    prep.run_id='toy';prep.stages=dict(real.sources)
    for op,args in [('run_the_loop',{}),('confirm_best',{}),('run_screening_trial',{}),
                    ('evaluate_baseline_reference',{}),('submit_native_job',{'stage':'train'}),
                    ('derive_a_command',{'stage':'evaluate'})]:
        result=prep._do_operation(op,**args)
        assert result['failure_category']=='workflow_restore_before_policy_trial'


def test_native_schema_advertises_workflow_namespace(tmp_path):
    import re
    from autosim.research.agent_runtime import _evidence_tool_schema
    pattern=_evidence_tool_schema()['inputSchema']['properties']['evidence_id']['pattern']
    assert re.fullmatch(pattern,'workflow-patch:'+'a'*32)
    assert not re.fullmatch(pattern,'workflow-patch:../../data')


def test_source_namespace_is_not_native_evidence(tmp_path):
    from autosim.research.evidence_store import read_attempt_evidence
    real=fixture(tmp_path);row=apply(real)
    with pytest.raises(ValueError):read_attempt_evidence(real.output,'workflow-patch:'+row['id'])


def test_refused_reviewer_cannot_rewrite_the_sealed_request(tmp_path):
    real=fixture(tmp_path)
    def tamper(p):
        request=next((real.output/'workflow_patches').glob('*/request.json'))
        request.write_text('{}')
        return review(p)
    with pytest.raises(ValueError,match='changed'):apply(real,tamper)
    assert (real.repo/'collector.py').read_text().endswith('return 1\n')


def test_partial_rollback_stays_pending_and_can_resume(tmp_path,monkeypatch):
    real=fixture(tmp_path)
    (real.repo/'helper.py').write_text('def helper():\n    return 1\n')
    atomic_json(real.output/'workspace_snapshot.json',{'destination':str(real.repo),
        'source_entries':[['file','collector.py'],['file','helper.py']]})
    idea=real.library.get('collector_observer');old=dict(idea.change)
    idea.change={'patches':[old,{'file':'helper.py','find':'return 1','replace':'return 2'}]}
    idea.touches=['collector.py','helper.py'];real.library._write()
    row=apply(real);original=w.revert;calls=[]
    def fail_once(p,b,**kw):
        calls.append(p.file)
        if len(calls)==2:raise OSError('simulated rollback interruption')
        return original(p,b,**kw)
    monkeypatch.setattr(w,'revert',fail_once)
    with pytest.raises(OSError):w.restore(real,patch_id=row['id'],reason='test')
    assert w.view(real.output,real.run_id)[0]['status']=='prepared'
    monkeypatch.setattr(w,'revert',original)
    assert w.restore(real,patch_id=row['id'],reason='resume')['status']=='rolled_back'
    assert all((real.repo/name).read_text().endswith('return 1\n') for name in ['collector.py','helper.py'])
