"""Prospective setup repair is not retroactive alteration of experiment results."""
import json
import pytest

from autosim.research import protocol_revision as pr
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.confirmation_reservation import search_started


def fixture(tmp_path):
    source=tmp_path/'original'; source.mkdir()
    repo=tmp_path/'project'; repo.mkdir()
    before='def success(state):\n    return state.done\n'
    (source/'eval.py').write_text(before)
    (repo/'eval.py').write_text(before)
    output=tmp_path/'out'; root=output/'research'/'derived'
    atomic_json(root/'protocol_frozen.json',{'hashes':{str(repo/'eval.py'):digest(repo/'eval.py')}})
    atomic_json(root/'reference_protocols'/'original.json',{'conditions':'preserve'})
    atomic_json(output/'workspace_snapshot.json',{'source':str(source),'destination':str(repo)})
    (repo/'eval.py').write_text(before+'# native identity witness, not metric changes\n')
    return source,repo,output,root


def planned(repo,output):
    return pr.plan(output,repo,'derived',{str(repo/'eval.py'):digest(repo/'eval.py')})


def approved(plan):
    return {'plan_id':plan['plan_id'],**dict.fromkeys(pr.CHECKS,True)}


def blocked(root):
    attempt='a'*32
    receipt={'attempt_id':attempt,'node_id':'evaluate','status':'blocked',
             'termination_reason':'protocol_changed','returncode':None}
    atomic_json(root/'attempts'/attempt/'receipt.json',receipt)
    atomic_json(root/'measurements'/'reference_old.json',
                {'ok':False,'evaluate':{'attempt_id':attempt}})
    return receipt


def test_reviewed_revision_preserves_original_protocol_and_failures(tmp_path):
    source,repo,out,root=fixture(tmp_path); blocked(root)
    originals={p:p.read_bytes() for p in root.rglob('*.json')}
    assert not search_started(out,'derived')
    plan=planned(repo,out)
    assert not (root/'active_protocol_revision.json').exists()
    revision=pr.activate(out,repo,'derived',plan,approved(plan),'Instrumented evaluator was frozen too early')
    frozen,active=pr.effective(root)
    assert active['id']==revision['id'] and not active['formal_paper_eligible']
    assert read_json(frozen)['hashes']==plan['hashes']
    assert all(p.read_bytes()==data for p,data in originals.items())
    assert (source/'eval.py').read_text().endswith('return state.done\n')
    assert pr.reference_path(root,'original')!=root/'reference_protocols/original.json'
    assert len(pr.reference_path(root,'original').stem)<=64


@pytest.mark.parametrize('kind',['score_zero','unknown','ran','running','session','comparison','confirmation','pid'])
def test_no_revision_after_started_or_unknown_experiment(tmp_path,kind):
    _,repo,out,root=fixture(tmp_path)
    receipt=blocked(root)
    if kind=='score_zero': atomic_json(root/'measurements/baseline.json',{'ok':True,'value':0})
    elif kind=='unknown': atomic_json(root/'measurements/unknown.json',{'ok':False})
    elif kind in ('ran','running','pid'):
        receipt.update({'ran':True} if kind=='ran' else {'status':'running'} if kind=='running' else {'pid':12})
        atomic_json(root/'attempts'/('a'*32)/'receipt.json',receipt)
    else:
        name={'session':'controller_session.json','comparison':'comparison_protocol.json',
              'confirmation':'confirmation_attempts/c.json'}[kind]
        atomic_json(root/name,{})
    with pytest.raises(ValueError): planned(repo,out)
    assert not (root/'active_protocol_revision.json').exists()


@pytest.mark.parametrize('kind',['rejected','wrong_plan','race','no_old_bytes','added_file','symlink','jobs'])
def test_revision_requires_exact_review_authentic_sources_and_no_jobs(tmp_path,kind,monkeypatch):
    source,repo,out,root=fixture(tmp_path); plan=planned(repo,out); review=approved(plan)
    if kind=='rejected':review[pr.CHECKS[0]]=False
    elif kind=='wrong_plan':review['plan_id']='f'*64
    elif kind=='race':(repo/'eval.py').write_text('new source\n')
    elif kind=='no_old_bytes':(source/'eval.py').write_text('lost original\n')
    elif kind=='added_file':plan['hashes'][str(repo/'another.py')]='f'*64
    elif kind=='symlink':
        path=root/'protocol_frozen.json';path.unlink();path.symlink_to(source/'eval.py')
    else:
        monkeypatch.setattr('autosim.research.native_jobs.active_jobs',lambda out:[{'status':'running'}])
    with pytest.raises((ValueError,OSError)):
        pr.activate(out,repo,'derived',plan,review,'test')
    assert not (root/'active_protocol_revision.json').exists()


@pytest.mark.parametrize('kind',['parent','child','row','pointer','directory_symlink'])
def test_effective_protocol_validates_immutable_chain(tmp_path,kind):
    _,repo,out,root=fixture(tmp_path);plan=planned(repo,out)
    row=pr.activate(out,repo,'derived',plan,approved(plan),'test')
    directory=root/'protocol_revisions'/row['id']
    if kind=='parent': atomic_json(root/'protocol_frozen.json',{'hashes':{}})
    elif kind=='child':atomic_json(directory/'protocol_frozen.json',{'hashes':{}})
    elif kind=='row':atomic_json(directory/'revision.json',{'id':row['id']})
    elif kind=='pointer':atomic_json(root/'active_protocol_revision.json',{'id':'../escape'})
    else:
        (directory/'revision.json').unlink();(directory/'revision.json').symlink_to(root/'protocol_frozen.json')
    with pytest.raises(ValueError):pr.effective(root)


def test_confirmation_symlink_cannot_hide_boundary(tmp_path):
    _,repo,out,root=fixture(tmp_path)
    elsewhere=tmp_path/'hidden';elsewhere.mkdir()
    (root/'confirmation_attempts').symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError,match='unsafe'):planned(repo,out)
    assert search_started(out,'derived')


def test_unresolved_detached_evaluation_also_closes_reservation(tmp_path):
    _,_,out,root=fixture(tmp_path)
    atomic_json(root/'attempts'/('f'*32)/'receipt.json',
                {'node_id':'evaluate','status':'running','returncode':None})
    assert search_started(out,'derived')


def test_main_operation_uses_independent_review_not_operator_approval(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from autosim.research.prepare import Preparation
    from autosim.research.redlines import RedLines
    _,repo,out,root=fixture(tmp_path)
    class Reviewer:
        def chat_with_metadata(self,system,user,**kwargs):
            assert kwargs['read_only'] and not kwargs['include_research_context']
            packet=json.loads(user)
            assert packet['changes'][0]['diff'] and packet['plan_id']
            return json.dumps(approved(packet)),{}
    real=Preparation(repo=repo,output=out,client=Reviewer(),scouting=tmp_path/'scouting')
    monkeypatch.setattr(real,'_research_controller',lambda:
        (SimpleNamespace(red_lines=lambda:RedLines(protected=('eval.py',)),run_root=root),{}))
    assert real._presearch_protocol_revision_view()['status']=='review_available'
    result=real.do('revise_presearch_protocol',reason='source-backed pre-search witness setup')
    assert result['outcome']=='revised' and not result['formal_paper_eligible'],result
    assert real._presearch_protocol_revision_view()['status']=='not_available'


def test_new_controller_runs_cpu_operation_and_binds_revision_receipt(tmp_path):
    import sys
    from pathlib import Path
    from tests.test_derived_research import space, sources_for
    from autosim.research.derived_research import DerivedResearch
    from autosim.research.declarative_backend import DeclarativeBackend
    from autosim.research.compute_decision import ComputeDecision
    from autosim.research.receipt_verifier import verify_measurement
    from autosim.research.workspace_snapshot import create
    source=tmp_path/'original';source.mkdir()
    (source/'evaluate.py').write_text('original = True\n')
    out=tmp_path/'out';repo=out/'checkout';root=out/'research/derived'
    create(source,repo,tracked_only=False)
    def research(repo):
        sources=sources_for(('train','evaluate'))
        answer={'stages':{stage:dict(available=True,entrypoint='x.py',invocation='python x.py',
                                     artifact='models/*.pth') for stage in sources}}
        return DerivedResearch(repo=repo,output=out,
            backend=DeclarativeBackend(repo=repo,answer=answer,sources=sources,parameters={}),
            interpreter=Path(sys.executable),space=space(),client=None,stages=sources,run_id='derived',
            compute=ComputeDecision(device='cpu',device_index=0,environment={},why='CPU contract fixture'))
    # The stock fixture's protected file set is discovered, not injected below.
    real=research(repo)
    from autosim.research.redlines import protected_hashes
    atomic_json(root/'protocol_frozen.json',{'hashes':protected_hashes(repo,real.red_lines(),exclude_roots=(real.run_root,))})
    original=(root/'protocol_frozen.json').read_bytes()
    (repo/'evaluate.py').write_text('original = True\n# identity instrumentation\n')
    plan=pr.plan(out,repo,'derived',protected_hashes(repo,real.red_lines(),exclude_roots=(real.run_root,)))
    row=pr.activate(out,repo,'derived',plan,approved(plan),'toy pre-search setup repair')
    resumed=research(repo)
    result=resumed.measure(settings={},label='baseline')
    assert result['ok'],result
    assert result['protocol_revision_id']==row['id']
    receipt=read_json(root/'attempts'/result['evaluate']['attempt_id']/'receipt.json')
    assert receipt['protocol_revision_id']==row['id']
    assert (root/'protocol_frozen.json').read_bytes()==original
    report=verify_measurement(root,'baseline')
    assert report['status']=='consistent' and report['checks']['protocol_revision'],report
    with pytest.raises(ValueError):pr.plan(out,repo,'derived',plan['hashes'])
