from copy import deepcopy

import pytest
from autosim.research.ideas import Idea
from tests.test_derived_research import research

from autosim.research import idea_reproposal as m


def fixture(root):
    real=research(root);(root/'train.py').write_text('old\n')
    old=Idea(label='batch observer',granularity='code',mechanism='Observe actual completed updates',
        change={'file':'train.py','find':'old','replace':'new'},risk='low',crosses='none',
        why='Original support experiment',evidence=['observed source'],touches=['train.py'])
    real.library.add(old);real.library.note(old.label,worked=False,outcome='zero score, working observer')
    offer=m.reproposal_offers(real.library)[0]
    request={'label':'next seed support','reuse_idea_label':old.label,
        'expected_idea_sha256':offer['expected_idea_sha256'],'reason':'Explicit replication; trial settings selected separately'}
    return real,old,request


def test_exact_reproposal_reenters_normal_redline_audit_without_execution(tmp_path):
    real,old,request=fixture(tmp_path);before=deepcopy(old.as_dict())
    result=real.propose_research_idea(request)
    assert result['status']=='cleared'
    copied=real.library.get(request['label'])
    assert copied.parent==old.label and copied.change==old.change and copied.times_tried==0
    assert old.as_dict()==before
    assert (tmp_path/'train.py').read_text()=='old\n'
    assert not list((real.run_root/'measurements').glob('*.json'))


def test_changed_current_preimage_is_refused_before_new_library_entry(tmp_path):
    real,old,request=fixture(tmp_path);(tmp_path/'train.py').write_text('different current source\n')
    result=real.propose_research_idea(request)
    assert result['status']=='invalid'
    assert real.library.get(request['label']) is None


def test_reproposal_cannot_skip_current_redline_decision(tmp_path):
    real,old,request=fixture(tmp_path);calls=[]
    def reject(lines):
        calls.append(lines);real.library.get(request['label']).status='rejected'
    real._audit_the_batch=reject
    result=real.propose_research_idea(request)
    assert result['status']=='rejected' and len(calls)==1
    assert old.status=='tried'


def test_stale_origin_refused_without_replacing_prior_result(tmp_path):
    real,old,request=fixture(tmp_path);old.why='independently changed origin'
    result=real.propose_research_idea(request)
    assert result['status']=='invalid' and 'identity' in result['why']
    assert real.library.get(request['label']) is None


def test_main_frontdoor_accepts_exact_reproposal_without_executing(tmp_path,monkeypatch):
    from tests.test_operation_contracts import preparation
    real,old,request=fixture(tmp_path)
    controller=preparation(tmp_path,choices=[{'do':'propose_research_idea',
        'why':'repeat exact support payload','arguments':{'idea':request}}])
    monkeypatch.setattr(controller,'state',lambda:{'available':['propose_research_idea'],
                                                  'state_revision':0})
    monkeypatch.setattr(controller,'_research_controller',
        lambda:pytest.fail('frontdoor validation must not create/execute research'))
    choice=controller.choose()
    assert choice['arguments']=={'idea':request}
    assert real.library.get(request['label']) is None
    assert (tmp_path/'train.py').read_text()=='old\n'


def test_main_frontdoor_repairs_stale_reproposal_hash_before_action(tmp_path,monkeypatch):
    from tests.test_operation_contracts import preparation
    real,old,request=fixture(tmp_path);bad={**request,'expected_idea_sha256':'0'*64}
    controller=preparation(tmp_path,choices=[
        {'do':'propose_research_idea','why':'stale snapshot','arguments':{'idea':bad}},
        {'do':'propose_research_idea','why':'fresh snapshot','arguments':{'idea':request}}])
    monkeypatch.setattr(controller,'state',lambda:{'available':['propose_research_idea'],
                                                  'state_revision':0})
    choice=controller.choose()
    assert choice['arguments']=={'idea':request}
    assert len(controller.client.seen)==2
    assert real.library.get(request['label']) is None
