from copy import deepcopy

import pytest
from autosim.research.ideas import Idea,IdeaLibrary

from autosim.research import idea_reproposal as m


def fixture(tmp_path):
    library=IdeaLibrary(tmp_path/'ideas.json')
    old=Idea(label='observed-support-patch',granularity='code',mechanism='Observe completed updates',
        change={'file':'train.py','find':'old','replace':'new'},risk='low',crosses='none',
        why='Original experiment',evidence=['source:one'],touches=['train.py'])
    library.add(old);library.note(old.label,worked=False,outcome='real score zero, witness fixed')
    offer=m.reproposal_offers(library)[0]
    request={'label':'explicit-replica','reuse_idea_label':old.label,
             'expected_idea_sha256':offer['expected_idea_sha256'],'reason':'Replicate with separately selected trial settings'}
    return library,old,request


def test_exact_payload_is_copied_without_reviving_or_mutating_original(tmp_path):
    library,old,request=fixture(tmp_path);before=deepcopy(old.as_dict());bytes_before=library.path.read_bytes()
    copied,parent=m.expand_reproposal(request,library)
    assert parent==old.label and copied['change']==old.change
    assert copied['label']=='explicit-replica' and copied['why']==request['reason']
    assert old.as_dict()==before and library.path.read_bytes()==bytes_before
    assert library.get('explicit-replica') is None
    copied['change']['replace']='mutated copy'
    assert old.change['replace']=='new'
    assert 'training_settings' not in copied and 'score' not in copied
    assert copied['evidence'][-1].startswith('exact_prior_proposal:')


@pytest.mark.parametrize('change',[{'expected_idea_sha256':'0'*64},{'reuse_idea_label':'other-run'},
    {'label':'observed-support-patch'},{'reason':''},{'label':'x'*161},
    {'reason':'x'*2001},{'expected_idea_sha256':None},{'extra':'not allowed'}])
def test_incorrect_identity_or_unbounded_requests_refused(tmp_path,change):
    library,old,request=fixture(tmp_path);request.update(change)
    with pytest.raises(ValueError):m.expand_reproposal(request,library)


def test_changed_original_payload_does_not_match_old_offer(tmp_path):
    library,old,request=fixture(tmp_path);old.change['replace']='new source'
    with pytest.raises(ValueError,match='identity changed'):m.expand_reproposal(request,library)


def test_unexecuted_idea_is_not_a_reproduction_offer(tmp_path):
    library,old,request=fixture(tmp_path);old.times_tried=0
    assert m.reproposal_offers(library)==[]
    with pytest.raises(ValueError,match='previously executed'):m.expand_reproposal(request,library)


def test_offers_are_bounded_recent_ids_not_a_hidden_best_selector(tmp_path):
    library,old,request=fixture(tmp_path)
    for i in range(12):
        idea=deepcopy(old);idea.label='replica-'+str(i);library.ideas.append(idea)
    offers=m.reproposal_offers(library)
    assert len(offers)==8 and offers[-1]['label']=='replica-11'
    assert all('not installed source' in row['authority'] for row in offers)
    assert all('change' not in row and 'score' not in row for row in offers)


@pytest.mark.parametrize('field,value',[('evidence',None),('touches',{}),('change','not a patch')])
def test_malformed_history_never_crashes_the_fact_projection(tmp_path,field,value):
    library,old,request=fixture(tmp_path);setattr(old,field,value)
    assert m.reproposal_offers(library)[0]['status']=='unavailable'
    with pytest.raises(ValueError):m.expand_reproposal(request,library)


def test_reproposal_helper_is_readable_but_not_live_self_editable():
    from autosim.research.harness_repair import READABLE,ALLOWED
    assert 'research/idea_reproposal.py' in READABLE
    assert 'research/idea_reproposal.py' not in ALLOWED
