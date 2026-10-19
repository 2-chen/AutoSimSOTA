"""One reference selection repairs transport, never independent scientific judgment."""
import json

import pytest

from autosim.research import recorder
from autosim.research.common import read_json
from tests.test_recorder import snapshot, Writer


class Reviewed(Writer):
    def __init__(self, mode='valid'):
        super().__init__(); self.review_mode=mode; self.reviews=0; self.original_reason='当前证据不支持这个说法，不能据旧记录断言。'

    def chat_with_metadata(self, system, user, **kwargs):
        packet=json.loads(user)
        if packet.get('phase')=='repair_issue_references':
            self.calls.append((packet,kwargs))
            assert packet['original_review']['issues'][0]['reason']==self.original_reason
            answer={'issue_citations':[{'issue_index':0,'section':'finding','evidence_ids':['run']}]}
            if self.review_mode=='unknown_ref':answer['issue_citations'][0]['evidence_ids']=['invented']
            if self.review_mode=='changed_verdict':answer['verdict']='supported'
            if self.review_mode=='dropped':answer['issue_citations']=[]
            if self.review_mode=='wrong_index':answer['issue_citations'][0]['issue_index']=True
            return json.dumps(answer),{'role':'supervisor','reference_selection':True}
        if packet.get('phase')=='review_report':
            self.calls.append((packet,kwargs));self.reviews+=1
            if self.reviews==2 and self.review_mode!='repeated_typo':
                return json.dumps({'verdict':'supported','issues':[]}),{'role':'supervisor'}
            return json.dumps({'verdict':'unsupported','issues':[{
                'section':'found','reason':self.original_reason,'evidence_ids':['run']}]}),{'role':'supervisor'}
        return super().chat_with_metadata(system,user,**kwargs)


def test_section_typo_has_one_bounded_selection_and_keeps_original_denial(tmp_path):
    held=snapshot(tmp_path);writer=Reviewed()
    result=recorder.narrate(tmp_path,held,client=writer,timeout=60)
    assert result['status']=='written'
    assert result['fact_review_format_repair_attempted']
    assert [r['answer']['verdict']for r in result['fact_reviews']]==['unsupported','supported']
    first=result['fact_reviews'][0]['answer']['issues'][0]
    assert first=={'section':'finding','reason':writer.original_reason,'evidence_ids':['run']}
    repair=result['fact_review_reference_repair']
    assert repair['original_answer']['issues'][0]['section']=='found'
    assert repair['original_answer']['verdict']=='unsupported'
    picks=[(p,o)for p,o in writer.calls if p.get('phase')=='repair_issue_references']
    assert len(picks)==1 and picks[0][1]['max_tokens']==700
    assert picks[0][1]['read_only'] and not picks[0][1]['resume_session']
    assert 0<picks[0][1]['timeout']<=30


@pytest.mark.parametrize('mode',['unknown_ref','changed_verdict','dropped','wrong_index','repeated_typo'])
def test_bad_or_repeated_reference_correction_never_publishes_or_retries_forever(tmp_path,mode):
    held=snapshot(tmp_path);writer=Reviewed(mode)
    assert recorder.narrate(tmp_path,held,client=writer,timeout=60)=={}
    assert not (tmp_path/'report/narrative.json').exists()
    trigger=read_json(tmp_path/'report/recorder_trigger.json')
    assert trigger['status']=='failed' and trigger['fact_review_format_repair_attempted']
    assert len([p for p,_ in writer.calls if p.get('phase')=='repair_issue_references'])==1
    count=len(writer.calls)
    recorder.narrate(tmp_path,held,client=writer,timeout=60)
    assert len(writer.calls)==count


def test_readiness_and_score_cannot_be_changed_in_the_reference_map(tmp_path):
    held=snapshot(tmp_path);writer=Reviewed('changed_verdict')
    recorder.narrate(tmp_path,held,client=writer)
    trigger=read_json(tmp_path/'report/recorder_trigger.json')
    original=trigger['fact_review_reference_repair']['original_answer']
    assert original['verdict']=='unsupported' and original['issues'][0]['reason']==writer.original_reason
    assert not recorder.pending_requests(tmp_path)
