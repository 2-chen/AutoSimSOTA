import hashlib
import json

import pytest

from autosim.research.candidate_source_guard import historical_mismatches, preserve
from autosim.research.codepatch import Patch
from autosim.research.common import atomic_json, read_json


@pytest.mark.parametrize('state',['candidate','preimage','unrelated','missing','symlink'])
def test_only_exact_audited_preimage_can_be_reasserted(tmp_path,state):
    patch=Patch('glue.py','VALUE = 1','VALUE = 2')
    before='VALUE = 1\n'
    path=tmp_path/'glue.py'
    if state!='missing':
        path.write_text('VALUE = 2\n' if state=='candidate' else before if state=='preimage' else 'VALUE = 3\n')
    if state=='symlink':
        target=tmp_path/'other.py';path.rename(target);path.symlink_to(target)
    if state in {'unrelated','missing','symlink'}:
        with pytest.raises(ValueError):preserve(tmp_path,{'glue.py':(patch,before)})
        return
    result=preserve(tmp_path,{'glue.py':(patch,before)})
    assert result['status']=='verified' and path.read_text()=='VALUE = 2\n'
    assert result['files'][0]['reapplied_after_staging']==(state=='preimage')


@pytest.mark.parametrize('rewrite',['preimage','unrelated'])
def test_native_training_cannot_silently_lose_its_code_intervention(tmp_path,rewrite):
    from tests.test_derived_research import research
    from autosim.research.declarative_backend import DeclarativeBackend
    from autosim.research.ideas import Idea
    real=research(tmp_path)
    file=tmp_path/'glue.py';file.write_text('VALUE = 1\n')
    idea=Idea(label='change learning',granularity='code',touches=['glue.py'],
              change={'file':'glue.py','find':'VALUE = 1','replace':'VALUE = 2'})
    assert real.code_change(idea,transaction_round=1)['applied']
    real.backend.sources['train']='def stage_argv_train(i):\n    return [i["python"], "-c", "import glue; print(glue.VALUE)"]'
    real.backend=DeclarativeBackend(repo=tmp_path,answer=real.backend.answer,
                                    sources=real.backend.sources,parameters={})
    value=1 if rewrite=='preimage' else 3
    real.backend.stages['train']['staging']=[f'printf "VALUE = {value}\\n" > glue.py']
    result=real.run_stage('train',settings={})
    receipt=read_json(real.run_root/'attempts'/result['attempt_id']/'receipt.json')
    if rewrite=='unrelated':
        assert result['ran'] is False
        assert receipt['termination_reason']=='candidate_source_changed_by_staging'
    else:
        assert result['returncode']==0 and result['said'].strip()=='2'
        assert receipt['candidate_source_integrity']['files'][0]['reapplied_after_staging'] is True


def history(root,*,mismatch=True):
    expected=hashlib.sha256(b'candidate').hexdigest()
    actual=hashlib.sha256(b'baseline').hexdigest() if mismatch else expected
    atomic_json(root/'snapshots/snapshots.json',{'snapshots':[{'name':'round-1','files':{'glue.py':expected}}]})
    atomic_json(root/'code_changes/round-1.json',{'idea_label':'learning','patches':[{'file':'glue.py'}]})
    atomic_json(root/'measurements/round_1.json',{'ok':True,'metric_value':0.98,
        'source_state':{'overlay_files':{'glue.py':actual}}})


def test_historical_candidate_audit_is_additive_not_a_score_rewrite(tmp_path):
    history(tmp_path)
    measurement=tmp_path/'measurements/round_1.json'
    before=measurement.read_bytes()
    assert historical_mismatches(tmp_path)[0]['label']=='round_1'
    assert measurement.read_bytes()==before
    history(tmp_path,mismatch=False)
    assert historical_mismatches(tmp_path)==[]


def test_historical_candidate_warning_is_visible_to_main(tmp_path):
    from tests.test_prepare import preparation
    real=preparation(tmp_path)
    root=real.output/'research'/real.run_id
    history(root)
    atomic_json(root/'controller_session.json',{'schema_version':1,'run_id':real.run_id,
        'repository':str(real.repo),'status':'paused','history':[],'rounds':1,'next_round':2})
    warnings=real.state()['research_progress']['candidate_source_integrity_warnings']
    assert warnings[0]['status']=='candidate_source_mismatch'


@pytest.mark.parametrize('operation',['run_the_loop','confirm_best'])
def test_lost_code_intervention_cannot_consume_heldout_before_repair(tmp_path,monkeypatch,operation):
    from tests.test_prepare import preparation
    real=preparation(tmp_path)
    real.base_settings={'_confirm':True}
    root=real.output/'research'/real.run_id
    history(root)
    rows=[{'round':1,'status':'measured','label':'round_1','metric_value':0.98}]
    atomic_json(root/'controller_session.json',{'schema_version':1,'run_id':real.run_id,
        'repository':str(real.repo),'status':'finalizing','history':rows,'rounds':1,'next_round':2})
    atomic_json(root/'research_report.json',{'run_id':real.run_id,'repo':str(real.repo),
        'run_status':'paused','rounds':rows})
    monkeypatch.setattr(real,'_research_controller',lambda:pytest.fail('no held-out or new trainer'))
    result=getattr(real,'_step_'+operation)()
    assert result['failure_category']=='candidate_source_invalid'
