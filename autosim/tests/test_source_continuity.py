import hashlib
import json

import pytest

from autosim.research.common import atomic_json
from autosim.research.snapshot import Snapshots
from autosim.research.source_continuity import view


def fixture(tmp_path):
    output = tmp_path/'run'; repo = output/'checkout'; repo.mkdir(parents=True)
    root = output/'research/derived'
    source = repo/'train.py'; source.write_text('reader = old\n')
    snapshots = Snapshots(root/'snapshots')
    snapshots.capture(['train.py'],repo=repo,name='before-round-1')
    source.write_text('reader = new\n')
    snapshots.capture(['train.py'],repo=repo,name='round-1')
    atomic_json(root/'code_changes/round-1.json',{'round':1,'status':'applied',
        'snapshot':'before-round-1','patches':[{'file':'train.py','find':'old','replace':'new'}]})
    measurement = root/'measurements/round_1_handoff_correction.json'
    atomic_json(measurement,{'label':measurement.stem,'ok':True})
    return output,repo,source,root,measurement


@pytest.mark.parametrize('text,status,installed',[
    ('reader = new\n','matches_postimage',True),
    ('reader = old\n','matches_preimage_not_postimage',False),
    ('reader = other\n','differs_from_both',False),
    (None,'missing',False)])
def test_current_source_not_transaction_label_controls_observed_status(tmp_path,text,status,installed):
    output,repo,source,root,measurement = fixture(tmp_path)
    if text is None:source.unlink()
    else:source.write_text(text)
    watched=[measurement,root/'code_changes/round-1.json',root/'snapshots/snapshots.json']
    before={p:p.read_bytes() for p in watched}
    rows=view(output,repo,'derived')
    assert len(rows)==1
    assert rows[0]['recorded_transaction_status']=='applied'
    assert rows[0]['files'][0]['status']==status
    assert rows[0]['all_changed_files_match_postimage'] is installed
    assert 'parameter-only' in rows[0]['rule'] and 'not' in rows[0]['authority']
    assert {p:p.read_bytes() for p in watched}==before


@pytest.mark.parametrize('fault',['blob','patch','symlink','confirmation','not_ok','wrong_label','outside_repo'])
def test_unverified_or_out_of_scope_records_never_claim_installation(tmp_path,fault):
    output,repo,source,root,measurement=fixture(tmp_path)
    if fault=='blob':
        key=hashlib.sha256(b'reader = new\n').hexdigest()
        (root/'snapshots/blobs'/key).write_text('corrupt')
    elif fault=='patch':
        p=root/'code_changes/round-1.json';row=json.loads(p.read_text())
        row['patches'][0]['replace']='wrong';atomic_json(p,row)
    elif fault=='symlink':
        external=tmp_path/'outside.py';external.write_text('reader = new\n')
        source.unlink();source.symlink_to(external)
    elif fault=='outside_repo':repo=tmp_path
    else:
        row=json.loads(measurement.read_text())
        row.update({'confirmation':True} if fault=='confirmation' else
            {'ok':False} if fault=='not_ok' else {'label':'round_999'})
        atomic_json(measurement,row)
    assert view(output,repo,'derived')==[]


def test_main_common_decision_facts_include_source_continuity(tmp_path,monkeypatch):
    from tests.test_prepare import preparation
    real=preparation(tmp_path)
    expected=[{'all_changed_files_match_postimage':False,'rule':'fixture: archived is not installed'}]
    monkeypatch.setattr('autosim.research.source_continuity.view',lambda *a:expected)
    assert real._decision_facts()['current_source_continuity']==expected
