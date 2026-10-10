from pathlib import Path

import pytest

from autosim.research import measurement_adoption as ma
from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.snapshot import Snapshots
from autosim.research.workspace_snapshot import _state_identity
from tests.test_derived_research import ambiguous_policy_research


def fixture(tmp_path):
    real = ambiguous_policy_research(tmp_path)
    baseline = real.measure(settings={}, label='baseline')
    assert baseline['ok']
    old = real.snapshots.capture(['ppo.py'],repo=tmp_path,name='round-1')
    real.snapshots.advance(old.name,score=0.5,scale='success_rate')
    label = 'round_2_handoff_correction'
    row = real.measure(settings={},label=label)
    assert row['ok'] and row['metric_value']==0.75
    source = real.snapshots.capture(['ppo.py'],repo=tmp_path,name='source-state-'+label)
    entries=[['file','ppo.py',(tmp_path/'ppo.py').stat().st_size,digest(tmp_path/'ppo.py'),420]]
    workspace={'schema_version':2,'destination':str(tmp_path),
               'source_entries':entries,'source_tree_fingerprint':object_digest(entries),
               'copy_mode':'tracked_worktree','untracked_assets_omitted':True}
    atomic_json(real.output/'workspace_snapshot.json',workspace)
    row['source_state']={'source_snapshot':source.name,'overlay_files':source.files,
        'overlay_modes':{},'overlay_directories':{},'copy_mode':'tracked_worktree',
        'untracked_assets_omitted':True,'source_tree_fingerprint':workspace['source_tree_fingerprint'],
        'identity_sha256':_state_identity(workspace,source.name,source.files,{},{})}
    row['handoff_correction']={'training_reused':True}
    atomic_json(real.run_root/'measurements'/f'{label}.json',row)
    history=[{'label':'baseline'},{'label':'round_1'}]
    atomic_json(real.run_root/'controller_session.json',{'status':'completed','action':'','history':history})
    atomic_json(real.run_root/'research_report.json',{'run_status':'completed',
        'rounds':history,'best':real.snapshots.best().as_dict(),'best_export':{'old':True}})
    return real, ma.offered(real.run_root)[0]


def request(row):
    return {k:row[k] for k in ('label','expected_best','measurement_sha256')}


def test_adoption_preserves_history_source_baseline_and_measurements_without_training(tmp_path,monkeypatch):
    real,row=fixture(tmp_path)
    pinned=[real.run_root/'measurements/baseline.json',real.run_root/'measurements'/f"{row['label']}.json",
            real.run_root/'controller_session.json',tmp_path/'ppo.py']
    before={p:p.read_bytes() for p in pinned}
    oldrows=real.snapshots.history()
    monkeypatch.setattr(real,'measure',lambda **kw:pytest.fail('adoption reran experiment'))
    result=ma.adopt(real,**request(row))
    assert result['ok'] and real.snapshots.best().name=='round-2_handoff_correction'
    assert real.snapshots.best().score==0.75
    assert real.snapshots.history()[:-1]==oldrows
    assert all(p.read_bytes()==v for p,v in before.items())
    assert read_json(real.run_root/'research_report.json')['best_export']=={}
    assert ma.offered(real.run_root)==[]
    assert ma.adopt(real,**request(row))['already_applied']


@pytest.mark.parametrize('fault',['measurement','policy','log','source','protocol','incumbent','active','heldout','history','symlink'])
def test_adoption_refuses_unverified_or_changed_inputs_without_best_mutation(tmp_path,monkeypatch,fault):
    real,row=fixture(tmp_path)
    root=real.run_root
    m=read_json(root/'measurements'/f"{row['label']}.json")
    if fault=='measurement':
        m['metric_value']=1;atomic_json(root/'measurements'/f"{row['label']}.json",m)
    if fault=='policy': Path(m['policy_artifact']['path']).write_text('changed')
    if fault=='log':
        receipt=read_json(root/'attempts'/m['evaluate']['attempt_id']/'receipt.json')
        Path(receipt['log']).write_text('success_rate: 0.75\nforged')
    if fault=='source': (real.snapshots.blobs/next(iter(m['source_state']['overlay_files'].values()))).write_text('changed')
    if fault=='protocol': atomic_json(root/'comparison_protocol.json',{})
    if fault=='incumbent': real.snapshots._best='other';real.snapshots._write()
    if fault=='active': monkeypatch.setattr('autosim.research.native_jobs.active_jobs',lambda output:['unfinished'])
    if fault=='heldout': atomic_json(root/'confirmation_attempts/seen.json',{})
    if fault=='history': atomic_json(root/'research_report.json',{'rounds':[]})
    if fault=='symlink':
        p=root/'controller_session.json';data=p.read_bytes();p.unlink()
        other=tmp_path/'session.json';other.write_bytes(data);p.symlink_to(other)
    before=(root/'snapshots/snapshots.json').read_bytes()
    with pytest.raises((ValueError,OSError)):
        ma.adopt(real,**request(row))
    assert (root/'snapshots/snapshots.json').read_bytes()==before


def test_prepared_commit_recovers_exact_partial_projection_without_repeating_work(tmp_path,monkeypatch):
    real,row=fixture(tmp_path)
    original=ma.atomic_json
    calls=[]
    def interrupt(path,value):
        if Path(path).name=='research_report.json':
            calls.append(1);raise OSError('simulated interruption')
        return original(path,value)
    monkeypatch.setattr(ma,'atomic_json',interrupt)
    with pytest.raises(OSError): ma.adopt(real,**request(row))
    receipt_path=next((real.run_root/'measurement_adoptions').glob('*.json'))
    assert read_json(receipt_path)['status']=='prepared'
    assert Snapshots(real.run_root/'snapshots').best().score==0.75
    assert ma.offered(real.run_root)[0]['kind']=='reconcile_prepared_adoption'
    monkeypatch.setattr(ma,'atomic_json',original)
    result=ma.adopt(real,**request(row))
    assert result['ok'] and read_json(receipt_path)['status']=='applied'
    assert read_json(real.run_root/'research_report.json')['best']['score']==0.75


def test_interrupted_commit_does_not_clobber_unrelated_projection_changes(tmp_path,monkeypatch):
    real,row=fixture(tmp_path)
    original=ma.atomic_json
    def interrupt(path,value):
        if Path(path).name=='research_report.json': raise OSError('interrupt')
        return original(path,value)
    monkeypatch.setattr(ma,'atomic_json',interrupt)
    with pytest.raises(OSError): ma.adopt(real,**request(row))
    monkeypatch.setattr(ma,'atomic_json',original)
    atomic_json(real.run_root/'research_report.json',{'unrelated':'edit'})
    with pytest.raises(ValueError,match='outside its transaction'):
        ma.adopt(real,**request(row))
    assert read_json(real.run_root/'research_report.json')=={'unrelated':'edit'}
