import json
from pathlib import Path

import pytest

from autosim.research import frozen_policy_trial as trial
from autosim.research.common import atomic_json,digest,read_json
from tests.test_derived_research import ambiguous_policy_research


def fixture(tmp_path,*,directory=False):
    real=ambiguous_policy_research(tmp_path)
    if directory:
        sources=dict(real.sources)
        sources['train']=sources['train'].replace("(p/'one.pth').write_text('one'); (p/'two.pth').write_text('two');",
            "(p/'one.pth').mkdir(); (p/'two.pth').mkdir(); (p/'two.pth'/'model.bin').write_text('two');")
        sources['evaluate']=sources['evaluate'].replace('p.read_text()',"(p/'model.bin').read_text()")
        from autosim.research.declarative_backend import DeclarativeBackend
        real.backend=DeclarativeBackend(repo=tmp_path,answer=real.backend.answer,sources=sources,parameters={})
        real.sources=sources
    row=real.measure(settings={},label='baseline')
    assert row['ok'],row
    request={'label':'baseline','measurement_sha256':digest(real.run_root/'measurements/baseline.json')}
    return real,row,request


def metadata_fixture(tmp_path):
    archive = tmp_path / 'archive/policy'
    archive.mkdir(parents=True)
    atomic_json(archive / 'config.json', {'n_action_steps': 32, 'horizon': 33})
    atomic_json(archive / 'train_config.json', {'seed': 0})
    (archive / 'model.safetensors').write_bytes(b'not parsed as metadata')
    atomic_json(archive / 'weights.json', {'tensor': [123, 456]})
    path = tmp_path / 'measurements/baseline.json'
    atomic_json(path, {'policy_artifact': {'path': str(archive), 'sha256': 'a'*64}})
    return archive, path, {'label': 'baseline', 'measurement_sha256': digest(path)}


def test_frozen_policy_metadata_binds_parent_and_reads_only_bounded_configuration(tmp_path, monkeypatch):
    archive, path, request = metadata_fixture(tmp_path)
    original = Path.read_bytes
    def read_bytes(p):
        assert p.name not in {'model.safetensors', 'weights.json'}
        return original(p)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    original_text = Path.read_text
    def read_text(p, *args, **kwargs):
        assert p.name not in {'model.safetensors', 'weights.json'}
        return original_text(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_text)
    rows = trial.metadata_sources(tmp_path, request)
    assert {row['path'] for row in rows} == {'config.json', 'train_config.json'}
    assert json.loads(rows[0]['text'])['n_action_steps'] == 32
    assert all(row['policy_sha256'] == 'a'*64 and row['complete'] for row in rows)
    assert all(row['sha256'] == digest(archive/row['path']) for row in rows)
    assert trial.metadata_sources(tmp_path, {**request, 'measurement_sha256': 'b'*64}) == []
    assert trial.metadata_sources(tmp_path, {**request, 'label': '../escape'}) == []


def test_frozen_policy_metadata_bounds_inventory_and_delivery(tmp_path):
    archive, _, request = metadata_fixture(tmp_path)
    for index in range(10):
        atomic_json(archive/f'{index:03}_config.json', {'padding': 'x'*8000})
    rows = trial.metadata_sources(tmp_path, request)
    assert len(rows) <= 8 and sum(len(row['text'].encode()) for row in rows) <= 32768
    for index in range(129):
        atomic_json(archive/f'{index:03}_ignored.json', {'not_configuration': True})
    assert trial.metadata_sources(tmp_path, request) == []


@pytest.mark.parametrize('mode', ['symlink_file', 'symlink_archive', 'external', 'file_only', 'large', 'invalid'])
def test_frozen_policy_metadata_never_reads_unbound_or_unsafe_configuration(tmp_path, mode):
    archive, path, request = metadata_fixture(tmp_path)
    if mode == 'symlink_file':
        outside = tmp_path / 'secret.json'
        atomic_json(outside, {'private': True})
        (archive / 'config.json').unlink()
        (archive / 'config.json').symlink_to(outside)
    elif mode == 'symlink_archive':
        alias = tmp_path / 'alias'
        alias.symlink_to(archive, target_is_directory=True)
        atomic_json(path, {'policy_artifact': {'path': str(alias)}})
    elif mode == 'external':
        atomic_json(path, {'policy_artifact': {'path': str(tmp_path.parent)}})
    elif mode == 'file_only':
        atomic_json(path, {'policy_artifact': {'path': str(archive/'model.safetensors')}})
    elif mode == 'large':
        (archive/'config.json').write_text(' ' * 16385)
        (archive/'train_config.json').write_text(' ' * 16385)
    else:
        (archive/'config.json').write_text('invalid json')
    request['measurement_sha256'] = digest(path)
    assert trial.metadata_sources(tmp_path, request) == []


def test_frozen_policy_metadata_changes_review_identity_without_relaxing_routing(tmp_path):
    from autosim.research import candidate_execution_review as admission
    from tests.test_candidate_execution_review import fixture as source_fixture
    archive, _, request = metadata_fixture(tmp_path)
    (tmp_path / 'code').mkdir()
    repo, idea, stages = source_fixture(tmp_path / 'code')
    rows = trial.metadata_sources(tmp_path, request)
    packet = admission.packet_for(tmp_path, repo, stages, idea, reuse_policy=request, policy_metadata=rows)
    assert next(s for s in packet['sources'] if s['id'] == rows[0]['id'])['text'] == rows[0]['text']
    assert 'a'*64 not in json.dumps(packet)
    atomic_json(archive/'config.json', {'n_action_steps': 8})
    changed = admission.packet_for(tmp_path, repo, stages, idea, reuse_policy=request,
                                  policy_metadata=trial.metadata_sources(tmp_path, request))
    assert changed != packet
    with pytest.raises(ValueError, match='workflow and intervention'):
        admission.validate({'verdict': 'applicable', 'reason': 'Metadata is not routing proof.',
                            'source_evidence': [{'source_id': rows[0]['id'], 'quote': rows[0]['text']}]}, packet)


@pytest.mark.parametrize('directory',[False,True])
def test_frozen_policy_only_runs_new_evaluation_and_preserves_identity(tmp_path,monkeypatch,directory):
    real,parent,request=fixture(tmp_path,directory=directory)
    assert Path(parent['policy_artifact']['path']).is_dir() is directory
    selected=trial.load(real,request)
    assert selected['identity']['policy_sha256']==parent['policy_artifact']['sha256']
    original=real.run_stage;stages=[]
    def run(stage,**kwargs):
        stages.append(stage)
        assert stage=='evaluate','frozen policy unexpectedly retrained'
        return original(stage,**kwargs)
    monkeypatch.setattr(real,'run_stage',run)
    monkeypatch.setattr(real,'_select_policy_artifact',lambda *a,**k:pytest.fail('reselected checkpoint'))
    pinned=real.run_root/'measurements/baseline.json';before=pinned.read_bytes()
    result=real.measure(settings={},label='frozen_trial',_frozen_policy_request=request)
    assert result['ok'],result
    assert stages==['evaluate'] and result['train']['reused'] is True
    assert result['train']['attempt_id']==parent['train']['attempt_id']
    assert result['evaluate']['attempt_id']!=parent['evaluate']['attempt_id']
    assert result['policy_artifact']==parent['policy_artifact'] and pinned.read_bytes()==before
    assert result['policy_reuse']['no_new_training_or_data_claim'] is True
    assert result['baseline_origin']=='locally_trained'
    from autosim.research.receipt_verifier import verify_measurement
    assert verify_measurement(real.run_root,'frozen_trial')['status']=='consistent'


@pytest.mark.parametrize('fault',['measurement','policy','native_log','request','heldout','settings','selection','released','symlink'])
def test_unverified_or_changed_frozen_policy_refused_before_native_work(tmp_path,monkeypatch,fault):
    real,parent,request=fixture(tmp_path)
    root=real.run_root
    if fault=='measurement':
        parent['metric_value']=1;atomic_json(root/'measurements/baseline.json',parent)
    elif fault=='policy': Path(parent['policy_artifact']['path']).write_text('changed')
    elif fault=='native_log':
        receipt=read_json(root/'attempts'/parent['train']['attempt_id']/'receipt.json')
        Path(receipt['log']).write_text('changed')
    elif fault=='request': request['extra']='other policy'
    elif fault=='heldout': atomic_json(root/'confirmation_attempts/seen.json',{})
    elif fault=='settings':
        parent['settings']={'steps':999};atomic_json(root/'measurements/baseline.json',parent)
        request['measurement_sha256']=digest(root/'measurements/baseline.json')
    elif fault=='selection':
        parent['train']['artifact']['selected_path']='models/one.pth'
        atomic_json(root/'measurements/baseline.json',parent)
        request['measurement_sha256']=digest(root/'measurements/baseline.json')
    elif fault=='released':
        parent['baseline_origin']='released_reference';atomic_json(root/'measurements/baseline.json',parent)
        request['measurement_sha256']=digest(root/'measurements/baseline.json')
    elif fault=='symlink':
        path=root/'measurements';path.rename(root/'elsewhere');path.symlink_to(root/'elsewhere',target_is_directory=True)
    monkeypatch.setattr(real,'run_stage',lambda *a,**k:pytest.fail('invalid frozen policy launched native work'))
    result=real.measure(settings={},label='invalid_frozen_trial',_frozen_policy_request=request)
    assert result['ok'] is False


def test_main_selected_formal_code_round_reuses_policy_without_training(tmp_path,monkeypatch):
    from autosim.research.ideas import Idea,CLEARED
    real=ambiguous_policy_research(tmp_path)
    report=real.run(rounds=1,yield_after_action=True,main_controller_owns_selection=True)
    assert report['run_status']=='paused'
    path=real.run_root/'measurements/baseline.json';parent=read_json(path)
    request={'label':'baseline','measurement_sha256':digest(path)}
    original=(tmp_path/'ppo.py').read_text()
    idea=Idea(label='native-deployment-test',granularity='code',risk='low',crosses='none',
        mechanism='exercise formal frozen-policy flow with a tiny deployment fixture',
        change={'file':'ppo.py','find':original,'replace':original+'# deployment variant\n'},
        why='fixture only; not a metric improvement claim',status=CLEARED)
    real.library.add(idea)
    stage=real.run_stage;calls=[]
    def run(name,**kwargs):
        calls.append(name);assert name=='evaluate'
        return stage(name,**kwargs)
    monkeypatch.setattr(real,'run_stage',run)
    report=real.run(rounds=1,yield_after_action=True,main_controller_owns_selection=True,
        selected_idea_label=idea.label,candidate_frozen_policy=request)
    assert calls==['evaluate']
    result=read_json(real.run_root/'measurements/round_1.json')
    assert result['ok'],result
    assert result['train']['attempt_id']==parent['train']['attempt_id']
    assert result['settings']==parent['settings']
    assert result['policy_reuse']['measurement_sha256']==request['measurement_sha256']
    assert read_json(path)==parent and report['rounds'][1]['label']=='round_1'


@pytest.mark.parametrize('extra',[{'settings':{'train.n_epochs':2}}, {'dataset':'other'}, {'confirmation':True}])
def test_frozen_trial_cannot_claim_new_training_data_or_confirmation(tmp_path,monkeypatch,extra):
    real,parent,request=fixture(tmp_path)
    monkeypatch.setattr(real,'run_stage',lambda *a,**k:pytest.fail('conflicting trial launched native work'))
    result=real.measure(**({'settings':{},'label':'conflict',**extra}),_frozen_policy_request=request)
    assert result['ok'] is False


def test_offer_does_not_create_snapshot_storage(tmp_path):
    assert trial.offered(tmp_path)==[]
    assert list(tmp_path.iterdir())==[]


def test_offer_control_identity_survives_model_projection_without_policy_hash(tmp_path):
    from autosim.research.common import sanitize_model_payload
    real, parent, request = fixture(tmp_path)
    atomic_json(real.run_root/'snapshots/snapshots.json', {'best': 'baseline'})
    raw = trial.offered(real.run_root)
    visible = sanitize_model_payload({'research_options': {'frozen_policy_trials': raw}},
                                     local_roots=(real.output, real.repo))['research_options']['frozen_policy_trials'][0]
    assert {key: visible[key] for key in request} == request
    assert visible['policy_sha256'] != raw[0]['policy_sha256']


def test_trial_cannot_overwrite_parent_measurement(tmp_path,monkeypatch):
    real,parent,request=fixture(tmp_path)
    path=real.run_root/'measurements/baseline.json';before=path.read_bytes()
    monkeypatch.setattr(real,'run_stage',lambda *a,**k:pytest.fail('overwrote parent'))
    from autosim.research.derived_research import ResearchStateError
    with pytest.raises(ResearchStateError):
        real.measure(settings={},label='baseline',_frozen_policy_request=request)
    assert path.read_bytes()==before


@pytest.mark.parametrize('field',['measurement_sha256','training_receipt_sha256','policy_sha256','training_attempt_id'])
def test_independent_verifier_rechecks_frozen_policy_lineage(tmp_path,field):
    from autosim.research.receipt_verifier import verify_measurement
    real,parent,request=fixture(tmp_path)
    result=real.measure(settings={},label='frozen_trial',_frozen_policy_request=request)
    assert result['ok']
    assert verify_measurement(real.run_root,'frozen_trial')['checks']['frozen_policy_parent']
    result['policy_reuse'][field]='0'*64
    atomic_json(real.run_root/'measurements/frozen_trial.json',result)
    assert verify_measurement(real.run_root,'frozen_trial')['status']=='inconsistent'
