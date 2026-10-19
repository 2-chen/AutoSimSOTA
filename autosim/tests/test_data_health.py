import json
from pathlib import Path

import pytest

from autosim.research.data_health import lerobot_v2_profile, native_profile_code


def dataset(tmp_path, *, count=2):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    root = tmp_path / 'data'
    (root / 'meta').mkdir(parents=True)
    (root / 'frames').mkdir()
    info = {'codebase_version':'v2.1','total_episodes':count,'total_frames':count*2,
            'data_path':'frames/{episode_index}.parquet',
            'features':{'action':{'dtype':'float32','shape':[1]}}}
    (root / 'meta/info.json').write_text(json.dumps(info))
    episodes=[]
    for index in range(count):
        episodes.append({'episode_index':index,'length':2})
        pq.write_table(pa.table({'action':[[.1],[.2]],'timestamp':[0.,.04],
            'episode_index':[index,index],'frame_index':[0,1]}),root/'frames'/f'{index}.parquet')
    (root / 'meta/episodes.jsonl').write_text('\n'.join(map(json.dumps,episodes)))
    return root


def test_full_and_bounded_partial_scans_never_certify_expert_quality(tmp_path):
    root = dataset(tmp_path)
    before = (root/'meta/info.json').read_bytes()
    full = lerobot_v2_profile(root)
    assert full['scope']=='full' and not full['issues']
    assert full['counts']['parquet_frames']==4
    part = lerobot_v2_profile(root,max_episodes=1)
    assert part['scope']=='sampled' and part['population_episodes']==2
    assert 'native loader consumption' in full['unchecked']
    assert (root/'meta/info.json').read_bytes()==before


def test_missing_partial_episode_is_not_silently_counted_as_healthy(tmp_path):
    root=dataset(tmp_path)
    (root/'frames/1.parquet').unlink()
    row=lerobot_v2_profile(root)
    assert row['issues'] and row['counts']['parquet_frames']==2


def test_metadata_cannot_read_outside_the_bound_dataset(tmp_path):
    root=dataset(tmp_path)
    info=json.loads((root/'meta/info.json').read_text())
    info['data_path']='../private/{episode_index}.parquet'
    (root/'meta/info.json').write_text(json.dumps(info))
    assert any('escapes' in x['reason'] for x in lerobot_v2_profile(root)['issues'])


def test_dimension_nan_and_time_mismatch_are_reported(tmp_path):
    root=dataset(tmp_path,count=1)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({'action':[[float('nan'),.2],[.3,.4]],
        'timestamp':[0.,0.],'episode_index':[0,0],'frame_index':[0,2]}),root/'frames/0.parquet')
    issues=lerobot_v2_profile(root)['issues']
    assert any('nonfinite' in x['reason'] for x in issues)
    assert any('dimension' in x['reason'] for x in issues)
    assert any('timestamps' in x['reason'] for x in issues)


@pytest.mark.parametrize('root',['/tmp/secret','../other','a/../../other','a\\b'])
def test_profile_request_never_grants_host_paths(root):
    with pytest.raises(ValueError):native_profile_code({'format':'lerobot_v2.1','root':root})


def test_embedded_checker_is_self_contained_and_bounded(tmp_path):
    root=dataset(tmp_path)
    code=native_profile_code({'format':'lerobot_v2.1','root':str(root.relative_to(tmp_path)),
                              'max_episodes':1,'decode_videos':False})
    assert len(code.encode())<16000
    import subprocess,sys
    done=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,capture_output=True,text=True,timeout=15)
    assert done.returncode==0,done.stderr
    assert json.loads(done.stdout)['scope']=='sampled'


def test_manifest_binds_actual_files_and_native_metadata_without_inventing_labels(tmp_path):
    from autosim.research.data_quality import validate_manifest, selection
    from autosim.research.experiment_bundle import artifact_identity
    root = dataset(tmp_path)
    p = root/'meta/episodes.jsonl'
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    for i, row in enumerate(rows):
        row.update(training_split='train', expert='reset_expert', scene=f'scene:{i}',
                   family='source:1', initial_pose=i / 10)
    p.write_text('\n'.join(json.dumps(row) for row in rows))
    fields = dict(source_family='family', source_instance='scene', split='training_split',
        supervision='expert', conditions={'pose':dict(metadata_field='initial_pose', unit='metres', source_ref='native-reset')})
    profile = lerobot_v2_profile(root, include_manifest=True, manifest_fields=fields)
    manifest = validate_manifest(profile['episode_manifest'], root=root)
    assert artifact_identity(root) == (manifest['identity_field'], manifest['identity'])
    result = selection(manifest, target=dict(field='pose',op='between',value=[.05,.2],unit='metres'))
    assert result['selected_episode_ids'] == ['1']
    unknown = lerobot_v2_profile(root, include_manifest=True)['episode_manifest']
    assert all(row['supervision'] is None and row['source_family'] is None for row in unknown['episodes'])


def test_manifest_keeps_incomplete_episode_visible_but_out_of_usable_view(tmp_path):
    from autosim.research.data_quality import selection
    root = dataset(tmp_path)
    (root/'frames/1.parquet').unlink()
    profile = lerobot_v2_profile(root, include_manifest=True)
    assert profile['manifest_status'] == 'complete'
    assert profile['episode_manifest']['episodes'][1]['committed'] is False
    result = selection(profile['episode_manifest'], require_split=None)
    assert result['selected_episode_ids'] == ['0']
    assert result['rejected'] == {'uncommitted_or_format_unverified':1}


def test_embedded_manifest_checker_outputs_a_complete_byte_bound_manifest(tmp_path):
    from autosim.research.experiment_bundle import artifact_identity
    import subprocess, sys
    root = dataset(tmp_path)
    code = native_profile_code({'format':'lerobot_v2.1','root':'data','include_manifest':True})
    assert len(code.encode()) < 16000
    result = subprocess.run([sys.executable,'-c',code],cwd=tmp_path,capture_output=True,text=True,timeout=15)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)['episode_manifest']
    assert artifact_identity(root) == (manifest['identity_field'], manifest['identity'])
