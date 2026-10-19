"""Actual parquet/video timing and diagnostic repetition, never synthetic scores."""
import json
from fractions import Fraction

import pytest

from autosim.research.data_health import lerobot_v2_profile, native_profile_code
from tests.test_data_health import dataset


@pytest.mark.parametrize('fault', ['nan', 'null', 'ragged'])
def test_nested_numeric_fields_fail_format_validation(tmp_path, fault):
    root = dataset(tmp_path, count=1)
    import pyarrow as pa
    import pyarrow.parquet as pq
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features']['pose'] = {'dtype': 'float32', 'shape': [2, 2]}
    info_path.write_text(json.dumps(info))
    pose = [[1., 0.], [0., 1.]]
    if fault == 'nan': pose[1][0] = float('nan')
    if fault == 'null': pose[1][0] = None
    if fault == 'ragged': pose[1] = [1.]
    table = pq.read_table(root / 'frames/0.parquet').to_pydict()
    table['pose'] = [pose, [[1., 0.], [0., 1.]]]
    pq.write_table(pa.table(table), root / 'frames/0.parquet')
    result = lerobot_v2_profile(root, include_manifest=True)
    assert any('pose' in issue['reason'] for issue in result['issues'])
    assert result['episode_manifest']['episodes'][0]['format_valid'] is False


def test_action_repetition_is_not_automatic_rejection_or_coverage_certification(tmp_path):
    root = dataset(tmp_path)
    result = lerobot_v2_profile(root, include_manifest=True)
    assert not result['issues']
    assert all(row['format_valid'] for row in result['episode_manifest']['episodes'])
    assert result['diagnostics']['unique_exact_action_sequences'] == 1
    assert result['diagnostics']['repeated_action_episodes'] == 1
    assert result['diagnostics']['repeated_action_groups_preview'] == [[0, 1]]
    partial = lerobot_v2_profile(root, max_episodes=1)
    assert partial['diagnostics']['scope'] == 'sampled'
    assert partial['diagnostics']['repeated_action_episodes'] == 0


def test_declared_fps_is_checked_but_not_invented(tmp_path):
    root = dataset(tmp_path, count=1)
    assert not lerobot_v2_profile(root)['diagnostics']['declared_fps_checked']
    path = root / 'meta/info.json'
    info = json.loads(path.read_text())
    info['fps'] = 25
    path.write_text(json.dumps(info))
    assert not lerobot_v2_profile(root)['issues']
    info['fps'] = 50
    path.write_text(json.dumps(info))
    assert any('frame rate' in row['reason'] for row in lerobot_v2_profile(root)['issues'])


@pytest.mark.parametrize('fps', [0, -1, True, None])
def test_invalid_rate_does_not_mark_episode_format_valid(tmp_path, fps):
    root = dataset(tmp_path, count=1)
    path = root / 'meta/info.json'
    info = json.loads(path.read_text())
    info['fps'] = fps
    path.write_text(json.dumps(info))
    result = lerobot_v2_profile(root, include_manifest=True)
    assert any('frame rate' in row['reason'] for row in result['issues'])
    assert result['episode_manifest']['episodes'][0]['format_valid'] is False


@pytest.mark.parametrize('offset', [0, 1])
def test_real_cpu_video_pts_must_match_parquet_even_when_counts_match(tmp_path, offset):
    av = pytest.importorskip('av')
    np = pytest.importorskip('numpy')
    root = dataset(tmp_path, count=1)
    path = root / 'meta/info.json'
    info = json.loads(path.read_text())
    info.update(fps=25, video_path='video/{episode_index}.mp4')
    info['features']['camera'] = {'dtype': 'video', 'shape': [16, 16, 3],
                                'names': ['height', 'width', 'channel']}
    path.write_text(json.dumps(info))
    (root / 'video').mkdir()
    with av.open(str(root / 'video/0.mp4'), 'w') as writer:
        stream = writer.add_stream('mpeg4', rate=25)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        for n in range(2):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), n * 100, dtype=np.uint8), format='rgb24')
            frame.pts, frame.time_base = n + offset, Fraction(1, 25)
            for packet in stream.encode(frame): writer.mux(packet)
        for packet in stream.encode(): writer.mux(packet)
    result = lerobot_v2_profile(root, include_manifest=True)
    assert result['counts']['videos_decoded'] == 1
    assert result['counts']['parquet_frames'] == 2
    if offset:
        assert any('presentation timestamp' in row['reason'] for row in result['issues'])
        assert not result['episode_manifest']['episodes'][0]['format_valid']
    else:
        assert not result['issues']


def test_embedded_temporal_checker_is_still_bounded_and_self_contained(tmp_path):
    import subprocess
    import sys
    root = dataset(tmp_path)
    code = native_profile_code({'format': 'lerobot_v2.1', 'root': 'data',
                                'include_manifest': True, 'decode_videos': False})
    assert len(code.encode()) < 16000
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['diagnostics']['repeated_action_episodes'] == 1


def test_large_legal_native_metadata_mapping_still_fits_diagnostic_contract():
    fields = {key: key + 'x' * 130 for key in ('source_family', 'source_instance', 'split', 'supervision')}
    fields['conditions'] = {str(i): {'metadata_field': 'x' * 130, 'unit': 'm',
                                   'source_ref': 'y' * 220} for i in range(3)}
    code = native_profile_code({'format': 'lerobot_v2.1', 'root': 'x' * 500,
                                'include_manifest': True, 'manifest_fields': fields})
    assert len(code.encode()) < 16000
