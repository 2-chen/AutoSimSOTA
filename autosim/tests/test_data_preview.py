"""Existing native data videos are previews, never another policy measurement."""
import json

import pytest

from autosim.research import data_preview, data_versions, recorder
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.experiment_bundle import artifact_identity
from tests.test_data_health import dataset
from tests.test_data_lineage import producer, REVIEW
from tests.test_partial_collection import manifest_probe
from tests.test_data_versions import registered
from tests.test_prepare import preparation


def native_data(tmp_path):
    av = pytest.importorskip('av')
    np = pytest.importorskip('numpy')
    repo, output, _, _ = registered(tmp_path)
    root = dataset(output)
    info = read_json(root/'meta/info.json')
    info.update(video_path='videos/{episode_index}.mp4')
    info['features']['observation.images.cam'] = {'dtype':'video','shape':[16,16,3],
                                                'names':['height','width','channels']}
    atomic_json(root/'meta/info.json',info)
    (root/'videos').mkdir()
    for i in range(2):
        with av.open(str(root/'videos'/f'{i}.mp4'),'w') as writer:
            stream = writer.add_stream('libx264',rate=25)
            stream.width=16; stream.height=16; stream.pix_fmt='yuv420p'
            for _ in range(2):
                frame=av.VideoFrame.from_ndarray(np.full((16,16,3),i*60,dtype=np.uint8),format='rgb24')
                for packet in stream.encode(frame): writer.mux(packet)
            for packet in stream.encode(): writer.mux(packet)
    ident,_=producer(output,root,status='timed_out')
    profile=manifest_probe(output,root)
    card=data_versions.register(output,repo,dict(id='samples',dataset_ref='data',source_refs=[],reason='native fixture',
        asset={'kind':'collected','producer_evidence_id':ident,'manifest_evidence_id':profile}),REVIEW)
    return repo,output,root,card


def request(output):
    held={'event_revision':'a'*64,'revision':'b'*64,'measurements':[],
          'data_versions':data_versions.catalog(output)}
    recorder._enqueue(output,held,[dict(kind='dataset_preview',data_version_id='samples',purpose='观察原生轨迹和相机')])
    pending = recorder.pending_requests(output)
    return pending[0] if pending else None


def test_data_preview_reuses_bounded_native_media_without_raw_mutation(tmp_path):
    repo,output,root,_=native_data(tmp_path)
    before=artifact_identity(root)
    pending=request(output)
    recorder.resolve_request(output,pending['id'],status='capturing',reason='show data')
    result=data_preview.capture(output,repo,version_id='samples',card_sha256=pending['data_version_sha256'])
    recorder.complete_capture(output,pending['id'],result)
    assert result['status']=='captured' and result['gpu_access'] is False
    assert len(result['media'])==2 and result['preview_bytes']>0
    shown=data_preview.view(output)
    assert len(shown)==2 and shown[0]['episode_id']=='0'
    assert all(row['path'].startswith('report/data_previews/') for row in shown)
    assert artifact_identity(root)==before
    assert read_json(output/'producer'/('a'*32)/'receipt.json')['status']=='timed_out'
    panel=recorder.data_panel(output,{'revision':'c'*64,'data_versions':data_versions.catalog(output),
                                    'data_previews':shown,'measurements':[]})
    assert any('播放数据录像' in line for line in panel)
    request(output)
    assert recorder.pending_requests(output)==[]  # same version, no repeated requests
    copied=output/shown[0]['path'];copied.write_bytes(b'changed')
    assert len(data_preview.view(output))==1


def test_scheduler_handles_data_preview_before_first_policy_score(tmp_path):
    repo,output,_,_=native_data(tmp_path)
    pending=request(output)
    real=preparation(tmp_path);real.repo=repo;real.output=output
    real.client.supports_recorder=True
    result=real._step_review_report_demo(request_id=pending['id'],decision='capture',reason='展示真实数据')
    assert result['outcome']=='captured'
    assert len(data_preview.view(output))==2
    assert not (output/'research/derived/measurements').exists()


def test_data_preview_cannot_use_changed_card_or_reopen_heldout(tmp_path):
    repo,output,_,_=native_data(tmp_path)
    pending=request(output)
    with pytest.raises(ValueError):
        data_preview.capture(output,repo,version_id='samples',card_sha256='0'*64)
    atomic_json(output/'research/derived/confirmation_attempts/a.json',{})
    real=preparation(tmp_path);real.repo=repo;real.output=output;real.client.supports_recorder=True
    result=real._step_review_report_demo(request_id=pending['id'],decision='capture',reason='invalid reopen')
    assert result['outcome']=='rejected'
    assert not (output/'report/data_previews').exists()
