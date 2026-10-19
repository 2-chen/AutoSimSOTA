"""An Agent source read must reach review, but never become runtime proof."""
import pytest

from autosim.research import candidate_execution_review as admission
from autosim.research import checkout_source_citations as checkout
from autosim.research import native_source_citations as native
from autosim.research.common import atomic_json, read_json
from autosim.research.ideas import Idea


def captured(tmp_path, monkeypatch, kind):
    output=tmp_path/'run'; repo=output/'checkout'; repo.mkdir(parents=True)
    context={'identity':'fixture-native-context','interpreter':str(output/'env/bin/python')}
    monkeypatch.setattr(native,'load_context',lambda *args:context)
    if kind=='checkout':
        path=repo/'adapter.py'
        path.write_text('def load(config):\n    return reader(config.root)\n')
        handle=checkout.capture(output,repo,{'ref':'adapter.py:1-2'})
        store='checkout_source_citations'
    else:
        path=output/'env/lib/adapter.py'; path.parent.mkdir(parents=True)
        path.write_text('def select(batch, count):\n    return predict(batch)[:, :count]\n')
        handle=native.capture(output,repo,{'root_index':0,'path':'lib/adapter.py'})
        store='native_source_citations'
    return output,repo,path,handle,store,context


@pytest.mark.parametrize('kind',['checkout','native'])
def test_explicit_source_capture_reaches_independent_packet_without_execution(tmp_path,monkeypatch,kind):
    output,repo,path,handle,store,context=captured(tmp_path,monkeypatch,kind)
    before=path.read_bytes()
    rows=admission.source_citation_observations(output,repo,[handle['evidence_id']])
    assert len(rows)==1 and rows[0]['text']==path.read_text()
    assert rows[0]['source_ref']==handle['ref'] and rows[0]['complete'] is True
    assert 'not an execution edge' in rows[0]['authority']
    (repo/'config.yml').write_text('steps: 50\n')
    idea=Idea(label='fixture',granularity='code',mechanism='change config',
        change={'file':'config.yml','find':'50','replace':'4'})
    monkeypatch.setattr(admission,'native_observations',lambda *a,**k:[])
    unseen=admission.packet_for(output,repo,{},idea)
    assert not any(s['id'].startswith('captured_source_') for s in unseen['sources'])
    packet=admission.packet_for(output,repo,{},idea,
        additional_evidence_refs=[handle['ref']])
    supplied=[s for s in packet['sources'] if s['id'].startswith('captured_source_')]
    assert len(supplied)==1 and supplied[0]['evidence_id']==handle['evidence_id']
    assert path.read_bytes()==before
    assert not (output/'research').exists()


@pytest.mark.parametrize('kind',['checkout','native'])
@pytest.mark.parametrize('fault',['source','snapshot','metadata','missing_seal','bad_status',
                                  'symlink_snapshot','cross_run','context','wrong_sealed_log'])
def test_changed_or_unowned_source_is_not_forwarded(tmp_path,monkeypatch,kind,fault):
    output,repo,path,handle,store,context=captured(tmp_path,monkeypatch,kind)
    identity=handle['evidence_id']; snapshot=output/store/(identity+'.source')
    if fault=='source': path.write_text('# changed backing source\n')
    elif fault=='snapshot': snapshot.write_text('# changed snapshot\n')
    elif fault=='metadata':
        record=read_json(output/store/(identity+'.json')); record['id']='f'*32
        atomic_json(output/store/(identity+'.json'),record)
    elif fault=='missing_seal': (output/'evidence'/(identity+'.json')).unlink()
    elif fault=='bad_status':
        record=read_json(output/'evidence'/(identity+'.json'));record['status']='failed'
        atomic_json(output/'evidence'/(identity+'.json'),record)
    elif fault=='wrong_sealed_log':
        other=output/'other.log';other.write_bytes(snapshot.read_bytes())
        record=read_json(output/'evidence'/(identity+'.json'));record['log_ref']='other.log'
        atomic_json(output/'evidence'/(identity+'.json'),record)
    elif fault=='symlink_snapshot':
        original=output/'saved.source';snapshot.rename(original);snapshot.symlink_to(original)
    elif fault=='cross_run':
        output=tmp_path/'other';output.mkdir()
    elif fault=='context':
        if kind=='checkout':
            repo=tmp_path/'other-checkout';repo.mkdir()
        else:context['identity']='changed-native-context'
    assert admission.source_citation_observations(output,repo,[identity])==[]


def test_source_capture_delivery_is_deduplicated_bounded_and_not_a_store_scan(tmp_path,monkeypatch):
    output,repo,path,handle,store,context=captured(tmp_path,monkeypatch,'native')
    assert admission.source_citation_observations(output,repo,[])==[]
    assert admission.source_citation_observations(output,repo,['../escape','z'*32])==[]
    rows=admission.source_citation_observations(output,repo,[handle['evidence_id']]*20)
    assert len(rows)==1
    path.write_text('# public source\n'+'#'+('x'*65480)+'\n')
    handles=[native.capture(output,repo,{'root_index':0,'path':'lib/adapter.py'})
             for _ in range(3)]
    rows=admission.source_citation_observations(output,repo,[h['evidence_id'] for h in handles])
    assert len(rows)==2 and sum(len(s['text'].encode()) for s in rows)<=128*1024
    path.write_text('#'+('x'*65537))
    large=native.capture(output,repo,{'root_index':0,'path':'lib/adapter.py'})
    assert admission.source_citation_observations(output,repo,[large['evidence_id']])==[]
