"""Historical candidates retain their paid producer after a safe code rollback."""
from pathlib import Path

import pytest

from autosim.research import checkpoint_handoff as ch, handoff_correction as hc
from autosim.research import measurement_recovery as mr
from autosim.research.codepatch import Patch, apply_many, revert
from autosim.research.common import atomic_json, digest, read_json
from tests.test_measurement_recovery import failed_bundle


def rolled_back_candidate(tmp_path):
    real, _ = failed_bundle(tmp_path)
    producer = real.repo/'produce.py'
    producer.write_text('import helper\n'+producer.read_text())
    (real.repo/'helper.py').write_text('COMPATIBILITY = 1\n')
    session = read_json(real.run_root/'controller_session.json')
    atomic_json(real.run_root/'controller_session.json', {
        **session, 'schema_version':1, 'repository':str(real.repo)})
    patch = Patch('produce.py', 'completed += 1',
                  'completed += 1  # archived candidate intervention')
    real.snapshots.capture([patch.file], repo=real.repo, name='before-round-2')
    original = apply_many([patch], repo=real.repo)
    real.snapshots.capture([patch.file], repo=real.repo, name='round-2')
    atomic_json(real.run_root/'code_changes/round-2.json', {
        'round':2, 'status':'applied', 'snapshot':'before-round-2',
        'patches':[patch.as_dict()]})
    failed = real.measure(settings={}, label='round_2')
    assert not failed['ok'] and failed['where'] == 'evaluate'
    revert(patch, original[patch.file], repo=real.repo)
    # Match a real paused caller with its exact isolated repository.
    session = read_json(real.run_root/'controller_session.json')
    atomic_json(real.run_root/'controller_session.json', {
        **session, 'repository':str(real.repo)})
    return real, failed, patch


def test_historical_candidate_catalog_and_recovery_do_not_retrain(tmp_path, monkeypatch):
    real, failed, patch = rolled_back_candidate(tmp_path)
    source_sha = digest(real.repo/patch.file)
    measurement_sha = digest(real.run_root/'measurements/round_2.json')
    receipt = real.run_root/'attempts'/failed['train']['attempt_id']/'receipt.json'
    receipt_sha = digest(receipt)
    offered = ch.catalog(real.output)
    candidate = next(row for row in offered['producers'] if row.get('label') == 'round_2')
    assert candidate['source_provenance']['kind'] == 'archived_candidate_snapshot'
    assert candidate['progress']['evidence']['native_counter']['value'] == 17
    assert digest(real.repo/patch.file) == source_sha  # inventory is read-only
    packet = next(row for row in mr.offered(real.output, real.run_id) if row['label'] == 'round_2')
    assert packet['training_attempt_id'] == failed['train']['attempt_id']
    assert any(row['label'] == 'round_2' for row in hc.eligible(real.run_root))
    native = real.run_stage
    calls = []
    def evaluation_only(stage, **kw):
        assert stage == 'evaluate'
        calls.append(stage)
        assert 'archived candidate intervention' in (real.repo/patch.file).read_text()
        return native(stage, **kw)
    monkeypatch.setattr(real, 'run_stage', evaluation_only)
    result = hc.correct(real, label='round_2',
        checkpoint_ref=candidate['output_ref']+'/arbitrary_native_bundle')
    assert result['ok'] and result['metric_value'] == .6, result
    assert calls == ['evaluate']
    assert result['train']['attempt_id'] == failed['train']['attempt_id']
    assert digest(receipt) == receipt_sha
    assert digest(real.run_root/'measurements/round_2.json') == measurement_sha
    assert digest(real.repo/patch.file) == source_sha
    assert not any(row['label'] == 'round_2' for row in mr.offered(real.output, real.run_id))


@pytest.mark.parametrize('fault', ['missing_blob', 'corrupt_blob', 'preimage_drift',
                                 'patch_drift', 'dependency_drift', 'counter_drift'])
def test_historical_source_failure_is_visible_and_never_offers_recovery(tmp_path, fault):
    real, failed, patch = rolled_back_candidate(tmp_path)
    after = real.snapshots.get('round-2')
    blob = real.snapshots.blobs/after.files[patch.file]
    if fault == 'missing_blob':
        blob.unlink()
    elif fault == 'corrupt_blob':
        blob.write_text('corrupt archived source')
    elif fault == 'preimage_drift':
        (real.repo/patch.file).write_text('print("unrelated edit")')
    elif fault == 'patch_drift':
        transaction = real.run_root/'code_changes/round-2.json'
        row = read_json(transaction)
        row['patches'][0]['replace'] += ' altered'
        atomic_json(transaction, row)
    elif fault == 'dependency_drift':
        # Restorable top-level code does not authorize a changed dependency.
        (real.repo/'helper.py').write_text('COMPATIBILITY = 2\n')
    else:
        receipt = read_json(real.run_root/'attempts'/failed['train']['attempt_id']/'receipt.json')
        output = Path(receipt['output_directory'])
        atomic_json(output/'progress.json', {'completed':999})
    result = ch.catalog(real.output)
    assert not any(row.get('label') == 'round_2' for row in result.get('producers', []))
    assert any(row.get('measurement_ref', '').endswith('/round_2.json') and row['because']
               for row in result['pending_producers'])
    assert not any(row['label'] == 'round_2' for row in mr.offered(real.output, real.run_id))


def test_archived_import_graph_is_traced_without_live_source_writes(tmp_path):
    from autosim.research.training_progress_audit import sources
    from autosim.research.common import atomic_text
    output = tmp_path/'run'
    repo = output/'checkout'
    repo.mkdir(parents=True)
    (repo/'producer.py').write_text('import new_helper\n')
    (repo/'old_helper.py').write_text('original_work = 17\n')
    (repo/'new_helper.py').write_text('changed_work = 99\n')
    old = 'import old_helper\n'
    import hashlib
    key = hashlib.sha256(old.encode()).hexdigest()
    blob = output/'research/derived/snapshots/blobs'/key
    atomic_text(blob, old)
    rows = sources(repo, output, ['producer.py'], source_overrides={repo/'producer.py':blob})
    assert rows[0]['content'] == old
    assert rows[1]['path'] == repo/'old_helper.py'
    assert not any('changed_work' in row['content'] for row in rows)
    assert (repo/'producer.py').read_text() == 'import new_helper\n'


def test_state_shares_one_inventory_but_next_state_revalidates_sources(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    real, _, patch = rolled_back_candidate(tmp_path)
    prep = Preparation(repo=real.repo,output=real.output,scouting=real.output/'scouting',
                       client=real.client)
    monkeypatch.setattr(prep,'_unscored_baseline_recovery_status',lambda:{'available':False})
    calls=[]
    original=ch.formal_producers
    def count(*a,**kw):
        import traceback
        calls.append([row.name for row in traceback.extract_stack()[-7:-1]])
        return original(*a,**kw)
    monkeypatch.setattr(ch,'formal_producers',count)
    facts=prep.state()
    assert len(calls)==1, calls
    assert any(r['label']=='round_2' for r in facts['research_progress']['development_recoveries'])
    (real.repo/patch.file).write_text('print("later unrelated edit")')
    facts=prep.state()
    assert len(calls)==2  # No cross-state cache or stale positive admission.
    assert not any(r['label']=='round_2' for r in facts['research_progress']['development_recoveries'])


def test_initialized_study_does_not_hydrate_engine_to_offer_presearch_revision(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    real,_=failed_bundle(tmp_path)
    atomic_json(real.run_root/'protocol_frozen.json',{})
    prep=Preparation(repo=real.repo,output=real.output,scouting=real.output/'scouting',client=real.client)
    monkeypatch.setattr(prep,'_research_controller',lambda:pytest.fail('unnecessary engine hydration'))
    assert prep._presearch_protocol_revision_view()['status']=='not_available'
