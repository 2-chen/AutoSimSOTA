"""Regression contracts for interrupted registration and model-visible evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosim.research import baseline_reference as refs
from autosim.research.common import sanitize_model_payload
from autosim.research.experiment_bundle import freeze_artifact, artifact_identity
from tests.test_main_agent import make, Client


def test_reference_archive_reconciles_exact_copy_without_overwriting(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text('{"native":true}')
    (source/'weights.bin').write_bytes(b'policy bytes')
    destination = tmp_path/'run/baseline_references/public/policy'
    first = freeze_artifact(source, destination, max_bytes=1024)
    inode = (destination/'weights.bin').stat().st_ino
    resumed = refs.freeze_reference(source, destination, max_bytes=1024)
    assert resumed['sha256'] == first['sha256']
    assert resumed['reconciled_existing_copy']
    assert (destination/'weights.bin').stat().st_ino == inode
    (source/'weights.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='conflicts'):
        refs.freeze_reference(source, destination, max_bytes=1024)
    assert (destination/'weights.bin').read_bytes() == b'policy bytes'


def test_reference_copy_never_reuses_links_or_mutable_inodes(tmp_path):
    import os
    source = tmp_path/'weights.bin'
    source.write_bytes(b'policy')
    destination = tmp_path/'policy'
    destination.symlink_to(source)
    with pytest.raises(ValueError, match='unsafe'):
        refs.freeze_reference(source, destination, max_bytes=1024)
    destination.unlink()
    os.link(source, destination)
    with pytest.raises(ValueError, match='inode'):
        refs.freeze_reference(source, destination, max_bytes=1024)


def test_reference_review_maps_exact_visible_quote_not_desanitized_guess():
    original = [{'ref':'README.md', 'quote':'Read /private/resources/policy.bin for native weights.'}]
    shown = sanitize_model_payload([{**original[0], 'source_id':'reference-source-1'}])
    assert shown[0]['quote'] != original[0]['quote']
    citation = {'source_id':'reference-source-1', 'quote':shown[0]['quote']}
    assert refs.resolve_review_citations(original, shown, [citation]) == original
    for changed in ({**citation,'quote':'not supplied'}, {**citation,'source_id':'invented'}):
        with pytest.raises(ValueError):
            refs.resolve_review_citations(original, shown, [changed])
    # A lossy projection cannot be resolved by guessing which original it meant.
    with pytest.raises(ValueError):
        refs.resolve_review_citations(original*2, shown*2,
            [{'ref':shown[0]['ref'],'quote':shown[0]['quote']}])


def fixture(tmp_path, monkeypatch):
    class Reviewer(Client):
        accept = True
        corrupt = False
        def chat_with_metadata(self, system, user, **kwargs):
            payload = json.loads(user)
            shown = payload['reference']['source_refs'][0]
            if self.corrupt:
                weight.write_bytes(b'changed during review')
            return json.dumps(dict(sources_supported=self.accept, protocol_matches=self.accept,
                tolerance_supported=self.accept, release_identity_supported=self.accept,
                why='audited source and protocol', citations=[{
                    'source_id':shown['source_id'], 'quote':shown['quote']}])), {}
    client = Reviewer()
    real = make(tmp_path, client)
    (real.repo/'README.md').write_text('Native reference success 0.37, weights at /private/policy.bin.')
    weight = real.output/'release.bin'
    weight.write_bytes(b'public release')
    spec = dict(id='public', kind='released_checkpoint', reason='Verify published reference',
        source_refs=[dict(ref='README.md',quote=(real.repo/'README.md').read_text())],
        release_url='https://example.org/public',revision='abc',checkpoint_ref='release.bin',
        episodes=100,expected_metric=0.37,tolerance=0.05)
    monkeypatch.setattr(real, '_research_controller', lambda: (
        SimpleNamespace(metric_spec=SimpleNamespace(as_dict=lambda:{'name':'success_rate'})),None))
    return real, client, weight, spec


def test_refused_review_does_not_create_orphan_archive(tmp_path, monkeypatch):
    real, client, _, spec = fixture(tmp_path, monkeypatch)
    client.accept = False
    result = real._step_configure_baseline_reference(reference=spec)
    assert result['reference_status'] == 'rejected', result
    assert not (real.output/'baseline_references/public/policy').exists()


def test_legacy_orphan_retry_publishes_reviewed_reference(tmp_path, monkeypatch):
    real, _, weight, spec = fixture(tmp_path, monkeypatch)
    dest = real.output/'baseline_references/public/policy'
    freeze_artifact(weight, dest, max_bytes=1024)
    result = real._step_configure_baseline_reference(reference=spec)
    assert result['reference_status'] == 'ready', result
    assert refs.handoff(real.output, 'public')['selected_path'] == str(dest)
    assert artifact_identity(weight) == artifact_identity(dest)


def test_changed_reference_during_review_cannot_be_published(tmp_path, monkeypatch):
    real, client, _, spec = fixture(tmp_path, monkeypatch)
    client.corrupt = True
    result = real._step_configure_baseline_reference(reference=spec)
    assert result['outcome'] == 'not attempted'
    assert 'changed during review' in result['because']
    assert not (real.output/'baseline_references/public/policy').exists()


def test_storage_contract_refusal_offers_identity_route_not_gc_workaround(tmp_path, monkeypatch):
    real = make(tmp_path)
    monkeypatch.setattr('autosim.research.storage_lifecycle.idle',lambda *a:None)
    def reject(*a, **kwargs):
        raise ValueError('metadata, evidence and source cannot be garbage-collected')
    monkeypatch.setattr('autosim.research.storage_lifecycle.validate_registration',reject)
    result = real.do('register_storage_artifact', evidence_id='e'*32,
        refs=['metadata.json'],role='terminal',reason='protect output',source_evidence=[])
    assert result['outcome'] == 'not attempted', result
    assert result['recoverable']
    assert result['verified_train_artifacts']['status'] == 'unavailable'
    assert any('derive_a_command' in x for x in result['next_actions'])
    assert not list((real.output/'storage/artifacts').glob('*.json'))
