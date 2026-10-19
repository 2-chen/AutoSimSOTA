"""A failed producer is immutable; separately checked commits may be retained."""
import json

import pytest

from autosim.research import data_lineage, data_versions
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.data_health import lerobot_v2_profile
from autosim.research.evidence_store import capture_attempt_evidence
from tests.test_data_lineage import producer, REVIEW
from tests.test_data_health import dataset
from tests.test_data_versions import registered


def manifest_probe(output, root, *, changes=None):
    profile = lerobot_v2_profile(root, include_manifest=True)
    ident = 'b'*32
    ref = f'native_diagnostics/{ident}.json'
    log = output/f'native_diagnostics/{ident}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(profile))
    receipt = dict(data_profile_generator='autosim_lerobot_v2_readonly_manifest_v1',
        data_profile=profile, gpu_access=False, repository_source_stable=True,
        data_profile_request={'format':'lerobot_v2.1','root':str(root.relative_to(output)), 'include_manifest':True})
    receipt.update(changes or {})
    atomic_json(output/ref, receipt)
    capture_attempt_evidence(output, attempt_id=ident, log=log, receipt_ref=ref,
                            status='completed', returncode=0, termination_reason='cpu_native_diagnostic')
    return ident


def partial(tmp_path):
    repo, output, _, _ = registered(tmp_path)
    root = dataset(output)
    ident, _ = producer(output, root, status='failed')
    probe = manifest_probe(output, root)
    spec = dict(id='retained', dataset_ref='data', reason='retain committed training candidates; expert semantics independently reviewed',
        source_refs=[], asset={'kind':'collected','producer_evidence_id':ident, 'manifest_evidence_id':probe})
    return repo, output, root, ident, probe, spec


@pytest.mark.parametrize('generator', ['autosim_lerobot_v2_readonly_manifest_v1',
                                      'autosim_lerobot_v2_readonly_manifest_v2'])
def test_failed_collection_is_retained_without_rewriting_failure_or_claiming_labels(tmp_path, generator):
    repo, output, root, ident, _, spec = partial(tmp_path)
    manifest_probe(output, root, changes={'data_profile_generator': generator})
    failed = output/f'producer/{ident}/receipt.json'
    before = digest(failed)
    card = data_versions.register(output, repo, spec, REVIEW)
    assert card['asset']['producer']['partial_output'] is True
    assert card['asset']['producer']['producer_status'] == 'failed'
    assert card['asset']['profile']['funnel']['committed_episodes'] == 2
    assert card['asset']['profile']['manifest']['unknown_supervision'] == 2
    assert card['asset']['profile']['training_scope'] == 'requires_independent_registration_review'
    assert read_json(failed)['status'] == 'failed' and digest(failed) == before
    assert data_versions.resolve(output, repo, dict(dataset_root='data-version:retained'))[1][0]['id'] == 'retained'


@pytest.mark.parametrize('change', ['no_manifest','sampled','unsafe_generator','gpu','source_changed','context_changed',
                                  'changed_log','changed_receipt','changed_data','wrong_parent_scope'])
def test_partial_recovery_refuses_unverified_or_changed_native_commits(tmp_path, change):
    repo, output, root, ident, probe, spec = partial(tmp_path)
    if change == 'no_manifest': spec['asset'].pop('manifest_evidence_id')
    elif change in ('unsafe_generator','gpu','source_changed','context_changed'):
        key, value = {'unsafe_generator':('data_profile_generator','model_claim'),
                      'gpu':('gpu_access',True),'source_changed':('repository_source_stable',False),
                      'context_changed':('native_context_stable',False)}[change]
        manifest_probe(output, root, changes={key:value})
    elif change == 'sampled':
        profile = lerobot_v2_profile(root,max_episodes=1,include_manifest=True)
        manifest_probe(output, root, changes={'data_profile':profile})
    elif change == 'changed_log':
        p=output/f'native_diagnostics/{probe}.log'; p.write_text(p.read_text()+'changed')
    elif change == 'changed_receipt':
        p=output/f'producer/{ident}/receipt.json'; row=read_json(p); row['stage']='train';atomic_json(p,row)
    elif change == 'changed_data': (root/'frames/0.parquet').write_bytes(b'changed')
    else: spec['asset']['producer_scope']='budget_parent'
    with pytest.raises((ValueError, OSError)):
        data_versions.register(output, repo, spec, REVIEW)
    assert not (output/'data_versions/retained.json').exists()


def test_recovery_manifest_stays_bound_after_registration(tmp_path):
    repo, output, _, _, probe, spec = partial(tmp_path)
    data_versions.register(output, repo, spec, REVIEW)
    path = output/f'native_diagnostics/{probe}.json'
    value = read_json(path); value['data_profile']['counts']['parquet_frames']=999
    atomic_json(path,value)
    with pytest.raises(ValueError):
        data_versions.resolve(output, repo, dict(dataset_root='data-version:retained'))


def test_completed_native_collector_can_use_separate_manifest_without_log_patch(tmp_path):
    repo, output, root, _, probe, spec = partial(tmp_path)
    ident, _ = producer(output, root, status='completed', ident='c'*32, profile={})
    spec['asset']['producer_evidence_id'] = ident
    before = digest(output/f'producer/{ident}/receipt.json')
    card = data_versions.register(output, repo, spec, REVIEW)
    assert card['asset']['producer']['partial_output'] is False
    assert card['asset']['producer']['native_manifest'] is True
    assert card['asset']['profile_witness']['evidence_id'] == probe
    assert digest(output/f'producer/{ident}/receipt.json') == before
