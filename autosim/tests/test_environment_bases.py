"""Prevalidated families are useful evidence, never automatic benchmark readiness."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from autosim.research import environment_bases as bases, environment_pool as pool, provision
from autosim.research.common import atomic_json, atomic_text, digest, read_json
from autosim.research.evidence_store import capture_attempt_evidence


@pytest.fixture
def base(tmp_path):
    prefix = tmp_path / 'base'
    subprocess.run([sys.executable, '-I', '-S', '-m', 'venv', '--without-pip', str(prefix)], check=True)
    site = next((prefix / 'lib').glob('python*/site-packages'))
    (site / 'dangerous.pth').write_text(f"import pathlib; pathlib.Path({str(tmp_path / 'HOOK_EXECUTED')!r}).touch()\n")
    return prefix


def mocked_probe(monkeypatch, *, ok=True, leased=True):
    monkeypatch.setattr(provision, 'platform_facts', lambda **k: {'os': 'Linux', 'gpus': ['test-driver-uuid']})

    def run(command, **kwargs):
        output = kwargs['output'].parent
        spec = bases.probe_spec('python-toolchain', pool.describe(Path(read_json(output / 'env/overlay.json')['base_prefix'])))
        identity = 'a' * 32
        log = output / 'probe.log'
        atomic_text(log, '$ ' + command + '\n' + bases.RESULT + json.dumps({'capabilities': spec['capabilities']}) + '\n')
        ref = 'receipt.json'
        receipt = {'ok': ok, 'returncode': 0 if ok else 1, 'command': command,
                   'gpu_access': leased, 'gpu_lease_id': 'lease' if leased else None,
                   **capture_attempt_evidence(output, attempt_id=identity, log=log, receipt_ref=ref,
                       status='completed' if ok else 'failed', returncode=0 if ok else 1,
                       termination_reason='test_probe')}
        atomic_json(output / ref, receipt)
        return receipt

    monkeypatch.setattr(provision, 'run', run)


def verified(base, tmp_path, monkeypatch):
    mocked_probe(monkeypatch)
    output = tmp_path / 'check'
    result = bases.verify(base, profile='python-toolchain', output=output, store=tmp_path / 'store')
    assert result['status'] == 'verified', result
    return output, result


def base_bytes(base):
    return {str(path.relative_to(base)): (str(path.readlink()) if path.is_symlink() else digest(path))
            for path in base.rglob('*') if path.is_file() or path.is_symlink()}


def test_real_cpu_check_no_download_no_old_hook_no_base_mutation(base, tmp_path):
    if not shutil.which('bwrap'):
        pytest.skip('native boundary requires bubblewrap')
    before = base_bytes(base)
    result = bases.verify(base, profile='python-toolchain', output=tmp_path / 'check', store=tmp_path / 'store')
    assert result['status'] == 'verified', result
    assert result['capabilities'] == ['local_io', 'python_subprocess']
    assert not (tmp_path / 'HOOK_EXECUTED').exists()
    assert base_bytes(base) == before
    assert not (tmp_path / 'check/task_gpu_budget.json').exists()
    assert not (tmp_path / 'check/environment.json').exists()  # no task readiness forged
    assert not (tmp_path / 'check/agent').exists()  # no LLM bill
    assert result['wall_budget']['wall_seconds'] == 180


def test_venv_registration_is_allowed_for_readonly_overlay(base, tmp_path):
    before = base_bytes(base)
    row = pool.register_base(tmp_path / 'store', base)
    assert not row['cloneable']
    view = pool.short_catalog(pool.discover(tmp_path / 'store', prefixes=[]))[0]
    assert view['overlay_reusable'] and view['base_verification']['status'] == 'unverified'
    assert base_bytes(base) == before
    assert str(base) not in json.dumps(view)


@pytest.mark.parametrize('profile', ['torch-cuda', 'mujoco-headless'])
def test_gpu_profiles_require_explicit_short_budget_before_creation(base, tmp_path, profile):
    with pytest.raises(ValueError, match='explicit --gpu-seconds'):
        bases.verify(base, profile=profile, output=tmp_path / 'check', store=tmp_path / 'store')
    assert not (tmp_path / 'check').exists()


@pytest.mark.parametrize('wall,gpu', [(float('nan'), 0), (float('inf'), 0), (0, 0), (601, 0), (True, 0), (180, 601), (180, -1)])
def test_short_budget_bounds(base, tmp_path, wall, gpu):
    with pytest.raises(ValueError):
        bases.verify(base, profile='python-toolchain', output=tmp_path / 'check', store=tmp_path / 'store', wall_seconds=wall, gpu_seconds=gpu)


def test_existing_output_is_never_overwritten(base, tmp_path):
    output = tmp_path / 'check'
    output.mkdir()
    sentinel = output / 'keep'
    sentinel.write_text('existing')
    with pytest.raises(ValueError, match='fresh output'):
        bases.verify(base, profile='python-toolchain', output=output, store=tmp_path / 'store')
    assert sentinel.read_text() == 'existing'


def test_failed_check_is_evidence_not_registered_success(base, tmp_path, monkeypatch):
    mocked_probe(monkeypatch, ok=False)
    result = bases.verify(base, profile='python-toolchain', output=tmp_path / 'check', store=tmp_path / 'store')
    assert result['status'] == 'failed' and result['evidence_id']
    assert not list((tmp_path / 'store/registered').glob('*'))
    assert read_json(tmp_path / 'check/base_check.json')['status'] == 'failed'


@pytest.mark.parametrize('target', ['probe.log', 'receipt.json', 'checkout/probe.py', 'base_verification.json'])
def test_tampered_proof_is_stale_not_readiness(base, tmp_path, monkeypatch, target):
    output, _ = verified(base, tmp_path, monkeypatch)
    (output / target).write_text('tampered')
    rows = pool.discover(tmp_path / 'store', prefixes=[], machine=provision.platform_facts())
    assert rows[0]['base_verification']['status'] == 'stale'
    assert not rows[0]['base_verification']['capabilities']


def test_changed_base_driver_or_probe_invalidates_certificate(base, tmp_path, monkeypatch):
    output, _ = verified(base, tmp_path, monkeypatch)
    proof = [str(output / 'base_verification.json')]
    assert bases.verification_view(base, proof, {'gpus': ['changed']})['status'] == 'stale'
    old_code = bases.PYTHON
    monkeypatch.setattr(bases, 'PYTHON', old_code + '\n# revised probe\n')
    assert bases.verification_view(base, proof, provision.platform_facts())['status'] == 'stale'
    monkeypatch.setattr(bases, 'PYTHON', old_code)
    site = next((base / 'lib').glob('python*/site-packages'))
    (site / 'dangerous.pth').write_text('# changed hook\n')
    assert bases.verification_view(base, proof, provision.platform_facts())['status'] == 'stale'


def test_machine_or_log_cannot_be_asserted_verified_by_registration(base, tmp_path):
    proof = tmp_path / 'fake.json'
    atomic_json(proof, {'status': 'verified'})
    with pytest.raises(ValueError, match='stale, failed or missing'):
        pool.register_base(tmp_path / 'store', base, proof=proof)


def test_verified_candidates_are_short_and_only_recommended(base, tmp_path, monkeypatch):
    output, _ = verified(base, tmp_path, monkeypatch)
    run = tmp_path / 'run'
    pool.configure(run, tmp_path / 'store')
    rows = pool.catalog(run, {}, provision.platform_facts())
    row = next(row for row in pool.short_catalog(rows) if row['origin'] == 'registered_base')
    assert row['base_verification']['checks'][0]['profile'] == 'python-toolchain'
    assert str(base) not in json.dumps(row) and str(output) not in json.dumps(row)
    assert row['readiness'] == 'unverified'  # task still needs probes
    assert 'YOUR decision' in provision.PLAN_SYSTEM


def test_new_base_registration_refreshes_existing_short_catalog(base, tmp_path, monkeypatch):
    monkeypatch.setattr(provision, 'platform_facts', lambda **k: {})
    run, store = tmp_path / 'run', tmp_path / 'store'
    pool.configure(run, store)
    pool.catalog(run, {}, {})
    pool.register_base(store, base, label='new-base')
    assert any(row.get('label') == 'new-base' for row in pool.candidate_view(run))


def test_current_proof_tampering_detected_even_with_cached_catalog(base, tmp_path, monkeypatch):
    output, _ = verified(base, tmp_path, monkeypatch)
    run = tmp_path / 'run'
    pool.configure(run, tmp_path / 'store')
    pool.catalog(run, {}, provision.platform_facts())
    (output / 'probe.log').write_text('changed')
    row = next(row for row in pool.candidate_view(run) if row['origin'] == 'registered_base')
    assert row['base_verification']['status'] == 'stale'


def test_version_variants_and_all_probe_templates_compile():
    for profile, version in [('python-toolchain', ''), ('torch-cuda', ''),
                             ('mujoco-headless', ''), ('sapien-headless', '2.2.2'), ('sapien-headless', '3.0.2')]:
        spec = bases.probe_spec(profile, {'packages': {'sapien': version}})
        compile(spec['code'], '<base-probe>', 'exec')
        assert 'LIBERO' not in spec['code'] and 'RoboSyn' not in spec['code'] and 'RoboTwin' not in spec['code']
    with pytest.raises(ValueError, match='supported SAPIEN'):
        bases.probe_spec('sapien-headless', {'packages': {'sapien': '4.0'}})


def test_run_overlays_are_not_registered_as_independent_bases(base, tmp_path):
    atomic_json(base / 'overlay.json', {})
    with pytest.raises(ValueError, match='underlying base'):
        pool.register_base(tmp_path / 'store', base)
    with pytest.raises(ValueError, match='underlying base'):
        bases.verify(base, profile='python-toolchain', output=tmp_path / 'check', store=tmp_path / 'store')


def test_successful_overlay_delta_is_offered_on_original_base(base, tmp_path, monkeypatch):
    monkeypatch.setattr(provision, 'platform_facts', lambda **k: {})
    run = tmp_path / 'run'
    pool.configure(run, tmp_path / 'store')
    pool.register_base(tmp_path / 'store', base)
    prefix = run / 'env'
    (prefix / 'bin').mkdir(parents=True)
    (prefix / 'bin/python').write_text('fixture')
    atomic_json(prefix / 'overlay.json', {'base_fingerprint': pool.describe(base)['fingerprint']})
    atomic_json(run / 'environment.json', {'verdict': {'passed': True}, 'interpreter': str(prefix / 'bin/python')})
    row = pool.describe(base)
    row['portable_packages'] = {'small-gap': '1.2'}
    assert pool.publish_template(run, row=row, manifests={}, machine={})['status'] == 'published'
    rows = pool.catalog(run, {}, {})
    original = next(row for row in rows if row.get('prefix') == str(base))
    assert original['successful_incremental_pins'] == {'small-gap': '1.2'}
    assert original['prior_setup_scope'].startswith('prior_declared_consumers_only')


def test_cli_verification_registration_and_argument_errors(base, tmp_path, capsys):
    assert pool.main(['register', '--prefix', str(base), '--store', str(tmp_path / 'store')]) == 0
    assert json.loads(capsys.readouterr().out)['readiness'] == 'unverified'
    with pytest.raises(SystemExit) as error:
        pool.main(['verify', '--prefix', str(base)])
    assert error.value.code == 2


def test_recorder_explains_base_evidence_scope_in_chinese(tmp_path):
    from autosim.research.recorder import render, make_snapshot
    from autosim.research.run_record import build_report_view
    atomic_json(tmp_path / 'environment_pool.json', {'enabled': True})
    atomic_json(tmp_path / 'environment_selection.json', {'base_verification': {
        'status': 'verified', 'capabilities': ['cuda_compute', 'offscreen_frame']}})
    snapshot = make_snapshot(tmp_path, build_report_view(tmp_path, 'derived', status='running'), {})
    text = render(tmp_path, snapshot, {})
    assert 'GPU 运算、离屏画面' in text
    assert '不代表本仓库任务、数据采集或训练评测已经通过' in text


def test_default_publication_does_not_copy_a_large_environment(base, tmp_path, monkeypatch):
    monkeypatch.setattr(provision, 'platform_facts', lambda **k: {})
    run = tmp_path / 'run'
    repo = run / 'checkout'
    repo.mkdir(parents=True)
    pool.configure(run, tmp_path / 'store')
    def forbidden(*a, **kw):
        raise AssertionError('large snapshot was not authorized')
    monkeypatch.setattr(pool, 'publish_snapshot', forbidden)
    monkeypatch.setattr(pool, 'publish_wheels', lambda *a, **kw: {})
    result = provision._finish(run, repo, pool.describe(base)['python'], [], [], ['native probe'],
                              {'passed': True}, interpreter=base / 'bin/python', manifests={})
    assert result['verdict']['passed']
    saved = read_json(run / 'environment_cache_publication.json')['snapshot']
    assert saved['status'] == 'template_only' and saved['portable_template']['status'] == 'published'
    assert not list((tmp_path / 'store/snapshots').iterdir())


def test_large_snapshot_publication_requires_explicit_policy(base, tmp_path, monkeypatch):
    monkeypatch.setattr(provision, 'platform_facts', lambda **k: {})
    run = tmp_path / 'run'
    repo = run / 'checkout'
    repo.mkdir(parents=True)
    pool.configure(run, tmp_path / 'store', publish_snapshots=True)
    calls = []
    monkeypatch.setattr(pool, 'publish_snapshot', lambda *a, **kw: calls.append(kw) or {'status': 'published'})
    monkeypatch.setattr(pool, 'publish_wheels', lambda *a, **kw: {})
    provision._finish(run, repo, pool.describe(base)['python'], [], [], [], {'passed': True},
                      interpreter=base / 'bin/python', manifests={})
    assert len(calls) == 1
    with pytest.raises(ValueError, match='snapshot policy changed'):
        pool.configure(run, tmp_path / 'store', publish_snapshots=False)


def test_failed_driver_query_is_not_a_gpu_model(monkeypatch):
    def query(argv, **kw):
        if argv[0] == 'nvidia-smi':
            return subprocess.CompletedProcess(argv, 9, 'NVIDIA driver inaccessible', '')
        return subprocess.CompletedProcess(argv, 0, 'Cuda compilation tools, release 12.8', '')
    monkeypatch.setattr(provision, 'bounded_run', query)
    facts = provision.platform_facts()
    assert facts['gpus'] == [] and 'not proof of absent hardware' in facts['gpu_query_error']


def test_cpu_certificate_does_not_depend_on_gpu_visibility(base, tmp_path, monkeypatch):
    output, _ = verified(base, tmp_path, monkeypatch)
    machine = provision.platform_facts()
    machine.update(gpus=[], gpu_query_error='driver inaccessible')
    view = bases.verification_view(base, [str(output / 'base_verification.json')], machine)
    assert view['status'] == 'verified' and view['capabilities'] == ['local_io', 'python_subprocess']


def test_bounded_torch_table_is_forwarded_without_unbounded_discovery(monkeypatch):
    from autosim.research import compute_decision
    captured = {}
    def discover(**kwargs):
        captured.update(kwargs)
        return {'gpus': [], 'allowed': []}
    monkeypatch.setattr(compute_decision, 'discover', discover)
    decision = compute_decision.decide(torch_rows=[], prefer='cpu')
    assert captured['torch_rows'] == [] and decision.device == 'cpu'


def test_check_report_links_frame_only_on_pass(tmp_path):
    (tmp_path / 'checkout').mkdir()
    (tmp_path / 'checkout/frame.png').write_bytes(b'fixture png')
    result = {'profile': 'mujoco-headless', 'status': 'failed', 'reason': 'render error',
              'wall_budget': {'elapsed_wall_seconds': 1}, 'evidence_id': 'b' * 32}
    bases.write_report(tmp_path, result)
    assert '实际渲染画面' not in (tmp_path / 'CHECK.md').read_text()
    result.update(status='verified', capabilities=bases.CAPABILITIES['mujoco-headless'])
    bases.write_report(tmp_path, result)
    text = (tmp_path / 'CHECK.md').read_text()
    assert '实际渲染画面' in text and '不是目标仓库的实验结果' in text


def test_prevalidated_base_enters_real_incremental_provision_without_installs(base, tmp_path, monkeypatch):
    if not shutil.which('bwrap'):
        pytest.skip('native boundary requires bubblewrap')
    mocked_probe(monkeypatch)
    # This phase's certificate is a fixture; consumers below use the real executor.
    result = bases.verify(base, profile='python-toolchain', output=tmp_path / 'check', store=tmp_path / 'store')
    assert result['status'] == 'verified'
    monkeypatch.undo()
    from autosim.research.compute_decision import ComputeDecision
    run, repo = tmp_path / 'run', tmp_path / 'run/checkout'
    repo.mkdir(parents=True)
    pool.configure(run, tmp_path / 'store')
    # Keep test hardware facts consistent with the fixture certificate.
    monkeypatch.setattr(provision, 'platform_facts', lambda **k: {'os': 'Linux', 'gpus': ['test-driver-uuid']})
    row = pool.describe(base)
    probe = '{python} -c "print(73)"'
    class Agent:
        def chat_with_metadata(self, system, payload, **kw):
            assert row['id'] in payload and 'python_subprocess' in payload
            assert str(base) not in payload
            return json.dumps({'python': row['python'], 'commands': [], 'probes': [probe],
                'resource_requests': [{'command': probe, 'resource': 'cpu', 'why': 'native interpreter consumer'}],
                'base_environment_id': row['id'], 'base_environment_mode': 'overlay',
                'environment_selection_reason': 'matching Python and verified interpreter; current native probe required',
                'reasoning': 'reuse first, no dependency gaps'}), {}
    monkeypatch.setattr(provision, 'load_recipe', lambda **kw: None)
    result = provision.build(repo, client=Agent(), prefix=run / 'env', output=run,
                             manifests={}, assets={}, max_rounds=1, step_timeout=20,
                             compute=ComputeDecision('cpu', 0))
    assert result['verdict']['passed'], result
    record = read_json(run / 'environment_selection.json')
    assert record['base_verification']['status'] == 'verified'
    assert record['mode'] == 'overlay' and 'current native probe required' in record['reason']
    assert not [receipt for receipt in result['record'] if 'pip install' in receipt.get('command', '')]
