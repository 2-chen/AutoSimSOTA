"""Reuse dependencies without executing inherited hooks or exposing writable bases."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autosim.research import environment_overlay as overlay, environment_pool as pool, provision
from autosim.research.common import atomic_json
from autosim.research.native_context import publish_context, load_context


@pytest.fixture
def fixture(tmp_path):
    base = tmp_path/'base'
    subprocess.run([sys.executable, '-I', '-S', '-m', 'venv', '--without-pip', str(base)], check=True)
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    site = base/'lib'/f'python{version}'/'site-packages'
    (site/'dependency.py').write_text('VALUE = 73\n')
    (site/'malicious.pth').write_text(f"import pathlib; pathlib.Path({str(tmp_path/'HOOK_RAN')!r}).touch()\n")
    (site/'sitecustomize').mkdir()
    (site/'sitecustomize/__init__.py').write_text(
        f"from pathlib import Path; Path({str(tmp_path/'HOOK_RAN')!r}).touch()\n")
    old = tmp_path/'old_source'; old.mkdir()
    (old/'__init__.py').write_text("VALUE = 'old'\n")
    addon = tmp_path/'extra_source'; addon.mkdir()
    (addon/'__init__.py').write_text('VALUE = 19\n')
    (site/'__editable___task_finder.py').write_text(
        f'MAPPING = {{"task": {str(old)!r}, "addon": {str(addon)!r}}}\nraise RuntimeError("never execute")\n')
    (site/'__editable__task.pth').write_text('import __editable___task_finder\n')
    dist = site/'task-1.dist-info'; dist.mkdir()
    (dist/'METADATA').write_text('Name: task\nVersion: 1\n')
    (dist/'top_level.txt').write_text('task\n')
    atomic_json(dist/'direct_url.json', {'url':old.as_uri(), 'dir_info':{'editable':True}})
    output = tmp_path/'run'; repo = output/'checkout'; repo.mkdir(parents=True)
    (repo/'task').mkdir(); (repo/'task/__init__.py').write_text("VALUE = 'current'\n")
    return base, site, output, repo, version


def make(fixture, *, ids=None):
    base, site, output, repo, version = fixture
    chosen = pool.describe(base)
    view = overlay.inspect(base, version, repo=repo)
    bindings = ids if ids is not None else [b['id'] for b in view['bindings'] if b['module'] == 'addon']
    return overlay.create(chosen, prefix=output/'env', output=output, repo=repo, source_binding_ids=bindings)


def test_overlay_metadata_observation_is_single_read_and_not_stale(fixture,monkeypatch):
    import importlib.metadata as metadata
    base,site,_,repo,version=fixture
    original=metadata.Distribution.metadata.fget
    reads=[]
    def observed(distribution):
        reads.append(str(distribution._path))
        return original(distribution)
    monkeypatch.setattr(metadata.Distribution,'metadata',property(observed))
    first=overlay.inspect(base,version,repo=repo)
    assert len(reads)==1
    (site/'task-1.dist-info/METADATA').write_text('Name: task\nVersion: 2\n')
    second=overlay.inspect(base,version,repo=repo)
    assert len(reads)==2 and second['fingerprint']!=first['fingerprint']


def test_reuse_rebind_and_never_execute_old_hooks(fixture):
    base, site, output, repo, version = fixture
    before = {p.name:p.read_bytes() for p in site.iterdir() if p.is_file()}
    result = make(fixture)
    text = subprocess.check_output([result['interpreter'], '-c',
        'import dependency,task,addon,sys; print(dependency.VALUE,task.VALUE,addon.VALUE); print(sys.prefix)'], text=True)
    assert text.splitlines() == ['73 current 19', str(output/'env')]
    assert not (base.parent/'HOOK_RAN').exists()
    assert before == {p.name:p.read_bytes() for p in site.iterdir() if p.is_file()}
    assert not (output/'env/dependencies/task-1.dist-info').exists()
    assert all(b['origin'] == 'checkout' for b in overlay.inspect(base,version,repo=repo)['bindings'] if b['module']=='task')
    assert not list((output/'env/dependencies').glob('*.pth'))
    (output/'env/lib'/f'python{version}'/'site-packages/dependency.py').write_text('VALUE = 99\n')
    assert subprocess.check_output([result['interpreter'], '-c', 'import dependency; print(dependency.VALUE)'], text=True).strip() == '99'
    assert (site/'dependency.py').read_text() == 'VALUE = 73\n'


def test_external_source_requires_selection(fixture):
    result = make(fixture, ids=[])
    attempt = subprocess.run([result['interpreter'], '-c', 'import addon'], capture_output=True, text=True)
    assert attempt.returncode != 0 and 'ModuleNotFoundError' in attempt.stderr


def test_frozen_checkout_import_rebinding_preserves_original_prefix_and_local_overrides(fixture):
    from autosim.research.workspace_snapshot import create
    from autosim.research.common import isolated_argv, digest
    base, site, output, repo, version = fixture
    result = make(fixture)
    prefix = output / 'env'
    (prefix / 'lib' / f'python{version}' / 'site-packages/dependency.py').write_text('VALUE = 99\n')
    source = base.parent / 'frozen_source'; (source / 'task').mkdir(parents=True)
    (source / 'task/__init__.py').write_text("VALUE = 'frozen'\n")
    fresh = output / 'confirmation_sources/baseline/checkout'
    create(source, fresh)
    original = prefix / 'lib' / f'python{version}' / 'site-packages/autosim_overlay_sources.py'
    before = digest(original)
    binding = overlay.isolated_source_bindings(Path(result['interpreter']), output=output,
                                               original_repo=repo, repo=fresh)
    checked = overlay.verify_isolated_source_bindings(Path(result['interpreter']),
        output=output, repo=fresh, binding=binding)
    assert checked['status'] == 'verified' and checked['gpu_access'] is False
    argv = isolated_argv([result['interpreter'], '-c',
        'import task,dependency,addon; print(task.VALUE,dependency.VALUE,addon.VALUE)'],
        repo=fresh, output=output, allow_gpu=False, import_binding_override=binding)
    text = subprocess.check_output(argv, cwd=fresh, text=True)
    assert text.strip() == 'frozen 99 19'
    nested = fresh / 'task'
    nested_argv = isolated_argv([result['interpreter'], '-c', 'import pathlib; print(pathlib.Path.cwd().name)'],
        repo=fresh, output=output, allow_gpu=False, import_binding_override=binding,
        native_working_directory=nested)
    assert subprocess.check_output(nested_argv, cwd=nested, text=True).strip() == 'task'
    assert digest(original) == before
    assert subprocess.check_output([result['interpreter'], '-c', 'import task; print(task.VALUE)'],
                                  cwd=repo, text=True).strip() == 'current'
    Path(binding['source']).write_text('# corrupted\n')
    with pytest.raises(ValueError, match='import binding'):
        isolated_argv([result['interpreter'], '-c', 'pass'], repo=fresh, output=output,
                      import_binding_override=binding)


def test_unknown_binding_refused_before_creation(fixture):
    with pytest.raises(ValueError, match='binding ID'):
        make(fixture, ids=['0'*32])
    assert not (fixture[2]/'env').exists()


def test_duplicate_module_and_metadata_change_refused(fixture):
    base, site, output, repo, version = fixture
    options = overlay.inspect(base, version, repo=repo)['bindings']
    with pytest.raises(ValueError, match='one source'):
        make(fixture, ids=[next(b['id'] for b in options if b['module']=='task')]*2)
    chosen = pool.describe(base)
    chosen['overlay_fingerprint'] = overlay.inspect(base, version, repo=repo)['fingerprint']
    (site/'changed.pth').write_text('# changed\n')
    with pytest.raises(ValueError, match='metadata changed'):
        overlay.create(chosen, prefix=output/'env', output=output, repo=repo, source_binding_ids=[])


def test_native_context_explicit_mounts_and_detects_base_change(fixture):
    base, site, output, repo, version = fixture
    result = make(fixture)
    row = publish_context(output, repo, Path(result['interpreter']), {})
    assert str(base) in row['dependency_roots']
    assert str(base.parent/'extra_source') in row['dependency_roots']
    assert load_context(output, repo)['identity'] == row['identity']
    # Persistence must survive another process/hash seed, not just a same-process set.
    subprocess.run([sys.executable, '-c', 'from pathlib import Path; from autosim.research.native_context import load_context; '
        f'load_context(Path({str(output)!r}),Path({str(repo)!r}))'], check=True)
    (site/'changed.pth').write_text('# changed\n')
    with pytest.raises(ValueError, match='metadata changed'):
        load_context(output, repo)


def test_overlay_plan_can_probe_without_install_and_select_sources(fixture):
    base, site, output, repo, version = fixture
    row = pool.describe(base)
    view = overlay.inspect(base, version, repo=repo)
    row.update(overlay_reusable=True, source_binding_options=overlay.binding_catalog(view))
    candidate = pool.short_catalog([row])[0]
    class Agent:
        def chat_with_metadata(self, system, payload, **kwargs):
            assert 'source_binding_options' in payload
            return json.dumps({'python':version, 'commands':[], 'probes':["{python} -c 'import task,dependency'"],
                'base_environment_id':row['id'], 'base_environment_mode':'overlay',
                'environment_selection_reason':'installed dependencies match; replace stale editable imports',
                'source_binding_ids':[]}), {}
    plan = provision.plan(Agent(), repo, manifests={}, assets={'environment_candidates':[candidate]})
    assert plan['commands'] == []
    assert plan['probes']


def test_overlay_creation_has_sealed_evidence(fixture):
    base, site, output, repo, version = fixture
    result = provision._overlay_selected_base(pool.describe(base), prefix=output/'env',
        output=output, repo=repo, timeout=60, source_binding_ids=[])
    assert result['ok']
    assert (output/result['evidence_ref']).is_file()


def test_primary_wheel_not_hidden_by_secondary_local_build(fixture, monkeypatch):
    base, site, output, repo, version = fixture
    secondary = base.parent/'secondary'; secondary.mkdir()
    for root, number in ((site, '2'), (secondary, '1')):
        dist = root/f'dependency-{number}.dist-info'; dist.mkdir()
        (dist/'METADATA').write_text(f'Name: dependency\nVersion: {number}\n')
        (dist/'top_level.txt').write_text('dependency\n')
    atomic_json(secondary/'dependency-1.dist-info/direct_url.json',
        {'url':'file:///build/dependency', 'dir_info':{}})
    (secondary/'dependency.py').write_text('VALUE = 0\n')
    monkeypatch.setattr(overlay, 'sites', lambda *a:[site,secondary])
    result = make(fixture)
    text = subprocess.check_output([result['interpreter'], '-c',
        'import dependency,importlib.metadata as m; print(dependency.VALUE,m.version("dependency"))'], text=True)
    assert text.strip() == '73 2'
    assert not (output/'env/dependencies/dependency-1.dist-info').exists()


def test_native_diagnostic_keeps_borrowed_dependencies_readonly(fixture):
    from autosim.research.agent_runtime import _McpExecutor
    base, site, output, repo, version = fixture
    result = make(fixture)
    atomic_json(output/'agent_fixture.json', {'schema_version':1,'workspace':str(repo)})
    publish_context(output, repo, Path(result['interpreter']), {})
    code = ('import dependency,task,addon\nfrom pathlib import Path\n'
            f'p=Path({str(site/"dependency.py")!r})\n'
            'try:\n p.write_text("corrupt")\nexcept OSError:\n print("readonly",dependency.VALUE,task.VALUE,addon.VALUE)\n'
            'else:\n raise RuntimeError("base unexpectedly writable")\n')
    receipt = _McpExecutor(workspace=repo,output=output).native_probe(
        {'code':code,'purpose':'verify readonly dependency bindings'})
    assert receipt['returncode'] == 0, receipt
    assert receipt['stdout'].strip() == 'readonly 73 current 19'
    assert (site/'dependency.py').read_text() == 'VALUE = 73\n'


def test_provision_overlay_probes_first_without_install(fixture, monkeypatch):
    import shlex
    base, site, output, repo, version = fixture
    pool.configure(output, base.parent/'store')
    row = pool.describe(base)
    row.update(overlay_reusable=True, source_binding_options=overlay.binding_catalog(
        overlay.inspect(base,version,repo=repo)))
    class Agent:
        def chat_with_metadata(self, *a, **kw):
            return json.dumps({'python':version,'commands':[],
                'probes':["{python} -c 'import dependency,task; assert dependency.VALUE==73; assert task.VALUE==\"current\"'"],
                'base_environment_id':row['id'],'base_environment_mode':'overlay',
                'environment_selection_reason':'probe existing dependencies before install'}), {}
    monkeypatch.setattr(pool,'catalog',lambda *a:[row])
    monkeypatch.setattr(provision,'load_recipe',lambda **kw:None)
    monkeypatch.setattr(provision,'run',lambda *a,**kw:pytest.fail('unexpected installation'))
    def probe(command, **kwargs):
        for name,value in kwargs['values'].items():
            command=command.replace('{'+name+'}',value)
        attempt=subprocess.run(shlex.split(command),capture_output=True,text=True)
        return {'ok':attempt.returncode==0,'command':command,'seconds':0,'excerpt':attempt.stderr}
    monkeypatch.setattr(provision,'probe_environment',probe)
    monkeypatch.setattr(provision,'_finish',lambda *a,**kw:{'verdict':a[6]})
    result = provision.build(repo,client=Agent(),prefix=output/'env',output=output,
        manifests={},assets={},max_rounds=1)
    assert result['verdict']['passed'], result
    assert (output/'env/overlay.json').exists()


def test_pip_override_is_local_and_does_not_uninstall_borrowed_package(fixture):
    import shlex
    import zipfile
    from autosim.research.common import run_local_environment
    base, site, output, repo, version = fixture
    dist = site/'dependency-1.dist-info'; dist.mkdir()
    (dist/'METADATA').write_text('Name: dependency\nVersion: 1\n')
    (dist/'RECORD').write_text('dependency.py,,\ndependency-1.dist-info/METADATA,,\n')
    result = make(fixture)
    python = result['interpreter']
    subprocess.run([python,'-m','ensurepip','--default-pip'],check=True,capture_output=True,timeout=60)
    code = ('from pip._internal.metadata import get_default_environment; '
            'd=get_default_environment().get_distribution("dependency"); assert not d.local')
    subprocess.run([python,'-c',code],check=True)
    wheel = output/'dependency-2-py3-none-any.whl'
    with zipfile.ZipFile(wheel,'w') as package:
        files = {'dependency.py':'VALUE = 99\n',
            'dependency-2.dist-info/METADATA':'Metadata-Version: 2.1\nName: dependency\nVersion: 2\n',
            'dependency-2.dist-info/WHEEL':'Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
        files['dependency-2.dist-info/RECORD']=''.join(name+',,\n' for name in files)+'dependency-2.dist-info/RECORD,,\n'
        for name,text in files.items():
            package.writestr(name,text)
    publish_context(output,repo,Path(python),{})
    command = shlex.join([python,'-m','pip','install','--no-index','--no-deps',str(wheel)])
    receipt = provision.run(command,env=run_local_environment(output),cwd=repo,
        output=output/'pip-override.log',timeout=60)
    assert receipt['ok'], receipt
    text = subprocess.check_output([python,'-c','import dependency,importlib.metadata as m; print(dependency.VALUE,m.version("dependency"))'],text=True)
    assert text.strip() == '99 2'
    assert (site/'dependency.py').read_text() == 'VALUE = 73\n'
    assert (dist/'METADATA').read_text() == 'Name: dependency\nVersion: 1\n'


def test_interrupted_creation_recovers_same_prefix_without_recreating_venv(fixture, monkeypatch):
    write = overlay.atomic_text
    def crash(path, text):
        if path.name == 'autosim_overlay_sources.py':
            raise RuntimeError('simulated process loss')
        return write(path, text)
    monkeypatch.setattr(overlay, 'atomic_text', crash)
    with pytest.raises(RuntimeError, match='process loss'):
        make(fixture)
    assert json.loads((fixture[2]/'env/overlay_creation.json').read_text())['status'] == 'venv_ready'
    monkeypatch.setattr(overlay, 'atomic_text', write)
    monkeypatch.setattr(overlay.subprocess, 'run', lambda *a, **kw: pytest.fail('venv was recreated'))
    result = make(fixture)
    assert result['ok']
    assert make(fixture)['recovered']


def test_unowned_existing_prefix_is_never_overwritten(fixture):
    prefix = fixture[2]/'env'; prefix.mkdir()
    sentinel = prefix/'user_data'; sentinel.write_text('keep')
    with pytest.raises(ValueError):
        make(fixture)
    assert sentinel.read_text() == 'keep'


def test_binding_order_has_canonical_identity(fixture):
    base, site, output, repo, version = fixture
    ids = [b['id'] for b in overlay.inspect(base, version, repo=repo)['bindings']]
    first = make(fixture, ids=ids)
    second = make(fixture, ids=list(reversed(ids)))
    assert first['overlay']['definition_identity'] == second['overlay']['definition_identity']
    assert second['recovered']


def test_refresh_preserves_local_install_and_changes_context_identity(fixture):
    base, site, output, repo, version = fixture
    result = make(fixture)
    first = publish_context(output, repo, Path(result['interpreter']), {})
    local = output/'env/lib'/f'python{version}'/'site-packages/dependency.py'
    local.write_text('VALUE = 99\n')
    (site/'task-1.dist-info/METADATA').write_text('Name: task\nVersion: 2\n')
    with pytest.raises(ValueError, match='metadata changed'):
        load_context(output, repo)
    refreshed = overlay.refresh_selected(output/'env', output, repo)
    second = publish_context(output, repo, Path(result['interpreter']), {})
    assert first['identity'] != second['identity']
    assert Path(result['overlay']['dependency_view']).is_dir()
    assert Path(refreshed['overlay']['dependency_view']).is_dir()
    assert list((output/'env/overlay_history').glob('*.json'))
    assert local.read_text() == 'VALUE = 99\n'
    assert subprocess.check_output([result['interpreter'], '-c',
        'import dependency,addon,task; print(dependency.VALUE,addon.VALUE,task.VALUE)'], text=True).strip() == '99 19 current'


def test_nested_dependency_requires_explicit_binding_and_skips_hooks(fixture):
    base, site, output, repo, version = fixture
    nested = site/'arbitrary.prefix/lib'/f'python{version}'/'site-packages'
    nested.mkdir(parents=True)
    (nested/'native_dep.py').write_text('VALUE=42\n')
    (nested/'sitecustomize.py').write_text('raise RuntimeError("must not execute")\n')
    (nested/'__pycache__').mkdir()
    (nested/'unsafe.pth').write_text('import nonexistent_hook\n')
    dist = nested/'native_dep-1.dist-info'; dist.mkdir()
    (dist/'METADATA').write_text('Name: native-dep\nVersion: 1\n')
    (dist/'top_level.txt').write_text('native_dep\n')
    options = overlay.inspect(base, version, repo=repo)['bindings']
    assert not any(b['module'] in {'sitecustomize','__pycache__'} for b in options)
    selected = next(b for b in options if b['module'] == 'native_dep')
    make(fixture, ids=[])
    assert subprocess.run([str(output/'env/bin/python'), '-c', 'import native_dep'], capture_output=True).returncode != 0
    result = overlay.create(pool.describe(base),prefix=output/'selected_env',output=output,repo=repo,
        source_binding_ids=[selected['id']])
    assert subprocess.check_output([result['interpreter'], '-c',
        'import native_dep,importlib.metadata as m; print(native_dep.VALUE,m.version("native-dep"))'],text=True).strip() == '42 1'


def test_nested_symlink_outside_base_not_offered(fixture):
    base, site, output, repo, version = fixture
    external = base.parent/'external'; external.mkdir()
    (external/'unsafe.py').write_text('VALUE=1\n')
    holder = site/'vendor/lib'/f'python{version}'; holder.mkdir(parents=True)
    (holder/'site-packages').symlink_to(external, target_is_directory=True)
    assert not any(b['module'] == 'unsafe' for b in overlay.inspect(base,version,repo=repo)['bindings'])


def test_startup_tampering_invalidates_native_context(fixture):
    result = make(fixture)
    output, repo, version = fixture[2:]
    publish_context(output, repo, Path(result['interpreter']), {})
    (output/'env/lib'/f'python{version}'/'site-packages/autosim_overlay.pth').write_text('import os\n')
    with pytest.raises(ValueError, match='startup files changed'):
        load_context(output, repo)


def provision_fixture(fixture, monkeypatch, *, bindings=()):
    """Scripted planner and real CPU consumer imports; no model, GPU or download."""
    import shlex
    base, site, output, repo, version = fixture
    pool.configure(output, base.parent/'store')
    row = pool.describe(base)
    row.update(overlay_reusable=True, source_binding_options=overlay.binding_catalog(
        overlay.inspect(base, version, repo=repo)))
    monkeypatch.setattr(pool, 'catalog', lambda *a: [row])
    monkeypatch.setattr(provision, 'load_recipe', lambda **kw: None)
    monkeypatch.setattr(provision, 'run', lambda *a, **kw: pytest.fail('unexpected installation'))
    planned = {'python':version, 'commands':[], 'probes':["{python} -c 'import dependency,task,addon'"],
        'base_environment_id':row['id'], 'base_environment_mode':'overlay',
        'environment_selection_reason':'verify installed consumers', 'source_binding_ids':list(bindings)}
    monkeypatch.setattr(provision, 'plan', lambda *a, **kw: dict(planned))
    monkeypatch.setattr(provision, 'resume', lambda *a, **kw: ([], None))
    monkeypatch.setattr(provision, 'diagnose', lambda *a, **kw: {})
    monkeypatch.setattr(pool, 'publish_wheels', lambda *a, **kw: {})
    monkeypatch.setattr(pool, 'publish_snapshot', lambda *a, **kw: {})
    def probe(command, **kwargs):
        for name, value in kwargs['values'].items():
            command = command.replace('{'+name+'}', value)
        attempt = subprocess.run(shlex.split(command), capture_output=True, text=True)
        return {'ok':attempt.returncode == 0,'returncode':attempt.returncode,
            'command':command,'seconds':0,'excerpt':attempt.stderr}
    monkeypatch.setattr(provision, 'probe_environment', probe)
    return row, planned


def test_explicit_base_reuses_supplied_prepared_overlay_and_local_delta(fixture, monkeypatch):
    base, site, output, repo, version = fixture
    prepared = make(fixture)
    addon = next(b['id'] for b in overlay.inspect(base, version, repo=repo)['bindings'] if b['module']=='addon')
    row, planned = provision_fixture(fixture, monkeypatch, bindings=[addon])
    local = output/'env/lib'/f'python{version}'/'site-packages/local_delta.py'
    local.write_text('VALUE = 123\n')
    planned['probes'] = ["{python} -c 'import dependency,task,addon,local_delta; assert local_delta.VALUE == 123'"]
    result = provision.build(repo,client=None,prefix=output/'env',output=output,
        python=prepared['interpreter'],manifests={},assets={},max_rounds=1,
        base_environment_id=row['id'],base_environment_mode='overlay',environment_selection_reason='reuse supplied base')
    assert result['verdict']['passed'], result
    assert result['interpreter'] == prepared['interpreter']
    assert local.read_text() == 'VALUE = 123\n'
    assert not list((output/'environments').glob('*/bin/python'))
    assert not (base.parent/'HOOK_RAN').exists()


def test_main_can_select_prepared_overlay_after_failed_other_prefix(fixture, monkeypatch):
    base, site, output, repo, version = fixture
    prepared = make(fixture)
    row, planned = provision_fixture(fixture, monkeypatch)
    kwargs = dict(client=None,prefix=output/'env',output=output,manifests={},assets={},max_rounds=1,
        base_environment_id=row['id'],base_environment_mode='overlay',environment_selection_reason='source backed choice')
    first = provision.build(repo,source_binding_ids=[],**kwargs)
    assert not first['verdict']['passed']
    addon = next(b['id'] for b in overlay.inspect(base,version,repo=repo)['bindings'] if b['module']=='addon')
    planned['source_binding_ids'] = [addon]
    second = provision.build(repo,reuse_environment_ref='env',**kwargs)
    assert second['verdict']['passed'], second
    assert second['interpreter'] == prepared['interpreter'] != first['interpreter']
    assert Path(first['interpreter']).is_file()
    assert len(list((output/'environments').glob('*/bin/python')))==1


def test_invalid_caller_binding_sealed_before_model_or_prefix_change(fixture, monkeypatch):
    base, site, output, repo, version = fixture
    row, planned = provision_fixture(fixture,monkeypatch)
    monkeypatch.setattr(provision,'plan',lambda *a,**kw: pytest.fail('bad caller must not buy a plan'))
    with pytest.raises(provision.EnvironmentPlanError,match='unknown source_binding_ids: mistaken-id') as caught:
        provision.build(repo,client=None,prefix=output/'env',output=output,manifests={},assets={},
            base_environment_id=row['id'],base_environment_mode='overlay',
            environment_selection_reason='select explicit SDK',source_binding_ids=['mistaken-id'])
    failure=caught.value.planning_failure
    assert failure['repair_owner']=='main_agent' and failure['retry_original_arguments'] is False
    assert failure['native_operation_status']=='not_started'
    from autosim.research.evidence_store import read_attempt_evidence
    assert read_attempt_evidence(output,failure['evidence_id'])['status']=='proposal_rejected'
    assert not list((output/'environments').glob('*'))
    assert not (output/'environment_selection.json').exists()


def test_native_fix_can_return_selection_advice_without_switching_or_installing(fixture, monkeypatch):
    base,site,output,repo,version=fixture
    prepared=make(fixture)
    original_resume=provision.resume
    row,planned=provision_fixture(fixture,monkeypatch)
    proposal={'step':'build_the_environment','arguments':{
        'base_environment_id':row['id'],'base_environment_mode':'overlay',
        'reuse_environment_ref':'env','environment_selection_reason':'reuse same ABI and preserved local closure'}}
    class Client:
        def chat_with_metadata(self,system,user,**kwargs):
            assert json.loads(user)['prepared_overlays'][0]['ref']=='env'
            return json.dumps({'proposed_action':proposal,'reasoning':'preserved modules'}),{}
    client=Client();client.output=output
    transcript=[]
    commands,probes=original_resume(client,repo,[],
        {'evidence_id':'a'*32,'template':'{python} -c "import missing"','ok':False},
        manifests={},transcript=transcript,values={'python':prepared['interpreter'],'repo':str(repo)},
        current_probes=['{python} -c "import missing"'])
    assert commands==[] and probes is None
    assert transcript[-1]['kind']=='repair_selection_advice'
    assert transcript[-1]['proposed_action']==proposal
    assert transcript[-1]['native_operation_launched'] is False
    assert not (output/'environment_selection.json').exists()
    assert not list((output/'environments').glob('*'))


@pytest.mark.parametrize('fault',['invalid_id','escape','install_mix','missing_reason'])
def test_fix_selection_advice_rejects_unbounded_or_invalid_choice(fixture,monkeypatch,fault):
    base,site,output,repo,version=fixture
    make(fixture);row,_=provision_fixture(fixture,monkeypatch)
    args={'base_environment_id':row['id'],'base_environment_mode':'overlay',
          'reuse_environment_ref':'env','environment_selection_reason':'preserved local closure'}
    if fault=='invalid_id':args['source_binding_ids']=['bad-id']
    elif fault=='escape':args['reuse_environment_ref']='../other'
    elif fault=='install_mix':args['repair_proposal']={'failure_evidence_id':'a'*32}
    else:args['environment_selection_reason']=''
    with pytest.raises(ValueError):
        provision.selection_advice(output,repo,{'step':'build_the_environment','arguments':args},manifests={})
    assert not (output/'environment_selection.json').exists()


@pytest.mark.parametrize('change',['startup','base','escape','bindings'])
def test_prepared_overlay_selection_rejects_changed_identity(fixture, monkeypatch, change):
    base, site, output, repo, version = fixture
    make(fixture)
    row, planned = provision_fixture(fixture,monkeypatch)
    ref='env'; ids=None
    if change=='startup':(output/'env/lib'/f'python{version}'/'site-packages/autosim_overlay.pth').write_text('import os\n')
    elif change=='base':(site/'task-1.dist-info/METADATA').write_text('Name: task\nVersion: 2\n')
    elif change=='escape':ref='../base'
    else:ids=[]
    monkeypatch.setattr(provision,'plan',lambda *a,**kw: pytest.fail('invalid reuse must not buy a plan'))
    with pytest.raises(provision.EnvironmentPlanError):
        provision.build(repo,client=None,prefix=output/'env',output=output,manifests={},assets={},
            base_environment_id=row['id'],base_environment_mode='overlay',
            environment_selection_reason='reuse verified metadata',reuse_environment_ref=ref,source_binding_ids=ids)
    assert not (base.parent/'HOOK_RAN').exists()


def test_crash_before_first_probe_continues_without_model_or_new_prefix(fixture, monkeypatch):
    from autosim.research import native_context
    base, site, output, repo, version = fixture
    addon = next(b['id'] for b in overlay.inspect(base,version,repo=repo)['bindings'] if b['module']=='addon')
    provision_fixture(fixture, monkeypatch, bindings=[addon])
    publish = native_context.publish_context
    monkeypatch.setattr(native_context, 'publish_context', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError, match='crash'):
        provision.build(repo,client=None,prefix=output/'env',output=output,manifests={},assets={},max_rounds=1)
    assert (output/'provision_cursor.json').is_file()
    assert (output/'environment_bootstrap.json').is_file()
    monkeypatch.setattr(native_context, 'publish_context', publish)
    monkeypatch.setattr(provision, 'plan', lambda *a, **kw: pytest.fail('resume requested another plan'))
    result = provision.build(repo,client=None,prefix=output/'env',output=output,manifests={},assets={},max_rounds=1)
    assert result['verdict']['passed'], result
    assert result['interpreter'] == str(output/'env/bin/python')
    assert json.loads((output/'environment_bootstrap.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('change', ['none', 'bindings', 'context'])
def test_repair_may_reassert_only_exact_accepted_base(fixture, monkeypatch, change):
    from autosim.research import native_context
    base, site, output, repo, version = fixture
    row, planned = provision_fixture(fixture, monkeypatch)
    publish = native_context.publish_context
    monkeypatch.setattr(native_context, 'publish_context',
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError, match='crash'):
        provision.build(repo, client=None, prefix=output/'env', output=output,
                        manifests={}, assets={}, max_rounds=1,
                        base_environment_id=row['id'], base_environment_mode='overlay',
                        environment_selection_reason='initial explicit selection')
    monkeypatch.setattr(native_context, 'publish_context', publish)
    selection = (output/'environment_selection.json').read_bytes()
    selected_prefix = Path(json.loads(selection)['prefix'])
    prefixes = list((output/'environments').glob('*'))
    kwargs = dict(client=None, prefix=output/'env', output=output, manifests={},
                  assets={}, max_rounds=1, base_environment_id=row['id'],
                  base_environment_mode='overlay', environment_selection_reason='keep existing base',
                  repair_proposal={'failure_evidence_id':'fixture'})
    if change == 'bindings':
        kwargs['source_binding_ids'] = [next(b['id'] for b in
            overlay.inspect(base, version, repo=repo)['bindings'] if b['module'] == 'addon')]
    elif change == 'context':
        kwargs['assets'] = {'research_context': {'changed':True}}
    # The valid reassertion must reach ordinary repair validation, not plan or
    # environment creation. Stop there to avoid faking native failure evidence.
    def reached_existing_interpreter(python):
        assert python == str(selected_prefix/'bin/python')
        raise RuntimeError('existing interpreter reached')
    monkeypatch.setattr(provision, 'interpreter_is_given', reached_existing_interpreter)
    if change == 'none':
        with pytest.raises(RuntimeError, match='existing interpreter reached'):
            provision.build(repo, **kwargs)
    else:
        with pytest.raises(ValueError, match='exact accepted environment continuation'):
            provision.build(repo, **kwargs)
    assert (output/'environment_selection.json').read_bytes() == selection
    assert list((output/'environments').glob('*')) == prefixes


def test_same_base_accepts_corrected_bindings_but_rejects_identical_retry(fixture, monkeypatch):
    base, site, output, repo, version = fixture
    row, planned = provision_fixture(fixture, monkeypatch)
    kwargs = dict(client=None,prefix=output/'env',output=output,manifests={},assets={},max_rounds=1,
        base_environment_id=row['id'],base_environment_mode='overlay',environment_selection_reason='correct binding')
    first = provision.build(repo,source_binding_ids=[],**kwargs)
    assert not first['verdict']['passed']
    addon = next(b['id'] for b in overlay.inspect(base,version,repo=repo)['bindings'] if b['module']=='addon')
    planned['source_binding_ids'] = [addon]
    second = provision.build(repo,source_binding_ids=[addon],**kwargs)
    assert second['verdict']['passed'], second
    assert first['interpreter'] != second['interpreter']
    with pytest.raises(ValueError,match='already attempted'):
        provision.build(repo,source_binding_ids=[addon],**kwargs)


def test_omitted_bindings_preserve_explicit_checkout_and_external_plan(fixture, monkeypatch):
    from autosim.research import native_context
    base,site,output,repo,version=fixture
    options=overlay.inspect(base,version,repo=repo)['bindings']
    ids=[b['id'] for b in options if b['module'] in {'task','addon'}]
    row,planned=provision_fixture(fixture,monkeypatch,bindings=ids)
    publish=native_context.publish_context
    monkeypatch.setattr(native_context,'publish_context',lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('crash')))
    kwargs=dict(client=None,prefix=output/'env',output=output,manifests={},assets={},max_rounds=1,
                base_environment_id=row['id'],base_environment_mode='overlay',environment_selection_reason='preserve same plan')
    with pytest.raises(RuntimeError,match='crash'):provision.build(repo,**kwargs)
    selection=(output/'environment_selection.json').read_bytes()
    interpreter=json.loads(selection)['prefix']+'/bin/python'
    retained=Path(interpreter).parent.parent/'retained-local-delta';retained.write_text('keep')
    monkeypatch.setattr(native_context,'publish_context',publish)
    monkeypatch.setattr(provision,'plan',lambda *a,**kw: pytest.fail('same definition must not buy a fresh plan'))
    result=provision.build(repo,**kwargs)
    assert result['verdict']['passed'] and result['interpreter']==interpreter
    assert retained.read_text()=='keep'
    assert (output/'environment_selection.json').read_bytes()==selection
    assert len(list((output/'environments').glob('*/bin/python')))==1


def test_prepare_does_not_report_stale_overlay_as_already_done(fixture, monkeypatch):
    from autosim.research.prepare import Preparation
    base, site, output, repo, version = fixture
    result = make(fixture)
    publish_context(output,repo,Path(result['interpreter']),{})
    atomic_json(output/'environment.json', {'verdict':{'passed':True},'probes':[]})
    obj = Preparation.__new__(Preparation)
    obj.output, obj.repo, obj.interpreter = output, repo, Path(result['interpreter'])
    obj.steps, obj.run_id, obj.declaration = [], 'fixture', None
    monkeypatch.setattr(obj,'_surveyed_runnable_stages',lambda:{})
    assert obj._step_build_the_environment()['outcome'] == 'already done'
    (output/'native_context.json').unlink()
    assert obj._step_build_the_environment()['outcome'] == 'not attempted'
    publish_context(output,repo,Path(result['interpreter']),{})
    (output/'native_context.json').write_text('invalid JSON')
    assert obj._step_build_the_environment()['outcome'] == 'not attempted'
    publish_context(output,repo,Path(result['interpreter']),{})
    (site/'task-1.dist-info/METADATA').write_text('Name: task\nVersion: 2\n')
    assert obj._step_build_the_environment()['outcome'] == 'not attempted'


@pytest.mark.parametrize('context_state', ['present','missing','corrupt'])
def test_build_revalidates_changed_dependencies_with_original_consumers(fixture, monkeypatch, context_state):
    base, site, output, repo, version = fixture
    addon = next(b['id'] for b in overlay.inspect(base,version,repo=repo)['bindings'] if b['module']=='addon')
    row, planned = provision_fixture(fixture, monkeypatch, bindings=[addon])
    first = provision.build(repo,client=None,prefix=output/'env',output=output,manifests={},assets={},max_rounds=1)
    assert first['verdict']['passed']
    first_context = load_context(output, repo)['identity']
    local = output/'env/lib'/f'python{version}'/'site-packages/dependency.py'
    local.write_text('VALUE = 99\n')
    (site/'task-1.dist-info/METADATA').write_text('Name: task\nVersion: 2\n')
    if context_state == 'missing':
        (output/'native_context.json').unlink()
    elif context_state == 'corrupt':
        (output/'native_context.json').write_text('invalid JSON')
    planned.pop('base_environment_id')
    planned['python'] = first['interpreter']
    second = provision.build(repo,client=None,prefix=output/'env',output=output,python=first['interpreter'],
        manifests={},assets={},max_rounds=1)
    assert second['verdict']['passed'], second
    assert second['interpreter'] == first['interpreter']
    assert second['verified_native_context'] != first_context
    assert local.read_text() == 'VALUE = 99\n'
    transcript = json.loads((output/'transcript.json').read_text())['rows']
    assert any(r.get('kind') == 'environment_identity_changed' and r.get('evidence_id') for r in transcript)
    assert any(r.get('kind') == 'environment_revalidated' for r in transcript)
    assert len([r for r in transcript if r.get('kind')=='probe']) == 2


def test_failed_revalidation_does_not_preserve_old_passing_verdict(fixture, monkeypatch):
    base, site, output, repo, version = fixture
    addon = next(b['id'] for b in overlay.inspect(base,version,repo=repo)['bindings'] if b['module']=='addon')
    _, planned = provision_fixture(fixture, monkeypatch, bindings=[addon])
    first = provision.build(repo,client=None,prefix=output/'env',output=output,manifests={},assets={},max_rounds=1)
    assert first['verdict']['passed']
    (site/'task-1.dist-info/METADATA').write_text('Name: task\nVersion: 2\n')
    (site/'dependency.py').unlink()
    planned.pop('base_environment_id')
    planned['python'] = first['interpreter']
    second = provision.build(repo,client=None,prefix=output/'env',output=output,python=first['interpreter'],
        manifests={},assets={},max_rounds=1)
    assert not second['verdict']['passed'], second
    assert not json.loads((output/'environment.json').read_text())['verdict']['passed']


def test_environment_refresh_refuses_while_native_job_is_active(fixture, monkeypatch):
    from autosim.research import native_jobs
    base, site, output, repo, version = fixture
    addon = next(b['id'] for b in overlay.inspect(base,version,repo=repo)['bindings'] if b['module']=='addon')
    _, planned = provision_fixture(fixture, monkeypatch, bindings=[addon])
    first = provision.build(repo,client=None,prefix=output/'env',output=output,manifests={},assets={},max_rounds=1)
    before = (output/'env/overlay.json').read_bytes()
    (site/'task-1.dist-info/METADATA').write_text('Name: task\nVersion: 2\n')
    planned.pop('base_environment_id'); planned['python'] = first['interpreter']
    monkeypatch.setattr(native_jobs,'active_jobs',lambda *a:[{'id':'training'}])
    result = provision.build(repo,client=None,prefix=output/'env',output=output,python=first['interpreter'],
        manifests={},assets={},max_rounds=1)
    assert result['verdict']['reason'] == 'overlay_revalidation_requires_idle_jobs'
    assert not result['verdict']['passed']
    assert (output/'env/overlay.json').read_bytes() == before
