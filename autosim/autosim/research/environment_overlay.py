"""Read-only dependency reuse with run-owned writes and explicit editable rebinding.

Discovered Python hooks are parsed as data, never executed. This is a live dependency
reference, not a portable snapshot or a benchmark-readiness verdict.
"""
import ast
import importlib.metadata as metadata
import json
import re
import subprocess
from pathlib import Path

from .common import atomic_json, atomic_text, digest, object_digest, read_json


def nested_bindings(dependency_sites: list[Path], version: str) -> list[dict]:
    """Offer ABI-matched nested installations as data, not executable .pth hooks.

    A bounded structural scan handles vendor prefixes without naming any vendor.
    Actual source locations stay executor-side; selecting import modules is the Agent's job.
    """
    result = []
    discovered = set()
    for site in dependency_sites:
        for nested in sorted(site.glob(f'*/lib/python{version}/site-packages'))[:128]:
            if not nested.resolve().is_relative_to(site.resolve()) or nested.resolve() in discovered:
                continue
            discovered.add(nested.resolve())
            for distribution in metadata.distributions(path=[str(nested)]):
                if not Path(distribution._path).resolve().is_relative_to(site.resolve()):
                    continue
                names = (distribution.read_text('top_level.txt') or '').splitlines()
                if not names:
                    names = sorted({str(f).split('/')[0].removesuffix('.py') for f in (distribution.files or [])})
                for name in names:
                    if name in {'sitecustomize','usercustomize','__pycache__'} or not name.isidentifier() or any((root/name).exists() or (root/(name+'.py')).exists()
                                                    for root in dependency_sites):
                        continue
                    target = nested/name
                    if not target.is_dir():
                        target = nested/(name+'.py')
                    if not target.is_file() and not target.is_dir():
                        # Native extension entrypoints can be identified without importing.
                        extensions = sorted(nested.glob(name+'.*.so'))
                        if len(extensions) != 1:
                            continue
                        target = extensions[0]
                    if not target.resolve().is_relative_to(site.resolve()):
                        continue
                    row = {'module':name, 'path':str(target.resolve()),
                        'origin':'existing_nested_dependency',
                        'metadata_path':str(Path(distribution._path).resolve()),
                        'distribution':distribution.metadata.get('Name'), 'version':distribution.version}
                    result.append({**row, 'id':object_digest(row)[:32]})
    # Some binary prefixes have no distribution metadata. Offer real packages, still
    # requiring explicit selection and native verification, rather than claiming absence.
    for site in dependency_sites:
        for nested in sorted(site.glob(f'*/lib/python{version}/site-packages'))[:128]:
            if not nested.resolve().is_relative_to(site.resolve()):
                continue
            for target in sorted(nested.iterdir())[:512]:
                name = target.name if target.is_dir() else target.name.removesuffix('.py')
                if (name in {'sitecustomize','usercustomize','__pycache__'} or not name.isidentifier() or not (target.is_dir() or target.suffix == '.py') or
                        not target.resolve().is_relative_to(site.resolve()) or
                        any(b['module']==name for b in result) or
                        any((root/name).exists() or (root/(name+'.py')).exists() for root in dependency_sites)):
                    continue
                row = {'module':name,'path':str(target.resolve()),'origin':'existing_nested_dependency'}
                result.append({**row,'id':object_digest(row)[:32]})
    return result


def sites(prefix: Path, version: str) -> list[Path]:
    prefix = prefix.absolute()
    result = [prefix/'lib'/f'python{version}'/'site-packages']
    config = prefix/'pyvenv.cfg'
    if config.is_file():
        text = config.read_text()
        home = re.search(r'(?m)^home\s*=\s*(.+)$', text)
        enabled = re.search(r'(?mi)^include-system-site-packages\s*=\s*true\s*$', text)
        if home and enabled:
            base = Path(home[1].strip()).parent
            result.append(base/'lib'/f'python{version}'/'site-packages')
    return list(dict.fromkeys(p.resolve() for p in result if p.is_dir()))


def editable_maps(site: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(site.glob('__editable__*finder.py')):
        if path.stat().st_size > 256*1024:
            continue
        try:
            tree = ast.parse(path.read_text())
            for node in tree.body:
                target = node.target if isinstance(node, ast.AnnAssign) else (
                    node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
                if not isinstance(target, ast.Name) or target.id != 'MAPPING':
                    continue
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    continue
                for name, location in value.items():
                    if (isinstance(name, str) and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', name)
                            and isinstance(location, str) and Path(location).is_absolute()):
                        result[name] = Path(location)
        except (ValueError, SyntaxError, TypeError, OSError):
            continue
    return result


def inspect(prefix: Path, version: str, *, repo: Path | None = None) -> dict:
    dependency_sites = sites(prefix, version)
    maps = {}
    excluded = set()
    skipped_hooks = []
    distributions = []
    for site in dependency_sites:
        for name, location in editable_maps(site).items():
            maps.setdefault(name, location)
        for distribution in metadata.distributions(path=[str(site)]):
            direct = distribution.read_text('direct_url.json')
            distributions.append((str(site), distribution.metadata.get('Name'), distribution.version, direct))
            try:
                local = json.loads(direct) if direct else {}
            except (ValueError, TypeError):
                local = {}
            if not isinstance(local, dict):
                local = {}
            info = local.get('dir_info')
            # Non-editable installs built from a local directory are installed copies,
            # not live source pointers. Excluding them globally can hide a valid wheel
            # in the primary venv merely because the inherited base has build provenance.
            if isinstance(info, dict) and info.get('editable'):
                excluded.add(Path(distribution._path).name)
                top = distribution.read_text('top_level.txt') or ''
                excluded.update(line.strip() for line in top.splitlines() if line.strip().isidentifier())
                if not top:
                    excluded.update(str(file).split('/')[0] for file in (distribution.files or [])
                                    if not str(file).startswith(('../', '__editable__')))
        skipped_hooks.extend(str(p) for p in site.glob('*.pth'))
    excluded.update(name.split('.')[0] for name in maps)
    bindings = []
    for name, source in sorted(maps.items()):
        # A stale editable path still identifies the import name. Bind current checkout
        # first when it has that module; external code requires an explicit catalog ID.
        checkout_bound = False
        if repo is not None:
            checkout = repo.joinpath(*name.split('.'))
            for target in (checkout, checkout.with_suffix('.py')):
                if target.exists() and target.resolve().is_relative_to(repo.resolve()):
                    row = {'module':name, 'path':str(target.resolve()), 'origin':'checkout'}
                    bindings.append({**row, 'id':object_digest(row)[:32]})
                    checkout_bound = True
                    break
        if not checkout_bound and source.exists() and source.resolve() != Path('/'):
            row = {'module':name, 'path':str(source.resolve()), 'origin':'existing_editable_source'}
            bindings.append({**row, 'id':object_digest(row)[:32]})
    for binding in nested_bindings(dependency_sites, version):
        name = binding['module']
        checkout = repo.joinpath(*name.split('.')) if repo is not None else None
        if checkout is not None and (checkout.is_dir() or checkout.with_suffix('.py').is_file()):
            continue
        if not any(b['module'] == name for b in bindings):
            bindings.append(binding)
    return {'sites':[str(p) for p in dependency_sites], 'excluded_entries':sorted(excluded),
            'bindings':bindings, 'skipped_hooks':sorted(skipped_hooks),
            'fingerprint':object_digest({'sites':[str(p) for p in dependency_sites],
                'maps':{k:str(v) for k,v in maps.items()}, 'excluded':sorted(excluded),
                'distributions':sorted(distributions, key=str),
                'hooks':{p:digest(Path(p)) for p in sorted(skipped_hooks)}}),
            'reusable':bool(dependency_sites), 'readiness':'requires_native_consumers'}


def binding_catalog(view: dict) -> list[dict]:
    return [{key:row[key] for key in ('id','module','origin','distribution','version') if key in row}
            for row in view['bindings']]


def overlay_identity(prefix: Path):
    manifest = prefix/'overlay.json'
    if not manifest.is_file() or manifest.is_symlink():
        return None
    row = read_json(manifest)
    return object_digest({k:row.get(k) for k in ('base_prefix','base_fingerprint',
        'dependency_fingerprint','dependency_view','bindings','startup_hashes')})


def readonly_roots(prefix: Path, output: Path, repo: Path) -> list[Path]:
    """Validate a controller-created overlay and mount only its dependency/code roots."""
    from .common import read_json
    manifest = prefix/'overlay.json'
    if not manifest.exists():
        return []
    if manifest.is_symlink() or not prefix.resolve().is_relative_to(output.resolve()):
        raise ValueError('unsafe overlay manifest')
    row = read_json(manifest)
    if row.get('prefix') != str(prefix.absolute()) or row.get('mode') != 'overlay':
        raise ValueError('overlay belongs to a different prefix')
    from .environment_pool import describe
    base = Path(row['base_prefix'])
    if describe(base)['fingerprint'] != row['base_fingerprint']:
        raise ValueError('borrowed environment metadata changed; revalidate consumers')
    view = inspect(base, row['python'], repo=repo)
    if row.get('dependency_fingerprint') != view['fingerprint']:
        raise ValueError('borrowed environment metadata changed; revalidate consumers')
    options = {b['id']:b for b in view['bindings']}
    if any(options.get(b['id']) != b for b in row['bindings']):
        raise ValueError('overlay source binding changed')
    if row['dependency_sites'] != view['sites']:
        raise ValueError('overlay dependency sites changed')
    roots = [base.resolve(), (base/'bin/python').resolve().parent.parent]
    roots.extend(Path(p).resolve() for p in row['dependency_sites'])
    roots.extend(Path(b['path']).resolve() for b in row['bindings'] if b['origin'] != 'checkout')
    result = []
    for root in sorted(set(roots), key=lambda p: (len(p.parts), str(p))):
        if root in {Path('/'),Path('/home'),Path('/tmp'),Path('/usr'),Path('/usr/local')}:
            # System Python is already visible in the read-only system root.
            if root in {Path('/usr'),Path('/usr/local')}:
                continue
            raise ValueError('dependency root is too broad')
        if root.is_relative_to(output.resolve()) or output.resolve().is_relative_to(root):
            raise ValueError('external dependency root overlaps writable run')
        if not any(root.is_relative_to(parent) for parent in result):
            result.append(root)
    if row.get('dependency_view'):
        view_path = Path(row['dependency_view'])
        legacy = prefix.parent/'.borrowed_dependencies'/prefix.name
        generation = legacy/row.get('definition_identity','invalid')
        if view_path != legacy and not re.fullmatch(r'[0-9a-f]{64}', row.get('definition_identity','')):
            raise ValueError('unsafe dependency generation identity')
        if (view_path not in {legacy,generation} or view_path.is_symlink() or
                not view_path.is_dir() or not view_path.resolve().is_relative_to(output.resolve())):
            raise ValueError('unsafe run-owned dependency view')
        result.append(view_path.resolve())
    for relative, expected in (row.get('startup_hashes') or {}).items():
        path = prefix/relative
        if not path.resolve().is_relative_to(prefix.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError('overlay startup files changed; controlled repair required')
    return result


def runtime_library_dirs(prefix: Path, output: Path, repo: Path) -> list[str]:
    """The selected interpreter's native runtime is part of its dependency closure.

    Never scan the host or select a library by benchmark/error name. System Python
    keeps system resolution; a borrowed non-system interpreter brings its own lib.
    All returned directories are already inside validated read-only mounts.
    """
    roots = readonly_roots(prefix, output, repo)
    if not roots:
        return []
    row = read_json(prefix/'overlay.json')
    base = Path(row['base_prefix'])
    runtime = (base/'bin/python').resolve(strict=True).parent.parent
    if runtime in {Path('/usr'), Path('/usr/local')}:
        return []
    library = (runtime/'lib').resolve(strict=True)
    if (not library.is_dir() or not library.is_relative_to(runtime) or
            not any(library.is_relative_to(root) for root in roots)):
        raise ValueError('borrowed interpreter native libraries escaped validated roots')
    return [str(library)]


FINDER = '''import importlib.abc, importlib.util, importlib.machinery, sys
from pathlib import Path
MAPPING = {mapping!r}
class BoundSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MAPPING:
            return None
        source = Path(MAPPING[fullname])
        if source.is_file():
            return importlib.util.spec_from_file_location(fullname, source)
        entry = source / '__init__.py'
        if entry.is_file():
            return importlib.util.spec_from_file_location(fullname, entry, submodule_search_locations=[str(source)])
        spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
        spec.submodule_search_locations = [str(source)]
        return spec
sys.meta_path.insert(0, BoundSources())
'''


def create(chosen: dict, *, prefix: Path, output: Path, repo: Path,
           source_binding_ids: list[str], timeout: float = 60, refresh: bool = False) -> dict:
    from .environment_pool import describe
    source = Path(chosen['prefix']).absolute()
    fresh = describe(source)
    if fresh['fingerprint'] != chosen['fingerprint']:
        raise ValueError('base environment metadata changed; refresh catalog')
    view = inspect(source, chosen['python'], repo=repo)
    if chosen.get('overlay_fingerprint') and chosen['overlay_fingerprint'] != view['fingerprint']:
        raise ValueError('base environment metadata changed; refresh catalog')
    if not view['reusable']:
        raise ValueError('no dependency sites available for overlay')
    if (not isinstance(source_binding_ids, list) or len(source_binding_ids) > 64 or
            any(not isinstance(v, str) for v in source_binding_ids)):
        raise ValueError('source_binding_ids must be a bounded list of catalog IDs')
    available = {row['id']:row for row in view['bindings']}
    if any(identity not in available for identity in source_binding_ids):
        raise ValueError('source binding ID unavailable; refresh catalog')
    selected = [available[i] for i in source_binding_ids]
    names = [row['module'] for row in selected]
    if len(names) != len(set(names)):
        raise ValueError('select only one source for each module')
    # Current checkout bindings are the default, including stale old editable imports.
    selected += [row for row in view['bindings'] if row['origin'] == 'checkout' and row['module'] not in names]
    selected.sort(key=lambda row: row['module'])
    prefix = prefix.absolute()
    if (prefix.is_symlink() or not prefix.resolve().is_relative_to(output.resolve()) or
            prefix == output.resolve() or source.resolve().is_relative_to(output.resolve())):
        raise ValueError('overlay needs a run-owned prefix and an external read-only base')
    for binding in selected:
        if binding['origin'] != 'checkout':
            target = Path(binding['path']).resolve()
            if target.is_relative_to(output.resolve()) or output.resolve().is_relative_to(target):
                raise ValueError('external source binding overlaps writable run; use checkout binding')
    definition = {'prefix':str(prefix),'repo':str(repo.resolve()),'base_prefix':str(source),
        'base_fingerprint':fresh['fingerprint'],'dependency_fingerprint':view['fingerprint'],
        'python':chosen['python'],'bindings':selected}
    identity = object_digest(definition)
    transaction = prefix/'overlay_creation.json'
    manifest = prefix/'overlay.json'
    previous = read_json(manifest) if manifest.is_file() and not manifest.is_symlink() else {}
    if previous and not refresh:
        # Handles a crash after the final manifest but before the caller's receipt/cursor.
        if any(previous.get(k) != v for k,v in definition.items() if k != 'repo'):
            raise ValueError('existing overlay belongs to another definition; explicitly revise it')
        readonly_roots(prefix, output, repo)
        return {'ok':True,'returncode':0,'command':'create_readonly_overlay','recovered':True,
                'overlay':previous,'interpreter':str(prefix/'bin/python')}
    if refresh:
        if not previous or previous.get('base_prefix') != str(source) or previous.get('python') != chosen['python']:
            raise ValueError('controlled refresh cannot change the base interpreter or ABI')
        history = prefix/'overlay_history'/f"{object_digest(previous)}.json"
        atomic_json(history, previous)
    elif prefix.exists():
        if transaction.is_symlink() or not transaction.is_file() or read_json(transaction).get('definition') != definition:
            raise ValueError('existing prefix is not a matching interrupted overlay transaction')
    prefix.mkdir(parents=True, exist_ok=True)
    status = read_json(transaction).get('status') if transaction.is_file() else None
    atomic_json(transaction, {'definition':definition,'identity':identity,'status':status or 'preparing'})
    if not refresh and status not in {'venv_ready','prepared'}:
        # Replaying only this run-owned, not-yet-published venv construction is safe.
        # Never run ensurepip or inherited hooks, nor reset an already usable prefix.
        subprocess.run([str(source/'bin/python'), '-I', '-S', '-m', 'venv', '--copies', '--without-pip', str(prefix)],
                       check=True, capture_output=True, timeout=timeout)
        atomic_json(transaction, {'definition':definition,'identity':identity,'status':'venv_ready'})
    site = prefix/'lib'/f"python{chosen['python']}"/'site-packages'
    if not site.is_dir():
        raise ValueError('overlay Python ABI disagrees with catalog')
    # Keep borrowed distribution locations OUTSIDE sys.prefix: pip then treats them
    # as external, installs overrides locally and does not try uninstalling base files.
    dependencies = prefix.parent/'.borrowed_dependencies'/prefix.name/identity
    if dependencies.is_symlink():
        raise ValueError('unsafe dependency view')
    if str(dependencies.resolve()).startswith(str(prefix.resolve())):
        raise ValueError('dependency view must not share the interpreter prefix pathname')
    dependencies.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_distributions = set()
    for directory in view['sites']:
        suppressed_metadata = set()
        for distribution in metadata.distributions(path=[directory]):
            name = re.sub(r'[-_.]+', '-', str(distribution.metadata.get('Name') or '').lower())
            if name in seen_distributions:
                suppressed_metadata.add(Path(distribution._path).name)
            elif Path(distribution._path).name not in view['excluded_entries']:
                seen_distributions.add(name)
        for path in sorted(Path(directory).iterdir()):
            if (path.name in view['excluded_entries'] or path.name in suppressed_metadata or path.name.startswith('__editable__') or
                    path.suffix in {'.pth','.egg-link'} or
                    path.name.split('.')[0] in {'sitecustomize','usercustomize','__pycache__'}):
                continue
            target = dependencies/path.name
            if not target.exists() and not target.is_symlink():
                target.symlink_to(path.resolve(), target_is_directory=path.is_dir())
                count += 1
            elif not target.is_symlink() or target.resolve() != path.resolve():
                # Higher priority sites intentionally win; validate only their first link.
                if not any((Path(s)/path.name).exists() and target.resolve() == (Path(s)/path.name).resolve()
                           for s in view['sites'][:view['sites'].index(directory)]):
                    raise ValueError('interrupted dependency view entry changed')
    for binding in selected:
        if binding.get('metadata_path'):
            origin = Path(binding['metadata_path'])
            target = dependencies/origin.name
            if not target.exists() and not target.is_symlink():
                target.symlink_to(origin, target_is_directory=True)
    alias = prefix/'dependencies'
    if alias.exists() and not alias.is_symlink():
        # Legacy overlays used a real directory here. Do not overwrite it; the new
        # trusted .pth uses its immutable generation directly.
        pass
    elif not alias.is_symlink():
        alias.symlink_to(dependencies, target_is_directory=True)
    elif refresh:
        replacement = prefix/'dependencies.next'
        if replacement.exists() or replacement.is_symlink():
            if not replacement.is_symlink() or replacement.resolve() != dependencies.resolve():
                raise ValueError('unsafe dependency alias replacement')
        else:
            replacement.symlink_to(dependencies, target_is_directory=True)
        replacement.replace(alias)
    atomic_text(site/'autosim_overlay_sources.py', FINDER.format(mapping={row['module']:row['path'] for row in selected}))
    atomic_text(site/'autosim_overlay.pth', str(dependencies)+'\nimport autosim_overlay_sources\n')
    # pip's shebang and writes target this prefix, never the borrowed environment.
    atomic_text(prefix/'bin/pip', '#!'+str(prefix/'bin/python')+'\nfrom pip._internal.cli.main import main\nraise SystemExit(main())\n')
    (prefix/'bin/pip').chmod(0o755)
    record = {'schema_version':1, 'mode':'overlay', 'prefix':str(prefix), 'base_prefix':str(source),
        'definition_identity':identity,
        'base_fingerprint':fresh['fingerprint'], 'dependency_fingerprint':view['fingerprint'],
        'dependency_view':str(dependencies),
        'python':chosen['python'], 'dependency_entries':count,
        'dependency_sites':view['sites'], 'bindings':selected, 'skipped_hooks':view['skipped_hooks'],
        'excluded_entries':view['excluded_entries'], 'readiness':'requires_native_consumers',
        'policy':'live read-only dependencies; editable sources explicitly rebound; local installs only'}
    record['startup_hashes'] = {str(p.relative_to(prefix)):digest(p) for p in
        (site/'autosim_overlay_sources.py', site/'autosim_overlay.pth')}
    atomic_json(prefix/'overlay.json', record)
    atomic_json(transaction, {'definition':definition,'identity':identity,'status':'prepared'})
    return {'ok':True, 'returncode':0, 'command':'create_readonly_overlay',
            'overlay':record, 'interpreter':str(prefix/'bin/python')}


def refresh_selected(prefix: Path, output: Path, repo: Path) -> dict:
    """Explicit preparation revalidation preserves local installs and old generations."""
    from .environment_pool import describe
    manifest = prefix/'overlay.json'
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError('no controlled overlay to revalidate')
    row = read_json(manifest)
    if row.get('prefix') != str(prefix.absolute()):
        raise ValueError('overlay prefix identity changed')
    chosen = describe(Path(row['base_prefix']))
    available = inspect(Path(row['base_prefix']),chosen['python'],repo=repo)['bindings']
    selected = []
    for previous in row['bindings']:
        if previous['origin'] == 'checkout':
            continue
        matching = [b for b in available if all(b.get(k)==previous.get(k) for k in ('module','path','origin'))]
        if len(matching) != 1:
            raise ValueError('approved module source moved; Agent must revise its binding IDs')
        selected.append(matching[0]['id'])
    return create(chosen,prefix=prefix,output=output,repo=repo,source_binding_ids=selected,refresh=True)
