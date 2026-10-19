"""Optional benchmark discovery does not dereference every neighbor successfully."""
from pathlib import Path

import pytest

from autosim.research.common import BENCHMARK_ROOT_ENV, benchmark_roots, find_benchmark


@pytest.mark.parametrize('fault', ['root', 'child', 'marker', 'direct'])
def test_inaccessible_neighbor_does_not_hide_the_actual_checkout(tmp_path, monkeypatch, fault):
    search = tmp_path / 'search'; search.mkdir()
    denied = search / 'a-private'; denied.mkdir()
    good = search / 'z-public' / 'native' / 'scripts'
    good.mkdir(parents=True)
    (good / 'eval.py').write_text('pass\n')
    monkeypatch.setenv(BENCHMARK_ROOT_ENV, str(search))
    original_dir, original_file = Path.is_dir, Path.is_file
    def is_dir(path):
        if fault == 'root' and path == tmp_path / 'workspace':
            raise PermissionError('unreadable optional root')
        if fault == 'child' and path == denied:
            raise PermissionError('unreadable optional neighbor')
        return original_dir(path)
    def is_file(path):
        target = (search / 'native' / 'scripts' / 'eval.py' if fault == 'direct'
                  else denied / 'native' / 'scripts' / 'eval.py')
        if fault in {'marker', 'direct'} and path == target:
            raise PermissionError('unreadable optional marker')
        return original_file(path)
    monkeypatch.setattr(Path, 'is_dir', is_dir)
    monkeypatch.setattr(Path, 'is_file', is_file)
    assert find_benchmark('native', tmp_path / 'workspace', marker='scripts/eval.py') == good.parent


def test_directory_marker_is_not_proof_of_a_checkout(tmp_path, monkeypatch):
    (tmp_path / 'native/scripts/eval.py').mkdir(parents=True)
    monkeypatch.setenv(BENCHMARK_ROOT_ENV, str(tmp_path))
    assert find_benchmark('native', tmp_path, marker='scripts/eval.py') is None


def test_bad_optional_root_does_not_change_configured_root_order(tmp_path, monkeypatch):
    known = tmp_path / 'known'; known.mkdir()
    bad = tmp_path / 'loop'
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        if path == bad:
            raise RuntimeError('symlink loop')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve)
    monkeypatch.setenv(BENCHMARK_ROOT_ENV, f'{bad}:{known}:{known}')
    roots = benchmark_roots(tmp_path)
    assert roots[0] == known and roots.count(known) == 1 and bad not in roots
