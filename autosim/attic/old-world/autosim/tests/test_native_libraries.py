"""Where the native libraries are, derived from a file the platform already ships.

`/home/wbc/miniconda3/envs/robotwin5090/lib` was written into seven files by absolute path.
The reason it is worth a module rather than a search-and-replace is what happens when it is
wrong: `LD_LIBRARY_PATH` ignores a directory that is not there, so on another machine every
one of those sites keeps working, keeps setting the variable, and stops providing anything.
"""

import os
from pathlib import Path

from autosim.research.native import (base_environments, bundled_nvidia_libraries, library_dirs,
                                     provides_native_runtime)


def a_venv(prefix: Path, home: Path) -> Path:
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "pyvenv.cfg").write_text(
        f"home = {home}/bin\ninclude-system-site-packages = true\nversion = 3.10.20\n")
    return prefix


def with_runtime(prefix: Path, *names: str) -> Path:
    lib = prefix / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    for name in names:
        (lib / name).write_text("")
    return prefix


def test_the_environment_a_venv_was_built_from_is_read_from_its_own_file(tmp_path):
    """The constant was derivable all along, and the derivation gives exactly it.

    `.venv/pyvenv.cfg` says `home = /home/wbc/miniconda3/envs/robotwin5090/bin`, which is the
    environment whose `lib` the engine resolves cudnn against -- the path seven files named by
    hand. What made it look like a machine fact was that nobody had asked the file.
    """
    base = tmp_path / "miniconda3/envs/robotwin5090"
    venv = a_venv(tmp_path / "proj/.venv", base)
    assert base_environments(venv) == [base]


def test_the_chain_is_followed_to_its_end(tmp_path):
    """A venv made from a venv made from a conda environment is a normal arrangement."""
    conda = tmp_path / "miniconda3/envs/robotwin5090"
    inner = a_venv(tmp_path / "proj/.venv", conda)
    outer = a_venv(tmp_path / "proj/.venv_robotwin", inner)
    assert base_environments(outer) == [inner, conda]


def test_a_chain_that_points_at_itself_stops(tmp_path):
    looped = tmp_path / "proj/.venv"
    a_venv(looped, looped)
    assert base_environments(looped) == []


def test_the_runtime_is_found_where_it_actually_is(tmp_path):
    """The interpreters' own `lib` exists and carries none of it, which is why the search
    happens at all -- a rule of "if they have a lib, stop looking" would find nothing."""
    conda = with_runtime(tmp_path / "miniconda3/envs/robotwin5090", "libcudnn.so.9")
    venv = with_runtime(a_venv(tmp_path / "proj/.venv", conda))          # has a lib, no cudnn
    assert not provides_native_runtime(venv / "lib")
    answer = library_dirs(bases=[venv], sys_prefix=venv,
                          environ={"CONDA_ROOT": str(tmp_path / "miniconda3")})
    assert str(conda / "lib") in answer["directories"]


def test_a_directory_symlinked_to_another_is_not_counted_twice(tmp_path):
    """conda ships `lib/python3.1 -> python3.10`, so a glob over `python3.*` finds every
    library twice under two names -- and a loader stats each entry on every lookup."""
    conda = tmp_path / "miniconda3/envs/robotwin5090"
    (conda / "lib/python3.10/site-packages/nvidia/cudnn/lib").mkdir(parents=True)
    (conda / "lib/python3.1").symlink_to("python3.10")
    # The glob reaches it twice, under two names, for one directory.
    assert len(list(conda.glob("lib/python*/site-packages/nvidia/*/lib"))) == 2
    found = bundled_nvidia_libraries([conda])
    assert len(found) == 1, found


def test_one_directory_per_library_from_the_first_prefix_that_has_it(tmp_path):
    """Fifty entries doing the work of a dozen, and the copies further down unreachable."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    for prefix in (first, second):
        for library in ("cublas", "cudnn"):
            (prefix / f"lib/python3.10/site-packages/nvidia/{library}/lib").mkdir(parents=True)
    found = bundled_nvidia_libraries([first, second])
    assert len(found) == 2
    assert all(str(first) in str(path) for path in found)


def test_what_the_environment_names_comes_first_and_is_reported_when_absent(tmp_path):
    named = with_runtime(tmp_path / "named")
    missing = tmp_path / "not-here"
    answer = library_dirs(bases=[], sys_prefix=tmp_path,
                          environ={"AUTOSIM_EXTRA_LIBRARY_PATH": os.pathsep.join(
                              [str(named / "lib"), str(missing)])})
    assert answer["directories"][0] == str(named / "lib")
    assert answer["absent"] == [str(missing)]
    assert answer["from_environment"] == [str(named / "lib")]


def test_the_answer_says_where_each_directory_came_from(tmp_path):
    """A caller that records this can answer "why did this run load the library it loaded"."""
    answer = library_dirs(bases=[], sys_prefix=tmp_path, environ={})
    assert set(answer) >= {"directories", "absent", "from_environment", "from_interpreters",
                           "from_this_machine", "base_environments"}
