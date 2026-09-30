"""Where this machine actually keeps the native libraries the engine links against.

`/home/wbc/miniconda3/envs/robotwin5090/lib` was written into seven files: the runtime's
environment builder, the RoboTwin job launcher, an official-comparison entry point and four
validation scripts. It is one machine's conda environment, named absolutely, and it provides
the cudnn and CUDA runtime that the simulator's native extension resolves symbols against.

The reason this is worth a module rather than a search-and-replace is what happens when it is
wrong. `LD_LIBRARY_PATH` **ignores a directory that is not there**. On another machine every
one of those seven sites keeps working, keeps setting the variable, and stops providing
anything -- and the failure surfaces much later, as a loader error about a symbol in a library
nobody removed. A constant that fails silently is worse than one that fails loudly.

It also cannot simply be derived. The obvious candidate, `CONDA_PREFIX`, is the environment
this *process* runs in, and on the machine this was written for that is the base environment
while the libraries are in a different one -- so the derivation returns something that looks
right, is not, and would again fail only later. What the line meant was "some environment on
this machine has the CUDA runtime", and that is a question with an answer: look.

So the order is: what the caller asked for, then what the interpreters in play carry, then the
conda environments on this machine that actually contain cudnn. Deterministic, in that order,
and the choice is returned so a caller can record it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

#: A directory that is a symlink to another is the same directory, and a list that dedupes by
#: string keeps both. conda ships `lib/python3.1 -> python3.10` beside the real one, so a glob
#: over `python3.*` finds every library twice under two names -- and a real `python3.1` install
#: resolves to itself and is kept, which is right.

#: Names a caller may set instead of the machine being searched: one or more directories,
#: separated the way this platform separates paths. This is where a fact about one machine
#: belongs -- visible in the environment rather than invisible in the source.
EXTRA_ENV = "AUTOSIM_EXTRA_LIBRARY_PATH"

#: What the search is looking for. A directory with no cudnn in it is not the environment
#: anybody meant, so it is not offered as one.
_PROVIDES = ("libcudnn", "libcublas", "libcuda.so", "libnvinfer")


def extra_library_paths(environ: dict[str, str] | None = None) -> list[Path]:
    """The directories the environment names, if it names any."""
    raw = (environ or os.environ).get(EXTRA_ENV) or ""
    return [Path(part.strip()).expanduser() for part in raw.split(os.pathsep) if part.strip()]


def base_environments(venv: Path, *, limit: int = 4) -> list[Path]:
    """The environments an interpreter was built from, following `pyvenv.cfg`.

    A virtual environment records the interpreter that created it, in `home`. Following that
    chain gives the environment whose native libraries a process started from this one will
    resolve symbols against -- and on the machine this module was written for, it gives
    exactly the directory that had been written into seven files by hand:

        .venv/pyvenv.cfg    home = /home/wbc/miniconda3/envs/robotwin5090/bin

    So the constant was derivable all along, from a file the platform already ships. What made
    it look like a machine fact was that nobody had asked the file.

    The chain is followed rather than read once, because a venv made from a venv made from a
    conda environment is a normal arrangement and the libraries are at the end of it.
    """
    out: list[Path] = []
    current = Path(venv)
    for _ in range(limit):
        config = current / "pyvenv.cfg"
        if not config.is_file():
            break
        home = None
        try:
            for line in config.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "home":
                    home = Path(value.strip())
                    break
        except OSError:
            break
        if home is None:
            break
        base = home.parent                      # `<prefix>/bin/python` -> `<prefix>`
        if base in out or base == current:
            break
        out.append(base)
        current = base
    return out


def interpreter_libraries(bases: Iterable[Path]) -> list[Path]:
    """`<prefix>/lib` for each environment a process will be started under.

    A virtual environment's libraries sit beside its binary, so an interpreter path is enough
    to find them. This is always right and never sufficient: the cudnn the simulator needs is
    usually in the environment that built the extension rather than in the one running it.
    """
    found: list[Path] = []
    for base in bases:
        lib = Path(base) / "lib"
        if lib.is_dir() and lib not in found:
            found.append(lib)
    return found


def bundled_nvidia_libraries(prefixes: Iterable[Path]) -> list[Path]:
    """A Python environment's own CUDA runtime, which is not on any loader's path.

    `nvidia-cudnn-cu12` and its siblings install into `site-packages/nvidia/<library>/lib`,
    where nothing looks for a `.so` unless it is told to. This is the one entry below that is
    a *nested* directory rather than a `<prefix>/lib`, and the difference matters: the loader
    does not search recursively, so naming `<prefix>/lib` where the libraries are two levels
    further down provides nothing at all.
    """
    # `python3.*` and not `python*`: the looser pattern also matches `python3.1`, which is a
    # real directory in some environments and is not this interpreter's.
    #
    # And one directory per library, from the first prefix that has it. A loader stats every
    # entry on the path for every symbol it cannot resolve, so a list built by concatenating
    # every prefix's copy of cublas, cudnn, nccl and nine more is fifty-odd entries doing the
    # work of a dozen -- and the copies further down can never be reached anyway.
    found: list[Path] = []
    claimed: set[str] = set()
    for prefix in prefixes:
        try:
            candidates = sorted(Path(prefix).glob("lib/python*/site-packages/nvidia/*/lib"))
        except OSError:
            continue
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            path = candidate.resolve()
            library = candidate.parent.name
            if library in claimed or path in found:
                continue
            claimed.add(library)
            found.append(path)
    return found


def provides_native_runtime(lib: Path) -> bool:
    try:
        names = [entry.name for entry in lib.iterdir()]
    except OSError:
        return False
    return any(name.startswith(_PROVIDES) for name in names)


def program_from_environments(pattern: str, *, bases: Iterable[Path] = ()) -> Path | None:
    """A binary a Python package ships, found in one of the environments in play.

    `imageio_ffmpeg` installs an `ffmpeg` under its own package directory, and four validation
    scripts named one machine's copy of it by absolute path. Which environment it lands in is
    a property of the machine; that it is that package's binary is a property of the package.
    """
    for root in [Path(base) for base in bases] + [Path(sys.prefix)]:
        for base in [root, *base_environments(root)]:
            for match in sorted(base.glob(f"lib/python3.*/site-packages/{pattern}")):
                if match.exists():
                    return match
    return None


def conda_environments_with_cuda(*, limit: int = 4) -> list[Path]:
    """Conda environments on this machine whose `lib` carries a CUDA runtime.

    Sorted, so the same machine gives the same answer twice. Bounded, because this runs while
    building an environment and a person with forty conda environments should not pay for all
    of them.
    """
    roots: list[Path] = []
    for candidate in (os.environ.get("CONDA_ROOT"), os.environ.get("CONDA_PREFIX_1"),
                      Path.home() / "miniconda3", Path.home() / "anaconda3",
                      Path("/opt/conda")):
        if not candidate:
            continue
        root = Path(candidate)
        envs = root / "envs"
        if envs.is_dir():
            roots.append(envs)
    found: list[Path] = []
    for envs in sorted(set(roots)):
        try:
            children = sorted(path for path in envs.iterdir() if path.is_dir())
        except OSError:
            continue
        for child in children:
            lib = child / "lib"
            if provides_native_runtime(lib):
                found.append(lib)
                if len(found) >= limit:
                    return found
    return found


def library_dirs(bases: Iterable[Path] = (), *, environ: dict[str, str] | None = None,
                 sys_prefix: Path | None = None) -> dict[str, list[str]]:
    """The loader's search path, in the order it should be tried, with its reasoning.

    Returns the directories and, separately, the ones that were asked for and were not there.
    A caller that records both can answer "why did this run load the library it loaded" and
    "what was missing on the machine that failed".
    """
    named = extra_library_paths(environ)
    present = [path for path in named if path.is_dir()]
    absent = [str(path) for path in named if not path.is_dir()]

    # Resolved first: `.venv` and `/home/wbc/.../.venv` are the same directory and two
    # different strings, and a list that dedupes by string keeps both.
    roots = [Path(base).expanduser().resolve() for base in bases]
    # What the interpreters were built from, before what they carry: a venv's own `lib` holds
    # its site-packages, and the native runtime the extension links against is in the
    # environment underneath it.
    for base in [*roots, Path(sys_prefix or sys.prefix).expanduser().resolve()]:
        roots.extend(base_environments(base))
    roots = list(dict.fromkeys(roots))
    # The platform's own environment first: those are the libraries the extension was built
    # against, and a different environment's copy of the same library is a way to load a
    # version nothing was tested with.
    derived = [*bundled_nvidia_libraries(roots), *interpreter_libraries(roots)]
    # The interpreters' own directories do not always carry the runtime -- on the machine this
    # was written for they carry none of it -- so a wider look happens when they do not.
    covered = any(provides_native_runtime(lib) for lib in [*present, *derived])
    searched = [] if covered else [
        lib for lib in conda_environments_with_cuda() if lib not in present and lib not in derived]

    ordered: list[Path] = []
    for path in [*present, *derived, *searched]:
        if path.is_dir() and path not in ordered:
            ordered.append(path)
    return {"directories": [str(path) for path in ordered],
            "base_environments": [str(path) for path in roots],
            "absent": absent,
            "from_environment": [str(path) for path in present],
            "from_interpreters": [str(path) for path in derived],
            "from_this_machine": [str(path) for path in searched]}

