"""Durable records, immutable specifications, and bounded subprocess execution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_local_environment(output: Path, base: dict[str, str] | None = None, *,
                          read_native_context: bool = True) -> dict[str, str]:
    """Keep implicit home/cache writes inside a run, not in the operator's home.

    This is a path hygiene layer, not an OS sandbox: a command naming an absolute path can
    still escape it and must be independently constrained before execution.
    """
    root = Path(output).absolute()
    home = root / "home"
    cache = home / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(base or os.environ)
    env.update(HOME=str(home), XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
               PIP_CACHE_DIR=str(cache / "pip"), UV_CACHE_DIR=str(cache / "uv"),
               CONDA_PKGS_DIRS=str(cache / "conda"))
    context = root / "native_context.json"
    if read_native_context and context.is_file() and not context.is_symlink():
        row = read_json(context)
        identity = {k: v for k, v in row.items() if k not in {"identity", "updated_at"}}
        if object_digest(identity) != row.get("identity") or not isinstance(row.get("paths"), dict):
            raise ValueError("native configuration identity changed")
        if row.get('runtime_library_dirs'):
            from .native_context import load_context
            checked = load_context(root, Path(row['repo']))
            paths = checked['runtime_library_dirs'] + [
                item for item in env.get('LD_LIBRARY_PATH', '').split(':') if item]
            env['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(paths))
        for key, value in row["paths"].items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("invalid native configuration mapping")
            if ((key == "HOME" or key.endswith(("_CONFIG_PATH", "_CONFIG_HOME"))) and
                    Path(value).is_absolute() and Path(value).resolve().is_relative_to(root.resolve())):
                env[key] = value
    from .environment_pool import wheel_links, store_for
    pool = store_for(root)
    if pool is not None:
        view_path = root / "package_cache_view.json"
        if view_path.is_file():
            view = read_json(view_path)
            if view.get("root") != str(pool):
                raise ValueError("package cache view differs from configured store")
            links = view.get("links") or []
        else:
            links = wheel_links(root)
            atomic_json(view_path, {"root": str(pool), "links": links, "verified_at": now()})
        if links:
            env["PIP_FIND_LINKS"] = " ".join(links)
            env["UV_FIND_LINKS"] = env["PIP_FIND_LINKS"]
    return env


def isolated_argv(argv: list[str], *, output: Path, repo: Path,
                  require_pid_namespace: bool = False,
                  read_only_roots: tuple[Path, ...] = (),
                  allow_gpu: bool = True,
                  native_environment: dict[str, str] | None = None,
                  writable_paths: tuple[Path, ...] | None = None) -> list[str]:
    """Run controlled commands in a PID namespace with explicit writable paths.

    The namespace matters even when the process group is new: a benchmark can call
    ``setsid()`` and leave a child outside that group. With the namespace's init tied to
    bubblewrap, exiting/killing the tracked launcher causes the kernel to kill the namespace
    members too. Isolated-copy runs additionally require the repository to live under the
    run directory. On hosts without bubblewrap, direct execution is refused unless the
    operator explicitly sets ``AUTOSIM_ALLOW_PROCESS_GROUP_ONLY=1``; that opt-in only
    guarantees process-group cleanup and does not contain ``setsid()`` descendants.
    """
    output = Path(output).resolve()
    repo = Path(repo).resolve()
    snapshot_path = repo.parent / "workspace_snapshot.json"
    if not snapshot_path.is_file():
        snapshot_path = output / "workspace_snapshot.json"
    isolated_copy = snapshot_path.is_file()
    if isolated_copy and not repo.is_relative_to(output):
        raise ValueError("isolated checkout is outside its run directory")
    if argv:
        executable = Path(str(argv[0]))
        # Only an absolute target can be checked without the caller's cwd. Relative argv[0]
        # is resolved by Popen after it applies that cwd, which may differ from the repo root.
        if (executable.is_absolute() and executable.is_file() and
                not os.access(executable, os.X_OK)):
            raise PermissionError(13, "Permission denied", str(executable))
    bubblewrap = shutil.which("bwrap")
    if not bubblewrap:
        if isolated_copy:
            raise ValueError("isolated-copy execution requires bubblewrap")
        if require_pid_namespace:
            raise ValueError("benchmark command requires a bubblewrap PID namespace")
        if os.environ.get("AUTOSIM_ALLOW_PROCESS_GROUP_ONLY") != "1":
            raise ValueError(
                "bubblewrap is unavailable; refusing weak process-group-only execution; "
                "set AUTOSIM_ALLOW_PROCESS_GROUP_ONLY=1 only if escaped descendants are an "
                "accepted limitation")
        return [str(part) for part in argv]
    if repo == Path("/") or output == Path("/"):
        raise ValueError("repository and run output may not be the filesystem root")
    output.mkdir(parents=True, exist_ok=True)
    writable = sorted(set(writable_paths) if writable_paths is not None else
                      {Path("/tmp"), repo, output}, key=lambda path: len(path.parts))
    if writable_paths is not None and any(Path(path) != Path("/tmp") and not Path(path).resolve().is_relative_to(output)
           for path in writable):
        raise ValueError("native writable path escaped run output")
    command = [bubblewrap, "--ro-bind", "/", "/"]
    if writable_paths is not None:
        # Preview scripts do not get the host's writable /tmp or host home contents.
        for hidden in (Path("/tmp"), Path("/home"), Path("/root"), Path("/run")):
            if hidden.exists(): command.extend(("--tmpfs", str(hidden)))
        command.extend(("--ro-bind", str(output), str(output),
                        "--ro-bind", str(repo), str(repo)))
        command.append("--unshare-net")
    for path in writable:
        if writable_paths is not None and Path(path) == Path("/tmp"):
            continue
        command.extend(("--bind", str(path), str(path)))
    from .environment_pool import store_for
    pool = store_for(output)
    context_path = output/'native_context.json'
    if context_path.is_file() and not context_path.is_symlink():
        context = read_json(context_path)
        if context.get('repo') == str(repo) and context.get('dependency_roots'):
            from .native_context import load_context
            native = load_context(output, repo)
            read_only_roots = (*read_only_roots, *(Path(p) for p in native.get('dependency_roots', [])))
    for path in (*read_only_roots, *((pool,) if pool is not None else ())):
        resolved = Path(path).resolve(strict=True)
        if resolved == Path("/") or output.is_relative_to(resolved):
            raise ValueError("read-only environment root overlaps writable output")
        command.extend(("--ro-bind", str(resolved), str(resolved)))
    if isolated_copy:
        manifest = read_json(snapshot_path)
        if manifest.get("source"):
            source = Path(manifest["source"]).resolve(strict=True)
            if source.is_relative_to(output) or output.is_relative_to(source):
                raise ValueError("source and output must be disjoint")
            command.extend(("--ro-bind", str(source), str(source)))
    from .workspace_resources import mount_args, bindings_for
    for binding in bindings_for(repo):
        command.extend(("--ro-bind", binding["source"], binding["source"]))
    command.extend(mount_args(repo))
    if native_environment and native_environment.get('HOME'):
        # Native SDKs may derive fixed-width identifiers from absolute cache
        # paths. Use a short namespace alias of the SAME owned home, not a fresh
        # cache, a relocated checkout or writable access to external resources.
        home = Path(native_environment['HOME']).resolve()
        if not home.is_relative_to(output) or home == output:
            raise ValueError('native HOME must be confined to a run-owned subdirectory')
        if writable_paths is not None and not any(
                home.is_relative_to(Path(path).resolve()) for path in writable
                if Path(path) != Path('/tmp')):
            raise ValueError('native home alias cannot widen restricted writable paths')
        home.mkdir(parents=True, exist_ok=True)
        alias = Path('/tmp') / ('as-h-' + object_digest(str(home))[:16])
        alias.mkdir(mode=0o700, exist_ok=True)
        if (alias.is_symlink() or not alias.is_dir() or
                alias.stat().st_uid != os.getuid() or alias.stat().st_mode & 0o077 or
                any(alias.iterdir())):
            raise ValueError('unsafe native home alias mount point')
        command.extend(('--bind', str(home), str(alias)))
        mapped = {}
        for key in ('HOME', 'XDG_CACHE_HOME', 'HF_HOME', 'PIP_CACHE_DIR',
                    'UV_CACHE_DIR', 'CONDA_PKGS_DIRS'):
            value = native_environment.get(key)
            if value and (value == str(home) or value.startswith(str(home) + '/')):
                mapped[key] = str(alias) + value[len(str(home)):]
                command.extend(('--setenv', key, mapped[key]))
        mapping = {'schema_version': 1, 'source': str(home), 'namespace_alias': str(alias),
                   'environment': mapped, 'authority': 'same run-owned native cache bytes'}
        mapping_ref = output / 'native_home_mappings' / f'{object_digest(mapping)}.json'
        if mapping_ref.parent.is_symlink() or mapping_ref.is_symlink():
            raise ValueError('unsafe native home mapping receipt')
        atomic_json(mapping_ref, mapping)
    command.extend(("--dev-bind", "/dev", "/dev") if allow_gpu else ("--dev", "/dev"))
    command.extend(("--unshare-pid",
                    "--die-with-parent", "--proc", "/proc", "--", *map(str, argv)))
    return command


def inspection_mountpoint(repo: Path) -> Path:
    """Choose an existing mount point that can safely be overlaid with the checkout."""
    checkout = Path(repo).resolve(strict=True)
    if checkout == Path("/"):
        raise ValueError("inspection checkout may not be the filesystem root")
    for candidate in (Path("/mnt"), Path("/media"), Path("/srv")):
        if (candidate.is_dir() and not checkout.is_relative_to(candidate) and
                not candidate.is_relative_to(checkout)):
            return candidate
    raise ValueError("no existing isolated mount point is available for inspection")


def inspection_argv(argv: list[str], *, repo: Path, directory: Path,
                    environment: dict[str, str]) -> list[str]:
    """Build a fail-closed sandbox command for an LLM-requested diagnostic inspection.

    Unlike benchmark stages, an inspection only needs to read the checkout and ask a
    bounded question of a tool. It therefore gets no writable bind mounts, no host home,
    temp, runtime sockets, devices, or network. The checkout is visible read-only at a
    stable in-sandbox path; callers must map checkout arguments to the selected existing
    mount point.
    A missing bubblewrap binary is an explicit refusal, never a process-group fallback.
    """
    bubblewrap = shutil.which("bwrap")
    if not bubblewrap:
        raise ValueError("read-only inspection requires bubblewrap; refusing host execution")
    checkout = Path(repo).resolve(strict=True)
    stage_dir = Path(directory).resolve(strict=False)
    if not checkout.is_dir() or not stage_dir.is_relative_to(checkout):
        raise ValueError("inspection working directory must be inside the checkout")
    if not argv:
        raise ValueError("inspection command is empty")
    mountpoint = inspection_mountpoint(checkout)

    # Bind before masking /tmp or /home: a legitimate checkout may itself live below one of
    # those host paths. Overlay it on an existing mount point so the helper never needs to
    # create a directory on the read-only root filesystem.
    command = [bubblewrap, "--ro-bind", "/", "/", "--ro-bind", str(checkout),
               str(mountpoint)]
    for hidden in (Path("/home"), Path("/root"), Path("/run"), Path("/tmp"),
                   Path("/mnt"), Path("/media"), Path("/srv")):
        if hidden.exists() and hidden != mountpoint:
            command.extend(("--tmpfs", str(hidden)))
    from .workspace_resources import mount_args
    command.extend(mount_args(checkout, mounted_at=mountpoint))
    command.extend(("--dir", "/tmp/home", "--dir", "/tmp/cache",
                    "--unshare-pid", "--unshare-net", "--die-with-parent",
                    "--proc", "/proc", "--dev", "/dev", "--cap-drop", "ALL",
                    "--clearenv", "--setenv", "HOME", "/tmp/home", "--setenv",
                    "TMPDIR", "/tmp", "--setenv", "XDG_CACHE_HOME", "/tmp/cache",
                    "--chdir", str(mountpoint) + (
                        ("/" + stage_dir.relative_to(checkout).as_posix())
                        if stage_dir != checkout else "")))
    for name, value in sorted(environment.items()):
        if not name or "\x00" in name or "\x00" in value:
            continue
        command.extend(("--setenv", name, value))

    def map_checkout_path(value: str) -> str:
        prefix = str(checkout)
        if value == prefix:
            return str(mountpoint)
        if value.startswith(prefix + os.sep):
            return str(mountpoint) + value[len(prefix):]
        return value

    command.extend(("--", *(map_checkout_path(str(part)) for part in argv)))
    return command


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    """Replace a human-readable report without exposing a partial rewrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def redact(text: str) -> str:
    return re.sub(r"(?:hf_|sk-)[A-Za-z0-9_-]{16,}", "[REDACTED]", text)


_MODEL_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?<![\w.-])['\"]?([A-Z0-9_.-]*(?:API[_-]?KEY|ACCESS[_-]?KEY|TOKEN|SECRET|"
    r"PASSWORD|PASSWD|CREDENTIALS?)[A-Z0-9_.-]*)['\"]?\s*([=:])\s*"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;}\]]+)")
_MODEL_SECRET_TOKEN = re.compile(
    r"(?i)(?:hf_[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{12,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})")
_MODEL_BEARER = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]+=*")
_MODEL_PRIVATE_URL = re.compile(
    r"(?i)https?://(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})"
    r"(?::\d+)?(?:/[^\s\"'<>]*)?")
_MODEL_FILE_URL = re.compile(r"(?i)file://[^\s\"'<>]+")
_MODEL_MACHINE_PATH = re.compile(
    r"(?i)(?<![\w:/\\])/(?:home|root|tmp|var/tmp|run/user)(?:/[^\s\"'<>;,)}\]]*)?")
_MODEL_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:\\(?:[^\s\"'<>|?*]+\\?)+")
_MODEL_UNC_PATH = re.compile(r"(?<![\w])\\\\[^\s\"'<>|?*]+")
_MODEL_PARENT_PATH = re.compile(r"(?<![\w.])(?:\.\./){1,}[\w.@~+/-]+")


def _sanitize_serialized_json(value: Any, scrub) -> str | None:
    """Scrub decoded strings, never JSON escape sequences mistaken for paths."""
    if not isinstance(value, str) or not value.lstrip().startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(value)
    except (ValueError, RecursionError):
        return None

    def walk(item, depth=0):
        if depth > 24:
            return "[NESTING_LIMIT]"
        if isinstance(item, dict):
            result = {}
            for index, (key, child) in enumerate(item.items()):
                name = scrub(key)
                if name in result:
                    name = f"{name}_{index + 1}"
                result[name] = walk(child, depth + 1)
            return result
        if isinstance(item, list):
            return [walk(child, depth + 1) for child in item]
        return scrub(item) if isinstance(item, str) else item

    return json.dumps(walk(parsed), ensure_ascii=False)


def sanitize_model_text(text: Any, *, local_roots: tuple[Path, ...] = ()) -> str:
    """Remove credentials and private host paths before text crosses an LLM boundary.

    This textual scrub is a final safety net, not a substitute for projecting structured
    reports: survey asset paths must be removed by ``summarise_for_model``. Common private
    home/temp/runtime prefixes and caller-supplied checkout roots are hidden, while
    task-relevant paths such as a benchmark's configured ``/data`` mount remain usable.
    """
    structured = _sanitize_serialized_json(text, lambda item: sanitize_model_text(
        item, local_roots=local_roots))
    if structured is not None:
        return structured
    value = str(text if text is not None else "")
    roots = sorted((str(Path(root).expanduser().resolve()) for root in local_roots),
                   key=len, reverse=True)
    for root in roots:
        if root and root != "/":
            value = value.replace(root, "{repo}")
    value = _MODEL_SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", value)
    value = _MODEL_BEARER.sub(r"\1[REDACTED]", value)
    value = _MODEL_SECRET_TOKEN.sub("[REDACTED]", value)
    value = _MODEL_PRIVATE_URL.sub("[PRIVATE_URL]", value)
    value = _MODEL_FILE_URL.sub("[LOCAL_PATH]", value)
    value = _MODEL_WINDOWS_PATH.sub("[LOCAL_PATH]", value)
    value = _MODEL_UNC_PATH.sub("[LOCAL_PATH]", value)
    value = _MODEL_MACHINE_PATH.sub("[LOCAL_PATH]", value)
    value = _MODEL_PARENT_PATH.sub("[OUTSIDE_PATH]", value)
    return value


_MODEL_LOCAL_RESOURCE_FIELDS = frozenset({
    "artifact", "artifacts", "artifact_path", "artifact_paths", "artifact_pattern",
    "candidate_paths", "selected_path", "verified_artifact", "policy_artifact",
    "result_artifact", "metric_artifact_evidence", "checkpoint", "checkpoint_path",
    "checkpoint_paths", "dataset", "dataset_path", "demo", "demo_path", "asset_path",
    "trajectory", "trajectory_path", "video_path", "media_path", "episode_id", "episode_ids",
    "episode_keys", "initial_state_hashes", "checkpoint_sha256", "artifact_sha256",
    "policy_sha256", "dataset_sha256", "demo_sha256", "trajectory_hashes",
    "relative_path", "sha256", "size_bytes", "mtime",
})
_MODEL_LOCAL_RESOURCE_REF = re.compile(
    r"(?<![\w.-])(?:[\w.@+~*?={}\[\]-]+[/\\])+"
    r"[\w.@+~*?={}\[\]-]+\.(?:pt|pth|ckpt|safetensors|bin|onnx|pkl|pickle|"
    r"h5|hdf5|msgpack|flax|mp4|mkv|avi|mov|npz|npy|jsonl|json|csv|parquet)"
    r"(?:[\w.+~-]*)?(?![\w.-])|"
    r"(?<![\w.-])[\w.@+~*?={}\[\]-]+\.(?:pt|pth|ckpt|safetensors|bin|onnx|"
    r"pkl|pickle|h5|hdf5|msgpack|flax|mp4|mkv|avi|mov|npz|npy|jsonl|json|csv|"
    r"parquet)(?:[\w.+~-]*)?(?![\w.-])", re.IGNORECASE)
_MODEL_ABSOLUTE_PATH = re.compile(r"(?<![\w:/])/(?:[\w.@+~-]+/)*[\w.@+~-]+/?")
_MODEL_EXECUTION_HANDLE_END = re.compile(r"\{(?:repo|run_path_\d+)\}$")
_MODEL_EPISODE_ID = re.compile(
    # A field identifier or ordinary prose is not an observed episode identity.
    # Keep explicit assignments and concrete ID-like labels private, but do not
    # corrupt `episode_id`, `episode_ids`, `episode_index` or `episode_keys` syntax.
    r"(?i)\b(?:episode(?:[_ -]?id)?\s*[:=#]\s*[\w.-]+|"
    r"episode(?:[_-]?id)?[_-](?!(?:id|ids|index|keys)\b)[\w.-]+|"
    r"episode(?:[_ -]?id)?\s+(?=[\w.-]*\d)[\w.-]+)")
_MODEL_RESOURCE_ARGUMENT = re.compile(
    r"(?i)(--?(?:checkpoint|ckpt|model(?:-path)?|policy(?:-path)?|dataset|data|demo|"
    r"trajectory|video|output|asset-path|file)(?:=|\s+))(?P<quote>['\"]?)"
    r"(?P<value>[^\s'\";&|]+)(?P=quote)")


def sanitize_model_payload_text(value: Any, *, local_roots: tuple[Path, ...] = ()) -> str:
    """Scrub a free-text prompt value, including relative local artifacts and episode IDs."""
    structured = _sanitize_serialized_json(value, lambda item: sanitize_model_payload_text(
        item, local_roots=local_roots))
    if structured is not None:
        return structured
    text = sanitize_model_text(value, local_roots=local_roots)
    # Preserve the argument name before the generic artifact matcher sees a joined token
    # such as ``--dataset=/private/data/episodes.hdf5``. Without this ordering it hides the
    # flag along with the path, depriving a model of useful command semantics.
    text = _MODEL_RESOURCE_ARGUMENT.sub(r"\1\g<quote>[LOCAL_RESOURCE]\g<quote>", text)
    text = _MODEL_LOCAL_RESOURCE_REF.sub("[LOCAL_RESOURCE]", text)
    text = _MODEL_EPISODE_ID.sub("episode [LOCAL_ID]", text)
    def absolute_path(match):
        # Known handle suffixes are run-relative identities, not host paths.
        # Repeated projection must retain the exact consumer/script distinction.
        if _MODEL_EXECUTION_HANDLE_END.search(text, max(0, match.start()-64), match.start()):
            return match.group()
        return "[LOCAL_PATH]"
    return _MODEL_ABSOLUTE_PATH.sub(absolute_path, text)


def sanitize_model_payload(value: Any, *, local_roots: tuple[Path, ...] = (),
                           field: str = "", depth: int = 0) -> Any:
    """Project structured run data before an external model boundary.

    Stage/metric semantics remain available, while filesystem identities, local artifact
    names, per-episode IDs and file hashes are withheld. Local verifiers must retain the
    original object and resolve any opaque candidate IDs themselves.
    """
    if depth > 24:
        return "[NESTING_LIMIT]"
    normalized = str(field).casefold().replace("-", "_")
    if normalized == "shipped_checkpoint":
        return bool(value)
    if normalized == "report_ref":
        return "state.research_progress.report" if value else None
    if normalized in _MODEL_LOCAL_RESOURCE_FIELDS:
        # Schema previews may describe a private field's type, never its value.
        # This narrow descriptor cannot contain identities, paths or results.
        if (isinstance(value, dict) and set(value) == {"type"} and
                isinstance(value.get("type"), str) and
                value.get("type") in {"string", "integer", "number", "boolean", "null"}):
            return dict(value)
        if normalized in {"artifact", "artifacts", "artifact_path", "artifact_paths",
                          "artifact_pattern", "checkpoint", "checkpoint_path",
                          "checkpoint_paths", "policy_artifact", "result_artifact",
                          "metric_artifact_evidence"}:
            return "[DECLARED_LOCAL_ARTIFACT]" if value else ""
        if normalized in {"episode_id", "episode_ids", "episode_keys"}:
            return {"present": bool(value), "values_omitted": True}
        return "[LOCAL_RESOURCE_IDENTITY_OMITTED]" if value else None
    if normalized in {"path", "repository_path", "local_path", "interpreter"}:
        return "[LOCAL_PATH]" if value else ""
    if normalized == "interpreter_already_present":
        return bool(value)
    if isinstance(value, dict):
        projected: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            original_name = str(key)
            if original_name.casefold().startswith("_local_"):
                continue
            original_normalized = original_name.casefold().replace("-", "_")
            name = (original_name if original_normalized in _MODEL_LOCAL_RESOURCE_FIELDS else
                    sanitize_model_payload_text(original_name, local_roots=local_roots))
            if name in projected:
                name = f"{name}_{index + 1}"
            if original_name.casefold() == "base_python_hint" and isinstance(item, dict):
                projected[name] = {"exists": bool(item.get("exists"))}
                continue
            projected[name] = sanitize_model_payload(
                item, local_roots=local_roots, field=original_name, depth=depth + 1)
        return projected
    if isinstance(value, list):
        return [sanitize_model_payload(item, local_roots=local_roots,
                                       field=field, depth=depth + 1)
                for item in value[:500]]
    if isinstance(value, tuple):
        return [sanitize_model_payload(item, local_roots=local_roots,
                                       field=field, depth=depth + 1)
                for item in value[:500]]
    if isinstance(value, str):
        return sanitize_model_payload_text(value, local_roots=local_roots)
    return value


def immutable_json(path: Path, value: Any) -> None:
    """Treat tuples/lists identically after JSON serialization, never redefine inputs."""
    if path.exists():
        if object_digest(read_json(path)) != object_digest(value):
            raise ValueError(f"immutable artifact changed: {path}")
        return
    atomic_json(path, value)


def event(path: Path, kind: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(redact(json.dumps({"time": now(), "event": kind, **payload}, ensure_ascii=False)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def freeze_files(paths: list[Path]) -> dict[str, str]:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen files missing: {missing}")
    return {str(p.absolute()): digest(p) for p in sorted(set(paths))}


def assert_frozen(hashes: dict[str, str]) -> None:
    bad = [p for p, h in hashes.items() if not Path(p).is_file() or digest(Path(p)) != h]
    if bad:
        raise RuntimeError(f"frozen protocol changed: {bad[:10]}")


def bounded_run(command: list[str] | str, *, cwd: Path, env: dict[str, str],
                timeout: int | float, shell: bool = False,
                input: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a bounded probe and terminate its whole process group on timeout."""
    from .process_executor import run_process
    result = run_process(command, cwd=cwd, env=env, timeout=float(timeout),
                         shell=shell, input=input)
    if result.error is not None:
        raise result.error
    if result.timed_out:
        raise subprocess.TimeoutExpired(command, timeout, result.stdout, result.stderr)
    # A launcher that exits zero while its workers continue has not completed the probe.
    return subprocess.CompletedProcess(command, 125 if result.orphaned_children else
                                       result.returncode,
                                       result.stdout, result.stderr)


def run_command(command: list[str], *, cwd: Path, env: dict[str, str],
                output: Path, timeout: int, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Never retry a partial run or infer success merely from exit code zero.

    ``metadata`` (the device binding, the job name) is stamped onto the record and onto
    every partial write of it, but never enters ``request``/``request_hash``: the timeout
    may shrink when a run is resumed, so the request hash already excludes it, and a job
    re-bound to a *different* device is a different fact that must be refused rather than
    silently overwritten.
    """
    metadata = dict(metadata or {})
    collisions = set(metadata) & {"command", "cwd", "timeout", "request_hash", "status",
                                  "started_at", "finished_at", "elapsed_seconds", "pid",
                                  "returncode", "error", "readings", "success_rate"}
    if collisions:
        raise ValueError(f"metadata may not shadow process record fields: {sorted(collisions)}")
    output.mkdir(parents=True, exist_ok=True)
    request = {"command": command, "cwd": str(cwd), "timeout": timeout}
    request_hash = object_digest(request)
    result_path = output / "process.json"
    if result_path.exists():
        previous = read_json(result_path)
        if previous["command"] != command or previous["cwd"] != str(cwd):
            raise RuntimeError(f"refusing to reuse output for a different command: {output}")
        conflicts = {key: value for key, value in metadata.items() if previous.get(key, value) != value}
        if conflicts:
            raise RuntimeError(f"refusing to rebind an existing process record: {output} {conflicts}")
        if metadata:
            atomic_json(result_path, {**previous, **metadata})
        if previous["status"] == "completed":
            return {**previous, **metadata}
        raise RuntimeError(f"prior process incomplete/failed; audit and use a new attempt directory: {output}")
    record = {**request, **metadata, "request_hash": request_hash, "started_at": now(), "status": "running"}
    atomic_json(result_path, record)
    started = time.monotonic()
    from .process_executor import run_process

    def started_process(process: subprocess.Popen[Any]) -> None:
        record["pid"] = process.pid
        atomic_json(result_path, record)

    with (output / "stdout.log").open("w", encoding="utf-8") as log:
        try:
            result = run_process(command, cwd=cwd, env=env, timeout=timeout,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 on_start=started_process)
            if result.error is not None:
                record.update(returncode=None, status="failed_to_start" if not result.launched
                              else "failed", error=f"{type(result.error).__name__}: {result.error}")
            elif result.timed_out:
                record.update(returncode=None, status="interrupted", error="TimeoutExpired")
            elif result.orphaned_children:
                record.update(returncode=result.returncode, status="failed",
                              error="launcher exited while child processes remained")
            else:
                record.update(returncode=result.returncode,
                              status="completed" if result.returncode == 0 else "failed")
        except (KeyboardInterrupt, SystemExit) as exc:
            record.update(returncode=None, status="interrupted", error=type(exc).__name__)
            raise
        finally:
            identity = output / "worker_identity.json"
            if identity.exists():
                record.update(read_json(identity))
            # What the program said about itself, kept. Before this the record of a finished
            # process was its exit code, its duration and a path to a log file -- so a
            # training run that printed its loss every hundred steps was recorded as having
            # printed nothing, and "did it learn anything" had no answer short of opening the
            # log by hand. Every process in the system goes through here: training,
            # evaluation, collection, environment builds.
            from .readings import named_numbers, success_rate, tail_of
            said = tail_of(output / "stdout.log")
            record.update(readings=named_numbers(said), success_rate=success_rate(said))
            record.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
            atomic_json(result_path, record)
    if record["status"] != "completed":
        raise RuntimeError(f"process {record['status']} ({record.get('returncode')}): {output / 'stdout.log'}")
    return record


#: Colon-separated list of extra directories to search for benchmark checkouts.
BENCHMARK_ROOT_ENV = "AUTOSIM_BENCHMARK_ROOT"


def benchmark_roots(workspace: Path) -> list[Path]:
    """Directories to search for a benchmark checkout, most specific first.

    Benchmarks are separate checkouts and where they live is the operator's choice:
    beside the project, grouped with others in a shared directory, or anywhere named
    by ``AUTOSIM_BENCHMARK_ROOT``. Deriving a single path from the project layout made
    moving a benchmark a code change; searching is what keeps it a preference.
    """
    # Absolute first: `Path(".").parent` is `Path(".")`, so a relative workspace silently
    # loses the parent as a search root and the nested walk resolves against the wrong
    # directory. Everything downstream assumes these are real locations.
    workspace = Path(workspace).expanduser().resolve()
    roots: list[Path] = []
    configured = os.environ.get(BENCHMARK_ROOT_ENV)
    if configured:
        roots.extend(Path(part).expanduser() for part in configured.split(os.pathsep) if part)
    roots.extend([workspace, workspace.parent])
    seen, unique = set(), []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            unique.append(resolved)
    return unique


def find_benchmark(name: str, workspace: Path, *, marker: str, depth: int = 2) -> Path | None:
    """Locate a checkout by name and a file that proves what it is.

    Searches each root directly and then one level of subdirectories, so a layout
    that groups benchmarks under a shared directory works as well as one that keeps
    them side by side. Nothing here assumes a particular depth or an ancestor count.
    """
    for root in benchmark_roots(workspace):
        direct = root / name
        if (direct / marker).exists():
            return direct.absolute()
        if depth > 1:
            try:
                children = sorted(child for child in root.iterdir() if child.is_dir())
            except OSError:
                continue
            for child in children:
                nested = child / name
                if (nested / marker).exists():
                    return nested.absolute()
    return None
