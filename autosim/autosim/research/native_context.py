"""Executor-owned environment identity for read-only native CPU diagnostics.

Paths are never supplied by the model. Configuration bindings are run-owned and
read-only; credentials and the rest of the output directory are not mounted.
"""
from pathlib import Path
import shlex
from .common import atomic_json, object_digest, now, read_json


def inspection_availability(output: Path):
    """Published interpreter context is not proof that CLI exposes its tool."""
    context = Path(output) / 'native_context.json'
    published = context.is_file() and not context.is_symlink()
    tool = 'mcp__autosim_exec__inspect_native_environment'
    path = Path(output) / 'agent/tool_capabilities.json'
    row = read_json(path) if path.is_file() and not path.is_symlink() and path.stat().st_size <= 32000 else {}
    if tool in (row.get('advertised_tools') or []):
        status = 'advertised'
    elif tool in (row.get('configured_tools') or []):
        status = 'missing_from_cli'
    else:
        status = 'not_observed_for_capable_role'
    return {'available': published and status == 'advertised', 'context_published': published,
        'cli_tool_status': status, 'observed_role': row.get('role'),
        'guidance': 'Context publication is not CLI tool discovery or environment readiness. '
            'A missing configured tool is a tool-schema/runtime issue, not a reason to reinstall SDK. '
            'Native execution revalidates context and authorized source bindings.'}


def publish_context(output: Path, repo: Path, interpreter: Path, environment: dict):
    output = output.resolve()
    # Keep only run-local path configuration, never arbitrary inherited secrets.
    paths = {k: str(v) for k, v in environment.items() if
             (k == "HOME" or k.endswith(("_CONFIG_PATH", "_CONFIG_HOME"))) and
             Path(str(v)).is_absolute() and Path(str(v)).resolve().is_relative_to(output)}
    record = {"schema_version": 1, "repo": str(repo.resolve()),
              "interpreter": str(interpreter.absolute()), "paths": paths,
              "capability": "cpu_diagnostic_not_simulation_readiness"}
    from .environment_overlay import readonly_roots, overlay_identity, runtime_library_dirs
    record['dependency_roots'] = [str(p) for p in readonly_roots(interpreter.parent.parent, output, repo)]
    identity = overlay_identity(interpreter.parent.parent)
    if identity is not None:
        record['overlay_identity'] = identity
    libraries = runtime_library_dirs(interpreter.parent.parent, output, repo)
    if libraries:
        record['runtime_library_dirs'] = libraries
    record["identity"] = object_digest(record)
    atomic_json(output / "native_context.json", {**record, "updated_at": now()})
    return record


def load_context(output: Path, workspace: Path):
    path = output / "native_context.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("native context has not been published")
    row = read_json(path)
    if not isinstance(row, dict) or not isinstance(row.get("paths"), dict) or not isinstance(row.get("interpreter"), str):
        raise ValueError("invalid native context record")
    if row.get("repo") != str(workspace.resolve()) or row.get("schema_version") != 1:
        raise ValueError("native context belongs to a different checkout")
    record = {k: v for k, v in row.items() if k not in {"identity", "updated_at"}}
    if object_digest(record) != row.get("identity"):
        raise ValueError("native context identity changed")
    python = Path(row["interpreter"])
    prefix = python.parent.parent
    if (not python.is_file() or prefix == output or prefix.is_symlink() or
            not prefix.resolve().is_relative_to(output.resolve())):
        raise ValueError("native interpreter must be inside this run's environment")
    for key, value in row["paths"].items():
        if (not isinstance(key, str) or not isinstance(value, str) or not
                (key == "HOME" or key.endswith(("_CONFIG_PATH", "_CONFIG_HOME")))):
            raise ValueError("invalid native configuration mapping")
        target = Path(value)
        if not target.is_absolute() or target.is_symlink() or not target.resolve().is_relative_to(output.resolve()) or target == output:
            raise ValueError("native configuration path escaped this run")
    from .environment_overlay import readonly_roots, overlay_identity, runtime_library_dirs
    if row.get('dependency_roots', []) != [str(p) for p in readonly_roots(prefix, output, workspace)]:
        raise ValueError('native dependency bindings changed; republish after controlled repair')
    current_overlay = overlay_identity(prefix)
    if row.get('overlay_identity') is not None and row['overlay_identity'] != current_overlay:
        raise ValueError('native overlay identity changed; republish after controlled repair')
    if row.get('runtime_library_dirs', []) != runtime_library_dirs(prefix, output, workspace):
        raise ValueError('native loader bindings changed; republish and revalidate consumers')
    return row


def publish_interpreter_hint(output: Path, repo: Path, interpreter: Path | None):
    """A safe run-local hint permits CPU source reading, not environment admission.

    Otherwise a fresh study cannot inspect the known SDK until an environment build,
    while its protocol survey needs those SDK sources before any native evaluation.
    External paths are still not grants; CLI constructs the owned overlay first.
    """
    output, repo = Path(output).resolve(), Path(repo).resolve()
    path = output/'native_context.json'
    if path.exists() or path.is_symlink():
        return {'status':'existing_context_preserved', 'readiness':'requires_native_consumers'}
    if interpreter is None:
        return {'status':'no_hint', 'readiness':'unverified'}
    python = Path(interpreter).absolute()
    prefix = python.parent.parent
    if (not python.is_file() or prefix.is_symlink() or prefix == output
            or not prefix.resolve().is_relative_to(output)):
        return {'status':'hint_not_run_owned', 'readiness':'unverified'}
    from .environment_overlay import readonly_roots
    try:
        readonly_roots(prefix, output, repo)
        record = publish_context(output, repo, python, {})
        load_context(output, repo)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        return {'status':'hint_context_refused', 'why':str(exc)[:500], 'readiness':'unverified'}
    result = {'status':'diagnostic_hint_published', 'context_identity':record['identity'],
              'readiness':'unverified_requires_native_consumers', 'gpu_access':False,
              'authority':'provided run-owned interpreter for read-only CPU/source inspection only'}
    atomic_json(output/'native_context_hint.json',result)
    return result


def record_command_configuration(output: Path, command: str, environment: dict):
    """Persist explicit leading run-local config assignments, never parse Python text."""
    path = output / "native_context.json"
    if not path.is_file() or path.is_symlink():
        return
    try:
        tokens = shlex.split(command)
    except ValueError:
        return
    if tokens and tokens[0] == "env":
        tokens = tokens[1:]
    overrides = {}
    for token in tokens:
        if "=" not in token:
            break
        key, value = token.split("=", 1)
        if key.endswith(("_CONFIG_PATH", "_CONFIG_HOME")):
            target = Path(value)
            if target.is_absolute() and target.resolve().is_relative_to(output.resolve()):
                overrides[key] = value
    if overrides:
        row = read_json(path)
        publish_context(output, Path(row["repo"]), Path(row["interpreter"]),
                        {**environment, **row["paths"], **overrides})
