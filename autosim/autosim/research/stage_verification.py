"""Repository-independent resource handoff and live native verification evidence."""
from contextlib import nullcontext
from pathlib import Path
import json
import subprocess
import os
import time
import uuid
from datetime import datetime, timezone

from .common import atomic_json, now, redact, read_json
from .workspace_resources import bindings_for


def resource_context(repo: Path) -> dict:
    """Expose bounded names/kinds from bindings, never data or guessed dataset selection."""
    from .resource_inventory import inspect
    rows = []
    for binding in bindings_for(repo):
        top = inspect(repo.parent, target=binding['target'], limit=24)
        children = []
        for entry in top['entries']:
            if entry['kind'] == 'directory' and len(children) < 8:
                sub = inspect(repo.parent, target=binding['target'], directory=entry['name'], limit=16)
                children.append({'directory':entry['name'], 'entries':sub['entries'],
                                 'evidence_ref':f"resource_inspections/{sub['id']}.json"})
        rows.append({'target':binding['target'], 'entries':top['entries'],
                     'children':children, 'truncated':top['truncated'],
                     'evidence_ref':f"resource_inspections/{top['id']}.json"})
    return {'resources':rows, 'instruction':
        'Bindings are visible at i["repo"] + "/" + target inside native execution. '
        'Host readers see empty placeholders, not missing data. Choose the precise native '
        'consumer root from source and metadata, including required nested dataset directories; '
        'do not reuse the original repository default when its path was not bound. '
        'These names are discovery evidence, not loader readiness or an automatic selection. '
        'If none is compatible, request acquisition explicitly rather than silently falling '
        'back to online downloads. No resource contents are included.'}


def missing_checkout_paths(argv, *, repo: Path, output: Path | None) -> list[str]:
    """Catch nonexistent local inputs before an SDK silently falls back to the network."""
    bindings = bindings_for(repo)
    missing = []
    for token in argv:
        value = str(token).split('=', 1)[-1].strip('"\'')
        path = Path(value)
        if not path.is_absolute() or not path.is_relative_to(repo):
            continue
        if output is not None and path.is_relative_to(output):
            continue
        actual = path
        for binding in bindings:
            target = repo / binding['target']
            if path.is_relative_to(target):
                actual = Path(binding['source']) / path.relative_to(target)
                break
        if not actual.exists():
            missing.append(str(path.relative_to(repo)))
    return sorted(set(missing))


def recent_attempts(output: Path, limit: int = 6) -> list[dict]:
    """Read receipt-bound native evidence; this is NOT formal measurement history."""
    from .evidence_store import read_attempt_evidence
    rows = []
    files = sorted((output/'derivation_native').glob('*/receipt.json'),
                   key=lambda path:path.stat().st_mtime, reverse=True)
    for path in files[:limit]:
        try:
            if (path.is_symlink() or path.parent.is_symlink() or
                    not path.resolve().is_relative_to(output.resolve()) or path.stat().st_size > 98304):
                continue
            record = read_json(path)
            identity = path.parent.name
            if record.get('id') != identity:
                continue
            row = {key:record.get(key) for key in ('id','stage','status','started_at',
                'finished_at','seconds','argv','verification_output')}
            row['receipt_ref'] = path.relative_to(output).as_posix()
            if record.get('evidence_id') == identity:
                evidence_path = output/'evidence'/f'{identity}.json'
                if evidence_path.is_symlink():
                    continue
                sealed = read_json(evidence_path)
                verified = read_attempt_evidence(output, identity,
                    offset=max(0,int(sealed.get('log_bytes') or 0)-1600), limit=1600)
                row.update(status=verified['status'],returncode=verified['returncode'],
                    evidence_id=identity,evidence_ref=f'evidence/{identity}.json',
                    excerpt=verified['text'],evidence_verified=True)
            else:
                row.update(status='running_unverified',evidence_verified=False)
                # Live output is deliberately separate from sealed evidence and
                # never supplies a return code, completion or measurement verdict.
                log = path.parent/'native.log'
                if (not log.is_symlink() and log.is_file() and
                        log.resolve().is_relative_to(output.resolve())):
                    with os.fdopen(os.open(log, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
                        size = stream.seek(0, 2)
                        stream.seek(max(0, size-4096))
                        tail = stream.read(4096).decode('utf-8', 'replace')
                    row['live_telemetry'] = {'tail':tail, 'observed_at':now(),
                        'observed_bytes':size, 'log_ref':log.relative_to(output).as_posix(),
                        'authority':'unsealed live excerpt; not completion, liveness or score evidence'}
                try:
                    started = datetime.fromisoformat(str(record.get('started_at') or ''))
                    if started.tzinfo is not None:
                        row['elapsed_wall_seconds'] = max(0, (
                            datetime.now(timezone.utc)-started).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    pass
            row['authority'] = 'native stage verification, not baseline/candidate score'
            row['receipt_scope'] = (
                'Native launch/exit/capture only. Absence of artifact or metric_artifact '
                'fields is not an artifact/identity validation verdict. Read the actual '
                'derivation rejection before inferring a postcondition failure.')
            rows.append(row)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return rows


def run_native(argv, *, repo: Path, output: Path, cwd: Path, env: dict,
               timeout: float, stdin: str | None, window=None,
               stage: str = '', verification_output: str = ''):
    """Stream the real output to a receipt-backed log, including partial timeout output."""
    from .common import isolated_argv
    from .process_executor import run_process_stream
    from .evidence_store import capture_attempt_evidence
    identity = uuid.uuid4().hex
    directory = output / 'derivation_native' / identity
    directory.mkdir(parents=True, exist_ok=False)
    log = directory / 'native.log'
    receipt_path = directory / 'receipt.json'
    receipt_ref = receipt_path.relative_to(output).as_posix()
    record = {'id':identity, 'status':'running', 'started_at':now(),
              'stage':stage, 'verification_output':verification_output,
              'argv':list(argv), 'cwd':str(cwd), 'timeout_seconds':timeout,
              'log_ref':log.relative_to(output).as_posix()}
    atomic_json(receipt_path, record)
    started = time.monotonic()
    attempt = None
    context = window(timeout) if window is not None else nullcontext((timeout, False, {}))
    try:
        with log.open('w', encoding='utf-8') as stream, context as (allowed, gpu, device_env):
            def line(text):
                stream.write(redact(text) + '\n')
                stream.flush()
            record.update(gpu_allowed=gpu, effective_timeout_seconds=allowed)
            atomic_json(receipt_path, record)
            attempt = run_process_stream(
                isolated_argv(list(argv), output=output, repo=repo,
                              native_environment=env,
                              require_pid_namespace=True, allow_gpu=gpu),
                cwd=cwd, env={**env, **device_env, 'PYTHONUNBUFFERED':'1'}, timeout=allowed,
                input_bytes=stdin.encode() if stdin is not None else None,
                on_stdout_line=line, on_stderr_line=line)
        if attempt.error is not None:
            attempt.error.evidence_ref = f'evidence/{identity}.json'
            raise attempt.error
        if attempt.timed_out:
            error = subprocess.TimeoutExpired(argv, timeout, attempt.stdout, attempt.stderr)
            error.evidence_ref = f'evidence/{identity}.json'
            raise error
        result = subprocess.CompletedProcess(argv, 125 if attempt.orphaned_children else
                                             attempt.returncode, attempt.stdout, attempt.stderr)
        result.evidence_ref = f'evidence/{identity}.json'
        result.native_returncode = attempt.returncode
        result.native_cleanup = {'orphaned_children':attempt.orphaned_children,
                                 'containment_mode':attempt.containment_mode,
                                 'audit':attempt.cleanup_diagnostics}
        return result
    finally:
        record.update(finished_at=now(), seconds=time.monotonic()-started,
            status=('interrupted' if attempt is None else 'timed_out' if attempt.timed_out else
                    'failed' if attempt.error or attempt.orphaned_children or attempt.returncode else 'completed'),
            returncode=attempt.returncode if attempt else None)
        if attempt is not None:
            record.update(effective_returncode=125 if attempt.orphaned_children else attempt.returncode,
                orphaned_children=attempt.orphaned_children,
                containment_mode=attempt.containment_mode,
                cleanup_diagnostics=attempt.cleanup_diagnostics,
                capture_error=({'type':type(attempt.error).__name__,
                                'message':redact(str(attempt.error))[:1500]}
                               if attempt.error is not None else None),
                stdout_tail=redact(str(attempt.stdout or ''))[-2000:],
                stderr_tail=redact(str(attempt.stderr or ''))[-4000:])
            if attempt.error is not None or attempt.orphaned_children:
                # Seal the controller-side reason alongside native output: a zero
                # native exit otherwise leaves the next Agent only an unexplained
                # failed status. This is execution audit, never a reported metric.
                audit = {key: record[key] for key in ('returncode', 'effective_returncode',
                    'capture_error', 'orphaned_children', 'containment_mode',
                    'cleanup_diagnostics')}
                with log.open('a', encoding='utf-8') as stream:
                    stream.write('\n[AutoSim execution audit; not a metric] ' +
                                 json.dumps(audit, ensure_ascii=False) + '\n')
        if log.is_file():
            record.update(capture_attempt_evidence(output, attempt_id=identity, log=log,
                receipt_ref=receipt_ref, status=record['status'], returncode=record['returncode'],
                termination_reason=record['status']))
        atomic_json(receipt_path, record)
