"""Independent, additive review of launched confirmation engineering failures.

A native launch is not proof of rollout, and a missing score is not proof of no
exposure. Only an independent review backed by positive execution/source evidence
and a sealed CPU repair probe can authorize the SAME frozen operation again.
"""
import re
import uuid
import time
from pathlib import Path

from .common import atomic_text, digest, immutable_json, now, object_digest, read_json
from .evidence_store import _log_slice


def _load(output, relative):
    path = output / relative
    if (path.is_symlink() or not path.resolve().is_relative_to(output.resolve()) or
            path.stat().st_size > 256 * 1024):
        raise ValueError('unsafe confirmation recovery record')
    return read_json(path), digest(path)


def packet(root: Path, output: Path, *, diagnosis_only: bool = False) -> dict:
    """Supervisor-private packet; raw native log never goes to the Scheduler/Fix."""
    records = {}
    row, sha = _load(output, (root / 'confirmation.json').relative_to(output))
    records[(root / 'confirmation.json').relative_to(output).as_posix()] = sha
    result_path = root / 'measurements/confirmation.json'
    result, sha = _load(output, result_path.relative_to(output))
    records[result_path.relative_to(output).as_posix()] = sha
    if row.get('ok') is not False or result.get('ok') is not False:
        raise ValueError('not a failed confirmation')
    if not diagnosis_only and any(result.get(key) is not None for key in ('metric_value', 'baseline_metric_value')):
        raise ValueError('a scored held-out result cannot enter engineering recovery')
    for key in ('candidate_label', 'settings', 'baseline_evaluate'):
        if row.get(key) != result.get(key):
            raise ValueError('confirmation projections disagree')
    from .experiment_bundle import artifact_identity
    for key in ('policy_artifact', 'baseline_policy_artifact'):
        artifact = result.get(key) or {}
        if artifact:
            path = Path(artifact['path'])
            if not path.resolve().is_relative_to(output):
                raise ValueError('frozen policy escaped run')
            field, identity = artifact_identity(path)
            if identity != artifact.get(field):
                raise ValueError('frozen policy bytes changed')
    attempts = sorted((root / 'confirmation_attempts').glob('*.json'))
    if not attempts or len(attempts) > 64:
        raise ValueError('missing or excessive immutable attempt history')
    native = []
    for path in attempts:
        old, sha = _load(output, path.relative_to(output))
        records[path.relative_to(output).as_posix()] = sha
        if not diagnosis_only and (old.get('ok') is True or any(old.get(k) is not None for k in ('metric_value', 'baseline_metric_value'))):
            raise ValueError('historical held-out score prevents setup recovery')
        if (old.get('held_out_exposure') == 'unknown' and not old.get('attempt_id') and
                not (old.get('baseline_evaluate') or {}).get('attempt_id')):
            raise ValueError('unknown historic exposure without native evidence')
        for value in (old.get('baseline_evaluate') or {}, {'attempt_id': old.get('attempt_id')}):
            identity = value.get('attempt_id')
            if not identity:
                continue
            if not re.fullmatch('[0-9a-f]{32}', str(identity)):
                raise ValueError('invalid native attempt')
            evidence, evidence_sha = _load(output, Path('evidence') / (identity + '.json'))
            records['evidence/' + identity + '.json'] = evidence_sha
            receipt, receipt_sha = _load(output, Path(evidence['receipt_ref']))
            records[evidence['receipt_ref']] = receipt_sha
            if diagnosis_only and receipt.get('returncode') == 0:
                continue
            def missing_metric(value):
                return not value or (isinstance(value, dict) and
                    value.get('status') == 'missing' and value.get('matched') == 0 and
                    not value.get('candidate_paths') and not value.get('path'))
            if (receipt.get('attempt_id') != identity or receipt.get('ran') is not True or
                    receipt.get('returncode') in (None, 0) or
                    (not diagnosis_only and (not missing_metric(evidence.get('metric_artifact')) or
                    not missing_metric(receipt.get('metric_artifact'))))):
                raise ValueError('not a sealed unsuccessful native launch')
            log = output / evidence['log_ref']
            if not log.resolve().is_relative_to(output.resolve()):
                raise ValueError('native log escaped run')
            _, _, raw = _log_slice(log, offset=None, limit=12000,
                expected_bytes=evidence['log_bytes'], expected_sha256=evidence['log_sha256'])
            records[evidence['log_ref']] = evidence['log_sha256']
            observation = {}
            observation_ref = 'native_stack_observations/' + identity + '.json'
            if (output / observation_ref).exists():
                observation, sha = _load(output, Path(observation_ref))
                if observation.get('attempt_id') != identity:
                    raise ValueError('native stack observation belongs to another attempt')
                records[observation_ref] = sha
                stack_log = observation['log_ref']
                if _load_hash(output, stack_log) != observation['log_sha256']:
                    raise ValueError('native stack observation changed')
                records[stack_log] = observation['log_sha256']
                _, _, stack = _log_slice(output / stack_log, offset=None, limit=12000,
                    expected_sha256=observation['log_sha256'])
                observation = {**observation, 'frames': stack.decode(errors='replace')}
            repair = {}
            repair_ref = 'native_repair_evidence/' + identity + '.json'
            if (output / repair_ref).exists():
                repair, sha = _load(output, Path(repair_ref))
                if repair.get('failed_attempt_id') != identity:
                    raise ValueError('repair evidence belongs to another native attempt')
                records[repair_ref] = sha
                repair_id = repair['evidence_id']
                if not re.fullmatch('[0-9a-f]{32}', str(repair_id)):
                    raise ValueError('invalid related repair evidence')
                observed, sha = _load(output, Path('evidence') / (repair_id + '.json'))
                records['evidence/' + repair_id + '.json'] = sha
                detail, sha = _load(output, Path(observed['receipt_ref']))
                records[observed['receipt_ref']] = sha
                if _load_hash(output, observed['log_ref']) != observed['log_sha256']:
                    raise ValueError('related repair log changed')
                records[observed['log_ref']] = observed['log_sha256']
                repair = {**repair, 'native_receipt': detail, 'log_sha256': observed['log_sha256']}
            state = (old.get('baseline_source_state') if identity ==
                     (old.get('baseline_evaluate') or {}).get('attempt_id') else old.get('source_state'))
            if not state:
                # Legacy failed post-run tree checks discarded the expected identity.
                # Recover the measured reference only; probe_connections must still
                # verify the entire copied tree before it can issue any CPU proof.
                label = 'baseline' if identity == (old.get('baseline_evaluate') or {}).get('attempt_id') else old.get('candidate_label')
                if isinstance(label, str) and re.fullmatch('[A-Za-z0-9_-]{1,80}', label):
                    measured_ref = (root / 'measurements' / (label + '.json')).relative_to(output)
                    if (output / measured_ref).exists():
                        measured, sha = _load(output, measured_ref)
                        records[measured_ref.as_posix()] = sha
                        state = measured.get('source_state') if measured.get('ok') is True else None
            native.append({'attempt_id': identity, 'receipt': receipt, 'source_state': state,
                           'terminal_log': raw.decode('utf-8', errors='replace'),
                           'readonly_process_observation': observation,
                           'related_engineering_repair': repair})
    if not native:
        raise ValueError('no launched engineering failure to review')
    for name in ('comparison_protocol.json', 'protocol_frozen.json'):
        frozen, sha = _load(output, (root / name).relative_to(output))
        records[(root / name).relative_to(output).as_posix()] = sha
        if name == 'protocol_frozen.json':
            for path, expected in (frozen.get('hashes') or {}).items():
                relative = Path(path).relative_to(output).as_posix()
                if _load_hash(output, relative) != expected:
                    raise ValueError('protected source changed before engineering review')
                records[relative] = expected
    return {'subject_sha256': object_digest(records), 'records': records,
            'native_failures': native, 'authority': 'Supervisor private; not research feedback'}


def engineering_summary(root: Path, output: Path) -> dict:
    """Keep only traceback locations/types; never arbitrary exception text or scores."""
    private = packet(root, output, diagnosis_only=True)
    frames = []
    exceptions = []
    workspace = read_json(output / 'workspace_snapshot.json') if (output / 'workspace_snapshot.json').is_file() else {}
    development = Path(workspace.get('destination') or output / 'checkout')
    for failure in private['native_failures']:
        text = failure['terminal_log']
        trace = text[text.rfind('Traceback (most recent call last):'):]
        if not text or not trace.startswith('Traceback'):
            assertions = re.findall(r'Assertion at\s*:\s*([^\n]+?):(\d+)\s*:', text)
            symbols = re.findall(r'#\d+\s*:\s*([A-Za-z_][A-Za-z0-9_:]*)\(', text)
            if assertions:
                exceptions.append('NativeAssertion')
                for path, line in assertions[-4:]:
                    frames.append({'file': Path(path).name, 'source_ref': Path(path).name + ':' + line,
                                   'origin': 'external_dependency', 'line': int(line),
                                   'function': symbols[0] if symbols else 'native_assertion'})
            continue
        for path, line, function in re.findall(r'  File "([^"\n]+)", line (\d+), in ([^\n]+)', trace):
            if len(frames) >= 32:
                break
            frame = Path(path)
            failed_checkout = Path(failure['receipt'].get('cwd') or output / 'missing')
            origin, ref = 'external_dependency', frame.name
            for base, category in ((failed_checkout, 'failed_confirmation_checkout'),
                                   (development, 'development_checkout')):
                if frame.is_absolute() and frame.is_relative_to(base):
                    origin, ref = category, frame.relative_to(base).as_posix()
                    break
            frames.append({'file': frame.name, 'source_ref': ref + ':' + line,
                           'origin': origin, 'line': int(line), 'function': function[:100]})
        names = re.findall(r'^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):', trace, re.M)
        exceptions.extend(names[-1:])
    return {'subject_sha256': private['subject_sha256'], 'traceback_frames': frames,
            'exception_types': exceptions, 'exposure': 'requires_independent_review',
            'guidance': 'Diagnose the cited source/API/resource connection on CPU using development inputs. '
                        'Do not read held-out logs, run rollout, change policy/protocol, or retry confirmation.'}


def probe_connections(root: Path, output: Path, *, interpreter: Path, original_repo: Path,
                      snapshots, timeout: float = 30) -> dict:
    """Probe the failed frozen source namespace, not the working checkout or held-out argv.

    This only verifies import roots/resource mounts. The Supervisor still decides
    whether this covers the original failure; it is not a simulator reset test.
    """
    from .environment_overlay import isolated_source_bindings, verify_isolated_source_bindings
    from .workspace_snapshot import verify_state
    from .evidence_store import capture_attempt_evidence
    private = packet(root, output)
    observations = []
    deadline = time.monotonic() + timeout
    runtime_records = {}
    for failure in private['native_failures']:
        checkout = Path(failure['receipt']['cwd'])
        if (not checkout.resolve().is_relative_to(output / 'confirmation_sources') or
                checkout.name != 'checkout'):
            raise ValueError('original frozen confirmation checkout unavailable')
        state = failure.get('source_state')
        if not isinstance(state, dict) or not state:
            raise ValueError('failed confirmation lacks verifiable source state')
        manifest = read_json(checkout.parent / 'workspace_snapshot.json')
        if not verify_state(checkout, manifest, state, snapshots):
            raise ValueError('failed source copy changed')
        binding = isolated_source_bindings(interpreter, output=output,
                                          original_repo=original_repo, repo=checkout)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError('CPU connection probe window exhausted')
        observed = verify_isolated_source_bindings(interpreter, output=output, repo=checkout,
                                                  binding=binding, timeout=remaining)
        if observed['status'] != 'verified' or not verify_state(checkout, manifest, state, snapshots):
            raise ValueError('CPU connection probe not verified or source changed')
        observations.append({'attempt_id': failure['attempt_id'], **observed})
        for path in (checkout.parent / 'workspace_snapshot.json', Path(binding['source']),
                     Path(binding['target']), Path(binding['prefix']) / 'overlay.json'):
            runtime_records[path.relative_to(output).as_posix()] = digest(path)
    identity = uuid.uuid4().hex
    relative = 'native_diagnostics/' + identity
    row = {'status': 'completed', 'returncode': 0, 'gpu_access': False,
           'repository_source_stable': True, 'failure_subject_sha256': private['subject_sha256'],
           'runtime_records': runtime_records,
           'capability': 'frozen_source_import_resource_connections_only', 'observations': observations}
    immutable_json(output / (relative + '.json'), row)
    import json
    atomic_text(output / (relative + '.log'), json.dumps(row, ensure_ascii=False))
    evidence = capture_attempt_evidence(output, attempt_id=identity,
        log=output / (relative + '.log'), receipt_ref=relative + '.json', status='completed',
        returncode=0, termination_reason='cpu_native_diagnostic')
    return {'evidence_id': identity, 'evidence_ref': 'evidence/' + identity + '.json',
            'scope': row['capability'], 'observations': observations}


def review(root: Path, output: Path, *, diagnostic_evidence_id: str, reviewer) -> dict:
    """The controller calls a separate Supervisor; a caller cannot supply its verdict."""
    private = packet(root, output)
    if not re.fullmatch('[0-9a-f]{32}', diagnostic_evidence_id):
        raise ValueError('CPU diagnostic evidence ID required')
    evidence, sha = _load(output, Path('evidence') / (diagnostic_evidence_id + '.json'))
    receipt, receipt_sha = _load(output, Path(evidence['receipt_ref']))
    if evidence.get('termination_reason') == 'operation_contract_failed':
        # Accept the stable diagnosis envelope as well as its CPU leaf; resolve
        # only the executor-sealed, hash-bound edge, never a caller-supplied path.
        original, original_sha = _load(output, Path(receipt.get('_local_input_ref') or 'invalid'))
        if (receipt.get('operation') != 'confirm_best_engineering' or
                original_sha != receipt.get('_local_input_sha256') or
                original.get('operation') != 'confirm_best_engineering' or
                (original.get('context') or {}).get('subject_sha256') != private['subject_sha256']):
            raise ValueError('diagnosis envelope does not bind this confirmation failure')
        private['records'].update({'evidence/' + diagnostic_evidence_id + '.json': sha,
            evidence['receipt_ref']: receipt_sha, receipt['_local_input_ref']: original_sha})
        diagnostic_evidence_id = ((original['context'].get('connection_probe') or {}).get('evidence_id'))
        if not re.fullmatch('[0-9a-f]{32}', str(diagnostic_evidence_id or '')):
            raise ValueError('diagnosis has no successful CPU probe; obtain cpu_native_diagnostic evidence first')
        evidence, sha = _load(output, Path('evidence') / (diagnostic_evidence_id + '.json'))
        receipt, receipt_sha = _load(output, Path(evidence['receipt_ref']))
    if (evidence.get('termination_reason') != 'cpu_native_diagnostic' or
            receipt.get('gpu_access') is not False or receipt.get('returncode') != 0 or
            receipt.get('repository_source_stable') is not True):
        raise ValueError('repair requires successful source-stable cpu_native_diagnostic evidence; '
                         'use the sealed diagnosis context.connection_probe.evidence_id')
    if receipt.get('failure_subject_sha256') not in (None, private['subject_sha256']):
        raise ValueError('CPU repair probe belongs to another confirmation failure')
    for relative, sha in (receipt.get('runtime_records') or {}).items():
        if _load_hash(output, relative) != sha:
            raise ValueError('CPU repair runtime changed')
        private['records'][relative] = sha
    log = output / evidence['log_ref']
    if not log.resolve().is_relative_to(output.resolve()):
        raise ValueError('CPU diagnostic log escaped run')
    _, _, raw = _log_slice(log, offset=None, limit=12000,
        expected_bytes=evidence['log_bytes'], expected_sha256=evidence['log_sha256'])
    private['records'].update({'evidence/' + diagnostic_evidence_id + '.json': sha,
                              evidence['receipt_ref']: receipt_sha,
                              evidence['log_ref']: evidence['log_sha256']})
    verdict = reviewer({**private, 'diagnostic_evidence_id': diagnostic_evidence_id,
                        'repair_probe': {'receipt': receipt,
                                                   'terminal_log': raw.decode(errors='replace')}})
    if packet(root, output)['subject_sha256'] != private['subject_sha256']:
        raise ValueError('confirmation evidence changed during independent review')
    required = ('setup_only_positive_evidence', 'no_rollout_or_metric_exposure',
                'repair_probe_covers_original_failure', 'frozen_policy_protocol_preserved')
    approved = (isinstance(verdict, dict) and verdict.get('verdict') == 'setup_only' and
                all(verdict.get(k) is True for k in required) and
                isinstance(verdict.get('source_refs'), list) and bool(verdict['source_refs']) and
                isinstance(verdict.get('positive_evidence'), str) and
                1 <= len(verdict['positive_evidence']) <= 12000)
    source_hashes = verdict.pop('_source_hashes', {}) if isinstance(verdict, dict) else {}
    for path, sha in source_hashes.items():
        relative = Path(path).relative_to(output).as_posix()
        if _load_hash(output, relative) != sha:
            raise ValueError('review source changed')
        private['records'][relative] = sha
    reused = verdict.get('reused_independent_review_ref') if isinstance(verdict, dict) else None
    if reused:
        if not str(reused).startswith((root / 'confirmation_execution_reviews').relative_to(output).as_posix() + '/'):
            raise ValueError('invalid cached independent review reference')
        _, sha = _load(output, Path(reused))
        private['records'][reused] = sha
    format_issues = []
    if not isinstance(verdict, dict):
        format_issues.append('review must be an object')
    elif verdict.get('verdict') == 'setup_only' and not approved:
        if not isinstance(verdict.get('positive_evidence'), str) or not 1 <= len(verdict.get('positive_evidence', '')) <= 12000:
            format_issues.append('positive_evidence must be 1..12000 characters')
        if not isinstance(verdict.get('source_refs'), list) or not verdict.get('source_refs'):
            format_issues.append('source_refs must contain exact source citations')
    row = {'schema_version': 1, 'at': now(), 'subject_sha256': private['subject_sha256'],
           'records': private['records'], 'diagnostic_evidence_id': diagnostic_evidence_id,
           'review_authority': 'independent_supervisor', 'approved': approved,
           'review': verdict, 'validation_issues': format_issues,
           'scope': 'same frozen confirmation only; development stays closed'}
    identity = object_digest(row)
    immutable_json(root / 'confirmation_execution_reviews' / (identity + '.json'), row)
    return {'approved': approved, 'review_id': identity,
            'because': 'reviewed setup-only recovery' if approved else
                       'review format invalid: ' + '; '.join(format_issues) if format_issues else
                       'exposure or repair not established'}


def cached_review(root: Path, output: Path, *, diagnostic_evidence_id: str) -> tuple[dict, str] | None:
    """Reuse a real independent response only for unchanged reviewed bytes/CPU input.

    Useful after presentation-validation corrections: do not buy another identical
    Supervisor turn or change the original failed validation record.
    """
    private = packet(root, output)
    for path in sorted((root / 'confirmation_execution_reviews').glob('*.json'))[-64:]:
        row, _ = _load(output, path.relative_to(output))
        if (row.get('review_authority') != 'independent_supervisor' or
                row.get('subject_sha256') != private['subject_sha256'] or
                row.get('diagnostic_evidence_id') != diagnostic_evidence_id):
            continue
        records = row.get('records') or {}
        if (all(records.get(k) == v for k, v in private['records'].items()) and
                all(_load_hash(output, k) == v for k, v in records.items()) and
                isinstance(row.get('review'), dict)):
            return row['review'], path.relative_to(output).as_posix()
    return None


def approved(root: Path, output: Path) -> bool:
    """Every historical launch must be reviewed; changing any evidence closes the gate."""
    try:
        reviews = sorted((root / 'confirmation_execution_reviews').glob('*.json'))
        if not reviews:
            return False
        private = packet(root, output)
        for path in reviews[-64:]:
            row, _ = _load(output, path.relative_to(output))
            if (row.get('review_authority') != 'independent_supervisor' or
                    row.get('approved') is not True or
                    row.get('subject_sha256') != private['subject_sha256']):
                continue
            records = row.get('records') or {}
            if all(records.get(k) == v for k, v in private['records'].items()) and all(
                    _load_hash(output, k) == v for k, v in records.items()):
                return True
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return False


def _load_hash(output, relative):
    path = output / relative
    if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
        raise ValueError('unsafe recovery evidence')
    return digest(path)
