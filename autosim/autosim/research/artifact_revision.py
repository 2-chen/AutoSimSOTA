"""Explicit, source-reviewed output locators; never infer parents or change judges.

A marker file proves only that file was found. Changing its scope is a new
declaration which needs a new native verification, not a reinterpretation of
an old receipt. The Agent chooses the format-independent path/glob.
"""
from pathlib import Path

from .common import immutable_json, now, object_digest


class ArtifactRevisionRefused(ValueError):
    def __init__(self, message, ref, verdict):
        super().__init__(message)
        self.review_ref, self.review = ref, verdict


def validate(stage, revision):
    from .declarative_backend import artifact_pattern_problem
    if stage not in {'collect', 'prepare_data', 'train', 'evaluate'}:
        raise ValueError('artifact_revision requires a declared native output stage')
    if (not isinstance(revision, dict) or
            set(revision) != {'pattern', 'reason', 'source_refs'}):
        raise ValueError('artifact_revision requires exactly pattern, reason, source_refs')
    pattern, reason = revision['pattern'], revision['reason']
    if (not isinstance(pattern, str) or not pattern or pattern != pattern.strip()
            or len(pattern) > 512 or '\\' in pattern or '\x00' in pattern
            or Path(pattern).is_absolute() or '..' in Path(pattern).parts
            or not Path(pattern).parts
            or any(char in Path(pattern).parts[0] for char in '*?[')
            or artifact_pattern_problem(pattern)):
        raise ValueError('artifact_revision needs a confined relative path/glob with a literal top-level directory')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError('artifact_revision requires a concise nonempty reason')
    refs = revision['source_refs']
    if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
        raise ValueError('artifact_revision requires 1..12 exact source citations')
    return dict(revision)


def assert_prebaseline(output, run_id='derived'):
    """Policy/score locators can be corrected only before formal research starts."""
    import re
    output = Path(output)
    if (not isinstance(run_id, str) or run_id in {'.', '..'} or
            not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', run_id)):
        raise ValueError('invalid research identity for output locator review')
    root = output / 'research' / run_id
    for path in (output, output/'research', root, root/'measurements'):
        if path.is_symlink():
            raise ValueError('unsafe research boundary for output locator review')
        if path.exists() and not path.is_dir():
            raise ValueError('research boundary must contain directories, not files')
    paths = [root/'controller_session.json', root/'comparison_protocol.json',
             root/'research_report.json']
    if (any(path.exists() or path.is_symlink() for path in paths) or
            ((root/'measurements').exists() and any((root/'measurements').iterdir()))):
        raise ValueError('train/evaluate output locator revision is pre-baseline only; '
                         'frozen measurements and research contracts cannot be changed')


def review(output, repo, client, stage, row, revision, *, run_id='derived'):
    """Return an unverified row and immutable review, without installing either."""
    import json
    from .agent_client import role_scope
    from .baseline_reference import source_evidence
    from .execution_derive import _object
    from .source_citations import project, resolve, INSTRUCTION
    revision = validate(stage, revision)
    scoring_locator = stage in {'train', 'evaluate'}
    if scoring_locator:
        assert_prebaseline(output, run_id)
    if Path(output).is_symlink() or (Path(output) / 'artifact_revisions').is_symlink():
        raise ValueError('unsafe artifact revision store')
    if revision['pattern'] == row.get('artifact'):
        raise ValueError('artifact pattern is unchanged; derive without artifact_revision instead')
    citations = source_evidence(repo, revision['source_refs'], output=output)
    packet, shown = project({'stage': stage, 'prior_declaration': row,
        'requested_revision': {**revision, 'source_refs': citations},
        'authority': 'proposed output scope, not producer evidence or an installed adapter'},
        citations, local_roots=(repo, output))
    with role_scope(client, 'supervisor'):
        answer, _ = client.chat_with_metadata(
            ('Review a proposed pre-baseline native output LOCATOR correction, not a score '
             'or a change to training/evaluation. Only artifact path/glob may change; all '
             'other declared fields, native code, task, checkpoint identity, inputs, learning '
             'settings, evaluator, episode/seed selection and metric definitions must remain '
             'unchanged. Require exact native writer source connecting runtime caller inputs '
             '(including variable checkpoint/model names) to the proposed reusable path. '
             'For train require the real native model bundle boundary; for evaluate require '
             'the evaluator-produced metrics artifact, not a fabricated replacement or '
             'another run. A source-supported locator does not establish a score, full '
             'rollout, consumed data or initialized weights. Return an additional Boolean '
             'native_semantics_preserved; reject if this cannot be established. '
             if scoring_locator else
             'Review a proposed acquisition artifact declaration, not a score. Require exact '
            'native source evidence connecting the selected collector/converter to the proposed '
            'output path/glob and its precise dataset boundary. A marker file alone cannot '
            'authorize its parent, a broad ancestor, existing resources or unrelated datasets. '
            'Do not infer a dataset layout from a filename. Reject unsupported scope expansion. '
            'This declares a reusable interface: the same declaration must also match later '
            'formal invocations AFTER the verification probe. Reject a guessed next sequential '
            'dataset name when the probe consumes that name and the collector advances its '
            'counter on every invocation. Prefer the source-backed reusable output scope, '
            'not a prediction of one invocation\'s transient directory. ')+
            'The output locator must be reusable for later source-supported caller inputs, '
            'not a guess at one transient verification filename. '
            'New native verification with freshness/ownership checks remains mandatory; old '
            'receipts cannot be changed or upgraded. Return JSON with booleans '
            'artifact_scope_supported, source_output_connection_supported, '
            'new_native_verification_required, plus why and citations.' + INSTRUCTION,
            json.dumps(packet, ensure_ascii=False), max_tokens=1800,
            read_only=True, include_research_context=False)
    verdict = _object(answer)
    verdict['citations'] = resolve(citations, shown, verdict.get('citations'))
    if source_evidence(repo, citations, output=output) != citations:
        raise ValueError('artifact declaration sources changed during review')
    if scoring_locator:
        assert_prebaseline(output, run_id)
    approved = all(verdict.get(k) is True for k in ('artifact_scope_supported',
        'source_output_connection_supported', 'new_native_verification_required'))
    if scoring_locator:
        approved = approved and verdict.get('native_semantics_preserved') is True
    payload = {'schema_version': 1, 'created_at': now(), 'stage': stage,
        'prior_artifact': row.get('artifact'), 'revision': {**revision, 'source_refs': citations},
        'review': verdict, 'status': 'reviewed_unverified' if approved else 'review_rejected',
        'authority': 'new native verification required; no historical producer authority',
        **({'scope': 'prebaseline_output_locator_only', 'run_id':run_id}
           if scoring_locator else {})}
    ident = object_digest(payload)
    ref = f'artifact_revisions/{ident}.json'
    immutable_json(Path(output) / ref, {'id': ident, **payload})
    if not approved:
        raise ArtifactRevisionRefused('artifact revision not approved: ' +
            str(verdict.get('why', 'unsupported scope'))[:1000], ref, verdict)
    return {**row, 'artifact': revision['pattern'], 'artifact_revision_ref': ref}, ref
