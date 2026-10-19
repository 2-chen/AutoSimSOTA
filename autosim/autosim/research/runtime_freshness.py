"""Current controller/budget facts versus historical observations, without erasing evidence."""
from functools import lru_cache
from pathlib import Path

from .common import digest, object_digest, read_json
from .prompt_library import prompt_identity


@lru_cache(maxsize=1)
def controller_identity():
    root = Path(__file__).parent
    # Automatically cover new admission validators, never mutable run artifacts.
    sources = {path.name: digest(path) for path in sorted(root.glob('*.py'))}
    return object_digest({'sources': sources, 'prompts': prompt_identity()})


def current_model_budget(output, run_id):
    path = Path(output)/'agent/cost_ledger.json'
    if not path.is_file():
        return {'status':'unavailable', 'reason':'no persistent model ledger'}
    try:
        from .agent_budget import AgentCostLedger
        if path.is_symlink():
            raise ValueError('unsafe model ledger')
        row = read_json(path)
        ledger = AgentCostLedger(output,run_id=run_id,limit_usd=row['limit_usd'],
                                 cost_basis=row.get('cost_basis','cli_reported'))
        return {'status':'verified_current', **ledger.snapshot(),
                'authority':'current ledger; historical role snapshots are not admission decisions'}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        return {'status':'unavailable','reason':type(exc).__name__,
                'authority':'unverified; retain holds and do not infer available allowance'}


def budget_observation(observation, current):
    result = dict(observation or {})
    original = result.get('run_budget') or {}
    if (result.get('status') == 'budget_exhausted' and
            current.get('status') == 'verified_current' and
            isinstance(original,dict) and isinstance(original.get('remaining_usd'),(int,float)) and
            current['remaining_usd'] > original['remaining_usd'] + 1e-9):
        result.update(status='historical_budget_exhausted', original_status='budget_exhausted',
            validity='historical_not_current_admission',
            invalidation_reason='current safe allowance exceeds the recorded blocked allowance',
            budget_snapshot_scope='historical; use current_model_budget, not this run_budget')
    return result


def failure_context(output, row):
    """Version a failed controller action without deleting or declaring it repaired."""
    recorded = None
    ref = row.get('receipt_ref')
    if isinstance(ref, str):
        path = Path(output)/ref
        try:
            if (not path.is_symlink() and path.resolve().is_relative_to(Path(output).resolve())
                    and path.is_file() and path.stat().st_size <= 2*1024**2):
                from .action_receipt import verify_action_receipt
                receipt = verify_action_receipt(read_json(path))
                recorded = (receipt.get('resource_limits') or {}).get('enforced', {}).get('controller_runtime_id')
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    if recorded is None and row.get('evidence_id'):
        try:
            import json
            from .evidence_store import read_attempt_evidence
            proof = read_attempt_evidence(output, row['evidence_id'], limit=12000)
            recorded = (json.loads(proof['text']).get('context') or {}).get('controller_runtime_id')
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return {'receipt_ref':ref, 'outcome':row.get('outcome'),
            'evidence_id':row.get('evidence_id'), 'recorded_at':row.get('finished_at') or row.get('at'),
            'failure_category':row.get('failure_category'), 'repair_owner':row.get('repair_owner'),
            'recorded_controller_runtime_id':recorded,
            'validity':'observed_in_current_runtime' if recorded == controller_identity()
                       else 'unverified_in_current_runtime',
            'repair_status':'not_proven; bounded native revalidation required'}


def controller_context(output, steps, *, train_source=None):
    identity = controller_identity()
    result = {'controller_runtime_id':identity,
        'guidance':'Historical failure is not proof the current controller still fails. '
                   'When runtime identity differs or is unrecorded, consider ONE bounded '
                   'source-backed revalidation using the trusted executor; preserve old evidence, '
                   'do not mark it repaired, do not replay unknown live operations.'}
    failures, seen = [], set()
    for row in reversed(steps):
        arguments = row.get('arguments') or {}
        key = (row.get('step'), arguments.get('stage') if isinstance(arguments, dict) else None)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        if row.get('outcome') in {'raised','failed','rejected','no runnable command','checkpoint failure','not attempted','blocked'}:
            failures.append({'step':key[0], 'stage':key[1], **failure_context(output,row)})
    result['previous_action_failures'] = failures[:8]
    # A changed whole-controller hash does not explain WHAT changed. Keep a tiny,
    # source-bound admission description in the live decision context so retained
    # "route closed" lessons cannot masquerade as current executor contracts.
    # This is not repair success or permission to replay unknown operations.
    root = Path(__file__).parent
    result['current_admission_contracts'] = {
        'completed_development_score_recovery': {
            'revision':1,'source_ref':'research/completed_score_recovery.py',
            'source_sha256':digest(root/'completed_score_recovery.py'),
            'rule':'An offered handoff_corrections item with '
                'kind=completed_evaluation_missing_locator means native evaluation already '
                'completed, but its result was not located. Use correct_policy_handoff with '
                'original label and artifact_revision={pattern,reason,source_refs} ONLY. '
                'Do not pass checkpoint_ref/job_id and do not repeat evaluation. Main chooses '
                'a source-backed attempt-owned locator, independent review verifies output '
                'semantics and unchanged metric, and existing load/rollout/episode/byte '
                'checks re-read the same result on CPU. The original failed reading, native '
                'receipt, baseline and incumbent remain intact. Main may separately adopt '
                'an offered verified improvement; recovery itself does not adopt or claim SOTA.'},
        'selected_source_review_delivery': {
            'revision':1,'source_ref':'research/candidate_execution_review.py',
            'source_sha256':digest(root/'candidate_execution_review.py'),
            'rule':'Explicit current research_decision.input_evidence and idea evidence may '
                'select checkout-source/native-source capture IDs. The execution reviewer now '
                'receives the original sealed current source, not just Main prose or CPU diagnostic '
                'output. Backing source, snapshot, environment and seal must match. At most 12 '
                'selected handles, 64KiB per source, 128KiB total; capture a bounded checkout line '
                'page when needed. No store scan, path inference or scientific approval follows. '
                'Use exact capture IDs instead of repeating an uncertain packet whose native '
                'consumer source was absent. A real negative judgment still stands.'},
        'data_registration_phase_boundary': {
            'revision':1,'source_ref':'research/prepare.py',
            'source_sha256':digest(root/'prepare.py'),
            'rule':'register_data_version reviews immutable data identity, frozen training-domain '
                'permission, provenance and a source-supported native root-to-reader route. It does '
                'NOT require prior training, an installed batch emitter or a populated '
                'AUTOSIM_TRAIN_SELECTION_JSON channel. Register compatible bytes first, then select '
                'the handle explicitly in a source-reviewed candidate. Candidate consumer review '
                'and actual training_batch witnesses remain separate mandatory scoring checks. '
                'A successful process or registration never proves data were consumed. '
                'Do not run a paid default-data training merely to install a dead selection emitter. '
                'Before preparing/converting data inspect actual meta schema and the native reader '
                'requirements. A converter in the repository is not evidence conversion is needed. '
                'An empty input staging slot is NOT a selected existing dataset: bind the original '
                'producer data explicitly through the supported input contract. A missing local root '
                'must not silently become a Hub download. Never relabel a dataset version to fool '
                'a converter, fabricate a manifest, or alter an already frozen experiment.'},
        'marker_owned_dataset_diagnostics': {
            'source_ref':'research/diagnostic_artifacts.py',
            'source_sha256':digest(root/'diagnostic_artifacts.py'),
            'rule':'A sealed collect/prepare_data artifact may be a marker file, not the dataset. '
                'Use data_profile_request root=@producer and producer={evidence_id,artifact_index,'
                'dataset_root_relative} to explicitly select its containing dataset directory '
                'relative to THAT producer output. No parent traversal, symlinks, whole producer/run '
                'root, unrelated data or writes are granted. Full-tree identity is checked before/after. '
                'Use include_manifest=true and full video decoding for complete lineage evidence. '
                'New collection registration uses asset.kind=collected with producer_evidence_id '
                'and manifest_evidence_id; omitting asset leaves legacy unknown, not a provenance repair. '
                'Registration still needs independent training-scope/loader/source review and is NOT '
                'actual training_batch consumption. Do not recollect merely to replace a marker declaration.'},
        'data_consumer_review_inputs': {
            'revision': 2, 'source_ref': 'research/data_versions.py',
            'source_sha256': digest(root/'data_versions.py'),
            'rule': 'An uncertain consumer review is not a global source-read denial or '
                'proof that the data route is impossible. Binding setting keys and the '
                'consumption parser are public framework contracts: read current '
                'research/data_versions.py (resolve / verify_consumption) using '
                'framework_source_request; artifact_identity is in the readable '
                'research/experiment_bundle.py. If the reviewer lacks these facts, '
                'capture the relevant exact source pages and cite their evidence IDs in '
                'a revised, source-backed proposal. Repeating an unchanged cached uncertain '
                'packet supplies no new evidence. This guide is not review approval, '
                'actual data consumption or permission to echo expected identities as proof. '
                'A post-training unverified consumer is a Data/Fix investigation, not a '
                'failed optimizer or another bundle choice. Recovery now seals that distinct '
                'failure ID with repair_owner=data; do not replay an unchanged ambiguous log. '
                'For multiworker writers, a constructor-time PID/instance ID may be copied '
                'before worker creation. Inspect actual worker lifetime and atomic framed '
                'writes or a single collector; throttling print is not atomicity. Prefer a '
                'CPU multiworker contract probe before another paid training. Sample reads '
                'are not completed training_batch evidence. Preserve old data/model/logs; '
                'Main chooses new source-backed evidence or an audited candidate, not '
                'invented reader IDs or relaxed verification.'},
        'interruption_ownership': {
            'revision':1,'source_ref':'research/prepare.py',
            'source_sha256':digest(root/'prepare.py'),
            'rule':'The outer choose operation is a pure Main decision, not a native stage '
                'or an owner of an inherited research round. New choose records explicitly '
                'set parent_action null. Legacy choose recoveries without native attempt, '
                'receipt/process reference or process identity reconcile only that decision; '
                'they must NOT rewrite any research session or infer native completion. '
                'Explicit parent links and native process evidence retain ownership checks. '
                'Completed candidate discard also closes its shared research-phase parent '
                'after the exact rollback and session/report commit.'},
        'development_round_allocation': {
            'revision':2,'source_ref':'research/round_extension.py',
            'source_sha256':digest(root/'round_extension.py'),
            'rule':'extend_research_rounds accepts idle paused, completed OR finalizing '
                'development sessions before any held-out exposure. An exhausted local '
                'round allocation is not an exhausted whole-study budget. Finalizing after '
                'discarding the last candidate is eligible when session/report match and '
                'no pending action, active job or worker owns the workspace. Main chooses '
                'additional_rounds and reason within unchanged total model/GPU/wall budgets. '
                'This does not adopt a score, approve a candidate or replay native work.'},
        'consumer_review_response': {
            'revision':5,'source_ref':'research/data_consumer_review.py',
            'source_sha256':digest(root/'data_consumer_review.py'),
            'rule':'A completed provider call is NOT a valid consumer review unless its '
                'JSON booleans, reason and supplied citation IDs validate. One format-only '
                'correction is allowed on unchanged evidence; a valid negative verdict '
                'is not retried. Malformed answers are sealed as consumer_review_unavailable, '
                'not false data/loader findings. Legacy cached format failures alone may '
                'receive a versioned format-v3 review (with format-v2 approvals/refusals retained) '
                'without deleting old evidence or altering source just to invalidate a cache. '
                'A nonempty reason is bounded at 8192 characters, not an artificial 1500; '
                'authenticated sealed older paid responses are revalidated first against '
                'the same current sources and strict booleans/citations, with upgrade provenance, '
                'before any new model call. No genuine negative judgment is retried. Current exhausted corrections '
                'remain cached unavailable. Recovery still requires the applied candidate '
                'transaction and current immutable data/source/environment; no training, '
                'actual batch consumption or score follows from a source review alone. '
                'Current source-context v2 additionally binds complete bodies of ALREADY '
                'cited checkout CODE files (aggregate 64 KiB, explicit omissions), not new '
                'SDK access, raw JSON datasets or invented evidence. A genuinely expanded '
                'packet may receive a new independent judgment without deleting the old '
                'refusal; still-current legacy approvals stay valid. Global rollout episodes '
                'are not dataset episode counts, and original data permission does not '
                'certify current routing, quality or actual batches. Source-context v3 '
                'also binds at most 16 KiB of current public first-party I/O contract '
                '(registered handle resolution, typed selection metadata, environment '
                'export and native isolation handoff). Registered version/slot association '
                'remains explicit after private path redaction; source admission does '
                'not require future runtime byte/batch counters or certify consumption. '
                'No new SDK/resource access or original registration changes are implied; '
                'genuine legacy context-v2 approvals remain usable while current.'},
        'frozen_evaluator_commit': {
            'revision':1,'source_ref':'research/frozen_invocation.py',
            'source_sha256':digest(root/'frozen_invocation.py'),
            'rule':'A successful diagnostic must not replace a frozen formal evaluator '
                'source, parameters or effective stage row. Keep the original formal '
                'adapter; checkpoint selection is a producer-owned caller input, not '
                'permission to regenerate the judge. correct_policy_handoff(label=...) '
                'selects the ORIGINAL failed measurement label from development_recoveries, '
                'not corrected_label or a new output name. The executor creates the new label.'},
        'framework_source_access': {
            'revision': 1, 'source_ref': 'research/harness_repair.py',
            'source_sha256': digest(root/'harness_repair.py'),
            'rule': 'research task source_paths means BENCHMARK checkout files only. '
                'Do not put framework catalog paths in that argument. For public framework '
                'source use inspect_native_environment(purpose=..., framework_source_request='
                '{path: catalog_source_ref, query: optional_literal, limit: 100}); this is '
                'read-only, no native process or framework edit. ONLY propose_harness_repair '
                'uses framework-catalog source_paths. Current catalog includes '
                'research/training_work.py and research/decision.py; an older refused read '
                'does not prove those entries remain unreadable.'},
        'candidate_settings': {
            'revision': 4, 'source_ref': 'research/candidate_settings.py',
            'source_sha256': digest(root/'candidate_settings.py'),
            'rule': 'Sparse param/algo ideas override ONLY explicitly requested training axes. '
                'Validation-filled defaults never overwrite the measured base seed or other '
                'omitted settings. Synchronous and detached candidates use merge_training. '
                'training_settings.seed explicitly selects TRAINING randomness only when '
                'the verified trainer builder consumes its seed slot; the executor stores '
                'it as _autosim_training_seed and supplies it only to train. The original '
                'settings.seed still governs frozen evaluation. The scoped value is '
                'never forwarded as an invented native flag. Training replicas need not '
                'reuse the evaluation seed; direct evaluation-seed mutations remain forbidden. '
                'Real evaluation-protocol changes remain forbidden. '
                'research.sources[train] is the generated stage argv builder '
                '(Preparation.stages passed to DerivedResearch), NOT a checkout trainer '
                'module. consumes_input examines that BUILDER source; adding a get call '
                'to the native trainer does not change this input-admission predicate. '
                'Read workflow_train in the execution review and research/derived_research.py '
                'for the actual caller. Environment-variable names invented in a candidate '
                'are not injected by this contract: establish their writer before relying '
                'on them. Axis.accepts is publicly readable in research/adapter_protocol.py. '
                'These facts neither approve a candidate nor prove native consumption.'},
        'training_selection_transport': {
            'revision': 2, 'source_ref': 'research/data_versions.py',
            'source_sha256': digest(root/'data_versions.py'),
            'rule': 'Resolved registered training selections have a separate typed channel: '
                'inputs.selected_training_data=[{id,setting,path}]. The executor supplies '
                'registered labels and canonical setting keys, NOT expected hashes or counters. '
                'A previously verified argv builder does not automatically adopt a new input. '
                'The SAME labels also reach the native process as JSON in executor-owned '
                'AUTOSIM_TRAIN_SELECTION_JSON, written by staging_environment in '
                'research/declarative_backend.py AND the actual _run_stage environment in '
                'research/derived_research.py (verification: execution_derive.local_environment); '
                'an empty selection overwrites ambient '
                'values with []. A source-reviewed reader/bridge can consume this channel '
                'without changing the frozen argv source or inventing a new training axis. '
                'Existing research requires AUDITED CODE CANDIDATES, not research_task(edit), '
                'for checkout changes. Init/Fix may inspect and propose the exact patch; Main '
                'uses propose_research_idea then the independently reviewed run_the_loop as offered, '
                'and reconciles/discards an interrupted candidate before replacing it. '
                'Do not rederive a different train builder and silently mix session inputs. '
                'Before research exists, native option/config derivation is also possible. '
                'This is an adapter/reader connection, not a new research axis '
                'or permission to pass arbitrary training_settings. Native readers must hash '
                'actual loaded bytes and count actual batches. Do not infer impossibility '
                'from a pre-deployment missing-label failure. Immutable boundary rows remain '
                'history: if recovery succeeds, register a new campaign ID preserving the '
                'original comparison conditions instead of overwriting a boundary or claiming '
                'the old experiment completed.'},
        'interrupted_candidate_resolution': {
            'revision':1, 'source_ref':'research/prepare.py',
            'source_sha256':digest(root/'prepare.py'),
            'rule':'A completed discard applies to its exact round/idea, not every later '
                'candidate. A reconciled pre-native interrupted candidate may expose '
                'discard_interrupted_candidate even when an earlier completed resolution '
                'remains in history. Main chooses explicitly; the executor archives the '
                'earlier resolution and rechecks source rollback without adopting a score '
                'or replaying native work. Missing ownership/receipts remain fail-closed.'},
        'candidate_contract_delivery': {
            'revision':2,'source_ref':'research/candidate_execution_review.py',
            'source_sha256':digest(root/'candidate_execution_review.py'),
            'rule':'Current authenticated framework pages explicitly cited by the Agent '
                'are forwarded within the declared twelve-reference bound, not silently '
                'cut off after three pages. Explicit input_evidence in the CURRENT Main '
                'decision is forwarded too, without rewriting immutable idea citations; '
                'new current authenticated pages change the review packet/cache identity, '
                'not Main prose, unknown IDs or incidental process references. Bounded '
                'validation rechecks the identical selected-evidence packet. Recent action '
                'summaries retain the independent Supervisor verdict, record and readable '
                'execution-review evidence ID; a missing native attempt is not a Main-only '
                'judgment. A previously uncertain candidate whose cited '
                'payload/writer source was omitted may now have a different review packet '
                'with the same idea and inputs. Read its current execution review/catalog '
                'before inventing another label or repeating source investigation. Complete '
                'source transport is neither independent approval nor native consumption.'},
        'candidate_uncertainty_validation': {
            'revision':1,'source_ref':'research/candidate_validation_trial.py',
            'source_sha256':digest(root/'candidate_validation_trial.py'),
            'rule':'An authenticated uncertain source-plausibility verdict is not a '
                'scientific veto, but is NOT approval. For an audited candidate in the '
                'paused standard train/evaluate workflow, Main may explicitly choose '
                'run_the_loop with training_settings and execution_uncertainty containing '
                'reason, postcondition and positive stage_window_seconds. This is a '
                'Main-selected experiment, not another review-only retry. Choose native '
                'work explicitly (steps when consumed), total-resource headroom and '
                'observable postconditions; a short window may legitimately fail. The '
                'executor rechecks the exact review/source/input identity and seals the '
                'decision separately; it never rewrites the verdict. Inapplicable, '
                'unavailable/malformed review, unaudited code, job adoption, policy reuse '
                'and protocol edits cannot use this path. Existing actual-data, training '
                'progress, policy-byte/load, rollout and score checks still decide results. '
                'The window clamps EACH native stage within unchanged hard totals; it '
                'does not add budget, permit unverified scoring or require a fixed work size.'},
        'record_data_campaign_arm': {
            'revision': 3, 'source_ref': 'research/prepare.py',
            'source_sha256': digest(root/'prepare.py'),
            'rule': 'Boundary mode omits measurement_label and both data-version arguments; '
                'source_refs input uses {ref,quote}, while independent Supervisor selects '
                'supplied source_id objects. One citation-only format correction may preserve '
                'the original independent verdict. Negative scientific checks take priority '
                'over citation-format errors: Main/Data receive the denial and missing '
                'prerequisites, not an alleged Supervisor-only closed gate. A boundary '
                'is not a completed experiment.'}}
    if isinstance(train_source, str) and train_source:
        # Give the current loaded caller, not an inferred benchmark file or a
        # stale disk adapter. This is readable state, never an approval/probe.
        from .common import sanitize_model_payload_text
        import hashlib
        bound = result['current_admission_contracts']['candidate_settings']
        bound['current_train_builder'] = {
            'namespace': 'run_stage_argv_builder_not_native_trainer',
            'source_sha256': hashlib.sha256(train_source.encode()).hexdigest(),
            'complete': len(train_source.encode()) <= 8192,
            'authority': 'currently loaded caller source; not native execution or consumption'}
        if bound['current_train_builder']['complete']:
            bound['current_train_builder']['text'] = sanitize_model_payload_text(train_source)
        else:
            bound['current_train_builder']['reading_guidance'] = (
                'Builder exceeds the inline bound; use the workflow_train source '
                'in the sealed execution review request. Do not infer missing arguments '
                'from an omitted excerpt.')
    result['historical_admission_revalidation'] = {
        'operations': sorted({row['step'] for row in failures[:8]
            if row['validity'] == 'unverified_in_current_runtime'}),
        'authority': 'historical refusal is not a current admission decision or verified repair',
        'next_decision': 'Before preserving a closed-route or stop-condition lesson, compare '
            'its failed runtime with current contracts and source. A relevant deployed change '
            'may justify ONE bounded executor revalidation chosen by Main. Do not replay '
            'unknown live work, relax protocol, or infer success from this description.'}
    for row in reversed(steps):
        if row.get('step') != 'build_the_environment':
            continue
        if row.get('outcome') not in {'raised','failed','checkpoint failure','repair proposal rejected'}:
            break
        ref = row.get('receipt_ref')
        recorded = None
        if isinstance(ref,str):
            path = Path(output)/ref
            try:
                if (not path.is_symlink() and path.resolve().is_relative_to(Path(output).resolve()) and
                        path.is_file() and path.stat().st_size <= 2*1024**2):
                    from .action_receipt import verify_action_receipt
                    receipt = verify_action_receipt(read_json(path))
                    recorded = (receipt.get('resource_limits') or {}).get('enforced',{}).get('controller_runtime_id')
            except (OSError,ValueError,TypeError,AttributeError):
                pass
        result['previous_setup_failure'] = {'receipt_ref':ref,'outcome':row.get('outcome'),
            'recorded_controller_runtime_id':recorded,
            'validity': 'observed_in_current_runtime' if recorded == identity else
                        'unverified_in_current_runtime',
            'repair_status':'not_proven; bounded native revalidation required'}
        break
    return result
