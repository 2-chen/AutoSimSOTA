"""What has to be true before a research loop can run, as one loop rather than four programs.

Five things have to happen before a benchmark can be researched: read the checkout, write down
what it can do, build an environment, find a command for each stage, and run the loop. They
were five programs. Each wrote its JSON somewhere, and each next program knew where to look
and what shape to expect -- so the sequence was a contract between programs, maintained by
hand, and it failed at the seams rather than inside any of them.

The failures were real and none of them were about a benchmark. A driver that read
`execution.json` crashed with `FileNotFoundError` when the step that writes it had not run. A
build reported `passed: true` on an empty record because the step that planned it was skipped
when a caller supplied an interpreter. The declaration was chosen by newest mtime, and the one
that was alphabetically first won when two were equally new. Provisioning was reachable only
from a script under `tools/`, so `autosim research` could not build an environment at all --
it required that someone had already run something else, and said `run provisioning first`.

Native execution stays in validated operations, but the coding-agent Scheduler also owns a
persistent global working plan and can investigate arbitrary questions or delegate them to
specialists. Its decision turns are read-only; editable investigations have explicit action
boundaries and invalidate prior verification. A failed stage returns evidence to the main
Agent rather than prescribing either a repository-specific repair or immediate termination.

The records are still written, by the operation that produced them, and they are what the
state is read from. They are not the interface any more: nothing here parses a document to
find out what step to take. The state is a reading of the same records a person reads, and a
step that fails leaves the record exactly as legible as one that succeeds.
"""

from __future__ import annotations

import json
import hashlib
import csv
import math
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import benchmark_bundle, execution_derive, provision, run_record, scout, supervisor
from .budget import BudgetedClient, RunBudget
from .action_receipt import ActionReceipt, write_action_receipt
from .common import (atomic_json, atomic_text, digest, now, object_digest, read_json,
                     redact, sanitize_model_payload as _controller_model_value,
                     sanitize_model_payload_text as _controller_model_text,
                     sanitize_model_text)
from .codepatch import patches_from_change
from .main_agent import memory_view
from .scheduling import view as scheduling_view
from .compute_decision import ComputeDecision, decide
from .declaration import space_from, usable_declarations
from .declarative_backend import DeclarativeBackend
from .decisions import Decisions, ResearchDecision
from .derived_research import DerivedResearch
from .devices import NoCompatibleDevice
from .ideas import (Idea, IdeaLibrary, declared_space_compatibility,
                    execution_compatibility)
from .metric_contract import MetricSpec, resolve_metric_artifact
from .process_executor import (capture_process_identity, inspect_process_identity,
                              observe_process_starts, terminate_recorded_process)
from .research_state import (ResearchStateError, ResearchStatePersistenceError,
                             ResearchStateStore)
from .skills import GUIDE, read_selected_skills, skills_catalog, skills_reference
from . import recorder
from .snapshot import Snapshots


def _structured_result_schema(path: Path) -> dict[str, Any]:
    """Describe field names and types for metric mapping without exporting result values."""
    path = Path(path)
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            return {"error": "schema preview refused: file exceeds 8 MiB"}
        if path.suffix.lower() == ".json":
            document = read_json(path)

            def shape(value: Any, depth: int = 0) -> Any:
                if isinstance(value, dict):
                    if depth >= 4:
                        return {"type": "object", "keys": sorted(map(str, value))[:80]}
                    return {"type": "object", "fields": {
                        str(key): shape(item, depth + 1)
                        for key, item in sorted(value.items(), key=lambda row: str(row[0]))[:80]}}
                if isinstance(value, list):
                    samples = [shape(item, depth + 1) for item in value[:2]]
                    unique: list[Any] = []
                    for sample in samples:
                        if sample not in unique:
                            unique.append(sample)
                    return {"type": "array", "length": len(value),
                            "item_shapes": unique}
                if isinstance(value, bool):
                    return {"type": "boolean"}
                if isinstance(value, (int, float)):
                    return {"type": "number"}
                if value is None:
                    return {"type": "null"}
                if isinstance(value, str):
                    return {"type": "string"}
                return {"type": type(value).__name__}

            return shape(document)
        if path.suffix.lower() == ".csv":
            types: dict[str, set[str]] = {}
            with path.open(newline="", encoding="utf-8-sig") as stream:
                rows = csv.DictReader(stream)
                fields = list(rows.fieldnames or [])
                samples = 0
                for row in rows:
                    samples += 1
                    for name in fields:
                        value = str(row.get(name) or "").strip()
                        if not value:
                            observed = "empty"
                        elif value.lower() in {"true", "false"}:
                            observed = "boolean"
                        else:
                            try:
                                float(value)
                                observed = "number"
                            except ValueError:
                                observed = "string"
                        types.setdefault(name, set()).add(observed)
                    if samples >= 12:
                        break
            return {"type": "csv", "columns": {
                name: sorted(types.get(name) or {"empty"}) for name in fields},
                "sampled_rows": samples}
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        return {"error": f"schema preview unavailable: {type(exc).__name__}"}
    return {"error": "unsupported structured result type"}


#: The steps, in the order they usually happen. The order is a default and not a rule: the
#: model may take them in any order it can justify, and may take one twice.
OPERATIONS: dict[str, str] = {
    "configure_screening": "declare a fidelity study: study={id,budget_axis,rungs,eta,min_peers,evidence,reason}; final rung matches formal training budget, evaluation stays frozen.",
    "run_screening_trial": "execute one audited param idea at an explicit study_id/rung with idea_label, window_seconds, reason; separate receipts and scores, never formal best.",
    "inspect_screening": "read study_id and advisory same-rung promotion recommendations; Scheduler decides promotion and must revalidate full-protocol candidates.",
    "submit_research_task": "submit a read-only specialist assignment asynchronously: role, task, expected_result; independent session, shared total model budget. Results are merged only by Scheduler.",
    "cancel_research_task": "discard an active read-only task result using task_id and reason; in-flight provider charges remain accounted.",
    "wait_for_jobs": "wait for native/agent completion events without model polling; provide no arguments. Use when dependencies are running and no useful independent work remains.",
    "review_report_demo": "resolve a Recorder request with request_id, decision (capture or "
                          "decline), and reason; only Scheduler may authorize native capture.",
    "capture_environment_demo": "fulfill a pending environment_smoke request before scoring. "
        "Provide request_id, code (native Python recording script), purpose, source_refs "
        "(checkout-relative files), timeout_seconds and resource (cpu or gpu). "
        "Read actual repository APIs; write video to os.environ['AUTOSIM_DEMO_DIR']. "
        "This preview is unscored and cannot claim policy/task identity or improvement.",
    "update_research_plan": "persist the main Agent's global working plan and unresolved "
                            "questions; this is memory, not a change to the frozen objective.",
    "research_task": "investigate an arbitrary evidence-backed question yourself or delegate "
                     "it to an AutoSOTA role using its tools; return findings to the main "
                     "Agent without launching benchmark jobs or certifying success.",
    "retry_failed_action": "after AgentFix changed the isolated checkout, rerun the exact "
                           "failed preparation operation and judge its new receipt; do not "
                           "turn a patch or CPU probe into a verified recovery.",
    "submit_native_job": "submit a verified native stage as a detached, budgeted job; optional resources={cpu,memory_mib,gpu,priority}. "
                         "provide a positive local window and evidence-based reason. A "
                         "completed producer can be adopted as an initial baseline only "
                         "when its receipt matches the frozen settings and protocol.",
    "inspect_native_job": "read a submitted job's liveness, result and local budget.",
    "adjust_native_job": "extend or shorten a running job's local window within the hard "
                         "frozen wall, GPU and model budgets, with a reason based on progress.",
    "cancel_native_job": "request cancellation of a stalled job; keep its attempt evidence.",
    "read_the_checkout": "survey the repository: which stages exist, what their entry points "
                         "are, and which files say so. Writes execution.json.",
    "declare": "read the repository and write down what it can do -- its tasks, its data, its "
               "evaluation, and the settings that may vary. Writes a declaration.",
    "build_the_environment": "make or re-verify an environment the stages can run in, "
                             "Optional max_operations (1..64) lets Scheduler batch stable "
                             "setup work; failures always return control. Default is one. "
                             "recording only the commands that worked. After a later native "
                             "stage failure, re-run its capability probes and give the "
                             "failure evidence to the planner; do not recreate a verified "
                             "interpreter. Writes environment.json and recipe.json.",
    "derive_a_command": "find a command that runs one stage, revising the invocation until it "
                        "does. Optional timeout_seconds requests a local verification window "
                        "within the unchanged run hard budgets. Writes what it kept "
                        "to derived_stages.json.",
    "reconcile_interrupted_action": "inspect the prior attempt receipt and process identity; "
                                    "stop only its verified process group, preserve unknown "
                                    "outcome, and report evidence before another attempt.",
    "discard_interrupted_candidate": "after reviewing a reconciled candidate boundary, "
                                     "explicitly discard its measurement and roll back only "
                                     "an exact, still-matching source transaction; then resume "
                                     "at the next research round.",
    "bind_metric": "bind the primary metric to a verified score command's actual output "
                   "and repository source; do not guess a score from a label.",
    "recover_unscored_baseline": "revalidate an unscored baseline's exact completed train "
                                 "receipt and native policy selection, then evaluate that "
                                 "already-produced policy without retraining.",
    "run_the_loop": "run the research loop on the stages that have commands.",
    "generate_research_ideas": "ask for and audit a fresh batch of candidate research ideas; "
                              "this does not run a benchmark stage.",
    "propose_research_idea": "submit one evidence-backed idea from the main controller for "
                             "deterministic audit; this does not run a benchmark stage.",
    "confirm_best": "evaluate the already-frozen best policy on the explicitly requested "
                    "held-out protocol; this action must not train or change the policy.",
    "stop": "nothing further can be done, and say why.",
}

_MONITORED_FAILURES = frozenset({
    "raised", "rejected", "blocked", "no usable declaration", "no interpreter recorded",
    "no runnable command", "no such stage", "not bound", "not recovered",
    "not confirmed", "no number came out",
})

MONITOR_SYSTEM = """You are AgentMonitor, outside the preparation and research action loop.

Inspect the completed action or scheduler segment, its persisted evidence, and (when useful)
the repository. Give a high-level progress assessment and guidance to the Scheduler. Do not
edit files, run commands, select an experiment, change a metric, or decide the benchmark
result. Do not prescribe a specific code patch or claim a repair occurred. The Scheduler
retains all action authority.

Return exactly one JSON object with `progress` (`progress`, `stalled`, `blocked`, or
`uncertain`), a concise `summary`, high-level `guidance` (empty if none), and relative
`evidence_refs` from the shown records or checkout. Cite evidence; do not invent it."""

FIX_SYSTEM = """You are AgentFix, the repair role for one failed preparation action.

Use the cited failure receipt, current repository and relevant skills to diagnose the cause.
You may inspect and make the smallest justified repair in this run's isolated checkout, then
run bounded CPU-only diagnostics that test the same failure. Never change the benchmark task,
evaluator, metric, test split, success threshold, or protected test inputs. Do not launch a
training job, benchmark evaluation, GPU workload, or data collection. If the failure is not
repairable within these limits, make no speculative edit and explain the missing evidence or
external prerequisite. A patch or a passing probe is not proof that the original action now
works; report only what you actually inspected or ran.

If failed_action contains a stable evidence_id, call read_evidence with that ID before
diagnosing. The original log is run-owned and may not be visible inside the checkout.
For environment failures, also read environment.latest_failure.evidence_id before proposing
a fix. Installation success and environment-round exhaustion do not replace that failure.
Inspect native configuration initialization for unattended input/EOF failures; prepare
run-local configuration with consistent paths and revalidate the original failing operation.
Put CPU diagnostic environments in /tmp/diagnostics, never inside the source checkout.

Return exactly one JSON object with `assessment` (`repair_attempted`, `no_safe_repair`,
`blocked`, or `uncertain`), concise `summary`, a list of `changes`, and relative
`evidence_refs`. Evidence references are model citations and are not independently verified."""

_FIX_ASSISTANCE_STEPS = frozenset({
    "read_the_checkout", "declare", "build_the_environment", "derive_a_command",
    "bind_metric",
})
_FIX_ASSISTANCE_OUTCOMES = frozenset({
    "raised", "failed", "no usable declaration", "no interpreter recorded",
    "no runnable command", "no such stage", "not bound",
})

STEP_SYSTEM = """You are the main AgentScheduler, responsible for the whole research run,
from repository onboarding through baseline, optimization, verification and final handoff.

You are given what is known so far: the records that exist, what each step produced, and what
failed and why. Choose the next step.

Own a coherent global plan, not just the next missing pipeline stage. `main_agent` contains
your persisted working plan and specialist handoffs across sessions. Reconcile new evidence
with that memory; keep unknowns explicit and revise contradicted hypotheses. If supported,
use `update_research_plan` with arguments.plan containing exactly objective (string),
hypotheses, open_questions, next_actions and evidence_refs (lists of strings). This does not
change the benchmark's frozen objective or confer permissions.
Before freezing the experiment, ask Objective which secondary metrics the native evaluator
reports and which must not regress. If supported by source, declare research_goal.guardrail_metrics
with explicit name/direction/unit/source and max_regression. Use the same archived result as
the primary metric, or a labelled native log. Do not invent thresholds from metric names.
Missing secondary readings are unknown, not permission to accept a primary-only win.

When the fixed operations cannot resolve an uncertainty, use `research_task` with arguments
role, task and expected_result (all strings). Role scheduler means investigate yourself;
resource/objective/monitor/ideator/supervisor are read-only specialists; init/fix can inspect,
repair and CPU-probe before research starts. Assign a concrete question and acceptance
evidence, not a guessed answer. Each task may use multiple tools; after its handoff YOU decide
whether to inspect more, retry a native operation, revise the plan, or stop. During an existing
research session only read-only research_task roles are allowed; code candidates still pass
through the audited idea/snapshot/evaluation path. A specialist report is not proof of recovery.
Do not spend the budget merely rewriting plans. Prefer actual evidence-producing work.
This decision turn cannot edit or run arbitrary shell commands. You may inspect the actual
selected environment with inspect_native_environment: CPU only, read-only source/config,
no network, writes only diagnostic scratch; its receipt is not simulation readiness.
Put edits, installation, GPU work and native benchmark actions in explicit validated operations.
Skill selection is your prior choice, cached with its reasons while the context is unchanged.
Set refresh_skill_selection=true in your decision JSON to request a fresh skill choice next turn.
Environment preparation executes one native operation by default, then hands evidence back
to you. Supply arguments.max_operations (1..64) to batch a coherent preparation chunk.
Any failure yields after a bounded Fix proposal: inspect its stable evidence before proceeding.

The steps, and what each needs:

{operations}

Choose by what is missing and by what failed. A few things are worth knowing:

* `report_demo_requests` are Recorder suggestions, not authorized jobs. Use
  `review_report_demo` with request_id, decision="capture" or "decline", and reason.
  Scheduler alone budgets and approves capture from its frozen development measurement.
  A pending environment_smoke request instead uses capture_environment_demo with a source-backed
  recording script; no scored measurement is required, and it is explicitly unaudited/unscored.
  For scored demos no custom commands or settings are accepted. Preserve confirmation resources; never
  use held-out results to design a demo or replay a capture with an unknown outcome.

* `monitor_observation`, when present, is an independent high-level assessment, not a diagnosis
  or permission. Use its cited evidence as a prompt to inspect the records; the Scheduler alone
  chooses the next action.

* **A step that failed is not a reason to stop.** The record says why it failed. If the reason
  is about one stage, another stage may still be derivable. If the environment could not be
  built, the checkout may still be readable and the declaration still worth writing.
* If AgentFix reports a concrete repair, select `retry_failed_action` to re-execute the
  original failed preparation operation once. Read its new receipt before calling it fixed;
  if the same failure remains, choose a discriminating new action rather than looping.
  Exception: a failed candidate research round has already entered the selection history and
  may have rolled back its source. Its Fix handoff is read-only diagnosis, never an authorized
  `retry_failed_action`. Inspect the cited evidence ID, formulate an audited repair idea via
  `propose_research_idea` (or select a matching existing idea), then run it as a NEW round.
  Compare its native train/evaluate receipts and metric to the prior failed operation before
  claiming the repair worked. If the idea cannot be represented within the frozen protocol,
  state the boundary instead of changing the declaration mid-session.
* **A step already done does not need doing again** unless something it depended on changed.
* Data acquisition is a user-prioritized research option. During onboarding, resolve its
  feasibility before freezing the declaration: choose the catalog skill
  `autosimsota.connecting-native-data-collection` when relevant, and use `research_task`
  with role `resource` to trace the native producer, expert/planner/policy prerequisites,
  conversion and actual trainer loader. Record unknowns, evidence and next actions in
  `update_research_plan`; missing assets are not proof of a missing collector. Init/Fix can
  repair onboarding connections; request native probes through the trusted stage executor.
  When automatic collection is allowed, resources are available and the trainer can consume
  its output, prioritize a small collection-to-loader probe and an early audited data
  intervention over routine hyperparameter exploration. Keep the original-data baseline
  identifiable; if data is absent, collection may bootstrap the first baseline. A probe is
  not evidence of a score improvement. Preserve native training/held-out splits, count failed
  collection attempts, and reserve time for conversion, training and evaluation inside the
  hard budget. Decide whether to scale from observed yield and downstream score. Explain any
  deferral (fixed-data protocol, human-only control, missing assets, inadequate policy yield,
  or poor expected value). Do not assume collection always helps or force it into online RL.
* **Derive the path that can produce a score for the selected task.** `prepare_data` and
  `collect` are optional; if the available trainer learns directly from simulator interaction
  and no demonstration dataset is present, derive `train` and `evaluate` before attempting
  a demonstration conversion or collection stage.
* If path selection rejects an incompatible train→score checkpoint handoff, choose
  `read_the_checkout` at most once to inspect a different native family; do not build the
  rejected combination or treat matching checkpoint suffixes as compatibility. Then compare
  `workflow_selection`'s input digest with the current `surveyed_stages`: a rejection bound to
  older inputs is not a fact about the new survey. If the current survey now exposes a
  plausible source-supported path, retry `build_the_environment` so its selector evaluates
  that path; do not keep rereading unchanged source.
* `surveyed_stages` are source-derived candidates, not verified commands. Compare their
  entrypoints/invocations with `stages_the_checkout_has` before deciding what is missing.
  `environment.base_python_hint` is an available input to provisioning, not proof that the
  simulator environment passed its probes; use `build_the_environment` to verify it.
* A passing environment claim is not permanent. If a later selected native stage fails,
  inspect the probe commands and their receipts. When the failure could be an environment
  contradiction, `build_the_environment` re-runs the capability check in the recorded
  interpreter and gives the stage failure to the provision planner; successful installs are
  replayed only when absent, and the existing prefix is not recreated. Do not assume every
  bad invocation is an environment problem; use the stage evidence to choose.
* If the evaluator consumes a checkpoint and none is present, derive and verify `train`
  first. Its verification must produce a fresh policy artifact before the evaluator can
  be tested against it.
* A new coding-agent run requires evidence from the *native policy loader*, not just an
  archived checkpoint path. Inspect the loader and arrange a protocol-preserving witness
  at the real load site, without changing policy actions, evaluation data or scoring.
  It must emit one `AUTOSIM_POLICY_LOADED {{"path":"<absolute loaded path>",
  "content_sha256":"<sha256 of loaded file>"}}` line (use `sha256` for a directory
  artifact). The executor compares that line to the frozen artifact. If the repository
  cannot expose a trustworthy witness under the protected-code rules, report that
  limitation; do not label an unverified evaluation as a score. A candidate that changes
  model family must train or acquire a matching policy before evaluation.
  At native episode completion, also emit `AUTOSIM_ROLLOUT_COMPLETED
  {{"episode_id":"<stable episode identifier>","policy_sha256":"<loaded policy hash>"}}`.
  The same identity must cover the rollout that yielded the metric. Add this witness at
  the actual episode boundary, not in a wrapper that merely prints plausible lines.
  At the native metric aggregation site, emit exactly one
  `AUTOSIM_METRIC_REPORTED {{"policy_sha256":"<loaded policy hash>",
  "episode_ids":["<the same completed episode IDs>"],"value":<native score>}}`.
  The executor only checks identity and numeric consistency; it does not prescribe
  benchmark-specific aggregation or let an Agent change the scoring definition.
* For command derivation, `timeout_seconds` is a *local* request, not a new run budget.
  Derive native work units and request an adequate window when a complete epoch/task
  requires longer than the default probe. The executor clips every attempt to the
  remaining hard run budget; repeated timeout with unchanged work is not a reason to
  keep rewriting argv.
* For a verified trainer whose full native work exceeds a model turn, use
  `submit_native_job` with `stage`, `window_seconds`, `reason`, and optional
  `resources` (cpu, memory_mib, gpu, priority). New scheduling runs also admit verified
  collection, data preparation and other native stages. Inspect its progress
  before `adjust_native_job` or `cancel_native_job`; these actions take `job_id` and a
  reason (adjustment also takes `window_seconds` from now). A completed detached trainer
  is not a score; to adopt it as the initial baseline, call `run_the_loop` with `job_id`.
  The executor rechecks the original training receipt, settings and artifact before native
  evaluation. Never run the same stage concurrently in its shared output namespace.
  Independent CPU preparation may overlap GPU work; queued work does not consume its
  local execution window or GPU time, but still counts against the hard wall deadline.
* Use `submit_research_task` for an independent read-only specialist question,
  with role, task, expected_result. It has a separate source snapshot/session
  but shares the run's model ledger. Do not delegate sequentially dependent questions.
  A stale report is a lead to recheck, not evidence about the current checkout.
  When there is no useful independent work, choose `wait_for_jobs` (no arguments)
  instead of repeatedly spending model calls inspecting unchanged state.
* Native jobs own their source/environment until their outcome is resolved. Do not
  request edits, environment recreation, derivation or formal research during that lease.
  A parallel execution_graph must declare max_workers (1..4), resources for every node,
  and parallel_safety evidence about independent sibling outputs; dependencies and
  artifacts, not node names, decide which work can overlap.
* When several audited parameter ideas compete, optionally configure_screening with
  study (id, budget_axis, increasing rungs, eta, min_peers, evidence, reason). Identify
  the actual native training budget axis from source; never guess epochs or reduce the
  evaluation protocol. The final rung equals the baseline's full training budget.
  run_screening_trial uses study_id, idea_label, rung (zero-based), window_seconds,
  reason. inspect_screening uses study_id. Same-rung native rollout scores give advisory
  promotions after min_peers; early loss alone cannot discard a robotics policy.
  Screening does not update best or certification: return a promising idea to the full
  audited research loop, then independently confirm. Code/algo ideas retain their full
  source transaction; do not turn this optional lane into a parameter-only optimizer.
* If `recovery_after_interruption` is present, `reconcile_interrupted_action` is the only
  permitted next step. The previous action's outcome is unknown. Reconcile its receipts,
  outputs, and process state before proposing that it run again; absence of a completion
  record is not proof that no side effect occurred.
* **A runnable evaluator is not yet a score.** If a score command is verified but its
  primary metric is unbound, choose `bind_metric` before `run_the_loop`.
* `to_measure.ready` means the evaluator and metric contract can be attempted; it does not
  mean a prior measurement produced a number. If
  `research_progress.unscored_baseline_recovery.available` is true, choose
  `recover_unscored_baseline` before another `run_the_loop`: it
  verifies the original train receipt and source/candidate evidence, then evaluates that
  existing policy without retraining. Do not re-run `bind_metric` when the metric binding is
  already verified, and do not start another train attempt to repair a selection-only failure.
* **The run exists to produce a number**, and `to_measure` says whether it can: when it reads
  `ready: true` and no research session is paused, `run_the_loop` starts or safely resumes
  the baseline. A paused session has a different contract: inspect its exact options and
  select one, propose a novel idea, or request suggestions before resuming. A step that failed
  is not a reason to repair it first -- a declaration already on file is not improved by
  re-drafting one, and a run that spends its steps fixing what is only *untidy* never measures
  anything.
* `research_progress.status == "paused"` means one bounded research action finished: inspect
  its evidence and `research_options`; `run_the_loop` may resume at the recorded next round
  only with `arguments.idea_label` set to one exact currently available option. This is the
  main research controller's decision: the inner engine validates and executes that label but
  must not choose another idea. Existing candidates are suggestions, not a closed menu: the
  main controller may submit `propose_research_idea` with its own evidence-backed idea for
  audit. Prefer `generate_research_ideas` first when `research_options.items` is empty and
  `excluded_incompatible` shows that the old pool targets settings outside the verified stage
  interface; a fresh batch is conditioned on the current task, failure, and selected stage
  parameters. Do not propose a parameter the declaration does not expose merely because the
  native command accepts it. Once a research session exists (including `paused`), its task,
  metric, evaluation protocol and optimization space are frozen: never choose `declare` to
  change those inputs mid-session. If a fresh batch still has no selectable ideas because every
  stage-compatible setting is outside the declared space, record that boundary and stop; a
  corrected declaration requires a new run before its baseline. Do not widen the space just to
  force a candidate. `arguments.idea` must be exactly one object with these nine keys:
  `label`, `granularity` (`param`, `code`, or `algo`), `mechanism`, `change` (always a JSON
  object, never prose), `risk` (`low`, `medium`, or `high`), `crosses` (`none` or a red-line
  ID), `why`, `evidence` (a non-empty list of shown references), and `touches` (a list).
  For `param` and `algo`, set `touches` to `[]`; only a `code` idea may name repository-relative
  paths, which must exactly match its bounded patch. A parameter `change` is a map of declared
  axis names to in-range values. Copy each axis name exactly as shown in `declared_space`; do
  not add a `train.`/`training.` prefix based on its section. For example, if the training axis
  is named `n_epochs`, use `{{"change":{{"n_epochs":2}},"touches":[]}}`; use a dotted name only
  when the declared axis itself contains that exact dot. Do not wrap the map in a string or
  attach a source file. An accepted proposal appears in
  `research_options`; it still needs an explicit later `idea_label` selection. If no options
  remain after a fresh compatible batch, use the shown declaration and stage contract to
  explain the boundary or propose only a change that passes both contracts. Do not repeat the
  completed action. A new research session runs only the baseline before returning control.
  For quantitative training ideas, if
  `research_progress.last_observation.actual_training` is present, treat its receipt-backed
  parameter values as what actually produced the measured policy. A surveyed invocation,
  documented default, or declared-space default is not evidence that the baseline used that
  value. Do not describe a candidate as changing "from X" unless the actual training receipt
  supports X; if it is absent, inspect its cited receipt or state that the baseline setting is
  unknown before making a causal claim.
  `interrupted` means
  the process was reconciled but the research outcome is
  still unknown: only a baseline interruption with `safe_to_resume: true` may be explicitly
  resumed, and its receipt-verified measurement is reused when present. An interrupted
  candidate is not replayable until a pending-action checkpoint exists; inspect its evidence
  or stop with that boundary. `running` or unreadable/identity-mismatched state has unknown
  outcome and must be reconciled or stopped, never relaunched blindly. `finalizing` may resume
  only report and export finalization; `completed` must not train again.
* An interrupted candidate can be discarded only after its process/receipt reconciliation.
  `discard_interrupted_candidate` is an explicit main-controller decision not to adopt any
  candidate metric; the kernel may continue only after exact source rollback is verified
  where a code transaction exists. Missing transaction evidence or a diverged checkout keeps
  the session interrupted. The saved measurement remains in the audit record, not history as
  an accepted score. Provide `arguments.reason` citing the shown boundary/evidence.
* If completed research has `confirmation_requested` and confirmation is still available,
  choose `confirm_best`; it evaluates the frozen winner without training. Otherwise review the
  recorded result and stop or return to an evidence-backed preparation action. A completed
  inner research loop is not itself the end of this controller's run.
* **`stop` is for when nothing available would change anything.** Say what is missing and what
  would have to be true for it to be possible. A run that stops should leave a reader knowing
  exactly what blocked it.

`method_library` contains candidate methods retrieved from metadata, not executable policy.
The main Agent selected the bodies to read from a short directory; lexical recommendations
were hints, not the selector. For each selected body other than the library guide,
compare its preconditions and `invalid_when` with this run's evidence, then return one
`method_review` row with its exact skill ID, verdict (`use`, `decline`, or
`insufficient_evidence`), a short reason, and only evidence references shown in the state.
Do not force a skill into the plan; the controller may reason without one. This is your
judgment, not kernel verification, and a skill grants no tool, filesystem, network, process,
or compute permission.

Return one structured research decision as JSON. Echo the exact decision-relevant
`state_revision` in the state you read. Run/agent event sequence numbers are audit metadata;
agent-runtime telemetry does not change `state_revision`, but any decision-relevant run-state
change does. Include `question` (the uncertainty this action resolves), `hypothesis` (what you
expect to learn or establish), `evidence_refs` (only source paths/state fields actually shown),
`resource_limits` (limits justified by the state; do not ask for more than remains),
`expected_outputs`, `postconditions`, and a concrete `stop_condition`. Empty evidence is better
than an invented citation; missing resource data must remain unknown. The kernel enforces its
own budgets and action preconditions regardless of this proposal.

Return exactly one JSON object: {{"state_revision": <integer>, "do": "<one permitted step>",
"why": "<reasoning>", "question": "<uncertainty>", "hypothesis": "<testable expectation>",
"evidence_refs": ["<shown path or state field>"], "resource_limits": {{}},
"expected_outputs": ["<expected output>"], "postconditions": ["<checkable condition>"],
"stop_condition": "<when this line of work should stop>",
"method_review": [{{"id":"<selected skill id>","verdict":"use|decline|insufficient_evidence",
"why":"<applicability reasoning>","evidence_refs":["<shown reference>"]}}],
"arguments": {{<only when needed: "stage":
"<one surveyed stage>" and optional positive "timeout_seconds" for derive_a_command,
"idea_label": "<exact offered label>" for a paused run_the_loop, or
"job_id": "<completed baseline trainer job ID>" for the initial run_the_loop,
"idea": {{...}} for propose_research_idea, "reason":
"<evidence-backed discard rationale>" for discard_interrupted_candidate,
"role", "task", "expected_result" for submit_research_task,
"study" for configure_screening; "study_id", "idea_label", "rung", "window_seconds",
"reason" for run_screening_trial; "study_id" for inspect_screening;
"task_id", "reason" for cancel_research_task; no arguments for wait_for_jobs;
"stage", "window_seconds", "reason", optional "resources" for submit_native_job,
"job_id" for inspect_native_job, "job_id", "window_seconds", "reason" for
adjust_native_job, or "job_id", "reason" for cancel_native_job>}}}}."""


SKILL_SELECTION_SYSTEM = (
    "You are the main research Agent choosing which method-library entries to read before "
    "your next research decision. The catalog contains only short descriptions, not method "
    "bodies. Return exactly one JSON object: "
    '{"skill_reads":[{"id":"<catalog ID>","why":"<why this method may help the current '
    'uncertainty>"}]}. Choose zero to three distinct IDs from the catalog. You may choose '
    "entries without a keyword recommendation; recommendations are hints, not decisions. "
    "Do not claim a method applies until you have read it and compared its preconditions "
    "with current evidence. No tools or repository edits are needed for this selection."
)


def _tail_of_said(said: str, *, limit: int = 400) -> str:
    """The end of a program's output, because that is where the cause is.

    A Python traceback ends with the exception; a compiler error ends with the error. Taking
    the first N characters of one shows the frames that led there and cuts the line that says
    what happened -- which is what happened to RoboTwin's record: the log showed
    `from detr.main import (...)` and not the `TypeError` six frames below it, so the recorded
    reason for eighty-eight rejected commands was a fragment.
    """
    text = said.strip()
    if len(text) <= limit:
        return text
    return "…" + text[-limit:]


def _latest_per_step(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per step, saying what is true of it now.

    The state used to carry every attempt, and on RoboTwin that was 217 entries of which 210
    were `derive:evaluate` rejections. The scheduler read the history and stopped, saying the
    derivation had "consumed an enormous number of attempts cycling through the same two root
    causes without converging" -- **while the line beside it said `evaluate: has a command`.**
    It had converged; the state buried the fact under the record of getting there.

    A scheduler decides from what is true now. The history is not lost -- every attempt is in
    `derivation_attempts/<stage>.json` and in the run's document -- but it is not what the
    next decision is made from, and handing it over as though it were is how a run that
    succeeded concludes that it failed.
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in steps:
        step = str(row.get("step") or "")
        held = latest.setdefault(step, {"step": step, "outcome": row.get("outcome") or "",
                                        "because": str(row.get("because") or "")[:300],
                                        "attempts": 0})
        held["attempts"] += 1
        # Outcome and reason are one record and are replaced together. Taking the latest of
        # each independently pairs a success with the failure before it -- `accepted`, because
        # `ValueError: the command did not run` -- which reads as a contradiction and is the
        # shape of a reader concluding the opposite of what happened.
        held["outcome"] = row.get("outcome") or held["outcome"]
        held["because"] = str(row.get("because") or "")[:300]
    return list(latest.values())


def _last_stage_failure_after_build(steps: list[dict[str, Any]], output: Path, run_id: str
                                    ) -> dict[str, Any] | None:
    """Find the newest failed native-stage observation since environment verification.

    A stage failure is evidence to reconsider a passing environment, not proof that the
    environment is wrong. Keep it available to the model with its original stage/attempt
    reference so the controller can decide whether reprovisioning is useful. This includes
    both command-verification failures and a later measured research-stage failure.
    """
    last_build = max((index for index, row in enumerate(steps)
                      if isinstance(row, dict) and row.get("step") == "build_the_environment"),
                     default=-1)
    latest_action = next((row for row in reversed(steps[last_build + 1:])
                          if isinstance(row, dict) and row.get("step") in {
                              "derive_a_command", "run_the_loop"}), None)
    if latest_action is None:
        return None
    if latest_action.get("step") == "run_the_loop":
        root = output / "research" / run_id
        document: dict[str, Any] = {}
        for name in ("controller_session.json", "research_report.json"):
            path = root / name
            try:
                if (not path.is_symlink() and path.is_file() and
                        path.resolve(strict=True).is_relative_to(output.resolve()) and
                        path.stat().st_size <= 8 * 1024 * 1024):
                    candidate = read_json(path)
                    if isinstance(candidate, dict) and candidate.get("run_id") == run_id:
                        document = candidate
                        break
            except (OSError, ValueError, TypeError, RuntimeError):
                continue
        rows = document.get("history")
        if not isinstance(rows, list):
            rows = document.get("rounds")
        if not isinstance(rows, list):
            rows = []
        observation = _last_research_observation(rows, run_id, research_root=root)
        failure = observation.get("failure") if isinstance(observation, dict) else None
        if not isinstance(failure, dict) or failure.get("stage") == "policy_artifact":
            return None
        stage = str(failure.get("stage") or "")
        if not stage or stage in {"research", "controller", "confirmation"}:
            return None
        return {"stage": stage[:100], "outcome": str(observation.get("status") or "")[:80],
                "because": str(failure.get("reason") or observation.get("why_not") or "")[-700:],
                "failure_kind": "research_stage_failure",
                "evidence_ref": str(failure.get("evidence") or
                                     observation.get("measurement_ref") or "")[:300]}

    arguments = latest_action.get("arguments")
    stage = str(arguments.get("stage") or "") if isinstance(arguments, dict) else ""
    outcome = str(latest_action.get("outcome") or "")
    if outcome not in {"no runnable command", "failed", "raised", "rejected"}:
        return None
    result: dict[str, Any] = {
        "stage": stage[:100], "outcome": outcome[:80],
        "because": str(latest_action.get("because") or "")[-700:],
        "evidence_ref": (f"derivation_attempts/{stage}.json"
                         if re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", stage) else ""),
    }
    if result["evidence_ref"]:
        path = output / result["evidence_ref"]
        try:
            if (not path.is_symlink() and path.is_file() and
                    path.resolve(strict=True).is_relative_to(output.resolve()) and
                    path.stat().st_size <= 2 * 1024 * 1024):
                document = read_json(path)
                rows = document.get("attempts") if isinstance(document, dict) else None
                latest = next((row for row in reversed(rows or [])
                               if isinstance(row, dict) and row.get("status") != "accepted"), {})
                excerpt = str(latest.get("error") or latest.get("said") or "")
                if excerpt:
                    result["failure_excerpt"] = excerpt[-900:]
                if latest.get("failure_kind"):
                    result["failure_kind"] = str(latest["failure_kind"])[:100]
        except (OSError, ValueError, TypeError, RuntimeError):
            pass
    return result


def _coding_agent_turn_trace(metadata: Any, decision_attempt_id: str
                             ) -> dict[str, Any] | None:
    """Accept only runtime-issued, decision-bound references to a Scheduler turn."""
    if not isinstance(metadata, dict):
        return None
    turn_id = str(metadata.get("turn_id") or "")
    process_ref = str(metadata.get("process_ref") or "")
    if (metadata.get("role") != "scheduler" or
            metadata.get("decision_attempt_id") != decision_attempt_id or
            not re.fullmatch(r"[0-9a-f]{32}", turn_id) or
            process_ref != f"agent/processes/{turn_id}.json"):
        return None
    event_count = metadata.get("event_count")
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 1:
        event_count = None
    return {"turn_id": turn_id, "process_ref": process_ref,
            "event_count": event_count}


def _safe_agent_trace_refs(values: Any) -> list[str]:
    """Keep only canonical run-local event/process references in decision records."""
    if not isinstance(values, list):
        return []
    refs: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        if re.fullmatch(r"agent/events\.jsonl#(?:decision_attempt_id|turn_id)="
                        r"[0-9a-f]{32}", value):
            refs.append(value)
            continue
        match = re.fullmatch(r"agent/processes/([0-9a-f]{32})\.json", value)
        if match:
            refs.append(value)
    return list(dict.fromkeys(refs))


def _safe_receipt_parameters(argv: Any) -> list[dict[str, str]]:
    """Project scalar CLI settings without forwarding paths or credential-like values."""
    if not isinstance(argv, list):
        return []
    rows: list[dict[str, str]] = []
    pending = ""

    def append(name: str, value: str) -> None:
        nonlocal rows
        name = name.strip()
        if (not re.fullmatch(r"--?[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name) or
                re.search(r"(?:token|secret|password|credential|auth|checkpoint|weight)|"
                          r"(?:^|[-_])[^_-]*key(?:$|[-_])|"
                          r"(?:^|[-_])(?:api[-_]?key|path|file|dir|root|dataset|demo|asset)"
                          r"(?:$|[-_])", name, re.IGNORECASE)):
            return
        value = redact(str(value).strip())
        if not value:
            value = "<empty>"
        elif not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,128}", value):
            value = "<non-scalar-or-path>"
        rows.append({"name": name, "value": value})

    numeric = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
    for raw in argv[:256]:
        if not isinstance(raw, (str, int, float)):
            continue
        token = str(raw).strip()
        if not token:
            continue
        if pending:
            if token.startswith("-") and not numeric.fullmatch(token):
                append(pending, "true")
                pending = ""
            else:
                append(pending, token)
                pending = ""
                continue
        if token.startswith("-"):
            if "=" in token:
                name, value = token.split("=", 1)
                append(name, value)
            else:
                pending = token
    if pending:
        append(pending, "true")
    return rows[:96]


def _last_research_observation(rows: list[Any], run_id: str, *,
                               research_root: Path | None = None
                               ) -> dict[str, Any] | None:
    """Keep the last outcome, actual scalar score, and safe receipt facts for the controller."""
    row = next((item for item in reversed(rows) if isinstance(item, dict)), None)
    if row is None:
        return None
    label = row.get("label")
    if not label and row.get("round") == 0:
        label = "baseline"
    if not label and isinstance(row.get("round"), int) and row["round"] > 0:
        label = f"round_{row['round']}"
    if not label:
        return None
    status = row.get("status")
    if not status:
        status = "measured" if row.get("measured") is True else "not measured"
    observation = {"round": row.get("round"), "label": label,
                   "status": str(status)[:120],
                   "why_not": str(row.get("why_not") or "")[:500],
                   "measurement_ref": (f"research/{run_id}/measurements/{label}.json"
                                       if isinstance(label, str) and label else "")}
    for key in ("idea", "kind", "verdict", "undone"):
        if key in row:
            observation[key] = row[key]
    for key in ("metric_name", "metric_value", "success_rate", "measured"):
        value = row.get(key)
        if key in {"metric_value", "success_rate"}:
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                observation[key] = value
        elif isinstance(value, (str, bool)):
            observation[key] = value
    settings = row.get("settings")
    if isinstance(settings, dict):
        # Settings are useful context but can contain paths, credentials or raw task data.
        # Keep only short scalar values whose keys do not look sensitive.
        safe_settings = []
        for key, value in list(settings.items())[:64]:
            name = str(key)
            if (not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", name) or
                    re.search(r"token|secret|password|credential|auth|key|path|file|dir|"
                              r"dataset|demo|asset|checkpoint|weight", name,
                              re.IGNORECASE)):
                continue
            if isinstance(value, (str, int, float, bool)):
                rendered = redact(str(value))
                if ("[REDACTED]" not in rendered and
                        re.fullmatch(r"[A-Za-z0-9_.:+-]{1,128}", rendered)):
                    safe_settings.append({"name": name, "value": rendered})
        if safe_settings:
            observation["settings"] = safe_settings

    failure = row.get("failure")
    if isinstance(failure, dict):
        observation["failure"] = {key: failure[key] for key in
                                  ("stage", "reason", "ran", "status", "returncode",
                                   "termination_reason", "attempt_id", "evidence")
                                  if key in failure}

    # The native command discovered from source is not necessarily the command that
    # produced the measured baseline: preparation may have used a short verification budget
    # or reduced resource settings. Expose the exact attempt's safe scalar parameters so the
    # main controller reasons from what ran, not from a repository default or surveyed argv.
    if research_root is not None and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", str(label)):
        try:
            root = Path(research_root).resolve(strict=True)
            measurement_path = root / "measurements" / f"{label}.json"
            if (measurement_path.is_symlink() or not measurement_path.is_file() or
                    not measurement_path.resolve(strict=True).is_relative_to(root) or
                    measurement_path.stat().st_size > 2 * 1024 * 1024):
                return observation
            measurement = read_json(measurement_path)
            if not isinstance(measurement, dict) or measurement.get("label") != label:
                return observation
            for stage_name in ("train", "evaluate"):
                stage_row = measurement.get(stage_name) or {}
                evidence_id = str(stage_row.get("evidence_id") or "") if isinstance(
                    stage_row, dict) else ""
                if re.fullmatch(r"[a-f0-9]{32}", evidence_id):
                    observation[f"{stage_name}_evidence_id"] = evidence_id
                    observation[f"{stage_name}_evidence_ref"] = f"evidence/{evidence_id}.json"
            metric_name = measurement.get("metric", {}).get("name") \
                if isinstance(measurement.get("metric"), dict) else None
            metric_value = measurement.get("metric_value")
            if isinstance(metric_name, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}",
                                                              metric_name):
                observation["metric_name"] = metric_name
            if (isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool)
                    and math.isfinite(metric_value)):
                observation["metric_value"] = metric_value
            train = measurement.get("train")
            attempt_id = (str(train.get("attempt_id") or "")
                          if isinstance(train, dict) else "")
            if re.fullmatch(r"[a-f0-9]{32}", attempt_id):
                receipt_ref = f"research/{run_id}/attempts/{attempt_id}/receipt.json"
                receipt_path = root / "attempts" / attempt_id / "receipt.json"
                actual = {"attempt_id": attempt_id,
                          "receipt_ref": receipt_ref,
                          "reused": train.get("reused") is True}
                if (not receipt_path.is_symlink() and receipt_path.is_file() and
                        receipt_path.resolve(strict=True).is_relative_to(root) and
                        receipt_path.stat().st_size <= 512 * 1024):
                    receipt = read_json(receipt_path)
                    if (isinstance(receipt, dict) and
                            receipt.get("attempt_id") == attempt_id and
                            receipt.get("node_id") == "train"):
                        actual.update({"status": str(receipt.get("status") or "unknown")[:40],
                                       "returncode": receipt.get("returncode")
                                       if isinstance(receipt.get("returncode"), int) else None,
                                       "parameters": _safe_receipt_parameters(
                                           receipt.get("argv"))})
                observation["actual_training"] = actual
        except (OSError, RuntimeError, TypeError, ValueError):
            # State remains useful when optional evidence cannot be read; its reference stays
            # visible and the absence is not converted into a claim about the attempt.
            pass
    return observation


class Preparation:
    """One checkout, the five steps, and the records they leave."""

    def __init__(self, *, repo: Path, output: Path, client: Any, scouting: Path,
                 run_id: str = "derived", base_settings: dict[str, Any] | None = None,
                 out: Any = None, keep_only: bool = False,
                 wall_seconds: float | None = None,
                 interpreter_hint: Path | None = None):
        self.repo = Path(repo).expanduser().resolve()
        self.output = Path(output).expanduser().resolve()
        self.client = client
        self.run_id = run_id
        self.state_store = ResearchStateStore(self.output, run_id=self.run_id,
                                              repository=self.repo)
        #: Where declarations are kept. A directory rather than a file: several scouts may
        #: have run against one checkout, and which of their readings is right is a question
        #: the state reports rather than one this code answers by picking the newest.
        self.scouting = Path(scouting)
        self.base_settings = dict(base_settings or {})
        #: Where the running commentary goes. A caller may pass a stream to capture it; the
        #: default is stdout, because the reader who needs it is watching a terminal.
        self.out = out if out is not None else sys.stdout
        self.keep_only = keep_only
        self.wall_seconds = wall_seconds
        self.interpreter_hint = (Path(interpreter_hint).expanduser().absolute()
                                 if interpreter_hint is not None else None)
        self.budget: RunBudget | None = None
        self._step_budget = 8
        self.steps: list[dict[str, Any]] = []
        self.last_decision: dict[str, Any] = {}
        self.current_action: dict[str, Any] = {}
        self.last_action: dict[str, Any] = {}
        self.monitor_observation: dict[str, Any] = {}
        self.fix_observation: dict[str, Any] = {}
        self.main_agent: dict[str, Any] = {"plan": {}, "handoffs": []}
        self.recovery_after_interruption: dict[str, Any] = {}
        self.state_persistence_error = ""
        self.state_revision = 0
        self.decision_revision = 0
        #: Every attempt the derivation made in the current step, written when the step ends.
        self._attempts: list[dict[str, Any]] = []
        self._derivation_evidence: dict[str, Any] = {}
        self.declaration: dict[str, Any] = {}
        self.interpreter: Path | None = None
        self.execution: dict[str, Any] = {}
        self.stages: dict[str, str] = {}
        self.parameters: dict[str, Any] = {}
        #: stage -> the row it was *verified* as. See `_resume_from_records`.
        self.verified: dict[str, dict[str, Any]] = {}
        requested_device = str(self.base_settings.get("device") or "").strip()
        try:
            self.decision = decide(
                prefer=requested_device if requested_device and
                requested_device.lower() != "auto" else None)
        except NoCompatibleDevice as exc:
            # The controller can still read, declare and report while no GPU may safely be
            # used. Keep the refusal as state instead of crashing before the durable run
            # journal exists; execution-capable steps check this value and fail closed.
            self.decision = ComputeDecision(
                device="unavailable", device_index=0,
                environment={"CUDA_VISIBLE_DEVICES": ""}, why=str(exc),
                evidence={"resource_unavailable": True,
                          "error": f"{type(exc).__name__}: {exc}",
                          **dict(exc.evidence)})
        self.decisions: Decisions | None = None
        self._resume_from_records()
        try:
            self.decisions = Decisions(self.output)
        except (OSError, ValueError, TypeError) as exc:
            self.decisions = None
            self.state_persistence_error = redact(
                f"controller decision log is not trustworthy: "
                f"{type(exc).__name__}: {exc}")[:400]

    def _resume_from_records(self) -> None:
        """Pick up whatever earlier runs left, because that is what the records are for.

        The state has always been read from them -- `state()` exists to show a reader what is
        on disk -- but the *steps* did not read them, so a preparation that had already built
        an environment and verified two commands would build the environment again. On LIBERO
        that was not merely wasteful: the build's first command is a `conda create` into the
        prefix that already held the interpreter, so a run that had lost nothing recreated the
        environment and destroyed the verified one on its way past.

        Every line here is a record read, not a decision made: a declaration on file, an
        interpreter the environment reported, commands an earlier derivation verified. A step
        that finds its work already done says so and the model chooses again.
        """
        try:
            shared_state = self.state_store.load()
        except ResearchStateError as exc:
            shared_state = None
            self.state_persistence_error = redact(str(exc))[:400]
        if isinstance(shared_state, dict):
            rows = shared_state.get("steps")
            if isinstance(rows, list):
                self.steps = [dict(row) for row in rows if isinstance(row, dict)]
            revision = shared_state.get("state_revision")
            if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 0:
                self.state_revision = revision
            decision_revision = shared_state.get("decision_revision")
            if (isinstance(decision_revision, int) and
                    not isinstance(decision_revision, bool) and decision_revision >= 0):
                self.decision_revision = decision_revision
            for key in ("last_decision", "last_action"):
                value = shared_state.get(key)
                if isinstance(value, dict):
                    setattr(self, key, dict(value))
            phases = shared_state.get("phases")
            main_phase = phases.get("main_agent") if isinstance(phases, dict) else None
            if isinstance(main_phase, dict) and isinstance(main_phase.get("memory"), dict):
                self.main_agent = dict(main_phase["memory"])
            monitor_phase = phases.get("monitor") if isinstance(phases, dict) else None
            monitor_observation = (monitor_phase.get("latest_observation")
                                   if isinstance(monitor_phase, dict) else None)
            if isinstance(monitor_observation, dict):
                self.monitor_observation = dict(monitor_observation)
            fix_phase = phases.get("fix") if isinstance(phases, dict) else None
            fix_observation = (fix_phase.get("latest_observation")
                               if isinstance(fix_phase, dict) else None)
            if isinstance(fix_observation, dict):
                self.fix_observation = dict(fix_observation)
            saved_recovery = shared_state.get("recovery_after_interruption")
            if isinstance(saved_recovery, dict):
                self.recovery_after_interruption = dict(saved_recovery)
            action = shared_state.get("current_action")
            phases = shared_state.get("phases")
            research_phase = (phases.get("research") if isinstance(phases, dict) else {})
            parent_action = (action.get("parent_action")
                             if isinstance(action, dict) and
                             isinstance(action.get("parent_action"), dict) else
                             research_phase.get("parent_action")
                             if isinstance(research_phase, dict) else None)
            if (shared_state.get("status") == "running" and isinstance(action, dict) and
                    action.get("status") == "running" and
                    not self.recovery_after_interruption):
                identity = action.get("process_identity")
                if not isinstance(identity, dict):
                    # A stage-start event is committed before its subprocess is launched;
                    # the on_start observer then writes the process identity into the
                    # attempt receipt before appending the identity event. If the controller
                    # dies or the event append fails in that narrow window, the receipt is
                    # the durable identity source. Resolve only an in-run, non-symlink path
                    # whose run and attempt identities match the active action.
                    evidence_ref = action.get("receipt_ref") or action.get("process_ref")
                    if isinstance(evidence_ref, str) and evidence_ref:
                        raw_evidence = self.output / evidence_ref
                        candidate = raw_evidence.resolve()
                        if (not raw_evidence.is_symlink() and
                                candidate.is_relative_to(self.output.resolve()) and
                                candidate.is_file()):
                            try:
                                receipt = read_json(candidate)
                            except (OSError, ValueError, TypeError):
                                receipt = {}
                            expected_attempt = action.get("attempt_id")
                            if (isinstance(receipt, dict) and
                                    receipt.get("run_id") == self.run_id and
                                    receipt.get("attempt_id") == expected_attempt and
                                    isinstance(receipt.get("process_identity"), dict)):
                                identity = receipt["process_identity"]
                self.recovery_after_interruption = {
                    "action": dict(action),
                    "parent_action": (dict(parent_action)
                                      if isinstance(parent_action, dict) else None),
                    "process_identity": dict(identity) if isinstance(identity, dict) else None,
                    "process_status": (inspect_process_identity(identity)
                                      if isinstance(identity, dict) else
                                      {"status": "not_recorded",
                                       "why": "no durable process identity was recorded"}),
                    "reason": "the prior run ended while this action was active; its outcome "
                              "is unknown and must be reconciled from receipts before repeating it"}
        history_paths = (() if shared_state is not None or self.state_persistence_error else
                         (self.output / f"preparation_{self.run_id}.json",))
        for history_path in history_paths:
            try:
                history = read_json(history_path) if history_path.is_file() else {}
            except (OSError, ValueError, TypeError):
                history = {}
            if not isinstance(history, dict):
                continue
            same_run = not history.get("run_id") or history.get("run_id") == self.run_id
            try:
                same_repository = Path(str(history.get("repository") or "")).resolve() == \
                    self.repo
            except (OSError, RuntimeError, ValueError):
                same_repository = False
            if not (same_run and same_repository):
                continue
            rows = history.get("steps")
            if not isinstance(rows, list):
                continue
            self.steps = [dict(row) for row in rows if isinstance(row, dict)]
            break

        found, _ = self._declaration_candidates()
        if found:
            _, _, self.declaration = max(found, key=lambda row: row[0])
        try:
            self.interpreter = provision.env_python(self.output)
        except Exception:                                            # noqa: BLE001
            self.interpreter = None
        kept = self.output / "derived_stages.json"
        if kept.is_file():
            try:
                for stage, row in json.loads(kept.read_text(encoding="utf-8")).items():
                    if isinstance(row, dict) and row.get("source"):
                        self.stages[stage] = str(row["source"])
                        self.parameters[stage] = row.get("parameters") or {}
                        # **And the row it was verified as.** A stage is two records of one
                        # thing: what a fresh reading of the checkout says about its entry
                        # point, and what the derivation settled -- the working directory, the
                        # environment, the flags -- in order to get it to run. Both are called
                        # the stage's row and only the second has the `PATH` that made the
                        # command work.
                        #
                        # RoboTwin's evaluation ran without it and died on
                        # `ModuleNotFoundError: No module named 'einops'`: the recorded
                        # environment put `.venv_robotwin/bin` first on `PATH`, the re-read row
                        # put nothing there, and `python3` in the benchmark's shell script
                        # resolved to an interpreter without the dependency. The command was
                        # verified and then run as a different command.
                        settled = dict(row.get("row") or {})
                        if settled:
                            self.verified[stage] = settled
            except (OSError, ValueError, TypeError):
                pass
        bound = self.output / "metric_binding.json"
        if bound.is_file() and self.declaration:
            try:
                record = read_json(bound)
                target = str(record.get("score_target") or "evaluate")
                source = self.stages.get(target, "")
                if (source and record.get("stage_source_sha256") ==
                        hashlib.sha256(source.encode()).hexdigest()):
                    self._set_primary_metric(record["primary_metric"])
                    if "guardrail_metrics" in record:
                        self._set_metric_guardrails(record["guardrail_metrics"])
            except (OSError, ValueError, TypeError, KeyError):
                pass

    def _set_primary_metric(self, metric: dict[str, Any]) -> None:
        declared = {**self.declaration}
        goal = {**(declared.get("research_goal") or {})}
        goal["primary_metric"] = metric
        declared["research_goal"] = goal
        MetricSpec.from_declaration(declared)
        self.declaration = declared

    def _set_metric_guardrails(self, metrics: list[dict[str, Any]]) -> None:
        from .metric_guardrails import specifications
        goal = {**(self.declaration.get("research_goal") or {}), "guardrail_metrics": metrics}
        declared = {**self.declaration, "research_goal": goal}
        specifications(declared)
        self.declaration = declared

    def _metric_bound(self) -> bool:
        if self.main_agent.get("metric_revalidation_required"):
            return False
        try:
            if not isinstance(((self.declaration.get("research_goal") or {}).get(
                    "primary_metric") or (self.declaration.get("task_contract") or {}).get(
                    "primary_metric")), dict):
                return False
            MetricSpec.from_declaration(self.declaration)
            return True
        except (TypeError, ValueError):
            return False

    # -- the state, read from the records ------------------------------------------------

    def _source_repository(self) -> Path:
        """Follow bounded snapshot provenance to the original repository checkout."""
        current = self.repo
        snapshot = self.output / "workspace_snapshot.json"
        visited: set[Path] = set()
        # A run may be made from another run's isolated checkout. Following only the
        # immediate snapshot makes the benchmark name "checkout", so its just-written
        # declaration cannot be found by the name-based candidate index. Follow the bounded
        # provenance chain to the original source directory, checking every destination and
        # refusing cycles rather than trusting an arbitrary snapshot path.
        for _ in range(8):
            if not snapshot.is_file():
                break
            try:
                row = read_json(snapshot)
                source_value = str(row.get("source") or "").strip()
                if not source_value:
                    break
                destination = Path(str(row.get("destination") or "")).expanduser().resolve()
                source = Path(source_value).expanduser().resolve()
                if destination != current or not source.name or source == current:
                    break
                if source in visited:
                    break
                visited.add(source)
                current = source
                # A parent run's manifest describes the isolated source we just followed.
                # Only follow it when it explicitly names this path as its destination.
                snapshot = current.parent / "workspace_snapshot.json"
            except (OSError, ValueError, TypeError):
                break
        return current

    def _benchmark_name(self) -> str:
        """Use the original repository identity when an isolated copy is named checkout."""
        return self._source_repository().name.lower()

    def _declaration_candidates(self) -> tuple[list[tuple[float, str, dict]], list[tuple[str, str]]]:
        return usable_declarations(self.scouting, self._benchmark_name())

    def _score_target(self) -> str:
        answer = self.execution
        if not answer and (self.output / "execution.json").is_file():
            try:
                answer = read_json(self.output / "execution.json")
            except (OSError, ValueError):
                answer = {}
        graph = (answer or {}).get("execution_graph") or {}
        return str(graph.get("score_target") or "evaluate")

    def _surveyed_runnable_stages(self) -> dict[str, dict[str, Any]]:
        """Return only stage claims explicitly marked runnable by the repository survey."""
        answer = self.execution
        if not answer and (self.output / "execution.json").is_file():
            try:
                answer = read_json(self.output / "execution.json")
            except (OSError, ValueError, TypeError):
                answer = {}
        rows = (answer or {}).get("stages") or {}
        if not isinstance(rows, dict):
            return {}
        return {str(name): row for name, row in rows.items()
                if isinstance(row, dict) and row.get("available") is True}

    def _declaration_refresh_is_justified(
            self, *, research_progress: dict[str, Any] | None = None,
            research_options: dict[str, Any] | None = None) -> bool:
        """Offer declaration only when it is missing or new evidence warrants a refresh."""
        if (research_progress or {}).get("status") in {
                "paused", "running", "finalizing", "completed", "interrupted",
                "unreadable", "identity_mismatch"}:
            return False
        if not self.declaration:
            return True
        declared_at = max((index for index, row in enumerate(self.steps)
                           if row.get("step") == "declare"), default=-1)
        for row in self.steps[declared_at + 1:]:
            if row.get("step") == "read_the_checkout" and row.get("outcome") == "done":
                return True
            if (row.get("step") in {"build_the_environment", "derive_a_command",
                                    "bind_metric"} and
                    row.get("outcome") in {"raised", "failed", "rejected"}):
                return True
        return False

    def _verified_score_output_available(self) -> bool:
        """Whether metric binding has an accepted score-stage result to inspect."""
        target = self._score_target()
        if target not in self.stages:
            return False
        try:
            attempts = read_json(self.output / "derivation_attempts" / f"{target}.json")
        except (OSError, TypeError, ValueError):
            return False
        rows = attempts.get("attempts") if isinstance(attempts, dict) else None
        latest = next((row for row in reversed(rows or []) if isinstance(row, dict)), {})
        if latest.get("status") != "accepted":
            return False
        artifact = latest.get("verified_artifact")
        candidates = (artifact.get("structured_candidates")
                      if isinstance(artifact, dict) else None)
        return bool(str(latest.get("said") or "").strip() or candidates)

    def _research_loop_ready(self) -> bool:
        """Share the exact deterministic readiness condition used by the research action."""
        return bool(self.declaration and self._score_target() in self.stages and
                    self.interpreter and self._metric_bound() and
                    not self._selected_train_command_missing())

    def _selected_train_command_missing(self) -> bool:
        """A selected producer cannot be replaced by an incidental checkpoint on disk."""
        path = self.output / "selected_path.json"
        if path.is_symlink():
            return True
        if not path.is_file():
            return False
        try:
            selected = read_json(path)
        except (OSError, ValueError, TypeError):
            return True
        if not isinstance(selected, dict) or not isinstance(selected.get("stages"), list):
            return True
        return "train" in selected["stages"] and "train" not in self.stages

    def _research_options(self, progress: dict[str, Any] | None,
                          execution: dict[str, Any]) -> dict[str, Any]:
        """Expose only audited, currently selectable ideas to the main controller.

        The library is an input to the controller, not an inner selector. When a missing
        score path means only structural/code ideas can help, do not present parameter ideas
        as executable choices. The research kernel repeats this eligibility check before it
        accepts the chosen label.
        """
        if not progress or progress.get("status") != "paused":
            return {"status": "not_paused", "items": []}
        root = self.output / "research" / self.run_id
        path = root / "ideas.json"
        if path.is_symlink():
            return {"status": "unsafe_idea_library", "items": [],
                    "evidence_ref": f"research/{self.run_id}/ideas.json"}
        if not path.is_file():
            return {"status": "empty", "items": [],
                    "evidence_ref": f"research/{self.run_id}/ideas.json"}
        try:
            document = read_json(path)
            if not isinstance(document, dict) or document.get("schema_version") != 1:
                raise ValueError("idea library identity/schema is invalid")
            library = IdeaLibrary(path)
            session = read_json(root / "controller_session.json")
            if (not isinstance(session, dict) or session.get("run_id") != self.run_id or
                    Path(str(session.get("repository") or "")).expanduser().resolve() !=
                    self.repo or session.get("status") != "paused"):
                raise ValueError("paused research session identity changed")
        except (OSError, ValueError, TypeError) as exc:
            return {"status": "unreadable", "items": [],
                    "evidence_ref": f"research/{self.run_id}/ideas.json",
                    "why": f"{type(exc).__name__}: {exc}"[:300]}

        history = session.get("history")
        if not isinstance(history, list):
            return {"status": "unreadable", "items": [],
                    "evidence_ref": f"research/{self.run_id}/controller_session.json",
                    "why": "paused history is not a list"}
        generation_attempted = False
        generation_path = root / "idea_generation.json"
        if generation_path.is_symlink():
            generation_attempted = True
        elif generation_path.is_file():
            try:
                generation = read_json(generation_path)
                generation_attempted = (
                    not isinstance(generation, dict) or
                    generation.get("next_round") == session.get("next_round"))
            except (OSError, ValueError, TypeError):
                generation_attempted = True
        measuring = any(isinstance(row, dict) and
                        row.get("metric_value", row.get("success_rate")) is not None
                        for row in history)
        graph = execution.get("execution_graph")
        commands_exist = False
        if isinstance(graph, dict) and graph.get("score_target"):
            try:
                from .execution_graph import ExecutionGraph
                parsed = ExecutionGraph(graph)
                commands_exist = all(self._stage_has_command(name)
                                     for name in parsed.order_for(
                                         str(graph["score_target"])))
            except (KeyError, TypeError, ValueError):
                commands_exist = False
        else:
            source_policy = ((self.declaration.get("task_contract") or {}).get(
                "policy_representation") == "source")
            commands_exist = (self._stage_has_command("evaluate") and
                              (self._stage_has_command("train") or
                               bool(self._shipped_checkpoint()) or source_policy))
        kinds_wanted = (None if measuring or commands_exist else {"code", "algo"})
        training_parameters = self.parameters.get("train")
        if not training_parameters:
            training_parameters = ((execution.get("stages") or {}).get("train") or {}).get(
                "parameters")
        try:
            declared_space = space_from(self.declaration)
        except (KeyError, TypeError, ValueError):
            declared_space = None
        options = []
        excluded: list[dict[str, str]] = []
        for idea in library.usable():
            if kinds_wanted is not None and idea.granularity not in kinds_wanted:
                continue
            compatible, why = execution_compatibility(
                idea, stage_parameters=training_parameters)
            if compatible and idea.granularity in {"param", "algo"}:
                if declared_space is None:
                    compatible, why = False, "the declaration has no usable optimization space"
                else:
                    compatible, why = declared_space_compatibility(
                        idea, space=declared_space)
            if compatible:
                options.append(idea)
            else:
                excluded.append({"label": idea.label, "because": why})
        return {"status": "available" if options else "empty", "items": [
                    {"label": idea.label, "granularity": idea.granularity,
                     "mechanism": idea.mechanism, "change": idea.change,
                     "risk": idea.risk, "why": idea.why,
                     "evidence": idea.evidence, "times_tried": idea.times_tried}
                    for idea in options],
                "eligible_granularities": (sorted(kinds_wanted) if kinds_wanted else
                                           ["param", "code", "algo"]),
                "excluded_incompatible": excluded[:40],
                "generation_attempted": generation_attempted,
                "next_round": session.get("next_round"),
                "evidence_ref": f"research/{self.run_id}/ideas.json"}

    def _stage_has_command(self, stage: str) -> bool:
        return stage in self.stages and bool(self.stages.get(stage))

    def state(self) -> dict[str, Any]:
        """What is known, as facts. Read from the records a person would read.

        Every line here is a reading and not a decision: how many candidates exist and which
        ones were refused and why, what the last step's failure said, which stages have a
        command. The decision is the model's and it is made from this.
        """
        found, refused = self._declaration_candidates()
        held = json.loads((self.output / "environment.json").read_text(encoding="utf-8")) \
            if (self.output / "environment.json").is_file() else {}
        kept = json.loads((self.output / "derived_stages.json").read_text(encoding="utf-8")) \
            if (self.output / "derived_stages.json").is_file() else {}
        answer = self.execution or (json.loads((self.output / "execution.json").read_text(
            encoding="utf-8")) if (self.output / "execution.json").is_file() else {})
        probe_context = {"stage_paths": {
            name: {"entrypoint": str(row.get("entrypoint") or "")}
            for name, row in self._surveyed_runnable_stages().items()
            if row.get("entrypoint")}}
        probes = held.get("probes") or []
        probe_faults = [
            {"index": index, "because": violation[:300]}
            for index, probe in enumerate(probes[:32]) if isinstance(probe, str)
            for violation in [provision.probe_stage_violation(probe, probe_context)]
            if violation]
        runtime_failure = _last_stage_failure_after_build(self.steps, self.output, self.run_id)
        held_verdict = held.get("verdict") or {}
        if probe_faults:
            verification_status = "stale_probe_contract"
        elif held_verdict.get("passed") and runtime_failure:
            verification_status = "passed_but_later_stage_failed"
        elif held_verdict.get("passed"):
            verification_status = "passed"
        elif held:
            verification_status = "failed"
        else:
            verification_status = "unknown"
        stages = {name: ("has a command" if name in self.stages else
                         ("kept from an earlier run" if name in kept else "no command"))
                  for name in (answer.get("stages") or {})}
        surveyed_stages = {
            name: {
                "available": bool(row.get("available")),
                "entrypoint": str(row.get("entrypoint") or ""),
                "invocation": str(row.get("invocation") or "")[:700],
                "artifact": str(row.get("artifact") or "")[:300],
                "level": str(row.get("level") or "")[:160],
                "why": str(row.get("why") or "")[:400],
            }
            for name, row in (answer.get("stages") or {}).items()
            if isinstance(row, dict)
        }
        score_target = self._score_target()
        score_ready = score_target in self.stages
        score_row = (answer.get("stages") or {}).get(score_target) or {}
        needs_checkpoint = "checkpoint" in str(score_row.get("invocation") or "").lower()
        workflow_input_digest = object_digest({
            "execution": answer, "declaration": self.declaration,
            "task": self._declared_task(),
        })
        workflow_selection: dict[str, Any] = {
            "status": "not_selected", "input_digest": workflow_input_digest,
            "score_target": score_target,
            "requested_task": self._declared_task(),
        }
        selected_path_file = self.output / "selected_path.json"
        if selected_path_file.is_symlink():
            workflow_selection["status"] = "unsafe_selection_record"
        elif selected_path_file.is_file():
            try:
                selected_path = read_json(selected_path_file)
            except (OSError, ValueError, TypeError):
                selected_path = {}
                workflow_selection["status"] = "unreadable_selection_record"
            if isinstance(selected_path, dict):
                review = selected_path.get("handoff_review")
                review = review if isinstance(review, dict) else {}
                workflow_selection["status"] = (
                    "selected_current_inputs" if
                    selected_path.get("input_digest") == workflow_input_digest and
                    review.get("compatible") is True
                    else "selected_for_different_inputs")
                workflow_selection["selected_stages"] = selected_path.get("stages") or []
                workflow_selection["reason"] = str(selected_path.get("why") or "")[:600]
                workflow_selection["handoff_review"] = review
        if workflow_selection["status"] == "not_selected":
            attempts_file = self.output / "path_selection_attempts.json"
            if attempts_file.is_symlink():
                workflow_selection["status"] = "unsafe_attempt_record"
            elif attempts_file.is_file():
                try:
                    prior = read_json(attempts_file)
                except (OSError, ValueError, TypeError):
                    prior = {}
                if isinstance(prior, dict):
                    prior_rows = prior.get("rows") or prior.get("attempts") or []
                    if isinstance(prior_rows, list) and prior_rows:
                        bound_digest = prior.get("input_digest")
                        workflow_selection["status"] = (
                            "rejected_current_inputs" if
                            bound_digest == workflow_input_digest else
                            "rejections_unbound_to_current_inputs" if not bound_digest else
                            "rejections_for_different_inputs")
                        workflow_selection["attempt_input_digest"] = bound_digest
                        workflow_selection["prior_rejections"] = [
                            {"stages": row.get("stages"),
                             "why_rejected": str(row.get("why_rejected") or "")[:600]}
                            for row in prior_rows[-3:] if isinstance(row, dict)
                        ]
        train_row = (answer.get("stages") or {}).get("train") or {}
        target_row = (answer.get("stages") or {}).get(score_target) or {}
        train_row = train_row if isinstance(train_row, dict) else {}
        target_row = target_row if isinstance(target_row, dict) else {}
        workflow_selection["train_score_same_entrypoint"] = bool(
            train_row.get("entrypoint") and
            train_row.get("entrypoint") == target_row.get("entrypoint"))
        missing = []
        if not self.declaration:
            missing.append("a declaration")
        if not self.interpreter:
            missing.append("an interpreter")
        if not score_ready:
            missing.append(f"a verified {score_target} score command")
        if self._selected_train_command_missing():
            missing.append("a verified train command for the selected workflow")
        if not self._metric_bound():
            missing.append("an explicit primary metric bound to native output")
        score_reason = ("missing: " + ", ".join(missing) if missing else
                        f"the research loop can start: score target {score_target} has a "
                        "verified command and declaration, with the primary metric bound; "
                        "the first successful measurement is still unproven")
        research_progress: dict[str, Any] | None = None
        research_root = self.output / "research" / self.run_id
        research_report_path = research_root / "research_report.json"
        research_session_path = research_root / "controller_session.json"
        report: dict[str, Any] | None = None
        report_error = False
        if research_report_path.is_file():
            try:
                candidate = read_json(research_report_path)
                if (isinstance(candidate, dict) and candidate.get("run_id") == self.run_id and
                        Path(str(candidate.get("repo") or "")).expanduser().resolve() ==
                        self.repo):
                    report = candidate
                else:
                    research_progress = {"status": "identity_mismatch",
                                         "evidence": str(research_report_path)}
            except (OSError, ValueError, TypeError):
                # A malformed report is not a new experiment opportunity.
                report_error = True
        if research_session_path.is_file() and research_progress is None:
            try:
                session = read_json(research_session_path)
                if (not isinstance(session, dict) or session.get("schema_version") != 1 or
                        session.get("run_id") != self.run_id or
                        Path(str(session.get("repository") or "")).expanduser().resolve() !=
                        self.repo):
                    research_progress = {"status": "identity_mismatch",
                                         "evidence": str(research_session_path)}
                else:
                    status = str(session.get("status") or "unreadable")
                    rows = session.get("history") or []
                    if not isinstance(rows, list):
                        rows = []
                    if status not in {"paused", "running", "finalizing", "completed",
                                      "interrupted"}:
                        status = "unreadable"
                    if status == "completed" and report is None:
                        status = "unreadable"
                    best = (report or {}).get("best") or {}
                    if not isinstance(best, dict):
                        best = {}
                    research_progress = {
                        "status": status,
                        "round_count": len(rows),
                        "measured_rounds": sum(
                            1 for row in rows if isinstance(row, dict) and
                            (row.get("status") == "measured" or row.get("measured") is True)),
                        "best_candidate": best.get("name") or best.get("label"),
                        "next_round": session.get("next_round"),
                        "planned_rounds": session.get("rounds"),
                        # Keep the held-out split opaque to the Scheduler. It needs the
                        # lifecycle status to know whether more work is legal, never the
                        # score, held-out settings, candidate comparison or failure detail.
                        "confirmation": self._confirmation_controller_view(
                            (report or {}).get("confirmation")),
                        "report_ref": f"research/{self.run_id}/research_report.json",
                    }
                    interruption = session.get("interruption")
                    reconciliation = session.get("reconciliation")
                    if status == "interrupted" and isinstance(interruption, dict):
                        research_progress["interruption"] = dict(interruption)
                        if isinstance(interruption.get("boundary"), dict):
                            research_progress["interruption_boundary"] = dict(
                                interruption["boundary"])
                        pending_action = session.get("pending_action")
                        if isinstance(pending_action, dict):
                            research_progress["pending_action"] = dict(pending_action)
                        research_progress["reconciliation"] = (
                            dict(reconciliation) if isinstance(reconciliation, dict) else {})
                        resolution = session.get("interruption_resolution")
                        if isinstance(resolution, dict):
                            research_progress["interruption_resolution"] = dict(resolution)
                        research_progress["safe_to_resume"] = (
                            interruption.get("controller_action") == "baseline" and
                            isinstance(reconciliation, dict) and
                            reconciliation.get("status") == "reconciled")
                    observation = _last_research_observation(
                        rows, self.run_id, research_root=research_root)
                    if observation:
                        research_progress["last_observation"] = observation
            except (OSError, ValueError, TypeError):
                research_progress = {"status": "unreadable",
                                     "evidence": str(research_session_path)}
        elif research_progress is None and report is not None:
            rows = report.get("rounds") or []
            best = report.get("best") or {}
            if not isinstance(rows, list):
                rows = []
            if not isinstance(best, dict):
                best = {}
            # Reports predating resumable sessions had no run_status and are terminal.
            status = str(report.get("run_status") or "completed")
            if status not in {"paused", "completed", "running", "finalizing",
                              "interrupted"}:
                status = "unreadable"
            research_progress = {
                "status": status,
                "round_count": len(rows),
                "measured_rounds": sum(
                    1 for row in rows if isinstance(row, dict) and
                    (row.get("status") == "measured" or row.get("measured") is True)),
                "best_candidate": best.get("name") or best.get("label"),
                "confirmation": self._confirmation_controller_view(
                    report.get("confirmation")),
                "report_ref": f"research/{self.run_id}/research_report.json",
            }
            observation = _last_research_observation(
                rows, self.run_id, research_root=research_root)
            if observation:
                research_progress["last_observation"] = observation
            if report.get("next_round") is not None:
                research_progress["next_round"] = report["next_round"]
            if report.get("planned_rounds") is not None:
                research_progress["planned_rounds"] = report["planned_rounds"]
        elif report_error and research_progress is None:
            research_progress = {"status": "unreadable",
                                 "evidence": str(research_report_path)}
        evaluation_verdict = self._load_supervisor_verdict()
        if research_progress is not None:
            recovery = self._unscored_baseline_recovery_status()
            research_progress["unscored_baseline_recovery"] = recovery
            observation = research_progress.get("last_observation") or {}
            failure = observation.get("failure") or {}
            if (failure.get("stage") == "policy_artifact" or
                    recovery.get("available") is True):
                score_reason = ("the evaluator command and primary metric binding are already "
                                "verified, but the recorded baseline is unscored because its "
                                "policy artifact was not safely selected; inspect the train "
                                "receipt and use recover_unscored_baseline if the persisted "
                                "recovery contract says it is available")
        metric_binding = self._metric_binding_retry_status()
        metric_binding["structured_upgrade_available"] = \
            self._structured_metric_upgrade_available()
        if metric_binding.get("retry_blocked"):
            score_reason = ("the verified evaluator output did not establish the declared "
                            "metric; do not repeat bind_metric on unchanged evidence. Re-derive "
                            f"and verify the {metric_binding['score_target']} command to obtain "
                            "new native output, then bind the metric from that output")
        research_options = self._research_options(research_progress, answer)
        from .workspace_resources import resource_view
        return {
            "repository": str(self.repo),
            "requested_task": self._declared_task(),
            "workspace_resources": resource_view(self.repo),
            "declarations_on_file": {
                "usable": [{"name": name, "axes": sum(len(v) for v in
                                                      (doc.get("optimization_space") or {}).values())}
                           for _, name, doc in found],
                "refused": [{"name": name, "why": why} for name, why in refused],
                "chosen": str(self.declaration.get("benchmark") or ""),
            },
            "stages_the_checkout_has": stages,
            "surveyed_stages": surveyed_stages,
            "workflow_selection": workflow_selection,
            "score_dependency": ("derive train first: the selected workflow requires its "
                                 "policy producer; an incidental checkpoint from a failed "
                                 "attempt is not a verified training command"
                                 if self._selected_train_command_missing() else
                                 "derive train first: evaluator names a checkpoint and "
                                 "no score command has been verified" if needs_checkpoint and
                                 "train" in (answer.get("stages") or {}) and
                                 "train" not in self.stages else "none known"),
            "environment": {
                "interpreter": str(self.interpreter) if self.interpreter else "",
                "base_python_hint": ({
                    "path": str(self.interpreter_hint),
                    "exists": self.interpreter_hint.is_file(),
                } if self.interpreter_hint else None),
                "probes": len(held.get("probes") or []),
                "verification_status": verification_status,
                "verdict_reason": str(held_verdict.get("reason") or "")[:240],
                "probe_contract_faults": probe_faults[:8],
                "probe_results": [
                    {"probe": str(row.get("probe") or row.get("command") or "")[:400],
                     "passed": row.get("ok") is True,
                     "failure_kind": str(row.get("failure_kind") or "")[:80],
                     "excerpt": str(row.get("excerpt") or "")[-500:]}
                    for row in (held.get("record") or [])
                    if isinstance(row, dict) and row.get("kind") == "probe"][-8:],
                "stage_failure_after_verification": runtime_failure,
                # Named for what it is: how many commands the *build* recorded. It used to be
                # `commands_recorded`, and a scheduler read it beside
                # `stages_the_checkout_has: {'evaluate': 'has a command'}` and concluded that
                # no stage had a command -- because a build that reused an existing
                # interpreter records none, however many commands the derivation went on to
                # verify. Two fields whose names both answer "how many commands" is a state
                # that contradicts itself to anyone reading it.
                "build_commands": len([row for row in held.get("record", [])
                    if isinstance(row, dict) and row.get("ok") and row.get("kind") != "probe"]),
                "partial_status": held.get("status") or "",
                "latest_installation_attempt": held.get("latest_attempt") or {},
                "latest_failure": provision.latest_failure(self.output, held),
            },
            "device": {"chosen": self.decision.device, "why": self.decision.why},
            "task_gpu_budget": self._task_gpu_budget_view(),
            "scheduling": scheduling_view(self.output),
            "steps_taken": _latest_per_step(self.steps),
            "last_decision": self.last_decision or None,
            "current_action": self.current_action or None,
            "last_action": self.last_action or None,
            "monitor_observation": self.monitor_observation or None,
            "fix_observation": self.fix_observation or None,
            "native_jobs": self._native_job_view(),
            "agent_tasks": self._agent_task_view(),
            "main_agent": memory_view(self.main_agent),
            "report_demo_requests": recorder.pending_requests(self.output),
            "recovery_after_interruption": self.recovery_after_interruption or None,
            "state_persistence_error": self.state_persistence_error or None,
            # This revision advances only for facts that can change the next legal
            # research decision. The shared run_state.json state_revision remains the
            # all-event audit counter; coding-agent telemetry must not stale its own
            # parent Scheduler decision.
            "state_revision": self.decision_revision,
            "available": self._available_operations(research_progress, research_options),
            "metric_binding": metric_binding,
            # What the run is for, and how far it is from being able to do it. Not a
            # decision -- the same two conditions `run_the_loop` checks before it will start
            # -- but a reader that has to infer "therefore I may run the loop" from a list of
            # facts will instead go and fix whichever fact looks broken. RoboTwin's scheduler
            # did exactly that: two usable declarations on file, an interpreter, four stages
            # with commands, and it spent its steps re-declaring because `declare` was the
            # only step that had failed.
            "to_measure": {
                "ready": not missing,
                "because": score_reason,
                "what_it_would_do": "use available native stages; training is optional when "
                                     "a verified policy already exists",
            },
            "research_progress": research_progress,
            "evaluation_verdict": evaluation_verdict,
            "research_options": research_options,
            "confirmation_requested": bool(self.base_settings.get("_confirm")),
        }

    @staticmethod
    def _confirmation_controller_view(value: Any) -> dict[str, Any]:
        """Expose lifecycle state without returning any held-out result to Scheduler."""
        if not isinstance(value, dict):
            return {"status": "not_recorded", "recorded": False}
        status = str(value.get("status") or "unknown")[:40]
        return {"status": status,
                "recorded": status in {"taken", "attempted_and_failed"}}

    def _load_supervisor_verdict(self) -> dict[str, Any] | None:
        """Read the hash-checked, run-bound independent review projection, if present."""
        path = self.output / "evaluation_verdict.json"
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024):
            return None
        try:
            record = read_json(path)
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(record, dict):
            return None
        held = dict(record)
        claimed = held.pop("review_sha256", None)
        if (record.get("schema_version") != 1 or record.get("run_id") != self.run_id or
                Path(str(record.get("repository") or "")).expanduser().resolve() != self.repo or
                not isinstance(record.get("review"), dict) or
                not isinstance(claimed, str) or object_digest(held) != claimed):
            return None
        review = record["review"]
        return {"verdict": review.get("verdict"),
                "summary": str(review.get("summary") or "")[:900],
                "evidence_refs": [str(ref)[:120] for ref in
                                  (review.get("evidence_refs") or [])[:20]
                                  if isinstance(ref, str)],
                "packet_sha256": str(record.get("packet_sha256") or ""),
                "review_sha256": claimed,
                "reviewed_at": record.get("reviewed_at")}

    def _review_final_study(self) -> dict[str, Any]:
        """Run an independent read-only evidence audit, cached by exact audit packet."""
        packet_error = ""
        try:
            packet, allowed_refs = supervisor.build_audit_packet(
                output=self.output, repo=self.repo, run_id=self.run_id)
        except Exception as exc:  # noqa: BLE001 - report an incomplete audit as uncertainty.
            packet_error = type(exc).__name__
            packet, allowed_refs = ({"run_id": self.run_id,
                                     "audit_packet_status": "unavailable",
                                     "scope_note": "No audit evidence was safely assembled."}, [])
        packet_sha256 = supervisor.packet_digest(packet)
        projection_path = self.output / "evaluation_verdict.json"
        reviews_root = self.output / "supervisor_reviews"
        if projection_path.is_symlink() or reviews_root.is_symlink():
            raise ValueError("Supervisor review destination is unsafe")
        existing = self._load_supervisor_verdict()
        if existing and existing.get("packet_sha256") == packet_sha256:
            return existing
        reviews_root.mkdir(parents=True, exist_ok=True)
        immutable_path = reviews_root / f"{packet_sha256}.json"

        record: dict[str, Any] | None = None
        if immutable_path.is_symlink():
            raise ValueError("content-addressed Supervisor review is a symlink")
        if immutable_path.is_file():
            if immutable_path.stat().st_size > 1024 * 1024:
                raise ValueError("content-addressed Supervisor review exceeds size limit")
            cached = read_json(immutable_path)
            if (isinstance(cached, dict) and cached.get("packet_sha256") == packet_sha256 and
                    cached.get("run_id") == self.run_id and
                    cached.get("repository") == str(self.repo)):
                unsigned = dict(cached)
                cached_hash = unsigned.pop("review_sha256", None)
                if cached_hash == object_digest(unsigned):
                    record = cached
                else:
                    raise ValueError("content-addressed Supervisor review hash is invalid")
            else:
                raise ValueError("content-addressed Supervisor review identity is invalid")

        if record is None:
            model_meta: dict[str, Any] = {}
            try:
                if packet_error:
                    raise ValueError(f"audit packet unavailable ({packet_error})")
                safe_packet = _controller_model_value(
                    packet, local_roots=(self.repo, self.output))
                # Defense in depth: the packet builder itself withholds held-out numbers,
                # and the final external-model payload excludes even the non-numeric
                # confirmation summary so the result cannot echo it back to the Scheduler.
                safe_packet.pop("confirmation", None)
                prompt = json.dumps(safe_packet, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"))
                from .agent_client import role_scope
                remaining = self.budget.remaining() if self.budget else 90.0
                if remaining <= 0:
                    raise TimeoutError("run wall-clock budget exhausted before Supervisor review")
                with role_scope(self.client, "supervisor"):
                    content, metadata = self.client.chat_with_metadata(
                        supervisor.SYSTEM, prompt, max_tokens=2400,
                        timeout=min(90.0, remaining), retries=0)
                answer = execution_derive._object(str(content))
                review = supervisor.validate_response(
                    answer, allowed_refs=allowed_refs, packet=packet)
                model_meta = metadata if isinstance(metadata, dict) else {}
            except Exception as exc:  # noqa: BLE001 - a failed audit is uncertainty, not run failure.
                review = {"verdict": "uncertain",
                          "summary": "Independent review could not be completed; "
                                     f"the reviewer returned {type(exc).__name__}.",
                          "findings": [{"severity": "info", "area": "review_execution",
                                        "summary": "No independent validity conclusion was "
                                                   "produced; the run result remains unverified.",
                                        "evidence_refs": []}],
                          "evidence_refs": [],
                          "deterministic_measurement_check": {
                              "status": "not_reviewed", "scored_measurements": 0,
                              "receipt_statuses": []},
                          "verdict_scope": "evidence validity only; improvement and SOTA are separate claims"}
                model_meta = {"status": "failed", "failure_category": type(exc).__name__}
            review["summary"] = _controller_model_text(
                review.get("summary") or "", local_roots=(self.repo, self.output))[:1600]
            for finding in review.get("findings") or []:
                finding["summary"] = _controller_model_text(
                    finding.get("summary") or "", local_roots=(self.repo, self.output))[:800]
            record = {"schema_version": 1, "run_id": self.run_id,
                      "repository": str(self.repo), "reviewed_at": now(),
                      "packet_sha256": packet_sha256, "review": review,
                      "reviewer": {key: model_meta[key] for key in
                                   ("model", "runtime", "role", "status",
                                    "total_cost_usd", "failure_category")
                                   if key in model_meta and
                                   isinstance(model_meta[key],
                                              (str, int, float, bool, type(None)))}}
            record["review_sha256"] = object_digest(record)
            from .common import immutable_json
            immutable_json(immutable_path, record)

        # The top-level file is a small current projection; full, immutable reviews remain
        # in the digest-named history above. Never put held-out metric values in this record.
        atomic_json(projection_path, record)
        compact = self._load_supervisor_verdict() or {}
        try:
            self.state_store.record(
                "supervisor", "final_review_completed",
                status=str((record.get("review") or {}).get("verdict") or "uncertain"),
                details={"packet_sha256": packet_sha256,
                         "review_sha256": record.get("review_sha256"),
                         "verdict": (record.get("review") or {}).get("verdict")},
                phase_state={"evaluation_verdict": compact,
                             "packet_sha256": packet_sha256},
                event_patch={"evaluation_verdict": compact,
                             "packet_sha256": packet_sha256},
                decision_relevant=False)
        except Exception as exc:  # noqa: BLE001 - surfaced by the durable state guard.
            self.state_persistence_error = redact(
                f"Supervisor review event could not be persisted: {type(exc).__name__}")[:400]
        return compact

    def _available_operations(self, research_progress: dict[str, Any] | None = None,
                              research_options: dict[str, Any] | None = None) -> list[str]:
        progress = research_progress
        if progress is None:
            # State callers also need the persisted research summary; avoid recursively
            # building the full state while calculating the operation list.
            root = self.output / "research" / self.run_id
            session_path = root / "controller_session.json"
            path = root / "research_report.json"
            if session_path.is_file():
                try:
                    session = read_json(session_path)
                    if (isinstance(session, dict) and session.get("run_id") == self.run_id and
                            Path(str(session.get("repository") or "")).expanduser().resolve() ==
                            self.repo and session.get("status") in {
                                "paused", "running", "finalizing", "completed",
                                "interrupted"}):
                        progress = {"status": session.get("status"),
                                    "confirmation": {}}
                        interruption = session.get("interruption")
                        reconciliation = session.get("reconciliation")
                        if isinstance(interruption, dict):
                            progress["interruption"] = dict(interruption)
                            if isinstance(interruption.get("boundary"), dict):
                                progress["interruption_boundary"] = dict(
                                    interruption["boundary"])
                        if isinstance(reconciliation, dict):
                            progress["reconciliation"] = dict(reconciliation)
                        if isinstance(session.get("interruption_resolution"), dict):
                            progress["interruption_resolution"] = dict(
                                session["interruption_resolution"])
                        if (session.get("status") == "interrupted" and
                                isinstance(interruption, dict)):
                            progress["safe_to_resume"] = (
                                interruption.get("controller_action") == "baseline" and
                                isinstance(reconciliation, dict) and
                                reconciliation.get("status") == "reconciled")
                    else:
                        progress = {"status": "identity_mismatch"}
                except (OSError, ValueError, TypeError):
                    progress = {"status": "unreadable"}
            if progress is None and path.is_file():
                try:
                    row = read_json(path)
                    if (isinstance(row, dict) and row.get("run_id") == self.run_id and
                            Path(str(row.get("repo") or "")).expanduser().resolve() == self.repo):
                        progress = {"status": row.get("run_status") or "completed",
                                    "confirmation": row.get("confirmation") or {}}
                    else:
                        progress = {"status": "unreadable"}
                except (OSError, ValueError, TypeError):
                    progress = {"status": "unreadable"}
        if progress is not None and "unscored_baseline_recovery" not in progress:
            progress["unscored_baseline_recovery"] = (
                self._unscored_baseline_recovery_status())
        if research_options is None:
            research_options = self._research_options(progress, self.execution)
        available = list(OPERATIONS)
        from .scheduling import policy
        if not policy(self.output) or progress is None or progress.get("status") != "paused":
            for name in ("configure_screening", "run_screening_trial", "inspect_screening"):
                available.remove(name)
        if not policy(self.output) or not hasattr(self.client, "fork_readonly") or self.keep_only:
            for name in ("submit_research_task", "cancel_research_task", "wait_for_jobs"):
                available.remove(name)
        from .recorder import pending_requests
        if (not getattr(self.client, "supports_recorder", False) or self.keep_only or
                not pending_requests(self.output)):
            available.remove("review_report_demo")
        if (not getattr(self.client, "supports_recorder", False) or self.keep_only or
                not any(r.get("kind") == "environment_smoke" for r in pending_requests(self.output))):
            available.remove("capture_environment_demo")
        if not getattr(self.client, "supports_main_agent", False) or self.keep_only:
            available.remove("research_task")
            available.remove("update_research_plan")
            for name in ("submit_native_job", "inspect_native_job",
                         "adjust_native_job", "cancel_native_job"):
                available.remove(name)
        else:
            from .native_jobs import active_jobs
            active = active_jobs(self.output)
            if not (self.stages if policy(self.output) else "train" in self.stages):
                available.remove("submit_native_job")
            if active:
                if not policy(self.output) or len(active) >= policy(self.output).get("native_slots", 1):
                    available.remove("submit_native_job")
                for name in ("run_the_loop", "derive_a_command", "confirm_best", "build_the_environment", "retry_failed_action", "configure_screening", "run_screening_trial", "declare", "bind_metric", "recover_unscored_baseline"):
                    if name in available:
                        available.remove(name)
            else:
                for name in ("adjust_native_job", "cancel_native_job"):
                    available.remove(name)
        if not self._repair_retry_available():
            available.remove("retry_failed_action")
        if not self._declaration_refresh_is_justified(
                research_progress=progress, research_options=research_options):
            available.remove("declare")
        if not self.declaration:
            failed_declarations = sum(
                row.get("step") == "declare" and row.get("outcome") in {
                    "no usable declaration", "raised", "failed", "rejected"}
                for row in self.steps)
            if failed_declarations >= 2:
                # `declare` already includes five schema-repair calls. Two complete failed
                # actions with no usable document are enough evidence to stop this line; the
                # model must not spend the rest of the run alternating declaration, survey,
                # and build actions whose prerequisites are unchanged.
                for name in ("declare", "read_the_checkout"):
                    if name in available:
                        available.remove(name)
            # A declaration is the provisioning contract, and an interpreter is what lets a
            # derived command be checked. Hide their dependents instead of offering actions
            # whose kernel preconditions are guaranteed to return "not attempted".
            for name in ("build_the_environment", "derive_a_command", "bind_metric"):
                if name in available:
                    available.remove(name)
        elif not self.interpreter and "derive_a_command" in available:
            available.remove("derive_a_command")
        if "bind_metric" in available and (
                (self._metric_bound() and not self._structured_metric_upgrade_available()) or
                self._metric_binding_retry_status().get("retry_blocked")):
            available.remove("bind_metric")
        elif "bind_metric" in available and not self._verified_score_output_available():
            available.remove("bind_metric")
        if ("read_the_checkout" in available and
                not self._checkout_read_is_justified()):
            available.remove("read_the_checkout")
        if not self._surveyed_runnable_stages():
            # Provisioning and command derivation both require a source-derived stage to
            # act on. Offering them before one exists turns a missing survey result into
            # repeated no-op controller steps that cannot add evidence.
            for name in ("build_the_environment", "derive_a_command"):
                if name in available:
                    available.remove(name)
        if not ((progress or {}).get("unscored_baseline_recovery") or {}).get("available"):
            available.remove("recover_unscored_baseline")
        else:
            # A reusable completed train attempt is the only safe next measurement path.
            # Do not let another idea obscure or duplicate it before the master resolves the
            # already-paid baseline attempt.
            available.remove("run_the_loop")
        if not self._research_loop_ready() and "run_the_loop" in available:
            available.remove("run_the_loop")
        if not self.recovery_after_interruption:
            available.remove("reconcile_interrupted_action")
        if progress and progress.get("status") in {
                "completed", "running", "unreadable", "identity_mismatch"}:
            if "run_the_loop" in available:
                available.remove("run_the_loop")
        if (progress and progress.get("status") == "interrupted" and
                not progress.get("safe_to_resume") and "run_the_loop" in available):
            available.remove("run_the_loop")
        if not self._interrupted_candidate_discard_available(progress):
            available.remove("discard_interrupted_candidate")
        if progress and progress.get("status") == "paused":
            has_options = (research_options.get("status") == "available" and
                           bool(research_options.get("items")))
            if not has_options and "run_the_loop" in available:
                available.remove("run_the_loop")
            if (has_options or research_options.get("generation_attempted") and
                    "generate_research_ideas" in available):
                available.remove("generate_research_ideas")
            if research_options.get("status") not in {"empty", "available"}:
                for name in ("generate_research_ideas", "propose_research_idea"):
                    if name in available:
                        available.remove(name)
            budget_left = self.budget.remaining() if self.budget else 1.0
            has_actionable_research = (
                (has_options and "run_the_loop" in available) or
                "generate_research_ideas" in available)
            if has_actionable_research and budget_left > 0 and "stop" in available:
                available.remove("stop")
        else:
            for name in ("generate_research_ideas", "propose_research_idea"):
                if name in available:
                    available.remove(name)
        confirmation_state = (progress or {}).get("confirmation") or {}
        if (not self.base_settings.get("_confirm") or not progress or
                progress.get("status") != "completed" or
                confirmation_state.get("status") != "available_not_taken"):
            available.remove("confirm_best")
        if (getattr(self.client, "supports_main_agent", False) and not self.keep_only and
                not self.main_agent.get("plan")):
            # Let source discovery happen first, but require an Agent-authored roadmap
            # before any environment mutation or research workload is submitted.
            guarded = {"build_the_environment", "run_the_loop", "submit_native_job", "capture_environment_demo",
                       "run_screening_trial", "derive_a_command"}
            available = [name for name in available if name not in guarded]
        return available

    @staticmethod
    def _interrupted_candidate_discard_available(
            progress: dict[str, Any] | None) -> bool:
        """Only a reconciled, identity-matched candidate may receive a discard decision."""
        if not isinstance(progress, dict) or progress.get("status") != "interrupted":
            return False
        interruption = progress.get("interruption")
        reconciliation = progress.get("reconciliation")
        boundary = progress.get("interruption_boundary")
        if (not isinstance(interruption, dict) or
                not isinstance(reconciliation, dict) or
                not isinstance(boundary, dict) or
                reconciliation.get("status") != "reconciled" or
                boundary.get("status") != "unresolved" or
                boundary.get("pending_action_identity") != "matched"):
            return False
        action = str(interruption.get("controller_action") or "")
        attempt = boundary.get("attempt_evidence") or {}
        measurement = boundary.get("candidate_measurement") or {}
        resolution = progress.get("interruption_resolution") or {}
        return (action.startswith("round_") and action[6:].isdigit() and
                boundary.get("controller_action") == action and
                boundary.get("round") == int(action[6:]) and
                boundary.get("phase") in {"idea_selected", "preparing_idea", "prepared"} and
                isinstance(measurement, dict) and measurement.get("status") == "missing" and
                isinstance(attempt, dict) and not attempt.get("attempt_id") and
                not attempt.get("receipt_ref") and
                attempt.get("receipt_status") == "not_recorded" and
                attempt.get("process_status") == "not_recorded" and
                isinstance(boundary.get("idea"), dict) and
                bool(boundary["idea"].get("label")) and
                (not isinstance(resolution, dict) or
                 resolution.get("status") != "completed"))

    def _metric_binding_retry_status(self) -> dict[str, Any]:
        """Require fresh score-command evidence before repeating a failed metric bind."""
        target = self._score_target()
        output = ""
        try:
            rows = read_json(self.output / "derivation_attempts" / f"{target}.json").get(
                "attempts") or []
            output = next((str(row.get("said") or "") for row in reversed(rows)
                           if row.get("status") == "accepted"), "")
        except (OSError, TypeError, ValueError):
            pass
        if self._metric_bound():
            return {"status": "bound", "retry_blocked": False,
                    "score_target": target}
        failures = [(index, row) for index, row in enumerate(self.steps)
                    if isinstance(row, dict) and row.get("step") == "bind_metric" and
                    row.get("outcome") in {"not bound", "raised", "rejected"}]
        if not failures:
            return {"status": "unbound", "retry_blocked": False,
                    "score_target": target,
                    "verified_output_excerpt": output[-600:]}
        last_failure = failures[-1][0]
        fresh_score_run = any(
            isinstance(row, dict) and row.get("step") == "derive_a_command" and
            row.get("outcome") == "done" and
            (row.get("arguments") or {}).get("stage") == target
            for row in self.steps[last_failure + 1:])
        return {"status": "unbound", "retry_blocked": not fresh_score_run,
                "score_target": target,
                "reason": "a failed bind can only be retried after a new verified score-stage run",
                "verified_output_excerpt": output[-600:]}

    def _structured_metric_upgrade_available(self) -> bool:
        """Offer a one-time evidence-backed upgrade from a log aggregate to episode data."""
        try:
            spec = MetricSpec.from_declaration(self.declaration)
            if spec.source != "log":
                return False
            target = self._score_target()
            attempts = read_json(self.output / "derivation_attempts" / f"{target}.json")
            accepted = next((row for row in reversed(attempts.get("attempts") or [])
                             if row.get("status") == "accepted"), {})
            evidence = accepted.get("verified_artifact") or {}
            candidates = evidence.get("structured_candidates") or []
            if not candidates:
                return False
            binding_path = self.output / "metric_binding.json"
            binding = read_json(binding_path) if binding_path.is_file() else {}
            reviewed = str(binding.get("reviewed_structured_candidates_sha256") or "")
            return reviewed != object_digest(candidates)
        except (OSError, TypeError, ValueError, KeyError):
            return False

    def _unscored_baseline_recovery_status(self) -> dict[str, Any]:
        """Expose only receipt-bound recoveries that cannot repeat completed training."""
        root = self.output / "research" / self.run_id
        measurement_path = root / "measurements" / "baseline.json"
        retry_after_archive_failure = False
        try:
            measurement = read_json(measurement_path)
            if (not isinstance(measurement, dict) or
                    measurement.get("label") != "baseline" or
                    measurement.get("ok") is True):
                return {"available": False,
                        "why": "no failed baseline measurement is recorded"}
            if measurement.get("where") == "policy_archive":
                failed_recovery = measurement.get("recovery") or {}
                attempt_hint = str(failed_recovery.get("training_attempt_id") or "")
                if (failed_recovery.get("kind") !=
                        "reevaluate_after_prestart_resource_block" or
                        failed_recovery.get("training_reused") is not True or
                        not re.fullmatch(r"[a-f0-9]{32}", attempt_hint) or
                        not str(measurement.get("archive_error") or "").startswith(
                            "FileExistsError:") or measurement.get("evaluate")):
                    return {"available": False,
                            "why": "policy archival failure is not the bounded pre-evaluation retry case"}
                attempt_root = root / "attempts" / attempt_hint
                backup_path = attempt_root / "baseline_unscored.json"
                first_state_path = attempt_root / "baseline_recovery.json"
                backup_resolved = backup_path.resolve(strict=True)
                backup_resolved.relative_to(root.resolve())
                original = read_json(backup_resolved)
                first_state = read_json(first_state_path)
                backup_ref = f"attempts/{attempt_hint}/baseline_unscored.json"
                stored_original_digest = first_state.get("original_measurement_sha256")
                if (backup_path.is_symlink() or not isinstance(original, dict) or
                        original.get("where") != "evaluate" or original.get("ok") is True or
                        first_state.get("status") != "failed" or
                        first_state.get("recovery_kind") !=
                        "reevaluate_after_prestart_resource_block" or
                        first_state.get("training_attempt_id") != attempt_hint or
                        (stored_original_digest is not None and
                         stored_original_digest != object_digest(original)) or
                        first_state.get("measurement_sha256") != object_digest(measurement) or
                        first_state.get("evaluation_attempt_id") or
                        failed_recovery.get("original_measurement_ref") != backup_ref or
                        failed_recovery.get("superseded_evaluation_attempt_id") !=
                        (original.get("evaluate") or {}).get("attempt_id") or
                        (original.get("train") or {}).get("attempt_id") != attempt_hint or
                        not (original.get("policy_artifact") or {}).get("path")):
                    return {"available": False, "attempt_id": attempt_hint,
                            "why": "prior recovery lacks exact pre-evaluation archive-failure evidence"}
                measurement = original
                retry_after_archive_failure = True
            recovery_kind = ""
            blocked_evaluation_id = ""
            if (measurement.get("where") == "policy_artifact" and
                    measurement.get("status") == "unscored"):
                recovery_kind = "revalidate_policy_selection"
                outcome = measurement.get("training_outcome") or {}
                attempt_id = str(outcome.get("attempt_id") or
                                 measurement.get("attempt_id") or "")
            elif measurement.get("where") == "evaluate":
                recovery_kind = "reevaluate_after_prestart_resource_block"
                train = measurement.get("train") or {}
                attempt_id = str(train.get("attempt_id") or "")
                evaluation = measurement.get("evaluate") or {}
                blocked_evaluation_id = str(evaluation.get("attempt_id") or "")
                if (not re.fullmatch(r"[a-f0-9]{32}", blocked_evaluation_id) or
                        not re.fullmatch(r"[a-f0-9]{32}", attempt_id)):
                    return {"available": False,
                            "why": "failed evaluation is not linked to exact train/evaluate attempts"}
                raw_evaluation = root / "attempts" / blocked_evaluation_id / "receipt.json"
                evaluation_path = raw_evaluation.resolve()
                if (raw_evaluation.is_symlink() or
                        not evaluation_path.is_relative_to(root.resolve()) or
                        not evaluation_path.is_file()):
                    return {"available": False, "attempt_id": attempt_id,
                            "why": "the blocked evaluation receipt is missing or unsafe"}
                evaluation_receipt = read_json(evaluation_path)
                if (evaluation_receipt.get("run_id") != self.run_id or
                        evaluation_receipt.get("attempt_id") != blocked_evaluation_id or
                        evaluation_receipt.get("node_id") != "evaluate" or
                        evaluation_receipt.get("status") != "blocked" or
                        evaluation_receipt.get("ran") is not False or
                        evaluation_receipt.get("command_started") is not False or
                        evaluation_receipt.get("process_identity") or
                        evaluation_receipt.get("termination_reason") not in {
                            "gpu_unavailable", "gpu_lease_unavailable", "gpu_lease_busy"} or
                        evaluation_receipt.get("settings_digest") != object_digest(
                            measurement.get("settings") or {})):
                    return {"available": False, "attempt_id": attempt_id,
                            "why": "evaluation was not proven blocked before process start"}
                if not self.decision.on_gpu:
                    return {"available": False, "attempt_id": attempt_id,
                            "why": "the matching GPU is not currently available for a pinned retry"}
            else:
                return {"available": False,
                        "why": "baseline failure is not an approved recovery boundary"}
            if not re.fullmatch(r"[a-f0-9]{32}", attempt_id):
                return {"available": False,
                        "why": "baseline failure has no exact completed train attempt"}
            session = read_json(root / "controller_session.json")
            report = read_json(root / "research_report.json")
            history = session.get("history") if isinstance(session, dict) else None
            rows = report.get("rounds") if isinstance(report, dict) else None
            if (not isinstance(session, dict) or not isinstance(report, dict) or
                    session.get("run_id") != self.run_id or
                    report.get("run_id") != self.run_id or
                    session.get("status") not in {"paused", "completed"} or
                    report.get("run_status") != session.get("status") or
                    not isinstance(history, list) or not history or
                    not isinstance(rows, list) or not rows or
                    not isinstance(history[0], dict) or not isinstance(rows[0], dict) or
                    history[0].get("label") != "baseline" or
                    rows[0].get("label") != "baseline" or
                    any(row.get("measured") is True or
                        isinstance(row.get("metric_value"), (int, float))
                        for row in history[1:] if isinstance(row, dict))):
                return {"available": False, "attempt_id": attempt_id,
                        "why": "controller state is not paused/completed with an unscored "
                               "baseline and no later scored arm"}
            attempt_dir = root / "attempts" / attempt_id
            recovery_state_path = (attempt_dir / "baseline_recovery_attempt_2.json"
                                   if retry_after_archive_failure else
                                   attempt_dir / "baseline_recovery.json")
            if recovery_state_path.exists():
                return {"available": False, "attempt_id": attempt_id,
                        "why": "this bounded recovery attempt already exists; inspect its receipt "
                               "rather than replaying it"}
            raw_receipt = attempt_dir / "receipt.json"
            receipt_path = raw_receipt.resolve()
            if (raw_receipt.is_symlink() or not receipt_path.is_relative_to(root.resolve()) or
                    not receipt_path.is_file()):
                return {"available": False, "attempt_id": attempt_id,
                        "why": "the exact train receipt is missing or escapes the run root"}
            receipt = read_json(receipt_path)
            protocol_path = root / "comparison_protocol.json"
            settings = measurement.get("settings")
            cwd = Path(str(receipt.get("working_directory") or "")).resolve()
            artifact = receipt.get("artifact") or {}
            recorded_protocol = str(receipt.get("comparison_protocol_sha256") or "")
            if (not isinstance(settings, dict) or receipt.get("attempt_id") != attempt_id or
                    receipt.get("run_id") != self.run_id or
                    receipt.get("node_id") != "train" or receipt.get("stage") != "train" or
                    receipt.get("status") != "completed" or receipt.get("returncode") != 0 or
                    receipt.get("termination_reason") != "normal_exit" or
                    receipt.get("settings_digest") != object_digest(settings) or
                    not protocol_path.is_file() or
                    (recorded_protocol and recorded_protocol != digest(protocol_path)) or
                    not cwd.is_relative_to(self.repo) or
                    not self.interpreter or
                    (receipt.get("argv") or [None])[0] != str(self.interpreter) or
                    artifact.get("checked") is not True or
                    not isinstance(artifact.get("matched"), int) or
                    artifact.get("matched", 0) < 1 or
                    (receipt.get("training_progress") or {}).get("status") != "observed"):
                return {"available": False, "attempt_id": attempt_id,
                        "why": "train receipt, settings, protocol, interpreter, or fresh "
                               "artifact evidence does not match"}
            selection_ref = ""
            if recovery_kind == "revalidate_policy_selection":
                selection_ref = str((measurement.get("artifact_selection") or {}).get(
                    "selection_ref") or "")
                raw_selection = self.output / selection_ref
                selection_path = raw_selection.resolve()
                if (not selection_ref or raw_selection.is_symlink() or
                        not selection_path.is_relative_to(root.resolve()) or
                        not selection_path.is_file()):
                    return {"available": False, "attempt_id": attempt_id,
                            "why": "the original policy-selection record is missing or unsafe"}
                selection = read_json(selection_path)
                if (selection.get("attempt_id") != attempt_id or
                        selection.get("status") != "abstained"):
                    return {"available": False, "attempt_id": attempt_id,
                            "why": "the original selection record is not a matching abstention"}
            else:
                from .devices import normalize_uuid
                train_uuid = str(receipt.get("gpu_device_uuid") or "")
                selected = self.decision.evidence.get("selected_device") or {}
                current_uuid = str(selected.get("uuid") or "")
                if (not train_uuid or not current_uuid or
                        normalize_uuid(train_uuid) != normalize_uuid(current_uuid)):
                    return {"available": False, "attempt_id": attempt_id,
                            "why": "the baseline checkpoint's physical GPU is not the "
                                   "currently selected GPU"}
                policy = measurement.get("policy_artifact") or {}
                raw_policy = Path(str(policy.get("path") or ""))
                try:
                    policy_path = raw_policy.resolve(strict=True)
                    policy_path.relative_to(root.resolve())
                    from .experiment_bundle import artifact_identity
                    identity_field, identity = artifact_identity(policy_path)
                except (OSError, RuntimeError, ValueError):
                    return {"available": False, "attempt_id": attempt_id,
                            "why": "the exact frozen baseline policy is missing or unsafe"}
                if (raw_policy.is_symlink() or not policy_path.is_file() or
                        policy.get(identity_field) != identity):
                    return {"available": False, "attempt_id": attempt_id,
                            "why": "the frozen baseline policy identity changed"}
            return {"available": True, "attempt_id": attempt_id,
                    "recovery_kind": recovery_kind,
                    "blocked_evaluation_attempt_id": blocked_evaluation_id,
                    "recovery_attempt": 2 if retry_after_archive_failure else 1,
                    "recovery_state_ref": str(recovery_state_path.relative_to(root)),
                    **({"selection_ref": selection_ref} if selection_ref else {}),
                    "measurement_ref": f"research/{self.run_id}/measurements/baseline.json",
                    "why": ("the exact completed training receipt, frozen policy, blocked "
                            "evaluation receipt, matching physical GPU and protocol permit "
                            "evaluation-only recovery" if recovery_kind.startswith("reevaluate")
                            else "completed train receipt and protocol-bound fresh artifacts "
                                 "can be revalidated and evaluated without retraining")}
        except (OSError, TypeError, ValueError, AttributeError, IndexError):
            return {"available": False,
                    "why": "baseline recovery evidence is malformed or unreadable"}

    def _checkout_read_is_justified(self) -> bool:
        """Allow a fresh survey initially and after new actionable failure evidence.

        A repository read with no new failure or source evidence is the same operation on the
        same input. Repeating it only spends controller budget and can create a loop in which
        the model keeps rewriting a plausible stage list but never tries the newly available
        next action. The model remains responsible for choosing the response to a new failure.
        """
        if self.main_agent.get("checkout_needs_resurvey"):
            return True
        read_indexes = [index for index, row in enumerate(self.steps)
                        if row.get("step") == "read_the_checkout"]
        if not read_indexes:
            return True
        latest_read = max(read_indexes)
        latest_read_receipt = str(self.steps[latest_read].get("receipt_ref") or "")
        fix_receipt = str(self.fix_observation.get("failure_receipt_ref") or "")
        if (latest_read_receipt and fix_receipt == latest_read_receipt and
                self.fix_observation.get("status") == "assessed" and
                self.fix_observation.get("assessment") == "repair_attempted"):
            # A verified reference to an AgentFix attempt on this exact survey failure is
            # new evidence that can justify retrying the survey. The Scheduler still chooses
            # whether to retry; a Fix narrative alone is not a successful survey result.
            return True
        return any(row.get("step") != "read_the_checkout" and
                   row.get("outcome") in {"raised", "failed", "rejected"}
                   for row in self.steps[latest_read + 1:])

    def _persist_run_state(self, *, status: str) -> None:
        """Atomically checkpoint the preparation controller's facts and handoff state."""
        # Once a write's commit status is uncertain, do not issue another state write as if
        # it were a safe retry. The event may have reached disk even when atomic_json raised
        # (for example, a directory fsync failure after replace); restart/replay is the only
        # component that can resolve that ambiguity from durable evidence.
        if self.state_persistence_error:
            return
        facts = self.state()
        phase_state = {"state": facts, "steps": self.steps,
                       "last_decision": self.last_decision or None,
                       "current_action": self.current_action or None,
                       "process_identity": self.current_action.get("process_identity") or None,
                       "last_action": self.last_action or None,
                       "recovery_after_interruption": self.recovery_after_interruption or None,
                       "state_persistence_error": self.state_persistence_error or None}
        action = self.current_action or self.last_action
        event = ("action_started" if self.current_action.get("status") == "running" else
                 "action_completed" if self.last_action.get("finished_at") else
                 "state_checkpoint")
        try:
            stored = self.state_store.record(
                "preparation", event, status=status,
                details={"action": action, "decision": self.last_decision or None,
                         "step_count": len(self.steps),
                         "facts_sha256": object_digest(facts)},
                phase_state=phase_state,
                event_patch={"last_decision": self.last_decision or None,
                             "current_action": self.current_action or None,
                             "process_identity":
                                 self.current_action.get("process_identity") or None,
                             "last_action": self.last_action or None,
                             "recovery_after_interruption":
                                 self.recovery_after_interruption or None,
                             "step_count": len(self.steps)})
            self.state_revision = int(stored.get("state_revision") or self.state_revision)
            self.decision_revision = int(stored.get("decision_revision") or
                                          self.decision_revision)
        except (ResearchStateError, OSError, TypeError, ValueError) as exc:
            self.state_persistence_error = redact(
                f"run_state checkpoint failed: {type(exc).__name__}: {exc}")[:400]
            self._say(self.state_persistence_error)

    def _state_failure_report(self) -> dict[str, Any]:
        """Stop safely when durable run state can no longer be trusted or written."""
        why = self.state_persistence_error or "durable run state is unavailable"
        failure = {"step": "run_state", "outcome": "blocked", "because": why}
        if not self.steps or self.steps[-1] != failure:
            self.steps.append(failure)
        report = {"schema_version": 1, "created_at": now(), "run_id": self.run_id,
                  "repository": str(self.repo), "status": "internal_error",
                  "verified_level": ("L1" if self.verified else "L0"
                                     if self.execution else "unknown"),
                  "state_persistence_error": why, "steps": self.steps,
                  "state": self.state(), "stages_with_commands": sorted(self.stages)}
        # The primary event/snapshot path may be unavailable. Keep a best-effort local
        # diagnostic, but never run another repository action to make the report prettier.
        atomic_json(self.output / f"preparation_{self.run_id}.json", report)
        atomic_json(self.output / "feasibility.json", {
            "schema_version": 1, "repository": str(self.repo),
            "status": "internal_error", "last_verified_level": report["verified_level"],
            "conclusion_scope": "this run and this environment",
            "blocking_conditions": [{"requirement": "durable run state",
                                     "observed": why,
                                     "evidence_refs": ["run_events.json", "run_state.json"]}]})
        self._refresh_document(status="internal_error", current=why)
        return report

    # -- the steps -----------------------------------------------------------------------

    def _observe_process_start(self, process: subprocess.Popen[Any]) -> None:
        """Persist an implicit bounded-helper subprocess under the active prep action."""
        attempt_id = uuid.uuid4().hex
        identity = capture_process_identity(
            process, run_id=self.run_id, attempt_id=attempt_id, argv=process.args)
        process_ref = f"processes/{attempt_id}.json"
        process_path = self.output / process_ref
        process_path.parent.mkdir(parents=True, exist_ok=True)
        action = {**self.current_action, "step": self.current_action.get(
            "step") or "unscoped_process", "status": "running",
                  "attempt_id": attempt_id, "process_ref": process_ref,
                  "process_identity": identity}
        atomic_json(process_path, {"schema_version": 1, "run_id": self.run_id,
                                   "attempt_id": attempt_id, "status": "running",
                                   "started_at": now(), "action": self.current_action,
                                   "process_identity": identity,
                                   "argv_sha256": identity.get("argv_sha256")})
        self.current_action = action
        try:
            stored = self.state_store.record(
                "preparation", "process_started", status="running",
                details={"action": action, "process_ref": process_ref,
                         "process_identity": identity},
                phase_state={"current_action": action,
                             "process_identity": identity},
                event_patch={"current_action": action,
                             "process_identity": identity})
            self.state_revision = int(stored.get("state_revision") or self.state_revision)
            self.decision_revision = int(stored.get("decision_revision") or
                                          self.decision_revision)
        except (ResearchStateError, OSError, TypeError, ValueError) as exc:
            self.state_persistence_error = redact(
                f"process identity checkpoint failed: {type(exc).__name__}: {exc}")[:400]
            # The start observer exception makes run_process terminate the process group.
            raise

    def _new_process_evidence(self, *, decision_id: str, step: str,
                              before: set[str]) -> tuple[list[str], list[str]]:
        """Find safe child-process records created by exactly one controller action."""
        directory = self.output / "processes"
        if directory.is_symlink() or not directory.is_dir():
            return [], []
        root = self.output.resolve()
        attempt_ids: list[str] = []
        evidence_refs: list[str] = []
        try:
            paths = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return [], []
        for path in paths:
            attempt_id = path.stem
            if (attempt_id in before or not re.fullmatch(r"[0-9a-f]{32}", attempt_id) or
                    path.suffix != ".json" or path.is_symlink()):
                continue
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
                if resolved.parent != directory.resolve() or not path.is_file():
                    continue
                record = read_json(path)
            except (OSError, ValueError, TypeError, RuntimeError):
                continue
            action = record.get("action")
            identity = record.get("process_identity")
            if not isinstance(action, dict) or not isinstance(identity, dict):
                continue
            if (record.get("run_id") != self.run_id or
                    record.get("attempt_id") != attempt_id or
                    identity.get("run_id") != self.run_id or
                    identity.get("attempt_id") != attempt_id or
                    action.get("decision_id") != decision_id or
                    action.get("step") != step):
                continue
            attempt_ids.append(attempt_id)
            evidence_refs.append(f"processes/{attempt_id}.json")
        return attempt_ids, evidence_refs

    def _existing_evidence_refs(self, *references: str) -> list[str]:
        """Return only regular, run-confined evidence files for action receipts."""
        root = self.output.resolve()
        result: list[str] = []
        for reference in references:
            relative = Path(reference)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            path = root / relative
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                        if parent != root and parent.is_relative_to(root)):
                continue
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError, RuntimeError):
                continue
            if path.is_file() and not path.is_symlink():
                result.append(relative.as_posix())
        return list(dict.fromkeys(result))

    def _candidate_interruption_boundary(self, *, controller_action: str,
                                         pending_action: Any,
                                         evidence_ref: str | None,
                                         evidence_status: str,
                                         process_status: str) -> dict[str, Any] | None:
        """Persist what is known about an interrupted candidate without adopting it.

        A measurement file alone is not enough to finish a research round: the controller
        may still need to score the idea, update best-state bookkeeping, undo a code change,
        and append the decision to history. The boundary records which parts can be checked.
        Only a code patch interrupted before any benchmark attempt can be restored here, and
        only from its exact transaction/snapshot; later side effects stay unresolved for an
        explicit main-controller decision or human review.
        """
        prefix = "round_"
        suffix = controller_action[len(prefix):] if controller_action.startswith(prefix) else ""
        if not suffix.isdigit():
            return None
        if not isinstance(pending_action, dict):
            pending_action = {}
        round_index = int(suffix)
        pending_identity = ("missing" if not pending_action else
                            "matched" if pending_action.get("round") == round_index else
                            "mismatch")
        idea = pending_action.get("idea")
        idea = idea if isinstance(idea, dict) else {}
        changed = pending_action.get("changed")
        changed = changed if isinstance(changed, dict) else {}
        changed_files = changed.get("files")
        if not isinstance(changed_files, list):
            changed_files = [changed.get("file")] if changed.get("file") else []
        changed_files = sorted({str(path) for path in changed_files
                                if isinstance(path, str) and path})
        action_record = self.recovery_after_interruption.get("action") or {}
        if not isinstance(action_record, dict):
            action_record = {}

        label = f"round_{round_index}"
        measurement_ref = f"research/{self.run_id}/measurements/{label}.json"
        raw_measurement_path = self.output / measurement_ref
        measurement_path = raw_measurement_path.resolve()
        output_root = self.output.resolve()
        measurement_parents = [self.output / "research",
                               self.output / "research" / self.run_id,
                               self.output / "research" / self.run_id / "measurements"]
        if (any(path.is_symlink() for path in measurement_parents) or
                raw_measurement_path.is_symlink()):
            measurement_state = {"status": "symlink_refused", "ref": measurement_ref}
        elif not measurement_path.is_relative_to(output_root):
            measurement_state = {"status": "unsafe_path"}
        elif not measurement_path.is_file():
            measurement_state = {"status": "missing", "ref": measurement_ref}
        else:
            try:
                from .receipt_verifier import verify_measurement
                checked = verify_measurement(self.output / "research" / self.run_id, label)
                measurement_state = {
                    "status": str(checked.get("status") or "unverifiable"),
                    "ref": measurement_ref,
                    "attempt_id": checked.get("attempt_id"),
                    "checks": checked.get("checks") or {},
                    "issues": checked.get("issues") or [],
                    "limitation": checked.get("limitation"),
                }
            except (OSError, ValueError, TypeError) as exc:
                measurement_state = {
                    "status": "unverifiable", "ref": measurement_ref,
                    "issue": f"{type(exc).__name__}: {exc}"[:300],
                }

        committed_proof = DerivedResearch.verify_committed_candidate_finalization(
            run_root=self.output / "research" / self.run_id,
            run_id=self.run_id, repo=self.repo, round_index=round_index)
        if committed_proof.get("status") == "verified_committed":
            finalization = committed_proof["finalization"]
            row = committed_proof["history_row"]
            return {
                "schema_version": 1,
                "status": "resolved_committed",
                "reason_code": "round_finalization_committed_and_verified",
                "controller_action": controller_action,
                "round": round_index,
                "pending_action_identity": "committed_transaction_verified",
                "phase": "committed",
                "idea": {"label": finalization.get("idea_label"),
                         "granularity": row.get("kind"),
                         "mechanism": row.get("mechanism")},
                "attempt_evidence": {
                    "attempt_id": action_record.get("attempt_id"),
                    "receipt_ref": evidence_ref,
                    "receipt_status": evidence_status,
                    "process_status": process_status,
                    "outcome_known": True,
                },
                "candidate_measurement": measurement_state,
                "round_finalization": finalization,
                "round_history_sha256": object_digest(row),
                "receipt_proof": committed_proof.get("receipt_proof"),
                "automatic_replay": False,
                "automatic_measurement_adoption": False,
                "automatic_source_rollback": False,
                "required_next": "rebuild only the report projection from the committed "
                                 "session, then let the main controller decide the next round; "
                                 "do not rerun this candidate",
            }

        rollback = {"status": "not_attempted",
                    "because": "the candidate was not interrupted in the pre-change phase",
                    "paths": []}
        if (pending_identity == "matched" and
                pending_action.get("phase") == "preparing_idea" and
                measurement_state.get("status") == "missing" and
                evidence_ref is None and evidence_status == "not_recorded" and
                process_status == "not_recorded" and not action_record.get("attempt_id") and
                not action_record.get("process_identity")):
            rollback = self._rollback_interrupted_code_change(
                round_index=round_index, idea_label=str(idea.get("label") or ""))
            if (rollback.get("status") == "no_transaction_record" and
                    idea.get("granularity") == "code" and
                    pending_action.get("source_transaction_protocol") == 1):
                rollback = {"status": "no_source_change",
                            "because": "the durable phase marker proves this transaction "
                                      "protocol had not recorded or applied a source patch",
                            "paths": []}
            if not changed_files and rollback.get("paths"):
                changed_files = sorted(set(map(str, rollback["paths"])))
        rollback_succeeded = rollback.get("status") in {
            "rolled_back", "already_rolled_back", "no_source_change"}
        finalization = pending_action.get("round_finalization")
        finalization_summary = None
        if isinstance(finalization, dict):
            steps = finalization.get("steps")
            if isinstance(steps, dict):
                finalization_summary = {
                    "schema_version": finalization.get("schema_version"),
                    "transaction_id": finalization.get("transaction_id"),
                    "status": finalization.get("status"),
                    "round": finalization.get("round"),
                    "idea_label": finalization.get("idea_label"),
                    "measurement_sha256": finalization.get("measurement_sha256"),
                    "result_sha256": finalization.get("result_sha256"),
                    "steps": {str(name): {
                        "status": (str(value.get("status") or "malformed")
                                   if isinstance(value, dict) else "malformed"),
                        **({"evidence": dict(value.get("evidence") or {})}
                           if isinstance(value, dict) and
                           isinstance(value.get("evidence"), dict) else {})}
                        for name, value in steps.items()},
                }
            else:
                finalization_summary = {
                    "status": "malformed",
                    "issue": "step ledger is missing or not an object",
                }
        in_flight_finalization_steps = sorted(
            name for name, row in ((finalization_summary or {}).get("steps") or {}).items()
            if isinstance(row, dict) and row.get("status") in {"started", "unknown"})
        return {
            "schema_version": 1,
            "status": "unresolved",
            "reason_code": "candidate_interrupted_before_controller_commit",
            "controller_action": controller_action,
            "round": round_index,
            "pending_action_identity": pending_identity,
            "phase": str(pending_action.get("phase") or "unknown"),
            "idea": {key: idea.get(key) for key in ("label", "granularity", "mechanism")
                     if idea.get(key) is not None},
            "attempt_evidence": {
                "attempt_id": action_record.get("attempt_id"),
                "receipt_ref": evidence_ref,
                "receipt_status": evidence_status,
                "process_status": process_status,
                "outcome_known": False,
            },
            "candidate_measurement": measurement_state,
            **({"round_finalization": finalization_summary,
               "in_flight_finalization_steps": in_flight_finalization_steps}
               if finalization_summary is not None else {}),
            "changed_source_paths": changed_files,
            "automatic_replay": False,
            "automatic_measurement_adoption": False,
            "automatic_source_rollback": rollback_succeeded,
            "source_rollback": rollback,
            "unknowns": ([
                "whether the interrupted round completed every controller-side postcondition",
            ] + ([] if rollback_succeeded else [
                "whether a source change can be reverted without overwriting later edits",
            ])),
            "required_next": (
                "source rollback was verified, but the candidate round remains unresolved: "
                "inspect its records and make a new controller decision; do not adopt a score"
                if rollback_succeeded else
                "inspect the linked receipt, measurement verifier result, and source diff; "
                "preserve this run as unresolved until a separately verified recovery or "
                "human-reviewed boundary decision"),
        }

    def _rollback_interrupted_code_change(self, *, round_index: int,
                                          idea_label: str) -> dict[str, Any]:
        """Restore an interrupted pre-measurement patch only from its durable transaction.

        No manifest means this is an older/non-code transaction and is not enough evidence to
        write source. A valid manifest binds the exact repaired patch to a content-addressed
        pre-change snapshot. The snapshot API accepts only original or exact-patched bytes and
        refuses symlinks, corruption, and later edits.
        """
        if not idea_label or round_index < 1:
            return {"status": "refused", "because": "candidate identity is incomplete",
                    "paths": []}
        output_root = self.output.resolve()
        research_root = self.output / "research" / self.run_id
        raw_manifest_dir = research_root / "code_changes"
        raw_manifest = raw_manifest_dir / f"round-{round_index}.json"
        paths: list[str] = []
        try:
            if (research_root.is_symlink() or raw_manifest_dir.is_symlink() or
                    raw_manifest.is_symlink()):
                raise ValueError("code-change transaction path contains a symlink")
            resolved_root = research_root.resolve()
            resolved_manifest = raw_manifest.resolve()
            if (not resolved_root.is_relative_to(output_root) or
                    not resolved_manifest.is_relative_to(resolved_root)):
                raise ValueError("code-change transaction path escapes the run directory")
            if not resolved_manifest.is_file():
                return {"status": "no_transaction_record",
                        "because": "no durable code-change transaction exists; source was "
                                  "left untouched", "paths": []}
            manifest = read_json(resolved_manifest)
            if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
                    manifest.get("round") != round_index or
                    manifest.get("idea_label") != idea_label or
                    manifest.get("snapshot") != f"before-round-{round_index}"):
                raise ValueError("code-change transaction identity does not match")
            paths = sorted({str(row.get("file")) for row in manifest.get("patches", [])
                            if isinstance(row, dict) and row.get("file")})
            if manifest.get("status") == "rolled_back":
                return {"status": "already_rolled_back",
                        "because": "the exact candidate source snapshot was already restored",
                        "paths": paths,
                        "transaction_ref": str(raw_manifest.relative_to(research_root))}
            if manifest.get("status") not in {"applying", "applied"}:
                raise ValueError("code-change transaction is not in a recoverable phase")
            patches = patches_from_change({"patches": manifest.get("patches")})
            if not patches or len(paths) != len(patches):
                raise ValueError("code-change transaction has invalid patch records")
            snapshots = Snapshots(research_root / "snapshots")
            moved = snapshots.restore_patched(str(manifest["snapshot"]), patches,
                                              repo=self.repo)
            manifest.update(status="rolled_back", rolled_back_at=now(),
                            rollback_paths=moved)
            atomic_json(resolved_manifest, manifest)
            return {"status": "rolled_back" if moved else "already_rolled_back",
                    "because": ("candidate source was restored from its pre-change snapshot"
                                if moved else "candidate source already matched its snapshot"),
                    "paths": paths, "restored_paths": moved,
                    "transaction_ref": str(raw_manifest.relative_to(research_root))}
        except (OSError, TypeError, ValueError, KeyError, RuntimeError) as exc:
            return {"status": "refused",
                    "because": f"{type(exc).__name__}: {redact(str(exc))[:300]}",
                    "paths": paths}

    def _step_reconcile_interrupted_action(self) -> dict[str, Any]:
        """Close an interrupted attempt only after checking its owned process and receipt."""
        recovery = dict(self.recovery_after_interruption or {})
        action = recovery.get("action") if isinstance(recovery.get("action"), dict) else {}
        if not recovery:
            return {"outcome": "already reconciled",
                    "because": "there is no interrupted action in the current run state"}
        def is_research_action_name(value: Any) -> bool:
            name = str(value or "")
            return name == "baseline" or (name.startswith("round_") and
                                           name[6:].isdigit())

        identity = recovery.get("process_identity") or action.get("process_identity")
        process_state = (inspect_process_identity(identity) if isinstance(identity, dict)
                         else dict(recovery.get("process_status") or
                                   {"status": "not_recorded"}))
        process_status = str(process_state.get("status") or "unverifiable")
        if process_status == "matching_running":
            process_state = terminate_recorded_process(identity)
            process_status = str(process_state.get("status") or "unverifiable")
            if process_status != "terminated":
                return {"outcome": "blocked",
                        "because": "the recorded attempt process could not be confirmed stopped; "
                                   f"{process_state.get('why') or process_status}"}
        elif process_status in {"identity_mismatch", "unverifiable", "termination_failed",
                                "still_running"}:
            return {"outcome": "blocked",
                    "because": "process identity is not safe to reconcile automatically; "
                               f"{process_state.get('why') or process_status}"}
        elif process_status == "not_recorded":
            # A model decision can be interrupted without starting a repository command. For
            # any action that could have external side effects, missing process identity is a
            # hard stop rather than a guess that the process was never launched. A durable
            # parent research action is different: it owns the controller checkpoint and can
            # be marked interrupted without claiming that its hidden work completed.
            if (action.get("step") not in {"choose", "discard_interrupted_candidate"} and
                    not is_research_action_name(action.get("step"))):
                return {"outcome": "blocked",
                        "because": "the interrupted action may have started a command, but no "
                                   "durable process identity exists; automatic retry is unsafe"}

        evidence_ref = str(action.get("receipt_ref") or action.get("process_ref") or "")
        evidence_status = "not_recorded"
        if evidence_ref:
            raw_candidate = self.output / evidence_ref
            candidate = raw_candidate.resolve()
            if (raw_candidate.is_symlink() or
                    not candidate.is_relative_to(self.output.resolve())):
                return {"outcome": "blocked",
                        "because": "the recorded evidence reference escapes the run directory"}
            if candidate.is_file():
                try:
                    receipt = read_json(candidate)
                except (OSError, ValueError, TypeError) as exc:
                    return {"outcome": "blocked",
                            "because": f"the interrupted process record is unreadable: "
                                       f"{type(exc).__name__}: {exc}"}
                if (receipt.get("run_id") != self.run_id or
                        receipt.get("attempt_id") != action.get("attempt_id")):
                    return {"outcome": "blocked",
                            "because": "process record identity does not match the interrupted "
                                       "attempt"}
                evidence_status = str(receipt.get("status") or "unknown")
                if evidence_status == "running":
                    receipt.update(status="interrupted",
                                   termination_reason="controller_restart_outcome_unknown",
                                   finished_at=now(),
                                   process_reconciliation=process_state)
                    atomic_json(candidate, receipt)
                    evidence_status = "interrupted"
        elif action.get("attempt_id"):
            return {"outcome": "blocked",
                    "because": "the interrupted attempt has no process/receipt reference"}

        # A benchmark stage is nested inside a durable research action. Reconciling only the
        # stage receipt leaves controller_session.json marked `running`, which permanently
        # prevents the main controller from making an evidence-based next decision. Link the
        # exact parent action to this reconciled attempt; never infer that the research action
        # itself completed.
        parent = recovery.get("parent_action") or action.get("parent_action")
        if not isinstance(parent, dict):
            parent = action if is_research_action_name(action.get("step")) else {}
        controller_action = str(parent.get("step") or "")
        is_research_action = is_research_action_name(controller_action)
        if is_research_action:
            research_root = (self.output / "research" / self.run_id).resolve()
            if not research_root.is_relative_to(self.output.resolve()):
                return {"outcome": "blocked",
                        "because": "research session path escapes the run output"}
            session_path = research_root / "controller_session.json"
            if session_path.is_symlink() or not session_path.is_file():
                return {"outcome": "blocked",
                        "because": "the interrupted research session is absent or unsafe; "
                                   "the stage receipt was reconciled but controller state "
                                   "cannot be advanced"}
            try:
                session = read_json(session_path)
            except (OSError, ValueError, TypeError) as exc:
                return {"outcome": "blocked",
                        "because": "the interrupted research session is unreadable: "
                                   f"{type(exc).__name__}: {exc}"}
            if (not isinstance(session, dict) or session.get("schema_version") != 1 or
                    session.get("run_id") != self.run_id or
                    Path(str(session.get("repository") or "")).expanduser().resolve() !=
                    self.repo):
                return {"outcome": "blocked",
                        "because": "the interrupted research session identity is invalid"}
            old_reconciliation = session.get("reconciliation")
            same_interrupted_attempt = (
                session.get("status") == "interrupted" and
                isinstance(old_reconciliation, dict) and
                old_reconciliation.get("attempt_id") == action.get("attempt_id") and
                session.get("action") == controller_action)
            same_committed_recovery = (
                session.get("status") in {"paused", "finalizing"} and
                isinstance(old_reconciliation, dict) and
                old_reconciliation.get("status") == "committed_round_verified" and
                old_reconciliation.get("attempt_id") == action.get("attempt_id") and
                session.get("action") in {"", controller_action})
            matching_active_action = (session.get("status") == "running" and
                                      session.get("action") == controller_action)
            if (not (same_interrupted_attempt or same_committed_recovery or
                     matching_active_action) and
                    (session.get("status") != "running" or
                     session.get("action") != controller_action)):
                return {"outcome": "blocked",
                        "because": "the active research action does not match the reconciled "
                                   "stage parent; refusing to rewrite controller state"}
            interruption = {
                "controller_action": controller_action,
                "stage": str(action.get("step") or ""),
                "attempt_id": action.get("attempt_id"),
                "evidence_ref": evidence_ref or None,
                "evidence_status": evidence_status,
                "outcome_known": False,
            }
            pending_action = session.get("pending_action")
            if isinstance(pending_action, dict):
                interruption["pending_action"] = dict(pending_action)
            boundary = self._candidate_interruption_boundary(
                controller_action=controller_action, pending_action=pending_action,
                evidence_ref=evidence_ref or None, evidence_status=evidence_status,
                process_status=process_status)
            if boundary is not None:
                interruption["boundary"] = boundary
            reconciliation = {
                "status": "reconciled", "at": now(),
                "process_status": process_status,
                "process_reconciliation": process_state,
                "attempt_id": action.get("attempt_id"),
                "evidence_ref": evidence_ref or None,
                "evidence_status": evidence_status,
            }

            if (isinstance(boundary, dict) and
                    boundary.get("status") == "resolved_committed"):
                finalization = boundary.get("round_finalization")
                round_index = boundary.get("round")
                rounds = session.get("rounds")
                next_round = session.get("next_round")
                if (not isinstance(finalization, dict) or
                        isinstance(round_index, bool) or not isinstance(round_index, int) or
                        isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0 or
                        isinstance(next_round, bool) or not isinstance(next_round, int) or
                        next_round != round_index + 1 or next_round > rounds + 1 or
                        not isinstance(session.get("history"), list)):
                    return {"outcome": "blocked",
                            "because": "verified candidate finalization does not match the "
                                       "durable controller round/history cursor"}
                if session.get("status") in {"paused", "finalizing"} and not (
                        isinstance(old_reconciliation, dict) and
                        old_reconciliation.get("status") == "committed_round_verified" and
                        old_reconciliation.get("attempt_id") == action.get("attempt_id") and
                        (old_reconciliation.get("round_finalization") or {}).get(
                            "transaction_id") == finalization.get("transaction_id")):
                    return {"outcome": "blocked",
                            "because": "a later controller state cannot be rewritten by an "
                                       "unrelated interrupted attempt"}

                stopped_because = str(session.get("stopped_because") or "")
                next_status = ("finalizing" if next_round > rounds or stopped_because
                               else "paused")
                interruption["outcome_known"] = True
                interruption["boundary"] = {
                    **boundary,
                    "attempt_evidence": {
                        **(boundary.get("attempt_evidence") or {}),
                        "outcome_known": True,
                    },
                }
                boundary = interruption["boundary"]
                reconciliation = {
                    **reconciliation,
                    "status": "committed_round_verified",
                    "outcome_known": True,
                    "round": round_index,
                    "next_round": next_round,
                    "next_status": next_status,
                    "round_finalization": {
                        "transaction_id": finalization.get("transaction_id"),
                        "history_row_sha256": finalization.get("history_row_sha256"),
                        "measurement_sha256": finalization.get("measurement_sha256"),
                    },
                }

                report_path = research_root / "research_report.json"
                if report_path.is_symlink():
                    return {"outcome": "blocked",
                            "because": "the research report is a symlink; verified session "
                                       "was not advanced"}
                prior_report = None
                if report_path.exists():
                    try:
                        prior_report = read_json(report_path)
                    except (OSError, ValueError, TypeError) as exc:
                        return {"outcome": "blocked",
                                "because": "the research report cannot be rebuilt after "
                                           f"reconciliation: {type(exc).__name__}: {exc}"}
                    if (not isinstance(prior_report, dict) or
                            prior_report.get("run_id") != self.run_id or
                            Path(str(prior_report.get("repo") or "")).expanduser().resolve() !=
                            self.repo):
                        return {"outcome": "blocked",
                                "because": "the research report identity is invalid after "
                                           "reconciliation"}

                rubric_path = research_root / "rubric.json"
                if rubric_path.is_symlink() or not rubric_path.is_file():
                    return {"outcome": "blocked",
                            "because": "committed finalization has no safe persisted rubric "
                                       "for a report-only rebuild"}
                try:
                    rubric_document = read_json(rubric_path)
                except (OSError, ValueError, TypeError) as exc:
                    return {"outcome": "blocked",
                            "because": "persisted rubric cannot be read for report rebuild: "
                                       f"{type(exc).__name__}: {exc}"}
                if (not isinstance(rubric_document, dict) or
                        not isinstance(rubric_document.get("checks"), list) or
                        not rubric_document["checks"]):
                    return {"outcome": "blocked",
                            "because": "persisted rubric is not sufficient for a report-only "
                                       "rebuild; no model call or benchmark action is allowed"}
                try:
                    research, _ = self._research_controller()
                    research.build_rubric()
                    report = research._controller_report(
                        history=session["history"], stopped_because=stopped_because,
                        run_status=next_status, next_round=next_round, rounds=rounds)
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    return {"outcome": "blocked",
                            "because": "the committed research report projection could not be "
                                       f"rebuilt: {type(exc).__name__}: {redact(str(exc))[:300]}"}
                report.update(
                    interruption=interruption,
                    interruption_boundary=boundary,
                    reconciliation=reconciliation,
                    recovered_candidate_finalization={
                        "status": "verified_committed",
                        "round": round_index,
                        "transaction_id": finalization.get("transaction_id"),
                        "history_row_sha256": finalization.get("history_row_sha256"),
                        "receipt_proof": boundary.get("receipt_proof"),
                        "automatic_replay": False,
                    },
                    updated_at=now())

                # The report is a projection. Write it first so a crash leaves the durable
                # session in an interrupted state and this exact proof can rebuild it again.
                atomic_json(report_path, report)
                session.update(status=next_status, action="", pending_action=None,
                               interruption=interruption, interruption_boundary=boundary,
                               reconciliation=reconciliation,
                               last_interruption=interruption, updated_at=now())
                atomic_json(session_path, session)
                document_warning = ""
                try:
                    run_record.generate(
                        research_root,
                        title=f"{self.declaration.get('benchmark') or self.run_id} / derived")
                except (OSError, ValueError, TypeError) as exc:
                    document_warning = ("RUN.md projection could not be refreshed: "
                                        f"{type(exc).__name__}")
                result = {"step": "reconcile_interrupted_action",
                          "previous_step": action.get("step"),
                          "research_action": controller_action,
                          "attempt_id": action.get("attempt_id"),
                          "process_status": process_status,
                          "evidence_status": evidence_status,
                          "evidence_ref": evidence_ref or None,
                          "outcome_known": True,
                          "round": round_index,
                          "transaction_id": finalization.get("transaction_id"),
                          "status": next_status,
                          "next_round": next_round,
                          "required_next": "main controller may choose the next action; do not "
                                           "rerun the committed candidate",
                          **({"document_warning": document_warning}
                             if document_warning else {})}
                self.recovery_after_interruption = {}
                self.last_action = {**result,
                                    "outcome": "committed candidate round verified",
                                    "because": "all candidate measurement, history, receipt, "
                                               "and controller-side finalization evidence matched; "
                                               "only the report/session projection was rebuilt",
                                    "finished_at": now()}
                return {**result, "outcome": "committed candidate round verified",
                        "because": "the candidate transaction is committed and verified; the "
                                   "report was rebuilt without replaying benchmark work"}

            session.update(status="interrupted", action=controller_action,
                           interruption=interruption, reconciliation=reconciliation,
                           updated_at=now())
            atomic_json(session_path, session)
            report_path = research_root / "research_report.json"
            if report_path.is_file():
                if report_path.is_symlink():
                    return {"outcome": "blocked",
                            "because": "the research report is a symlink; interruption "
                                       "was persisted in the session but report projection "
                                       "was not updated"}
                try:
                    report = read_json(report_path)
                except (OSError, ValueError, TypeError) as exc:
                    return {"outcome": "blocked",
                            "because": "the research report cannot be updated after "
                                       f"reconciliation: {type(exc).__name__}: {exc}"}
                if (not isinstance(report, dict) or report.get("run_id") != self.run_id or
                        Path(str(report.get("repo") or "")).expanduser().resolve() !=
                        self.repo):
                    return {"outcome": "blocked",
                            "because": "the research report identity is invalid after "
                                       "reconciliation"}
                report.update(run_status="interrupted", interruption=interruption,
                              reconciliation=reconciliation, updated_at=now())
                if isinstance(interruption.get("boundary"), dict):
                    report["interruption_boundary"] = interruption["boundary"]
                atomic_json(report_path, report)

        result = {"step": "reconcile_interrupted_action",
                  "previous_step": action.get("step"),
                  "research_action": controller_action if is_research_action else None,
                  "attempt_id": action.get("attempt_id"),
                  "process_status": process_status,
                  "evidence_status": evidence_status,
                  "evidence_ref": evidence_ref or None,
                  "outcome_known": False}
        self.recovery_after_interruption = {}
        self.last_action = {**result, "outcome": "interrupted; outcome unknown",
                            "because": "local process state was reconciled; inspect the "
                                       "attempt receipt, log, and outputs before retrying",
                            "finished_at": now()}
        return {**result, "outcome": "interrupted; outcome unknown",
                "because": "the active process was reconciled and its evidence preserved, but "
                           "completion and side effects are not inferred; inspect "
                           f"{evidence_ref or 'the recorded action evidence'} before retrying"}

    def do(self, step: str, **arguments: Any) -> dict[str, Any]:
        """Observe throughput and suppress identical failed derivations, not repair attempts."""
        from .scheduling import policy, note
        if not policy(self.output):
            return self._do_operation(step, **arguments)
        started = time.monotonic()
        context_key = None
        path = self.output / "scheduling_failures.json"
        failures = read_json(path) if path.is_file() else {}
        if step == "derive_a_command":
            from .native_jobs import source_identity
            stamp = source_identity(self.output, self.repo)
            if stamp:
                context_key = object_digest({"step": step, "arguments": arguments,
                    "source": stamp, "execution": self.execution,
                    "environment": digest(self.output / "environment.json")
                        if (self.output / "environment.json").is_file() else ""})
        if context_key and failures.get(context_key, {}).get("count", 0) >= 2:
            return {"outcome": "not attempted", "because":
                "identical failed derivation already attempted twice; use Fix and retry_failed_action, change verified inputs/window, or resolve resource evidence",
                "previous_failure": failures[context_key],
                "evidence_refs": ["scheduling_failures.json"]}
        result = self._do_operation(step, **arguments)
        if context_key and result.get("outcome") == "no runnable command" and not result.get("resource_block"):
            previous = failures.get(context_key) or {}
            failures[context_key] = {"count": int(previous.get("count", 0))+1,
                "step": step, "arguments": arguments, "because": result.get("because"),
                "evidence_refs": result.get("evidence_refs") or [], "at": now()}
            atomic_json(path, failures)
        note(self.output, kind="controller_action", identity=str(self.current_action.get("decision_id") or uuid.uuid4().hex),
             seconds=time.monotonic()-started, operation=step, status=result.get("outcome"))
        return result

    def _do_operation(self, step: str, **arguments: Any) -> dict[str, Any]:
        """Take one step. Never raises: a step that fails is a step that failed.

        What it returns is what happened, in the shape the state reads: an outcome, and the
        reason it did not work when it did not. A failed step that raised would take the run
        down from inside the loop that was supposed to be deciding what to do about it.
        """
        if step not in OPERATIONS:
            return {"outcome": "no such step", "because": f"{step!r} is not one of the steps"}
        from .native_jobs import active_jobs
        if active_jobs(self.output) and step in {"build_the_environment", "derive_a_command", "declare", "bind_metric",
                "retry_failed_action", "run_the_loop", "confirm_best", "recover_unscored_baseline", "configure_screening", "run_screening_trial"}:
            return {"outcome": "blocked", "because": "unresolved native jobs own source/environment; wait or cancel first"}
        if self.state_persistence_error:
            return {"outcome": "blocked", "because":
                    "durable run state is unavailable; no further action is safe: "
                    f"{self.state_persistence_error}"}
        if self.keep_only and step not in {"run_the_loop", "recover_unscored_baseline",
                                           "confirm_best", "stop",
                                           "reconcile_interrupted_action"}:
            return {"outcome": "not attempted", "because":
                    f"--keep-only prohibits {step}; only the recorded stages may be run"}
        if self.recovery_after_interruption and step != "reconcile_interrupted_action":
            return {"outcome": "blocked",
                    "because": "the previous action must be reconciled before any new action; "
                               "its outcome or side effects are unknown"}
        if step in {"research_task", "update_research_plan"}:
            from .main_agent import READ_ONLY_ROLES, validate_plan, validate_task
            if not getattr(self.client, "supports_main_agent", False):
                return {"outcome": "rejected", "because": "main-agent runtime is unavailable"}
            try:
                if step == "research_task":
                    validate_task(arguments)
                    if (self.state().get("research_progress") and
                            arguments["role"] not in READ_ONLY_ROLES):
                        return {"outcome": "rejected", "because":
                                "existing research requires a read-only task role; code "
                                "changes must use the audited candidate path"}
                else:
                    if set(arguments) != {"plan"}:
                        raise ValueError("update_research_plan requires exactly plan")
                    validate_plan(arguments["plan"])
            except ValueError as exc:
                return {"outcome": "rejected", "because": str(exc)}
        if step == "declare":
            facts = self.state()
            if not self._declaration_refresh_is_justified(
                    research_progress=facts.get("research_progress"),
                    research_options=facts.get("research_options")):
                return {"outcome": "not attempted", "because":
                        "a usable declaration is already on file and no new survey, "
                        "actionable failure, or stage/declaration mismatch justifies "
                        "repeating it"}
        if step == "bind_metric":
            if not self._verified_score_output_available():
                return {"outcome": "not attempted",
                        "because": "metric binding requires a verified score command and its "
                                   "latest accepted native output or structured artifact"}
            retry = self._metric_binding_retry_status()
            if retry.get("retry_blocked"):
                return {"outcome": "not attempted",
                        "because": "metric binding already failed on the current verified "
                                  "score output; re-derive that score stage before retrying",
                        "score_target": retry.get("score_target")}
        if step == "read_the_checkout" and not self._checkout_read_is_justified():
            return {"outcome": "not attempted",
                    "because": "the current checkout was already surveyed and no new actionable "
                              "failure evidence arrived since that survey; use the current "
                              "surveyed entrypoints and choose the next missing operation"}
        if step == "discard_interrupted_candidate":
            reason = arguments.get("reason")
            if (not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 1000):
                return {"outcome": "not attempted", "because":
                        "discard requires a concise, evidence-backed reason (1–1000 chars)"}
            progress = self.state().get("research_progress") or {}
            if not self._interrupted_candidate_discard_available(progress):
                return {"outcome": "not attempted", "because":
                        "discard is available only for a reconciled candidate with a matched "
                        "unresolved pending-action boundary"}
        if step == "run_the_loop":
            recovery = self._unscored_baseline_recovery_status()
            if recovery.get("available"):
                return {"outcome": "not attempted",
                        "because": "a completed baseline train receipt can be recovered "
                                  "without retraining; choose recover_unscored_baseline before "
                                  "starting another research arm",
                    "recovery_attempt_id": recovery.get("attempt_id")}
        if step == "retry_failed_action" and not self._repair_retry_available():
            return {"outcome": "not attempted", "because":
                    "no unused, repair-attempted AgentFix handoff names a failed operation"}
        if step in {"submit_native_job", "inspect_native_job", "adjust_native_job",
                    "cancel_native_job"}:
            if not getattr(self.client, "supports_main_agent", False):
                return {"outcome": "rejected", "because":
                        "detached native jobs require the AutoSOTA main-agent runtime"}
        if step in {"run_the_loop", "generate_research_ideas", "propose_research_idea"}:
            current = self.state()
            progress = current.get("research_progress") or {}
            options = current.get("research_options") or {}
            if step in {"generate_research_ideas", "propose_research_idea"} and \
                    progress.get("status") != "paused":
                return {"outcome": "not attempted",
                        "because": "research ideas may only be generated or proposed while "
                                  "the persisted research session is paused"}
            if step == "generate_research_ideas" and options.get("items"):
                return {"outcome": "not attempted",
                        "because": "currently selectable ideas already exist; the main "
                                  "controller must decide among them or propose its own idea"}
            if step == "generate_research_ideas" and options.get("generation_attempted"):
                return {"outcome": "not attempted",
                        "because": "a fresh suggestion batch was already requested for this "
                                  "paused research round; formulate a different main-controller "
                                  "proposal or state why no safe action remains"}
            if step == "run_the_loop" and progress.get("status") == "paused":
                label = arguments.get("idea_label")
                selectable = {str(row.get("label")) for row in options.get("items") or []
                              if isinstance(row, dict)}
                if not isinstance(label, str) or label not in selectable:
                    return {"outcome": "not attempted",
                            "because": "a paused run requires the exact label of a currently "
                                      "audited research option; the inner engine will not "
                                      "choose another idea",
                            "available_idea_labels": sorted(selectable)}
            elif step == "run_the_loop" and arguments.get("idea_label"):
                return {"outcome": "not attempted",
                        "because": "idea_label is only valid when resuming a paused candidate "
                                  "round; baseline, recovery, and finalization do not consume it"}
            if step == "run_the_loop" and arguments.get("job_id") and \
                    (progress.get("status") or arguments.get("idea_label")):
                return {"outcome": "not attempted", "because":
                        "a detached trainer can only be adopted before the initial baseline"}
        handler = getattr(self, f"_step_{step}", None)
        if handler is None:
            return {"outcome": "no such step", "because": f"{step} has no implementation"}
        # The scheduler chooses an operation, but cannot extend its input contract by
        # inventing arguments. A stray `stage` on a repository survey used to raise and
        # consume another preparation step even though the survey itself needed no input.
        arguments = (arguments if step in {"research_task", "submit_research_task", "cancel_research_task", "update_research_plan", "review_report_demo", "capture_environment_demo",
                                         "configure_screening", "run_screening_trial", "inspect_screening",
                                         "submit_native_job", "inspect_native_job",
                                         "adjust_native_job", "cancel_native_job"}
                     else {"max_operations": arguments["max_operations"]}
                     if step == "build_the_environment" and "max_operations" in arguments
                     else {"stage": arguments["stage"], **(
                         {"timeout_seconds": arguments["timeout_seconds"]}
                         if "timeout_seconds" in arguments else {})}
                     if step == "derive_a_command"
                     and isinstance(arguments.get("stage"), str) and arguments["stage"]
                     else {"idea_label": arguments["idea_label"]}
                     if step == "run_the_loop" and
                     isinstance(arguments.get("idea_label"), str) and
                     arguments["idea_label"]
                     else {"job_id": arguments["job_id"]}
                     if step == "run_the_loop" and
                     isinstance(arguments.get("job_id"), str) and
                     arguments["job_id"]
                     else {"idea": arguments["idea"]}
                     if step == "propose_research_idea" and
                     isinstance(arguments.get("idea"), dict)
                     else {"reason": arguments["reason"]}
                     if step == "discard_interrupted_candidate" and
                     isinstance(arguments.get("reason"), str)
                     else {"because": arguments["because"]} if step == "stop"
                     and isinstance(arguments.get("because"), str) else {})
        try:
            # The opt-in coding-agent runtime uses one persistent session per AutoSOTA
            # responsibility.  Legacy chat clients do not expose a role scope and remain
            # behavior-compatible.  Scope the whole operation so nested calls made while
            # preparing/evaluating it retain the same role identity.
            from .agent_client import role_scope
            role = {
                "read_the_checkout": "resource",
                "declare": "objective",
                "build_the_environment": "init",
                "derive_a_command": "init",
                "bind_metric": "objective",
                "reconcile_interrupted_action": "monitor",
                "discard_interrupted_candidate": "monitor",
                "recover_unscored_baseline": "fix",
                "run_the_loop": "scheduler",
                "generate_research_ideas": "ideator",
                "propose_research_idea": "ideator",
                "confirm_best": "supervisor",
                "stop": "monitor",
            }.get(step, "scheduler")
            fix_handoff: dict[str, Any] | None = None
            if step == "build_the_environment":
                # A later native-stage failure can invalidate a previously passing
                # environment.  That is a repair/reverification handoff, not initial
                # setup: preserve the failure evidence and let AgentFix own the probe
                # plan.  Ordinary first-time provisioning remains AgentInit's job.
                failure = _last_stage_failure_after_build(
                    self.steps, self.output, self.run_id)
                if failure:
                    role = "fix"
                    fix_handoff = {
                        key: failure[key] for key in
                        ("stage", "outcome", "because", "failure_kind", "evidence_ref")
                        if key in failure
                    }
            self._publish_main_context(self.state())
            with role_scope(self.client, role):
                from .initialization_audit import preserve
                preserve(self.output, self.repo)
                result = handler(**arguments)
            if fix_handoff and isinstance(result, dict):
                result = {**result, "delegated_role": "fix",
                          "fix_handoff": fix_handoff}
            return result
        except Exception as exc:                                     # noqa: BLE001
            if self._attempts:
                self._record_attempts(step, arguments)
            if isinstance(exc, ResearchStatePersistenceError):
                self.state_persistence_error = redact(
                    f"research run-state event failed: {type(exc).__name__}: {exc}")[:400]
            agent_status = str(getattr(exc, "status", "") or "")[:80]
            failure_category = str(getattr(exc, "failure_category", "") or "")[:120]
            from .agent_client import run_model_budget_exhausted
            is_model_budget = run_model_budget_exhausted(exc)
            outcome = ("exhausted" if is_model_budget else
                       "rejected" if isinstance(exc, ResearchStateError) else "raised")
            result = {"outcome": outcome,
                      "because": redact(f"{type(exc).__name__}: {exc}")[:600]}
            if step == "build_the_environment":
                progress_path = self.output / "provision_progress.json"
                if progress_path.is_file() and not progress_path.is_symlink():
                    progress = read_json(progress_path)
                    progress.update(status="interrupted", updated_at=now(),
                                    failure_category=failure_category,
                                    interruption=result["because"])
                    atomic_json(progress_path, progress)
                    result["provision_progress_ref"] = "provision_progress.json"
                    result["evidence_ids"] = [row["evidence_id"] for row in
                        progress.get("attempts", []) if row.get("evidence_id")][-20:]
            if agent_status:
                result["agent_status"] = agent_status
            if failure_category:
                result["failure_category"] = failure_category
            run_budget = getattr(exc, "run_budget", None)
            if isinstance(run_budget, dict):
                result["run_budget"] = {
                    key: run_budget[key] for key in (
                        "run_id", "limit_usd", "spent_usd", "reserved_usd",
                        "remaining_usd", "unknown_entries", "entry_count")
                    if key in run_budget}
            return result

    def _handoff_scheduler_decision_failure(self, *, failure: dict[str, Any]) -> bool:
        """Persist a failed Scheduler turn, ask Monitor, and allow a bounded retry.

        No benchmark action is inferred to have run. This path is deliberately unavailable
        after cost-budget exhaustion or state-store failure; Monitor must not be used to
        bypass either hard boundary.
        """
        if (self.state_persistence_error or
                not callable(getattr(self.client, "as_role", None))):
            return False
        # Stable pre-launch evidence is separate from model/session event references:
        # no CLI process (and hence no such event) necessarily exists yet.
        fingerprint = failure.get("failure_fingerprint")
        if fingerprint:
            guard_path = self.output / "agent/runtime_recovery_gate.json"
            if guard_path.is_symlink():
                raise ValueError("runtime recovery gate is unsafe")
            gate = read_json(guard_path) if guard_path.is_file() else {}
            transient = failure.get("failure_category") in {
                "TimeoutError", "ConnectionError", "BrokenPipeError", "ConnectionResetError",
                "ConnectionAbortedError", "provider_or_tool_error", "wall_timeout", "admission_wait"}
            attempts = int(gate.get("attempts", 0)) + 1 if gate.get("fingerprint") == fingerprint else 1
            if gate.get("fingerprint") == fingerprint and (not transient or attempts > 3):
                failure["recovery"] = {"status": "blocked", "reason":
                    "same startup fault already assessed; repair the recorded cause before retry"}
                atomic_json(self.output / "runtime_blocker.json", failure)
                return False
            atomic_json(guard_path, {"fingerprint": fingerprint, "at": now(), "attempts": attempts,
                                    "evidence_id": failure.get("evidence_id")})
            if failure.get("failure_category") == "workspace_guard":
                try:
                    from .runtime_recovery import recover
                    result = recover(self.client, failure, timeout=min(
                        self.client.timeout, self.budget.remaining() if self.budget else 300))
                except Exception as exc:
                    result = {"status": "blocked", "reason": redact(str(exc))[:600]}
                failure["recovery"] = result
                atomic_json(self.output / "runtime_blocker.json", failure)
                return result.get("status") == "revalidated"
        action_id = "scheduler-decision-" + uuid.uuid4().hex
        failed_action = {**failure, "action_id": action_id,
                         "receipt_ref": "", "receipt_sha256": ""}
        self.current_action = {}
        self.last_action = {**failed_action, "finished_at": now()}
        self._persist_run_state(status="running")
        if self.state_persistence_error:
            return False
        self._refresh_document(status="running", current="Scheduler decision recovery")
        observation = self._monitor_failed_action(
            action_id=action_id, action=failed_action)
        if observation is None:
            return False
        failure_view = {
            key: observation.get(key) for key in (
                "status", "progress", "summary", "guidance", "evidence_refs")
        }
        if self.steps and self.steps[-1].get("step") == "choose":
            self.steps[-1] = {**self.steps[-1], "monitor": failure_view}
        self.last_action["monitor_status"] = observation.get("status")
        self.last_action["monitor_progress"] = observation.get("progress")
        self.last_action["monitor_summary"] = observation.get("summary")
        exhausted = observation.get("status") == "budget_exhausted"
        status = "budget_exhausted" if exhausted else "running"
        self._persist_run_state(status=status)
        if self.state_persistence_error:
            return False
        self._refresh_document(
            status=status,
            current=("model budget exhausted during Monitor recovery" if exhausted else
                     "Scheduler will reconsider after Monitor assessment"))
        return not exhausted

    def _monitor_failed_action(self, *, action_id: str,
                               action: dict[str, Any]) -> dict[str, Any] | None:
        """Ask the read-only AgentMonitor for high-level guidance after a failed action.

        This is an observer handoff, not a second Scheduler: its assessment is persisted as
        advisory evidence and the next research action remains the Scheduler's decision.
        Legacy chat clients keep their previous behavior.
        """
        if not callable(getattr(self.client, "as_role", None)):
            return None
        facts = self.state()
        self._publish_main_context(facts)
        payload = {
            "run_id": self.run_id,
            "action_id": action_id,
            "failed_action": action,
            "recent_actions": (facts.get("steps_taken") or [])[-6:],
            "environment": facts.get("environment"),
            "surveyed_stages": facts.get("surveyed_stages"),
            "to_measure": facts.get("to_measure"),
            "research_progress": facts.get("research_progress"),
            "available_next_actions": facts.get("available"),
            "previous_monitor_observation": self.monitor_observation or None,
        }
        payload_text = json.dumps(_controller_model_value(
            payload, local_roots=(self.repo, self.output)),
            ensure_ascii=False, default=str)
        status = "assessed"
        failure_category = ""
        metadata: dict[str, Any] = {}
        try:
            from .agent_client import role_scope
            timeout = (max(1, min(180, int(self.budget.remaining())))
                       if self.budget else 180)
            with role_scope(self.client, "monitor"):
                content, metadata = self.client.chat_with_metadata(
                    MONITOR_SYSTEM, payload_text, max_tokens=900, timeout=timeout,
                    thinking="disabled")
            from .execution_derive import _object
            answer = _object(content)
            progress = str(answer.get("progress") or "")
            summary = redact(str(answer.get("summary") or "").strip())[:700]
            guidance = redact(str(answer.get("guidance") or "").strip())[:900]
            if progress not in {"progress", "stalled", "blocked", "uncertain"} or not summary:
                raise ValueError("AgentMonitor response failed its structured handoff contract")
            raw_refs = answer.get("evidence_refs")
            refs = []
            if isinstance(raw_refs, list):
                for value in raw_refs[:20]:
                    if not isinstance(value, str):
                        continue
                    reference = redact(value.strip())[:300]
                    ref_path = Path(reference)
                    if (reference and not ref_path.is_absolute() and
                            ".." not in ref_path.parts and "\\" not in reference):
                        refs.append(reference)
            observation = {
                "action_id": action_id, "status": status, "progress": progress,
                "summary": summary, "guidance": guidance,
                "evidence_refs": refs,
                "evidence_validation": "model_citations_not_independently_verified",
                "recorded_at": now(),
            }
            cost = metadata.get("total_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                observation["reported_cost_usd"] = float(cost)
        except Exception as exc:                                          # noqa: BLE001
            agent_status = str(getattr(exc, "status", "") or "")[:80]
            failure_category = str(getattr(exc, "failure_category", "") or
                                   type(exc).__name__)[:120]
            from .agent_client import run_model_budget_exhausted
            status = ("budget_exhausted" if run_model_budget_exhausted(exc)
                      else "unavailable")
            observation = {
                "action_id": action_id, "status": status,
                "progress": "uncertain",
                "summary": ("AgentMonitor could not assess this action; no monitor conclusion "
                            "is inferred."),
                "guidance": "",
                "evidence_refs": [str(action.get("receipt_ref") or "")][:1],
                "failure_category": failure_category,
                "agent_status": agent_status,
                "recorded_at": now(),
            }
            run_budget = getattr(exc, "run_budget", None)
            if isinstance(run_budget, dict):
                observation["run_budget"] = {
                    key: run_budget[key] for key in (
                        "run_id", "limit_usd", "spent_usd", "reserved_usd",
                        "remaining_usd", "unknown_entries", "entry_count")
                    if key in run_budget}
        self.monitor_observation = observation
        try:
            stored = self.state_store.record(
                "monitor", "failed_action_assessed", status=status,
                details={"action_id": action_id,
                         "progress": observation.get("progress"),
                         "summary": observation.get("summary"),
                         "guidance": observation.get("guidance"),
                         "evidence_refs": observation.get("evidence_refs", []),
                         "failure_category": failure_category or None},
                phase_state={"latest_observation": observation})
            self.state_revision = int(stored.get("state_revision") or self.state_revision)
            self.decision_revision = int(stored.get("decision_revision") or
                                          self.decision_revision)
        except (ResearchStateError, OSError, TypeError, ValueError) as exc:
            self.state_persistence_error = redact(
                f"AgentMonitor handoff could not be persisted: "
                f"{type(exc).__name__}: {exc}")[:400]
        return observation

    def _fix_failed_action(self, *, action_id: str,
                           action: dict[str, Any]) -> dict[str, Any] | None:
        """Delegate one recoverable onboarding failure to AgentFix.

        Fix is deliberately an intervention, not proof of recovery: it may edit only the
        isolated checkout and run bounded diagnostics. The next Scheduler decision must
        inspect its report and choose whether to retry the original action. Expensive native
        train/evaluate jobs and confirmation actions are never replayed by this hook.
        """
        if (not getattr(self.client, "supports_agent_fix", False) or
                not callable(getattr(self.client, "as_role", None))):
            return None
        step = str(action.get("step") or "")
        outcome = str(action.get("outcome") or "")
        candidate_failure = step == "run_the_loop" and bool(action.get("candidate_failure"))
        if ((step not in _FIX_ASSISTANCE_STEPS and not candidate_failure) or
                (outcome not in _FIX_ASSISTANCE_OUTCOMES and not candidate_failure) or
                action.get("delegated_role") == "fix" or self.state_persistence_error):
            return None

        fix_attempt_id = uuid.uuid4().hex[:16]
        parent_receipt = str(action.get("receipt_ref") or "")
        facts = self.state()
        payload = {
            "run_id": self.run_id,
            "fix_attempt_id": fix_attempt_id,
            "failed_action_id": action_id,
            "failed_action": action,
            "failure_receipt_ref": parent_receipt,
            "recent_actions": (facts.get("steps_taken") or [])[-6:],
            "environment": facts.get("environment"),
            "surveyed_stages": facts.get("surveyed_stages"),
            "to_measure": facts.get("to_measure"),
            "monitor_observation": self.monitor_observation or None,
            "fix_observation": self.fix_observation or None,
            "boundary": (("Candidate measurement has already been committed and its source "
                          "may have been rolled back. Diagnose only; do not edit the checkout "
                          "or replay the candidate. Return an evidence-backed repair idea "
                          "for the Scheduler to audit as a NEW experiment round.")
                         if candidate_failure else
                         "Only inspect/repair this isolated checkout and run bounded CPU "
                         "diagnostics. Do not launch native training, evaluation, data "
                         "collection, or GPU jobs. The original action will not be retried "
                         "until the Scheduler reviews this report.")
        }
        self._publish_main_context(facts)
        payload_text = json.dumps(_controller_model_value(
            payload, local_roots=(self.repo, self.output)), ensure_ascii=False, default=str)
        started = time.monotonic()
        prior_action = dict(self.current_action)
        self.current_action = {"step": "agent_fix", "status": "running",
                               "started_at": now(), "parent_action_id": action_id,
                               "failure_receipt_ref": parent_receipt}
        self._persist_run_state(status="running")
        if self.state_persistence_error:
            return None
        self._refresh_document(status="running", current="AgentFix diagnosis/repair")

        status = "assessed"
        assessment = "uncertain"
        summary = "AgentFix did not produce a verifiable repair report."
        changes: list[str] = []
        refs: list[str] = []
        runtime_refs: list[str] = []
        failure_category = ""
        agent_status = ""
        run_budget: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        try:
            from .agent_client import role_scope
            local_limit = float(getattr(self.client, "timeout", 300))
            timeout = max(1, min(local_limit, self.budget.remaining())) if self.budget else local_limit
            with role_scope(self.client, "fix"):
                content, metadata = self.client.chat_with_metadata(
                    (FIX_SYSTEM + ("\nFor this committed candidate failure, diagnosis is "
                                   "read-only. Do not edit files. A repair must be a new "
                                   "audited candidate, not a hidden replay."
                                   if candidate_failure else "")),
                    payload_text, max_tokens=1400, timeout=timeout,
                    read_only=candidate_failure,
                    thinking="disabled")
            trace = metadata if isinstance(metadata, dict) else {}
            turn_id = str(trace.get("turn_id") or "")
            process_ref = str(trace.get("process_ref") or "")
            if (trace.get("role") == "fix" and
                    re.fullmatch(r"[0-9a-f]{32}", turn_id) and
                    process_ref == f"agent/processes/{turn_id}.json"):
                runtime_refs = [f"agent/events.jsonl#turn_id={turn_id}", process_ref]
            from .execution_derive import _object
            answer = _object(content)
            assessment = str(answer.get("assessment") or "")
            if assessment not in {"repair_attempted", "no_safe_repair", "blocked", "uncertain"}:
                raise ValueError("AgentFix response has an unknown assessment")
            if candidate_failure and assessment == "repair_attempted":
                assessment = "uncertain"
            raw_summary = answer.get("summary")
            if not isinstance(raw_summary, str) or not raw_summary.strip():
                raise ValueError("AgentFix response has no summary")
            summary = redact(raw_summary.strip())[:700]
            if candidate_failure and answer.get("assessment") == "repair_attempted":
                summary = ("Read-only candidate diagnosis cannot certify a repair. " +
                           summary)[:700]
            raw_changes = answer.get("changes")
            if isinstance(raw_changes, list):
                changes = [redact(item.strip())[:300] for item in raw_changes[:12]
                           if isinstance(item, str) and item.strip()]
            raw_refs = answer.get("evidence_refs")
            if isinstance(raw_refs, list):
                for value in raw_refs[:20]:
                    if not isinstance(value, str):
                        continue
                    reference = redact(value.strip())[:300]
                    ref_path = Path(reference)
                    if (reference and not ref_path.is_absolute() and
                            ".." not in ref_path.parts and "\\" not in reference):
                        refs.append(reference)
        except Exception as exc:  # noqa: BLE001 - the original Scheduler still owns recovery.
            agent_status = str(getattr(exc, "status", "") or "")[:80]
            failure_category = str(getattr(exc, "failure_category", "") or
                                   type(exc).__name__)[:120]
            from .agent_client import run_model_budget_exhausted
            budget_failure = run_model_budget_exhausted(exc)
            status = "budget_exhausted" if budget_failure else "unavailable"
            summary = ("AgentFix could not run because the model budget is exhausted."
                       if budget_failure else
                       "AgentFix could not complete; the failure remains unresolved.")
            run_budget = getattr(exc, "run_budget", None)
            if isinstance(run_budget, dict):
                run_budget = {
                    key: run_budget[key] for key in (
                        "run_id", "limit_usd", "spent_usd", "reserved_usd",
                        "remaining_usd", "unknown_entries", "entry_count")
                    if key in run_budget}
            else:
                run_budget = {}

        observation = {
            "fix_attempt_id": fix_attempt_id,
            "failed_action_id": action_id,
            "failure_receipt_ref": parent_receipt or None,
            "status": status,
            "assessment": assessment,
            "summary": summary,
            "changes": changes,
            "evidence_refs": refs,
            "runtime_evidence_refs": runtime_refs,
            "evidence_validation": "model_report_not_independently_verified",
            "failure_category": failure_category or None,
            "agent_status": agent_status or None,
            "run_budget": run_budget,
            "elapsed_wall_seconds": max(0.0, time.monotonic() - started),
            "recorded_at": now(),
        }
        self.fix_observation = observation
        self.current_action = prior_action
        try:
            stored = self.state_store.record(
                "fix", "failed_action_repair_attempted", status=status,
                details={key: observation.get(key) for key in (
                    "fix_attempt_id", "failed_action_id", "failure_receipt_ref",
                    "assessment", "summary", "changes", "evidence_refs",
                    "runtime_evidence_refs", "failure_category", "agent_status")},
                phase_state={"latest_observation": observation})
            self.state_revision = int(stored.get("state_revision") or self.state_revision)
            self.decision_revision = int(stored.get("decision_revision") or
                                          self.decision_revision)
        except (ResearchStateError, OSError, TypeError, ValueError) as exc:
            self.state_persistence_error = redact(
                f"AgentFix handoff could not be persisted: {type(exc).__name__}")[:400]
        if self.last_action:
            self.last_action = {**self.last_action, "fix_status": status,
                                "fix_attempt_id": fix_attempt_id}
        self._persist_run_state(status=("budget_exhausted" if status == "budget_exhausted"
                                       else "running"))
        self._refresh_document(status=("budget_exhausted" if status == "budget_exhausted"
                                       else "running"), current="AgentFix result recorded")
        return observation

    def _publish_main_context(self, facts: dict[str, Any]) -> None:
        publish = getattr(self.client, "set_research_context", None)
        if callable(publish):
            # state() already withholds held-out outcomes. Do not project raw research
            # records here: specialists must share the same information boundary.
            publish({key: value for key, value in facts.items()
                     if key not in {"current_action", "last_decision"}})

    def _save_main_memory(self, memory: dict[str, Any], *, event: str) -> None:
        # This is LOCAL durable evidence, not an outbound model projection. Applying the
        # artifact scrub here would destroy valid .json/.jsonl receipt references forever.
        # External prompts still go through _controller_model_value independently.
        def scrub(value: Any) -> Any:
            if isinstance(value, str):
                return sanitize_model_text(value, local_roots=(self.repo, self.output))
            if isinstance(value, dict):
                return {key: scrub(item) for key, item in value.items()}
            if isinstance(value, list):
                return [scrub(item) for item in value]
            return value
        safe = scrub(memory)
        try:
            stored = self.state_store.record(
                "main_agent", event, status="running",
                details={"authority": "model_memory_not_verified_facts"},
                phase_state={"memory": safe})
        except (ResearchStateError, OSError, TypeError, ValueError) as exc:
            raise ResearchStatePersistenceError("main-agent memory could not be persisted") from exc
        self.main_agent = safe
        self.state_revision = int(stored.get("state_revision") or self.state_revision)
        self.decision_revision = int(stored.get("decision_revision") or self.decision_revision)

    def _native_job_view(self) -> list[dict[str, Any]]:
        from .native_jobs import active_jobs, status
        root = self.output / "native_jobs"
        if not root.is_dir() or root.is_symlink():
            return []
        rows = []
        active = set(active_jobs(self.output))
        directories = sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        selected = [p for p in directories if p.name in active]
        selected += [p for p in directories if p.name not in active][:5]
        for directory in selected:
            try:
                job = status(self.output, directory.name)
                result = job.get("result") or {}
                stage_result = result.get("stage_result") or {}
                rows.append({"job_id": job["job_id"], "stage": job.get("stage"),
                             "status": job.get("status"),
                             "deadline_epoch": job.get("deadline_epoch"),
                             "worker": job.get("worker"),
                             "progress": job.get("progress"), "resources": job.get("resources"),
                             "wait_reason": job.get("wait_reason"),
                             "window_seconds": job.get("window_seconds"),
                             "failure": result.get("error") or stage_result.get("why"),
                             "result_ref": f"native_jobs/{job['job_id']}/result.json" if result else None,
                             "attempt_id": stage_result.get("attempt_id"),
                             "evidence_id": stage_result.get("evidence_id") or result.get("evidence_id"),
                             "job_ref": job.get("job_ref")})
            except (OSError, ValueError, TypeError, KeyError):
                continue
        return rows

    def _agent_task_view(self) -> list[dict[str, Any]]:
        from .agent_tasks import records
        all_rows = records(self.output)
        active = [r for r in all_rows if r["status"] in {"submitted", "running", "outcome_unknown"}]
        inactive = [r for r in all_rows if r not in active][-12:]
        return [{key: row.get(key) for key in ("id", "role", "task", "status", "stale", "collected_at", "error")}
                for row in [*active, *inactive]]

    def _collect_agent_tasks(self):
        from .agent_tasks import acknowledge, collect
        reported = False
        for row in collect(self.output, self.repo):
            if any(item.get("task_id") == row["id"] for item in self.main_agent.get("handoffs") or []):
                acknowledge(self.output, row["id"])
                continue
            handoff = {"task_id": row["id"], "role": row["role"], "task": row["task"],
                "expected_result": row["expected_result"], "report": row["report"],
                "stale": row["stale"], "input_state_revision": row["input_state_revision"],
                "runtime_evidence_refs": [row["worker_ref"] + "/agent/events.jsonl"], "at": now()}
            self._save_main_memory({**self.main_agent, "handoffs":
                [*(self.main_agent.get("handoffs") or []), handoff][-12:]}, event="async_task_reported")
            acknowledge(self.output, row["id"])
            reported = True
        return reported

    def _step_submit_research_task(self, *, role: str, task: str, expected_result: str):
        from .agent_tasks import submit
        try:
            row = submit(self.output, self.repo, self.client,
                {"role": role, "task": task, "expected_result": expected_result}, self.state(),
                min(240, self.budget.remaining()) if self.budget else 240)
            return {"outcome": "submitted", "because": "read-only investigation runs independently",
                    "task_id": row["id"], "evidence_refs": [f"agent_tasks/{row['id']}/task.json"]}
        except (OSError, ValueError) as exc:
            return {"outcome": "rejected", "because": str(exc)[:400]}

    def _step_cancel_research_task(self, *, task_id: str, reason: str):
        from .agent_tasks import cancel
        try:
            cancel(self.output, task_id, reason)
            return {"outcome": "cancel_requested", "because": "in-flight usage remains charged; result will not be adopted"}
        except (OSError, ValueError) as exc:
            return {"outcome": "rejected", "because": str(exc)[:400]}

    def _step_wait_for_jobs(self):
        from .scheduling import policy
        def signature():
            return object_digest([[{key: row.get(key) for key in
                ("job_id", "status", "attempt_id", "deadline_epoch")}
                for row in self._native_job_view()], self._agent_task_view()])
        before = signature()
        end = time.monotonic() + min(policy(self.output).get("wait_seconds", 30),
                                    self.budget.remaining() if self.budget else 30)
        while time.monotonic() < end:
            time.sleep(min(1, max(0, end-time.monotonic())))
            self._collect_agent_tasks()
            if signature() != before:
                return {"outcome": "event", "because": "a background task changed state"}
        return {"outcome": "waiting", "because": "dependencies remain active; no model poll was made"}

    def _screening_controller(self):
        progress = self.state().get("research_progress") or {}
        if progress.get("status") != "paused":
            raise ValueError("screening requires a scored baseline and paused development session")
        from .scheduling import policy
        if not policy(self.output):
            raise ValueError("this run did not enable scheduling")
        research, _ = self._research_controller()
        return research

    def _step_configure_screening(self, *, study: dict):
        from .screening import configure
        research = self._screening_controller()
        row = configure(research, study, research._base_settings(self.base_settings))
        return {"outcome": "configured", "because": "fidelity protocol frozen separately from formal selection",
                "study_id": row["id"], "evidence_refs": [f"research/{self.run_id}/screening/{row['id']}/study.json"]}

    def _step_run_screening_trial(self, *, study_id: str, idea_label: str, rung: int,
                                  window_seconds: float, reason: str):
        from .screening import trial
        research = self._screening_controller()
        result = trial(research, study_id=study_id, idea_label=idea_label, rung=rung,
                       window_seconds=window_seconds, reason=reason)
        return {"outcome": result["status"], "because": result.get("why") or "screening receipt verified; not a formal improvement",
                "screening_result": result, "evidence_refs": [
                    f"research/{self.run_id}/screening/{study_id}/trials/{result['trial_id']}/trial.json"]}

    def _step_inspect_screening(self, *, study_id: str):
        from .screening import inspect
        return {"outcome": "inspected", "because": "recommendations do not override Scheduler",
                "screening": inspect(self._screening_controller(), study_id)}

    def _repair_retry_available(self) -> bool:
        fix = self.fix_observation
        if fix.get("status") != "assessed" or fix.get("assessment") != "repair_attempted":
            return False
        fix_id = str(fix.get("fix_attempt_id") or "")
        failed_ref = str(fix.get("failure_receipt_ref") or "")
        if not fix_id or not failed_ref:
            return False
        if any(row.get("step") == "retry_failed_action" and
               row.get("fix_attempt_id") == fix_id for row in self.steps):
            return False
        return any(row.get("receipt_ref") == failed_ref and
                   row.get("step") in _FIX_ASSISTANCE_STEPS for row in self.steps)

    def _task_gpu_budget_view(self) -> dict[str, Any] | None:
        from .task_budget import TaskGPUBudget
        if (self.output / "task_gpu_budget.json").is_file():
            return TaskGPUBudget(self.output).snapshot()
        return None

    def _step_submit_native_job(self, *, stage: str, window_seconds: float,
                                reason: str, resources: dict | None = None) -> dict[str, Any]:
        from .native_jobs import submit
        from .repository_budget import RepositoryBudget
        if stage not in self.stages:
            return {"outcome": "not attempted", "because":
                    "only a verified native stage may be submitted"}
        # Legacy studies retain their original repository reservation. New task-scoped
        # studies reserve GPU time only when the worker acquires the physical device.
        from .derive_and_run import ROOT
        reservation = RepositoryBudget(
            ROOT / "autoresearch_runs" / "resource_budgets", repo=self.repo)
        try:
            if self.budget is None:
                raise ValueError("hard run budget is unavailable")
            if not (self.output / "task_gpu_budget.json").is_file():
                reservation.reserve(self.output, wall_seconds=self.budget.limit)
        except (OSError, ValueError, TypeError) as exc:
            return {"outcome": "blocked", "because":
                    f"repository GPU reservation refused job: {exc}"[:500]}
        settings = {key: value for key, value in self.base_settings.items()
                    if not key.startswith("_")}
        try:
            job = submit(self.output, repo=self.repo, stage=stage,
                         settings=settings, requested_seconds=window_seconds,
                         reason=reason, run_id=self.run_id, resources=resources)
        except (OSError, ValueError, TypeError) as exc:
            return {"outcome": "rejected", "because":
                    f"native job refused: {type(exc).__name__}: {exc}"[:600]}
        return {"outcome": "submitted", "because": "verified native stage is running "
                "under the existing hard budget", **job,
                "evidence_refs": [job["job_ref"]]}

    def _step_inspect_native_job(self, *, job_id: str) -> dict[str, Any]:
        from .native_jobs import status
        try:
            job = status(self.output, job_id)
        except (OSError, ValueError, TypeError) as exc:
            return {"outcome": "rejected", "because":
                    f"job status unavailable: {type(exc).__name__}: {exc}"[:500]}
        return {"outcome": "observed", "because": f"native job {job['status']}",
                **job, "evidence_refs": [job["job_ref"]]}

    def _step_adjust_native_job(self, *, job_id: str, window_seconds: float,
                                reason: str) -> dict[str, Any]:
        from .native_jobs import adjust
        try:
            job = adjust(self.output, job_id, seconds_from_now=window_seconds,
                         reason=reason)
        except (OSError, ValueError, TypeError) as exc:
            return {"outcome": "rejected", "because":
                    f"job adjustment refused: {type(exc).__name__}: {exc}"[:500]}
        return {"outcome": "adjusted", "because": "local job window changed; "
                "hard run/repository budget unchanged", **job,
                "evidence_refs": [job["job_ref"]]}

    def _step_cancel_native_job(self, *, job_id: str, reason: str) -> dict[str, Any]:
        from .native_jobs import cancel
        try:
            job = cancel(self.output, job_id, reason=reason)
        except (OSError, ValueError, TypeError) as exc:
            return {"outcome": "rejected", "because":
                    f"job cancellation refused: {type(exc).__name__}: {exc}"[:500]}
        return {"outcome": "cancellation_requested", "because":
                "worker will stop its process group and preserve the attempt receipt",
                **job, "evidence_refs": [job["job_ref"]]}

    def _step_retry_failed_action(self) -> dict[str, Any]:
        """Revalidate the same operation, once, after a concrete Fix handoff."""
        if not self._repair_retry_available():
            return {"outcome": "not attempted", "because":
                    "no unused repair handoff matches a failed action"}
        fix = dict(self.fix_observation)
        original = next(row for row in reversed(self.steps)
                        if row.get("receipt_ref") == fix["failure_receipt_ref"])
        original_step = str(original["step"])
        arguments = original.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        result = self.do(original_step, **arguments)
        return {"outcome": "reverified" if result.get("outcome") == "done" else
                "reverification_failed",
                "because": (f"original {original_step} returned "
                            f"{result.get('outcome')}: {result.get('because') or ''}")[:800],
                "fix_attempt_id": fix.get("fix_attempt_id"),
                "replayed_step": original_step,
                "original_outcome": result.get("outcome"),
                "evidence_refs": result.get("evidence_refs") or []}

    def _step_review_report_demo(self, *, request_id: str, decision: str,
                               reason: str) -> dict[str, Any]:
        from . import recorder
        if not getattr(self.client, "supports_recorder", False):
            return {"outcome": "rejected", "because": "Recorder integration is unavailable"}
        if decision not in {"capture", "decline"} or not isinstance(reason, str) or not reason.strip():
            return {"outcome": "rejected", "because": "choose capture or decline and explain why"}
        request = next((r for r in recorder.pending_requests(self.output) if r["id"] == request_id), None)
        if request is None:
            return {"outcome": "rejected", "because": "request is not pending"}
        if decision == "decline":
            recorder.resolve_request(self.output, request_id, status="declined", reason=reason)
            return {"outcome": "declined", "because": reason}
        if request.get("kind") == "environment_smoke":
            return {"outcome": "needs native plan", "because":
                    "Use capture_environment_demo with source-backed recording code; "
                    "no scored measurement is required and no score will be inferred."}
        from .native_jobs import active_jobs
        progress = self.state().get("research_progress") or {}
        if (active_jobs(self.output) or progress.get("status") != "paused" or
                (progress.get("confirmation") or {}).get("status") in {"taken", "attempted_and_failed"}):
            return {"outcome": "blocked", "because": "capture requires an idle paused development session"}
        if self.budget and self.budget.remaining() <= 0:
            return {"outcome": "blocked", "because": "run wall budget is exhausted"}
        relative = str(request.get("measurement_ref") or "")
        if not relative.startswith(f"research/{self.run_id}/measurements/"):
            return {"outcome": "rejected", "because": "measurement belongs to another run"}
        measurement = recorder.read(self.output, relative)
        path = self.output / relative
        if (not measurement or measurement.get("confirmation") is True or
                measurement.get("ok") is not True or
                digest(path) != request.get("measurement_sha256")):
            return {"outcome": "rejected", "because": "development measurement changed or is unscored"}
        research, _ = self._research_controller()
        # Pending -> capturing is persisted before side effects; interrupted captures are
        # never automatically replayed. Existing executor applies protocol/GPU budgets.
        recorder.resolve_request(self.output, request_id, status="capturing", reason=reason)
        try:
            result = research.capture_demo(trigger="recorder_request", measurement=measurement,
                                           settings=measurement.get("settings") or {})
        except Exception as exc:
            result = {"status": "unknown", "why": type(exc).__name__}
        with run_record._report_lock(self.output / "report" / "demo_requests.json"):
            store = recorder.read(self.output, "report/demo_requests.json")
            for row in store.get("requests", []):
                if row.get("id") == request_id:
                    row.update(status=result.get("status") or "unknown", result=result,
                               resolved_at=now())
            atomic_json(self.output / "report" / "demo_requests.json", store)
        return {"outcome": result.get("status") or "unknown", "because": result.get("why") or reason,
                "attempt_ids": [result["attempt_id"]] if result.get("attempt_id") else []}

    def _step_capture_environment_demo(self, *, request_id: str, code: str, purpose: str,
                                       source_refs: list[str], timeout_seconds: float,
                                       resource: str) -> dict[str, Any]:
        from . import recorder, environment_demo
        from .native_jobs import active_jobs
        request = next((r for r in recorder.pending_requests(self.output)
                        if r.get("id") == request_id and r.get("kind") == "environment_smoke"), None)
        if request is None or active_jobs(self.output):
            return {"outcome": "blocked", "because": "missing pending preview request or native job owns checkout"}
        confirmation = (self.state().get("research_progress") or {}).get("confirmation") or {}
        if confirmation.get("status") in {"taken", "attempted_and_failed"}:
            return {"outcome": "blocked", "because": "confirmation boundary forbids further development previews"}
        if resource not in {"cpu", "gpu"} or not 0 < float(timeout_seconds) <= 1800:
            return {"outcome": "rejected", "because": "invalid preview resource/window"}
        window = min(float(timeout_seconds), self.budget.remaining()) if self.budget else float(timeout_seconds)
        if window <= 0 or resource == "gpu" and not self.decision.on_gpu:
            return {"outcome": "blocked", "because": "preview resource or total wall budget unavailable"}
        recorder.resolve_request(self.output, request_id, status="capturing", reason=purpose)
        try:
            result = environment_demo.capture(self.output, self.repo, code=code, purpose=purpose,
                source_refs=source_refs, timeout=window,
                compute=self.decision if resource == "gpu" else None)
        except Exception as exc:
            from .runtime_recovery import seal_failure
            fault = seal_failure(self.output, self.repo, exc, role="scheduler",
                                 decision_attempt_id=None)
            result = {"status": "unknown", "why": redact(str(exc))[:600],
                      "evidence_id": fault["evidence_id"], "evidence_ref": fault["evidence_ref"]}
        with run_record._report_lock(self.output / "report/demo_requests.json"):
            store = recorder.read(self.output, "report/demo_requests.json")
            for row in store.get("requests", []):
                if row.get("id") == request_id:
                    row.update(status=result["status"], result=result, resolved_at=now())
            atomic_json(self.output / "report/demo_requests.json", store)
        return {"outcome": result["status"], "because": result.get("why") or
                "环境预览已记录；它不是正式测量，任务和策略身份尚未审计。",
                "evidence_id": result.get("evidence_id"),
                "evidence_refs": [result["evidence_ref"]] if result.get("evidence_ref") else []}

    def _step_update_research_plan(self, *, plan: dict[str, Any]) -> dict[str, Any]:
        from .main_agent import validate_plan
        self._save_main_memory({**self.main_agent, "plan": validate_plan(plan)},
                               event="plan_updated")
        return {"outcome": "recorded", "because": "global working plan persisted; "
                "frozen benchmark objective and verified facts are unchanged",
                "evidence_refs": ["run_state.json#phases.main_agent.memory.plan"]}

    def _step_research_task(self, *, role: str, task: str,
                            expected_result: str) -> dict[str, Any]:
        from .agent_client import role_scope
        from .main_agent import READ_ONLY_ROLES, TASK_SYSTEM, validate_report
        from .execution_derive import _object

        from .native_jobs import active_jobs
        if role not in READ_ONLY_ROLES and active_jobs(self.output):
            return {"outcome": "blocked", "because":
                    "an unresolved native job owns this checkout; editable research must wait"}

        task_id = uuid.uuid4().hex
        if role not in READ_ONLY_ROLES:
            self._save_main_memory({**self.main_agent, "checkout_needs_resurvey": True,
                                   "metric_revalidation_required": True,
                                   "last_editable_task_id": task_id},
                                   event="editable_investigation_started")
            # A tool-using turn can mutate source even if it times out or produces malformed
            # JSON. Invalidate BEFORE giving it control, not after its self-reported success.
            # Preserve the old command evidence, but require native verification again.
            kept_path = self.output / "derived_stages.json"
            archive = self.output / "superseded_stages" / f"{task_id}.json"
            kept = read_json(kept_path) if kept_path.is_file() else {}
            if kept or self.stages:
                try:
                    atomic_json(archive, {"reason": "editable main-agent investigation",
                                         "kept": kept, "stages": self.stages,
                                         "parameters": self.parameters,
                                         "verified": self.verified})
                    atomic_json(kept_path, {})
                except (OSError, ValueError, TypeError) as exc:
                    raise ResearchStatePersistenceError(
                        "could not invalidate commands before editable investigation") from exc
                self.stages.clear()
                self.parameters.clear()
                self.verified.clear()
        facts = self.state()
        self._publish_main_context(facts)
        payload = _controller_model_value({
            "assignment": {"role": role, "task": task, "expected_result": expected_result},
            "state": facts,
        }, local_roots=(self.repo, self.output))
        timeout = max(1, min(240, int(self.budget.remaining()))) if self.budget else 240
        with role_scope(self.client, role):
            content, metadata = self.client.chat_with_metadata(
                TASK_SYSTEM, json.dumps(payload, ensure_ascii=False, default=str),
                max_tokens=4000, timeout=timeout, include_research_context=False)
        report = validate_report(_object(content))
        runtime_refs: list[str] = []
        metadata = metadata if isinstance(metadata, dict) else {}
        turn_id = str(metadata.get("turn_id") or "")
        if (metadata.get("role") == role and re.fullmatch(r"[0-9a-f]{32}", turn_id) and
                metadata.get("process_ref") == f"agent/processes/{turn_id}.json"):
            runtime_refs = [f"agent/events.jsonl#turn_id={turn_id}", metadata["process_ref"]]
        handoff = {"task_id": task_id, "role": role, "task": task,
                   "expected_result": expected_result, "report": report,
                   "runtime_evidence_refs": runtime_refs,
                   "input_state_revision": facts.get("state_revision"), "at": now()}
        self._save_main_memory({**self.main_agent, "handoffs":
                               [*(self.main_agent.get("handoffs") or []), handoff][-12:]},
                               event="research_task_reported")
        # The hash-chained state log keeps older reports even when the prompt window rolls.
        return {"outcome": "reported", "because": report["summary"],
                "delegated_role": role, "verification_level": None,
                "evidence_refs": ["run_state.json#phases.main_agent.memory.handoffs",
                                  *runtime_refs]}

    def _step_read_the_checkout(self) -> dict[str, Any]:
        """Survey the repository and record which stages exist and how they are entered."""
        rejected_path = self.output / "path_selection_attempts.json"
        prior = read_json(rejected_path) if rejected_path.is_file() else {}
        result = execution_derive.run(
            self.repo, client=self.client, output=self.output,
            context={"task": self._declared_task(),
                     "assets": self.declaration.get("assets") or {},
                     "task_contract": self.declaration.get("task_contract") or {},
                     "rejected_handoffs": (prior.get("rows") or [])[-3:]})
        self.execution = result
        if self.main_agent.get("checkout_needs_resurvey"):
            self._save_main_memory({**self.main_agent, "checkout_needs_resurvey": False},
                                   event="checkout_resurveyed")
        stages = sorted(name for name, row in (result.get("stages") or {}).items()
                        if row.get("available"))
        return {"outcome": "done" if stages else "nothing available",
                "because": f"stages with an entry point: {', '.join(stages) or 'none'}",
                "stages": stages,
                "evidence_refs": self._existing_evidence_refs("execution.json")}

    def _step_declare(self) -> dict[str, Any]:
        """Read the repository and write down what it can do."""
        out = self.scouting / f"{self._benchmark_name()}_prepared"
        if self.scouting == self.output / "scouting":
            # Formal AutoSOTA Resource evidence belongs to this run. Do not let a
            # pre-existing symlink silently redirect declarations outside its durable
            # evidence bundle.
            if self.scouting.is_symlink() or out.is_symlink():
                return {"outcome": "blocked",
                        "because": "run-local scouting path must not be a symlink"}
            try:
                self.scouting.mkdir(parents=True, exist_ok=True)
                if not self.scouting.resolve(strict=True).is_relative_to(
                        self.output.resolve(strict=True)):
                    return {"outcome": "blocked",
                            "because": "run-local scouting path escaped the run output"}
            except (OSError, RuntimeError, ValueError) as exc:
                return {"outcome": "blocked",
                        "because": f"run-local scouting path is unavailable: "
                                   f"{type(exc).__name__}"}
        scout.run(self.repo, client=self.client, output=out)
        found, refused = self._declaration_candidates()
        if not found:
            return {"outcome": "no usable declaration",
                    "because": "; ".join(f"{n}: {w}" for n, w in refused)[:600] or
                               "the scout produced no declaration"}
        # Every usable candidate is reported, not the newest one. Which reading of a
        # repository is right is a judgement about evidence, and `state()` shows them all.
        _, name, document = max(found, key=lambda row: row[0])
        self.declaration = document
        return {"outcome": "done",
                "because": f"chose {name} of {len(found)} usable; "
                           f"{len(refused)} refused",
                "declaration": name}

    def _step_build_the_environment(self, *, max_operations: int = 1) -> dict[str, Any]:
        """Build or re-verify an environment, keeping only the commands that worked.

        A valid, unrebutted passing record is reused. A later failed native-stage derivation
        or a probe that violates the current capability contract reopens verification. The
        provisioner receives the actual already-verified interpreter, so it can replay only
        missing successful setup commands and test a corrected probe without recreating the
        environment prefix.
        """
        if isinstance(max_operations, bool) or not isinstance(max_operations, int) or not 1 <= max_operations <= 64:
            return {"outcome": "rejected", "because": "max_operations must be an integer within 1..64"}
        record = self.output / "environment.json"
        held: dict[str, Any] = {}
        if record.is_file():
            try:
                loaded = json.loads(record.read_text(encoding="utf-8"))
                held = loaded if isinstance(loaded, dict) else {}
            except (OSError, ValueError):
                held = {}
        probe_document: dict[str, Any] = {}
        plan_path = self.output / "plan.json"
        try:
            if plan_path.is_file() and not plan_path.is_symlink():
                loaded_plan = read_json(plan_path)
                probe_document = loaded_plan if isinstance(loaded_plan, dict) else {}
        except (OSError, ValueError, TypeError):
            probe_document = {}
        stage_rows = self._surveyed_runnable_stages()
        probe_context = {"stage_paths": {
            name: {"entrypoint": str(row.get("entrypoint") or "")}
            for name, row in stage_rows.items() if row.get("entrypoint")}}
        probes_to_check = held.get("probes") or probe_document.get("probes") or []
        probe_faults = [provision.probe_stage_violation(probe, probe_context)
                        for probe in probes_to_check if isinstance(probe, str)]
        probe_faults = [fault for fault in probe_faults if fault]
        later_stage_failure = _last_stage_failure_after_build(
            self.steps, self.output, self.run_id)
        native_path = self.output / "native_context.json"
        native_identity = read_json(native_path).get("identity") if native_path.is_file() else None
        verified_stages = held.get("verified_entrypoints") or {}
        verified_context = {"stage_paths": {name: row for name, row in probe_context["stage_paths"].items()
                                            if name in verified_stages}}
        context_changed = bool(held.get("verified_native_context") and
            (held["verified_native_context"] != native_identity or
             verified_stages != provision.source_context_hashes(self.repo, verified_context)))
        if ((held.get("verdict") or {}).get("passed") and self.interpreter and
                not probe_faults and not later_stage_failure and not context_changed):
            return {"outcome": "already done",
                    "because": f"the record holds a passing build and the interpreter at "
                               f"{self.interpreter}; nothing to rebuild",
                    "interpreter": str(self.interpreter)}
        if not self.declaration:
            return {"outcome": "not attempted",
                    "because": "no declaration yet: the build is told what this benchmark "
                               "needs to import, and that is what a declaration records"}
        assets = self.declaration.get("assets") or {}
        stage_rows = self._surveyed_runnable_stages()
        if not stage_rows:
            return {"outcome": "not attempted", "because":
                    "no surveyed runnable stage; read the checkout before provisioning"}
        selected = self._select_execution_path()
        available = selected["stages"]
        stage_paths = {name: {"entrypoint": stage_rows[name].get("entrypoint"),
                              "invocation": stage_rows[name].get("invocation"),
                              "artifact": stage_rows[name].get("artifact")}
                       for name in available}
        needed_assets = {name: assets[name] for name in selected["asset_keys"]}
        asset_present = {}
        for name, asset in needed_assets.items():
            declared = str(asset.get("path") or "") if isinstance(asset, dict) else ""
            candidate = Path(declared) if declared else None
            if candidate is not None and not candidate.is_absolute():
                candidate = self.repo / candidate
            asset_present[name] = bool(candidate and candidate.exists() and
                                       not candidate.is_symlink())
        planner_assets = {**needed_assets, "research_context": {
                            "task": self._declared_task(),
                            "available_stages": available,
                            "stage_paths": stage_paths,
                            "selected_asset_keys": selected["asset_keys"],
                            "asset_present_at_declared_path": asset_present,
                            "writable_output": str(self.output.resolve()),
                            "selected_path_reason": selected["why"],
                            "dataset_present": asset_present.get("dataset", False),
                            "checkpoint_present": asset_present.get("checkpoint", False)}}
        if later_stage_failure:
            planner_assets["observed_stage_failure"] = later_stage_failure
        existing_python = self.interpreter
        if existing_python is None:
            recorded_python = str(held.get("interpreter") or "").strip()
            candidate = Path(recorded_python).expanduser() if recorded_python else None
            if candidate is not None and candidate.is_file():
                existing_python = candidate
        build_result = provision.build(self.repo, client=self.client, prefix=self.output / "env",
                        output=self.output,
                        assets=planner_assets,
                        budget=self.budget, max_operations=max_operations, compute=self.decision,
                        python=(str(existing_python) if existing_python else
                                str(self.interpreter_hint) if self.interpreter_hint else None))
        self.interpreter = provision.env_python(self.output)
        if (build_result or {}).get("status") == "yielded":
            latest = build_result.get("latest_attempt") or {}
            handoff = {"task_id": "provision-" + str(latest.get("attempt_id") or uuid.uuid4().hex),
                "role": "init", "task": "原生环境准备", "expected_result": "能力证据或可定位故障",
                "report": {"summary": "原生准备操作已返回；不代表仿真就绪。",
                    "findings": [str(latest.get("excerpt") or "")[:1000]],
                    "uncertainties": ["尚未完成正式 baseline"], "evidence_refs": [latest["evidence_ref"]] if latest.get("evidence_ref") else [],
                    "recommended_next_actions": ["审核待办、配置身份与原操作复验结果"],
                    "authority": "executor_receipt_not_performance_claim"}, "at": now()}
            self._save_main_memory({**self.main_agent, "handoffs":
                [*(self.main_agent.get("handoffs") or []), handoff][-12:]}, event="provision_checkpoint")
            return {"outcome": "checkpoint failure" if latest.get("ok") is False else "checkpoint", "because":
                    "一次原生准备操作已封存，控制权返回 Scheduler；环境尚未核验。",
                    "pending_commands": build_result.get("pending_commands"),
                    "latest_failure": build_result.get("latest_failure"),
                    "evidence_refs": self._existing_evidence_refs(
                        "environment.json", "provision_cursor.json", "native_context.json")}
        old_verdict = held.get("verdict") if isinstance(held.get("verdict"), dict) else {}
        return {"outcome": "done" if self.interpreter else "no interpreter recorded",
                "because": f"interpreter: {self.interpreter}" if self.interpreter else
                           ("the re-verification did not restore a passing environment"
                            if old_verdict.get("passed") else
                            "the build finished without recording an interpreter"),
                "verification_reopened": bool(later_stage_failure or probe_faults),
                "observed_stage_failure": ({"stage": later_stage_failure.get("stage"),
                                             "evidence_ref": later_stage_failure.get(
                                                 "evidence_ref")}
                                            if later_stage_failure else None),
                "evidence_refs": self._existing_evidence_refs(
                    "environment.json", "selected_path.json")}

    def _select_execution_path(self) -> dict[str, Any]:
        """Choose one producer-to-score path before provisioning optional workflows."""
        answer = self.execution or read_json(self.output / "execution.json")
        rows = {name: row for name, row in (answer.get("stages") or {}).items()
                if isinstance(row, dict) and row.get("available")}
        target = self._score_target()
        assets = self.declaration.get("assets") or {}
        identity = object_digest({"execution": answer, "declaration": self.declaration,
                                  "task": self._declared_task()})
        path = self.output / "selected_path.json"
        if path.is_file():
            saved = read_json(path)
            if (saved.get("input_digest") == identity and
                    (saved.get("handoff_review") or {}).get("compatible") is True):
                if ("checkpoint" in (saved.get("asset_keys") or []) and
                        "train" in (saved.get("stages") or []) and
                        not self._shipped_checkpoint()):
                    saved["asset_keys"] = [key for key in saved["asset_keys"]
                                           if key != "checkpoint"]
                    atomic_json(path, saved)
                return saved
        prompt = ("Select the minimal native producer-to-score workflow for this one task. "
                  "A repository may expose several unrelated training families; their "
                  "dependencies are not cumulative. For a first measurement, prefer the "
                  "valid path with fewer unavailable external assets and shorter bounded "
                  "verification cost, without claiming it is the best final method. "
                  "Use source-backed stage invocations "
                  "and the task contract. If an online trainer interacts directly with the "
                  "simulator, do not include demonstration conversion or download. Include "
                  "only external prerequisites that must exist before the selected path; "
                  "a checkpoint produced by a selected train stage is a graph edge, not an "
                  "asset to download or build during environment provisioning. Scene/runtime assets "
                  "are still required when a selected stage consumes them. Return JSON "
                  "{\"stages\": [stage names in dependency order], \"asset_keys\": "
                  "[keys from declared assets], \"why\": source-backed reason}. "
                  "The score target must be included. If an artifact evaluator has no "
                  "shipped checkpoint and training exists, include train. Never invent a "
                  "stage, asset key or runnable command. A `collect` stage that only writes "
                  "trajectories is not a policy/checkpoint producer and cannot replace train.")
        from .execution_derive import _object
        payload = {"task": self._declared_task(), "score_target": target,
                   "policy_representation": (self.declaration.get(
                       "task_contract") or {}).get("policy_representation"),
                   "shipped_checkpoint": self._shipped_checkpoint(),
                   "stages": rows, "assets": assets}
        attempts: list[dict[str, Any]] = []

        def persist_rejections() -> None:
            atomic_json(self.output / "path_selection_attempts.json",
                        {"input_digest": identity, "score_target": target,
                         "rows": attempts})

        for _ in range(3):
            model_payload = _controller_model_value(
                {**payload, "prior_rejections": attempts[-2:]},
                local_roots=(self.repo, self.output))
            content, _ = self.client.chat_with_metadata(
                _controller_model_text(prompt, local_roots=(self.repo, self.output)),
                json.dumps(model_payload, ensure_ascii=False)[:24000], max_tokens=1200,
                timeout=max(1, min(180, int(self.budget.remaining())))
                if self.budget else 180, thinking="disabled")
            picked = _object(content)
            stages = picked.get("stages")
            keys = picked.get("asset_keys")
            if (not isinstance(stages, list) or not stages or
                    any(not isinstance(name, str) or name not in rows for name in stages) or
                    len(set(stages)) != len(stages) or target not in stages or
                    not isinstance(keys, list) or
                    any(not isinstance(name, str) or name not in assets for name in keys) or
                    len(set(keys)) != len(keys)):
                attempts.append({"stages": stages, "why_rejected":
                                 "selected execution path must name only available stages, "
                                 "declared asset keys, and the score target"})
                persist_rejections()
                continue
            shipped_checkpoint = self._shipped_checkpoint()
            if ("train" in rows and not shipped_checkpoint and
                    (self.declaration.get("task_contract") or {}).get(
                        "policy_representation") == "artifact" and "train" not in stages):
                attempts.append({"stages": stages, "why_rejected":
                                 "artifact score path without shipped policy must include train"})
                persist_rejections()
                continue
            score_row = rows[target]
            score_inputs = " ".join([
                str(score_row.get("invocation") or ""),
                " ".join(str(parameter.get("name") or "")
                         for parameter in (score_row.get("parameters") or [])
                         if isinstance(parameter, dict)),
            ]).lower().replace("_", "-")
            policy_representation = str((self.declaration.get("task_contract") or {}).get(
                "policy_representation") or "").lower()
            score_needs_policy = (
                policy_representation in {"artifact", "checkpoint", "model", "weights"} or
                any(token in score_inputs for token in
                    ("checkpoint", "ckpt", "model-path", "policy-path")))

            def declares_policy_artifact(stage_name: str) -> bool:
                artifact = str(rows[stage_name].get("artifact") or "").lower()
                prefix = artifact.split("*", 1)[0].split("{", 1)[0]
                suffix = Path(prefix).suffix
                return (any(token in artifact for token in
                            ("checkpoint", "ckpt", "policy", "model")) or
                        suffix in {".pt", ".pth", ".ckpt", ".pkl", ".pickle",
                                   ".safetensors", ".msgpack", ".flax"})

            has_policy_producer = any(
                stage_name != target and declares_policy_artifact(stage_name)
                for stage_name in stages)
            if (score_needs_policy and not shipped_checkpoint and
                    "train" not in stages and "checkpoint" not in keys and
                    not has_policy_producer):
                attempts.append({"stages": stages, "why_rejected":
                                 "the selected score command consumes a policy/checkpoint, "
                                 "but the path has no shipped checkpoint, no selected "
                                 "checkpoint asset, and no selected stage whose declared "
                                 "artifact is a policy; a trajectory-producing collect "
                                 "stage cannot supply the score policy"})
                persist_rejections()
                continue
            if "checkpoint" in keys and "train" in stages and not shipped_checkpoint:
                keys = [name for name in keys if name != "checkpoint"]
            if not str(picked.get("why") or "").strip():
                attempts.append({"stages": stages, "why_rejected":
                                 "selected execution path needs source-backed reasoning"})
                persist_rejections()
                continue
            # An external prerequisite cannot simultaneously be an output of a stage
            # selected for this same path. Reject the graph here, before provisioning
            # repeatedly tries to acquire a file that only a future node can write.
            selected_rows = {name: rows[name] for name in stages}
            produced_asset = None
            for key in keys:
                asset = assets.get(key)
                if not isinstance(asset, dict):
                    continue
                violation = provision.future_stage_output_violation(
                    str(asset.get("path") or ""), {"stage_paths": selected_rows})
                if violation:
                    produced_asset = (key, violation)
                    break
            if produced_asset:
                attempts.append({"stages": stages, "why_rejected":
                                 f"asset {produced_asset[0]} is not external: "
                                 f"{produced_asset[1]}"})
                persist_rejections()
                continue
            review = self._review_policy_handoff(rows, stages, target, str(picked["why"]))
            if review.get("compatible") is False:
                attempts.append({"stages": stages, "why_rejected": str(review.get("why"))[:600]})
                persist_rejections()
                continue
            selected = {"input_digest": identity, "stages": stages,
                        "asset_keys": keys, "why": str(picked["why"])[:1200],
                        "handoff_review": review}
            atomic_json(path, selected)
            return selected
        raise ValueError("no source-supported train-to-score policy handoff after three "
                         f"selection attempts: {attempts[-1] if attempts else 'none'}")

    def _review_policy_handoff(self, rows: dict[str, Any], stages: list[str],
                               target: str, reason: str) -> dict[str, Any]:
        """Ask for an explicit save/load compatibility proof across distinct entrypoints."""
        if "train" not in stages or target == "train":
            return {"compatible": True, "why": "no train-to-score artifact handoff"}
        train, score = rows["train"], rows[target]
        if train.get("entrypoint") == score.get("entrypoint"):
            return {"compatible": True, "why": "same native entrypoint; load is still tested later"}

        def excerpt(row: dict[str, Any]) -> str:
            relative = Path(str(row.get("entrypoint") or ""))
            source = (self.repo / relative).resolve()
            if (not source.is_relative_to(self.repo.resolve()) or not source.is_file() or
                    source.stat().st_size > 2 * 1024**2):
                return "source unavailable; do not assume compatibility"
            lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
            hits = [index for index, line in enumerate(lines) if any(
                word in line.lower() for word in
                ("checkpoint", "state_dict", "torch.save", "torch.load", "policy"))]
            indexes = sorted({near for index in hits[:50] for near in
                              range(max(0, index - 1), min(len(lines), index + 2))})
            return "\n".join(f"{index + 1}: {lines[index]}" for index in indexes)[:6000]

        question = ("Determine whether the selected native train stage's exact checkpoint "
                    "format is loadable by the selected native score stage. Same file suffix "
                    "or both being policies is not evidence. If source does not prove a "
                    "producer→consumer match, return compatible=false. Return JSON with "
                    "compatible (boolean) and why (specific save/load evidence or mismatch).")
        evidence = {"train": train, "score": score, "selection_reason": reason,
                    "train_source_excerpt": excerpt(train),
                    "score_source_excerpt": excerpt(score)}
        content, _ = self.client.chat_with_metadata(
            _controller_model_text(question, local_roots=(self.repo, self.output)),
            json.dumps(_controller_model_value(
                evidence, local_roots=(self.repo, self.output)),
                ensure_ascii=False)[:22000], max_tokens=1000,
            timeout=max(1, min(180, int(self.budget.remaining())))
            if self.budget else 180, thinking="disabled")
        from .execution_derive import _object
        review = _object(content)
        if not isinstance(review.get("compatible"), bool) or not str(review.get("why") or ""):
            return {"compatible": False, "why": "handoff review lacked a definite evidence-backed answer"}
        return {"compatible": review["compatible"], "why": str(review["why"])[:1200]}

    def _step_derive_a_command(self, *, stage: str = "",
                               timeout_seconds: int | float | None = None) -> dict[str, Any]:
        """Find a command that runs one stage, revising the invocation until it does.

        The invocation is what gets revised, not the argv: a failure the invocation caused
        cannot be repaired by generating the command again. That is `make_runnable`'s whole
        job and this step only supplies it with the facts it needs.
        """
        if self.decision.device == "unavailable":
            return {"outcome": "blocked", "because":
                    "command verification may execute simulator code, but the GPU resource "
                    f"decision is unavailable: {self.decision.why}",
                    "resource_block": self.decision.evidence}
        if not self.interpreter:
            return {"outcome": "not attempted",
                    "because": "no interpreter: a command cannot be verified without one"}
        answer = self.execution or (
            json.loads((self.output / "execution.json").read_text(encoding="utf-8"))
            if (self.output / "execution.json").is_file() else {})
        rows = {name: row for name, row in (answer.get("stages") or {}).items()
                if row.get("available")}
        if not rows:
            return {"outcome": "not attempted",
                    "because": "no stage of this checkout has an entry point"}
        wanted = stage or sorted(rows)[0]
        if wanted not in rows:
            return {"outcome": "no such stage",
                    "because": f"{wanted} is not a stage of this checkout; "
                               f"it has {sorted(rows)}"}
        if timeout_seconds is not None:
            if (not isinstance(timeout_seconds, (int, float)) or
                    isinstance(timeout_seconds, bool) or
                    not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
                return {"outcome": "rejected", "because":
                        "timeout_seconds must be a positive finite number"}
        requested_timeout = (float(timeout_seconds) if timeout_seconds is not None else
                             900.0 if wanted == "train" else 300.0)
        if self.budget:
            requested_timeout = min(requested_timeout, self.budget.remaining())
        if requested_timeout <= 0:
            return {"outcome": "blocked", "because": "run wall-clock budget exhausted"}
        from .survey import survey
        checkpoint = execution_derive.checkpoint_for_verification(
            survey(self.repo), declaration=self.declaration)
        requires_checkpoint = (wanted == self._score_target() and
                               "checkpoint" in str(rows[wanted].get("invocation") or "").lower())
        if requires_checkpoint and not checkpoint["path"] and "train" in rows:
            return {"outcome": "not attempted", "because":
                    "the native score command consumes a checkpoint, but none exists; "
                    "derive and verify train first so its fresh policy artifact can be used"}
        inputs = {**execution_derive.blank_inputs(), "python": str(self.interpreter),
                  "repo": str(self.repo), "task": self._declared_task(),
                  "dataset": "", "checkpoint": checkpoint["path"],
                  # One step can be smaller than a trainer's rollout batch and silently
                  # produce an initialization checkpoint with zero optimizer updates.
                  # This is a bounded verification budget, not the research budget.
                  "output": str(self.output / "v"),
                  "steps": execution_derive.TRAINING_VERIFICATION_STEPS, "episodes": 1,
                  "device": self.decision.device, "device_index": self.decision.device_index}
        self._attempts = []
        source, settled, log, row = execution_derive.make_runnable(
            self.client, wanted, rows[wanted], repo=self.repo,
            repository_files=answer.get("read") or [], inputs_for_verify=inputs,
            base_environment=self.decision.environment,
            attempts=execution_derive.ARGV_DERIVATION_ATTEMPTS,
            rounds=execution_derive.ARGV_DERIVATION_ROUNDS,
            require_step_control=wanted == "train",
            require_evaluation_progress=wanted == self._score_target(),
            # A native epoch may include a full data pass and simulator evaluation.
            # This is still a per-attempt ceiling inside the persisted run/GPU budget;
            # 180 seconds incorrectly classified legitimate epoch-only trainers as
            # impossible before a single positive update could finish.
            verification_timeout=requested_timeout,
            remaining_seconds=(self.budget.remaining if self.budget else None),
            on_event=lambda stage, entries: self._note(stage, entries))
        derivation_evidence = self._record_attempts(wanted, inputs)
        if source is None:
            why = next((str(one.get("error") or "") for one in reversed(log)
                        if one.get("status") in {
                            "no command can fix this",
                            "same failure repeated; stopping bounded derivation"}), "")
            return {"outcome": "no runnable command",
                    "because": why[:600] or f"{wanted} has no command that finishes",
                    "evidence_id": derivation_evidence.get("evidence_id"),
                    "evidence_refs": [derivation_evidence["evidence_ref"]]
                    if derivation_evidence else self._existing_evidence_refs(
                        f"derivation_attempts/{wanted}.json")}
        self.stages[wanted] = source
        self.parameters[wanted] = {name: {"value": value}
                                   for name, value in (settled or {}).items()}
        verified = {**row, "parameters": [
            {"name": name, "value": value} for name, value in (settled or {}).items()]}
        self.verified[wanted] = verified
        kept_path = self.output / "derived_stages.json"
        kept = json.loads(kept_path.read_text(encoding="utf-8")) if kept_path.is_file() else {}
        kept[wanted] = {"source": source, "parameters": self.parameters[wanted],
                        "row": verified}
        atomic_json(kept_path, kept)
        remaining = [name for name in rows if name not in self.stages]
        return {"outcome": "done", "because": f"{wanted} runs",
                "still_without_a_command": remaining,
                "evidence_refs": self._existing_evidence_refs(
                    f"derivation_attempts/{wanted}.json", "derived_stages.json")}

    def _step_bind_metric(self) -> dict[str, Any]:
        """Bind a native log or fresh structured result to a source-backed metric."""
        target = self._score_target()
        source = self.stages.get(target)
        if not source:
            return {"outcome": "not attempted", "because": "no verified score command"}
        path = self.output / "derivation_attempts" / f"{target}.json"
        try:
            rows = read_json(path).get("attempts") or []
        except (OSError, ValueError, TypeError):
            rows = []
        accepted = next((row for row in reversed(rows)
                         if row.get("status") == "accepted"), {})
        said = str(accepted.get("said") or "")
        artifact_evidence = accepted.get("verified_artifact") or {}
        structured_candidates = artifact_evidence.get("structured_candidates") or []
        if not said.strip() and not structured_candidates:
            return {"outcome": "not bound",
                    "because": "the verified score command has neither native output nor a "
                              "fresh structured result artifact"}

        def resolve_proposal(spec: MetricSpec) -> dict[str, Any]:
            if spec.source == "log":
                return {"status": "log", "reading": spec.read_log(said)}
            result = resolve_metric_artifact(
                spec,
                roots={
                    "working_directory": Path(str(artifact_evidence.get(
                        "working_directory") or "")),
                    "output": Path(str(artifact_evidence.get("output_directory") or "")),
                    "policy_parent": Path(str(artifact_evidence.get("policy_parent") or "")),
                },
                started_at=float(artifact_evidence.get("attempt_started") or 0),
                allowed_roots=[self.repo, self.output])
            if result.get("status") != "matched":
                return {**result, "reading": {"value": None}}
            recorded = next((item for item in structured_candidates
                             if item.get("root") == spec.artifact_root and
                             str(Path(str(item.get("path") or "")).resolve()) ==
                             result.get("path")), None)
            if (recorded is None or recorded.get("sha256") != result.get("sha256") or
                    Path(str(recorded.get("path") or "")).is_symlink()):
                return {"status": "not_in_verified_candidates", "matched": 0,
                        "reading": {"value": None}}
            result["verification_attempt"] = artifact_evidence.get("attempt_started")
            result["reading"] = spec.read(said=said, artifact=Path(result["path"]))
            return result

        if self._metric_bound():
            existing = MetricSpec.from_declaration(self.declaration)
            reading = resolve_proposal(existing).get("reading") or {"value": None}
            if (reading.get("value") is not None and
                    not (existing.source == "log" and structured_candidates)):
                return {"outcome": "already done", "because": "declared metric occurs in "
                        "the current verified native output/artifact"}
        entrypoint = str((self.verified.get(target) or {}).get("entrypoint") or "")
        candidate = (self.repo / entrypoint).resolve()
        if not candidate.is_relative_to(self.repo) or not candidate.is_file():
            return {"outcome": "not bound", "because": "score entrypoint source unavailable"}
        code = candidate.read_text(encoding="utf-8", errors="replace")[:24000]
        candidate_previews = []
        model_candidates: dict[str, dict[str, Any]] = {}
        for item in structured_candidates[:12]:
            path = Path(str(item.get("path") or ""))
            try:
                if (path.is_symlink() or not path.is_file() or
                        digest(path) != str(item.get("sha256") or "")):
                    continue
                candidate_id = f"candidate_{len(candidate_previews) + 1}"
                model_candidates[candidate_id] = item
                candidate_previews.append({
                    "candidate_id": candidate_id,
                    "root": item.get("root"),
                    "suffix": item.get("suffix"), "size_bytes": item.get("size_bytes"),
                    "schema_only_no_result_values": _structured_result_schema(path),
                })
            except OSError:
                continue

        def localize_candidate_alias(proposal: Any) -> tuple[Any, str | None]:
            if not isinstance(proposal, dict):
                return proposal, None
            if str(proposal.get("source") or "log").casefold() not in {"json", "csv"}:
                return proposal, None
            alias = str(proposal.get("artifact_candidate") or "")
            item = model_candidates.get(alias)
            if item is None:
                return proposal, "structured metric did not select a listed opaque candidate"
            if proposal.get("artifact_pattern") not in (None, ""):
                return proposal, "structured metric returned a local path instead of a candidate ID"
            requested_root = proposal.get("artifact_root")
            if requested_root not in (None, "", item.get("root")):
                return proposal, "structured metric candidate root disagrees with the verified row"
            localized = {key: value for key, value in proposal.items()
                         if key not in {"artifact_candidate", "artifact_pattern"}}
            localized["artifact_root"] = item["root"]
            localized["artifact_pattern"] = item["relative_path"]
            return localized, None

        prompt = (
            "Infer the one primary native task metric from the evaluator source and the "
            "fresh verified output evidence. The `verified_output_shape` has numeric values "
            "redacted; use it only to identify exact log labels and semantics. Structured "
            "candidate schemas contain field names/types, not episode result values. "
            "When a fresh structured candidate represents completed native episodes with "
            "identities and per-episode outcomes, prefer it over an aggregate log metric; "
            "the structured reading will be checked against any corresponding log aggregate. "
            "Return JSON with `primary_metric` and a source-backed `evidence`. Optionally "
            "include guardrail_metrics: at most 16 secondary metrics with the same mapping "
            "fields, explicit max_regression in native units, and source-backed evidence "
            "for meaning/direction/tolerance. Use the SAME structured artifact_candidate "
            "as primary, or an exact native log label. Omit metrics/thresholds not justified "
            "by the experiment constraints. These become immutable before research and "
            "missing readings prevent primary-only wins. The primary metric "
            "may use source `log`, `json`, or `csv`. A structured source must select exactly "
            "one listed opaque `artifact_candidate` such as `candidate_1`; do not return any "
            "path, file name, glob, digest or episode value. The local verifier maps that "
            "candidate ID to its frozen artifact. For JSON use "
            "`json_key` for a scalar path or record array, and for per-episode arrays use "
            "`json_value_key`, `episode_id_column`, optional `task_column`, and an explicit "
            "`aggregation`; declare `min_samples` only when the artifact's actual rows are "
            "the completed samples. For CSV use `csv_column` and explicit aggregation. "
            "Only declare task/episode identity when the source and schema establish its "
            "meaning; never infer completed episode counts from a requested setting or an "
            "aggregate log line. Optional bounds are `min` and `max`. Return `null` if "
            "the native meaning is not established. Example shape: {\"primary_metric\": "
            "{\"name\":\"...\",\"direction\":\"maximize| minimize\",\"unit\":\"...\","
            "\"source\":\"json\",\"artifact_candidate\":\"candidate_1\","
            "\"json_key\":\"...\","
            "\"json_value_key\":\"...\",\"episode_id_column\":\"...\","
            "\"task_column\":\"...\",\"aggregation\":\"mean\"},"
            "\"evidence\":\"...\"}.")
        output_shape = re.sub(
            r"(?<![A-Za-z_])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
            "<number>", said[-10000:])
        payload_data = {"task": self._declared_task(), "entrypoint": entrypoint,
                        "source": code, "verified_output_shape": output_shape,
                        "fresh_structured_candidates": candidate_previews}
        payload_data = _controller_model_value(
            payload_data, local_roots=(self.repo, self.output))
        payload = json.dumps(payload_data, ensure_ascii=False)
        request_timeout = (max(1, min(180, int(self.budget.remaining())))
                           if self.budget else 180)
        content, _ = self.client.chat_with_metadata(
            prompt, payload, max_tokens=1200, timeout=request_timeout,
            thinking="disabled")
        from .execution_derive import _object
        answer = _object(content)
        metric, metric_alias_error = localize_candidate_alias(answer.get("primary_metric"))
        # A model can identify an exact aggregate log label while overlooking that the
        # same evaluator invocation also emitted fresh per-record results. Give it one
        # bounded chance to reconcile the two sources. This is deliberately evidence-led:
        # we do not guess field names or aggregate values here, and the resulting mapping
        # still has to resolve to one hashed candidate and agree with the log below.
        if (isinstance(metric, dict) and str(metric.get("source") or "log").lower() ==
                "log" and candidate_previews):
            review_prompt = (prompt + "\n\nThe proposal selected a log aggregate even though "
                             "fresh structured candidates were supplied. Re-examine those "
                             "schemas and the evaluator source once: if one candidate maps "
                             "completed native records to this same metric, return that "
                             "structured mapping (including record identity, value field, "
                             "aggregation and minimum observed sample count). If none does, "
                             "keep the log source and explicitly explain why each candidate "
                             "cannot establish this metric. Do not infer values from schemas.")
            review_payload = json.dumps(_controller_model_value({**payload_data,
                                         "prior_proposal": metric,
                                         "prior_evidence": answer.get("evidence")},
                                        local_roots=(self.repo, self.output)),
                                        ensure_ascii=False)
            try:
                review_content, _ = self.client.chat_with_metadata(
                    review_prompt, review_payload, max_tokens=1200,
                    timeout=request_timeout, thinking="disabled")
                reviewed_answer = _object(review_content)
                if isinstance(reviewed_answer.get("primary_metric"), dict):
                    answer = reviewed_answer
                    metric, metric_alias_error = localize_candidate_alias(
                        answer["primary_metric"])
            except (OSError, RuntimeError, TimeoutError, ValueError):
                # The first source-backed proposal remains eligible; a failed optional
                # review must not erase it or turn transport trouble into a fake metric.
                pass
        if metric_alias_error:
            return {"outcome": "not bound", "because": metric_alias_error}
        if not isinstance(metric, dict) or not str(answer.get("evidence") or "").strip():
            excerpt = output_shape.strip().replace("\n", " | ")[-400:]
            return {"outcome": "not bound", "because":
                    "verified native output/artifact schema and source did not establish one "
                    "exact metric; "
                    f"output excerpt: {excerpt or '(empty)'}; re-derive and verify the score "
                    "command before retrying metric binding"}
        trial = {**self.declaration,
                 "research_goal": {**(self.declaration.get("research_goal") or {}),
                                   "primary_metric": metric}}
        spec = MetricSpec.from_declaration(trial)
        verified_metric = resolve_proposal(spec)
        reading = verified_metric.get("reading") or {"value": None}
        if reading.get("value") is None:
            return {"outcome": "not bound", "because": "proposed metric could not be "
                    "read from the exact verified output using its declared mapping; "
                    f"verification={verified_metric.get('status')}; re-derive a fresh score "
                    "artifact before retrying metric binding"}
        if spec.source != "log":
            log_reading = spec.read_log(said)
            if log_reading.get("value") is not None and not math.isclose(
                    float(log_reading["value"]), float(reading["value"]),
                    rel_tol=1e-6, abs_tol=1e-8):
                return {"outcome": "not bound", "because":
                        "structured native aggregate disagrees with the evaluator's exact "
                        "log metric; preserve the conflict and do not bind either source"}
        guardrails = answer.get("guardrail_metrics", (self.declaration.get("research_goal") or {}).get("guardrail_metrics", []))
        if not isinstance(guardrails, list) or len(guardrails) > 16:
            return {"outcome": "not bound", "because": "secondary objectives must be a bounded list"}
        localized_guardrails = []
        for guardrail in guardrails:
            if not isinstance(guardrail, dict) or not str(guardrail.get("evidence") or "").strip():
                return {"outcome": "not bound", "because": "secondary objective lacks source/constraint evidence"}
            secondary, error = localize_candidate_alias(guardrail)
            # Already localized declaration rows need not choose an opaque candidate again.
            if error and guardrail.get("artifact_pattern"):
                secondary, error = guardrail, None
            if error:
                return {"outcome": "not bound", "because": error}
            secondary_spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": secondary}})
            if (secondary_spec.source != "log" and
                    (secondary_spec.source, secondary_spec.artifact_root, secondary_spec.artifact_pattern) !=
                    (spec.source, spec.artifact_root, spec.artifact_pattern)):
                return {"outcome": "not bound", "because": "secondary result must share the verified primary archive"}
            if (resolve_proposal(secondary_spec).get("reading") or {}).get("value") is None:
                return {"outcome": "not bound", "because": "secondary metric cannot be read from verified native output"}
            localized_guardrails.append(secondary)
        from .metric_guardrails import specifications
        specifications({"research_goal": {"guardrail_metrics": localized_guardrails}})
        binding = {
            "score_target": target, "stage_source_sha256": hashlib.sha256(
                source.encode()).hexdigest(), "primary_metric": metric,
            "evidence": str(answer["evidence"])[:1000], "guardrail_metrics": localized_guardrails,
            "verified_output_sha256": hashlib.sha256(said.encode()).hexdigest(),
            "verified_derivation_attempt": accepted.get("attempt"),
            "reviewed_structured_candidates_sha256": (
                object_digest(structured_candidates) if structured_candidates else ""),
            "verified_artifact": ({key: verified_metric.get(key) for key in
                                   ("path", "sha256", "size_bytes", "mtime", "root",
                                    "pattern", "matched", "freshness_checked")
                                   if key in verified_metric}
                                  if spec.source in {"json", "csv"} else {})}
        atomic_json(self.output / "metric_binding.json", binding)
        self._set_primary_metric(metric)
        self._set_metric_guardrails(localized_guardrails)
        if self.main_agent.get("metric_revalidation_required"):
            self._save_main_memory({**self.main_agent, "metric_revalidation_required": False},
                                   event="metric_revalidated")
        return {"outcome": "done", "because": f"bound {spec.name} to verified native "
                f"{spec.source} output ({reading.get('episodes_completed', 1)} sample(s))"}

    def _record_attempts(self, stage: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Write down every attempt at one stage, with what the program said.

        `make_runnable` returns this log and nothing kept it: `provision.build` writes a
        transcript of its rounds, and the derivation -- which runs the benchmark's own
        program forty times and is the more expensive of the two -- wrote nothing. So the
        only surviving trace of a failed derivation was the last outcome, and a reader asking
        "what did the loop actually see" had nowhere to look. It is also the archive the next
        run's failure memory reads from.
        """
        directory = Path(self.output) / "derivation_attempts"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stage}.json"
        atomic_json(path,
                    {"stage": stage, "at": now(),
                     "arguments": {k: str(v)[:400] for k, v in arguments.items()},
                     "attempts": self._attempts})
        from .evidence_store import capture_attempt_evidence
        evidence_id = uuid.uuid4().hex
        try:
            sealed_path = self.output / "evidence" / "logs" / f"{evidence_id}.json"
            sealed_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, sealed_path)
            self._derivation_evidence = capture_attempt_evidence(
                self.output, attempt_id=evidence_id, log=sealed_path,
                receipt_ref=("action_receipts/" +
                             str(self.current_action.get("decision_id") or "") + ".json"),
                status="accepted" if any(row.get("status") == "accepted"
                                         for row in self._attempts) else "failed",
                returncode=None, termination_reason="command_derivation")
        except (OSError, ValueError, TypeError):
            self._derivation_evidence = {}
        return dict(self._derivation_evidence)

    def _research_controller(self) -> tuple[DerivedResearch, dict[str, Any]]:
        """Rebuild the research controller from verified, persisted run inputs."""
        execution_path = self.output / "execution.json"
        answer = self.execution or (json.loads(execution_path.read_text(encoding="utf-8"))
                                    if execution_path.is_file() else {"stages": {}})
        stage_names = set((answer.get("stages") or {})) | set(self.stages)
        answer = {**answer, "stages": {
            name: {**((answer.get("stages") or {}).get(name) or {}),
                   **(self.verified.get(name) or {}), "available": name in self.stages}
            for name in stage_names}}
        backend = DeclarativeBackend(repo=self.repo, answer=answer, sources=self.stages,
                                     parameters=self.parameters)
        research = DerivedResearch(repo=self.repo, output=self.output, backend=backend,
                                   interpreter=self.interpreter, space=space_from(self.declaration),
                                   client=self.client, stages=self.stages, run_id=self.run_id,
                                   compute=self.decision,
                                   benchmark=str(self.declaration.get("benchmark") or ""),
                                   declaration=self.declaration,
                                   require_training_progress=True,
                                   require_policy_consumption=bool(getattr(
                                       self.client, "supports_main_agent", False)),
                                   checkpoint=self._shipped_checkpoint(),
                                   controller_decision_id=str(
                                       self.current_action.get("decision_id") or ""))
        return research, answer

    def _step_run_the_loop(self, *, idea_label: str = "", job_id: str = "") -> dict[str, Any]:
        """Run one bounded research action, then return control to Preparation."""
        report_path = self.output / "research" / self.run_id / "research_report.json"
        session_path = report_path.with_name("controller_session.json")
        prior: dict[str, Any] | None = None
        session: dict[str, Any] | None = None
        if report_path.exists():
            try:
                prior = read_json(report_path)
            except (OSError, ValueError, TypeError) as exc:
                return {"outcome": "blocked", "because":
                        "existing research report is unreadable; refusing to overwrite it: "
                        f"{type(exc).__name__}: {exc}"}
            if (not isinstance(prior, dict) or prior.get("run_id") != self.run_id or
                    Path(str(prior.get("repo") or "")).expanduser().resolve() != self.repo):
                return {"outcome": "blocked", "because":
                        "existing research report identity does not match this run; refusing "
                        "to launch another experiment"}
        if session_path.is_file():
            try:
                session = read_json(session_path)
            except (OSError, ValueError, TypeError) as exc:
                return {"outcome": "blocked", "because":
                        "existing research session is unreadable; refusing to relaunch: "
                        f"{type(exc).__name__}: {exc}"}
            if (not isinstance(session, dict) or session.get("run_id") != self.run_id or
                    Path(str(session.get("repository") or "")).expanduser().resolve() !=
                    self.repo):
                return {"outcome": "blocked", "because":
                        "existing research session identity does not match this run"}
            session_status = str(session.get("status") or "unreadable")
            if session_status == "paused" and not idea_label:
                return {"outcome": "not attempted", "because":
                        "the main controller must select one exact audited idea before a "
                        "paused candidate round can resume"}
            if session_status != "paused" and idea_label:
                return {"outcome": "not attempted", "because":
                        "idea_label cannot be applied to baseline recovery or finalization"}
            if session_status == "interrupted":
                interruption = session.get("interruption")
                reconciliation = session.get("reconciliation")
                if (not isinstance(interruption, dict) or
                        interruption.get("controller_action") != "baseline" or
                        not isinstance(reconciliation, dict) or
                        reconciliation.get("status") != "reconciled"):
                    return {"outcome": "blocked", "because":
                            "this interrupted research action is not safely resumable; inspect "
                            "its pending decision, process receipt, source changes, and outputs; "
                            "candidate-round replay is intentionally not automatic"}
                resume_interrupted = True
            elif session_status not in {"paused", "finalizing"}:
                return {"outcome": "blocked", "because":
                        f"research session status is {session_status}; reconcile its "
                        "action before any retry"}
            else:
                resume_interrupted = False
        else:
            resume_interrupted = False
        if prior is not None:
            report_status = str(prior.get("run_status") or "completed")
            if report_status == "completed" and not resume_interrupted:
                return {"outcome": "already done", "because":
                        "a terminal research report already exists; inspect it or use the "
                        "separate confirmation action rather than repeating training"}
            if report_status not in {"paused", "finalizing", "interrupted", "completed"}:
                return {"outcome": "blocked", "because":
                        f"research status is {report_status}; reconcile its action before "
                        "any retry"}
            if report_status == "interrupted" and not resume_interrupted:
                return {"outcome": "blocked", "because":
                        "the research report records an interruption without a verified "
                        "resumable controller session"}
        target = self._score_target()
        if not self._research_loop_ready():
            return {"outcome": "not attempted",
                    "because": f"the loop needs a declaration, interpreter, and verified "
                               f"score command for {target}, plus an explicit primary "
                               "metric and any producer required by the selected workflow; "
                               "an incidental checkpoint from failed training is not enough"}
        research, answer = self._research_controller()
        baseline_training_attempt_id = ""
        if job_id:
            from .native_jobs import status as native_job_status, source_identity
            try:
                job = native_job_status(self.output, job_id)
                request = read_json(self.output / "native_jobs" / job_id / "request.json")
                result = (job.get("result") or {}).get("stage_result") or {}
                attempt = str(result.get("attempt_id") or "")
                settings = research._base_settings({
                    k: v for k, v in self.base_settings.items() if not k.startswith("_")})
                if (job.get("status") != "completed" or
                        request.get("repo") != str(self.repo) or
                        request.get("run_id") != self.run_id or
                        request.get("settings") != settings or
                        not request.get("source_identity") or
                        request.get("source_identity") != source_identity(
                            self.output, self.repo) or
                        result.get("stage") != "train" or
                        research._verified_training_receipt(attempt, settings) is None):
                    raise ValueError("job result does not match a verified baseline receipt")
                baseline_training_attempt_id = attempt
            except (OSError, ValueError, TypeError, KeyError) as exc:
                return {"outcome": "not attempted", "because":
                        f"detached baseline adoption refused: {type(exc).__name__}: {exc}"}
        attempt_root = research.run_root / "attempts"
        attempts_before = ({path.parent.name for path in attempt_root.glob("*/receipt.json")}
                           if attempt_root.is_dir() else set())

        def new_attempt_evidence() -> tuple[list[str], list[str]]:
            attempts = (sorted(attempt_root.glob("*/receipt.json"))
                        if attempt_root.is_dir() else [])
            selected = [path for path in attempts
                        if path.parent.name not in attempts_before]
            ids = [path.parent.name for path in selected]
            refs = [str(path.resolve().relative_to(self.output.resolve()))
                    for path in selected]
            return ids, refs

        def candidate_failure_from(report: dict[str, Any]) -> dict[str, Any]:
            if not idea_label:
                return {}
            last = (report.get("rounds") or [])[-1:]
            row = last[0] if last and isinstance(last[0], dict) else {}
            if (row.get("idea") != idea_label or row.get("status") == "measured" or
                    isinstance(row.get("metric_value", row.get("success_rate")),
                               (int, float))):
                return {}
            index = row.get("round")
            if not isinstance(index, int) or isinstance(index, bool) or index < 1:
                return {}
            label = f"round_{index}"
            measure_path = research.run_root / "measurements" / f"{label}.json"
            measurement = (read_json(measure_path) if measure_path.is_file() and
                           not measure_path.is_symlink() else {})
            if not isinstance(measurement, dict):
                measurement = {}
            stage = measurement.get("evaluate") or {}
            if not isinstance(stage, dict):
                stage = {}
            if not stage.get("evidence_id"):
                stage = measurement.get("train") or {}
            if not isinstance(stage, dict):
                stage = {}
            return {"idea_label": idea_label,
                    "measurement_ref": (str(measure_path.relative_to(self.output))
                                        if measure_path.is_file() else ""),
                    "evidence_id": stage.get("evidence_id"),
                    "stage_attempt_id": stage.get("attempt_id"),
                    "why": str((measurement or {}).get("why") or
                               row.get("why_not") or "")[:500]}

        graph = answer.get("execution_graph") or {}
        score_target = graph.get("score_target") if isinstance(graph, dict) else None
        def verified_native_level() -> str:
            """An aggregate number alone does not establish native task/episode semantics."""
            from .receipt_verifier import verify_measurement

            for path in sorted((research.run_root / "measurements").glob("*.json")):
                try:
                    row = read_json(path)
                except (OSError, ValueError):
                    continue
                if not isinstance(row, dict) or row.get("ok") is not True:
                    continue
                reading = row.get("metric_reading") or {}
                metric = row.get("metric") or {}
                settings = row.get("settings") or {}
                keys = reading.get("episode_keys") if isinstance(reading, dict) else None
                if (metric.get("source") in {"csv", "json"} and
                        metric.get("episode_id_column") and
                        (metric.get("task_column") or settings.get("task")) and
                        isinstance(keys, list) and keys and len(set(keys)) == len(keys) and
                        reading.get("episodes_completed") == len(keys) and
                        verify_measurement(research.run_root, path.stem).get("status") ==
                        "consistent"):
                    return "L2"
            return "L1" if self.verified else "L0"

        if score_target and not research.available("evaluate"):
            report = research.run(
                rounds=int(self.base_settings.get("_rounds", 2)),
                settings={k: v for k, v in self.base_settings.items()
                          if k not in ("_rounds", "_confirm")},
                yield_after_action=True, max_rounds_per_action=1,
                resume_interrupted=resume_interrupted,
                selected_idea_label=idea_label,
                main_controller_owns_selection=True,
                baseline_training_attempt_id=baseline_training_attempt_id,
            )
            rounds = report.get("rounds") or []
            attempt_ids, evidence_refs = new_attempt_evidence()
            valid = [row for row in rounds if isinstance(
                row.get("metric_value", row.get("success_rate")), (int, float))]
            level = verified_native_level() if valid else "unknown"
            status = report.get("run_status", "completed")
            candidate_failure = candidate_failure_from(report)
            return {"outcome": ("yielded" if status == "paused" else
                                "done" if valid else "no number came out"),
                    "because": (f"explicit graph: {len(rounds)} recorded row(s), "
                                f"{len(valid)} with a metric; status={status}, "
                                f"next_round={report.get('next_round')}; native task/episode "
                                f"evidence {'verified' if level == 'L2' else 'not verified'}"),
                    "metric_value": (valid[-1].get("metric_value", valid[-1].get("success_rate"))
                                     if valid else None),
                    "verification_level": level,
                    "research_attempted": len(rounds) > 1,
                    "research_status": status,
                    "next_round": report.get("next_round"),
                    "attempt_ids": attempt_ids,
                    "evidence_refs": evidence_refs,
                    **({"candidate_failure": candidate_failure,
                       "evidence_id": candidate_failure.get("evidence_id")}
                      if candidate_failure else {}),
                    **({"selected_idea_label": idea_label} if idea_label else {}),
                    "independently_confirmed": False}
        report = research.run(rounds=int(self.base_settings.get("_rounds", 2)),
                              settings={k: v for k, v in self.base_settings.items()
                                        if k not in ("_rounds", "_confirm")},
                              confirm=False, yield_after_action=True,
                              max_rounds_per_action=1,
                              resume_interrupted=resume_interrupted,
                              selected_idea_label=idea_label,
                              main_controller_owns_selection=True,
                              baseline_training_attempt_id=baseline_training_attempt_id)
        measured = [row for row in report.get("rounds") or []
                    if isinstance(row.get("metric_value", row.get("success_rate")),
                                  (int, float))]
        candidate_failure = candidate_failure_from(report)
        attempt_ids, evidence_refs = new_attempt_evidence()
        level = verified_native_level() if measured else "unknown"
        status = report.get("run_status", "completed")
        return {"outcome": ("yielded" if status == "paused" else
                            "done" if measured else "no number came out"),
                "because": f"{len(report.get('rounds') or [])} recorded row(s), "
                           f"{len(measured)} with a number; native task/episode evidence "
                           f"{'verified' if level == 'L2' else 'not verified'}; "
                           f"status={status}, next_round={report.get('next_round')}",
                "verification_level": level,
                "research_status": status,
                "next_round": report.get("next_round"),
                "attempt_ids": attempt_ids,
                "evidence_refs": evidence_refs,
                **({"candidate_failure": candidate_failure,
                   "evidence_id": candidate_failure.get("evidence_id")}
                  if candidate_failure else {}),
                **({"selected_idea_label": idea_label} if idea_label else {}),
                "objective": (report.get("objective") or {}).get("score")}

    def _step_generate_research_ideas(self) -> dict[str, Any]:
        """Ask for a fresh audited candidate batch without running a benchmark stage."""
        state = self.state()
        if (state.get("research_progress") or {}).get("status") != "paused":
            return {"outcome": "not attempted",
                    "because": "idea generation requires a paused research session"}
        if (state.get("research_options") or {}).get("items"):
            return {"outcome": "not attempted",
                    "because": "select an available idea or propose a main-controller idea "
                               "instead of replacing a usable candidate pool"}
        options_state = state.get("research_options") or {}
        if options_state.get("generation_attempted"):
            return {"outcome": "not attempted",
                    "because": "a suggestion batch was already requested for this paused "
                              "research round"}
        generation_path = (self.output / "research" / self.run_id /
                           "idea_generation.json")
        if generation_path.is_symlink():
            return {"outcome": "blocked",
                    "because": "idea generation receipt is a symlink; refusing to overwrite it"}
        atomic_json(generation_path, {
            "schema_version": 1, "next_round": options_state.get("next_round"),
            "status": "started", "at": now(),
            "evidence_digest": object_digest({
                "last_observation": (state.get("research_progress") or {}).get(
                    "last_observation"),
                "research_options": options_state.get("items") or []}),
        })
        research, _ = self._research_controller()
        research.prepare_idea_options(extend=True, generate_if_empty=False)
        updated_state = self.state()
        options = (updated_state.get("research_options") or {}).get("items") or []
        atomic_json(generation_path, {
            "schema_version": 1, "next_round": options_state.get("next_round"),
            "status": "completed", "at": now(),
            "selectable_ideas": len(options),
            "ideas_digest": object_digest(options),
        })
        return {"outcome": "ideas available" if options else "no selectable ideas",
                "because": (f"audited a fresh suggestion batch; {len(options)} idea(s) are "
                            "currently selectable" if options else
                            "the fresh batch produced no selectable ideas; use the evidence "
                            "to propose a different action or state the boundary"),
                "idea_count": len(options),
                "evidence_ref": f"research/{self.run_id}/ideas.json"}

    def _step_propose_research_idea(self, *, idea: dict[str, Any]) -> dict[str, Any]:
        """Audit a novel main-controller idea without authorizing it to execute."""
        if (self.state().get("research_progress") or {}).get("status") != "paused":
            return {"outcome": "not attempted",
                    "because": "a novel idea may only be proposed while research is paused"}
        research, _ = self._research_controller()
        result = research.propose_research_idea(idea)
        return {"outcome": result.get("status") or "invalid",
                "because": result.get("why") or "idea audit completed",
                "idea_label": (result.get("idea") or {}).get("label") or
                              idea.get("label"),
                "idea_status": (result.get("idea") or {}).get("status") or
                               result.get("status"),
                "evidence_ref": f"research/{self.run_id}/ideas.json"}

    def _step_discard_interrupted_candidate(self, *, reason: str) -> dict[str, Any]:
        """Close only a reconciled pre-stage candidate, without adopting its measurement.

        This is an explicit main-controller decision. It can resume research after a crash
        before any benchmark stage started; later candidate interruptions remain unresolved
        because their external side effects are not generally reversible.
        """
        research_root = self.output / "research" / self.run_id
        session_path = research_root / "controller_session.json"
        report_path = research_root / "research_report.json"
        ideas_path = research_root / "ideas.json"
        if any(path.is_symlink() for path in (self.output / "research", research_root,
                                              session_path, report_path, ideas_path)):
            return {"outcome": "blocked",
                    "because": "research session, report, ideas, or root contains a symlink"}
        if not research_root.resolve().is_relative_to(self.output.resolve()):
            return {"outcome": "blocked",
                    "because": "research records escape the current run output directory"}
        try:
            session = read_json(session_path)
            report = read_json(report_path)
            ideas_document = read_json(ideas_path)
        except (OSError, ValueError, TypeError) as exc:
            return {"outcome": "blocked", "because":
                    f"interrupted candidate records are unavailable: {type(exc).__name__}: {exc}"}
        if (not isinstance(session, dict) or session.get("schema_version") != 1 or
                session.get("run_id") != self.run_id or
                Path(str(session.get("repository") or "")).expanduser().resolve() != self.repo or
                not isinstance(report, dict) or report.get("run_id") != self.run_id or
                Path(str(report.get("repo") or "")).expanduser().resolve() != self.repo or
                not isinstance(ideas_document, dict) or
                not isinstance(ideas_document.get("ideas"), list)):
            return {"outcome": "blocked",
                    "because": "interrupted candidate record identities or schema are invalid"}
        interruption = session.get("interruption")
        reconciliation = session.get("reconciliation")
        boundary = (interruption.get("boundary") if isinstance(interruption, dict) else None)
        resolution = session.get("interruption_resolution")
        progress = {"status": session.get("status"), "interruption": interruption,
                    "reconciliation": reconciliation,
                    "interruption_boundary": boundary,
                    "interruption_resolution": resolution}
        if not self._interrupted_candidate_discard_available(progress):
            if isinstance(resolution, dict) and resolution.get("status") == "completed":
                return {"outcome": "already resolved", "because":
                        "this candidate was already explicitly discarded",
                        "resolution_id": resolution.get("resolution_id")}
            return {"outcome": "blocked",
                    "because": "only a reconciled, identity-matched pre-stage candidate with "
                              "no receipt, process, or measurement may be discarded"}
        if report.get("run_status") not in {"interrupted", "paused", "finalizing"}:
            return {"outcome": "blocked",
                    "because": "research report status conflicts with interrupted session"}
        if (not isinstance(session.get("history"), list) or
                not isinstance(session.get("kinds"), list) or
                not isinstance(report.get("rounds"), list)):
            return {"outcome": "blocked",
                    "because": "research history or kind ledger is malformed"}
        interruption = dict(interruption)
        boundary = dict(boundary)
        round_index = int(boundary["round"])
        pending = interruption.get("pending_action")
        if (not isinstance(pending, dict) or pending.get("round") != round_index or
                pending.get("phase") != boundary.get("phase")):
            return {"outcome": "blocked",
                    "because": "the durable pending action does not match its interruption boundary"}
        idea = pending.get("idea") if isinstance(pending.get("idea"), dict) else {}
        idea_label = str(idea.get("label") or "")
        idea_row = next((row for row in ideas_document["ideas"]
                         if isinstance(row, dict) and row.get("label") == idea_label), None)
        if not idea_label or not isinstance(idea_row, dict):
            return {"outcome": "blocked",
                    "because": "the interrupted candidate is absent from the audited idea library"}
        raw_total_rounds = session.get("rounds") or report.get("planned_rounds")
        if (isinstance(raw_total_rounds, bool) or
                not isinstance(raw_total_rounds, int) or raw_total_rounds < round_index):
            return {"outcome": "blocked",
                    "because": "planned research round count is missing or inconsistent"}

        resolution_id = ""
        if isinstance(resolution, dict) and resolution.get("status") == "discarding":
            if (resolution.get("round") != round_index or
                    resolution.get("idea_label") != idea_label or
                    not resolution.get("resolution_id")):
                return {"outcome": "blocked",
                        "because": "an in-progress discard journal has a different candidate identity"}
            resolution_id = str(resolution["resolution_id"])
            reason = str(resolution.get("reason") or reason)
        else:
            decision_id = str(self.current_action.get("decision_id") or
                              self.last_decision.get("decision_id") or "")
            if not decision_id:
                return {"outcome": "blocked",
                        "because": "discard must be owned by a persisted main-controller decision"}
            resolution_id = object_digest({"round": round_index, "idea_label": idea_label,
                                           "decision_id": decision_id})[:32]
            resolution = {"status": "discarding", "round": round_index,
                          "idea_label": idea_label, "resolution_id": resolution_id,
                          "decision_id": decision_id,
                          "reason": redact(reason.strip())[:1000], "started_at": now()}
            session.update(interruption_resolution=resolution, updated_at=now())
            atomic_json(session_path, session)

        history = list(session.get("history") or [])
        session_rows = [row for row in history if isinstance(row, dict) and
                        row.get("round") == round_index]
        report_rounds = list(report.get("rounds") or [])
        report_rows = [row for row in report_rounds if isinstance(row, dict) and
                       row.get("round") == round_index]
        if (any(row.get("interruption_resolution_id") != resolution_id
                for row in session_rows) or
                any(row.get("interruption_resolution_id") != resolution_id
                    for row in report_rows)):
            return {"outcome": "blocked",
                    "because": "research history already contains a conflicting row for "
                              "the interrupted round"}

        phase = str(boundary.get("phase") or "")
        source_rollback = {"status": "not_required", "paths": []}
        if str(idea.get("granularity") or "") == "code":
            if phase == "idea_selected":
                # The engine checkpoints idea_selected before entering _prepare_idea; no
                # source operation can have started in this phase.
                source_rollback = {"status": "not_applied",
                                   "because": "the code-change phase had not started",
                                   "paths": []}
            elif (phase == "preparing_idea" and
                  pending.get("source_transaction_protocol") == 1 and
                  not (research_root / "code_changes").is_symlink() and
                  not (research_root / "code_changes" /
                       f"round-{round_index}.json").is_symlink() and
                  not (research_root / "code_changes" /
                       f"round-{round_index}.json").exists()):
                source_rollback = {"status": "no_source_change",
                                   "because": "the durable phase marker proves this "
                                              "transaction protocol had not recorded or "
                                              "applied a patch",
                                   "paths": []}
            else:
                source_rollback = self._rollback_interrupted_code_change(
                    round_index=round_index, idea_label=idea_label)
                if source_rollback.get("status") not in {
                        "rolled_back", "already_rolled_back"}:
                    resolution = {**resolution, "status": "blocked",
                                  "last_error": source_rollback.get("because"),
                                  "updated_at": now()}
                    session.update(interruption_resolution=resolution, updated_at=now())
                    atomic_json(session_path, session)
                    return {"outcome": "blocked",
                            "because": "candidate source could not be restored without "
                                      f"overwriting unknown changes: {source_rollback.get('because')}",
                            "source_rollback": source_rollback,
                            "evidence_ref": f"research/{self.run_id}/code_changes/"
                                           f"round-{round_index}.json"}
        elif (research_root / "code_changes" / f"round-{round_index}.json").exists():
            return {"outcome": "blocked",
                    "because": "a source patch transaction exists for an idea not typed as code"}

        library = IdeaLibrary(ideas_path)
        library.note(idea_label, worked=False,
                     outcome=f"explicitly discarded after interrupted pre-stage action: "
                             f"{redact(reason.strip())[:300]}",
                     event_id=resolution_id)
        stored_idea = library.get(idea_label)
        if stored_idea is None or resolution_id not in stored_idea.outcome_event_ids:
            raise ResearchStatePersistenceError(
                "discarded candidate outcome was not durably recorded")

        existing = session_rows
        round_row = {
            "round": round_index, "status": "interrupted candidate discarded",
            "idea": idea_label, "kind": idea.get("granularity"),
            "mechanism": idea.get("mechanism"),
            "verdict": "the main controller discarded this interrupted candidate; "
                       "no candidate metric was adopted",
            "interruption_resolution_id": resolution_id,
            "interruption_phase": phase,
            "measurement_not_adopted": dict(boundary.get("candidate_measurement") or {}),
            "source_rollback": source_rollback,
        }
        if existing:
            if (len(existing) != 1 or
                    existing[0].get("interruption_resolution_id") != resolution_id):
                return {"outcome": "blocked",
                        "because": "research history already contains a conflicting row for "
                                  "the interrupted round"}
            history = [round_row if isinstance(row, dict) and
                       row.get("round") == round_index else row for row in history]
        else:
            history.append(round_row)
        kinds = list(session.get("kinds") or [])
        kind = str(idea.get("granularity") or "")
        if kind:
            kinds.append(kind)
        total_rounds = raw_total_rounds
        next_round = round_index + 1
        next_status = "finalizing" if next_round > total_rounds else "paused"
        resolution = {**resolution, "status": "completed", "completed_at": now(),
                      "source_rollback": source_rollback,
                      "round_history_ref": f"research/{self.run_id}/controller_session.json"}
        boundary.update(status="resolved_discarded", resolution_id=resolution_id,
                        resolved_at=now())
        interruption["boundary"] = boundary
        report_rounds = [row for row in report_rounds if not (
            isinstance(row, dict) and row.get("round") == round_index)]
        report_rounds.append(round_row)
        report.update(rounds=report_rounds, run_status=next_status,
                      next_round=next_round, interruption=interruption,
                      interruption_boundary=boundary,
                      interruption_resolution=resolution, updated_at=now())
        atomic_json(report_path, report)

        session.update(status=next_status, action="", history=history, kinds=kinds,
                       next_round=next_round, pending_action=None,
                       interruption=interruption, interruption_resolution=resolution,
                       stopped_because="", updated_at=now())
        atomic_json(session_path, session)
        document_warning = ""
        try:
            run_record.generate(
                research_root,
                title=f"{self.declaration.get('benchmark') or self.run_id} / derived")
        except (OSError, ValueError, TypeError) as exc:
            document_warning = f"RUN.md projection could not be refreshed: {type(exc).__name__}"
        return {"outcome": "candidate discarded",
                "because": "the main controller explicitly declined the candidate result; "
                          "the candidate metric remains unadopted and research may continue "
                          "from the next round",
                "round": round_index, "idea_label": idea_label,
                "resolution_id": resolution_id,
                "source_rollback": source_rollback,
                "measurement_ref": boundary.get("candidate_measurement", {}).get("ref"),
                **({"document_warning": document_warning} if document_warning else {}),
                "evidence_refs": [f"research/{self.run_id}/research_report.json",
                                  f"research/{self.run_id}/controller_session.json",
                                  f"research/{self.run_id}/ideas.json"]}

    def _step_recover_unscored_baseline(self) -> dict[str, Any]:
        """Continue a policy-artifact failure from its completed train receipt, not a new train."""
        eligibility = self._unscored_baseline_recovery_status()
        if not eligibility.get("available"):
            return {"outcome": "not attempted",
                    "because": str(eligibility.get("why") or
                                   "no recoverable unscored baseline is recorded")}
        research, _ = self._research_controller()
        result = research.recover_unscored_baseline()
        if result.get("ok"):
            return {"outcome": "done",
                    "because": "evaluated the receipt-bound baseline policy without "
                               "retraining; the original unscored measurement and selection "
                               "abstention remain linked in the run record",
                    "metric_value": result.get("metric_value"),
                    "training_reused": True,
                    "training_attempt_id": (result.get("recovery") or {}).get(
                        "training_attempt_id"),
                    "evaluation_attempt_id": (result.get("evaluate") or {}).get("attempt_id")}
        return {"outcome": "not recovered",
                "because": str(result.get("why") or "policy evidence did not revalidate"),
                "training_reused": bool((result.get("recovery") or {}).get(
                    "training_reused")),
                "evidence": result.get("recovery_state") or result.get("artifact_selection")}

    def _step_confirm_best(self) -> dict[str, Any]:
        """Use the requested held-out split on the frozen winner, without training."""
        if not self.base_settings.get("_confirm"):
            return {"outcome": "not attempted",
                    "because": "independent confirmation was not explicitly requested"}
        report_path = self.output / "research" / self.run_id / "research_report.json"
        try:
            report = read_json(report_path)
        except (OSError, ValueError, TypeError) as exc:
            return {"outcome": "not attempted",
                    "because": "no readable completed research report: "
                               f"{type(exc).__name__}: {exc}"}
        if (not isinstance(report, dict) or report.get("run_id") != self.run_id or
                Path(str(report.get("repo") or "")).expanduser().resolve() != self.repo):
            return {"outcome": "blocked",
                    "because": "research report identity does not match this run"}
        research, _ = self._research_controller()
        before = research.confirmation_state()
        if before.get("status") != "available_not_taken":
            return {"outcome": "not attempted",
                    "because": f"confirmation is {before.get('status')}: "
                               f"{before.get('why') or 'not available'}"}
        result = research.confirm()
        after = research.confirmation_state()
        report["confirmation"] = after
        report["confirmation_action"] = {
            "at": now(),
            "where": result.get("where"),
            "attempt_id": (result.get("evaluate") or {}).get("attempt_id"),
            "training_performed": False,
        }
        atomic_json(report_path, report)
        return {"outcome": "measurement_recorded" if result.get("ok") else
                "measurement_unavailable",
                "because": ("the authorized held-out evaluation produced a record; its "
                            "values are withheld from Scheduler and available in the final "
                            "run report" if result.get("ok") else
                            "the held-out evaluation produced no usable score; details remain "
                            "in the run evidence and the Scheduler does not receive held-out "
                            "values"),
                "training_performed": False}

    def _step_stop(self, *, because: str = "") -> dict[str, Any]:
        return {"outcome": "stopped", "because": because or "the model chose to stop"}

    def _shipped_checkpoint(self) -> str:
        """A checkpoint the benchmark ships, if this machine has one.

        The same fact `derive_a_command` needs to verify an evaluator: a stage whose command
        reads `i["checkpoint"]` cannot be run at all with nothing in that slot. It is read
        here as well because the *loop* needs it -- an evaluator with no trainer still
        measures something when there are weights to score.
        """
        try:
            from .survey import survey
            from .policy_provenance import source_snapshot_policy
            report = survey(self.repo)
            if (self.output / "workspace_snapshot.json").is_file():
                declared = ((self.declaration.get("assets") or {}).get("checkpoint") or {})
                candidates = [declared.get("path"), *(
                    row.get("path") for row in report.get("model_artifacts") or []
                    if isinstance(row, dict))]
                for value in candidates:
                    if not value:
                        continue
                    classified = source_snapshot_policy(
                        Path(str(value)), repo=self.repo, output=self.output)
                    if classified.get("status") == "source_snapshot":
                        return str(classified["path"])
                return ""
            found = execution_derive.checkpoint_for_verification(
                report, declaration=self.declaration)
        except Exception:                                            # noqa: BLE001
            return ""
        return str(found.get("path") or "")

    def _declared_task(self) -> str:
        """The task the declaration names, read rather than written here."""
        explicit = self.base_settings.get("task") or self.base_settings.get("task_name")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        for axes in (self.declaration.get("optimization_space") or {}).values():
            for axis in axes or []:
                if not isinstance(axis, dict):
                    continue
                name = str(axis.get("name", "")).lower()
                if (name.endswith("benchmark_name") or name in ("task_name", "task")) \
                        and axis.get("default"):
                    return str(axis["default"])
        return ""

    def has_run_out_of_its_own_records(self, name: str) -> bool:
        """Whether a step's product is already on disk, for the steps not covered by a guard.

        A reader asking "why is this taking an hour" should be able to see the answer without
        the run having to end first, and the records are where the answer is.
        """
        return (self.output / name).is_file()

    def _say(self, text: str) -> None:
        """Print as it happens, because the alternative is an hour of silence.

        A preparation takes as long as it takes: a build solves a conda environment, a
        derivation runs a benchmark forty times. Until this existed the whole thing printed
        one JSON document when it finished, so a reader watching a log saw nothing at all --
        and a run that is working and a run that is stuck are the same thing to look at. The
        RoboTwin derivation had per-attempt output and this did not, which is backwards: this
        is the part that decides whether anything can run at all.
        """
        print(f"  {text}", file=self.out, flush=True)

    def _note(self, stage: str, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            # The whole entry, not the summary. What a reader needs from a failed attempt is
            # the program's own output, and the line that explains a traceback is the last
            # one -- a fixed prefix of the message shows everything except the cause.
            self._attempts.append({**entry, "error": redact(str(entry.get("error") or ""))})
            self.steps.append({"step": f"derive:{stage}", "outcome": entry.get("status") or "",
                               "because": _tail_of_said(str(entry.get("error") or ""))})
            self._say(f"{stage}: {entry.get('status') or ''}"
                      + (f"\n    {redact(str(entry.get('error') or ''))[:400]}"
                         if entry.get("error") else ""))

    @staticmethod
    def _controller_skill_query(facts: dict[str, Any]) -> str:
        """Build a routing hint from state labels, never raw logs or hidden results.

        Keyword ranking may recommend entries, but the main Agent chooses what to read.
        """
        progress = facts.get("research_progress") or {}
        observation = progress.get("last_observation") or {}
        parts = [
            "master research controller planning failure diagnosis experiment design "
            "runtime validation resource budget intervention adaptation",
            str((facts.get("to_measure") or {}).get("ready") or ""),
            str(progress.get("status") or ""), str(observation.get("status") or ""),
            str(observation.get("label") or ""),
        ]
        parts.extend(str(name) for name in facts.get("available") or [])
        parts.extend(str(name) for name in (facts.get("surveyed_stages") or {}))
        parts.extend(str(name) for name in (facts.get("stages_the_checkout_has") or {}))
        for item in (facts.get("steps_taken") or [])[-4:]:
            if isinstance(item, dict):
                parts.extend(str(item.get(key) or "") for key in ("step", "outcome"))
        return " ".join(part for part in parts if part)

    def _controller_methods(self, facts: dict[str, Any]) -> dict[str, Any]:
        """Legacy chat path: keep bounded, advisory retrieval for non-agent clients."""
        return skills_reference(
            benchmark=str(self.declaration.get("benchmark") or "") or None,
            task=self._declared_task() or None,
            query=self._controller_skill_query(facts), max_selected=3,
            include_index=False)

    def _select_controller_methods(self, facts: dict[str, Any], *,
                                   decision_attempt_id: str,
                                   traces: list[dict[str, Any]]
                                   ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Let the main Agent choose IDs from a short catalog, then read only those bodies."""
        benchmark = str(self.declaration.get("benchmark") or "") or None
        task = self._declared_task() or None
        catalog = skills_catalog(benchmark=benchmark, task=task,
                                 query=self._controller_skill_query(facts))
        available = {str(row["id"]): row for row in catalog["entries"]}
        cache_path = self.output / "agent/controller_skill_selection.json"
        failure = (read_json(self.output / "environment.json").get("latest_failure") or {}) if (self.output / "environment.json").is_file() else {}
        cache_key = object_digest({"catalog": catalog["entries"], "task": task,
            "available": facts.get("available"), "plan": self.main_agent.get("plan"),
            "failure": {k: failure.get(k) for k in ("failure_kind", "excerpt")}})
        cached = read_json(cache_path) if cache_path.is_file() and not cache_path.is_symlink() else {}
        if cached.get("key") == cache_key:
            methods = read_selected_skills(cached["ids"], benchmark=benchmark, task=task)
            hashes = {r["id"]: r.get("body_sha256") for r in methods["selection"]}
            if hashes == cached.get("hashes"):
                for item in methods["selection"]:
                    identity = item["id"]
                    item.update(selected_by="main_agent", selection_reason=cached["reasons"].get(identity),
                        selection_reused="unchanged catalog/problem; prior Agent choice, not keyword decision")
                return methods, catalog, []
        # The selection turn needs enough state to identify the current uncertainty, not
        # every artifact preview that the eventual research decision will inspect.
        selection_context = {key: facts.get(key) for key in (
            "state_revision", "available", "to_measure", "research_progress",
            "steps_taken", "requested_task")}
        selection_context["steps_taken"] = (selection_context.get("steps_taken") or [])[-4:]
        user = json.dumps({"decision_context": _controller_model_value(
                               selection_context, local_roots=(self.repo, self.output)),
                           "skill_catalog": catalog}, ensure_ascii=False, default=str)
        request: dict[str, Any] = {
            "max_tokens": 700,
            "timeout": max(1, min(240, int(self.budget.remaining())))
                       if self.budget else 240,
            "thinking": "disabled", "include_research_context": False,
            "read_only": True, "auto_skills": False,
        }
        if callable(getattr(self.client, "as_role", None)):
            request["decision_attempt_id"] = decision_attempt_id
        from .agent_client import role_scope
        from .execution_derive import _object
        with role_scope(self.client, "scheduler"):
            content, metadata = self.client.chat_with_metadata(
                SKILL_SELECTION_SYSTEM, user, **request)
        trace = _coding_agent_turn_trace(metadata, decision_attempt_id)
        if trace is not None:
            traces.append(trace)
        problems: list[str] = []
        try:
            answer = _object(content)
        except (TypeError, ValueError):
            answer = {}
            problems.append("skill selection response was not JSON; no body was loaded")
        raw = answer.get("skill_reads")
        if not isinstance(raw, list):
            raw = []
            problems.append("skill_reads was not a list; no body was loaded")
        if len(raw) > 3:
            raw = []
            problems.append("skill_reads exceeded three; no body was loaded")
        ids: list[str] = []
        reasons: dict[str, str] = {}
        for item in raw:
            identity = str(item.get("id") or "") if isinstance(item, dict) else ""
            why = str(item.get("why") or "").strip()[:400] if isinstance(item, dict) else ""
            if identity not in available or not why or identity in reasons:
                problems.append("invalid, duplicate, or unexplained skill selection was ignored")
                continue
            ids.append(identity)
            reasons[identity] = why
        methods = read_selected_skills(ids, benchmark=benchmark, task=task)
        for item in methods["selection"]:
            identity = str(item["id"])
            item.update(selected_by="main_agent", selection_reason=reasons[identity],
                        recommended_by_keyword=available[identity]["recommended_by_keyword"])
        if len(methods["selection"]) != len(ids):
            problems.append("a selected skill body could not be read; it was withheld")
        if not problems:
            if cache_path.is_symlink():
                raise ValueError("controller skill cache is a symlink")
            atomic_json(cache_path, {"key": cache_key, "ids": ids, "reasons": reasons,
                "hashes": {r["id"]: r.get("body_sha256") for r in methods["selection"]}})
        return methods, catalog, problems

    @staticmethod
    def _review_controller_methods(reference: dict[str, Any],
                                  answer: dict[str, Any]) -> tuple[list[dict[str, Any]],
                                                                  list[str]]:
        """Keep skill applicability as a model judgment, distinct from tool verification."""
        selected = reference.get("selection") or []
        eligible = {str(item.get("id")) for item in selected if isinstance(item, dict) and
                    str(item.get("id")) != f"autosimsota.{GUIDE}"}
        reviews: dict[str, dict[str, Any]] = {}
        faults: list[str] = []
        raw = answer.get("method_review")
        if raw is not None and not isinstance(raw, list):
            faults.append("method_review is not a list")
            raw = []
        for item in raw or []:
            if not isinstance(item, dict):
                faults.append("method_review contains a non-object entry")
                continue
            identity = str(item.get("id") or "")
            if identity not in eligible:
                faults.append("method_review names an unselected skill")
                continue
            if identity in reviews:
                faults.append(f"method_review repeats {identity}")
                continue
            verdict = str(item.get("verdict") or "")
            why = str(item.get("why") or "").strip()[:600]
            refs = item.get("evidence_refs")
            refs = ([str(value)[:300] for value in refs if isinstance(value, str)][:20]
                    if isinstance(refs, list) else [])
            if (verdict not in {"use", "decline", "insufficient_evidence"} or not why or
                    (verdict != "insufficient_evidence" and not refs)):
                faults.append(f"method_review is incomplete for {identity}")
                continue
            reviews[identity] = {
                "verdict": verdict, "why": why, "evidence_refs": refs,
                "verdict_source": "model",
                "evidence_validation": "references_not_independently_verified",
            }
        merged: list[dict[str, Any]] = []
        missing: list[str] = []
        for item in selected:
            if not isinstance(item, dict):
                continue
            identity = str(item.get("id") or "")
            if identity == f"autosimsota.{GUIDE}":
                merged.append({**item, "verdict": "orientation_only",
                               "verdict_source": "kernel_default"})
                continue
            review = reviews.get(identity)
            if review is None:
                missing.append(identity)
                merged.append({**item, "verdict": "not_assessed",
                               "verdict_source": "kernel_default",
                               "evidence_validation": "no_applicability_verdict"})
            else:
                merged.append({**item, **review})
        return merged, faults + [f"no applicability verdict for {identity}"
                                 for identity in missing]

    # -- the loop ------------------------------------------------------------------------

    def choose(self) -> dict[str, Any]:
        """Ask what to do next, given the state.

        Native execution stays in validated operations. The main Agent may also maintain
        a global plan and assign open-ended tool-using investigations instead of trying
        to force every uncertainty into a fixed provisioning operation.
        """
        from .initialization_audit import preserve
        preserve(self.output, self.repo)
        new_reports = self._collect_agent_tasks()
        if not new_reports and self.last_action.get("step") == "wait_for_jobs" and self.last_action.get("outcome") == "waiting":
            from .native_jobs import active_jobs
            if any(row["status"] in {"queued", "running"} for row in self._native_job_view()) or any(row["status"] in {"submitted", "running"} for row in self._agent_task_view()):
                return {"do": "wait_for_jobs", "arguments": {}, "why": "wait for background completion events", "decision_by": "tool"}
        if self.state_persistence_error:
            return {"do": "stop", "why": "durable run state is unavailable: "
                    f"{self.state_persistence_error}", "arguments": {},
                    "decision_by": "tool"}
        if self.recovery_after_interruption:
            return {"do": "reconcile_interrupted_action",
                    "why": "a prior action has unknown outcome; reconcile its receipt and "
                           "verified process identity before any new attempt",
                    "arguments": {}, "decision_by": "tool"}
        if self.keep_only:
            progress = self.state().get("research_progress")
            available = self._available_operations(progress)
            # A completed research report can still have an explicitly recoverable, unscored
            # baseline. Resolve that receipt-bound measurement before treating the report as
            # terminal; the ordinary run loop is intentionally unavailable in this state.
            if "recover_unscored_baseline" in available:
                return {"do": "recover_unscored_baseline",
                        "why": "reuse the verified completed train receipt and evaluate the "
                              "existing baseline policy without retraining",
                        "arguments": {}, "decision_by": "tool"}
            if progress and progress.get("status") == "completed":
                return {"do": "confirm_best" if "confirm_best" in available else "stop",
                        "why": "research is already recorded; do not repeat its training "
                        "or evaluation" if "confirm_best" not in available else
                        "the recorded best can now receive the explicitly requested "
                        "confirmation evaluation", "arguments": {}, "decision_by": "tool"}
            ready = self._research_loop_ready()
            can_resume = "run_the_loop" in available
            return {"do": "run_the_loop" if ready and can_resume else "stop",
                    "why": ("resume the next bounded action from verified records" if
                            ready and can_resume else
                            f"research status {progress.get('status') if progress else 'new'} "
                            "is not safely runnable, or kept records lack a declaration, "
                            "runnable stage, or interpreter"),
                    "arguments": {}, "decision_by": "tool"}
        facts = self.state()
        self._publish_main_context(facts)
        base_revision = int(facts.get("state_revision") or 0)
        permitted = set(facts.get("available") or [])
        operations = "\n".join(f"* `{name}` -- {text}" for name, text in OPERATIONS.items()
                               if name in permitted)
        local_roots = (self.repo, self.output)
        controller_facts = _controller_model_value(facts, local_roots=local_roots)
        from .execution_derive import _object
        requested = "stop"
        row: dict[str, Any] = {}
        rejection = ""
        decision_attempt_id = uuid.uuid4().hex
        coding_agent_traces: list[dict[str, Any]] = []
        selection_issues: list[str] = []
        if getattr(self.client, "supports_main_agent", False):
            methods, _catalog, selection_issues = self._select_controller_methods(
                facts, decision_attempt_id=decision_attempt_id,
                traces=coding_agent_traces)
        else:
            methods = self._controller_methods(facts)
        controller_methods = _controller_model_value(methods, local_roots=local_roots)
        payload: Any = json.dumps({**controller_facts,
                                   "method_library": controller_methods},
                                  ensure_ascii=False, default=str)

        def runtime_evidence_refs() -> list[str]:
            if not coding_agent_traces:
                return []
            refs = [f"agent/events.jsonl#decision_attempt_id={decision_attempt_id}"]
            for trace in coding_agent_traces:
                turn_id = str(trace["turn_id"])
                refs.extend((f"agent/events.jsonl#turn_id={turn_id}",
                             str(trace["process_ref"])))
            return list(dict.fromkeys(refs))

        for repair in range(2):
            from .agent_client import role_scope
            with role_scope(self.client, "scheduler"):
                request = {
                    "max_tokens": 4000 if getattr(self.client, "supports_main_agent", False)
                                  else 1500,
                    "timeout": max(1, min(240, int(self.budget.remaining())))
                    if self.budget else 240,
                    "thinking": "disabled",
                }
                if callable(getattr(self.client, "as_role", None)):
                    request["decision_attempt_id"] = decision_attempt_id
                if getattr(self.client, "supports_main_agent", False):
                    request["include_research_context"] = False
                    request["read_only"] = True
                    request["auto_skills"] = False
                content, metadata = self.client.chat_with_metadata(
                    STEP_SYSTEM.format(operations=operations), payload, **request)
                trace = _coding_agent_turn_trace(metadata, decision_attempt_id)
                if trace is not None:
                    coding_agent_traces.append(trace)
            try:
                row = _object(content)
            except (TypeError, ValueError) as exc:
                # A malformed model response is a recoverable decision-protocol failure,
                # not a repository failure and not an executed action. Feed the same
                # persisted state back once with a precise correction request. Do not echo
                # arbitrary model text into another prompt: the parse category is enough to
                # repair the output shape and avoids re-projecting any accidental secrets.
                rejection = ("the response did not contain one parseable JSON object "
                             f"({type(exc).__name__}: {str(exc)[:120]}); no action ran")
                if repair == 0:
                    payload = json.dumps({
                        **controller_facts,
                        "method_library": controller_methods,
                        "rejected_decision": {"parse_error": type(exc).__name__},
                        "rejection": rejection,
                        "permitted_actions": sorted(permitted),
                        "instruction": "Repair the response format only: return exactly one "
                                      "JSON object matching the requested decision schema, "
                                      f"using state_revision {base_revision} and only a "
                                      "permitted action. No action has run yet."
                    }, ensure_ascii=False, default=str)
                continue
            requested = str(row.get("do") or "stop")
            if row.get("refresh_skill_selection") is True:
                cache = self.output / "agent/controller_skill_selection.json"
                if cache.is_file() and not cache.is_symlink():
                    atomic_json(cache, {})
            proposed_revision = row.get("state_revision")
            stale_revision = (proposed_revision is not None and
                              (isinstance(proposed_revision, bool) or
                               not isinstance(proposed_revision, int) or
                               proposed_revision != base_revision))
            if requested in permitted and not stale_revision:
                latest = self.state_store.load()
                latest_revision = int((latest or {}).get("decision_revision") or 0)
                if latest_revision != base_revision:
                    return {"do": "stop",
                            "why": f"state changed during decision: read revision "
                                   f"{base_revision}, current revision is {latest_revision}; "
                                   "discard this decision and re-read state",
                            "arguments": {}, "decision_by": "tool",
                            "decision_failure": {
                                "kind": "stale_state", "state_revision": base_revision,
                                "current_revision": latest_revision,
                                "coding_agent_trace": coding_agent_traces,
                                "runtime_evidence_refs": runtime_evidence_refs()}}
                evidence = row.get("evidence_refs")
                evidence = ([str(item)[:300] for item in evidence if isinstance(item, str)][:20]
                            if isinstance(evidence, list) else [])
                expected = row.get("expected_outputs")
                expected = ([str(item)[:300] for item in expected if isinstance(item, str)][:20]
                            if isinstance(expected, list) else [])
                postconditions = row.get("postconditions")
                postconditions = ([str(item)[:400] for item in postconditions
                                   if isinstance(item, str)][:20]
                                  if isinstance(postconditions, list) else [])
                resource_limits = row.get("resource_limits")
                resource_limits = (resource_limits if isinstance(resource_limits, dict) else {})
                missing = [name for name in (
                    "state_revision", "question", "hypothesis", "evidence_refs",
                    "resource_limits", "expected_outputs", "postconditions", "stop_condition")
                    if name not in row]
                for name in ("question", "hypothesis", "stop_condition"):
                    if (not isinstance(row.get(name), str) or not row[name].strip()) and \
                            name not in missing:
                        missing.append(name)
                for name, expected_type in (("evidence_refs", list),
                                            ("resource_limits", dict),
                                            ("expected_outputs", list),
                                            ("postconditions", list)):
                    if not isinstance(row.get(name), expected_type) and name not in missing:
                        missing.append(name)
                method_selection, method_review_missing = \
                    self._review_controller_methods(methods, row)
                return {"do": requested, "why": str(row.get("why") or "")[:600],
                        "arguments": (row.get("arguments")
                                      if isinstance(row.get("arguments"), dict) else {}),
                        "decision_by": "model",
                        "research_decision": {
                            "state_revision": base_revision,
                            "question": str(row.get("question") or "")[:600],
                            "hypothesis": str(row.get("hypothesis") or "")[:800],
                            "input_evidence": evidence,
                            "resource_limits": resource_limits,
                            "expected_outputs": expected,
                            "postconditions": postconditions,
                            "stop_condition": str(row.get("stop_condition") or "")[:600],
                            "contract_missing": sorted(set(missing)),
                            "method_review_missing": method_review_missing,
                            "skill_selection_issues": selection_issues,
                            "coding_agent_trace": coding_agent_traces,
                            "runtime_evidence_refs": runtime_evidence_refs(),
                        },
                        "method_selection": method_selection}
            rejection = (f"state_revision must equal {base_revision}, got "
                         f"{proposed_revision!r}" if stale_revision else
                         f"{requested!r} is not currently permitted")
            if repair == 0:
                safe_rejected_decision = _controller_model_value(
                    row, local_roots=local_roots)
                payload = json.dumps({
                    **controller_facts,
                    "method_library": controller_methods,
                    "rejected_decision": safe_rejected_decision,
                    "rejection": rejection + "; no action ran",
                    "permitted_actions": sorted(permitted),
                    "instruction": "Choose again using only one of permitted_actions and echo "
                                   "the exact state_revision."
                }, ensure_ascii=False, default=str)
        return {"do": "stop",
                "why": f"controller decision was rejected twice ({rejection}); "
                       f"currently permitted actions are {sorted(permitted)}. "
                       "No action was executed.",
                "arguments": {}, "decision_by": "tool",
                "decision_failure": {
                    "kind": "decision_rejected", "attempts": 2,
                    "reason": rejection[:300],
                    "coding_agent_trace": coding_agent_traces,
                    "runtime_evidence_refs": runtime_evidence_refs()}}

    def _record_controller_decision(self, step: str, choice: dict[str, Any],
                                    arguments: dict[str, Any]) -> str:
        """Persist the outer controller's choice against the state revision it read."""
        if self.decisions is None:
            raise ValueError("controller decision log is unavailable")
        maker = str(choice.get("decision_by") or "model")
        if maker not in {"model", "tool", "human"}:
            maker = "tool"
        activity = f"preparation:{step}"
        if arguments:
            activity += " " + json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        used = ["run_state.json#phases.preparation.state"]
        if (self.output / "feasibility.json").is_file():
            used.append("feasibility.json")
        used.extend(_safe_agent_trace_refs(choice.get("runtime_evidence_refs")))
        method_selection = [dict(item) for item in choice.get("method_selection") or []
                            if isinstance(item, dict)]
        used.extend(
            "skill:" + str(item.get("id") or "") + "@" +
            str(item.get("version") or "") + "#" +
            str(item.get("body_sha256") or "")
            for item in method_selection if item.get("verdict") == "use")
        contract = choice.get("research_decision") or {}
        if not isinstance(contract, dict):
            contract = {}
        used.extend(_safe_agent_trace_refs(
            contract.get("runtime_evidence_refs")))
        for trace in contract.get("coding_agent_trace") or []:
            if not isinstance(trace, dict):
                continue
            turn_id = str(trace.get("turn_id") or "")
            process_ref = str(trace.get("process_ref") or "")
            if (re.fullmatch(r"[0-9a-f]{32}", turn_id) and
                    process_ref == f"agent/processes/{turn_id}.json"):
                used.extend((f"agent/events.jsonl#turn_id={turn_id}", process_ref))
        used = list(dict.fromkeys(used))
        wall_remaining = self.budget.remaining() if self.budget else None
        return self.decisions.record(ResearchDecision(
            activity=activity, by=maker,
            agent=str(getattr(self.client, "model", "autosim.prepare") or ""),
            why=str(choice.get("why") or ""), used=used,
            state_revision=int(contract.get("state_revision", self.decision_revision)),
            action=step,
            question=str(contract.get("question") or "")[:600],
            hypothesis=str(contract.get("hypothesis") or "")[:800],
            input_evidence=list(contract.get("input_evidence") or []),
            resource_limits={
                "model_proposal": (contract.get("resource_limits")
                                   if isinstance(contract.get("resource_limits"), dict) else {}),
                "enforced": {"wall_seconds_remaining": wall_remaining,
                             "controller_actions_remaining": max(
                                 0, self._step_budget - len(self.steps))}},
            expected_outputs=list(contract.get("expected_outputs") or []),
            postconditions=list(contract.get("postconditions") or []),
            stop_condition=str(contract.get("stop_condition") or "")[:600],
            contract_missing=list(contract.get("contract_missing") or []),
            method_selection=method_selection,
            method_review_missing=list(contract.get("method_review_missing") or []),
            skill_selection_issues=list(contract.get("skill_selection_issues") or [])))

    def _resolve_controller_decision(self, decision_id: str,
                                     outcome: dict[str, Any]) -> None:
        """Close the decision only after its action result is known and recorded."""
        if not decision_id or self.decisions is None:
            return
        try:
            self.decisions.resolve(decision_id, outcome)
        except (OSError, ValueError, TypeError) as exc:
            self.state_persistence_error = redact(
                f"controller decision outcome could not be persisted: "
                f"{type(exc).__name__}: {exc}")[:400]

    def _refresh_document(self, *, status: str, current: str = "") -> None:
        """Show preparation progress from the moment a run starts."""
        last = self.steps[-1] if self.steps else {}
        monitor = self.monitor_observation
        latest_progress = (monitor.get("summary") if monitor else
                           last.get("because") or "run created")
        level = ("L2" if any(row.get("step") == "run_the_loop" and
                             row.get("verification_level") == "L2"
                             for row in self.steps) else "L1" if self.verified else
                 "L0" if self.execution else "unknown")
        source_repository = self._source_repository()
        lines = [f"# Research run: {source_repository.name}", "", f"Updated: {now()}",
                 "<!-- AUTOSIM_LIVE_START -->", f"Status: {status}",
                 f"Current action: {current or 'none'}", "<!-- AUTOSIM_LIVE_END -->",
                 f"Latest progress: {redact(str(latest_progress))[:300]}",
                 f"Preparation budget: {len([row for row in self.steps if not str(row.get('step', '')).startswith('derive:')])}/{self._step_budget} steps",
                 f"Highest verified level: {level}", "",
                 f"Repository identity: `{source_repository.name}`",
                 f"Source repository: `{source_repository}`",
                 f"Execution checkout: `{self.repo}`", "",
                 "[Feasibility record](feasibility.json)", ""]
        from .workspace_resources import resource_view
        resources = resource_view(self.repo)
        if resources:
            from html import escape
            lines.extend(["## Code isolation and resource access", "",
                          "[Workspace manifest](workspace_snapshot.json)", "",
                          "<pre>" + escape(json.dumps(resources, ensure_ascii=False,
                                                     indent=2)) + "</pre>", ""])
        if self.main_agent.get("plan") or self.main_agent.get("handoffs"):
            from html import escape
            # Render as text, not model-supplied Markdown/HTML links. The authoritative
            # measurements and media remain in the existing deterministic sections.
            memory_text = json.dumps(self.main_agent, ensure_ascii=False, indent=2)
            lines += ["## Main Agent working memory", "",
                      "Model-authored plan and handoffs; not verified execution results.", "",
                      "<details><summary>Global plan and recent investigations</summary>", "",
                      "<pre>" + escape(memory_text) + "</pre>", "", "</details>", ""]
        lines += ["## Preparation", ""]
        if self.budget:
            budget = self.budget.record()
            lines.insert(7, f"Wall budget: {budget['elapsed_wall_seconds']:.1f}/"
                            f"{budget['wall_seconds']:.1f}s; remaining "
                            f"{budget['remaining_wall_seconds']:.1f}s")
            if budget.get("budget_scope") == "task":
                lines += ["", "本研究任务独立计费：GPU 作业设备占用已结算 "
                          f"{budget['settled_gpu_seconds'] / 3600:.3f} / "
                          f"{budget['gpu_cap_seconds'] / 3600:g} GPU 小时；在途预留 "
                          f"{budget['reserved_gpu_seconds'] / 3600:.3f} GPU 小时。"
                          "预留不是实际消耗；LLM 等待不计入 GPU 账本。"
                          "墙钟期限独立计算，暂停期间期限继续流逝。", ""]
        if self.steps:
            lines += ["| Step | Outcome | Evidence |", "| --- | --- | --- |"]
            for entry in self.steps:
                reason = redact(str(entry.get("because") or "")).replace("|", "\\|")
                monitor_row = entry.get("monitor")
                if isinstance(monitor_row, dict):
                    progress = redact(str(monitor_row.get("progress") or "uncertain"))[:40]
                    summary = redact(str(monitor_row.get("summary") or ""))[:180]
                    guidance = redact(str(monitor_row.get("guidance") or ""))[:220]
                    reason += (f"; AgentMonitor `{progress}`: {summary}"
                               + (f" Guidance: {guidance}" if guidance else ""))
                    reason = reason.replace("|", "\\|")
                fix_row = entry.get("fix")
                if isinstance(fix_row, dict):
                    fix_status = run_record._markdown_text(
                        fix_row.get("status") or "unknown", limit=40)
                    fix_assessment = run_record._markdown_text(
                        fix_row.get("assessment") or "uncertain", limit=40)
                    fix_summary = run_record._markdown_text(
                        fix_row.get("summary") or "", limit=220)
                    reason += (f"; AgentFix `{fix_status}/{fix_assessment}`: "
                               f"{fix_summary}")
                    reason = reason.replace("|", "\\|")
                lines.append(f"| {entry.get('step', '')} | {entry.get('outcome', '')} | "
                             f"{reason[:650]}"
                             f"{' [receipt](' + str(entry['receipt_ref']) + ')' if entry.get('receipt_ref') else ''} |")
        else:
            lines.append("No step has completed yet.")
        if last.get("receipt_ref"):
            lines.extend(["", f"Latest action receipt: [{last['receipt_ref']}]"
                          f"({last['receipt_ref']}) — SHA256 "
                          f"`{last.get('receipt_sha256') or ''}`."])
        if self.recovery_after_interruption:
            action = self.recovery_after_interruption.get("action") or {}
            lines.extend(["", "## Recovery required", "",
                          f"Previous action `{action.get('step', 'unknown')}` has no "
                          "verified outcome. Reconcile its receipts/output before repeating it."])
        if monitor:
            lines.extend(["", "## AgentMonitor", "",
                          f"Status: `{redact(str(monitor.get('status') or 'unknown'))[:40]}` · "
                          f"progress: `{redact(str(monitor.get('progress') or 'uncertain'))[:40]}`",
                          redact(str(monitor.get("summary") or ""))[:700]])
            guidance = redact(str(monitor.get("guidance") or ""))[:900]
            if guidance:
                lines.extend(["", f"Scheduler guidance (advisory): {guidance}"])
            refs = [redact(str(ref))[:300] for ref in
                    (monitor.get("evidence_refs") or [])[:20] if isinstance(ref, str)]
            if refs:
                lines.extend(["", "Evidence references (model citations; not independently "
                              "verified): " + ", ".join(f"`{ref}`" for ref in refs) + "."])
            lines.extend(["", "AgentMonitor is read-only; the Scheduler retains action authority."])
        fix = self.fix_observation
        if fix:
            lines.extend(["", "## AgentFix", "",
                          f"Status: `{redact(str(fix.get('status') or 'unknown'))[:40]}` · "
                          f"assessment: `{redact(str(fix.get('assessment') or 'uncertain'))[:40]}`",
                          run_record._markdown_text(fix.get("summary") or "", limit=700),
                          "The repair report is model-generated and is not independent proof "
                          "that the original action now succeeds; Scheduler must verify it."])
            changes = [redact(str(value))[:260] for value in
                       (fix.get("changes") or [])[:12] if isinstance(value, str)]
            if changes:
                lines.extend(["", "Reported changes:"])
                lines.extend(f"- {run_record._markdown_text(value, limit=260)}"
                             for value in changes)
            refs = [redact(str(ref))[:300] for ref in
                    (fix.get("evidence_refs") or [])[:20] if isinstance(ref, str)]
            refs.extend(_safe_agent_trace_refs(fix.get("runtime_evidence_refs")))
            if refs:
                lines.extend(["", "Evidence references (model citations; not independently "
                              "verified): " + ", ".join(f"`{ref}`" for ref in
                                                         (run_record._markdown_text(ref, limit=300)
                                                          for ref in dict.fromkeys(refs))) + "."])
        evaluation = self._load_supervisor_verdict()
        if evaluation:
            lines.extend(["", "## AgentSupervisor final review", "",
                          f"Verdict: `{evaluation.get('verdict') or 'uncertain'}`",
                          redact(str(evaluation.get("summary") or ""))[:1000],
                          "Scope: evidence validity only. This is not a claim of improvement "
                          "or SOTA; confirmation-set values are not included in the review."])
            refs = [str(ref)[:120] for ref in evaluation.get("evidence_refs") or []
                    if isinstance(ref, str)]
            if refs:
                lines.extend(["", "Evidence IDs: " + ", ".join(f"`{ref}`" for ref in refs)])
            lines.extend(["", f"Review record: `supervisor_reviews/"
                          f"{evaluation.get('packet_sha256')}.json` · SHA256 "
                          f"`{evaluation.get('review_sha256')}`."])
        if self.decisions is not None and self.decisions.path.is_file():
            summary = self.decisions.summary()
            lines.extend(["", f"[Controller decisions](decisions.json) — "
                          f"{summary['decisions']} recorded, "
                          f"{summary['unresolved']} unresolved."])
        lines += ["", "<!-- AUTOSIM_RESEARCH_START -->"]
        lines.extend(run_record._research_overview(self.output, self.run_id,
                                                   status=status, current_action=current))
        lines.append("<!-- AUTOSIM_RESEARCH_END -->")
        if (self.output / "benchmark_bundle" / "manifest.json").is_file():
            lines += ["", "[Benchmark bundle](benchmark_bundle/manifest.json)"]
        lines += ["", "## Current boundary", ""]
        if status != "completed":
            lines.append("The run has not established an independently verified improvement.")
        else:
            lines.append("See the experiment record for the measurement and its limits.")
        content = "\n".join(lines) + "\n"
        try:
            from . import recorder
            view = recorder.read(self.output, "report/view.json")
            context = {"actions": [{key: row.get(key) for key in (
                "step", "outcome", "because", "why", "receipt_ref")}
                for row in self.steps[-8:] if row.get("step") != "confirm_best"],
                "plan": self.main_agent.get("plan") or {},
                "native_jobs": self._native_job_view(), "agent_tasks": self._agent_task_view()}
            remaining = self.budget.remaining() if self.budget else 60
            writer = self.client if remaining >= 10 and self.steps else None
            if status == "infrastructure_blocked":
                writer = None
            presentation = recorder.refresh(self.output, view, context=context,
                                             client=writer, timeout=min(60, remaining))
            before, live_tail = content.split(run_record._LIVE_START, 1)
            live, rest = live_tail.split(run_record._LIVE_END, 1)
            content = (f"# {source_repository.name} 研究进展\n\n" + run_record._LIVE_START + live +
                       run_record._LIVE_END + "\n\n" + presentation +
                       "\n\n<details><summary>技术详情与完整证据</summary>\n\n" + rest + "\n</details>\n")
        except Exception as exc:
            atomic_json(self.output / "report" / "presentation_error.json", {
                "at": now(), "error": type(exc).__name__})
        run_record.publish_document(self.output / "RUN.md", content)

    def _with_heartbeat(self, action: Any, *, current: str) -> Any:
        """Keep the Markdown timestamp live while an LLM call or stage blocks."""
        stopped = threading.Event()

        def refresh() -> None:
            while not stopped.wait(45):
                try:
                    run_record.refresh_live_status(
                        self.output,
                        live_root=self.output / "research" / self.run_id,
                        fallback_status="running", fallback_current=current)
                except OSError:
                    pass

        worker = threading.Thread(target=refresh, name="autosim-run-document", daemon=True)
        worker.start()
        try:
            return action()
        finally:
            stopped.set()
            worker.join(timeout=1)

    def _run_cycle(self, *, max_steps: int = 8) -> dict[str, Any]:
        """Take one bounded Scheduler segment; the outer Monitor may resume it.

        A step that fails is recorded and the loop goes on, because the next decision is made
        from the state and the state now includes the failure. The budget is a cost bound and
        not a judgement. Reaching this segment's action cap is a resumable yield, not evidence
        that the research goal has completed or that the repository is infeasible.
        """
        steps_before_cycle = len(self.steps)
        self._say(f"preparing {self.repo}")
        self._step_budget = max_steps
        self.output.mkdir(parents=True, exist_ok=True)
        if self.wall_seconds is not None or (self.output / "budget.json").is_file():
            try:
                self.budget = RunBudget(self.output, wall_seconds=self.wall_seconds)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                why = redact(f"budget record unusable: {type(exc).__name__}: {exc}")[:500]
                self.steps.append({"step": "budget", "outcome": "raised", "because": why})
                report = {"schema_version": 1, "created_at": now(),
                          "run_id": self.run_id, "repository": str(self.repo),
                          "status": "internal_error",
                          "verified_level": "unknown", "steps": self.steps}
                atomic_json(self.output / f"preparation_{self.run_id}.json", report)
                atomic_json(self.output / "feasibility.json", {
                    "schema_version": 1, "repository": str(self.repo),
                    "status": "internal_error", "last_verified_level": "unknown",
                    "blocking_conditions": [{"requirement": "valid budget record",
                                             "observed": why,
                                             "evidence_refs": ["budget.json"]}],
                    "conclusion_scope": "this run and this environment"})
                self.current_action = {}
                self.last_action = {"step": "budget", "outcome": "raised",
                                    "because": why, "finished_at": now()}
                self._persist_run_state(status="internal_error")
                self._refresh_document(status="internal_error", current="budget validation")
                return report
            if not isinstance(self.client, BudgetedClient):
                self.client = BudgetedClient(self.client, self.budget)
        if not (self.output / "feasibility.json").is_file():
            atomic_json(self.output / "feasibility.json", {
                "schema_version": 1, "repository": str(self.repo), "status": "running",
                "last_verified_level": "unknown",
                "conclusion_scope": "this run and this environment",
                "blocking_conditions": []})
        self._persist_run_state(status="running")
        if self.state_persistence_error:
            return self._state_failure_report()
        initial_action = ("reconcile interrupted action" if self.recovery_after_interruption
                          else "initial investigation")
        self._refresh_document(status="running", current=initial_action)
        held = self.state()
        self._say("starting from: "
                  f"{len(held['declarations_on_file']['usable'])} declaration(s), "
                  f"interpreter {'set' if held['environment']['interpreter'] else 'none'}, "
                  f"stages {held['stages_the_checkout_has'] or 'unread'}")
        for _ in range(max_steps):
            if (self.budget and self.budget.remaining() <= 0 and
                    not self.recovery_after_interruption):
                self.steps.append({"step": "budget", "outcome": "exhausted",
                                   "because": "the run's wall-clock budget expired"})
                self.last_action = dict(self.steps[-1])
                self._persist_run_state(status="budget_exhausted")
                break
            self.current_action = {"step": "choose", "status": "running",
                                   "started_at": now()}
            self._persist_run_state(status="running")
            if self.state_persistence_error:
                return self._state_failure_report()
            # The controller should not have to wait for the 45-second heartbeat to see
            # that its decision request is now the active action.
            self._refresh_document(status="running", current="choose")
            try:
                choice = self._with_heartbeat(self.choose, current="choose next action")
            except Exception as exc:                                 # noqa: BLE001
                because = redact(f"{type(exc).__name__}: {exc}")[:400]
                agent_status = str(getattr(exc, "status", "") or "")[:80]
                failure_category = str(getattr(exc, "failure_category", "") or "")[:120]
                infrastructure_blocked = agent_status == "infrastructure_blocked"
                from .agent_client import run_model_budget_exhausted
                is_model_budget = run_model_budget_exhausted(exc)
                outcome = "exhausted" if is_model_budget else "could not be asked"
                failure = {"step": "choose", "outcome": outcome, "because": because}
                runtime_failure = getattr(exc, "runtime_failure", {}) or {}
                if runtime_failure:
                    failure.update({"evidence_id": runtime_failure.get("evidence_id"),
                        "evidence_ref": runtime_failure.get("evidence_ref"),
                        "error_detail": runtime_failure.get("message"),
                        "failure_fingerprint": runtime_failure.get("fingerprint"),
                        "unsafe_entries": runtime_failure.get("entries", [])})
                decision_attempt_id = str(
                    getattr(exc, "decision_attempt_id", "") or "")
                if re.fullmatch(r"[0-9a-f]{32}", decision_attempt_id):
                    runtime_refs = ([runtime_failure["evidence_ref"]] if runtime_failure else [
                        f"agent/events.jsonl#decision_attempt_id={decision_attempt_id}"])
                    turn_id = str(getattr(exc, "turn_id", "") or "")
                    process_ref = str(getattr(exc, "process_ref", "") or "")
                    if (re.fullmatch(r"[0-9a-f]{32}", turn_id) and
                            process_ref == f"agent/processes/{turn_id}.json"):
                        runtime_refs.extend((f"agent/events.jsonl#turn_id={turn_id}",
                                             process_ref))
                    failure["runtime_evidence_refs"] = runtime_refs
                if agent_status:
                    failure["agent_status"] = agent_status
                if failure_category:
                    failure["failure_category"] = failure_category
                run_budget = getattr(exc, "run_budget", None)
                if isinstance(run_budget, dict):
                    failure["run_budget"] = {
                        key: run_budget[key] for key in (
                            "run_id", "limit_usd", "spent_usd", "reserved_usd",
                            "remaining_usd", "unknown_entries", "entry_count")
                        if key in run_budget}
                self.steps.append(failure)
                self.current_action = {}
                self.last_action = {**failure, "finished_at": now()}
                if infrastructure_blocked:
                    # Every role uses the same runtime. Asking Monitor/Fix to repair a
                    # runtime that cannot start only repeats the same host failure.
                    self.last_action["outcome"] = "infrastructure_blocked"
                    self.steps[-1]["outcome"] = "infrastructure_blocked"
                    self._persist_run_state(status="infrastructure_blocked")
                    break
                if (not is_model_budget and not self.state_persistence_error and
                        callable(getattr(self.client, "as_role", None))):
                    if self._handoff_scheduler_decision_failure(failure=failure):
                        continue
                    if (failure.get("recovery") or {}).get("status") == "blocked":
                        failure["outcome"] = "infrastructure_blocked"
                        self.last_action = {**failure, "finished_at": now()}
                        self.current_action = {}
                        self._persist_run_state(status="infrastructure_blocked")
                        break
                    if self.state_persistence_error:
                        return self._state_failure_report()
                    if self.monitor_observation.get("status") == "budget_exhausted":
                        break
                why = ("the run-level model-cost budget is exhausted" if is_model_budget else
                       f"controller could not be asked: {because}")
                try:
                    decision_id = self._record_controller_decision(
                        "stop", {"decision_by": "tool", "why": why,
                                 "runtime_evidence_refs": failure.get(
                                     "runtime_evidence_refs", [])}, {})
                    self.last_decision = {"step": "stop", "decision_id": decision_id,
                                          "why": why,
                                          "arguments": {}, "at": now()}
                    self._resolve_controller_decision(decision_id, {
                        "step": "choose", "outcome": outcome,
                        "because": because})
                except (OSError, ValueError, TypeError) as log_exc:
                    self.state_persistence_error = redact(
                        f"controller decision could not be persisted: "
                        f"{type(log_exc).__name__}: {log_exc}")[:400]
                self._persist_run_state(status=("budget_exhausted" if is_model_budget else
                                                "running"))
                if self.state_persistence_error:
                    return self._state_failure_report()
                break
            decision_failure = choice.get("decision_failure")
            if (isinstance(decision_failure, dict) and
                    callable(getattr(self.client, "as_role", None))):
                failure = {
                    "step": "choose", "outcome": "decision_rejected",
                    "because": str(choice.get("why") or
                                   "Scheduler decision failed validation")[:500],
                    "failure_category": str(decision_failure.get("kind") or
                                             "decision_rejected")[:100],
                    "attempts": decision_failure.get("attempts", 1),
                }
                runtime_refs = _safe_agent_trace_refs(
                    decision_failure.get("runtime_evidence_refs"))
                if runtime_refs:
                    failure["runtime_evidence_refs"] = runtime_refs
                self.steps.append(failure)
                if self._handoff_scheduler_decision_failure(failure=failure):
                    continue
                if self.state_persistence_error:
                    return self._state_failure_report()
                if self.monitor_observation.get("status") == "budget_exhausted":
                    break
                # The structured decision still says `stop`; if Monitor is unavailable,
                # retain that fail-closed behavior rather than inventing a Scheduler action.
            step = choice["do"]
            arguments = {k: v for k, v in (choice["arguments"] or {}).items()
                         if v is not None and v != ""}
            try:
                decision_id = self._record_controller_decision(step, choice, arguments)
            except (OSError, ValueError, TypeError) as exc:
                self.state_persistence_error = redact(
                    f"controller decision could not be persisted: "
                    f"{type(exc).__name__}: {exc}")[:400]
                return self._state_failure_report()
            self.last_decision = {"step": step, "why": choice["why"],
                                  "arguments": arguments, "decision_id": decision_id,
                                  "at": now()}
            if isinstance(choice.get("research_decision"), dict):
                self.last_decision["research_decision"] = choice["research_decision"]
            if choice.get("method_selection"):
                self.last_decision["method_selection"] = choice["method_selection"]
            if step == "stop":
                self.steps.append({"step": "stop", "outcome": "stopped",
                                   "because": choice["why"]})
                self.current_action = {}
                self.last_action = {**self.last_decision, "outcome": "stopped",
                                    "finished_at": now()}
                self._resolve_controller_decision(decision_id, {
                    "step": "stop", "outcome": "stopped", "because": choice["why"]})
                self._persist_run_state(status="stopped")
                if self.state_persistence_error:
                    return self._state_failure_report()
                self._say(f"stop — {choice['why']}")
                self._refresh_document(status="stopped", current=choice["why"])
                break
            if (self.budget and self.budget.remaining() <= 0 and
                    step != "reconcile_interrupted_action"):
                self.steps.append({"step": "budget", "outcome": "exhausted",
                                   "because": "the wall-clock budget expired before the step"})
                self.current_action = {}
                self.last_action = {**self.last_decision, "outcome": "budget exhausted before "
                                   "action", "finished_at": now()}
                self._resolve_controller_decision(decision_id, {
                    "step": step, "outcome": "budget exhausted before action",
                    "because": "the wall-clock budget expired before the step"})
                self._persist_run_state(status="budget_exhausted")
                break
            self._say(f"{step}{' ' + str(arguments) if arguments else ''}"
                      f"\n    because {choice['why'][:200]}")
            self.current_action = {**self.last_decision, "status": "running",
                                   "started_at": now()}
            self._persist_run_state(status="running")
            if self.state_persistence_error:
                return self._state_failure_report()
            self._refresh_document(status="running", current=step)
            action_started_monotonic = time.monotonic()
            wall_remaining_before = self.budget.remaining() if self.budget else None
            process_dir = self.output / "processes"
            try:
                preexisting_process_ids = {
                    item.stem for item in process_dir.iterdir()
                    if item.suffix == ".json" and re.fullmatch(r"[0-9a-f]{32}", item.stem)
                } if process_dir.is_dir() and not process_dir.is_symlink() else set()
            except OSError:
                preexisting_process_ids = set()
            with observe_process_starts(self._observe_process_start):
                result = self._with_heartbeat(
                    lambda: self.do(step, **arguments), current=step)
            step_result = {"step": step, "outcome": result.get("outcome") or "",
                           "because": result.get("because") or "",
                           "verification_level": result.get("verification_level"),
                           "why": choice["why"][:300]}
            for key in ("attempt_ids", "evidence_refs", "training_attempt_id",
                        "evaluation_attempt_id", "resolution_id", "round",
                        "idea_label", "source_rollback", "measurement_ref",
                        "failure_category", "agent_status", "run_budget",
                        "delegated_role", "fix_handoff", "fix_attempt_id",
                        "replayed_step", "original_outcome", "evidence_id",
                        "candidate_failure", "task_id", "study_id", "screening_result", "screening"):
                value = result.get(key)
                if value:
                    step_result[key] = list(value) if isinstance(value, tuple) else value
            process_attempt_ids, process_evidence_refs = self._new_process_evidence(
                decision_id=str(decision_id), step=step, before=preexisting_process_ids)
            if process_attempt_ids:
                existing_ids = step_result.get("attempt_ids")
                existing_ids = (existing_ids if isinstance(existing_ids, list) else
                                [existing_ids] if isinstance(existing_ids, str) else [])
                step_result["attempt_ids"] = list(dict.fromkeys(
                    [str(item) for item in existing_ids if item] + process_attempt_ids))[:80]
            if process_evidence_refs:
                existing_refs = step_result.get("evidence_refs")
                existing_refs = (existing_refs if isinstance(existing_refs, list) else
                                 [existing_refs] if isinstance(existing_refs, str) else [])
                step_result["evidence_refs"] = list(dict.fromkeys(
                    [str(item) for item in existing_refs if item] + process_evidence_refs))[:80]
            if (step == "derive_a_command" and
                    isinstance(arguments.get("stage"), str) and arguments["stage"]):
                step_result["arguments"] = {"stage": arguments["stage"], **(
                    {"timeout_seconds": arguments["timeout_seconds"]}
                    if "timeout_seconds" in arguments else {})}
            elif (step == "run_the_loop" and
                  isinstance(arguments.get("idea_label"), str) and
                  arguments["idea_label"]):
                step_result["arguments"] = {"idea_label": arguments["idea_label"]}
            elif (step == "propose_research_idea" and
                  isinstance(arguments.get("idea"), dict)):
                step_result["arguments"] = {
                    "idea_label": str(arguments["idea"].get("label") or "")}
            elif (step == "discard_interrupted_candidate" and
                  isinstance(arguments.get("reason"), str)):
                step_result["arguments"] = {
                    "reason": redact(arguments["reason"])[:600]}
            contract = choice.get("research_decision")
            contract = contract if isinstance(contract, dict) else {}
            attempt_values: list[str] = []
            for key in ("attempt_ids", "training_attempt_id", "evaluation_attempt_id",
                        "resolution_id"):
                value = step_result.get(key)
                values = value if isinstance(value, list) else [value]
                attempt_values.extend(str(item)[:200] for item in values
                                      if isinstance(item, (str, int)) and str(item))
            evidence_values = step_result.get("evidence_refs")
            evidence_values = ([str(item)[:400] for item in evidence_values
                                if isinstance(item, str)][:40]
                               if isinstance(evidence_values, list) else [])
            elapsed = max(0.0, time.monotonic() - action_started_monotonic)
            action_id = str(decision_id)
            receipt_ref, receipt = write_action_receipt(
                self.output,
                ActionReceipt(
                    action_id=action_id, run_id=self.run_id,
                    repository=str(self.repo), action=step, decision_id=decision_id,
                    state_revision=int(contract.get("state_revision", self.state_revision)),
                    started_at=str(self.current_action.get("started_at") or ""),
                    finished_at=now(),
                    status=str(step_result.get("outcome") or "unknown")[:120],
                    reason=redact(str(step_result.get("because") or choice.get("why") or
                                      "action returned no rationale"))[:1000],
                    arguments=(step_result.get("arguments")
                               if isinstance(step_result.get("arguments"), dict) else {}),
                    evidence_refs=evidence_values,
                    child_attempt_ids=list(dict.fromkeys(attempt_values))[:80],
                    reported_postconditions=list(contract.get("postconditions") or [])[:20],
                    postcondition_verification="not_independently_verified",
                    verification_level=step_result.get("verification_level"),
                    resource_limits={
                        "model_proposal": (contract.get("resource_limits")
                                           if isinstance(contract.get("resource_limits"), dict)
                                           else {}),
                        "enforced": {
                            "wall_seconds_remaining_before": wall_remaining_before,
                            "controller_actions_remaining": max(
                                0, self._step_budget - len(self.steps)),
                        }},
                    costs={"wall_seconds": elapsed, "gpu_seconds": None,
                           "gpu_seconds_status": "not_separately_metered"},
                    role_handoff=({"to": step_result.get("delegated_role"),
                                   "trigger": "verified_environment_contradicted_by_native_stage",
                                   "failure": step_result.get("fix_handoff")}
                                  if step_result.get("delegated_role") == "fix" and
                                  isinstance(step_result.get("fix_handoff"), dict) else {})))
            step_result["receipt_ref"] = receipt_ref
            step_result["receipt_sha256"] = receipt["receipt_sha256"]
            self.steps.append(step_result)
            self.current_action = {}
            self.last_action = {**self.last_decision, "outcome": step_result["outcome"],
                                "because": step_result["because"],
                                "verification_level": step_result["verification_level"],
                                "receipt_ref": receipt_ref,
                                "receipt_sha256": receipt["receipt_sha256"],
                                "finished_at": now()}
            self._resolve_controller_decision(decision_id, {
                **step_result})
            self._persist_run_state(status="running")
            if self.state_persistence_error:
                return self._state_failure_report()
            self._say(f"→ {result.get('outcome') or ''}"
                      + (f" — {str(result.get('because') or '')[:300]}"
                         if result.get("because") else ""))
            self._refresh_document(status="running", current=f"{step}: {result.get('outcome')}")
            if (step_result["outcome"] in (_MONITORED_FAILURES | {"checkpoint failure"}) and
                    step_result["outcome"] != "exhausted"):
                observation = self._monitor_failed_action(
                    action_id=action_id,
                    action={**step_result, "receipt_ref": receipt_ref,
                            "receipt_sha256": receipt["receipt_sha256"]})
                if observation is not None:
                    step_result["monitor"] = {
                        "status": observation.get("status"),
                        "progress": observation.get("progress"),
                        "summary": observation.get("summary"),
                        "guidance": observation.get("guidance"),
                        "evidence_refs": observation.get("evidence_refs", []),
                    }
                    self.steps[-1] = step_result
                    self.last_action["monitor_status"] = observation.get("status")
                    self.last_action["monitor_progress"] = observation.get("progress")
                    self.last_action["monitor_summary"] = observation.get("summary")
                    self._persist_run_state(status=(
                        "budget_exhausted" if observation.get("status") == "budget_exhausted"
                        else "running"))
                    if self.state_persistence_error:
                        return self._state_failure_report()
                    self._refresh_document(
                        status=("budget_exhausted" if
                                observation.get("status") == "budget_exhausted" else "running"),
                        current="AgentMonitor assessment")
                    if observation.get("status") == "budget_exhausted":
                        break
            if (((step_result["outcome"] in _FIX_ASSISTANCE_OUTCOMES and
                  step in _FIX_ASSISTANCE_STEPS) or
                 (step == "run_the_loop" and step_result.get("candidate_failure"))) and
                    step_result.get("delegated_role") != "fix"):
                failed_step_index = len(self.steps) - 1
                fix_observation = self._with_heartbeat(
                    lambda: self._fix_failed_action(
                        action_id=action_id,
                        action={**step_result, "receipt_ref": receipt_ref,
                                "receipt_sha256": receipt["receipt_sha256"]}),
                    current="AgentFix diagnosis/repair")
                if fix_observation is not None:
                    step_result["fix"] = {
                        "status": fix_observation.get("status"),
                        "assessment": fix_observation.get("assessment"),
                        "summary": fix_observation.get("summary"),
                        "fix_attempt_id": fix_observation.get("fix_attempt_id"),
                        "runtime_evidence_refs": fix_observation.get(
                            "runtime_evidence_refs", []),
                    }
                    self.steps[failed_step_index] = step_result
                    self.last_action["fix_status"] = fix_observation.get("status")
                    self.last_action["fix_attempt_id"] = fix_observation.get(
                        "fix_attempt_id")
                    self._persist_run_state(status="running")
                    self._refresh_document(status="running",
                                           current="AgentFix result recorded")
                    if self.state_persistence_error:
                        return self._state_failure_report()
                    if fix_observation.get("status") == "budget_exhausted":
                        self._persist_run_state(status="budget_exhausted")
                        self._refresh_document(
                            status="budget_exhausted",
                            current="model budget exhausted during AgentFix")
                        break
            if step_result["outcome"] == "exhausted":
                # A depleted model/experiment budget cannot be repaired by asking another
                # role to reason. Close the run explicitly and do not launch more turns.
                self._persist_run_state(status="budget_exhausted")
                self._refresh_document(status="budget_exhausted",
                                       current="budget exhausted; no further action launched")
                break
            if step == "reconcile_interrupted_action":
                # Reconciliation establishes only that the local process is no longer
                # running; it deliberately leaves the attempt outcome unknown. Return control
                # after this one action so the operator can inspect its receipt/output before
                # any model decision can launch a duplicate external side effect.
                break
        report = {"schema_version": 1, "created_at": now(), "run_id": self.run_id,
                  "repository": str(self.repo),
                  "steps": self.steps, "state": self.state(),
                  "stages_with_commands": sorted(self.stages)}
        last = self.steps[-1] if self.steps else {}
        measured = any(row.get("step") == "run_the_loop" and row.get("outcome") == "done" and
                       row.get("verification_level") == "L2"
                       for row in self.steps)
        progress = report["state"].get("research_progress") or {}
        research_completed = progress.get("status") == "completed"
        confirmation_pending = (bool(self.base_settings.get("_confirm")) and
                                progress.get("confirmation", {}).get("status") ==
                                "available_not_taken")
        budget_exhausted = ((self.budget and self.budget.remaining() <= 0) or
                            last.get("outcome") == "exhausted" or
                            self.monitor_observation.get("status") == "budget_exhausted" or
                            self.fix_observation.get("status") == "budget_exhausted")
        actions_in_cycle = len(self.steps) - steps_before_cycle
        explicitly_stopped = (last.get("step") == "stop" or
                              last.get("outcome") == "stopped")
        interrupted_reconciled = last.get("step") == "reconcile_interrupted_action"
        if research_completed and measured and not confirmation_pending:
            report["status"] = "completed"
        elif budget_exhausted:
            report["status"] = "budget_exhausted"
        elif last.get("outcome") == "infrastructure_blocked":
            report["status"] = "infrastructure_blocked"
        elif last.get("outcome") == "raised":
            report["status"] = "internal_error"
        elif explicitly_stopped:
            report["status"] = "adaptation_unresolved"
        elif interrupted_reconciled:
            report["status"] = "paused"
        elif actions_in_cycle >= max_steps:
            report["status"] = "action_limit"
        else:
            report["status"] = "adaptation_unresolved"
        if report["status"] == "completed":
            # The Supervisor runs only after the trusted loop and score receipts are
            # complete. Its validity verdict is a separate axis: an unavailable reviewer
            # becomes `uncertain`, never an execution failure or a Scheduler-owned score.
            try:
                report["evaluation_verdict"] = self._review_final_study()
                report["state"] = self.state()
            except Exception as exc:  # noqa: BLE001 - keep an otherwise complete run inspectable.
                report["evaluation_verdict"] = {
                    "verdict": "uncertain",
                    "summary": "Independent review could not be persisted; "
                               f"the controller returned {type(exc).__name__}.",
                    "evidence_refs": [],
                }
                report["supervisor_review_error"] = type(exc).__name__
                report["state"] = self.state()
        report["actions_this_cycle"] = actions_in_cycle
        report["verified_level"] = ("L2" if measured else "L1" if self.verified
                                    else "L0" if self.execution else "unknown")
        atomic_json(self.output / f"preparation_{self.run_id}.json", report)
        blocking = ([] if measured or report["status"] in {"action_limit", "paused"} else [{
            "requirement": str(last.get("step") or "adaptation"),
            "observed": str(last.get("because") or "no valid evaluation was completed"),
            "evidence_refs": [f"preparation_{self.run_id}.json"],
            "alternatives_checked": [],
            "attempted_steps": [str(row.get("step")) for row in self.steps[:-1]
                                if row.get("step")],
            "untried_options": ["further repository investigation or additional budget"],
            "resume_when": "the observed condition is resolved and the stage is reverified",
        }])
        atomic_json(self.output / "feasibility.json", {
            "schema_version": 1, "repository": str(self.repo),
            "status": report["status"], "last_verified_level": report["verified_level"],
            "scientific_validity": (report.get("evaluation_verdict") or {}).get("verdict"),
            "conclusion_scope": "this run and this environment",
            "last_observation": last, "stages_with_commands": sorted(self.stages),
            "blocking_conditions": blocking,
            "still_available": (report["state"].get("available") or [])
            if report["status"] in {"action_limit", "paused"} else
            ["repository_analysis"] if not measured else
            ["research_iteration", "independent_confirmation"]})
        try:
            benchmark_bundle.build(repo=self.repo, output=self.output,
                                   declaration=self.declaration, run_id=self.run_id)
        except (OSError, ValueError, TypeError) as exc:
            report["bundle_error"] = redact(f"{type(exc).__name__}: {exc}")[:400]
            atomic_json(self.output / f"preparation_{self.run_id}.json", report)
        self.current_action = {}
        self._persist_run_state(status=report["status"])
        if self.state_persistence_error:
            return self._state_failure_report()
        self._refresh_document(status=report["status"], current="finished")
        return report

    def _supervision_fingerprint(self) -> str:
        """Hash decision-relevant progress, excluding timestamps and runtime telemetry."""
        facts = self.state()
        progress = facts.get("research_progress") or {}
        options = facts.get("research_options") or {}
        return object_digest({
            "declaration": object_digest(self.declaration),
            "execution": object_digest(self.execution),
            "interpreter": str(self.interpreter) if self.interpreter else None,
            "stages": self.stages,
            "verified": self.verified,
            "environment": facts.get("environment"),
            "metric_binding": facts.get("metric_binding"),
            "research_progress": {key: progress.get(key) for key in (
                "status", "round_count", "measured_rounds", "best_candidate",
                "next_round", "planned_rounds", "confirmation")},
            "research_options": {
                "status": options.get("status"),
                "generation_attempted": options.get("generation_attempted"),
                "labels": [str(row.get("label") or "") for row in
                           options.get("items", []) if isinstance(row, dict)],
            },
            "to_measure": facts.get("to_measure"),
            "native_jobs": [{key: row.get(key) for key in ("job_id", "status", "attempt_id")}
                            for row in facts.get("native_jobs") or []],
        })

    def run(self, *, max_steps: int = 8, max_relaunch: int = 0) -> dict[str, Any]:
        """Supervise bounded Scheduler segments until a terminal state or relaunch limit.

        AutoSOTA's outer Monitor continues a coding-agent session when a bounded segment
        yields but the goal, wall budget, and model budget remain active. Two consecutive
        segments without decision-relevant state change trigger a read-only Monitor handoff;
        it advises the Scheduler and never chooses or executes a repair itself.
        """
        if max_steps <= 0 or max_relaunch < 0:
            raise ValueError("max_steps must be positive and max_relaunch non-negative")
        cycles = 0
        waiting_segments = 0
        stagnant_cycles = 0
        stagnation_reviews = 0
        report: dict[str, Any] = {}
        while cycles <= max_relaunch:
            before = self._supervision_fingerprint()
            step_start = len(self.steps)
            report = self._run_cycle(max_steps=max_steps)
            from .scheduling import policy
            cycle_steps = self.steps[step_start:]
            if (policy(self.output) and report.get("status") == "action_limit" and cycle_steps
                    and all(row.get("step") == "wait_for_jobs" for row in cycle_steps)
                    and self.budget and self.budget.remaining() > 0):
                waiting_segments += 1
                continue  # Event waiting is not a fresh Scheduler/model relaunch.
            cycles += 1
            if report.get("status") != "action_limit":
                break
            after = self._supervision_fingerprint()
            from .native_jobs import active_jobs
            pending = any(row["status"] in {"queued", "running"} for row in self._native_job_view()) or any(row["status"] in {"submitted", "running"} for row in self._agent_task_view())
            if before == after and not pending:
                stagnant_cycles += 1
            else:
                stagnant_cycles = 0
            if stagnant_cycles >= 2:
                last = self.last_action or {}
                action_id = str(last.get("decision_id") or
                                object_digest({"run_id": self.run_id,
                                               "cycle": cycles,
                                               "fingerprint": after})[:32])
                observation = self._monitor_failed_action(
                    action_id=action_id,
                    action={"step": "scheduler_segment", "outcome": "no_progress",
                            "because": "two bounded Scheduler segments ended without a "
                                       "decision-relevant state transition",
                            "last_action": last,
                            "receipt_ref": last.get("receipt_ref", "")})
                stagnation_reviews += 1
                stagnant_cycles = 0
                self._refresh_document(status=("budget_exhausted" if observation and
                                               observation.get("status") == "budget_exhausted"
                                               else "running"),
                                       current="AgentMonitor stagnation assessment")
                if observation and observation.get("status") == "budget_exhausted":
                    report["status"] = "budget_exhausted"
                    break
            if cycles > max_relaunch:
                break

        report["steps"] = self.steps
        report["state"] = self.state()
        report["supervision"] = {
            "scheduler_segments": cycles,
            "automatic_relaunches": max(0, cycles - 1),
            "max_relaunch": max_relaunch,
            "stagnation_reviews": stagnation_reviews,
        }
        if policy(self.output):
            report["supervision"]["event_wait_segments"] = waiting_segments
        if report.get("status") == "action_limit":
            last = self.steps[-1] if self.steps else {}
            if max_relaunch == 0 and last.get("outcome") in _MONITORED_FAILURES:
                # Preserve one-shot diagnostic semantics for callers that explicitly asked
                # for no automatic continuation. The formal AutoSOTA CLI supplies a bounded
                # relaunch budget and lets the Scheduler respond to the Monitor handoff.
                report["status"] = "adaptation_unresolved"
                self._persist_run_state(status="adaptation_unresolved")
            else:
                report["status"] = "paused"
                report["pause_reason"] = (
                    "automatic Scheduler relaunch limit reached; this run is resumable with "
                    "the same frozen inputs and budget")
                last_failure = next((row for row in reversed(self.steps)
                                     if row.get("outcome") in {"could not be asked", "no interpreter recorded"}), {})
                report["pause_cause"] = {key: last_failure.get(key) for key in (
                    "because", "failure_category", "error_detail", "evidence_id", "evidence_ref")}
                report["state"] = self.state()
                self._persist_run_state(status="paused")
                feasibility_path = self.output / "feasibility.json"
                feasibility = (read_json(feasibility_path)
                               if feasibility_path.is_file() and
                               not feasibility_path.is_symlink() else {})
                feasibility.update(status="paused", blocking_conditions=[],
                                   resume_when="resume this persisted AutoSOTA run; inputs and "
                                               "budget remain frozen")
                atomic_json(feasibility_path, feasibility)
        else:
            self._persist_run_state(status=str(report.get("status") or "internal_error"))
        atomic_json(self.output / f"preparation_{self.run_id}.json", report)
        if self.state_persistence_error:
            return self._state_failure_report()
        self._refresh_document(status=str(report.get("status") or "internal_error"),
                               current=str(report.get("pause_reason") or "finished"))
        return report
