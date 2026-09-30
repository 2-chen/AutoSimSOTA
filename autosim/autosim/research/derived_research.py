"""The research protocol, over whatever the derivation found and the build produced.

The protocol existed twice, each time welded to one benchmark: a RoboSyn file that collects,
admixes, trains, evaluates and selects, and a RoboTwin file that does a shorter version of
the same thing. A third benchmark meant a third file, and LIBERO would have been it.

Everything the protocol needs is now produced rather than written. The environment is built
by `provision`, the stages and their invocations are derived by `execution_derive`, the
commands are generated and verified by `execution_derive.generate_argv`, and they run through
`DeclarativeBackend`. What was missing is the part that decides *what to do next* -- and that
part does not vary by benchmark. Which stages exist does, and that is what it reads.

Two things make this general rather than a third copy.

**It asks which stages exist.** A benchmark with no way to produce new trajectories is not a
benchmark that cannot be researched; `available("collect")` is false and the loop uses the
families it has. LIBERO is exactly that case, and the loop should reach it without being
told.

**A proposal travels as settings, not as a command line.** The controller varies names from
the benchmark's own declared space -- `train.loss_scale`, an epoch count, a policy family --
and each benchmark spells those differently. The argv function is where that spelling lives,
so the layer that decides *what* to vary never learns *how* it is written down.

What this does not yet do is establish statistical improvement. It can rank development
measurements by a declared metric, but paired seeds, independent confirmation and intervals
remain separate acceptance work. A score in this loop is not an AutoSOTA claim.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import textwrap
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .adapter_protocol import OptimizationSpace
from . import (awareness, media_manifest, objective, redlines as redline_rules,
               run_record, telemetry)
from .budget import RunBudget
from .codepatch import Patch, apply_many, check_many, patches_from_change
from .codepatch import diff as patch_diff, revert as revert_patch
from .compute_decision import ComputeDecision, decide, recheck_gpu, reconsider
from .declaration import values_the_program_contradicts
from .decisions import Decision, Decisions
from .experiment_bundle import artifact_identity, export_best, freeze_artifact
from .execution_graph import ExecutionGraph
from .evidence_store import capture_attempt_evidence
from .ideas import (Idea, IdeaLibrary, declared_space_compatibility, evidence_stalled,
                    execution_compatibility, needs_a_leap)
from .ideas import repair as repair_idea, select as select_idea
from .ideas import build as build_ideas, leap as leap_idea
from .metric_contract import MetricSpec, resolve_metric_artifact
from .policy_consumption import (verify_policy_consumption, verify_rollout_evidence,
                                 verify_metric_lineage)
from .process_executor import (ProcessAttempt, capture_process_identity,
                               requested_containment, run_process)
from .readings import (explicitly_zero_training_work, named_numbers, success_rate,
                       success_reading, tail_of, training_progress)
from . import selection
from .significance import compare
from .common import (assert_frozen, atomic_json, bounded_run, digest, immutable_json,
                     isolated_argv, now, object_digest, read_json, redact,
                     run_local_environment, sanitize_model_text)
from .decision import build_request, validate_proposal
from .skills import skills_reference
from .snapshot import Snapshots
from .declarative_backend import (DeclarativeBackend, artifact_pattern_problem,
                                 invocation_stdin)
from .devices import NoCompatibleDevice
from .gpu_lease import GPULeaseBusy, GPUResourceLease
from .research_state import (ResearchStateError, ResearchStatePersistenceError,
                             ResearchStateStore)


#: Stages the protocol would use if the benchmark has them. Which ones it *does* have is the
#: derivation's answer, and a missing one is a family the loop works without.
STAGE_ROLES = {
    "train": "produce a policy from the benchmark's own demonstrations",
    "evaluate": "score a policy on the benchmark's own initial states and success predicate",
    "collect": "produce new trajectories, when the benchmark can do so without a person",
    "prepare_data": "turn shipped data into what the trainer reads, when that is a step",
}

_ROUND_FINALIZATION_STEPS = (
    "proposal_resolution", "rubric_pre_update", "round_event", "best_state",
    "demo_capture", "source_rollback", "idea_outcome", "benchmark_questions",
    "rubric_final", "history_session_commit",
)


def _source_quote_present(source: str, quote: str) -> bool:
    """Match an exact source excerpt while allowing common block indentation to be omitted.

    Models often return a code excerpt without the indentation shared by its enclosing
    function. Normalize only the excerpt's common leading indentation, then require the full
    line sequence to match contiguously; do not normalize internal whitespace or tokens.
    """
    candidate = quote.strip()
    if not candidate:
        return False
    if candidate in source:
        return True
    expected = textwrap.dedent(candidate).splitlines()
    source_lines = source.splitlines()
    if not expected:
        return False
    for start in range(len(source_lines) - len(expected) + 1):
        actual = textwrap.dedent("\n".join(
            source_lines[start:start + len(expected)])).splitlines()
        if actual == expected:
            return True
    return False


@contextmanager
def _live_stage_document(root: Path, *, telemetry_writer: Any = None,
                         report_root: Path | None = None, run_id: str = "derived",
                         telemetry_poll_seconds: float = 15.0,
                         heartbeat_seconds: float = 45.0):
    """Keep status and optional native training telemetry current during a stage."""
    stopped = threading.Event()

    def refresh_status() -> None:
        try:
            run_record.refresh_live_status(root)
            run_record.refresh_live_status(root, destination=root.parent.parent / "RUN.md")
        except Exception:  # noqa: BLE001 - reporting must never stop the benchmark process
            pass

    def refresh_projection() -> None:
        if report_root is None:
            return
        try:
            run_record.refresh_research_projection(report_root, run_id)
        except Exception:  # noqa: BLE001 - telemetry/reporting is observational only
            pass

    def poll_telemetry() -> None:
        if telemetry_writer is None:
            return
        before = (telemetry_writer.sample_count, telemetry_writer._published_status,
                  tuple(telemetry_writer.errors))
        try:
            state = telemetry_writer.poll()
        except Exception as exc:  # noqa: BLE001 - isolate every observer failure
            try:
                telemetry_writer.note_error(f"observer_{type(exc).__name__}")
            except Exception:  # noqa: BLE001
                pass
            return
        after = (telemetry_writer.sample_count, state.get("status"),
                 tuple(state.get("errors") or []))
        if before != after:
            refresh_projection()

    def refresh() -> None:
        last_telemetry_poll = time.monotonic()
        last_heartbeat = last_telemetry_poll
        while not stopped.wait(max(0.05, min(telemetry_poll_seconds,
                                             heartbeat_seconds))):
            current = time.monotonic()
            if (telemetry_writer is not None and telemetry_poll_seconds > 0 and
                    current - last_telemetry_poll >= telemetry_poll_seconds):
                poll_telemetry()
                last_telemetry_poll = current
            if current - last_heartbeat >= heartbeat_seconds:
                refresh_status()
                last_heartbeat = current

    refresh_status()
    poll_telemetry()
    refresh_projection()
    worker = threading.Thread(target=refresh, name="autosim-stage-document", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=1)


def _load_confirmation(run_root: Path) -> dict:
    """The confirmation this run already took, if it took one.

    Read from the file rather than from memory so that a resumed run cannot take a second
    one -- the whole value of a held-out measurement is that it happened once.
    """
    try:
        row = read_json(Path(run_root) / "confirmation.json")
    except (OSError, ValueError):
        return {}
    return row if isinstance(row, dict) and row.get("ok") else {}


class DerivedResearch:
    """One benchmark's research loop, assembled from what was derived about it."""

    #: How long a single stage may run. Not a budget on the *research* -- the loop takes as
    #: many rounds as it is given -- but a ceiling on one command, so a stage that hangs is
    #: a failed round rather than the end of the session. A lifelong benchmark training ten
    #: tasks at a publishable epoch count runs for a day, so this is set well past that: the
    #: two hours it used to be would have killed every real measurement.
    STAGE_TIMEOUT_SECONDS = 60 * 60 * 72

    #: The label of the idea that means "the controller does not want another round". A
    #: constant rather than a magic string because two places read it: the fallback that makes
    #: it, and the loop that has to tell it apart from a move worth running.
    CONTROLLER_STOPPED = "(the controller stopped)"

    def __init__(self, *, repo: Path, output: Path, backend: DeclarativeBackend,
                 interpreter: Path, space: OptimizationSpace, client: Any,
                 stages: dict[str, str], run_id: str = "derived",
                 compute: ComputeDecision | None = None, benchmark: str = "",
                 declaration: dict[str, Any] | None = None,
                 checkpoint: str = "", require_training_progress: bool = False,
                 require_policy_consumption: bool = False,
                 job_control_path: Path | None = None,
                 controller_decision_id: str = ""):
        self.repo = Path(repo).expanduser().resolve()
        self.output = Path(output)
        self.backend = backend
        self.interpreter = Path(interpreter)
        self.space = space
        self.client = client
        self.sources = stages
        self.run_id = run_id
        self.state_store = ResearchStateStore(self.output, run_id=self.run_id,
                                              repository=self.repo)
        # Which device this runs on, decided from the machine rather than assumed. Late
        # enough that a caller can supply one it has already recorded, early enough that the
        # first stage and the last use the same answer.
        self.compute = compute
        self.benchmark = benchmark
        # What the benchmark says about itself, when a caller has it. Not required -- a loop
        # with no declaration still runs, with the standing red lines and a bare rubric -- but
        # it is what lets the red lines carry the benchmark's own evaluator and what lets the
        # rubric ask questions in the benchmark's terms rather than in this system's alone.
        self.declaration = dict(declaration or {})
        self.metric_spec = MetricSpec.from_declaration(self.declaration)
        from .metric_guardrails import specifications
        self.guardrail_specs = specifications(self.declaration)
        self.source_policy = ((self.declaration.get("task_contract") or {}).get(
            "policy_representation") == "source")
        graph_doc = backend.answer.get("execution_graph")
        self.execution_graph = graph_doc if isinstance(graph_doc, dict) and graph_doc.get(
            "score_target") else None
        if self.execution_graph:
            graph = ExecutionGraph(self.execution_graph)
            target = str(self.execution_graph["score_target"])
            if target not in graph.nodes or graph.nodes[target].role != "evaluate":
                raise ValueError("execution graph score_target must name an evaluate node")
        #: A checkpoint the benchmark ships, when one was found. What a run scores when it
        #: has an evaluator and no trainer of its own -- see `measure`.
        self.checkpoint = str(checkpoint or "")
        self.require_training_progress = require_training_progress
        self.require_policy_consumption = require_policy_consumption
        self.job_control_path = Path(job_control_path) if job_control_path else None
        # The preparation-level decision owns this bounded research action. Keep its ID on
        # every nested research action and stage receipt so the outer reasoning record can be
        # followed to the exact process attempt without making the inner controller a second
        # owner of the run-wide decision.
        self.controller_decision_id = str(controller_decision_id or "")
        self.run_root = self.output / "research" / run_id
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.budget = RunBudget.existing(self.output)
        frozen_path = self.run_root / "protocol_frozen.json"
        self._frozen_error = ""
        self._held_out_error = ""
        try:
            if frozen_path.is_file():
                self._frozen = dict(read_json(frozen_path).get("hashes") or {})
            else:
                self._frozen = redline_rules.protected_hashes(
                    self.repo, self.red_lines(), exclude_roots=(self.run_root,))
                atomic_json(frozen_path, {"schema_version": 1, "hashes": self._frozen,
                                          "created_at": now()})
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            self._frozen, self._frozen_error = {}, f"{type(exc).__name__}: {exc}"
        events_path = self.run_root / "events.json"
        try:
            held_events = read_json(events_path) if events_path.is_file() else {}
            rows = held_events.get("rows") if isinstance(held_events, dict) else []
            self.events: list[dict[str, Any]] = ([dict(row) for row in rows
                                                  if isinstance(row, dict)]
                                                 if isinstance(rows, list) else [])
        except (OSError, TypeError, ValueError) as exc:
            self.events = [{"at": now(), "event": "prior_event_log_unreadable",
                            "why": redact(f"{type(exc).__name__}: {exc}")[:300]}]
        self.active_research_action: dict[str, Any] = {}
        self.state_persistence_error = ""
        # What the run decided and why. Beside the events rather than inside them: an event
        # says something happened, a decision says someone chose it and on what grounds.
        self.decisions = Decisions(self.run_root)
        #: The id of the most recent proposal, so the round that runs it can close it.
        self.last_proposal_decision: str = ""
        # The ideas this run could try, which is where a round's work now comes from -- an
        # idea of a kind, chosen from a batch, rather than one change invented per round
        # against the declared space. The loop could only ever tune because a proposal was
        # drawn from the space, and the space has only values in it.
        self.library = IdeaLibrary(self.run_root / "ideas.json")
        # What each change looked like before it was made, so a change that does not work
        # leaves nothing behind. Content-addressed and per-run; see `snapshot`.
        self.snapshots = Snapshots(self.run_root / "snapshots")
        #: Every idea's audit, kept for the record. Refusals as well as clearances: a run that
        #: lists only what it did cannot be told from one that considered nothing else.
        self.audits: list[dict[str, Any]] = []
        #: How far the run has got, when the number at the end is not the only thing known.
        self.rubric = objective.plain()
        self.rubric_built = False
        #: The benchmark files this run has written to. What a snapshot has to hold: a
        #: snapshot of the whole checkout would be a second copy of the benchmark, and the
        #: parts of it that matter are the parts something changed.
        self.touched: set[str] = set()
        #: file -> (the patch applied to it, the text that was there). Held so a change that
        #: did not work can be taken back exactly instead of in the other direction.
        self._held: dict[str, tuple[Patch, str]] = {}
        #: Idea label -> the declared-space proposal it came from, when it came from there.
        #: The proposal's collection half is not part of the idea's `change`, because
        #: `collection` is also the name of a declared section and a settings map carrying
        #: that key would be read as an axis rather than as a request for data.
        self._proposals_by_idea: dict[str, dict[str, Any]] = {}
        #: Labels that came from the declared space rather than from the library. The two
        #: sources are not interchangeable when a round cannot run: the space would answer
        #: the same way again, and the library has other ideas in it.
        self._from_declared_space: set[str] = set()
        run_record.generate(self.run_root, title=f"{self.benchmark or self.run_id} / derived")

    # -- the lines, the library of ideas, and how far the run has got --------------------

    def red_lines(self) -> redline_rules.RedLines:
        """What this run may not do, checked against every idea before anything runs.

        `from_declaration` when there is a declaration, because the benchmark's own evaluator
        is the one file that certainly may not be edited and only the declaration knows which
        it is. The standing six hold either way.
        """
        if self.declaration:
            return redline_rules.from_declaration(self.declaration, repo=self.repo)
        return redline_rules.RedLines(benchmark=self.benchmark)

    def _repository_files(self) -> dict[str, list[str]]:
        """The benchmark's own text files, grouped by top-level directory.

        Here because a `code` idea names a file, and a model that has not been told which
        files there are names one it has inferred from a README. The grouping and the walk are
        `survey`'s, which already learned that a depth-first listing of a checkout with a
        vendored library in it is a listing of the vendored library. Bounded, because this
        goes into a prompt: the point is that the model can see the shape of the checkout and
        name a real path, not that it can read the checkout here -- `ideas.repair` shows it the
        file once it has named one.
        """
        try:
            from .survey import _candidate_sources
            groups = _candidate_sources(self.repo, max_depth=4)
        except Exception:                                            # noqa: BLE001
            return {}
        out: dict[str, list[str]] = {}
        for key, paths in sorted(groups.items()):
            out[key] = [str(one.relative_to(self.repo)) for one in sorted(paths)[:40]]
        return out

    def _material(self) -> dict[str, Any]:
        """Everything a batch of ideas -- or a rubric -- is made from.

        The stages and how they run, because a `code` idea is about a file this benchmark
        has and the model has to know which files there are; the declared space, because a
        `param` idea has to name a value that exists; and what the benchmark says about
        itself, because that is where its own evaluation boundary is written down.
        """
        stages: dict[str, Any] = {}
        for stage, role in STAGE_ROLES.items():
            row = dict(self.backend.stages.get(stage) or {})
            stages[stage] = {"role": role, "available": self.available(stage),
                             "entrypoint": row.get("entrypoint"),
                             "argv": row.get("argv"),
                             "verified_parameters": sorted(
                                 str(key) for key in
                                 (self.backend.parameters.get(stage) or {})),
                             "why": redact(str(row.get("why") or ""))[:600]}
        return {
            "benchmark": self.benchmark,
            "repository": str(self.repo),
            "stages": stages,
            "selected_stage_parameters": {
                stage: stages[stage]["verified_parameters"]
                for stage in self.sources if stage in stages and self.available(stage)},
            "files_in_the_checkout": self._repository_files(),
            "declared_space": self.space.describe(),
            "what_the_benchmark_declares": {
                "capabilities": self.declaration.get("capabilities"),
                "task_contract": self.declaration.get("task_contract"),
                "evidence": redact(str(self.declaration.get("evidence") or ""))[:4000],
            },
            "what_may_not_change": self.red_lines().states_itself(),
            "methods": skills_reference(
                benchmark=self.benchmark or None,
                query="benchmark task objective source stage entrypoint assets metric "
                      "evaluation protocol capabilities data collection training feasibility"),
        }

    def build_rubric(self, *, on_event: Any = None) -> Any:
        """The objective: the spine, plus this benchmark's own questions under it.

        Once per run and not once per round. It is a reading of the benchmark, and the
        benchmark does not change between rounds -- regenerating it would make two rounds'
        scores incomparable, which is the one thing a scoreboard may not do.
        """
        if self.rubric_built:
            return self.rubric
        path = self.run_root / "rubric.json"
        held = read_json(path) if path.is_file() else None
        if isinstance(held, dict) and held.get("checks"):
            self.rubric = objective.Rubric([
                objective.Check(name=str(one.get("name") or ""),
                                question=str(one.get("question") or ""),
                                weight=float(one.get("weight") or 0.0),
                                passed=one.get("passed"),
                                because=str(one.get("because") or ""),
                                children=[objective.Check(
                                    name=str(two.get("name") or ""),
                                    question=str(two.get("question") or ""),
                                    weight=float(two.get("weight") or 0.0),
                                    passed=two.get("passed"),
                                    because=str(two.get("because") or ""),
                                    by=str(two.get("by") or ""))
                                    for two in (one.get("children") or [])])
                for one in held["checks"]])
            self.rubric_built = True
            return self.rubric
        self.rubric = objective.decompose(self.client, self._material(), on_event=on_event)
        self.rubric_built = True
        self._write_rubric()
        return self.rubric

    def _write_rubric(self) -> None:
        atomic_json(self.run_root / "rubric.json", self.rubric.as_dict())

    def _answer_the_run_s_own_questions(self, *, rounds: list[dict[str, Any]],
                                        current: dict[str, Any]) -> int:
        """Ask the run what it found, for the questions it wrote itself.

        The spine is answered by code from facts the code holds -- whether a number came out,
        whether a stage ran. The children are the benchmark's own, and no code can answer
        them: "does the run log which config file it loaded" is a question about a record
        that exists and a fact only this benchmark's reader knows how to find. The run wrote
        the questions when it read the repository; this is the other half of that act.

        The material is what the run has *recorded* -- the rounds, their verdicts, the readings
        the measurements produced -- and not what it intended. A question answered from intent
        is the failure mode this whole layer exists to prevent, so the prompt says so and the
        answers are marked `by: model` in the record.
        """
        material = {
            "rounds": [{key: row.get(key) for key in
                        ("round", "status", "idea", "kind", "verdict", "why_not",
                         "success_rate", "varied", "repair", "undone")}
                       for row in rounds],
            "the_last_measurement": {
                "where": current.get("where"), "success_rate": current.get("success_rate"),
                "metric_value": current.get("metric_value"), "ok": current.get("ok"),
                # Measurement receipts may contain concrete policy/result paths, filenames,
                # source hashes, or task identifiers. The rubric only needs to know whether
                # any fresh artifact was observed; keep those local details out of this prompt.
                "artifact_available": self._measurement_has_artifact(current)},
            "stages_this_benchmark_has": self.describe(),
        }
        return objective.answer_children(self.client, self.rubric, material=material,
                                         on_event=self._note)

    @staticmethod
    def _measurement_has_artifact(current: dict[str, Any]) -> bool:
        """Project artifact receipts to a boolean before they enter a model prompt."""
        if current.get("policy_artifact") or current.get("result_artifact"):
            return True
        for key in ("artifact",):
            value = current.get(key)
            if isinstance(value, dict) and (value.get("matched") or value.get("path")):
                return True
        for stage in ("train", "evaluate"):
            row = current.get(stage)
            if not isinstance(row, dict):
                continue
            artifact = row.get("artifact")
            if isinstance(artifact, dict) and (artifact.get("matched") or
                                               artifact.get("selected_path")):
                return True
            if row.get("policy_artifact") or row.get("result_artifact"):
                return True
        return False

    def _ran_anything(self) -> bool:
        """Whether a stage has actually been run, as opposed to merely having a command.

        The difference is the spine's first rung and the reason it is not free. A benchmark
        whose commands were generated but never executed is a benchmark with a plan; one in
        which something ran and wrote a log is one where the environment is real. Read off
        the run root -- the logs the stages wrote and the measurements read from them -- and
        not off a flag the loop sets about itself.
        """
        if (self.run_root / "measurements").is_dir() \
                and any((self.run_root / "measurements").glob("*.json")):
            return True
        for stage in STAGE_ROLES:
            if (self.run_root / stage / "output.log").is_file():
                return True
        return False

    def rubric_facts(self, *, rounds: list[dict[str, Any]],
                     baseline: float | None, current: dict[str, Any]) -> dict[str, Any]:
        """What the run knows about itself, under the spine's names.

        Every answer here is read off the record rather than asserted. In particular
        `improvement` needs a *spread* -- two measurements of the same thing -- and reports
        "not reached" when there is only one reading, because a single number has no spread
        and a difference smaller than one is not an improvement.
        """
        ran = [row for row in rounds if row.get("status") == "measured"]
        rates = [row.get("metric_utility", row.get("success_rate")) for row in rounds
                 if isinstance(row.get("metric_utility", row.get("success_rate")), (int, float))]
        here = self._metric_utility(current)
        with_commands = sorted(s for s in self.sources if self.available(s))
        facts: dict[str, Any] = {
            "environment": {
                "passed": bool(with_commands) and self._ran_anything(),
                "because": (f"commands exist for {', '.join(with_commands)} and stage logs are "
                            f"on disk" if with_commands and self._ran_anything() else
                            f"commands exist for {', '.join(with_commands)} but nothing has "
                            f"run" if with_commands else
                            "no stage has a command at all, so nothing can run")},
            "stages": {
                "passed": bool(self.sources) and all(
                    self.available(s) for s in self.sources),
                "because": ("every stage this benchmark has, has a command"
                            if self.sources else "this benchmark declares no stages"),
            },
            "measurement": {
                "passed": isinstance(here, (int, float)),
                "because": (f"the evaluation returned {here}" if isinstance(here, (int, float))
                            else f"no number came out: {current.get('where', 'the')} stage did "
                                 f"not produce one")},
            "comparison": {
                "passed": isinstance(baseline, (int, float)) and bool(ran),
                "because": (f"baseline {baseline} against {len(ran)} measured round(s)"
                            if isinstance(baseline, (int, float)) and ran else
                            "there is nothing to compare yet: "
                            + ("no baseline number" if not isinstance(baseline, (int, float))
                               else "no round was measured"))},
        }
        if len(rates) >= 2 and isinstance(here, (int, float)) \
                and isinstance(baseline, (int, float)):
            spread = max(rates) - min(rates)
            gap = here - baseline
            facts["improvement"] = {
                "passed": gap > spread,
                "because": f"the candidate is {gap:+.4f} against a spread of {spread:.4f} "
                           f"between the {len(rates)} readings taken"}
        return facts

    def _advance_best(self, name: str, *, score: float | None, scale: str, why: str) -> None:
        """Record where the run is, and point `_best` at it when it is the best so far.

        The score is the success rate when there is one and the objective's completion when
        there is not, which is what lets a run that cannot measure yet still have a best state:
        a run that got the evaluator's invocation right is further along than one that did not,
        and without this there is nothing to prefer and nothing to go back to. The two are
        never compared -- `advance` refuses -- because a completion of 0.6 is not better than a
        success rate of 0.4.
        """
        self.snapshots.capture(sorted(self.touched), repo=self.repo, name=name, why=why)
        self.snapshots.advance(name, score=score, scale=scale, why=why)

    def _capture_workspace_source_state(self, label: str) -> dict[str, Any] | None:
        """Freeze the complete isolated checkout state used by one measurement, if available."""
        path = self.output / "workspace_snapshot.json"
        if path.is_symlink():
            raise ValueError("workspace source manifest is a symlink")
        if not path.exists():
            return None
        if not path.is_file():
            raise ValueError("workspace source manifest is not a regular file")
        manifest = read_json(path)
        from .workspace_snapshot import capture_state
        return capture_state(self.repo, manifest, self.snapshots,
                             name=f"source-state-{label}")

    # -- what this benchmark can do ------------------------------------------------------

    def available(self, stage: str) -> bool:
        """Asked of the backend, which holds the command and the answer it came from.

        The caller's stage list is a request; whether a command exists is the fact, and the
        two can disagree -- a list that names a stage whose command failed to generate would
        otherwise be reported as available and then fail at the point of use.
        """
        return self.backend.available(stage)

    def describe(self) -> dict[str, Any]:
        return {stage: {"available": self.available(stage),
                        "role": STAGE_ROLES.get(stage) or
                                (self.backend.stages.get(stage) or {}).get("role"),
                        "entrypoint":
                        (self.backend.stages.get(stage) or {}).get("entrypoint")}
                for stage in dict.fromkeys((*STAGE_ROLES, *self.backend.stages))}

    # -- running one stage ---------------------------------------------------------------

    def stage_directory(self, stage: str) -> Path:
        """Where a stage is told to write, at a path the benchmark can express.

        The system chooses this directory, so the system has to choose one the benchmark's
        own argument grammar can accept. A checkout under a directory whose name is not
        ASCII -- which is where this one lives -- produces an output path that hydra's
        override lexer refuses at the equals sign, and the failure names the harness's own
        directory rather than anything about the benchmark. Rounds were spent on it: the
        reviser read the traceback correctly and still could not act, because the offending
        value was not the benchmark's to choose.

        So a stage is given an ASCII alias to the actual run-owned directory. The real files
        stay below the run root: the survey, media manifest, export and run relocation can
        all find them without following an arbitrary external directory symlink.
        """
        logical = self.run_root / stage
        if all(ord(character) < 128 for character in str(logical)):
            return logical
        base = Path(os.environ.get("AUTOSIM_STAGE_ROOT")
                    or "/tmp/autosim/stages")
        alias = base / object_digest(str(logical))[:16] / stage
        if logical.is_symlink():
            # Read-only compatibility with runs created under the old, reversed mapping.
            # Do not silently migrate in-flight outputs or lose their old provenance.
            resolved = logical.resolve()
            if not resolved.is_dir():
                raise RuntimeError(f"stage output alias is broken: {logical}")
            return resolved
        logical.mkdir(parents=True, exist_ok=True)
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.is_symlink():
            if alias.resolve() != logical.resolve():
                raise RuntimeError(f"stage output alias points elsewhere: {alias}")
        elif alias.exists():
            raise RuntimeError(f"stage output alias already exists: {alias}")
        else:
            alias.symlink_to(logical, target_is_directory=True)
        return alias

    def run_graph(self, document: dict[str, Any], *, target: str,
                  settings: dict[str, Any]) -> dict[str, Any]:
        """Run an explicit dependency graph of verified native commands.

        This is an execution facility, not a performance measurement: an evaluator node
        still needs its metric, protocol and candidate identity checked by ``measure`` or a
        future graph-aware scorer before it can be ranked.
        """
        graph = ExecutionGraph(document)
        from .scheduling import policy
        max_workers = document.get("max_workers", 1) if policy(self.output) else 1
        if max_workers != 1 and not str(document.get("parallel_safety") or "").strip():
            raise ValueError("parallel graph needs Agent evidence that sibling outputs/resources are independent")
        for name in graph.order_for(target):
            if not self.available(name):
                return {"target": target, "status": "blocked", "where": name,
                        "why": "node has no derived executable command", "nodes": []}
        immutable_json(self.run_root / "execution_graph_frozen.json", document)
        attempt = uuid.uuid4().hex
        record_path = self.run_root / "graph_runs" / f"{attempt}.json"
        checkpoint_sources = {source for node in graph.nodes.values()
                              for key, source in node.bindings if key == "checkpoint"}
        frozen_artifacts: dict[str, dict[str, Any]] = {}
        import threading
        event_lock = threading.RLock()
        started_at = now()
        atomic_json(record_path, {"target": target, "status": "running", "nodes": [],
                                  "started_at": started_at})
        last_state: dict[str, Any] = {"nodes": []}

        def remember(state: dict[str, Any]) -> None:
            last_state.update(state)
            atomic_json(record_path, {**state, "graph_attempt_id": attempt,
                                      "started_at": started_at, "updated_at": now()})

        def artifact_of(record: dict[str, Any]) -> Path | None:
            name = str(record.get("stage") or "")
            if name in checkpoint_sources:
                record = self._select_policy_artifact(record)
            raw = self._recorded_artifact(record)
            if raw is None or name not in checkpoint_sources:
                return raw
            cap = int(os.environ.get("AUTOSIM_ARTIFACT_COPY_LIMIT_BYTES", str(2 * 1024**3)))
            if cap <= 0:
                raise ValueError("artifact copy limit must be positive")
            suffix = raw.suffix if raw.is_file() else ""
            archived = freeze_artifact(
                raw, self.run_root / "graph_runs" / attempt / "artifacts" / name /
                f"policy{suffix}", max_bytes=cap)
            frozen_artifacts[name] = archived
            return Path(archived["path"])

        def invoke(name, inputs):
            resources = graph.nodes[name].resources
            if max_workers == 1 or resources is None:
                return self.run_stage(name, settings=settings,
                    _protocol_guard=graph.nodes[name].role == "evaluate", **inputs)
            import copy
            from .scheduling import HostLease, note
            lease = HostLease(attempt + "-" + name, resources, limits=policy(self.output))
            original_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set()
            wait_started = time.monotonic()
            while not lease.acquire():
                if self.budget is None or self.budget.remaining() <= 0:
                    raise TimeoutError("graph resource admission reached hard deadline")
                time.sleep(min(1, self.budget.remaining()))
            try:
                if lease.cores and hasattr(os, "sched_setaffinity"):
                    os.sched_setaffinity(0, set(lease.cores))
                note(self.output, kind="graph_resource_wait", identity=attempt + "-" + name,
                     seconds=time.monotonic()-wait_started, status="admitted")
                worker = copy.copy(self)
                def append_event(*args, **kwargs):
                    with event_lock:
                        return self._append_local_event(*args, **kwargs)
                worker._append_local_event = append_event
                worker.job_resources = resources
                local_settings = dict(settings)
                if not resources["gpu"]:
                    local_settings["device"] = "cpu"
                    worker.compute = ComputeDecision(device="cpu", device_index=0,
                        environment={"CUDA_VISIBLE_DEVICES": ""},
                        why="Agent declared CPU graph node")
                return worker.run_stage(name, settings=local_settings,
                    _protocol_guard=graph.nodes[name].role == "evaluate", **inputs)
            finally:
                lease.release()
                if original_affinity and hasattr(os, "sched_setaffinity"):
                    os.sched_setaffinity(0, original_affinity)

        try:
            result = graph.execute(
                target,
                invoke=invoke, artifact_of=artifact_of, on_step=remember, max_workers=max_workers)
            result.update(graph_attempt_id=attempt, frozen_artifacts=frozen_artifacts)
            remember(result)
            return result
        except Exception as exc:  # noqa: BLE001
            result = {"target": target, "status": "internal_error",
                      "nodes": last_state.get("nodes") or [],
                      "why": redact(f"{type(exc).__name__}: {exc}")[:600]}
            remember(result)
            return result

    def measure_graph(self, document: dict[str, Any], *, target: str,
                      settings: dict[str, Any], label: str) -> dict[str, Any]:
        """Score an explicit graph's evaluate node without assuming its stage name.

        A checkpoint edge evaluates a frozen producer artifact. A graph without one needs
        an explicit source-policy declaration; otherwise an unbound policy is too easy to
        mistake for a candidate. This is a baseline measurement, not a research loop or an
        independent confirmation.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
            raise ValueError("measurement label must be a short path-safe identifier")
        graph = ExecutionGraph(document)
        if target not in graph.nodes or graph.nodes[target].role != "evaluate":
            raise ValueError("graph measurement target must have evaluate role")
        violation = self._comparison_protocol_violation(settings, target=target,
                                                        freeze=True)
        if violation:
            return self._comparison_refusal(label, settings, violation)
        policy_source = dict(graph.nodes[target].bindings).get("checkpoint")
        if policy_source is None and not self.source_policy:
            failed = {"label": label, "ok": False, "where": "policy_identity",
                      "why": "evaluate node has no checkpoint binding and no declared "
                             "source-defined controller", "settings": dict(settings),
                      "metric_value": None}
            atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
            return failed
        source_state: dict[str, Any] | None = None
        try:
            source_state = self._capture_workspace_source_state(label)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            failed = {"label": label, "ok": False, "where": "source_identity",
                      "why": f"the graph source state cannot be frozen: "
                             f"{type(exc).__name__}: {exc}",
                      "settings": dict(settings), "metric_value": None,
                      "source_state": {"status": "unavailable"}}
            atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
            return failed
        executed = self.run_graph(document, target=target, settings=settings)
        if executed.get("status") != "completed":
            failed = {"label": label, "ok": False, "where": executed.get("where") or target,
                      "why": executed.get("why") or "graph did not complete",
                      "graph_run": executed, "settings": dict(settings),
                      "metric_value": None}
            atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
            return failed
        score_node = next(row for row in executed["nodes"] if row["id"] == target)
        score_attempt = str(score_node["attempt_id"])
        scored = read_json(self.run_root / "attempts" / score_attempt / "receipt.json")
        policy = (executed.get("frozen_artifacts") or {}).get(policy_source) or {}
        if policy_source and not policy:
            failed = {"label": label, "ok": False, "where": "policy_archive",
                      "why": "graph did not freeze the checkpoint-producing node",
                      "graph_run": executed, "metric_value": None}
            atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
            return failed
        result_archive: dict[str, Any] = {}
        result_error = ""
        metric_artifact_evidence: dict[str, Any] = {}
        if self.metric_spec.source in {"json", "csv"}:
            artifact, metric_artifact_evidence = self._metric_artifact_for_record(scored)
            try:
                cap = int(os.environ.get("AUTOSIM_RESULT_COPY_LIMIT_BYTES",
                                         str(64 * 1024**2)))
                if cap <= 0 or artifact is None or not artifact.is_file():
                    raise ValueError("graph result needs one existing file and a positive cap")
                result_archive = freeze_artifact(
                    artifact, self.run_root / "experiments" / label / score_attempt /
                    f"result{artifact.suffix}", max_bytes=cap)
                result_archive["content_sha256"] = digest(Path(result_archive["path"]))
            except (OSError, ValueError, RuntimeError) as exc:
                result_error = f"{type(exc).__name__}: {exc}"
            self._attach_metric_artifact_evidence(scored, metric_artifact_evidence,
                                                  result_archive)
        reading = self.metric_spec.read(
            said=str(scored.get("said") or ""),
            artifact=Path(result_archive["path"]) if result_archive else None)
        source_state_error = ""
        if source_state is not None:
            try:
                after_source_state = self._capture_workspace_source_state(label)
                if (after_source_state or {}).get("identity_sha256") != \
                        source_state.get("identity_sha256"):
                    source_state_error = "source tree changed while the graph was being measured"
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                source_state_error = ("source tree could not be verified after graph evaluation: "
                                      f"{type(exc).__name__}: {exc}")
        valid = (reading.get("value") is not None and
                 (self.metric_spec.source == "log" or
                  metric_artifact_evidence.get("status") == "matched") and
                 not source_state_error)
        result = {"label": label, "ok": valid,
                  "where": "" if valid else "source_identity" if source_state_error else target,
                  "why": "" if valid else source_state_error or result_error or
                         "evaluate metric could not be read",
                  "settings": dict(settings), "graph_run": executed,
                  "scored": "a frozen graph checkpoint" if policy else
                            "a source-defined controller with no weight artifact",
                  "policy_artifact": policy, "evaluated_policy_path":
                      str(policy.get("path") or ""),
                  "restorable_policy": bool(policy),
                  "source_state": source_state or {},
                  "source_policy_artifact": ({"kind": "source_tree", **source_state}
                                             if not policy and self.source_policy and
                                             source_state else {}),
                  "result_artifact": result_archive,
                  "metric_artifact_evidence": metric_artifact_evidence,
                  "result_archive_error": result_error,
                  "evaluate": {"attempt_id": score_attempt,
                               "returncode": scored.get("returncode"),
                               "said": str(scored.get("said") or "")[-800:]},
                  "metric": self.metric_spec.as_dict(),
                  "metric_reading": reading,
                  "metric_value": reading["value"] if valid else None,
                  "metric_utility": (self.metric_spec.utility(reading["value"])
                                     if valid else None),
                  "success_rate": reading["value"] if valid and
                                  self.metric_spec.name == "success_rate" else None}
        self._attach_guardrails(result, label=label, said=str(scored.get("said") or ""),
                                artifact=Path(result_archive["path"]) if result_archive else None)
        atomic_json(self.run_root / "measurements" / f"{label}.json", result)
        run_record.generate(self.run_root,
                            title=f"{self.benchmark or self.run_id} / derived")
        return result

    def compute_for(self, stage: str, settings: dict[str, Any]) -> ComputeDecision:
        """This stage's device. Decided once, reused, and revisable from a failure.

        Decided rather than defaulted: the machine this runs on is not known when the code is
        written, so any value written into the code is a guess about a machine nobody has
        seen. A caller that names a device in the settings is taken at its word -- it has a
        reason the resource table cannot show -- and everything else is read from the table.

        Held rather than re-decided per stage, because two stages of one measurement that
        land on different cards are two experiments.
        """
        if self.compute is not None:
            return self.compute
        asked = str(settings.get("device") or "").strip()
        self.compute = decide() if (not asked or asked.lower() == "auto") else decide(prefer=asked)
        # `by="tool"`: nobody reasoned about this, a rule was applied to a resource table. The
        # record says which, because "the system chose this" and "the model chose this" are
        # different claims and a reader weighing them needs to know which one they have.
        self.decisions.record(Decision(
            activity=f"run on {self.compute.device}", by="tool",
            agent="compute_decision", why=self.compute.why,
            used=["the resource table"], outcome={"state": "known",
                                                  "what": {"device": self.compute.device,
                                                           "on_gpu": self.compute.on_gpu}}))
        self._append_local_event({"at": now(), "event": "compute",
                                  "device": self.compute.device, "why": self.compute.why,
                                  "evidence": self.compute.evidence})
        return self.compute

    def _inputs(self, stage: str, *, settings: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        """The system's vocabulary, filled for this stage.

        Built on `blank_inputs` so every name the vocabulary defines is present: a generated
        function reads the names it was shown, and one the caller forgot is a `KeyError`
        inside that function rather than a fact about the benchmark.
        """
        from .execution_derive import blank_inputs
        inputs = blank_inputs()
        training_steps = settings.get("train.n_epochs")
        if training_steps is None:
            training_steps = settings.get("steps")
        if training_steps is None:
            from .execution_derive import TRAINING_VERIFICATION_STEPS
            training_steps = TRAINING_VERIFICATION_STEPS
        inputs.update({
            "python": str(self.interpreter), "repo": str(self.repo),
            "task": settings.get("task") or self._task() or "",
            # Not the checkout. This used to be `str(self.repo)`, which is a guess that the
            # benchmark's data lives where its code does -- false for LIBERO, whose datasets
            # are in a sibling directory, and false in a way the caller cannot see: the value
            # goes into `i`, the caller's inputs take precedence over the values the
            # derivation settled, and so a correct `folder` that the system had discovered
            # and recorded was silently replaced by the checkout path. The runner supplies
            # what it knows, which here is nothing; the derivation's own answer fills it.
            "dataset": "", "output": str(self.stage_directory(stage)),
            "steps": int(training_steps),
            "episodes": int(settings.get("eval.n_eval") or settings.get("episodes") or 0),
            "seed": int(settings.get("seed") or 0),
            "device": str(settings.get("device") or ""),
            "device_index": int(settings.get("device_index") or 0), "setting": "",
            # These are protocol/execution controls with dedicated input slots above, not
            # benchmark hyperparameters. Forwarding them through both channels makes a
            # generated trainer append invented flags such as --task or --episodes.
            "settings": {key: value for key, value in settings.items()
                         if key not in {"task", "task_name", "steps", "episodes",
                                        "seed", "device", "device_index"}},
        })
        inputs.update(overrides)
        return inputs

    def _task(self) -> Any:
        """The benchmark's task, if the derivation named one."""
        row = self.backend.stages.get("evaluate") or {}
        for parameter in (row.get("parameters") or []):
            if isinstance(parameter, dict) and str(parameter.get("name", "")).endswith(
                    ("task", "benchmark")):
                return parameter.get("value")
        return None

    def run_stage(self, stage: str, *, settings: dict[str, Any],
                  timeout: int | None = None, _reconsidered: bool = False,
                  _protocol_guard: bool = False, confirmation: bool = False,
                  **overrides: Any) -> dict[str, Any]:
        """Run one stage only after a fresh occupancy check and cooperative GPU lease."""
        if not self.available(stage) or (self.budget and self.budget.remaining() <= 0):
            return self._run_stage(stage, settings=settings, timeout=timeout,
                                   _reconsidered=_reconsidered,
                                   _protocol_guard=_protocol_guard,
                                   confirmation=confirmation, **overrides)
        try:
            decision = self.compute_for(stage, settings)
        except NoCompatibleDevice as exc:
            return self._record_resource_block(stage, settings, exc,
                                               reason_code="gpu_unavailable")
        if decision.device == "unavailable":
            refusal = NoCompatibleDevice(
                decision.why or "GPU resource is unavailable",
                evidence=decision.evidence)
            return self._record_resource_block(stage, settings, refusal,
                                               reason_code="gpu_unavailable")
        if not decision.on_gpu:
            return self._run_stage(stage, settings=settings, timeout=timeout,
                                   _reconsidered=_reconsidered,
                                   _protocol_guard=_protocol_guard,
                                   confirmation=confirmation, **overrides)

        selected = decision.evidence.get("selected_device") or {}
        device_uuid = str(selected.get("uuid") or "")
        lease_id = uuid.uuid4().hex
        lease: GPUResourceLease | None = None
        try:
            settle = (min(5.0, max(0.0, self.budget.remaining()))
                      if self.budget else 5.0)
            recheck_gpu(decision, settle_seconds=settle)
            lease = GPUResourceLease(
                device_uuid,
                metadata={"run_id": self.run_id, "stage": stage,
                          "lease_id": lease_id}).acquire()
        except NoCompatibleDevice as exc:
            return self._record_resource_block(stage, settings, exc,
                                               reason_code="gpu_unavailable")
        except GPULeaseBusy as exc:
            return self._record_resource_block(stage, settings, exc,
                                               reason_code="gpu_lease_busy",
                                               evidence={"selected_device": selected,
                                                         "lease_id": lease_id})
        except OSError as exc:
            # If the lock cannot be acquired or validated, fail closed: running would turn
            # the serialization policy into a best-effort convention.
            return self._record_resource_block(stage, settings, exc,
                                               reason_code="gpu_lease_unavailable",
                                               evidence={"selected_device": selected,
                                                         "lease_id": lease_id})
        task_gpu = None
        gpu_started = time.monotonic()
        gpu_deadline = None
        try:
            if (self.output / "task_gpu_budget.json").is_file():
                from .task_budget import TaskGPUBudget
                task_gpu = TaskGPUBudget(self.output)
                try:
                    from .scheduling import policy
                    total_left = task_gpu.snapshot()["available_gpu_seconds"]
                    window = None
                    if policy(self.output):
                        window = min(float(self.STAGE_TIMEOUT_SECONDS if timeout is None else timeout), total_left)
                        if self.job_control_path:
                            control = read_json(self.job_control_path)
                            window = min(window, max(.1, control["deadline_epoch"]-time.time()))
                    reserved = task_gpu.reserve(lease_id, device_uuid, requested_seconds=window)
                    gpu_deadline = gpu_started + (total_left if self.job_control_path else reserved["reserved_seconds"])
                    if self.job_control_path:
                        from .scheduling import locked
                        with locked(self.job_control_path.with_name("control.lock")):
                            control = read_json(self.job_control_path)
                            control["gpu_lease_id"] = lease_id
                            atomic_json(self.job_control_path, control)
                except (OSError, ValueError) as exc:
                    task_gpu = None
                    return self._record_resource_block(
                        stage, settings, exc, reason_code="gpu_task_budget_unavailable")
            return self._run_stage(
                stage, settings=settings, timeout=timeout,
                _reconsidered=_reconsidered, _protocol_guard=_protocol_guard,
                confirmation=confirmation, _gpu_lease_id=lease_id,
                _gpu_uuid=device_uuid, _gpu_deadline=gpu_deadline, **overrides)
        finally:
            try:
                if task_gpu is not None:
                    task_gpu.finish(lease_id, time.monotonic() - gpu_started)
            finally:
                lease.release()

    def _record_resource_block(self, stage: str, settings: dict[str, Any],
                               error: BaseException, *, reason_code: str,
                               evidence: dict[str, Any] | None = None
                               ) -> dict[str, Any]:
        """Persist a resource refusal as a blocked attempt, never as a simulator failure."""
        attempt_id = uuid.uuid4().hex
        why = redact(f"{type(error).__name__}: {error}")[:800]
        if evidence is None:
            evidence = dict(getattr(error, "evidence", {}) or {})
        receipt = {
            "schema_version": 1, "run_id": self.run_id, "attempt_id": attempt_id,
            "node_id": stage, "status": "blocked", "started_at": now(),
            "finished_at": now(), "returncode": None, "ran": False,
            "command_started": False, "argv": [],
            "containment_requirement": "not_started",
            "termination_reason": reason_code, "why": why,
            "settings_digest": object_digest(settings),
            "controller_decision_id": self.controller_decision_id or None,
            "resource_block": {"kind": reason_code, "evidence": evidence},
        }
        atomic_json(self.run_root / "attempts" / attempt_id / "receipt.json", receipt)
        try:
            self.decisions.record(Decision(
                activity=f"do not start {stage} without an exclusive GPU",
                by="tool", agent="compute_decision",
                why=why, used=["GPU occupancy table", "GPU lease"]
                if reason_code.startswith("gpu_") else ["GPU occupancy table"],
                produced=[f"attempts/{attempt_id}/receipt.json"],
                outcome={"state": "known", "what": {"blocked": True,
                                                          "reason": reason_code,
                                                          "attempt_id": attempt_id}}))
            self._append_local_event(
                {"at": now(), "event": "stage_blocked", "stage": stage,
                 "status": "blocked", "returncode": None, "attempt_id": attempt_id,
                 "termination_reason": reason_code, "why": why,
                 "resource_block": receipt["resource_block"]},
                shared_event="stage_blocked", shared_status="blocked")
        except ResearchStateError:
            # The local receipt is already durable and explicitly says no process started.
            # The state-store error remains attached to the controller for normal recovery.
            pass
        try:
            run_record.generate(self.run_root,
                                title=f"{self.benchmark or self.run_id} / derived")
        except (OSError, ValueError, RuntimeError):
            pass
        return {"stage": stage, "ran": False, "status": "blocked",
                "returncode": None, "attempt_id": attempt_id, "why": why,
                "termination_reason": reason_code,
                "resource_block": receipt["resource_block"]}

    def _run_stage(self, stage: str, *, settings: dict[str, Any],
                   timeout: int | None = None, _reconsidered: bool = False,
                   _protocol_guard: bool = False, confirmation: bool = False,
                   _gpu_lease_id: str | None = None, _gpu_uuid: str | None = None,
                   _gpu_deadline: float | None = None,
                   **overrides: Any) -> dict[str, Any]:
        """Generate, run and report one stage, with what came out of it."""
        if not self.available(stage):
            return {"stage": stage, "ran": False,
                    "why": f"the derivation reports {stage} unavailable for this benchmark"}
        timeout = self.STAGE_TIMEOUT_SECONDS if timeout is None else timeout
        if self.budget:
            left = self.budget.remaining()
            if left <= 0:
                self.budget.record()
                return {"stage": stage, "ran": False, "status": "blocked",
                        "returncode": None, "why": "run wall-clock budget exhausted"}
            timeout = min(timeout, left)
        decision = self.compute_for(stage, settings)
        # The decision wins over the ambient environment and loses to what the stage itself
        # declares, because a benchmark that names a variable for its own reasons knows
        # something this table does not.
        inputs = self._inputs(stage, settings=settings, device=decision.device,
                              device_index=decision.device_index, **overrides)
        policy_parent: Path | None = None
        if stage == "evaluate":
            policy_input = str(overrides.get("checkpoint") or inputs.get("checkpoint") or "")
            experiment_input = str(overrides.get("experiment_dir") or "")
            try:
                policy_path = Path(policy_input).expanduser().resolve(strict=True)
                experiment_path = Path(experiment_input).expanduser().resolve(strict=True)
                parent = policy_path.parent.resolve(strict=True)
                if (policy_path.is_file() and experiment_path == policy_path and
                        parent.is_relative_to(self.run_root.resolve())):
                    policy_parent = parent
            except (OSError, RuntimeError, ValueError):
                pass
        surroundings = run_local_environment(
            self.output, {**decision.environment, **self.backend.environment(stage)})
        # Through the backend rather than through the generated function directly: the
        # backend is what supplies the values the derivation settled for this repository --
        # a benchmark's own flag names, its task name -- and calling the function without
        # them asks it for something the runner has no way to spell.
        argv = self.backend.argv(stage, inputs)
        resources = getattr(self, "job_resources", None)
        if resources:
            count = str(resources["cpu"])
            surroundings.update({key: count for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        started = time.monotonic()
        before = time.time()
        attempt_id = uuid.uuid4().hex
        # Python's timestamp/size-based bytecode validation can accept a stale .pyc when a
        # source patch has the same length and lands in the same timestamp tick as the
        # baseline. Each attempt gets a fresh cache namespace so a code experiment measures
        # the source state it was given rather than an earlier compiled module.
        surroundings["PYTHONPYCACHEPREFIX"] = str(self.run_root / "pycache" / attempt_id)
        receipt_path = self.run_root / "attempts" / attempt_id / "receipt.json"
        safe_argv = [redact(str(part)) for part in argv]
        receipt = {"schema_version": 1, "run_id": self.run_id, "attempt_id": attempt_id,
                   "node_id": stage, "status": "running", "started_at": now(),
                   "started_epoch": before,
                   "argv": safe_argv, "cwd": str(self.backend.directory(stage)),
                   "output_directory": str(self.stage_directory(stage).resolve()),
                   "policy_output_root": str(policy_parent) if policy_parent else "",
                   "gpu_lease_id": _gpu_lease_id,
                   "gpu_device_uuid": _gpu_uuid,
                   "timeout_seconds": timeout,
                   "containment_requirement": "pid_namespace",
                   "settings_digest": object_digest(settings),
                   "controller_decision_id": self.controller_decision_id or None}
        comparison_path = self.run_root / "comparison_protocol.json"
        if (stage == "evaluate" or _protocol_guard) and comparison_path.is_file():
            try:
                receipt["comparison_protocol_sha256"] = digest(comparison_path)
            except OSError:
                receipt["comparison_protocol_sha256"] = ""
        atomic_json(receipt_path, receipt)
        if stage == "evaluate" or _protocol_guard:
            violation = (self._protocol_violation() or
                         self._comparison_protocol_violation(settings, target=stage,
                                                             confirmation=confirmation))
            if violation:
                atomic_json(receipt_path, {**receipt, "status": "blocked",
                            "termination_reason": "protocol_changed", "finished_at": now(),
                            "returncode": None, "why": violation})
                return {"stage": stage, "ran": False, "status": "blocked",
                        "returncode": None, "why": violation, "attempt_id": attempt_id}
        # Written down before the command runs and removed after. A run that never comes back
        # leaves this behind, and that is the only way "working" is distinguishable from
        # "stuck" from outside -- ten hours passed once because it was not.
        aw = awareness
        expected = aw.cost_of(self.events, stage)
        aw.began(self.run_root, stage=stage, argv=safe_argv, expected_seconds=expected,
                 device=decision.device)
        # In its own session, so that a timeout can take the whole tree with it. A benchmark
        # that parallelises by spawning processes -- LIBERO starts twenty per evaluation --
        # leaves those processes behind when only the parent is killed, and they keep a core
        # each: nine of them, one a day old, took this machine's load average to 28 and made
        # every later run slow enough to look hung. It took hours to diagnose, and what it
        # looked like was a benchmark that deadlocks.
        # Whatever the repository assumes and does not have, made true first. In the loop and
        # not only at derivation, because this is where the stage actually runs and a stage
        # verified against an unstaged repository was verified against something else. The
        # commands are required to be safe to run twice; what they did is in the record.
        for command in self.backend.staging(stage):
            staging_timeout = min(600, self.budget.remaining()) if self.budget else 600
            if _gpu_deadline is not None:
                staging_timeout = min(staging_timeout, _gpu_deadline - time.monotonic())
            if self.job_control_path:
                local_control = read_json(self.job_control_path)
                staging_timeout = min(staging_timeout, local_control["deadline_epoch"]-time.time())
                if local_control.get("cancelled"):
                    staging_timeout = 0
            if staging_timeout <= 0:
                if self.budget:
                    self.budget.record()
                atomic_json(receipt_path, {**receipt, "status": "blocked",
                            "termination_reason": "run_budget_exhausted",
                            "finished_at": now(), "returncode": None})
                aw.ended(self.run_root)
                run_record.generate(self.run_root)
                return {"stage": stage, "ran": False, "status": "blocked",
                        "returncode": None, "why": "run wall-clock budget exhausted",
                        "attempt_id": attempt_id}
            try:
                with _live_stage_document(self.run_root):
                    staged = bounded_run(isolated_argv(
                        ["sh", "-c", command], output=self.output, repo=self.repo,
                        require_pid_namespace=True),
                                         timeout=staging_timeout,
                                         cwd=self.backend.directory(stage),
                                         env={**os.environ, **surroundings})
            except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
                staged = None
                self._append_local_event({"at": now(), "event": "staging_failed",
                                          "stage": stage, "command": redact(command)[:300],
                                          "why": f"{type(exc).__name__}: {exc}"})
            else:
                self._append_local_event({"at": now(), "event": "staged", "stage": stage,
                                          "command": redact(command)[:300],
                                          "returncode": staged.returncode,
                                          "said": redact((staged.stdout or "") +
                                                         (staged.stderr or ""))[-400:]})
            if staged is not None and staged.returncode != 0:
                record = {"stage": stage, "ran": False, "why":
                          f"staging did not succeed: {redact((staged.stdout or '') + (staged.stderr or ''))[-400:]}",
                          "staging": redact(command)}
                atomic_json(receipt_path, {**receipt, "status": "failed",
                            "termination_reason": "staging_failed", "finished_at": now(),
                            "returncode": staged.returncode})
                aw.ended(self.run_root)
                run_record.generate(self.run_root)
                return record
            if staged is None:
                atomic_json(receipt_path, {**receipt, "status": "failed",
                            "termination_reason": "staging_error", "finished_at": now(),
                            "returncode": None})
                aw.ended(self.run_root)
                run_record.generate(self.run_root)
                return {"stage": stage, "ran": False,
                        "why": f"staging could not finish: {redact(command)[:300]}"}
        if stage == "evaluate" or _protocol_guard:
            violation = self._protocol_violation()
            if violation:
                aw.ended(self.run_root)
                atomic_json(receipt_path, {**receipt, "status": "blocked",
                            "termination_reason": "protocol_changed_by_staging",
                            "finished_at": now(), "returncode": None, "why": violation})
                run_record.generate(self.run_root)
                return {"stage": stage, "ran": False, "status": "blocked",
                        "returncode": None, "why": violation, "attempt_id": attempt_id}
        if self.budget or _gpu_deadline is not None:
            left = self.budget.remaining() if self.budget else float("inf")
            if _gpu_deadline is not None:
                left = min(left, _gpu_deadline - time.monotonic())
            if left <= 0:
                if self.budget:
                    self.budget.record()
                atomic_json(receipt_path, {**receipt, "status": "blocked",
                            "termination_reason": "run_budget_exhausted_after_staging",
                            "finished_at": now(), "returncode": None})
                aw.ended(self.run_root)
                run_record.generate(self.run_root)
                return {"stage": stage, "ran": False, "status": "blocked",
                        "returncode": None, "why": "run wall-clock budget exhausted",
                        "attempt_id": attempt_id}
            timeout = min(timeout, left)
        fed = invocation_stdin(self.backend.stages.get(stage) or {})
        # The stage's output goes to a file as it is produced, and is read back from there.
        # It used to be piped and read only when the process ended, so **a stage that was
        # running printed nothing anyone could read**: the liveness marker said it was alive
        # and there was no way to see what it was doing. That is the shape of the incident
        # this system spent a day on -- four hours of a training run that had finished task 0
        # in eleven minutes, with nothing written down -- and the marker alone does not fix
        # it. `tail_of` reads the file back with the same bound the pipe had.
        stage_directory = self.stage_directory(stage)
        stage_directory.mkdir(parents=True, exist_ok=True)
        attempt_log = stage_directory / "attempts" / attempt_id / "output.log"
        attempt_log.parent.mkdir(parents=True, exist_ok=True)
        log = stage_directory / "output.log"
        if log.is_file() and not log.is_symlink():
            # Preserve a pre-receipt log instead of overwriting the only record of it.
            log.rename(attempt_log.parent / "previous_output.log")
        link = stage_directory / f"output.log.{attempt_id}.next"
        link.symlink_to(attempt_log)
        os.replace(link, log)
        # `stderr` is merged into the same file rather than kept in a second pipe, and that is
        # a change in what `said` contains: the two streams used to be concatenated, stderr
        # first, which is not the order they were written in. A traceback with its surrounding
        # stdout in the wrong order is harder to read, and reading it is the only thing that
        # happens to it.
        stream = attempt_log.open("w", encoding="utf-8", errors="replace")
        timed_out, refused = False, ""
        process_identity_record: dict[str, Any] = {}
        # The command is argv *and* the environment it ran under, and only one of the two was
        # being written down. A stage that failed left its argv, its output and nothing about
        # the shell it ran in -- so a failure on a missing import could not be told from a
        # wrong `PATH`, and the run could not be re-created by hand to find out. What is kept
        # is the part that differs from the ambient environment, which is the part the
        # derivation established and the part a reader needs.
        differing = {name: ("[REDACTED]" if any(secret in name.upper() for secret in
                    ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY"))
                    else redact(value)) for name, value in sorted(surroundings.items())
                     if os.environ.get(name) != value}
        receipt_ref = str(receipt_path.resolve().relative_to(self.output.resolve()))
        log_ref = str(attempt_log.resolve().relative_to(self.output.resolve()))
        try:
            self._record_shared_event(
                "stage_started",
                details={"stage": stage, "attempt_id": attempt_id,
                         "receipt_ref": receipt_ref, "log_ref": log_ref,
                         "controller_decision_id": self.controller_decision_id or None,
                         "argv_sha256": object_digest(safe_argv),
                         "settings_sha256": object_digest(settings),
                         "timeout_seconds": timeout, "device": decision.device},
                phase_state={"current_action": {"step": stage, "status": "running",
                                                 "attempt_id": attempt_id,
                                                 "receipt_ref": receipt_ref,
                                                 "log_ref": log_ref},
                             "parent_action": self.active_research_action or None},
                event_patch={"current_action": {"step": stage, "status": "running",
                                                "attempt_id": attempt_id,
                                                "receipt_ref": receipt_ref,
                                                "log_ref": log_ref},
                             "parent_action": self.active_research_action or None})
        except ResearchStateError as exc:
            # This checkpoint is the gate before launching a benchmark process. Preserve a
            # durable refusal receipt and close the liveness marker, then stop the controller.
            stream.close()
            aw.ended(self.run_root)
            atomic_json(receipt_path, {**receipt, "status": "blocked",
                                       "termination_reason": "run_state_unavailable",
                                       "finished_at": now(), "why": str(exc)[:500]})
            raise

        def record_process_start(process: subprocess.Popen[Any]) -> None:
            process_identity_record.update(capture_process_identity(
                process, run_id=self.run_id, attempt_id=attempt_id, argv=safe_argv))
            receipt["process_identity"] = process_identity_record
            atomic_json(receipt_path, receipt)
            action = {"step": stage, "status": "running", "attempt_id": attempt_id,
                      "receipt_ref": receipt_ref, "log_ref": log_ref,
                      "process_identity": process_identity_record}
            self._record_shared_event(
                "stage_process_started",
                details={"stage": stage, "attempt_id": attempt_id,
                         "process_identity": process_identity_record},
                phase_state={"current_action": action,
                             "process_identity": process_identity_record},
                event_patch={"current_action": action,
                             "process_identity": process_identity_record})

        telemetry_writer = None
        telemetry_error = ""
        if stage == "train":
            try:
                telemetry_spec = self.declaration.get("telemetry_spec")
                if telemetry_spec is None:
                    telemetry_spec = (self.declaration.get("task_contract") or {}).get(
                        "telemetry_spec")
                telemetry_writer = telemetry.LiveTelemetryWriter(
                    output=self.output, run_root=self.run_root, repo=self.repo,
                    working_directory=Path(self.backend.directory(stage)),
                    stage_directory=stage_directory, log_path=attempt_log,
                    stage=stage, attempt_id=attempt_id, started_epoch=before,
                    telemetry_spec=telemetry_spec)
                run_record.refresh_research_projection(self.output, self.run_id)
            except Exception as exc:  # noqa: BLE001 - telemetry is never a stage precondition
                telemetry_error = f"writer_{type(exc).__name__}"
                self._append_local_event({"at": now(), "event": "telemetry_unavailable",
                                          "attempt_id": attempt_id,
                                          "stage": stage, "error": telemetry_error})
        try:
            try:
                execution_argv = isolated_argv(
                    argv, output=self.output, repo=self.repo, require_pid_namespace=True)
            except (OSError, ValueError) as exc:
                # An unavailable isolation boundary or invalid launcher is a refusal before
                # benchmark start; it must not be recorded as a failed benchmark attempt.
                attempt = ProcessAttempt(False, None, error=exc,
                                         containment_mode="unavailable")
                code, timed_out = None, False
                refused = f"{type(exc).__name__}: {exc}"
            else:
                receipt["containment_mode"] = requested_containment(execution_argv)
                atomic_json(receipt_path, receipt)
                try:
                    attempt = run_process(
                        execution_argv,
                        cwd=self.backend.directory(stage),
                        env={**os.environ, **surroundings}, timeout=timeout,
                        input=fed, stdout=stream, stderr=subprocess.STDOUT,
                        during=_live_stage_document(
                            self.run_root, telemetry_writer=telemetry_writer,
                            report_root=self.output, run_id=self.run_id),
                        on_start=record_process_start,
                        live_control=(lambda: read_json(self.job_control_path))
                        if self.job_control_path else None)
                    code, timed_out = attempt.returncode, attempt.timed_out
                    if attempt.cancelled:
                        refused = "job cancelled by controller"
                    if attempt.orphaned_children:
                        code = None
                        refused = "launcher exited while child processes were still running"
                    if attempt.error is not None:
                        refused = f"{type(attempt.error).__name__}: {attempt.error}"
                except (OSError, ResearchStateError, ValueError) as exc:
                    # run_process kills a started session before propagating observer
                    # failures. Only a nonpositive timeout can fail before its Popen call.
                    launched = timeout > 0
                    attempt = ProcessAttempt(
                        launched, None, error=exc,
                        containment_mode=receipt.get("containment_mode", "unavailable"))
                    code, timed_out = None, False
                    refused = (f"could not persist or execute process identity: "
                               f"{type(exc).__name__}: {exc}")
        finally:
            # Closed before the file is read: the last buffer of a long run is the part that
            # says why it ended.
            stream.close()
            aw.ended(self.run_root)
        telemetry_summary = None
        if telemetry_writer is not None:
            telemetry_status = ("timed_out" if timed_out else
                                "completed" if not refused and code == 0 else "failed")
            try:
                telemetry_state = telemetry_writer.finalize(telemetry_status)
                telemetry_summary = {
                    "status": telemetry_state.get("status"),
                    "sample_count": telemetry_state.get("sample_count", 0),
                    "metrics": [str(row.get("metric_name") or "unknown")
                                for row in telemetry_state.get("series") or []],
                    "data_ref": str(telemetry_writer.state_path.resolve().relative_to(
                        self.output.resolve())),
                    "chart_ref": str(telemetry_writer.chart_path.resolve().relative_to(
                        self.output.resolve())),
                    "errors": list(telemetry_state.get("errors") or []),
                }
            except Exception as exc:  # noqa: BLE001 - do not turn a successful train into failure
                telemetry_error = f"finalize_{type(exc).__name__}"
                telemetry_summary = {
                    "status": "unavailable",
                    "sample_count": telemetry_writer.sample_count,
                    "metrics": sorted(telemetry_writer.series)[:12],
                    "errors": [telemetry_error],
                }
        elif telemetry_error:
            telemetry_summary = {"status": "unavailable", "sample_count": 0,
                                 "metrics": [], "errors": [telemetry_error]}
        if refused:
            said = refused
        elif timed_out:
            said = (f"did not finish within {timeout}s\n--- what it produced ---\n"
                    + tail_of(log))
        else:
            said = tail_of(log)

        artifact = self.backend.check_artifact(stage, self.run_root / stage, since=before)
        if not artifact.get("matched"):
            # The benchmark may write somewhere the caller has no way to name. LIBERO's
            # training tree is built from config values and its own working directory, so the
            # artifact a stage promised appears beside the command rather than under the
            # output directory the caller chose. Look there too, and only at files this run
            # wrote: the checkout already holds every earlier run's output, and a check that
            # yesterday's checkpoint can satisfy is not a check.
            beside = self.backend.artifact_beside(stage, self.backend.directory(stage),
                                                  since=before)
            if beside.get("matched"):
                artifact = beside
        metric_result_artifact: dict[str, Any] = {"status": "not_declared", "matched": 0}
        if stage == "evaluate" and self.metric_spec.source in {"json", "csv"}:
            if self.metric_spec.artifact_pattern:
                metric_result_artifact = resolve_metric_artifact(
                    self.metric_spec,
                    roots={"working_directory": Path(str(self.backend.directory(stage))),
                           "output": stage_directory,
                           "policy_parent": policy_parent or
                           (self.run_root / "__missing_policy_parent__")},
                    started_at=before, allowed_roots=[self.repo, self.run_root])
            else:
                matched = int(artifact.get("matched") or 0)
                metric_result_artifact = {
                    "status": "matched" if matched == 1 else
                              "ambiguous" if matched > 1 else "missing",
                    "matched": matched, "source": "declared_stage_artifact",
                }
        record = {"stage": stage, "ran": attempt.launched, "argv": safe_argv,
                  "attempt_id": attempt_id, "returncode": code,
                  "started_epoch": before,
                  "containment_mode": attempt.containment_mode,
                  "process_identity": process_identity_record or None,
                  "status": "timed_out" if timed_out else
                            "failed" if refused or code != 0 else "completed",
                  "termination_reason": "cancelled" if attempt.cancelled else
                                        "timeout" if timed_out else
                                        "orphaned_children" if attempt.orphaned_children else
                                        "launch_error" if refused else
                                        "nonzero_exit" if code != 0 else "normal_exit",
                  # The command is argv *and* the shell it ran in, and only one of the two was
                  # written down. A stage that failed left its argv, its output and nothing
                  # about the environment -- so a failure on a missing module could not be
                  # told from a wrong `PATH`, and the run could not be re-created by hand to
                  # find out. What is kept is the part that differs from the ambient
                  # environment: the part the derivation established.
                  "environment": differing,
                  "working_directory": str(self.backend.directory(stage)),
                  "output_directory": str(stage_directory.resolve()),
                  "policy_output_root": str(policy_parent) if policy_parent else "",
                  "compute_environment": dict(decision.environment),
                  "gpu_device_uuid": _gpu_uuid,
                  "device": decision.device, "device_why": decision.why,
                  "seconds": round(time.monotonic() - started, 1),
                  "artifact": artifact,
                  "metric_result_artifact": metric_result_artifact,
                  # Every number the program gave a name to, so a later reader -- and a method
                  # that ranks candidates cheaply -- has more than one number at the end.
                  "readings": self._readings(said),
                  "said": said[-1500:]}
        if telemetry_summary is not None:
            record["telemetry"] = telemetry_summary
        if stage == "train" and self.require_training_progress:
            record["training_progress"] = training_progress(tail_of(log))
        media_capture_started = time.monotonic()
        captured_media = media_manifest.capture_attempt(
            self.run_root, stage_directory, since=before)
        # Some native evaluators put videos beside the frozen policy they consumed,
        # not in the stage output directory. `experiment_dir` is a run-owned policy
        # archive supplied by measure(); only that exact directory is additionally
        # scanned, and only fresh files can be attributed to this attempt.
        if stage == "evaluate" and policy_parent:
            captured_media.extend(media_manifest.capture_attempt(
                self.run_root, policy_parent, since=before))
        record["media"] = list({item["path"]: item for item in captured_media}.values())
        record["media_capture_seconds"] = round(
            max(0.0, time.monotonic() - media_capture_started), 3)
        if stage == "evaluate" or _protocol_guard:
            violation = self._protocol_violation()
            if violation:
                record.update(status="failed", termination_reason="protocol_changed",
                              why=violation)
        score_from_log = (stage == "evaluate" or _protocol_guard) and \
            self.metric_spec.source == "log"
        score_from_structured = (stage == "evaluate" and
                                 self.metric_spec.source in {"json", "csv"})
        # A log-scored evaluator may also declare a video/trajectory glob. That media is
        # useful evidence, but it is not the source of the score; a mistaken media glob
        # must not erase a valid native number. JSON/CSV scoring still requires its file.
        try:
            sealed_evidence = capture_attempt_evidence(
                self.output, attempt_id=attempt_id, log=attempt_log,
                receipt_ref=receipt_ref, status=str(record["status"]),
                returncode=code,
                termination_reason=str(record["termination_reason"]),
                artifact=artifact, metric_artifact=metric_result_artifact)
            record.update(evidence_id=sealed_evidence["evidence_id"],
                          evidence_ref=sealed_evidence["evidence_ref"])
        except (OSError, ValueError, TypeError) as exc:
            # An uninspectable native run cannot be used as a scientific score.
            record.update(status="failed", termination_reason="evidence_capture_failed",
                          evidence_error=f"{type(exc).__name__}: {exc}")
        canonical_log = (self.run_root / stage / "attempts" / attempt_id / "output.log")
        atomic_json(receipt_path, {**receipt, **record, "finished_at": now(),
                                   "log": str(canonical_log),
                                   "postconditions": [{"id": "declared_artifact",
                                                       "passed": (None if score_from_log or
                                                                  score_from_structured else
                                                                  bool(artifact.get("matched"))
                                                                  if artifact.get("checked")
                                                                  else None),
                                                       "required_for_score": not score_from_log
                                                       and not score_from_structured}]
                                   + ([{"id": "metric_result_artifact",
                                        "passed": metric_result_artifact.get("status") ==
                                        "matched", "required_for_score": True}]
                                      if score_from_structured else [])
                                   + ([{"id": "training_progress",
                                        "passed": record["training_progress"]["status"] ==
                                        "observed"}]
                                      if stage == "train" and self.require_training_progress
                                      else [])})
        if stage == "train":
            try:
                run_record.refresh_research_projection(self.output, self.run_id)
            except Exception:  # noqa: BLE001 - report failure cannot alter stage evidence
                pass
        if self.state_persistence_error:
            # Keep the failed attempt receipt, but do not continue to another stage/round
            # after losing the authoritative run-wide journal.
            raise ResearchStateError(self.state_persistence_error)
        if self.budget:
            self.budget.record()
        self._append_local_event({"at": now(), "event": "stage", **{k: record[k] for k in
                                                                      ("stage", "returncode",
                                                                       "seconds", "device",
                                                                       "device_why")}})
        action = {"step": stage, "attempt_id": attempt_id,
                  "outcome": record.get("status"), "returncode": record.get("returncode"),
                  "termination_reason": record.get("termination_reason"),
                  "seconds": record.get("seconds"), "receipt_ref": receipt_ref,
                  "finished_at": now()}
        self._record_shared_event(
            "stage_completed",
            details={"stage": stage, "attempt_id": attempt_id,
                     "status": record.get("status"),
                     "returncode": record.get("returncode"),
                     "termination_reason": record.get("termination_reason"),
                     "seconds": record.get("seconds"), "receipt_ref": receipt_ref,
                     "artifact_checked": (record.get("artifact") or {}).get("checked"),
                     "artifact_matches": (record.get("artifact") or {}).get("matched")},
            phase_state={"current_action": self.active_research_action or None,
                         "last_action": action,
                         "parent_action": self.active_research_action or None,
                         "process_identity": None},
            event_patch={"current_action": self.active_research_action or None,
                         "last_action": action,
                         "parent_action": self.active_research_action or None,
                         "process_identity": None})
        # Look at what was just written, before moving on. A record that contradicts itself
        # is cheapest to catch at the moment it is made.
        found = aw.observe(self.run_root, events=self.events,
                           measurements=sorted((self.run_root / "measurements").glob("*.json"))
                           if (self.run_root / "measurements").is_dir() else ())
        if found:
            for observation in found:
                self._append_local_event({"at": now(), "event": "observed",
                                          **observation.as_dict()})
            aw.record(self.run_root, found)
        # The document, rebuilt from what is now on disk. The observer above already reads the
        # records at this point, and this is the same reading assembled for a person instead of
        # for a check -- so a run in progress has a readable account of itself at every stage
        # boundary rather than only at the end, which is when a run that never ends has
        # nothing. No model call here: the written half is cached and refreshed at round
        # boundaries, so this costs a file rewrite and never a request.
        run_record.generate(self.run_root, title=f"{self.benchmark or self.run_id} / derived")
        return record

    # -- the protocol --------------------------------------------------------------------

    def collect(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Produce new trajectories, and say where they are.

        The proposal carries a collection half -- the validator reads its `enabled` flag, the
        space declares collection axes, and the protocol's notes say that half decides what
        data exists -- and this loop read none of it. The controller was handed a handle that
        did not exist.

        A proposal that asks for data and does not get it ends the round rather than falling
        back to the old data: measuring the old data under a hypothesis about new data records
        an answer to a question nobody asked, which is the failure this loop exists to avoid.
        """
        if not self.available("collect"):
            return {"ran": False, "why": "the derivation reports collect unavailable for this "
                                         "benchmark, so a proposal cannot ask for data"}
        started = time.time()
        result = self.run_stage("collect", settings=settings)
        if not result.get("ran") or result.get("returncode") != 0:
            return {**result, "ran": False,
                    "why": f"the collector did not finish ({result.get('returncode')})"}
        produced = self._recorded_artifact(result)
        if produced is None:
            beside = self.backend.artifact_beside("collect", self.backend.directory("collect"),
                                                  since=started)
            produced = Path(self.backend.directory("collect")) / beside["examples"][0] \
                if beside.get("matched") == 1 else None
        if produced is None:
            return {**result, "ran": False,
                    "why": "the collector finished and wrote nothing the derivation named"}
        return {**result, "ran": True, "data": str(produced)}

    def _prepare_once(self, settings: dict[str, Any]) -> None:
        """Run the benchmark's data-preparation stage, once per run.

        Once, because it turns what the benchmark ships into what its trainer reads and that
        does not depend on what is being varied -- running it per measurement would re-do the
        same conversion for every arm. Recorded, because what it produced is what the trainer
        is about to read. A failure here is not raised: the trainer that follows will say what
        is missing, and it says it better than this would.
        """
        if not self.available("prepare_data") or getattr(self, "_prepared", False):
            return
        self._prepared = True
        result = self.run_stage("prepare_data", settings=settings)
        self._append_local_event({"at": now(), "event": "prepared_data",
                                  "ran": bool(result.get("ran")),
                                  "returncode": result.get("returncode"),
                                  "seconds": result.get("seconds"),
                                  "why": result.get("why")})

    def capture_demo(self, *, trigger: str, measurement: dict[str, Any] | None,
                     settings: dict[str, Any]) -> dict[str, Any]:
        """Ask a declared native recording stage for one real demo and keep its lineage."""
        stage = str((self.declaration.get("task_contract") or {}).get("demo_stage") or
                    "record_demo")
        if not self.available(stage):
            return {"status": "unavailable", "why": "no native recording stage was derived"}
        policy = (measurement or {}).get("policy_artifact") or {}
        if not policy.get("path"):
            return {"status": "policy_unavailable",
                    "why": "a frozen measured policy is required for attributed demo capture"}
        policy_path = Path(str(policy.get("path") or ""))
        if not policy_path.is_absolute():
            policy_path = self.run_root / policy_path
        try:
            if not policy_path.resolve().is_relative_to(self.run_root):
                raise ValueError("recording policy is outside this run")
            identity_field, actual = artifact_identity(policy_path)
            if policy.get(identity_field) != actual:
                raise ValueError("recording policy does not match a frozen measurement")
        except (OSError, ValueError):
            return {"status": "policy_unavailable",
                    "why": "a frozen measured policy is required for attributed demo capture"}
        violation = self._protocol_violation()
        if violation:
            return {"status": "blocked", "why": violation}
        evaluation = (measurement or {}).get("evaluate") or {}
        evaluation_attempt_id = str(evaluation.get("attempt_id") or "")
        evaluation_receipt_sha256 = ""
        comparison_protocol_sha256 = ""
        if re.fullmatch(r"[0-9a-f]{32}", evaluation_attempt_id):
            evaluation_receipt = (self.run_root / "attempts" / evaluation_attempt_id /
                                  "receipt.json")
            try:
                if (not evaluation_receipt.is_symlink() and
                        evaluation_receipt.resolve(strict=True).is_relative_to(self.run_root)):
                    held_receipt = read_json(evaluation_receipt)
                    if (isinstance(held_receipt, dict) and
                            held_receipt.get("run_id") == self.run_id and
                            held_receipt.get("attempt_id") == evaluation_attempt_id and
                            held_receipt.get("node_id") == "evaluate"):
                        evaluation_receipt_sha256 = digest(evaluation_receipt)
                        protocol_hash = held_receipt.get("comparison_protocol_sha256")
                        if isinstance(protocol_hash, str) and re.fullmatch(
                                r"[0-9a-f]{64}", protocol_hash):
                            comparison_protocol_sha256 = protocol_hash
            except (OSError, ValueError, TypeError, RuntimeError):
                pass
        measurement_settings = (measurement or {}).get("settings") or {}
        task = (settings.get("task") or settings.get("task_name") or
                measurement_settings.get("task") or measurement_settings.get("task_name"))
        seed = (settings["seed"] if "seed" in settings else
                measurement_settings.get("seed"))
        capture_started = time.monotonic()
        ran = self.run_stage(stage, settings=settings, checkpoint=str(policy_path),
                             experiment_dir=str(policy_path),
                             previous_artifact=str(policy_path))
        capture_seconds = round(max(0.0, time.monotonic() - capture_started), 3)
        media = ran.get("media") or []
        after = self._protocol_violation()
        status = ("captured" if ran.get("status") == "completed" and media and not after
                  else "protocol_changed" if after else "no_media_or_stage_failed")
        row = {"at": now(), "trigger": trigger, "status": status,
               "stage": stage, "attempt_id": ran.get("attempt_id"),
               "evaluation_attempt_id": evaluation_attempt_id or None,
               "evaluation_receipt_sha256": evaluation_receipt_sha256 or None,
               "comparison_protocol_sha256": comparison_protocol_sha256 or None,
               "policy_sha256": actual, "policy_path": str(policy_path),
               "measurement_label": (measurement or {}).get("label"),
               "task": task, "seed": seed, "episode_id": None,
               "episode_identity_status": "not_reported_by_native_recording_stage",
               "settings": dict(settings), "media": media,
               "seconds": capture_seconds, "stage_seconds": ran.get("seconds"),
               "media_capture_seconds": ran.get("media_capture_seconds"),
               "why": after or ran.get("why") or ""}
        path = self.run_root / "media_events.json"
        try:
            old = read_json(path) if path.is_file() else {}
        except (OSError, ValueError):
            old = {}
        atomic_json(path, {"schema_version": 1, "rows": list(old.get("rows") or []) + [row]})
        run_record.generate(self.run_root,
                            title=f"{self.benchmark or self.run_id} / derived")
        return row

    def confirm(self, *, settings: dict[str, Any] | None = None,
                label: str = "confirmation") -> dict[str, Any]:
        """Evaluate the frozen winning policy on the reserved settings, without training."""
        held_out = self._held_out()
        if self._held_out_error:
            return {"label": label, "ok": False, "where": "confirmation",
                    "why": self._held_out_error, "metric_value": None, "ran": False}
        if not held_out:
            return {"label": label, "ok": False, "where": "confirmation",
                    "why": "no held-out settings were declared, so there is nothing this run "
                           "can confirm on; its best number stays the maximum of the arms it "
                           "measured",
                    "metric_value": None, "ran": False}
        existing = _load_confirmation(self.run_root)
        if existing:
            return {"label": label, "ok": False, "where": "confirmation",
                    "why": f"this run already confirmed at {existing.get('label')} "
                           f"({existing.get('metric_value')}); a second one would be a search "
                           f"of the held-out set",
                    "metric_value": None, "ran": False}
        for attempt_path in (self.run_root / "confirmation_attempts").glob("*.json"):
            try:
                prior = read_json(attempt_path)
            except (OSError, ValueError):
                continue
            if (isinstance(prior, dict) and
                    ((prior.get("baseline_evaluate") or {}).get("attempt_id") or
                     prior.get("attempt_id"))):
                return {"label": label, "ok": False, "where": "confirmation",
                        "why": "the held-out evaluator was already launched in a failed "
                               "confirmation attempt; repeating it would search the "
                               "held-out set", "metric_value": None, "ran": False}
        frozen = self._frozen_search_settings()
        if frozen is None:
            return {"label": label, "ok": False, "where": "confirmation",
                    "why": "no comparison protocol has been frozen, so there is no search to "
                           "confirm against", "metric_value": None, "ran": False}
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
            raise ValueError("confirmation label must be a short path-safe identifier")
        confirmation_settings = {**frozen, **held_out}
        if settings and any(confirmation_settings.get(key) != value for key, value in
                            settings.items()):
            return {"label": label, "ok": False, "where": "confirmation",
                    "why": "confirmation settings may only use the frozen held-out protocol",
                    "metric_value": None, "ran": False}

        def recorded(result: dict[str, Any]) -> dict[str, Any]:
            row = {"schema_version": 1, "label": label, "held_out": held_out,
                   "settings": confirmation_settings, "at": now(),
                   "candidate_label": result.get("candidate_label"),
                   "policy_artifact": result.get("policy_artifact") or {},
                   "source_state": result.get("source_state") or {},
                   "baseline_source_state": result.get("baseline_source_state") or {},
                   "source_isolation": result.get("source_isolation") or {},
                   "baseline_metric_value": result.get("baseline_metric_value"),
                   "baseline_evaluate": result.get("baseline_evaluate") or {},
                   "comparison": result.get("confirmation_comparison") or {},
                   "metric_value": result.get("metric_value"),
                   "metric_utility": self._metric_utility(result),
                   "ok": bool(result.get("ok")),
                   "attempt_id": (result.get("evaluate") or {}).get("attempt_id"),
                   "why": result.get("why") or ""}
            # Each failed exposure is durable even if a later attempt succeeds.
            immutable_json(self.run_root / "confirmation_attempts" /
                           f"{uuid.uuid4().hex}.json", row)
            atomic_json(self.run_root / "confirmation.json", row)
            atomic_json(self.run_root / "measurements" / f"{label}.json", result)
            run_record.generate(self.run_root,
                                title=f"{self.benchmark or self.run_id} / derived")
            return result

        if self.execution_graph:
            return recorded({"label": label, "ok": False, "where": "confirmation",
                             "why": "graph confirmation of a frozen policy is not supported",
                             "settings": confirmation_settings, "metric_value": None,
                             "ran": False, "confirmation": True})
        candidates: list[tuple[str, dict[str, Any]]] = []
        for path in sorted((self.run_root / "measurements").glob("*.json")):
            if path.stem == label:
                continue
            try:
                row = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(row, dict) and row.get("ok") is True and row.get("metric_value") is not None:
                candidates.append((path.stem, row))
        best = self.snapshots.best()
        selected = (best.name.replace("-", "_", 1) if best is not None else "")
        if best is not None and best.scale == "progress":
            selected = ""
        if not selected and candidates and best is None:
            selected = max(candidates, key=lambda one: (
                self._metric_utility(one[1]) if self._metric_utility(one[1]) is not None
                else float("-inf")))[0]
        winner = next((row for name, row in candidates if name == selected), None)
        if winner is None:
            return recorded({"label": label, "ok": False, "where": "confirmation",
                             "why": "there is no scored frozen best candidate to confirm",
                             "settings": confirmation_settings, "metric_value": None,
                             "ran": False, "confirmation": True})
        policy = winner.get("policy_artifact") or {}
        policy_path = Path(str(policy.get("path") or ""))
        if not policy_path.is_absolute():
            policy_path = self.run_root / policy_path
        def frozen_policy_matches(path: Path, archived: dict[str, Any]) -> bool:
            if not path.resolve().is_relative_to(self.run_root):
                return False
            try:
                field, actual = artifact_identity(path)
            except (OSError, ValueError):
                return False
            return bool(archived.get(field)) and archived[field] == actual

        def frozen_source_matches(row: dict[str, Any]) -> bool:
            state = row.get("source_state") if isinstance(row.get("source_state"), dict) else {}
            workspace_path = self.output / "workspace_snapshot.json"
            if (not state or workspace_path.is_symlink() or not workspace_path.is_file()):
                return False
            try:
                workspace = read_json(workspace_path)
                name = str(state.get("source_snapshot") or "")
                overlay = state.get("overlay_files")
                expected = object_digest({
                    "schema_version": 1,
                    "source_tree_fingerprint": workspace.get("source_tree_fingerprint"),
                    "source_snapshot": name,
                    "overlay_files": overlay,
                    "overlay_modes": state.get("overlay_modes") or {},
                    "overlay_directories": state.get("overlay_directories") or {},
                    "copy_mode": state.get("copy_mode", "full_worktree"),
                    "untracked_assets_omitted": bool(
                        state.get("untracked_assets_omitted")),
                }) if name and isinstance(overlay, dict) else ""
                snapshot = self.snapshots.get(name) if name else None
                if (workspace.get("schema_version") != 2 or
                    workspace.get("destination") != str(self.repo) or
                    state.get("source_tree_fingerprint") !=
                    workspace.get("source_tree_fingerprint") or
                    not expected or state.get("identity_sha256") != expected or
                    snapshot is None or dict(snapshot.files) != overlay):
                    return False
                for key in snapshot.files.values():
                    if key:
                        blob = self.snapshots.blobs / key
                        if not blob.is_file() or digest(blob) != key:
                            return False
                if str(row.get("scored") or "").startswith("a source-defined controller"):
                    source_policy = row.get("source_policy_artifact") or {}
                    if (source_policy.get("kind") != "source_tree" or
                            source_policy.get("identity_sha256") != expected):
                        return False
                return True
            except (OSError, TypeError, ValueError, AttributeError):
                return False

        winner_source_state = bool(winner.get("source_state"))
        baseline_row = next((row for name, row in candidates if name == "baseline"), None)
        baseline_policy = (baseline_row.get("policy_artifact") or {}) if baseline_row else {}
        baseline_source_state = bool((baseline_row or {}).get("source_state"))
        if (winner_source_state != baseline_source_state or
                (winner_source_state and
                 (not frozen_source_matches(winner) or
                  not frozen_source_matches(baseline_row or {})))):
            return recorded({"label": label, "ok": False, "where": "source_identity",
                             "why": "baseline and selected policy do not both have matching, "
                                    "content-addressed source states for independent confirmation",
                             "candidate_label": selected, "settings": confirmation_settings,
                             "metric_value": None, "ran": False, "confirmation": True})
        source_state_confirmation = winner_source_state and baseline_source_state
        winner_is_source = str(winner.get("scored") or "").startswith(
            "a source-defined controller")
        if (not frozen_policy_matches(policy_path, policy) if policy else
                not (winner_is_source and source_state_confirmation)):
            return recorded({"label": label, "ok": False, "where": "policy_identity",
                             "why": "the selected candidate's frozen policy identity cannot be verified",
                             "candidate_label": selected, "settings": confirmation_settings,
                             "metric_value": None, "ran": False, "confirmation": True})
        if baseline_row is None:
            return recorded({"label": label, "ok": False, "where": "baseline_identity",
                             "why": "no scored frozen baseline is available for held-out comparison",
                             "candidate_label": selected, "settings": confirmation_settings,
                             "metric_value": None, "ran": False, "confirmation": True})
        baseline_path = Path(str(baseline_policy.get("path") or ""))
        if not baseline_path.is_absolute():
            baseline_path = self.run_root / baseline_path
        baseline_is_source = str(baseline_row.get("scored") or "").startswith(
            "a source-defined controller")
        if (not frozen_policy_matches(baseline_path, baseline_policy) if baseline_policy else
                not (baseline_is_source and source_state_confirmation)):
            return recorded({"label": label, "ok": False, "where": "baseline_identity",
                             "why": "the frozen baseline policy identity cannot be verified",
                             "candidate_label": selected, "settings": confirmation_settings,
                             "metric_value": None, "ran": False, "confirmation": True})
        # The evaluator may import changed benchmark files. Refuse if the winning snapshot
        # is no longer the source state in the checkout; a later round cannot silently
        # confirm an earlier policy under different code.
        if best is not None and not source_state_confirmation:
            for relative, key in best.files.items():
                current = self.repo / relative
                if (digest(current) if current.is_file() else "") != key:
                    return recorded({"label": label, "ok": False, "where": "source_identity",
                                     "why": "the selected candidate's source snapshot is not "
                                            "the current checkout state",
                                     "candidate_label": selected,
                                     "settings": confirmation_settings,
                                     "metric_value": None, "ran": False,
                                     "confirmation": True})
            baseline_snapshot = self.snapshots.get("baseline")
            if selected != "baseline" and best.files and (
                    baseline_snapshot is None or baseline_snapshot.files != best.files):
                return recorded({"label": label, "ok": False,
                                 "where": "baseline_source_identity",
                                 "why": "baseline and selected candidate require different "
                                        "source states; confirmation cannot evaluate both "
                                        "fairly in one mutable checkout",
                                 "candidate_label": selected,
                                 "settings": confirmation_settings,
                                 "metric_value": None, "ran": False,
                                 "confirmation": True})
        violation = self._comparison_protocol_violation(confirmation_settings,
                                                        target="evaluate", confirmation=True)
        if violation:
            return recorded({"label": label, "ok": False, "where": "comparison_protocol",
                             "why": violation, "candidate_label": selected,
                             "settings": confirmation_settings, "metric_value": None,
                             "ran": False, "confirmation": True})
        def in_frozen_source_state(row: dict[str, Any], role: str,
                                   action: Any) -> dict[str, Any]:
            state = row.get("source_state") if isinstance(row.get("source_state"), dict) else {}
            if not state:
                return action()
            from .workspace_snapshot import apply_state, create, verify_state

            workspace_path = self.output / "workspace_snapshot.json"
            workspace = read_json(workspace_path)
            parent = self.output / "confirmation_sources" / f"{role}-{uuid.uuid4().hex}"
            checkout = parent / "checkout"
            copied = create(Path(workspace["source"]), checkout,
                            max_bytes=max(1, int(workspace["bytes"])),
                            resources=workspace.get("resource_bindings", []),
                            tracked_only=workspace.get("copy_mode") == "tracked_worktree")
            if copied.get("source_tree_fingerprint") != state.get("source_tree_fingerprint"):
                raise ValueError("the confirmed source base differs from the measured base")
            apply_state(checkout, state, self.snapshots)
            if not verify_state(checkout, copied, state, self.snapshots):
                raise ValueError("the reconstructed confirmation source differs from its identity")

            original_repo, original_backend, original_sources = self.repo, self.backend, self.sources
            old_root, new_root = str(original_repo), str(checkout)

            def rebase(value: Any) -> Any:
                if isinstance(value, str):
                    return value.replace(old_root, new_root)
                if isinstance(value, list):
                    return [rebase(item) for item in value]
                if isinstance(value, dict):
                    return {key: rebase(item) for key, item in value.items()}
                return value

            sources = rebase(original_backend.sources)
            self.repo = checkout
            self.sources = sources
            self.backend = DeclarativeBackend(
                repo=checkout, answer=rebase(original_backend.answer), sources=sources,
                parameters=rebase(original_backend.parameters))
            try:
                result = action()
                if not verify_state(checkout, copied, state, self.snapshots):
                    return {**result, "ok": False, "metric_value": None,
                            "why": "source tree changed during independent confirmation"}
                result["source_state"] = state
                result["source_isolation"] = "fresh_copy_from_verified_base_and_overlay"
                return result
            finally:
                self.repo, self.backend, self.sources = (
                    original_repo, original_backend, original_sources)

        def score_frozen(path: Path, role: str,
                         measured_row: dict[str, Any]) -> dict[str, Any]:
            def evaluate() -> dict[str, Any]:
                scored = self.run_stage("evaluate", settings=confirmation_settings,
                                        checkpoint=str(path) if path else "",
                                        experiment_dir=str(path) if path else "",
                                        previous_artifact=str(path) if path else "",
                                        confirmation=True)
                metric_artifact, metric_artifact_evidence = self._metric_artifact_for_record(
                    scored)
                result_archive: dict[str, Any] = {}
                if self.metric_spec.source in {"json", "csv"} and metric_artifact is not None:
                    try:
                        cap = int(os.environ.get("AUTOSIM_RESULT_COPY_LIMIT_BYTES",
                                                str(64 * 1024**2)))
                        if cap <= 0 or not metric_artifact.is_file():
                            raise ValueError("result copy requires a positive limit and one file")
                        result_archive = freeze_artifact(
                            metric_artifact, self.run_root / "experiments" / label / role /
                            str(scored.get("attempt_id") or uuid.uuid4().hex) /
                            f"result{metric_artifact.suffix}", max_bytes=cap)
                    except (OSError, ValueError, RuntimeError):
                        result_archive = {}
                if self.metric_spec.source in {"json", "csv"}:
                    self._attach_metric_artifact_evidence(scored, metric_artifact_evidence,
                                                          result_archive)
                reading = self.metric_spec.read(
                    said=scored.get("said", ""),
                    artifact=Path(result_archive["path"]) if result_archive else None)
                ok = (bool(scored.get("ran")) and scored.get("returncode") == 0
                      and scored.get("status") == "completed"
                      and (self.metric_spec.source == "log" or
                           metric_artifact_evidence.get("status") == "matched")
                      and reading.get("value") is not None)
                from .metric_guardrails import read_values
                secondary = read_values(getattr(self, "guardrail_specs", []),
                    said=scored.get("said", ""), artifact=Path(result_archive["path"]) if result_archive else None,
                    primary_spec=self.metric_spec)
                return {"ok": ok, "metric_value": reading.get("value") if ok else None,
                        "secondary_metric_readings": secondary,
                        "metric_reading": reading, "result_artifact": result_archive,
                        "metric_artifact_evidence": metric_artifact_evidence,
                        "evaluate": {"attempt_id": scored.get("attempt_id"),
                                     "returncode": scored.get("returncode"),
                                     "said": str(scored.get("said") or "")[-800:]}}

            try:
                return in_frozen_source_state(measured_row, role, evaluate)
            except (OSError, TypeError, ValueError, RuntimeError, KeyError) as exc:
                return {"ok": False, "metric_value": None, "metric_reading": {},
                        "result_artifact": {}, "evaluate": {},
                        "why": f"source-isolated confirmation could not run: "
                               f"{type(exc).__name__}: {exc}"}

        baseline_result = score_frozen(baseline_path, "baseline", baseline_row)
        if selected == "baseline":
            candidate_result = baseline_result
        elif baseline_result["ok"]:
            candidate_result = score_frozen(policy_path, "candidate", winner)
        else:
            candidate_result = {"ok": False, "metric_value": None,
                                "metric_reading": {}, "result_artifact": {},
                                "evaluate": {}}
        ok = baseline_result["ok"] and candidate_result["ok"]
        from .metric_guardrails import compare as compare_guardrails
        confirmation_guardrails = compare_guardrails(getattr(self, "guardrail_specs", []),
            candidate_result.get("secondary_metric_readings") or {},
            baseline_result.get("secondary_metric_readings") or {})
        improvement = (self.metric_spec.utility(candidate_result["metric_value"]) -
                       self.metric_spec.utility(baseline_result["metric_value"])) if ok else None
        comparison: dict[str, Any] = {
            "utility_difference": improvement, "verdict": "not_established",
            "why": "paired native episode evidence and uncertainty analysis are required "
                   "before claiming improvement"}
        if ok and self.metric_spec.episode_id_column and self.metric_spec.initial_state_hash_column:
            left = baseline_result["metric_reading"]
            right = candidate_result["metric_reading"]
            left_keys = list(left.get("episode_keys") or [])
            right_keys = list(right.get("episode_keys") or [])
            left_states = list(left.get("initial_state_hashes") or [])
            right_states = list(right.get("initial_state_hashes") or [])
            left_values = list(left.get("episode_values") or [])
            right_values = list(right.get("episode_values") or [])
            if (left_keys and set(left_keys) == set(right_keys) and
                    len(left_keys) == len(left_states) == len(left_values) and
                    len(right_keys) == len(right_states) == len(right_values)):
                baseline_episodes = dict(zip(left_keys, zip(left_states, left_values)))
                candidate_episodes = dict(zip(right_keys, zip(right_states, right_values)))
                same_states = all(baseline_episodes[key][0] == candidate_episodes[key][0]
                                  for key in left_keys)
                heldout_states = set(left_states)
                development_states: set[str] = set()
                development_verified = True
                for _, development in candidates:
                    archived_result = development.get("result_artifact") or {}
                    result_path = Path(str(archived_result.get("path") or ""))
                    if not result_path.is_absolute():
                        result_path = self.run_root / result_path
                    if (not result_path.resolve().is_relative_to(self.run_root) or
                            not result_path.is_file() or
                            not archived_result.get("content_sha256") or
                            digest(result_path) != archived_result["content_sha256"]):
                        development_verified = False
                        break
                    reread = self.metric_spec.read(said="", artifact=result_path)
                    if (reread.get("value") != development.get("metric_value") or
                            not reread.get("initial_state_hashes")):
                        development_verified = False
                        break
                    development_states.update(reread["initial_state_hashes"])
                if not same_states:
                    comparison["why"] = "baseline and candidate do not share native initial-state hashes"
                elif not development_verified:
                    comparison["why"] = "development initial-state evidence cannot be verified from frozen result bytes"
                elif heldout_states & development_states:
                    comparison["why"] = "held-out initial states overlap development measurements"
                else:
                    baseline_values = [baseline_episodes[key][1] for key in left_keys]
                    candidate_values = [candidate_episodes[key][1] for key in left_keys]
                    if self.metric_spec.direction == "minimize":
                        baseline_values = [-value for value in baseline_values]
                        candidate_values = [-value for value in candidate_values]
                    statistical = compare(
                        baseline_values, candidate_values,
                        kind="binary" if self.metric_spec.name == "success_rate" else "continuous",
                        paired=True).as_dict()
                    comparison = {"utility_difference": improvement,
                                  "verdict": "not_established",
                                  "statistical_evidence": statistical,
                                  "paired_initial_states": len(left_keys),
                                  "why": "paired statistics are available, but native initial-state "
                                         "hash provenance and full protocol semantics have not "
                                         "been independently audited"}
            else:
                comparison["why"] = "held-out baseline and candidate lack matching native episodes"
        return recorded({"label": label, "ok": ok,
                         "where": "" if ok else "evaluate",
                         "why": "" if ok else
                         "held-out baseline or candidate did not produce a verified metric",
                         "confirmation": True, "candidate_label": selected,
                         "settings": confirmation_settings,
                         "policy_artifact": policy,
                         "source_state": candidate_result.get("source_state") or {},
                         "baseline_source_state": baseline_result.get("source_state") or {},
                         "source_isolation": {
                             "baseline": baseline_result.get("source_isolation") or
                                         "mutable_checkout",
                             "candidate": candidate_result.get("source_isolation") or
                                           "mutable_checkout"},
                         "evaluated_policy_path": str(policy_path),
                         "train": {"ran": False},
                         "baseline_policy_artifact": baseline_policy,
                         "baseline_metric_value": baseline_result["metric_value"],
                         "baseline_metric_reading": baseline_result["metric_reading"],
                         "baseline_evaluate": baseline_result["evaluate"],
                         "evaluate": candidate_result["evaluate"],
                         "result_artifact": candidate_result["result_artifact"],
                         "metric": self.metric_spec.as_dict(),
                         "metric_reading": candidate_result["metric_reading"],
                         "metric_value": candidate_result["metric_value"],
                         "metric_utility": self.metric_spec.utility(
                             candidate_result["metric_value"]) if ok else None,
                         "confirmation_comparison": (comparison if confirmation_guardrails["status"] == "passed" else
                             {**comparison, "verdict": "not_established", "why": "secondary metric guardrails are missing or violated"}),
                         "guardrails": confirmation_guardrails,
                         "success_rate": candidate_result["metric_value"] if ok and
                         self.metric_spec.name == "success_rate" else None})

    def confirmation_state(self) -> dict[str, Any]:
        """What is known about the held-out measurement without taking one.

        For the report of a run that was not asked to confirm: whether it could have, whether
        it already did, and -- when it could have and did not -- that the run's best number is
        therefore still the maximum of its arms.
        """
        try:
            row = read_json(self.run_root / "confirmation.json")
        except (OSError, ValueError):
            row = None
        row = row if isinstance(row, dict) else {}
        if row.get("ok"):
            return {"status": "taken", "label": row.get("label"),
                    "held_out": row.get("held_out") or {},
                    "metric_value": row.get("metric_value"),
                    "baseline_metric_value": row.get("baseline_metric_value"),
                    "candidate_label": row.get("candidate_label"),
                    "comparison": row.get("comparison") or {},
                    "at": row.get("at", "")}
        if row:
            # Attempted and did not produce a number. Not the same as never having tried, and
            # not the same as having confirmed. An evaluator actually launched on held-out
            # states spends the split even if it failed before returning a usable score.
            exposed = bool((row.get("baseline_evaluate") or {}).get("attempt_id") or
                           row.get("attempt_id"))
            return {"status": "attempted_and_failed", "held_out": row.get("held_out") or {},
                    "why": row.get("why") or "the held-out measurement produced no number",
                    "retryable": not exposed}
        held_out = self._held_out()
        if self._held_out_error:
            return {"status": "unavailable", "why": self._held_out_error}
        if not held_out:
            return {"status": "not_declared",
                    "why": "this run declared no held-out settings, so it has no episodes the "
                           "search did not see and nothing to confirm on"}
        return {"status": "available_not_taken", "held_out": held_out,
                "why": "a held-out set was declared and the run did not measure it, so the "
                       "best number above is the maximum of the search's own arms"}

    def _frozen_search_settings(self) -> dict[str, Any] | None:
        """The settings the search measured at, as frozen before it started."""
        path = self.run_root / "comparison_protocol.json"
        try:
            frozen = read_json(path).get("settings") if path.is_file() else None
        except (OSError, ValueError):
            return None
        return dict(frozen) if isinstance(frozen, dict) else None

    def measure(self, *, settings: dict[str, Any], label: str, dataset: str | None = None,
                confirmation: bool = False, _reuse_training_attempt_id: str = "",
                _baseline_recovery: bool = False,
                _reuse_policy_artifact: dict[str, Any] | None = None) -> dict[str, Any]:
        """One arm of the run, noted in the selection record on the way out.

        Wrapped rather than instrumented. `_measure` returns from eight places -- a stage that
        could not run, a trainer that failed, an artifact that could not be frozen, a policy
        that was never produced -- and every one of them is an arm the loop took. A note
        written at each return is a note missed at the ninth, and the arms that go missing
        that way are the failures, which is the direction that makes a search look smaller
        than it was and its best number look better than it is.

        So the counting happens where every return has to pass, whatever `_measure` decides.
        """
        if not confirmation and any((self.run_root / "confirmation_attempts").glob("*.json")):
            return self._comparison_refusal(
                label, settings, "search cannot resume after the held-out set was exposed")
        measurement_started = time.monotonic()
        result = self._measure(settings=settings, label=label, dataset=dataset,
                               confirmation=confirmation,
                               reuse_training_attempt_id=_reuse_training_attempt_id,
                               reuse_policy_artifact=_reuse_policy_artifact)
        from .scheduling import policy, note
        if policy(self.output):
            note(self.output, kind="formal_measurement", identity=label,
                 seconds=time.monotonic()-measurement_started, valid=bool(result.get("ok")),
                 confirmation=confirmation, metric_value=result.get("metric_value"),
                 elapsed_since_run_start=max(0, time.time()-self.budget.started_epoch) if self.budget else None)
        try:
            protocol = self.run_root / "comparison_protocol.json"
            selection.note(
                self.run_root,
                protocol_sha256=(digest(protocol) if protocol.is_file() else ""),
                label=label, started=True,
                metric_value=result.get("metric_value"),
                metric_utility=self._metric_utility(result),
                baseline=label == "baseline" or _baseline_recovery,
                settings=self._frozen_search_settings() or result.get("settings") or settings,
                metric_kind=("binary" if self.metric_spec.name == "success_rate"
                             else "continuous"),
                actual_episodes=(result.get("metric_reading") or {}).get("episodes_completed"),
                successes=(result.get("metric_reading") or {}).get("successes"))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            # The number was measured; the accounting for it is not worth the measurement.
            # A run that loses its count has under-reported its own search, and that is a
            # better outcome than one that loses the measurement instead.
            self._note("selection_record_unwritten",
                       [{"label": label, "why": f"{type(exc).__name__}: {exc}"}])
        self._checkpoint_candidate_measurement(label=label, result=result)
        return result

    def _checkpoint_candidate_measurement(self, *, label: str,
                                         result: dict[str, Any]) -> bool:
        """Record the exact point where a candidate metric exists but is not yet adopted.

        This is deliberately a phase checkpoint, not a commit: after this method the caller
        still has rubric, best-state, demo, source rollback, idea-library and history work to
        do. Recovery may verify this measurement but must not treat it as a completed round.
        """
        match = re.fullmatch(r"round_([1-9][0-9]*)", label)
        if not match:
            return False
        session_path = self.run_root / "controller_session.json"
        if not session_path.exists():
            return False
        if session_path.is_symlink():
            raise ResearchStateError("candidate measurement checkpoint session is a symlink")
        measurement_path = self.run_root / "measurements" / f"{label}.json"
        if measurement_path.is_symlink():
            raise ResearchStateError("candidate measurement checkpoint is a symlink")
        if not measurement_path.is_file():
            return False
        try:
            session = read_json(session_path)
            stored_result = read_json(measurement_path)
        except (OSError, ValueError, TypeError) as exc:
            raise ResearchStateError("candidate measurement checkpoint cannot be read: "
                                     f"{type(exc).__name__}: {exc}") from exc
        if (not isinstance(session, dict) or session.get("schema_version") != 1 or
                session.get("run_id") != self.run_id or
                Path(str(session.get("repository") or "")).expanduser().resolve() !=
                self.repo):
            raise ResearchStateError("candidate measurement checkpoint session identity is invalid")
        pending = session.get("pending_action")
        round_index = int(match.group(1))
        if (session.get("status") != "running" or session.get("action") != label or
                not isinstance(pending, dict) or pending.get("round") != round_index):
            return False
        expected_result_hash = object_digest(result)
        if (not isinstance(stored_result, dict) or stored_result.get("label") != label or
                object_digest(stored_result) != expected_result_hash):
            raise ResearchStateError("candidate measurement file does not match the returned result")
        phase = str(pending.get("phase") or "")
        measurement_hash = digest(measurement_path)
        if phase == "measurement_recorded":
            if (pending.get("measurement_sha256") != measurement_hash or
                    pending.get("result_sha256") != expected_result_hash):
                raise ResearchStateError("candidate measurement checkpoint conflicts with its file")
            finalization = pending.get("round_finalization")
            if isinstance(finalization, dict) and (
                    finalization.get("measurement_sha256") != measurement_hash or
                    finalization.get("result_sha256") != expected_result_hash):
                raise ResearchStateError(
                    "candidate finalization journal conflicts with its measurement")
            return True
        if phase != "measurement_starting":
            return False
        idea = pending.get("idea") if isinstance(pending.get("idea"), dict) else {}
        transaction_id = object_digest({
            "run_id": self.run_id, "round": round_index,
            "idea_label": idea.get("label"),
            "measurement_sha256": measurement_hash,
            "result_sha256": expected_result_hash,
        })[:32]
        finalization = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "status": "in_progress",
            "round": round_index,
            "idea_label": str(idea.get("label") or ""),
            "measurement_ref": f"measurements/{label}.json",
            "measurement_sha256": measurement_hash,
            "result_sha256": expected_result_hash,
            "steps": {name: {"status": "pending"}
                      for name in _ROUND_FINALIZATION_STEPS},
            "created_at": now(),
        }
        pending = dict(pending)
        pending.update(
            phase="measurement_recorded",
            measurement_ref=f"measurements/{label}.json",
            measurement_sha256=measurement_hash,
            result_sha256=expected_result_hash,
            measurement_ok=bool(result.get("ok")),
            metric_value=result.get("metric_value"),
            round_finalization=finalization,
            recorded_at=now())
        session.update(pending_action=pending, updated_at=now())
        atomic_json(session_path, session)
        return True

    def _update_candidate_finalization(self, *, label: str, step: str, status: str,
                                       evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write-ahead one post-measurement side effect without guessing after a crash.

        `started` is durable before the effect; `completed` is written only after the caller
        returns. A restart that finds `started` cannot infer whether an external side effect
        happened and must leave that step unresolved rather than replay it.
        """
        if step not in _ROUND_FINALIZATION_STEPS:
            raise ValueError(f"unknown candidate-finalization step: {step}")
        if status not in {"started", "completed", "skipped", "unknown"}:
            raise ValueError(f"invalid candidate-finalization state: {status}")
        match = re.fullmatch(r"round_([1-9][0-9]*)", label)
        if not match:
            raise ValueError("candidate-finalization label is invalid")
        session_path = self.run_root / "controller_session.json"
        if session_path.is_symlink() or not session_path.is_file():
            raise ResearchStateError("candidate finalization session is missing or unsafe")
        try:
            session = read_json(session_path)
        except (OSError, ValueError, TypeError) as exc:
            raise ResearchStateError("candidate finalization session cannot be read: "
                                     f"{type(exc).__name__}: {exc}") from exc
        if (not isinstance(session, dict) or session.get("schema_version") != 1 or
                session.get("run_id") != self.run_id or
                Path(str(session.get("repository") or "")).expanduser().resolve() !=
                self.repo):
            raise ResearchStateError("candidate finalization session identity is invalid")
        pending = session.get("pending_action")
        round_index = int(match.group(1))
        idea = pending.get("idea") if isinstance(pending, dict) else None
        if (session.get("status") != "running" or session.get("action") != label or
                not isinstance(pending, dict) or pending.get("round") != round_index or
                pending.get("phase") != "measurement_recorded" or
                not isinstance(idea, dict)):
            raise ResearchStateError("candidate finalization is outside its recorded phase")
        finalization = pending.get("round_finalization")
        if (not isinstance(finalization, dict) or
                finalization.get("schema_version") != 1 or
                finalization.get("round") != round_index or
                finalization.get("idea_label") != str(idea.get("label") or "") or
                finalization.get("measurement_sha256") != pending.get("measurement_sha256") or
                finalization.get("result_sha256") != pending.get("result_sha256") or
                not re.fullmatch(r"[a-f0-9]{32}",
                                 str(finalization.get("transaction_id") or ""))):
            raise ResearchStateError("candidate finalization journal identity is invalid")
        steps = finalization.get("steps")
        if not isinstance(steps, dict) or set(steps) != set(_ROUND_FINALIZATION_STEPS):
            raise ResearchStateError("candidate finalization journal steps are malformed")
        row = steps.get(step)
        if not isinstance(row, dict) or row.get("status") not in {
                "pending", "started", "completed", "skipped", "unknown"}:
            raise ResearchStateError("candidate finalization step record is malformed")
        previous = str(row.get("status"))
        encoded = json.dumps(dict(evidence or {}), ensure_ascii=False, sort_keys=True,
                             default=str)
        if len(encoded) > 4000:
            safe_evidence = {"sha256": object_digest(encoded),
                             "truncated": True}
        else:
            safe_evidence = json.loads(encoded)
        if status == "started":
            if previous == "started":
                return session
            if previous != "pending":
                raise ResearchStateError(
                    f"cannot start finalization step {step} from {previous}")
            row = {"status": "started", "started_at": now()}
        elif status == "skipped":
            if previous == "skipped":
                return session
            if previous != "pending":
                raise ResearchStateError(
                    f"cannot skip finalization step {step} from {previous}")
            row = {"status": "skipped", "evidence": safe_evidence, "finished_at": now()}
        elif status == "completed":
            if previous == "completed":
                if object_digest(row.get("evidence") or {}) != object_digest(safe_evidence):
                    raise ResearchStateError(
                        f"completed finalization step {step} has conflicting evidence")
                return session
            if previous != "started":
                raise ResearchStateError(
                    f"cannot complete finalization step {step} from {previous}")
            row = {"status": "completed", "evidence": safe_evidence,
                   "finished_at": now(), "started_at": row.get("started_at")}
        else:  # unknown
            if previous == "unknown":
                return session
            if previous != "started":
                raise ResearchStateError(
                    f"cannot mark finalization step {step} unknown from {previous}")
            row = {"status": "unknown", "evidence": safe_evidence,
                   "finished_at": now(), "started_at": row.get("started_at")}
        steps = dict(steps)
        steps[step] = row
        finalization = dict(finalization)
        finalization["steps"] = steps
        finalization["status"] = (
            "unresolved" if any(one.get("status") == "unknown"
                                 for one in steps.values()) else
            "postprocessing_complete" if all(one.get("status") in {"completed", "skipped"}
                                             for one in steps.values()) else
            "in_progress")
        pending = dict(pending)
        pending["round_finalization"] = finalization
        session.update(pending_action=pending, updated_at=now())
        atomic_json(session_path, session)
        return session

    def _run_candidate_finalization_step(self, *, label: str, step: str,
                                         action: Any) -> Any:
        """Run one journaled controller-side effect, leaving uncertain work unresolved."""
        session_path = self.run_root / "controller_session.json"
        if not session_path.exists():
            # The legacy all-in-one research call has no outer bounded-action session. The
            # write-ahead contract applies to the resumable main-controller path only.
            return action()
        self._update_candidate_finalization(label=label, step=step, status="started")
        try:
            result = action()
        except BaseException as exc:
            try:
                self._update_candidate_finalization(
                    label=label, step=step, status="unknown",
                    evidence={"exception": type(exc).__name__,
                              "because": redact(str(exc))[:500]})
            except Exception:
                pass
            raise
        try:
            safe_result = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            safe_result = str(result)
        evidence: dict[str, Any] = {
            "result_type": type(result).__name__,
            "result_sha256": object_digest(safe_result),
        }
        if isinstance(safe_result, dict):
            summary = {key: safe_result[key] for key in (
                "status", "outcome", "ok", "attempt_id", "returncode", "metric_value",
                "success_rate", "seconds", "why", "advanced", "sha256", "event_id",
                "event_sha256", "sequence", "decision_id", "outcome_state", "undone",
                "paths", "remaining_hashes", "restored_paths", "times_tried",
                "idea_status", "answered_questions") if key in result}
            if summary:
                evidence["summary"] = summary
                evidence["summary_sha256"] = object_digest(summary)
            encoded_result = json.dumps(safe_result, ensure_ascii=False, sort_keys=True)
            if len(encoded_result) <= 2500:
                evidence["result"] = safe_result
        elif isinstance(result, (str, int, float, bool)) or result is None:
            evidence["value"] = result
        self._update_candidate_finalization(label=label, step=step,
                                            status="completed", evidence=evidence)
        return result

    def _skip_candidate_finalization_step(self, *, label: str, step: str,
                                          because: str) -> None:
        if not (self.run_root / "controller_session.json").exists():
            return
        self._update_candidate_finalization(
            label=label, step=step, status="skipped", evidence={"because": because[:500]})

    def _candidate_finalization_id(self, *, label: str) -> str:
        session_path = self.run_root / "controller_session.json"
        if not session_path.is_file() or session_path.is_symlink():
            return ""
        try:
            session = read_json(session_path)
            pending = session.get("pending_action") if isinstance(session, dict) else None
            finalization = (pending.get("round_finalization")
                            if isinstance(pending, dict) else None)
            if (isinstance(pending, dict) and pending.get("phase") == "measurement_recorded" and
                    pending.get("round") == int(label.removeprefix("round_")) and
                    isinstance(finalization, dict)):
                return str(finalization.get("transaction_id") or "")
        except (OSError, ValueError, TypeError):
            pass
        return ""

    @staticmethod
    def verify_committed_candidate_finalization(*, run_root: Path, run_id: str,
                                                repo: Path, round_index: int) -> dict[str, Any]:
        """Read-only proof that one candidate round's controller transaction committed.

        This is intentionally stricter than finding a measurement file. It binds the durable
        session row, exact measurement bytes, all post-measurement step receipts, source
        rollback result where applicable, and the native evaluation receipt. It performs no
        replay and writes nothing; callers may use it to project a committed session after an
        outer controller was interrupted.
        """
        root = Path(run_root)
        repository = Path(repo).resolve()
        if (root.is_symlink() or not root.is_dir() or round_index < 1 or
                not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", run_id)):
            return {"status": "unverified", "because": "run identity or root is unsafe"}
        root = root.resolve()

        def safe_file(relative: str) -> Path:
            raw = root / relative
            if raw.is_symlink():
                raise ValueError(f"record is a symlink: {relative}")
            current = root
            for part in Path(relative).parts[:-1]:
                current = current / part
                if current.is_symlink():
                    raise ValueError(f"record parent is a symlink: {relative}")
            resolved = raw.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError(f"record is missing or escapes run root: {relative}")
            return resolved

        def read_record(relative: str) -> Any:
            return read_json(safe_file(relative))

        try:
            session = read_record("controller_session.json")
            if (not isinstance(session, dict) or session.get("schema_version") != 1 or
                    session.get("run_id") != run_id or
                    Path(str(session.get("repository") or "")).expanduser().resolve() !=
                    repository):
                raise ValueError("controller session identity does not match")
            if session.get("pending_action") is not None:
                raise ValueError("controller session still has an uncommitted pending action")
            if not isinstance(session.get("history"), list):
                raise ValueError("controller history is malformed")
            matching = [row for row in session.get("round_finalizations", [])
                        if isinstance(row, dict) and row.get("round") == round_index]
            if len(matching) != 1:
                raise ValueError("there is not exactly one finalization for this round")
            finalization = matching[0]
            if (finalization.get("status") != "committed" or
                    finalization.get("schema_version") != 1 or
                    finalization.get("idea_label") == ""):
                raise ValueError("round finalization is not committed or has no candidate identity")
            transaction_id = str(finalization.get("transaction_id") or "")
            if not re.fullmatch(r"[a-f0-9]{32}", transaction_id):
                raise ValueError("round finalization transaction ID is malformed")
            measurement_path = safe_file(f"measurements/round_{round_index}.json")
            measurement = read_json(measurement_path)
            measurement_hash = digest(measurement_path)
            result_hash = object_digest(measurement)
            if (not isinstance(measurement, dict) or
                    measurement.get("label") != f"round_{round_index}" or
                    finalization.get("measurement_sha256") != measurement_hash or
                    finalization.get("result_sha256") != result_hash):
                raise ValueError("saved measurement does not match the committed journal")
            expected_id = object_digest({
                "run_id": run_id, "round": round_index,
                "idea_label": finalization.get("idea_label"),
                "measurement_sha256": measurement_hash,
                "result_sha256": result_hash,
            })[:32]
            if expected_id != transaction_id:
                raise ValueError("finalization ID does not bind the candidate measurement")
            history_rows = [row for row in session.get("history", [])
                            if isinstance(row, dict) and row.get("round") == round_index and
                            row.get("label") == f"round_{round_index}" and
                            row.get("finalization_id") == transaction_id]
            if len(history_rows) != 1:
                raise ValueError("controller history lacks the committed candidate row")
            history_row = history_rows[0]
            history_hash = object_digest(history_row)
            if (finalization.get("history_row_sha256") != history_hash or
                    (finalization.get("steps") or {}).get(
                        "history_session_commit", {}).get("evidence", {}).get(
                            "history_row_sha256") != history_hash):
                raise ValueError("committed history row hash does not match its journal")
            steps = finalization.get("steps")
            if (not isinstance(steps, dict) or
                    set(steps) != set(_ROUND_FINALIZATION_STEPS) or
                    any(not isinstance(row, dict) or
                        row.get("status") not in {"completed", "skipped"}
                        for row in steps.values())):
                raise ValueError("one or more controller finalization steps are unresolved")
            from .receipt_verifier import verify_measurement
            receipt_proof = verify_measurement(root, f"round_{round_index}")
            if receipt_proof.get("status") != "consistent":
                raise ValueError("measurement receipt verification is not consistent: " +
                                 str(receipt_proof.get("status") or "unknown"))

            def result_of(step_name: str) -> Any:
                step = steps.get(step_name) or {}
                if step.get("status") == "skipped":
                    return None
                evidence = step.get("evidence") or {}
                if not isinstance(evidence, dict):
                    raise ValueError(f"{step_name} has no completed result evidence")
                summary = evidence.get("summary")
                if summary is not None and (
                        not isinstance(summary, dict) or
                        object_digest(summary) != evidence.get("summary_sha256")):
                    raise ValueError(f"{step_name} summary evidence hash is invalid")
                if "result" not in evidence:
                    if isinstance(summary, dict):
                        return summary
                    raise ValueError(f"{step_name} has no completed result evidence")
                result = evidence["result"]
                if object_digest(result) != evidence.get("result_sha256"):
                    raise ValueError(f"{step_name} result evidence hash is invalid")
                return result

            best_result = result_of("best_state")
            if not isinstance(best_result, dict):
                raise ValueError("best-state result evidence is missing")
            snapshots_path = safe_file("snapshots/snapshots.json")
            if digest(snapshots_path) != best_result.get("sha256"):
                raise ValueError("snapshot/best record changed since finalization")
            rubric_result = result_of("rubric_final")
            if (not isinstance(rubric_result, dict) or
                    digest(safe_file("rubric.json")) != rubric_result.get("sha256")):
                raise ValueError("final rubric record changed since finalization")

            event_result = result_of("round_event")
            if not isinstance(event_result, dict):
                raise ValueError("round event evidence is missing")
            events = read_record("events.json")
            event_rows = events.get("rows") if isinstance(events, dict) else None
            if not isinstance(event_rows, list) or not any(
                    isinstance(row, dict) and row.get("event_id") ==
                    event_result.get("event_id") and row.get("event_sha256") ==
                    event_result.get("event_sha256") for row in event_rows):
                raise ValueError("round event is absent from the local event ledger")

            idea_result = result_of("idea_outcome")
            if not isinstance(idea_result, dict):
                raise ValueError("idea outcome evidence is missing")
            ideas = read_record("ideas.json")
            idea_rows = ideas.get("ideas") if isinstance(ideas, dict) else None
            matching_ideas = [row for row in idea_rows or []
                              if isinstance(row, dict) and row.get("label") ==
                              finalization.get("idea_label")]
            if (len(matching_ideas) != 1 or
                    idea_result.get("event_id") not in
                    matching_ideas[0].get("outcome_event_ids", []) or
                    idea_result.get("times_tried") != matching_ideas[0].get("times_tried")):
                raise ValueError("idea outcome ledger does not match finalization evidence")

            proposal_result = result_of("proposal_resolution")
            if proposal_result is not None:
                if not isinstance(proposal_result, dict):
                    raise ValueError("proposal-decision result evidence is malformed")
                decisions = read_record("decisions.json")
                decision_rows = decisions.get("rows") if isinstance(decisions, dict) else None
                if not any(isinstance(row, dict) and
                           row.get("id") == proposal_result.get("decision_id") and
                           (row.get("outcome") or {}).get("state") == "known"
                           for row in decision_rows or []):
                    raise ValueError("proposal decision outcome is absent from decision log")

            rollback_result = result_of("source_rollback")
            if isinstance(rollback_result, dict):
                hashes = rollback_result.get("remaining_hashes") or {}
                if not isinstance(hashes, dict):
                    raise ValueError("source rollback hash record is malformed")
                for relative, expected_hash in hashes.items():
                    relative_path = Path(str(relative))
                    target = repository / relative_path
                    if (relative_path.is_absolute() or
                            target.is_symlink() or
                            not target.resolve().is_relative_to(repository)):
                        raise ValueError("source rollback path escapes repository")
                    if any(parent.is_symlink() for parent in
                           [repository, *(repository / relative_path).parents]):
                        raise ValueError("source rollback path contains a symlink")
                    if not target.is_file() or digest(target) != expected_hash:
                        raise ValueError("source changed after the recorded rollback")

            demo_result = result_of("demo_capture")
            if isinstance(demo_result, dict) and demo_result.get("attempt_id"):
                attempt_id = str(demo_result["attempt_id"])
                if not re.fullmatch(r"[a-f0-9]{32}", attempt_id):
                    raise ValueError("demo attempt ID is malformed")
                receipt = read_record(f"attempts/{attempt_id}/receipt.json")
                if (not isinstance(receipt, dict) or receipt.get("run_id") != run_id or
                        receipt.get("attempt_id") != attempt_id or
                        receipt.get("status") not in {"completed", "failed", "blocked",
                                                       "timed_out"}):
                    raise ValueError("demo attempt receipt is not terminal")
                media = read_record("media_events.json")
                media_rows = media.get("rows") if isinstance(media, dict) else None
                if not any(isinstance(row, dict) and row.get("attempt_id") == attempt_id
                           for row in media_rows or []):
                    raise ValueError("demo attempt has no media-event record")
            return {"status": "verified_committed", "round": round_index,
                    "transaction_id": transaction_id,
                    "history_row": history_row, "finalization": finalization,
                    "receipt_proof": receipt_proof}
        except (OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
            return {"status": "unverified", "because": redact(str(exc))[:500]}

    def _verified_training_receipt(self, attempt_id: str,
                                   settings: dict[str, Any]) -> dict[str, Any] | None:
        """Load one completed train receipt only when its identity still binds this run."""
        if not re.fullmatch(r"[a-f0-9]{32}", attempt_id):
            return None
        root = self.run_root.resolve()
        raw = root / "attempts" / attempt_id / "receipt.json"
        if raw.is_symlink():
            return None
        path = raw.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return None
        try:
            receipt = read_json(path)
            protocol_path = self.run_root / "comparison_protocol.json"
            cwd = Path(str(receipt.get("working_directory") or "")).resolve()
            argv = receipt.get("argv")
            recorded_protocol = str(receipt.get("comparison_protocol_sha256") or "")
            if (not isinstance(receipt, dict) or receipt.get("attempt_id") != attempt_id or
                    receipt.get("run_id") != self.run_id or
                    receipt.get("node_id") != "train" or receipt.get("stage") != "train" or
                    receipt.get("status") != "completed" or receipt.get("returncode") != 0 or
                    receipt.get("termination_reason") != "normal_exit" or
                    receipt.get("settings_digest") != object_digest(settings) or
                    not protocol_path.is_file() or
                    (recorded_protocol and recorded_protocol != digest(protocol_path)) or
                    not cwd.is_relative_to(self.repo.resolve()) or
                    not isinstance(argv, list) or len(argv) < 2 or
                    str(argv[0]) != str(self.interpreter)):
                return None
            artifact = receipt.get("artifact") or {}
            progress = receipt.get("training_progress") or {}
            if (artifact.get("checked") is not True or
                    not isinstance(artifact.get("matched"), int) or
                    artifact.get("matched", 0) < 1 or
                    progress.get("status") != "observed"):
                return None
            return receipt
        except (OSError, TypeError, ValueError, AttributeError):
            return None

    def recover_unscored_baseline(self) -> dict[str, Any]:
        """Finish a baseline from its exact train receipt without training a second time.

        Supported boundaries are a source-selection abstention and an evaluation refused
        before process start for a temporary GPU resource conflict. The latter is eligible
        only while the same physical GPU is available and the exact frozen policy still
        matches the completed training attempt.
        """
        if self.execution_graph or self.available("train") is False:
            return {"ok": False, "where": "recovery", "ran": False,
                    "why": "receipt-based policy recovery is not supported for this workflow"}
        measurement_path = self.run_root / "measurements" / "baseline.json"
        try:
            current_measurement = read_json(measurement_path)
        except (OSError, TypeError, ValueError) as exc:
            return {"ok": False, "where": "recovery", "ran": False,
                    "why": f"baseline measurement is unavailable: {type(exc).__name__}"}
        if (not isinstance(current_measurement, dict) or
                current_measurement.get("label") != "baseline" or
                current_measurement.get("ok") is True):
            return {"ok": False, "where": "recovery", "ran": False,
                    "why": "baseline is not a failed measurement"}
        original = current_measurement
        retry_after_archive_failure = False
        if current_measurement.get("where") == "policy_archive":
            recovery = current_measurement.get("recovery") or {}
            attempt_hint = str(recovery.get("training_attempt_id") or "")
            if (recovery.get("kind") != "reevaluate_after_prestart_resource_block" or
                    recovery.get("training_reused") is not True or
                    not re.fullmatch(r"[a-f0-9]{32}", attempt_hint) or
                    not str(current_measurement.get("archive_error") or "").startswith(
                        "FileExistsError:") or current_measurement.get("evaluate")):
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "failed policy archival is not an approved evaluation-only retry"}
            attempt_root = self.run_root / "attempts" / attempt_hint
            backup_path = attempt_root / "baseline_unscored.json"
            prior_state_path = attempt_root / "baseline_recovery.json"
            try:
                backup_resolved = backup_path.resolve(strict=True)
                backup_resolved.relative_to(self.run_root.resolve())
                original = read_json(backup_resolved)
                prior_state = read_json(prior_state_path)
                current_digest = object_digest(current_measurement)
                original_digest = object_digest(original)
                stored_original_digest = (prior_state.get("original_measurement_sha256")
                                          if isinstance(prior_state, dict) else None)
                if (backup_path.is_symlink() or not isinstance(original, dict) or
                        original.get("where") != "evaluate" or
                        original.get("ok") is True or
                        not isinstance(prior_state, dict) or
                        prior_state.get("status") != "failed" or
                        prior_state.get("recovery_kind") !=
                        "reevaluate_after_prestart_resource_block" or
                        prior_state.get("training_attempt_id") != attempt_hint or
                        (stored_original_digest is not None and
                         stored_original_digest != original_digest) or
                        prior_state.get("measurement_sha256") != current_digest or
                        prior_state.get("evaluation_attempt_id") or
                        recovery.get("original_measurement_ref") !=
                        f"attempts/{attempt_hint}/baseline_unscored.json" or
                        recovery.get("superseded_evaluation_attempt_id") !=
                        (original.get("evaluate") or {}).get("attempt_id") or
                        (original.get("train") or {}).get("attempt_id") != attempt_hint or
                        current_measurement.get("evaluate") or
                        (current_measurement.get("recovery") or {}).get(
                            "training_attempt_id") != attempt_hint or
                        (current_measurement.get("recovery") or {}).get(
                            "training_reused") is not True):
                    raise ValueError("prior recovery is not a verified pre-evaluation archive failure")
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": f"prior recovery cannot be safely retried: {type(exc).__name__}: {exc}"}
            retry_after_archive_failure = True
        resource_retry = original.get("where") == "evaluate"
        policy_retry = (original.get("where") == "policy_artifact" and
                        original.get("status") == "unscored")
        if not (resource_retry or policy_retry):
            return {"ok": False, "where": "recovery", "ran": False,
                    "why": "baseline failure is not a supported recovery boundary"}
        outcome = original.get("training_outcome") or {}
        train_outcome = original.get("train") or {}
        attempt_id = str(outcome.get("attempt_id") or original.get("attempt_id") or
                         (train_outcome.get("attempt_id") if resource_retry else ""))
        settings = original.get("settings")
        if not isinstance(settings, dict):
            return {"ok": False, "where": "recovery", "ran": False,
                    "why": "failed baseline has no settings bound to its train receipt"}
        blocked_evaluation_id = ""
        if resource_retry:
            if (train_outcome.get("attempt_id") != attempt_id or
                    train_outcome.get("returncode") != 0):
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "failed measurement does not identify one completed train attempt"}
            evaluation = original.get("evaluate") or {}
            blocked_evaluation_id = str(evaluation.get("attempt_id") or "")
            if not re.fullmatch(r"[a-f0-9]{32}", blocked_evaluation_id):
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "failed evaluation has no exact resource-block receipt identity"}
            raw_block = self.run_root / "attempts" / blocked_evaluation_id / "receipt.json"
            try:
                blocked_path = raw_block.resolve(strict=True)
                blocked_path.relative_to(self.run_root.resolve())
                blocked_receipt = read_json(blocked_path)
            except (OSError, ValueError, TypeError, RuntimeError):
                blocked_receipt = {}
            if (raw_block.is_symlink() or not isinstance(blocked_receipt, dict) or
                    blocked_receipt.get("run_id") != self.run_id or
                    blocked_receipt.get("attempt_id") != blocked_evaluation_id or
                    blocked_receipt.get("node_id") != "evaluate" or
                    blocked_receipt.get("status") != "blocked" or
                    blocked_receipt.get("ran") is not False or
                    blocked_receipt.get("command_started") is not False or
                    blocked_receipt.get("process_identity") or
                    blocked_receipt.get("termination_reason") not in {
                        "gpu_unavailable", "gpu_lease_unavailable", "gpu_lease_busy"} or
                    blocked_receipt.get("settings_digest") != object_digest(settings)):
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "evaluation was not proven refused before process start"}
        trained = self._verified_training_receipt(attempt_id, settings)
        if trained is None:
            return {"ok": False, "where": "recovery", "ran": False,
                    "why": "completed training receipt no longer matches run, settings, "
                           "protocol, interpreter, or checkout identity"}
        recovery_kind = "revalidate_policy_selection"
        if policy_retry:
            selection_info = original.get("artifact_selection") or {}
            selection_ref = str(selection_info.get("selection_ref") or "")
            raw_selection = self.output / selection_ref
            try:
                selection_path = raw_selection.resolve()
                selection_doc = read_json(selection_path)
            except (OSError, TypeError, ValueError):
                selection_doc = {}
                selection_path = Path()
            if (raw_selection.is_symlink() or not selection_path.is_relative_to(
                    self.run_root.resolve()) or not isinstance(selection_doc, dict) or
                    selection_doc.get("attempt_id") != attempt_id or
                    selection_doc.get("status") != "abstained"):
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "the failed selection record is missing, changed, or not an "
                               "abstention; no policy can be safely rebound"}
        else:
            recovery_kind = "reevaluate_after_prestart_resource_block"
            from .devices import normalize_uuid
            selected_device = ((self.compute.evidence.get("selected_device") or {})
                               if self.compute is not None else {})
            train_uuid = str(trained.get("gpu_device_uuid") or "")
            current_uuid = str(selected_device.get("uuid") or "")
            if (self.compute is None or not self.compute.on_gpu or not train_uuid or
                    not current_uuid or normalize_uuid(train_uuid) !=
                    normalize_uuid(current_uuid)):
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "the exact training GPU is not currently available for evaluation"}
            policy = original.get("policy_artifact") or {}
            raw_policy = Path(str(policy.get("path") or ""))
            try:
                policy_path = raw_policy.resolve(strict=True)
                policy_path.relative_to(self.run_root.resolve())
                identity_field, actual_identity = artifact_identity(policy_path)
            except (OSError, ValueError, RuntimeError):
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "the exact frozen baseline policy is missing or unsafe"}
            if (raw_policy.is_symlink() or not policy_path.is_file() or
                    policy.get(identity_field) != actual_identity):
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "the frozen baseline policy identity changed"}
            source_path = Path(str(policy.get("source") or ""))
            if not source_path.is_absolute():
                source_path = Path(str(trained.get("working_directory") or self.repo)) / source_path
            try:
                actual_source = source_path.resolve(strict=True)
                cwd = Path(str(trained.get("working_directory") or "")).resolve()
                candidates = (trained.get("artifact") or {}).get("candidate_paths") or []
                exact_candidates = {
                    (Path(candidate) if Path(candidate).is_absolute() else cwd / candidate)
                    for candidate in candidates if isinstance(candidate, str)}
                exact_candidates = {candidate.resolve(strict=True)
                                    for candidate in exact_candidates}
            except (OSError, ValueError, TypeError, RuntimeError):
                exact_candidates = set()
                actual_source = Path()
            if actual_source not in exact_candidates:
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "the frozen baseline policy is not one of the exact fresh "
                               "outputs named by its train receipt"}

        session_path = self.run_root / "controller_session.json"
        report_path = self.run_root / "research_report.json"
        try:
            session = read_json(session_path)
            report = read_json(report_path)
            history = session.get("history") if isinstance(session, dict) else None
            rows = report.get("rounds") if isinstance(report, dict) else None
            session_status = session.get("status") if isinstance(session, dict) else None
            report_status = report.get("run_status") if isinstance(report, dict) else None
            if (not isinstance(session, dict) or not isinstance(report, dict) or
                    session.get("run_id") != self.run_id or
                    report.get("run_id") != self.run_id or
                    session_status not in {"paused", "completed"} or
                    report_status != session_status or
                    not isinstance(history, list) or not history or
                    not isinstance(rows, list) or not rows or
                    not isinstance(history[0], dict) or not isinstance(rows[0], dict) or
                    history[0].get("label") != "baseline" or
                    rows[0].get("label") != "baseline" or
                    any(row.get("measured") is True or
                        isinstance(row.get("metric_value"), (int, float))
                        for row in history[1:] if isinstance(row, dict))):
                raise ValueError("controller state is not recoverable: it must contain the "
                                 "same paused/completed unscored baseline and no later score")
        except (OSError, TypeError, ValueError, AttributeError) as exc:
            return {"ok": False, "where": "recovery", "ran": False,
                    "why": f"baseline controller state cannot be safely resumed: "
                           f"{type(exc).__name__}: {exc}"}

        attempt_dir = self.run_root / "attempts" / attempt_id
        recovery_state_path = (attempt_dir / "baseline_recovery_attempt_2.json"
                               if retry_after_archive_failure else
                               attempt_dir / "baseline_recovery.json")
        backup_path = attempt_dir / "baseline_unscored.json"
        if measurement_path.is_file():
            current = read_json(measurement_path)
            recovered = (current.get("recovery") or {}) if isinstance(current, dict) else {}
            if (isinstance(current, dict) and current.get("ok") is True and
                    recovered.get("training_attempt_id") == attempt_id and
                    recovered.get("training_reused") is True):
                return current
        if recovery_state_path.is_file():
            try:
                prior_state = read_json(recovery_state_path)
            except (OSError, TypeError, ValueError):
                prior_state = {}
            return {"ok": False, "where": "recovery", "ran": False,
                    "why": "this bounded baseline recovery attempt already exists; inspect its stage "
                           "receipt before any further action",
                    "recovery_state": prior_state}
        if retry_after_archive_failure:
            if not backup_path.is_file() or backup_path.is_symlink():
                return {"ok": False, "where": "recovery", "ran": False,
                        "why": "the immutable unscored baseline backup is missing or unsafe"}
        else:
            immutable_json(backup_path, original)
        if any((self.run_root / "confirmation_attempts").glob("*.json")):
            return {"ok": False, "where": "recovery", "ran": False,
                    "why": "held-out evaluation was already exposed; baseline recovery is "
                           "not permitted"}

        atomic_json(recovery_state_path, {
            "schema_version": 1, "status": "running", "label": "baseline",
            "recovery_attempt": 2 if retry_after_archive_failure else 1,
            "training_attempt_id": attempt_id,
            "recovery_kind": recovery_kind,
            "blocked_evaluation_attempt_id": blocked_evaluation_id,
            "settings_digest": object_digest(settings),
            "original_measurement_sha256": object_digest(original), "started_at": now(),
        })
        self._set_research_action("recover_unscored_baseline", details={
            "training_attempt_id": attempt_id, "training_reused": True})
        result = self.measure(settings=settings, label="baseline",
                              _reuse_training_attempt_id=attempt_id,
                              _baseline_recovery=True,
                              _reuse_policy_artifact=(original.get("policy_artifact")
                                                      if resource_retry else None))
        result["recovery"] = {
            "training_attempt_id": attempt_id, "training_reused": True,
            "kind": recovery_kind,
            "recovery_attempt": 2 if retry_after_archive_failure else 1,
            "superseded_evaluation_attempt_id": blocked_evaluation_id,
            "original_measurement_ref":
                f"attempts/{attempt_id}/baseline_unscored.json",
        }
        atomic_json(measurement_path, result)
        atomic_json(recovery_state_path, {
            "schema_version": 1, "status": "completed" if result.get("ok") else "failed",
            "recovery_attempt": 2 if retry_after_archive_failure else 1,
            "label": "baseline", "training_attempt_id": attempt_id,
            "training_reused": True,
            "recovery_kind": recovery_kind,
            "superseded_evaluation_attempt_id": blocked_evaluation_id,
            "evaluation_attempt_id": (result.get("evaluate") or {}).get("attempt_id"),
            "measurement_sha256": object_digest(result), "finished_at": now(),
        })
        if not result.get("ok"):
            self._set_research_action("recover_unscored_baseline", status="completed",
                                      details={"training_attempt_id": attempt_id,
                                               "training_reused": True,
                                               "measurement_ok": False})
            return result

        prior_failure = {key: history[0][key] for key in ("why_not", "failure")
                         if key in history[0]}
        recovered_row = {**history[0], "measured": True, "status": "measured",
                         "metric_value": result.get("metric_value"),
                         "metric_utility": result.get("metric_utility"),
                         "success_rate": result.get("success_rate"),
                         "recovered_from": {
                             "measurement_ref":
                                 f"research/{self.run_id}/attempts/{attempt_id}/"
                                 "baseline_unscored.json",
                             "training_attempt_id": attempt_id,
                             "training_reused": True,
                             "original_failure": prior_failure,
                         }}
        recovered_row.pop("failure", None)
        recovered_row.pop("why_not", None)
        history[0] = recovered_row
        rows[0] = recovered_row
        baseline_rate = self._metric_utility(result)
        current_label = str(session.get("current_measurement_label") or "baseline")
        session.update(history=history, baseline_rate=baseline_rate,
                       current_measurement_label=current_label,
                       status=session_status,
                       action=(session.get("action") or "") if session_status == "paused"
                       else session.get("action", ""), updated_at=now())
        self._advance_best("baseline", score=baseline_rate,
                           scale=("success_rate" if self.metric_spec.name == "success_rate"
                                  else "metric:" + self.metric_spec.name),
                           why="recovered by evaluating the already completed baseline train "
                               "attempt without retraining")
        rubric = self.build_rubric(on_event=self._note)
        self.rubric = objective.evaluate(
            rubric, self.rubric_facts(rounds=history, baseline=baseline_rate,
                                      current=result))
        self._write_rubric()
        best = self.snapshots.best()
        report.update(rounds=rows, best=best.as_dict() if best else None,
                      objective=self.rubric.as_dict(), run_status=report_status,
                      next_round=session.get("next_round", report.get("next_round")))
        atomic_json(session_path, session)
        atomic_json(report_path, report)
        self._append_local_event({
            "at": now(), "event": "unscored_baseline_recovered",
            "training_attempt_id": attempt_id,
            "evaluation_attempt_id": (result.get("evaluate") or {}).get("attempt_id"),
            "training_reused": True, "metric_value": result.get("metric_value"),
            "original_measurement_ref": f"attempts/{attempt_id}/baseline_unscored.json",
        })
        self._set_research_action("recover_unscored_baseline", status="completed",
                                  details={"training_attempt_id": attempt_id,
                                           "evaluation_attempt_id":
                                               (result.get("evaluate") or {}).get("attempt_id"),
                                           "training_reused": True,
                                           "measurement_ok": True})
        run_record.generate(self.run_root, title=f"{self.benchmark or self.run_id} / derived")
        return result

    def _measure(self, *, settings: dict[str, Any], label: str, dataset: str | None = None,
                 confirmation: bool = False,
                 reuse_training_attempt_id: str = "",
                 reuse_policy_artifact: dict[str, Any] | None = None) -> dict[str, Any]:
        """Train and score, which is what one arm of a comparison consists of.

        The two stages are run in the order the benchmark needs them, and the second is
        given the first's artifact -- the checkpoint a trainer wrote is the checkpoint an
        evaluator scores, and carrying it is the runner's job rather than a name the
        derivation has to guess.
        """
        if self.execution_graph:
            if dataset:
                return {"label": label, "ok": False, "where": "graph_dataset_binding",
                        "why": "external dataset override is not declared as a graph edge",
                        "metric_value": None}
            return self.measure_graph(self.execution_graph,
                                      target=str(self.execution_graph["score_target"]),
                                      settings=settings, label=label)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
            raise ValueError("measurement label must be a short path-safe identifier")
        violation = self._comparison_protocol_violation(settings, target="evaluate",
                                                        freeze=True,
                                                        confirmation=confirmation)
        if violation:
            return self._comparison_refusal(label, settings, violation)
        # Before anything is started: a measurement needs both halves, and the cost of
        # discovering that afterwards is whatever the first half costs. RoboTwin's trainer is
        # six thousand epochs and hours of GPU; with no command derived for `evaluate`, the
        # loop began that training anyway and would have recorded no number at the end of it.
        # A stage that cannot run is a finding, and a finding does not need a day of compute
        # to be true.
        #
        # **A benchmark that ships a checkpoint can be scored without training one.** The
        # evaluation is then the whole measurement, and what it measures is the benchmark's
        # own policy -- which is a *baseline*, not a candidate: no round can improve on it by
        # being measured against it, because the round would be measuring the same shipped
        # weights again. What it buys is the measurement rung: a benchmark with an evaluator
        # and a released policy produces its own number instead of reporting that it has no
        # number, and the run knows what there is to beat. The record says which of the two
        # it scored, because a reader comparing two runs has to know.
        scored_what = ("a checkpoint this run trained" if self.available("train")
                       else "the checkpoint the benchmark ships" if self.checkpoint
                       else "a source-defined controller with no weight artifact"
                       if self.source_policy else "")
        missing = [name for name in ("train", "evaluate") if not self.available(name)]
        if missing and not ((self.checkpoint or self.source_policy) and
                            self.available("evaluate") and missing == ["train"]):
            why = (f"{' and '.join(missing)} cannot be run: no command was derived for "
                   f"{'it' if len(missing) == 1 else 'them'}. A measurement needs both halves, "
                   f"so the first was not started.")
            failed = {"label": label, "ok": False, "where": missing[0],
                      "settings": dict(settings), "success_rate": None,
                      "ran": False, "why": why}
            (self.run_root / "measurements").mkdir(parents=True, exist_ok=True)
            atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
            self._append_local_event({"at": now(), "event": "measurement_not_started",
                                      "label": label, "missing": missing, "why": why})
            return failed
        # The step before the trainer, when the benchmark has one and has not yet had it run.
        # `prepare_data`'s role is "turn the benchmark's shipped demonstrations into the form
        # its trainer reads" -- and the loop had it available and never invoked it, so a
        # benchmark whose trainer reads a table that step generates failed on a missing key
        # that no command could supply. RoboTwin's did, twice, with the diagnosis right both
        # times: "the config that would hold it is a file, not a field, and the stage must be
        # run so that key exists". A stage nothing runs is a stage the loop does not have.
        if reuse_training_attempt_id:
            trained = self._verified_training_receipt(reuse_training_attempt_id, settings)
            if trained is None:
                failed = {"label": label, "ok": False, "where": "policy_artifact",
                          "settings": dict(settings), "metric_value": None,
                          "why": "the completed training receipt no longer matches this "
                                 "measurement's settings, protocol, or run identity",
                          "ran": False, "status": "unscored",
                          "training_outcome": {"status": "unverified",
                                               "attempt_id": reuse_training_attempt_id}}
                atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
                return failed
            trained = self._select_policy_artifact(trained, revalidate_abstained=True)
        else:
            self._prepare_once(settings)
            trained = (self.run_stage("train", settings=settings,
                                      **({"dataset": dataset} if dataset else {}))
                       if self.available("train")
                       else {"ran": False, "returncode": None, "why":
                             "no train command; evaluating the declared source controller"
                             if self.source_policy and not self.checkpoint else
                             "no command was derived for train, and the benchmark ships a "
                             "checkpoint, so that is what was scored"})
        if self.available("train"):
            if not reuse_training_attempt_id:
                trained = self._select_policy_artifact(trained)
        train_artifact = trained.get("artifact") or {}
        train_readings = trained.get("readings") or {}
        # A new checkpoint can be an untrained initialization. If the native trainer
        # explicitly says it planned zero iterations/updates, its zero-exit and fresh file
        # are insufficient evidence that the requested training actually happened.
        zero_work = explicitly_zero_training_work(train_readings)
        progress_unverified = (self.require_training_progress and self.available("train") and
                               (trained.get("training_progress") or {}).get("status") !=
                               "observed")
        policy_artifact_unresolved = (
            trained.get("status") == "completed" and trained.get("returncode") == 0 and
            bool(train_artifact.get("matched")) and
            self._recorded_artifact(trained) is None)
        if self.available("train") and (
                not trained.get("ran") or trained.get("returncode") != 0
                or not train_artifact.get("checked") or not train_artifact.get("matched")
                or zero_work or progress_unverified or policy_artifact_unresolved):
            # Written down like any other measurement, because a stage that failed is the
            # most informative thing this loop produces and it used to be returned without
            # being recorded: the report said a stage "did not run a command that finished"
            # and kept no copy of what the command said when it did not.
            failed = {**trained, "label": label, "ok": False,
                      "where": "policy_artifact" if policy_artifact_unresolved else "train",
                      "settings": dict(settings), "success_rate": None,
                      **({"status": "unscored",
                          "training_outcome": {"status": trained.get("status"),
                                               "returncode": trained.get("returncode"),
                                               "attempt_id": trained.get("attempt_id")}}
                         if policy_artifact_unresolved else {}),
                      "metric_value": None, "metric_utility": None,
                      "why": ("trainer explicitly reported zero updates/iterations" if
                              zero_work else "positive native training progress not verified" if
                              progress_unverified else
                              (trained.get("artifact_selection") or {}).get("why") or
                              "training did not produce a verified artifact" if
                              trained.get("returncode") == 0 else
                              "training did not exit successfully"),
                      "zero_work_reported": zero_work,
                      "training_progress": trained.get("training_progress")}
            atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
            return failed
        produced = (self._recorded_artifact(trained) if trained.get("ran")
                    else Path(self.checkpoint) if self.checkpoint else None)
        if not (self.source_policy and not self.available("train") and not self.checkpoint) and (
                produced is None or not produced.exists()):
            policy_artifact_unresolved = (trained.get("status") == "completed" and
                                          trained.get("returncode") == 0)
            failed = {**trained, "label": label, "ok": False,
                      "where": "policy_artifact" if policy_artifact_unresolved else "train",
                      "settings": dict(settings), "success_rate": None,
                      **({"status": "unscored",
                          "training_outcome": {"status": trained.get("status"),
                                               "returncode": trained.get("returncode"),
                                               "attempt_id": trained.get("attempt_id")}}
                         if policy_artifact_unresolved else {}),
                      "metric_value": None, "metric_utility": None,
                      "why": ((trained.get("artifact_selection") or {}).get("why") or
                              "no unique, existing policy artifact belongs to this training attempt")}
            atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
            return failed
        archived: dict[str, Any] = {}
        archive_error = ""
        if reuse_policy_artifact is not None:
            raw_policy = Path(str(reuse_policy_artifact.get("path") or ""))
            try:
                archived_path = raw_policy.resolve(strict=True)
                archived_path.relative_to(self.run_root.resolve())
                identity_field, identity = artifact_identity(archived_path)
                source_path = Path(str(reuse_policy_artifact.get("source") or ""))
                if not source_path.is_absolute():
                    source_path = Path(str(trained.get("working_directory") or self.repo)) / source_path
                if (raw_policy.is_symlink() or not archived_path.is_file() or
                        reuse_policy_artifact.get(identity_field) != identity or
                        produced is None or
                        source_path.resolve(strict=True) != produced.resolve(strict=True)):
                    raise ValueError("reused policy artifact no longer matches the completed train receipt")
                archived = dict(reuse_policy_artifact)
            except (OSError, ValueError, RuntimeError) as exc:
                archive_error = f"{type(exc).__name__}: {exc}"
        elif produced is not None:
            try:
                cap = int(os.environ.get("AUTOSIM_ARTIFACT_COPY_LIMIT_BYTES", str(2 * 1024**3)))
                if cap <= 0:
                    raise ValueError("artifact copy limit must be positive")
                suffix = produced.suffix if produced.is_file() else ""
                attempt = str(trained.get("attempt_id") or uuid.uuid4().hex)
                archived = freeze_artifact(
                    produced, self.run_root / "experiments" / label / attempt /
                    f"policy{suffix}", max_bytes=cap)
            except (OSError, ValueError, RuntimeError) as exc:
                archive_error = f"{type(exc).__name__}: {exc}"
        if produced is not None and not archived:
            # A score of bytes we failed to preserve cannot become an incumbent or be
            # confirmed later. Do not spend evaluation GPU time on that unrepeatable arm.
            failed = {"label": label, "ok": False, "where": "policy_archive",
                      "settings": dict(settings), "success_rate": None,
                      "policy_artifact": {}, "archive_error": archive_error,
                      "why": "policy bytes could not be frozen before evaluation",
                      "train": {"returncode": trained.get("returncode"),
                                "artifact": trained.get("artifact")}}
            atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
            return failed
        # Evaluate the archived bytes, not a mutable path in the benchmark checkout. A later
        # stage may overwrite `produced`; the metric must refer to the candidate we retain.
        evaluated_policy = Path(archived["path"]) if archived else produced
        source_state: dict[str, Any] | None = None
        try:
            source_state = self._capture_workspace_source_state(label)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            failed = {"label": label, "ok": False, "where": "source_identity",
                      "why": f"the evaluation source state cannot be frozen: "
                             f"{type(exc).__name__}: {exc}",
                      "settings": dict(settings), "metric_value": None,
                      "source_state": {"status": "unavailable"},
                      "policy_artifact": archived,
                      "train": {"returncode": trained.get("returncode"),
                                "attempt_id": trained.get("attempt_id"),
                                "artifact": trained.get("artifact")}}
            atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
            return failed
        scored = self.run_stage("evaluate", settings=settings,
                                checkpoint=str(evaluated_policy or ""),
                                experiment_dir=str(evaluated_policy or ""),
                                previous_artifact=str(evaluated_policy or ""),
                                confirmation=confirmation)
        policy_consumption: dict[str, Any] = {"status": "not_required"}
        rollout_evidence: dict[str, Any] = {"status": "not_required"}
        if self.require_policy_consumption and archived:
            eval_attempt = str(scored.get("attempt_id") or "")
            candidate_log = (self.run_root / "evaluate" / "attempts" /
                             eval_attempt / "output.log")
            try:
                policy_consumption = verify_policy_consumption(
                    candidate_log, Path(str(archived["path"])))
            except (OSError, ValueError, RuntimeError) as exc:
                policy_consumption = {"status": "unverified", "reason":
                                      f"policy load proof unavailable: {type(exc).__name__}"}
            if policy_consumption.get("status") == "verified":
                try:
                    rollout_evidence = verify_rollout_evidence(
                        candidate_log, policy_sha256=str(policy_consumption["sha256"]))
                except (OSError, ValueError, RuntimeError) as exc:
                    rollout_evidence = {"status": "unverified", "reason":
                                        f"rollout proof unavailable: {type(exc).__name__}"}
        source_state_error = ""
        if source_state is not None:
            try:
                after_source_state = self._capture_workspace_source_state(label)
                if (after_source_state or {}).get("identity_sha256") != \
                        source_state.get("identity_sha256"):
                    source_state_error = "source tree changed while the evaluation was running"
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                source_state_error = ("source tree could not be verified after evaluation: "
                                      f"{type(exc).__name__}: {exc}")
        result_archive: dict[str, Any] = {}
        result_archive_error = ""
        metric_artifact, metric_artifact_evidence = self._metric_artifact_for_record(scored)
        if self.metric_spec.source in {"json", "csv"} and metric_artifact is not None:
            try:
                cap = int(os.environ.get("AUTOSIM_RESULT_COPY_LIMIT_BYTES", str(64 * 1024**2)))
                if cap <= 0 or not metric_artifact.is_file():
                    raise ValueError("result copy requires a positive limit and one file")
                result_archive = freeze_artifact(
                    metric_artifact,
                    self.run_root / "experiments" / label /
                    str(scored.get("attempt_id") or uuid.uuid4().hex) /
                    f"result{metric_artifact.suffix}", max_bytes=cap)
                result_archive["content_sha256"] = digest(Path(result_archive["path"]))
            except (OSError, ValueError, RuntimeError) as exc:
                result_archive_error = f"{type(exc).__name__}: {exc}"
        if self.metric_spec.source in {"json", "csv"}:
            self._attach_metric_artifact_evidence(scored, metric_artifact_evidence,
                                                  result_archive)
        # A stage that never ran has no return code, and reading its absence as success is
        # how a benchmark missing an evaluator would report a number nobody measured.
        reading = self.metric_spec.read(said=scored.get("said", ""),
                                        artifact=(Path(result_archive["path"])
                                                  if result_archive else None))
        metric_lineage: dict[str, Any] = {"status": "not_required"}
        if self.require_policy_consumption and archived and reading["value"] is not None:
            try:
                metric_lineage = verify_metric_lineage(
                    candidate_log, policy_sha256=str(policy_consumption.get("sha256") or ""),
                    episode_ids=rollout_evidence.get("episode_ids") or [],
                    value=reading["value"])
            except (OSError, ValueError, RuntimeError) as exc:
                metric_lineage = {"status": "unverified", "reason":
                                  f"metric lineage unavailable: {type(exc).__name__}"}
        score_artifact_required = self.metric_spec.source in {"json", "csv"}
        scored_ok = (bool(scored.get("ran")) and scored.get("returncode") == 0
                     and scored.get("status") == "completed"
                     and (not self.require_policy_consumption or not archived or
                          (policy_consumption.get("status") == "verified" and
                           rollout_evidence.get("status") == "verified"))
                     and (not self.require_policy_consumption or not archived or
                          metric_lineage.get("status") == "verified")
                     and (not score_artifact_required or
                          (bool(result_archive) and
                           metric_artifact_evidence.get("status") == "matched"))
                     and reading["value"] is not None and not source_state_error)
        result = {"label": label, "ok": scored_ok,
                  "where": ("source_identity" if source_state_error else
                            "evaluate" if not scored_ok else ""),
                  "why": (source_state_error or
                          (policy_consumption.get("reason") if
                           self.require_policy_consumption and archived and
                           policy_consumption.get("status") != "verified" else "") or
                          (rollout_evidence.get("reason") if
                           self.require_policy_consumption and archived and
                           rollout_evidence.get("status") != "verified" else "") or
                          (metric_lineage.get("reason") if
                           self.require_policy_consumption and archived and
                           metric_lineage.get("status") != "verified" else "") or
                          "evaluation did not exit normally or yield a verified native "
                          "metric result" if not scored_ok else ""),
                  "scored": scored_what,
                  "settings": {k: settings[k] for k in sorted(settings)},
                  "policy_artifact": archived,
                  "policy_consumption": policy_consumption,
                  "rollout_evidence": rollout_evidence,
                  "metric_lineage": metric_lineage,
                  "source_state": source_state or {},
                  "source_policy_artifact": ({"kind": "source_tree", **source_state}
                                             if not archived and self.source_policy and
                                             source_state else {}),
                  "evaluated_policy_path": str(evaluated_policy or ""),
                  "archive_error": archive_error,
                  "result_artifact": result_archive,
                  "metric_artifact_evidence": metric_artifact_evidence,
                  "result_archive_error": result_archive_error,
                  "restorable_policy": bool(archived),
                  **({"dataset": dataset} if dataset else {}),
                  "train": {"returncode": trained.get("returncode"),
                            "attempt_id": trained.get("attempt_id"),
                            "evidence_id": trained.get("evidence_id"),
                            "evidence_ref": trained.get("evidence_ref"),
                            "reused": bool(reuse_training_attempt_id),
                            "artifact": trained.get("artifact"),
                            "artifact_selection": trained.get("artifact_selection")},
                  "evaluate": {"returncode": scored.get("returncode"),
                               "attempt_id": scored.get("attempt_id"),
                               "evidence_id": scored.get("evidence_id"),
                               "evidence_ref": scored.get("evidence_ref"),
                               "device": scored.get("device"),
                               "gpu_device_uuid": scored.get("gpu_device_uuid"),
                               "compute_environment": scored.get("compute_environment") or {},
                               "said": scored.get("said", "")[-800:]},
                  "metric": self.metric_spec.as_dict(),
                  "metric_value": reading["value"] if scored_ok else None,
                  "metric_utility": (self.metric_spec.utility(reading["value"])
                                     if scored_ok else None),
                  "success_rate": (reading["value"] if scored_ok and
                                   self.metric_spec.name == "success_rate" else None),
                  # Where that number came from. A program reporting one rate per task
                  # prints them separated by `|`, and the mean of them was stored as a bare
                  # float with nothing saying it was a mean -- and that float is what every
                  # candidate is ranked by. With the line and the values beside it, a reader
                  # can tell an average over three tasks from one task scoring the same, and
                  # can see it was read at all rather than taken on faith.
                  "metric_reading": reading,
                  **({"success_reading": reading}
                     if self.metric_spec.name == "success_rate" else {}),
                  # What to rank on when the final number is not the only thing known. Both
                  # stages, because a trainer that reports loss has reported something usable
                  # before anything has been evaluated at all.
                  # The readings, by the stage that printed them. A reader has to be able to
                  # tell what a number is evidence *of*: `loss` from a trainer that ran for
                  # three hours is a reading, and `line` and `cuda` scraped from an evaluator's
                  # traceback are the shape a crash leaves. Merged into one map they are
                  # indistinguishable, and the document used to resolve the ambiguity by
                  # hiding the whole column whenever no success rate came out -- which hid the
                  # trainer's real readings on exactly the runs that got furthest.
                  "readings_by_stage": {
                      "train": self._readings(trained.get("said", "")) if trained.get("ran")
                               else {},
                      "evaluate": self._readings(scored.get("said", "")) if scored.get("ran")
                                  else {}},
                  "readings": {**self._readings(trained.get("said", "")),
                               **self._readings(scored.get("said", ""))}}
        self._attach_guardrails(result, label=label, said=str(scored.get("said") or ""),
                                artifact=Path(result_archive["path"]) if result_archive else None)
        atomic_json(self.run_root / "measurements" / f"{label}.json", result)
        # The stage boundary document precedes the scored measurement. Refresh it once the
        # metric and exact evaluation attempt are recorded, so media lineage is current while
        # the run is still live, not only after the final research report.
        run_record.generate(self.run_root, title=f"{self.benchmark or self.run_id} / derived")
        return result

    def _artifact_path(self, stage: str, settings: dict[str, Any]) -> Path | None:
        """Where the stage said it would write, resolved against this run.

        The artifact is a glob and may contain placeholders the stage's own inputs supply,
        so it is resolved the way the command was rather than assumed -- and a placeholder
        can be filled with an absolute path, which makes the whole thing absolute. That is
        not hypothetical: LIBERO's own tree is `./experiments/{benchmark}/{algo}/...` and the
        run that produced a real FWT of 0.488 finished, wrote `result.pt`, and *then* this
        raised `NotImplementedError: Non-relative patterns are unsupported` -- so the number
        existed and the system had no record of it. `Path.glob` refuses an absolute pattern;
        the stdlib `glob` does not, and is what a resolved pattern needs.
        """
        pattern = self.backend.artifact_pattern(stage)
        if not pattern:
            return None
        base = self.run_root / stage
        try:
            resolved = pattern.format(**(self._inputs(stage, settings=settings) | {
                "hydra_run_dir": "", "experiment_dir": ""}))
        except (KeyError, IndexError, ValueError):
            resolved = pattern
        if artifact_pattern_problem(resolved):
            return None
        if not any(char in resolved for char in "*?["):
            target = Path(resolved)
            return target if target.is_absolute() else base / resolved
        try:
            if Path(resolved).is_absolute():
                matches = sorted(Path(match) for match in glob.glob(resolved, recursive=True))
            else:
                matches = sorted(base.glob(resolved))
        except (NotImplementedError, ValueError):
            return None
        return matches[-1] if matches else None

    def _selection_candidates(self, record: dict[str, Any]
                              ) -> tuple[list[str], str]:
        """Return the complete fresh candidate set, reconstructing older receipts only
        when the persisted match count and attempt time let us prove the same set."""
        artifact = record.get("artifact") or {}
        matched = int(artifact.get("matched") or 0)
        candidates = artifact.get("candidate_paths")
        if (isinstance(candidates, list) and not artifact.get("candidate_paths_truncated")
                and len(candidates) == matched):
            return [str(path) for path in candidates], ""
        if matched <= 1 or artifact.get("candidate_paths_truncated") or matched > 128:
            return [], "the complete candidate set is unavailable or exceeds the selection bound"

        attempt_id = str(record.get("attempt_id") or "")
        receipt_path = self.run_root / "attempts" / attempt_id / "receipt.json"
        try:
            receipt = read_json(receipt_path)
            started = datetime.fromisoformat(
                str(receipt["started_at"]).replace("Z", "+00:00")).timestamp()
            pattern = str(artifact["pattern"])
            stage = str(record.get("stage") or "")
            base = (Path(str(record.get("cwd") or self.backend.directory(stage)))
                    if artifact.get("found_beside_the_command") else
                    self.run_root / stage)
            target = Path(pattern)
            target = target if target.is_absolute() else base / target
            paths = sorted(Path(path) for path in glob.glob(str(target), recursive=True))
            paths = [path for path in paths if path.exists() and
                     path.stat().st_mtime >= started]
            if len(paths) != matched:
                # Older receipts kept only three examples. A later attempt may have added
                # fresh matches to a sibling run directory, changing the broad glob even
                # though the original candidate directory is still intact. Narrow only
                # when every recorded example proves one common parent and that parent's
                # complete fresh set exactly reproduces the old count; otherwise abstain.
                examples = artifact.get("examples") or []
                example_paths = []
                for example in examples:
                    candidate = Path(str(example))
                    candidate = (candidate if candidate.is_absolute() else base / candidate)
                    example_paths.append(candidate.resolve(strict=True))
                parents = {candidate.parent for candidate in example_paths}
                if (example_paths and len(parents) == 1 and
                        all(candidate in paths and candidate.stat().st_mtime >= started
                            for candidate in example_paths)):
                    parent = next(iter(parents))
                    scoped = [candidate for candidate in paths
                              if candidate.resolve().parent == parent]
                    if len(scoped) == matched:
                        paths = scoped
            if len(paths) != matched:
                return [], "fresh artifact candidates no longer match the completed receipt"
            rendered = ([str(path) for path in paths] if Path(pattern).is_absolute() else
                        [str(path.relative_to(base)) for path in paths])
            examples = artifact.get("examples") or []
            if rendered[:len(examples)] != [str(path) for path in examples]:
                return [], "reconstructed candidates disagree with paths recorded in the receipt"
            artifact["candidate_paths"] = rendered
            artifact["candidate_paths_reconstructed"] = True
            record["artifact"] = artifact
            return rendered, ""
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return [], f"could not reconstruct candidates from this attempt: " \
                       f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _selection_excerpt(source: str) -> str:
        """Keep source context around native save/load semantics within a bounded prompt."""
        lines = source.splitlines()
        numbered = [f"{index + 1}: {line}" for index, line in enumerate(lines)]
        full = "\n".join(numbered)
        if len(full) <= 9000:
            return full
        relevant = [index for index, line in enumerate(lines)
                    if re.search(r"save|checkpoint|ckpt|best|final|weight|policy|model",
                                 line, re.IGNORECASE)]
        chosen: set[int] = set()
        for index in relevant:
            chosen.update(range(max(0, index - 2), min(len(lines), index + 3)))
            excerpt = "\n".join(numbered[row] for row in sorted(chosen))
            if len(excerpt) >= 9000:
                break
        if not chosen:
            return full[:9000]
        return "\n".join(numbered[row] for row in sorted(chosen))[:9000]

    def _policy_selection_sources(self, stage: str
                                  ) -> tuple[list[dict[str, str]], str]:
        row = self.backend.stages.get(stage) or {}
        workdir = Path(self.backend.directory(stage))
        paths: list[Path] = []

        def add(path: Path) -> None:
            try:
                resolved = path.resolve(strict=True)
                if (resolved.is_file() and resolved.is_relative_to(self.repo.resolve())
                        and resolved not in paths):
                    paths.append(resolved)
            except OSError:
                return

        entrypoint = str(row.get("entrypoint") or "").strip()
        if entrypoint:
            path = Path(entrypoint)
            add(path if path.is_absolute() else self.repo / path)
            add(path if path.is_absolute() else workdir / path)
        for root in (workdir, self.repo):
            try:
                for readme in sorted(root.glob("README*")):
                    add(readme)
            except OSError:
                continue
        evidence = json.dumps(row.get("parameters") or [], ensure_ascii=False, default=str)
        for relative in re.findall(
                r"(?<![\w./-])(?:[\w.-]+/)*[\w.-]+\.(?:py|md|ya?ml|toml|json)",
                evidence):
            add(self.repo / relative)
            if len(paths) >= 5:
                break
        sources = []
        for path in paths[:5]:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            relative = str(path.relative_to(self.repo.resolve()))
            sources.append({"path": relative, "sha256": digest(path), "text": text,
                            "excerpt": self._selection_excerpt(text)})
        if not sources:
            return [], "no repository source file was available to establish native selection semantics"
        return sources, ""

    def _select_policy_artifact(self, record: dict[str, Any], *,
                                retry_unavailable: bool = False,
                                revalidate_abstained: bool = False) -> dict[str, Any]:
        """Use the LLM to interpret native selection evidence, then verify its choice
        against the exact fresh files emitted by this attempt before evaluation."""
        artifact = dict(record.get("artifact") or {})
        matched = int(artifact.get("matched") or 0)
        if matched <= 1 or record.get("status") != "completed" or record.get("returncode") != 0:
            return record
        record["artifact"] = artifact
        attempt_id = str(record.get("attempt_id") or "")
        candidates, candidate_error = self._selection_candidates(record)
        candidate_ids = {f"candidate_{index + 1}": path
                         for index, path in enumerate(candidates)}
        candidate_id_by_path = {path: candidate_id
                                for candidate_id, path in candidate_ids.items()}

        def redact_candidate_references(value: str) -> tuple[str, set[str]]:
            """Replace produced checkpoint references before constructing any model input."""
            text = str(value)
            found: set[str] = set()
            replacements: dict[str, str] = {}
            basename_owners: dict[str, set[str]] = {}
            for candidate_id, candidate in candidate_ids.items():
                path = Path(candidate)
                variants = {candidate, path.as_posix(), path.name}
                try:
                    variants.add(str(path.resolve()))
                except OSError:
                    pass
                for variant in variants:
                    if variant:
                        replacements[variant] = candidate_id
                basename_owners.setdefault(path.name, set()).add(candidate_id)
            for basename, owners in basename_owners.items():
                if len(owners) > 1:
                    # A bare duplicate filename cannot identify one of the fresh files.
                    replacements[basename] = "[AMBIGUOUS_CANDIDATE]"
            for variant in sorted(replacements, key=len, reverse=True):
                if variant in text:
                    replacement = replacements[variant]
                    if replacement in candidate_ids:
                        found.add(replacement)
                    text = text.replace(variant, replacement)
            text = sanitize_model_text(text, local_roots=(self.repo, self.run_root))
            # Mask non-candidate checkpoint names and wildcard patterns as well. The model
            # needs candidate IDs, not local artifact naming conventions or paths.
            text = re.sub(
                r"(?i)(?<![\w])(?:[\w.@+~*?-]+[/\\])*[\w.@+~*?-]*\.(?:pt|pth|ckpt|"
                r"safetensors|bin|onnx|pkl|pickle|h5|hdf5|msgpack|flax)(?:[\w.+~-]*)",
                "[CHECKPOINT_REF]", text)
            return text, found

        raw_training_log = str(record.get("said") or "")[-6000:]
        safe_training_log, observed_candidate_ids = redact_candidate_references(raw_training_log)
        selection_dir = self.run_root / "attempts" / attempt_id
        selection_path = selection_dir / "artifact_selection.json"
        candidate_sha256 = object_digest(candidates)
        sources, source_error = self._policy_selection_sources(str(record.get("stage") or ""))
        source_hashes = {source["path"]: source["sha256"] for source in sources}

        def same_inputs(saved: dict[str, Any]) -> bool:
            return (saved.get("candidate_paths_sha256") == candidate_sha256 and
                    saved.get("source_hashes") == source_hashes)

        def transient_failure(saved: dict[str, Any]) -> bool:
            if saved.get("status") == "unavailable":
                return any(word in str(saved.get("why") or "").lower() for word in (
                    "transport", "timeout", "timed out", "connection", "proxy", "http 5"))
            # Older receipts used `abstained` for transport failures. Preserve those records,
            # but let an explicit recovery action create a new resolver attempt.
            return (saved.get("status") == "abstained" and any(
                word in str(saved.get("why") or "").lower() for word in (
                    "transport", "timeout", "timed out", "connection", "proxy", "http 5")))

        # A failed/abstaining resolver record is immutable evidence, but a later controller
        # version may be able to recover a candidate set that the old resolver could not.
        # Give that new resolution its own file instead of overwriting history. A previously
        # selected result, on the other hand, can never be silently superseded.
        try:
            if selection_path.is_file():
                previous = read_json(selection_path)
                if previous.get("attempt_id") != attempt_id:
                    raise ValueError("persisted selection belongs to a different attempt")
                if (revalidate_abstained and previous.get("status") == "abstained" and
                        not same_inputs(previous)):
                    record["artifact_selection"] = {
                        "status": "abstained", "selection_ref": selection_ref,
                        "why": "cannot revalidate the saved decision because its source or "
                               "candidate inputs changed"}
                    return record
                if not same_inputs(previous) or (
                        retry_unavailable and transient_failure(previous) and
                        not candidate_error and not source_error):
                    if previous.get("status") == "selected":
                        raise ValueError("a selected result cannot be rebound or retried")
                    index = 1
                    while True:
                        alternate = selection_dir / f"artifact_selection_{index}.json"
                        if not alternate.is_file():
                            selection_path = alternate
                            break
                        prior = read_json(alternate)
                        if prior.get("attempt_id") != attempt_id:
                            raise ValueError("persisted selection belongs to a different attempt")
                        if same_inputs(prior):
                            if (retry_unavailable and transient_failure(prior) and
                                    not candidate_error and not source_error):
                                index += 1
                                continue
                            selection_path = alternate
                            break
                        if prior.get("status") == "selected":
                            raise ValueError(
                                "a selected result cannot be rebound to changed inputs")
                        index += 1
        except (OSError, TypeError, ValueError) as exc:
            reason = f"persisted artifact selection is invalid: {type(exc).__name__}: {exc}"
            record["artifact_selection"] = {"status": "invalid", "why": reason}
            return record
        selection_ref = str(selection_path.relative_to(self.output))

        def selection_record(status: str, *, answer: dict[str, Any] | None = None,
                             why: str = "", provider: dict[str, Any] | None = None,
                             response_sha256: str = "") -> dict[str, Any]:
            selected = str((answer or {}).get("selected_path") or "")
            return {"schema_version": 1, "attempt_id": attempt_id,
                    "stage": str(record.get("stage") or ""), "status": status,
                    "candidate_paths": candidates,
                    "candidate_paths_sha256": candidate_sha256,
                    "source_hashes": source_hashes,
                    "selected_path": selected or None,
                    "native_rule": str((answer or {}).get("native_rule") or "")[:1000],
                    "evidence": (answer or {}).get("evidence") or [],
                    "candidate_trace": str((answer or {}).get("candidate_trace") or "")[:1000],
                    "why": why[:1000], "provider": dict(provider or {}),
                    "response_sha256": response_sha256, "at": now()}

        safe_source_texts = {
            source["path"]: redact_candidate_references(source["text"])[0]
            for source in sources
        }
        safe_source_excerpt_texts = {
            source["path"]: "\n".join(
                re.sub(r"^\d+: ", "", line)
                for line in self._selection_excerpt(
                    safe_source_texts[source["path"]]).splitlines())
            for source in sources
        }

        def validate(answer: dict[str, Any], *, allow_legacy: bool = False
                     ) -> tuple[str | None, str]:
            if answer.get("decision") == "abstain":
                return None, str(answer.get("why") or "model abstained: native selection is unclear")
            if answer.get("decision") != "select":
                return None, "selection response must explicitly select or abstain"
            candidate_id = str(answer.get("candidate_id") or "")
            selected = candidate_ids.get(candidate_id, "")
            if not selected and allow_legacy:
                legacy_path = str(answer.get("selected_path") or "")
                if legacy_path in candidate_id_by_path:
                    selected = legacy_path
                    candidate_id = candidate_id_by_path[legacy_path]
            if not selected:
                return None, "selection must name an exact opaque candidate ID from this attempt"
            native_rule = str(answer.get("native_rule") or "").strip()
            if len(native_rule) < 12:
                return None, "selection did not explain the repository-native rule"
            citations = answer.get("evidence")
            if not isinstance(citations, list) or not citations:
                return None, "selection has no source-code evidence"
            source_by_path = {source["path"]: source for source in sources}
            verified_citations = []
            for citation in citations[:3]:
                if not isinstance(citation, dict):
                    return None, "source evidence must be structured"
                path = str(citation.get("source_path") or "")
                quote = str(citation.get("quote") or "").strip()
                source = source_by_path.get(path)
                if source is None or not _source_quote_present(
                        safe_source_excerpt_texts[path], quote):
                    # Old immutable decisions may quote a pre-alias source excerpt. Permit
                    # that only while validating an already-saved local response; live model
                    # responses must cite exactly the safe excerpt shown in their request.
                    if (not allow_legacy or source is None or
                            not _source_quote_present(source["text"], quote)):
                        return None, "source citation is not an exact quote from a current repository file"
                verified_citations.append({"source_path": path, "quote": quote,
                                           "why": str(citation.get("why") or "")[:500]})
            trace = str(answer.get("candidate_trace") or "").strip()
            safe_trace, _ = redact_candidate_references(trace)
            if (not safe_trace or safe_trace not in safe_training_log or
                    candidate_id not in safe_trace or
                    candidate_id not in observed_candidate_ids):
                return None, "training log does not identify the selected fresh candidate"
            answer["evidence"] = verified_citations
            answer["candidate_id"] = candidate_id
            answer.pop("selected_path", None)
            answer["candidate_trace"] = safe_trace
            return selected, ""

        try:
            if selection_path.is_file():
                previous = read_json(selection_path)
                if (previous.get("attempt_id") != attempt_id or
                        previous.get("candidate_paths_sha256") != candidate_sha256 or
                        previous.get("source_hashes") != source_hashes):
                    raise ValueError("persisted selection no longer matches its candidates or source")
                answer = previous.get("answer") or {}
                if revalidate_abstained and previous.get("status") == "abstained":
                    # Re-read the original model response rather than trusting an edited
                    # selection record. A later validator may safely accept a previously
                    # rejected citation, but the original decision remains immutable and the
                    # revalidation gets its own receipt.
                    exchange_path = self.run_root / "exchanges" / (
                        f"artifact_selection_{attempt_id}.json")
                    exchange = read_json(exchange_path) if exchange_path.is_file() else {}
                    content = str(exchange.get("response") or "")
                    if (not content or object_digest(content) !=
                            str(previous.get("response_sha256") or "")):
                        record["artifact_selection"] = {
                            "status": "abstained", "selection_ref": selection_ref,
                            "why": "saved model response is unavailable or does not match "
                                   "the original selection digest"}
                        return record
                    from .execution_derive import _object
                    answer = _object(content)
                selected, why = validate(answer, allow_legacy=True)
                if selected:
                    answer["selected_path"] = selected
                if previous.get("status") == "selected" and not selected:
                    record["artifact_selection"] = {
                        "status": "invalid", "selection_ref": selection_ref,
                        "why": "saved selected decision no longer passes current local validation: "
                              f"{why}"}
                    return record
                if previous.get("status") == "selected" and selected:
                    artifact["selected_path"] = selected
                    artifact["selection_ref"] = selection_ref
                    record.update(artifact=artifact,
                                  artifact_selection={"status": "selected",
                                                      "selected_path": selected,
                                                      "selection_ref": selection_ref,
                                                      "native_rule": previous.get("native_rule")})
                elif revalidate_abstained and previous.get("status") == "abstained" and selected:
                    original_ref = selection_ref
                    index = 1
                    while True:
                        alternate = selection_dir / f"artifact_selection_{index}.json"
                        if not alternate.exists():
                            break
                        index += 1
                    selection_path = alternate
                    selection_ref = str(selection_path.relative_to(self.output))
                    recovered = selection_record(
                        "selected", answer=answer,
                        why="previously abstained response passed the current source/candidate "
                            "verifier without changing its inputs",
                        provider=previous.get("provider") or {},
                        response_sha256=str(previous.get("response_sha256") or ""))
                    recovered["answer"] = answer
                    recovered["supersedes_selection_ref"] = original_ref
                    recovered["revalidation"] = {
                        "kind": "source_and_candidate_revalidation",
                        "prior_status": "abstained",
                        "current_inputs_unchanged": True,
                    }
                    atomic_json(selection_path, recovered)
                    artifact["selected_path"] = selected
                    artifact["selection_ref"] = selection_ref
                    record.update(artifact=artifact,
                                  artifact_selection={"status": "selected",
                                                      "selected_path": selected,
                                                      "selection_ref": selection_ref,
                                                      "native_rule": recovered["native_rule"]})
                else:
                    record["artifact_selection"] = {
                        "status": str(previous.get("status") or "abstained"),
                                                     "selection_ref": selection_ref,
                                                     "why": previous.get("why") or why}
                return record
        except (OSError, TypeError, ValueError) as exc:
            reason = f"persisted artifact selection is invalid: {type(exc).__name__}: {exc}"
            record["artifact_selection"] = {"status": "invalid", "why": reason}
            return record

        if candidate_error or source_error:
            result = selection_record("unavailable", why=candidate_error or source_error)
            atomic_json(selection_path, {**result, "answer": {}})
            record["artifact_selection"] = {"status": "unavailable",
                                             "selection_ref": selection_ref,
                                             "why": result["why"]}
            return record
        if not callable(getattr(self.client, "chat_with_metadata", None)):
            result = selection_record("unavailable",
                                      why="no LLM client is available to resolve native artifact semantics")
            atomic_json(selection_path, {**result, "answer": {}})
            record["artifact_selection"] = {"status": "unavailable",
                                             "selection_ref": selection_ref,
                                             "why": result["why"]}
            return record

        system = (
            "Resolve which single fresh policy artifact the repository itself intends its "
            "evaluator to consume. This is not a metric-selection task. Never choose by "
            "lexicographic order or modification time alone, and do not assume that a file "
            "named final/latest/best is semantically correct unless the supplied source proves "
            "it. Select only an exact opaque candidate_id. Candidate IDs are local aliases; "
            "never infer or invent a filesystem path or filename. Cite exact quotations from "
            "the supplied repository-source excerpts that establish the native save/load or "
            "best-policy rule, and quote a training-log trace naming the candidate ID. If the "
            "rule is ambiguous or evidence is insufficient, abstain. Return one JSON object "
            "with decision ('select' or 'abstain'), candidate_id, "
            "native_rule, evidence ([{source_path, quote, why}]), candidate_trace, and why."
        )
        safe_sources = [{"path": source["path"], "sha256": source["sha256"],
                         "excerpt": self._selection_excerpt(safe_source_texts[source["path"]])}
                        for source in sources]
        payload = {"stage": record.get("stage"), "entrypoint":
                   (self.backend.stages.get(str(record.get("stage"))) or {}).get("entrypoint"),
                   "candidate_ids": list(candidate_ids),
                   "training_log_tail": safe_training_log,
                   "repository_sources": safe_sources}
        user = sanitize_model_text(
            json.dumps(payload, ensure_ascii=False, default=str),
            local_roots=(self.repo, self.run_root))
        try:
            content, provider = self.client.chat_with_metadata(
                sanitize_model_text(system), user, max_tokens=1200, timeout=90,
                thinking="disabled")
        except Exception as exc:  # noqa: BLE001 - transport failure is recoverable, not a guess
            why = f"native artifact selection transport unavailable: {type(exc).__name__}: {exc}"
            result = selection_record("unavailable", why=why)
            result["answer"] = {}
            try:
                atomic_json(selection_path, result)
            except OSError:
                pass
            record["artifact_selection"] = {"status": "unavailable",
                                             "selection_ref": selection_ref, "why": why}
            self._append_local_event({"at": now(), "event": "policy_artifact_selection",
                                      "attempt_id": attempt_id, "status": "unavailable",
                                      "selection_ref": selection_ref, "why": why})
            return record
        if isinstance(provider, dict) and provider.get("available") is False:
            why = "LLM client has no configured provider credentials"
            result = selection_record("unavailable", why=why, provider=provider)
            result["answer"] = {}
            atomic_json(selection_path, result)
            record["artifact_selection"] = {"status": "unavailable",
                                             "selection_ref": selection_ref, "why": why}
            self._append_local_event({"at": now(), "event": "policy_artifact_selection",
                                      "attempt_id": attempt_id, "status": "unavailable",
                                      "selection_ref": selection_ref, "why": why})
            return record
        try:
            self._record_exchange(f"artifact_selection_{attempt_id}", system=system,
                                  user=user, content=content, metadata=provider or {})
            from .execution_derive import _object
            answer = _object(content)
            selected, why = validate(answer)
            if selected:
                # The exact selected path is retained only in local receipts, never the
                # model request/response schema or exchange prompt.
                answer["selected_path"] = selected
            status = "selected" if selected else "abstained"
            result = selection_record(status, answer=answer, why=why,
                                      provider=provider or {},
                                      response_sha256=object_digest(content))
            result["answer"] = answer
            atomic_json(selection_path, result)
        except Exception as exc:  # noqa: BLE001 - failure to select remains fail-closed
            why = f"native artifact selection failed: {type(exc).__name__}: {exc}"
            result = selection_record("abstained", why=why)
            result["answer"] = {}
            try:
                atomic_json(selection_path, result)
            except OSError:
                pass
            record["artifact_selection"] = {"status": "abstained",
                                             "selection_ref": selection_ref, "why": why}
            self._append_local_event({"at": now(), "event": "policy_artifact_selection",
                                      "attempt_id": attempt_id, "status": "abstained",
                                      "why": why})
            return record

        self._append_local_event({"at": now(), "event": "policy_artifact_selection",
                                  "attempt_id": attempt_id, "status": status,
                                  "selected_path": selected, "selection_ref": selection_ref,
                                  "why": why})
        if selected:
            artifact["candidate_paths"] = candidates
            artifact["selected_path"] = selected
            artifact["selection_ref"] = selection_ref
            record.update(artifact=artifact,
                          artifact_selection={"status": "selected",
                                              "selected_path": selected,
                                              "selection_ref": selection_ref,
                                              "native_rule": result["native_rule"]})
        else:
            record["artifact_selection"] = {"status": "abstained",
                                             "selection_ref": selection_ref,
                                             "why": why or "the model could not establish a native selection rule"}
        return record

    def _metric_artifact_for_record(self, record: dict[str, Any]
                                    ) -> tuple[Path | None, dict[str, Any]]:
        """Find one result file from this score attempt, separate from its policy/media output."""
        if self.metric_spec.artifact_pattern:
            resolved = resolve_metric_artifact(
                self.metric_spec,
                roots={"working_directory": Path(str(record.get("working_directory") or
                                                       self.backend.directory("evaluate"))),
                       "output": Path(str(record.get("output_directory") or
                                         self.stage_directory("evaluate"))),
                       "policy_parent": Path(str(record.get("policy_output_root") or
                                                 self.run_root / "__missing_policy_parent__"))},
                started_at=float(record.get("started_epoch") or 0),
                allowed_roots=[self.repo, self.run_root])
            return (Path(resolved["path"]), resolved) if resolved.get("status") == "matched" \
                else (None, resolved)
        path = self._recorded_artifact(record)
        if path is None or not path.is_file() or path.is_symlink():
            return None, {"status": "missing", "matched": 0,
                          "why": "no unique verified result artifact for this attempt"}
        try:
            stat = path.stat()
            return path, {"status": "matched", "matched": 1, "path": str(path.resolve()),
                          "sha256": digest(path), "size_bytes": stat.st_size,
                          "mtime": stat.st_mtime, "source": "declared_stage_artifact"}
        except OSError as exc:
            return None, {"status": "unreadable", "matched": 0,
                          "why": type(exc).__name__}

    def _attach_metric_artifact_evidence(self, record: dict[str, Any],
                                         evidence: dict[str, Any],
                                         archive: dict[str, Any]) -> None:
        """Bind the selected result bytes and their frozen copy to the exact eval receipt."""
        attempt_id = str(record.get("attempt_id") or "")
        if not re.fullmatch(r"[a-f0-9]{32}", attempt_id):
            return
        receipt_path = self.run_root / "attempts" / attempt_id / "receipt.json"
        try:
            receipt = read_json(receipt_path)
            if (receipt.get("attempt_id") != attempt_id or
                    receipt.get("node_id") != "evaluate"):
                return
            frozen_path = Path(str(archive.get("path") or ""))
            if frozen_path.is_absolute() and frozen_path.resolve().is_relative_to(
                    self.run_root.resolve()):
                frozen_ref = str(frozen_path.resolve().relative_to(self.run_root.resolve()))
            else:
                frozen_ref = ""
            receipt["metric_result_artifact"] = {
                "status": evidence.get("status"), "path": evidence.get("path"),
                "root": evidence.get("root"), "pattern": evidence.get("pattern"),
                "sha256": evidence.get("sha256"), "size_bytes": evidence.get("size_bytes"),
                "mtime": evidence.get("mtime"),
                "frozen_copy_ref": frozen_ref,
                "frozen_copy_sha256": archive.get("content_sha256", ""),
            }
            atomic_json(receipt_path, receipt)
        except (OSError, ValueError, TypeError):
            return

    def _recorded_artifact(self, record: dict[str, Any]) -> Path | None:
        """Use one artifact this exact stage attempt verified, never a later glob hit."""
        artifact = record.get("artifact") or {}
        if artifact.get("matched") == 1:
            examples = artifact.get("examples") or []
            candidate = str(examples[0]) if examples else ""
        else:
            selection = record.get("artifact_selection") or {}
            candidate = str(artifact.get("selected_path") or "")
            candidates = artifact.get("candidate_paths") or []
            if (selection.get("status") != "selected" or not candidate or
                    candidate not in candidates or
                    selection.get("selected_path") != candidate):
                return None
            reference = str(selection.get("selection_ref") or "")
            raw_reference = self.output / reference
            selection_path = raw_reference.resolve()
            if (raw_reference.is_symlink() or
                    not selection_path.is_relative_to(self.run_root.resolve()) or
                    not selection_path.is_file()):
                return None
            try:
                saved = read_json(selection_path)
            except (OSError, TypeError, ValueError):
                return None
            if (saved.get("status") != "selected" or
                    saved.get("selected_path") != candidate or
                    saved.get("candidate_paths_sha256") != object_digest(candidates)):
                return None
        if not candidate:
            return None
        path = Path(candidate)
        if not path.is_absolute():
            base = (self.backend.directory(str(record.get("stage")))
                    if artifact.get("found_beside_the_command") else
                    self.run_root / str(record.get("stage")))
            path = base / path
        try:
            resolved = path.resolve(strict=True)
            if (not resolved.is_file() and not resolved.is_dir()) or not any(
                    resolved.is_relative_to(root.resolve())
                    for root in (self.repo, self.output)):
                return None
        except OSError:
            return None
        return path

    def _protocol_violation(self) -> str:
        if self._frozen_error:
            return f"could not freeze evaluation protocol: {self._frozen_error}"
        try:
            assert_frozen(self._frozen)
            current = redline_rules.protected_hashes(
                self.repo, self.red_lines(), exclude_roots=(self.run_root,))
            added = sorted(set(current) - set(self._frozen))
            if added:
                return f"frozen protocol gained protected files: {added[:10]}"
        except (OSError, RuntimeError, ValueError) as exc:
            return str(exc)
        return ""

    def _comparison_refusal(self, label: str, settings: dict[str, Any], why: str
                            ) -> dict[str, Any]:
        failed = {"label": label, "ok": False, "where": "comparison_protocol",
                  "why": why, "settings": dict(settings), "metric_value": None,
                  "ran": False}
        atomic_json(self.run_root / "measurements" / f"{label}.json", failed)
        return failed

    def _comparison_protocol_violation(self, settings: dict[str, Any], *,
                                       target: str, freeze: bool = False,
                                       confirmation: bool = False) -> str:
        """Lock evaluation identity without locking candidate training parameters.

        The generic names are deliberately conservative. Repositories with nonstandard
        evaluator switches must declare them as research_goal.protocol_keys; until then
        this is only a partial protocol guarantee, not independent confirmation.
        """
        path = self.run_root / "comparison_protocol.json"
        if not freeze and not path.exists():
            return ""
        goal = self.declaration.get("research_goal") or {}
        extra = goal.get("protocol_keys") or []
        if not isinstance(extra, list) or any(not isinstance(key, str) or not key
                                             for key in extra):
            return "research_goal.protocol_keys must be a list of nonempty names"
        exact = {"task", "tasks", "suite", "benchmark", "seed", "seeds",
                 "episodes", "n_episodes", "horizon", "max_episode_steps",
                 "initial_states", "eval_seed", *extra}
        selected = {key: value for key, value in settings.items()
                    if key in exact or key.startswith(("eval.", "evaluation."))}
        held_out = self._held_out()
        if self._held_out_error:
            return self._held_out_error
        # What the *search* measures at, which is the frozen protocol once one exists and this
        # call's own settings while the protocol is being frozen. Comparing the reservation
        # against the caller's settings instead would make every attempt to use the held-out
        # value look like a declaration that reserves nothing, and the run would be told to
        # fix its declaration when what it did was search on the held-out episodes.
        frozen_settings = None
        if path.exists():
            try:
                old = read_json(path)
                if isinstance(old, dict) and isinstance(old.get("settings"), dict):
                    frozen_settings = old["settings"]
            except (OSError, ValueError):
                frozen_settings = None
        searching = selected if frozen_settings is None else frozen_settings
        # A reservation whose value is the value the search already measures at reserves
        # nothing: it reads like a held-out split and the confirmation would be a second
        # measurement of the same episodes, reported as independent of them.
        same = sorted(key for key in held_out if searching.get(key) == held_out[key])
        if same:
            return (f"research_goal.confirmation reserves {same} at the value the search "
                    f"already measures at, which reserves nothing")
        # A measurement that is *not* the confirmation may not use the held-out values. This is
        # the whole of the split: the settings the search may look at and the settings it may
        # not are both fixed before the first measurement, and the only thing the loop can do
        # with the second set is measure it once, at the end, and report what came back.
        if held_out and not confirmation:
            borrowed = sorted(key for key in held_out if selected.get(key) == held_out[key])
            if borrowed:
                return (f"the search measured at the held-out {'/'.join(borrowed)} value(s) "
                        f"reserved for confirmation")
        stage = self.backend.stages.get(target) or {}
        source = self.backend.sources.get(target) or ""
        projection = {"schema_version": 1, "target": target,
                      "settings": selected,
                      "confirmation": held_out,
                      "protocol_keys": extra,
                      "task_contract": self.declaration.get("task_contract") or {},
                      "metric": self.metric_spec.as_dict(),
                      "evaluator_stage": stage,
                      "evaluator_parameters": self.backend.parameters.get(target) or {},
                      "evaluator_argv_source": source,
                      "coverage": "declared_and_generic_keys_only"}
        if getattr(self, "guardrail_specs", []):
            projection["guardrail_metrics"] = self.guardrail_specs
        try:
            if path.exists():
                old = read_json(path)
                if not isinstance(old, dict) or not isinstance(old.get("settings"), dict):
                    raise ValueError("frozen comparison protocol is malformed")
                # The settings this call is entitled to have. For every measurement but the
                # confirmation that is exactly the frozen set, which is what makes the freeze a
                # freeze. For the confirmation it is the frozen set with the held-out values
                # applied -- and nothing else, so a confirmation that drifted on some other key
                # is a fresh evaluation of a different question reported as a confirmation of
                # this one.
                allowed = {**old["settings"], **held_out} if confirmation else old["settings"]
                drifted = sorted(key for key in set(selected) | set(allowed)
                                 if selected.get(key) != allowed.get(key))
                if drifted:
                    return ("a confirmation measurement may differ from the frozen protocol "
                            f"only in the held-out keys, and differs in {drifted}"
                            if confirmation else
                            f"immutable comparison protocol changed: settings {drifted}")
                # Everything else the protocol fixes: the evaluator, the task contract, the
                # metric, and the split itself. The settings are taken out of both sides
                # because `drifted` above has already judged them, against the set this call
                # is entitled to rather than against the frozen one -- comparing them again
                # here would refuse every confirmation, which is the one measurement whose
                # settings are supposed to differ.
                without = {**projection, "settings": None}
                if object_digest(without) != object_digest({**old, "settings": None}):
                    return ("immutable comparison protocol changed: evaluator, task contract, "
                            "metric, or held-out split")
            elif freeze:
                immutable_json(path, projection)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            return f"comparison protocol cannot be verified: {type(exc).__name__}: {exc}"
        return ""

    def _held_out(self) -> dict[str, Any]:
        """The settings reserved for confirmation, declared rather than inferred.

        Which setting decides *which episodes* an evaluation runs is a fact about a benchmark:
        a seed for one, a task list for another, a fixed initial-state bank for a third, where
        the answer is that nothing varies and no split is possible. This system has no way to
        read that off the code, and a guess at it would reserve the wrong thing -- so the
        declaration supplies it, and a run that does not declare one is not given a split it
        did not choose.

        `research_goal.confirmation` names the keys and the values:
        `{"seed": 101}` means the search measures at the frozen seed and the one confirmation
        measures at 101. An empty or absent block means no split, and the run's best number is
        then reported as the maximum of draws it is.

        Two ways a declared split can be vacuous, both refused at freeze time because both look
        like a split and are not one: a key the protocol does not freeze, whose held-out value
        would be silently ignored, and a held-out value equal to the frozen one, which would
        reserve an episode set the search is already measuring.
        """
        goal = self.declaration.get("research_goal") or {}
        self._held_out_error = ""
        declared = goal.get("confirmation")
        if not declared:
            return {}
        if not isinstance(declared, dict) or any(not isinstance(key, str) or not key
                                                 for key in declared):
            self._held_out_error = ("research_goal.confirmation must be an object mapping a "
                                    "setting name to the value reserved for confirmation")
            return {}
        extra = goal.get("protocol_keys") or []
        exact = {"task", "tasks", "suite", "benchmark", "seed", "seeds",
                 "episodes", "n_episodes", "horizon", "max_episode_steps",
                 "initial_states", "eval_seed", *extra}
        outside = sorted(key for key in declared
                         if key not in exact and not key.startswith(("eval.", "evaluation.")))
        if outside:
            self._held_out_error = (
                f"research_goal.confirmation reserves {outside}, which the comparison protocol "
                f"does not freeze, so the reservation would have no effect")
            return {}
        return dict(declared)

    #: A success reading as benchmarks actually print one: a word, a separator of any of
    #: `:`, `=`, `.` or nothing, then the number. Two quantities wear the same word and are
    #: not the same as the thing being measured, so both are excluded rather than averaged
    #: in -- `best succ: 0.90`, which is the best epoch so far rather than the last one, and
    #: `succ. AoC 0.00`, which LIBERO computes over all tasks seen with counterbalancing.
    #: Capturing the run of digits also picks up a per-task table (`[All task succ.] 0.10 |
    #: 0.20 |`), which is why several values are reduced to their mean below.
    @staticmethod
    def _readings(said: str) -> dict[str, float]:
        """The numbers the program named. See `readings.named_numbers`.

        Kept as a method because the loop calls it, but the reading itself lives in one
        place. Two implementations of "which number did the program print" would drift, and
        this repository has already paid for that twice: a duplicated `error_excerpt`
        silently discarded the fix made to the first copy, and a duplicated `_offending_call`
        did the same to a second.
        """
        return named_numbers(said)

    @staticmethod
    def _was_measured(result: dict[str, Any]) -> bool:
        """Did a number come out, or did the loop only intend to produce one?

        A stage that never ran has no return code, and `None in (0, None)` is true -- so the
        absence of a run read as success, and the round was written down as "measured". The
        report of the LIBERO run says exactly that, for a round in which no stage ran.
        """
        return bool(result.get("ok")) and (result.get("metric_value") is not None or
                                            result.get("success_rate") is not None)

    @staticmethod
    def _failure_context(result: dict[str, Any]) -> dict[str, Any]:
        """Keep the measurement's actual failure and the evidence needed to inspect it.

        The measurement layer distinguishes, for example, a command that never launched
        from a successful training command whose output glob matched six checkpoints. The
        research report must not collapse both into "the stage did not finish".
        """
        stage = str(result.get("where") or result.get("stage") or "measurement")
        stage_record = result.get(stage)
        if not isinstance(stage_record, dict):
            stage_record = {}
        reason = str(result.get("why") or "").strip()
        if not reason:
            if result.get("ran") and result.get("returncode") == 0:
                reason = "command completed, but no valid measurement was established"
            elif result.get("ran"):
                reason = ("command did not produce a valid measurement "
                          f"(returncode={result.get('returncode')})")
            else:
                reason = "command was not run"

        failure: dict[str, Any] = {"stage": stage, "reason": reason}
        for key in ("ran", "status", "returncode", "termination_reason", "attempt_id"):
            if key in result:
                failure[key] = result[key]
            elif key in stage_record:
                failure[key] = stage_record[key]
        artifact = result.get("artifact") or stage_record.get("artifact")
        if isinstance(artifact, dict):
            summary = {key: artifact[key] for key in
                       ("checked", "matched", "freshness_checked", "found_beside_the_command")
                       if key in artifact}
            for key in ("pattern", "examples"):
                if key in artifact:
                    values = artifact[key] if isinstance(artifact[key], list) else [artifact[key]]
                    summary[key] = [redact(str(value))[:300] for value in values[:3]]
            if summary:
                failure["artifact"] = summary
        attempt_id = str(result.get("attempt_id") or stage_record.get("attempt_id") or "")
        if attempt_id:
            failure["evidence"] = {
                "receipt": f"attempts/{attempt_id}/receipt.json",
                "log": f"{stage}/attempts/{attempt_id}/output.log"}

        said = str(result.get("said") or stage_record.get("said") or "")[-800:]
        return {"why_not": f"{stage}: {reason}", "failure": failure,
                **({"said": said} if said else {})}

    @staticmethod
    def _metric_utility(result: dict[str, Any]) -> float | None:
        if (result.get("guardrails") or {}).get("status") in {"unknown", "violated"}:
            return None
        value = result.get("metric_utility")
        return value if isinstance(value, (int, float)) else result.get("success_rate")

    def _attach_guardrails(self, result: dict, *, label: str, said: str, artifact: Path | None):
        from .metric_guardrails import read_values, compare
        specs = getattr(self, "guardrail_specs", [])
        if not specs: return
        result["secondary_metric_readings"] = read_values(specs, said=said, artifact=artifact,
                                                         primary_spec=self.metric_spec)
        baseline_path = self.run_root / "measurements/baseline.json"
        baseline = result if label == "baseline" else read_json(baseline_path) if baseline_path.is_file() else {}
        result["guardrails"] = compare(specs, result["secondary_metric_readings"],
                                         baseline.get("secondary_metric_readings") or {})
        result["eligible_for_best"] = result["guardrails"]["status"] == "passed"

    @staticmethod
    def _success_reading(said: str) -> dict[str, Any]:
        """The reading and its source. See `readings.success_reading`."""
        return success_reading(said)

    @staticmethod
    def _success_rate(said: str) -> float | None:
        """The last labelled success reading the program printed. See `readings`."""
        return success_rate(said)

    def baseline(self, *, settings: dict[str, Any] | None = None,
                 training_attempt_id: str = "") -> dict[str, Any]:
        return self.measure(settings=self._base_settings(settings), label="baseline",
                            _reuse_training_attempt_id=training_attempt_id)

    @staticmethod
    def _base_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
        """The smallest run that measures anything, with the benchmark's own choices kept.

        `device` was pinned to `"cpu"` here, which is not a neutral default: LIBERO's own
        configuration says `cuda`, and a baseline that takes about ninety seconds on this
        machine's card ran for over an hour without finishing on the CPU the caller never
        asked for. A default that overrides the benchmark's is not conservative, it is a
        different experiment -- and the difference was the whole wall clock.
        """
        return dict(settings or {})

    def propose(self, evidence: dict[str, Any], *, evidence_id: str,
                round_index: int = 0) -> dict[str, Any] | None:
        """Ask the decision layer what to vary, against the benchmark's declared space."""
        # The method library, which every other path passes and this one did not. It is
        # reference and not policy -- nothing validates against it and the controller may
        # contradict any entry -- so passing it cannot make a proposal invalid. Not passing it
        # means the controller reasons with no methods at all, which it was doing.
        evidence_keys = " ".join(str(key) for key in sorted(evidence))
        evidence_stages = " ".join(
            str(row.get("stage") or row.get("family") or "")
            for row in (evidence.get("attempts") or []) if isinstance(row, dict))
        method_library = skills_reference(
            benchmark=self.benchmark or None,
            query=("research intervention hypothesis experiment comparison metric budget "
                   "failure diagnosis adaptation " + evidence_keys + " " + evidence_stages))
        system, user = build_request(self.space, evidence=evidence, evidence_id=evidence_id,
                                     method_library=method_library)
        from .agent_client import role_scope
        with role_scope(self.client, "ideator"):
            content, metadata = self.client.chat_with_metadata(
                system, user, max_tokens=3000, timeout=240, thinking="disabled")
        # The response and what was asked for, kept. This used to be `content, _ = ...`, and
        # the underscore held the provider metadata: which model answered, how many tokens it
        # took, whether it finished. That is the only record of what the controller was shown
        # and what it said, and it was being discarded one line after it arrived -- so a
        # proposal could be read back but not explained.
        self._record_exchange(f"round_{round_index}", system=system, user=user, content=content,
                              metadata=metadata or {},
                              method_selection=method_library.get("selection", []))
        text = content.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            raw = json.loads(text[text.find("{"):text.rfind("}") + 1])
            proposal = validate_proposal(raw, space=self.space, evidence_id=evidence_id)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.decisions.record(Decision(
                activity="propose a change", by="model", agent=self._model_name(),
                used=[f"evidence:{evidence_id}"], round=round_index,
                why=f"the proposal was refused: {type(exc).__name__}: {exc}",
                outcome={"state": "known", "what": {"accepted": False, "error": str(exc)[:300]}}))
            self._append_local_event({"at": now(), "event": "proposal_rejected",
                                      "error": redact(
                                          f"{type(exc).__name__}: {exc}")[:400]})
            return None
        # Held so the round that runs it can close it. A decision whose outcome nobody fills
        # in stays open in the record, and reads as exactly that -- which is a finding about
        # the run rather than a decision that had no consequences.
        self.last_proposal_decision = self.decisions.record(Decision(
            activity="propose: " + str(proposal.get("hypothesis") or "")[:120], by="model",
            agent=self._model_name(), used=[f"evidence:{evidence_id}"], round=round_index,
            why=str(proposal.get("hypothesis") or ""),
            produced=[str(self.run_root / "exchanges" / f"round_{round_index}.json")]))
        return proposal

    # -- choosing what to try, and changing the benchmark's own source -------------------

    def _note(self, event: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._append_local_event(
            {"at": now(), "event": event, "rows": rows},
            shared_event="controller_event",
            shared_details={"controller_event": event, "row_count": len(rows)})

    def _write_audits(self) -> None:
        atomic_json(self.run_root / "audits.json",
                    {"at": now(), "red_lines": self.red_lines().as_dict(),
                     "audits": self.audits})

    def choose_idea(self, *, history: list[str], evidence: dict[str, Any],
                    round_index: int = 0,
                    kinds_wanted: tuple[str, ...] | None = None) -> Any | None:
        """The next thing to try: chosen from the library, and audited before anything runs.

        The library is filled once, from a reading of the benchmark, rather than re-invented
        as one proposal per round -- because a proposal drawn from the declared space can only
        ever name values, and the three changes that mattered on RoboTwin were not values.
        With multiple admissible ideas, selection considers the latest evidence through a
        bounded model call; a deterministic choice is only the failure fallback.

        **The whole batch is audited, before any of it is selected from.** That is AutoSOTA's
        ordering and it is the better one: an idea that meets a red line is marked refused in
        the library and is never a candidate, so the loop cannot walk a library of refusals
        and the record can show, per idea, that it was refused rather than that it was never
        reached. `assert_frozen` already refuses a changed evaluator, but at the moment it is
        used -- after the idea was proposed, selected and applied -- which spends the round and
        leaves a reader looking at a failed run rather than a refused idea.

        A run whose library came back empty still has a move: `_idea_from_proposal`, the
        declared space, when the stages that would read a parameter actually exist.
        """
        self.prepare_idea_options()
        idea = None
        if needs_a_leap(history) and evidence_stalled(evidence):
            # A type streak alone is not failure: three parameter rounds that improve a
            # result are not stuck. Leap only when the measured result or failure diagnosis
            # also stopped changing.
            from .agent_client import role_scope
            with role_scope(self.client, "ideator"):
                fresh = leap_idea(self.client, {**self._material(), "tried": history[-3:]},
                                  avoid=history[-1], on_event=self._note)
            if fresh is not None:
                known = self.library.get(fresh.label) is not None
                idea = self.library.add(fresh)
                self.library.audit(lines)
                self.library.record_leap(
                    because=f"three `{history[-1]}` moves in a row did not make the stage run",
                    tried=list(history[-3:]), produced=fresh.label)
                # A leap that came back under a label the library already holds is not a leap;
                # a leap that met a red line is not one either; and where the run can only be
                # moved by a particular kind of change, a leap of another kind is the round
                # that was already tried under a new name. All three fall through.
                if known or idea.status != redline_rules.CLEARED or (
                        kinds_wanted is not None and idea.granularity not in kinds_wanted):
                    idea = None
        if idea is None:
            idea = self._choose_from_library(history=history, evidence=evidence,
                                             round_index=round_index,
                                             kinds_wanted=kinds_wanted)
        if idea is None and kinds_wanted is None:
            # Only when the required stages exist. A parameter can rescue an OOM or invalid
            # setting even before the first score, but cannot create a missing command.
            idea = self._idea_from_proposal(evidence=evidence, round_index=round_index)
        return idea

    def prepare_idea_options(self, *, extend: bool = False,
                             generate_if_empty: bool = True) -> list[dict[str, Any]]:
        """Build/audit a candidate pool without selecting what the run will execute.

        The outer research controller uses this as an option-producing step. In particular,
        the preparation-level main agent must make the consequential choice of which idea to
        run; this method may ask the model to suggest a pool, but it cannot authorize a stage.
        """
        lines = self.red_lines()
        if (not self.library.ideas and generate_if_empty) or extend:
            from .agent_client import role_scope
            with role_scope(self.client, "ideator"):
                rows = build_ideas(self.client, self._material(), on_event=self._note)
            before = len(self.library.ideas)
            for row in rows:
                self.library.add(row)
            batch_name = ("library" if before == 0 else
                          f"library_refresh_{len(self.events) + 1}")
            self._record_exchange(batch_name,
                                  system=f"[the library draft] {len(rows)} ideas",
                                  user=json.dumps(self._material(), ensure_ascii=False,
                                                  default=str)[:20000],
                                  content=json.dumps([one.as_dict() for one in rows],
                                                     ensure_ascii=False)[:20000],
                                  metadata={})
            self._note("library_refreshed" if before else "library", [{
                "status": "the candidate library was extended" if before else
                          "the library was drafted",
                "ideas_returned": len(rows),
                "ideas_added": len(self.library.ideas) - before,
                "of_each": {kind: sum(1 for one in rows if one.granularity == kind)
                            for kind in ("param", "code", "algo")}}])
        self._audit_the_batch(lines)
        return [one.as_dict() for one in self.library.usable()]

    def propose_research_idea(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Record and audit one idea supplied by the outer research controller.

        This is deliberately separate from selection and execution: the main controller can
        originate an idea outside the generated pool, but it cannot make that idea runnable
        merely by proposing it. Shape/space validation and red-line audit happen before it is
        exposed as a selectable option; source edits are still rechecked at application time.
        """
        required = {"label", "granularity", "mechanism", "change", "risk", "crosses",
                    "why", "evidence", "touches"}
        if not isinstance(candidate, dict) or set(candidate) != required:
            return {"status": "invalid", "why":
                    f"idea fields must be exactly {sorted(required)}"}
        label = candidate.get("label")
        if not isinstance(label, str) or not label.strip() or len(label.strip()) > 160:
            return {"status": "invalid", "why": "label must be 1–160 characters"}
        granularity = candidate.get("granularity")
        if granularity not in {"param", "code", "algo"}:
            return {"status": "invalid", "why": "granularity must be param, code, or algo"}
        risk = candidate.get("risk")
        if risk not in {"low", "medium", "high"}:
            return {"status": "invalid", "why": "risk must be low, medium, or high"}
        for key in ("mechanism", "why"):
            if not isinstance(candidate.get(key), str) or not candidate[key].strip():
                return {"status": "invalid", "why": f"{key} must be non-empty text"}
        if not isinstance(candidate.get("change"), dict):
            return {"status": "invalid", "why": "change must be a JSON object"}
        for key in ("evidence", "touches"):
            value = candidate.get(key)
            if (not isinstance(value, list) or
                    any(not isinstance(item, str) or not item.strip() for item in value)):
                return {"status": "invalid", "why": f"{key} must be a list of non-empty strings"}
        if not candidate["evidence"]:
            return {"status": "invalid", "why": "evidence must cite at least one observed fact"}
        if candidate["touches"] and granularity != "code":
            return {"status": "invalid", "why": "only code ideas may name touched files"}
        idea = Idea(label=label.strip(), granularity=granularity,
                    mechanism=candidate["mechanism"].strip(), change=dict(candidate["change"]),
                    risk=risk, crosses=str(candidate.get("crosses") or "").strip(),
                    why=candidate["why"].strip(),
                    evidence=[item.strip() for item in candidate["evidence"]],
                    touches=[item.strip() for item in candidate["touches"]])
        if self.library.get(idea.label) is not None:
            existing = self.library.get(idea.label)
            return {"status": "duplicate", "idea": existing.as_dict(),
                    "why": "idea labels are stable identities; use a new label for a new proposal"}

        if idea.granularity == "code":
            patches = patches_from_change(idea.change)
            if not patches:
                return {"status": "invalid", "why":
                        "code changes need {file, find, replace} or a bounded patches list"}
            actual_files = [patch.file for patch in patches]
            if idea.touches and set(idea.touches) != set(actual_files):
                return {"status": "invalid", "why":
                        "touches must exactly match the file paths in change"}
            idea.touches = actual_files
            problems = check_many(patches, repo=self.repo)
            if problems:
                return {"status": "invalid", "why": "; ".join(problems)[:600]}
        else:
            evidence_id = object_digest({"main_proposal": idea.evidence,
                                         "state_revision": self.controller_decision_id})
            try:
                envelope = self._envelope_for(idea, evidence_id=evidence_id)
                from .decision import validate_proposal
                validate_proposal(envelope, space=self.space, evidence_id=evidence_id)
            except (KeyError, TypeError, ValueError) as exc:
                return {"status": "invalid", "why":
                        f"change does not fit the declared action contract: {exc}"[:600]}

        self.library.add(idea)
        self._audit_the_batch(self.red_lines())
        held = self.library.get(idea.label)
        if held is None:
            return {"status": "unrecorded", "why": "idea was not present after library audit"}
        self._note("main_controller_idea_proposed", [{
            "label": held.label, "status": held.status,
            "controller_decision_id": self.controller_decision_id,
            "granularity": held.granularity, "outcome": held.outcome,
        }])
        return {"status": held.status, "idea": held.as_dict(),
                "why": held.outcome or "proposal passed the current action and red-line audit"}

    def _choose_from_library(self, *, history: list[str], evidence: dict[str, Any],
                             round_index: int,
                             kinds_wanted: tuple[str, ...] | None) -> Any | None:
        """Reason about the current failure and audited candidates, without bypassing audit.

        The model may only return a label already in the admissible pool. Its explanation is
        saved for a reader, but cannot turn a refused idea into an executable one.
        """
        candidates = [one for one in self.library.usable()
                      if execution_compatibility(
                          one, stage_parameters=self.backend.parameters.get("train"))[0]]
        if kinds_wanted is not None:
            candidates = [one for one in candidates if one.granularity in kinds_wanted]
        compatible_library = IdeaLibrary(self.library.path)
        compatible_library.ideas = candidates
        compatible_library.leaps = list(self.library.leaps)
        fallback = select_idea(compatible_library,
                               history=history if evidence_stalled(evidence) else (),
                               of_kind=kinds_wanted)
        if len(candidates) < 2 or not callable(getattr(self.client, "chat_with_metadata", None)):
            return fallback
        system = ("Select exactly one previously audited research idea based on the current "
                  "benchmark evidence. Return JSON {\"label\":\"exact candidate label\","
                  "\"why\":\"evidence-based reason\"}. Only labels in candidates are "
                  "allowed. Never change evaluation rules, metrics, seeds, or protected files.")
        material = {
            "evidence": evidence, "recent_kinds": history[-6:],
            "kinds_wanted": kinds_wanted,
            "candidates": [{"label": one.label, "kind": one.granularity,
                            "mechanism": one.mechanism, "risk": one.risk,
                            "why": one.why, "evidence": one.evidence,
                            "times_tried": one.times_tried, "last_outcome": one.outcome}
                           for one in candidates[:24]],
        }
        user = json.dumps(material, ensure_ascii=False, default=str)[:24000]
        try:
            from .agent_client import role_scope
            with role_scope(self.client, "scheduler"):
                content, metadata = self.client.chat_with_metadata(
                    system, user, max_tokens=600, timeout=90, thinking="disabled")
            self._record_exchange(f"select_{round_index}", system=system, user=user,
                                  content=content, metadata=metadata or {})
            from .execution_derive import _object
            answer = _object(content)
            chosen = next((one for one in candidates[:24]
                           if one.label == answer.get("label")), None)
            if chosen is None:
                raise ValueError("selected label is not an audited candidate")
            self._note("idea_selected", [{"label": chosen.label, "by": "model",
                                          "why": redact(str(answer.get("why") or ""))[:600]}])
            return chosen
        except Exception as exc:  # noqa: BLE001 - a model failure must not stop a run
            self._note("idea_selection_fallback", [{"label": fallback.label if fallback else None,
                  "why": redact(f"{type(exc).__name__}: {exc}")[:300]}])
            return fallback

    def _idea_from_proposal(self, *, evidence: dict[str, Any],
                            round_index: int = 0) -> Any | None:
        """The declared-space path, wrapped as an idea so the loop has one shape to run.

        Kept because it always answers: the library is built by a model call and can come back
        empty, and a loop with one source of proposals is a loop that stops when that source
        does. A value cannot create a missing stage, but can fix an executable stage that
        fails on its current settings; it remains a real fallback move.

        `evidence` is the loop's own and not a copy made here. A proposal is made against what
        was known when it was made, and a second evidence dict assembled at this depth would
        leave out the round that has just been measured -- which is the whole point of running
        a protocol rather than a sweep.
        """
        from .ideas import Idea
        try:
            proposal = self.propose(evidence, evidence_id=object_digest(evidence),
                                    round_index=round_index)
        except Exception as exc:                                     # noqa: BLE001
            # A proposal is asked for over the network, and a network that is down is not a
            # finding about the benchmark. Unguarded, one failed call ends the run from inside
            # the step that was supposed to give the loop something to do -- which is what
            # happened here the first time this path was exercised.
            self._note("proposal_failed", [{
                "status": "the declared-space proposal could not be made",
                "error": redact(f"{type(exc).__name__}: {exc}")[:400]}])
            return None
        if not proposal:
            return None
        if proposal.get("decision") == "stop":
            # The controller's own stop, in the shape the loop now reads: an idea it will not
            # run, carrying the reason.
            return Idea(label=self.CONTROLLER_STOPPED, granularity="param",
                        mechanism=str(proposal.get("hypothesis") or "")[:300],
                        change={}, risk="low",
                        why=str(proposal.get("hypothesis") or ""),
                        status=redline_rules.CLEARED,
                        outcome=str(proposal.get("hypothesis") or "the controller stopped"))
        idea = Idea(label=f"declared-space {round_index}: "
                          f"{str(proposal.get('hypothesis') or 'proposal')[:60]}",
                    granularity="param",
                    mechanism=str(proposal.get("hypothesis") or "")[:300],
                    change=dict(proposal.get("training") or {}),
                    why=str(proposal.get("hypothesis") or ""),
                    status=redline_rules.CLEARED)
        # The proposal's collection half, held against the label. Not folded into `change`:
        # `collection` is also the name of a declared section, and a settings map carrying a
        # key `collection` would be read as an axis.
        self._proposals_by_idea[idea.label] = proposal
        self._from_declared_space.add(idea.label)
        return idea

    def _envelope_for(self, idea: Any, *, evidence_id: str) -> dict[str, Any]:
        """The idea as the protocol's own proposal shape, so the space can check it.

        A library idea is not a full proposal and should not have to be. `decision`,
        `proposal_id` and `expected_validation` are the protocol's envelope -- the loop has
        already decided to run this, the label identifies it, and the idea's own `why` is what
        it expects to see. Making the model write those would be asking it to fill in forms
        about a decision it is not making.

        What the space must check is the part that *is* the change: the axis names and their
        values. Axes the idea does not name are filled from what the space declares. Every
        required axis has a default -- a space without one does not build -- so "unnamed" has
        an answer, and the answer is the benchmark's own choice.
        """
        from .decision import EVIDENCE_FIELD
        sections = set(self.space.sections())
        change = dict(idea.change or {})
        if any(key in sections for key in change):
            named = {name: dict(change.get(name) or {}) for name in sections}
            remainder = {key: value for key, value in change.items() if key not in sections}
        else:
            # A bare settings map, which is how a `param` idea is written.
            named = {name: {} for name in sections}
            named["training"] = dict(change)
            remainder = {}
        envelope: dict[str, Any] = {
            "decision": "experiment",
            "proposal_id": idea.label[:80],
            "hypothesis": idea.mechanism or idea.label,
            "expected_validation": idea.why or idea.mechanism or idea.label,
            EVIDENCE_FIELD: evidence_id,
        }
        for section, axes in self.space.sections().items():
            supplied = dict(named.get(section) or {}) if section in named else {}
            supplied.update({key: value for key, value in remainder.items()
                             if any(axis.name == key for axis in axes)})
            for axis in axes:
                if axis.group or axis.name in supplied or axis.optional:
                    continue
                if axis.default is not None:
                    supplied[axis.name] = axis.default
                elif axis.accepts(None):
                    # Required, with nothing declared to put there, and the axis says null is
                    # a value it takes. Naming it explicitly rather than leaving it out, so
                    # the record shows the choice instead of an absence a reader has to
                    # interpret.
                    supplied[axis.name] = None
            envelope[section] = supplied
        return envelope

    def _audit_the_batch(self, lines: Any) -> None:
        """Audit everything not yet audited, and record it for the run's document."""
        verdicts = self.library.audit(lines)
        if not verdicts:
            return
        for found in verdicts:
            held = self.library.get(found.idea)
            self.audits.append({"idea": found.idea, "verdict": found.verdict,
                                "line": found.line, "because": found.because,
                                "granularity": held.granularity if held else "",
                                "file": str((held.change or {}).get("file") or "")
                                if held else "",
                                "risk": held.risk if held else ""})
        self._write_audits()
        self._note("audit", [{"idea": one.idea, "verdict": one.verdict, "line": one.line}
                             for one in verdicts])

    def _prepare_idea(self, idea: Any, *, evidence: dict[str, Any],
                      round_index: int | None = None) -> dict[str, Any]:
        """Make this idea runnable -- change the file, or fit the declared space.

        Both refusals are one situation: the model wrote something the system could not read.
        And both get one answer: show it what was wrong and the facts it did not have, and let
        it correct itself. For a `code` idea the missing fact is the file -- it wrote an exact
        `find` without ever having seen it. For a `param` or `algo` idea it is the sections and
        axes the benchmark actually declares.

        That is `ideas.repair`, and it is the framework's answer. The alternative is a rule per
        spelling, and this file had three of them before this replaced them: rules that moved a
        section out of a section, joined a stage name to an axis, and un-doubled a repeated
        key. Each covered the spellings its author had seen. This covers the ones nobody has.
        """
        refused, repair_note, changed = "", "", {}
        # Which refusal it was, so the record says what happened rather than that something
        # did. Two ways in, one channel out.
        status = "the change could not be applied"
        for attempt in (1, 2):
            if idea.granularity == "code":
                # Only `code` changes a file. An `algo` idea names what the run does -- which
                # stages, on what data -- and that travels the way a declared-space proposal
                # does. Sending it to `code_change` asked it for a `find` it had never been
                # given.
                changed = self.code_change(idea, transaction_round=round_index)
                if not changed.get("applied"):
                    refused = redact(str(changed.get("why") or ""))[:400]
            elif idea.label not in self._from_declared_space:
                # Through the validator the loop always used. A value idea names settings, and
                # the space is what says which names exist. An `algo` idea goes the same way:
                # what it names is which stages run and on what data, and both are declared
                # settings rather than free text. A proposal straight from `propose` was
                # validated when it was made and is not checked twice.
                try:
                    self._proposals_by_idea[idea.label] = validate_proposal(
                        self._envelope_for(idea, evidence_id=object_digest(evidence)),
                        space=self.space, evidence_id=object_digest(evidence))
                except (ValueError, KeyError) as exc:
                    refused = redact(f"{type(exc).__name__}: {exc}")[:400]
                    status = "the idea does not fit the declared space"
            if not refused:
                return {"refused": False, "changed": changed,
                        "proposal": self._proposals_by_idea.get(idea.label)
                                    or {"hypothesis": idea.mechanism}}
            from .agent_client import role_scope
            with role_scope(self.client, "fix"):
                repaired = repair_idea(self.client, idea, [refused], repo=self.repo,
                                       space=self.space, on_event=self._note)
            # What the model said when it corrected the change, or why it could not. A repair
            # that gave up leaves its reason on the idea, and that reason is the finding.
            repair_note = idea.outcome or "the change could not be made to fit"
            if repaired is None:
                break
            refused = ""
        return {"refused": True, "why": refused, "repair": repair_note, "status": status}

    def code_change(self, idea: Any, *,
                    transaction_round: int | None = None) -> dict[str, Any]:
        """Apply this idea's change to the benchmark's own source, or say why it cannot be.

        The change is a find-and-replace and is checked before it is applied: the file has to
        contain the text exactly once, and the result has to parse. Both are faults that would
        otherwise be discovered by running, which is the more expensive way to find them --
        and `find` occurring twice is the one that matters, because the change would then be
        applied to one of two identical places and the run would not know which.

        What was there is kept, so undoing is exact rather than a second guess. The snapshot
        is taken before the patch and not after it: a snapshot taken afterwards records the
        changed state, which is not a thing anyone needs to be able to return to.
        """
        patches = patches_from_change(idea.change)
        if not patches:
            return {"applied": False,
                    "why": "the change needs {file, find, replace} or a bounded patches list"}
        # The idea's self-reported `touches` list is not the actual patch. Check the file in
        # the executable change as well; otherwise a model can omit an evaluator from its
        # description and pass the earlier audit while still editing that evaluator.
        from types import SimpleNamespace
        files = [patch.file for patch in patches]
        actual = redline_rules.audit(SimpleNamespace(
            label=str(getattr(idea, "label", "code patch")), touches=files,
            crosses="none"), self.red_lines())
        if not actual.cleared:
            return {"applied": False, "why": actual.because, "file": files[0],
                    "red_line": actual.line}
        problems = check_many(patches, repo=self.repo)
        if problems:
            return {"applied": False, "why": "; ".join(problems), "file": files[0]}
        snapshot_name = (f"before-round-{transaction_round}" if transaction_round is not None
                         else f"before-{idea.label}")
        manifest_path: Path | None = None
        if transaction_round is not None:
            if (isinstance(transaction_round, bool) or
                    not isinstance(transaction_round, int) or transaction_round < 1):
                raise ValueError("code-change transaction round must be a positive integer")
            manifest_root = self.run_root / "code_changes"
            if manifest_root.is_symlink():
                raise ValueError("code-change transaction directory is a symlink")
            manifest_root.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_root / f"round-{transaction_round}.json"
            if manifest_path.is_symlink():
                raise ValueError("code-change transaction record is a symlink")
            if manifest_path.exists():
                raise ValueError("code-change transaction record already exists")
        self.snapshots.capture(files, repo=self.repo,
                               name=snapshot_name,
                               why=f"before applying {idea.label}")
        if manifest_path is not None:
            # This is committed before the first source write. If the controller dies while
            # applying a multi-file idea, recovery has the exact repaired patch and its
            # pre-change snapshot rather than an earlier, possibly repaired-away proposal.
            atomic_json(manifest_path, {
                "schema_version": 1, "round": transaction_round,
                "idea_label": str(idea.label), "snapshot": snapshot_name,
                "patches": [one.as_dict() for one in patches],
                "status": "applying", "created_at": now(),
            })
        before = apply_many(patches, repo=self.repo)
        if manifest_path is not None:
            manifest = read_json(manifest_path)
            manifest.update(status="applied", applied_at=now())
            atomic_json(manifest_path, manifest)
        for patch in patches:
            self.touched.add(patch.file)
            self._held[patch.file] = (patch, before[patch.file])
        return {"applied": True, "file": files[0], "files": files,
                **({"snapshot": snapshot_name,
                   "transaction_ref": str(manifest_path.relative_to(self.run_root))}
                   if manifest_path is not None else {}),
                "diff": "\n".join(patch_diff(patch) for patch in patches)}

    def undo_code_change(self, file: str) -> bool:
        """Put that file back exactly as it was, and say whether anything moved.

        Called when a change did not work. A change that is left behind is not a lost round:
        it is a checkout the next round runs on top of, and the round after that on top of
        both -- so the run's later failures are attributed to the wrong thing, and the number
        it finally reports came from a benchmark nobody chose.
        """
        held = self._held.pop(file, None)
        if held is None:
            return False
        patch, before = held
        revert_patch(patch, before, repo=self.repo)
        return True

    def _summarize(self) -> None:
        """Rebuild the run's document, with a freshly written summary.

        Called at the end of a stage of work rather than continuously: a summary costs a model
        call and a summary per stage would be four summaries of the same round. A round is the
        unit a person wants summarised.
        """
        # Main-agent runs narrate at the Scheduler boundary, not through a second
        # legacy free-form Scheduler summary turn.
        run_record.generate(self.run_root, client=(None if getattr(
            self.client, "supports_recorder", False) else self.client), refresh_summary=True,
                            title=f"{self.benchmark or self.run_id} / derived")

    def _record_shared_event(self, event: str, *, details: dict[str, Any],
                             phase_state: dict[str, Any],
                             event_patch: dict[str, Any] | None = None,
                             status: str = "running") -> None:
        """Project a research event into the run-wide, hash-chained event stream."""
        if self.state_persistence_error:
            raise ResearchStatePersistenceError(self.state_persistence_error)
        try:
            self.state_store.record("research", event, status=status, details=details,
                                    phase_state=phase_state, event_patch=event_patch)
        except (ResearchStateError, OSError, TypeError, ValueError) as exc:
            self.state_persistence_error = redact(
                f"run_state event {event} failed: {type(exc).__name__}: {exc}")[:400]
            warning = {"at": now(), "event": "run_state_event_failed",
                       "why": self.state_persistence_error[:300],
                       "intended_event": event}
            self.events.append(warning)
            try:
                atomic_json(self.run_root / "events.json", {"rows": self.events})
            except OSError:
                pass
            # A warning in the local log is not a substitute for the shared recovery
            # journal. Propagate so both controllers stop and the next run reconciles the
            # last durable action instead of executing another stage.
            raise ResearchStatePersistenceError(self.state_persistence_error) from exc

    def _append_local_event(self, row: dict[str, Any], *, shared_event: str | None = None,
                            shared_details: dict[str, Any] | None = None,
                            shared_status: str = "running") -> dict[str, Any]:
        """Persist one detailed event and link it from the run-wide event stream."""
        local = {"sequence": len(self.events) + 1, "event_id": uuid.uuid4().hex, **row}
        local_hash = object_digest(local)
        local["event_sha256"] = local_hash
        self.events.append(local)
        atomic_json(self.run_root / "events.json", {"rows": self.events})
        event_name = shared_event or str(local.get("event") or "research_event")
        details = {"local_event_ref": str((self.run_root / "events.json").relative_to(self.output)),
                   "local_event_sequence": local["sequence"],
                   "local_event_id": local["event_id"],
                   "local_event_sha256": local_hash,
                   **dict(shared_details or {})}
        summary = {key: local[key] for key in
                   ("event", "stage", "round", "label", "status", "returncode", "why")
                   if key in local}
        self._record_shared_event(
            event_name, details=details,
            phase_state={"last_local_event": summary,
                         "last_local_event_sequence": local["sequence"]},
            event_patch={"last_local_event": summary,
                         "last_local_event_sequence": local["sequence"]},
            status=shared_status)
        return local

    def _set_research_action(self, step: str, *, status: str = "running",
                             details: dict[str, Any] | None = None) -> None:
        """Publish which part of the research controller owns the next decision."""
        at = now()
        action = {"step": step, "status": status, "updated_at": at,
                  **dict(details or {})}
        if self.controller_decision_id:
            action["controller_decision_id"] = self.controller_decision_id
        if status == "running":
            action.setdefault("started_at", at)
            self.active_research_action = action
            current_action, last_action = action, None
        else:
            self.active_research_action = {}
            current_action, last_action = None, action
        self._record_shared_event(
            "research_action_started" if status == "running" else "research_action_finished",
            details={"action": action},
            phase_state={"current_action": current_action, "last_action": last_action},
            event_patch={"current_action": current_action, "last_action": last_action},
            status="running" if status == "running" else status)

    def _model_name(self) -> str:
        return str(getattr(self.client, "model", "") or "model")

    def _record_exchange(self, name: str, *, system: str, user: str, content: str,
                         metadata: dict[str, Any],
                         method_selection: list[dict[str, Any]] | None = None) -> None:
        """What the controller -- or the library's author -- was shown and what it answered.

        Named rather than numbered, because the library is drafted once before any round and
        an exchange filed as `round_0` would read as a round that happened. Bounded, because
        the prompt carries the method library and the response is prose -- but bounded far
        above what a summary would hold, because the point is that a reader can go and look.
        A hash proves an answer was not altered; only the answer says what it was.
        """
        directory = self.run_root / "exchanges"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / f"{name}.json",
                    {"exchange": name, "at": now(),
                     "system": system[:20000], "user": user[:20000],
                     "response": content[:20000],
                     "provider": metadata,
                     "method_selection": method_selection or [],
                     "response_sha256": object_digest(content)})

    def _research_session_identity(self, *, rounds: int,
                                   settings: dict[str, Any]) -> str:
        """Bind a resumable search to the protocol inputs it was created with."""
        return object_digest(self._research_session_inputs(rounds=rounds,
                                                           settings=settings))

    def _research_session_inputs(self, *, rounds: int,
                                 settings: dict[str, Any]) -> dict[str, Any]:
        """The exact immutable inputs to a search session, named for mismatch diagnosis."""
        return {
            "run_id": self.run_id, "repository": str(self.repo),
            "rounds": rounds, "settings": settings,
            "declaration": self.declaration,
            "metric": self.metric_spec.as_dict(),
            "stages": self.sources,
            "execution_graph": self.execution_graph,
        }

    def _research_session_input_hashes(self, *, rounds: int,
                                       settings: dict[str, Any]) -> dict[str, str]:
        """Persist component digests so a rejected resume says which input class changed."""
        return {name: object_digest(value) for name, value in
                self._research_session_inputs(rounds=rounds, settings=settings).items()}

    def _controller_report(self, *, history: list[dict[str, Any]],
                           stopped_because: str, run_status: str,
                           next_round: int, rounds: int) -> dict[str, Any]:
        best = self.snapshots.best()
        return {"schema_version": 1, "created_at": now(), "repo": str(self.repo),
                "run_id": self.run_id, "run_status": run_status,
                "rounds": [dict(row) for row in history],
                **({"stopped_because": stopped_because} if stopped_because else {}),
                "objective": self.rubric.as_dict(),
                "red_lines": self.red_lines().as_dict(),
                "ideas": self.library.summary(),
                "best": best.as_dict() if best is not None else None,
                "confirmation": self.confirmation_state(),
                "available_stages": sorted(s for s in self.sources if self.available(s)),
                "next_round": next_round, "planned_rounds": rounds}

    def run(self, *, rounds: int = 2, settings: dict[str, Any] | None = None,
            confirm: bool = False, yield_after_action: bool = False,
            max_rounds_per_action: int = 1,
            resume_interrupted: bool = False,
            selected_idea_label: str = "",
            main_controller_owns_selection: bool = False,
            baseline_training_attempt_id: str = "") -> dict[str, Any]:
        """Baseline, then one idea per round, each measured against the last.

        A round that cannot be proposed or cannot run ends the loop rather than being
        retried, and says which of the two it was: a benchmark the controller will not or
        cannot propose against is a finding about the declaration, not a transient fault.

        What a round works on comes from the library rather than from a fresh proposal against
        the declared space, and that is the difference between a loop that can only tune and
        one that can repair. The declared space holds values. The three diagnoses RoboTwin
        produced in a single run -- an entrypoint that is a shell script and a command built as
        `python eval.sh`; a staging script that has to find its inputs by content and name the
        destination the way the loader opens it -- are not values, and a loop whose only move
        is to name a value can state each of them correctly and then stop.

        And it no longer stops when the baseline cannot be measured. That was the right
        response to a proposal drawn from the space, because a value cannot make an evaluation
        run; it is the wrong response to an idea, because the idea most worth having in that
        state is the one that makes the evaluation run. What replaces the stop is the
        objective: the run reports how far it got, and works on the rung it is on.
        """
        if rounds < 0 or max_rounds_per_action < 1:
            raise ValueError("rounds must be nonnegative and action round limit positive")
        if yield_after_action and confirm:
            raise ValueError("yielding research uses the separate confirmation action")
        if main_controller_owns_selection and (
                not yield_after_action or max_rounds_per_action != 1):
            raise ValueError("main-controlled idea selection requires one bounded action")
        base_settings = self._base_settings(settings)
        session_path = self.run_root / "controller_session.json"
        if baseline_training_attempt_id:
            if session_path.exists() or not self._verified_training_receipt(
                    baseline_training_attempt_id, base_settings):
                raise ResearchStateError("detached baseline training receipt is missing, "
                                         "mismatched, or research already started")
        report_path = self.run_root / "research_report.json"
        session_input_hashes = self._research_session_input_hashes(
            rounds=rounds, settings=base_settings)
        session_identity = self._research_session_identity(rounds=rounds,
                                                           settings=base_settings)
        session: dict[str, Any] | None = None
        resume_finalization = False
        restart_interrupted_baseline = False
        recovered_baseline: dict[str, Any] | None = None
        if session_path.is_file():
            try:
                session = read_json(session_path)
            except (OSError, ValueError, TypeError) as exc:
                raise ResearchStateError(
                    f"resumable research state cannot be read: {type(exc).__name__}: {exc}")
            if (not isinstance(session, dict) or session.get("schema_version") != 1 or
                    session.get("run_id") != self.run_id or
                    Path(str(session.get("repository") or "")).expanduser().resolve() !=
                    self.repo):
                raise ResearchStateError("resumable research state identity is invalid")
            if session.get("input_identity") != session_identity:
                saved_hashes = session.get("input_component_sha256")
                if isinstance(saved_hashes, dict):
                    changed = sorted(name for name in
                                     set(saved_hashes) | set(session_input_hashes)
                                     if saved_hashes.get(name) != session_input_hashes.get(name))
                    fields = ", ".join(changed[:12]) or "identity metadata"
                    raise ResearchStateError(
                        "research inputs changed since the paused action "
                        f"({fields}); start a new run rather than mixing protocols")
                raise ResearchStateError(
                    "research inputs changed since the paused action; this older session has "
                    "no component digests to isolate the mismatch, so start a new run rather "
                    "than mixing protocols")
            saved_hashes = session.get("input_component_sha256")
            if (saved_hashes is not None and
                    (not isinstance(saved_hashes, dict) or
                     saved_hashes != session_input_hashes)):
                raise ResearchStateError(
                    "research session component digests do not match its input identity; "
                    "refusing to resume altered state")
            if session.get("status") == "completed":
                try:
                    completed = read_json(report_path)
                except (OSError, ValueError, TypeError) as exc:
                    raise ResearchStateError("completed research report cannot be read: "
                                             f"{type(exc).__name__}: {exc}")
                if (not isinstance(completed, dict) or
                        completed.get("run_id") != self.run_id or
                        completed.get("run_status") != "completed"):
                    raise ResearchStateError("completed research state has no matching report")
                return completed
            if session.get("status") == "running":
                raise ResearchStateError(
                    "a research action has unknown outcome; inspect its attempt receipt and "
                    "outputs before starting a new run")
            if session.get("status") == "interrupted":
                interruption = session.get("interruption")
                reconciliation = session.get("reconciliation")
                if not resume_interrupted:
                    raise ResearchStateError(
                        "an interrupted research action requires an explicit controller "
                        "decision after receipt reconciliation")
                if (not isinstance(interruption, dict) or
                        not isinstance(reconciliation, dict) or
                        reconciliation.get("status") != "reconciled"):
                    raise ResearchStateError(
                        "interrupted research state has no verified process reconciliation")
                if interruption.get("controller_action") != "baseline":
                    pending = session.get("pending_action")
                    pending_summary = (
                        f"; pending round={pending.get('round')} phase="
                        f"{pending.get('phase')} idea="
                        f"{(pending.get('idea') or {}).get('label')}"
                        if isinstance(pending, dict) else "")
                    raise ResearchStateError(
                        "candidate action was interrupted; its pending decision is preserved "
                        "for inspection, but safe completion/rollback is not yet implemented; "
                        "automatic replay is unsafe and the run must preserve this as an "
                        "unresolved boundary" + pending_summary)
                baseline_path = self.run_root / "measurements" / "baseline.json"
                if baseline_path.exists():
                    from .receipt_verifier import verify_measurement
                    verified = verify_measurement(self.run_root, "baseline")
                    if verified.get("status") != "consistent":
                        raise ResearchStateError(
                            "an interrupted baseline has an existing but unverifiable "
                            "measurement; refusing to overwrite or adopt it")
                    try:
                        recovered_baseline = read_json(baseline_path)
                    except (OSError, ValueError, TypeError) as exc:
                        raise ResearchStateError(
                            "interrupted baseline measurement cannot be read: "
                            f"{type(exc).__name__}: {exc}")
                    if (not isinstance(recovered_baseline, dict) or
                            recovered_baseline.get("label") != "baseline"):
                        raise ResearchStateError(
                            "interrupted baseline measurement identity is invalid")
                    measured = self._was_measured(recovered_baseline)
                    if not measured:
                        raise ResearchStateError(
                            "baseline receipt is consistent but has no valid metric")
                    baseline_rate = self._metric_utility(recovered_baseline)
                    history = [{"round": 0, "label": "baseline",
                                "success_rate": recovered_baseline.get("success_rate"),
                                "metric_value": recovered_baseline.get("metric_value"),
                                "metric_utility": baseline_rate,
                                "metric_name": self.metric_spec.name,
                                "settings": recovered_baseline.get("settings"),
                                "measured": True}]
                    session.update(status="paused", action="", history=history, kinds=[],
                                   baseline_rate=baseline_rate,
                                   current_measurement_label="baseline", next_round=1,
                                   stopped_because="", failure_demo_attempted=False,
                                   recovered_baseline_from_interruption=True,
                                   updated_at=now())
                    atomic_json(session_path, session)
                else:
                    # The baseline had no durable scored measurement. The outer controller
                    # has explicitly chosen to resume after inspecting the reconciled attempt;
                    # preserve old receipts and start a new attempt under the same protocol.
                    restart_interrupted_baseline = True
                    session.update(status="running", action="baseline", updated_at=now())
                    atomic_json(session_path, session)
            if session.get("status") not in {"paused", "finalizing"}:
                if not restart_interrupted_baseline:
                    raise ResearchStateError(
                        "research session is neither paused, finalizing, nor completed")
            resume_finalization = session.get("status") == "finalizing"

        rubric = self.build_rubric(on_event=self._note)
        if recovered_baseline is not None:
            baseline_rate = self._metric_utility(recovered_baseline)
            self.rubric = objective.evaluate(
                rubric, self.rubric_facts(rounds=session.get("history") or [],
                                          baseline=baseline_rate,
                                          current=recovered_baseline))
            self._write_rubric()
            metric_scale = ("success_rate" if self.metric_spec.name == "success_rate" else
                            "metric:" + self.metric_spec.name)
            self._advance_best("baseline", score=baseline_rate, scale=metric_scale,
                               why="recovered from a receipt-verified baseline measurement")
        if session is None or restart_interrupted_baseline:
            # Existing measurement records without their controller checkpoint may belong to
            # a legacy or interrupted loop. Never assume they are safe to repeat or adopt.
            measurements = self.run_root / "measurements"
            if yield_after_action and measurements.is_dir() and any(measurements.glob("*.json")):
                raise ResearchStateError("measurement records exist without a resumable "
                                         "controller checkpoint; refusing to retrain")
            self._set_research_action("build_rubric_and_baseline")
            self._set_research_action("baseline")
            if yield_after_action and session is None:
                session = {"schema_version": 1, "run_id": self.run_id,
                           "repository": str(self.repo), "input_identity": session_identity,
                           "input_component_sha256": session_input_hashes,
                           "status": "running", "action": "baseline", "rounds": rounds,
                           "base_settings": base_settings, "next_round": 1,
                           "current_measurement_label": "", "history": [], "kinds": [],
                           "stopped_because": "", "baseline_rate": None,
                           "failure_demo_attempted": False, "updated_at": now()}
                atomic_json(session_path, session)
            elif yield_after_action and session is not None:
                session.update(status="running", action="baseline",
                               current_measurement_label="", history=[], kinds=[],
                               updated_at=now())
                atomic_json(session_path, session)
            current = (self.baseline(settings=base_settings,
                                     training_attempt_id=baseline_training_attempt_id)
                       if baseline_training_attempt_id else
                       self.baseline(settings=base_settings))
            baseline_measured = self._was_measured(current)
            self._set_research_action("baseline", status="completed",
                                      details={"measured": baseline_measured,
                                               "attempt_id": current.get("attempt_id"),
                                               "measurement_ref":
                                                   f"research/{self.run_id}/measurements/baseline.json"})
            baseline_failure = ({} if baseline_measured else self._failure_context(current))
            baseline_rate = self._metric_utility(current) if baseline_measured else None
            history = [{"round": 0, "label": "baseline",
                        "success_rate": current.get("success_rate"),
                        "metric_value": current.get("metric_value"),
                        "metric_utility": self._metric_utility(current),
                        "metric_name": self.metric_spec.name,
                        "settings": current.get("settings"),
                        "measured": baseline_measured,
                        **baseline_failure}]
            self.rubric = objective.evaluate(rubric, self.rubric_facts(
                rounds=history, baseline=baseline_rate, current=current))
            self._write_rubric()
            metric_scale = ("success_rate" if self.metric_spec.name == "success_rate" else
                            "metric:" + self.metric_spec.name)
            self._advance_best("baseline", score=baseline_rate,
                               scale=metric_scale,
                               why="the benchmark's own configuration, unchanged")
            if baseline_measured:
                self.capture_demo(trigger="first_native_baseline", measurement=current,
                                  settings=self._frozen_search_settings() or base_settings)
            if not baseline_measured:
                self._advance_best("baseline-progress", score=self.rubric.score(),
                                   scale="progress",
                                   why="as far as the benchmark's own configuration got")
                self._note("no_baseline", [{
                    "status": "no baseline number came out, so the run works on the objective",
                    "at": objective.render(self.rubric).splitlines()[0],
                    "frontier": (self.rubric.frontier().question
                                 if self.rubric.frontier() is not None else ""),
                    **baseline_failure}])
            self._summarize()
            kinds: list[str] = []
            stopped_because = ""
            failure_demo_attempted = False
            next_round = 1
            if session is not None:
                session.update(status="paused" if yield_after_action and rounds else "running",
                               action="" if yield_after_action and rounds else "finalizing",
                               history=history, kinds=kinds, baseline_rate=baseline_rate,
                               current_measurement_label="baseline", next_round=next_round,
                               updated_at=now())
                atomic_json(session_path, session)
            if yield_after_action and rounds:
                if main_controller_owns_selection:
                    # Audit only what was already supplied. The outer controller decides
                    # whether to request a generated batch or formulate its own proposal.
                    self.prepare_idea_options(generate_if_empty=False)
                report = self._controller_report(
                    history=history, stopped_because="", run_status="paused",
                    next_round=next_round, rounds=rounds)
                atomic_json(report_path, report)
                self._set_research_action("research_paused", status="paused",
                                          details={"after": "baseline",
                                                   "next_round": next_round,
                                                   "measurement_ref":
                                                       f"research/{self.run_id}/measurements/baseline.json"})
                run_record.generate(self.run_root,
                                    title=f"{self.benchmark or self.run_id} / derived")
                return report
        else:
            self._set_research_action("resume_research_finalization" if resume_finalization
                                      else "resume_research_controller",
                                      details={"next_round": session.get("next_round")})
            history = session.get("history")
            if (not isinstance(history, list) or not history or
                    not isinstance(history[0], dict) or history[0].get("label") != "baseline"):
                raise ResearchStateError("paused research history has no verified baseline")
            next_round = int(session.get("next_round") or 0)
            if next_round < 1 or next_round > rounds + 1:
                raise ResearchStateError("paused research next-round index is invalid")
            kinds = [str(row.get("kind")) for row in history[1:]
                     if isinstance(row, dict) and row.get("kind")]
            stopped_because = str(session.get("stopped_because") or "")
            baseline_rate = session.get("baseline_rate")
            failure_demo_attempted = bool(session.get("failure_demo_attempted"))
            current_label = str(session.get("current_measurement_label") or "")
            if not re.fullmatch(r"(?:baseline|round_[1-9][0-9]*)", current_label):
                raise ResearchStateError("paused research has an invalid current measurement id")
            measurement_path = self.run_root / "measurements" / f"{current_label}.json"
            try:
                current = read_json(measurement_path)
            except (OSError, ValueError, TypeError) as exc:
                raise ResearchStateError("paused research measurement is unavailable: "
                                         f"{type(exc).__name__}: {exc}")
            if not isinstance(current, dict) or current.get("label") != current_label:
                raise ResearchStateError("paused research measurement identity does not match")
            if resume_finalization and next_round <= rounds:
                raise ResearchStateError("finalizing research still has unprocessed rounds")

        if main_controller_owns_selection:
            if session is None:
                if selected_idea_label:
                    raise ResearchStateError(
                        "a fresh run starts with its baseline and cannot consume a candidate label")
            elif resume_interrupted and selected_idea_label:
                raise ResearchStateError(
                    "an interrupted baseline recovery cannot consume a candidate idea label")
            elif session.get("status") == "finalizing" and selected_idea_label:
                raise ResearchStateError("finalization cannot consume a candidate idea label")
            elif session.get("status") == "paused" and next_round <= rounds:
                if not selected_idea_label:
                    raise ResearchStateError(
                        "the main controller must select one exact audited idea before the "
                        "paused round can resume")
                measuring = any(
                    row.get("metric_value", row.get("success_rate")) is not None
                    for row in history if isinstance(row, dict))
                if self.execution_graph:
                    graph = ExecutionGraph(self.execution_graph)
                    target = str(self.execution_graph["score_target"])
                    commands_exist = all(self.available(name)
                                         for name in graph.order_for(target))
                else:
                    commands_exist = self.available("evaluate") and (
                        self.available("train") or bool(self.checkpoint) or
                        self.source_policy)
                kinds_wanted = (None if measuring or commands_exist else {"code", "algo"})
                eligible = [one for one in self.library.usable()
                            if (kinds_wanted is None or one.granularity in kinds_wanted) and
                            execution_compatibility(
                                one, stage_parameters=self.backend.parameters.get("train"))[0] and
                            declared_space_compatibility(one, space=self.space)[0]]
                if not any(one.label == selected_idea_label for one in eligible):
                    raise ResearchStateError(
                        "selected idea label is not in the current audited, stage-compatible "
                        "option set; session remains paused")
            elif selected_idea_label:
                raise ResearchStateError(
                    "the research plan has no remaining candidate round")

        metric_scale = ("success_rate" if self.metric_spec.name == "success_rate" else
                        "metric:" + self.metric_spec.name)
        stop_round = (min(rounds, next_round + max_rounds_per_action - 1)
                      if yield_after_action else rounds)

        def checkpoint_session(**updates: Any) -> None:
            """Persist the current bounded action before/after it can change run state."""
            if session is None:
                return
            if session_path.is_symlink() or not session_path.is_file():
                raise ResearchStateError("controller session disappeared or became a symlink")
            try:
                durable = read_json(session_path)
            except (OSError, ValueError, TypeError) as exc:
                raise ResearchStateError("controller session cannot be reloaded: "
                                         f"{type(exc).__name__}: {exc}") from exc
            if (not isinstance(durable, dict) or durable.get("schema_version") != 1 or
                    durable.get("run_id") != self.run_id or
                    Path(str(durable.get("repository") or "")).expanduser().resolve() !=
                    self.repo):
                raise ResearchStateError("controller session identity changed during research")
            # The measurement and finalization helpers update this file independently. Merge
            # from its latest durable revision so a later checkpoint cannot erase their journal.
            session.clear()
            session.update(durable)
            requested_pending = updates.get("pending_action", ...)
            finalization_pending = session.get("pending_action")
            if (requested_pending is None and isinstance(finalization_pending, dict) and
                    isinstance(finalization_pending.get("round_finalization"), dict)):
                finalization = dict(finalization_pending["round_finalization"])
                transaction_id = str(finalization.get("transaction_id") or "")
                steps = finalization.get("steps")
                rows = updates.get("history")
                round_index = finalization.get("round")
                committed_row = next((row for row in reversed(rows or [])
                                      if isinstance(row, dict) and
                                      row.get("round") == round_index and
                                      row.get("finalization_id") == transaction_id), None)
                history_step = (steps.get("history_session_commit")
                                if isinstance(steps, dict) else None)
                if (not isinstance(steps, dict) or
                        set(steps) != set(_ROUND_FINALIZATION_STEPS) or
                        not isinstance(committed_row, dict) or
                        not isinstance(history_step, dict) or
                        history_step.get("status") != "started"):
                    raise ResearchStateError(
                        "candidate history commit lacks its finalization journal or matching row")
                unfinished = sorted(name for name, state in steps.items()
                                    if name != "history_session_commit" and
                                    (not isinstance(state, dict) or
                                     state.get("status") not in {"completed", "skipped"}))
                if unfinished:
                    raise ResearchStateError(
                        "candidate history commit has unfinished side effects: " +
                        ", ".join(unfinished))
                steps = dict(steps)
                steps["history_session_commit"] = {
                    "status": "completed",
                    "evidence": {"history_row_sha256": object_digest(committed_row)},
                    "finished_at": now(),
                    "started_at": history_step.get("started_at"),
                }
                finalization.update(status="committed", steps=steps,
                                    committed_at=now(),
                                    history_row_sha256=object_digest(committed_row))
                existing_finalizations = session.get("round_finalizations") or []
                if not isinstance(existing_finalizations, list):
                    raise ResearchStateError("controller finalization ledger is malformed")
                committed = list(existing_finalizations)
                previous = next((one for one in committed if isinstance(one, dict) and
                                 one.get("transaction_id") == transaction_id), None)
                if previous is not None and previous.get("history_row_sha256") != \
                        object_digest(committed_row):
                    raise ResearchStateError(
                        "candidate finalization ID already commits a different history row")
                if previous is None:
                    committed.append(finalization)
                updates["round_finalizations"] = committed
            session.update(**updates, updated_at=now())
            atomic_json(session_path, session)

        if session is not None and session.get("status") == "paused":
            session.update(status="running", action=f"round_{next_round}",
                           updated_at=now())
            atomic_json(session_path, session)
        for index in range(next_round, stop_round + 1):
            checkpoint_session(status="running", action=f"round_{index}",
                               pending_action=None)
            self._set_research_action(f"round_{index}")
            if self.budget and self.budget.remaining() <= 0:
                stopped_because = "run wall-clock budget exhausted"
                self.budget.record()
                self._set_research_action(f"round_{index}", status="budget_exhausted",
                                          details={"why": stopped_because})
                break
            # A copy, because an idea is chosen against what was known when it was chosen.
            # Handing over the live list means the record of a decision keeps changing after
            # the decision -- and whatever writes that record down writes rounds that had
            # not happened yet.
            evidence = {"baseline": current.get("metric_value", current.get("success_rate")),
                        "primary_metric": self.metric_spec.as_dict(),
                        "round_history": [dict(row) for row in history],
                        "stage_availability": {s: row["available"] for s, row in
                                               self.describe().items()}}
            # A parameter cannot create an absent train/evaluate command. It *can* rescue an
            # existing command that fails before producing a number (OOM, bad batch size,
            # wrong task setting). Narrow actions only for a genuinely missing command.
            #
            # Read from the rounds so far and not from the baseline, because the baseline's
            # answer stops being the run's state the moment a round produces a number. Taken
            # once it said "no baseline could be measured" for a round that was choosing
            # against a measurement made two rounds earlier.
            measuring = any(row.get("metric_value", row.get("success_rate")) is not None
                            for row in history)
            # `algo` is wanted here and not only `code`, because an `algo` idea names what the
            # run does -- which stages, on what data -- and that is as able to make a stage run
            # as a change to a file. What an `algo` idea is *not* is a file patch, and the
            # first version of this sent both kinds to `code_change`, where four of seven
            # rounds died asking an algo idea for a `find` it never had.
            if self.execution_graph:
                graph = ExecutionGraph(self.execution_graph)
                target = str(self.execution_graph["score_target"])
                commands_exist = all(self.available(name) for name in graph.order_for(target))
            else:
                commands_exist = self.available("evaluate") and (
                    self.available("train") or bool(self.checkpoint) or self.source_policy)
            wanted = None if measuring or commands_exist else ("code", "algo")
            if main_controller_owns_selection:
                if index != next_round:
                    raise ResearchStateError(
                        "main-controlled research only accepts one selected idea per action")
                idea = self.library.get(selected_idea_label)
                if idea is None or idea.status != redline_rules.CLEARED or (
                        wanted is not None and idea.granularity not in wanted):
                    raise ResearchStateError(
                        "selected idea is no longer an eligible audited option; refusing to "
                        "change the persisted research decision")
                self._note("main_controller_selected_idea", [{
                    "label": idea.label, "granularity": idea.granularity,
                    "controller_decision_id": self.controller_decision_id,
                    "evidence_sha256": object_digest(evidence),
                    "eligible_options_sha256": object_digest(
                        [one.as_dict() for one in self.library.usable()
                         if wanted is None or one.granularity in wanted]),
                }])
            else:
                idea = self.choose_idea(history=kinds, evidence=evidence,
                                        round_index=index, kinds_wanted=wanted)
            if idea is None:
                stopped_because = (
                    "no baseline could be measured and no round since has produced a number, "
                    "so no admissible idea or declared-space proposal was available"
                    if not measuring else
                    "every idea in the library had been tried or refused, and the declared "
                    "space offered nothing that fits")
                history.append({"round": index,
                                "status": "no valid proposal" if not self.library.ideas
                                          else "no idea to try",
                                "why_not": stopped_because})
                checkpoint_session(history=history, kinds=kinds, next_round=index + 1,
                                   pending_action=None)
                self._set_research_action(f"round_{index}", status="completed",
                                          details={"outcome": "no_admissible_idea",
                                                   "why": stopped_because})
                break
            if idea.label == self.CONTROLLER_STOPPED:
                stopped_because = (f"research controller elected to stop: "
                                   f"{idea.outcome or idea.mechanism}")
                history.append({"round": index, "status": "controller stopped",
                                "idea": idea.mechanism, "kind": "param",
                                "verdict": idea.outcome})
                checkpoint_session(history=history, kinds=kinds, next_round=index + 1,
                                   pending_action=None)
                self._set_research_action(f"round_{index}", status="completed",
                                          details={"outcome": "controller_stopped",
                                                   "why": stopped_because})
                break
            checkpoint_session(pending_action={
                "round": index, "phase": "idea_selected", "idea": idea.as_dict(),
                "evidence_sha256": object_digest(evidence), "updated_at": now()})
            kinds.append(idea.granularity)
            before_measured = self._was_measured(current)
            checkpoint_session(pending_action={
                "round": index, "phase": "preparing_idea", "idea": idea.as_dict(),
                "evidence_sha256": object_digest(evidence),
                "source_transaction_protocol": 1, "updated_at": now()})
            prepared = self._prepare_idea(idea, evidence=evidence, round_index=index)
            if prepared["refused"]:
                # The round is spent and the loop goes on. Ending the run here would make one
                # idea the end of the research. The repair's own answer travels with it,
                # because "the file does not contain what the idea assumed" is a finding about
                # the idea and this is where a reader looks for it.
                history.append({"round": index, "status": prepared["status"],
                                "idea": idea.label, "kind": idea.granularity,
                                "why_not": prepared["why"], "repair": prepared["repair"],
                                # The change as it stood when it was refused, after any repair.
                                # Without it the record says a change did not fit and not one
                                # word about what it was.
                                "change": dict(idea.change or {})})
                checkpoint_session(history=history, kinds=kinds, next_round=index + 1,
                                   pending_action=None)
                self.library.note(idea.label, worked=False,
                                  outcome=redact(str(prepared["why"]))[:300])
                self._summarize()
                self._set_research_action(f"round_{index}", status="completed",
                                          details={"outcome": "idea_refused",
                                                   "idea": idea.label,
                                                   "why": redact(str(prepared["why"]))[:300]})
                continue
            changed, proposal = prepared["changed"], prepared["proposal"]
            # Captured now, before the stage runs: the state about to be measured is the one
            # worth being able to return to, and after the round the state may already have
            # been taken back.
            self.snapshots.capture(sorted(self.touched), repo=self.repo,
                                   name=f"round-{index}",
                                   why=f"before measuring {idea.label} ({idea.granularity})")
            varied = {**base_settings}
            for key, value in (proposal.get("training") or {}).items():
                if isinstance(value, dict):
                    varied.update({f"{key}.{k}": v for k, v in value.items()})
                else:
                    varied[key] = value
            checkpoint_session(pending_action={
                "round": index, "phase": "prepared", "idea": idea.as_dict(),
                "evidence_sha256": object_digest(evidence),
                "changed": dict(changed), "proposal": dict(proposal),
                "settings": dict(varied), "updated_at": now()})
            # What this round asked for and will not get. A declared-space proposal carries a
            # `collection` half -- the validator checks its `enabled` flag, the space declares
            # collection axes, and the protocol's own notes say that half decides what data
            # exists -- and this loop never read it. The controller was handed a handle that
            # does not exist, and nothing said so: the request was validated, accepted, and
            # then dropped in silence. Whether the system can act on it is a separate question
            # from whether it may pretend it did.
            #
            # An idea out of the library carries no such half: a `code` idea changes a file and
            # an `algo` idea names stages, and neither is a request for data. The half is read
            # wherever it exists rather than assumed to exist.
            # What the *author* asked for, not what the envelope was filled with. A library
            # idea's envelope carries every required collection axis at its declared default,
            # because the space check needs them -- so reading the request off the validated
            # proposal made every idea a request for data, and on RoboTwin four rounds of
            # eight died asking a benchmark with no collector to collect. The defaults say
            # what this benchmark does when nobody says otherwise; they are not a request.
            held_proposal = self._proposals_by_idea.get(idea.label) or {}
            own = dict(idea.change or {})
            requested = (dict(held_proposal.get("collection") or {})
                         if idea.label in self._from_declared_space
                         else dict(own.get("collection") or {}))
            asked_for = {name: value for name, value in requested.items()
                         if value not in (None, "", False, {}, [])}
            collected = None
            if asked_for:
                collected = self.collect({**base_settings, **asked_for})
                self._append_local_event({"at": now(), "event": "collection", "round": index,
                                          "asked_for": sorted(asked_for),
                                          "ran": collected.get("ran"),
                                          "why": collected.get("why")})
                if not collected.get("ran"):
                    # A round, not the end of the run -- this is the twin of the stop that was
                    # removed from the no-baseline case and left standing here. The finding is
                    # real and specific: this benchmark cannot produce new trajectories, so an
                    # idea that needs data it does not have cannot be run. Ending the loop for
                    # it makes one such idea the end of the research, when what it is is one
                    # idea the benchmark cannot accommodate -- and the library holds others.
                    history.append({"round": index, "status": "nothing was measured",
                                    "idea": idea.label, "kind": idea.granularity,
                                    "why_not": collected.get("why"),
                                    "asked_for_and_did_not_get": sorted(asked_for)})
                    checkpoint_session(history=history, kinds=kinds, next_round=index + 1,
                                       pending_action=None)
                    self.library.note(idea.label, worked=False,
                                      outcome=f"the benchmark cannot supply the data this "
                                              f"asked for: {collected.get('why')}")
                    if idea.label in self._from_declared_space:
                        # Except when the idea came from the declared space, because then the
                        # next round asks the same controller the same question and gets the
                        # same answer -- a loop, not a search. That was the whole content of
                        # the stop this replaces, and it is right about the space and wrong
                        # about the library.
                        stopped_because = (
                            f"the declared space asks for data this benchmark cannot supply: "
                            f"{collected.get('why')}")
                        self._set_research_action(f"round_{index}", status="completed",
                                                  details={"outcome": "collection_unavailable",
                                                           "why": collected.get("why")})
                        break
                    self._summarize()
                    self._set_research_action(f"round_{index}", status="completed",
                                              details={"outcome": "collection_unavailable",
                                                       "why": collected.get("why")})
                    continue
            checkpoint_session(pending_action={
                "round": index, "phase": "measurement_starting", "idea": idea.as_dict(),
                "evidence_sha256": object_digest(evidence),
                "changed": dict(changed), "proposal": dict(proposal),
                "settings": dict(varied), "asked_for": sorted(asked_for),
                "collected": (dict(collected) if collected else None),
                "updated_at": now()})
            result = self.measure(settings=varied, label=f"round_{index}",
                                  dataset=collected.get("data") if collected else None)
            round_label = f"round_{index}"
            finalization_id = self._candidate_finalization_id(label=round_label)
            if self.last_proposal_decision:
                proposal_decision_id = self.last_proposal_decision
                self._run_candidate_finalization_step(
                    label=round_label, step="proposal_resolution",
                    action=lambda: (
                        self.decisions.resolve(
                            proposal_decision_id,
                            {"ran": result.get("ok"),
                             "success_rate": result.get("success_rate"),
                             "metric_value": result.get("metric_value"),
                             "varied": {k: varied[k] for k in sorted(varied)
                                        if base_settings.get(k) != varied[k]}}),
                        {"decision_id": proposal_decision_id,
                         "outcome_state": "known"})[1])
                self.last_proposal_decision = ""
            else:
                self._skip_candidate_finalization_step(
                    label=round_label, step="proposal_resolution",
                    because="this candidate has no outstanding proposal decision")
            measured = self._was_measured(result)
            helped, verdict = self._did_it_help(idea=idea, before_measured=before_measured,
                                                measured=measured, result=result,
                                                baseline=baseline_rate)
            # The rubric first, because the state that was measured is the one worth scoring
            # and the state that was measured is the one about to be undone.
            def update_rubric_before_best() -> dict[str, Any]:
                self.rubric = objective.evaluate(rubric, self.rubric_facts(
                    rounds=history + [{
                        "status": "measured" if measured else "nothing was measured",
                        "success_rate": result.get("success_rate"),
                        "metric_utility": self._metric_utility(result)}],
                    baseline=baseline_rate, current=result))
                self._write_rubric()
                path = self.run_root / "rubric.json"
                return {"sha256": digest(path), "score": round(self.rubric.score(), 6)}

            self._run_candidate_finalization_step(
                label=round_label, step="rubric_pre_update",
                action=update_rubric_before_best)
            progress = self.rubric.score()

            def record_round_event() -> dict[str, Any]:
                event = self._note(
                    "round", [{"round": index, "idea": idea.label,
                               "kind": idea.granularity, "verdict": verdict,
                               "progress": round(progress, 4),
                               "frontier": (self.rubric.frontier().question
                                            if self.rubric.frontier() is not None else "")}])
                return {"event_id": event.get("event_id"),
                        "event_sha256": event.get("event_sha256"),
                        "sequence": event.get("sequence")}

            self._run_candidate_finalization_step(
                label=round_label, step="round_event", action=record_round_event)
            # What was measured, recorded before anything is taken back -- and recorded on one
            # of two scales, never a mixture. The success rate is the answer; the completion is
            # how far a run with no answer has got. `advance` refuses to compare them.
            score, scale = ((self._metric_utility(result), metric_scale) if measured
                            else (progress, "progress"))
            def update_best_state() -> dict[str, Any]:
                advanced = self.snapshots.advance(
                    f"round-{index}", score=score, scale=scale,
                    why=f"{idea.label}: {verdict}")
                path = self.run_root / "snapshots" / "snapshots.json"
                return {"advanced": advanced,
                        "sha256": digest(path) if path.is_file() else ""}

            best_evidence = self._run_candidate_finalization_step(
                label=round_label, step="best_state", action=update_best_state)
            became_best = bool(best_evidence.get("advanced"))
            if measured and became_best:
                self._run_candidate_finalization_step(
                    label=round_label, step="demo_capture",
                    action=lambda: self.capture_demo(
                        trigger="new_best", measurement=result,
                        settings=self._frozen_search_settings() or varied))
            elif not measured and not failure_demo_attempted:
                failure_demo_attempted = True
                self._run_candidate_finalization_step(
                    label=round_label, step="demo_capture",
                    action=lambda: self.capture_demo(
                        trigger="representative_failure", measurement=result,
                        settings=self._frozen_search_settings() or varied))
            else:
                self._skip_candidate_finalization_step(
                    label=round_label, step="demo_capture",
                    because="no new-best or representative-failure demo was due")
            undone = False
            if not helped and idea.granularity in ("code", "algo") and changed.get("file"):
                # A change to the benchmark's source that did not work is taken back, and taken
                # back here rather than at the end of the run. Left in place it is a checkout
                # the next round runs on top of and the round after that on top of both -- so
                # the run's later failures get attributed to the wrong thing, and any number it
                # finally reports came from a benchmark nobody chose.
                def rollback_source() -> dict[str, Any]:
                    paths = list(reversed(changed.get("files") or [
                        str(changed.get("file"))]))
                    restored = [self.undo_code_change(file) for file in paths]
                    return {"undone": all(restored),
                            "paths": paths,
                            "remaining_hashes": {
                                file: digest(self.repo / file)
                                for file in paths if (self.repo / file).is_file()}}

                rollback_evidence = self._run_candidate_finalization_step(
                    label=round_label, step="source_rollback", action=rollback_source)
                undone = bool(rollback_evidence.get("undone"))
            else:
                self._skip_candidate_finalization_step(
                    label=round_label, step="source_rollback",
                    because="the idea did not require a source rollback")

            idea_event_id = object_digest({"run_id": self.run_id, "round": index,
                                           "idea_label": idea.label,
                                           "finalization_id": finalization_id,
                                           "step": "idea_outcome"})[:32]

            def record_idea_outcome() -> dict[str, Any]:
                self.library.note(idea.label, worked=helped, outcome=verdict,
                                  event_id=idea_event_id)
                stored = self.library.get(idea.label)
                return {"event_id": idea_event_id,
                        "times_tried": stored.times_tried if stored else None,
                        "status": stored.status if stored else "missing"}

            self._run_candidate_finalization_step(
                label=round_label, step="idea_outcome", action=record_idea_outcome)
            # And the questions only this benchmark's reader can answer, from what the run has
            # recorded -- after the undo, so the record they are answered from is the state
            # that was actually kept rather than the one that was measured and taken back.
            def answer_benchmark_questions() -> dict[str, Any]:
                answered = self._answer_the_run_s_own_questions(
                    rounds=history + [{"round": index,
                                       "status": ("measured" if measured else
                                                  "nothing was measured"),
                                       "idea": idea.label, "kind": idea.granularity,
                                       "verdict": verdict, "undone": undone,
                                       "success_rate": result.get("success_rate"),
                                       "metric_value": result.get("metric_value"),
                                       "metric_utility": self._metric_utility(result),
                                       "varied": {k: varied[k] for k in sorted(varied)
                                                  if base_settings.get(k) != varied[k]}}],
                    current=result if not undone else current)
                return {"answered_questions": answered}

            self._run_candidate_finalization_step(
                label=round_label, step="benchmark_questions",
                action=answer_benchmark_questions)
            self._run_candidate_finalization_step(
                label=round_label, step="rubric_final",
                action=lambda: (self._write_rubric(),
                                {"sha256": digest(self.run_root / "rubric.json")})[1])
            history.append({"round": index,
                            "status": "measured" if measured else "nothing was measured",
                            "label": f"round_{index}",
                            "success_rate": result.get("success_rate"),
                            "metric_value": result.get("metric_value"),
                            "metric_utility": self._metric_utility(result),
                            "metric_name": self.metric_spec.name,
                            "idea": idea.label,
                            "kind": idea.granularity,
                            "risk": idea.risk,
                            "mechanism": idea.mechanism,
                            "verdict": verdict,
                            "undone": undone,
                            "progress": round(progress, 4),
                            **({} if measured else
                               self._failure_context(result)),
                            "varied": {k: varied[k] for k in sorted(varied)
                                       if base_settings.get(k) != varied[k]},
                            "asked_for_and_did_not_get": sorted(asked_for) if (
                                asked_for and not collected) else [],
                            **({"collected": collected.get("data")} if collected else {}),
                            **({"finalization_id": finalization_id}
                               if finalization_id else {}),
                            "hypothesis": proposal.get("hypothesis")})
            if finalization_id:
                self._update_candidate_finalization(
                    label=round_label, step="history_session_commit", status="started")
            checkpoint_session(history=history, kinds=kinds,
                               current_measurement_label=f"round_{index}",
                               next_round=index + 1, pending_action=None)
            current = result
            # After the round, so the document a person reads mid-run describes the run as far
            # as it has got rather than as far as it got last time.
            self._summarize()
            self._set_research_action(f"round_{index}", status="completed",
                                      details={"outcome": "measured" if measured else
                                               "measurement_failed",
                                               "idea": idea.label,
                                               "measurement_ref":
                                                   f"research/{self.run_id}/measurements/"
                                                   f"round_{index}.json"})

        # Yield the controller after one bounded research action.  A refused idea or an
        # unavailable collection can still be a useful observation for the outer controller;
        # only an explicit terminal decision, exhausted rounds, or exhausted budget finalizes
        # the research session.
        completed_rounds = min(
            rounds, max((int(row.get("round", 0)) for row in history
                         if isinstance(row, dict) and
                         isinstance(row.get("round", 0), int)), default=0))
        should_pause = (yield_after_action and not stopped_because and
                        next_round <= rounds and completed_rounds < rounds)
        if session is not None and should_pause:
            current_label = str(current.get("label") or
                                session.get("current_measurement_label") or "")
            if not re.fullmatch(r"(?:baseline|round_[1-9][0-9]*)", current_label):
                raise ResearchStateError("cannot checkpoint research without a current measurement")
            next_round = completed_rounds + 1
            session.update(status="paused", action="", history=history, kinds=kinds,
                           baseline_rate=baseline_rate,
                           current_measurement_label=current_label,
                           next_round=next_round,
                           failure_demo_attempted=failure_demo_attempted,
                           stopped_because="", updated_at=now())
            report = self._controller_report(
                history=history, stopped_because="", run_status="paused",
                next_round=next_round, rounds=rounds)
            atomic_json(session_path, session)
            atomic_json(report_path, report)
            self._summarize()
            self._set_research_action(
                "research_paused", status="paused",
                details={"after": f"round_{completed_rounds}",
                         "next_round": next_round,
                         "measurement_ref":
                             f"research/{self.run_id}/measurements/{current_label}.json"})
            run_record.generate(self.run_root,
                                title=f"{self.benchmark or self.run_id} / derived")
            return report

        if session is not None:
            # Finalization is restartable and never repeats a benchmark action. Keeping a
            # distinct state lets recovery finish export/report projection after interruption.
            session.update(status="finalizing", action="select_best_confirm_and_export",
                           history=history, kinds=kinds, baseline_rate=baseline_rate,
                           current_measurement_label=str(current.get("label") or
                                                         session.get("current_measurement_label") or
                                                         "baseline"),
                           next_round=max(rounds + 1, next_round),
                           failure_demo_attempted=failure_demo_attempted,
                           stopped_because=stopped_because, updated_at=now())
            atomic_json(session_path, session)
        self._set_research_action("select_best_confirm_and_export")
        best = self.snapshots.best()
        # The held-out measurement, when the run was authorized to spend it and declared a set
        # to spend it on. Not automatic: it costs what a full evaluation costs, and it can only
        # be taken once, so it is a decision rather than a final step. `confirm()` refuses for
        # every reason it might not be possible and the refusal is recorded like any other
        # finding -- a run that declares a split and never uses it has left its one checked
        # number on the table, and the report says so.
        # Through `confirmation_state()` either way, so the report carries one shape. When the
        # measurement was taken the state is read back from the file `confirm()` wrote, which
        # is also what a later reader of the run directory would find.
        if confirm:
            confirmed = self.confirm()
            if confirmed.get("ok"):
                self.capture_demo(trigger="final_confirmation", measurement=confirmed,
                                  settings=confirmed.get("settings") or {})
        confirmation = self.confirmation_state()
        try:
            exported = export_best(self.run_root, best.as_dict() if best is not None else None)
        except (OSError, ValueError, RuntimeError) as exc:
            exported = {"status": "export_failed", "why": f"{type(exc).__name__}: {exc}"}
        report = {"schema_version": 1, "created_at": now(), "repo": str(self.repo),
                  "run_status": "completed",
                  "run_id": self.run_id, "rounds": history,
                  **({"stopped_because": stopped_because} if stopped_because else {}),
                  "objective": self.rubric.as_dict(),
                  "red_lines": self.red_lines().as_dict(),
                  "ideas": self.library.summary(),
                  "best": best.as_dict() if best is not None else None,
                  "best_export": exported,
                  "confirmation": confirmation,
                  "available_stages": sorted(s for s in self.sources if self.available(s))}
        atomic_json(report_path, report)
        if session is not None:
            session.update(status="completed", action="", updated_at=now(),
                           stopped_because=stopped_because)
            atomic_json(session_path, session)
        # Once more, because a round that ended the loop by breaking out never reached the
        # line above -- and a loop that stopped early is the case most worth summarising.
        self._summarize()
        phase_status = ("budget_exhausted" if "budget" in stopped_because.lower()
                        else "completed")
        self._set_research_action(
            "research_complete", status=phase_status,
            details={"report_ref": f"research/{self.run_id}/research_report.json",
                     "report_sha256": object_digest(report),
                     "round_count": len(history),
                     "measurement_count": sum(
                         1 for row in history if row.get("measured") or
                         row.get("status") == "measured"),
                     "best_label": best.name if best is not None else None})
        # The summary above was built before this terminal event existed. Re-project the
        # deterministic state once more so a finished run cannot leave RUN.md showing an
        # active finalization action.
        run_record.generate(self.run_root, title=f"{self.benchmark or self.run_id} / derived")
        return report

    @staticmethod
    def _did_it_help(*, idea: Any, before_measured: bool, measured: bool,
                     result: dict[str, Any], baseline: float | None) -> tuple[bool, str]:
        """Whether the round moved the thing the run is trying to move, and why.

        Three states and they are not the same. A round that made the evaluation produce a
        number where none came out has done the most a round can do here, whatever the number
        is. A round that produced a number and beat the baseline is an improvement. A round
        that produced a number no better than the baseline's has not failed -- it has measured
        nothing new -- but it has not earned the right to keep its change either.
        """
        rate = DerivedResearch._metric_utility(result)
        if measured and (result.get("guardrails") or {}).get("status") in {"unknown", "violated"}:
            return False, "primary metric is measured, but secondary objectives are missing or violated"
        if measured and not before_measured:
            return True, (f"the evaluation produced a number where none came out before "
                          f"({rate}), which is the rung the run was on")
        if measured and isinstance(rate, (int, float)):
            if isinstance(baseline, (int, float)):
                if rate > baseline:
                    return True, f"{rate} against a baseline of {baseline}"
                return False, (f"{rate} against a baseline of {baseline} -- measured, and no "
                               f"better than what was already there")
            return True, f"{rate}, with no baseline to compare against"
        failure = DerivedResearch._failure_context(result)
        return False, f"no valid measurement: {failure['why_not']}"
