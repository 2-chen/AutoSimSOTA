"""RoboSyn adapter; one device per subprocess, artifacts belong to an explicit experiment."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .common import (assert_frozen, atomic_json, digest, find_benchmark, immutable_json,
                     object_digest, read_json, run_command)
from .devices import DEFAULT_DEVICE_ENV, SimDeviceSelection, select, shard_count
from .registry import TaskSpec
from .native_lifecycle import retryable_startup, startup_attempt_limit


def _this_package() -> Path:
    """The installed package, not the runtime's workspace.

    Worth knowing before relying on this digest: it does not vary with the run's
    workspace, repo or output, so two runtimes in different directories share one identity.
    """
    return Path(__file__).resolve().parents[1]


def _activated_component_identity() -> str | None:
    from .execution_activation import active_execution_identity
    activated = active_execution_identity()
    return activated.get("component_sha256") if activated else None


def execution_code_digest(package: Path) -> str:
    """Content-addressed identity of a package tree, keyed by relative path.

    Extracted from the method so the property that matters can be tested directly: adding,
    removing or *renaming* a module changes the digest even when the code is byte-identical,
    because the paths are part of the key. That is the reason generalizing the execution
    layer invalidates resume for runs started before it, and it is not something to
    discover by watching a resume fail.
    """
    files = sorted(Path(package).rglob("*.py"))
    identity = object_digest({str(p.relative_to(package)): digest(p) for p in files})
    component_identity = _activated_component_identity()
    if component_identity:
        identity = object_digest({"package": identity, "lifecycle_component": component_identity})
    return identity


def retryable_evaluation_startup(directory: Path) -> bool:
    """Only native failures positively recorded before the first evaluation reset."""
    return retryable_startup(directory)


def retryable_collection_startup(directory: Path) -> bool:
    """A collection process that died before its first reset left no collection state.

    The historical failure is DexSim aborting (rc -6/-11) while constructing the environment.
    Nothing was sampled and no attempt seed was consumed, so restarting the same block in a
    fresh directory cannot change the seed stream.  Any collection that got past its first
    reset is score-bearing evidence and is never restarted.
    """
    process = directory / "process/process.json"
    if not process.is_file():
        return False
    record = read_json(process)
    return (record.get("status") == "failed" and record.get("returncode") in {-11, -6}
            and not (directory / "collection.json").exists()
            and not (directory / "scene_resets.jsonl").exists())


def startup_receipt(directory: Path) -> dict:
    """What a failed probe's own records say, quoted rather than interpreted.

    A probe that was killed while the engine was still building the environment
    and a probe whose engine aborted are different findings, and only the receipt
    tells them apart: ``process.json`` carries the termination (``TimeoutExpired``
    versus a native signal) and ``startup.json`` the phase. Reports that assert a
    renderer fault without quoting this have invented their cause.
    """
    receipt: dict = {}
    process = directory / "process/process.json"
    if process.is_file():
        record = read_json(process)
        receipt.update({key: record.get(key) for key in
                        ("status", "returncode", "error", "timeout", "elapsed_seconds", "pid")})
        receipt["pid"] = record.get("worker_pid", receipt.get("pid"))
    startup = directory / "startup.json"
    if startup.is_file():
        receipt["startup_phase"] = read_json(startup).get("phase")
    receipt["initializations_recorded"] = (directory / "initializations.jsonl").is_file()
    return receipt


def startup_census(directory: Path) -> list:
    """Every phase receipt an evaluation wrote, in order, with its own-process census.

    ``startup.json`` keeps only the latest phase, and the latest phase is never the answer
    to "when did card 0 gain its 525 MiB": memory already held before the engine is
    constructed belongs to a library import, the same memory appearing while it is
    constructed belongs to the engine.  The two findings have opposite fixes, so the ladder
    carries the whole timeline rather than one number.  Unreadable or malformed lines are
    skipped -- a diagnostic that raises would turn a measurement into a crash.
    """
    rows: list = []
    try:
        text = (directory / "startup_census.jsonl").read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


@dataclass
class Runtime:
    workspace: Path
    output: Path
    gpu: str = "0"
    deadline: float | None = None
    repo_path: Path | None = None
    eval_repo_path: Path | None = None
    python_path: Path | None = None
    selection: SimDeviceSelection | None = None
    plan: dict | None = None
    job: str | None = None
    shard_min_episodes: int = 8
    cost_model: dict = field(default_factory=dict)
    execution_resources: dict = field(default_factory=dict)
    assigned_devices: tuple[dict, ...] = ()
    compute_ledger: object | None = None
    native_barrier: dict | None = None
    harness_config: dict = field(default_factory=dict)
    lease_journal: str | None = None
    audit_kernel_registry: Path | None = None
    audit_optimizer: Any = None

    def _unbound_compute(self) -> bool:
        return bool(self.plan and self.plan.get("unified_scheduler") and self.job is None)

    def _execution_code_digest(self) -> str:
        """Identity of the code that produced a run's artifacts.

        Keyed on ``{relative path: sha256}`` over every module in the package, so it is a
        digest of *where* code lives as much as of what it says: moving a method from one
        module to another changes the keys and therefore the digest, even though the
        behaviour is identical.

        Three consumers make that expensive. It is `HarnessArtifacts.revision`, compared on
        resume, so a run started before such a move refuses to continue
        (``ArtifactConflict: incomplete attempt belongs to another executor revision``). It
        is frozen into the device probe's `probe_contract.json`. And it keys the profile
        store behind `PhaseJob.code_digest`, whose measured percentiles decide which jobs
        run together.

        Consequence for anyone refactoring across module boundaries: a fresh run reproduces
        the same artifact tree, and *resuming a run started beforehand does not work*. That
        is a property of this function, not a defect to work around; `execution_activation`
        exists to pin a run's identity when one must be continued across such a change.
        """
        # A Runtime leaf can call adapters and data validators in other modules.
        # Hash the package once per source version, so changing one of those modules
        # cannot silently reuse a result merely because runtime.py stayed unchanged.
        package = _this_package()
        files = sorted(package.rglob("*.py"))
        signature = (tuple((str(p.relative_to(package)), p.stat().st_size, p.stat().st_mtime_ns)
                          for p in files), _activated_component_identity())
        cached = getattr(self, "_execution_code_cache", None)
        if cached and cached[0] == signature:
            return cached[1]
        identity = execution_code_digest(package)
        self._execution_code_cache = (signature, identity)
        return identity

    @staticmethod
    def _profile_digest(category: str, request: dict, *, block_size: int | None = None) -> str:
        if category not in {"training", "collection", "evaluation"}:
            return object_digest(request)
        # Input identity fences replay; timing samples pool comparable workload shapes.
        # Seeds and trained weight values do not create a new tensor/work-count shape.
        shape = {k:request.get(k) for k in ("spec","steps","episodes","attempts","target",
            "profile","policy","config","collection_mode","mode","prefix_min","prefix_max","replan","return")}
        shape.update(category=category, block_size=block_size)
        if category == "training":
            params = request.get("params") or {}
            defaults = {"batch_size":32,"chunk_size":50,"n_action_steps":50,
                        "image_augmentation_profile":"none","num_workers":None}
            shape["training_shape"] = {k:params.get(k,v) for k,v in defaults.items()}
        return object_digest(shape)

    def _scheduled_call(self, category: str, output: Path, callback, *, request: dict,
                        estimate_seconds: float = 60, gpu_count: int = 1):
        from .accounting import BudgetLedger
        from .job_runner import PhaseJob, run_phase
        from .resource_contracts import ResourceRequest
        if self.compute_ledger is None:
            self.compute_ledger = BudgetLedger(self.output / "compute/budget.json",
                wall_limit_seconds=max(1, (self.deadline - time.monotonic()) if self.deadline else 86400),
                gpu_hours_limit=self.plan.get("gpu_hours_limit"), devices=self.plan["usable"])
        count = max(1, self.plan.get("max_parallel_jobs", 1))
        params = self.plan.get("execution_parameters", {})
        desired = max(params.get("omp_threads", 4), params.get("loader_workers", 0)+1 if category == "training" else 1)
        cores = max(1, min(desired, int(self.plan["host_capacity"]["cpu_cores"] // count)))
        resource = ResourceRequest(cpu_cores=cores, ram_mib=4096,
                                   per_gpu_memory_mib=(12000 if category == "training" else 2048) if gpu_count else 0,
                                   io_slots=1 if gpu_count == 0 else 0,
                                   capabilities=(("train",) if category == "training" else ("physics","render")) if gpu_count else ())
        phase = f"{category}_{object_digest(str(output.absolute()))[:16]}"
        if category == "training":
            phase += f"_{request.get('steps')}_{'resume' if request.get('resume') else 'initial'}"
        job = PhaseJob("work", category, callback, device_count=gpu_count, resources=resource,
                       estimate_seconds=estimate_seconds, reserve_seconds=0,
                       input_digest=object_digest(request), code_digest=self._execution_code_digest(),
                       profile_digest=self._profile_digest(category,request))
        return run_phase(phase=phase, jobs=[job], runtime=self, plan=self.plan,
                         output=self.output / "compute/schedule", ledger=self.compute_ledger,
                         run_id=object_digest(str(self.output))[:16])["work"]

    @property
    def platform_root(self) -> Path:
        """Where this installation's own assets live: ``.venv``, ``EmbodiChain``, ``embodichain_data``.

        This is a property of the AutoSimSOTA installation, not of where the benchmark
        repository happens to sit.  Keying off ``workspace/RoboSynChallenge`` made the
        root depend on an unrelated checkout: moving a benchmark outside the project
        directory silently redirected the venv, the simulator package, the asset root
        and the library path at once.  Set ``AUTOSIM_PLATFORM_ROOT`` to override.
        """
        override = os.environ.get("AUTOSIM_PLATFORM_ROOT")
        if override:
            return Path(override).expanduser().absolute()
        for candidate in (self.workspace, self.workspace / "AutoSimSOTA"):
            if (candidate / ".venv").is_dir() or (candidate / "EmbodiChain").is_dir():
                return candidate
        # No installation assets found; preserve the historical standalone answer.
        return self.workspace / "AutoSimSOTA" if (self.workspace / "AutoSimSOTA").is_dir() else self.workspace

    @property
    def repo(self) -> Path:
        if self.repo_path is not None:
            return self.repo_path.absolute()
        # The benchmark is a separate checkout, and where it lives is the operator's
        # choice: beside the project, grouped with others in a shared directory, or named
        # by AUTOSIM_BENCHMARK_ROOT. Found rather than derived, so reorganising the
        # checkouts is not a code change.
        found = find_benchmark("RoboSynChallenge", self.workspace,
                               marker="scripts/eval_policy.py")
        if found is not None:
            return found
        # Nothing on disk: answer with the historical sibling path so the error a caller
        # eventually raises names a plausible location instead of an empty one.
        return (self.workspace / "RoboSynChallenge").absolute()

    @property
    def eval_repo(self) -> Path:
        if self.eval_repo_path is not None:
            return self.eval_repo_path.absolute()
        candidate = self.repo.with_name(self.repo.name + "_eval_clean")
        return candidate if candidate.is_dir() else self.repo

    @property
    def python(self) -> Path:
        # Do not resolve the venv executable symlink.
        return self.python_path or self.platform_root / ".venv/bin/python"

    @property
    def train_python(self) -> Path:
        """The ACT trainer's own interpreter, not the platform venv.

        RoboSynChallenge ships ``policy/act`` as a uv project with its own venv, pinned
        to LeRobot 0.3.3; the patched entry point states the contract in its docstring:
        "Train LeRobot v0.3.3 ACTPolicy on a canonical v2.1 dataset". The platform venv
        cannot serve this stage. It carries embodichain, which requires lerobot>=0.4.4
        whose reader rejects every v2.1 dataset -- including the benchmark's own official
        dataset -- with BackwardCompatibilityError ("The dataset you requested ... is in
        2.1 format. We introduced a new format since v3.0"). Simulation, collection and
        evaluation stay in the platform venv; only policy training moves.
        """
        trainer = self.repo / "policy/act/.venv/bin/python"
        return trainer if trainer.is_file() else self.python

    def for_job(self, selection: SimDeviceSelection | None, *, job: str,
                output: Path | None = None) -> "Runtime":
        """A per-job view bound to one device; the shared object is never mutated.

        Every heavy job gets its own writable caches on *local* disk, so two concurrent
        jobs cannot collide over the fixed ``/tmp`` paths this adapter used to hardcode.
        Native engine caches are partitioned by GPU and workspace, then reused under
        an exclusive device lease. Immutable model weights and extracted assets stay
        shared; generated URDFs and process temporary files remain private.
        """
        bound = replace(self, selection=selection, job=job, output=Path(output) if output else self.output)
        if selection is not None:
            root = Path("/tmp/autosim_robosyn_jobs") / f"{job}-{object_digest(str(bound.output))[:12]}"
            # Model weights remain in the pre-existing shared cache; isolating XDG
            # must not force every worker to download the same immutable backbone.
            caches = {"TORCH_HOME":os.environ.get("TORCH_HOME", str(
                Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "torch"))}
            for name, leaf in (("XDG_CACHE_HOME", "xdg_cache"), ("MPLCONFIGDIR", "matplotlib"),
                               ("TMPDIR", "tmp")):
                (root / leaf).mkdir(parents=True, exist_ok=True)
                caches[name] = str(root / leaf)
            if self.plan and self.plan.get("unified_scheduler"):
                cache_identity = object_digest({"gpu":selection.uuid,"host":os.uname().nodename,
                                                 "workspace":str(self.platform_root),
                                                 "namespace": self.harness_config.get("_native_cache_namespace")})[:12]
                caches["AUTOSIM_NATIVE_CACHE"] = str(Path("/tmp/as_cache") / cache_identity)
                # DexSim material identifiers have a fixed native string bound. Keep the
                # runtime asset prefix short, and isolate generated URDFs between workers.
                assets = Path("/tmp/asd") / object_digest(str(bound.output))[:12]
                assets.mkdir(parents=True, exist_ok=True)
                (assets / "assembled").mkdir(exist_ok=True)
                for leaf in ("download", "extract"):
                    link = assets / leaf
                    target = (self.platform_root / "embodichain_data" / leaf).resolve()
                    if not link.exists() and not link.is_symlink():
                        link.symlink_to(target, target_is_directory=True)
                caches["EMBODICHAIN_DATA_ROOT"] = str(assets)
            bound.selection = replace(selection, extra_env=caches)
        return bound

    def environment(self, repo: Path | None = None,
                    selection: SimDeviceSelection | None = None) -> dict[str, str]:
        chosen = self.selection if selection is None else selection
        env = os.environ.copy()
        # ``cuda_visible=None`` means "the set this process already has" -- for an *identity*
        # selection that is every card, and the engine's index is a valid CUDA ordinal only
        # because nothing renumbered the devices.  Collapsing it to ``self.gpu`` ("0") would
        # renumber them behind the engine's back: measured on the 5090 pool, an identity
        # selection with one visible card aborts in ``OptixDevice.cpp`` at
        # ``cuDeviceGet(&m_cudaDevice, 3) -> CUDA_ERROR_INVALID_DEVICE`` ("Invalid device ID: 3.
        # Available devices: 0-0").  So: narrow to the leased card for pinned_index, and leave
        # the outer value (or the variable's absence) alone for identity.
        visible = self.gpu if chosen is None else chosen.cuda_visible
        env.update(PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
                   AUTOSIM_EXPERT_SETTLE_STEPS="75",
                   EMBODICHAIN_DATA_ROOT=str((self.platform_root / "embodichain_data").resolve()),
                   XDG_CACHE_HOME="/tmp/robosyn_autoresearch_cache",
                   MPLCONFIGDIR="/tmp/robosyn_autoresearch_matplotlib", OMP_NUM_THREADS="4")
        if visible is None:
            # Preserve the outer variable exactly, including an explicit empty mask.
            pass
        else:
            env["CUDA_VISIBLE_DEVICES"] = visible
        if chosen is not None:
            env.update(chosen.extra_env)
            # A device plan is what makes "the default device" ambiguous, so a plan is also
            # what turns on aligning it: every GPU child this run spawns inherits this and the
            # three entry points that build an environment (the evaluator shim, the collection
            # worker, the policy child) call ``devices.align_process_defaults`` before they
            # touch CUDA.  Without it an unqualified ``cuda`` in a library resolves to card 0
            # -- a neighbour's card under identity addressing.  Absent on the legacy path,
            # where ordinal 0 already is the leased card, which is what keeps that path
            # byte-identical.
            env[DEFAULT_DEVICE_ENV] = str(chosen.torch_index)
        if self.execution_resources:
            cores = max(1, int(self.execution_resources.get("cpu_cores") or 1))
            env.update(OMP_NUM_THREADS=str(cores), MKL_NUM_THREADS=str(cores),
                       OPENBLAS_NUM_THREADS=str(cores), NUMEXPR_NUM_THREADS=str(cores))
            if self.execution_resources.get("gpu_count") == 0:
                env["CUDA_VISIBLE_DEVICES"] = ""
        if self.native_barrier:
            if "address" not in self.native_barrier:
                raise ValueError("legacy native file barriers are no longer supported")
            env["AUTOSIM_NATIVE_COORDINATOR"] = json.dumps({**self.native_barrier,
                "worker": self.job, "uuid": chosen.uuid if chosen else None})
        else:
            env.pop("AUTOSIM_NATIVE_COORDINATOR", None)
        env.pop("AUTOSIM_NATIVE_BARRIER_DIR", None)
        env.pop("AUTOSIM_NATIVE_BARRIER_COUNT", None)
        if self.lease_journal:
            env["AUTOSIM_JOB_JOURNAL_ROOT"] = self.lease_journal
        for key in list(env):
            upper = key.upper()
            if upper.endswith(("_KEY", "_TOKEN")) or any(part in upper for part in
                    ("API_KEY", "AUTH_TOKEN", "ACCESS_TOKEN", "ACCESS_KEY", "SECRET", "PASSWORD")):
                env.pop(key)
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in [
            self.platform_root / "autosim", repo or self.repo,
            self.platform_root / "EmbodiChain"])
        libraries = [self.platform_root / ".venv/lib/python3.10/site-packages/nvidia/cudnn/lib",
                     Path("/home/wbc/miniconda3/envs/robotwin5090/lib")]
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(p) for p in libraries] +
                                                  ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        return env

    def collector_device_flags(self, selection: SimDeviceSelection | None = None) -> list[str]:
        """Device flags in ``run_env.py``'s spelling: ``--gpu_id`` and ``--renderer``.

        The engine's ``gpu_id`` is a *physical/NVML* index while everything torch-side is a
        *CUDA* index, and ``CUDA_VISIBLE_DEVICES`` is what makes the two diverge. So the
        engine index is passed as the physical one, ``device`` is left unset --
        ``sim_manager`` only overwrites ``gpu_id`` from torch when ``device`` is a
        qualified ``cuda:N``, and we must never let it.
        """
        chosen = self.selection if selection is None else selection
        if chosen is None:
            return []
        return ["--gpu_id", str(chosen.vulkan_gpu_id), "--renderer", chosen.renderer]

    def evaluation_device_flags(self, selection: SimDeviceSelection | None = None) -> list[str]:
        """The same injection in ``autosim.research.evaluation``'s spelling.

        Two spellings exist because two different CLIs are driven here, and the evaluator
        is not the collector: it takes ``--device-gpu-id`` for the engine index *and*
        ``--device-torch-index`` for the process-local one. Passing the collector's
        ``--gpu_id`` to the evaluator is an argparse error, not a wrong device, and the
        first multi-GPU probe spent an arm discovering exactly that.
        """
        chosen = self.selection if selection is None else selection
        if chosen is None:
            return []
        return ["--device-gpu-id", str(chosen.vulkan_gpu_id), "--renderer", chosen.renderer,
                "--device-torch-index", str(chosen.torch_index)]

    def device_metadata(self, selection: SimDeviceSelection | None = None) -> dict:
        """Which device a process record belongs to; empty on the legacy path."""
        chosen = self.selection if selection is None else selection
        metadata = {"job": self.job} if self.job else {}
        if chosen is not None:
            metadata.update(device_uuid=chosen.uuid, device_index=chosen.index,
                            device_mode=chosen.mode, device_torch_index=chosen.torch_index,
                            renderer=chosen.renderer)
        return metadata

    def run(self, command: list[str], output: Path, timeout: int, *, evaluation=False,
            selection: SimDeviceSelection | None = None, metadata: dict | None = None) -> dict:
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining < 1:
                raise RuntimeError("task process-time budget exhausted")
            timeout = min(timeout, max(1, int(remaining)))
        if shutil.disk_usage(self.output).free < 12 * 1024**3:
            raise RuntimeError("less than 12 GiB free: refusing further data/checkpoint writes")
        repo = self.eval_repo if evaluation else self.repo
        stamped = self.device_metadata(selection)
        stamped.update(metadata or {})
        return run_command(command, cwd=repo, env=self.environment(repo, selection), output=output,
                           timeout=timeout, metadata=stamped or None)

    def collect(self, spec: TaskSpec, destination: Path, *, episodes: int,
                master_seed: int, profile: str = "full_random", timeout=1800) -> Path:
        result = self.collect_bounded(
            spec, destination, attempt_budget=episodes * 12,
            target_episodes=episodes, master_seed=master_seed,
            profile=profile, timeout=timeout)
        if result["capability_state"] != "target_reached":
            raise RuntimeError(f"collector did not reach target: {destination / 'collection.json'}")
        if result["dataset_root"] is None:
            raise RuntimeError(f"collector returned no dataset: {destination / 'collection.json'}")
        return Path(result["dataset_root"])

    def collect_bounded(self, spec: TaskSpec, destination: Path, *,
                        attempt_budget: int, master_seed: int,
                        profile: str = "full_random",
                        target_episodes: int | None = None,
                        collection_mode: str = "expert",
                        correction_checkpoint: Path | None = None,
                        correction_prefix_min: int = 20,
                        correction_prefix_max: int = 120,
                        correction_replan_steps: int = 10,
                        correction_safe_return_steps: int = 20,
                        timeout=1800) -> dict:
        """Collect under an attempt budget and preserve legitimate partial yield.

        This is the research-facing contract.  Zero yield and partial yield are
        evidence, not process failures; callers decide whether they are useful.
        The legacy ``collect`` method remains strict for workflows that require a
        fixed number of accepted episodes.
        """
        if attempt_budget < 1:
            raise ValueError("attempt_budget must be positive")
        if target_episodes is None:
            target_episodes = attempt_budget
        if not 1 <= target_episodes <= attempt_budget:
            raise ValueError("target_episodes must be in [1, attempt_budget]")
        if collection_mode not in {"expert", "policy_correction"}:
            raise ValueError("unknown collection_mode")
        if collection_mode == "policy_correction":
            if not spec.correction_supported:
                raise ValueError(f"{spec.name} has no validated correction adapter")
            if correction_checkpoint is None:
                raise ValueError("policy_correction requires correction_checkpoint")
            if not 1 <= correction_prefix_min <= correction_prefix_max:
                raise ValueError("invalid correction prefix range")
        manifest = destination / "collection.json"
        selections = self.collection_shard_selections(attempt_budget, collection_mode=collection_mode)
        if self._unbound_compute():
            selections = selections[:target_episodes]
            if len(selections) < 2:
                selections = []
        if selections:
            return self._collect_bounded_sharded(
                spec, destination, attempt_budget=attempt_budget, master_seed=master_seed,
                profile=profile, target_episodes=target_episodes,
                collection_mode=collection_mode, timeout=timeout, selections=selections)
        if self._unbound_compute():
            request = {"spec": spec.signature, "attempts": attempt_budget, "seed": master_seed,
                       "profile": profile, "target": target_episodes, "mode": collection_mode,
                       "checkpoint": digest(correction_checkpoint / "model.safetensors") if correction_checkpoint else None,
                       "prefix_min": correction_prefix_min, "prefix_max": correction_prefix_max,
                       "replan": correction_replan_steps, "return": correction_safe_return_steps}
            return self._scheduled_call("collection", destination, lambda bound: bound.collect_bounded(
                spec, destination, attempt_budget=attempt_budget, master_seed=master_seed, profile=profile,
                target_episodes=target_episodes, collection_mode=collection_mode,
                correction_checkpoint=correction_checkpoint, correction_prefix_min=correction_prefix_min,
                correction_prefix_max=correction_prefix_max, correction_replan_steps=correction_replan_steps,
                correction_safe_return_steps=correction_safe_return_steps, timeout=timeout),
                request=request, estimate_seconds=min(timeout, 360 + attempt_budget * 40))
        module = "autosim.research.collection_worker"
        command = [str(self.python), "-m", module, "--gym_config", spec.gym_config,
                   "--action_config", spec.action_config, "--num_envs", "1", "--headless",
                   "--max_episodes", str(target_episodes), "--collection_seed", str(master_seed),
                   "--collection_profile", profile, "--collection_mode", collection_mode,
                   "--collection_manifest", str(manifest), "--dataset_save_path", str(destination / "data"),
                   "--collection_max_attempts", str(attempt_budget), "--collection_quiet"]
        if collection_mode == "policy_correction":
            command += ["--correction_checkpoint", str(correction_checkpoint),
                        "--correction_prefix_min", str(correction_prefix_min),
                        "--correction_prefix_max", str(correction_prefix_max),
                        "--correction_replan_steps", str(correction_replan_steps),
                        "--correction_safe_return_steps", str(correction_safe_return_steps)]
            if self.selection is not None:
                # The policy runs in torch space; the engine does not. Only the policy
                # index moves with CUDA_VISIBLE_DEVICES.
                command += ["--correction_device", f"cuda:{self.selection.torch_index}"]
        command += self.collector_device_flags()
        contract_path = destination / "expert_contract.json"
        if not contract_path.exists():
            atomic_json(contract_path, {"task": spec.name, "adapter": spec.expert_adapter,
                        "collection_mode": collection_mode,
                        "parent_checkpoint": (str(correction_checkpoint) if correction_checkpoint else None),
                        "parent_model_sha256": (digest(correction_checkpoint / "model.safetensors")
                                                if correction_checkpoint else None),
                        "training_only": True, "success_function_modified": False,
                        "expert_success_cadence": "one_original_judge_query_after_each_actual_action_like_policy_adapter",
                        "terminal_hold_max_steps": 75,
                        "terminal_hold": "real recorded actions until original judge success, bounded below task horizon",
                        "retrospective_capture": (destination / "process/process.json").exists(),
                        "collector_source_sha256": digest(self.repo / "scripts/run_env.py"),
                        "adapter_source_sha256": digest(Path(__file__).with_name("collection_worker.py"))})
        self.run(command, destination / "process", timeout)
        return self._finalize_collection(
            spec, destination, attempt_budget=attempt_budget, target_episodes=target_episodes,
            profile=profile, collection_mode=collection_mode)

    def _finalize_collection(self, spec: TaskSpec, directory: Path, *, attempt_budget: int,
                             target_episodes: int, profile: str,
                             collection_mode: str) -> dict:
        """Read one collection directory back and publish its bounded result.

        Shared by the serial collection and by each shard of a spread one, so both judge a
        manifest, a partially-spent attempt budget and the realized-scene evidence by the
        same rules.  A shard calls this with *its own* block size and target share, which is
        what makes its result readable on its own and mergeable afterwards.
        """
        manifest = directory / "collection.json"
        data = read_json(manifest)
        budget_exhausted = (
            data.get("status") == "failed"
            and data.get("error") in {
                f"RuntimeError: Collection exceeded {attempt_budget} expert attempts.",
                f"RuntimeError: Correction collection exceeded {attempt_budget} attempts.",
            }
            and len(data.get("attempts", [])) == attempt_budget
        )
        if data["status"] != "completed" and not budget_exhausted:
            raise RuntimeError(f"collector process did not complete: {manifest}")
        if data["task"] != spec.name or len(data.get("dataset_paths", [])) > 1:
            raise RuntimeError(f"unexpected collection contract: {manifest}")
        from .diagnostic_audit import audit_collection_scene_evidence
        scene_audit = audit_collection_scene_evidence(
            manifest, directory / "scene_resets.jsonl")
        atomic_json(directory / "scene_evidence_audit.json", scene_audit)
        if not scene_audit["passed"]:
            raise RuntimeError(
                f"collection scene evidence audit failed: "
                f"{directory / 'scene_evidence_audit.json'}")
        accepted = len(data.get("successful_episode_seeds", []))
        capability_state = (
            "target_reached" if accepted >= target_episodes
            else "partial_yield" if accepted else "no_observed_yield"
        )
        result = {
            "schema_version": 1,
            "kind": "bounded_collection_result",
            "task": spec.name,
            "profile": profile,
            "collection_mode": collection_mode,
            "attempt_budget": attempt_budget,
            "target_episodes": target_episodes,
            "attempts_consumed": len(data.get("attempts", [])),
            "accepted_episodes": accepted,
            "capability_state": capability_state,
            "termination": "attempt_budget_exhausted" if budget_exhausted else "target_reached",
            "dataset_root": (data.get("dataset_paths") or [None])[0],
            "manifest": str(manifest.resolve()),
            "scene_evidence_audit": str(
                (directory / "scene_evidence_audit.json").resolve()),
            "zero_yield_does_not_prove_incapability": accepted == 0,
        }
        atomic_json(directory / "bounded_collection_result.json", result)
        return result

    def prepare_data(self, spec: TaskSpec, root: Path, output: Path, *,
                     full_video_decode: bool = False) -> dict:
        if self._unbound_compute():
            from .data_version import fingerprint_dataset
            content = fingerprint_dataset(root, self.output / "data_versions")
            return self._scheduled_call("audit", output, lambda bound: bound.prepare_data(
                spec, root, output, full_video_decode=full_video_decode), gpu_count=0,
                request={"spec": spec.signature, "root": str(root),
                         "content": content["content_id"], "full_video_decode": full_video_decode})
        from autosim.robosyn_data import validate_dataset

        info = read_json(root / "meta/info.json")
        if info["codebase_version"] == "v3.0":
            # The official converter renames the source, keeping a v3 backup.
            # Refuse any existing intermediate directory rather than overwriting it.
            for suffix in ("_v3.0", "_v2.1"):
                if root.with_name(root.name + suffix).exists():
                    raise FileExistsError(f"conversion intermediate exists: {root}{suffix}")
            self.run([str(self.python), "scripts/convert_lerobot3.0_to_2.1.py",
                      "--repo-id", root.name, "--root", str(root.parent)], output / "conversion", 1800)
        aliases = {c: (f"observation.images.{c}", f"{c}.color") for c in spec.cameras}
        # A recorded frame is not necessarily a simulator control step. Do not
        # enforce the evaluation step limit on training recording lengths.
        from .data_execution import audit_kernel_context
        registry = self.audit_kernel_registry
        if registry is None and self.audit_optimizer is not None:
            registry = self.audit_optimizer.prepare(root, remaining_seconds=(
                max(0, self.deadline-time.monotonic()) if self.deadline else 0))
        audit_started = time.perf_counter()
        with audit_kernel_context(registry) as patch:
            result = validate_dataset(root, state_dim=spec.state_dim, action_dim=spec.action_dim,
                                      max_episode_frames=None, camera_aliases=aliases,
                                      full_video_decode=full_video_decode)
        result["timings"] = {"dataset_validation_seconds": time.perf_counter()-audit_started}
        if patch:
            result["execution_patch"] = patch
        result["task_signature"] = spec.signature
        result["recorded_fps"] = read_json(root / "meta/info.json").get("fps")
        result["info_sha256"] = digest(root / "meta/info.json")
        atomic_json(output / "data_audit.json", result)
        if not result["passed"]:
            raise RuntimeError(f"data audit failed: {result['errors']}")
        from .data_version import fingerprint_dataset
        fingerprint_started = time.perf_counter()
        result["content_version"] = fingerprint_dataset(root, self.output / "data_versions")
        result["timings"]["fingerprint_seconds"] = time.perf_counter()-fingerprint_started
        atomic_json(output / "data_audit.json", result)
        return result

    def train(self, spec: TaskSpec, root: Path, output: Path, *, steps: int,
              params: dict | None = None, mixture: Path | None = None,
              seed: int = 1000, resume: bool = False,
              pretrained: Path | None = None) -> Path:
        if self._unbound_compute():
            from .data_version import training_versions
            versions = training_versions(root, mixture, self.output / "data_versions")
            request = {"spec": spec.signature, "dataset": str(root), "steps": steps, "params": params,
                       "content_ids": [row["content_id"] for row in versions],
                       "dataset_info": digest(root / "meta/info.json"),
                       "mixture": digest(mixture) if mixture else None, "seed": seed, "resume": resume,
                       "weights": digest(pretrained / "model.safetensors") if pretrained else None}
            return self._scheduled_call("training", output, lambda bound: bound.train(
                spec, root, output, steps=steps, params=params, mixture=mixture,
                seed=seed, resume=resume, pretrained=pretrained), request=request,
                estimate_seconds=max(60, steps * 1.5))
        params = dict(params or {})
        allowed = {"batch_size", "chunk_size", "n_action_steps", "optimizer_lr", "kl_weight",
                   "dropout", "action_loss_profile", "image_augmentation_profile", "num_workers"}
        if set(params) - allowed:
            raise ValueError(f"unapproved training parameters: {set(params) - allowed}")
        defaults = dict(batch_size=32, chunk_size=50, n_action_steps=50, optimizer_lr=1e-5,
                        kl_weight=10.0, dropout=0.1, action_loss_profile="legacy_mask_mean",
                        image_augmentation_profile="none", num_workers=4)
        defaults.update(params)
        # Worker RNG streams can affect stochastic image transforms. Preserve their
        # historical worker count until a per-sample RNG contract is verified.
        if (self.execution_resources and "num_workers" not in params
                and defaults["image_augmentation_profile"] == "none"):
            requested = self.cost_model.get("execution_parameters", {}).get("loader_workers", 4)
            defaults["num_workers"] = min(requested, max(0, int(self.execution_resources.get("cpu_cores", 1)) - 1))
        if not 1 <= int(defaults["n_action_steps"]) <= int(defaults["chunk_size"]):
            raise ValueError("action execution horizon must be in [1, chunk_size]")
        if resume and pretrained is not None:
            raise ValueError("resume and pretrained initialization are mutually exclusive")
        train_dir = output / "train"
        from .data_version import training_versions
        versions = training_versions(root, mixture, self.output / "data_versions")
        immutable_json(output / "observation_contract.json", spec.as_dict())
        command = [str(self.train_python), "policy/act/scripts/train.py", "--dataset-root", str(root),
                   "--observation-contract", str(output / "observation_contract.json"),
                   "--output-dir", str(train_dir), "--device", "cuda", "--steps", str(steps),
                   "--seed", str(seed), "--save-freq", str(min(20000, steps)), "--log-freq", "100"]
        for key, value in defaults.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        if mixture:
            command += ["--dataset-mixture-manifest", str(mixture),
                        "--data-pipeline-audit", str(output / "training_data_audit.json"),
                        "--sampling-exposure-audit", str(output / "training_exposure_audit.json")]
        if pretrained is not None:
            command += ["--pretrained-policy", str(pretrained)]
        if resume:
            command += ["--resume"]
        recipe = {"task_signature": spec.signature,
                    "steps": steps, "save_freq": min(20000, steps),
                    "seed": seed, "params": defaults, "dataset": str(root),
                    "training_data_content_ids": {row["root"]: row["content_id"] for row in versions},
                    "dataset_info_sha256": digest(root / "meta/info.json"),
                    "mixture_sha256": digest(mixture) if mixture else None,
                    "pretrained_checkpoint": str(pretrained) if pretrained else None,
                    "pretrained_model_sha256": (
                        digest(pretrained / "model.safetensors") if pretrained else None),
                    "pretrained_config_sha256": (
                        digest(pretrained / "config.json") if pretrained else None),
                    "trainer_sha256": digest(self.repo / "policy/act/scripts/train.py"),
                    "training_exposure_audit": (
                        str(output / "training_exposure_audit.json") if mixture else None
                    ),
                    "mixture": str(mixture) if mixture else None, "resume": resume}
        recipe_path = output / f"recipe_{steps}.json"
        if recipe_path.exists() and read_json(recipe_path) != recipe:
            raise RuntimeError("training inputs changed: use a new experiment directory")
        atomic_json(recipe_path, recipe)
        # ``--device cuda`` is ordinal 0 of the training process's own view. Under identity
        # addressing that view is the whole node, so the trainer would land on card 0
        # whatever card the job leased -- the same split the engine had, one space over.
        # Narrowing the view to the leased card makes ordinal 0 mean the leased card again,
        # and keeps a training job off its neighbours (training and simulation never share
        # a device).  ``training_selection`` is the identity on a filtered selection.
        self.run(command, output / f"process_train_{steps}", max(1800, int(steps * 1.5)),
                 selection=self.training_selection())
        checkpoint = train_dir / "checkpoints" / f"{steps:06d}" / "pretrained_model"
        if not (checkpoint / "model.safetensors").is_file():
            raise RuntimeError(f"training returned without checkpoint: {checkpoint}")
        return checkpoint

    def training_selection(self) -> SimDeviceSelection | None:
        """The selection a training process runs under: one visible card, ordinal 0.

        Training never touches the engine, so it does not need the whole visible set -- and
        under identity addressing it *must not* have it, because ``--device cuda`` would then
        mean physical card 0 rather than the leased card.  Narrowing the visible set to the
        leased card makes the ordinal honest for both modes and keeps training off every other
        device, which is what "training and simulation never share a device" asks for.
        """
        if self.selection is None:
            return None
        visible = self.selection.uuid if self.plan and self.plan.get("unified_scheduler") else str(self.selection.index)
        return replace(self.selection, cuda_visible=visible, torch_index=0)

    def shard_selections(self, episodes: int, *, purpose: str = "development") -> list[SimDeviceSelection]:
        """The devices this evaluation may spread over; a single entry means "run it whole".

        Sharding pays a full environment construction per shard, so it is only chosen when
        the device plan allows it *and* the cost model says the extra fixed cost is repaid
        (see ``devices.shard_count``).  A smoke probe and a resumed bound job stay single.
        """
        if self.plan is None or self.selection is not None or purpose == "smoke":
            return []
        usable = self.plan.get("usable") or []
        if self._unbound_compute() and usable:
            count = min(8, max(1, episodes // self.shard_min_episodes))
            return [select(usable[i % len(usable)], self.plan["mode"]) for i in range(count)] if count > 1 else []
        if len(usable) < 2:
            return []
        mode = self.plan.get("mode") or "pinned_index"
        count = shard_count(episodes, len(usable),
                            max_parallel_jobs=int(self.plan.get("max_parallel_jobs") or 1),
                            cost_model=self.cost_model,
                            min_episodes_per_shard=self.shard_min_episodes)
        if count <= 1:
            return []
        chosen = []
        for device in usable[:count]:
            # Prefer the addressing the plan froze; recomputing it could only diverge.
            frozen = device.get("selection")
            chosen.append(SimDeviceSelection(**frozen) if frozen else select(device, mode))
        return chosen

    def collection_shard_selections(self, attempts: int, *,
                                    collection_mode: str = "expert") -> list[SimDeviceSelection]:
        """The devices one collection may spread over; a single entry means "run it whole".

        Only the expert track is shardable.  ``policy_correction`` draws its prefix lengths
        from a *second* stream (``RandomState(seed ^ 0x5A17C0DE)``, one draw per attempt) and
        sizes its work from the recorder's current episode count, so a block start would have
        to shift that stream and that count too; until that is verified it stays a single
        process rather than becoming an undeclared divergence.  The cost model is the
        evaluation one unless the caller tunes the collection keys: a shard pays the same
        environment construction per device, and its attempts are the marginal cost.
        """
        if self.plan is None or self.selection is not None or collection_mode != "expert":
            return []
        usable = self.plan.get("usable") or []
        if self._unbound_compute() and usable:
            count = min(8, max(1, attempts // self.shard_min_episodes))
            return [select(usable[i % len(usable)], self.plan["mode"]) for i in range(count)] if count > 1 else []
        if len(usable) < 2:
            return []
        model = self.cost_model or {}
        count = shard_count(
            attempts, len(usable),
            max_parallel_jobs=int(self.plan.get("max_parallel_jobs") or 1),
            cost_model={"construction_seconds": model.get("collection_construction_seconds",
                                                          model.get("construction_seconds", 360.0)),
                        "marginal_seconds": model.get("collection_marginal_seconds",
                                                      model.get("marginal_seconds", 40.0))},
            min_episodes_per_shard=self.shard_min_episodes)
        if count <= 1:
            return []
        mode = self.plan.get("mode") or "pinned_index"
        chosen = []
        for device in usable[:count]:
            frozen = device.get("selection")
            chosen.append(SimDeviceSelection(**frozen) if frozen else select(device, mode))
        return chosen

    def _collect_bounded_sharded(self, spec: TaskSpec, destination: Path, *, attempt_budget: int,
                                 master_seed: int, profile: str, target_episodes: int,
                                 collection_mode: str, timeout: int,
                                 selections: list[SimDeviceSelection]) -> dict:
        """Collect the attempt budget as contiguous blocks, one process per device, then merge.

        The division is frozen before any shard starts, so a resume cannot re-divide the
        attempts, and the merge re-derives the attempt stream and refuses any shard whose
        attempts are not exactly its own block of it.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from .collection_shards import (build_collection_shard_plan, collection_attempt_bank,
                                        merge_collection_shards, shard_dir)

        bank = collection_attempt_bank(master_seed, attempt_budget + 1)
        plan = build_collection_shard_plan(
            attempts=attempt_budget, bank=bank, master_seed=master_seed,
            target_episodes=target_episodes,
            selections=[selection.as_dict() for selection in selections])
        if self._unbound_compute():
            return self._dispatch_shards("collection", destination, plan, lambda bound, i, block:
                bound._collect_bounded_shard(index=i, spec=spec, selection=bound.selection,
                    shard_root=shard_dir(destination, i), block=block, count=plan["count"],
                    master_seed=master_seed, profile=profile, collection_mode=collection_mode,
                    timeout=timeout), merge=lambda results: merge_collection_shards(
                        output=destination, plan=plan, bank=bank, task=spec.name, profile=profile,
                        collection_mode=collection_mode, attempts=attempt_budget,
                        target_episodes=target_episodes), filename="collection_shard_plan.json",
                request={"spec": spec.signature, "profile": profile, "mode": collection_mode})
        immutable_json(destination / "collection_shard_plan.json", plan)
        results: dict[int, dict] = {}
        failures: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(selections)) as pool:
            futures = {
                pool.submit(self._collect_bounded_shard, index=index, spec=spec,
                            selection=selection, shard_root=shard_dir(destination, index),
                            block=plan["blocks"][index], count=len(selections),
                            master_seed=master_seed, profile=profile,
                            collection_mode=collection_mode, timeout=timeout): index
                for index, selection in enumerate(selections)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except BaseException as exc:      # recorded, then re-raised below as a group
                    failures[index] = f"{type(exc).__name__}: {exc}"
        if failures:
            raise RuntimeError(f"collection shards failed: {failures}")
        merged = merge_collection_shards(
            output=destination, plan=plan, bank=bank, task=spec.name, profile=profile,
            collection_mode=collection_mode, attempts=attempt_budget,
            target_episodes=target_episodes)
        merged["shard_devices"] = [selection.uuid for selection in selections]
        merged["startup_attempts_per_shard"] = [results[index]["attempt"]
                                                for index in sorted(results)]
        atomic_json(destination / "bounded_collection_result.json", merged)
        return merged

    def _collect_bounded_shard(self, *, index: int, spec: TaskSpec,
                               selection: SimDeviceSelection, shard_root: Path, block: dict,
                               count: int, master_seed: int, profile: str, collection_mode: str,
                               timeout: int, startup_attempts: int = 3) -> dict:
        """One contiguous block of the attempt stream, in its own process and device."""
        bound = self.for_job(selection, job=f"{self.job or 'collection'}-shard{index}",
                             output=shard_root)
        shard_root.mkdir(parents=True, exist_ok=True)
        offset, size = int(block["offset"]), int(block["size"])
        shard_target = int(block["target_episodes"])
        first_attempt = 1
        for attempt in range(first_attempt, startup_attempts + 1):
            destination = shard_root if attempt == 1 else shard_root / f"startup_attempt_{attempt}"
            destination.mkdir(parents=True, exist_ok=True)
            manifest = destination / "collection.json"
            command = [str(bound.python), "-m", "autosim.research.collection_worker",
                       "--gym_config", spec.gym_config, "--action_config", spec.action_config,
                       "--num_envs", "1", "--headless",
                       "--max_episodes", str(shard_target),
                       "--collection_seed", str(master_seed),
                       "--collection_profile", profile, "--collection_mode", collection_mode,
                       "--collection_manifest", str(manifest),
                       "--dataset_save_path", str(destination / "data"),
                       "--collection_max_attempts", str(size),
                       "--collection_seed_offset", str(offset), "--collection_quiet"]
            command += bound.collector_device_flags()
            contract_path = destination / "expert_contract.json"
            if not contract_path.exists():
                atomic_json(contract_path, {
                    "task": spec.name, "adapter": spec.expert_adapter,
                    "collection_mode": collection_mode, "parent_checkpoint": None,
                    "parent_model_sha256": None, "training_only": True,
                    "success_function_modified": False,
                    "expert_success_cadence": "one_original_judge_query_after_each_actual_action_like_policy_adapter",
                    "terminal_hold_max_steps": 75,
                    "terminal_hold": "real recorded actions until original judge success, bounded below task horizon",
                    "retrospective_capture": (destination / "process/process.json").exists(),
                    "collector_source_sha256": digest(self.repo / "scripts/run_env.py"),
                    "adapter_source_sha256": digest(Path(__file__).with_name("collection_worker.py")),
                    "attempt_block": {"index": index, "count": count, "offset": offset,
                                      "size": size, "target_episodes": shard_target},
                })
            try:
                bound.run(command, destination / "process", timeout)
            except RuntimeError:
                if attempt == startup_attempts or not retryable_collection_startup(destination):
                    raise
                atomic_json(shard_root / f"startup_attempt_{attempt}_retry_decision.json", {
                    "reason": "native_crash_before_first_collection_reset",
                    "shard_index": index, "next_attempt": attempt + 1,
                    "accepted_episodes": 0, "attempt_budget_consumed": 0,
                    "same_collection_seed_master": True})
                continue
            result = self._finalize_collection(
                spec, destination, attempt_budget=size, target_episodes=shard_target,
                profile=profile, collection_mode=collection_mode)
            return {"index": index, "attempt": attempt, "offset": offset, "size": size,
                    "device_uuid": selection.uuid, "directory": str(destination),
                    "accepted_episodes": result["accepted_episodes"],
                    "attempts_consumed": result["attempts_consumed"]}
        # Unreachable: the loop's last iteration either returns or re-raises the cause.
        raise AssertionError(f"collection shard {index} left the startup loop without a verdict")

    def _evaluation_first_attempt(self, directory: Path, startup_attempts: int) -> int:
        """Resume provenance for one evaluation directory (whole run or one shard)."""
        first_attempt = 1
        published = directory / "evaluation_metrics.json"
        if published.is_file():
            previous = read_json(published)
            if previous.get("execution_mode") == "real_simulation":
                first_attempt = int(previous.get("startup_attempt_count", 1))
                if first_attempt not in set(range(1, startup_attempts + 1)):
                    raise ValueError("invalid startup attempt provenance")
        else:
            prior_process = directory / "process/process.json"
            if prior_process.is_file():
                prior = read_json(prior_process)
                # A launcher path failure before Popen produced a historical
                # ``running`` record without a pid. Preserve it and continue in
                # a separate attempt directory; no simulator reset occurred.
                never_started = (prior.get("status") in {"running", "failed_to_start"}
                                 and "pid" not in prior
                                 and not (directory / "initializations.jsonl").exists()
                                 and not (directory / "startup.json").exists())
                retryable_startup = retryable_evaluation_startup(directory)
                if not never_started and not retryable_startup:
                    raise RuntimeError(f"unclassified prior evaluation attempt: {prior_process}")
                first_attempt = 2
                atomic_json(directory / "startup_attempt_1_recovery.json", {
                    "reason": ("launcher_failed_before_process_creation" if never_started
                               else "native_startup_failure_before_first_reset"),
                    "prior_process_status": prior.get("status"),
                    "completed_evaluation_episodes": 0,
                    "same_checkpoint_and_seed_bank": True,
                    "next_attempt": 2,
                })
        return first_attempt

    def _finalize_evaluation(self, data: dict, spec: TaskSpec, output: Path, *,
                             purpose: str, episodes: int) -> dict:
        """The post-conditions a merged payload must satisfy exactly like a native one."""
        if data.get("execution_mode") != "real_simulation" or data.get("purpose") != purpose:
            raise RuntimeError("evaluation worker did not certify a real result under the requested purpose")
        rows = data["episodes"]
        if len(rows) != episodes or len({r["episode_seed"] for r in rows}) != episodes:
            raise RuntimeError("evaluation has missing/duplicate episode seeds")
        if data["config"]["task"] != spec.name or data["config"]["timeout_action_steps"] != spec.max_episode_steps:
            raise RuntimeError("evaluation task/horizon differs from contract")
        atomic_json(output / "evaluation_metrics.json", data)
        return data

    def evaluate(self, spec: TaskSpec, checkpoint: Path, output: Path, *, episodes: int,
                 master_seed: int, purpose: str = "development", policy: str = "act",
                 startup_attempts: int = 3, smoke_timeout: int = 300) -> dict:
        if purpose not in {"smoke", "development", "selection_validation",
                           "final_confirmation", "confirmation", "final"}:
            raise ValueError(f"invalid evaluation purpose: {purpose}")
        startup_attempts = startup_attempt_limit(startup_attempts, self.harness_config,
                                                coordinated=bool(self.native_barrier))
        if smoke_timeout < 30:
            raise ValueError("smoke_timeout must be at least 30 seconds")
        immutable_json(output / "evaluation_request.json", {
            "task_signature": spec.signature, "checkpoint": str(checkpoint),
            "weight_sha256": digest(checkpoint / "model.safetensors"),
            "model_config_sha256": digest(checkpoint / "config.json"),
            "episodes": episodes, "master_seed": master_seed, "purpose": purpose, "policy": policy})
        selections = self.shard_selections(episodes, purpose=purpose)
        if selections:
            return self._evaluate_sharded(spec, checkpoint, output, episodes=episodes,
                                          master_seed=master_seed, purpose=purpose, policy=policy,
                                          startup_attempts=startup_attempts,
                                          smoke_timeout=smoke_timeout, selections=selections)
        if self._unbound_compute():
            return self._scheduled_call("evaluation", output, lambda bound: bound.evaluate(
                spec, checkpoint, output, episodes=episodes, master_seed=master_seed, purpose=purpose,
                policy=policy, startup_attempts=startup_attempts, smoke_timeout=smoke_timeout),
                request={"spec": spec.signature, "weights": digest(checkpoint / "model.safetensors"),
                         "episodes": episodes, "seed": master_seed, "purpose": purpose, "policy": policy},
                estimate_seconds=360 + episodes * 40)
        first_attempt = self._evaluation_first_attempt(output, startup_attempts)
        for attempt in range(first_attempt, startup_attempts + 1):
            destination = output if attempt == 1 else output / f"startup_attempt_{attempt}"
            command = [str(self.python), "-m", "autosim.research.evaluation", "--task", spec.name,
                       "--checkpoint", str(checkpoint), "--output", str(destination), "--episodes", str(episodes),
                       "--seed", str(master_seed), "--purpose", purpose, "--policy", policy]
            if self.selection is not None:
                command += self.evaluation_device_flags()
            try:
                timeout = smoke_timeout if purpose == "smoke" else max(1800, episodes * 60)
                self.run(command, destination / "process", timeout, evaluation=True)
            except RuntimeError:
                if attempt == startup_attempts or not retryable_evaluation_startup(destination):
                    raise
                assert_frozen(read_json(destination / "protocol.json")["frozen_files"])
                atomic_json(destination / "startup_retry_decision.json", {
                    "reason": "native_crash_before_first_evaluation_reset", "next_attempt": attempt + 1,
                    "completed_evaluation_episodes": 0, "same_checkpoint_and_seed_bank": True})
                continue
            assert_frozen(read_json(destination / "protocol.json")["frozen_files"])
            data = read_json(destination / "evaluation_metrics.json")
            data["artifact_directory"] = str(destination)
            data["startup_attempt_count"] = attempt
            break
        return self._finalize_evaluation(data, spec, output, purpose=purpose, episodes=episodes)

    def _evaluate_sharded(self, spec: TaskSpec, checkpoint: Path, output: Path, *, episodes: int,
                          master_seed: int, purpose: str, policy: str, startup_attempts: int,
                          smoke_timeout: int, selections: list[SimDeviceSelection]) -> dict:
        """Run the bank in contiguous slices, one process per device, then merge.

        The division is frozen before anything starts, so a resume cannot re-divide the
        episodes; the merge re-derives the seed bank and refuses any slice that is not
        exactly its own block of it.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from autosim.robosyn_data import evaluation_seed_bank

        from .evaluation_merge import build_shard_plan, merge_evaluation_shards, shard_dir

        bank = evaluation_seed_bank(master_seed, episodes)
        plan = build_shard_plan(episodes=episodes, bank=bank,
                                selections=[selection.as_dict() for selection in selections])
        if self._unbound_compute():
            merged = self._dispatch_shards("evaluation", output, plan, lambda bound, i, block:
                bound._run_shard(index=i, spec=spec, checkpoint=checkpoint, selection=bound.selection,
                    shard_root=shard_dir(output, i), block=block, count=plan["count"], episodes=episodes,
                    master_seed=master_seed, purpose=purpose, policy=policy,
                    startup_attempts=startup_attempts, smoke_timeout=smoke_timeout),
                merge=lambda results: merge_evaluation_shards(output=output, purpose=purpose,
                    master_seed=master_seed, episodes=episodes, bank=bank,
                    merge_command=["autosim.research.evaluation", "scheduler-merged"]),
                filename="shard_plan.json", request={"spec": spec.signature,
                    "weights": digest(checkpoint / "model.safetensors"),
                    "config": digest(checkpoint / "config.json"), "purpose": purpose, "policy": policy})
            return self._finalize_evaluation(merged, spec, output, purpose=purpose, episodes=episodes)
        immutable_json(output / "shard_plan.json", plan)
        results: dict[int, dict] = {}
        failures: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(selections)) as pool:
            futures = {
                pool.submit(self._run_shard, index=index, spec=spec, checkpoint=checkpoint,
                            selection=selection, shard_root=shard_dir(output, index),
                            block=plan["blocks"][index], count=len(selections), episodes=episodes,
                            master_seed=master_seed, purpose=purpose, policy=policy,
                            startup_attempts=startup_attempts, smoke_timeout=smoke_timeout): index
                for index, selection in enumerate(selections)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except BaseException as exc:      # recorded, then re-raised below as a group
                    failures[index] = f"{type(exc).__name__}: {exc}"
        if failures:
            raise RuntimeError(f"evaluation shards failed: {failures}")
        merged = merge_evaluation_shards(
            output=output, purpose=purpose, master_seed=master_seed, episodes=episodes, bank=bank,
            merge_command=["autosim.research.evaluation (merged from shards)",
                           f"--task {spec.name}", f"--episodes {episodes}", f"--seed {master_seed}"])
        merged["artifact_directory"] = str(output)
        merged["shard_count"] = len(selections)
        merged["startup_attempt_count"] = max(row["attempt"] for row in results.values())
        merged["startup_attempts_per_shard"] = [results[index]["attempt"]
                                               for index in sorted(results)]
        merged["shard_devices"] = [selection.uuid for selection in selections]
        return self._finalize_evaluation(merged, spec, output, purpose=purpose, episodes=episodes)

    def _dispatch_shards(self, category, output, plan, callback, *, merge, filename, request):
        from .accounting import BudgetLedger
        from .job_runner import PhaseJob, run_phase
        from .resource_contracts import ResourceRequest
        # Logical blocks are protocol inputs; device placement is a separate execution receipt.
        for block in plan["blocks"]:
            for key in ("device_uuid", "device_index", "selection"):
                block.pop(key, None)
        plan["placement_independent"] = True
        path = output / filename
        if path.exists():
            previous = read_json(path)
            # Preserve the original block boundaries when resuming under another resource shape.
            for key in ("episodes", "attempts", "target_episodes", "master_seed", "bank_sha256"):
                if previous.get(key) != plan.get(key):
                    raise ValueError(f"shard protocol changed: {key}")
            plan.clear(); plan.update(previous)
        immutable_json(path, plan)
        if self.compute_ledger is None:
            self.compute_ledger = BudgetLedger(self.output / "compute/budget.json",
                wall_limit_seconds=max(1, self.deadline-time.monotonic()) if self.deadline else 86400,
                gpu_hours_limit=self.plan.get("gpu_hours_limit"), devices=self.plan["usable"])
        count = max(1, self.plan.get("max_parallel_jobs") or 1)
        resources = ResourceRequest(cpu_cores=max(1, min(4, int(self.plan["host_capacity"]["cpu_cores"] // count))),
                                    ram_mib=4096, per_gpu_memory_mib=2048,
                                    capabilities=("physics", "render", "policy_inference") if category == "evaluation"
                                    else ("physics", "render"))
        jobs = [PhaseJob(f"block_{i:03d}", category, lambda bound, i=i, block=block: callback(bound, i, block),
                        resources=resources, estimate_seconds=360 + block["size"] * 40, reserve_seconds=0,
                        input_digest=object_digest({"plan": plan, "block": block, "request": request}),
                        code_digest=self._execution_code_digest(),
                        profile_digest=self._profile_digest(category,request,block_size=block["size"]))
                for i, block in enumerate(plan["blocks"])]
        phase = f"{category}_{object_digest(str(output.absolute()))[:16]}"
        results = run_phase(phase=phase, jobs=jobs, runtime=self, plan=self.plan,
                            output=self.output / "compute/schedule", ledger=self.compute_ledger,
                            run_id=object_digest(str(self.output))[:16])
        merged = merge(results)
        merged.update(artifact_directory=str(output), shard_count=plan["count"],
                      startup_attempt_count=max(r["attempt"] for r in results.values()),
                      shard_devices=[r["device_uuid"] for r in results.values()])
        if category == "collection":
            atomic_json(output / "bounded_collection_result.json", merged)
        return merged

    def _run_shard(self, *, index: int, spec: TaskSpec, checkpoint: Path,
                   selection: SimDeviceSelection, shard_root: Path, block: dict, count: int,
                   episodes: int, master_seed: int, purpose: str, policy: str,
                   startup_attempts: int, smoke_timeout: int) -> dict:
        """One contiguous block of the bank, in its own process and on its own device."""
        bound = self.for_job(selection, job=f"{self.job or purpose}-shard{index}", output=shard_root)
        shard_root.mkdir(parents=True, exist_ok=True)
        offset, size = int(block["offset"]), int(block["size"])
        immutable_json(shard_root / "evaluation_request.json", {
            "task_signature": spec.signature, "checkpoint": str(checkpoint),
            "weight_sha256": digest(checkpoint / "model.safetensors"),
            "model_config_sha256": digest(checkpoint / "config.json"),
            "episodes": size, "master_seed": master_seed, "purpose": purpose, "policy": policy,
            "shard": {"index": index, "count": count, "seed_offset": offset, "episodes": size}})
        first_attempt = self._evaluation_first_attempt(shard_root, startup_attempts)
        for attempt in range(first_attempt, startup_attempts + 1):
            destination = shard_root if attempt == 1 else shard_root / f"startup_attempt_{attempt}"
            destination.mkdir(parents=True, exist_ok=True)
            command = [str(bound.python), "-m", "autosim.research.evaluation", "--task", spec.name,
                       "--checkpoint", str(checkpoint), "--output", str(destination),
                       "--episodes", str(size), "--seed", str(master_seed), "--purpose", purpose,
                       "--policy", policy, "--seed-offset", str(offset),
                       "--shard-index", str(index), "--shard-count", str(count)]
            command += bound.evaluation_device_flags()
            try:
                timeout = smoke_timeout if purpose == "smoke" else max(1800, episodes * 60)
                bound.run(command, destination / "process", timeout, evaluation=True)
            except RuntimeError:
                if attempt == startup_attempts or not retryable_evaluation_startup(destination):
                    raise
                assert_frozen(read_json(destination / "protocol.json")["frozen_files"])
                atomic_json(destination / "startup_retry_decision.json", {
                    "reason": "native_crash_before_first_evaluation_reset",
                    "next_attempt": attempt + 1, "shard_index": index,
                    "completed_evaluation_episodes": 0, "same_checkpoint_and_seed_bank": True})
                continue
            assert_frozen(read_json(destination / "protocol.json")["frozen_files"])
            return {"index": index, "directory": str(destination), "attempt": attempt,
                    "device_uuid": selection.uuid}
        # Unreachable: the loop's last iteration either returns or re-raises the cause, and
        # ``_evaluation_first_attempt`` never returns a number above ``startup_attempts``.
        raise AssertionError(f"shard {index} left the startup loop without a verdict")
