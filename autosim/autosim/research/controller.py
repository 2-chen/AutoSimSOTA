"""Durable bounded research cycles; final evaluation is a separate locked phase."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import time

from autosim.robosyn_mvp import gpu_lock
from .analysis import analyze
from .common import (assert_frozen, atomic_json, digest, event, exclusive, freeze_files,
                     now, object_digest, read_json, redact)
from .ledger import SeedLedger, compare
from .proposals import Proposal, propose
from .registry import TASK_IDS, load_task
from .runtime import Runtime


_NON_SOURCE_DIRECTORIES = frozenset({
    ".venv", "venv", ".git", ".hg", ".svn", "__pycache__", ".cache",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".conda",
    "site-packages", "dist-packages", "node_modules",
})


def source_tree_files(root: Path, suffix: str) -> list[Path]:
    """Enumerate project source without entering environments or linked trees."""
    root = Path(root)
    if root.is_symlink():
        return []
    files = []
    for directory, children, names in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        children[:] = sorted(name for name in children
                             if name not in _NON_SOURCE_DIRECTORIES
                             and not (base / name).is_symlink()
                             and not (base / name / "pyvenv.cfg").is_file())
        files.extend(base / name for name in sorted(names) if name.endswith(suffix))
    return files


def controller_source_files(runtime: Runtime) -> list[Path]:
    """Freeze scientific code/config; dependency identity belongs to environment contracts."""
    files = source_tree_files(runtime.eval_repo / "robosynchallenge", ".py")
    files += source_tree_files(runtime.eval_repo / "configs", ".json")
    files += source_tree_files(runtime.workspace / "AutoSimSOTA/EmbodiChain/embodichain/lab", ".py")
    files += list(Path(__file__).parent.glob("*.py"))
    files += list(Path(__file__).parent.parent.glob("robosyn*.py"))
    files += source_tree_files(runtime.repo / "policy/act", ".py")
    files += [runtime.eval_repo / "scripts/eval_policy.py",
              runtime.repo / "policy/act/scripts/train.py",
              runtime.repo / "scripts/run_env.py",
              runtime.eval_repo / "policy/act/deploy_policy.py"]
    return sorted(set(files))


@dataclass(frozen=True)
class ResearchConfig:
    rounds: int = 2
    screen_steps: int = 20000
    final_steps: int = 80000
    development_episodes: int = 40
    confirmation_episodes: int = 100
    collect_per_round: int = 50
    train_seed: int = 1000
    hours_per_task: float = 40
    allow_llm: bool = False
    pilot: bool = False

    def validate(self):
        if not 2 <= self.rounds <= 8:
            raise ValueError("research must run 2–8 bounded rounds")
        if not 0 < self.screen_steps <= self.final_steps <= 160000:
            raise ValueError("invalid training step budget")
        if min(self.development_episodes, self.confirmation_episodes, self.collect_per_round) <= 0:
            raise ValueError("episode budgets must be positive")
        if not 0 < self.hours_per_task <= 168:
            raise ValueError("task time ceiling must be in (0,168] hours")
        return self


def task_cost(directory: Path) -> float:
    return sum(float(read_json(p).get("elapsed_seconds", 0)) for p in directory.rglob("process.json"))


def make_mixture(roots: list[Path], path: Path, profiles: list[str], *, pilot=False) -> Path:
    if len(roots) != len(profiles) or len(set(str(p.resolve()) for p in roots)) != len(roots):
        raise ValueError("mixture roots/profiles must align and roots must be unique")
    entries = []
    for index, (root, profile) in enumerate(zip(roots, profiles)):
        info = read_json(root / "meta/info.json")
        entries.append({"root": str(root), "profile": profile,
                        "source_kind": ("self_collected_pilot" if pilot else "official_full_random")
                                       if index == 0 else ("self_collected_full_random" if profile == "full_random"
                                                            else "targeted_training_slice"),
                        "episode_count": info["total_episodes"], "frame_count": info["total_frames"],
                        "info_sha256": digest(root / "meta/info.json")})
    data = {"schema_version": 3, "kind": "robosyn_task_mixture", "datasets": entries,
            "sampling": "proportional_to_frames", "total_episodes": sum(e["episode_count"] for e in entries),
            "total_frames": sum(e["frame_count"] for e in entries)}
    if path.exists() and read_json(path) != data:
        raise RuntimeError("refusing to redefine a data version")
    atomic_json(path, data)
    return path


def import_known_seeds(runtime: Runtime, ledger: SeedLedger):
    """Read IDs only, never feed historical test performance to the proposer."""
    roots = [runtime.workspace / "autosim/output", runtime.repo / "eval_result"]
    for root in roots:
        for path in root.rglob("*.json"):
            if any(x in path.parts for x in ("source_before", "assets", "checkpoints", "data")):
                continue
            try:
                if path.stat().st_size > 10 * 1024 * 1024:
                    continue
                data = read_json(path)
                if not isinstance(data, dict):
                    continue
                task = data.get("task") or data.get("config", {}).get("task")
                if task not in TASK_IDS:
                    continue
                if data.get("kind") == "robosyn_expert_collection":
                    seeds = [r["seed"] for r in data.get("resets", [])]
                else:
                    seeds = [r["episode_seed"] for r in data.get("episodes", []) if "episode_seed" in r]
                if seeds:
                    ledger.retire(task, seeds, str(path))
            except (ValueError, TypeError, KeyError, OSError):
                continue


class ResearchController:
    def __init__(self, runtime: Runtime, config: ResearchConfig):
        self.runtime, self.config = runtime, config.validate()
        self.root = runtime.output / ("pilot_research" if config.pilot else "research") / f"train_seed_{config.train_seed}"
        self.root.mkdir(parents=True, exist_ok=True)
        lock_config = self.root / "research_config.json"
        if lock_config.exists() and read_json(lock_config) != asdict(config):
            raise ValueError("cannot change the configuration of an existing research run")
        atomic_json(lock_config, asdict(config))
        self.ledger = SeedLedger(runtime.output / "seeds.sqlite")
        import_known_seeds(runtime, self.ledger)

    def _base_data(self, task: str) -> tuple[Path, Path | None]:
        inventory = read_json(self.runtime.output / "task_inventory.json")
        item = next(r for r in inventory if r["task"] == task)
        if self.config.pilot:
            smoke = read_json(self.runtime.output / "smoke_status.json")["tasks"][task]
            if smoke.get("status") == "passed":
                return Path(smoke["dataset"]), Path(item["official_checkpoint"]) if item["checkpoint_available"] else None
        asset_path = self.runtime.output / "asset_status.json"
        assets = read_json(asset_path)["tasks"].get(task, {}) if asset_path.exists() else {}
        if item["dataset_available"]:
            root = Path(item["official_dataset"])
        elif assets.get("dataset", {}).get("status") == "completed":
            root = Path(assets["dataset"]["path"])
        else:
            raise FileNotFoundError(f"official dataset not ready: {task}")
        checkpoint = None
        if item["checkpoint_available"]:
            checkpoint = Path(item["official_checkpoint"])
        elif assets.get("model", {}).get("status") == "completed":
            checkpoint = Path(assets["model"]["path"])
        return root, checkpoint

    def run_task(self, task: str) -> dict:
        directory = self.root / task
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "state.json"
        state = read_json(path) if path.exists() else {"task": task, "created_at": now(), "rounds": [],
                    "status": "pending", "final_evaluation": "not_run", "improvement": "not_established"}
        if state["status"] in {"completed_development", "failed"}:
            return state
        spec = load_task(self.runtime.repo, task)
        config = self.config
        protocol_path = directory / "frozen_protocol.json"
        if protocol_path.exists():
            frozen = read_json(protocol_path)
        else:
            frozen = freeze_files(controller_source_files(self.runtime))
            atomic_json(protocol_path, frozen)

        def gate(stage):
            assert_frozen(frozen)
            cost = task_cost(directory)
            if cost >= config.hours_per_task * 3600:
                raise RuntimeError("task research budget exhausted; no final success claim")
            self.runtime.deadline = time.monotonic() + config.hours_per_task * 3600 - cost
            state.update(stage=stage, status="running", elapsed_process_seconds=cost)
            atomic_json(path, state)
            print(f"[{task}] {stage}, accounted process time {cost / 3600:.2f} h", flush=True)

        # Seeds depend only on task and predeclared study, not performance.
        seed_base = 80_000_000 + list(TASK_IDS).index(task) * 1_000_000
        if config.pilot:
            seed_base += 20_000_000
        seed_base += (config.train_seed - 1000) * 10000

        def evaluate(checkpoint, label, purpose="development", bank=1):
            count = config.development_episodes if purpose == "development" else config.confirmation_episodes
            suffix = f"{purpose}_bank_{bank}" if purpose == "development" else purpose
            seed = seed_base + (1 if bank == 1 else 3) if purpose == "development" else seed_base + 2
            self.ledger.reserve(task, f"{self.root.name}:{'pilot' if config.pilot else 'full'}:{suffix}", purpose, seed, count)
            gate(f"evaluate:{label}:{purpose}")
            destination = directory / "evaluations" / f"{label}_{suffix}"
            result = self.runtime.evaluate(spec, checkpoint, destination, episodes=count, master_seed=seed, purpose=purpose)
            return result, destination

        try:
            smoke = read_json(self.runtime.output / "smoke_status.json")["tasks"].get(task, {})
            collect_enabled = smoke.get("status") == "passed"
            if not collect_enabled:
                fallback_path = self.runtime.output / "official_policy_smoke_status.json"
                fallback = read_json(fallback_path)["tasks"].get(task, {}) if fallback_path.exists() else {}
                if fallback.get("status") != "passed":
                    raise RuntimeError("neither full-collection nor official-data policy smoke gate passed")
            state.update(automatic_collection_validated=collect_enabled,
                         research_mode="data_augmentation" if collect_enabled else "official_data_only_fallback",
                         control_description="random_collection_fixed_recipe" if collect_enabled else "official_data_fixed_recipe_no_collection")
            root, released = self._base_data(task)
            gate("official_data_audit")
            self.runtime.prepare_data(spec, root, directory / "official_data_audit")
            gate("controlled_official_data_training")
            baseline = self.runtime.train(spec, root, directory / "controlled_baseline",
                                          steps=config.final_steps, seed=config.train_seed)
            baseline_metrics, baseline_eval = evaluate(baseline, "controlled_baseline")
            incumbent, incumbent_metrics, incumbent_eval = baseline, baseline_metrics, baseline_eval
            if released is not None:
                released_metrics, _ = evaluate(released, "released_checkpoint")
                state["released_checkpoint_summary"] = released_metrics["summary"]
            seen = {r["proposal_signature"] for r in state["rounds"]}
            winners = []
            auto_roots, auto_profiles = [root], ["full_random"]
            random_roots, random_profiles = [root], ["full_random"]
            # Each arm gets the same trajectory cap and the same number of
            # screening trainings plus one long training. All attempts count.
            for index in range(config.rounds):
                label = f"round_{index + 1}"
                saved = next((r for r in state["rounds"] if r["index"] == index), None)
                if saved:
                    if saved["auto_data"] is not None:
                        auto_roots.append(Path(saved["auto_data"]))
                        auto_profiles.append(saved["proposal"]["collection_profile"])
                        random_roots.append(Path(saved["random_data"]))
                        random_profiles.append("full_random")
                    winners.append(saved)
                    if saved["promoted_for_development"]:
                        incumbent = Path(saved["checkpoint"])
                        incumbent_eval = Path(saved["evaluation"])
                        incumbent_metrics = read_json(incumbent_eval / "evaluation_metrics.json")
                    continue
                gate(f"analyze:{label}")
                analysis = analyze(incumbent_eval, directory / label / "failure_analysis.json")
                proposal, provider = propose(analysis, seen, allow_llm=config.allow_llm)
                if proposal is None:
                    state["stop_reason"] = "bounded proposal library exhausted"
                    break
                seen.add(proposal.signature)
                atomic_json(directory / label / "proposal.json", {**proposal.as_dict(), "provider": provider})
                data_roots = {"auto": None, "random": None}
                arms = (("auto", proposal.collection_profile), ("random", "full_random")) if collect_enabled else ()
                for arm, profile in arms:
                    master = seed_base + 100 + index * 10 + (0 if arm == "auto" else 1)
                    self.ledger.reserve(task, f"{self.root.name}:{config.pilot}:{label}:{arm}:collection",
                                        "collection", master, config.collect_per_round * 40 + 10)
                    gate(f"collect:{label}:{arm}")
                    dest = directory / label / arm / "collection"
                    shard = self.runtime.collect(spec, dest, episodes=config.collect_per_round,
                                                 master_seed=master, profile=profile, timeout=7200)
                    self.runtime.prepare_data(spec, shard, dest)
                    data_roots[arm] = shard
                if collect_enabled:
                    auto_roots.append(data_roots["auto"])
                    auto_profiles.append(proposal.collection_profile)
                    random_roots.append(data_roots["random"])
                    random_profiles.append("full_random")
                auto_mix = make_mixture(auto_roots, directory / label / "auto_mixture.json", auto_profiles, pilot=config.pilot and collect_enabled)
                random_mix = make_mixture(random_roots, directory / label / "random_mixture.json", random_profiles, pilot=config.pilot and collect_enabled)
                gate(f"train:{label}:auto")
                candidate = self.runtime.train(spec, root, directory / label / "auto", steps=config.screen_steps,
                                               params=proposal.params, mixture=auto_mix, seed=config.train_seed)
                metrics, eval_dir = evaluate(candidate, label + "_auto")
                gate(f"train:{label}:random_control")
                random_candidate = self.runtime.train(spec, root, directory / label / "random", steps=config.screen_steps,
                                                      mixture=random_mix, seed=config.train_seed)
                random_metrics, random_eval = evaluate(random_candidate, label + "_random")
                paired = compare(metrics, incumbent_metrics)
                promoted = paired["success_delta"] > 0  # Development only, never final acceptance.
                saved = {"index": index, "proposal": proposal.as_dict(), "proposal_signature": proposal.signature,
                         "auto_data": str(data_roots["auto"]) if collect_enabled else None,
                         "random_data": str(data_roots["random"]) if collect_enabled else None,
                         "collection_executed": collect_enabled,
                         "auto_mixture": str(auto_mix), "random_mixture": str(random_mix),
                         "checkpoint": str(candidate), "evaluation": str(eval_dir),
                         "random_checkpoint": str(random_candidate), "random_evaluation": str(random_eval),
                         "summary": metrics["summary"], "random_summary": random_metrics["summary"],
                         "comparison_to_incumbent": paired, "promoted_for_development": promoted}
                state["rounds"].append(saved)
                winners.append(saved)
                if promoted:
                    incumbent, incumbent_metrics, incumbent_eval = candidate, metrics, eval_dir
                atomic_json(path, state)
            if not winners:
                raise RuntimeError("no completed experimental rounds")
            # A second development bank checks ranking stability. It remains
            # development data, not an untouched final test.
            for row in winners:
                for arm in ("auto", "random"):
                    key = "checkpoint" if arm == "auto" else "random_checkpoint"
                    data, destination = evaluate(Path(row[key]), f"round_{row['index'] + 1}_{arm}", bank=2)
                    row[f"{arm}_second_bank"] = {"evaluation": str(destination), "summary": data["summary"]}
                    first = row["summary" if arm == "auto" else "random_summary"]["success_rate"]
                    row[f"{arm}_selection_score"] = (first + data["summary"]["success_rate"]) / 2
                atomic_json(path, state)
            # Choose using development data only, then train both selected arms
            # to the same final step budget and confirm on a separate bank.
            selected = max(winners, key=lambda r: (r["auto_selection_score"], -r["summary"]["average_action_steps"]))
            selected_random = max(winners, key=lambda r: (r["random_selection_score"], -r["random_summary"]["average_action_steps"]))
            long_checkpoints = {}
            for arm, selected_row in (("auto", selected), ("random", selected_random)):
                gate(f"long_training:{arm}")
                long_checkpoints[arm] = self.runtime.train(spec, root, directory / f"round_{selected_row['index'] + 1}" / arm,
                    steps=config.final_steps, seed=config.train_seed,
                    params=selected_row["proposal"]["params"] if arm == "auto" else {},
                    mixture=Path(selected_row[f"{arm}_mixture"]), resume=config.final_steps > config.screen_steps)
            confirmed_baseline, _ = evaluate(baseline, "controlled_baseline", "confirmation")
            confirmations = {}
            for arm, checkpoint in long_checkpoints.items():
                result, result_dir = evaluate(checkpoint, arm + "_selected", "confirmation")
                confirmations[arm] = {"checkpoint": str(checkpoint), "evaluation": str(result_dir),
                                      "summary": result["summary"], "vs_controlled_baseline": compare(result, confirmed_baseline)}
            state.update(status="completed_development", stage="awaiting_locked_final_test",
                         baseline_checkpoint=str(baseline), selected=confirmations,
                         completed_at=now(), elapsed_process_seconds=task_cost(directory),
                         final_evaluation="not_run", improvement="not_established",
                         pilot_only=config.pilot, completed_rounds=len(winners))
            atomic_json(path, state)
            return state
        except Exception as exc:
            state.update(status="failed", error=redact(f"{type(exc).__name__}: {exc}"),
                         elapsed_process_seconds=task_cost(directory), finished_at=now())
            atomic_json(path, state)
            return state

    def run(self, tasks: list[str]) -> dict:
        status = {}
        with exclusive(self.runtime.output / "suite.lock"), gpu_lock(self.runtime.gpu):
            for task in tasks:
                status[task] = self.run_task(task)
                atomic_json(self.root / "suite_status.json", status)
                print(f"[{task}] research status: {status[task]['status']}", flush=True)
        return status
