"""RoboSyn backend policy-free discovery and capability implementation.

Task semantics live here and in :mod:`registry`; the research controller only
consumes contracts and verified operation names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapter_protocol import Axis, OptimizationSpace
from .common import digest, read_json
from .contracts import CapabilityRecord
from .registry import EVENT_FAMILIES, TASK_IDS, TaskSpec, inventory, load_task


PROFILE_FAMILIES = {
    "targeted_camera": {"camera"},
    "targeted_appearance": {"appearance"},
    "targeted_clutter": {"clutter"},
    "targeted_recovery": {"robot_pose", "clutter"},
    "composite_hard": set(EVENT_FAMILIES.values()),
}
TRAINING_READBACK_PROFILES = {"targeted_clutter", "targeted_recovery"}


class RoboSynAdapter:
    benchmark = "RoboSynChallenge"

    def __init__(self, repo: Path):
        self.repo = repo.absolute()

    @staticmethod
    def recognise(repo: Path) -> dict[str, Any]:
        """Is this checkout RoboSynChallenge at all, before asking anything about a task.

        Identity is a question about the repository and a task is a question inside it, so
        this asks the first one on markers that do not depend on which task you want. It
        exists because the two were asked in the wrong order: a checkout that was not
        RoboSyn produced "no task has both an official ACT checkpoint and dataset", which
        is a true sentence about a checkout the system should never have been reading.
        """
        repo = Path(repo)
        markers = {
            "pyproject": repo / "pyproject.toml",
            "native_evaluator": repo / "scripts" / "eval_policy.py",
            "task_package": repo / "robosynchallenge" / "tasks",
        }
        missing = sorted(name for name, path in markers.items() if not path.exists())
        named = False
        if "pyproject" not in missing:
            text = markers["pyproject"].read_text(encoding="utf-8", errors="replace").lower()
            named = "robosynchallenge" in text
        return {"repo": str(repo.absolute()), "benchmark": RoboSynAdapter.benchmark,
                "recognised": not missing and named,
                "missing_markers": missing, "pyproject_names_the_project": named,
                "markers": {name: str(path) for name, path in markers.items()},
                "how_to_read_it_anyway":
                    f"autosim scout {repo} --run-id <id> reads an unfamiliar benchmark and "
                    f"reports what it can and cannot do"}

    def task_inventory(self) -> list[dict[str, Any]]:
        return inventory(self.repo)

    def select_task(self, requested: str) -> str:
        if requested != "auto":
            if requested not in TASK_IDS:
                raise ValueError(f"unknown RoboSyn task: {requested}")
            return requested
        ready = [row["task"] for row in self.task_inventory()
                 if row.get("contract_status") == "passed"
                 and row.get("checkpoint_available") and row.get("dataset_available")]
        if not ready:
            raise FileNotFoundError("no task has both an official ACT checkpoint and dataset")
        # Stable registry order makes auto-selection reproducible.
        return ready[0]

    def discover(self, task: str) -> dict[str, Any]:
        evidence = {
            "pyproject": self.repo / "pyproject.toml",
            "native_evaluator": self.repo / "scripts/eval_policy.py",
            "task_config": self.repo / "configs" / task / "random" / "gym_config.json",
            "action_config": self.repo / "configs" / task / "action_config.json",
            "task_source": self.repo / "robosynchallenge" / "tasks" / task / f"{task}.py",
        }
        missing = [name for name, path in evidence.items() if not path.is_file()]
        if missing:
            raise ValueError(f"repository is not an unambiguous RoboSyn checkout for {task}; missing {missing}")
        if "robosynchallenge" not in evidence["pyproject"].read_text(
                encoding="utf-8", errors="replace").lower():
            raise ValueError("pyproject does not identify RoboSynChallenge")
        spec = load_task(self.repo, task)
        return {
            "schema_version": 2,
            "benchmark": self.benchmark,
            "selected_task": task,
            "recognition": "verified_known_adapter",
            "task_contract": spec.as_dict(),
            "task_signature": spec.signature,
            "evidence": {name: {"path": str(path), "sha256": digest(path)}
                         for name, path in evidence.items()},
            "inventory": self.task_inventory(),
        }

    def resolve_assets(self, task: str) -> dict[str, Any]:
        checkpoint = self.repo / "checkpoints" / f"ACT_sim_{task}"
        dataset = (self.repo / "lerobot_dataset" / "RoboSynChallenge"
                   / f"cobotmagic_Sim_{task}")
        required = [checkpoint / "model.safetensors", checkpoint / "config.json",
                    dataset / "meta/info.json"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"official assets are incomplete for {task}: {missing}")
        info = read_json(dataset / "meta/info.json")
        episodes = int(info.get("total_episodes", -1))
        frames = int(info.get("total_frames", -1))
        if episodes <= 0 or frames <= 0:
            raise ValueError(f"official dataset metadata is invalid for {task}")
        data_files = len(list((dataset / "data").glob("**/*.parquet")))
        video_files = len(list((dataset / "videos").glob("**/*.mp4")))
        video_features = sum(value.get("dtype") == "video"
                             for value in info.get("features", {}).values())
        if info.get("codebase_version") == "v2.1":
            expected_data = episodes
            expected_videos = episodes * video_features
            if data_files != expected_data or video_files != expected_videos:
                raise FileNotFoundError(
                    f"official dataset download is incomplete for {task}: "
                    f"parquet={data_files}/{expected_data}, videos={video_files}/{expected_videos}")
        return {
            "task": task,
            "official_checkpoint": str(checkpoint),
            "official_dataset": str(dataset),
            "checkpoint_weight_sha256": digest(checkpoint / "model.safetensors"),
            "checkpoint_config_sha256": digest(checkpoint / "config.json"),
            "dataset_info_sha256": digest(dataset / "meta/info.json"),
            "official_episodes": episodes,
            "official_frames": frames,
            "data_file_count": data_files,
            "video_file_count": video_files,
            "sources": {
                "checkpoint": f"https://huggingface.co/RoboSynChallenge/ACT_sim_{task}",
                "dataset": f"https://huggingface.co/datasets/RoboSynChallenge/cobotmagic_Sim_{task}",
            },
        }

    def asset_inventory(self, task: str) -> dict[str, Any]:
        checkpoint = self.repo / "checkpoints" / f"ACT_sim_{task}"
        dataset = (self.repo / "lerobot_dataset" / "RoboSynChallenge"
                   / f"cobotmagic_Sim_{task}")
        dataset_complete = False
        dataset_counts: dict[str, int] = {}
        info_path = dataset / "meta/info.json"
        if info_path.is_file():
            info = read_json(info_path)
            episodes = int(info.get("total_episodes", 0))
            video_features = sum(value.get("dtype") == "video"
                                 for value in info.get("features", {}).values())
            data_files = len(list((dataset / "data").glob("**/*.parquet")))
            video_files = len(list((dataset / "videos").glob("**/*.mp4")))
            dataset_counts = {"episodes_declared": episodes, "parquet_files": data_files,
                              "video_files": video_files,
                              "video_files_expected": episodes * video_features}
            dataset_complete = (info.get("codebase_version") == "v2.1"
                                and data_files == episodes
                                and video_files == episodes * video_features)
        return {
            "task": task,
            "checkpoint": {
                "path": str(checkpoint),
                "available": (checkpoint / "model.safetensors").is_file(),
                "source_repo_id": f"RoboSynChallenge/ACT_sim_{task}",
                "repo_type": "model",
            },
            "dataset": {
                "path": str(dataset),
                "available": dataset_complete,
                "counts": dataset_counts,
                "source_repo_id": f"RoboSynChallenge/cobotmagic_Sim_{task}",
                "repo_type": "dataset",
            },
        }

    def collection_profiles(self, spec: TaskSpec) -> set[str]:
        families = set(spec.event_families)
        profiles = {name for name, required in PROFILE_FAMILIES.items()
                    if name in TRAINING_READBACK_PROFILES and required <= families}
        # composite_hard preserves the full official distribution and is
        # available whenever at least one randomized family exists.
        if spec.name == "click_bell" and families:
            profiles.add("composite_hard")
        return profiles

    def capabilities(self, spec: TaskSpec) -> list[CapabilityRecord]:
        profiles = sorted(self.collection_profiles(spec))
        families = set(spec.event_families)
        diagnostic_only = sorted(
            name for name, required in PROFILE_FAMILIES.items()
            if name not in profiles and name != "composite_hard"
            and required and required <= families
        )
        records = [
            CapabilityRecord("native_evaluation", spec.name, "declared",
                             {"entrypoint": str(self.repo / "scripts/eval_policy.py")}),
            CapabilityRecord("official_expert", spec.name, "declared",
                             {"adapter": spec.expert_adapter}),
            CapabilityRecord("original_distribution_collection", spec.name, "declared"),
            CapabilityRecord("official_dataset_quality", spec.name, "declared"),
            CapabilityRecord("targeted_collection", spec.name,
                             "declared" if profiles else "unsupported",
                             {"training_profiles": profiles,
                              "diagnostic_only_unknown_profiles": diagnostic_only}),
            CapabilityRecord("policy_prefix_expert_takeover", spec.name,
                             "declared" if spec.correction_supported else "unsupported",
                             limitation=None if spec.correction_supported else
                             "task-specific continuity probe has not passed"),
            CapabilityRecord("act_training", spec.name, "declared",
                             {"parameter_space": "registered_v1"}),
        ]
        records.extend(CapabilityRecord(
            f"targeted_collection.{profile}", spec.name, "declared",
            {"profile": profile},
            "full official-distribution alias, not a factor-isolation intervention"
            if profile == "composite_hard" else None)
            for profile in profiles)
        records.extend(CapabilityRecord(
            f"targeted_collection.{profile}", spec.name, "unknown",
            {"profile": profile, "training_admission": "forbidden"},
            "realized camera/material parameters are not available for readback")
            for profile in diagnostic_only)
        return records

    #: What this benchmark's trainer implements. Facts about the ACT trainer shipped in
    #: the repository, declared here rather than written into the decision layer.
    TRAINING_BOUNDS = {
        "optimizer_lr": (1e-8, 1e-1),
        "n_action_steps": (1, 50),
        "kl_weight": (0.0, 1e5),
        "dropout": (0.0, 0.99),
        "batch_size": (1, 4096),
    }
    #: (implemented values, the one this benchmark has reason to believe works). The second
    #: is only the skeleton's starting point; every value in the first is proposable.
    TRAINING_OPTIONS = {
        "action_loss_profile": (("legacy_mask_mean", "valid_mean"), "valid_mean"),
        "image_augmentation_profile": (("none", "clutter_occlusion_mild", "photometric_mild",
                                        "photometric_strong", "camera_geometry_mild",
                                        "illumination_step_mild"), "photometric_mild"),
    }
    PHASE_BIN_NAMES = ("early", "middle", "late")

    @staticmethod
    def _valid_phase_weights(value: Any) -> bool:
        if not isinstance(value, dict) or not value:
            return False
        if any(not isinstance(v, (int, float)) or v <= 0 for v in value.values()):
            return False
        return set(value) in ({"early", "middle", "late"},
                              {"early", "approach", "contact_recovery"})

    @staticmethod
    def _valid_phase_bins(value: Any) -> bool:
        if not isinstance(value, list) or not value:
            return False
        spans = []
        for row in value:
            if not isinstance(row, dict) or set(row) != {"name", "start", "end", "weight"}:
                return False
            if float(row["weight"]) <= 0 or not 0.0 <= float(row["start"]) < float(row["end"]):
                return False
            spans.append((float(row["start"]), float(row["end"])))
        spans.sort()
        if spans[0][0] > 0.0 or spans[-1][1] < 1.0:
            return False
        return all(start <= end for (_, end), (start, _) in zip(spans, spans[1:]))

    def optimization_space(self, task: str) -> OptimizationSpace:
        """Everything the controller may vary for this task.

        Derived from the task's own contract where the legal values depend on it -- the
        collection profiles a task offers follow from the event families its config
        declares -- and from the trainer's implemented set otherwise.
        """
        spec = load_task(self.repo, task)
        profiles = tuple(sorted(self.collection_profiles(spec)))
        modes = ("expert", "policy_correction") if spec.correction_supported else ("expert",)
        attempts = 100  # the nominal round budget; the wall clock is the real ceiling
        # Every parameter here is optional in the proposal because the trainer supplies a
        # default for each (`runtime.Trainer` merges the proposal over its own defaults).
        # Declaring them required would overstate the contract: a proposal that names one
        # parameter and leaves the rest alone is executable, and the matched control arms
        # are exactly that shape. They are still listed in the skeleton, so the controller
        # sees them and their defaults.
        params = tuple(
            [Axis(name, "number", "trainer parameter", low=low, high=high, default=default,
                  group="params", optional=True)
             for (name, (low, high)), default in zip(
                 self.TRAINING_BOUNDS.items(), (1e-5, 25, 10.0, 0.1, 32))]
            + [Axis(name, "choice", "implemented choice", values=values,
                    default=preferred, group="params", optional=True)
               for name, (values, preferred) in self.TRAINING_OPTIONS.items()]
        )
        return OptimizationSpace(
            collection=(
                Axis("enabled", "choice", "whether this round collects new data",
                     values=(True, False), default=True),
                Axis("mode", "choice", "how the expert is driven",
                     values=modes, default=modes[0]),
                Axis("profile", "choice",
                     "which distribution the targeted collection draws from; the profiles a "
                     "task offers follow from the event families its config declares",
                     values=profiles, default=profiles[0] if profiles else "full_random"),
                Axis("targeted_attempts", "integer",
                     "attempts spent on the targeted distribution", low=0, high=10_000,
                     default=int(attempts * 0.75)),
                Axis("original_attempts", "integer",
                     "attempts spent on the original distribution", low=0, high=10_000,
                     default=int(attempts * 0.25)),
                Axis("target_episodes", "integer",
                     "episodes the targeted collection aims for", low=0, high=10_000,
                     default=int(attempts * 0.75)),
            ),
            training=(
                Axis("steps", "integer", "candidate training steps", low=1, high=1_000_000,
                     default=20_000),
                *params,
                Axis("targeted_sampling_mass", "number",
                     "share of training samples drawn from the targeted distribution",
                     low=0.001, high=1.0, default=0.5),
                Axis("phase_weights", "structure",
                     "a weight per progress bin, positive, keyed by the three registered "
                     "bin names", default={"early": 0.75, "middle": 1.0, "late": 3.0},
                     validator=self._valid_phase_weights),
                Axis("horizon_floor", "number",
                     "floor on the action-chunk validity weight", low=0.001, high=1.0,
                     default=0.25),
                Axis("phase_bins", "structure",
                     "optional: define your own progress bins instead of weighting the three "
                     "registered ones; each needs name, start, end and weight, and together "
                     "they must cover the episode from 0 to 1",
                     default=None, optional=True, validator=self._valid_phase_bins),
            ),
            # A third section: how many episodes this round's own evidence draws on. It is
            # neither a collection knob nor a training knob, so it is declared as its own
            # section rather than forced into one of the two.
            extra=(("resolution", (
                Axis("development_episodes", "integer",
                     "episodes this round is judged on; raise it when the success and failure "
                     "cohorts overlap too much to tell whether an intervention helped",
                     low=1, high=10_000, default=40),
                Axis("note", "structure",
                     "why this resolution can test the claim you are making",
                     default="", validator=lambda v: isinstance(v, str)),
            )),),
        )

    @staticmethod
    def challenge_context(spec: TaskSpec) -> dict[str, Any]:
        return {
            "instruction": spec.instruction,
            "roles": spec.roles,
            "event_families": spec.event_families,
            "observation_contract": {
                "state_dim": spec.state_dim,
                "cameras": spec.camera_shapes,
            },
            "action_contract": {
                "action_dim": spec.action_dim,
                "control_parts": spec.control_parts,
                "recorded_fps": spec.recorded_fps,
                "max_episode_steps": spec.max_episode_steps,
            },
        }
