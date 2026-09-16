"""Explicit production task contracts, derived from pinned benchmark configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .common import digest, object_digest, read_json


TASK_IDS = {
    "click_bell": "ClickBell", "table_rearrangement": "TableRearrangement",
    "water_pouring": "WaterPouring", "handle_basket": "HandleBasket",
    "items_handover": "ItemsHandover", "drawer_open_place": "DrawerOpenPlace",
    "mixer_operating": "MixerOperating", "item_assembly": "ItemAssembly",
    "manipulate_pipette": "ManipulatePipette", "sample_loading": "SampleLoading",
}

# Task-specific semantics stay here, not in the experiment controller.
TASK_ROLES = {
    "click_bell": {"articulations": ["button"], "objects": [], "family": "press"},
    "table_rearrangement": {"objects": ["fork", "spoon", "plate"], "family": "place_multiple"},
    "water_pouring": {"objects": ["bottle", "cup"], "family": "pour"},
    "handle_basket": {"objects": ["milk", "basket"], "family": "transport"},
    "items_handover": {"objects": ["pen", "holder"], "family": "handover"},
    "drawer_open_place": {"objects": ["duck"], "articulations": ["drawer"], "family": "articulated_place"},
    "mixer_operating": {"objects": ["beaker", "beaker_mixer"], "family": "place_then_contact"},
    "item_assembly": {"objects": ["guijiao1", "guijiao2"], "family": "align"},
    "manipulate_pipette": {"objects": ["beaker1"], "articulations": ["pipette"], "family": "tool_contact"},
    "sample_loading": {"objects": ["cube", "rack"], "family": "precise_place"},
}

#: Tasks whose benchmark expert entrypoint needs the training-only ActionBank binding.
#: A discovered capability of the upstream repo, not a policy choice.
EXPERT_ADAPTERS = {"handle_basket": "action_bank"}

#: Tasks with a validated policy-correction adapter. Everything else collects with the
#: official expert only; the controller is told which, rather than being handed a menu.
CORRECTION_CAPABLE = {"click_bell"}

EVENT_FAMILIES = {
    "randomize_light": "appearance", "randomize_visual_material": "appearance",
    "randomize_camera_intrinsics": "camera", "randomize_camera_extrinsics": "camera",
    "randomize_robot_eef_pose": "robot_pose", "randomize_robot_qpos": "robot_pose",
    "replace_distractor_slots_from_library": "clutter",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_id: str
    setting: str
    max_episode_steps: int
    state_dim: int
    action_dim: int
    cameras: tuple[str, ...]
    camera_shapes: dict[str, list[int]]
    control_parts: tuple[str, ...]
    recorded_fps: float
    instruction: str
    gym_config: str
    action_config: str
    config_hashes: dict[str, str]
    event_families: dict[str, list[str]]
    roles: dict[str, Any]
    correction_supported: bool = False
    expert_adapter: str = "official"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def signature(self) -> str:
        return object_digest(self.as_dict())


def _declared_uids(gym: dict, key: str) -> list[str]:
    """Entity uids the task's own config declares under `key`."""
    value = gym.get(key)
    if not isinstance(value, list):
        return []
    return [str(row["uid"]) for row in value
            if isinstance(row, dict) and row.get("uid") is not None]


def derive_roles(gym: dict, task: str) -> dict[str, Any]:
    """Read the task's entities out of its config instead of a hand-kept table.

    The config already names every object and articulation, and the telemetry layer
    reports them under exactly these names, so a table here was duplicated metadata
    that had to be hand-edited before a new task could run at all.
    """
    objects = [uid for uid in _declared_uids(gym, "rigid_object")
               if not uid.startswith("distractor")]
    articulations = _declared_uids(gym, "articulation")
    roles: dict[str, Any] = {"objects": sorted(objects), "articulations": sorted(articulations)}
    # `family` is a semantic label with no config equivalent. The table is a preset, not a
    # gate: an unlisted task gets an explicit "unclassified" rather than a missing key, so
    # telemetry stays the same shape and nothing downstream has to special-case it.
    preset = TASK_ROLES.get(task) or {}
    roles["family"] = preset.get("family", "unclassified")
    return roles


def load_task(repo: Path, task: str, setting: str = "random") -> TaskSpec:
    # No task allow-list: any task whose config exists is loadable. The config's own
    # `id` is the environment id, so a separate table could only disagree with it.
    if setting != "random":
        raise ValueError("ranking suite only supports the frozen random setting")
    gym_path = repo / "configs" / task / setting / "gym_config.json"
    action_path = repo / "configs" / task / "action_config.json"
    if not gym_path.is_file():
        raise ValueError(f"no {setting} gym config for task: {task}")
    gym, action = read_json(gym_path), read_json(action_path)
    if "action_config" in action:
        action = action["action_config"]
    parts = tuple(gym["env"]["control_parts"])
    dimension = sum(int(action["scope"][part]["dim"][0]) for part in parts)
    cameras = {s["uid"]: [int(s["height"]), int(s["width"]), 3]
               for s in gym["sensor"] if s.get("sensor_type") == "Camera"}
    if not cameras or dimension <= 0:
        raise ValueError(f"invalid observation/action contract for {task}")
    recorder = next(iter(gym["env"]["dataset"].values()))["params"]
    families: dict[str, list[str]] = {}
    for name, event in gym["env"].get("events", {}).items():
        family = EVENT_FAMILIES.get(event.get("func"), "task_semantics")
        families.setdefault(family, []).append(name)
    return TaskSpec(
        name=task, env_id=gym["id"], setting=setting,
        max_episode_steps=int(gym["max_episode_steps"]), state_dim=dimension,
        action_dim=dimension, cameras=tuple(cameras), camera_shapes=cameras,
        control_parts=parts, recorded_fps=float(recorder["robot_meta"]["control_freq"]),
        instruction=recorder["instruction"]["lang"], gym_config=str(gym_path.absolute()),
        action_config=str(action_path.absolute()), config_hashes={
            str(gym_path.absolute()): digest(gym_path), str(action_path.absolute()): digest(action_path)},
        event_families=families, roles=derive_roles(gym, task),
        # These two describe capabilities of the benchmark's own adapters, not preferences
        # the controller picks from, so they stay table-driven -- but as defaults, not gates:
        # an unlisted task gets the standard answer and still runs.
        correction_supported=task in CORRECTION_CAPABLE,
        expert_adapter=EXPERT_ADAPTERS.get(task, "official"),
    )


def filter_training_events(events: dict, keep_families: set[str]) -> dict:
    """Never remove unknown/semantic events, including constraints and placements."""
    unknown = keep_families - set(EVENT_FAMILIES.values())
    if unknown:
        raise ValueError(f"unknown randomization families: {unknown}")
    return {name: value for name, value in events.items()
            if EVENT_FAMILIES.get(value.get("func")) in keep_families
            or value.get("func") not in EVENT_FAMILIES}


def inventory(repo: Path) -> list[dict[str, Any]]:
    result = []
    for task in TASK_IDS:
        try:
            spec = load_task(repo, task)
            checkpoint = repo / "checkpoints" / f"ACT_sim_{task}"
            dataset = repo / "lerobot_dataset/RoboSynChallenge" / f"cobotmagic_Sim_{task}"
            dataset_available = False
            info_path = dataset / "meta/info.json"
            if info_path.is_file():
                info = read_json(info_path)
                episodes = int(info.get("total_episodes", 0))
                videos = sum(value.get("dtype") == "video"
                             for value in info.get("features", {}).values())
                dataset_available = (
                    info.get("codebase_version") == "v2.1"
                    and len(list((dataset / "data").glob("**/*.parquet"))) == episodes
                    and len(list((dataset / "videos").glob("**/*.mp4"))) == episodes * videos)
            result.append({"task": task, "contract": spec.as_dict(), "signature": spec.signature,
                           "contract_status": "passed", "pipeline_status": "not_run",
                           "improvement_status": "not_evaluated", "official_checkpoint": str(checkpoint),
                           "checkpoint_available": (checkpoint / "model.safetensors").is_file(),
                           "official_dataset": str(dataset),
                           "dataset_available": dataset_available})
        except Exception as exc:
            result.append({"task": task, "contract_status": "failed", "error": str(exc),
                           "pipeline_status": "blocked", "improvement_status": "not_evaluated"})
    return result
