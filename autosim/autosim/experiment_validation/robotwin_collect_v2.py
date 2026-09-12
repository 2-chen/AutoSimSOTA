"""RoboTwin collector that skips only the benchmark's declared unstable scene error."""

import argparse
import importlib
from pathlib import Path

from autosim.experiment_system.plugins import RoboTwinPlugin
from autosim.experiment_system.robotwin_worker import native_args
from autosim.research.common import atomic_json, digest


def classify_setup_exception(exc: BaseException, unstable_type: type[BaseException]) -> str | None:
    return "official_scene_unstable" if isinstance(exc, unstable_type) else None


def collect(repo: Path, task_name: str, destination: Path, episodes: int, seed: int):
    from envs.utils.create_actor import UnStableError

    config = native_args(repo, task_name, destination / "raw")
    records, accepted = [], []
    for offset in range(episodes * 12):
        if len(accepted) >= episodes:
            break
        seed_now, index = seed + offset, len(accepted)
        task = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
        row = {"seed": seed_now, "saved": False}
        try:
            plan_args = dict(config, need_plan=True, save_data=False)
            try:
                task.setup_demo(now_ep_num=index, seed=seed_now, **plan_args)
            except BaseException as exc:
                reason = classify_setup_exception(exc, UnStableError)
                if reason is None:
                    raise
                row.update(reason=reason, error=f"{type(exc).__name__}: {exc}")
                continue
            task.play_once()
            if not task.plan_success or not task.check_success():
                row["reason"] = "native_expert_failed"
                continue
            task.save_traj_data(index)
            task.close_env()
            replay_args = dict(plan_args, need_plan=False, save_data=True)
            # A replay-stage exception is not safe to skip because planner cache
            # and partial episode output already exist for this index.
            task.setup_demo(now_ep_num=index, seed=seed_now, **replay_args)
            trajectory = task.load_tran_data(index)
            replay_args.update(left_joint_path=trajectory["left_joint_path"],
                               right_joint_path=trajectory["right_joint_path"])
            task.set_path_lst(replay_args)
            task.play_once()
            replay_success = bool(task.check_success())
            task.close_env()
            if not replay_success:
                row["reason"] = "native_replay_failed"
                raise RuntimeError("replay left partial data; independent attempt and audit required")
            task.merge_pkl_to_hdf5_video()
            path = Path(config["save_path"]) / "data" / f"episode{index}.hdf5"
            if not path.is_file() or not path.stat().st_size:
                raise ValueError("native collector produced no HDF5")
            accepted.append(str(path))
            row.update(saved=True, path=str(path), sha256=digest(path))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            records.append(row)
            atomic_json(destination / "collection.json", {
                "status": "completed" if len(accepted) == episodes else "incomplete",
                "task": task_name, "target_episodes": episodes, "accepted_hdf5": accepted,
                "attempts": records, "maximum_attempts": episodes * 12,
                "native_judge_modified": False, "mock_planner_used": False,
                "retryable_setup_exception": "envs.utils.create_actor.UnStableError_only",
            })
            try:
                task.close_env()
            except Exception:
                pass
    if len(accepted) != episodes:
        raise RuntimeError("bounded native expert budget exhausted")
    return {"status": "completed", "hdf5": accepted,
            "verified_files": {path: digest(Path(path)) for path in accepted},
            "automatic_collection": True, "attempts": len(records),
            "rejected_unstable_scenes": sum(row.get("reason") == "official_scene_unstable" for row in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--task", choices=["open_laptop"], required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    contract = RoboTwinPlugin(args.repo).contract(args.task)
    contract.assert_sources()
    result = collect(args.repo, args.task, args.attempt, args.episodes, args.seed)
    result["contract_signature"] = contract.signature
    atomic_json(args.attempt / "result.json", result)
    contract.assert_sources()


if __name__ == "__main__":
    main()
