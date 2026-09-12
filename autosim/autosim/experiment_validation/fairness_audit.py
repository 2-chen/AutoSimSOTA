"""Independent evidence audit for equal-budget AutoResearch comparisons."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from autosim.research.common import atomic_json, digest, freeze_files, now, read_json
from autosim.research.ledger import compare
from .queue_manifest import versioned_manifest


TASKS = ("click_bell", "water_pouring", "drawer_open_place")


def command_value(process: dict, flag: str):
    command = process.get("command", [])
    matches = [index for index, value in enumerate(command) if value == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise ValueError(f"missing or repeated command flag: {flag}")
    return command[matches[0] + 1]


def command_flag(process: dict, flag: str) -> bool:
    """Require a boolean command flag to be absent or present exactly once."""
    count = process.get("command", []).count(flag)
    if count > 1:
        raise ValueError(f"repeated command flag: {flag}")
    return count == 1


def _metric(path: Path) -> dict:
    return read_json(path / "evaluation_metrics.json")


def _train_process(arm: Path, steps: int) -> dict:
    return read_json(arm / f"process_train_{steps}/process.json")


def collection_seed_set(record: dict) -> tuple[set[int], bool]:
    values = [int(value) for value in record.get("successful_episode_seeds", [])]
    return set(values), len(values) == len(set(values))


def selected_round(rows: list[dict], arm: str) -> dict:
    if arm not in {"auto", "random"}:
        raise ValueError(f"unknown arm: {arm}")
    score_key = f"{arm}_selection_score"
    summary_key = "summary" if arm == "auto" else "random_summary"
    return max(rows, key=lambda row: (float(row[score_key]),
                                      -float(row[summary_key]["average_action_steps"])))


def audit_task(research_root: Path, task: str) -> dict:
    config = read_json(research_root / "research_config.json")
    state_path = research_root / task / "state.json"
    if not state_path.is_file():
        return {"task": task, "status": "pending", "reason": "research state missing", "checks": []}
    state = read_json(state_path)
    checks, evidence = [], {str(state_path.absolute()): digest(state_path)}
    auto_new_episodes = random_new_episodes = 0
    auto_screen_updates = random_screen_updates = 0
    auto_collection_attempts = random_collection_attempts = 0
    auto_collection_seconds = random_collection_seconds = 0.0
    collection_seed_groups: list[tuple[str, set[int]]] = []
    development_seed_banks: dict[int, set[int]] = {}

    def check(name: str, passed: bool, **detail):
        checks.append({"name": name, "passed": bool(passed), **detail})

    for row in state.get("rounds", []):
        index = int(row["index"]) + 1
        auto_data = Path(row["auto_data"]) if row.get("auto_data") else None
        random_data = Path(row["random_data"]) if row.get("random_data") else None
        if auto_data and random_data:
            auto_info_path, random_info_path = auto_data / "meta/info.json", random_data / "meta/info.json"
            auto_info, random_info = read_json(auto_info_path), read_json(random_info_path)
            auto_collection_path = auto_data.parents[1] / "collection.json"
            random_collection_path = random_data.parents[1] / "collection.json"
            auto_collection, random_collection = read_json(auto_collection_path), read_json(random_collection_path)
            auto_collection_process_path = auto_data.parents[1] / "process/process.json"
            random_collection_process_path = random_data.parents[1] / "process/process.json"
            auto_collection_process = read_json(auto_collection_process_path)
            random_collection_process = read_json(random_collection_process_path)
            evidence.update({str(auto_info_path.absolute()): digest(auto_info_path),
                             str(random_info_path.absolute()): digest(random_info_path),
                             str(auto_collection_path.absolute()): digest(auto_collection_path),
                             str(random_collection_path.absolute()): digest(random_collection_path),
                             str(auto_collection_process_path.absolute()): digest(auto_collection_process_path),
                             str(random_collection_process_path.absolute()): digest(random_collection_process_path)})
            check(f"round_{index}:new_successful_episode_budget",
                  auto_info["total_episodes"] == random_info["total_episodes"] == config["collect_per_round"],
                  auto=auto_info["total_episodes"], random=random_info["total_episodes"],
                  expected=config["collect_per_round"])
            auto_seeds, auto_unique = collection_seed_set(auto_collection)
            random_seeds, random_unique = collection_seed_set(random_collection)
            expected_auto_profile = row["proposal"]["collection_profile"]
            check(f"round_{index}:collection_receipts",
                  auto_collection.get("status") == random_collection.get("status") == "completed" and
                  auto_collection.get("target_successful_episodes") ==
                  random_collection.get("target_successful_episodes") == config["collect_per_round"] and
                  len(auto_seeds) == auto_info["total_episodes"] and
                  len(random_seeds) == random_info["total_episodes"] and
                  auto_collection.get("profile") == expected_auto_profile and
                  random_collection.get("profile") == "full_random",
                  auto_status=auto_collection.get("status"), random_status=random_collection.get("status"),
                  auto_profile=auto_collection.get("profile"), random_profile=random_collection.get("profile"),
                  auto_success_seeds=len(auto_seeds), random_success_seeds=len(random_seeds))
            check(f"round_{index}:collection_seed_disjointness",
                  auto_unique and random_unique and not (auto_seeds & random_seeds),
                  auto_unique=auto_unique, random_unique=random_unique,
                  cross_arm_overlap=len(auto_seeds & random_seeds))
            expected_attempt_cap = config["collect_per_round"] * 12
            auto_cap = int(command_value(auto_collection_process, "--collection_max_attempts"))
            random_cap = int(command_value(random_collection_process, "--collection_max_attempts"))
            check(f"round_{index}:collection_ex_ante_attempt_cap",
                  auto_cap == random_cap == expected_attempt_cap,
                  auto=auto_cap, random=random_cap, expected=expected_attempt_cap)
            check(f"round_{index}:collection_process_completion",
                  auto_collection_process.get("status") == random_collection_process.get("status") == "completed" and
                  auto_collection_process.get("returncode") == random_collection_process.get("returncode") == 0,
                  auto_status=auto_collection_process.get("status"),
                  random_status=random_collection_process.get("status"),
                  auto_returncode=auto_collection_process.get("returncode"),
                  random_returncode=random_collection_process.get("returncode"))
            auto_master = int(command_value(auto_collection_process, "--collection_seed"))
            random_master = int(command_value(random_collection_process, "--collection_seed"))
            check(f"round_{index}:collection_master_seed_lineage",
                  auto_master == int(auto_collection.get("master_seed", -1)) and
                  random_master == int(random_collection.get("master_seed", -1)) and
                  auto_master != random_master,
                  auto_command=auto_master, auto_manifest=auto_collection.get("master_seed"),
                  random_command=random_master, random_manifest=random_collection.get("master_seed"))
            auto_attempts = int(auto_collection.get("expert_attempt_count", -1))
            random_attempts = int(random_collection.get("expert_attempt_count", -1))
            check(f"round_{index}:collection_attempt_lineage",
                  auto_attempts == len(auto_collection.get("attempts", [])) and
                  random_attempts == len(random_collection.get("attempts", [])) and
                  config["collect_per_round"] <= auto_attempts <= auto_cap and
                  config["collect_per_round"] <= random_attempts <= random_cap,
                  auto_attempts=auto_attempts, random_attempts=random_attempts)
            auto_collection_attempts += auto_attempts
            random_collection_attempts += random_attempts
            auto_collection_seconds += float(auto_collection_process.get("elapsed_seconds", 0.0))
            random_collection_seconds += float(random_collection_process.get("elapsed_seconds", 0.0))
            collection_seed_groups.extend(((f"round_{index}:auto", auto_seeds),
                                           (f"round_{index}:random", random_seeds)))
            auto_new_episodes += int(auto_info["total_episodes"])
            random_new_episodes += int(random_info["total_episodes"])
        else:
            check(f"round_{index}:collection_disabled_symmetrically", auto_data is None and random_data is None,
                  auto=str(auto_data) if auto_data else None, random=str(random_data) if random_data else None)

        auto_mix_path, random_mix_path = Path(row["auto_mixture"]), Path(row["random_mixture"])
        auto_mix, random_mix = read_json(auto_mix_path), read_json(random_mix_path)
        evidence.update({str(auto_mix_path.absolute()): digest(auto_mix_path),
                         str(random_mix_path.absolute()): digest(random_mix_path)})
        check(f"round_{index}:cumulative_episode_budget",
              auto_mix["total_episodes"] == random_mix["total_episodes"],
              auto=auto_mix["total_episodes"], random=random_mix["total_episodes"])

        auto_arm = Path(row["checkpoint"]).parents[3]
        random_arm = Path(row["random_checkpoint"]).parents[3]
        auto_process = _train_process(auto_arm, config["screen_steps"])
        random_process = _train_process(random_arm, config["screen_steps"])
        auto_steps = int(command_value(auto_process, "--steps"))
        random_steps = int(command_value(random_process, "--steps"))
        auto_screen_updates += auto_steps
        random_screen_updates += random_steps
        for arm, process in (("auto", auto_process), ("random", random_process)):
            process_path = (auto_arm if arm == "auto" else random_arm) / f"process_train_{config['screen_steps']}/process.json"
            evidence[str(process_path.absolute())] = digest(process_path)
        check(f"round_{index}:screening_update_budget",
              auto_steps == random_steps == config["screen_steps"],
              auto=auto_steps, random=random_steps,
              expected=config["screen_steps"])
        check(f"round_{index}:training_seed",
              int(command_value(auto_process, "--seed")) == int(command_value(random_process, "--seed")) == config["train_seed"],
              auto=int(command_value(auto_process, "--seed")), random=int(command_value(random_process, "--seed")))

        for bank, auto_key, random_key in ((1, "evaluation", "random_evaluation"),
                                           (2, "auto_second_bank", "random_second_bank")):
            if bank == 2 and (auto_key not in row or random_key not in row):
                if state.get("status") == "completed_development":
                    check(f"round_{index}:development_bank_2_pairing", False, reason="missing second bank")
                continue
            auto_eval = Path(row[auto_key] if bank == 1 else row[auto_key]["evaluation"])
            random_eval = Path(row[random_key] if bank == 1 else row[random_key]["evaluation"])
            auto_metrics, random_metrics = _metric(auto_eval), _metric(random_eval)
            paired = compare(auto_metrics, random_metrics)
            check(f"round_{index}:development_bank_{bank}_pairing", paired["episodes"] == config["development_episodes"],
                  episodes=paired["episodes"], expected=config["development_episodes"])
            seeds = {int(item["episode_seed"]) for item in auto_metrics["episodes"]}
            if bank in development_seed_banks:
                check(f"round_{index}:development_bank_{bank}_suite_consistency",
                      seeds == development_seed_banks[bank], episodes=len(seeds))
            else:
                development_seed_banks[bank] = seeds
            for path in (auto_eval / "evaluation_metrics.json", random_eval / "evaluation_metrics.json"):
                evidence[str(path.absolute())] = digest(path)
        if "auto_second_bank" in row and "random_second_bank" in row:
            auto_first = _metric(Path(row["evaluation"]))["summary"]["success_rate"]
            auto_second = _metric(Path(row["auto_second_bank"]["evaluation"]))["summary"]["success_rate"]
            random_first = _metric(Path(row["random_evaluation"]))["summary"]["success_rate"]
            random_second = _metric(Path(row["random_second_bank"]["evaluation"]))["summary"]["success_rate"]
            expected_auto_score = (float(auto_first) + float(auto_second)) / 2
            expected_random_score = (float(random_first) + float(random_second)) / 2
            check(f"round_{index}:selection_score_recomputation",
                  float(row["auto_selection_score"]) == expected_auto_score and
                  float(row["random_selection_score"]) == expected_random_score,
                  auto_stored=row["auto_selection_score"], auto_recomputed=expected_auto_score,
                  random_stored=row["random_selection_score"], random_recomputed=expected_random_score)

    completed_rounds = len(state.get("rounds", []))
    if completed_rounds:
        check("total_acquisition_budget",
              auto_new_episodes == random_new_episodes == completed_rounds * config["collect_per_round"],
              auto=auto_new_episodes, random=random_new_episodes,
              expected=completed_rounds * config["collect_per_round"])
        check("total_screening_update_budget",
              auto_screen_updates == random_screen_updates == completed_rounds * config["screen_steps"],
              auto=auto_screen_updates, random=random_screen_updates,
              expected=completed_rounds * config["screen_steps"])
        total_seed_count = sum(len(seeds) for _, seeds in collection_seed_groups)
        union_seed_count = len(set().union(*(seeds for _, seeds in collection_seed_groups))) if collection_seed_groups else 0
        check("total_collection_seed_disjointness",
              bool(collection_seed_groups) and union_seed_count == total_seed_count,
              groups=[{"name": name, "successful_seeds": len(seeds)} for name, seeds in collection_seed_groups],
              total_seed_count=total_seed_count, union_seed_count=union_seed_count,
              overlap_count=total_seed_count - union_seed_count)
        bank_1, bank_2 = development_seed_banks.get(1, set()), development_seed_banks.get(2, set())
        check("development_bank_disjointness",
              len(bank_1) == len(bank_2) == config["development_episodes"] and not (bank_1 & bank_2),
              bank_1_unique=len(bank_1), bank_2_unique=len(bank_2), overlap=len(bank_1 & bank_2),
              expected_per_bank=config["development_episodes"])

    selected_data = None
    if state.get("status") == "completed_development":
        selected = state["selected"]
        auto_arm = Path(selected["auto"]["checkpoint"]).parents[3]
        random_arm = Path(selected["random"]["checkpoint"]).parents[3]
        auto_process = _train_process(auto_arm, config["final_steps"])
        random_process = _train_process(random_arm, config["final_steps"])
        check("selected_long_training_update_budget",
              int(command_value(auto_process, "--steps")) == int(command_value(random_process, "--steps")) == config["final_steps"],
              auto=int(command_value(auto_process, "--steps")), random=int(command_value(random_process, "--steps")),
              expected=config["final_steps"])
        check("selected_long_training_process_completion",
              auto_process.get("status") == random_process.get("status") == "completed" and
              auto_process.get("returncode") == random_process.get("returncode") == 0,
              auto_status=auto_process.get("status"), auto_returncode=auto_process.get("returncode"),
              random_status=random_process.get("status"), random_returncode=random_process.get("returncode"))
        check("selected_long_training_seed",
              int(command_value(auto_process, "--seed")) ==
              int(command_value(random_process, "--seed")) == config["train_seed"],
              auto=int(command_value(auto_process, "--seed")),
              random=int(command_value(random_process, "--seed")), expected=config["train_seed"])
        expected_resume = config["final_steps"] > config["screen_steps"]
        auto_resume, random_resume = command_flag(auto_process, "--resume"), command_flag(random_process, "--resume")
        check("selected_long_training_resume_mode",
              auto_resume == random_resume == expected_resume,
              auto=auto_resume, random=random_resume, expected=expected_resume)
        auto_round = next(row for row in state["rounds"]
                          if Path(row["checkpoint"]).parents[3] == auto_arm)
        random_round = next(row for row in state["rounds"]
                            if Path(row["random_checkpoint"]).parents[3] == random_arm)
        expected_auto_round, expected_random_round = (selected_round(state["rounds"], arm)
                                                       for arm in ("auto", "random"))
        check("selected_round_matches_development_ranking",
              auto_round["index"] == expected_auto_round["index"] and
              random_round["index"] == expected_random_round["index"],
              auto_selected=int(auto_round["index"]) + 1,
              auto_expected=int(expected_auto_round["index"]) + 1,
              random_selected=int(random_round["index"]) + 1,
              random_expected=int(expected_random_round["index"]) + 1)
        auto_mix_path, random_mix_path = Path(auto_round["auto_mixture"]), Path(random_round["random_mixture"])
        auto_selected_episodes = int(read_json(auto_mix_path)["total_episodes"])
        random_selected_episodes = int(read_json(random_mix_path)["total_episodes"])
        selected_data = {"auto_round": int(auto_round["index"]) + 1,
                         "random_round": int(random_round["index"]) + 1,
                         "auto_episodes": auto_selected_episodes,
                         "random_episodes": random_selected_episodes,
                         "identical_selected_training_episode_count": auto_selected_episodes == random_selected_episodes,
                         "interpretation": "independent development selection under equal total research budgets"}
        evidence.update({str(auto_mix_path.absolute()): digest(auto_mix_path),
                         str(random_mix_path.absolute()): digest(random_mix_path)})
        auto_eval, random_eval = Path(selected["auto"]["evaluation"]), Path(selected["random"]["evaluation"])
        paired = compare(_metric(auto_eval), _metric(random_eval))
        check("held_out_confirmation_pairing", paired["episodes"] == config["confirmation_episodes"],
              episodes=paired["episodes"], expected=config["confirmation_episodes"])
        for path in (auto_arm / f"process_train_{config['final_steps']}/process.json",
                     random_arm / f"process_train_{config['final_steps']}/process.json",
                     auto_eval / "evaluation_metrics.json", random_eval / "evaluation_metrics.json"):
            evidence[str(path.absolute())] = digest(path)

    failed = [row for row in checks if not row["passed"]]
    complete = state.get("status") == "completed_development"
    return {"task": task, "status": "failed" if failed else ("passed" if complete else "partial"),
            "research_status": state.get("status"), "completed_rounds": completed_rounds,
            "checks": checks, "failed_checks": len(failed), "evidence": evidence,
            "selected_training_data": selected_data,
            "acquisition_cost": {
                "auto_actual_attempts": auto_collection_attempts,
                "random_actual_attempts": random_collection_attempts,
                "actual_attempts_equal": auto_collection_attempts == random_collection_attempts,
                "auto_elapsed_seconds": auto_collection_seconds,
                "random_elapsed_seconds": random_collection_seconds,
                "budget_definition": "equal successful-episode targets and equal predeclared maximum attempt caps; actual early-stopped attempts/time are reported, not assumed equal",
            },
            "equal_total_research_budget_established": complete and not failed,
            "equal_budget_established": complete and not failed}


def write_audit(root: Path, research_root: Path) -> dict:
    rows = []
    for task in TASKS:
        try:
            rows.append(audit_task(research_root, task))
        except Exception as exc:
            rows.append({"task": task, "status": "failed", "failed_checks": 1,
                         "equal_budget_established": False,
                         "error": f"{type(exc).__name__}: {exc}"})
    complete = all(row.get("equal_total_research_budget_established") for row in rows)
    value = {"updated_at": now(), "kind": "autoresearch_equal_budget_evidence_audit",
             "task_count": len(TASKS), "tasks": rows, "all_three_tasks_equal_budget": complete,
             "policy_effect_claim": False, "official_leaderboard_result": False, "sota_claim": False}
    atomic_json(root / "fairness_audit.json", value)
    lines = ["# AutoResearch 三任务预算公平性审计", "",
             "本表审计实际数据、训练命令和评测文件；它不依据配置声明推断公平性。", "",
             "| 任务 | 研究状态 | 完成轮数 | 实际检查 | 失败检查 | 预注册预算公平 | 实际采集 attempts | 最终入模 episodes |",
             "| --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['task']} | {row.get('research_status', row['status'])} | "
                     f"{row.get('completed_rounds', 0)} | {len(row.get('checks', []))} | "
                     f"{row.get('failed_checks', 0)} | "
                     f"{'是' if row.get('equal_total_research_budget_established') else '否/待完成'} | "
                     f"{('自动 ' + str(row['acquisition_cost']['auto_actual_attempts']) + ' / 随机 ' + str(row['acquisition_cost']['random_actual_attempts'])) if row.get('acquisition_cost') else '待采集'} | "
                     f"{('自动 ' + str(row['selected_training_data']['auto_episodes']) + ' / 随机 ' + str(row['selected_training_data']['random_episodes'])) if row.get('selected_training_data') else '待选择'} |")
    lines += ["", f"三任务实际等预算证据完成：{'是' if complete else '否'}。",
              "", "注意：这里的采集公平性精确指每轮相同的成功轨迹目标和预注册最大attempt上限；由于达到成功目标后必须停止，实际attempt数和耗时可不同，表中单独报告。两组的筛选更新、长训练更新和确认评测上限相同。两组可独立选择不同轮次，因此最终入模 episode 数也单独报告。这些公平性证据不等于自动组更有效；性能结论由独立的100局同种子确认结果决定。"]
    (root / "fairness_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def queue(root: Path, research_root: Path, comparison_state: Path, max_hours: float, poll_seconds: float):
    if not 0 < max_hours <= 336 or not 1 <= poll_seconds <= 60:
        raise ValueError("invalid bounded queue settings")
    local = Path(__file__).resolve()
    versioned_manifest(root, {"tasks": list(TASKS), "comparison_state": str(comparison_state),
                              "sources": freeze_files([local, local.with_name("queue_manifest.py")]),
                              "policy_effect_claim": False, "sota_claim": False})
    state_path = root / "fairness_state.json"
    prior = read_json(state_path) if state_path.is_file() else {}
    if prior.get("status") in {"completed", "requires_review", "queue_wait_budget_exhausted"}:
        return prior
    deadline = float(prior.get("deadline_epoch", time.time() + max_hours * 3600))
    state = {**prior, "status": "running", "started_at": prior.get("started_at", now()),
             "deadline_epoch": deadline}
    if prior:
        state.update(resume_count=int(prior.get("resume_count", 0)) + 1, resumed_at=now())
    while time.time() < deadline:
        comparison = read_json(comparison_state) if comparison_state.is_file() else {}
        write_audit(root, research_root)
        if comparison.get("status") == "three_task_comparison_complete":
            result = write_audit(root, research_root)
            state.update(status="completed" if result["all_three_tasks_equal_budget"] else "requires_review",
                         all_three_tasks_equal_budget=result["all_three_tasks_equal_budget"], updated_at=now())
            atomic_json(state_path, state)
            return state
        if comparison.get("status") in {"requires_review", "queue_wait_budget_exhausted"}:
            state.update(status="requires_review", upstream_status=comparison.get("status"), updated_at=now())
            atomic_json(state_path, state)
            return state
        state.update(stage="waiting_three_task_comparison", upstream_status=comparison.get("status", "missing"),
                     updated_at=now())
        atomic_json(state_path, state)
        time.sleep(poll_seconds)
    state.update(status="queue_wait_budget_exhausted", updated_at=now())
    atomic_json(state_path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "queue"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--comparison-state", type=Path)
    parser.add_argument("--max-hours", type=float, default=240)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.command == "audit":
        write_audit(args.root.absolute(), args.research_root.absolute())
    else:
        if args.comparison_state is None:
            parser.error("queue requires --comparison-state")
        queue(args.root.absolute(), args.research_root.absolute(), args.comparison_state.absolute(),
              args.max_hours, args.poll_seconds)
