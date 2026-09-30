"""Global seed reservations and strict same-protocol comparisons."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from autosim.robosyn_data import evaluation_seed_bank
from .common import object_digest


class SeedLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS banks (task TEXT, name TEXT, purpose TEXT, "
                        "master INTEGER, count INTEGER, hash TEXT, PRIMARY KEY(task,name))")
        self.db.execute("CREATE TABLE IF NOT EXISTS seeds (task TEXT, seed INTEGER, bank TEXT, "
                        "PRIMARY KEY(task,seed))")
        self.db.commit()

    def reserve(self, task: str, name: str, purpose: str, master: int, count: int) -> list[int]:
        if purpose not in {"collection", "smoke", "development", "selection_validation",
                           "final_confirmation", "confirmation", "final", "retired"}:
            raise ValueError("unknown seed bank purpose")
        if count <= 0:
            raise ValueError("seed bank must be nonempty")
        seeds = evaluation_seed_bank(master, count)
        if len(set(seeds)) != count:
            raise ValueError("seed bank contains duplicates; choose a new master before running")
        signature = object_digest(seeds)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            existing = self.db.execute("SELECT purpose,master,count,hash FROM banks WHERE task=? AND name=?",
                                       (task, name)).fetchone()
            if existing:
                if existing != (purpose, master, count, signature):
                    raise ValueError("cannot redefine a reserved seed bank")
                return seeds
            self.db.execute("INSERT INTO banks VALUES (?,?,?,?,?,?)", (task, name, purpose, master, count, signature))
            try:
                self.db.executemany("INSERT INTO seeds VALUES (?,?,?)", ((task, s, name) for s in seeds))
            except sqlite3.IntegrityError as exc:
                raise ValueError("seed bank overlaps known training/evaluation samples") from exc
        return seeds

    def retire(self, task: str, seeds: list[int], origin: str) -> None:
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO seeds VALUES (?,?,?)",
                                ((task, int(s), "retired:" + origin) for s in seeds))

    def close(self):
        self.db.close()


def compare(candidate: dict, baseline: dict) -> dict:
    for data in (candidate, baseline):
        if data.get("execution_mode") != "real_simulation":
            raise ValueError("only real simulation results can enter policy selection")
        rows = data["episodes"]
        seeds = [r["episode_seed"] for r in rows]
        if not rows or len(set(seeds)) != len(rows):
            raise ValueError("missing/duplicate episode samples")
        if len(rows) != data["summary"]["episode_count"]:
            raise ValueError("summary and episode count disagree")
        if sum(bool(r["success"]) for r in rows) != data["summary"]["success_count"]:
            raise ValueError("summary and episode successes disagree")
    for key in ("task", "setting", "timeout_action_steps", "seed"):
        if candidate["config"].get(key) != baseline["config"].get(key):
            raise ValueError(f"incompatible evaluation config: {key}")
    if candidate.get("harness") != baseline.get("harness") or candidate.get("purpose") != baseline.get("purpose"):
        raise ValueError("incompatible harness/purpose")
    a, b = candidate["episodes"], baseline["episodes"]
    if [r["episode_seed"] for r in a] != [r["episode_seed"] for r in b]:
        raise ValueError("paired seed coverage/order must match exactly")
    wins = sum(bool(x["success"]) and not bool(y["success"]) for x, y in zip(a, b))
    losses = sum(bool(y["success"]) and not bool(x["success"]) for x, y in zip(a, b))
    n = wins + losses
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0
    return {"episodes": len(a), "candidate_only_successes": wins, "baseline_only_successes": losses,
            "success_delta": (wins - losses) / len(a), "one_sided_p_value": p,
            "interpretation": "same-seed comparison; simulator determinism must be checked separately"}


def holm(p_values: dict[str, float], alpha=0.05) -> dict[str, bool]:
    result, rejected = {}, True
    for rank, (name, p) in enumerate(sorted(p_values.items(), key=lambda item: item[1])):
        rejected = rejected and p <= alpha / (len(p_values) - rank)
        result[name] = rejected
    return result
