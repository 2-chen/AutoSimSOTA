"""Does a cheap screening run rank candidates the way an expensive one does?

Every method that screens at a small budget and promotes the survivors rests on one
assumption: that the ranking at the small budget predicts the ranking at the large one. If it
does not, promotion is a coin toss that discards good candidates and keeps bad ones -- worse
than not screening, because the screening is paid for and the idea is lost.

The assumption is checkable in an afternoon and nobody had checked it here. This runs the same
candidates at two budgets -- one promotion step apart, which is what a ladder would actually
do -- and reports whether the orders agree, and on which quantity.

Usage: `.venv/bin/python tools/check_screen_correlation.py [--dry-run] [--parallel 2]`

Writes `screen_correlation.json` next to the run's research directory. Each run is a full
LIBERO-10 lifelong measurement, so this is hours, not minutes: the cheap rung is about an hour
per candidate and the expensive one about five.
"""

import json
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autosim"))

from autosim.research import execution_derive, provision                  # noqa: E402
from autosim.research.compute_decision import decide                      # noqa: E402
from autosim.research.declarative_backend import DeclarativeBackend       # noqa: E402
from autosim.research.derived_research import DerivedResearch             # noqa: E402
from autosim.research.execution_derive import blank_inputs                # noqa: E402

REPO = Path("/home/wbc/下载/autoresearch/test/LIBERO")
OUTPUT = ROOT / "autoresearch_runs/provisioning/libero"

#: The candidates, as a 2x2: two algorithms at two seeds.
#:
#: **Overridden by group, not by field.** The declared axis is `lifelong.algo` and the
#: declared values are the algorithm names, and setting the field to one of them produces a
#: configuration the algorithm cannot use: in LIBERO each algorithm's own hyperparameters live
#: in its own group file (`er.yaml` carries `n_memories`, `ewc.yaml` carries `e_lambda`,
#: `packnet.yaml` carries `prune_perc`), and `base.yaml` carries none of them. So
#: `lifelong.algo=ER` on top of `lifelong=base` leaves `n_memories` undefined and the run dies
#: in `er.py` with `AttributeError: 'EasyDict' object has no attribute 'n_memories'`. The
#: thing to override is the group: `lifelong=er`.
#:
#: This is the third defect of one kind in that space. `policy.policy_type` offers *file*
#: names where the field holds a *class* name; `lifelong.algo` offers *field* values where the
#: thing that works is the *group*; and two of the five algorithms cannot complete a run at
#: all on this checkout. A declaration is a claim, and these are the parts of it that running
#: it contradicted.
CANDIDATES = [{"lifelong": group, "seed": seed}
              for group in ("base", "er") for seed in (10000, 10001)]

#: A candidate is screened before it is scheduled: one epoch, and the question is only whether
#: task 0 *and its post-task evaluation* completed -- that evaluation is where EWC and PackNet
#: died, so a candidate that gets past it is one the remaining runs can be planned around.
#:
#: Thirty-five minutes, and a criterion that is not the clock. The first version used fifteen
#: minutes and "did task 1's checkpoint appear", and the work takes about fifteen minutes: it
#: passed by seconds on one run and failed on the next, which is a judgement made by luck. The
#: signal is now positive evidence that the evaluation finished -- LIBERO prints `[Task  0
#: succ.]` after it -- with more than twice the margin on the clock.
SMOKE_SECONDS = 35 * 60

#: The signal that separates a working candidate from a broken one, checked against the four
#: real runs before it was trusted. It is *task 1*, not task 0 -- and an earlier version of
#: this file had it the other way round.
#:
#: `[Task  0 succ.]` looks like the right evidence, since that is where the evaluation path
#: runs. It is a false positive for exactly the failure being screened for: ER builds its
#: memory buffer in `start_task`, the buffer is empty for task 0 so the code that reads
#: `n_memories` never executes, and the run dies on task *1*. The broken ER run printed
#: `[Task  0 succ.]` at line 412 and raised `AttributeError: 'EasyDict' object has no
#: attribute 'n_memories'` at line 424. A candidate has to get past the task that follows.
SMOKE_EVIDENCE = re.compile(r"task1_model")

#: A wall clock per run, from measured cost: a 5-epoch lifelong run takes about two hours
#: (four minutes of training per task, seven of evaluation), so twenty epochs is about eight.
#: With room above that, because a timeout that arrives before the work is done is the same as
#: no timeout at all. A timeout is a failure that can be seen; silence is not.
RUN_TIMEOUT = {5: 5 * 3600, 20: 14 * 3600}

#: Serial. Two concurrent LIBERO lifelong runs deadlock: each evaluation starts twenty
#: subprocesses, and forty of them contending is where the hangs came from -- a run that had
#: finished task 0 in eleven minutes sat for four hours with nothing more written, and an
#: earlier one sat for ten. Concurrency was the reason the first two attempts produced nothing,
#: and it is not worth the wall clock it saves.
PARALLEL = 1

#: The two rungs, one promotion step apart with η = 4 -- the step a ladder would actually
#: take. Testing the full 50-epoch protocol would cost four times as much to answer a question
#: about the step.
CHEAP, EXPENSIVE = 5, 20

#: Everything held fixed across the candidates. `policy` is absent deliberately: the
#: generated command already carries `policy=bc_transformer_policy` as a settled literal, and
#: the declared axis named `policy.policy_type` holds *file* names where the config field
#: wants *class* names -- setting it to any of its declared values produces a command that
#: dies in `get_policy_class`. That is a defect in the declaration, not in this experiment;
#: it is recorded rather than worked around silently.
BASE = {"benchmark_name": "LIBERO_10", "eval.n_eval": 20,
        "train.num_workers": 0, "eval.num_workers": 0}


def build(decision) -> tuple[DerivedResearch, dict]:
    interpreter = provision.env_python(OUTPUT)
    if interpreter is None:
        raise SystemExit("no successful build here; run provisioning first")
    execution = json.loads((OUTPUT / "execution.json").read_text(encoding="utf-8"))
    kept = json.loads((OUTPUT / "derived_stages.json").read_text(encoding="utf-8"))
    stages = {stage: row["source"] for stage, row in kept.items()}
    parameters = {stage: row["parameters"] for stage, row in kept.items()}
    for stage, row in kept.items():
        execution["stages"][stage] = row["row"]
    backend = DeclarativeBackend(repo=REPO, answer=execution, sources=stages,
                                 parameters=parameters)
    research = DerivedResearch(repo=REPO, output=OUTPUT, backend=backend,
                               interpreter=interpreter, space=None, client=None,
                               stages=stages, run_id="screen", compute=decision,
                               benchmark="LIBERO")
    return research, stages


def name_of(candidate: dict) -> str:
    """A candidate's name, from its own settings rather than a key this file remembers.

    The candidates are spelled differently once the axis turns out to be a group rather than a
    field -- `lifelong` instead of `lifelong.algo` -- and every place that reached for the old
    key became a `KeyError` in turn. The candidate is the authority on what distinguishes it.
    """
    return "_".join(f"{key.split('.')[-1]}={value}" for key, value in sorted(candidate.items()))


def spearman(left: list[float], right: list[float]) -> float:
    """Rank correlation, computed directly: the candidate counts here are single digits and a
    dependency for six numbers is not worth the install."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for position, index in enumerate(order):
            out[index] = float(position)
        return out
    a, b = ranks(left), ranks(right)
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    da = sum((x - mean_a) ** 2 for x in a) ** 0.5
    db = sum((y - mean_b) ** 2 for y in b) ** 0.5
    return cov / (da * db) if da and db else float("nan")


def main() -> int:
    dry = "--dry-run" in sys.argv
    parallel = PARALLEL
    if "--parallel" in sys.argv:
        parallel = int(sys.argv[sys.argv.index("--parallel") + 1])

    decision = decide()
    print("compute:", decision.device, "--", decision.why, flush=True)
    research, stages = build(decision)
    if "train" not in stages:
        raise SystemExit("train is not verified for this benchmark; derive it first")

    if dry:
        for item in [{"candidate": c, "epochs": e}
                     for e in (CHEAP, EXPENSIVE) for c in CANDIDATES]:
            settings = {**BASE, **item["candidate"], "train.n_epochs": item["epochs"]}
            argv = research.backend.argv(
                "train", research._inputs("train", settings=settings,
                                          device=decision.device,
                                          device_index=decision.device_index))
            # The whole command: truncating the tail cut off the one argument that
            # distinguishes the candidates, so the dry run printed four identical lines
            # for four different experiments.
            print(f"  {item['epochs']:>2} epochs  " + " ".join(argv))
        return 0


    # Screened before it is scheduled, always. Skipping this is what cost ten hours: four
    # candidates were taken from the declared space on the assumption that a declared value
    # is a usable value, and two of them cannot finish a single evaluation.
    kept_candidates = []
    for candidate in CANDIDATES:
        name = name_of(candidate)
        started = time.time()
        result = research.run_stage(
            "train", timeout=SMOKE_SECONDS,
            settings={**BASE, **candidate, "train.n_epochs": 1})
        # Only what this run wrote: the checkout holds every earlier run's checkpoints, and a
        # screen that yesterday's file can satisfy is not a screen.
        wrote = research.backend.artifact_beside("train", research.backend.directory("train"),
                                                 since=started)
        names = " ".join(wrote.get("examples") or [])
        # The evaluation finishing is the thing being screened for. `[Task 0 succ.]` is
        # LIBERO's own statement that it got through it; task 1's checkpoint is the same
        # fact one step later, and either is enough.
        ok = "task1_model" in names or result.get("returncode") == 0
        # The whole list, not the first hundred characters of it: a truncated print is what
        # made a criterion look right for the wrong reason a moment ago.
        print(f"  smoke {name}: {'runs' if ok else 'DOES NOT RUN'}"
              f"  (rc={result.get('returncode')}, {len(names.split())} file(s))", flush=True)
        for item in (wrote.get("examples") or []):
            print(f"      wrote {item}", flush=True)
        if not ok:
            print(f"      said: {str(result.get('said') or '')[-400:]}", flush=True)
        if ok:
            kept_candidates.append(candidate)
    if len(kept_candidates) < 3:
        raise SystemExit(f"only {len(kept_candidates)} candidate(s) completed a run; "
                         f"there is nothing to rank")
    candidates = kept_candidates

    plan = [{"candidate": candidate, "epochs": epochs}
            for epochs in (CHEAP, EXPENSIVE) for candidate in candidates]
    lock = threading.Lock()
    readings: list[dict] = []

    def measure(item: dict) -> dict:
        settings = {**BASE, **item["candidate"], "train.n_epochs": item["epochs"]}
        label = f"{name_of(item['candidate'])}_{item['epochs']}ep"
        result = research.run_stage("train", timeout=RUN_TIMEOUT[item["epochs"]],
                                    settings=settings)
        row = {"label": label, "epochs": item["epochs"], "candidate": item["candidate"],
               "returncode": result.get("returncode"),
               "readings": research._readings(result.get("said", "")),
               "success_rate": research._success_rate(result.get("said", "")),
               # Kept, because a run that failed is the most informative thing here and the
               # last attempt recorded only the numbers it managed to print.
               "said": (result.get("said") or "")[-2000:]}
        with lock:
            readings.append(row)
            print(f"  {label:>18} rc={row['returncode']} readings={row['readings']}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(measure, plan))

    def key(row):
        return name_of(row["candidate"])

    cheap = {key(r): r for r in readings if r["epochs"] == CHEAP}
    dear = {key(r): r for r in readings if r["epochs"] == EXPENSIVE}
    shared = sorted(set(cheap) & set(dear))
    report = {"cheap_epochs": CHEAP, "expensive_epochs": EXPENSIVE,
              "base_settings": BASE, "runs": readings}
    report["smoked"] = [name_of(c) for c in candidates]
    for quantity in ("loss", "succ"):
        pairs = [(cheap[name]["readings"].get(quantity), dear[name]["readings"].get(quantity),
                  name) for name in shared]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        if len(pairs) < 3:
            report.setdefault("verdict", {})[quantity] = "not enough readings to rank on"
            continue
        left = [p[0] for p in pairs]
        right = [p[1] for p in pairs]
        rho = spearman(left, right)
        report.setdefault("verdict", {})[quantity] = {
            "spearman": round(rho, 3), "n": len(pairs),
            "cheap": {p[2]: p[0] for p in pairs},
            "expensive": {p[2]: p[1] for p in pairs},
            "same_top": max(pairs, key=lambda p: p[0])[2] == max(pairs, key=lambda p: p[1])[2]}
    (OUTPUT / "screen_correlation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report.get("verdict"), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
