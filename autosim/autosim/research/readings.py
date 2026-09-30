"""The numbers a program gave a name to, read out of what it printed.

A training run prints its loss every hundred steps and the system kept none of it. The record
of a finished process was its exit code, its duration and a path to a log file nobody
opened -- so "did this run learn anything" had no answer short of re-reading the log by hand,
and the one piece of evidence a trainer produces continuously was the one thing not written
down. This reads it, once, in a place every process can reach.

Two rules, both learned from getting them wrong.

**Read by name, not by position.** The position of a number moves between epochs, between
benchmarks and between lines, and the name is the part the program chose deliberately:
`loss: 5.36`, `succ: 0.62`, `AoC = 0.11`. A reader that counts columns is a reader for one
program.

**A quantity that merely wears the same word is not the quantity.** `best succ: 0.90` is the
best epoch so far and `succ: 0.62` is this one; reading the first under the second's name
gives a right-looking number for the wrong question, which is worse than giving nothing.
`AoC` is excluded for the same reason -- it is computed over every task seen, with
counterbalancing, and it is not a success rate at this epoch.

Nothing here is specific to a benchmark: a program that names its numbers gets them recorded,
and one that names none gets an empty record rather than a guess.
"""

from __future__ import annotations

import re
from typing import Any

#: `name: 1.25`, `name = 1.25`, `name 1.25`. The lookbehinds drop `best succ`, which is a
#: different quantity under the same word -- see the module docstring.
_NAMED = re.compile(
    r"(?<!best )(?<!Best )(?<!max )(?<!Max )"
    r"\b([A-Za-z][A-Za-z_]{2,20})\s*[:=]?\s*"
    r"([-+]?[0-9]*\.?[0-9]+)")


def named_numbers(said: str, *, limit: int = 400_000) -> dict[str, float]:
    """Every number the program gave a name to, as it last said it.

    The last value per name, because a program that prints its loss every epoch has said
    what the loss is, and the last one is where it finished. Names are lowercased so that
    `Loss:` and `loss:` are one quantity rather than two.

    Nothing is filtered out. A first version dropped the words that are not measurements --
    `epoch`, `time`, `version` -- on the grounds that a method ranking candidates should not
    be tempted by a counter. That is a judgement made on the reader's behalf and it is the
    same mistake this repository keeps making in other clothes: a skeleton that filled its
    placeholders with zeros, a screening criterion that accepted the step before the one that
    fails. What a program said is a record; which parts of it matter is the caller's
    question, and a reader that has already answered it cannot be asked again.
    """
    readings: dict[str, float] = {}
    for match in _NAMED.finditer(said or ""):
        try:
            readings[match.group(1).lower()] = float(match.group(2))
        except ValueError:
            continue
    return readings


def explicitly_zero_training_work(readings: dict[str, float]) -> dict[str, float]:
    """Native planned-update counters that explicitly say training did no work.

    Absence is unknown, not proof of progress. These are generic counter semantics,
    not names of any simulator or benchmark; do not infer work from a fresh checkpoint.
    """
    planned = ("num_iterations", "total_iterations", "num_updates", "total_updates",
               "num_epochs", "total_epochs")
    return {name: readings[name] for name in planned
            if name in readings and readings[name] == 0}


def training_progress(said: str) -> dict[str, Any]:
    """Conservative evidence that a native training loop actually advanced.

    A new file and zero exit can be an initialization-only run. An absent progress
    signal is ``unknown``, never silently interpreted as an optimizer update.
    """
    readings = named_numbers(said)
    zero = explicitly_zero_training_work(readings)
    if zero:
        return {"status": "zero", "evidence": zero}
    for name in ("global_step", "optimizer_step", "update_step", "updates_completed"):
        if readings.get(name, 0) > 0:
            return {"status": "observed", "evidence": {name: readings[name]}}
    # A number of native RL trainers report steps-per-second (often abbreviated `SPS`)
    # after completing a rollout/update cycle rather than printing a tqdm bar or a final
    # `global_step`. A positive named throughput is direct evidence that native steps ran;
    # zero or absent throughput remains unknown rather than being treated as progress.
    if readings.get("sps", 0) > 0:
        return {"status": "observed",
                "evidence": {"steps_per_second": readings["sps"]}}
    # tqdm writes carriage-return-delimited progress bars. Its final N/N is a direct
    # reading of loop iterations completed, whereas 0it is a loop that never started.
    # tqdm prints either throughput (``1.00it/s``) or elapsed time per iteration
    # (``3.30s/it``). The latter is common for slow simulator steps; requiring only
    # ``it/s`` silently rejects completed work even when the native progress bar says
    # ``1/1024``. Accept both units while still requiring a tqdm-like bracketed rate.
    progress = re.findall(
        r"(?<!\d)(\d+)/(\d+)\s*\[[^\]\r\n]{0,180}"
        r"(?:\d+(?:\.\d+)?(?:it/s|s/it)|\?{1,2}it/s)"
        r"[^\]\r\n]{0,80}\]",
        said or "",
    )
    if progress:
        done, total = map(int, progress[-1])
        return {"status": "observed" if done > 0 else "zero",
                "evidence": {"tqdm_completed": done, "tqdm_total": total}}
    if re.search(r"(?:^|[\r\n])\s*0it\s*\[", said or ""):
        return {"status": "zero", "evidence": {"tqdm_completed": 0}}
    return {"status": "unknown", "evidence": {}}


_COMPLETED_EPISODES = (
    re.compile(r"\bevaluated\s+([\d,]+)\s+steps?\s+resulting\s+in\s+"
               r"([\d,]+)\s+episodes?\b", re.IGNORECASE),
    re.compile(r"\bcompleted(?:[ _]+)episodes?\s*[:=]\s*([\d,]+)\b", re.IGNORECASE),
)


def evaluation_progress(said: str) -> dict[str, Any]:
    """Read explicit completed-episode counters without treating requested counts as facts."""
    for line in reversed((said or "").splitlines()):
        for pattern in _COMPLETED_EPISODES:
            match = pattern.search(line)
            if not match:
                continue
            token = match.group(2) if len(match.groups()) > 1 else match.group(1)
            try:
                count = int(token.replace(",", ""))
            except ValueError:
                continue
            return {"status": "observed" if count > 0 else "zero",
                    "completed_episodes": count, "evidence": line.strip()[:300]}
    return {"status": "unknown", "completed_episodes": None, "evidence": ""}


#: A success reading as benchmarks print one: a word, a separator of any of `:`, `=`, `.` or
#: nothing, then the number. `AoC` is excluded rather than averaged in, and so is `best succ`,
#: which is the best epoch so far rather than the last one.
_SUCCESS = re.compile(
    r"(?<!best )(?<!max )\b(?:success[ _]+rate|success|succ)"
    r"(?!\s*\.?\s*AoC)\s*[.:=]?\s*\]?\s*"
    r"(?P<values>\d+(?:\.\d+)?(?![0-9.eE])\s*%?"
    r"(?:[ \t]*\|[ \t]*\d+(?:\.\d+)?(?![0-9.eE])\s*%?)*[ \t]*\|?)",
    re.IGNORECASE)


def success_rate(said: str) -> float | None:
    """The last labelled success reading the program printed, if it printed one.

    Only labelled ones. A program that ends with a bare `print(test_loss, success_rate)` has
    printed two unlabelled numbers, and those are not a reading a reader can take without
    guessing which is which -- so that returns `None`, which is reported as the number not
    having been read rather than as a zero.
    """
    return success_reading(said)["value"]


def success_reading(said: str) -> dict[str, Any]:
    """That reading, with what it was read *from*.

    A number without the line it came from cannot be checked. The failure this records for:
    a program that reports one success rate per task prints them separated by `|`, and this
    took the mean of them and stored the mean as a bare float. **Nothing said the number was
    an average**, over how many tasks, or what the per-task values were -- and that float is
    what the whole loop ranks candidates by, and what a threshold of `>= 0.03` is applied to.
    A reader who wanted to ask "is 0.2 an average of three tasks or one task scoring 0.2?"
    had no way to, and the two mean very different things about the policy.

    So the answer carries the line, the values it was reduced from, and whether it was
    reduced at all. The value is unchanged for every caller that wants only the number.
    """
    # Work one line at a time. A whitespace class containing newlines can turn a following
    # episode count into another task's score and make a failed evaluator appear successful.
    found = [(line, match) for line in (said or "").splitlines()
             for match in _SUCCESS.finditer(line)]
    if not found:
        return {"value": None, "read_from": "", "values": [], "averaged": False}
    line, match = found[-1]
    values: list[float] = []
    for token in match.group("values").replace("|", " ").split():
        try:
            number = float(token.rstrip("%"))
            value = number / 100 if token.endswith("%") else number
            # An unmarked 75 might be a percentage, a count, or another metric. It is not
            # a probability until the benchmark supplies a metric contract saying so.
            if not 0 <= value <= 1:
                return {"value": None, "read_from": match.group(0).strip(),
                        "values": [], "averaged": False, "why_not": "ambiguous rate unit"}
            values.append(value)
        except ValueError:
            continue
    if not values:
        return {"value": None, "read_from": "", "values": [], "averaged": False}
    excerpt = line[match.start():match.end()].strip()[:300]
    return {"value": sum(values) / len(values), "read_from": excerpt, "values": values,
            "averaged": len(values) > 1}


def tail_of(path, *, limit: int = 200_000) -> str:
    """The end of a log file, where a program's final numbers are.

    Bounded, because a training log is not. A program that reports every epoch has reported
    the same names many times and the last statement of each is what it settled on, so the
    end of the file carries everything this needs.
    """
    try:
        size = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            if size > limit:
                stream.seek(size - limit)
                stream.readline()               # discard the partial line at the seam
            return stream.read()
    except (OSError, AttributeError):
        return ""
