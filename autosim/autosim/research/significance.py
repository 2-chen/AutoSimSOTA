"""Whether a difference is worth believing, which is a different question from its size.

A research loop that reports "0.52 against 0.48, better" has reported two numbers and a
decision, and the decision does not follow from the numbers. Two measurements of the same
policy differ by more than that; a benchmark with forty episodes cannot resolve four
percentage points; and a run that tries thirty candidates and reports the best has reported
the maximum of thirty draws, not an improvement.

So nothing here decides *which* candidate wins. It answers the question a reader arrives
with -- is this difference larger than the noise this measurement has -- and it is allowed to
answer "not established", which is the answer most single measurements deserve.

**Bootstrap, not a table of quantiles.** The samples are episode outcomes or per-episode
returns from a benchmark whose distribution is nobody's business, and the count is usually
small. Resampling makes no assumption about either, costs nothing at these sizes, and -- with
the seed fixed -- gives the same interval twice, which matters more than the last decimal.
The one exact test here is McNemar's, because for paired binary outcomes it is a binomial and
computing it exactly is easier than approximating it.

**The verdicts are ordered and the weakest is the default.** `insufficient` when there is not
enough to say anything, `directional_only` when there is a difference in the sample and not
enough of one to establish it, `no_difference` when the interval says the sign is unknown,
and `better` / `worse` only when the whole interval is on one side of zero. A loop that can
only report the last two will report them about noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

#: Below this many observations a difference is reported as a direction and nothing more.
#: Eight is not a statistical threshold -- no threshold is honest at eight -- it is the point
#: below which the interval is so wide that stating one would be theatre. What it is really
#: saying is "this is a smoke test", which is what the plan asks small samples to stay.
DIRECTIONAL_BELOW = 8

#: Resamples. Four thousand puts the interval's own Monte-Carlo error far below the width of
#: any interval these sample sizes produce, and it is fast enough to run per round.
RESAMPLES = 4000


@dataclass
class Comparison:
    """Two sets of readings, and what can and cannot be said about their difference."""

    kind: str                       # "binary" | "continuous"
    paired: bool
    n_baseline: int
    n_candidate: int
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    difference: float | None = None
    interval: tuple[float, float] | None = None
    p_value: float | None = None
    test: str = ""
    verdict: str = "insufficient"
    why: str = ""
    #: What a reader must not read into this. Travels with the result rather than living in
    #: the reader's head, because the same numbers are quoted in the run's document.
    limitation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "paired": self.paired,
                "n_baseline": self.n_baseline, "n_candidate": self.n_candidate,
                "baseline_mean": self.baseline_mean, "candidate_mean": self.candidate_mean,
                "difference": self.difference,
                "interval": list(self.interval) if self.interval else None,
                "p_value": self.p_value, "test": self.test, "verdict": self.verdict,
                "why": self.why, "limitation": self.limitation}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap_difference(baseline: Sequence[float], candidate: Sequence[float], *,
                          paired: bool, alpha: float, seed: int) -> tuple[float, float]:
    """The interval on the difference, by resampling what was actually observed.

    Paired resamples the *pairs*, so the correlation between two measurements of the same
    episode is carried into the interval rather than assumed away. Unpaired resamples each
    side on its own, because there is no pairing to preserve and pretending there is would
    narrow the interval for a reason that does not hold.
    """
    rng = random.Random(seed)
    n = len(baseline)
    draws: list[float] = []
    for _ in range(RESAMPLES):
        if paired:
            picks = [rng.randrange(n) for _ in range(n)]
            draws.append(sum(candidate[i] - baseline[i] for i in picks) / n)
        else:
            a = [baseline[rng.randrange(n)] for _ in range(n)]
            m = len(candidate)
            b = [candidate[rng.randrange(m)] for _ in range(m)]
            draws.append(sum(b) / m - sum(a) / n)
    draws.sort()
    low = draws[max(0, int((alpha / 2) * RESAMPLES) - 1)]
    high = draws[min(RESAMPLES - 1, int((1 - alpha / 2) * RESAMPLES))]
    return low, high


def _mcnemar_exact(baseline: Sequence[float], candidate: Sequence[float]) -> tuple[float, str]:
    """The paired binary test, computed rather than approximated.

    Only the *discordant* pairs carry information: episodes both arms got right, or both got
    wrong, say nothing about which is better. Under the null the discordant ones split by a
    fair coin, which is a binomial and can be summed exactly -- so there is no reason to use
    the chi-square approximation its authors offered for hand computation.
    """
    wins = sum(1 for a, b in zip(baseline, candidate) if b > a)
    losses = sum(1 for a, b in zip(baseline, candidate) if a > b)
    if wins + losses == 0:
        return 1.0, "mcnemar_exact"
    n = wins + losses
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2 * tail), "mcnemar_exact"


def _two_proportion(baseline: Sequence[float], candidate: Sequence[float]) -> tuple[float, str]:
    """The unpaired binary test, by the normal approximation -- and it says so."""
    n, m = len(baseline), len(candidate)
    if not n or not m:
        return 1.0, "two_proportion"
    pooled = (sum(baseline) + sum(candidate)) / (n + m)
    if pooled in (0.0, 1.0):
        return 1.0, "two_proportion"
    standard = math.sqrt(pooled * (1 - pooled) * (1 / n + 1 / m))
    if standard == 0:
        return 1.0, "two_proportion"
    z = (sum(candidate) / m - sum(baseline) / n) / standard
    return math.erfc(abs(z) / math.sqrt(2)), "two_proportion (normal approximation)"


def compare(baseline: Sequence[float], candidate: Sequence[float], *,
            kind: str = "binary", paired: bool = True, alpha: float = 0.05,
            seed: int = 20260925) -> Comparison:
    """The difference between two sets of readings, and how far to trust it.

    `baseline` and `candidate` are per-episode outcomes (1.0 for a success, 0.0 for a failure)
    when `kind` is `binary`, and per-episode quantities when it is `continuous`. Paired means
    the two lists are the *same* episodes under two policies, which is the case a benchmark
    with a fixed initial-state bank gives and the case that lets a small sample say something.
    """
    left = [float(one) for one in baseline]
    right = [float(one) for one in candidate]
    result = Comparison(kind=kind, paired=paired, n_baseline=len(left), n_candidate=len(right))
    if paired and len(left) != len(right):
        result.why = (f"a paired comparison needs one reading per episode on both sides: "
                      f"{len(left)} against {len(right)}")
        result.limitation = "the two sets are not the same episodes, so nothing is paired"
        return result
    if not left or not right:
        result.why = "one side has no readings"
        return result

    result.baseline_mean = _mean(left)
    result.candidate_mean = _mean(right)
    result.difference = result.candidate_mean - result.baseline_mean

    if kind == "binary":
        for side, name in ((left, "baseline"), (right, "candidate")):
            bad = [one for one in side if one not in (0.0, 1.0)]
            if bad:
                result.why = (f"{name} has values that are not a success or a failure "
                              f"({bad[:3]}); a binary comparison needs outcomes")
                return result

    eligible = min(len(left), len(right)) if paired else min(len(left), len(right))
    low, high = _bootstrap_difference(left, right, paired=paired, alpha=alpha, seed=seed)
    result.interval = (low, high)

    if kind == "binary":
        result.p_value, result.test = (_mcnemar_exact(left, right) if paired
                                       else _two_proportion(left, right))
    else:
        result.p_value, result.test = None, ("bootstrap on paired differences" if paired
                                             else "bootstrap on two independent samples")

    # The ladder, weakest first, and nothing skips a rung.
    if eligible < DIRECTIONAL_BELOW:
        result.verdict = "directional_only"
        result.why = (f"{eligible} episodes is a smoke test: the interval spans "
                      f"{low:+.3f} to {high:+.3f}, which is wider than any difference this "
                      f"measurement could establish")
    elif low > 0 and high > 0 and (result.p_value is None or result.p_value <= alpha):
        result.verdict = "better"
        result.why = (f"the whole interval is above zero ({low:+.3f} to {high:+.3f})"
                      + (f", p={result.p_value:.4g}" if result.p_value is not None else ""))
    elif low < 0 and high < 0 and (result.p_value is None or result.p_value <= alpha):
        result.verdict = "worse"
        result.why = (f"the whole interval is below zero ({low:+.3f} to {high:+.3f})"
                      + (f", p={result.p_value:.4g}" if result.p_value is not None else ""))
    else:
        result.verdict = "no_difference"
        result.why = (f"the interval crosses zero ({low:+.3f} to {high:+.3f}), so the sign of "
                      f"the difference is not established"
                      + (f" (p={result.p_value:.4g})" if result.p_value is not None else ""))

    result.limitation = (
        "one comparison between two policies on the episodes that were run. It is not a "
        "claim about the benchmark's test set, about other tasks, or about a difference "
        "smaller than this interval; and a best-of-many selected this way is the maximum of "
        "many draws rather than a measured improvement.")
    if not paired:
        result.limitation += (" The two sides are not the same episodes, so a difference "
                              "includes whatever the episodes differed by.")
    return result


def across(values: Sequence[float], *, confidence: float = 0.95, seed: int = 20260925
           ) -> dict[str, Any]:
    """One set of readings, as a mean with an interval. For a baseline's own spread.

    The interval a repeat of the *same* condition would fall in -- which is the number a
    reader needs to judge whether a later difference means anything, and the one a report
    tends to omit.
    """
    numbers = [float(one) for one in values]
    if not numbers:
        return {"n": 0, "mean": None, "interval": None,
                "limitation": "no readings, so there is no spread to report"}
    rng = random.Random(seed)
    n = len(numbers)
    draws = sorted(sum(numbers[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(RESAMPLES))
    low = draws[max(0, int(((1 - confidence) / 2) * RESAMPLES) - 1)]
    high = draws[min(RESAMPLES - 1, int((1 - (1 - confidence) / 2) * RESAMPLES))]
    return {"n": n, "mean": sum(numbers) / n, "interval": [low, high],
            "confidence": confidence,
            "limitation": (f"{n} readings. This is the spread of this sample; it says nothing "
                           f"about episodes that were not run.")}


def _normal_quantile(confidence: float) -> float:
    """The two-sided normal quantile, by bisection on `erf` rather than a table.

    Exact to machine precision in a few dozen cheap iterations, which beats carrying the
    four-decimal table every statistics reference prints and then mis-transcribing one row.
    """
    low, high = 0.0, 8.0
    for _ in range(200):
        middle = (low + high) / 2
        if math.erf(middle / math.sqrt(2)) < confidence:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def proportion(successes: int, total: int, *, confidence: float = 0.95) -> dict[str, Any]:
    """A rate and the interval it deserves, from a count and a denominator.

    The Wilson interval rather than the textbook `p ± 1.96·sqrt(p(1-p)/n)`, because the
    textbook one is wrong in exactly the region benchmarks live: it collapses to zero width
    at 0 % and 100 %, and it extends past them. A benchmark where the policy succeeds on
    every episode has not established that it always will, and its interval has to say so.

    This is the interval a run can nearly always compute -- a benchmark prints a success rate
    and its settings say how many episodes it was over -- which makes it the difference
    between reporting a number and reporting a number a reader can weigh.
    """
    if total <= 0 or not 0 <= successes <= total:
        return {"successes": successes, "n": total, "rate": None, "interval": None,
                "limitation": "a rate needs a count and a denominator that contains it"}
    rate = successes / total
    z = _normal_quantile(confidence)
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"successes": successes, "n": total, "rate": rate,
            "interval": [max(0.0, centre - half), min(1.0, centre + half)],
            "confidence": confidence,
            "limitation": (f"{successes} of {total} episodes. The interval is over episodes "
                           f"like these; it is not a claim about the benchmark's test set, "
                           f"and at {total} episodes it is wide.")}


def compare_proportions(successes: int, n: int, other_successes: int, other_n: int, *,
                        alpha: float = 0.05) -> Comparison:
    """Two rates from two denominators, which is usually all a benchmark run leaves behind.

    The interval on the difference is Newcombe's, built from the two Wilson intervals rather
    than from a pooled variance. It behaves at 0 % and 100 % -- where a policy that fails
    every episode on one arm and succeeds on three of forty is the ordinary early-round case,
    and the standard formula divides by a variance of zero and reports certainty.
    """
    left = proportion(successes, n, confidence=1 - alpha)
    right = proportion(other_successes, other_n, confidence=1 - alpha)
    result = Comparison(kind="binary", paired=False, n_baseline=n, n_candidate=other_n)
    result.baseline_mean, result.candidate_mean = left["rate"], right["rate"]
    if left["rate"] is None or right["rate"] is None:
        result.why = left["limitation"] if left["rate"] is None else right["limitation"]
        return result
    result.difference = right["rate"] - left["rate"]
    (l1, u1), (l2, u2) = left["interval"], right["interval"]
    p1, p2 = left["rate"], right["rate"]
    low = result.difference - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    high = result.difference + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    result.interval = (low, high)
    # No p-value: with only the counts there is no discordant-pair structure to test, and the
    # interval is the whole result. Reporting one from a variance computed on aggregates would
    # be a number with a derivation a reader cannot check.
    result.test = "newcombe interval on the difference of two rates"
    if low > 0:
        result.verdict, result.why = "better", (
            f"the whole interval is above zero ({low:+.3f} to {high:+.3f})")
    elif high < 0:
        result.verdict, result.why = "worse", (
            f"the whole interval is below zero ({low:+.3f} to {high:+.3f})")
    else:
        result.verdict, result.why = "no_difference", (
            f"the interval on {p1:.3f} against {p2:.3f} crosses zero "
            f"({low:+.3f} to {high:+.3f}), so the sign is not established")
    result.limitation = (
        "two rates over episodes that were run, each with its own denominator. It is not a "
        "claim about the benchmark's test set, and it is not a paired comparison -- if the "
        "two arms ran different episodes, part of this difference is which episodes they were.")
    return result


def spread_between_runs(runs: Sequence[float]) -> dict[str, Any]:
    """How much a number moves between independent runs of the same condition.

    The comparison a research loop most needs and least often makes: it selects the best of
    several attempts and reports the improvement, and the spread between repeats of one
    condition is what that improvement has to beat. Reported as a range because with three
    or four runs nothing better can be said.
    """
    numbers = [float(one) for one in runs if one is not None]
    if len(numbers) < 2:
        return {"runs": len(numbers), "range": None, "spread": None,
                "limitation": ("one run has no spread. Without two runs of the same "
                               "condition there is no scale to read a difference against")}
    return {"runs": len(numbers), "minimum": min(numbers), "maximum": max(numbers),
            "range": max(numbers) - min(numbers),
            "spread": _mean(numbers),
            "limitation": (f"the spread of {len(numbers)} runs. A difference smaller than "
                           f"{max(numbers) - min(numbers):.4f} has been produced by this "
                           f"benchmark running the same thing twice.")}
