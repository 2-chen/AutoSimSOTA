---
name: report-paired-and-intervaled
description: Report paired outcomes and intervals; small-n aggregates cannot support small deltas.
scope: general
confidence: high — an audit of the benchmarks themselves, plus a statistical method validated on thousands of rollouts
evidence: |
  "What Are We Actually Benchmarking in Robot Manipulation?" (arXiv:2606.04233) audited
  LIBERO, CALVIN, SimplerEnv, RoboCasa and RoboTwin 2.0 and found that only 19.8% of
  LIBERO and 19.7% of SimplerEnv headline gains are provably statistically significant.
  Between the rejection-feasibility and rejection-guarantee thresholds, the same published
  aggregates are consistent with both significant and insignificant outcomes.

  Arithmetic: at ~90% success, 10 episodes gives a 95% interval spanning roughly ±19
  percentage points and 100 episodes still ~±6. Typical evaluation is 10-60 trials.

  N-SCORE (RSS 2026, arXiv:2603.13616), validated on 4,500+ hardware and 2,000 simulation
  rollouts: sequential testing over paired progress scores cut evaluation burden by ~70%
  versus batch testing, and on RoboArena recovered a full 4-policy ranking at α=0.05 using
  1,420 of ~2,560 evaluations. Partial-credit metrics separated policies faster than binary
  success.

  Reporting practice: success rate appears in ~98% of surveyed papers while confidence
  intervals have not improved (33.3% in 2023, 11.0% in 2025, 19.5% in 2026 YTD).
---

# Report paired and intervaled

A success rate without an interval is not a measurement, it is a number. Two policies
evaluated on the same seeds can be compared far more sharply than two evaluated
independently — and most of the field does not do this.

## How to apply

Three things, in order of how much they buy:

1. **Share the seeds.** Evaluate every arm on the same initial states and compare
   *per episode*, not in aggregate. This turns an unpaired comparison into a paired one
   and removes the initial-state variance entirely.
2. **Say the interval, not just the mean.** At the episode counts in common use the
   interval is wide enough to swallow most reported deltas. If your delta is inside the
   interval, say so — that is a result, and it is more useful than a point estimate.
3. **Prefer a graded score to a binary one.** Partial-credit measures separate policies
   with fewer rollouts than pass/fail, which matters directly when each rollout is
   expensive.

## What it does not mean

It does not mean a small experiment is worthless — it means naming its resolution. "28/40
against 31/40, interval overlapping" is honest and useful; "70% against 77.5%, improved"
is the reporting failure the audit measured. And a wide interval on a real effect is a
reason to run more episodes, not a reason to claim nothing.
